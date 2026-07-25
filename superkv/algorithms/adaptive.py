"""AdaptiveCompressor — auto-selects compression strategy.

Routes each token's K,V through the optimal pipeline based on:
  X-axis: K value range  →  DeltaKV INT8 vs TurboQuant
  Y-axis: sequence length →  pure quant vs quant+eviction  
  Z-axis: layer depth     →  eviction ratio (pyramid)

Boundaries (empirically tuned on Qwen3-8B + OLMoE-1B-7B):

  K range:
    |K|_max < 10   → TurboQuant (K3V2, ~6x, MSE 0.06)
    |K|_max >= 10  → DeltaKV INT8 (4.5x, MSE 0.005)

  Sequence length:
    < 4096         → pure quantization (no eviction)
    4096-32768     → quant + ChunkKV top-50%
    > 32768        → quant + ChunkKV top-25%

  Layer depth (of num_layers):
    0-33%          → keep 100% (shallow, precise)
    33-66%         → keep 50%  (middle)
    66-100%        → keep 25%  (deep, aggressive)
"""

from __future__ import annotations

import torch
import math

from superkv.engine.registry import KVCompressor, register_algorithm
from superkv.engine.registry import create_compressor
from superkv.algorithms.chunkkv import ChunkKVTracker
from superkv.engine.guards import check_kv


# ── Boundary constants ───────────────────────────────────────────────

K_RANGE_THRESHOLD = 10.0       # K.abs().max() boundary

SEQ_SHORT = 4096               # pure quant, no eviction
SEQ_LONG = 32768               # aggressive eviction

LAYER_SHALLOW = 0.33           # keep 100%
LAYER_DEEP = 0.66              # keep 25%, middle keeps 50%


@register_algorithm
class AdaptiveCompressor:
    """Auto-dispatches per-token K,V to optimal compression pipeline.

    Usage identical to any KVCompressor:
        c = create_compressor('adaptive', num_heads=8, head_dim=128,
                              num_layers=36)
        packed = c.compress(K, V, layer_idx=0, token_id=0)
    """

    name = "adaptive"
    version = "0.1"

    def __init__(self, num_heads: int, head_dim: int,
                 num_layers: int = 1,
                 seq_length: int = 0,
                 **kwargs):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.seq_length = seq_length or SEQ_LONG  # estimate

        # Underlying compressors (lazy init)
        self._deltakv = None
        self._turboquant = None

        # ChunkKV per layer depth tier
        self._chunkkv_shallow = None  # 100% keep → None
        self._chunkkv_middle = ChunkKVTracker(
            chunk_size=8, top_k=4, layer_reuse=True)
        self._chunkkv_deep = ChunkKVTracker(
            chunk_size=8, top_k=2, layer_reuse=True)

        # Stats
        self._dispatch_counts = {'deltakv': 0, 'turboquant': 0,
                                 'eviction': 0, 'total': 0}

    # ── Dispatch logic ───────────────────────────────────────────────

    def _choose_quant(self, K: torch.Tensor) -> str:
        """Choose quantization method based on K value range."""
        if K.abs().max().item() >= K_RANGE_THRESHOLD:
            return 'deltakv'
        return 'turboquant'

    def _choose_eviction(self, layer_idx: int
                         ) -> ChunkKVTracker | None:
        """Choose eviction tracker based on layer depth."""
        depth = layer_idx / max(self.num_layers - 1, 1)

        if depth < LAYER_SHALLOW:
            return None  # keep everything
        elif depth < LAYER_DEEP:
            return self._chunkkv_middle  # keep 50%
        else:
            return self._chunkkv_deep    # keep 25%

    def _should_evict(self) -> bool:
        """Whether to use eviction at all based on sequence length."""
        return self.seq_length >= SEQ_SHORT

    # ── KVCompressor protocol ────────────────────────────────────────

    def compress(self, K: torch.Tensor, V: torch.Tensor,
                 layer_idx: int = 0,
                 token_id: int | None = None) -> tuple | None:
        """Auto-dispatched compress."""
        K, V = check_kv(K, V, expected_heads=self.num_heads,
                        expected_dim=self.head_dim)

        self._dispatch_counts['total'] += 1

        # 1. Choose quantization method
        quant_method = self._choose_quant(K)
        self._dispatch_counts[quant_method] += 1

        # 2. Eviction check
        evict = None
        if self._should_evict():
            evict = self._choose_eviction(layer_idx)
            if evict is not None and token_id is not None:
                if not evict.should_keep(token_id, K, layer_idx):
                    self._dispatch_counts['eviction'] += 1
                    return None  # token evicted

        # 3. Dispatch — but ALWAYS feed keyframe to both compressors
        result = None
        for method_name, comp in [('deltakv', self._get_deltakv()),
                                    ('tq', self._get_turboquant())]:
            r = comp.compress(K, V, layer_idx=layer_idx, token_id=token_id)
            if method_name == quant_method:
                result = r  # keep this method's result
        
        if result is not None:
            return (quant_method, result)
        return None

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress — determine method from packed format."""
        # Packed format: (method_tag, actual_packed)
        method, inner_packed = packed
        if method == 'deltakv':
            return self._get_deltakv().decompress(inner_packed, layer_idx)
        else:
            return self._get_turboquant().decompress(inner_packed, layer_idx)

    # ── Lazy compressor init ─────────────────────────────────────────

    def _get_deltakv(self):
        if self._deltakv is None:
            self._deltakv = create_compressor(
                'deltakv', num_heads=self.num_heads, head_dim=self.head_dim,
                reference_stride=8, num_layers=self.num_layers)
        return self._deltakv

    def _get_turboquant(self):
        if self._turboquant is None:
            self._turboquant = create_compressor(
                'turboquant', num_heads=self.num_heads, head_dim=self.head_dim,
                bits_k=3, bits_v=2, num_layers=self.num_layers)
        return self._turboquant

    def memory_report(self) -> dict:
        dk = self._get_deltakv().memory_report()
        tq = self._get_turboquant().memory_report()
        return {
            'algorithm': self.name,
            'version': self.version,
            'dispatch': self._dispatch_counts,
            'deltakv_ratio': dk['compression_ratio'],
            'turboquant_ratio': tq['compression_ratio'],
        }

    def reset(self):
        if self._deltakv: self._deltakv.reset()
        if self._turboquant: self._turboquant.reset()
        self._chunkkv_middle.reset()
        self._chunkkv_deep.reset()
        self._dispatch_counts = {k: 0 for k in self._dispatch_counts}
