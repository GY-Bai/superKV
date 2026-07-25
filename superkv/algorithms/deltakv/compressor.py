"""DeltaKV compressor — implements the KVCompressor protocol.

V3: INT8 delta encoding for both K and V.
V4: Batch support + bounded memory (max_tokens).
"""

from __future__ import annotations

import torch
from collections import OrderedDict

from superkv.engine.registry import KVCompressor, register_algorithm
from superkv.algorithms.deltakv.core import (
    delta_encode_int8,
    delta_decode_int8,
    is_keyframe,
    Q4_0_BLOCK_SIZE,
)


@register_algorithm
class DeltaKVCompressor:
    """DeltaKV KV cache compressor (V4 batched + bounded)."""

    name = "deltakv"
    version = "4.0"

    def __init__(self, num_heads: int, head_dim: int,
                 reference_stride: int = 8,
                 num_layers: int = 1,
                 max_tokens: int = 0,
                 **kwargs):
        """
        Args:
            num_heads, head_dim: model dimensions
            reference_stride: keyframe interval
            num_layers: number of transformer layers
            max_tokens: max stored tokens per layer (0 = unlimited).
                        Oldest tokens evicted when exceeded.
        """
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.stride = reference_stride
        self.num_layers = num_layers
        self.max_tokens = max_tokens

        # Per-layer state: OrderedDict for FIFO eviction
        self._keyframes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._tokens: dict[int, OrderedDict] = {}  # token_id → packed
        self._total_original = 0
        self._total_compressed = 0

    # ── KVCompressor protocol ───────────────────────────────────────

    def compress(self, K: torch.Tensor, V: torch.Tensor,
                 layer_idx: int = 0,
                 token_id: int | None = None) -> tuple | None:
        """Compress K,V. Supports both per-token (2D) and batched (3D).

        Args:
            K: (n_heads, head_dim) or (batch, n_heads, head_dim)
            V: same shape as K
        """
        # Detect batch mode
        if K.dim() == 3:
            return self._compress_batch(K, V, layer_idx, token_id)
        return self._compress_single(K, V, layer_idx, token_id)

    def _compress_single(self, K, V, layer_idx, token_id):
        elem_bytes = K.element_size()
        self._total_original += K.numel() * elem_bytes + V.numel() * elem_bytes

        if token_id is not None and is_keyframe(token_id, self.stride):
            self._keyframes[layer_idx] = (K.clone(), V.clone())
            self._add_token(layer_idx, token_id, None)  # marker
            return None

        kf = self._keyframes.get(layer_idx)
        if kf is None:
            self._keyframes[layer_idx] = (K.clone(), V.clone())
            self._add_token(layer_idx, token_id, None)
            return None

        kf_k, kf_v = kf
        dk_int8, dk_scale = delta_encode_int8(K, kf_k)
        dv_int8, dv_scale = delta_encode_int8(V, kf_v)

        self._total_compressed += dk_int8.nbytes + dk_scale.nbytes
        self._total_compressed += dv_int8.nbytes + dv_scale.nbytes

        packed = (dk_int8, dk_scale, dv_int8, dv_scale)
        self._add_token(layer_idx, token_id, packed)
        return packed

    def _compress_batch(self, K, V, layer_idx, token_ids):
        """Compress a batch of tokens."""
        batch = K.shape[0]
        # If token_ids is an int, it's the base offset
        if isinstance(token_ids, int) or token_ids is None:
            base = token_ids or 0
            token_ids = [base + b for b in range(batch)]
        results = []
        for b in range(batch):
            r = self._compress_single(
                K[b], V[b], layer_idx, token_ids[b])
            results.append(r)
        return results if any(r is not None for r in results) else None

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        dk_int8, dk_scale, dv_int8, dv_scale = packed
        kf_k, kf_v = self._keyframes[layer_idx]
        K = delta_decode_int8(kf_k, dk_int8, dk_scale)
        V = delta_decode_int8(kf_v, dv_int8, dv_scale)
        return K, V

    def _add_token(self, layer_idx, token_id, packed):
        """Store token and evict oldest if needed."""
        if layer_idx not in self._tokens:
            self._tokens[layer_idx] = OrderedDict()
        store = self._tokens[layer_idx]
        key = token_id if token_id is not None else len(store)
        store[key] = packed
        if self.max_tokens > 0 and len(store) > self.max_tokens:
            # Evict oldest
            oldest_key = next(iter(store))
            del store[oldest_key]

    def memory_report(self) -> dict:
        ratio = (self._total_original / self._total_compressed
                 if self._total_compressed > 0 else 1.0)
        total_tokens = sum(len(v) for v in self._tokens.values())
        return {
            'algorithm': self.name,
            'version': self.version,
            'original_bytes': self._total_original,
            'compressed_bytes': self._total_compressed,
            'compression_ratio': ratio,
            'stored_tokens': total_tokens,
            'backend': 'pytorch',
        }

    def reset(self):
        self._keyframes.clear()
        self._tokens.clear()
        self._total_original = 0
        self._total_compressed = 0
