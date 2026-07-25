"""DeltaKV compressor — implements the KVCompressor protocol.

V3: Asymmetric K/V encoding.
  K: INT8 delta (residuals are small, INT8 is accurate)
  V: Q4_0 delta (values are small, Q4_0 works well)

Keyframe tokens are stored at full precision (every `reference_stride` tokens).
Non-keyframe tokens store:
  delta_K = K_curr - K_keyframe  → INT8 with per-head scale
  delta_V = V_curr - V_keyframe  → Q4_0 with per-block scale
"""

from __future__ import annotations

import torch
from collections import defaultdict

from superkv.engine.registry import KVCompressor, register_algorithm
from superkv.algorithms.deltakv.core import (
    delta_encode_q4_0,
    delta_decode_q4_0,
    delta_encode_int8,
    delta_decode_int8,
    is_keyframe,
    Q4_0_BLOCK_SIZE,
)


@register_algorithm
class DeltaKVCompressor:
    """DeltaKV KV cache compressor (V3 asymmetric)."""

    name = "deltakv"
    version = "3.0"

    def __init__(self, num_heads: int, head_dim: int,
                 reference_stride: int = 8,
                 num_layers: int = 1,
                 normalized: bool = True,
                 max_sparse_tokens: int = 256,
                 **kwargs):  # accept legacy params silently
        assert head_dim % Q4_0_BLOCK_SIZE == 0

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.stride = reference_stride
        self.num_layers = num_layers
        self.normalized = normalized
        self.max_sparse = max_sparse_tokens

        self._keyframes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._compressed: dict[int, list[tuple]] = {}
        # Per-token scales: (dk_scale, sv or None)
        self._scales: dict[int, list[tuple]] = {}
        self._total_original = 0
        self._total_compressed = 0

    # ── KVCompressor protocol ───────────────────────────────────────

    def compress(self, K: torch.Tensor, V: torch.Tensor,
                 layer_idx: int = 0,
                 token_id: int | None = None) -> tuple | None:
        elem_bytes = K.element_size()
        self._total_original += K.numel() * elem_bytes + V.numel() * elem_bytes

        # Keyframe: store full precision
        if token_id is not None and is_keyframe(token_id, self.stride):
            self._keyframes[layer_idx] = (K.clone(), V.clone())
            self._total_compressed += K.numel() * elem_bytes + V.numel() * elem_bytes
            return None

        kf = self._keyframes.get(layer_idx)
        if kf is None:
            self._keyframes[layer_idx] = (K.clone(), V.clone())
            self._total_compressed += K.numel() * elem_bytes + V.numel() * elem_bytes
            return None

        kf_k, kf_v = kf

        # K: INT8 delta (residuals are ~±50, INT8 handles this well)
        dk_int8, dk_scale = delta_encode_int8(K, kf_k)
        self._total_compressed += dk_int8.nbytes + dk_scale.nbytes

        # V: Q4_0 delta (V values are small, Q4_0 is accurate)
        q_v, d_v = delta_encode_q4_0(V, kf_v)
        self._total_compressed += q_v.nbytes + d_v.nbytes

        result = (dk_int8, q_v, d_v, dk_scale)
        self._compressed.setdefault(layer_idx, []).append(result)
        return result

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        dk_int8, q_v, d_v, dk_scale = packed
        kf_k, kf_v = self._keyframes[layer_idx]

        K = delta_decode_int8(kf_k, dk_int8, dk_scale)
        V = delta_decode_q4_0(kf_v, q_v, d_v)
        return K, V

    def memory_report(self) -> dict:
        ratio = (self._total_original / self._total_compressed
                 if self._total_compressed > 0 else 1.0)
        return {
            'algorithm': self.name,
            'version': self.version,
            'original_bytes': self._total_original,
            'compressed_bytes': self._total_compressed,
            'compression_ratio': ratio,
            'backend': 'pytorch',
        }

    def reset(self):
        self._keyframes.clear()
        self._compressed.clear()
        self._scales.clear()
        self._total_original = 0
        self._total_compressed = 0
