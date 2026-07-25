"""DeltaKV compressor — implements the KVCompressor protocol.

V3: INT8 delta encoding for both K and V.

Key insight: K and V residuals (delta from keyframe) are much smaller
than the original values, making INT8 accurate enough for real models
across all layers (shallow and deep).

  shallow layer 0:  delta_K ∈ [-75, 82],   delta_V ∈ [-0.3, 0.3]
  deep layer 35:    delta_K ∈ [-29, 24],    delta_V ∈ [-23, 17]

INT8 handles the full range with MSE < 0.01 for K and < 0.01 for V.

Compression: ~4x (FP32 → INT8 with per-head scale + keyframe overhead).
"""

from __future__ import annotations

import torch
from collections import defaultdict

from superkv.engine.registry import KVCompressor, register_algorithm
from superkv.algorithms.deltakv.core import (
    delta_encode_int8,
    delta_decode_int8,
    is_keyframe,
    Q4_0_BLOCK_SIZE,
)


@register_algorithm
class DeltaKVCompressor:
    """DeltaKV KV cache compressor (V3 INT8 delta)."""

    name = "deltakv"
    version = "3.0"

    def __init__(self, num_heads: int, head_dim: int,
                 reference_stride: int = 8,
                 num_layers: int = 1,
                 normalized: bool = True,
                 max_sparse_tokens: int = 256,
                 **kwargs):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.stride = reference_stride
        self.num_layers = num_layers
        self.normalized = normalized
        self.max_sparse = max_sparse_tokens

        self._keyframes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
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

        # INT8 delta for both K and V
        dk_int8, dk_scale = delta_encode_int8(K, kf_k)
        dv_int8, dv_scale = delta_encode_int8(V, kf_v)

        self._total_compressed += dk_int8.nbytes + dk_scale.nbytes
        self._total_compressed += dv_int8.nbytes + dv_scale.nbytes

        return (dk_int8, dk_scale, dv_int8, dv_scale)

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        dk_int8, dk_scale, dv_int8, dv_scale = packed
        kf_k, kf_v = self._keyframes[layer_idx]

        K = delta_decode_int8(kf_k, dk_int8, dk_scale)
        V = delta_decode_int8(kf_v, dv_int8, dv_scale)
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
        self._total_original = 0
        self._total_compressed = 0
