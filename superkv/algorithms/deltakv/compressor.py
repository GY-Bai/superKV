"""DeltaKV compressor — implements the KVCompressor protocol.

Manages keyframe storage, residual encoding, Q4_0 quantization,
and sparse token selection for each transformer layer.
"""

from __future__ import annotations

import torch
from collections import defaultdict

from superkv.engine.registry import KVCompressor, register_algorithm
from superkv.algorithms.deltakv.core import (
    delta_encode_q4_0,
    delta_decode_q4_0,
    delta_encode_q4_0_normalized,
    delta_decode_q4_0_normalized,
    is_keyframe,
    Q4_0_BLOCK_SIZE,
)


@register_algorithm
class DeltaKVCompressor:
    """DeltaKV KV cache compressor.

    For each layer, maintains:
      - last_keyframe: K, V tensors of the most recent keyframe token
      - compressed: list of Q4_0-packed residuals since last keyframe
    """

    name = "deltakv"
    version = "2.0"

    def __init__(self, num_heads: int, head_dim: int,
                 reference_stride: int = 8,
                 num_layers: int = 1,
                 normalized: bool = True,
                 max_sparse_tokens: int = 256):
        """
        Args:
            num_heads: number of KV attention heads
            head_dim: dimension per head (must be divisible by 32)
            reference_stride: keyframe interval (stride=1 = no compression)
            num_layers: number of transformer layers
            normalized: use per-head normalized Q4_0 (recommended for real models)
            max_sparse_tokens: max tokens to select for sparse attention
        """
        assert head_dim % Q4_0_BLOCK_SIZE == 0, (
            f"head_dim {head_dim} must be divisible by {Q4_0_BLOCK_SIZE}")

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.stride = reference_stride
        self.num_layers = num_layers
        self.normalized = normalized
        self.max_sparse = max_sparse_tokens

        # Per-layer state
        self._keyframes: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self._compressed: dict[int, list[tuple]] = defaultdict(list)
        self._scales: dict[int, list[torch.Tensor]] = defaultdict(list)
        self._total_original = 0
        self._total_compressed = 0

    # ── KVCompressor protocol ───────────────────────────────────────

    def compress(self, K: torch.Tensor, V: torch.Tensor,
                 layer_idx: int = 0,
                 token_id: int | None = None) -> tuple:
        """Compress a single token's K, V.

        If token is a keyframe, store it as reference.
        Otherwise, encode residual from last keyframe.

        Args:
            K: (num_heads, head_dim) float32
            V: (num_heads, head_dim) float32
            layer_idx: layer index
            token_id: position in sequence (used for keyframe check)

        Returns:
            (q_k, d_k, q_v, d_v, head_scale_k, head_scale_v) or None if keyframe
        """
        elem_bytes = K.element_size()
        self._total_original += K.numel() * elem_bytes + V.numel() * elem_bytes

        if token_id is not None and is_keyframe(token_id, self.stride):
            # Keyframe: store full-precision
            self._keyframes[layer_idx] = (K.clone(), V.clone())
            self._total_compressed += K.numel() * elem_bytes + V.numel() * elem_bytes
            return None  # nothing compressed

        kf = self._keyframes.get(layer_idx)
        if kf is None:
            # No keyframe yet — treat this token as first keyframe
            self._keyframes[layer_idx] = (K.clone(), V.clone())
            self._total_compressed += K.numel() * elem_bytes + V.numel() * elem_bytes
            return None

        kf_k, kf_v = kf

        if self.normalized:
            q_k, d_k, sk = delta_encode_q4_0_normalized(K, kf_k)
            q_v, d_v, sv = delta_encode_q4_0_normalized(V, kf_v)
            self._scales[layer_idx].append((sk, sv))
            self._total_compressed += q_k.nbytes + d_k.nbytes + sk.nbytes
            self._total_compressed += q_v.nbytes + d_v.nbytes + sv.nbytes
        else:
            q_k, d_k = delta_encode_q4_0(K, kf_k)
            q_v, d_v = delta_encode_q4_0(V, kf_v)
            self._total_compressed += q_k.nbytes + d_k.nbytes
            self._total_compressed += q_v.nbytes + d_v.nbytes

        result = (q_k, d_k, q_v, d_v)
        self._compressed[layer_idx].append(result)
        return result

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress a single token's K, V from packed format."""
        q_k, d_k, q_v, d_v = packed
        kf = self._keyframes[layer_idx]
        if self.normalized and self._scales[layer_idx]:
            sk, sv = self._scales[layer_idx].pop(0)
            K = delta_decode_q4_0_normalized(kf[0], q_k, d_k, sk)
            V = delta_decode_q4_0_normalized(kf[1], q_v, d_v, sv)
        else:
            K = delta_decode_q4_0(kf[0], q_k, d_k)
            V = delta_decode_q4_0(kf[1], q_v, d_v)
        return K, V

    def memory_report(self) -> dict:
        """Return compression statistics."""
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
        """Clear all stored KV cache."""
        self._keyframes.clear()
        self._compressed.clear()
        self._scales.clear()
        self._total_original = 0
        self._total_compressed = 0
