"""TurboQuant compressor — implements the KVCompressor protocol.

TurboQuant (Google ICLR 2026):
  Random orthogonal rotation + Lloyd-Max optimal scalar quantization.
  Asymmetric K/V: K uses more bits than V (e.g., K=3bit, V=2bit).

V2: Hybrid K encoding.
  Lloyd-Max codebook is designed for N(0,1) — fails when K ∈ [-200,200].
  Auto-detect: if K.abs().max() > 50, switch to INT8 direct quantization
  (bypass Lloyd-Max). V always uses TurboQuant V2.
"""

from __future__ import annotations

import torch

from superkv.engine.registry import KVCompressor, register_algorithm
from superkv.algorithms.turboquant.core import (
    generate_rotation_matrix,
    turboquant_quantize,
    turboquant_dequantize,
)


@register_algorithm
class TurboQuantCompressor:
    """TurboQuant KV cache compressor with adaptive K encoding."""

    name = "turboquant"
    version = "0.2"

    def __init__(self, num_heads: int, head_dim: int,
                 bits_k: int = 3, bits_v: int = 2,
                 num_layers: int = 1,
                 k_auto_scale: bool = True,
                 device: str = "cpu"):
        assert 2 <= bits_k <= 4
        assert 2 <= bits_v <= 4

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.num_layers = num_layers
        self.k_auto_scale = k_auto_scale
        self.device = torch.device(device)

        self._Pi_k: dict[int, torch.Tensor] = {}
        self._Pi_v: dict[int, torch.Tensor] = {}
        self._compressed: dict[int, list[tuple]] = {}
        self._total_original = 0
        self._total_compressed = 0

    def _get_Pi(self, layer_idx: int, which: str) -> torch.Tensor:
        cache = self._Pi_k if which == "k" else self._Pi_v
        if layer_idx not in cache:
            seed = 42 + layer_idx * 2 + (0 if which == "k" else 1)
            cache[layer_idx] = generate_rotation_matrix(
                self.head_dim, device=self.device, seed=seed)
        return cache[layer_idx]

    # ── KVCompressor protocol ────────────────────────────────────────

    def compress(self, K: torch.Tensor, V: torch.Tensor,
                 layer_idx: int = 0,
                 token_id: int | None = None) -> tuple:
        self._total_original += K.numel() * 4 + V.numel() * 4

        # V: always TurboQuant
        Pi_v = self._get_Pi(layer_idx, "v")
        if Pi_v.device != V.device:
            Pi_v = Pi_v.to(V.device); self._Pi_v[layer_idx] = Pi_v
        idx_v, norm_v = turboquant_quantize(V.float(), Pi_v, self.bits_v)
        self._total_compressed += idx_v.nbytes + norm_v.nbytes

        # K: auto-detect wide range → INT8; otherwise → Lloyd-Max
        if self.k_auto_scale and K.abs().max() > 50:
            # Wide range: bypass Lloyd-Max, use INT8 directly
            from superkv.algorithms.deltakv.core import (
                delta_encode_int8, delta_decode_int8)
            k_int8, k_scale = delta_encode_int8(K.float(), torch.zeros_like(K))
            self._total_compressed += k_int8.nbytes + k_scale.nbytes
            result = ('int8', k_int8, k_scale, idx_v, norm_v)
        else:
            Pi_k = self._get_Pi(layer_idx, "k")
            if Pi_k.device != K.device:
                Pi_k = Pi_k.to(K.device); self._Pi_k[layer_idx] = Pi_k
            idx_k, norm_k = turboquant_quantize(K.float(), Pi_k, self.bits_k)
            self._total_compressed += idx_k.nbytes + norm_k.nbytes
            result = ('tq', idx_k, norm_k, idx_v, norm_v)

        self._compressed.setdefault(layer_idx, []).append(result)
        return result

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = packed[0]

        if mode == 'int8':
            _, k_int8, k_scale, idx_v, norm_v = packed
            from superkv.algorithms.deltakv.core import delta_decode_int8
            K = delta_decode_int8(
                torch.zeros(self.num_heads, self.head_dim,
                           device=k_int8.device),
                k_int8, k_scale)
            Pi_v = self._to_device(self._get_Pi(layer_idx, "v"), idx_v.device)
            V = turboquant_dequantize(
                idx_v, norm_v, Pi_v, self.bits_v,
                (self.num_heads, self.head_dim))
        else:
            _, idx_k, norm_k, idx_v, norm_v = packed
            Pi_k = self._to_device(self._get_Pi(layer_idx, "k"), idx_k.device)
            Pi_v = self._to_device(self._get_Pi(layer_idx, "v"), idx_v.device)
            K = turboquant_dequantize(
                idx_k, norm_k, Pi_k, self.bits_k,
                (self.num_heads, self.head_dim))
            V = turboquant_dequantize(
                idx_v, norm_v, Pi_v, self.bits_v,
                (self.num_heads, self.head_dim))

        return K, V

    @staticmethod
    def _to_device(tensor, device):
        return tensor if tensor.device == device else tensor.to(device)

    def memory_report(self) -> dict:
        ratio = (self._total_original / self._total_compressed
                 if self._total_compressed > 0 else 1.0)
        return {
            'algorithm': self.name, 'version': self.version,
            'original_bytes': self._total_original,
            'compressed_bytes': self._total_compressed,
            'compression_ratio': ratio,
            'bits_k': self.bits_k, 'bits_v': self.bits_v,
            'backend': 'pytorch',
        }

    def reset(self):
        self._compressed.clear()
        self._Pi_k.clear(); self._Pi_v.clear()
        self._total_original = 0; self._total_compressed = 0
# Legacy imports retained for backward compat
