"""TurboQuant compressor — implements the KVCompressor protocol.

TurboQuant (Google ICLR 2026):
  Random orthogonal rotation + Lloyd-Max optimal scalar quantization.
  Asymmetric K/V: K uses more bits than V (e.g., K=3bit, V=2bit).

Each layer gets its own rotation matrix Π (deterministic, seeded by layer_idx).
Compression ratio: (16 × head_dim) / (bits × head_dim/8) ≈ 128/bits ×
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
    """TurboQuant KV cache compressor.

    Pros:
      - Near-optimal quantization (Lloyd-Max codebook)
      - Simple: one rotation matrix per layer
      - Strong theoretical guarantees (Johnson-Lindenstrauss)
    Cons:
      - Rotation matrix costs head_dim² bytes per layer
      - No residual encoding (treats each token independently)
      - Random rotation must be invertible (float32 precision)
    """

    name = "turboquant"
    version = "0.1"

    def __init__(self, num_heads: int, head_dim: int,
                 bits_k: int = 3, bits_v: int = 2,
                 num_layers: int = 1,
                 device: str = "cpu"):
        """
        Args:
            num_heads: KV attention heads
            head_dim:  dimension per head
            bits_k:    bit width for keys (default: 3)
            bits_v:    bit width for values (default: 2)
            num_layers: number of transformer layers
            device:    torch device for rotation matrices
        """
        assert 2 <= bits_k <= 4, f"bits_k={bits_k} must be 2-4"
        assert 2 <= bits_v <= 4, f"bits_v={bits_v} must be 2-4"
        assert head_dim >= 64, "head_dim too small for TurboQuant"

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.bits_k = bits_k
        self.bits_v = bits_v
        self.num_layers = num_layers
        self.device = torch.device(device)

        # Per-layer rotation matrices (generated once, small memory)
        self._Pi_k: dict[int, torch.Tensor] = {}
        self._Pi_v: dict[int, torch.Tensor] = {}

        # Storage
        self._compressed: dict[int, list[tuple]] = {}
        self._total_original = 0
        self._total_compressed = 0

    def _get_Pi(self, layer_idx: int, which: str) -> torch.Tensor:
        """Get or create rotation matrix for a layer."""
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
        """Compress a token's K, V using TurboQuant.

        Independent per-token — no keyframes needed.

        Args:
            K: (n_heads, head_dim) float32
            V: (n_heads, head_dim) float32
            layer_idx: which transformer layer
            token_id: ignored (TurboQuant is per-token)

        Returns:
            (idx_k, norm_k, idx_v, norm_v) packed representation
        """
        self._total_original += (K.numel() * 4 + V.numel() * 4)

        Pi_k = self._get_Pi(layer_idx, "k")
        Pi_v = self._get_Pi(layer_idx, "v")

        # Ensure matrices are on the right device
        if Pi_k.device != K.device:
            Pi_k = Pi_k.to(K.device)
            self._Pi_k[layer_idx] = Pi_k
        if Pi_v.device != V.device:
            Pi_v = Pi_v.to(V.device)
            self._Pi_v[layer_idx] = Pi_v

        idx_k, norm_k = turboquant_quantize(K.float(), Pi_k, self.bits_k)
        idx_v, norm_v = turboquant_quantize(V.float(), Pi_v, self.bits_v)

        # Estimate compressed size
        self._total_compressed += (idx_k.nbytes + norm_k.nbytes +
                                    idx_v.nbytes + norm_v.nbytes)

        result = (idx_k, norm_k, idx_v, norm_v)
        self._compressed.setdefault(layer_idx, []).append(result)
        return result

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress a token's K, V."""
        idx_k, norm_k, idx_v, norm_v = packed
        Pi_k = self._get_Pi(layer_idx, "k")
        Pi_v = self._get_Pi(layer_idx, "v")

        # Ensure matrices are on same device as packed data
        if Pi_k.device != idx_k.device:
            Pi_k = Pi_k.to(idx_k.device)
        if Pi_v.device != idx_v.device:
            Pi_v = Pi_v.to(idx_v.device)

        orig_shape = (self.num_heads, self.head_dim)
        K = turboquant_dequantize(idx_k, norm_k, Pi_k, self.bits_k, orig_shape)
        V = turboquant_dequantize(idx_v, norm_v, Pi_v, self.bits_v, orig_shape)
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
            'bits_k': self.bits_k,
            'bits_v': self.bits_v,
            'backend': 'pytorch',
        }

    def reset(self):
        self._compressed.clear()
        self._Pi_k.clear()
        self._Pi_v.clear()
        self._total_original = 0
        self._total_compressed = 0
