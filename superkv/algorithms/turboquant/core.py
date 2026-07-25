"""TurboQuant core — random rotation + optimal scalar quantization.

TurboQuant (Google ICLR 2026):
  1. Normalize x to unit sphere
  2. Apply random orthogonal rotation Π (QR of Gaussian matrix)
  3. Quantize each coordinate with Lloyd-Max optimal codebook
  4. Dequantize: lookup centroids → rotate back → rescale

This module provides the low-level quantize/dequantize operations.
The TurboQuantCompressor in compressor.py wraps these into the
KVCompressor protocol with per-layer rotation matrices.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Rotation matrix generation
# ═══════════════════════════════════════════════════════════════════════

def generate_rotation_matrix(
    dim: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 42,
) -> torch.Tensor:
    """Generate random orthogonal matrix Π ∈ R^{d×d} via QR decomposition.

    For head_dim=128: 128×128 × 4 bytes = 64 KB, negligible overhead.
    The matrix is shared across all heads in a layer.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    G = torch.randn(dim, dim, generator=rng, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    # Fix sign so det = +1 (proper rotation)
    diag_sign = torch.sign(torch.diag(R))
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device=device, dtype=dtype)


# ═══════════════════════════════════════════════════════════════════════
# Lloyd-Max codebook (optimal for Gaussian source)
# ═══════════════════════════════════════════════════════════════════════

def _lloyd_max_codebook(bits: int, n_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Lloyd-Max optimal quantizer for standard Gaussian N(0,1).

    Uses the iterative Lloyd-Max algorithm:
      1. Start with uniform centroids in [-3, 3]
      2. Iterate: boundaries = midpoint of adjacent centroids
      3.          centroids = conditional expectation E[x | x in interval]

    Args:
        bits: number of bits (2^bits quantization levels)
        n_iters: Lloyd-Max iterations
    Returns:
        centroids: (2^bits,)  float32
        boundaries: (2^bits + 1,) float32  (decision boundaries, ±∞ at ends)
    """
    n_levels = 2 ** bits
    # Initialize uniformly
    centroids = torch.linspace(-3.0, 3.0, n_levels, dtype=torch.float64)

    # Standard normal PDF and CDF
    sqrt2 = math.sqrt(2.0)
    sqrt2pi = math.sqrt(2.0 * math.pi)

    def normal_pdf(x):
        return torch.exp(-0.5 * x ** 2) / sqrt2pi

    def normal_cdf(x):
        return 0.5 * (1.0 + torch.erf(x / sqrt2))

    for _ in range(n_iters):
        # Boundaries: midpoint of adjacent centroids
        boundaries = torch.zeros(n_levels + 1, dtype=torch.float64)
        boundaries[0] = -float('inf')
        boundaries[-1] = float('inf')
        boundaries[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])

        # Update centroids: E[x | x in (b_l, b_{l+1})]
        for i in range(n_levels):
            a = boundaries[i]
            b = boundaries[i + 1]
            if a == -float('inf') and b == float('inf'):
                centroids[i] = 0.0
            elif a == -float('inf'):
                pdf_b = normal_pdf(b)
                cdf_b = normal_cdf(b)
                centroids[i] = -pdf_b / cdf_b
            elif b == float('inf'):
                pdf_a = normal_pdf(a)
                cdf_a = normal_cdf(a)
                centroids[i] = pdf_a / (1.0 - cdf_a)
            else:
                pdf_a = normal_pdf(a)
                pdf_b = normal_pdf(b)
                cdf_a = normal_cdf(a)
                cdf_b = normal_cdf(b)
                centroids[i] = (pdf_a - pdf_b) / (cdf_b - cdf_a)

    return centroids.float(), boundaries.float()


# Precompute common bit-width codebooks
_CODEBOOK_CACHE: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def get_codebook(bits: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Get Lloyd-Max codebook for given bit width (cached)."""
    if bits not in _CODEBOOK_CACHE:
        _CODEBOOK_CACHE[bits] = _lloyd_max_codebook(bits)
    return _CODEBOOK_CACHE[bits]


# ═══════════════════════════════════════════════════════════════════════
# Packing utilities
# ═══════════════════════════════════════════════════════════════════════

def _pack_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack integer indices into uint8 bytes.

    bits=1: 8 per byte
    bits=2: 4 per byte
    bits=3,4: 2 per byte (padded to 4-bit)
    """
    d = indices.shape[-1]
    batch_shape = indices.shape[:-1]

    vals_per_byte = {1: 8, 2: 4, 3: 2, 4: 2}.get(bits, 1)
    pack_bits = {3: 4}.get(bits, bits)  # 3-bit → pad to 4-bit

    if vals_per_byte == 1:
        return indices.to(torch.uint8)

    # Pad to multiple of vals_per_byte
    padded_d = ((d + vals_per_byte - 1) // vals_per_byte) * vals_per_byte
    if padded_d > d:
        indices = F.pad(indices.to(torch.uint8), (0, padded_d - d), value=0)

    reshaped = indices.to(torch.uint8).reshape(
        *batch_shape, -1, vals_per_byte)
    shifts = (torch.arange(vals_per_byte, device=indices.device, dtype=torch.uint8)
              * pack_bits)
    packed = (reshaped << shifts).sum(dim=-1, dtype=torch.uint8)
    return packed


def _unpack_indices(packed: torch.Tensor, bits: int, d: int) -> torch.Tensor:
    """Unpack bit-packed indices back to integer tensor."""
    vals_per_byte = {1: 8, 2: 4, 3: 2, 4: 2}.get(bits, 1)
    pack_bits = {3: 4}.get(bits, bits)

    if vals_per_byte == 1:
        return packed.long()

    batch_shape = packed.shape[:-1]
    mask = (1 << pack_bits) - 1
    shifts = (torch.arange(vals_per_byte, device=packed.device, dtype=torch.uint8)
              * pack_bits)
    unpacked = ((packed.unsqueeze(-1) >> shifts) & mask)
    unpacked = unpacked.reshape(*batch_shape, -1)
    return unpacked[..., :d].long()


# ═══════════════════════════════════════════════════════════════════════
# Public quantize / dequantize
# ═══════════════════════════════════════════════════════════════════════

def turboquant_quantize(
    x: torch.Tensor,
    Pi: torch.Tensor,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TurboQuant quantize one tensor (K or V) for a single layer.

    Args:
        x:    (n_heads, head_dim)  float32
        Pi:   (head_dim, head_dim) rotation matrix
        bits: bit width (2, 3, or 4)

    Returns:
        indices:  bit-packed uint8, shape depending on bits and head_dim
        norms:    (n_heads,) float32 — per-head L2 norm for rescaling
    """
    centroids, boundaries = get_codebook(bits)
    boundaries = boundaries.to(x.device)
    centroids = centroids.to(x.device)

    # Per-head norms
    norms = x.norm(dim=-1, keepdim=False)  # (n_heads,)

    # Normalize to unit sphere
    x_unit = x / (norms.unsqueeze(-1) + 1e-10)

    # Apply random rotation
    y = torch.matmul(x_unit.float(), Pi.T)  # (n_heads, head_dim)

    # Scale to N(0,1): after unit-sphere projection, each coordinate
    # has variance ≈ 1/head_dim. Multiply by sqrt(head_dim) to match
    # the standard normal distribution the Lloyd-Max codebook expects.
    d = x.shape[-1]
    y = y * math.sqrt(d)

    # Quantize: searchsorted into boundary bins
    # boundaries[1:-1] are the thresholds between centroids
    decision = boundaries[1:-1].contiguous()
    indices = torch.searchsorted(decision, y.contiguous())

    # Pack
    packed_indices = _pack_indices(indices, bits)

    return packed_indices, norms


def turboquant_dequantize(
    indices: torch.Tensor,
    norms: torch.Tensor,
    Pi: torch.Tensor,
    bits: int,
    orig_shape: tuple,
) -> torch.Tensor:
    """TurboQuant dequantize.

    Args:
        indices:   bit-packed uint8
        norms:     (n_heads,) float32
        Pi:        (head_dim, head_dim) rotation matrix
        bits:      bit width
        orig_shape:(n_heads, head_dim)

    Returns:
        x_recon:   (n_heads, head_dim) float32
    """
    centroids, _ = get_codebook(bits)
    centroids = centroids.to(indices.device)

    n_heads, head_dim = orig_shape

    # Unpack
    idx_flat = _unpack_indices(indices, bits, head_dim)  # (n_heads, head_dim)

    # Lookup centroids
    y_hat = centroids[idx_flat]  # (n_heads, head_dim)

    # Undo the sqrt(d) scaling applied during quantization
    d = orig_shape[-1]
    y_hat = y_hat / math.sqrt(d)

    # Rotate back
    x_hat = torch.matmul(y_hat.float(), Pi)  # (n_heads, head_dim)

    # Rescale
    x_hat = x_hat * norms.unsqueeze(-1)

    return x_hat


def turboquant_roundtrip_mse(
    x: torch.Tensor, Pi: torch.Tensor, bits: int
) -> float:
    """Convenience: quantize → dequantize → MSE."""
    idx, norms = turboquant_quantize(x, Pi, bits)
    x_recon = turboquant_dequantize(idx, norms, Pi, bits, x.shape)
    return F.mse_loss(x_recon, x).item()
