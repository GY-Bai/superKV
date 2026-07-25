"""TurboQuant: Random rotation + optimal scalar quantization.

Google ICLR 2026. Compresses KV cache using:
  1. Random orthogonal rotation (QR decomposition)
  2. Lloyd-Max optimal quantization codebook
  3. Asymmetric bit-widths (K=3bit, V=2bit by default)

Reference: arXiv:2504.19874, vLLM PR #38479
"""

from superkv.algorithms.turboquant.core import (
    generate_rotation_matrix,
    turboquant_quantize,
    turboquant_dequantize,
    turboquant_roundtrip_mse,
    get_codebook,
    _lloyd_max_codebook,
)
from superkv.algorithms.turboquant.compressor import TurboQuantCompressor

__all__ = [
    "generate_rotation_matrix",
    "turboquant_quantize",
    "turboquant_dequantize",
    "turboquant_roundtrip_mse",
    "get_codebook",
    "_lloyd_max_codebook",
    "TurboQuantCompressor",
]
