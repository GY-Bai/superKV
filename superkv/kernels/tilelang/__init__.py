"""TileLang-accelerated kernels for KV compression.

Lazy compiled via @tilelang.jit. Falls back to PyTorch on error.
"""

from superkv.kernels.tilelang.q4_0_kernel import (
    q4_0_quantize,
    q4_0_dequantize,
    _check_tilelang,
)
from superkv.kernels.tilelang.q4_0_kernel import (
    _pytorch_fallback_quantize,
    _pytorch_fallback_dequantize,
)

__all__ = [
    "q4_0_quantize",
    "q4_0_dequantize",
    "_check_tilelang",
]
