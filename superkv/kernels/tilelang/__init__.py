"""TileLang-accelerated kernels for KV compression.

Kernels are lazily compiled via @tilelang.jit. Pre-compiled for
common shapes. Falls back to PyTorch on compile errors.
"""

from superkv.kernels.tilelang.q4_0_kernel import (
    q4_0_quantize,
    q4_0_dequantize,
    _check_tilelang,
)
from superkv.kernels.tilelang.attention_kernel import (
    sparse_attention_scores,
    sparse_attention,
)

__all__ = [
    "q4_0_quantize",
    "q4_0_dequantize",
    "sparse_attention_scores",
    "sparse_attention",
    "_check_tilelang",
]
