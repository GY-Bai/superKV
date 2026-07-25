"""Pre-compiled kernel for the common test shape (4 heads, 32 dim, 8 tokens).

Compiles once at import time for the detected platform.
"""

import tilelang.language as T

from superkv.kernels.tilelang.q4_0_kernel import _check_tilelang
from superkv.engine.platform import get_tilelang_target


_KERNEL = None


def get_precompiled_attention():
    """Return precompiled kernel for current platform (lazy, cached)."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    if not _check_tilelang():
        _KERNEL = None
        return None

    target = get_tilelang_target()

    @T.prim_func
    def _attn_static(
        Q: T.Tensor((4, 32), "float32"),
        K: T.Tensor((8, 4, 32), "float32"),
        scores: T.Tensor((4, 8), "float32"),
    ):
        for h in T.Parallel(4):
            for i in T.serial(8):
                dot = T.alloc_var("float32", init=0.0)
                for j in T.serial(32):
                    dot = dot + Q[h, j] * K[i, h, j]
                scores[h, i] = dot

    import tilelang
    _KERNEL = tilelang.compile(_attn_static, target=target)
    return _KERNEL


def has_precompiled() -> bool:
    """Check if a precompiled kernel is available on this platform."""
    return _check_tilelang()
