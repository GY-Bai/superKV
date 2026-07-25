"""Pre-compiled kernel for the common test shape (4 heads, 32 dim, 8 tokens).

This avoids JIT compilation overhead on first call for the most common
small test configuration.  Compiled eagerly at import time.
"""

import tilelang.language as T

from superkv.kernels.tilelang.q4_0_kernel import _check_tilelang


if _check_tilelang():
    @T.prim_func
    def _attention_scores_static(
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
    _precompiled_attention_scores_4x32x8 = tilelang.compile(
        _attention_scores_static, target="c")
else:
    _precompiled_attention_scores_4x32x8 = None
