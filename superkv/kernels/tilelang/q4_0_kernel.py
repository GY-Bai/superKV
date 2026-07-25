"""TileLang kernels for KV cache compression.

Q4_0 quantize/dequantize using T.Kernel + T.alloc_fragment pattern
(compatible with CPU c and CUDA targets). Lazy-compiled via @tilelang.jit.

Reference: tilelang examples/cast/example_per_token_cast_to_fp8.py
"""

from __future__ import annotations

import torch
import math

from superkv.engine.platform import get_tilelang_target

_TARGET = get_tilelang_target()
Q4_0_BLOCK = 32


# ═══════════════════════════════════════════════════════════════════════
# Q4_0 Quantize
# ═══════════════════════════════════════════════════════════════════════

def _compile_q4_0_quantize():
    """Lazily compile Q4_0 quantize using T.Kernel pattern.

    Grid: n_blocks × 1, each block processes one Q4_0 group of 32 elements.
    """
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target=_TARGET)
    def _q4_0_quantize(x, n_blocks):
        NB = T.const("n_blocks")
        x: T.Tensor((NB, 32), "float32")
        q_out = T.empty((NB, 16), "uint8")
        d_out = T.empty((NB,), "float16")

        with T.Kernel(NB, threads=32) as bx:
            # Load 32 elements into local fragment
            frag = T.alloc_fragment((32,), "float32")
            T.copy(x[bx, :], frag)

            # Find max abs via reduce
            amax = T.alloc_fragment((1,), "float32")
            T.reduce_absmax(frag, amax, dim=0)
            d = amax[0] / T.float32(7.0)
            d_out[bx] = T.cast(d, "float16")

            # Quantize: normalize, round, clamp, pack
            for j in T.serial(16):
                v0 = T.round(frag[j * 2] / d) + T.float32(8.0)
                v1 = T.round(frag[j * 2 + 1] / d) + T.float32(8.0)
                b0 = T.cast(
                    T.clamp(v0, T.float32(1.0), T.float32(15.0)), "uint8")
                b1 = T.cast(
                    T.clamp(v1, T.float32(1.0), T.float32(15.0)), "uint8")
                q_out[bx, j] = b0 | (b1 << T.uint8(4))

        return q_out, d_out

    return _q4_0_quantize


def _compile_q4_0_dequantize():
    """Lazily compile Q4_0 dequantize using T.Kernel pattern."""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target=_TARGET)
    def _q4_0_dequantize(q_in, d_in, n_blocks):
        NB = T.const("n_blocks")
        q_in: T.Tensor((NB, 16), "uint8")
        d_in: T.Tensor((NB,), "float16")
        x_out = T.empty((NB, 32), "float32")

        with T.Kernel(NB, threads=32) as bx:
            d = T.cast(d_in[bx], "float32")

            for j in T.serial(16):
                lo = T.cast(q_in[bx, j] & T.uint8(0x0F), "float32")
                hi = T.cast(q_in[bx, j] >> T.uint8(4), "float32")
                x_out[bx, j * 2] = (lo - T.float32(8.0)) * d
                x_out[bx, j * 2 + 1] = (hi - T.float32(8.0)) * d

        return x_out

    return _q4_0_dequantize


# ── Lazy singletons ─────────────────────────────────────────────────

_q4_0_quantize_kernel = None
_q4_0_dequantize_kernel = None
_tilelang_available = None


def _check_tilelang():
    global _tilelang_available
    if _tilelang_available is not None:
        return _tilelang_available
    try:
        import tilelang.language as T  # noqa: F401
        _tilelang_available = True
    except Exception:
        _tilelang_available = False
    return _tilelang_available


def q4_0_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    global _q4_0_quantize_kernel
    if not _check_tilelang():
        return _pytorch_fallback_quantize(x)

    orig_shape = x.shape
    n_blocks = x.numel() // Q4_0_BLOCK
    x_flat = x.reshape(n_blocks, Q4_0_BLOCK).float()

    try:
        if _q4_0_quantize_kernel is None:
            _q4_0_quantize_kernel = _compile_q4_0_quantize()
        q, d = _q4_0_quantize_kernel(x_flat, n_blocks=n_blocks)
    except Exception:
        return _pytorch_fallback_quantize(x)

    q_shape = orig_shape[:-1] + (Q4_0_BLOCK // 2,)
    d_shape = orig_shape[:-1] + (1,)
    return q.reshape(q_shape), d.reshape(d_shape)


def q4_0_dequantize(q: torch.Tensor, d: torch.Tensor,
                    orig_shape: tuple) -> torch.Tensor:
    global _q4_0_dequantize_kernel
    if not _check_tilelang():
        return _pytorch_fallback_dequantize(q, d, orig_shape)

    n_blocks = d.numel()
    q_flat = q.reshape(n_blocks, Q4_0_BLOCK // 2)
    d_flat = d.reshape(n_blocks)

    try:
        if _q4_0_dequantize_kernel is None:
            _q4_0_dequantize_kernel = _compile_q4_0_dequantize()
        x_out = _q4_0_dequantize_kernel(q_flat, d_flat, n_blocks=n_blocks)
    except Exception:
        return _pytorch_fallback_dequantize(q, d, orig_shape)

    return x_out.reshape(orig_shape)


def _pytorch_fallback_quantize(x):
    from superkv.algorithms.deltakv.core import quantize_q4_0
    return quantize_q4_0(x)


def _pytorch_fallback_dequantize(q, d, orig_shape):
    from superkv.algorithms.deltakv.core import dequantize_q4_0
    return dequantize_q4_0(q, d, orig_shape)
