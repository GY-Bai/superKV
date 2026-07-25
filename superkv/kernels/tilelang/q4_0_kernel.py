"""TileLang kernels for KV cache compression.

Kernels are lazily compiled via @tilelang.jit and cached per shape.
All kernels have identical Python fallback interfaces so they can be
swapped in without API changes.

Target dispatch:
  CPU   → tilelang target "c"
  GPU   → tilelang target "cuda"
  Metal → tilelang target "c" (fallback until PR#2767 merged)
"""

from __future__ import annotations

import torch
import math

from superkv.engine.platform import get_tilelang_target, detect_platform

_TARGET = get_tilelang_target()
Q4_0_BLOCK = 32  # elements per Q4_0 block


# ═══════════════════════════════════════════════════════════════════════
# Q4_0 Quantize — tilelang kernel  (Gate 9)
# ═══════════════════════════════════════════════════════════════════════

def _compile_q4_0_quantize():
    """Lazily compile the tilelang Q4_0 quantize kernel.

    Kernel:  quantize n × 32 float32 blocks → packed uint8 + float16 scales.
    For each block of 32 elements:  d = max_abs / 7,  each elem → 4-bit.
    """
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target=_TARGET)
    def _q4_0_quantize(x, n_blocks):
        """Q4_0 quantize kernel.

        Args:
            x:      (n_blocks, 32) float32 input
            n_blocks: number of Q4_0 blocks
        Returns:
            q_out:  (n_blocks, 16) uint8 packed  (2×int4 per byte)
            d_out:  (n_blocks,)   float16 scales
        """
        NB = T.const("n_blocks")
        x: T.Tensor((NB, 32), "float32")
        q_out = T.empty((NB, 16), "uint8")
        d_out = T.empty((NB,), "float16")

        for i in T.Parallel(NB):
            # Phase 1: find max abs value in this block
            amax = T.alloc_local("float32")
            T.set_local(amax, T.float32(0.0))
            for j in T.serial(32):
                T.set_local(amax,
                    T.max(T.get_local(amax), T.abs(x[i, j])))

            d = T.get_local(amax) / T.float32(7.0)
            d_out[i] = T.cast(d, "float16")

            # Phase 2: quantize + pack (2 × int4 per byte)
            for j in T.serial(16):
                v0 = T.round(x[i, j * 2] / d) + T.float32(8.0)
                v1 = T.round(x[i, j * 2 + 1] / d) + T.float32(8.0)
                b0 = T.cast(T.clamp(v0, T.float32(1.0), T.float32(15.0)), "uint8")
                b1 = T.cast(T.clamp(v1, T.float32(1.0), T.float32(15.0)), "uint8")
                q_out[i, j] = b0 | (b1 << T.uint8(4))

        return q_out, d_out

    return _q4_0_quantize


def _compile_q4_0_dequantize():
    """Lazily compile the tilelang Q4_0 dequantize kernel."""
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target=_TARGET)
    def _q4_0_dequantize(q_in, d_in, n_blocks):
        """Q4_0 dequantize kernel.

        Args:
            q_in:     (n_blocks, 16) uint8 packed
            d_in:     (n_blocks,)   float16 scales
            n_blocks: block count
        Returns:
            x_out:    (n_blocks, 32) float32
        """
        NB = T.const("n_blocks")
        q_in: T.Tensor((NB, 16), "uint8")
        d_in: T.Tensor((NB,), "float16")
        x_out = T.empty((NB, 32), "float32")

        for i in T.Parallel(NB):
            d = T.cast(d_in[i], "float32")
            for j in T.serial(16):
                lo = T.cast(q_in[i, j] & T.uint8(0x0F), "float32")
                hi = T.cast(q_in[i, j] >> T.uint8(4), "float32")
                x_out[i, j * 2]     = (lo - T.float32(8.0)) * d
                x_out[i, j * 2 + 1] = (hi - T.float32(8.0)) * d

        return x_out

    return _q4_0_dequantize


# ── Lazy module-level singletons ─────────────────────────────────────

_q4_0_quantize_kernel = None
_q4_0_dequantize_kernel = None
_tilelang_available = None


def _check_tilelang():
    """Check if tilelang is available and can compile kernels."""
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
    """Q4_0 quantize using tilelang kernel with PyTorch fallback.

    Args:
        x: float32, shape (..., 32) — last dim must be divisible by 32.
    Returns:
        q: uint8 packed, shape (..., 16)
        d: float16 scales, shape (..., 1)
    """
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

    # Reshape to match input pattern
    q_shape = orig_shape[:-1] + (Q4_0_BLOCK // 2,)
    d_shape = orig_shape[:-1] + (1,)
    return q.reshape(q_shape), d.reshape(d_shape)


def q4_0_dequantize(q: torch.Tensor, d: torch.Tensor,
                    orig_shape: tuple) -> torch.Tensor:
    """Q4_0 dequantize using tilelang kernel with PyTorch fallback."""
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


# ── PyTorch fallbacks (same code as algorithms/deltakv/core.py) ──────

def _pytorch_fallback_quantize(x: torch.Tensor
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch Q4_0 quantize."""
    from superkv.algorithms.deltakv.core import quantize_q4_0
    return quantize_q4_0(x)


def _pytorch_fallback_dequantize(q: torch.Tensor, d: torch.Tensor,
                                  orig_shape: tuple) -> torch.Tensor:
    """Pure PyTorch Q4_0 dequantize."""
    from superkv.algorithms.deltakv.core import dequantize_q4_0
    return dequantize_q4_0(q, d, orig_shape)
