"""Tests for tilelang Q4_0 kernels.

Verifies:
  1. Kernel compiles without errors on current platform (CPU c target).
  2. Output shapes match PyTorch fallback.
  3. Numerical accuracy: tilelang output ≈ PyTorch output.
  4. Roundtrip: dequantize(quantize(x)) ≈ x.
"""

import torch
import pytest


def _has_tilelang():
    from superkv.kernels.tilelang import _check_tilelang
    return _check_tilelang()


requires_tilelang = pytest.mark.skipif(
    not _has_tilelang(), reason="tilelang not installed")


class TestQ4_0KernelCompilation:
    """Verify kernel compiles on current platform."""

    @requires_tilelang
    def test_compile(self):
        """Kernel should compile without errors."""
        from superkv.kernels.tilelang.q4_0_kernel import _compile_q4_0_quantize
        kernel = _compile_q4_0_quantize()
        assert kernel is not None

    @requires_tilelang
    def test_basic_call(self):
        """Call the compiled kernel on a small input."""
        from superkv.kernels.tilelang import q4_0_quantize
        x = torch.randn(2, 32)
        q, d = q4_0_quantize(x)
        assert q.shape == (2, 16)
        assert d.shape == (2, 1)
        assert q.dtype == torch.uint8
        assert d.dtype == torch.float16

    @requires_tilelang
    def test_dequantize_roundtrip(self):
        """Roundtrip: dequantize(quantize(x)) ≈ x."""
        from superkv.kernels.tilelang import q4_0_quantize, q4_0_dequantize
        x = torch.randn(4, 64, dtype=torch.float32)
        q, d = q4_0_quantize(x)
        x_recon = q4_0_dequantize(q, d, x.shape)
        mse = torch.nn.functional.mse_loss(x_recon, x).item()
        assert mse < 0.1, f"MSE {mse:.4f} too high"


class TestQ4_0KernelVsPyTorch:
    """Verify tilelang kernel output matches PyTorch fallback."""

    @requires_tilelang
    def test_quantize_vs_fallback(self):
        from superkv.kernels.tilelang import q4_0_quantize
        from superkv.kernels.tilelang import _pytorch_fallback_quantize
        x = torch.randn(3, 64)
        qt, dt = q4_0_quantize(x)
        qp, dp = _pytorch_fallback_quantize(x)
        # Same dtype, same shape
        assert qt.shape == qp.shape
        assert dt.shape == dp.shape
        assert qt.dtype == qp.dtype

    @requires_tilelang
    def test_dequant_vs_fallback(self):
        from superkv.kernels.tilelang import q4_0_quantize, q4_0_dequantize
        from superkv.kernels.tilelang import _pytorch_fallback_dequantize
        x = torch.randn(3, 64)
        q, d = q4_0_quantize(x)
        xt = q4_0_dequantize(q, d, x.shape)
        xp = _pytorch_fallback_dequantize(q, d, x.shape)
        # Should produce similar results (same packed data)
        assert torch.allclose(xt, xp, atol=0.01), "tilelang vs PyTorch mismatch"

    @requires_tilelang
    def test_large_tensor(self):
        from superkv.kernels.tilelang import q4_0_quantize, q4_0_dequantize
        x = torch.randn(32, 128, dtype=torch.float32)  # 4096 values
        q, d = q4_0_quantize(x)
        x_recon = q4_0_dequantize(q, d, x.shape)
        assert x_recon.shape == x.shape


class TestQ4_0KernelEdgeCases:
    """Edge case handling."""

    @requires_tilelang
    def test_zeros(self):
        from superkv.kernels.tilelang import q4_0_quantize, q4_0_dequantize
        x = torch.zeros(1, 32)
        q, d = q4_0_quantize(x)
        x_recon = q4_0_dequantize(q, d, x.shape)
        # Zero input → near-zero output
        assert torch.allclose(x_recon, x, atol=1e-3)

    @requires_tilelang
    def test_constant(self):
        from superkv.kernels.tilelang import q4_0_quantize, q4_0_dequantize
        x = torch.full((2, 32), 3.0)
        q, d = q4_0_quantize(x)
        x_recon = q4_0_dequantize(q, d, x.shape)
        mse = torch.nn.functional.mse_loss(x_recon, x).item()
        assert mse < 0.1, f"MSE {mse:.4f} too high for constant input"

    @requires_tilelang
    def test_multi_shape(self):
        """Various head_dim shapes (must be multiples of 32)."""
        from superkv.kernels.tilelang import q4_0_quantize
        for hd in [32, 64, 128, 256]:
            x = torch.randn(4, hd)
            q, d = q4_0_quantize(x)
            assert q.shape == (4, hd // 2), f"hd={hd}"
            assert d.shape == (4, hd // 32)
