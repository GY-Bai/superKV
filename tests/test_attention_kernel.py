"""Tests for tilelang sparse attention kernel.

Verifies:
  1. Kernel compiles for various shapes.
  2. Output matches PyTorch einsum reference.
  3. Full sparse_attention pipeline produces correct dimensions.
  4. Performance baseline (tilelang vs PyTorch).
"""

import torch
import pytest
import math
import time


def _has_tilelang():
    from superkv.kernels.tilelang import _check_tilelang
    return _check_tilelang()


requires_tilelang = pytest.mark.skipif(
    not _has_tilelang(), reason="tilelang not installed")


# ═══════════════════════════════════════════════════════════════════════
# Reference implementation
# ═══════════════════════════════════════════════════════════════════════

def _ref_scores(Q, K_sparse):
    """Reference: torch.einsum Q@K^T."""
    return torch.einsum("hd,khd->hk", Q.float(), K_sparse.float())


def _ref_attention(Q, K_sparse, V_sparse, scale=None):
    """Reference full sparse attention."""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = _ref_scores(Q, K_sparse) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum("hi,ihj->hj", weights, V_sparse)


# ═══════════════════════════════════════════════════════════════════════
# Tests: compilation
# ═══════════════════════════════════════════════════════════════════════

class TestSparseAttentionCompile:
    @requires_tilelang
    def test_compile(self):
        """Kernel should compile without errors."""
        from superkv.kernels.tilelang.attention_kernel import \
            _compile_attention_scores
        kernel = _compile_attention_scores()
        assert kernel is not None

    @requires_tilelang
    def test_precompiled(self):
        """Pre-compiled kernel works on CPU, skips on GPU."""
        from superkv.kernels.tilelang._precompiled import (
            get_precompiled_attention, has_precompiled)
        from superkv.engine.platform import detect_platform
        if not has_precompiled():
            if detect_platform() != 'cpu':
                pytest.skip("precompiled kernel is CPU-only")
            pytest.skip("tilelang not available")
        kernel = get_precompiled_attention()
        assert kernel is not None
        Q = torch.randn(4, 32)
        K = torch.randn(8, 4, 32)
        scores = torch.zeros(4, 8, dtype=torch.float32)
        kernel(Q, K, scores)
        assert scores.shape == (4, 8)

    @requires_tilelang
    def test_basic_scores_shape(self):
        from superkv.kernels.tilelang import sparse_attention_scores
        Q = torch.randn(4, 32)
        K = torch.randn(8, 4, 32)
        scores = sparse_attention_scores(Q, K)
        assert scores.shape == (4, 8)


# ═══════════════════════════════════════════════════════════════════════
# Tests: correctness vs PyTorch reference
# ═══════════════════════════════════════════════════════════════════════

class TestSparseAttentionCorrectness:
    @requires_tilelang
    def test_scores_vs_reference_4x32x8(self):
        from superkv.kernels.tilelang import sparse_attention_scores
        Q = torch.randn(4, 32)
        K = torch.randn(8, 4, 32)
        scores_tl = sparse_attention_scores(Q, K)
        scores_ref = _ref_scores(Q, K)
        assert torch.allclose(scores_tl, scores_ref, atol=5e-5,
                              rtol=1e-4), "tilelang vs einsum mismatch"

    @requires_tilelang
    def test_scores_vs_reference_8x128_16(self):
        """Real model shape: 8 KV heads × 128 dim, 16 selected tokens."""
        from superkv.kernels.tilelang import sparse_attention_scores
        Q = torch.randn(8, 128)
        K = torch.randn(16, 8, 128)
        scores_tl = sparse_attention_scores(Q, K)
        scores_ref = _ref_scores(Q, K)
        assert torch.allclose(scores_tl, scores_ref, atol=1e-4,
                              rtol=1e-4), "8×128 mismatch"

    @requires_tilelang
    def test_full_attention_output(self):
        from superkv.kernels.tilelang import sparse_attention
        Q = torch.randn(4, 32)
        K = torch.randn(6, 4, 32)
        V = torch.randn(6, 4, 32)
        out = sparse_attention(Q, K, V)
        assert out.shape == (4, 32)
        assert not torch.isnan(out).any()

    @requires_tilelang
    def test_full_attention_vs_reference(self):
        from superkv.kernels.tilelang import sparse_attention
        Q = torch.randn(4, 32)
        K = torch.randn(8, 4, 32)
        V = torch.randn(8, 4, 32)
        out_tl = sparse_attention(Q, K, V)
        out_ref = _ref_attention(Q, K, V)
        assert torch.allclose(out_tl, out_ref, atol=1e-4,
                              rtol=1e-4), "full attention mismatch"


# ═══════════════════════════════════════════════════════════════════════
# Tests: various shapes (dynamic JIT)
# ═══════════════════════════════════════════════════════════════════════

class TestSparseAttentionShapes:
    @requires_tilelang
    def test_shape_4x64_12(self):
        from superkv.kernels.tilelang import sparse_attention_scores
        Q = torch.randn(4, 64)
        K = torch.randn(12, 4, 64)
        scores = sparse_attention_scores(Q, K)
        assert scores.shape == (4, 12)
        ref = _ref_scores(Q, K)
        assert torch.allclose(scores, ref, atol=1e-4, rtol=1e-4)

    @requires_tilelang
    def test_shape_16x128_32(self):
        from superkv.kernels.tilelang import sparse_attention_scores
        Q = torch.randn(16, 128)
        K = torch.randn(32, 16, 128)
        scores = sparse_attention_scores(Q, K)
        assert scores.shape == (16, 32)
        ref = _ref_scores(Q, K)
        assert torch.allclose(scores, ref, atol=1e-4, rtol=1e-4)

    @requires_tilelang
    def test_shape_8x256_4(self):
        from superkv.kernels.tilelang import sparse_attention_scores
        Q = torch.randn(8, 256)
        K = torch.randn(4, 8, 256)
        scores = sparse_attention_scores(Q, K)
        assert scores.shape == (8, 4)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark
# ═══════════════════════════════════════════════════════════════════════

@requires_tilelang
class TestSparseAttentionBenchmark:
    def test_completes_without_error(self):
        """Benchmark-like: run tilelang attention scores on realistic input."""
        from superkv.kernels.tilelang import sparse_attention_scores

        Q = torch.randn(8, 128)
        K = torch.randn(64, 8, 128)
        result = sparse_attention_scores(Q, K)
        assert result.shape == (8, 64)

    def test_tilelang_vs_pytorch_speed(self):
        """Quick speed comparison.

        Note: On CPU, tilelang-compiled C code wraps each operation
        in a function call, making it slower than PyTorch's optimized
        BLAS for small tensors. The value is in cross-platform
        portability (GPU/Metal/Ascend) and correctness.
        """
        from superkv.kernels.tilelang import sparse_attention_scores

        Q = torch.randn(8, 128)
        K = torch.randn(64, 8, 128)

        # Warmup
        for _ in range(3):
            sparse_attention_scores(Q, K)
            _ref_scores(Q, K)

        n_iter = 200
        t0 = time.perf_counter()
        for _ in range(n_iter):
            sparse_attention_scores(Q, K)
        t_tl = time.perf_counter() - t0

        t0 = time.perf_counter()
        for _ in range(n_iter):
            _ref_scores(Q, K)
        t_ref = time.perf_counter() - t0

        print(f"\n  tilelang: {t_tl*1000/n_iter:.3f}ms | "
              f"PyTorch: {t_ref*1000/n_iter:.3f}ms | "
              f"ratio: {t_ref/t_tl:.1f}x")
        # On CPU, tilelang may be slower (JIT overhead per small op).
        # On GPU/CUDA, tilelang should match or beat PyTorch.
        # This test just verifies tilelang completes without error.
        assert t_tl > 0 and t_ref > 0  # both completed
