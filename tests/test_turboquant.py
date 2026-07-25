"""Tests for TurboQuant algorithm.

Verifies:
  1. Lloyd-Max codebook: sum of centroids ≈ 0, monotonic
  2. Rotation matrix: orthonormal columns (Π @ Π^T ≈ I)
  3. Quantize/Dequantize roundtrip: MSE reasonable
  4. Compressor protocol: compress, decompress, memory report
  5. Asymmetric K/V: K 3-bit has lower MSE than K 2-bit
"""

import torch
import pytest

from superkv.engine.registry import list_algorithms, create_compressor


# ═══════════════════════════════════════════════════════════════════════
# Codebook tests
# ═══════════════════════════════════════════════════════════════════════

class TestLloydMaxCodebook:
    def test_codebook_2bit(self):
        from superkv.algorithms.turboquant import get_codebook
        centroids, boundaries = get_codebook(2)
        assert centroids.shape == (4,)
        assert boundaries.shape == (5,)
        assert boundaries[0] == -float('inf')
        assert boundaries[-1] == float('inf')

    def test_codebook_3bit(self):
        from superkv.algorithms.turboquant import get_codebook
        centroids, boundaries = get_codebook(3)
        assert centroids.shape == (8,)

    def test_codebook_4bit(self):
        from superkv.algorithms.turboquant import get_codebook
        centroids, boundaries = get_codebook(4)
        assert centroids.shape == (16,)

    def test_centroids_symmetric(self):
        """Lloyd-Max for Gaussian should be symmetric around 0."""
        from superkv.algorithms.turboquant import get_codebook
        for bits in [2, 3, 4]:
            centroids, _ = get_codebook(bits)
            assert abs(centroids.sum().item()) < 0.1, \
                f"{bits}-bit centroids not centered"

    def test_monotonic(self):
        """Centroids must be strictly increasing."""
        from superkv.algorithms.turboquant import get_codebook
        for bits in [2, 3, 4]:
            centroids, _ = get_codebook(bits)
            assert (centroids[1:] > centroids[:-1]).all(), \
                f"{bits}-bit not monotonic"


# ═══════════════════════════════════════════════════════════════════════
# Rotation matrix tests
# ═══════════════════════════════════════════════════════════════════════

class TestRotationMatrix:
    def test_orthogonal(self):
        from superkv.algorithms.turboquant import generate_rotation_matrix
        Pi = generate_rotation_matrix(64)
        # Π @ Π^T should ≈ I
        identity = Pi @ Pi.T
        expected = torch.eye(64, dtype=torch.float32)
        assert torch.allclose(identity, expected, atol=1e-5)

    def test_deterministic(self):
        from superkv.algorithms.turboquant import generate_rotation_matrix
        Pi1 = generate_rotation_matrix(64, seed=42)
        Pi2 = generate_rotation_matrix(64, seed=42)
        assert torch.equal(Pi1, Pi2)

    def test_different_seeds_different(self):
        from superkv.algorithms.turboquant import generate_rotation_matrix
        Pi1 = generate_rotation_matrix(64, seed=42)
        Pi2 = generate_rotation_matrix(64, seed=43)
        assert not torch.equal(Pi1, Pi2)


# ═══════════════════════════════════════════════════════════════════════
# Quantize / Dequantize tests
# ═══════════════════════════════════════════════════════════════════════

class TestTurboQuantRoundtrip:
    def test_roundtrip_3bit(self):
        from superkv.algorithms.turboquant import (
            generate_rotation_matrix,
            turboquant_quantize, turboquant_dequantize,
        )
        Pi = generate_rotation_matrix(128)
        x = torch.randn(8, 128)  # 8 KV heads, 128 dim
        idx, norms = turboquant_quantize(x, Pi, 3)
        x_recon = turboquant_dequantize(idx, norms, Pi, 3, x.shape)
        mse = torch.nn.functional.mse_loss(x_recon, x).item()
        # 3-bit should be decent
        assert mse < 0.5, f"MSE {mse:.4f} too high for 3-bit"

    def test_roundtrip_4bit(self):
        from superkv.algorithms.turboquant import (
            generate_rotation_matrix,
            turboquant_quantize, turboquant_dequantize,
        )
        Pi = generate_rotation_matrix(128)
        x = torch.randn(8, 128)
        idx, norms = turboquant_quantize(x, Pi, 4)
        x_recon = turboquant_dequantize(idx, norms, Pi, 4, x.shape)
        mse = torch.nn.functional.mse_loss(x_recon, x).item()
        assert mse < 0.2, f"MSE {mse:.4f} too high for 4-bit"

    def test_more_bits_lower_mse(self):
        """Higher bit-width should give lower MSE (for same input)."""
        from superkv.algorithms.turboquant import (
            generate_rotation_matrix,
            turboquant_roundtrip_mse,
        )
        Pi = generate_rotation_matrix(128)
        x = torch.randn(8, 128)
        mse2 = turboquant_roundtrip_mse(x, Pi, 2)
        mse3 = turboquant_roundtrip_mse(x, Pi, 3)
        mse4 = turboquant_roundtrip_mse(x, Pi, 4)
        assert mse2 > mse3 > mse4, \
            f"MSE should decrease with bits: 2b={mse2:.4f} 3b={mse3:.4f} 4b={mse4:.4f}"

    def test_near_zero_stays_near_zero(self):
        """Very small values should dequantize to near-zero."""
        from superkv.algorithms.turboquant import (
            generate_rotation_matrix,
            turboquant_quantize, turboquant_dequantize,
        )
        Pi = generate_rotation_matrix(64)
        x = torch.randn(4, 64) * 0.001
        idx, norms = turboquant_quantize(x, Pi, 3)
        x_recon = turboquant_dequantize(idx, norms, Pi, 3, x.shape)
        assert x_recon.abs().max() < 0.1


# ═══════════════════════════════════════════════════════════════════════
# Compressor protocol tests
# ═══════════════════════════════════════════════════════════════════════

class TestTurboQuantCompressor:
    def test_registered(self):
        assert "turboquant" in list_algorithms()

    def test_create(self):
        c = create_compressor("turboquant",
                              num_heads=8, head_dim=128)
        assert c.name == "turboquant"
        assert c.bits_k == 3
        assert c.bits_v == 2

    def test_compress_decompress(self):
        c = create_compressor("turboquant",
                              num_heads=8, head_dim=128,
                              bits_k=3, bits_v=2)
        K = torch.randn(8, 128)
        V = torch.randn(8, 128)
        packed = c.compress(K, V, layer_idx=0)
        K_recon, V_recon = c.decompress(packed, layer_idx=0)
        assert K_recon.shape == (8, 128)
        assert V_recon.shape == (8, 128)

    def test_multi_layer(self):
        c = create_compressor("turboquant",
                              num_heads=8, head_dim=128,
                              num_layers=4)
        # Each layer has different rotation matrix
        for l in range(4):
            packed = c.compress(torch.randn(8, 128), torch.randn(8, 128),
                                layer_idx=l)
            assert packed is not None

    def test_memory_report(self):
        c = create_compressor("turboquant",
                              num_heads=8, head_dim=128)
        for _ in range(20):
            c.compress(torch.randn(8, 128), torch.randn(8, 128),
                       layer_idx=0)
        report = c.memory_report()
        assert report['compression_ratio'] > 1.0
        # 3-bit K + 2-bit V + norms overhead ≈ 5-6x compression
        assert report['compression_ratio'] > 3.0

    def test_reset(self):
        c = create_compressor("turboquant",
                              num_heads=8, head_dim=128)
        c.compress(torch.randn(8, 128), torch.randn(8, 128),
                   layer_idx=0)
        c.reset()
        packed = c.compress(torch.randn(8, 128), torch.randn(8, 128),
                            layer_idx=0)
        assert packed is not None


class TestTurboQuantHybridK:
    def test_large_k_switches_to_int8(self):
        """K with wide range should use INT8 mode automatically."""
        from superkv.engine.registry import create_compressor
        c = create_compressor("turboquant", num_heads=8, head_dim=128, k_auto_scale=True)
        K = torch.randn(8, 128) * 100  # max ~300, triggers INT8 path
        V = torch.randn(8, 128) * 0.5
        packed = c.compress(K, V, layer_idx=0)
        Kr, Vr = c.decompress(packed, layer_idx=0)
        mse_k = torch.nn.functional.mse_loss(Kr, K).item()
        assert mse_k < 10, f"K MSE {mse_k:.2f} — INT8 should handle wide range"
        assert packed[0] == 'int8', f"Expected int8 mode, got {packed[0]}"

    def test_small_k_stays_turboquant(self):
        """Small K values should stay in TurboQuant mode."""
        from superkv.engine.registry import create_compressor
        c = create_compressor("turboquant", num_heads=8, head_dim=128, k_auto_scale=True)
        K = torch.randn(8, 128) * 3
        V = torch.randn(8, 128)
        packed = c.compress(K, V, layer_idx=0)
        assert packed[0] == 'tq', f"Expected tq mode, got {packed[0]}"

    def test_version(self):
        from superkv.engine.registry import create_compressor
        c = create_compressor("turboquant", num_heads=8, head_dim=128)
        assert c.version == "0.2"

    def test_disable_auto_scale(self):
        """k_auto_scale=False should always use TurboQuant path."""
        from superkv.engine.registry import create_compressor
        c = create_compressor("turboquant", num_heads=8, head_dim=128, k_auto_scale=False)
        K = torch.randn(8, 128) * 100
        V = torch.randn(8, 128)
        packed = c.compress(K, V, layer_idx=0)
        assert packed[0] == 'tq', "Should stay tq when auto_scale disabled"
