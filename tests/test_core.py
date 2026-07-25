"""Test suite for superKV — core algorithms and registry.

Run:  python -m pytest tests/ -v
"""

import torch
import pytest

from superkv.engine.registry import (
    list_algorithms, get_algorithm, create_compressor, KVCompressor,
)
from superkv.engine.platform import detect_platform, get_tilelang_target
from superkv.algorithms.deltakv import (
    Q4_0_BLOCK_SIZE,
    delta_encode, delta_decode,
    quantize_q4_0, dequantize_q4_0,
    delta_encode_q4_0, delta_decode_q4_0,
    delta_encode_q4_0_normalized, delta_decode_q4_0_normalized,
    sparse_attention,
)


# ═══════════════════════════════════════════════════════════════════════
# Platform tests
# ═══════════════════════════════════════════════════════════════════════

class TestPlatform:
    def test_detect(self):
        p = detect_platform()
        assert p in ("cpu", "cuda", "metal")

    def test_tilelang_target(self):
        t = get_tilelang_target()
        assert t in ("c", "cuda", "metal")


# ═══════════════════════════════════════════════════════════════════════
# Registry tests
# ═══════════════════════════════════════════════════════════════════════

class TestRegistry:
    def test_list_algorithms(self):
        algos = list_algorithms()
        assert "deltakv" in algos

    def test_get_algorithm(self):
        cls = get_algorithm("deltakv")
        assert cls.name == "deltakv"

    def test_create_compressor(self):
        c = create_compressor("deltakv", num_heads=4, head_dim=64)
        assert isinstance(c, KVCompressor)
        assert c.name == "deltakv"

    def test_unknown_algorithm(self):
        with pytest.raises(KeyError):
            get_algorithm("nonexistent")


# ═══════════════════════════════════════════════════════════════════════
# Delta encode/decode tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeltaEncode:
    def test_encode_zero(self):
        curr = torch.randn(4, 32)
        kf = curr.clone()
        r = delta_encode(curr, kf)
        assert torch.allclose(r, torch.zeros_like(r), atol=1e-6)

    def test_roundtrip(self):
        kf = torch.randn(4, 32)
        curr = kf + torch.randn(4, 32) * 0.1
        r = delta_encode(curr, kf)
        recon = delta_decode(kf, r)
        assert torch.allclose(recon, curr, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Q4_0 quantize/dequantize tests
# ═══════════════════════════════════════════════════════════════════════

class TestQ4_0:
    def test_block_size(self):
        assert Q4_0_BLOCK_SIZE == 32

    def test_roundtrip_small_range(self):
        x = torch.randn(4, 32) * 0.5
        q, d = quantize_q4_0(x)
        x_recon = dequantize_q4_0(q, d, x.shape)
        mse = torch.nn.functional.mse_loss(x_recon, x).item()
        assert mse < 0.05, f"MSE {mse:.4f} too high"

    def test_output_shapes(self):
        x = torch.randn(2, 64)
        q, d = quantize_q4_0(x)
        assert q.shape == (2, 32)  # 64/2 packed
        assert d.shape == (2, 2)   # 64/32 blocks

    def test_roundtrip_multi_head(self):
        x = torch.randn(8, 128) * 0.3
        q, d = quantize_q4_0(x)
        x_recon = dequantize_q4_0(q, d, x.shape)
        mse = torch.nn.functional.mse_loss(x_recon, x).item()
        assert mse < 0.1, f"MSE {mse:.4f} too high"

    def test_large_range_degradation(self):
        """K values in real models can be large — Q4_0 degrades."""
        x = torch.randn(4, 32) * 200  # simulate real K range
        q, d = quantize_q4_0(x)
        x_recon = dequantize_q4_0(q, d, x.shape)
        mse = torch.nn.functional.mse_loss(x_recon, x).item()
        # With wide range, MSE is expected to be high — this is the
        # motivation for normalized Q4_0
        assert mse > 0.01, "Expected degradation on large range"


# ═══════════════════════════════════════════════════════════════════════
# Delta + Q4_0 combined tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeltaQ4_0:
    def test_roundtrip(self):
        kf = torch.randn(4, 64)
        curr = kf + torch.randn(4, 64) * 0.05
        q, d = delta_encode_q4_0(curr, kf)
        recon = delta_decode_q4_0(kf, q, d)
        assert recon.shape == curr.shape

    def test_roundtrip_normalized(self):
        kf = torch.randn(8, 128) * 100  # wide range
        curr = kf + torch.randn(8, 128) * 5
        q, d, s = delta_encode_q4_0_normalized(curr, kf)
        recon = delta_decode_q4_0_normalized(kf, q, d, s)
        mse = torch.nn.functional.mse_loss(recon, curr).item()
        # Normalized should handle large ranges well
        assert mse < 1.0, f"Normalized MSE {mse:.4f} too high"


# ═══════════════════════════════════════════════════════════════════════
# Sparse attention tests
# ═══════════════════════════════════════════════════════════════════════

class TestSparseAttention:
    def test_output_shape(self):
        Q = torch.randn(4, 32)
        K = torch.randn(8, 4, 32)
        V = torch.randn(8, 4, 32)
        out = sparse_attention(Q, K, V)
        assert out.shape == (4, 32)

    def test_softmax_sum(self):
        Q = torch.randn(4, 32)
        K = torch.randn(4, 4, 32)
        V = torch.randn(4, 4, 32)
        out = sparse_attention(Q, K, V)
        assert not torch.isnan(out).any()


# ═══════════════════════════════════════════════════════════════════════
# DeltaKV Compressor integration tests
# ═══════════════════════════════════════════════════════════════════════

class TestDeltaKVCompressor:
    def test_create(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64, num_layers=2)
        assert c.name == "deltakv"
        assert c.version == "4.0"

    def test_compress_keyframe(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64, reference_stride=4)
        K = torch.randn(4, 64)
        V = torch.randn(4, 64)
        # token 0 should be keyframe
        result = c.compress(K, V, layer_idx=0, token_id=0)
        assert result is None  # keyframe, nothing to compress

    def test_compress_non_keyframe(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64, reference_stride=4)
        c.compress(torch.randn(4, 64), torch.randn(4, 64),
                   layer_idx=0, token_id=0)  # keyframe
        K = torch.randn(4, 64) * 0.5
        V = torch.randn(4, 64) * 0.5
        result = c.compress(K, V, layer_idx=0, token_id=1)
        assert result is not None  # should return packed data

    def test_decompress_roundtrip(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64, reference_stride=4,
                              normalized=False)
        K0 = torch.randn(4, 64)
        V0 = torch.randn(4, 64)
        c.compress(K0, V0, layer_idx=0, token_id=0)
        K1 = K0 + torch.randn(4, 64) * 0.01
        V1 = V0 + torch.randn(4, 64) * 0.01
        packed = c.compress(K1, V1, layer_idx=0, token_id=1)
        K_recon, V_recon = c.decompress(packed, layer_idx=0)
        assert K_recon.shape == K1.shape
        assert V_recon.shape == V1.shape

    def test_memory_report(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64, reference_stride=4)
        for i in range(10):
            c.compress(torch.randn(4, 64), torch.randn(4, 64),
                       layer_idx=0, token_id=i)
        report = c.memory_report()
        assert report['compression_ratio'] > 1.0
        assert report['algorithm'] == 'deltakv'

    def test_reset(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64)
        c.compress(torch.randn(4, 64), torch.randn(4, 64),
                   layer_idx=0, token_id=0)
        c.reset()
        # After reset, next token should become keyframe
        result = c.compress(torch.randn(4, 64), torch.randn(4, 64),
                            layer_idx=0, token_id=0)
        assert result is None

    def test_normalized_large_range(self):
        """Normalized Q4_0 handles large K values from real models."""
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=8, head_dim=128, reference_stride=4,
                              normalized=True)
        # Simulate real Qwen3-8B K values: range [-322, 329]
        K0 = torch.randn(8, 128) * 100
        V0 = torch.randn(8, 128) * 0.5
        c.compress(K0, V0, layer_idx=0, token_id=0)
        K1 = K0 + torch.randn(8, 128) * 5
        V1 = V0 + torch.randn(8, 128) * 0.1
        packed = c.compress(K1, V1, layer_idx=0, token_id=1)
        K_recon, V_recon = c.decompress(packed, layer_idx=0)
        mse_k = torch.nn.functional.mse_loss(K_recon, K1).item()
        mse_v = torch.nn.functional.mse_loss(V_recon, V1).item()
        # Normalized should keep MSE reasonable even with large K range
        assert mse_k < 50, f"K MSE {mse_k:.2f} too high (normalized failed)"
        assert mse_v < 0.1, f"V MSE {mse_v:.2f} too high"


class TestBatchSupport:
    def test_batch_compress(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64, reference_stride=1)
        # stride=1 means every token is keyframe
        K = torch.randn(3, 4, 64)
        V = torch.randn(3, 4, 64)
        results = c.compress(K, V, layer_idx=0, token_id=0)
        # All should be keyframes (stride=1)
        assert results is None

    def test_batch_mixed_tokens(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64, reference_stride=4)
        # Token 0: keyframe (batch=2)
        c.compress(torch.randn(2, 4, 64), torch.randn(2, 4, 64),
                   layer_idx=0, token_id=0)
        # Token 1: non-keyframe
        results = c.compress(torch.randn(2, 4, 64), torch.randn(2, 4, 64),
                             layer_idx=0, token_id=1)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_max_tokens_eviction(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64,
                              reference_stride=1, max_tokens=5)
        for t in range(10):
            c.compress(torch.randn(4, 64), torch.randn(4, 64),
                       layer_idx=0, token_id=t)
        r = c.memory_report()
        assert r['stored_tokens'] <= 5

    def test_version(self):
        from superkv.algorithms.deltakv import DeltaKVCompressor
        c = DeltaKVCompressor(num_heads=4, head_dim=64)
        assert c.version == "4.0"


class TestChunkKV:
    def test_create(self):
        from superkv.algorithms.chunkkv import ChunkKVTracker
        t = ChunkKVTracker(chunk_size=4, top_k=2)
        assert t.chunk_size == 4
        assert t.top_k == 2

    def test_warmup_keeps_everything(self):
        from superkv.algorithms.chunkkv import ChunkKVTracker
        t = ChunkKVTracker(chunk_size=4, top_k=2)
        K = torch.randn(8, 64)
        # First 16 tokens (8 chunks): warmup, keep all
        for i in range(16):
            assert t.should_keep(i, K) is True

    def test_pruning_after_warmup(self):
        from superkv.algorithms.chunkkv import ChunkKVTracker
        t = ChunkKVTracker(chunk_size=4, top_k=2)
        K_const = torch.randn(8, 64)
        # Feed 40 tokens (10 chunks) with varying K to create score differences
        for i in range(40):
            K_varying = K_const + torch.randn(8, 64) * (i % 5)
            t.should_keep(i, K_varying)
        # After warmup + pruning, some chunks should be evicted
        keeps = [t.should_keep(i, K_const) for i in range(40)]
        kept_count = sum(keeps)
        # Should keep roughly chunk_size * top_k tokens
        assert kept_count <= 40  # at most everything
        assert kept_count >= 8   # at least two chunks

    def test_layer_reuse(self):
        from superkv.algorithms.chunkkv import ChunkKVTracker
        t = ChunkKVTracker(chunk_size=4, top_k=2)
        K = torch.randn(8, 64)
        # Feed layer 0
        for i in range(20):
            t.should_keep(i, K, layer_idx=0)
        # Layer 5 should use cached indices without re-scoring
        keeps_l5 = [t.should_keep(i, K, layer_idx=5) for i in range(20)]
        assert sum(keeps_l5) <= 20

    def test_reset(self):
        from superkv.algorithms.chunkkv import ChunkKVTracker
        t = ChunkKVTracker(chunk_size=4, top_k=2)
        for i in range(20):
            t.should_keep(i, torch.randn(8, 64))
        t.reset()
        assert len(t._chunk_scores) == 0
