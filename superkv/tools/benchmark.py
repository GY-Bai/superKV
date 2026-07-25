"""Benchmark: compare KV compression algorithms.

Usage:
    uv run python -m superkv.tools.benchmark

Compares:
  - No compression (baseline)
  - DeltaKV (residual + Q4_0)
  - TurboQuant (rotation + optimal scalar quant)
  - DeltaKV + KeyframeEviction
  - TurboQuant + UniformEviction

Reports per-algorithm: compression ratio, MSE, throughput, memory.
"""

from __future__ import annotations

import torch
import time
import math
from typing import Any

from superkv.engine.registry import create_compressor, list_algorithms
from superkv.algorithms.eviction import (
    KeyframeEviction, UniformEviction, SimilarityEviction,
)


# ═══════════════════════════════════════════════════════════════════════
# Benchmark runner
# ═══════════════════════════════════════════════════════════════════════

def run_benchmark(
    seq_len: int = 256,
    num_heads: int = 8,
    head_dim: int = 128,
    num_layers: int = 1,
    device: str = "cpu",
    warmup: int = 3,
    n_iter: int = 10,
) -> list[dict[str, Any]]:
    """Run compression benchmark across all available algorithms.

    Simulates a realistic KV cache workload: sequence of tokens
    with temporally correlated K and V values.

    Returns list of per-algorithm results.
    """
    results = []

    # Generate synthetic sequence with temporal correlation
    # (adjacent tokens have similar K, V — realistic for LLMs)
    torch.manual_seed(42)
    base_k = torch.randn(num_heads, head_dim) * 0.5
    base_v = torch.randn(num_heads, head_dim) * 0.5
    seq_k = []
    seq_v = []
    for t in range(seq_len):
        noise_scale = 0.02 + 0.01 * math.sin(t / 10.0)
        seq_k.append(base_k + torch.randn(num_heads, head_dim) * noise_scale)
        seq_v.append(base_v + torch.randn(num_heads, head_dim) * noise_scale)
        # Slow drift of base
        base_k = base_k + torch.randn(num_heads, head_dim) * 0.01
        base_v = base_v + torch.randn(num_heads, head_dim) * 0.01

    # ── Baseline: no compression ─────────────────────────────────────
    results.append(_bench_baseline(seq_k, seq_v, warmup, n_iter))

    # ── DeltaKV ───────────────────────────────────────────────────────
    results.append(_bench_deltakv(seq_k, seq_v, num_heads, head_dim,
                                   warmup, n_iter))

    # ── TurboQuant ────────────────────────────────────────────────────
    results.append(_bench_turboquant(seq_k, seq_v, num_heads, head_dim,
                                      warmup, n_iter))

    # ── DeltaKV + KeyframeEviction ────────────────────────────────────
    for kf_stride in [4, 8]:
        results.append(_bench_deltakv_eviction(
            seq_k, seq_v, num_heads, head_dim, warmup, n_iter,
            evict=KeyframeEviction(keyframe_stride=kf_stride,
                                   residual_keep=1),
            label=f"DeltaKV + KF-{kf_stride}"))

    # ── TurboQuant + UniformEviction ──────────────────────────────────
    for stride in [2, 4]:
        results.append(_bench_turboquant_eviction(
            seq_k, seq_v, num_heads, head_dim, warmup, n_iter,
            evict=UniformEviction(stride=stride),
            label=f"TurboQuant + Uniform-{stride}"))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Per-algorithm benchmark helpers
# ═══════════════════════════════════════════════════════════════════════

def _bench_baseline(seq_k, seq_v, warmup, n_iter):
    """Baseline: store full FP32 K, V."""
    elem_bytes = 4  # float32
    total = sum(k.numel() * elem_bytes + v.numel() * elem_bytes
                for k, v in zip(seq_k, seq_v))

    # Time: just copying (no real work in baseline)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        stored = [(k.clone(), v.clone()) for k, v in zip(seq_k, seq_v)]
    elapsed = time.perf_counter() - t0

    return {
        'algorithm': 'No Compression',
        'original_mb': total / 1e6,
        'compressed_mb': total / 1e6,
        'ratio': 1.0,
        'mse_k': 0.0,
        'mse_v': 0.0,
        'throughput_tokens_per_s': n_iter * len(seq_k) / elapsed,
        'details': 'Baseline FP32',
    }


def _bench_deltakv(seq_k, seq_v, num_heads, head_dim, warmup, n_iter):
    """DeltaKV benchmark."""
    mses_k = []
    mses_v = []

    # Collect MSE separately (after warmup, single full pass)
    c = create_compressor("deltakv", num_heads=num_heads,
                          head_dim=head_dim, reference_stride=8,
                          normalized=True)
    for t, (k, v) in enumerate(zip(seq_k, seq_v)):
        packed = c.compress(k, v, layer_idx=0, token_id=t)
        if packed is not None:
            kr, vr = c.decompress(packed, layer_idx=0)
            mses_k.append(F.mse_loss(kr, k).item())
            mses_v.append(F.mse_loss(vr, v).item())
    report = c.memory_report()
    c.reset()

    # Benchmark throughput
    for _ in range(warmup):
        c2 = create_compressor("deltakv", num_heads=num_heads,
                               head_dim=head_dim, reference_stride=8,
                               normalized=True)
        for t, (k, v) in enumerate(zip(seq_k, seq_v)):
            c2.compress(k, v, layer_idx=0, token_id=t)
        c2.reset()
    del c2

    c3 = create_compressor("deltakv", num_heads=num_heads,
                           head_dim=head_dim, reference_stride=8,
                           normalized=True)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for t, (k, v) in enumerate(zip(seq_k, seq_v)):
            c3.compress(k, v, layer_idx=0, token_id=t)
        c3.reset()
    elapsed = time.perf_counter() - t0
    del c3

    avg_mse_k = sum(mses_k) / len(mses_k) if mses_k else 0
    avg_mse_v = sum(mses_v) / len(mses_v) if mses_v else 0

    return {
        'algorithm': 'DeltaKV (stride=8)',
        'original_mb': report['original_bytes'] / 1e6,
        'compressed_mb': report['compressed_bytes'] / 1e6,
        'ratio': report['compression_ratio'],
        'mse_k': avg_mse_k,
        'mse_v': avg_mse_v,
        'throughput_tokens_per_s': n_iter * len(seq_k) / elapsed,
        'details': f"normalized Q4_0, stride=8",
    }


def _bench_turboquant(seq_k, seq_v, num_heads, head_dim, warmup, n_iter):
    """TurboQuant benchmark."""
    for _ in range(warmup):
        c = create_compressor("turboquant", num_heads=num_heads,
                              head_dim=head_dim, bits_k=3, bits_v=2)
        for k, v in zip(seq_k[:10], seq_v[:10]):
            c.compress(k, v, layer_idx=0)
    del c

    c = create_compressor("turboquant", num_heads=num_heads,
                          head_dim=head_dim, bits_k=3, bits_v=2)
    mses_k = []
    mses_v = []
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for k, v in zip(seq_k, seq_v):
            packed = c.compress(k, v, layer_idx=0)
            kr, vr = c.decompress(packed, layer_idx=0)
        c.reset()
    elapsed = time.perf_counter() - t0

    # Re-run to collect MSE
    for k, v in zip(seq_k, seq_v):
        packed = c.compress(k, v, layer_idx=0)
        kr, vr = c.decompress(packed, layer_idx=0)
        mses_k.append(F.mse_loss(kr, k))
        mses_v.append(F.mse_loss(vr, v))

    report = c.memory_report()
    avg_mse_k = sum(mses_k).item() / len(mses_k)
    avg_mse_v = sum(mses_v).item() / len(mses_v)

    return {
        'algorithm': 'TurboQuant (K3V2)',
        'original_mb': report['original_bytes'] / 1e6,
        'compressed_mb': report['compressed_bytes'] / 1e6,
        'ratio': report['compression_ratio'],
        'mse_k': avg_mse_k,
        'mse_v': avg_mse_v,
        'throughput_tokens_per_s': n_iter * len(seq_k) / elapsed,
        'details': f"Lloyd-Max 3-bit K, 2-bit V",
    }


def _bench_deltakv_eviction(seq_k, seq_v, num_heads, head_dim,
                             warmup, n_iter, evict, label):
    """DeltaKV + eviction benchmark."""
    mses_k = []
    mses_v = []
    kept_count = 0

    for _ in range(warmup):
        c = create_compressor("deltakv", num_heads=num_heads,
                              head_dim=head_dim, reference_stride=8,
                              normalized=True)
        evict.reset()
        for t, (k, v) in enumerate(zip(seq_k, seq_v)):
            if evict.should_keep(t, k):
                c.compress(k, v, layer_idx=0, token_id=kept_count)
                kept_count += 1
    del c
    kept_count = 0
    evict.reset()

    c = create_compressor("deltakv", num_heads=num_heads,
                          head_dim=head_dim, reference_stride=8,
                          normalized=True)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for t, (k, v) in enumerate(zip(seq_k, seq_v)):
            if evict.should_keep(t, k):
                packed = c.compress(k, v, layer_idx=0, token_id=kept_count)
                if packed:
                    kr, vr = c.decompress(packed, layer_idx=0)
                    mses_k.append(F.mse_loss(kr, k))
                    mses_v.append(F.mse_loss(vr, v))
                kept_count += 1
        c.reset()
        evict.reset()
        kept_count = 0
    elapsed = time.perf_counter() - t0

    report = c.memory_report()
    evict_ratio = len(seq_k) / max(kept_count, 1)
    total_ratio = report['compression_ratio'] * evict_ratio

    return {
        'algorithm': label + f' (sparsity=1/{evict.stride if hasattr(evict, "stride") else "?"})',
        'original_mb': report['original_bytes'] / 1e6,
        'compressed_mb': report['compressed_bytes'] / 1e6,
        'ratio': total_ratio,
        'mse_k': sum(mses_k).item() / len(mses_k) if mses_k else 0,
        'mse_v': sum(mses_v).item() / len(mses_v) if mses_v else 0,
        'throughput_tokens_per_s': n_iter * len(seq_k) / elapsed,
        'details': f"DeltaKV + {type(evict).__name__}",
    }


def _bench_turboquant_eviction(seq_k, seq_v, num_heads, head_dim,
                                warmup, n_iter, evict, label):
    """TurboQuant + eviction benchmark."""
    mses_k = []
    mses_v = []

    for _ in range(warmup):
        c = create_compressor("turboquant", num_heads=num_heads,
                              head_dim=head_dim, bits_k=3, bits_v=2)
        for t, (k, v) in enumerate(zip(seq_k, seq_v)):
            if evict.should_keep(t):
                packed = c.compress(k, v, layer_idx=0)
    del c

    c = create_compressor("turboquant", num_heads=num_heads,
                          head_dim=head_dim, bits_k=3, bits_v=2)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        for t, (k, v) in enumerate(zip(seq_k, seq_v)):
            if evict.should_keep(t):
                packed = c.compress(k, v, layer_idx=0)
                kr, vr = c.decompress(packed, layer_idx=0)
                mses_k.append(F.mse_loss(kr, k))
                mses_v.append(F.mse_loss(vr, v))
        c.reset()
    elapsed = time.perf_counter() - t0

    report = c.memory_report()
    evict_ratio = evict.stride if hasattr(evict, 'stride') else 2
    total_ratio = report['compression_ratio'] * evict_ratio

    return {
        'algorithm': label,
        'original_mb': report['original_bytes'] / 1e6,
        'compressed_mb': report['compressed_bytes'] / 1e6,
        'ratio': total_ratio,
        'mse_k': sum(mses_k).item() / len(mses_k) if mses_k else 0,
        'mse_v': sum(mses_v).item() / len(mses_v) if mses_v else 0,
        'throughput_tokens_per_s': n_iter * len(seq_k) / elapsed,
        'details': f"TurboQuant + {type(evict).__name__}",
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

F = torch.nn.functional


def print_benchmark(results: list[dict]):
    """Pretty-print benchmark results."""
    header = f"{'Algorithm':<35} {'Ratio':>7} {'MSE_K':>8} {'MSE_V':>8} {'Tokens/s':>10}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        print(f"{r['algorithm']:<35} {r['ratio']:>6.1f}x "
              f"{r['mse_k']:>8.4f} {r['mse_v']:>8.4f} "
              f"{r['throughput_tokens_per_s']:>9.0f}")
    print(sep)

    # Best by ratio
    best = max(results, key=lambda r: r['ratio'])
    print(f"\nBest compression: {best['algorithm']} at {best['ratio']:.1f}x")

    # Best by MSE
    for which, key in [('K', 'mse_k'), ('V', 'mse_v')]:
        # Exclude baseline (MSE=0) and very high MSE
        candidates = [r for r in results if 0 < r[key] < 10]
        if candidates:
            best_mse = min(candidates, key=lambda r: r[key])
            print(f"Best {which} MSE:    {best_mse['algorithm']} at {best_mse[key]:.4f}")


if __name__ == "__main__":
    print("superKV Benchmark")
    print(f"  Available algorithms: {list_algorithms()}")
    print()

    results = run_benchmark(
        seq_len=256, num_heads=8, head_dim=128,
        warmup=2, n_iter=5,
    )
    print_benchmark(results)
