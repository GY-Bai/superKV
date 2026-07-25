# superKV

**Multi-algorithm KV cache compression toolkit.**

Cross-platform (CUDA / Metal / Ascend / CPU) via tilelang.
Pluggable vLLM V1 integration.

## Algorithms

| Algorithm | Method | Status |
|-----------|--------|--------|
| **DeltaKV** | Residual encoding + Q4_0 quantization + sparse attention | ✅ v0.1 |
| **TurboQuant** | Random rotation + Lloyd-Max optimal quantization (K3/V2) | ✅ v0.1 |
| **RocketKV** | Two-stage hybrid eviction | 📋 planned |
| **KV-Compress** | Paged variable-rate compression | 📋 planned |

## vLLM Plugin (Gate 8)

```python
from superkv.vllm_plugin import VLLMModelHook

hook = VLLMModelHook(algorithm="deltakv", reference_stride=8)
hook.install(model)  # hooks attention layers for KV compression

# Model now compresses KV cache via superKV
print(hook.report())  # {"compression_ratio": 4.2, ...}
```

## Quick Start

```python
from superkv import create_compressor, list_algorithms

# See what's available
print(list_algorithms())  # ['deltakv']

# Create a compressor
c = create_compressor("deltakv", num_heads=8, head_dim=128,
                      reference_stride=8, normalized=True)

# Compress KV cache
kf_k = torch.randn(8, 128)
kf_v = torch.randn(8, 128)
c.compress(kf_k, kf_v, layer_idx=0, token_id=0)  # keyframe

curr_k = kf_k + torch.randn(8, 128) * 0.1
curr_v = kf_v + torch.randn(8, 128) * 0.1
packed = c.compress(curr_k, curr_v, layer_idx=0, token_id=1)

# Decompress
k_recon, v_recon = c.decompress(packed, layer_idx=0)

# Check savings
print(c.memory_report())
```

## vLLM Integration

```python
import superkv.vllm_plugin  # registers DeltaKVSpec
# Then launch vLLM with superKV algorithms
```

## Architecture

```
superkv/
├── engine/        # Registry + platform detection
├── algorithms/    # Each algorithm = one subpackage
│   ├── deltakv/   # Residual + Q4_0 + sparse
│   └── turboquant/# Random rotation + scalar quant
├── kernels/       # tilelang-accelerated kernels
└── vllm_plugin/   # vLLM V1 KVCacheSpec registration
```

## Platform Support

| Platform | tilelang | PyTorch fallback |
|----------|----------|-----------------|
| NVIDIA CUDA | ✅ cuda | ✅ |
| Apple Metal | 🔜 pending PR#2767 | ✅ |
| Ascend NPU | 🔜 via tilelang | ✅ |
| x86/ARM CPU | ✅ c | ✅ |

## GPU Smoke Tests (RTX 3090, 24GB)

### Real Model Validation

| Model | Type | Attention | VRAM | DeltaKV V3 | K Range |
|-------|------|-----------|------|------------|---------|
| **Qwen3-8B** | Dense 8B | traditional ✅ | 16.4GB | 2.6x, MSE<0.006 | [-204, 218] |
| **OLMoE-1B-7B** | MoE 7B | traditional ✅ | 13.8GB | 2.7x, MSE<0.001 | [-17, 19] |

### Algorithm Comparison (Qwen3-8B, Layer 0)

| Algorithm | Compression | K MSE | V MSE | Notes |
|-----------|------------|-------|-------|-------|
| **DeltaKV V3** (INT8 delta) | 2.6x | 0.005 | 0.000 | Best accuracy on wide K ranges |
| **TurboQuant** (K3V2) | 9.8x | 11.0 | 0.000 | K explodes on [-204,218]; OK on smaller ranges |

### tilelang CUDA Kernel Performance

| Kernel | Input Shape | Time/iter | vs PyTorch |
|--------|------------|-----------|------------|
| Q4_0 quant+dequant | 1024×128 | 2.7ms | — |
| Attention scores | 8×128 × 256 tok | 0.2ms | cuBLAS |
| Attention (precompiled 4×32×8) | 4×32 × 8 tok | ~5µs | matches einsum |

### Attention Type Detection

| Model Type | Detection | Action |
|-----------|-----------|--------|
| Traditional (Qwen3-8B, OLMoE) | `traditional` | Hook and compress ✅ |
| Linear (GatedDeltaNet, Qwen3.5) | `linear` | Skip — no K,V to compress ✅ |
| MLA (DeepSeek) | `mla` | Skip — already compressed ✅ |
| Mamba/SSM | `mamba` | Skip — state space ✅ |

## License

MIT
