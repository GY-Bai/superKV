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

## License

MIT
