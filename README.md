# superKV

> Multi-algorithm KV cache compression toolkit.
> 多算法 KV 缓存压缩工具包。

**One line to 4-6x memory savings. | 一行代码，4-6 倍内存节省。**

89 tests verified · 跨平台（CUDA/CPU/Metal/Ascend） · vLLM plugin · 自适应策略

---

## Quick Start · 快速上手

```bash
pip install superkv  # or: git clone + uv sync
```

```python
from superkv import create_compressor

# Adaptive mode — auto-selects best strategy, zero config
# 自适应模式 — 自动选最优策略，零配置
c = create_compressor("adaptive", num_heads=8, head_dim=128, num_layers=36)

# Compress (as easy as a dict) · 压缩和字典一样简单
kf = torch.randn(8, 128)
vf = torch.randn(8, 128)
c.compress(kf, vf, layer_idx=0, token_id=0)          # token 0: keyframe 关键帧
packed = c.compress(kf + 0.1, vf + 0.1, layer_idx=0, token_id=1)  # compressed 压缩
k_hat, v_hat = c.decompress(packed, layer_idx=0)      # decompress 解压

print(c.memory_report())
# {'algorithm': 'adaptive', 'compression_ratio': 4.5, ...}
```

## How it works · 原理（30 秒）

KV cache is the memory bottleneck for LLM inference. superKV attacks it from three angles:
KV cache 是 LLM 推理的内存瓶颈，superKV 从三个方向压缩：

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│ Quantization│    │   Residual   │    │   Eviction   │
│ 量化        │ +  │   残差编码    │ +  │   分块淘汰    │
│ 2-6x        │    │  extra 2-4x  │    │  extra 2-10x │
└─────────────┘    └──────────────┘    └──────────────┘
       ↓                  ↓                   ↓
   TurboQuant          DeltaKV            ChunkKV
        AdaptiveCompressor (auto-select)
```

## Algorithms · 内置算法

| Algorithm 算法 | Description 描述 | Compression 压缩 | Accuracy 精度 | Use when 适用 |
|------|------|------|------|------|
| **adaptive** 🔥 | Auto-select best 自动选最优 | 4-6x | MSE<0.01 | Recommended 推荐 |
| **deltakv** | INT8 residual 残差编码 | 4.5x | MSE<0.01 | Wide K range K 值域大 |
| **turboquant** | Rotation + Lloyd-Max 旋转量化 | 6x | MSE<0.1 | Small K range K 值域小 |
| **chunkkv** | Semantic chunk eviction 语义分块 | 2-10x | — | Long context 长上下文 |
| **eviction** | Uniform/similarity pruning 均匀驱逐 | 2-8x | — | Quick prototype 快速原型 |

> 94 tests passed — M2 Mac + RTX 3090 verified

## Manual Selection · 手动选择

```python
# DeltaKV: for models with large K range (e.g. Qwen3-8B L0 K∈[-204,218])
# DeltaKV: 适合 K 值域大的模型
c = create_compressor("deltakv", num_heads=8, head_dim=128, reference_stride=8)

# TurboQuant: for small V values, compression-first
# TurboQuant: V 值小、压缩优先
c = create_compressor("turboquant", num_heads=8, head_dim=128, bits_k=3, bits_v=2)

# ChunkKV: long-context eviction overlay · 长上下文叠加驱逐
from superkv.algorithms.chunkkv import ChunkKVTracker
tracker = ChunkKVTracker(chunk_size=8, top_k=4, adaptive=True)  # elastic 弹性
tracker.should_keep(token_id, K)  # → True/False
```

## vLLM Plugin · vLLM 插件

```python
from superkv.vllm_plugin import VLLMModelHook

hook = VLLMModelHook(algorithm="adaptive")
hook.install(model)  # auto-hooks attention layers · 自动拦截
print(hook.report())
```

## GPU Benchmarks · GPU 实测

RTX 3090, Qwen3-8B (16.4GB VRAM, 36 layers, 8KV×128dim)

| Layer 层 | K Range 范围 | DeltaKV | TurboQuant+Adaptive |
|------|------|------|------|
| 0 | [-204, 218] | 4.5x MSE=0.006 | 6.1x MSE=0.095 |
| 12 | [-19, 21] | 4.5x MSE=0.001 | 6.1x MSE=0.001 |
| 35 | [-29, 25] | 4.5x MSE=0.002 | 6.1x MSE=0.001 |

## Architecture · 架构

```
superkv/
├── engine/            # Registry 注册表 + platform 平台 + NaN guard 守护
├── algorithms/        # Algorithm plugins 算法插件
│   ├── deltakv/       # INT8 residual + Q4_0
│   ├── turboquant/    # Rotation + Lloyd-Max
│   ├── chunkkv.py     # Semantic chunk eviction
│   ├── adaptive.py    # Auto-router 自动路由
│   ├── eviction.py    # Uniform/similarity pruning
│   └── transforms.py  # Pre-processing utilities
├── kernels/tilelang/  # Cross-platform GPU kernels
├── tools/             # Benchmark
└── vllm_plugin/       # vLLM integration
```

## Platform Support · 平台支持

| Platform | tilelang | PyTorch fallback |
|----------|----------|------------------|
| NVIDIA | ✅ | ✅ |
| Apple Metal | 🔜 | ✅ |
| Ascend NPU | 🔜 | ✅ |
| x86/ARM CPU | ✅ | ✅ |

## License

MIT
