# superKV

**Multi-algorithm KV cache compression toolkit.**

> 一行代码压缩 KV cache，4-6x 内存节省，89 tests 验证。
> 跨平台（CUDA / CPU / Metal / Ascend）| vLLM 插件 | 自适应策略

## 快速上手

```bash
pip install superkv  # 或: git clone + uv sync
```

```python
from superkv import create_compressor

# 自适应模式 — 自动选最优策略，零配置
c = create_compressor("adaptive", num_heads=8, head_dim=128, num_layers=36)

# 压缩（和普通 dict 一样简单）
kf = torch.randn(8, 128)  # 你的模型 K, V 就长这样
vf = torch.randn(8, 128)
c.compress(kf, vf, layer_idx=0, token_id=0)  # token 0: 关键帧
packed = c.compress(kf + 0.1, vf + 0.1, layer_idx=0, token_id=1)  # 压缩！
k_hat, v_hat = c.decompress(packed, layer_idx=0)  # 解压

print(c.memory_report())
# {'algorithm': 'adaptive', 'compression_ratio': 4.5, ...}
```

## 原理（30 秒版）

KV cache 是 LLM 推理的内存大户。superKV 用三种互补手段压缩：

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐
│  量化       │    │  残差编码     │    │  分块淘汰     │
│  每个值压小  │ +  │  只存差值     │ +  │  不重要不要   │
│  2-6x       │    │  额外 2-4x   │    │  额外 2-10x  │
└─────────────┘    └──────────────┘    └──────────────┘
       ↓                  ↓                   ↓
          TurboQuant        DeltaKV            ChunkKV
          自适应组合: AdaptiveCompressor
```

## 内置算法

| 算法 | 一句话 | 压缩率 | 精度 | 适用场景 |
|------|--------|--------|------|----------|
| **adaptive** 🔥 | 自动选最优 | 4-6x | MSE<0.01 | 推荐首选 |
| **deltakv** | INT8 残差 | 4.5x | MSE<0.01 | K 值范围大 |
| **turboquant** | 旋转+混合量化 | 6x | MSE<0.1 | K 值范围小 |
| **chunkkv** | 语义分块淘汰 | 2-10x | — | 长上下文 |
| **eviction** | 均匀/相似度驱逐 | 2-8x | — | 快速原型 |

> 94 tests 通过 — M2 Mac + RTX 3090 双平台验证

## 手动选择算法

```python
# DeltaKV: 适合 K 值范围大的模型（如 Qwen3-8B 浅层 K∈[-204,218]）
c = create_compressor("deltakv", num_heads=8, head_dim=128, reference_stride=8)

# TurboQuant: 适合 V 值小、压缩优先的场景
c = create_compressor("turboquant", num_heads=8, head_dim=128, bits_k=3, bits_v=2)

# ChunkKV: 长上下文叠加驱逐
from superkv.algorithms.chunkkv import ChunkKVTracker
tracker = ChunkKVTracker(chunk_size=8, top_k=4, adaptive=True)  # 弹性分块
tracker.should_keep(token_id, K)  # → True/False
```

## vLLM 插件

```python
from superkv.vllm_plugin import VLLMModelHook

hook = VLLMModelHook(algorithm="adaptive")
hook.install(model)  # 自动拦截 attention 层
# 模型现在自动压缩 KV cache
print(hook.report())
```

## GPU 实测（RTX 3090, Qwen3-8B）

```
模型: Qwen3-8B, 16.4GB VRAM, 36 layers, 8KV×128dim

Layer  K 范围       DeltaKV        TurboQuant+Adaptive
  0    [-204,218]   4.5x MSE=0.006  6.1x MSE=0.095
 12    [-19,21]     4.5x MSE=0.001  6.1x MSE=0.001
 35    [-29,25]     4.5x MSE=0.002  6.1x MSE=0.001
```

## 架构

```
superkv/
├── engine/            # 注册表 + 平台检测 + NaN 守护
├── algorithms/        # 算法插件
│   ├── deltakv/       # INT8 残差 + Q4_0
│   ├── turboquant/    # 旋转 + Lloyd-Max 量化
│   ├── chunkkv.py     # 语义分块淘汰
│   ├── adaptive.py    # 自动路由
│   ├── eviction.py    # 均匀/相似度驱逐
│   └── transforms.py  # 预处理工具
├── kernels/tilelang/  # 跨平台 GPU kernel
├── tools/             # benchmark
└── vllm_plugin/       # vLLM 集成
```

## 平台

| 平台 | tilelang | PyTorch |
|------|----------|---------|
| NVIDIA | ✅ | ✅ |
| Metal (M2) | 🔜 | ✅ |
| Ascend | 🔜 | ✅ |
| CPU | ✅ | ✅ |

## License

MIT
