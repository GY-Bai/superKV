"""superKV: Multi-algorithm KV cache compression toolkit.

Cross-platform (CUDA / Metal / Ascend / CPU) via tilelang.
Pluggable vLLM V1 integration via KVCacheSpec registry.

Algorithms:
  - DeltaKV:  Residual encoding + Q4_0 quantization + sparse attention
  - TurboQuant (planned): Random rotation + optimal scalar quantization
  - RocketKV (planned): Two-stage eviction
  - KV-Compress (planned): Paged variable-rate compression
"""

from superkv.engine.platform import detect_platform, get_tilelang_target
from superkv.engine.registry import list_algorithms, register_algorithm

__version__ = "0.1.0"
__all__ = [
    "detect_platform",
    "get_tilelang_target",
    "list_algorithms",
    "register_algorithm",
]
