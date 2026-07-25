"""Platform detection and tilelang target dispatch."""

import platform as _platform
from typing import Literal

Target = Literal["c", "cuda", "metal"]


def _has_torch_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def _has_torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def detect_platform() -> str:
    """Detect the current compute platform.

    Returns one of: 'cpu', 'cuda', 'metal'.
    """
    if _has_torch_cuda():
        return "cuda"
    if _has_torch_mps():
        return "metal"
    return "cpu"


def get_tilelang_target() -> Target:
    """Get the tilelang compilation target for the current platform.

    - CPU: 'c'
    - GPU (CUDA): 'cuda'
    - Apple Silicon (Metal): 'c' as fallback until PR #2767 is merged
    """
    plat = detect_platform()
    if plat == "cuda":
        return "cuda"
    # Metal support pending tile-ai/tilelang PR #2767
    # Use 'c' (CPU) as fallback for now
    return "c"
