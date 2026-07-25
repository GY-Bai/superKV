"""Platform detection and tilelang target dispatch."""

from typing import Literal

Target = Literal["c", "cuda", "metal"]


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _has_mps() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except Exception:
        return False


def detect_platform() -> str:
    """Detect the current compute platform."""
    if _has_cuda():
        return "cuda"
    if _has_mps():
        return "metal"
    return "cpu"


def get_tilelang_target() -> Target:
    """Get the tilelang compilation target."""
    if detect_platform() == "cuda":
        return "cuda"
    # Metal pending PR #2767; fallback to CPU
    return "c"
