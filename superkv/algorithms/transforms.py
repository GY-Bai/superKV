"""Pre-processing transforms for KV values.

Real-model K values span wide ranges (e.g., [-204, 218] on Qwen3-8B)
that cause quantization to lose precision. These transforms compress
the dynamic range before quantization and restore after dequantization.

Transforms:
  log:  f(x) = sign(x) * log(1 + |x|)     → maps [-200,200] to [-5.3,5.3]
  tanh: f(x) = tanh(x / s)                → saturates at ±1
  clip: f(x) = clamp(x, -c, c)           → hard clip
"""

import math
import torch


def apply_log_transform(x: torch.Tensor) -> torch.Tensor:
    """Compress dynamic range: f(x) = sign(x) * log(1 + |x|).

    Good for: values with heavy tails but meaningful small variations.
    Inverse:  f^{-1}(y) = sign(y) * (exp(|y|) - 1)
    """
    sign_x = torch.sign(x)
    abs_x = torch.abs(x)
    return sign_x * torch.log1p(abs_x)


def apply_log_inverse(y: torch.Tensor) -> torch.Tensor:
    """Restore from log-compressed space."""
    sign_y = torch.sign(y)
    abs_y = torch.abs(y)
    return sign_y * (torch.exp(abs_y) - 1.0)


def apply_tanh_transform(x: torch.Tensor,
                          scale: float | None = None) -> tuple[torch.Tensor, float]:
    """Apply tanh(x/scale). Scale defaults to max(|x|) if not given.

    Returns: (y, scale) — scale needed for inverse.
    """
    if scale is None:
        scale = x.abs().max().item()
        scale = max(scale, 1e-8)
    return torch.tanh(x / scale), scale


def apply_tanh_inverse(y: torch.Tensor, scale: float) -> torch.Tensor:
    """Restore from tanh-compressed space."""
    # atanh is numerically unstable near ±1, clamp
    y_clamped = torch.clamp(y, -0.999, 0.999)
    return scale * torch.atanh(y_clamped)


def apply_clip_transform(x: torch.Tensor,
                          threshold: float = 10.0) -> torch.Tensor:
    """Hard clip to [-threshold, threshold]."""
    return torch.clamp(x, -threshold, threshold)


# Map of known transforms
TRANSFORMS = {
    'log': (apply_log_transform, apply_log_inverse),
    'tanh': (apply_tanh_transform, apply_tanh_inverse),
    'clip': (apply_clip_transform, lambda y: y),
    None: (lambda x: x, lambda y: y),
}


def get_transform(name: str | None) -> tuple:
    """Get (transform_fn, inverse_fn) by name."""
    return TRANSFORMS.get(name, TRANSFORMS[None])
