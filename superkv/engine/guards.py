"""Input validation utilities for KV compressors.

Guard functions that check K,V tensors for common issues:
NaN, Inf, empty tensors, wrong shapes, extreme values.

Usage:
    from superkv.engine.guards import check_kv
    K, V = check_kv(K, V)  # returns cleaned or raises descriptive error
"""

from __future__ import annotations

import torch
import math
import logging

logger = logging.getLogger(__name__)

_NAN_WARNED = False


def check_kv(K: torch.Tensor, V: torch.Tensor,
             expected_heads: int | None = None,
             expected_dim: int | None = None,
             allow_batch: bool = True,
             clip_range: float | None = None,
             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and clean K,V tensors before compression.

    Args:
        K, V: input tensors
        expected_heads: if set, validate num_heads dimension
        expected_dim: if set, validate head_dim dimension
        allow_batch: if True, accept 3D (batch, heads, dim)
        clip_range: if set, clamp values to [-clip_range, clip_range]

    Returns:
        Cleaned (K, V). May modify in-place.

    Raises:
        ValueError: on invalid shapes or unrecoverable values.
    """
    _check_not_empty(K, "K")
    _check_not_empty(V, "V")

    # Shape validation
    if K.dim() == 3 and allow_batch:
        _check_shape(K, expected_heads, expected_dim, dim_offset=1)
    elif K.dim() == 2:
        _check_shape(K, expected_heads, expected_dim)
    else:
        raise ValueError(f"K must be 2D or 3D, got shape {K.shape}")

    if K.shape != V.shape:
        raise ValueError(f"K shape {K.shape} != V shape {V.shape}")

    # NaN/Inf detection
    global _NAN_WARNED
    if torch.isnan(K).any() or torch.isinf(K).any():
        nan_count = torch.isnan(K).sum().item()
        inf_count = torch.isinf(K).sum().item()
        msg = f"K contains {nan_count} NaN, {inf_count} Inf values. "
        if nan_count < K.numel() * 0.01:
            # Small contamination: replace with zeros
            K = torch.nan_to_num(K, nan=0.0, posinf=1e4, neginf=-1e4)
            msg += "Replaced with 0/±1e4."
        else:
            msg += "Too many — aborting."
            raise ValueError(msg)
        if not _NAN_WARNED:
            logger.warning(msg)
            _NAN_WARNED = True

    if torch.isnan(V).any() or torch.isinf(V).any():
        nan_count = torch.isnan(V).sum().item()
        inf_count = torch.isinf(V).sum().item()
        msg = f"V contains {nan_count} NaN, {inf_count} Inf values. "
        if nan_count < V.numel() * 0.01:
            V = torch.nan_to_num(V, nan=0.0, posinf=1e4, neginf=-1e4)
            msg += "Replaced with 0/±1e4."
        else:
            msg += "Too many — aborting."
            raise ValueError(msg)
        if not _NAN_WARNED:
            logger.warning(msg)
            _NAN_WARNED = True

    # Optional clipping
    if clip_range is not None:
        K = torch.clamp(K, -clip_range, clip_range)
        V = torch.clamp(V, -clip_range, clip_range)

    return K, V


def _check_not_empty(tensor, name):
    if tensor.numel() == 0:
        raise ValueError(f"{name} is empty (numel=0)")


def _check_shape(tensor, expected_heads, expected_dim, dim_offset=0):
    """dim_offset: 0 for 2D (heads,dim), 1 for 3D (batch,heads,dim)."""
    if tensor.dim() == 2:
        h_idx, d_idx = 0, 1
    else:
        h_idx, d_idx = 1, 2
    if expected_heads is not None and tensor.shape[h_idx] != expected_heads:
        raise ValueError(
            f"Expected {expected_heads} heads, got {tensor.shape[h_idx]}")
    if expected_dim is not None and tensor.shape[d_idx] != expected_dim:
        raise ValueError(
            f"Expected head_dim={expected_dim}, got {tensor.shape[d_idx]}")
