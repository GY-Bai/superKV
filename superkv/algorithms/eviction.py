"""Token eviction algorithms — select which KV tokens to keep.

Eviction complements quantization:
  1. Eviction reduces token count (sparse)
  2. Quantization compresses remaining tokens (dense)

Combined they achieve multiplicative compression:
  sparsity_rate × quantization_rate = total compression

Algorithms:
  - UniformEviction:   keep every Nth token (uniform stride)
  - SimilarityEviction: keep tokens whose K differs from neighbors
  - KeyframeEviction:  keep keyframe tokens, drop intermediates
                       (designed to pair with DeltaKV)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections import OrderedDict


# ═══════════════════════════════════════════════════════════════════════
# Base eviction tracker
# ═══════════════════════════════════════════════════════════════════════

class EvictionTracker:
    """Tracks which tokens to keep per layer.

    Usage:
        tracker = EvictionTracker(stride=4)
        for token_id in range(seq_len):
            should_keep = tracker.should_keep(token_id, K_current)
            if should_keep:
                compressor.compress(K, V, ...)
    """

    def should_keep(self, token_id: int,
                    K: torch.Tensor | None = None) -> bool:
        """Decide whether to keep this token."""
        raise NotImplementedError

    def kept_indices(self, seq_len: int) -> list[int]:
        """Return indices of kept tokens for a full sequence."""
        raise NotImplementedError

    def reset(self):
        pass


# ═══════════════════════════════════════════════════════════════════════
# Uniform Eviction
# ═══════════════════════════════════════════════════════════════════════

class UniformEviction(EvictionTracker):
    """Keep every Nth token. Simplest eviction, no overhead."""

    def __init__(self, stride: int = 4):
        assert stride >= 1
        self.stride = stride

    def should_keep(self, token_id: int,
                    K: torch.Tensor | None = None) -> bool:
        return token_id % self.stride == 0

    def kept_indices(self, seq_len: int) -> list[int]:
        return list(range(0, seq_len, self.stride))

    @property
    def sparsity(self) -> float:
        return 1.0 / self.stride


# ═══════════════════════════════════════════════════════════════════════
# Similarity Eviction
# ═══════════════════════════════════════════════════════════════════════

class SimilarityEviction(EvictionTracker):
    """Keep tokens whose K is dissimilar from the previous kept token.

    Proxy for attention importance: if a token's K is very similar
    to its neighbor, it's redundant → evict.

    Maintains a reference (last kept token's K). A new token is kept if
    its cosine similarity with the reference is below threshold.
    """

    def __init__(self, threshold: float = 0.98,
                 max_gap: int = 8):
        """
        Args:
            threshold: cosine similarity above which token is redundant
            max_gap:   force-keep after this many consecutive evictions
        """
        self.threshold = threshold
        self.max_gap = max_gap
        self._last_K: torch.Tensor | None = None
        self._gap = 0

    def should_keep(self, token_id: int,
                    K: torch.Tensor | None = None) -> bool:
        # First token always kept
        if token_id == 0:
            self._last_K = K.clone() if K is not None else None
            self._gap = 0
            return True

        # Force keep after max_gap
        if self._gap >= self.max_gap:
            self._last_K = K.clone() if K is not None else None
            self._gap = 0
            return True

        if K is None or self._last_K is None:
            return True

        # Cosine similarity per head, average across heads
        # K shape: (n_heads, head_dim)
        sim = F.cosine_similarity(
            K.float().flatten(), self._last_K.float().flatten(), dim=0)

        if sim < self.threshold:
            # This token is different → keep it
            self._last_K = K.clone()
            self._gap = 0
            return True

        self._gap += 1
        return False

    def kept_indices(self, seq_len: int) -> list[int]:
        """Simulate for reporting — approximate."""
        # Without actual K values, use stride-based estimate
        return list(range(seq_len))

    def reset(self):
        self._last_K = None
        self._gap = 0


# ═══════════════════════════════════════════════════════════════════════
# Keyframe Eviction (pairs with DeltaKV)
# ═══════════════════════════════════════════════════════════════════════

class KeyframeEviction(EvictionTracker):
    """Keep keyframes, drop intermediate tokens.

    Designed to pair with DeltaKV: DeltaKV already stores keyframes
    + residuals. KeyframeEviction additionally drops SOME residuals,
    trading precision for additional compression.

    Usage:
        # Keep keyframe + every 2nd residual (drop every other)
        evict = KeyframeEviction(keyframe_stride=8, residual_keep=2)
    """

    def __init__(self, keyframe_stride: int = 8, residual_keep: int = 1):
        """
        Args:
            keyframe_stride: interval between keyframes
            residual_keep:   keep 1 out of every N residuals
                             (1 = keep all, 2 = keep half, etc.)
        """
        self.kf_stride = keyframe_stride
        self.residual_keep = residual_keep

    def should_keep(self, token_id: int,
                    K: torch.Tensor | None = None) -> bool:
        # Always keep keyframes
        if token_id % self.kf_stride == 0:
            return True
        # Keep 1/residual_keep of non-keyframes
        return (token_id % self.kf_stride) % self.residual_keep == 0

    def kept_indices(self, seq_len: int) -> list[int]:
        kept = []
        for t in range(seq_len):
            if self.should_keep(t):
                kept.append(t)
        return kept
