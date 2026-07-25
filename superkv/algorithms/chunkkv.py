"""ChunkKV — Semantic chunk-based token eviction.

ChunkKV (NeurIPS 2025): groups consecutive tokens into fixed-size
chunks, scores each chunk via attention proxy, keeps top-K chunks.

Our implementation uses K-vector L2 distance as an attention proxy
(no true attention score dependency). This keeps ChunkKV within the
KVCompressor protocol — no hook into attention internals needed.

Layer-wise index reuse: chunk selection decisions from layer 0 are
reused on deeper layers, amortizing scoring cost across the model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import math
from collections import OrderedDict

from superkv.algorithms.eviction import EvictionTracker


class ChunkKVTracker(EvictionTracker):
    """Chunk-based KV cache eviction with attention proxy scoring.

    Algorithm:
      1. Group tokens into fixed-size chunks (e.g., 8 tokens/chunk)
      2. When a chunk is complete, compute its "importance score"
         via K-vector distance from previous chunk's mean K
      3. Keep top-K chunks by score, evict the rest
      4. Layer 0 computes scores; deeper layers reuse the indices

    Usage:
        tracker = ChunkKVTracker(chunk_size=8, top_k=4)
        for each token:
            if tracker.should_keep(token_id, K):
                compressor.compress(K, V, ...)
    """

    def __init__(self, chunk_size: int = 8, top_k: int = 4,
                 layer_reuse: bool = True,
                 adaptive: bool = False,
                 split_threshold: float = 0.95):
        """
        Args:
            chunk_size: base chunk size (max when adaptive=True)
            top_k: number of chunks to keep (after initial buffer)
            layer_reuse: if True, only layer 0 computes scores
            adaptive: if True, split chunks at K-vector mutation points
            split_threshold: cosine similarity below which we split
                             (lower = more aggressive splitting)
        """
        assert chunk_size >= 1 and top_k >= 1
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.layer_reuse = layer_reuse
        self.adaptive = adaptive
        self.split_threshold = split_threshold

        # State
        self._chunk_buffer: list[torch.Tensor] = []
        self._chunk_scores: OrderedDict[int, float] = {}
        self._kept_chunks: set[int] = set()
        self._prev_chunk_mean: torch.Tensor | None = None
        self._next_chunk_id = 0
        self._scored_layers: set[int] = set()
        self._layer_indices: dict[int, set[int]] = {}
        self._min_chunk_size = max(2, chunk_size // 4)  # floor for elastic

    # ── EvictionTracker interface ─────────────────────────────────

    def should_keep(self, token_id: int,
                    K: torch.Tensor | None = None,
                    layer_idx: int = 0) -> bool:
        """Decide whether to keep this token.

        Returns True for:
          - First top_k chunks (warmup, keep everything)
          - Tokens in chunks ranked top-K by score
        """
        # Layer reuse: if layer_idx > 0 and we have cached indices
        if self.layer_reuse and layer_idx > 0 and layer_idx in self._layer_indices:
            chunk_id = token_id // self.chunk_size
            return chunk_id in self._layer_indices[layer_idx]

        # Warmup: keep everything until we have enough chunks to select
        chunk_id = token_id // self.chunk_size
        if chunk_id < self.top_k * 2:
            self._ensure_chunk_scored(token_id, K, layer_idx)
            return True

        # Selection mode: keep only top-K chunks
        self._ensure_chunk_scored(token_id, K, layer_idx)
        self._maybe_prune(layer_idx)

        return chunk_id in self._kept_chunks

    def kept_indices(self, seq_len: int) -> list[int]:
        """Return all kept token indices for a full sequence."""
        kept = []
        for t in range(seq_len):
            chunk_id = t // self.chunk_size
            if chunk_id in self._kept_chunks:
                kept.append(t)
        return kept

    def reset(self):
        self._chunk_buffer.clear()
        self._chunk_scores.clear()
        self._kept_chunks.clear()
        self._prev_chunk_mean = None
        self._next_chunk_id = 0
        self._scored_layers.clear()
        self._layer_indices.clear()

    # ── Internal ─────────────────────────────────────────────────

    def _ensure_chunk_scored(self, token_id: int,
                              K: torch.Tensor | None,
                              layer_idx: int):
        """Score the chunk. With adaptive=True, may split early."""
        chunk_id = token_id // self.chunk_size

        if layer_idx in self._scored_layers and chunk_id in self._chunk_scores:
            return

        if K is None:
            return

        # Elastic split check: K突变 → 结束当前 chunk
        # 固定模式: chunk_size 一到就切
        # 弹性模式: 只有 K 突变或超过安全上限才切
        should_close = False
        if self.adaptive:
            # Elastic: split on mutation or safety cap
            if len(self._chunk_buffer) >= self._min_chunk_size:
                buf_k = torch.stack(self._chunk_buffer + [K])
                current_mean = buf_k.mean(dim=(0, 1))
                if self._prev_chunk_mean is not None:
                    sim = F.cosine_similarity(
                        current_mean.flatten(),
                        self._prev_chunk_mean.flatten(), dim=0)
                    if sim < self.split_threshold:
                        should_close = True  # K突然变了 → 切分
            # Safety cap: don't let a single chunk dominate
            if len(self._chunk_buffer) >= self.chunk_size * 8:
                should_close = True
        else:
            # Fixed: chunk_size tokens → close
            should_close = (token_id % self.chunk_size == self.chunk_size - 1)

        if should_close and len(self._chunk_buffer) > 0:
            self._chunk_buffer.append(K)
            chunk_k = torch.stack(self._chunk_buffer)
            self._chunk_buffer.clear()
            self._score_chunk(chunk_id, chunk_k, layer_idx)
        elif K is not None:
            self._chunk_buffer.append(K)

    def _score_chunk(self, chunk_id: int, chunk_k: torch.Tensor,
                     layer_idx: int):
        """Compute importance score for a chunk.

        Proxy: L2 distance between this chunk's mean K and previous chunk's mean.
        Larger distance = more novel = higher importance.
        """
        # Mean K over tokens and heads
        current_mean = chunk_k.mean(dim=(0, 1))  # (dim,)

        if self._prev_chunk_mean is not None:
            # Score = L2 distance from previous chunk (novelty)
            score = torch.norm(current_mean - self._prev_chunk_mean).item()
        else:
            score = 1.0  # first chunk always kept

        self._chunk_scores[chunk_id] = score
        self._prev_chunk_mean = current_mean

        # Layer reuse: only layer 0 does scoring
        if self.layer_reuse:
            self._scored_layers.add(layer_idx)

    def _maybe_prune(self, layer_idx: int):
        """Select top-K chunks and cache indices for layer reuse."""
        if len(self._chunk_scores) <= self.top_k:
            return

        # Sort chunks by score (descending)
        ranked = sorted(self._chunk_scores.items(),
                        key=lambda x: x[1], reverse=True)
        self._kept_chunks = {chunk_id for chunk_id, _ in ranked[:self.top_k]}

        # Cache for layer reuse
        if self.layer_reuse:
            self._layer_indices[layer_idx] = self._kept_chunks.copy()
            # Propagate to all layers
            for l in range(layer_idx + 1, 99):  # assume max 100 layers
                self._layer_indices[l] = self._kept_chunks.copy()

    @property
    def sparsity(self) -> float:
        """Approximate fraction of tokens kept."""
        if len(self._chunk_scores) == 0:
            return 1.0
        return min(self.top_k / max(len(self._chunk_scores), 1), 1.0)
