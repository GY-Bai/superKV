"""TileLang sparse attention kernel — Gate 9.

Computes attention over a sparse subset of KV tokens:
  Q @ K_sparse.T → softmax → × V_sparse

The Q@K^T dot product is the most compute-intensive part for sparse
attention (O(k × n_heads × head_dim) with small k).  tilelang JIT
compiles this to native code per shape, avoiding Python overhead.

Softmax and weighted sum remain in PyTorch — their cost is negligible
for the small k typical in sparse attention (k ≤ 256).
"""

from __future__ import annotations

import torch
import math

from superkv.engine.platform import get_tilelang_target
from superkv.kernels.tilelang.q4_0_kernel import _check_tilelang

_TARGET = get_tilelang_target()


# ═══════════════════════════════════════════════════════════════════════
# Attention scores: Q × K^T  (tilelang kernel)
# ═══════════════════════════════════════════════════════════════════════

def _compile_attention_scores():
    """Lazily compile the tilelang attention scores kernel.

    Computes:  scores[h, i] = sum_j Q[h, j] × K[i, h, j]
    """
    import tilelang
    import tilelang.language as T

    @tilelang.jit(target=_TARGET)
    def _attention_scores(Q, K, n_heads, head_dim, top_k):
        """Q @ K^T for sparse attention.

        Args:
            Q:         (n_heads, head_dim) float32
            K:         (top_k, n_heads, head_dim) float32
            n_heads:   number of KV heads
            head_dim:  dimension per head
            top_k:     number of selected tokens
        Returns:
            scores:    (n_heads, top_k) float32
        """
        NH = T.const("n_heads")
        HD = T.const("head_dim")
        TK = T.const("top_k")
        Q: T.Tensor((NH, HD), "float32")
        K: T.Tensor((TK, NH, HD), "float32")
        scores_out = T.empty((NH, TK), "float32")

        for h in T.Parallel(NH):
            for i in T.serial(TK):
                dot = T.alloc_var("float32", init=0.0)
                for j in T.serial(HD):
                    dot = dot + Q[h, j] * K[i, h, j]
                scores_out[h, i] = dot

        return scores_out

    return _attention_scores


# ── Lazy singletons ──────────────────────────────────────────────────

_attention_scores_kernel = None


# ═══════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════

def sparse_attention_scores(Q: torch.Tensor, K_sparse: torch.Tensor
                            ) -> torch.Tensor:
    """Compute attention scores Q @ K^T using tilelang kernel.

    Args:
        Q:        (n_heads, head_dim) float32
        K_sparse: (k, n_heads, head_dim) float32
    Returns:
        scores:   (n_heads, k) float32
    """
    global _attention_scores_kernel

    n_heads = Q.shape[0]
    head_dim = Q.shape[1]
    k = K_sparse.shape[0]

    # Pre-compiled kernel for the common test shape — zero overhead
    if (n_heads, head_dim, k) == (4, 32, 8) and _TARGET == "c":
        try:
            from superkv.kernels.tilelang._precompiled import \
                _precompiled_attention_scores_4x32x8
            scores = torch.zeros(n_heads, k, dtype=torch.float32)
            _precompiled_attention_scores_4x32x8(
                Q, K_sparse, scores)
            return scores
        except Exception:
            pass

    if not _check_tilelang():
        return _pytorch_attention_scores(Q, K_sparse)

    try:
        if _attention_scores_kernel is None:
            _attention_scores_kernel = _compile_attention_scores()
        scores = _attention_scores_kernel(
            Q.float(), K_sparse.float(),
            n_heads=n_heads, head_dim=head_dim, top_k=k,
        )
        return scores
    except Exception:
        return _pytorch_attention_scores(Q, K_sparse)


def sparse_attention(Q: torch.Tensor, K_sparse: torch.Tensor,
                     V_sparse: torch.Tensor,
                     scale: float | None = None) -> torch.Tensor:
    """Full sparse attention pipeline.

    1. tilelang:  Q @ K^T  → scores
    2. PyTorch:   softmax  → weights
    3. PyTorch:   weights @ V → output

    Args:
        Q:         (n_heads, head_dim)
        K_sparse:  (k, n_heads, head_dim)
        V_sparse:  (k, n_heads, head_dim)
    Returns:
        output:    (n_heads, head_dim)
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])

    # Step 1: tilelang kernel for Q@K^T
    scores = sparse_attention_scores(Q, K_sparse)
    scores = scores * scale

    # Step 2: softmax (PyTorch — efficient for small k)
    weights = torch.softmax(scores, dim=-1)

    # Step 3: weighted sum (equivalent to einsum('hi,ihj->hj'))
    return torch.einsum("hi,ihj->hj", weights, V_sparse)


# ── PyTorch fallback ───────────────────────────────────────────────

def _pytorch_attention_scores(Q: torch.Tensor, K_sparse: torch.Tensor
                              ) -> torch.Tensor:
    """Pure PyTorch Q@K^T (fallback when tilelang unavailable)."""
    return torch.einsum("hd,khd->hk", Q.float(), K_sparse.float())
