"""DeltaKV: Residual encoding + Q4_0 quantization + sparse attention.

Core operations:
  - Residual encoding:  delta = current - keyframe
  - Q4_0 quantization:  32-element blocks sharing one fp16 scale
  - Per-head normalization for large value ranges
  - Sparse attention:   top-k token selection via similarity

This is a standalone implementation using PyTorch. tilelang-accelerated
kernels live in superkv/kernels/tilelang/.
"""

import math
import torch


# ── Q4_0 block parameters ──────────────────────────────────────────
Q4_0_BLOCK_SIZE = 32


# ── Keyframe logic ──────────────────────────────────────────────────
def is_keyframe(token_id: int, stride: int) -> bool:
    """Check whether token_id is a keyframe (reference point)."""
    if stride <= 0:
        return True
    return token_id % stride == 0


# ── Residual encode / decode ────────────────────────────────────────
def delta_encode(curr: torch.Tensor, keyframe: torch.Tensor) -> torch.Tensor:
    """Compute residual: delta = current - keyframe."""
    return curr - keyframe


def delta_decode(keyframe: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Reconstruct: current = keyframe + residual."""
    return keyframe + residual


# ── Q4_0 Quantize / Dequantize ─────────────────────────────────────
def quantize_q4_0(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a float32 tensor to Q4_0 format.

    Groups of Q4_0_BLOCK_SIZE elements share one fp16 scale d.
    Values are bucketed into 16 levels (4-bit, symmetric around 8).

    Args:
        x: float32 tensor of any shape. Last dim must be divisible by 32.

    Returns:
        (q: uint8 packed tensor, d: float16 scales tensor)
          q shape: (*, last_dim // 2)  — 2 int4 per byte
          d shape: (*, last_dim // 32) — one scale per block
    """
    assert x.shape[-1] % Q4_0_BLOCK_SIZE == 0, (
        f"Last dim {x.shape[-1]} must be divisible by {Q4_0_BLOCK_SIZE}")
    orig_shape = x.shape

    # Flatten last dim into blocks
    n_blocks = x.numel() // Q4_0_BLOCK_SIZE
    x_flat = x.reshape(n_blocks, Q4_0_BLOCK_SIZE)

    # Per-block d = max(abs(x)) / 7  (range [−7, 7] for 4-bit signed)
    d_val = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
    d_val = d_val.to(torch.float16)

    # Normalize to [-7, 7] and quantize to [1, 15] (0 is unused)
    x_norm = x_flat / d_val
    q_vals = torch.clamp(torch.round(x_norm) + 8, 1, 15).to(torch.uint8)

    # Pack 2 int4 values per byte
    q_packed = torch.zeros(n_blocks, Q4_0_BLOCK_SIZE // 2,
                           dtype=torch.uint8, device=x.device)
    for i in range(0, Q4_0_BLOCK_SIZE, 2):
        q_packed[:, i // 2] = q_vals[:, i] | (q_vals[:, i + 1] << 4)

    # Reshape to match input shape pattern
    d_shape = orig_shape[:-1] + (orig_shape[-1] // Q4_0_BLOCK_SIZE,)
    q_shape = orig_shape[:-1] + (orig_shape[-1] // 2,)
    return q_packed.reshape(q_shape), d_val.reshape(d_shape)


def dequantize_q4_0(q: torch.Tensor, d: torch.Tensor,
                    orig_shape: tuple) -> torch.Tensor:
    """Dequantize Q4_0 format back to float32.

    Args:
        q: packed uint8, shape (*, last_dim // 2)
        d: float16 scales, shape (*, last_dim // 32)
        orig_shape: target float32 tensor shape

    Returns:
        float32 tensor of orig_shape
    """
    n_blocks = d.numel()
    q_flat = q.reshape(n_blocks, Q4_0_BLOCK_SIZE // 2)
    d_flat = d.reshape(n_blocks, 1)

    # Unpack nibbles
    q_low = q_flat & 0x0F
    q_high = q_flat >> 4

    # Interleave low/high: [l0,h0, l1,h1, ...]
    q_vals = torch.empty(n_blocks, Q4_0_BLOCK_SIZE, dtype=torch.uint8)
    q_vals[:, 0::2] = q_low
    q_vals[:, 1::2] = q_high

    return ((q_vals.float() - 8) * d_flat.float()).reshape(orig_shape)


# ── Combined Delta + Q4_0 ───────────────────────────────────────────
def delta_encode_q4_0(curr: torch.Tensor, keyframe: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual encode + Q4_0 quantize."""
    residual = delta_encode(curr, keyframe)
    return quantize_q4_0(residual)


def delta_decode_q4_0(keyframe: torch.Tensor,
                      q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Q4_0 dequantize + reconstruct from keyframe."""
    residual = dequantize_q4_0(q, d, keyframe.shape)
    return delta_decode(keyframe, residual)


# ── Per-head normalized variant (handles large value ranges) ────────
def delta_encode_q4_0_normalized(curr: torch.Tensor, keyframe: torch.Tensor
                                  ) -> tuple[torch.Tensor, torch.Tensor,
                                             torch.Tensor]:
    """Per-head normalized residual encode + Q4_0.

    Normalizes each head to [-1, 1] before quantization to avoid
    precision loss on real-model K values (range [-322, 329]).

    Returns:
        (q_packed, d_half, head_scale): Q4_0 data + per-head scale.
    """
    scale = keyframe.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    curr_norm = curr / scale
    kf_norm = keyframe / scale
    q, d = delta_encode_q4_0(curr_norm, kf_norm)
    return q, d, scale.squeeze(-1)


def delta_decode_q4_0_normalized(keyframe: torch.Tensor,
                                  q: torch.Tensor, d: torch.Tensor,
                                  head_scale: torch.Tensor) -> torch.Tensor:
    """Per-head normalized Q4_0 decode."""
    scale_exp = head_scale.unsqueeze(-1)
    residual = dequantize_q4_0(q, d, (keyframe.shape[0], keyframe.shape[1]))
    recon_norm = delta_decode(keyframe / scale_exp, residual)
    return recon_norm * scale_exp


# ── Sparse attention ────────────────────────────────────────────────
def sparse_attention(Q: torch.Tensor, K_sparse: torch.Tensor,
                     V_sparse: torch.Tensor,
                     scale: float | None = None) -> torch.Tensor:
    """Compute attention over a sparse subset of KV tokens.

    Args:
        Q: query, shape (n_heads, head_dim)
        K_sparse: selected keys, shape (k, n_heads, head_dim)
        V_sparse: selected values, shape (k, n_heads, head_dim)
        scale: custom scale (default: 1/sqrt(head_dim))

    Returns:
        (n_heads, head_dim) attention output.
    """
    if scale is None:
        scale = 1.0 / math.sqrt(Q.shape[-1])
    scores = torch.einsum('hd,khd->hk', Q, K_sparse) * scale
    weights = torch.softmax(scores, dim=-1)
    return torch.einsum('hk,khd->hd', weights, V_sparse)


# ── INT8 delta encoding (for K, which has large range) ──────────────

def delta_encode_int8(curr: torch.Tensor, keyframe: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head INT8 delta encoding.

    K residuals have smaller range than K itself (delta_K max_abs ~50
    vs K max_abs ~200), making INT8 accurate enough for reconstruction.

    Args:
        curr:     current K, shape (n_heads, head_dim)
        keyframe: reference K, same shape

    Returns:
        delta_int8: (n_heads, head_dim) int8
        scale:      (n_heads,) float32 — per-head scale for decoding
    """
    delta = curr - keyframe
    scale = delta.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    delta_int8 = torch.clamp(torch.round(delta / scale), -128, 127).to(torch.int8)
    return delta_int8, scale.squeeze(-1)


def delta_decode_int8(keyframe: torch.Tensor,
                      delta_int8: torch.Tensor,
                      scale: torch.Tensor) -> torch.Tensor:
    """Decode INT8 delta: keyframe + delta_int8 * scale."""
    return keyframe + delta_int8.float() * scale.unsqueeze(-1)
