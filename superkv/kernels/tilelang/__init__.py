"""TileLang kernel module — Q4_0 quantize and sparse attention."""

from superkv.engine.platform import get_tilelang_target

_TARGET = get_tilelang_target()


def get_q4_0_quantize_kernel():
    """Lazily compile a tilelang Q4_0 quantize kernel for the current platform.

    Returns a callable (tensor) -> (q_packed, d).
    Falls back to superkv.algorithms.deltakv.core.quantize_q4_0 on error.
    """
    try:
        import tilelang
        # TOOD: implement tilelang Q4_0 kernel
        raise NotImplementedError("tilelang Q4_0 kernel coming in v0.2")
    except Exception:
        from superkv.algorithms.deltakv.core import quantize_q4_0
        return quantize_q4_0


def get_sparse_attention_kernel():
    """Lazily compile a tilelang sparse attention kernel.

    Returns a callable (Q, K_sparse, V_sparse) -> output.
    Falls back to PyTorch einsum on error.
    """
    try:
        import tilelang
        # TOOD: implement tilelang sparse attention kernel
        raise NotImplementedError("tilelang attention kernel coming in v0.2")
    except Exception:
        from superkv.algorithms.deltakv.core import sparse_attention
        return sparse_attention
