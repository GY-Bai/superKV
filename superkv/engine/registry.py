"""Algorithm registry — discover and manage KV compression algorithms.

Each algorithm implements the KVCompressor protocol and can be
registered once. The registry supports querying available algorithms
and instantiating them by name.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class KVCompressor(Protocol):
    """Protocol that all KV compression algorithms must implement."""

    name: str
    version: str

    def compress(self, K: torch.Tensor, V: torch.Tensor,
                 layer_idx: int = 0,
                 token_id: int | None = None) -> tuple | None:
        """Compress K, V tensors for one attention layer.

        Supports both per-token and batched:
          Per-token: K shape (n_heads, head_dim)
          Batched:   K shape (batch, n_heads, head_dim)
        """
        ...

    def decompress(self, packed, layer_idx: int = 0
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress back to K, V tensors."""
        ...

    def memory_report(self) -> dict:
        """Return a dict with compression stats.
        
        Keys: 'original_bytes', 'compressed_bytes', 'compression_ratio',
              'algorithm', 'version', 'backend' (tilelang/pytorch).
        """
        ...

    def reset(self):
        """Reset internal state (e.g. keyframes)."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[KVCompressor]] = {}


def register_algorithm(cls: type[KVCompressor]) -> type[KVCompressor]:
    """Decorator: register a KVCompressor class."""
    name = getattr(cls, 'name', cls.__name__.lower())
    _REGISTRY[name] = cls
    return cls


def list_algorithms() -> list[str]:
    """Return names of all registered algorithms."""
    # Eagerly import algorithm modules to populate registry
    _ensure_algorithms_loaded()
    return sorted(_REGISTRY.keys())


def get_algorithm(name: str) -> type[KVCompressor]:
    """Get a registered algorithm class by name."""
    _ensure_algorithms_loaded()
    if name not in _REGISTRY:
        raise KeyError(
            f"Algorithm '{name}' not found. Available: {list_algorithms()}")
    return _REGISTRY[name]


def create_compressor(name: str, **kwargs) -> KVCompressor:
    """Instantiate a compressor by algorithm name."""
    cls = get_algorithm(name)
    return cls(**kwargs)


_algorithms_loaded = False


def _ensure_algorithms_loaded():
    global _algorithms_loaded
    if _algorithms_loaded:
        return
    import importlib
    for mod_name in ["superkv.algorithms.deltakv", "superkv.algorithms.turboquant"]:
        try:
            importlib.import_module(mod_name)
        except ImportError:
            pass
    _algorithms_loaded = True
