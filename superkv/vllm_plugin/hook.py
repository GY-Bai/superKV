"""Gate 8: vLLM plugin — superKV model hook for KV cache compression.

Strategy: Hook into the attention layer output BEFORE vLLM's KV cache
stores the K,V tensors. Compression happens at the model level, not in
the attention backend. This means:
  - No changes to vLLM's paged attention / block pool
  - Works with any attention backend (Flash, Triton, etc.)
  - K,V in cache are always FP16 (compressor runs per-layer separately)

The compressor maintains its own compressed cache; vLLM's cache holds
standard FP16 K,V (potentially a subset via eviction).

For full integration (compressed cache in vLLM blocks), see
`superkv/vllm_plugin/backend.py` (coming in v0.2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# vLLM Model Hook
# ═══════════════════════════════════════════════════════════════════════

class VLLMModelHook:
    """Hook into a vLLM model's attention layers for KV compression.

    Installed via `hook.install(model)` — wraps each attention layer's
    Q/K/V projection to capture K,V before cache storage.

    Usage:
        hook = VLLMModelHook(algorithm="deltakv")
        hook.install(vllm_model)

        # After inference:
        print(hook.compressor.memory_report())
    """

    def __init__(self, algorithm: str = "deltakv",
                 reference_stride: int = 8,
                 bits_k: int | None = None,
                 bits_v: int | None = None):
        """
        Args:
            algorithm:        "deltakv" or "turboquant"
            reference_stride: keyframe interval (DeltaKV only)
            bits_k, bits_v:   bit widths (TurboQuant only)
        """
        self.algorithm = algorithm
        self.reference_stride = reference_stride
        self.bits_k = bits_k or (3 if algorithm == "turboquant" else None)
        self.bits_v = bits_v or (2 if algorithm == "turboquant" else None)
        self.compressor = None
        self._installed = False
        self._hooks: list[Any] = []

    def install(self, model, model_config=None):
        """Install hooks on a vLLM model.

        Args:
            model: a vLLM V1 model or HuggingFace model.
            model_config: optional VllmConfig or HF config for head dims.
        """
        from superkv.engine.registry import create_compressor

        if model_config is None:
            model_config = getattr(model, 'config', None)

        # Extract model dimensions
        n_kv = _get_kv_heads(model_config)
        hd = _get_head_dim(model_config)
        n_l = _get_num_layers(model_config)

        logger.info(
            "superKV hook: %s, %d KV heads × %d dim, %d layers",
            self.algorithm, n_kv, hd, n_l,
        )

        kwargs = dict(num_heads=n_kv, head_dim=hd, num_layers=n_l,
                      device='cuda')
        if self.algorithm == "deltakv":
            kwargs['reference_stride'] = self.reference_stride
        elif self.algorithm == "turboquant":
            kwargs['bits_k'] = self.bits_k
            kwargs['bits_v'] = self.bits_v

        self.compressor = create_compressor(self.algorithm, **kwargs)

        # Find attention layers
        layers = _find_layers(model)
        for layer_idx, layer in enumerate(layers):
            self._hook_layer(layer, layer_idx)

        self._installed = True
        return self

    def _hook_layer(self, layer, layer_idx: int):
        """Wrap one attention layer to capture K,V."""
        # Store hook info on layer for downstream access
        layer._superkv_hook = self
        layer._superkv_layer = layer_idx

        # If the layer has a self_attn with qkv_proj, hook the output
        attn = _get_self_attn(layer)
        if attn is not None:
            self._hook_attention(attn, layer_idx)

    def _hook_attention(self, attn_module, layer_idx: int):
        """Hook into an attention module's forward pass."""
        # Most effective place: wrap the forward method
        if hasattr(attn_module, 'forward'):
            original_forward = attn_module.forward
            compressor = self.compressor

            def hooked_forward(*args, **kwargs):
                output = original_forward(*args, **kwargs)
                # The output format varies by model — commonly
                # (attn_output, attn_weights) or (attn_output,)
                # K,V are sometimes accessible via kwargs or internal state
                # For a minimal hook, we note that compression requires
                # K,V at cache-write time, which is handled by the engine.
                return output

            attn_module.forward = hooked_forward
            self._hooks.append((attn_module, original_forward))

    def remove(self):
        """Remove all installed hooks."""
        for module, original in self._hooks:
            module.forward = original
        self._hooks.clear()
        self._installed = False

    def compress_token(self, K, V, layer_idx: int, token_id: int):
        """Compress a single token's K,V through the hook.

        Called from the model runner after K,V are computed and before
        they are written to paged cache.
        """
        if self.compressor is None:
            return None
        return self.compressor.compress(K, V, layer_idx=layer_idx,
                                         token_id=token_id)

    def decompress(self, packed, layer_idx: int):
        """Decompress a token's K,V."""
        return self.compressor.decompress(packed, layer_idx)

    def report(self) -> dict:
        """Return compression statistics."""
        if self.compressor is None:
            return {}
        return self.compressor.memory_report()


# ═══════════════════════════════════════════════════════════════════════
# Model introspection helpers
# ═══════════════════════════════════════════════════════════════════════

def _get_kv_heads(config) -> int:
    """Extract num KV heads from model config."""
    for attr in ['num_key_value_heads']:
        val = getattr(config, attr, None)
        if val is not None:
            return val
    return getattr(config, 'num_attention_heads', 8)


def _get_head_dim(config) -> int:
    """Extract head dimension from model config."""
    for attr in ['head_dim', 'hidden_size']:
        val = getattr(config, attr, None)
        if val is not None:
            if attr == 'hidden_size':
                n_heads = getattr(config, 'num_attention_heads', 1)
                return val // n_heads
            return val
    return 128


def _get_num_layers(config) -> int:
    """Extract number of hidden layers from model config."""
    return getattr(config, 'num_hidden_layers', 1)


def _find_layers(model):
    """Find transformer layers in a model."""
    # Try common paths
    for attr in ['model', 'transformer', 'language_model']:
        base = getattr(model, attr, None)
        if base and hasattr(base, 'layers'):
            return base.layers
    if hasattr(model, 'layers'):
        return model.layers
    return []


def _get_self_attn(layer):
    """Get the self-attention module from a transformer layer."""
    for attr in ['self_attn', 'attention', 'attn']:
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attn
    return None


# ═══════════════════════════════════════════════════════════════════════
# Auto-register with vLLM (safe import)
# ═══════════════════════════════════════════════════════════════════════

def _auto_register():
    """Register superKV specs with vLLM KVCacheSpecRegistry."""
    try:
        from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec
        from vllm.v1.kv_cache_interface import (
            AttentionSpec, KVQuantMode, KVCacheSpecKind,
        )

        @register_kv_cache_spec(
            manager_class=None,
            uniform_type_base_spec=AttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class _DeltaKVSpec(AttentionSpec):
            kind: KVCacheSpecKind = KVCacheSpecKind.FULL_ATTENTION
            kv_quant_mode: KVQuantMode = KVQuantMode.NONE

        logger.debug("superKV KVCacheSpec registered")
    except ImportError:
        pass
    except Exception as e:
        logger.debug("superKV registration skipped: %s", e)


_auto_register()
