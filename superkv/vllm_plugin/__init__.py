"""vLLM V1 plugin — register superKV algorithms into vLLM's KV cache pipeline.

Usage:
    # In vLLM startup or config:
    import superkv.vllm_plugin  # registers DeltaKVSpec, TurboQuantSpec, etc.

    # Then launch vLLM with:
    vllm serve model --kv-cache-dtype deltakv_q4_0
"""

# The registration is triggered on import.
# Each algorithm gets its own KVCacheSpec registered via @register_kv_cache_spec.

def _register_all():
    """Register all superKV KVCacheSpec types with vLLM's registry."""
    try:
        from vllm.v1.kv_cache_spec_registry import register_kv_cache_spec
        from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheSpecKind
        from dataclasses import dataclass

        @register_kv_cache_spec(
            manager_class=None,  # TOOD: DeltaKVManager (Gate 8)
            uniform_type_base_spec=AttentionSpec,
        )
        @dataclass(frozen=True, kw_only=True)
        class DeltaKVAttentionSpec(AttentionSpec):
            """DeltaKV residual-compressed KV cache spec."""
            kind: KVCacheSpecKind = KVCacheSpecKind.FULL_ATTENTION
            reference_stride: int = 8
            top_k_budget: int = 256

    except ImportError:
        pass  # vLLM not installed — skip registration
    except Exception as e:
        import warnings
        warnings.warn(f"superKV vLLM plugin registration failed: {e}")


# Auto-register on import
_register_all()
