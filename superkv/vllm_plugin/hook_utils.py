"""Hook utilities — attention detection, model dims extraction.

Shared between hook.py and hook_v2.py.
"""

from __future__ import annotations

import types


def detect_attention_type(model, config=None) -> tuple[str, str]:
    """Detect attention type. Returns (type, reason)."""
    if config is None:
        config = getattr(model, 'config', None)

    arch = _get_arch(config)
    model_type = getattr(config, 'model_type', '').lower() if config else ''

    linear_patterns = ['gated_deltanet', 'deltanet', 'linear_attention',
                       'gated_linear', 'mamba2', 'hawk', 'rwkv']
    for pat in linear_patterns:
        if pat in arch.lower() or pat in model_type:
            return ('linear', f'config: {arch}')
    if 'deepseek' in arch.lower() or 'mla' in arch.lower():
        return ('mla', f'config: {arch}')
    if 'mamba' in arch.lower() or 'ssm' in arch.lower():
        return ('mamba', f'config: {arch}')

    layers = _find_layers(model)
    if layers:
        attn = _get_self_attn(layers[0])
        if attn is not None:
            has_k = hasattr(attn, 'k_proj')
            has_v = hasattr(attn, 'v_proj')
            if not has_k and not has_v:
                cls_name = attn.__class__.__name__.lower()
                for p in ['linear', 'delta', 'gated']:
                    if p in cls_name:
                        return ('linear', f'layer: {attn.__class__.__name__}')
                return ('unknown', 'no k_proj/v_proj')
            cls_name = attn.__class__.__name__
            if 'LinearAttention' in cls_name or 'DeltaNet' in cls_name:
                return ('linear', f'layer: {cls_name}')
            return ('traditional', 'has k_proj/v_proj')

    if arch and 'ForCausalLM' in arch:
        return ('traditional', f'inferred from {arch}')
    return ('unknown', 'could not inspect')


def extract_model_dims(config) -> dict:
    """Extract {n_kv, hd, n_layers} from model config."""
    n_kv = getattr(config, 'num_key_value_heads', None)
    if n_kv is None:
        n_kv = getattr(config, 'num_attention_heads', 8)
    hd = getattr(config, 'head_dim', None)
    if hd is None:
        hd = getattr(config, 'hidden_size', 128) // n_kv
    n_l = getattr(config, 'num_hidden_layers', 1)
    return {'n_kv': n_kv, 'hd': hd, 'n_layers': n_l}


def find_attention_modules(model) -> list[tuple[int, any]]:
    """Return list of (layer_idx, attention_module)."""
    layers = _find_layers(model)
    result = []
    for i, layer in enumerate(layers):
        attn = _get_self_attn(layer)
        if attn is not None:
            result.append((i, attn))
    return result


def _get_arch(config) -> str:
    if config is None:
        return ''
    archs = getattr(config, 'architectures', None)
    if archs and isinstance(archs, list) and len(archs) > 0:
        return archs[0]
    return getattr(config, 'model_type', '')


def _find_layers(model):
    for attr in ['model', 'transformer', 'language_model']:
        base = getattr(model, attr, None)
        if base and hasattr(base, 'layers'):
            return base.layers
    if hasattr(model, 'layers'):
        return model.layers
    return []


def _get_self_attn(layer):
    for attr in ['self_attn', 'attention', 'attn']:
        attn = getattr(layer, attr, None)
        if attn is not None:
            return attn
    return None
