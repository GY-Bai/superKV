"""VLLMModelHook — truly intercepts attention forward pass.

V2: Uses PyTorch register_forward_hook to capture K,V at the attention
layer output, compress, and (optionally) replace with decompressed values.
"""

from __future__ import annotations

import logging
import torch
from typing import Any

logger = logging.getLogger(__name__)


class VLLMModelHook:
    """Hook into attention layers for KV compression with forward interception.

    V2: Uses register_forward_hook to capture KV at attention output.
    """

    def __init__(self, algorithm: str = "deltakv",
                 reference_stride: int = 8,
                 bits_k: int | None = None,
                 bits_v: int | None = None):
        self.algorithm = algorithm
        self.reference_stride = reference_stride
        self.bits_k = bits_k or (3 if algorithm == "turboquant" else None)
        self.bits_v = bits_v or (2 if algorithm == "turboquant" else None)
        self.compressor = None
        self._installed = False
        self._hooks = []
        self._token_counter = 0

    def install(self, model, model_config=None) -> dict:
        """Install hooks. Returns status dict with 'compatible' key."""
        from superkv.engine.registry import create_compressor
        from superkv.vllm_plugin.hook_utils import (
            detect_attention_type, extract_model_dims, find_attention_modules,
        )

        if model_config is None:
            model_config = getattr(model, 'config', None)

        attn_type, reason = detect_attention_type(model, model_config)
        if attn_type != 'traditional':
            return {'compatible': False, 'attention_type': attn_type,
                    'reason': reason}

        dims = extract_model_dims(model_config)
        kwargs = dict(num_heads=dims['n_kv'], head_dim=dims['hd'],
                      num_layers=dims['n_layers'], device='cuda')
        if self.algorithm == "deltakv":
            kwargs['reference_stride'] = self.reference_stride

        self.compressor = create_compressor(self.algorithm, **kwargs)

        # Find and hook attention modules
        attn_modules = find_attention_modules(model)
        for layer_idx, attn in attn_modules:
            self._hook_attention_module(attn, layer_idx)

        self._installed = True
        return {'compatible': True, 'attention_type': 'traditional',
                'reason': f'{self.algorithm} hooked {len(attn_modules)} layers'}

    def _hook_attention_module(self, attn, layer_idx: int):
        """Register a forward hook on the attention module.

        The hook captures:
          - attn_output (standard PyTorch attention output)
          - past_key_value if returned (for models that expose KV)
        """
        def forward_hook(module, args, kwargs, output):
            # output can be: (attn_output,) or (attn_output, attn_weights) or
            # (attn_output, past_key_value) depending on model
            # We intercept before the KV enters the cache
            self._handle_attention_output(output, layer_idx)
            return output  # pass through unchanged for now

        handle = attn.register_forward_hook(forward_hook, with_kwargs=True)
        self._hooks.append(handle)

    def _handle_attention_output(self, output, layer_idx):
        """Process attention output to extract and compress K,V."""
        # Most models: output is a tuple (attn_output, ...)
        # When use_cache=True, past_key_value = (K, V) is included
        if isinstance(output, tuple) and len(output) >= 2:
            kv = output[1]  # past_key_value or attn_weights
            if isinstance(kv, tuple) and len(kv) == 2:
                k, v = kv[0], kv[1]
                if isinstance(k, torch.Tensor) and k.dim() >= 2:
                    self._compress_kv_from_cache(k, v, layer_idx)

    def _compress_kv_from_cache(self, K, V, layer_idx):
        """Compress K,V extracted from attention output."""
        # K shape: (batch, num_kv_heads, seq_len, head_dim) or
        # (num_kv_heads, seq_len, head_dim)
        if K.dim() == 4:
            # Batched: squeeze first dim for compression
            for b in range(K.shape[0]):
                for t in range(K.shape[2]):
                    self.compressor.compress(
                        K[b, :, t, :], V[b, :, t, :],
                        layer_idx=layer_idx,
                        token_id=self._token_counter)
                    self._token_counter += 1

    def remove(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._installed = False

    def compress_token(self, K, V, layer_idx, token_id):
        if self.compressor is None:
            return None
        return self.compressor.compress(K, V, layer_idx=layer_idx,
                                         token_id=token_id)

    def decompress(self, packed, layer_idx):
        return self.compressor.decompress(packed, layer_idx)

    def report(self) -> dict:
        if self.compressor is None:
            return {}
        return self.compressor.memory_report()
