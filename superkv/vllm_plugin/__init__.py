"""superKV vLLM V1 plugin.

Provides:
  - VLLMModelHook: model-level KV cache compression via attention layer hooks
  - KVCacheSpec registration: registered with vLLM's spec registry

Usage:
    import superkv.vllm_plugin  # auto-registers

    # Then attach hook to model:
    from superkv.vllm_plugin import VLLMModelHook
    hook = VLLMModelHook(algorithm="deltakv")
    hook.install(model)
    # model now compresses KV cache via superKV
"""

from superkv.vllm_plugin.hook import VLLMModelHook

__all__ = ["VLLMModelHook"]
