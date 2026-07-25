"""Tests for vLLM plugin integration.

These tests use HuggingFace transformers as a stand-in for vLLM
(since vLLM requires CUDA). The hook mechanism is identical — it
wraps attention layers to intercept K,V.

Run:  uv run pytest tests/test_vllm_plugin.py -v
"""

import torch
import pytest
import types


def _has_transformers():
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


requires_transformers = pytest.mark.skipif(
    not _has_transformers(), reason="transformers not installed")


# ═══════════════════════════════════════════════════════════════════════
# Hook installation
# ═══════════════════════════════════════════════════════════════════════

class TestVLLMModelHook:
    def test_create_hook(self):
        from superkv.vllm_plugin import VLLMModelHook
        hook = VLLMModelHook(algorithm="deltakv")
        assert hook.algorithm == "deltakv"
        assert hook.reference_stride == 8
        assert not hook._installed

    def test_create_turboquant_hook(self):
        from superkv.vllm_plugin import VLLMModelHook
        hook = VLLMModelHook(algorithm="turboquant", bits_k=3, bits_v=2)
        assert hook.algorithm == "turboquant"
        assert hook.bits_k == 3

    def test_install_no_model(self):
        """Should handle missing model gracefully."""
        from superkv.vllm_plugin import VLLMModelHook
        hook = VLLMModelHook()
        # Creating a hook without a model should not error
        assert hook.compressor is None

    def test_config_extraction(self):
        """Test extracting model dimensions from config."""
        from superkv.vllm_plugin.hook import (
            _get_kv_heads, _get_head_dim, _get_num_layers,
        )

        class FakeConfig:
            num_key_value_heads = 8
            num_attention_heads = 32
            head_dim = 128
            num_hidden_layers = 36

        cfg = FakeConfig()
        assert _get_kv_heads(cfg) == 8
        assert _get_head_dim(cfg) == 128
        assert _get_num_layers(cfg) == 36

    def test_config_fallback(self):
        """Config without KV heads should fallback to attention heads."""
        from superkv.vllm_plugin.hook import _get_kv_heads

        class FakeConfig:
            num_attention_heads = 16

        assert _get_kv_heads(FakeConfig()) == 16

    @pytest.mark.skip(reason="requires CUDA — transformers import broken on M2+Py3.13")
    def test_install_on_tiny_model(self):
        """Install hook on a tiny random model."""
        from transformers import AutoConfig, AutoModelForCausalLM
        from superkv.vllm_plugin import VLLMModelHook

        config = AutoConfig.for_model(
            model_type="llama",
            vocab_size=100, hidden_size=64,
            intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2,
        )
        model = AutoModelForCausalLM.from_config(config)
        hook = VLLMModelHook(algorithm="deltakv")
        hook.install(model)

        assert hook._installed
        assert hook.compressor is not None
        assert hook.compressor.name == "deltakv"

        # Cleanup
        hook.remove()
        assert not hook._installed

    def test_remove_before_install(self):
        """Remove should be safe even without install."""
        from superkv.vllm_plugin import VLLMModelHook
        hook = VLLMModelHook()
        hook.remove()  # should not error

    def test_report_before_install(self):
        """Report should return empty dict before install."""
        from superkv.vllm_plugin import VLLMModelHook
        hook = VLLMModelHook()
        assert hook.report() == {}

    def test_compress_without_install(self):
        """Compress should return None without compressor."""
        from superkv.vllm_plugin import VLLMModelHook
        hook = VLLMModelHook()
        result = hook.compress_token(
            torch.randn(4, 64), torch.randn(4, 64), 0, 0)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════
# Layer finding
# ═══════════════════════════════════════════════════════════════════════

class TestLayerFinding:
    def test_find_layers_standard(self):
        from superkv.vllm_plugin.hook import _find_layers

        class FakeModel:
            model = types.SimpleNamespace()
            model.layers = [1, 2, 3]

        assert len(_find_layers(FakeModel())) == 3

    def test_find_layers_direct(self):
        from superkv.vllm_plugin.hook import _find_layers

        class FakeModel:
            layers = [1, 2]

        assert len(_find_layers(FakeModel())) == 2

    def test_find_layers_empty(self):
        from superkv.vllm_plugin.hook import _find_layers

        class FakeModel:
            pass

        assert _find_layers(FakeModel()) == []

    def test_get_self_attn(self):
        from superkv.vllm_plugin.hook import _get_self_attn

        class FakeLayer:
            self_attn = "attn_module"

        assert _get_self_attn(FakeLayer()) == "attn_module"

    def test_get_self_attn_not_found(self):
        from superkv.vllm_plugin.hook import _get_self_attn

        class FakeLayer:
            pass

        assert _get_self_attn(FakeLayer()) is None
