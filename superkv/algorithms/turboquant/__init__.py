"""TurboQuant: Random rotation + optimal scalar quantization.

Google ICLR 2026. TurboQuant compresses KV caches using:
  1. Random orthogonal rotation (Hadamard transform)
  2. Optimal scalar quantization per group with learned scales

Status: PLACEHOLDER — to be implemented.
Reference: https://github.com/vllm-project/vllm/pull/38479
"""


@staticmethod
def _placeholder():
    raise NotImplementedError(
        "TurboQuant integration coming in superKV v0.2. "
        "Requires: random Hadamard rotation + asymmetric K/V quantization "
        "kernel. See vLLM PR #38479 for reference implementation."
    )
