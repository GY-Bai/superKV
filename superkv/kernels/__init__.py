"""TileLang-accelerated kernels for KV compression algorithms.

Each kernel is compiled once per shape via @tilelang.jit and cached.
Fallback to PyTorch when tilelang is not installed or compilation fails.
"""
