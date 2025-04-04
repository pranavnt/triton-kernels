"""GEMM Optimization - V1: Simple Triton GEMM

- Use direct memory access pattern (no advanced tiling)
- Use: `block_size_m`, `block_size_k`, `block_size_n`
"""