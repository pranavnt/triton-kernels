"""Matrix Transpose Optimization - V2: Coalesced Global Memory Access

- Improved loading pattern for contiguous memory access
- Performance: 1.40 ms
- Better, but still suffers from uncoalesced writes
"""

import torch
import triton
import triton.language as tl

import pdb

from triton_kernels.utils import create_test_matrix, benchmark_kernel, print_results
from triton_kernels.transpose.v0 import verify_transpose, transpose_v0
from triton_kernels.transpose.v1 import transpose_v1

BLOCK_SIZE = 32

@triton.jit
def transpose_kernel(
  X_ptr,
  Y_ptr,
  num_rows,
  num_cols,
  stride_x,
  stride_y,
  BLOCK_SIZE: tl.constexpr,
):
  ...

def transpose_v2(x):
  ...

if __name__ == "__main__":
  matrices = create_test_matrix(torch.float32, torch.device("cuda"))

  kernels = [transpose_v2]
  results = benchmark_kernel(kernels, matrices)
  print_results(results)
