"""Matrix Transpose Optimization - V1: Row-wise Partitioning

- Each thread block handles a row of the matrix
- Performance: 3.65 ms
- Poor performance due to uncoalesced memory accesses
"""

import torch
import triton
import triton.language as tl

from triton_kernels.utils import create_test_matrix, benchmark_kernel, print_results
from triton_kernels.transpose.v0 import verify_transpose

BLOCK_SIZE = 64

@triton.jit
def transpose_kernel(
  X_ptr,
  Y_ptr,
  num_rows,
  num_cols,
  BLOCK_SIZE: tl.constexpr,
):
  pid = tl.program_id(0)

  block_start = pid * BLOCK_SIZE
  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < num_rows
  x = tl.load(X_ptr + offsets[mask], mask=mask)

  offsets = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offsets < num_cols
  tl.store(Y_ptr + offsets[mask], x, mask=mask)

def transpose(x):
  num_rows, num_cols = x.shape
  output = torch.empty((num_cols, num_rows), device=x.device, dtype=x.dtype)
  grid = torch.zeros(triton.cdiv(num_rows, BLOCK_SIZE), device=x.device)
  transpose_kernel[grid](x, output, num_rows, num_cols, BLOCK_SIZE=BLOCK_SIZE)
  return output

if __name__ == "__main__":
  matrices = create_test_matrix(torch.float32, torch.device("cuda"))

  for matrix in matrices:
    correct = matrix.t().contiguous()
    y = transpose(matrix)
    assert torch.allclose(correct, y)

  # kernels = [transpose]
  # results = benchmark_kernel(kernels, matrices)
  # print_results(results)

  x
