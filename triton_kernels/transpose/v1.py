"""Matrix Transpose Optimization - V1: Row-wise Partitioning

- Each thread block handles a row of the matrix
- Performance: 3.65 ms
- Poor performance due to uncoalesced memory accesses
"""

import torch
import triton
import triton.language as tl

import pdb

from triton_kernels.utils import create_test_matrix, benchmark_kernel, print_results
from triton_kernels.transpose.v0 import verify_transpose, transpose_v0

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
  pid = tl.program_id(0)

  num_col_blocks = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
  row_idx = pid // num_col_blocks
  col_block_idx = pid % num_col_blocks

  col_start = col_block_idx * BLOCK_SIZE
  col_offsets = col_start + tl.arange(0, BLOCK_SIZE)
  col_mask = col_offsets < num_cols

  x_ptr = X_ptr + row_idx * stride_x + col_offsets
  row_chunk = tl.load(x_ptr, mask=col_mask)

  y_ptr = Y_ptr + col_offsets * stride_y + row_idx

  tl.store(y_ptr, row_chunk, mask=col_mask)

def transpose_v1(x):
    num_rows, num_cols = x.shape

    stride_x = x.stride(0)
    stride_y = num_rows

    num_col_blocks = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (num_rows * num_col_blocks,)

    output = torch.empty((num_cols, num_rows), device=x.device, dtype=x.dtype)

    transpose_kernel[grid](
      x, output,
      num_rows, num_cols,
      stride_x, stride_y,
      BLOCK_SIZE=BLOCK_SIZE
    )

    return output

if __name__ == "__main__":
  matrices = create_test_matrix(torch.float32, torch.device("cuda"))

  # for matrix in [torch.randn(64, 64, device="cuda")]:
  for matrix in matrices:
    correct = matrix.t().contiguous()
    y = transpose_v1(matrix)
    print(f"Checking matrix with shape: {matrix.shape}")
    print("Max diff: ", torch.max(torch.abs(correct - y)))
    print("avg diff: ", torch.mean(torch.abs(correct - y)))
    print("allclose: ", torch.allclose(correct, y))
    # break

  # benchmark
  kernels = [transpose_v1]
  results = benchmark_kernel(kernels, matrices)
  print_results(results)
