"""Matrix Transpose Optimization - V3: Tile-wise Partitioning with Shared Memory

- Load tiles into shared memory and transpose there
- Performance: 312 μs
- Still has bank conflicts in shared memory
"""

import torch
import triton
import triton.language as tl

from triton_kernels.utils import create_test_matrix, benchmark_kernel, print_results
from triton_kernels.transpose.v0 import verify_transpose, transpose_v0
from triton_kernels.transpose.v1 import transpose_v1
from triton_kernels.transpose.v2 import transpose_v2

BLOCK_SIZE = 16

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
  pid_0 = tl.program_id(axis=0)
  pid_1 = tl.program_id(axis=1)


  start_row = pid_0 * BLOCK_SIZE
  start_col = pid_1 * BLOCK_SIZE

  row_offsets = start_row + tl.arange(0, BLOCK_SIZE)
  col_offsets = start_col + tl.arange(0, BLOCK_SIZE)

  row_mask = row_offsets < num_rows
  col_mask = col_offsets < num_cols

  sm = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

  for i in range(BLOCK_SIZE):
    # save stuff to sm
    if start_row + i < num_rows:
      x_ptrs = X_ptr + (start_row + i) * stride_x + col_offsets

      row_data = tl.load(x_ptrs, mask=col_mask)
      sm[i, 0:BLOCK_SIZE] = row_data

  tl.sync()

  for i in range(BLOCK_SIZE):
    # load stuff from sm
    if start_col + i < num_cols:
      col_data = sm[0:BLOCK_SIZE, i]
      y_ptrs = Y_ptr + (start_col + i) * stride_y + row_offsets
      tl.store(y_ptrs, col_data, mask=row_mask)


def transpose_v3(x):
  num_rows, num_cols = x.shape

  stride_x = x.stride(0)
  stride_y = num_rows

  num_row_blocks = (num_rows + BLOCK_SIZE - 1) // BLOCK_SIZE
  num_col_blocks = (num_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

  grid = (num_row_blocks, num_col_blocks,)

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

  kernels = [transpose_v0, transpose_v1, transpose_v2, transpose_v3]
  results = benchmark_kernel(kernels, matrices)
  print_results(results)
