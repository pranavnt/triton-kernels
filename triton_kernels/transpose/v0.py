"""Matrix Transpose Optimization - V0: Basic PyTorch Implementation

- Using x.t() followed by contiguous()
- Performance: 0.561 ms (956 GB/s)
- Approximately 1/3 of optimal performance
"""

import torch
from triton_kernels.utils import benchmark_kernel, create_test_matrix, print_results

def transpose_v0(x):
    return x.t().contiguous()

if __name__ == "__main__":
  matrices = create_test_matrix(torch.float32, "cuda")
  results = benchmark_kernel(transpose_v0, matrices)
  print_results(results)