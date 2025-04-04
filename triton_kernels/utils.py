import torch

def create_test_matrices(dtype, device):
    M_shapes = [64, 512, 2048]
    N_shapes = [64, 512, 2048]
    K_shapes = [64, 512, 2048]

    matrices = []
    for M in M_shapes:
        for N in N_shapes:
            for K in K_shapes:
                matrices.append(torch.randn(M, K, dtype=dtype, device=device))
                matrices.append(torch.randn(K, N, dtype=dtype, device=device))
    return matrices

def create_test_matrix(dtype, device):
  M_shapes = [64, 512, 2048]
  N_shapes = [64, 512, 2048]

  matrices = []
  for M in M_shapes:
    for N in N_shapes:
      matrices.append(torch.randn(M, N, dtype=dtype, device=device))
  return matrices

def benchmark_kernel(kernels, matrices):
  if not isinstance(kernels, list):
    kernels = [kernels]

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  results = {k.__name__: {} for k in kernels}

  # Warmup
  for matrix in matrices:
    for kernel in kernels:
      for _ in range(5):  # Do 5 warmup iterations
        kernel(matrix)
  torch.cuda.synchronize()

  for matrix in matrices:
    for kernel in kernels:
      times = []
      for _ in range(10):
        start.record()
        kernel(matrix)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
      results[kernel.__name__][f"{matrix.shape}"] = sum(times) / len(times)

  return results

def print_results(results):
    # Get all unique shapes and kernel names
    shapes = set()
    kernel_names = list(results.keys())
    for kernel_results in results.values():
        shapes.update(kernel_results.keys())
    shapes = sorted(list(shapes))

    # Find maximum widths
    max_shape_width = max(len(shape) for shape in shapes)
    max_kernel_width = max(len(kernel) for kernel in kernel_names)

    # Print header
    header = "Shape".ljust(max_shape_width)
    for kernel in kernel_names:
        header += f" | {kernel:>10} (ms) | {'GB/s':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    # Print each matrix shape row
    for shape in shapes:
        # Calculate GB/s once per shape (assuming float32)
        shape_str = shape.replace('torch.Size','').replace('[',',').replace(']','')
        if shape_str.startswith('(,'): # Fix the invalid syntax
            shape_str = '(' + shape_str[2:]
        dims = eval(shape_str)  # This returns a tuple
        numel = 1
        for dim in dims:
            numel *= dim

        row = shape.ljust(max_shape_width)
        for kernel in kernel_names:
            time = results[kernel].get(shape, float('nan'))
            gb_per_sec = numel * 4 * 2 / (time * 1e6) if time != float('nan') else float('nan')
            row += f" | {time:>10.3f} | {gb_per_sec:>10.1f}"
        print(row)
