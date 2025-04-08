# Programming GPUs with Triton
## GPU Programming
### GPU Memory Hierarchy
The GPU memory system consists of several layers with different characteristics:
- Register file: ~600 TB/s bandwidth
- Shared memory ("Smem"): ~20 TB/s bandwidth, 228 KB per SM
- L1/L2 cache: ~10 TB/s bandwidth
- Global memory (HBM): 3 TB/s bandwidth, typically 80+ GB
![[CleanShot 2025-04-02 at 08.58.23@2x.png]]
### GPU Programming Model

| Concept       | Definition             | Corresponding Architecture | Communication    | Limits                        |
| ------------- | ---------------------- | -------------------------- | ---------------- | ----------------------------- |
| Thread        | Minimal execution unit | Function units             | Local            | Up to 255 registers           |
| Warp          | Group of 32 threads    | "SM tiles"                 | Register File    | 32 threads                    |
| Thread Blocks | Group of warps         | SM                         | Shared Memory    | Up to 32 warps (1024 threads) |
| Kernel        | Function on GPU        | GPU                        | L2/Global memory | Up to (2^32-1)³ Blocks        |

Visualization:
![[CleanShot 2025-04-02 at 08.58.49@2x.png]]

**Threads and Warps**
- Each thread uses functional units to perform work
- 32 threads form a warp, which runs in parallel
- Threads in a warp execute the same instructions at the same pace
- 4 warps can run on one SM simultaneously
- The scheduler swaps warps on and off SMs

**Thread Blocks**
- Blocks are groups of warps
- Blocks can be scheduled to SMs
- Within blocks, warps communicate through shared memory

**Kernels**
- A kernel consists of multiple thread blocks
- A kernel can launch more thread blocks than available SMs
- Block-to-block communication must use L2/Global memory
- Blocks should operate independently

Modern NVIDIA GPUs include specialized hardware units called Tensor Cores that accelerate matrix multiplication:
- A tensor core performs a small shape GEMM operation
- A warp (32 threads) collectively uses the tensor core
- Depending on the precision format, tensor cores can provide 8-256× speedup compared to traditional FP32 CUDA cores

Modern GPUs continue to evolve with new features:
- Unified memory address (P100+)
- NVLink for high-speed GPU interconnects (P100+)
- GPU clusters (H100+)
- Tensor Memory Accelerator (TMA) (H100+)
- NVSHARP for higher precision (H100+)
- FP4 and FP6 numerical formats (B100+)
### GPU Kernel Optimization Techniques
**Coalesced Memory Loading** Inside one warp, if memory access addresses are contiguous, the memory access is coalesced (batched). This allows data to be retrieved from global memory in one or a few transactions, significantly improving performance.

**Shared Memory**: Shared memory is much faster than global memory, with approximately 20 TB/s bandwidth compared to 3 TB/s. Strategic use of shared memory for frequently accessed data can greatly reduce memory bottlenecks.

**Avoiding Bank Conflicts**: Shared memory is divided into banks. When multiple threads in the same warp access the same memory bank, a bank conflict occurs, serializing the accesses and reducing performance. Careful memory access patterns can minimize these conflicts.

**Avoiding Branch Divergence**: When threads in a warp take different paths in conditional statements, the warp must execute all paths, masking out results for threads that don't follow a particular path. This is called branch divergence and significantly reduces performance.
## Triton Programming
Triton is a Python-based language for writing GPU kernels that offers several advantages over CUDA:

1. Higher-level abstraction that maintains performance
2. Automatic memory management and optimization
3. Seamless integration with PyTorch
4. Simplified programming model

Triton programs (kernels) follow this general structure:
```python
@triton.jit
def my_kernel(
    # Pointers to input/output data
    pointer_inputs,
    # Problem size and other scalar parameters
    scalar_inputs,
    # Constants that influence the program's shape
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID (often representing block ID)
    pid = tl.program_id(axis=0)

    # Calculate starting positions and offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create mask for boundary checking
    mask = offsets < scalar_input_size

    # Load data using pointers
    data = tl.load(pointer_input + offsets, mask=mask)

    # Perform computation
    result = compute(data)

    # Store results
    tl.store(pointer_output + offsets, result, mask=mask)
```

The Python function launching the kernel follows this pattern:
```python
def my_function(x: torch.Tensor, y: torch.Tensor):
    # Output tensor allocation
    output = torch.empty_like(x)
    n_elements = output.numel()

    # Define grid (number of blocks)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch kernel
    my_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output
```
#### Memory Hierarchy in Triton
Triton automatically manages different memory types but understanding them is still important:
1. **Register Memory**: Automatically managed by Triton
2. **Shared Memory**: Used for communication between threads in a block
    - Created using `tl.zeros()` or other initialization functions
    - Example: `shared_mem = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)`
3. **Global Memory**: Used for data storage and communication between blocks
    - Accessed via `tl.load()` and `tl.store()`
### Key Triton Functions and Optimization Techniques
#### Data Loading and Storing
```python
# Load from global memory to register (with masking)
data = tl.load(ptr + offsets, mask=mask, other=0.0)

# Store from register to global memory (with masking)
tl.store(ptr + offsets, data, mask=mask)
```
#### Tensor Operations
```python
# Matrix multiplication
C = tl.dot(A, B)  # Performs matrix multiplication

# Element-wise operations
result = tl.exp(x) + tl.log(y)
```
#### Tiling for Matrix Operations
Triton excels at tiled operations. For GEMM (General Matrix Multiplication):
```python
# Pseudocode for tiled 2D convolution in Triton
@triton.jit
def conv2d_kernel(
    input_ptr, filter_ptr, output_ptr,
    input_h, input_w, filter_h, filter_w, output_h, output_w,
    stride_ih, stride_iw, stride_oh, stride_ow,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_W: tl.constexpr
):
    # Get program ID for output block
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)

    # Calculate starting positions in output
    h_start = pid_h * BLOCK_SIZE_H
    w_start = pid_w * BLOCK_SIZE_W

    # Create offsets for output dimensions
    offs_h = h_start + tl.arange(0, BLOCK_SIZE_H)
    offs_w = w_start + tl.arange(0, BLOCK_SIZE_W)

    # Create masks for bounds checking
    mask_h = offs_h < output_h
    mask_w = offs_w < output_w

    # For each pixel in the output block
    for h in range(BLOCK_SIZE_H):
        for w in range(BLOCK_SIZE_W):
            if h + h_start >= output_h or w + w_start >= output_w:
                continue

            # Initialize accumulator for this output pixel
            acc = tl.zeros((1, 1), dtype=tl.float32)

            # Apply the filter by processing input tiles
            for fh in range(filter_h):
                for fw in range(filter_w):
                    # Calculate input coordinates
                    ih = h + h_start + fh
                    iw = w + w_start + fw

                    # Boundary check
                    if ih < input_h and iw < input_w:
                        # Load input and filter values
                        input_val = tl.load(input_ptr + ih * stride_ih + iw * stride_iw)
                        filter_val = tl.load(filter_ptr + fh * filter_w + fw)

                        # Accumulate weighted value
                        acc += input_val * filter_val

            # Store result for this output pixel
            out_pos = (h + h_start) * stride_oh + (w + w_start) * stride_ow
            tl.store(output_ptr + out_pos, acc)
```

Another common tiling pattern is for sliding window operations like blur or pooling:
```python
@triton.jit
def pooling_kernel(
    input_ptr, output_ptr,
    input_h, input_w, output_h, output_w,
    window_h, window_w, stride,
    BLOCK_SIZE: tl.constexpr
):
    # Process output in tiles
    pid = tl.program_id(0)

    # Number of elements per block
    n_elements = BLOCK_SIZE

    # Starting output index for this block
    start_idx = pid * n_elements

    # Offset within the block
    offsets = start_idx + tl.arange(0, n_elements)

    # Calculate 2D positions from linear index
    out_h = offsets // output_w
    out_w = offsets % output_w

    # Mask for valid outputs
    mask = (out_h < output_h) & (out_w < output_w)

    # Initialize output values
    results = tl.zeros((n_elements,), dtype=tl.float32)

    # For each output element in our block
    for i in range(n_elements):
        if not mask[i]:
            continue

        # Calculate input window top-left corner
        in_h_start = out_h[i] * stride
        in_w_start = out_w[i] * stride

        # Initialize for max pooling
        max_val = tl.zeros((1,), dtype=tl.float32) - float('inf')

        # Process the input window in tiles
        for h in range(window_h):
            for w in range(window_w):
                # Calculate input position
                ih = in_h_start + h
                iw = in_w_start + w

                # Boundary check
                if ih < input_h and iw < input_w:
                    # Load input value
                    idx = ih * input_w + iw
                    val = tl.load(input_ptr + idx)

                    # Update max value
                    max_val = tl.maximum(max_val, val)

        # Store result for this output element
        results[i] = max_val

    # Store results for the entire block
    tl.store(output_ptr + offsets, results, mask=mask)
```
#### Optimizing Memory Access Patterns
1. **Coalesced Memory Access**: Ensure adjacent threads access adjacent memory locations:
    ```python
    # Good: Adjacent threads access adjacent memory
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    data = tl.load(ptr + offsets, mask=mask)

    # Bad: Strided access pattern
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * stride
    data = tl.load(ptr + offsets, mask=mask)
    ```
2. **Avoiding Bank Conflicts in Shared Memory**: Use padding techniques:
    ```python
    # With padding to avoid bank conflicts
    BLOCK_SIZE = 32
    PADDED_SIZE = 33  # Add padding to avoid conflicts
    shared_mem = tl.zeros((BLOCK_SIZE, PADDED_SIZE), dtype=tl.float32)

    # Without padding (may have bank conflicts)
    shared_mem = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    ```

### Memory Addressing in Triton
Understanding how to calculate and manipulate memory addresses is crucial for efficient GPU programming with Triton. This section explores the key concepts and patterns for working with memory addresses.
#### Linear Memory and Pointer Arithmetic
Triton, like CUDA, operates on a linear memory model where multi-dimensional arrays are flattened into a 1D address space:
```python
# Accessing a 2D matrix element at [row, col]
element_ptr = base_ptr + row * stride_row + col * stride_col`
```

The stride values represent the memory distance between adjacent elements:
- `stride_row`: Number of bytes/elements to move to the next row
- `stride_col`: Number of bytes/elements to move to the next column (usually 1 for row-major storage)

For tensors with more dimensions:
```python
# Accessing a 3D tensor at [depth, row, col]
element_ptr = base_ptr + depth * stride_depth + row * stride_row + col * stride_col
```
#### Vectorized Memory Access with Triton
Triton provides powerful abstractions for vectorized memory operations:
```python
# Creating a range of indices
indices = tl.arange(0, BLOCK_SIZE)
# [0, 1, ... BLOCK_SIZE]
# base_idx + arange thing
# [base_idx + 0, base_idx + 1, ... BLOCK_SIZE]

# Simple offset calculation for 1D array
offsets = base_idx + indices
elements = tl.load(ptr + offsets, mask=mask)

# 2D matrix indexing with broadcasting
row_indices = row_start + tl.arange(0, BLOCK_SIZE_M)[:, None]  # Add dimension for broadcasting
col_indices = col_start + tl.arange(0, BLOCK_SIZE_N)[None, :]  # Add dimension for broadcasting

# Full matrix addressing
ptrs = base_ptr + row_indices * stride_row + col_indices * stride_col
elements = tl.load(ptrs, mask=mask)
```

#### Working with Strides
Strides determine the memory layout of a tensor. Understanding them is key to efficient memory access:
```python
# Accessing strides from PyTorch tensor
@triton.jit
def kernel(tensor_ptr, stride_0, stride_1, ...):
    # Use strides for addressing

# From Python
tensor = torch.zeros((M, N))
stride_0, stride_1 = tensor.stride(0), tensor.stride(1)
kernel[grid](tensor, stride_0, stride_1, ...)
```

Common stride patterns:
1. **Row-major (C-style)**: Elements in the same row are contiguous
    - `stride_row = N` (width of matrix)
    - `stride_col = 1`
2. **Column-major (Fortran-style)**: Elements in the same column are contiguous
    - `stride_row = 1`
    - `stride_col = M` (height of matrix)
3. **Transposed view**: Created when transposing without copying
    - Original: `stride_row = N`, `stride_col = 1`
    - Transposed: `stride_row = 1`, `stride_col = M`

#### Block-wise Memory Addressing
When working with blocks of data, address calculation follows this pattern:
```python
# Block starting position
block_row_start = block_row_id * BLOCK_SIZE_M
block_col_start = block_col_id * BLOCK_SIZE_N

# Element offsets within block
row_offsets = block_row_start + tl.arange(0, BLOCK_SIZE_M)
col_offsets = block_col_start + tl.arange(0, BLOCK_SIZE_N)

# Create 2D grid of pointers (using broadcasting)
ptrs = base_ptr + row_offsets[:, None] * stride_row + col_offsets[None, :] * stride_col
```
#### Memory Address Coalescing
For optimal performance, adjacent threads should access adjacent memory locations:
```python
# Good: Coalesced access (threads access adjacent memory)
offsets = tl.arange(0, BLOCK_SIZE)  # Each thread gets consecutive offset
coalesced_ptr = base_ptr + block_start + offsets
data = tl.load(coalesced_ptr)

# Bad: Strided access (threads access non-adjacent memory)
strided_ptr = base_ptr + block_start + offsets * large_stride
data = tl.load(strided_ptr)  # Poor performance due to non-coalesced access
```
#### Advanced Pointer Manipulation
Triton provides several ways to create complex memory access patterns:
```python
# Creating a 2D block of pointers with padding for shared memory
@triton.jit
def kernel(..., BLOCK_SIZE: tl.constexpr, PAD: tl.constexpr):
    # Normal loading (row-major, without padding)
    row_offs = tl.arange(0, BLOCK_SIZE)[:, None]
    col_offs = tl.arange(0, BLOCK_SIZE)[None, :]
    global_ptrs = input_ptr + row_offs * stride_row + col_offs * stride_col
    data = tl.load(global_ptrs)

    # Store to shared memory with padding to avoid bank conflicts
    # Each row has BLOCK_SIZE + PAD elements
    for i in range(BLOCK_SIZE):
        for j in range(BLOCK_SIZE):
            sm_idx = i * (BLOCK_SIZE + PAD) + j
            sm_ptr = shared_mem_ptr + sm_idx
            tl.store(sm_ptr, data[i, j])
```

#### Reasoning About Memory Alignment

Aligned memory access is important for performance:
```python
# Check if pointer is aligned to 16 bytes
is_aligned = (ptr % 16) == 0

# Ensure aligned loads by adjusting start pointer
aligned_start = (ptr + 15) // 16 * 16  # Round up to nearest 16-byte boundary

# For vectorized loads, ensure proper alignment
if is_aligned:
    # Use fast vectorized load
    vec_data = tl.load(ptr, mask=mask)
else:
    # Fall back to unaligned access pattern
    individual_data = [tl.load(ptr + i) for i in range(size)]
```
#### Common Pitfalls
1. **Incorrect stride calculation**: Always verify stride values match tensor layout
2. **Out-of-bounds access**: Always use masks for boundary checking
3. **Bank conflicts**: For shared memory, consider padding to avoid conflicts
4. **Non-coalesced access**: Ensure adjacent threads access adjacent memory
5. **Mixed addressing modes**: Be consistent with addressing patterns

By understanding and carefully managing memory addressing, you can significantly improve the performance of your Triton kernels.
### Advanced Techniques
#### Double Buffering
Double buffering overlaps computation and memory operations to hide latency:
```python
@triton.jit
def double_buffer_kernel(...):
    # Initialize two buffers
    buffer_a = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    buffer_b = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Load initial data into first buffer
    buffer_a = tl.load(ptr_a + offsets_0, mask=mask_0)

    # Process data in chunks
    for i in range(1, num_chunks):
        # Load next chunk into second buffer (overlapped with computation)
        buffer_b = tl.load(ptr_a + offsets_i, mask=mask_i)

        # Process data in first buffer
        result_a = compute(buffer_a)
        tl.store(ptr_out + offsets_prev, result_a, mask=mask_prev)

        # Swap buffers
        buffer_a, buffer_b = buffer_b, buffer_a
```

#### Auto-Tuning in Triton
Triton provides auto-tuning capabilities to find optimal configurations:
```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def my_kernel(...):
    # Kernel implementation
```
### Best Practices for Triton Programming
1. **Start Simple**: Begin with a basic implementation before optimizing
2. **Use Block-wise Operations**: Leverage tiling for efficient memory access
3. **Minimize Branching**: Avoid control flow divergence within kernels
4. **Use Proper Data Types**: Match precision requirements with appropriate types
5. **Maximize Occupancy**: Balance resource usage to allow multiple blocks per SM
6. **Profile Early and Often**: Use tools to identify bottlenecks
7. **Leverage Built-in Functions**: Use Triton's optimized operations where possible
8. **Auto-tune Parameters**: Use Triton's auto-tuning for optimal configurations
9. **Consider Data Layout**: Choose appropriate memory layouts for your problem
## Reasoning about Triton Programs
When reasoning through Triton programs visually, you can use a structured approach that helps you understand the execution model and memory access patterns. Here's how you might go about this:
### 1) Start with Block and Thread Organization
Visualize how your computation is divided:
- Draw a grid representing your output (2D typically works well)
- Highlight which parts of the output each thread block processes
- Show how individual threads within a block map to elements

For example, in a GEMM kernel:
```
Output Matrix C (6x6)
+---+---+---+---+---+---+
| B0| B0| B1| B1| B2| B2|  B0 = Block 0
+---+---+---+---+---+---+
| B0| B0| B1| B1| B2| B2|  Each block handles
+---+---+---+---+---+---+  a 2x2 tile
| B3| B3| B4| B4| B5| B5|
+---+---+---+---+---+---+
| B3| B3| B4| B4| B5| B5|
+---+---+---+---+---+---+
| B6| B6| B7| B7| B8| B8|
+---+---+---+---+---+---+
| B6| B6| B7| B7| B8| B8|
+---+---+---+---+---+---+
```
### 2) Visualize Memory Access Patterns
Draw how memory is accessed:
- Show the input tensors and how they're laid out in memory
- Draw arrows from threads to the memory locations they access
- Use color coding to identify coalesced vs non-coalesced accesses

For matrix transposition:
```
Input Matrix        Thread Block       Output Matrix
+--+--+--+--+       +--+--+            +--+--+--+--+
|A0|A1|A2|A3|       |T0|T1|            |A0|B0|C0|D0|
+--+--+--+--+       +--+--+            +--+--+--+--+
|B0|B1|B2|B3|  <->  |T2|T3|     ->     |A1|B1|C1|D1|
+--+--+--+--+       +--+--+            +--+--+--+--+
|C0|C1|C2|C3|                          |A2|B2|C2|D2|
+--+--+--+--+                          +--+--+--+--+
|D0|D1|D2|D3|                          |A3|B3|C3|D3|
+--+--+--+--+                          +--+--+--+--+
```

### 3) Trace the Data Flow for a Single Block
For one representative block:
- Follow the data from input to output
- Note where data is loaded, stored, and processed
- Identify synchronization points

This might look like:

```
Thread 0         Thread 1
Load A[0,0]      Load A[0,1]
Load B[0,0]      Load B[1,0]
Compute          Compute
Store C[0,0]     Store C[0,1]
```
### 4) Create Timeline Diagrams
For complex kernels, create a timeline showing:
- When loads happen
- When computation occurs
- When stores happen
- When synchronization occurs

Like this:
```
Time →
T0: [Load A][Load B][Compute][Store C]
T1: [Load A][Load B][Compute][Store C]
           ↑Sync   ↑Sync
```

### 5) Track Register and Shared Memory Usage
Create a visual representation of:
- Which data is in registers for each thread
- Which data is in shared memory
- How data moves between memory spaces

For example:

```
Registers (per thread)       Shared Memory (block-wide)
+------+------+             +------+------+
|a_val0|a_val1|             |a_tile|      |
+------+------+             +------+------+
|b_val0|b_val1|      <->    |b_tile|      |
+------+------+             +------+------+
|acc_00|acc_01|
+------+------+
```

### 6) Use Color Coding for Different Memory Transactions
Highlight potential performance issues:
- Green for coalesced accesses
- Red for non-coalesced accesses
- Yellow for bank conflicts
- Blue for cache-friendly patterns

### 7) Annotate with Performance Metrics
For key sections of your visual diagrams, add:
- Theoretical memory bandwidth utilization
- Arithmetic intensity
- Occupancy estimates

### 8. Practical Example: Visualizing a Transpose Kernel

Let's walk through visualizing a simple transpose:

```
Thread Block (4x4 threads)
+----+----+----+----+
| T0 | T1 | T2 | T3 |
+----+----+----+----+
| T4 | T5 | T6 | T7 |
+----+----+----+----+
| T8 | T9 | T10| T11|
+----+----+----+----+
| T12| T13| T14| T15|
+----+----+----+----+

Memory Access Pattern (loading):
Thread 0: Loads A[0,0]
Thread 1: Loads A[0,1]  ← Coalesced (good)
Thread 2: Loads A[0,2]
Thread 3: Loads A[0,3]

Memory Access Pattern (storing):
Thread 0: Stores B[0,0]
Thread 1: Stores B[1,0]  ← Non-coalesced (bad)
Thread 2: Stores B[2,0]
Thread 3: Stores B[3,0]
```

By using shared memory:

```
1. Coalesced load from global memory to shared memory
   [T0,T1,T2,T3] → [A0,A1,A2,A3] (row in global memory)

2. Shared memory with padding (to avoid bank conflicts)
   SM[0,0] SM[0,1] SM[0,2] SM[0,3] SM[0,4]
   SM[1,0] SM[1,1] SM[1,2] SM[1,3] SM[1,4]
   SM[2,0] SM[2,1] SM[2,2] SM[2,3] SM[2,4]
   SM[3,0] SM[3,1] SM[3,2] SM[3,3] SM[3,4]
   ← Extra column padding prevents conflicts

3. Coalesced store from shared memory to global memory
   [T0,T1,T2,T3] → [B0,B1,B2,B3] (row in global memory)
```

This visual approach helps identify performance bottlenecks and understand complex memory access patterns that might otherwise be difficult to track in code.
## Profiling GPU Kernels
Profiling is essential for understanding performance and identifying bottlenecks in GPU code. Here are systematic approaches to profiling:
### Performance Metrics to Monitor
1. **Execution Time**: The most basic metric
2. **Memory Bandwidth**: Measure how efficiently you're using memory bandwidth
3. **Compute Utilization**: Check if your kernel is compute-bound
4. **Occupancy**: Percentage of maximum possible warps that are active
5. **Memory Patterns**: Coalescing, bank conflicts, and cache hit rates
### Profiling Tools
#### PyTorch Profiler
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/my_model_trace'),
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    for step in range(10):
        # Run your code here
        output = my_function(input)
        prof.step()  # Record the next step

# Print summary
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

#### Basic Timing with CUDA Events
```python
# Create events
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

# Record start event
start.record()

# Run kernel multiple times
for i in range(100):
    output = my_function(input)

# Record end event
end.record()

# Wait for events to complete
torch.cuda.synchronize()

# Calculate elapsed time
elapsed_time = start.elapsed_time(end)
per_iteration_time = elapsed_time / 100
print(f"Average time per iteration: {per_iteration_time:.3f} ms")

# Calculate bandwidth
bytes_processed = 2 * input.element_size() * input.numel()  # Read + Write
bandwidth = bytes_processed / (per_iteration_time / 1000) / 1e9  # GB/s
print(f"Bandwidth: {bandwidth:.2f} GB/s")
```

#### Triton-Specific Profiling
Triton provides tools for benchmarking kernels:
```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as x-axis of the plot
        x_vals=[128 * i for i in range(1, 33)],  # Values for x-axis
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # Values for the line argument
        line_names=['Triton', 'PyTorch'],  # Names to use as labels for the lines
        styles=[('blue', '-'), ('green', '-')],  # Line styles
        ylabel='TFLOPS',  # Label for the y-axis
        plot_name='matmul-performance',  # Name for the plot
        args={},  # Values for other arguments not in x_names and line_arg
    )
)
def benchmark(M, N, K, provider):
    # Initialize input tensors
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b))
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: my_gemm(a, b))

    # Calculate TFLOPS (floating-point operations per second)
    flops = 2 * M * N * K  # FMA = 2 operations
    return flops / (ms * 1e-3) / 1e12
```

### Systematic Profiling Approach
1. **Establish Baselines**:
    - Compare against standard libraries (cuBLAS, PyTorch, etc.)
    - Determine theoretical maximum performance for your hardware
2. **Identify Bottlenecks**:
    - Is your kernel compute-bound or memory-bound?
    - Check memory access patterns and utilization
    - Look for divergent execution and low occupancy
3. **Measure Impact of Optimizations**:
    - Make one change at a time and measure its effect
    - Use the Roofline model to understand performance limits
    - Compare performance across different input sizes
4. **Iterate and Refine**:
    - Progressively optimize the most significant bottlenecks
    - Re-evaluate after each change
    - Document performance gains and techniques used

### Common Optimization Patterns Based on Profiling Results
1. **If memory bandwidth limited**:
    - Improve memory coalescing
    - Reduce global memory access by using shared memory or local registers
    - Consider using lower precision data types
    - Implement double buffering
2. **If compute limited**:
    - Increase arithmetic intensity (more computation per memory access)
    - Use specialized hardware (Tensor Cores)
    - Reduce thread divergence
    - Optimize mathematical operations
3. **If occupancy limited**:
    - Reduce register usage per thread
    - Optimize shared memory allocation
    - Adjust block sizes to maximize SM utilization
## Practice
### Practice: Matrix Transpose Optimization
**V0: Basic PyTorch Implementation**
- Using `x.t()` followed by `contiguous()`
- Performance: 0.561 ms (956 GB/s)
- Approximately 1/3 of optimal performance

**V1: Row-wise Partitioning**
- Each thread block handles a row of the matrix
- Performance: 3.65 ms
- Poor performance due to uncoalesced memory accesses

**V2: Coalesced Global Memory Access**
- Improved loading pattern for contiguous memory access
- Performance: 1.40 ms
- Better, but still suffers from uncoalesced writes

**V3: Tile-wise Partitioning with Shared Memory**
- Load tiles into shared memory and transpose there
- Performance: 312 μs
- Still has bank conflicts in shared memory

**V4: Shared Memory with Padding**
- Add padding to shared memory to avoid bank conflicts
- Performance: 280 μs (1.9 TB/s)
- Significant improvement, approaching theoretical limits
### Practice: GEMM
**V0: Basic Pytorch Implementation**
- Benchmark performance of this, and compare across different matrices

**V1: Simple Triton GEMM**
- Use direct memory access pattern (no advanced tiling)
- Use: `block_size_m`, `block_size_k`, `block_size_n`

**V2: Tiled GEMM**
- Use block-level tiling strategies
- Load tiles of input matrices into shared or register memory
- Process matrix multiplication tile by tile to improve cache efficiency
- Implement proper synchronization between tile operations if needed

**V3: Optimized GEMM**
- Double buffering for overlapping compute and memory operations
- Memory access pattern optimization (e.g., padding, alignment)
- Loop unrolling for compute-intensive sections
- Vectorized operations where possible
- Pipeline optimization to maximize throughput

**V4: Auto-tuning**
- Implement auto-tuning on block-sizes, and other parameters to make things fast across matrix sizes
- Benchmark against Pytorch for various matrix sizes


# CUDA Programming Study Guide
Based on your request, I'll provide a comprehensive guide on CUDA programming concepts and syntax that will help you understand how to write the kernels needed for your assignment. This guide won't provide direct solutions but will equip you with the knowledge needed to implement the required functionality.
## 1. CUDA Programming Model
### Core Concepts
- **Host**: The CPU and its memory
- **Device**: The GPU and its memory
- **Kernel**: Function that runs on the GPU
- **Thread**: Basic unit of execution on the GPU
- **Warp**: Group of 32 threads that execute in SIMT (Single Instruction, Multiple Thread) fashion
- **Block**: Group of threads that can communicate via shared memory
- **Grid**: Collection of blocks that make up a kernel launch
### Memory Hierarchy
- **Register Memory**: Fastest, private to each thread (tens of KB per SM)
- **Shared Memory**: Fast memory shared among threads in a block (~100KB per SM)
- **L1/L2 Cache**: Hardware-managed caches
- **Global Memory**: Largest but slowest memory, accessible by all threads (several GB)
## 2. CUDA Syntax Basics
### Kernel Definition
```cuda
__global__ void myKernel(float* input, float* output, int size) {
    // Code executed on the GPU
}
```
Key qualifiers:
- `__global__`: Function runs on the device, callable from host
- `__device__`: Function runs on the device, callable only from device
- `__host__`: Function runs on the host (default if no qualifier)
### Kernel Launch
```cuda
// Define grid dimensions
dim3 blockDim(256);  // Threads per block
dim3 gridDim((size + blockDim.x - 1) / blockDim.x);  // Number of blocks

// Launch kernel
myKernel<<<gridDim, blockDim>>>(d_input, d_output, size);
```
### Thread Indexing
```cuda
// Within a kernel:
int threadId = threadIdx.x;  // Thread ID within block (x dimension)
int blockId = blockIdx.x;    // Block ID (x dimension)
int globalId = blockIdx.x * blockDim.x + threadIdx.x;  // Global thread ID
```
### Multi-dimensional Thread Organization
```cuda
// 2D indexing
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
int idx = row * width + col;  // Linear index for 2D data
```
## 3. Memory Management
### Allocation and Deallocation
```cuda
// Device memory allocation
float *d_array;
cudaMalloc((void**)&d_array, size * sizeof(float));

// Device memory deallocation
cudaFree(d_array);

// Host memory allocation (pinned memory for faster transfers)
float *h_array;
cudaMallocHost((void**)&h_array, size * sizeof(float));
// or standard allocation
float *h_array = (float*)malloc(size * sizeof(float));

// Host memory deallocation
cudaFreeHost(h_array);  // For pinned memory
// or standard deallocation
free(h_array);
```

### Data Transfer

```cuda
// Host to device transfer
cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);

// Device to host transfer
cudaMemcpy(h_array, d_array, size * sizeof(float), cudaMemcpyDeviceToHost);

// Device to device transfer
cudaMemcpy(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice);

// Asynchronous transfer
cudaMemcpyAsync(d_array, h_array, size * sizeof(float),
                cudaMemcpyHostToDevice, stream);
```

### Shared Memory

```cuda
__global__ void sharedMemKernel(float* input, float* output, int size) {
    // Static shared memory declaration
    __shared__ float sharedData[256];

    // Or dynamic shared memory (size determined at kernel launch)
    extern __shared__ float dynamicShared[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < size) {
        sharedData[threadIdx.x] = input[idx];
    }

    // Synchronize threads in block
    __syncthreads();

    // Use shared data
    if (idx < size) {
        output[idx] = sharedData[threadIdx.x] * 2.0f;
    }
}

// Launch with dynamic shared memory
sharedMemKernel<<<gridDim, blockDim, sharedMemSize>>>(d_input, d_output, size);
```

## 4. Parallel Patterns

### Vector Addition

```cuda
__global__ void vectorAdd(float* a, float* b, float* c, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}
```

### Matrix Transpose

```cuda
__global__ void naiveTranspose(float* input, float* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        output[col * rows + row] = input[row * cols + col];
    }
}
```

### Optimized Matrix Transpose with Shared Memory

```cuda
__global__ void sharedTranspose(float* input, float* output, int rows, int cols) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Load data into shared memory
    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = input[y * cols + x];
    }

    __syncthreads();

    // Calculate transposed coordinates
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    // Write transposed data
    if (x < rows && y < cols) {
        output[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Reduction

```cuda
__global__ void reduction(float* input, float* output, int size) {
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

## 5. Optimization Techniques

### Coalesced Memory Access

- Access pattern where consecutive threads access consecutive memory locations
- Critical for performance as it allows for efficient memory transactions

```cuda
// Good: Coalesced access
int idx = blockIdx.x * blockDim.x + threadIdx.x;
float value = input[idx];  // Consecutive threads access consecutive elements

// Bad: Strided access
int idx = threadIdx.x * stride + blockIdx.x;
float value = input[idx];  // Threads access memory with stride gaps
```

### Bank Conflicts in Shared Memory

- Shared memory is divided into 32 banks (one for each thread in a warp)
- Bank conflicts occur when multiple threads in a warp access the same bank
- Avoid by using padding in shared memory declarations

```cuda
// With potential bank conflicts
__shared__ float sharedMem[TILE_SIZE][TILE_SIZE];

// Avoiding bank conflicts with padding
__shared__ float sharedMem[TILE_SIZE][TILE_SIZE + 1];
```

### Thread Divergence

- Occurs when threads in the same warp take different execution paths
- Minimize conditional statements that depend on thread ID

```cuda
// High divergence (bad)
if (threadIdx.x % 2 == 0) {
    // Half of threads do this
} else {
    // Half of threads do that
}

// Less divergence (better)
if (blockIdx.x % 2 == 0) {
    // All threads in the block do this
} else {
    // All threads in the block do that
}
```

### Loop Unrolling

```cuda
// Instead of:
for (int i = 0; i < 4; i++) {
    result += input[idx + i];
}

// Unroll to:
result += input[idx];
result += input[idx + 1];
result += input[idx + 2];
result += input[idx + 3];
```

### Occupancy

- Ratio of active warps to maximum possible warps on an SM
- Affected by register usage, shared memory usage, and block size
- Higher occupancy often leads to better performance by hiding latency

## 6. Error Handling

```cuda
// Check for errors after kernel launch
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Handle error
}

// Check for asynchronous errors
cudaDeviceSynchronize();
err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Handle error
}
```

## 7. Implementation for Your Assignment

Based on your code files, here are the key implementations you'll need:

### 1. SiLU Activation Function

```cuda
// In silu.cu
__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void silu_kernel(float* input, float* output, int n) {
    // TODO: Implement SiLU: x * sigmoid(x)
}

void silu(float* input, float* output, int n) {
    // TODO: Set up grid and block dimensions, launch kernel
}

// In main.cu
int main() {
    // TODO: Set up memory, launch kernel, verify results
}
```

### 2. RMS Normalization (Vector)

```cuda
// In rms_norm_vector.cu
__global__ void rms_norm_vector_kernel(float* input, float* weight, float* output,
                                      int cols, float epsilon) {
    // TODO: Implement RMS normalization for vectors
    // 1. Calculate sum of squares
    // 2. Calculate RMS (root mean square)
    // 3. Normalize and scale by weight
}

void rms_norm_vector(float* input, float* weight, float* output,
                     int cols, float epsilon) {
    // TODO: Set up grid and block dimensions, launch kernel
}

// In main.cu
int main() {
    // TODO: Set up memory, launch kernel, verify results
}
```

### 3. RMS Normalization (Matrix)

```cuda
// In rms_norm_matrix.cu
__global__ void rms_norm_matrix_kernel(float* input, float* weight, float* output,
                                      int rows, int cols, float epsilon) {
    // TODO: Implement RMS normalization for matrices
    // Process each row independently
}

void rms_norm_matrix(float* input, float* weight, float* output,
                     int rows, int cols, float epsilon) {
    // TODO: Set up grid and block dimensions, launch kernel
}

// In main.cu
int main() {
    // TODO: Set up memory, launch kernel, verify results
}
```

### 4. Copy First Column

```cuda
// In copy_first_column.cu
void copy_first_column(float* h_A, float* d_A, int rows, int cols) {
    // TODO: Implement copying first column from host to device
    // Options:
    // 1. Use cudaMemcpy for each element
    // 2. Create a temporary array and batch the copy
}

// In main.cu
int main() {
    // TODO: Set up memory, call function, verify results
}
```

## 8. Performance Considerations

1. **Memory Throughput**: Global memory bandwidth is often the bottleneck. Use coalesced access patterns and shared memory to reduce global memory traffic.

2. **Arithmetic Intensity**: Ratio of arithmetic operations to memory operations. Higher values generally indicate better performance.

3. **Occupancy**: Aim for high SM occupancy by balancing resources (registers, shared memory) and block size.

4. **Latency Hiding**: Use enough threads to hide memory and instruction latency.

5. **Synchronization**: Minimize `__syncthreads()` calls as they can stall execution.

6. **Divergence**: Minimize thread divergence by designing algorithms with uniform control flow.

7. **Block Size**: Choose block sizes that are multiples of 32 (warp size) for best performance.


By applying these concepts and techniques, you should be able to implement efficient CUDA kernels for the functions required in your assignment. Remember that optimization is an iterative process - start with a simple working implementation, then optimize for performance.