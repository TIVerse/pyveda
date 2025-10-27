# GPU Acceleration Guide

Comprehensive guide to using GPU acceleration in VedaRT.

## Overview

VedaRT provides automatic GPU offload for data-parallel operations with transparent CPU↔GPU data movement and intelligent cost-based decisions.

## Installation

Install with GPU support:

```bash
# For NVIDIA GPUs with CUDA
pip install vedart[gpu]

# This installs CuPy (recommended)
# Alternative: install numba separately
pip install numba
```

## Automatic GPU Offload

The simplest way to use GPU acceleration is with the `@gpu` decorator:

```python
import numpy as np
import vedart as veda

# Initialize with GPU enabled
config = veda.Config.builder().gpu(True).build()
veda.init(config)

@veda.gpu
def matrix_multiply(A, B):
    """Automatically uses GPU if beneficial."""
    return A @ B

# Small matrices - runs on CPU
A_small = np.ones((10, 10))
B_small = np.ones((10, 10))
C_small = matrix_multiply(A_small, B_small)

# Large matrices - runs on GPU
A_large = np.ones((5000, 5000))
B_large = np.ones((5000, 5000))
C_large = matrix_multiply(A_large, B_large)
```

## How GPU Offload Works

### 1. Cost-Based Decision

VedaRT evaluates whether GPU execution is beneficial:

```python
def should_offload(func, args):
    # Estimate data size
    data_size = sum(arg.nbytes for arg in args if hasattr(arg, 'nbytes'))
    
    # Small data: CPU faster (avoid transfer overhead)
    if data_size < 1_000_000:  # 1 MB threshold
        return False
    
    # Check GPU memory availability
    gpu_free_memory = get_gpu_memory_free()
    if data_size * 2 > gpu_free_memory:  # Need 2x for input + output
        return False
    
    return True
```

### 2. Automatic Data Transfer

- **Input**: Automatically transferred to GPU
- **Computation**: Executed on GPU
- **Output**: Automatically transferred back to CPU

### 3. Memory Management

VedaRT uses a memory pool to reduce allocation overhead:

```python
# Memory pool automatically:
# - Pre-allocates memory blocks
# - Reuses freed memory
# - Reduces fragmentation
```

## Backend Selection

VedaRT supports two GPU backends:

### CuPy (Recommended)

NumPy-compatible GPU arrays:

```python
import cupy as cp

@veda.gpu
def vector_add(a, b):
    # Works with NumPy or CuPy arrays
    return a + b

# Automatic conversion
result = vector_add(np.array([1, 2, 3]), np.array([4, 5, 6]))
```

### Numba CUDA

For custom kernels:

```python
from numba import cuda

@veda.gpu_kernel
@cuda.jit
def add_kernel(a, b, result):
    idx = cuda.grid(1)
    if idx < result.size:
        result[idx] = a[idx] + b[idx]
```

## GPU with Parallel Iterators

Combine GPU acceleration with parallel iteration:

```python
import numpy as np

# Process many matrices on GPU
matrices = [np.random.rand(1000, 1000) for _ in range(100)]

@veda.gpu
def process_matrix(M):
    # Eigenvalue decomposition on GPU
    return np.linalg.eigvals(M)

# Each matrix processed on GPU, distributed across iterations
eigenvalues = veda.par_iter(matrices)\
    .gpu_map(process_matrix)\
    .collect()
```

## Multi-GPU Support

For systems with multiple GPUs:

```python
# Future feature - specify device
@veda.gpu(device=0)
def process_on_gpu_0(data):
    return data * 2

@veda.gpu(device=1)
def process_on_gpu_1(data):
    return data ** 2
```

## Performance Tips

### 1. Batch Small Operations

Don't GPU-accelerate tiny operations:

```python
# Bad: Small data, high overhead
@veda.gpu
def add_small(a, b):
    return a + b

result = add_small(5, 3)  # Runs on CPU anyway

# Good: Batch operations
@veda.gpu
def process_batch(batch):
    return [item ** 2 for item in batch]

large_batch = np.arange(10_000)
results = process_batch(large_batch)  # Worth GPU offload
```

### 2. Minimize Transfers

Keep data on GPU across operations:

```python
# Bad: Multiple transfers
@veda.gpu
def step1(data):
    return data * 2

@veda.gpu  
def step2(data):
    return data + 1

result = step2(step1(data))  # GPU → CPU → GPU → CPU

# Good: Fused operation
@veda.gpu
def fused(data):
    return (data * 2) + 1

result = fused(data)  # Single GPU → CPU transfer
```

### 3. Use NumPy-Compatible Code

NumPy operations translate well to GPU:

```python
@veda.gpu
def efficient_operation(X):
    # All NumPy operations work on GPU
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    normalized = (X - mean) / std
    return normalized @ normalized.T
```

## Memory Management

### Monitor GPU Memory

```python
runtime = veda.get_runtime()

if runtime.gpu:
    stats = runtime.gpu.get_memory_stats()
    print(f"Used: {stats['used_mb']:.1f} MB")
    print(f"Free: {stats['free_mb']:.1f} MB")
    print(f"Total: {stats['total_mb']:.1f} MB")
    
    utilization = runtime.gpu.get_utilization()
    print(f"GPU Utilization: {utilization:.1f}%")
```

### Clear GPU Memory

```python
# CuPy memory pool
import cupy as cp

# Free all unused memory
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
```

## Debugging GPU Code

### Check Backend

```python
runtime = veda.get_runtime()

if runtime.gpu:
    print(f"Backend: {runtime.gpu.backend}")
    print(f"Devices: {runtime.gpu.device_count}")
    print(f"Available: {runtime.gpu.is_available()}")
else:
    print("GPU not available")
```

### Force CPU Execution

Temporarily disable GPU for debugging:

```python
config = veda.Config.builder().gpu(False).build()
veda.init(config)

# All @gpu functions run on CPU
```

### Profile GPU Operations

```python
import time

@veda.gpu
def gpu_operation(data):
    return data @ data.T

data = np.random.rand(5000, 5000)

start = time.perf_counter()
result = gpu_operation(data)
gpu_time = time.perf_counter() - start

# Compare with CPU
start = time.perf_counter()
result_cpu = data @ data.T
cpu_time = time.perf_counter() - start

print(f"GPU: {gpu_time:.3f}s")
print(f"CPU: {cpu_time:.3f}s")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")
```

## Common Patterns

### Image Processing

```python
@veda.gpu
def apply_filter(image, kernel):
    from scipy.ndimage import convolve
    return convolve(image, kernel)

images = load_many_images()
filtered = veda.par_iter(images)\
    .gpu_map(lambda img: apply_filter(img, gaussian_kernel))\
    .collect()
```

### Machine Learning

```python
@veda.gpu
def train_step(X, y, weights):
    # Forward pass
    predictions = X @ weights
    
    # Loss
    loss = ((predictions - y) ** 2).mean()
    
    # Gradient
    grad = 2 * X.T @ (predictions - y) / len(y)
    
    return loss, grad

# Training loop
for epoch in range(num_epochs):
    loss, grad = train_step(X_train, y_train, weights)
    weights -= learning_rate * grad
```

### Scientific Computing

```python
@veda.gpu
def monte_carlo_pi(n_samples):
    import numpy as np
    
    # Generate random points
    x = np.random.random(n_samples)
    y = np.random.random(n_samples)
    
    # Check if inside circle
    inside = (x**2 + y**2) <= 1.0
    
    # Estimate pi
    pi_estimate = 4.0 * inside.sum() / n_samples
    return pi_estimate

# Run with billions of samples on GPU
pi = monte_carlo_pi(1_000_000_000)
```

## Troubleshooting

### "GPU not available"

1. Check CUDA installation: `nvidia-smi`
2. Verify CuPy installation: `pip show cupy`
3. Check GPU config: `config.enable_gpu = True`

### "Out of memory"

1. Reduce batch size
2. Clear memory pool: `cp.get_default_memory_pool().free_all_blocks()`
3. Use smaller data types: `np.float32` instead of `np.float64`

### "Slower than CPU"

1. Data too small (< 1MB) - transfer overhead dominates
2. Too many small operations - batch them
3. CPU-optimized code (BLAS) - GPU not always faster

## Best Practices

1. **Profile First** - Measure before optimizing
2. **Batch Operations** - Combine small ops into larger ones
3. **Stay on GPU** - Minimize CPU↔GPU transfers
4. **NumPy Compatibility** - Use NumPy-style code
5. **Monitor Memory** - Track GPU memory usage
6. **Fallback to CPU** - Handle GPU unavailability gracefully

## Further Reading

- [CuPy Documentation](https://docs.cupy.dev/)
- [Numba CUDA Guide](https://numba.pydata.org/numba-doc/latest/cuda/index.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
