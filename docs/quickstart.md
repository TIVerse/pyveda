# PyVeda Quickstart Guide

Get started with PyVeda in minutes.

## Installation

```bash
pip install pyveda
```

### Optional Dependencies

```bash
# For GPU support
pip install pyveda[gpu]

# For development
pip install pyveda[dev]
```

## Basic Usage

### Parallel Map

Execute functions in parallel with automatic executor selection:

```python
import pyveda as veda

# Initialize runtime (optional, auto-initializes on first use)
veda.init()

# Parallel iteration
result = veda.par_iter(range(1000))\
    .map(lambda x: x ** 2)\
    .filter(lambda x: x % 2 == 0)\
    .sum()

print(result)

# Cleanup
veda.shutdown()
```

### Scoped Execution

Use structured concurrency with scopes:

```python
import veda

def compute(x):
    return x ** 2

with veda.scope() as s:
    # Spawn tasks
    futures = [s.spawn(compute, i) for i in range(10)]
    
    # Scope waits for all tasks to complete
    # when exiting the context

# All tasks are done here
results = [f.result() for f in futures]
```

### Configuration

Customize runtime behavior:

```python
import veda

# Build configuration
config = veda.Config.builder()\
    .threads(8)\
    .processes(4)\
    .gpu(True)\
    .telemetry(True)\
    .build()

# Initialize with config
veda.init(config)
```

## Key Features

### 1. Adaptive Scheduling

PyVeda automatically selects the best executor (threads, processes, async, GPU) based on:
- Task characteristics (CPU-bound, IO-bound, async)
- System load
- GPU availability

### 2. Parallel Iterators

Rayon-inspired chainable operations:

```python
result = veda.par_iter(data)\
    .map(transform)\
    .filter(predicate)\
    .fold(0, lambda a, b: a + b)
```

### 3. GPU Acceleration

Automatic GPU offload with decorator:

```python
import numpy as np

@veda.gpu
def matrix_multiply(A, B):
    return A @ B

# Automatically uses GPU if beneficial
result = matrix_multiply(np.ones((1000, 1000)), np.ones((1000, 1000)))
```

### 4. Telemetry

Built-in metrics and monitoring:

```python
config = veda.Config.builder().telemetry(True).build()
veda.init(config)

# ... run workload ...

# Get metrics snapshot
runtime = veda.get_runtime()
snapshot = runtime.telemetry.snapshot()

print(f"Tasks executed: {snapshot.tasks_executed}")
print(f"Avg latency: {snapshot.avg_latency_ms:.2f}ms")
print(f"Throughput: {snapshot.throughput_tasks_per_sec:.1f} tasks/sec")

# Export metrics
prometheus_text = snapshot.export_prometheus()
json_data = snapshot.export_json()
```

### 5. Deterministic Execution

Reproducible execution for testing:

```python
from veda.deterministic import deterministic

with deterministic(seed=42):
    result = veda.par_iter(range(100))\
        .map(complex_function)\
        .collect()
    
    # Same result every time with seed=42
```

## Iterator Operations

### Transformations

- `map(func)` - Apply function to each element
- `filter(pred)` - Keep elements matching predicate
- `flat_map(func)` - Map and flatten results
- `enumerate()` - Add indices
- `zip(other)` - Zip with another iterable
- `take(n)` - Take first n elements
- `skip(n)` - Skip first n elements
- `chunk(size)` - Group into fixed-size chunks

### Reductions

- `collect()` - Collect into list
- `fold(init, func)` - Reduce with initial value
- `reduce(func)` - Reduce without initial value
- `sum()` - Sum all elements
- `count()` - Count elements
- `min()` - Find minimum
- `max()` - Find maximum
- `any(pred)` - Check if any element matches
- `all(pred)` - Check if all elements match

### Advanced

- `gpu_map(func)` - GPU-accelerated map
- `async_map(func)` - Async function map
- `for_each(func)` - Execute for side effects
- `to_dict(key_func)` - Convert to dictionary

## Configuration Options

```python
Config.builder()
    .threads(n)                    # Thread pool size
    .processes(n)                  # Process pool size
    .policy(SchedulingPolicy)      # ADAPTIVE or DETERMINISTIC
    .gpu(bool)                     # Enable GPU
    .telemetry(bool)               # Enable telemetry
    .deterministic_seed(int)       # Seed for deterministic mode
    .task_queue_size(int)          # Max pending tasks
    .min_workers(int)              # Min workers for scaling
    .max_workers(int)              # Max workers for scaling
    .adaptive_interval_ms(int)     # Scaling check interval
    .cpu_threshold_percent(float)  # CPU threshold for executor selection
    .build()
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand internals
- Check [Examples](../examples/) for real-world use cases
- Explore the [API Reference](api_reference.md) for detailed documentation
- See [GPU Guide](gpu_guide.md) for GPU acceleration details
- Review [Telemetry](telemetry.md) for monitoring and observability
