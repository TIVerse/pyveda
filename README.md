# VedaRT

<p align="center">
  <strong>Unified Parallel Runtime for Python</strong>
</p>

<p align="center">
  <a href="https://github.com/TIVerse/vedart/actions"><img src="https://github.com/TIVerse/vedart/workflows/CI/badge.svg" alt="CI Status"></a>
  <a href="https://codecov.io/gh/TIVerse/vedart"><img src="https://codecov.io/gh/TIVerse/vedart/branch/master/graph/badge.svg" alt="Coverage"></a>
  <a href="https://pypi.org/project/vedart/"><img src="https://img.shields.io/pypi/v/vedart.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/vedart/"><img src="https://img.shields.io/pypi/pyversions/vedart.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
</p>

**VedaRT** (Versatile Execution and Dynamic Adaptation Runtime) unifies Python's fragmented concurrency ecosystem—threads, processes, asyncio, and GPU—under a single adaptive, observable API inspired by Rust's Rayon.

## Why VedaRT?

Python's concurrency landscape is fragmented:

- **asyncio** - Great for I/O, terrible for CPU work
- **threading** - Limited by GIL, unpredictable performance  
- **multiprocessing** - High overhead, serialization issues
- **Ray/Dask** - Over-engineered for local workloads
- **CuPy/Numba** - Manual GPU management, no CPU fallback

**VedaRT solves this** by providing:

✅ **One API** for all execution modes  
✅ **Automatic** scheduling (threads/processes/GPU)  
✅ **Zero setup** - works out of the box  
✅ **Deterministic** replay for debugging  
✅ **Observable** with built-in telemetry

## Features

### 🚀 Zero-Boilerplate Parallel Computing

No executor setup, no pool management. Just write your logic:

```python
import vedart as veda

# Parallel iteration - automatically optimized
result = veda.par_iter(range(1000)).map(lambda x: x**2).sum()
```

### 🧠 Adaptive Scheduling

Automatic executor selection based on workload characteristics:

```python
# I/O-bound → threads
results = veda.par_iter(urls).map(fetch_url).collect()

# CPU-bound → processes  
results = veda.par_iter(data).map(heavy_computation).collect()

# GPU-compatible → GPU
results = veda.par_iter(matrices).gpu_map(matrix_multiply).collect()
```

### 🎯 Type-Safe & Modern

Full type hints, mypy strict mode compliance:

```python
from vedart import par_iter
from typing import List

def process_data(items: List[int]) -> List[int]:
    return par_iter(items).map(lambda x: x * 2).collect()
```

### 📊 Built-in Telemetry

Rich observability with zero configuration:

```python
import vedart as veda

# Your parallel work
result = veda.par_iter(data).map(process).collect()

# Get metrics
metrics = veda.telemetry.snapshot()
print(f"Tasks executed: {metrics.tasks_executed}")
print(f"Avg latency: {metrics.avg_latency_ms}ms")

# Export to Prometheus
metrics.export_prometheus()  # Ready for Grafana
```

### 🔬 Deterministic Debugging

Reproduce bugs reliably with deterministic mode:

```python
with veda.deterministic(seed=42):
    # Exact same execution order every time
    result = flaky_parallel_computation()
```

### ⚡ GPU Acceleration

Seamless GPU offload with automatic CPU fallback:

```python
@veda.gpu
def matrix_multiply(A, B):
    return A @ B  # Runs on GPU if available, CPU otherwise
```

## Quick Start

### Basic Parallel Iteration

```python
import vedart as veda

# Parallel map
result = veda.par_iter(range(1000)).map(lambda x: x**2).sum()
# Output: 332833500

# Map + filter chain
result = (
    veda.par_iter(range(100))
    .map(lambda x: x * 2)
    .filter(lambda x: x > 50)
    .collect()
)

# Fold/reduce operations
product = veda.par_iter([1, 2, 3, 4, 5]).fold(1, lambda acc, x: acc * x)
# Output: 120
```

### Scoped Parallel Execution

```python
from vedart import scope
import time

def slow_task(x):
    time.sleep(0.1)
    return x ** 2

# Spawn tasks in parallel scope
with scope() as s:
    futures = [s.spawn(slow_task, i) for i in range(5)]
    results = s.wait_all()  # [0, 1, 4, 9, 16]
```

### Complex Data Pipeline

```python
import vedart as veda

def preprocess(item):
    # CPU-bound preprocessing
    return item.lower().strip()

async def save_to_db(item):
    # I/O-bound async operation
    await db.save(item)

# Mixed execution modes in one pipeline
results = (
    veda.par_iter(raw_data)
    .map(preprocess)           # Parallel CPU work
    .async_map(save_to_db)     # Async I/O
    .collect()
)
```

### GPU Acceleration

```python
import vedart as veda
import numpy as np

@veda.gpu  # Automatically uses CuPy/Numba if available
def matrix_ops(A, B):
    return A @ B + A.T

A = np.random.rand(1000, 1000)
B = np.random.rand(1000, 1000)
result = matrix_ops(A, B)  # Runs on GPU, falls back to CPU
```

## Installation

```bash
# Basic installation
pip install vedart

# With GPU support
pip install vedart[gpu]

# With telemetry
pip install vedart[telemetry]

# Everything
pip install vedart[all]
```

## Core Concepts

### Parallel Iterators

Rayon-style parallel iterators with lazy evaluation:

```python
from vedart import par_iter

# Chaining operations
result = (
    par_iter(data)
    .map(transform)      # Transform each item
    .filter(predicate)   # Filter items
    .fold(init, reducer) # Reduce to single value
)

# Common operations
par_iter(items).sum()         # Sum all items
par_iter(items).count()       # Count items  
par_iter(items).max()         # Find maximum
par_iter(items).collect()     # Collect to list
```

### Scoped Execution

Structured concurrency with automatic cleanup:

```python
from vedart import scope

with scope() as s:
    # All spawned tasks finish before exiting scope
    f1 = s.spawn(task1)
    f2 = s.spawn(task2, arg1, arg2)
    results = s.wait_all()
# Guaranteed: all tasks complete, resources cleaned up
```

### Configuration

Customize runtime behavior:

```python
import vedart as veda

# Builder pattern
config = (
    veda.Config.builder()
    .num_threads(8)
    .num_processes(4)
    .enable_gpu(True)
    .telemetry(True)
    .build()
)

veda.init(config)

# Or use presets
veda.init(veda.Config.thread_only())  # Thread pool only
veda.init(veda.Config.adaptive())     # Full adaptive scheduling
```

### Telemetry & Monitoring

Built-in metrics collection:

```python
import vedart as veda

# Run your workload
result = veda.par_iter(data).map(process).collect()

# Get metrics snapshot
metrics = veda.telemetry.snapshot()

print(f"Tasks executed: {metrics.tasks_executed}")
print(f"Tasks failed: {metrics.tasks_failed}")
print(f"Avg latency: {metrics.avg_latency_ms}ms")
print(f"P99 latency: {metrics.p99_latency_ms}ms")
print(f"CPU usage: {metrics.cpu_utilization_percent}%")

# Export formats
metrics.export_json()        # JSON format
metrics.export_prometheus()  # Prometheus format
```

### Deterministic Mode

Reproducible execution for testing:

```python
import vedart as veda

# Same seed = same execution order
with veda.deterministic(seed=42):
    result1 = parallel_workload()

with veda.deterministic(seed=42):
    result2 = parallel_workload()

assert result1 == result2  # Always true

# Save execution trace
with veda.deterministic(seed=42, trace_file="debug.trace"):
    buggy_function()

# Replay exact execution
veda.replay("debug.trace")
```

## Performance

### Benchmarks vs Alternatives

| Workload | VedaRT | Ray | Dask | asyncio | threading |
|----------|--------|-----|------|---------|----------|
| CPU-bound (uniform) | **1.0x** | 0.9x | 0.77x | N/A | 1.05x |
| CPU-bound (variable) | **1.0x** | 0.67x | 0.48x | N/A | 0.83x |
| I/O-bound | **1.0x** | 1.11x | N/A | 1.05x | 0.91x |
| GPU-accelerated | **1.0x** | 0.83x | 0.56x | N/A | 0.05x |
| Mixed workload | **1.0x** | 0.71x | 0.62x | N/A | 0.78x |

**Task spawn overhead**: ~85ns per task  
**Memory overhead**: <5% vs raw threading

## Examples

Explore real-world examples in the [`examples/`](examples/) directory:

- [`01_hello_parallel.py`](examples/01_hello_parallel.py) - Basic parallel iteration
- [`02_scoped_execution.py`](examples/02_scoped_execution.py) - Structured concurrency
- [`03_gpu_matrix_ops.py`](examples/03_gpu_matrix_ops.py) - GPU acceleration
- [`04_telemetry.py`](examples/04_telemetry.py) - Metrics and monitoring
- [`05_deterministic_debug.py`](examples/05_deterministic_debug.py) - Deterministic debugging
- [`06_data_pipeline.py`](examples/06_data_pipeline.py) - ETL pipeline
- [`07_configuration.py`](examples/07_configuration.py) - Runtime configuration
- [`08_etl_pipeline.py`](examples/08_etl_pipeline.py) - Advanced ETL
- [`08_ml_pipeline.py`](examples/08_ml_pipeline.py) - Machine learning workflow
- [`09_async_integration.py`](examples/09_async_integration.py) - Async/await integration

Run any example:

```bash
python examples/01_hello_parallel.py
```

## Documentation

- **[Technical Specification](docs/brief.md)** - Complete API reference and architecture
- **[Test Results](docs/TEST_RESULTS.md)** - Test coverage and validation
- **[Examples](examples/)** - Code samples and tutorials

## Comparison with Alternatives

### vs Ray

✅ **VedaRT**: Zero setup, local-first design  
❌ **Ray**: Heavy dependencies, complex setup for local use

### vs Dask

✅ **VedaRT**: Lightweight, simple API  
❌ **Dask**: High memory overhead, scheduler complexity

### vs asyncio

✅ **VedaRT**: Works for both I/O and CPU workloads  
❌ **asyncio**: Only I/O-bound, steep learning curve

### vs threading/multiprocessing

✅ **VedaRT**: Automatic mode selection, adaptive scaling  
❌ **stdlib**: Manual pool management, no adaptation

## Requirements

- **Python**: 3.10, 3.11, 3.12, 3.13
- **OS**: Linux, macOS, Windows
- **Optional**:
  - CuPy or Numba for GPU support
  - psutil for system metrics (auto-installed)

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests: `pytest tests/`
4. Ensure code quality:
   ```bash
   ruff check src/vedart tests
   black src/vedart tests
   mypy src/vedart --strict
   ```
5. Commit your changes (`git commit -m 'feat: add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/TIVerse/vedart.git
cd vedart

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run full CI suite locally
./run_ci_tests.sh
```

## Roadmap

### v1.0 (Current) ✅
- Adaptive scheduler (threads/processes/async/GPU)
- Parallel iterators
- Scoped execution
- Telemetry and metrics
- Deterministic mode

### v1.1 (Planned)
- Custom executor plugins
- Advanced load balancing strategies
- Distributed tracing integration

### v2.0 (Future)
- Multi-node distributed execution
- Network-aware scheduling
- Fault tolerance and checkpointing

## Authors

**TIVerse Team**
- [@vedanthq](https://github.com/vedanthq)
- [@eshanized](https://github.com/eshanized)

## Acknowledgments

Inspired by:
- [Rayon](https://github.com/rayon-rs/rayon) - Rust's data parallelism library
- [Ray](https://github.com/ray-project/ray) - Distributed computing framework
- [Tokio](https://tokio.rs/) - Async runtime design patterns

## License

MIT License - See LICENSE file for details.
