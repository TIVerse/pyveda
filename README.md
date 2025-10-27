# PyVeda

**Parallel runtime for Python**

PyVeda unifies Python's fragmented concurrency ecosystem (threads, processes, asyncio, GPU) under a single adaptive, observable API inspired by Rayon.

## Features

- 🚀 **Zero-boilerplate API** - Simple, intuitive parallel computing
- 🧠 **Adaptive scheduling** - Auto-tune thread/process/GPU selection
- 🎯 **Type-safe** - Full type hints and mypy strict compliance
- 📊 **Observable** - Built-in telemetry with Prometheus export
- 🔬 **Debuggable** - Deterministic mode for reproducible execution
- ⚡ **GPU acceleration** - Seamless CuPy/Numba integration

## Quick Start

```python
import pyveda as veda

# Parallel iteration
result = veda.par_iter(range(1000)).map(lambda x: x**2).sum()

# GPU acceleration
@veda.gpu
def matrix_multiply(A, B):
    return A @ B

# Deterministic debugging
with veda.deterministic(seed=42):
    result = flaky_computation()
```

## Installation

```bash
# Basic installation
pip install pyveda

# With GPU support
pip install pyveda[gpu]

# With telemetry
pip install pyveda[telemetry]

# Everything
pip install pyveda[all]
```

## Documentation

See `docs/brief.md` for complete specification and API reference.

## License

MIT License - See LICENSE file for details.
