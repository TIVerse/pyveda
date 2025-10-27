# PyVeda Documentation

Welcome to PyVeda - A unified Python parallel runtime with adaptive scheduling, GPU acceleration, and comprehensive observability.

## Getting Started

- **[Quickstart Guide](quickstart.md)** - Get up and running in 5 minutes
- **[Installation](quickstart.md#installation)** - Install PyVeda and dependencies
- **[Basic Examples](../examples/)** - Working code examples

## Core Concepts

- **[Architecture Overview](architecture.md)** - Understand PyVeda's design
- **[API Reference](api_reference.md)** - Complete API documentation
- **[Configuration](quickstart.md#configuration-options)** - Runtime configuration options

## Features

### Parallel Execution

- **Adaptive Scheduling** - Automatic executor selection (threads/processes/async/GPU)
- **Parallel Iterators** - Rayon-like chainable operations
- **Structured Concurrency** - Scoped execution with automatic cleanup

### GPU Acceleration

- **[GPU Guide](gpu_guide.md)** - Complete GPU acceleration guide
- **Automatic Offload** - Cost-based GPU execution decisions
- **Memory Management** - Efficient GPU memory pooling

### Observability

- **[Telemetry Guide](telemetry.md)** - Monitoring and metrics
- **Multiple Export Formats** - Prometheus, JSON, OpenTelemetry
- **Distributed Tracing** - Track task execution across systems

### Reliability

- **Deterministic Replay** - Reproducible execution for testing
- **Error Handling** - Comprehensive exception hierarchy
- **Resource Management** - Automatic cleanup and lifecycle management

## Examples

### Quick Example

```python
import pyveda as veda

# Parallel data processing
result = veda.par_iter(range(1000))\
    .map(lambda x: x ** 2)\
    .filter(lambda x: x % 2 == 0)\
    .sum()

print(result)
```

### GPU Acceleration

```python
import numpy as np

@veda.gpu
def matrix_multiply(A, B):
    return A @ B

# Automatically uses GPU for large matrices
result = matrix_multiply(np.ones((5000, 5000)), np.ones((5000, 5000)))
```

### Async Integration

```python
async def fetch_data(url):
    # Async HTTP request
    return await http_client.get(url)

# Parallel async execution
urls = ['http://api.example.com/1', 'http://api.example.com/2']
results = veda.par_iter(urls).async_map(fetch_data).collect()
```

### Telemetry

```python
config = veda.Config.builder().telemetry(True).build()
veda.init(config)

# Run workload
veda.par_iter(range(10000)).map(compute_task).collect()

# Get metrics
snapshot = veda.get_runtime().telemetry.snapshot()
print(f"Throughput: {snapshot.throughput_tasks_per_sec:.1f} tasks/sec")
print(f"P99 Latency: {snapshot.p99_latency_ms:.2f}ms")
```

## Guides

- **[Quickstart](quickstart.md)** - Basic usage and configuration
- **[Architecture](architecture.md)** - Internal design and components
- **[GPU Guide](gpu_guide.md)** - GPU acceleration and optimization
- **[Telemetry](telemetry.md)** - Monitoring and observability
- **[API Reference](api_reference.md)** - Complete API documentation

## Advanced Topics

### Performance Optimization

- Chunking strategies for parallel iterators
- Executor selection tuning
- GPU memory optimization
- Batching small operations

### Production Deployment

- Configuration best practices
- Monitoring and alerting
- Error handling patterns
- Resource limits

### Testing and Debugging

- Deterministic execution
- Replay traces
- Unit testing with PyVeda
- Performance profiling

## Community

- **GitHub**: [https://github.com/yourusername/pyveda](https://github.com/yourusername/pyveda)
- **Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## License

PyVeda is released under the MIT License. See [LICENSE](../LICENSE) for details.

## Changelog

See [CHANGELOG.md](../CHANGELOG.md) for version history.

---

**Next Steps:**

1. Follow the [Quickstart Guide](quickstart.md)
2. Explore [Examples](../examples/)
3. Read the [Architecture](architecture.md) to understand internals
4. Check [API Reference](api_reference.md) for detailed documentation
