# PyVeda API Reference

Complete API documentation for PyVeda.

## Core Functions

### `init(config: Optional[Config] = None) -> None`

Initialize the PyVeda runtime.

**Parameters:**
- `config`: Optional configuration object. If None, uses defaults.

**Example:**
```python
import pyveda as veda

config = veda.Config.builder().threads(4).build()
veda.init(config)
```

### `shutdown() -> None`

Shutdown the PyVeda runtime and cleanup resources.

**Example:**
```python
veda.shutdown()
```

### `get_runtime() -> Runtime`

Get the global runtime instance.

**Returns:**
- `Runtime` object with access to scheduler, telemetry, and GPU.

**Example:**
```python
runtime = veda.get_runtime()
print(runtime.config.num_threads)
```

### `spawn(func: Callable, *args, **kwargs) -> Future`

Spawn a task globally (non-scoped).

**Parameters:**
- `func`: Function to execute
- `*args`: Positional arguments
- `**kwargs`: Keyword arguments

**Returns:**
- `Future` for the result

**Example:**
```python
future = veda.spawn(compute, 42)
result = future.result()
```

## Configuration

### `Config`

Runtime configuration dataclass.

**Fields:**
- `num_threads: int = 4` - Thread pool size
- `num_processes: int = 0` - Process pool size (0 = disabled)
- `scheduling_policy: SchedulingPolicy = ADAPTIVE` - Scheduling strategy
- `enable_gpu: bool = False` - Enable GPU support
- `enable_telemetry: bool = False` - Enable telemetry
- `deterministic_seed: Optional[int] = None` - Seed for deterministic mode
- `task_queue_size: int = 1000` - Max pending tasks
- `min_workers: int = 1` - Min workers for adaptive scaling
- `max_workers: Optional[int] = None` - Max workers for adaptive scaling
- `adaptive_interval_ms: int = 1000` - Scaling check interval
- `cpu_threshold_percent: float = 80.0` - CPU threshold for executor selection

### `Config.builder() -> ConfigBuilder`

Create a configuration builder for fluent API.

**Returns:**
- `ConfigBuilder` instance

**Example:**
```python
config = veda.Config.builder()\
    .threads(8)\
    .gpu(True)\
    .telemetry(True)\
    .build()
```

### `ConfigBuilder`

Fluent builder for Config.

**Methods:**
- `threads(n: int) -> ConfigBuilder`
- `processes(n: int) -> ConfigBuilder`
- `policy(p: SchedulingPolicy) -> ConfigBuilder`
- `gpu(enabled: bool) -> ConfigBuilder`
- `telemetry(enabled: bool) -> ConfigBuilder`
- `deterministic_seed(seed: int) -> ConfigBuilder`
- `build() -> Config`

## Parallel Iterators

### `par_iter(iterable: Iterable[T]) -> ParallelIterator[T]`

Create a parallel iterator from an iterable.

**Parameters:**
- `iterable`: Input iterable

**Returns:**
- `ParallelIterator` instance

### `ParallelIterator[T]`

Rayon-inspired parallel iterator with chainable operations.

#### Transformations

**`map(func: Callable[[T], U]) -> ParallelIterator[U]`**

Apply function to each element in parallel.

**`filter(predicate: Callable[[T], bool]) -> ParallelIterator[T]`**

Keep elements matching predicate.

**`flat_map(func: Callable[[T], Iterable[U]]) -> ParallelIterator[U]`**

Map and flatten results.

**`gpu_map(func: Callable[[T], U]) -> ParallelIterator[U]`**

Apply function using GPU acceleration.

**`async_map(func: Callable[[T], U]) -> ParallelIterator[U]`**

Apply async function to elements.

**`enumerate() -> ParallelIterator[Tuple[int, T]]`**

Add indices to elements.

**`zip(other: Iterable[U]) -> ParallelIterator[Tuple[T, U]]`**

Zip with another iterable.

**`take(n: int) -> ParallelIterator[T]`**

Take first n elements.

**`skip(n: int) -> ParallelIterator[T]`**

Skip first n elements.

**`chunk(size: int) -> ParallelIterator[List[T]]`**

Group elements into fixed-size chunks.

#### Reductions

**`collect() -> List[T]`**

Execute pipeline and collect results into list.

**`fold(identity: U, op: Callable[[U, T], U]) -> U`**

Parallel fold with identity element.

**`reduce(op: Callable[[T, T], T]) -> Optional[T]`**

Parallel reduce without identity.

**`sum() -> T`**

Sum all elements (requires numeric type).

**`count() -> int`**

Count number of elements.

**`min() -> Optional[T]`**

Find minimum element.

**`max() -> Optional[T]`**

Find maximum element.

**`any(pred: Callable[[T], bool]) -> bool`**

Check if any element satisfies predicate.

**`all(pred: Callable[[T], bool]) -> bool`**

Check if all elements satisfy predicate.

**`for_each(func: Callable[[T], None]) -> None`**

Execute function on each element for side effects.

**`to_dict(key_func: Callable[[T], K]) -> Dict[K, T]`**

Convert to dictionary using key function.

## Scoped Execution

### `scope() -> Scope`

Create a scope for structured concurrency.

**Returns:**
- Context manager yielding `Scope` instance

**Example:**
```python
with veda.scope() as s:
    futures = [s.spawn(task, i) for i in range(10)]
    # Scope waits for all tasks on exit
```

### `Scope`

Structured concurrency scope.

**Methods:**

**`spawn(func: Callable, *args, **kwargs) -> Future`**

Spawn a task within the scope.

**`join() -> None`**

Wait for all spawned tasks to complete.

## GPU Support

### `@gpu`

Decorator for automatic GPU offload.

**Example:**
```python
import numpy as np

@veda.gpu
def matrix_multiply(A, B):
    return A @ B

result = matrix_multiply(np.ones((1000, 1000)), np.ones((1000, 1000)))
```

### `@gpu_kernel`

Mark function as GPU kernel (Numba backend).

**Example:**
```python
@veda.gpu_kernel
def add_kernel(a, b, c):
    # Numba CUDA kernel
    pass
```

### `GPURuntime`

GPU runtime manager.

**Methods:**

**`is_available() -> bool`**

Check if GPU is available.

**`get_memory_stats() -> Dict[str, float]`**

Get GPU memory statistics (used_mb, free_mb, total_mb).

**`get_utilization() -> float`**

Get GPU utilization percentage (0-100).

## Telemetry

### `TelemetrySystem`

Telemetry and metrics collection.

**Methods:**

**`snapshot() -> MetricsSnapshot`**

Create a metrics snapshot.

**`start() -> None`**

Start background collection.

**`stop() -> None`**

Stop background collection.

### `MetricsSnapshot`

Point-in-time metrics snapshot.

**Fields:**
- `timestamp: float` - Snapshot timestamp
- `tasks_executed: int` - Total tasks executed
- `tasks_failed: int` - Total tasks failed
- `tasks_pending: int` - Currently pending tasks
- `avg_latency_ms: float` - Average task latency
- `p50_latency_ms: float` - P50 latency
- `p95_latency_ms: float` - P95 latency
- `p99_latency_ms: float` - P99 latency
- `throughput_tasks_per_sec: float` - Task throughput
- `cpu_utilization_percent: float` - CPU utilization
- `memory_used_mb: float` - Memory used
- `gpu_utilization_percent: Optional[float]` - GPU utilization
- `gpu_memory_used_mb: Optional[float]` - GPU memory used

**Methods:**

**`export_prometheus() -> str`**

Export in Prometheus text format.

**`export_json() -> Dict`**

Export as JSON dictionary.

## Deterministic Execution

### `deterministic(seed: int, record: bool = False)`

Context manager for deterministic execution.

**Parameters:**
- `seed`: Random seed for reproducibility
- `record`: Whether to record execution trace

**Yields:**
- `ExecutionTrace` if recording, else None

**Example:**
```python
from pyveda.deterministic import deterministic

with deterministic(seed=42, record=True) as trace:
    result = veda.par_iter(data).map(func).collect()
    
    if trace:
        trace.save_to_file('execution.json')
```

### `ExecutionTrace`

Recorded execution trace.

**Methods:**

**`save_to_file(filepath: str) -> None`**

Save trace to JSON file.

**`load_from_file(filepath: str) -> None`**

Load trace from JSON file.

**`get_events() -> List[TaskScheduledEvent]`**

Get all recorded events.

## Types

### `Task`

Task dataclass representing a unit of work.

**Fields:**
- `id: str` - Unique task ID
- `func: Callable` - Function to execute
- `args: Tuple` - Positional arguments
- `kwargs: Dict` - Keyword arguments
- `priority: TaskPriority` - Task priority
- `is_async: bool` - Whether task is async
- `state: TaskState` - Current task state

### `TaskPriority`

Task priority enum.

**Values:**
- `HIGH = 0`
- `NORMAL = 1`
- `LOW = 2`

### `TaskState`

Task state enum.

**Values:**
- `PENDING`
- `RUNNING`
- `COMPLETED`
- `FAILED`
- `CANCELLED`

### `SchedulingPolicy`

Scheduling policy enum.

**Values:**
- `ADAPTIVE` - Dynamic executor selection
- `DETERMINISTIC` - Reproducible execution

## Exceptions

### `VedaError`

Base exception for all PyVeda errors.

### `SchedulerError`

Raised for scheduler-related errors.

### `GPUError`

Raised for GPU-related errors.

### `ConfigError`

Raised for configuration errors.
