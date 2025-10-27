# PyVeda Architecture

Understanding PyVeda's internal design and components.

## Overview

PyVeda is a unified Python parallel runtime that combines the best features of multiple execution models (threads, processes, async, GPU) with intelligent scheduling and comprehensive observability.

## Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                        Public API                            │
│  (par_iter, spawn, scope, @gpu, Config, etc.)               │
└───────────────────────┬─────────────────────────────────────┘
                        │
┌───────────────────────▼─────────────────────────────────────┐
│                      Runtime                                 │
│  - Global singleton                                          │
│  - Lifecycle management                                      │
│  - Component initialization                                  │
└─────────┬──────────┬──────────┬──────────┬──────────────────┘
          │          │          │          │
      ┌───▼──┐   ┌──▼───┐  ┌──▼────┐  ┌──▼────────┐
      │Sched │   │ GPU  │  │Telem  │  │Determin   │
      │uler  │   │      │  │etry   │  │istic      │
      └───┬──┘   └──────┘  └───────┘  └───────────┘
          │
   ┌──────┴──────────────────┐
   │     Executor Layer       │
   ├──────┬────┬─────┬────────┤
   │Thread│Proc│Async│  GPU   │
   └──────┴────┴─────┴────────┘
```

## Scheduler

The **Adaptive Scheduler** is the brain of PyVeda, responsible for:

### Responsibilities

1. **Task Routing** - Select optimal executor for each task
2. **Load Balancing** - Distribute work across executors
3. **Worker Scaling** - Dynamically adjust worker counts
4. **Metrics Collection** - Track performance statistics

### Selection Algorithm

```python
def select_executor(task):
    # 1. Check for GPU eligibility
    if GPU available and should_offload(task):
        return GPUExecutor
    
    # 2. Check for async tasks
    if task.is_async:
        return AsyncIOExecutor
    
    # 3. Adaptive CPU selection
    if cpu_load < threshold:
        return ThreadPoolExecutor  # For IO-bound
    else:
        return ProcessPoolExecutor  # For CPU-bound
```

### Adaptive Scaling

Uses **Little's Law** for worker scaling:

```
L = λ × W

where:
  L = optimal number of workers
  λ = arrival rate (tasks/sec)
  W = average service time (sec)
```

Implementation:
```python
arrival_rate = tasks_completed / time_interval
service_time = avg_latency_ms / 1000
optimal_workers = arrival_rate * service_time
```

### Metrics Tracking

Per-executor statistics:
- Tasks executed
- Tasks failed
- Average latency
- Latency histogram (P50, P95, P99)
- Throughput

## Executor Layer

PyVeda supports four executor types:

### 1. ThreadPoolExecutor

**Use Case:** IO-bound tasks, lightweight concurrency

**Characteristics:**
- Shared memory (no pickling overhead)
- GIL-limited for CPU tasks
- Low spawn overhead
- Best for: Network IO, file operations, small tasks

### 2. ProcessPoolExecutor

**Use Case:** CPU-bound tasks

**Characteristics:**
- True parallelism (bypasses GIL)
- Isolated memory (requires pickling)
- Higher spawn overhead
- Best for: Heavy computation, CPU-intensive work

### 3. AsyncIOExecutor

**Use Case:** Async/await coroutines

**Characteristics:**
- Event loop based
- Non-blocking IO
- Cooperative multitasking
- Best for: Async libraries, concurrent IO

### 4. GPUExecutor

**Use Case:** Data-parallel computation

**Characteristics:**
- CuPy or Numba backend
- Massive parallelism
- Automatic CPU↔GPU data transfer
- Best for: Matrix operations, large arrays

## GPU Runtime

Automatic GPU acceleration with cost-based offload decisions.

### Backend Detection

```python
# Priority order:
1. Try CuPy (preferred)
2. Try Numba CUDA
3. Fall back to CPU
```

### Offload Decision

Factors considered:
- Data size (transfer cost vs compute benefit)
- GPU availability
- GPU memory
- Function characteristics

```python
def should_offload(func, args):
    data_size = estimate_size(args)
    
    # Don't offload small data (transfer overhead)
    if data_size < MIN_SIZE_BYTES:
        return False
    
    # Check GPU memory
    if gpu_memory_available < data_size * 2:
        return False
    
    # Check if function is GPU-eligible
    if not is_gpu_compatible(func):
        return False
    
    return True
```

### Memory Management

**GPUMemoryPool** provides:
- Efficient allocation/deallocation
- Memory pooling to reduce overhead
- Automatic cleanup
- CuPy integration

## Telemetry System

Comprehensive observability with multiple export formats.

### Architecture

```
┌──────────────┐
│   Metrics    │
│  Collection  │
└──────┬───────┘
       │
   ┌───▼────┐
   │Counter │
   │Gauge   │
   │Histogram│
   └───┬────┘
       │
   ┌───▼────────┐
   │ Snapshot   │
   └───┬────────┘
       │
   ┌───▼────────┐
   │ Exporters  │
   ├────────────┤
   │Prometheus  │
   │JSON        │
   │OTLP        │
   └────────────┘
```

### Metric Types

**Counter** - Monotonically increasing value
- Tasks executed
- Tasks failed

**Gauge** - Point-in-time value
- Pending tasks
- CPU utilization
- Memory usage

**Histogram** - Distribution of values
- Task latency
- Percentiles (P50, P95, P99)

### Export Formats

1. **Prometheus** - Text format for Prometheus scraping
2. **JSON** - Structured data for custom integrations
3. **OTLP** - OpenTelemetry Protocol for standard observability

## Deterministic Execution

Reproducible execution for debugging and testing.

### Implementation

```python
class DeterministicScheduler:
    def __init__(self, seed):
        self.rng = random.Random(seed)  # Seeded RNG
        self.logical_clock = 0
        self.trace = ExecutionTrace()
    
    def submit(self, task):
        # Increment logical clock
        self.logical_clock += 1
        
        # Record event
        self.trace.record(TaskScheduledEvent(
            timestamp=self.logical_clock,
            task_id=task.id,
            executor_type="thread"
        ))
        
        # Deterministic executor selection
        return self.select_deterministic(task)
```

### Trace Replay

1. **Record Mode** - Save execution trace to file
2. **Replay Mode** - Enforce recorded scheduling order
3. **Comparison** - Verify results match

## Parallel Iterators

Rayon-inspired composable parallel operations.

### Design Philosophy

1. **Lazy Evaluation** - Operations chained without execution
2. **Automatic Parallelism** - Parallelization handled transparently
3. **Chunking Strategy** - Smart work distribution

### Execution Pipeline

```python
par_iter(data)           # Create iterator
  .map(transform)        # Add map operation
  .filter(predicate)     # Add filter operation
  .collect()             # Execute pipeline
```

Pipeline execution:
1. Chunk input data
2. Apply operations to each chunk in parallel
3. Aggregate results
4. Return final result

### Chunking Algorithm

```python
def auto_chunk_size(total_items):
    # Balance between parallelism and overhead
    workers = num_available_workers()
    min_chunk = 10
    max_chunk = 1000
    
    chunk_size = total_items // (workers * 4)
    return clamp(chunk_size, min_chunk, max_chunk)
```

## Task Lifecycle

```
┌─────────┐
│ CREATED │
└────┬────┘
     │
     ▼
┌─────────┐    ┌──────────┐
│ PENDING ├───►│ RUNNING  │
└─────────┘    └────┬─────┘
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
    ┌─────────┐          ┌─────────┐
    │COMPLETED│          │ FAILED  │
    └─────────┘          └─────────┘
```

1. **Created** - Task object instantiated
2. **Pending** - Submitted to scheduler (in queue)
3. **Running** - Executing on worker
4. **Completed** - Finished successfully
5. **Failed** - Exception raised

## Performance Characteristics

### Spawn Overhead

- **Thread**: ~10-50 μs
- **Process**: ~1-5 ms
- **Async**: ~1-10 μs
- **GPU**: ~100-500 μs (+ transfer time)

### Throughput

Adaptive scheduler overhead: ~5-10 μs per task

### Memory

- Base runtime: ~50 MB
- Per worker (thread): ~8 MB
- Per worker (process): ~100 MB+
- GPU pool: Configurable

## Concurrency Model

PyVeda uses a hybrid approach:

1. **Work Stealing** - Future enhancement for load balancing
2. **Task Queues** - Priority-based task scheduling
3. **Structured Concurrency** - Scopes for lifecycle management
4. **Async Integration** - Bridge between sync/async worlds

## Thread Safety

All components are thread-safe:
- Scheduler uses locks for state
- Telemetry uses atomic operations
- Executors handle concurrent submissions

## Future Enhancements

1. **Work Stealing** - Dynamic load rebalancing
2. **Multi-GPU** - Distribute across multiple GPUs
3. **Distributed** - Network-based task distribution
4. **JIT Compilation** - Native code generation for hot paths
5. **Memory Pooling** - Reduce allocation overhead
