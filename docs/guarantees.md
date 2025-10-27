# VedaRT Execution Guarantees

This document defines the formal guarantees provided by VedaRT's runtime system regarding task ordering, memory consistency, error handling, and deterministic execution.

---

## 📋 Table of Contents

- [Task Ordering](#task-ordering)
- [Memory Model](#memory-model)
- [Error Handling & Propagation](#error-handling--propagation)
- [Cancellation Semantics](#cancellation-semantics)
- [Deterministic Execution](#deterministic-execution)
- [Resource Management](#resource-management)
- [Thread Safety](#thread-safety)

---

## Task Ordering

### Parallel Iterators

**Guarantee:** VedaRT's `par_iter` operations provide **unordered execution** by default for maximum parallelism.

```python
import vedart as veda

# No ordering guarantee between map executions
results = veda.par_iter(range(100)).map(process).collect()
```

**Implications:**
- ✅ **Map operations** may execute in any order across workers
- ✅ **Filter operations** preserve relative ordering of items that pass the predicate
- ✅ **Collect** returns results in original input order (stable)
- ✅ **Fold/Reduce** operations are associative and commutative

**Example - Order Preservation:**
```python
# Input: [1, 2, 3, 4, 5]
# After map: may compute in parallel as [3, 1, 5, 2, 4]
# After collect: ALWAYS returns [f(1), f(2), f(3), f(4), f(5)]
result = veda.par_iter([1, 2, 3, 4, 5]).map(lambda x: x * 2).collect()
# result == [2, 4, 6, 8, 10]  ✅ Order preserved
```

### Scoped Execution

**Guarantee:** Tasks spawned within a `scope()` context have **no ordering guarantee** unless explicitly synchronized.

```python
with veda.scope() as s:
    f1 = s.spawn(task1)
    f2 = s.spawn(task2)
    # task1 and task2 may execute in parallel in any order
    results = s.wait_all()  # Blocks until ALL tasks complete
```

**Structured Concurrency Guarantee:**
- ✅ All spawned tasks **must complete** before scope exits
- ✅ Resources are **automatically cleaned up** on scope exit
- ✅ Exceptions in child tasks are **propagated** to parent scope
- ❌ No guarantee on **execution order** of spawned tasks

---

## Memory Model

VedaRT follows a **relaxed memory model** with specific guarantees for different executor types.

### Thread Executor (Shared Memory)

**Memory Visibility:** Python's GIL provides sequential consistency for thread executor.

```python
shared_list = []

def append_item(x):
    shared_list.append(x)  # ✅ Thread-safe due to GIL

veda.par_iter(range(100)).map(append_item).collect()
# All 100 items visible in shared_list (unordered)
```

**Guarantees:**
- ✅ Atomic operations on Python objects (list.append, dict.__setitem__)
- ✅ Reads always see writes from completed tasks
- ⚠️ Race conditions still possible with compound operations

### Process Executor (Message Passing)

**Memory Isolation:** Each process has isolated memory space.

```python
shared_list = []  # ❌ NOT shared across processes

def append_item(x):
    shared_list.append(x)  # Only modifies local process copy
    return x

results = veda.par_iter(range(100)).map(append_item).collect()
# shared_list remains empty in parent process
# results contains the computed values
```

**Guarantees:**
- ✅ Complete memory isolation (no race conditions)
- ✅ Results returned via serialization (pickle)
- ❌ Mutable shared state **not supported**
- ⚠️ Only pickle-able objects can be passed/returned

### Async Executor (Cooperative)

**Memory Model:** Single-threaded event loop with cooperative multitasking.

```python
shared_dict = {}

async def update_dict(key, value):
    shared_dict[key] = value  # ✅ Safe, no concurrent access

await veda.par_iter(items).async_map(update_dict).collect()
```

**Guarantees:**
- ✅ No race conditions (single-threaded)
- ✅ Atomic operations within a single await point
- ⚠️ Must not block the event loop

### GPU Executor (Device Memory)

**Memory Model:** Explicit host-device transfer with automatic synchronization.

```python
@veda.gpu
def matrix_multiply(A, B):
    return A @ B  # Runs on GPU, transfers managed automatically
```

**Guarantees:**
- ✅ Automatic host→device transfer before execution
- ✅ Automatic device→host transfer after execution
- ✅ Synchronization points at kernel boundaries
- ⚠️ Device memory not directly accessible from host

---

## Error Handling & Propagation

### Exception Propagation

**Guarantee:** All exceptions in child tasks are **propagated to the caller**.

```python
def failing_task(x):
    if x == 5:
        raise ValueError("Bad value")
    return x * 2

try:
    result = veda.par_iter(range(10)).map(failing_task).collect()
except ValueError as e:
    print(f"Caught: {e}")  # ✅ Exception propagated
```

**Behavior:**
- ✅ First exception encountered **stops iteration** (fail-fast)
- ✅ Exception includes **full traceback** from worker
- ✅ Other in-flight tasks may complete or be cancelled
- ✅ Resources are cleaned up before exception propagates

---

## Cancellation Semantics

### Scope Cancellation

**Guarantee:** Cancelling a scope attempts to cancel all child tasks.

```python
with veda.scope() as s:
    for i in range(100):
        s.spawn(long_running_task, i)
    # Ctrl+C sends cancellation to all tasks
```

**Behavior:**
- ✅ **Best-effort cancellation** (not guaranteed immediate)
- ✅ Thread executor: cooperative cancellation
- ✅ Process executor: SIGTERM, wait for cleanup
- ✅ Async executor: `asyncio.CancelledError`

---

## Deterministic Execution

### Deterministic Mode Guarantees

**Ordering Guarantee:** With `deterministic(seed=N)`, task **scheduling order** is deterministic.

```python
with veda.deterministic(seed=42):
    result1 = veda.par_iter(range(100)).map(random_func).collect()

with veda.deterministic(seed=42):
    result2 = veda.par_iter(range(100)).map(random_func).collect()

assert result1 == result2  # ✅ Same scheduling order
```

**What IS Guaranteed:**
- ✅ **Task scheduling order** is reproducible
- ✅ **RNG state** is seeded identically per task
- ✅ **Executor selection** follows same pattern
- ✅ **Worker assignment** is deterministic

**What IS NOT Guaranteed:**
- ❌ **Bitwise floating-point reproducibility** (hardware-dependent)
- ❌ **System call timing** (OS-dependent)
- ❌ **External I/O order** (network, disk)

---

## Resource Management

### Executor Lifecycle

**Guarantee:** Executors are created lazily and cleaned up automatically.

```python
import vedart as veda

result = veda.par_iter(range(10)).map(lambda x: x**2).collect()
# ✅ Thread pool created on first use
# ✅ Cleaned up on program exit
```

### Pool Sizing

**Guarantee:** Worker pools scale dynamically using Little's Law.

**Adaptive Behavior:**
- ✅ Pools grow when queue depth increases
- ✅ Pools shrink during idle periods
- ✅ Respects `Config.min_workers` and `Config.max_workers`

---

## Thread Safety

### Public API Thread Safety

**Guarantee:** All public VedaRT APIs are **thread-safe**.

```python
import threading
import vedart as veda

def worker():
    result = veda.par_iter(range(100)).map(lambda x: x**2).sum()

# ✅ Safe to call from multiple threads
threads = [threading.Thread(target=worker) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## Summary Table

| Feature | Guarantee Level | Notes |
|---------|----------------|-------|
| Task order | ❌ Unordered | Except `collect()` preserves input order |
| Memory isolation (process) | ✅ Complete | No shared state |
| Memory safety (thread) | ⚠️ GIL-protected | Python object atomicity only |
| Exception propagation | ✅ Always | Fail-fast by default |
| Resource cleanup | ✅ Automatic | Via scope or runtime shutdown |
| Cancellation | ⚠️ Best-effort | Not immediate |
| Deterministic scheduling | ✅ With seed | Not bitwise reproducible |
| Thread safety | ✅ All APIs | Internal state protected |

---

## References

- [Architecture Documentation](architecture.md)
- [API Reference](api_reference.md)
- [Telemetry Guide](telemetry.md)
