"""Benchmark comparing PyVeda with other frameworks."""

import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable, List

import pyveda as veda


def cpu_task(n: int) -> int:
    """CPU-intensive task for benchmarking.
    
    Args:
        n: Input value
        
    Returns:
        Computed result
    """
    return sum(i ** 2 for i in range(n))


def io_task(duration_ms: int) -> int:
    """IO-bound task simulation.
    
    Args:
        duration_ms: Sleep duration in milliseconds
        
    Returns:
        Duration
    """
    time.sleep(duration_ms / 1000.0)
    return duration_ms


def benchmark_framework(
    name: str,
    executor_fn: Callable,
    tasks: List,
    warmup: int = 10
) -> dict:
    """Benchmark a parallel framework.
    
    Args:
        name: Framework name
        executor_fn: Function that executes tasks
        tasks: List of task inputs
        warmup: Number of warmup iterations
        
    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup):
        executor_fn(tasks[:10])
    
    # Actual benchmark
    start = time.perf_counter()
    results = executor_fn(tasks)
    elapsed = time.perf_counter() - start
    
    throughput = len(tasks) / elapsed
    latency_ms = (elapsed / len(tasks)) * 1000
    
    return {
        'name': name,
        'tasks': len(tasks),
        'elapsed_sec': elapsed,
        'throughput_tasks_per_sec': throughput,
        'avg_latency_ms': latency_ms,
        'results': results
    }


def run_pyveda(tasks: List[int]) -> List[int]:
    """Run tasks with PyVeda.
    
    Args:
        tasks: Task inputs
        
    Returns:
        Results
    """
    return veda.par_iter(tasks).map(cpu_task).collect()


def run_thread_pool(tasks: List[int]) -> List[int]:
    """Run tasks with ThreadPoolExecutor.
    
    Args:
        tasks: Task inputs
        
    Returns:
        Results
    """
    with ThreadPoolExecutor(max_workers=4) as executor:
        return list(executor.map(cpu_task, tasks))


def run_process_pool(tasks: List[int]) -> List[int]:
    """Run tasks with ProcessPoolExecutor.
    
    Args:
        tasks: Task inputs
        
    Returns:
        Results
    """
    with ProcessPoolExecutor(max_workers=4) as executor:
        return list(executor.map(cpu_task, tasks))


def run_sequential(tasks: List[int]) -> List[int]:
    """Run tasks sequentially.
    
    Args:
        tasks: Task inputs
        
    Returns:
        Results
    """
    return [cpu_task(t) for t in tasks]


def main():
    """Run framework comparison benchmarks."""
    print("=" * 60)
    print("PyVeda Framework Comparison Benchmark")
    print("=" * 60)
    print()
    
    # Initialize PyVeda
    config = veda.Config.builder().threads(4).build()
    veda.init(config)
    
    # CPU-bound workload
    print("Workload: CPU-bound tasks (100 tasks, ~10ms each)")
    print("-" * 60)
    
    cpu_tasks = [1000] * 100
    
    frameworks = [
        ("Sequential", run_sequential),
        ("ThreadPoolExecutor", run_thread_pool),
        ("ProcessPoolExecutor", run_process_pool),
        ("PyVeda", run_pyveda),
    ]
    
    results = []
    for name, executor_fn in frameworks:
        print(f"\nBenchmarking {name}...", end=" ", flush=True)
        result = benchmark_framework(name, executor_fn, cpu_tasks)
        results.append(result)
        print("✓")
    
    # Print results table
    print("\nResults:")
    print("-" * 60)
    print(f"{'Framework':<25} {'Time (s)':<12} {'Throughput':<15} {'Latency (ms)':<12}")
    print("-" * 60)
    
    for result in results:
        print(
            f"{result['name']:<25} "
            f"{result['elapsed_sec']:<12.3f} "
            f"{result['throughput_tasks_per_sec']:<15.1f} "
            f"{result['avg_latency_ms']:<12.2f}"
        )
    
    # Calculate speedup
    baseline = results[0]['elapsed_sec']
    print("\nSpeedup vs Sequential:")
    print("-" * 60)
    for result in results[1:]:
        speedup = baseline / result['elapsed_sec']
        print(f"{result['name']:<25} {speedup:.2f}x")
    
    print()
    
    veda.shutdown()


if __name__ == "__main__":
    main()
