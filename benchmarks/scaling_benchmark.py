"""Benchmark scaling characteristics of PyVeda."""

import time
from typing import List, Tuple

import pyveda as veda


def compute_task(n: int) -> int:
    """Fixed-cost computation task.
    
    Args:
        n: Input value
        
    Returns:
        Result
    """
    return sum(i ** 2 for i in range(n))


def measure_scaling(
    num_tasks: int,
    num_workers: int,
    task_size: int = 1000
) -> Tuple[float, float]:
    """Measure performance with given worker count.
    
    Args:
        num_tasks: Number of tasks to execute
        num_workers: Number of worker threads
        task_size: Size parameter for tasks
        
    Returns:
        Tuple of (elapsed_time, throughput)
    """
    # Reinitialize runtime with specific worker count
    veda.shutdown()
    config = veda.Config.builder().threads(num_workers).build()
    veda.init(config)
    
    # Run benchmark
    tasks = [task_size] * num_tasks
    start = time.perf_counter()
    results = veda.par_iter(tasks).map(compute_task).collect()
    elapsed = time.perf_counter() - start
    
    throughput = num_tasks / elapsed
    return elapsed, throughput


def main():
    """Run scaling benchmarks."""
    print("=" * 60)
    print("PyVeda Scaling Benchmark")
    print("=" * 60)
    print()
    
    num_tasks = 100
    task_size = 1000
    worker_counts = [1, 2, 4, 8, 16]
    
    print(f"Workload: {num_tasks} tasks @ {task_size} iterations each")
    print("-" * 60)
    
    results: List[Tuple[int, float, float]] = []
    
    for num_workers in worker_counts:
        print(f"Testing with {num_workers} workers...", end=" ", flush=True)
        elapsed, throughput = measure_scaling(num_tasks, num_workers, task_size)
        results.append((num_workers, elapsed, throughput))
        print("✓")
    
    # Print results
    print("\nResults:")
    print("-" * 60)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 60)
    
    baseline_time = results[0][1]
    
    for num_workers, elapsed, throughput in results:
        speedup = baseline_time / elapsed
        efficiency = (speedup / num_workers) * 100
        
        print(
            f"{num_workers:<10} "
            f"{elapsed:<12.3f} "
            f"{throughput:<15.1f} "
            f"{speedup:<10.2f} "
            f"{efficiency:<10.1f}%"
        )
    
    # Analyze scaling
    print("\nScaling Analysis:")
    print("-" * 60)
    
    # Strong scaling efficiency (fixed workload)
    for i in range(1, len(results)):
        workers_prev, time_prev, _ = results[i-1]
        workers_curr, time_curr, _ = results[i]
        
        ideal_speedup = workers_curr / workers_prev
        actual_speedup = time_prev / time_curr
        efficiency = (actual_speedup / ideal_speedup) * 100
        
        print(
            f"{workers_prev} → {workers_curr} workers: "
            f"Speedup {actual_speedup:.2f}x (ideal: {ideal_speedup:.2f}x, "
            f"efficiency: {efficiency:.1f}%)"
        )
    
    print()
    veda.shutdown()


if __name__ == "__main__":
    main()
