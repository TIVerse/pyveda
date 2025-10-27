"""Basic performance benchmarks."""

import time

import pyveda as veda


def benchmark_parallel_sum(size):
    """Benchmark parallel sum operation."""
    start = time.perf_counter()
    result = veda.par_iter(range(size)).sum()
    elapsed = time.perf_counter() - start
    return elapsed, result


def benchmark_sequential_sum(size):
    """Benchmark sequential sum."""
    start = time.perf_counter()
    result = sum(range(size))
    elapsed = time.perf_counter() - start
    return elapsed, result


def benchmark_parallel_map(size):
    """Benchmark parallel map."""
    start = time.perf_counter()
    result = veda.par_iter(range(size)).map(lambda x: x * 2).collect()
    elapsed = time.perf_counter() - start
    return elapsed, len(result)


def main():
    """Run benchmarks."""
    print("VedaRT Benchmarks\n")
    print("=" * 60)
    
    sizes = [1000, 10000, 100000]
    
    for size in sizes:
        print(f"\nDataset size: {size:,}")
        print("-" * 60)
        
        # Sum benchmark
        seq_time, seq_result = benchmark_sequential_sum(size)
        par_time, par_result = benchmark_parallel_sum(size)
        
        print(f"Sum operation:")
        print(f"  Sequential: {seq_time*1000:.2f}ms")
        print(f"  Parallel:   {par_time*1000:.2f}ms")
        print(f"  Speedup:    {seq_time/par_time:.2f}x")
        print(f"  Correct:    {seq_result == par_result}")
        
        # Map benchmark
        map_time, map_count = benchmark_parallel_map(size)
        print(f"\nMap operation:")
        print(f"  Parallel:   {map_time*1000:.2f}ms")
        print(f"  Throughput: {map_count/map_time:,.0f} items/sec")


if __name__ == "__main__":
    veda.init()
    main()
    veda.shutdown()
