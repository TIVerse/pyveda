"""Scoped execution example."""

import time

import vedart as veda


def expensive_computation(n):
    """Simulate expensive computation."""
    time.sleep(0.1)
    return n ** 2


def main():
    """Demonstrate scoped parallel execution."""
    print("PyVeda - Scoped Execution\n")
    
    # Using scope for structured parallelism
    print("Processing tasks in parallel scope...")
    start = time.time()
    
    with veda.scope() as s:
        # Spawn multiple tasks
        futures = [s.spawn(expensive_computation, i) for i in range(5)]
        
        # Wait for all tasks to complete
        results = s.wait_all()
    
    elapsed = time.time() - start
    
    print(f"Results: {results}")
    print(f"Time: {elapsed:.2f}s (should be ~0.1s with parallelism)\n")
    
    # Compare with sequential execution
    print("Sequential execution for comparison...")
    start = time.time()
    results_seq = [expensive_computation(i) for i in range(5)]
    elapsed_seq = time.time() - start
    
    print(f"Results: {results_seq}")
    print(f"Time: {elapsed_seq:.2f}s\n")
    
    print(f"Speedup: {elapsed_seq / elapsed:.1f}x")


if __name__ == "__main__":
    main()
    veda.shutdown()
