"""Integration tests with real workloads."""

import time

import pytest

from pyveda.core.scope import scope
from pyveda.iter.parallel import par_iter


def test_data_processing_pipeline(cleanup_runtime):
    """Test realistic data processing pipeline."""
    # Simulate data processing
    data = range(1000)
    
    result = (
        par_iter(data)
        .map(lambda x: x * 2)  # Transform
        .filter(lambda x: x % 4 == 0)  # Filter
        .map(lambda x: x // 2)  # Normalize
        .collect()
    )
    
    # Verify correctness
    expected = [x for x in range(1000) if (x * 2) % 4 == 0]
    assert len(result) == len(expected)


def test_parallel_computation_scope(cleanup_runtime):
    """Test parallel computation with scopes."""
    def compute_chunk(start, end):
        return sum(range(start, end))
    
    with scope() as s:
        # Split work into chunks
        chunk_size = 250
        futures = []
        for i in range(0, 1000, chunk_size):
            futures.append(s.spawn(compute_chunk, i, min(i + chunk_size, 1000)))
        
        results = s.wait_all()
    
    total = sum(results)
    expected = sum(range(1000))
    assert total == expected


def test_nested_parallelism(cleanup_runtime):
    """Test nested parallel operations."""
    def process_batch(batch):
        # Inner parallel operation
        return par_iter(batch).map(lambda x: x ** 2).sum()
    
    # Outer parallel operation
    batches = [list(range(i * 10, (i + 1) * 10)) for i in range(10)]
    results = par_iter(batches).map(process_batch).collect()
    
    assert len(results) == 10
    assert all(isinstance(r, int) for r in results)


def test_mixed_operations(cleanup_runtime):
    """Test mixing different parallel primitives."""
    with scope() as s:
        # Spawn some tasks
        f1 = s.spawn(lambda: par_iter(range(100)).sum())
        f2 = s.spawn(lambda: par_iter(range(100, 200)).sum())
        
        results = s.wait_all()
    
    assert results[0] == sum(range(100))
    assert results[1] == sum(range(100, 200))
