"""Stress tests for memory leaks."""

import gc

import psutil
import pytest

from pyveda.iter.parallel import par_iter


def get_memory_mb():
    """Get current process memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


@pytest.mark.slow
def test_no_memory_leak_iterations(cleanup_runtime):
    """Test for memory leaks in repeated iterations."""
    # Warm up
    for _ in range(10):
        par_iter(range(100)).map(lambda x: x * 2).sum()
    
    gc.collect()
    initial_memory = get_memory_mb()
    
    # Run many iterations
    for _ in range(1000):
        par_iter(range(100)).map(lambda x: x * 2).sum()
    
    gc.collect()
    final_memory = get_memory_mb()
    
    # Memory growth should be minimal (< 20MB)
    memory_growth = final_memory - initial_memory
    assert memory_growth < 20, f"Memory grew by {memory_growth:.1f}MB"


@pytest.mark.slow
def test_no_memory_leak_scopes(cleanup_runtime):
    """Test for memory leaks in scopes."""
    from pyveda.core.scope import scope
    
    # Warm up
    for _ in range(10):
        with scope() as s:
            for _ in range(10):
                s.spawn(lambda: 42)
    
    gc.collect()
    initial_memory = get_memory_mb()
    
    # Run many scopes
    for _ in range(100):
        with scope() as s:
            for _ in range(10):
                s.spawn(lambda: 42)
    
    gc.collect()
    final_memory = get_memory_mb()
    
    memory_growth = final_memory - initial_memory
    assert memory_growth < 20, f"Memory grew by {memory_growth:.1f}MB"
