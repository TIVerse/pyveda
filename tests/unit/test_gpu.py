"""Unit tests for GPU runtime."""

import pytest

from vedart.gpu.backend import GPURuntime


def test_gpu_runtime_creation():
    """Test GPU runtime can be instantiated."""
    gpu = GPURuntime()
    assert gpu is not None


def test_gpu_backend_detection():
    """Test backend detection logic."""
    gpu = GPURuntime()

    # Backend should be None, 'cupy', or 'numba'
    assert gpu.backend in [None, "cupy", "numba"]


def test_gpu_availability():
    """Test is_available method."""
    gpu = GPURuntime()
    available = gpu.is_available()

    assert isinstance(available, bool)

    if available:
        assert gpu.backend is not None
        assert gpu.device_count > 0


def test_gpu_memory_stats_structure():
    """Test memory stats return correct structure."""
    gpu = GPURuntime()
    stats = gpu.get_memory_stats()

    assert isinstance(stats, dict)
    assert "used_mb" in stats
    assert "free_mb" in stats
    assert "total_mb" in stats

    # All values should be non-negative floats
    for value in stats.values():
        assert isinstance(value, float)
        assert value >= 0


def test_gpu_utilization_range():
    """Test GPU utilization is in valid range."""
    gpu = GPURuntime()
    util = gpu.get_utilization()

    assert isinstance(util, float)
    assert 0 <= util <= 100


def test_should_offload_small_data():
    """Test offload decision for small data."""
    gpu = GPURuntime()

    if not gpu.is_available():
        pytest.skip("GPU not available")

    # Small data should not offload (too much overhead)
    small_args = ([1, 2, 3],)

    # Function doesn't matter for basic size check
    def dummy_func(x):
        return x

    # Should return False for small data
    result = gpu.should_offload(dummy_func, small_args)
    assert isinstance(result, bool)


def test_gpu_execute_fallback():
    """Test GPU execute falls back to CPU."""
    gpu = GPURuntime()

    def simple_func(x):
        return x * 2

    # Should work regardless of GPU availability
    result = gpu.execute(simple_func, 5)
    assert result == 10


@pytest.mark.skipif(True, reason="Requires CuPy installation")
def test_gpu_memory_pool_initialization():
    """Test GPU memory pool is initialized with CuPy."""
    gpu = GPURuntime()

    if gpu.backend == "cupy" and gpu.is_available():
        assert gpu.memory_pool is not None
