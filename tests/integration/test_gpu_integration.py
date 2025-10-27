"""Integration tests for GPU functionality."""

import pytest

import vedart as veda


def test_gpu_runtime_initialization():
    """Test GPU runtime initialization."""
    # Runtime is already initialized in test environment
    runtime = veda.get_runtime()

    # GPU may or may not be available depending on hardware
    if hasattr(runtime, "gpu") and runtime.gpu and runtime.gpu.is_available():
        assert runtime.gpu.backend in ["cupy", "numba", "torch"]
        assert runtime.gpu.device_count > 0


def test_gpu_decorator_fallback():
    """Test GPU decorator falls back to CPU when GPU unavailable."""

    @veda.gpu
    def simple_computation(x):
        return x**2

    # Should work regardless of GPU availability
    result = simple_computation(5)
    assert result == 25


def test_gpu_memory_stats():
    """Test GPU memory statistics retrieval."""
    runtime = veda.get_runtime()

    if hasattr(runtime, "gpu") and runtime.gpu and runtime.gpu.is_available():
        stats = runtime.gpu.get_memory_stats()

        assert "used_mb" in stats
        assert "free_mb" in stats
        assert "total_mb" in stats
        assert all(v >= 0 for v in stats.values())

        utilization = runtime.gpu.get_utilization()
        assert 0 <= utilization <= 100


def test_gpu_executor_registration():
    """Test GPU executor is registered when available."""
    runtime = veda.get_runtime()

    if hasattr(runtime, "gpu") and runtime.gpu and runtime.gpu.is_available():
        from vedart.config import ExecutorType

        # GPU executor should be registered
        gpu_executor = runtime.scheduler._executors.get(ExecutorType.GPU)
        assert gpu_executor is not None
        assert gpu_executor.is_available()


@pytest.mark.skipif(not hasattr(veda, "gpu"), reason="GPU support not available")
def test_gpu_with_iterator():
    """Test GPU integration with parallel iterator."""
    pytest.importorskip("numpy")
    import numpy as np

    @veda.gpu
    def square_array(arr):
        return arr**2

    # Small arrays (CPU execution)
    arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    results = veda.par_iter(arrays).map(square_array).collect()

    assert len(results) == 2
    assert all(isinstance(r, np.ndarray) for r in results)
