"""GPU decorators for automatic offload."""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from vedart.exceptions import GPUError

logger = logging.getLogger(__name__)

T = TypeVar("T")


def gpu(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for automatic GPU offload.

    Automatically offloads function to GPU if beneficial,
    otherwise executes on CPU.

    Args:
        func: Function to wrap

    Returns:
        Wrapped function

    Example:
        @gpu
        def matrix_multiply(A, B):
            return A @ B
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Get GPU runtime
        from vedart.core.runtime import get_runtime

        runtime = get_runtime()

        if runtime.gpu and runtime.gpu.is_available():
            # Check if offload is beneficial
            if runtime.gpu.should_offload(func, args):
                try:
                    logger.debug(f"Offloading {func.__name__} to GPU")
                    return runtime.gpu.execute(func, *args, **kwargs)
                except Exception as e:
                    logger.warning(f"GPU execution failed, falling back to CPU: {e}")
                    # Fallback to CPU
                    return func(*args, **kwargs)

        # Execute on CPU
        return func(*args, **kwargs)

    return wrapper


def gpu_kernel(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator for GPU kernel functions.

    Marks function as a GPU kernel (requires Numba @cuda.jit).

    Args:
        func: Kernel function

    Returns:
        Wrapped function
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        from vedart.core.runtime import get_runtime

        runtime = get_runtime()

        if not runtime.gpu or not runtime.gpu.is_available():
            raise GPUError("GPU not available for kernel execution")

        if runtime.gpu.backend != "numba":
            raise GPUError("GPU kernels require Numba backend")

        # Execute kernel
        return func(*args, **kwargs)

    return wrapper
