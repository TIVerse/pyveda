"""GPU runtime backend with CuPy/Numba support."""

import logging
import sys
from typing import Any, Callable, Optional

from pyveda.exceptions import GPUError

logger = logging.getLogger(__name__)


class GPURuntime:
    """GPU runtime for automatic GPU offload.
    
    Detects available GPU backends (CuPy, Numba) and provides
    unified interface for GPU execution with cost-based offload.
    """

    def __init__(self) -> None:
        """Initialize GPU runtime."""
        self.backend: Optional[str] = None
        self.device_count = 0
        self._cp: Optional[Any] = None  # CuPy module
        self._numba_cuda: Optional[Any] = None  # Numba CUDA module
        self.memory_pool: Optional[Any] = None  # GPU memory pool

        self._detect_backend()
        self._init_memory_pool()
        logger.info(f"GPU runtime initialized: backend={self.backend}")

    def _detect_backend(self) -> bool:
        """Detect available GPU backend.
        
        Returns:
            True if GPU is available
        """
        # Try CuPy first
        try:
            import cupy as cp

            self._device_count = cp.cuda.runtime.getDeviceCount()
            if self._device_count > 0:
                self._available = True
                self._backend = "cupy"
                logger.info(f"GPU detected: CuPy with {self._device_count} device(s)")
                return True
        except Exception as e:
            logger.debug(f"CuPy not available: {e}")

        # Try Numba
        try:
            from numba import cuda

            if cuda.is_available():
                self._device_count = len(cuda.gpus)
                self._available = True
                self._backend = "numba"
                logger.info(f"GPU detected: Numba with {self._device_count} device(s)")
                return True
        except Exception as e:
            logger.debug(f"Numba CUDA not available: {e}")

        logger.info("No GPU backend available")
        return False

    def is_available(self) -> bool:
        """Check if GPU is available.
        
        Returns:
            True if GPU is available
        """
        return self._available

    @property
    def backend(self) -> Optional[str]:
        """Get the active backend name.
        
        Returns:
            Backend name or None
        """
        return self._backend

    @property
    def device_count(self) -> int:
        """Get number of available GPUs.
        
        Returns:
            Number of GPUs
        """
        return self._device_count

    def should_offload(self, func: Callable[..., Any], args: tuple[Any, ...]) -> bool:
        """Determine if function should be offloaded to GPU.
        
        Uses cost model based on data size and memory availability.
        
        Args:
            func: Function to execute
            args: Function arguments
            
        Returns:
            True if GPU offload is beneficial
        """
        if not self._available:
            return False

        # Estimate data size
        data_size = sum(self._estimate_size(arg) for arg in args)
        
        # Small data: transfer overhead too high
        if data_size < 1024 * 1024:  # < 1MB
            return False

        # Check GPU memory availability (FIXED logic)
        if self._backend == "cupy":
            try:
                import cupy as cp

                free, total = cp.cuda.Device().mem_info()
                used_ratio = 1.0 - (free / total)  # FIXED: correct computation
                
                if used_ratio > 0.9:  # > 90% used
                    logger.debug(f"GPU memory usage high: {used_ratio:.1%}")
                    return False
            except Exception as e:
                logger.warning(f"Failed to check GPU memory: {e}")
                return False

        return True

    def execute(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute function on GPU.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result (transferred back to CPU)
            
        Raises:
            GPUError: If execution fails
        """
        if not self._available:
            raise GPUError("GPU not available")

        try:
            if self._backend == "cupy":
                return self._execute_cupy(func, *args, **kwargs)
            elif self._backend == "numba":
                return self._execute_numba(func, *args, **kwargs)
            else:
                raise GPUError(f"Unknown backend: {self._backend}")
        except Exception as e:
            raise GPUError(f"GPU execution failed: {e}") from e

    def _execute_cupy(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute using CuPy.
        
        Args:
            func: Function to execute
            *args: Arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result
        """
        import cupy as cp

        # Transfer args to GPU
        gpu_args = []
        for arg in args:
            if hasattr(arg, "__array__"):
                gpu_args.append(cp.asarray(arg))
            else:
                gpu_args.append(arg)

        # Execute
        result = func(*gpu_args, **kwargs)

        # Transfer result back to CPU
        if hasattr(result, "get"):
            return result.get()
        return result

    def _execute_numba(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute using Numba.
        
        Args:
            func: Function to execute
            *args: Arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result
        """
        # Numba execution
        # Note: Function must be decorated with @cuda.jit
        return func(*args, **kwargs)

    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes.
        
        Args:
            obj: Object to estimate
            
        Returns:
            Estimated size in bytes
        """
        if hasattr(obj, "nbytes"):
            return obj.nbytes
        elif hasattr(obj, "__len__"):
            return len(obj) * 8  # Rough estimate
        return sys.getsizeof(obj)

    def _init_memory_pool(self) -> None:
        """Initialize GPU memory pool if CuPy is available."""
        if self.backend == "cupy" and self._cp is not None:
            try:
                from pyveda.gpu.memory import GPUMemoryPool
                self.memory_pool = GPUMemoryPool(self._cp)
                # Set CuPy to use the memory pool
                self._cp.cuda.set_allocator(self.memory_pool.malloc)
                logger.info("GPU memory pool initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU memory pool: {e}")

    def get_memory_stats(self) -> dict[str, float]:
        """Get GPU memory statistics.
        
        Returns:
            Dictionary with memory stats (used_mb, free_mb, total_mb)
        """
        if not self.is_available():
            return {'used_mb': 0.0, 'free_mb': 0.0, 'total_mb': 0.0}

        try:
            if self.backend == "cupy" and self._cp is not None:
                mempool = self._cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                total_bytes = mempool.total_bytes()
                
                # Get device memory info
                free_mem, total_mem = self._cp.cuda.runtime.memGetInfo()
                
                return {
                    'used_mb': used_bytes / (1024 ** 2),
                    'free_mb': free_mem / (1024 ** 2),
                    'total_mb': total_mem / (1024 ** 2),
                }
            elif self.backend == "numba" and self._numba_cuda is not None:
                # Numba doesn't expose memory stats easily
                return {'used_mb': 0.0, 'free_mb': 0.0, 'total_mb': 0.0}
        except Exception as e:
            logger.debug(f"Failed to get GPU memory stats: {e}")
        
        return {'used_mb': 0.0, 'free_mb': 0.0, 'total_mb': 0.0}

    def get_utilization(self) -> float:
        """Get GPU utilization percentage.
        
        Returns:
            Utilization percentage (0-100)
        """
        try:
            if self.backend == "cupy" and self._cp is not None:
                # Try to use nvidia-smi via subprocess
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                if result.returncode == 0:
                    return float(result.stdout.strip().split('\n')[0])
        except Exception:
            pass
        
        # Fallback: estimate from memory usage
        stats = self.get_memory_stats()
        if stats['total_mb'] > 0:
            return (stats['used_mb'] / stats['total_mb']) * 100.0
        return 0.0
