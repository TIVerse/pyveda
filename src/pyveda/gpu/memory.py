"""GPU memory management utilities."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class GPUMemoryPool:
    """GPU memory pool for efficient allocation.

    Provides memory pooling to reduce allocation overhead.
    """

    def __init__(self) -> None:
        """Initialize memory pool."""
        self._pool: Any | None = None
        self._backend: str | None = None
        self._init_pool()

    def _init_pool(self) -> None:
        """Initialize backend-specific pool."""
        try:
            import cupy as cp

            self._pool = cp.cuda.MemoryPool()
            cp.cuda.set_allocator(self._pool.malloc)
            self._backend = "cupy"
            logger.info("Initialized CuPy memory pool")
        except ImportError:
            logger.debug("CuPy not available for memory pooling")

    def get_memory_info(self) -> tuple[int, int]:
        """Get memory usage information.

        Returns:
            Tuple of (used_bytes, total_bytes)
        """
        if self._backend == "cupy":
            import cupy as cp

            free, total = cp.cuda.Device().mem_info()
            used = total - free
            return (used, total)
        return (0, 0)

    def clear_pool(self) -> None:
        """Clear the memory pool."""
        if self._pool and self._backend == "cupy":
            self._pool.free_all_blocks()
            logger.debug("Cleared GPU memory pool")
