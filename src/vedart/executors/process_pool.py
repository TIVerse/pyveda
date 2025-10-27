"""Process pool executor implementation."""

import logging
import os
import threading
from collections.abc import Callable
from concurrent.futures import Future
from concurrent.futures import ProcessPoolExecutor as StdProcessPoolExecutor
from typing import Any

from vedart.core.executor import BaseExecutor
from vedart.core.task import Task

logger = logging.getLogger(__name__)


def _execute_task_func(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Module-level function to execute task in subprocess.

    This needs to be at module level to be picklable.

    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Function result
    """
    return func(*args, **kwargs)


class ProcessPoolExecutor(BaseExecutor):
    """Executor using process pool for CPU-bound tasks.

    Wraps Python's ProcessPoolExecutor for true parallelism.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        name: str = "process-pool",
    ) -> None:
        """Initialize process pool executor.

        Args:
            max_workers: Maximum number of processes (None = cpu_count)
            name: Executor name for logging
        """
        super().__init__(name)

        if max_workers is None:
            # Use physical cores, not logical (avoid hyperthreading)
            max_workers = os.cpu_count() or 4

        self._max_workers = max_workers
        self._current_workers = max_workers
        self._pool = StdProcessPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()

        logger.info(f"ProcessPoolExecutor initialized with {max_workers} workers")

    def submit(self, task: Task) -> Future[Any]:
        """Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Future for the result
        """
        if self._shutdown:
            raise RuntimeError("Executor is shutdown")

        # Submit task function directly (must be picklable)
        return self._pool.submit(_execute_task_func, task.func, task.args, task.kwargs)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for pending tasks
        """
        with self._lock:
            if not self._shutdown:
                self._shutdown = True
                self._pool.shutdown(wait=wait)
                logger.info(f"{self.name} shutdown")

    def is_available(self) -> bool:
        """Check if executor is available.

        Returns:
            True if not shutdown
        """
        return not self._shutdown

    def scale(self, num_workers: int) -> None:
        """Scale the process pool.

        Note: ProcessPoolExecutor doesn't support dynamic scaling,
        so this is a no-op. A production implementation could
        recreate the pool.

        Args:
            num_workers: Desired worker count
        """
        if num_workers == self._current_workers:
            return

        # ProcessPoolExecutor doesn't support resizing
        # Production implementation would recreate pool
        logger.debug(
            f"Process pool scaling requested to {num_workers} (not implemented)"
        )
        self._current_workers = num_workers
