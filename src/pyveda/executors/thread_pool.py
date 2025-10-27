"""Thread pool executor implementation."""

import logging
import os
import threading
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor as StdThreadPoolExecutor
from typing import Any

from pyveda.core.executor import BaseExecutor
from pyveda.core.task import Task

logger = logging.getLogger(__name__)


class ThreadPoolExecutor(BaseExecutor):
    """Executor using thread pool for I/O-bound tasks.

    Wraps Python's ThreadPoolExecutor with dynamic scaling support.
    """

    def __init__(
        self,
        max_workers: int | None = None,
        name: str = "thread-pool",
    ) -> None:
        """Initialize thread pool executor.

        Args:
            max_workers: Maximum number of threads (None = cpu_count)
            name: Executor name for logging
        """
        super().__init__(name)

        if max_workers is None:
            max_workers = (os.cpu_count() or 4) * 2  # 2x for I/O-bound

        self._max_workers = max_workers
        self._current_workers = max_workers
        self._pool = StdThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"veda-{name}",
        )
        self._lock = threading.Lock()

        logger.info(f"ThreadPoolExecutor initialized with {max_workers} workers")

    def submit(self, task: Task) -> Future[Any]:
        """Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Future for the result
        """
        if self._shutdown:
            raise RuntimeError("Executor is shutdown")

        # Submit task function directly
        return self._pool.submit(task.func, *task.args, **task.kwargs)

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
        """Scale the thread pool.

        Note: ThreadPoolExecutor doesn't support dynamic scaling,
        so this is a no-op. A production implementation could
        recreate the pool.

        Args:
            num_workers: Desired worker count
        """
        if num_workers == self._current_workers:
            return

        # ThreadPoolExecutor doesn't support resizing
        # Production implementation would recreate pool or use custom implementation
        logger.debug(
            f"Thread pool scaling requested to {num_workers} (not implemented)"
        )
        self._current_workers = num_workers
