"""Scoped execution for structured parallelism."""

import logging
from collections.abc import Callable
from concurrent.futures import Future as StdFuture
from typing import Any, TypeVar

from vedart.core.task import Task, TaskPriority
from vedart.exceptions import VedaError

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Scope:
    """Structured parallelism scope.

    Provides scoped execution where all spawned tasks
    are awaited before exiting the scope.

    Example:
        with scope() as s:
            f1 = s.spawn(lambda: task1())
            f2 = s.spawn(lambda: task2())
            results = s.wait_all()
    """

    def __init__(self) -> None:
        self._futures: list[StdFuture[Any]] = []
        self._closed = False

    def spawn(
        self,
        func: Callable[..., T],
        *args: Any,
        priority: TaskPriority = TaskPriority.NORMAL,
        **kwargs: Any,
    ) -> StdFuture[T]:
        """Spawn a task within the scope.

        Args:
            func: Function to execute
            *args: Positional arguments
            priority: Task priority
            **kwargs: Keyword arguments

        Returns:
            Future for the result

        Raises:
            VedaError: If scope is closed
        """
        if self._closed:
            raise VedaError("Cannot spawn tasks on closed scope")

        # Import here to avoid circular dependency
        from vedart.core.runtime import get_runtime

        runtime = get_runtime()
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
        )
        future = runtime.scheduler.submit(task)
        self._futures.append(future)
        return future

    def wait_all(self, timeout: float | None = None) -> list[Any]:
        """Wait for all spawned tasks to complete.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            List of results in spawn order

        Raises:
            TimeoutError: If timeout expires
            Exception: If any task failed
        """
        results = []
        for future in self._futures:
            results.append(future.result(timeout=timeout))
        return results

    def __enter__(self) -> "Scope":
        """Enter the scope context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the scope, waiting for all tasks."""
        self._closed = True
        if not exc_type:
            # Only wait if no exception occurred
            try:
                self.wait_all()
            except Exception as e:
                logger.error(f"Task failed in scope: {e}")
                raise


def scope() -> Scope:
    """Create a new execution scope.

    Returns:
        New Scope instance
    """
    return Scope()


def spawn(
    func: Callable[..., T],
    *args: Any,
    priority: TaskPriority = TaskPriority.NORMAL,
    **kwargs: Any,
) -> StdFuture[T]:
    """Spawn a task on the runtime without a scope.

    Args:
        func: Function to execute
        *args: Positional arguments
        priority: Task priority
        **kwargs: Keyword arguments

    Returns:
        Future for the result
    """
    from vedart.core.runtime import get_runtime

    runtime = get_runtime()
    task = Task(
        func=func,
        args=args,
        kwargs=kwargs,
        priority=priority,
    )
    return runtime.scheduler.submit(task)
