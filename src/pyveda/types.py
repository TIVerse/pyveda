"""Type definitions and protocols for PyVeda."""

import sys
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

if sys.version_info >= (3, 11):
    pass
else:
    pass

# Type variables for generic operations
T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)

# Callable type aliases
MapFunc = Callable[[T], U]
FilterFunc = Callable[[T], bool]
FoldFunc = Callable[[U, T], U]
AsyncMapFunc = Callable[[T], Any]  # Returns Awaitable[U]


class Executor(Protocol):
    """Protocol defining executor interface.

    All executors (thread, process, async, GPU) must implement this interface.
    """

    def submit(self, task: "Task") -> "Future[Any]":
        """Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Future representing the result
        """
        ...

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        ...

    def is_available(self) -> bool:
        """Check if executor is available for work.

        Returns:
            True if executor can accept tasks
        """
        ...


class Future(Protocol[T_co]):
    """Protocol for future objects."""

    def result(self, timeout: float | None = None) -> T_co:
        """Get the result, blocking if necessary.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            The result value

        Raises:
            TimeoutError: If timeout expires
        """
        ...

    def done(self) -> bool:
        """Check if the future is done.

        Returns:
            True if the future has completed
        """
        ...

    def cancel(self) -> bool:
        """Attempt to cancel execution.

        Returns:
            True if cancellation succeeded
        """
        ...


class Task(Protocol):
    """Protocol for task objects."""

    id: str
    func: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
