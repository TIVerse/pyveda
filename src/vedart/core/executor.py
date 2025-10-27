"""Base executor interface and implementations."""

from abc import ABC, abstractmethod
from concurrent.futures import Future
from typing import Any

from vedart.core.task import Task


class BaseExecutor(ABC):
    """Abstract base class for all executors.

    Executors handle the actual task execution using different
    concurrency primitives (threads, processes, async, GPU).
    """

    def __init__(self, name: str) -> None:
        """Initialize executor.

        Args:
            name: Human-readable executor name
        """
        self.name = name
        self._shutdown = False

    @abstractmethod
    def submit(self, task: Task) -> Future[Any]:
        """Submit a task for execution.

        Args:
            task: Task to execute

        Returns:
            Future for the result
        """
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if executor is available for work.

        Returns:
            True if executor can accept tasks
        """
        pass

    @abstractmethod
    def scale(self, num_workers: int) -> None:
        """Scale the executor to the specified worker count.

        Args:
            num_workers: Desired number of workers
        """
        pass

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name={self.name})"
