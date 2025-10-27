"""Task representation and management."""

import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class TaskState(Enum):
    """State of a task in the execution pipeline."""

    PENDING = auto()
    SCHEDULED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class TaskPriority(Enum):
    """Priority levels for task scheduling."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Task:
    """Represents a unit of work to be executed.

    Attributes:
        id: Unique identifier for the task
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        priority: Task priority
        estimated_duration_ms: Estimated execution time
        is_async: Whether func is a coroutine
        state: Current task state
        result: Result value (after completion)
        error: Exception (if failed)
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    func: Callable[..., Any] = field(default=lambda: None)
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    estimated_duration_ms: float | None = None
    is_async: bool = False
    state: TaskState = TaskState.PENDING
    result: Any = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        """Auto-detect async functions if not explicitly set."""
        if not self.is_async and inspect.iscoroutinefunction(self.func):
            self.is_async = True

    def __lt__(self, other: "Task") -> bool:
        """Compare tasks by priority for priority queue."""
        return (
            self.priority.value > other.priority.value
        )  # Higher priority = lower value

    def execute(self) -> Any:
        """Execute the task synchronously.

        Returns:
            Result of the function call

        Raises:
            Any exception raised by the function
        """
        try:
            self.state = TaskState.RUNNING
            result = self.func(*self.args, **self.kwargs)
            self.result = result
            self.state = TaskState.COMPLETED
            return result
        except Exception as e:
            self.error = e
            self.state = TaskState.FAILED
            raise

    def cancel(self) -> bool:
        """Attempt to cancel the task.

        Returns:
            True if cancellation succeeded
        """
        if self.state in (TaskState.PENDING, TaskState.SCHEDULED):
            self.state = TaskState.CANCELLED
            return True
        return False

    def is_done(self) -> bool:
        """Check if task has completed execution.

        Returns:
            True if task is in terminal state
        """
        return self.state in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELLED,
        )
