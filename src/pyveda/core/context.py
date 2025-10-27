"""Execution context for tracking runtime state."""

import threading
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ExecutionContext:
    """Thread-local execution context.
    
    Tracks state for the current execution scope including
    active scheduler, telemetry, and deterministic mode.
    
    Attributes:
        task_id: ID of currently executing task
        thread_id: OS thread identifier
        deterministic_seed: Seed for deterministic execution
        telemetry_enabled: Whether to collect metrics
        metadata: Additional context metadata
    """

    task_id: Optional[str] = None
    thread_id: int = field(default_factory=threading.get_ident)
    deterministic_seed: Optional[int] = None
    telemetry_enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


# Thread-local storage for execution context
_context_storage = threading.local()


def get_context() -> ExecutionContext:
    """Get the current execution context.
    
    Returns:
        ExecutionContext for current thread
    """
    if not hasattr(_context_storage, "context"):
        _context_storage.context = ExecutionContext()
    return _context_storage.context


def set_context(context: ExecutionContext) -> None:
    """Set the execution context for current thread.
    
    Args:
        context: ExecutionContext to activate
    """
    _context_storage.context = context


def clear_context() -> None:
    """Clear the execution context for current thread."""
    if hasattr(_context_storage, "context"):
        delattr(_context_storage, "context")
