"""Execution tracing and replay for debugging."""

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Any, Generator, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class TaskScheduledEvent:
    """Event recording task scheduling."""

    timestamp: int
    task_id: str
    worker_id: int
    executor_type: str = "unknown"


class ExecutionTrace:
    """Records execution events for replay.
    
    Captures scheduling decisions and task execution
    order for deterministic replay.
    """

    def __init__(self) -> None:
        """Initialize execution trace."""
        self.events: List[TaskScheduledEvent] = []

    def record(self, event: TaskScheduledEvent) -> None:
        """Record an event.
        
        Args:
            event: Event to record
        """
        self.events.append(event)

    def get_events(self) -> List[TaskScheduledEvent]:
        """Get all recorded events.
        
        Returns:
            List of events
        """
        return self.events.copy()

    def clear(self) -> None:
        """Clear all events."""
        self.events.clear()

    def save_to_file(self, filepath: str) -> None:
        """Save trace to JSON file.
        
        Args:
            filepath: Path to save trace
        """
        with open(filepath, 'w') as f:
            data = [asdict(event) for event in self.events]
            json.dump(data, f, indent=2)
        logger.info(f"Trace saved to {filepath}")

    def load_from_file(self, filepath: str) -> None:
        """Load trace from JSON file.
        
        Args:
            filepath: Path to load trace from
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.events = [TaskScheduledEvent(**event) for event in data]
        logger.info(f"Trace loaded from {filepath} ({len(self.events)} events)")

    def to_dict(self) -> dict[str, Any]:
        """Export trace as dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            'events': [asdict(event) for event in self.events],
            'event_count': len(self.events)
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ExecutionTrace':
        """Create trace from dictionary.
        
        Args:
            data: Dictionary with trace data
            
        Returns:
            ExecutionTrace instance
        """
        trace = cls()
        trace.events = [TaskScheduledEvent(**event) for event in data.get('events', [])]
        return trace


@contextmanager
def deterministic(seed: int, record: bool = False) -> Generator[Optional[ExecutionTrace], None, None]:
    """Context manager for deterministic execution.
    
    Ensures reproducible execution by seeding the scheduler
    and using deterministic scheduling.
    
    Args:
        seed: Random seed
        record: Whether to record execution trace
        
    Yields:
        ExecutionTrace if recording, else None
        
    Example:
        with deterministic(seed=42) as trace:
            result = compute_flaky_operation()
            if trace:
                trace.save_to_file('execution.json')
    """
    from pyveda.config import Config, SchedulingPolicy
    from pyveda.core.runtime import get_runtime
    from pyveda.deterministic.scheduler import DeterministicScheduler

    runtime = get_runtime()
    old_scheduler = runtime.scheduler

    # Create trace if recording
    trace = ExecutionTrace() if record else None

    # Create deterministic scheduler
    det_config = Config(
        scheduling_policy=SchedulingPolicy.DETERMINISTIC,
        deterministic_seed=seed,
    )
    det_scheduler = DeterministicScheduler(det_config, seed, trace=trace)

    # Copy executor registrations
    for executor_type, executor in old_scheduler._executors.items():
        det_scheduler.register_executor(executor_type, executor)

    det_scheduler.start()

    # Swap scheduler
    runtime.scheduler = det_scheduler

    try:
        yield trace
    finally:
        # Restore original scheduler
        det_scheduler.shutdown()
        runtime.scheduler = old_scheduler

        logger.info(f"Deterministic execution completed ({det_scheduler.logical_clock} tasks)")
