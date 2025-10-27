"""Deterministic scheduler for reproducible execution."""

import logging
import random
from typing import Any, Dict, Optional, TYPE_CHECKING

from pyveda.config import Config
from pyveda.core.executor import BaseExecutor
from pyveda.core.task import Task

if TYPE_CHECKING:
    from pyveda.deterministic.replay import ExecutionTrace, TaskScheduledEvent

logger = logging.getLogger(__name__)


class DeterministicScheduler:
    """Deterministic scheduler for reproducible execution.
    
    Uses seeded random number generator to make scheduling
    decisions deterministic across runs.
    """

    def __init__(self, config: Config, seed: int, trace: Optional['ExecutionTrace'] = None) -> None:
        """Initialize deterministic scheduler.
        
        Args:
            config: Runtime configuration
            seed: Random seed for determinism
            trace: Optional execution trace for recording
        """
        self.config = config
        self.seed = seed
        self.rng = random.Random(seed)
        self.logical_clock = 0
        self._executors: Dict[str, BaseExecutor] = {}
        self._running = False
        self.trace = trace

        logger.info(f"Deterministic scheduler initialized with seed {seed}")

    def register_executor(self, executor_type: Any, executor: BaseExecutor) -> None:
        """Register an executor.
        
        Args:
            executor_type: Type of executor
            executor: Executor instance
        """
        self._executors[executor_type.value] = executor

    def start(self) -> None:
        """Start the scheduler."""
        self._running = True

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        self._running = False

    def submit(self, task: Task) -> Any:
        """Submit a task with deterministic executor selection.
        
        Args:
            task: Task to execute
            
        Returns:
            Future for the result
        """
        # Deterministically select executor
        executor = self._select_executor_deterministic(task)
        
        # Increment logical clock
        self.logical_clock += 1
        
        # Record event if tracing
        if self.trace is not None:
            from pyveda.deterministic.replay import TaskScheduledEvent
            event = TaskScheduledEvent(
                timestamp=self.logical_clock,
                task_id=task.id,
                worker_id=0,  # Simplified for now
                executor_type=self._get_executor_type_name(executor)
            )
            self.trace.record(event)
        
        # Submit to executor
        return executor.submit(task)

    def _get_executor_type_name(self, executor: BaseExecutor) -> str:
        """Get executor type name.
        
        Args:
            executor: Executor instance
            
        Returns:
            Executor type name
        """
        for exec_type, exec_inst in self._executors.items():
            if exec_inst is executor:
                return exec_type
        return "unknown"

    def _select_executor_deterministic(self, task: Task) -> BaseExecutor:
        """Select executor deterministically.
        
        Args:
            task: Task to schedule
            
        Returns:
            Selected executor
        """
        # For deterministic mode, only use thread executor
        # Process pool requires picklable functions (no lambdas)
        # This ensures determinism works with any functions
        thread_executor = self._executors.get("thread")
        if thread_executor:
            return thread_executor
        
        # Fallback to any available executor
        executor_list = list(self._executors.values())
        if not executor_list:
            raise RuntimeError("No executors registered")
        
        idx = self.rng.randint(0, len(executor_list) - 1)
        return executor_list[idx]

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "executors": {},
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "logical_clock": self.logical_clock,
        }
