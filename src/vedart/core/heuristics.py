"""Scheduler heuristics and decision policies.

This module contains the logic for selecting the optimal executor
for a given task based on workload characteristics and system state.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import psutil

from vedart.config import ExecutorType
from vedart.core.task import Task


class WorkloadType(Enum):
    """Classification of workload types."""

    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    ASYNC = "async"
    GPU_COMPATIBLE = "gpu_compatible"
    UNKNOWN = "unknown"


@dataclass
class TaskCharacteristics:
    """Characteristics of a task for scheduling decisions."""

    workload_type: WorkloadType
    estimated_duration_ms: float = 0.0
    data_size_bytes: int = 0
    is_async: bool = False
    requires_gpu: bool = False
    is_pure: bool = True  # Side-effect free
    priority: int = 0

    @property
    def is_small_task(self) -> bool:
        """Check if this is a small, lightweight task."""
        return self.estimated_duration_ms < 10.0 and self.data_size_bytes < 1024


@dataclass
class SystemState:
    """Current system resource state."""

    cpu_percent: float
    memory_percent: float
    cpu_count: int
    available_executors: set[ExecutorType]
    queue_depths: dict[ExecutorType, int]

    @property
    def is_cpu_saturated(self) -> bool:
        """Check if CPU is heavily utilized."""
        return self.cpu_percent > 80.0

    @property
    def is_memory_constrained(self) -> bool:
        """Check if memory is running low."""
        return self.memory_percent > 85.0


class SchedulerPolicy(Protocol):
    """Protocol for scheduler decision policies."""

    def select_executor(
        self,
        task: Task,
        characteristics: TaskCharacteristics,
        system_state: SystemState,
    ) -> ExecutorType | None:
        """Select the best executor for the task.

        Args:
            task: Task to schedule
            characteristics: Task characteristics
            system_state: Current system state

        Returns:
            Selected executor type or None if no suitable executor
        """
        ...


class AdaptivePolicy:
    """Adaptive scheduling policy using heuristics.

    This policy makes intelligent decisions based on:
    - Task characteristics (CPU/IO bound, size, async)
    - System state (CPU load, memory pressure)
    - Executor availability and queue depths
    - Historical performance metrics
    """

    def __init__(
        self,
        cpu_threshold: float = 70.0,
        gpu_threshold_bytes: int = 1024 * 1024,  # 1MB
        small_task_threshold_ms: float = 10.0,
    ):
        """Initialize adaptive policy.

        Args:
            cpu_threshold: CPU % threshold for process vs thread decision
            gpu_threshold_bytes: Minimum data size for GPU offload
            small_task_threshold_ms: Threshold for "small" tasks (use threads)
        """
        self.cpu_threshold = cpu_threshold
        self.gpu_threshold_bytes = gpu_threshold_bytes
        self.small_task_threshold_ms = small_task_threshold_ms

    def select_executor(
        self,
        task: Task,
        characteristics: TaskCharacteristics,
        system_state: SystemState,
    ) -> ExecutorType | None:
        """Select executor using adaptive heuristics."""

        # Priority 1: GPU for explicitly marked GPU tasks
        if self._should_use_gpu(task, characteristics, system_state):
            return ExecutorType.GPU

        # Priority 2: Async for async tasks
        if characteristics.is_async:
            if ExecutorType.ASYNC in system_state.available_executors:
                return ExecutorType.ASYNC

        # Priority 3: Choose between threads and processes
        return self._select_cpu_executor(task, characteristics, system_state)

    def _should_use_gpu(
        self,
        task: Task,
        characteristics: TaskCharacteristics,
        system_state: SystemState,
    ) -> bool:
        """Decide if GPU should be used."""
        # GPU not available
        if ExecutorType.GPU not in system_state.available_executors:
            return False

        # Explicitly marked for GPU
        if characteristics.requires_gpu:
            return True

        # Heuristic: Large data + numeric computation
        if (
            characteristics.workload_type == WorkloadType.GPU_COMPATIBLE
            and characteristics.data_size_bytes >= self.gpu_threshold_bytes
        ):
            return True

        return False

    def _select_cpu_executor(
        self,
        task: Task,
        characteristics: TaskCharacteristics,
        system_state: SystemState,
    ) -> ExecutorType | None:
        """Choose between thread and process pool."""

        # Small tasks: always use threads (lower overhead)
        if characteristics.estimated_duration_ms < self.small_task_threshold_ms:
            if ExecutorType.THREAD in system_state.available_executors:
                return ExecutorType.THREAD

        # I/O-bound: prefer threads
        if characteristics.workload_type == WorkloadType.IO_BOUND:
            if ExecutorType.THREAD in system_state.available_executors:
                return ExecutorType.THREAD

        # CPU-bound decisions
        if characteristics.workload_type == WorkloadType.CPU_BOUND:
            # High CPU load: avoid adding more threads (GIL contention)
            if system_state.cpu_percent > self.cpu_threshold:
                if ExecutorType.PROCESS in system_state.available_executors:
                    return ExecutorType.PROCESS
            else:
                # Low CPU: threads are fine (lower overhead)
                if ExecutorType.THREAD in system_state.available_executors:
                    return ExecutorType.THREAD

        # Consider queue depths for load balancing
        return self._balance_by_queue_depth(system_state)

    def _balance_by_queue_depth(self, system_state: SystemState) -> ExecutorType | None:
        """Select executor with lowest queue depth."""
        available = [
            ExecutorType.THREAD,
            ExecutorType.PROCESS,
        ]

        available = [ex for ex in available if ex in system_state.available_executors]

        if not available:
            return None

        # Select executor with minimal queue depth
        return min(available, key=lambda ex: system_state.queue_depths.get(ex, 0))


class GreedyPolicy:
    """Greedy policy: always use the fastest available executor.

    Priority order: GPU > Process > Thread > Async
    """

    def select_executor(
        self,
        task: Task,
        characteristics: TaskCharacteristics,
        system_state: SystemState,
    ) -> ExecutorType | None:
        """Select executor greedily."""

        # Async tasks must use async executor
        if characteristics.is_async:
            if ExecutorType.ASYNC in system_state.available_executors:
                return ExecutorType.ASYNC
            return None

        # Priority order for non-async tasks
        priority_order = [
            ExecutorType.GPU,
            ExecutorType.PROCESS,
            ExecutorType.THREAD,
        ]

        for executor_type in priority_order:
            if executor_type in system_state.available_executors:
                return executor_type

        return None


class RoundRobinPolicy:
    """Round-robin policy for load balancing."""

    def __init__(self):
        self._counter = 0

    def select_executor(
        self,
        task: Task,
        characteristics: TaskCharacteristics,
        system_state: SystemState,
    ) -> ExecutorType | None:
        """Select executor in round-robin fashion."""

        # Async tasks must use async executor
        if characteristics.is_async:
            if ExecutorType.ASYNC in system_state.available_executors:
                return ExecutorType.ASYNC
            return None

        # Round-robin over available executors
        available = sorted(system_state.available_executors, key=lambda x: x.value)

        if not available:
            return None

        selected = available[self._counter % len(available)]
        self._counter += 1

        return selected


class TaskAnalyzer:
    """Analyzes tasks to extract characteristics for scheduling."""

    @staticmethod
    def analyze(task: Task) -> TaskCharacteristics:
        """Analyze a task and extract characteristics.

        Args:
            task: Task to analyze

        Returns:
            Task characteristics for scheduling
        """
        workload_type = TaskAnalyzer._classify_workload(task)
        estimated_duration = TaskAnalyzer._estimate_duration(task)
        data_size = TaskAnalyzer._estimate_data_size(task)

        return TaskCharacteristics(
            workload_type=workload_type,
            estimated_duration_ms=estimated_duration,
            data_size_bytes=data_size,
            is_async=task.is_async,
            requires_gpu=TaskAnalyzer._requires_gpu(task),
            is_pure=TaskAnalyzer._is_pure(task),
        )

    @staticmethod
    def _classify_workload(task: Task) -> WorkloadType:
        """Classify task workload type."""
        # Check for async
        if task.is_async:
            return WorkloadType.ASYNC

        # Check function metadata
        func = task.func

        # GPU-decorated functions
        if hasattr(func, "__vedart_gpu__"):
            return WorkloadType.GPU_COMPATIBLE

        # Heuristic: check function name
        func_name = getattr(func, "__name__", "").lower()

        if any(
            keyword in func_name
            for keyword in ["io", "read", "write", "fetch", "download"]
        ):
            return WorkloadType.IO_BOUND

        if any(
            keyword in func_name
            for keyword in ["compute", "calculate", "process", "transform"]
        ):
            return WorkloadType.CPU_BOUND

        return WorkloadType.UNKNOWN

    @staticmethod
    def _estimate_duration(task: Task) -> float:
        """Estimate task duration in milliseconds."""
        # Check for hints in task metadata
        if hasattr(task.func, "__vedart_duration_hint__"):
            return task.func.__vedart_duration_hint__

        # Default: assume medium task
        return 50.0

    @staticmethod
    def _estimate_data_size(task: Task) -> int:
        """Estimate total data size in bytes."""
        import sys

        total_size = 0

        # Estimate args size
        for arg in task.args:
            total_size += sys.getsizeof(arg)

        # Estimate kwargs size
        for value in task.kwargs.values():
            total_size += sys.getsizeof(value)

        return total_size

    @staticmethod
    def _requires_gpu(task: Task) -> bool:
        """Check if task requires GPU."""
        return hasattr(task.func, "__vedart_gpu__")

    @staticmethod
    def _is_pure(task: Task) -> bool:
        """Check if task is a pure function (no side effects)."""
        # Conservative: assume impure unless marked
        if hasattr(task.func, "__vedart_pure__"):
            return task.func.__vedart_pure__

        return False


class SystemMonitor:
    """Monitors system state for scheduling decisions."""

    def __init__(self):
        self._last_cpu_check = 0.0
        self._cached_cpu_percent = 0.0

    def get_state(
        self,
        available_executors: set[ExecutorType],
        queue_depths: dict[ExecutorType, int],
    ) -> SystemState:
        """Get current system state.

        Args:
            available_executors: Set of available executor types
            queue_depths: Current queue depth for each executor

        Returns:
            Current system state
        """
        import time

        # Update CPU metrics (cached for performance)
        now = time.time()
        if now - self._last_cpu_check > 0.5:  # Update every 500ms
            self._cached_cpu_percent = psutil.cpu_percent(interval=0)
            self._last_cpu_check = now

        return SystemState(
            cpu_percent=self._cached_cpu_percent,
            memory_percent=psutil.virtual_memory().percent,
            cpu_count=psutil.cpu_count() or 1,
            available_executors=available_executors,
            queue_depths=queue_depths,
        )


def create_policy(name: str) -> SchedulerPolicy:
    """Factory function to create scheduling policies.

    Args:
        name: Policy name ("adaptive", "greedy", "round_robin")

    Returns:
        Scheduler policy instance

    Raises:
        ValueError: If policy name is unknown
    """
    policies = {
        "adaptive": AdaptivePolicy(),
        "greedy": GreedyPolicy(),
        "round_robin": RoundRobinPolicy(),
    }

    policy = policies.get(name.lower())
    if policy is None:
        raise ValueError(f"Unknown policy: {name}. Available: {list(policies.keys())}")

    return policy
