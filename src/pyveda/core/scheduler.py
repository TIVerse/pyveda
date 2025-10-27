"""Adaptive scheduler for task distribution."""

import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass, field
from queue import Empty, PriorityQueue
from typing import Any, Dict, Optional

import psutil

from pyveda.config import Config, ExecutorType, SchedulingPolicy
from pyveda.core.executor import BaseExecutor
from pyveda.core.task import Task
from pyveda.exceptions import SchedulerError

logger = logging.getLogger(__name__)


@dataclass
class ExecutorStats:
    """Statistics for an executor.
    
    Tracks metrics for adaptive scheduling decisions.
    """

    tasks_executed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0.0
    latency_samples: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        """Average task latency."""
        if not self.latency_samples:
            return 0.0
        return sum(self.latency_samples) / len(self.latency_samples)

    def record_task(self, latency_ms: float, failed: bool = False) -> None:
        """Record a completed task."""
        self.tasks_executed += 1
        if failed:
            self.tasks_failed += 1
        self.total_latency_ms += latency_ms
        self.latency_samples.append(latency_ms)
        # Keep only recent samples
        if len(self.latency_samples) > 1000:
            self.latency_samples = self.latency_samples[-1000:]


class AdaptiveScheduler:
    """Adaptive scheduler that selects executors and scales workers.
    
    Uses heuristics and system metrics to choose the best executor
    for each task and dynamically adjust worker counts.
    """

    def __init__(self, config: Config) -> None:
        """Initialize scheduler.
        
        Args:
            config: Runtime configuration
        """
        self.config = config
        self._executors: Dict[ExecutorType, BaseExecutor] = {}
        self._stats: Dict[ExecutorType, ExecutorStats] = defaultdict(ExecutorStats)
        self._task_queue: PriorityQueue[Task] = PriorityQueue(
            maxsize=config.task_queue_size
        )
        self._running = False
        self._adaptation_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Cached system metrics (updated periodically)
        self._cpu_percent = 0.0
        self._memory_percent = 0.0
        self._last_metrics_update = 0.0
        
        # Track executor state for delta computations
        self._last_executor_counts: Dict[ExecutorType, int] = {}
        self._last_snapshot_time = time.time()

    def register_executor(self, executor_type: ExecutorType, executor: BaseExecutor) -> None:
        """Register an executor with the scheduler.
        
        Args:
            executor_type: Type of executor
            executor: Executor instance
        """
        with self._lock:
            self._executors[executor_type] = executor
            logger.info(f"Registered executor: {executor_type.value}")

    def start(self) -> None:
        """Start the scheduler and adaptation loop."""
        if self._running:
            return

        self._running = True

        # Start adaptation thread if adaptive policy
        if self.config.scheduling_policy == SchedulingPolicy.ADAPTIVE:
            self._adaptation_thread = threading.Thread(
                target=self._adaptation_loop, daemon=True
            )
            self._adaptation_thread.start()
            logger.info("Started adaptive scheduling loop")

    def shutdown(self) -> None:
        """Shutdown the scheduler."""
        self._running = False
        if self._adaptation_thread:
            self._adaptation_thread.join(timeout=1.0)
        logger.info("Scheduler shutdown complete")

    def submit(self, task: Task) -> Future[Any]:
        """Submit a task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Future for the result
            
        Raises:
            SchedulerError: If no suitable executor available
        """
        executor = self._select_executor(task)
        if not executor:
            raise SchedulerError("No suitable executor available")

        # Track start time for metrics
        start_time = time.perf_counter()
        executor_type = self._get_executor_type(executor)

        # Submit to executor
        future = executor.submit(task)

        # Attach callback to record metrics
        def record_completion(fut: Future[Any]) -> None:
            try:
                latency_ms = (time.perf_counter() - start_time) * 1000
                # Check if future succeeded or failed
                if fut.exception() is not None:
                    if executor_type:
                        self._stats[executor_type].record_task(latency_ms, failed=True)
                else:
                    if executor_type:
                        self._stats[executor_type].record_task(latency_ms, failed=False)
            except Exception as e:
                logger.error(f"Error recording task metrics: {e}")

        # Wire the callback
        future.add_done_callback(record_completion)
        return future

    def _select_executor(self, task: Task) -> Optional[BaseExecutor]:
        """Select the best executor for a task.
        
        Args:
            task: Task to schedule
            
        Returns:
            Selected executor or None
        """
        # Update metrics if stale
        self._update_metrics()

        # Handle different policies
        if self.config.scheduling_policy == SchedulingPolicy.THREAD_ONLY:
            return self._executors.get(ExecutorType.THREAD)
        elif self.config.scheduling_policy == SchedulingPolicy.PROCESS_ONLY:
            return self._executors.get(ExecutorType.PROCESS)
        elif self.config.scheduling_policy == SchedulingPolicy.ASYNC_ONLY:
            return self._executors.get(ExecutorType.ASYNC)

        # Adaptive selection
        return self._adaptive_select(task)

    def _adaptive_select(self, task: Task) -> Optional[BaseExecutor]:
        """Adaptively select executor based on task and system state.
        
        Args:
            task: Task to schedule
            
        Returns:
            Selected executor
        """
        # Priority 1: GPU if enabled and beneficial
        gpu_executor = self._executors.get(ExecutorType.GPU)
        if gpu_executor and gpu_executor.is_available():
            # Check if GPU offload would be beneficial using GPU runtime
            # For now, rely on @gpu decorator marking, but could enhance with heuristics
            # based on task args size estimation
            try:
                from pyveda.core.runtime import get_runtime
                runtime = get_runtime()
                if runtime.gpu and runtime.gpu.should_offload(task.func, task.args):
                    return gpu_executor
            except Exception as e:
                logger.debug(f"GPU eligibility check failed: {e}")

        # Priority 2: Async if task is coroutine
        if task.is_async:
            async_executor = self._executors.get(ExecutorType.ASYNC)
            if async_executor and async_executor.is_available():
                return async_executor

        # Priority 3: Choose between thread and process based on CPU load
        thread_executor = self._executors.get(ExecutorType.THREAD)
        process_executor = self._executors.get(ExecutorType.PROCESS)

        # Use threads for low CPU load or I/O-bound tasks
        # Use processes for high CPU load
        if self._cpu_percent < self.config.cpu_threshold_percent:
            if thread_executor and thread_executor.is_available():
                return thread_executor

        # Fallback to processes for CPU-bound work
        if process_executor and process_executor.is_available():
            return process_executor

        # Final fallback to any available executor
        for executor in self._executors.values():
            if executor.is_available():
                return executor

        return None

    def _adaptation_loop(self) -> None:
        """Background loop for worker scaling."""
        while self._running:
            try:
                time.sleep(self.config.adaptive_interval_ms / 1000.0)
                self._adapt_workers()
            except Exception as e:
                logger.error(f"Adaptation loop error: {e}")

    def _adapt_workers(self) -> None:
        """Adapt worker counts using Little's Law."""
        now = time.time()
        time_delta = now - self._last_snapshot_time
        
        if time_delta < 0.001:  # Avoid division by zero
            return
        
        for executor_type, executor in self._executors.items():
            stats = self._stats[executor_type]

            if stats.tasks_executed < 10:
                # Not enough data yet
                continue

            # Compute arrival rate (tasks per second) using delta
            last_count = self._last_executor_counts.get(executor_type, 0)
            tasks_delta = stats.tasks_executed - last_count
            arrival_rate = tasks_delta / time_delta

            # Service time (average latency in seconds)
            service_time_sec = stats.avg_latency_ms / 1000.0

            if service_time_sec <= 0 or arrival_rate <= 0:
                continue

            # Little's Law: L = λ * W
            # optimal_workers = arrival_rate * service_time
            optimal = int(arrival_rate * service_time_sec)

            # Clamp to configured limits
            optimal = max(self.config.min_workers, optimal)
            if self.config.max_workers is not None:
                optimal = min(optimal, self.config.max_workers)
            else:
                optimal = min(optimal, os.cpu_count() or 8)

            # Scale executor
            try:
                executor.scale(optimal)
                logger.debug(f"Scaled {executor_type.value} to {optimal} workers")
            except Exception as e:
                logger.warning(f"Failed to scale {executor_type.value}: {e}")
            
            # Update last count
            self._last_executor_counts[executor_type] = stats.tasks_executed
        
        # Update last snapshot time
        self._last_snapshot_time = now

    def _update_metrics(self) -> None:
        """Update cached system metrics."""
        now = time.time()
        if now - self._last_metrics_update < 0.1:  # Update at most every 100ms
            return

        try:
            # Non-blocking CPU percent (uses cached value)
            self._cpu_percent = psutil.cpu_percent(interval=0)
            self._memory_percent = psutil.virtual_memory().percent
            self._last_metrics_update = now
        except Exception as e:
            logger.warning(f"Failed to update metrics: {e}")

    def _get_executor_type(self, executor: BaseExecutor) -> Optional[ExecutorType]:
        """Get the type of an executor."""
        for executor_type, ex in self._executors.items():
            if ex is executor:
                return executor_type
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "executors": {
                executor_type.value: {
                    "tasks_executed": stats.tasks_executed,
                    "tasks_failed": stats.tasks_failed,
                    "avg_latency_ms": stats.avg_latency_ms,
                }
                for executor_type, stats in self._stats.items()
            },
            "cpu_percent": self._cpu_percent,
            "memory_percent": self._memory_percent,
        }
