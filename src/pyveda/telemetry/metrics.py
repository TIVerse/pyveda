"""Metrics collection and aggregation."""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import psutil

logger = logging.getLogger(__name__)


class Counter:
    """Thread-safe counter metric."""

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize counter.

        Args:
            name: Metric name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._value = 0
        self._lock = threading.Lock()

    def inc(self, amount: int = 1) -> None:
        """Increment counter.

        Args:
            amount: Amount to increment
        """
        with self._lock:
            self._value += amount

    def get(self) -> int:
        """Get current value.

        Returns:
            Current counter value
        """
        with self._lock:
            return self._value

    def reset(self) -> None:
        """Reset counter to zero."""
        with self._lock:
            self._value = 0


class Gauge:
    """Thread-safe gauge metric."""

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize gauge.

        Args:
            name: Metric name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set gauge value.

        Args:
            value: New value
        """
        with self._lock:
            self._value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge.

        Args:
            amount: Amount to increment
        """
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge.

        Args:
            amount: Amount to decrement
        """
        with self._lock:
            self._value -= amount

    def get(self) -> float:
        """Get current value.

        Returns:
            Current gauge value
        """
        with self._lock:
            return self._value


class Histogram:
    """Thread-safe histogram for latency tracking."""

    def __init__(self, name: str, description: str = "") -> None:
        """Initialize histogram.

        Args:
            name: Metric name
            description: Human-readable description
        """
        self.name = name
        self.description = description
        self._samples: list[float] = []
        self._lock = threading.Lock()
        self._max_samples = 10000  # Keep last N samples

    def observe(self, value: float) -> None:
        """Record an observation.

        Args:
            value: Value to record
        """
        with self._lock:
            self._samples.append(value)
            # Keep only recent samples
            if len(self._samples) > self._max_samples:
                self._samples = self._samples[-self._max_samples :]

    def get_percentile(self, percentile: float) -> float:
        """Get percentile value.

        Args:
            percentile: Percentile (0-100)

        Returns:
            Percentile value
        """
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_samples = sorted(self._samples)
            idx = int(len(sorted_samples) * (percentile / 100.0))
            idx = min(idx, len(sorted_samples) - 1)
            return sorted_samples[idx]

    def get_mean(self) -> float:
        """Get mean value.

        Returns:
            Mean of samples
        """
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    def get_count(self) -> int:
        """Get sample count.

        Returns:
            Number of samples
        """
        with self._lock:
            return len(self._samples)


@dataclass
class MetricsSnapshot:
    """Snapshot of system metrics.

    Captures point-in-time metrics for export and monitoring.
    """

    timestamp: float
    tasks_executed: int
    tasks_failed: int
    tasks_pending: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_tasks_per_sec: float
    cpu_utilization_percent: float
    memory_used_mb: float
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: float | None = None

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus-formatted metrics
        """
        lines = [
            "# HELP veda_tasks_executed_total Total tasks executed",
            "# TYPE veda_tasks_executed_total counter",
            f"veda_tasks_executed_total {self.tasks_executed}",
            "",
            "# HELP veda_tasks_pending Current pending tasks",
            "# TYPE veda_tasks_pending gauge",
            f"veda_tasks_pending {self.tasks_pending}",
            "",
            "# HELP veda_latency_seconds Task latency",
            "# TYPE veda_latency_seconds summary",
            f'veda_latency_seconds{{quantile="0.5"}} {self.p50_latency_ms / 1000}',
            f'veda_latency_seconds{{quantile="0.99"}} {self.p99_latency_ms / 1000}',
            f"veda_latency_seconds_sum {self.avg_latency_ms * self.tasks_executed / 1000}",
            f"veda_latency_seconds_count {self.tasks_executed}",
            "",
            "# HELP veda_throughput_tasks_per_second Task throughput",
            "# TYPE veda_throughput_tasks_per_second gauge",
            f"veda_throughput_tasks_per_second {self.throughput_tasks_per_sec}",
            "",
            "# HELP veda_cpu_utilization_percent CPU utilization",
            "# TYPE veda_cpu_utilization_percent gauge",
            f"veda_cpu_utilization_percent {self.cpu_utilization_percent}",
            "",
            "# HELP veda_memory_used_mb Memory usage in MB",
            "# TYPE veda_memory_used_mb gauge",
            f"veda_memory_used_mb {self.memory_used_mb}",
        ]

        if self.gpu_utilization_percent is not None:
            lines.extend(
                [
                    "",
                    "# HELP veda_gpu_utilization_percent GPU utilization",
                    "# TYPE veda_gpu_utilization_percent gauge",
                    f"veda_gpu_utilization_percent {self.gpu_utilization_percent}",
                ]
            )

        return "\n".join(lines)

    def export_json(self) -> dict[str, Any]:
        """Export metrics as JSON.

        Returns:
            Dictionary of metrics
        """
        return {
            "timestamp": self.timestamp,
            "tasks": {
                "executed": self.tasks_executed,
                "pending": self.tasks_pending,
            },
            "latency_ms": {
                "avg": self.avg_latency_ms,
                "p50": self.p50_latency_ms,
                "p99": self.p99_latency_ms,
            },
            "throughput_tasks_per_sec": self.throughput_tasks_per_sec,
            "system": {
                "cpu_percent": self.cpu_utilization_percent,
                "memory_mb": self.memory_used_mb,
                "gpu_percent": self.gpu_utilization_percent,
            },
        }


class TelemetrySystem:
    """Telemetry system for collecting and exporting metrics.

    Runs a background thread to periodically collect metrics
    from the scheduler and system.
    """

    def __init__(self, scheduler: Any) -> None:
        """Initialize telemetry system.

        Args:
            scheduler: Scheduler to monitor
        """
        self.scheduler = scheduler
        self._running = False
        self._thread: threading.Thread | None = None

        # Metrics
        self.tasks_executed = Counter("tasks_executed", "Total tasks executed")
        self.tasks_pending = Gauge("tasks_pending", "Pending tasks")
        self.latency_histogram = Histogram("task_latency_ms", "Task latency in ms")

        # Snapshots
        self._snapshots: list[MetricsSnapshot] = []
        self._max_snapshots = 100

        # Track last values for delta computation
        self._last_task_count = 0
        self._last_snapshot_time = time.time()

    def start(self) -> None:
        """Start telemetry collection."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("Telemetry system started")

    def stop(self) -> None:
        """Stop telemetry collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Telemetry system stopped")

    def snapshot(self) -> MetricsSnapshot:
        """Create a metrics snapshot.

        Returns:
            Current metrics snapshot
        """
        # Get scheduler stats
        stats = self.scheduler.get_stats()
        now = time.time()
        time_delta = now - self._last_snapshot_time

        # Compute aggregate stats
        total_executed = sum(
            executor_stats["tasks_executed"]
            for executor_stats in stats["executors"].values()
        )
        total_failed = sum(
            executor_stats.get("tasks_failed", 0)
            for executor_stats in stats["executors"].values()
        )
        avg_latency = sum(
            executor_stats["avg_latency_ms"]
            for executor_stats in stats["executors"].values()
        ) / max(len(stats["executors"]), 1)

        # Compute throughput from delta
        tasks_delta = total_executed - self._last_task_count
        throughput = tasks_delta / time_delta if time_delta > 0 else 0.0

        # System metrics
        cpu_percent = stats.get("cpu_percent", 0.0)
        memory_mb = psutil.virtual_memory().used / (1024 * 1024)

        # GPU metrics if available
        gpu_util = None
        gpu_mem = None
        try:
            from pyveda.core.runtime import get_runtime

            runtime = get_runtime()
            if runtime.gpu and runtime.gpu.is_available():
                gpu_util = runtime.gpu.get_utilization()
                gpu_stats = runtime.gpu.get_memory_stats()
                gpu_mem = gpu_stats.get("used_mb", 0.0)
        except Exception:
            pass

        # Create snapshot
        snapshot = MetricsSnapshot(
            timestamp=now,
            tasks_executed=total_executed,
            tasks_failed=total_failed,
            tasks_pending=0,  # Would need task queue access
            avg_latency_ms=avg_latency,
            p50_latency_ms=self.latency_histogram.get_percentile(50),
            p95_latency_ms=self.latency_histogram.get_percentile(95),
            p99_latency_ms=self.latency_histogram.get_percentile(99),
            throughput_tasks_per_sec=throughput,
            cpu_utilization_percent=cpu_percent,
            memory_used_mb=memory_mb,
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mb=gpu_mem,
        )

        # Update tracking variables
        self._last_task_count = total_executed
        self._last_snapshot_time = now

        # Store snapshot
        self._snapshots.append(snapshot)
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

        return snapshot

    def _collection_loop(self) -> None:
        """Background loop for metrics collection."""
        while self._running:
            try:
                time.sleep(1.0)  # Collect every second
                self.snapshot()
            except Exception as e:
                logger.error(f"Telemetry collection error: {e}")

    def get_snapshots(self) -> list[MetricsSnapshot]:
        """Get all stored snapshots.

        Returns:
            List of snapshots
        """
        return self._snapshots.copy()
