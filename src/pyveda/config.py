"""Configuration management for PyVeda runtime."""

from dataclasses import dataclass
from enum import Enum


class SchedulingPolicy(Enum):
    """Scheduling policy for task execution."""

    ADAPTIVE = "adaptive"
    THREAD_ONLY = "thread_only"
    PROCESS_ONLY = "process_only"
    ASYNC_ONLY = "async_only"
    DETERMINISTIC = "deterministic"


class ExecutorType(Enum):
    """Available executor types."""

    THREAD = "thread"
    PROCESS = "process"
    ASYNC = "async"
    GPU = "gpu"


@dataclass
class Config:
    """Runtime configuration for PyVeda.

    Attributes:
        num_threads: Number of threads (None = cpu_count())
        num_processes: Number of processes (None = physical cores)
        scheduling_policy: Policy for task scheduling
        enable_gpu: Enable GPU executor
        enable_telemetry: Enable metrics collection
        deterministic_seed: Seed for deterministic mode (None = disabled)
        task_queue_size: Maximum pending tasks (0 = unlimited)
        adaptive_interval_ms: Milliseconds between adaptations
        min_workers: Minimum workers per executor
        max_workers: Maximum workers per executor
        chunk_size: Default chunk size for parallel iterators
        gpu_threshold_bytes: Minimum data size for GPU offload
        cpu_threshold_percent: CPU usage threshold for process fallback
    """

    num_threads: int | None = None
    num_processes: int | None = None
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.ADAPTIVE
    enable_gpu: bool = False
    enable_telemetry: bool = True
    deterministic_seed: int | None = None
    task_queue_size: int = 10000
    adaptive_interval_ms: int = 100
    min_workers: int = 1
    max_workers: int | None = None
    chunk_size: int | None = None
    gpu_threshold_bytes: int = 1024 * 1024  # 1MB
    cpu_threshold_percent: float = 70.0

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.num_threads is not None and self.num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        if self.num_processes is not None and self.num_processes < 1:
            raise ValueError("num_processes must be >= 1")
        if self.adaptive_interval_ms < 10:
            raise ValueError("adaptive_interval_ms must be >= 10")
        if self.min_workers < 1:
            raise ValueError("min_workers must be >= 1")
        if self.max_workers is not None and self.max_workers < self.min_workers:
            raise ValueError("max_workers must be >= min_workers")
        if not 0 <= self.cpu_threshold_percent <= 100:
            raise ValueError("cpu_threshold_percent must be in [0, 100]")

    @staticmethod
    def builder() -> "ConfigBuilder":
        """Create a builder for fluent configuration.

        Returns:
            New ConfigBuilder instance
        """
        return ConfigBuilder()

    @staticmethod
    def default() -> "Config":
        """Create default configuration.

        Returns:
            Config with default values
        """
        return Config()


class ConfigBuilder:
    """Fluent builder for Config objects."""

    def __init__(self) -> None:
        self._config = Config()

    def threads(self, count: int) -> "ConfigBuilder":
        """Set number of threads."""
        self._config.num_threads = count
        return self

    def processes(self, count: int) -> "ConfigBuilder":
        """Set number of processes."""
        self._config.num_processes = count
        return self

    def policy(self, policy: SchedulingPolicy) -> "ConfigBuilder":
        """Set scheduling policy."""
        self._config.scheduling_policy = policy
        return self

    def gpu(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable or disable GPU support."""
        self._config.enable_gpu = enabled
        return self

    def telemetry(self, enabled: bool = True) -> "ConfigBuilder":
        """Enable or disable telemetry."""
        self._config.enable_telemetry = enabled
        return self

    def deterministic(self, seed: int) -> "ConfigBuilder":
        """Enable deterministic mode with seed."""
        self._config.deterministic_seed = seed
        self._config.scheduling_policy = SchedulingPolicy.DETERMINISTIC
        return self

    def queue_size(self, size: int) -> "ConfigBuilder":
        """Set task queue size."""
        self._config.task_queue_size = size
        return self

    def adaptive_interval(self, ms: int) -> "ConfigBuilder":
        """Set adaptation interval in milliseconds."""
        self._config.adaptive_interval_ms = ms
        return self

    def worker_limits(self, min_workers: int, max_workers: int) -> "ConfigBuilder":
        """Set worker count limits."""
        self._config.min_workers = min_workers
        self._config.max_workers = max_workers
        return self

    def chunk_size(self, size: int) -> "ConfigBuilder":
        """Set default chunk size for parallel iterators."""
        self._config.chunk_size = size
        return self

    def build(self) -> Config:
        """Build the configuration.

        Returns:
            Configured Config instance
        """
        return self._config
