"""Global runtime management for VedaRT."""

import atexit
import logging
from typing import Any, Optional

from vedart.config import Config, SchedulingPolicy
from vedart.core.scheduler import AdaptiveScheduler
from vedart.exceptions import VedaError

logger = logging.getLogger(__name__)

# Global runtime singleton
_runtime: Optional["Runtime"] = None
_runtime_lock = __import__("threading").Lock()


class Runtime:
    """Global runtime managing scheduler, executors, and telemetry.

    This is the central coordination point for VedaRT. It owns
    the scheduler, all executors, and optional GPU/telemetry systems.

    Attributes:
        config: Runtime configuration
        scheduler: Task scheduler
        gpu: GPU runtime (if enabled)
        telemetry: Telemetry system (if enabled)
    """

    def __init__(self, config: Config) -> None:
        """Initialize runtime.

        Args:
            config: Runtime configuration
        """
        self.config = config

        # Initialize scheduler based on policy
        if config.scheduling_policy == SchedulingPolicy.DETERMINISTIC:
            from vedart.deterministic.scheduler import DeterministicScheduler

            self.scheduler = DeterministicScheduler(
                config, seed=config.deterministic_seed or 42
            )
        else:
            self.scheduler = AdaptiveScheduler(config)

        self.gpu: Any | None = None
        self.telemetry: Any | None = None

        # Initialize executors
        self._init_executors()

        # Initialize optional systems
        if config.enable_gpu:
            self._init_gpu()

        if config.enable_telemetry:
            self._init_telemetry()

        # Start scheduler
        self.scheduler.start()

        logger.info("Runtime initialized")

    def _init_executors(self) -> None:
        """Initialize and register executors."""
        from vedart.config import ExecutorType
        from vedart.executors.process_pool import ProcessPoolExecutor
        from vedart.executors.thread_pool import ThreadPoolExecutor

        # Thread pool executor
        thread_pool = ThreadPoolExecutor(
            max_workers=self.config.num_threads,
            name="thread-pool",
        )
        self.scheduler.register_executor(ExecutorType.THREAD, thread_pool)

        # Process pool executor
        process_pool = ProcessPoolExecutor(
            max_workers=self.config.num_processes,
            name="process-pool",
        )
        self.scheduler.register_executor(ExecutorType.PROCESS, process_pool)

        # Async executor (conditionally)
        try:
            from vedart.executors.async_executor import AsyncIOExecutor

            async_executor = AsyncIOExecutor(name="async-executor")
            self.scheduler.register_executor(ExecutorType.ASYNC, async_executor)
        except Exception as e:
            logger.warning(f"Failed to initialize async executor: {e}")

    def _init_gpu(self) -> None:
        """Initialize GPU runtime."""
        try:
            from vedart.config import ExecutorType
            from vedart.gpu.backend import GPURuntime
            from vedart.gpu.executor import GPUExecutor

            self.gpu = GPURuntime()
            if self.gpu.is_available():
                logger.info(f"GPU runtime initialized: {self.gpu.backend}")

                # Register GPU executor with scheduler
                gpu_executor = GPUExecutor(self.gpu, name="gpu-executor")
                self.scheduler.register_executor(ExecutorType.GPU, gpu_executor)
            else:
                logger.warning("GPU enabled but no GPU detected")
                self.gpu = None
        except ImportError:
            logger.warning("GPU support not available (install cupy/numba)")
            self.gpu = None
        except Exception as e:
            logger.error(f"Failed to initialize GPU runtime: {e}")
            self.gpu = None

    def _init_telemetry(self) -> None:
        """Initialize telemetry system."""
        try:
            from vedart.telemetry.metrics import TelemetrySystem

            self.telemetry = TelemetrySystem(self.scheduler)
            self.telemetry.start()
            logger.info("Telemetry system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize telemetry: {e}")
            self.telemetry = None

    def shutdown(self) -> None:
        """Shutdown the runtime and all subsystems."""
        logger.info("Shutting down runtime")

        # Stop telemetry
        if self.telemetry:
            try:
                self.telemetry.stop()
            except Exception as e:
                logger.error(f"Telemetry shutdown error: {e}")

        # Stop scheduler
        try:
            self.scheduler.shutdown()
        except Exception as e:
            logger.error(f"Scheduler shutdown error: {e}")

        # Shutdown executors
        from vedart.config import ExecutorType

        for executor_type in ExecutorType:
            executor = self.scheduler._executors.get(executor_type)
            if executor:
                try:
                    executor.shutdown(wait=True)
                except Exception as e:
                    logger.error(f"Executor {executor_type.value} shutdown error: {e}")

        logger.info("Runtime shutdown complete")


def init(config: Config | None = None) -> Runtime:
    """Initialize the global runtime.

    Args:
        config: Runtime configuration (uses defaults if None)

    Returns:
        Initialized Runtime instance

    Raises:
        VedaError: If runtime is already initialized
    """
    global _runtime

    with _runtime_lock:
        if _runtime is not None:
            raise VedaError("Runtime already initialized")

        if config is None:
            config = Config.default()

        _runtime = Runtime(config)

        # Register shutdown hook
        atexit.register(shutdown)

        return _runtime


def shutdown() -> None:
    """Shutdown the global runtime."""
    global _runtime

    with _runtime_lock:
        if _runtime is not None:
            _runtime.shutdown()
            _runtime = None


def get_runtime() -> Runtime:
    """Get the global runtime instance.

    Returns:
        Runtime instance

    Raises:
        VedaError: If runtime not initialized
    """
    global _runtime

    if _runtime is None:
        # Auto-initialize with defaults
        logger.info("Auto-initializing runtime with defaults")
        init()

    assert _runtime is not None
    return _runtime
