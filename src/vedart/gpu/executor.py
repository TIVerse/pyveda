"""GPU executor for automatic task offload."""

import logging
from concurrent.futures import Future as StdFuture
from typing import Any

from vedart.core.executor import BaseExecutor
from vedart.core.task import Task
from vedart.exceptions import GPUError

logger = logging.getLogger(__name__)


class GPUExecutor(BaseExecutor):
    """Executor for GPU-accelerated tasks.

    Offloads suitable tasks to GPU using cost model.
    Falls back to CPU if GPU is unavailable or not beneficial.
    """

    def __init__(self, gpu_runtime: Any, name: str = "gpu-executor") -> None:
        """Initialize GPU executor.

        Args:
            gpu_runtime: GPURuntime instance
            name: Executor name for logging
        """
        super().__init__(name)
        self._gpu_runtime = gpu_runtime
        logger.info("GPUExecutor initialized")

    def submit(self, task: Task) -> StdFuture[Any]:
        """Submit a task for GPU execution.

        Args:
            task: Task to execute

        Returns:
            Future for the result
        """
        if self._shutdown:
            raise RuntimeError("Executor is shutdown")

        # Check if GPU offload is beneficial
        if not self._gpu_runtime.is_available():
            raise GPUError("GPU not available")

        # Create future for result
        future = StdFuture()

        try:
            # Execute on GPU
            result = self._gpu_runtime.execute(task.func, *task.args, **task.kwargs)
            future.set_result(result)
        except Exception as e:
            logger.warning(
                f"GPU execution failed: {e}, CPU fallback not available in executor"
            )
            future.set_exception(e)

        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.

        Args:
            wait: Ignored for GPU executor
        """
        if self._shutdown:
            return

        self._shutdown = True
        logger.info(f"{self.name} shutdown")

    def is_available(self) -> bool:
        """Check if GPU executor is available.

        Returns:
            True if GPU runtime is available
        """
        return not self._shutdown and self._gpu_runtime.is_available()

    def scale(self, num_workers: int) -> None:
        """Scale the executor.

        GPU doesn't have workers in traditional sense, this is a no-op.

        Args:
            num_workers: Ignored
        """
        # GPU scaling could mean device selection in multi-GPU setup
        pass
