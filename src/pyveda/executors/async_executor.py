"""AsyncIO executor for coroutine execution."""

import asyncio
import logging
import threading
from concurrent.futures import Future as StdFuture
from typing import Any, Optional

from pyveda.core.executor import BaseExecutor
from pyveda.core.task import Task

logger = logging.getLogger(__name__)


class AsyncIOExecutor(BaseExecutor):
    """Executor for async/await coroutines.
    
    Runs an event loop in a dedicated thread and bridges
    to concurrent.futures.Future for synchronous API.
    """

    def __init__(self, name: str = "async-executor") -> None:
        """Initialize async executor.
        
        Args:
            name: Executor name for logging
        """
        super().__init__(name)

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._started = threading.Event()

        # Start event loop in background thread
        self._start_loop()

        logger.info("AsyncIOExecutor initialized")

    def _start_loop(self) -> None:
        """Start the asyncio event loop in a background thread."""

        def run_loop() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._started.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run_loop, daemon=True, name="veda-asyncio")
        self._thread.start()
        self._started.wait()  # Wait for loop to be ready

    def submit(self, task: Task) -> StdFuture[Any]:
        """Submit a task for execution.
        
        Args:
            task: Task to execute (must be async)
            
        Returns:
            Future for the result
        """
        if self._shutdown or not self._loop:
            raise RuntimeError("Executor is shutdown")

        # Create a future for the result
        future = StdFuture()

        async def execute_async() -> None:
            try:
                # Execute the function (handle both sync and async)
                result = task.func(*task.args, **task.kwargs)
                
                # Check if result is awaitable (coroutine)
                if hasattr(result, "__await__"):
                    result = await result
                
                future.set_result(result)
            except Exception as e:
                future.set_exception(e)

        # Schedule coroutine on the event loop
        asyncio.run_coroutine_threadsafe(execute_async(), self._loop)

        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.
        
        Args:
            wait: If True, wait for pending tasks
        """
        if self._shutdown:
            return

        self._shutdown = True

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if wait and self._thread:
            self._thread.join(timeout=5.0)

        logger.info(f"{self.name} shutdown")

    def is_available(self) -> bool:
        """Check if executor is available.
        
        Returns:
            True if loop is running
        """
        return not self._shutdown and self._loop is not None and self._loop.is_running()

    def scale(self, num_workers: int) -> None:
        """Scale the executor.
        
        AsyncIO doesn't have workers in the traditional sense,
        so this is a no-op.
        
        Args:
            num_workers: Ignored
        """
        # AsyncIO uses cooperative multitasking, no worker concept
        pass
