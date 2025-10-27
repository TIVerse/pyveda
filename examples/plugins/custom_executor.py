"""Custom Executor Plugin Example for VedaRT.

This example demonstrates how to create a custom executor backend
for VedaRT, enabling third-party execution strategies.

Use cases:
- Distributed execution (e.g., Redis queue, RabbitMQ)
- Cloud function execution (AWS Lambda, Google Cloud Functions)
- Custom scheduling policies
- Integration with existing job systems
"""

import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable
from queue import Queue, Empty
import threading

from vedart.core.executor import Executor
from vedart.core.task import Task
from vedart.config import ExecutorType


@dataclass
class CustomExecutorConfig:
    """Configuration for the custom executor."""
    
    max_workers: int = 4
    queue_size: int = 100
    timeout: float = 30.0
    name: str = "custom"


class CustomExecutor(Executor):
    """Custom executor implementation.
    
    This is a simple example showing how to implement a custom executor.
    In practice, this could connect to:
    - Message queues (Redis, RabbitMQ, Kafka)
    - Cloud services (AWS Lambda, Azure Functions)
    - Container orchestrators (Kubernetes, Docker Swarm)
    - Custom job schedulers (Slurm, PBS, SGE)
    """
    
    def __init__(self, config: CustomExecutorConfig):
        """Initialize the custom executor.
        
        Args:
            config: Executor configuration
        """
        super().__init__(name=config.name, executor_type=ExecutorType.THREAD)
        self.config = config
        self._task_queue: Queue[tuple[Task, Future]] = Queue(maxsize=config.queue_size)
        self._workers: list[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._start_workers()
    
    def _start_workers(self) -> None:
        """Start worker threads."""
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"{self.config.name}-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                task, future = self._task_queue.get(timeout=0.1)
                
                # Execute task
                try:
                    result = self._execute_task(task)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._task_queue.task_done()
                    
            except Empty:
                continue
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a single task.
        
        This is where you'd implement custom execution logic:
        - Send to remote queue
        - Invoke cloud function
        - Submit to job scheduler
        - etc.
        """
        # Simple local execution for demo
        return task.func(*task.args, **task.kwargs)
    
    def submit(self, task: Task) -> Future:
        """Submit a task for execution.
        
        Args:
            task: Task to execute
            
        Returns:
            Future representing the task result
        """
        if self._shutdown_event.is_set():
            raise RuntimeError(f"Executor {self.name} is shutdown")
        
        future: Future = Future()
        
        try:
            self._task_queue.put((task, future), timeout=self.config.timeout)
        except Exception as e:
            future.set_exception(e)
        
        return future
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor.
        
        Args:
            wait: Whether to wait for pending tasks
        """
        self._shutdown_event.set()
        
        if wait:
            # Wait for queue to empty
            self._task_queue.join()
            
            # Wait for workers to finish
            for worker in self._workers:
                worker.join(timeout=5.0)
        
        self._workers.clear()
    
    def map(self, func: Callable, *iterables) -> list[Any]:
        """Map a function over iterables.
        
        Args:
            func: Function to map
            *iterables: Input iterables
            
        Returns:
            List of results
        """
        tasks = []
        futures = []
        
        for args in zip(*iterables):
            task = Task(func=func, args=args, kwargs={})
            future = self.submit(task)
            tasks.append(task)
            futures.append(future)
        
        return [f.result() for f in futures]


# ============================================================================
# Example 1: Basic Usage
# ============================================================================

def example_basic_usage():
    """Demonstrate basic custom executor usage."""
    print("=" * 60)
    print("Example 1: Basic Custom Executor Usage")
    print("=" * 60)
    
    # Create custom executor
    config = CustomExecutorConfig(max_workers=4)
    executor = CustomExecutor(config)
    
    # Submit tasks
    def square(x: int) -> int:
        time.sleep(0.1)  # Simulate work
        return x ** 2
    
    # Submit multiple tasks
    futures = []
    for i in range(10):
        task = Task(func=square, args=(i,), kwargs={})
        future = executor.submit(task)
        futures.append(future)
    
    # Get results
    results = [f.result() for f in futures]
    print(f"Results: {results}")
    
    # Cleanup
    executor.shutdown()
    print("✓ Custom executor completed successfully\n")


# ============================================================================
# Example 2: Integration with VedaRT Runtime
# ============================================================================

def example_runtime_integration():
    """Show how to integrate custom executor with VedaRT runtime."""
    print("=" * 60)
    print("Example 2: Runtime Integration")
    print("=" * 60)
    
    import vedart as veda
    
    # Create custom executor
    config = CustomExecutorConfig(max_workers=4, name="my_custom")
    custom_executor = CustomExecutor(config)
    
    # Note: In a full implementation, you would register this with the runtime
    # For now, we demonstrate the interface compatibility
    
    def process_item(x: int) -> int:
        return x * 2 + 1
    
    # Use custom executor directly
    results = custom_executor.map(process_item, range(20))
    print(f"Processed {len(results)} items: {results[:5]}...")
    
    # Cleanup
    custom_executor.shutdown()
    print("✓ Runtime integration demonstrated\n")


# ============================================================================
# Example 3: Cloud Function Executor (Simulated)
# ============================================================================

class CloudFunctionExecutor(CustomExecutor):
    """Simulated cloud function executor.
    
    In a real implementation, this would:
    1. Serialize the task
    2. Invoke cloud function (AWS Lambda, etc.)
    3. Poll for result or use callback
    4. Deserialize and return
    """
    
    def _execute_task(self, task: Task) -> Any:
        """Simulate cloud function execution."""
        # Simulate network latency
        time.sleep(0.05)
        
        # In practice: 
        # - Serialize task with pickle/cloudpickle
        # - Invoke cloud API
        # - Wait for result
        # - Deserialize response
        
        print(f"  → Invoking cloud function for {task.func.__name__}")
        result = task.func(*task.args, **task.kwargs)
        
        # Simulate result retrieval
        time.sleep(0.05)
        
        return result


def example_cloud_executor():
    """Demonstrate cloud function executor."""
    print("=" * 60)
    print("Example 3: Cloud Function Executor (Simulated)")
    print("=" * 60)
    
    config = CustomExecutorConfig(max_workers=4, name="cloud")
    executor = CloudFunctionExecutor(config)
    
    def expensive_computation(n: int) -> int:
        """Simulate expensive computation."""
        result = sum(i * i for i in range(n))
        return result
    
    # Submit tasks
    print("Submitting tasks to 'cloud'...")
    results = executor.map(expensive_computation, [100, 200, 300, 400, 500])
    
    print(f"Results: {results}")
    
    executor.shutdown()
    print("✓ Cloud executor simulation completed\n")


# ============================================================================
# Example 4: Priority Queue Executor
# ============================================================================

from queue import PriorityQueue

class PriorityExecutor(Executor):
    """Executor with task priority support."""
    
    def __init__(self, max_workers: int = 4):
        super().__init__(name="priority", executor_type=ExecutorType.THREAD)
        self.max_workers = max_workers
        self._task_queue: PriorityQueue = PriorityQueue()
        self._workers: list[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._counter = 0  # For stable sorting
        self._start_workers()
    
    def _start_workers(self) -> None:
        """Start worker threads."""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"priority-worker-{i}",
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)
    
    def _worker_loop(self) -> None:
        """Worker thread main loop."""
        while not self._shutdown_event.is_set():
            try:
                _, _, task, future = self._task_queue.get(timeout=0.1)
                
                try:
                    result = task.func(*task.args, **task.kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._task_queue.task_done()
                    
            except Empty:
                continue
    
    def submit(self, task: Task, priority: int = 0) -> Future:
        """Submit task with priority (lower number = higher priority).
        
        Args:
            task: Task to execute
            priority: Task priority (default 0)
            
        Returns:
            Future representing the task result
        """
        future: Future = Future()
        
        # Use counter for stable sorting
        self._counter += 1
        self._task_queue.put((priority, self._counter, task, future))
        
        return future
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        self._shutdown_event.set()
        
        if wait:
            self._task_queue.join()
            for worker in self._workers:
                worker.join(timeout=5.0)
        
        self._workers.clear()


def example_priority_executor():
    """Demonstrate priority-based task execution."""
    print("=" * 60)
    print("Example 4: Priority Queue Executor")
    print("=" * 60)
    
    executor = PriorityExecutor(max_workers=2)
    
    def task_with_delay(name: str, duration: float) -> str:
        time.sleep(duration)
        return f"Completed: {name}"
    
    # Submit tasks with different priorities
    # Lower priority number = higher priority
    print("Submitting tasks (low priority tasks submitted first)...")
    
    low_priority = []
    for i in range(5):
        task = Task(func=task_with_delay, args=(f"Low-{i}", 0.1), kwargs={})
        future = executor.submit(task, priority=10)
        low_priority.append(future)
    
    # Give workers time to start processing
    time.sleep(0.05)
    
    # Submit high priority tasks
    high_priority = []
    for i in range(3):
        task = Task(func=task_with_delay, args=(f"HIGH-{i}", 0.1), kwargs={})
        future = executor.submit(task, priority=1)
        high_priority.append(future)
    
    # Collect results
    print("\nResults (notice high priority tasks complete first):")
    for future in high_priority + low_priority:
        print(f"  {future.result()}")
    
    executor.shutdown()
    print("✓ Priority executor completed\n")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("VedaRT Custom Executor Plugin Examples")
    print("=" * 60 + "\n")
    
    example_basic_usage()
    example_runtime_integration()
    example_cloud_executor()
    example_priority_executor()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  • Custom executors extend VedaRT's capabilities")
    print("  • Implement Executor protocol for compatibility")
    print("  • Use cases: cloud, distributed, custom scheduling")
    print("  • Priority queues, rate limiting, etc. are possible")
    print()
