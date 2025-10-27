# VedaRT Plugin System

This directory contains examples of custom executor plugins for VedaRT.

## Overview

VedaRT's plugin system allows you to extend the runtime with custom executors for specialized use cases:

- 🌐 **Distributed execution** (Redis, RabbitMQ, Kafka)
- ☁️ **Cloud functions** (AWS Lambda, Google Cloud Functions, Azure Functions)
- 🎯 **Custom scheduling** (priority queues, rate limiting, quotas)
- 🔗 **Integration with existing systems** (Slurm, PBS, Kubernetes)

## Creating a Custom Executor

### 1. Implement the Executor Protocol

```python
from vedart.core.executor import Executor
from vedart.core.task import Task
from concurrent.futures import Future

class MyCustomExecutor(Executor):
    def __init__(self, name: str = "custom"):
        super().__init__(name=name, executor_type=ExecutorType.THREAD)
    
    def submit(self, task: Task) -> Future:
        """Submit a task for execution."""
        # Your implementation here
        pass
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor."""
        # Your cleanup here
        pass
```

### 2. Register with Runtime (Optional)

```python
import vedart as veda

# Create your executor
custom_executor = MyCustomExecutor()

# Register it (if using runtime integration)
runtime = veda.get_runtime()
# runtime.register_executor(custom_executor)  # Future API
```

## Examples

### Basic Custom Executor
See `custom_executor.py` for a complete implementation including:
- Task queue management
- Worker thread pool
- Error handling
- Graceful shutdown

### Cloud Function Executor
Demonstrates how to integrate with cloud services:
- Task serialization
- Remote invocation
- Result retrieval

### Priority Queue Executor
Shows advanced scheduling:
- Priority-based task execution
- Stable sorting
- Preemptive scheduling

## Use Cases

### Distributed Computing
```python
class RedisQueueExecutor(Executor):
    """Execute tasks via Redis queue."""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        # ...
    
    def submit(self, task: Task) -> Future:
        # Serialize and push to Redis
        task_data = pickle.dumps((task.func, task.args, task.kwargs))
        self.redis.rpush("vedart:tasks", task_data)
        # Return future that polls for result
        # ...
```

### Rate-Limited Executor
```python
class RateLimitedExecutor(Executor):
    """Limit task submission rate."""
    
    def __init__(self, max_rate: float):
        self.limiter = RateLimiter(max_rate)
        # ...
    
    def submit(self, task: Task) -> Future:
        self.limiter.acquire()
        return super().submit(task)
```

### Hybrid Executor
```python
class HybridExecutor(Executor):
    """Route tasks to local or cloud based on cost."""
    
    def submit(self, task: Task) -> Future:
        if self._should_run_locally(task):
            return self.local_executor.submit(task)
        else:
            return self.cloud_executor.submit(task)
```

## Running Examples

```bash
# Run the custom executor example
python examples/plugins/custom_executor.py

# Output shows:
# - Basic executor usage
# - Runtime integration
# - Cloud simulation
# - Priority scheduling
```

## Best Practices

1. **Thread Safety**: Ensure your executor is thread-safe if used concurrently
2. **Resource Cleanup**: Always implement proper shutdown logic
3. **Error Handling**: Propagate exceptions through futures
4. **Monitoring**: Add metrics/logging for observability
5. **Documentation**: Document executor capabilities and limitations

## API Stability

⚠️ **Experimental**: The plugin API is currently experimental and may change in future versions.

For production use cases, please:
- Pin to specific VedaRT versions
- Test thoroughly with your workload
- Monitor for API deprecations

## Contributing

Have a useful executor implementation? Consider contributing:
1. Add to `examples/plugins/`
2. Include tests in `tests/plugins/`
3. Document use cases
4. Submit PR with examples

## References

- [Core Executor Protocol](../../src/vedart/core/executor.py)
- [Task Definition](../../src/vedart/core/task.py)
- [Scheduler Architecture](../../docs/architecture.md)
