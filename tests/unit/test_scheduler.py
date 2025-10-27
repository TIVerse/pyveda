"""Tests for scheduler functionality."""

from pyveda.config import Config, ExecutorType
from pyveda.core.scheduler import AdaptiveScheduler
from pyveda.core.task import Task
from pyveda.executors.thread_pool import ThreadPoolExecutor


def test_scheduler_initialization():
    """Test scheduler initialization."""
    config = Config(num_threads=2)
    scheduler = AdaptiveScheduler(config)
    assert scheduler.config == config


def test_scheduler_executor_registration():
    """Test executor registration."""
    config = Config(num_threads=2)
    scheduler = AdaptiveScheduler(config)

    executor = ThreadPoolExecutor(max_workers=2)
    scheduler.register_executor(ExecutorType.THREAD, executor)

    assert ExecutorType.THREAD in scheduler._executors


def test_scheduler_task_submission():
    """Test task submission."""
    config = Config(num_threads=2)
    scheduler = AdaptiveScheduler(config)

    executor = ThreadPoolExecutor(max_workers=2)
    scheduler.register_executor(ExecutorType.THREAD, executor)
    scheduler.start()

    task = Task(func=lambda x: x * 2, args=(5,))
    future = scheduler.submit(task)
    result = future.result(timeout=1.0)

    assert result == 10

    scheduler.shutdown()
    executor.shutdown()


def test_scheduler_stats():
    """Test scheduler statistics."""
    config = Config(num_threads=2)
    scheduler = AdaptiveScheduler(config)
    scheduler.start()

    stats = scheduler.get_stats()
    assert "executors" in stats
    assert "cpu_percent" in stats

    scheduler.shutdown()
