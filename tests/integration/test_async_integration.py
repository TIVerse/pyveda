"""Integration tests for async functionality."""

import asyncio

import pytest

import pyveda as veda


async def async_double(x):
    """Async function that doubles input."""
    await asyncio.sleep(0.001)  # Simulate async IO
    return x * 2


async def async_is_even(x):
    """Async predicate function."""
    await asyncio.sleep(0.001)
    return x % 2 == 0


def test_async_task_detection():
    """Test automatic async task detection."""
    from pyveda.core.task import Task
    
    # Sync function
    sync_task = Task(func=lambda x: x * 2, args=(5,))
    assert sync_task.is_async is False
    
    # Async function
    async_task = Task(func=async_double, args=(5,))
    assert async_task.is_async is True


def test_async_spawn():
    """Test spawning async tasks."""
    veda.init()
    
    async def async_compute(x):
        await asyncio.sleep(0.01)
        return x ** 2
    
    # Spawn async task
    future = veda.spawn(async_compute, 5)
    result = future.result()
    
    assert result == 25
    
    veda.shutdown()


def test_async_scope():
    """Test async tasks in scope."""
    veda.init()
    
    async def async_task(x):
        await asyncio.sleep(0.01)
        return x + 1
    
    with veda.scope() as s:
        futures = [s.spawn(async_task, i) for i in range(5)]
    
    # All tasks completed when scope exits
    results = [f.result() for f in futures]
    assert results == [1, 2, 3, 4, 5]
    
    veda.shutdown()


def test_async_iterator_integration():
    """Test async_map in parallel iterator."""
    veda.init()
    
    async def async_transform(x):
        await asyncio.sleep(0.001)
        return x * 3
    
    results = veda.par_iter(range(10))\
        .async_map(async_transform)\
        .collect()
    
    assert len(results) == 10
    assert results[0] == 0
    assert results[5] == 15
    
    veda.shutdown()


def test_mixed_async_sync_tasks():
    """Test mixing async and sync tasks."""
    veda.init()
    
    def sync_task(x):
        return x * 2
    
    async def async_task(x):
        await asyncio.sleep(0.001)
        return x * 3
    
    with veda.scope() as s:
        sync_futures = [s.spawn(sync_task, i) for i in range(5)]
        async_futures = [s.spawn(async_task, i) for i in range(5)]
    
    sync_results = [f.result() for f in sync_futures]
    async_results = [f.result() for f in async_futures]
    
    assert sync_results == [0, 2, 4, 6, 8]
    assert async_results == [0, 3, 6, 9, 12]
    
    veda.shutdown()


def test_async_executor_routing():
    """Test async tasks route to AsyncIOExecutor."""
    config = veda.Config.builder().threads(2).build()
    veda.init(config)
    
    async def async_op(x):
        await asyncio.sleep(0.001)
        return x
    
    # Execute async task
    future = veda.spawn(async_op, 42)
    result = future.result()
    
    assert result == 42
    
    # Check that async executor handled tasks
    runtime = veda.get_runtime()
    stats = runtime.scheduler.get_stats()
    
    # Should have async executor stats
    from pyveda.config import ExecutorType
    if ExecutorType.ASYNC in runtime.scheduler._executors:
        assert 'async' in stats['executors'] or ExecutorType.ASYNC in stats['executors']
    
    veda.shutdown()
