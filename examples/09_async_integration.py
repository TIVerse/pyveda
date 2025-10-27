"""Async/await integration with PyVeda example."""

import asyncio
import time

import pyveda as veda


async def async_fetch(url: str) -> dict:
    """Simulate async HTTP fetch.
    
    Args:
        url: URL to fetch
        
    Returns:
        Response data
    """
    # Simulate network delay
    await asyncio.sleep(0.1)
    return {
        'url': url,
        'status': 200,
        'data': f'Content from {url}',
        'timestamp': time.time()
    }


async def async_process(data: dict) -> dict:
    """Process fetched data asynchronously.
    
    Args:
        data: Raw data
        
    Returns:
        Processed data
    """
    await asyncio.sleep(0.05)
    return {
        'url': data['url'],
        'processed': data['data'].upper(),
        'length': len(data['data'])
    }


def sync_compute(x: int) -> int:
    """CPU-bound synchronous computation.
    
    Args:
        x: Input value
        
    Returns:
        Computed result
    """
    # Simulate heavy computation
    result = sum(i ** 2 for i in range(x * 100))
    return result


async def hybrid_task(task_id: int) -> dict:
    """Hybrid task combining async IO and sync compute.
    
    Args:
        task_id: Task identifier
        
    Returns:
        Task result
    """
    # Async fetch
    url = f"https://api.example.com/data/{task_id}"
    data = await async_fetch(url)
    
    # Sync compute (would block event loop if not handled properly)
    # PyVeda can offload this to thread pool
    compute_result = sync_compute(task_id)
    
    # Async process
    processed = await async_process(data)
    
    return {
        'task_id': task_id,
        'fetch_result': processed,
        'compute_result': compute_result
    }


def main():
    """Demonstrate async integration."""
    print("PyVeda - Async Integration Example\n")
    
    # Initialize runtime with async support
    config = veda.Config.builder()\
        .threads(4)\
        .build()
    veda.init(config)
    
    # Example 1: Parallel async operations with iterator
    print("1. Parallel async operations via iterator")
    urls = [f"https://example.com/page/{i}" for i in range(10)]
    
    start = time.time()
    results = veda.par_iter(urls)\
        .async_map(async_fetch)\
        .collect()
    
    elapsed = time.time() - start
    print(f"   Fetched {len(results)} URLs in {elapsed:.2f}s\n")
    
    # Example 2: Mix async and sync tasks
    print("2. Hybrid async/sync tasks")
    
    with veda.scope() as s:
        # Spawn async tasks
        async_futures = []
        for i in range(5):
            future = s.spawn(lambda i=i: asyncio.run(hybrid_task(i)))
            async_futures.append(future)
        
        # Spawn sync CPU tasks
        sync_futures = []
        for i in range(5):
            future = s.spawn(sync_compute, i * 10)
            sync_futures.append(future)
        
        # All tasks run concurrently (async in async executor, sync in thread pool)
        
    print(f"   Completed {len(async_futures)} async and {len(sync_futures)} sync tasks\n")
    
    # Example 3: Async pipeline
    print("3. Async data processing pipeline")
    
    start = time.time()
    pipeline_results = veda.par_iter(range(20))\
        .async_map(lambda i: async_fetch(f"https://api.example.com/{i}"))\
        .async_map(async_process)\
        .collect()
    
    elapsed = time.time() - start
    print(f"   Processed {len(pipeline_results)} items in {elapsed:.2f}s\n")
    
    print("✓ Async integration complete")


if __name__ == "__main__":
    main()
    veda.shutdown()
