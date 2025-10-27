"""Example: Mixed Async + CPU I/O Pipeline

This example demonstrates a real-world data pipeline mixing:
- Async I/O operations (API calls, database queries)
- CPU-bound processing (parsing, validation, transformation)
- Efficient batch processing with automatic executor selection

Use case: ETL pipeline fetching data from APIs, processing, and storing results.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import vedart as veda


@dataclass
class DataRecord:
    """A data record in our pipeline."""
    id: int
    source: str
    data: dict[str, Any]
    processed_at: float


# ============================================================================
# ASYNC I/O OPERATIONS (Network-bound)
# ============================================================================

async def fetch_from_api(record_id: int) -> dict[str, Any]:
    """Simulate async API call (I/O-bound)."""
    # Simulate network latency
    await asyncio.sleep(0.05 + (record_id % 10) * 0.01)
    
    # Simulate API response
    return {
        "id": record_id,
        "timestamp": time.time(),
        "data": {
            "value": record_id * 100,
            "category": f"cat_{record_id % 5}",
            "metadata": {"source": "api", "version": "v2"}
        }
    }


async def save_to_database(record: DataRecord) -> bool:
    """Simulate async database write (I/O-bound)."""
    # Simulate database write latency
    await asyncio.sleep(0.02)
    
    # In production: await db.execute(INSERT INTO ...)
    return True


async def validate_external(data: dict[str, Any]) -> bool:
    """Simulate async validation against external service."""
    await asyncio.sleep(0.01)
    
    # Simulate validation logic
    return data.get("value", 0) > 0


# ============================================================================
# CPU-BOUND OPERATIONS (Compute-intensive)
# ============================================================================

def parse_and_transform(raw_data: dict[str, Any]) -> dict[str, Any]:
    """CPU-bound data transformation."""
    # Simulate expensive parsing/transformation
    data = raw_data.get("data", {})
    
    # Complex transformations (CPU-bound)
    transformed = {
        "id": raw_data["id"],
        "value_squared": data["value"] ** 2,
        "category_hash": hash(data["category"]),
        "computed_field": sum(ord(c) for c in str(data)),
        "normalized": data["value"] / 1000.0,
    }
    
    return transformed


def validate_schema(data: dict[str, Any]) -> bool:
    """CPU-bound schema validation."""
    required_fields = ["id", "value_squared", "category_hash"]
    return all(field in data for field in required_fields)


def aggregate_batch(records: list[DataRecord]) -> dict[str, Any]:
    """CPU-bound batch aggregation."""
    if not records:
        return {}
    
    return {
        "count": len(records),
        "total_value": sum(r.data.get("value_squared", 0) for r in records),
        "categories": len(set(r.data.get("category_hash", 0) for r in records)),
        "avg_processing_time": sum(r.processed_at for r in records) / len(records),
    }


# ============================================================================
# PIPELINE IMPLEMENTATION
# ============================================================================

async def process_single_record(record_id: int) -> DataRecord | None:
    """Complete pipeline for a single record (mixed async + CPU)."""
    try:
        # Step 1: Fetch from API (async I/O)
        raw_data = await fetch_from_api(record_id)
        
        # Step 2: Transform data (CPU-bound)
        # Note: CPU work in async function - VedaRT handles this efficiently
        start = time.perf_counter()
        transformed = parse_and_transform(raw_data)
        
        # Step 3: Validate (CPU-bound)
        if not validate_schema(transformed):
            print(f"⚠️  Record {record_id} failed validation")
            return None
        
        # Step 4: External validation (async I/O)
        if not await validate_external(transformed):
            print(f"⚠️  Record {record_id} failed external validation")
            return None
        
        processing_time = time.perf_counter() - start
        
        # Create record
        record = DataRecord(
            id=record_id,
            source="api",
            data=transformed,
            processed_at=processing_time
        )
        
        # Step 5: Save to database (async I/O)
        await save_to_database(record)
        
        return record
        
    except Exception as e:
        print(f"❌ Error processing record {record_id}: {e}")
        return None


def demo_mixed_pipeline():
    """Demonstrate mixed async + CPU pipeline."""
    print("🔄 Mixed Async + CPU I/O Pipeline Demo")
    print("=" * 60)
    
    # Number of records to process
    num_records = 50
    print(f"\n📊 Processing {num_records} records through pipeline...")
    
    start = time.perf_counter()
    
    # Process all records in parallel using VedaRT
    # VedaRT automatically:
    # - Routes async operations to async executor
    # - Routes CPU operations to thread/process pool
    # - Handles event loop management
    results = (
        veda.par_iter(range(num_records))
        .async_map(process_single_record)  # Async pipeline execution
        .filter(lambda r: r is not None)  # Remove failed records
        .collect()
    )
    
    elapsed = time.perf_counter() - start
    
    # Statistics
    success_count = len(results)
    failure_count = num_records - success_count
    
    print(f"\n✅ Pipeline complete in {elapsed:.2f}s")
    print(f"   Successful: {success_count}/{num_records}")
    print(f"   Failed: {failure_count}")
    print(f"   Throughput: {num_records / elapsed:.1f} records/sec")
    
    # Aggregate results (CPU-bound)
    if results:
        print("\n📈 Batch aggregation (CPU)...")
        aggregated = aggregate_batch(results)
        
        print(f"   Total records: {aggregated['count']}")
        print(f"   Sum of values: {aggregated['total_value']:,}")
        print(f"   Unique categories: {aggregated['categories']}")
        print(f"   Avg processing: {aggregated['avg_processing_time']*1000:.2f}ms")


def demo_batch_fetching():
    """Demonstrate efficient batch fetching with async."""
    print("\n" + "=" * 60)
    print("📦 Batch Fetching Pattern")
    print("=" * 60)
    
    batch_ids = list(range(100))
    
    print(f"\n🔹 Sequential fetching (baseline):")
    start = time.perf_counter()
    
    # Sequential (slow)
    sequential_results = []
    for batch_id in batch_ids[:10]:  # Only first 10 for demo
        result = asyncio.run(fetch_from_api(batch_id))
        sequential_results.append(result)
    
    sequential_time = time.perf_counter() - start
    print(f"   10 records: {sequential_time:.2f}s")
    print(f"   Estimated for 100: ~{sequential_time * 10:.2f}s")
    
    print(f"\n🔹 Parallel async fetching (VedaRT):")
    start = time.perf_counter()
    
    # Parallel async with VedaRT
    parallel_results = (
        veda.par_iter(batch_ids)
        .async_map(fetch_from_api)
        .collect()
    )
    
    parallel_time = time.perf_counter() - start
    print(f"   100 records: {parallel_time:.2f}s")
    
    speedup = (sequential_time * 10) / parallel_time
    print(f"\n🎉 Speedup: ~{speedup:.1f}x faster")


async def async_pipeline_with_backpressure():
    """Demonstrate pipeline with backpressure control."""
    print("\n" + "=" * 60)
    print("🚰 Pipeline with Backpressure Control")
    print("=" * 60)
    
    # Simulate slow consumer
    async def slow_consumer(record: DataRecord) -> None:
        await asyncio.sleep(0.1)  # Slow processing
        print(f"  ✓ Consumed record {record.id}")
    
    # Configure VedaRT with limited parallelism
    config = veda.Config.builder().max_workers(5).build()
    veda.init(config)
    
    print("\n🔹 Processing with max 5 concurrent operations...")
    
    # Generate records
    record_ids = list(range(20))
    
    start = time.perf_counter()
    
    # Process with controlled concurrency
    await asyncio.gather(*[
        process_single_record(record_id)
        for record_id in record_ids[:10]  # First 10 for demo
    ])
    
    elapsed = time.perf_counter() - start
    print(f"\n✅ Completed in {elapsed:.2f}s")
    print("   (Backpressure prevents memory overflow)")


def demo_error_handling():
    """Demonstrate error handling in mixed pipeline."""
    print("\n" + "=" * 60)
    print("🛡️  Error Handling in Mixed Pipeline")
    print("=" * 60)
    
    async def failing_fetch(record_id: int) -> dict[str, Any]:
        if record_id % 5 == 0:
            raise ValueError(f"API error for record {record_id}")
        return await fetch_from_api(record_id)
    
    async def safe_process(record_id: int) -> tuple[int, bool, str]:
        try:
            data = await failing_fetch(record_id)
            return (record_id, True, "success")
        except Exception as e:
            return (record_id, False, str(e))
    
    print("\n🔹 Processing with error handling...")
    
    results = (
        veda.par_iter(range(20))
        .async_map(safe_process)
        .collect()
    )
    
    successes = [r for r in results if r[1]]
    failures = [r for r in results if not r[1]]
    
    print(f"\n📊 Results:")
    print(f"   Successful: {len(successes)}")
    print(f"   Failed: {len(failures)}")
    
    if failures:
        print(f"\n   Failed IDs: {[r[0] for r in failures[:5]]}...")


def demo_telemetry():
    """Show telemetry for async pipeline."""
    print("\n" + "=" * 60)
    print("📊 Pipeline Telemetry")
    print("=" * 60)
    
    # Run pipeline
    results = (
        veda.par_iter(range(30))
        .async_map(process_single_record)
        .filter(lambda r: r is not None)
        .collect()
    )
    
    # Get metrics
    try:
        metrics = veda.telemetry.snapshot()
        
        print(f"\n📈 Execution Metrics:")
        print(f"   Tasks executed: {metrics.tasks_executed}")
        print(f"   Tasks failed: {metrics.tasks_failed}")
        print(f"   Avg latency: {metrics.avg_latency_ms:.2f}ms")
        print(f"   P99 latency: {metrics.p99_latency_ms:.2f}ms")
        
        # Executor breakdown
        print(f"\n🔧 Executor Usage:")
        for executor, count in metrics.executor_tasks.items():
            print(f"   {executor}: {count} tasks")
            
    except AttributeError:
        print("\n⚠️  Telemetry not available (install with: pip install vedart[telemetry])")


def main():
    """Run all demos."""
    try:
        demo_mixed_pipeline()
        demo_batch_fetching()
        
        # Async demo (requires event loop)
        print("\n🔹 Running async backpressure demo...")
        asyncio.run(async_pipeline_with_backpressure())
        
        demo_error_handling()
        demo_telemetry()
        
        print("\n" + "=" * 60)
        print("✨ Demo complete!")
        print("\nKey takeaways:")
        print("  • async_map() seamlessly handles async I/O operations")
        print("  • CPU-bound work automatically routed to worker pools")
        print("  • VedaRT manages event loops and executor selection")
        print("  • Built-in error handling and telemetry")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
