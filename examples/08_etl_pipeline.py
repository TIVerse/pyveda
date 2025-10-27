"""ETL (Extract, Transform, Load) pipeline example."""

import time
from typing import Dict, List

import pyveda as veda


def extract_data(batch_id: int) -> List[Dict]:
    """Simulate data extraction from source.
    
    Args:
        batch_id: Batch identifier
        
    Returns:
        List of raw records
    """
    # Simulate IO-bound extraction
    time.sleep(0.01)
    return [
        {'id': batch_id * 100 + i, 'value': i * 2, 'category': 'A' if i % 2 == 0 else 'B'}
        for i in range(100)
    ]


def transform_record(record: Dict) -> Dict:
    """Transform a single record.
    
    Args:
        record: Raw record
        
    Returns:
        Transformed record
    """
    return {
        'id': record['id'],
        'processed_value': record['value'] ** 2,
        'category': record['category'],
        'timestamp': time.time()
    }


def load_batch(records: List[Dict]) -> int:
    """Simulate loading records to destination.
    
    Args:
        records: Transformed records
        
    Returns:
        Number of records loaded
    """
    # Simulate IO-bound load
    time.sleep(0.02)
    return len(records)


def main():
    """Demonstrate ETL pipeline with PyVeda."""
    print("PyVeda - ETL Pipeline Example\n")
    
    # Initialize runtime
    config = veda.Config.builder()\
        .threads(4)\
        .telemetry(True)\
        .build()
    veda.init(config)
    
    num_batches = 10
    start_time = time.time()
    
    print(f"Processing {num_batches} batches...\n")
    
    # Extract phase - parallel batch extraction
    print("1. Extract phase (parallel)")
    raw_batches = veda.par_iter(range(num_batches))\
        .map(extract_data)\
        .collect()
    
    # Flatten all records
    all_records = [record for batch in raw_batches for record in batch]
    print(f"   Extracted {len(all_records)} records\n")
    
    # Transform phase - parallel record transformation
    print("2. Transform phase (parallel)")
    transformed = veda.par_iter(all_records)\
        .map(transform_record)\
        .collect()
    print(f"   Transformed {len(transformed)} records\n")
    
    # Load phase - batch loading with chunking
    print("3. Load phase (chunked)")
    chunk_size = 100
    chunks = [transformed[i:i+chunk_size] for i in range(0, len(transformed), chunk_size)]
    
    loaded_counts = veda.par_iter(chunks)\
        .map(load_batch)\
        .collect()
    
    total_loaded = sum(loaded_counts)
    print(f"   Loaded {total_loaded} records in {len(chunks)} chunks\n")
    
    elapsed = time.time() - start_time
    print(f"Pipeline completed in {elapsed:.2f}s")
    print(f"Throughput: {total_loaded / elapsed:.0f} records/sec\n")
    
    # Show telemetry
    runtime = veda.get_runtime()
    if runtime.telemetry:
        snapshot = runtime.telemetry.snapshot()
        print("Telemetry:")
        print(f"  Tasks executed: {snapshot.tasks_executed}")
        print(f"  Avg latency: {snapshot.avg_latency_ms:.2f}ms")
        print(f"  Throughput: {snapshot.throughput_tasks_per_sec:.1f} tasks/sec")


if __name__ == "__main__":
    main()
    veda.shutdown()
