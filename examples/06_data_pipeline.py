"""Real-world data processing pipeline example."""

import pyveda as veda


def extract_data(file_id):
    """Simulate data extraction."""
    return [f"record_{file_id}_{i}" for i in range(100)]


def transform_record(record):
    """Transform a single record."""
    return record.upper()


def validate_record(record):
    """Validate a record."""
    return len(record) > 5


def main():
    """Demonstrate ETL pipeline with PyVeda."""
    print("PyVeda - Data Processing Pipeline\n")
    
    # Stage 1: Extract data from multiple sources
    print("Stage 1: Extract")
    file_ids = range(10)
    
    with veda.scope() as s:
        extract_futures = [s.spawn(extract_data, fid) for fid in file_ids]
        raw_data = s.wait_all()
    
    # Flatten
    all_records = [record for batch in raw_data for record in batch]
    print(f"  Extracted {len(all_records)} records\n")
    
    # Stage 2: Transform
    print("Stage 2: Transform (parallel)")
    transformed = veda.par_iter(all_records).map(transform_record).collect()
    print(f"  Transformed {len(transformed)} records\n")
    
    # Stage 3: Validate and filter
    print("Stage 3: Validate & Filter (parallel)")
    valid_records = veda.par_iter(transformed).filter(validate_record).collect()
    print(f"  Valid records: {len(valid_records)}\n")
    
    # Stage 4: Aggregate
    print("Stage 4: Aggregate")
    record_count = len(valid_records)
    print(f"  Final count: {record_count}")
    print(f"  Sample: {valid_records[:3]}")


if __name__ == "__main__":
    main()
    veda.shutdown()
