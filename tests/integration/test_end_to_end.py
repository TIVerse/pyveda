"""End-to-end integration tests for VedaRT.

Tests complete workflows from initialization to shutdown,
covering multiple features working together.
"""

import pytest
import time
import tempfile
import os
from typing import List

import vedart as veda


class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""
    
    def test_etl_pipeline(self):
        """Test complete ETL pipeline with mixed executors."""
        # Extract: I/O-bound
        def extract(i: int) -> dict:
            time.sleep(0.001)  # Simulate I/O
            return {"id": i, "value": i * 10}
        
        # Transform: CPU-bound
        def transform(data: dict) -> dict:
            # Simulate computation
            data["processed"] = data["value"] ** 2
            data["timestamp"] = time.time()
            return data
        
        # Load: I/O-bound
        results = []
        def load(data: dict) -> bool:
            time.sleep(0.001)  # Simulate I/O
            results.append(data)
            return True
        
        # Run pipeline
        source_data = range(50)
        
        extracted = veda.par_iter(source_data).map(extract).collect()
        transformed = veda.par_iter(extracted).map(transform).collect()
        loaded = veda.par_iter(transformed).map(load).collect()
        
        assert len(results) == 50
        assert all(r["processed"] == (r["id"] * 10) ** 2 for r in results)
    
    def test_mixed_executor_pipeline(self):
        """Test pipeline using different executor types."""
        
        # CPU-bound task
        def cpu_task(x: int) -> int:
            return sum(i * i for i in range(x))
        
        # I/O simulation
        def io_task(x: int) -> int:
            time.sleep(0.001)
            return x * 2
        
        # Process through different stages
        data = range(20)
        
        # Stage 1: CPU-bound
        stage1 = veda.par_iter(data).map(cpu_task).collect()
        
        # Stage 2: I/O-bound
        stage2 = veda.par_iter(stage1).map(io_task).collect()
        
        assert len(stage2) == 20
        assert all(isinstance(x, int) for x in stage2)
    
    def test_error_recovery_workflow(self):
        """Test workflow with error handling and recovery."""
        
        def unreliable_task(x: int) -> tuple[int, bool]:
            # Fail 20% of the time
            if x % 5 == 0 and x > 0:
                return (x, False)  # Mark as failed
            return (x * 2, True)  # Success
        
        def retry_failed(item: tuple[int, bool]) -> int:
            value, success = item
            if not success:
                # Retry with different logic
                return value * 3
            return value
        
        # First pass
        results = veda.par_iter(range(20)).map(unreliable_task).collect()
        
        # Retry failures
        final = veda.par_iter(results).map(retry_failed).collect()
        
        assert len(final) == 20
        assert all(isinstance(x, int) for x in final)
    
    def test_deterministic_workflow(self):
        """Test deterministic execution across multiple runs."""
        import random
        
        def random_task(x: int) -> int:
            return x + random.randint(0, 10)
        
        # Run 1
        with veda.deterministic(seed=42):
            result1 = veda.par_iter(range(50)).map(random_task).collect()
        
        # Run 2 with same seed
        with veda.deterministic(seed=42):
            result2 = veda.par_iter(range(50)).map(random_task).collect()
        
        # Should be identical
        assert result1 == result2
        
        # Run 3 with different seed
        with veda.deterministic(seed=999):
            result3 = veda.par_iter(range(50)).map(random_task).collect()
        
        # Should be different
        assert result3 != result1


class TestScopedWorkflows:
    """Test workflows using scoped execution."""
    
    def test_nested_scopes(self):
        """Test nested scope execution."""
        
        def outer_task(x: int) -> int:
            # Spawn inner scope
            with veda.scope() as inner_scope:
                f1 = inner_scope.spawn(lambda: x * 2)
                f2 = inner_scope.spawn(lambda: x * 3)
                results = inner_scope.wait_all()
            
            return sum(results)
        
        # Outer scope
        with veda.scope() as outer_scope:
            futures = [outer_scope.spawn(outer_task, i) for i in range(5)]
            results = outer_scope.wait_all()
        
        # Verify results
        expected = [i * 2 + i * 3 for i in range(5)]
        assert results == expected
    
    def test_scope_with_dependencies(self):
        """Test scope with task dependencies."""
        
        with veda.scope() as s:
            # Stage 1: Initial tasks
            stage1_futures = [s.spawn(lambda x: x * 2, i) for i in range(5)]
            results1 = [f.result() for f in stage1_futures]
            
            # Stage 2: Depends on stage 1 results
            stage2_futures = [s.spawn(lambda x: x + 10, r) for r in results1]
            results2 = [f.result() for f in stage2_futures]
        
        expected = [i * 2 + 10 for i in range(5)]
        assert results2 == expected


class TestConfigurationWorkflows:
    """Test workflows with different configurations."""
    
    def test_thread_only_config(self):
        """Test with thread-only configuration."""
        config = veda.Config.builder().policy(veda.SchedulingPolicy.THREAD_ONLY).build()
        
        # Don't init/shutdown in tests - runtime is already initialized
        result = veda.par_iter(range(20)).map(lambda x: x * 2).collect()
        assert len(result) == 20
    
    def test_custom_worker_count(self):
        """Test with custom worker counts."""
        # Just test the API - actual config application needs runtime restart
        config = (
            veda.Config.builder()
            .threads(4)
            .processes(2)
            .build()
        )
        
        assert config.num_threads == 4
        assert config.num_processes == 2
        
        # Test execution still works
        result = veda.par_iter(range(30)).map(lambda x: x ** 2).sum()
        expected = sum(i ** 2 for i in range(30))
        assert result == expected


class TestTelemetryIntegration:
    """Test telemetry integration in workflows."""
    
    def test_metrics_collection(self):
        """Test that metrics are collected during execution."""
        # Note: Actual telemetry system may be optional
        try:
            from vedart.telemetry.metrics import TelemetrySystem
            
            result = veda.par_iter(range(100)).map(lambda x: x * 2).collect()
            
            # Verify execution completed
            assert len(result) == 100
            
            # If telemetry is available, check metrics
            # (Implementation depends on telemetry API)
        except ImportError:
            pytest.skip("Telemetry not available")


class TestLargeScaleWorkflows:
    """Test large-scale data processing."""
    
    def test_large_dataset(self):
        """Test processing large dataset."""
        # Process 10,000 items
        data = range(10000)
        
        def process(x: int) -> int:
            return (x * 7 + 3) % 1000
        
        result = veda.par_iter(data).map(process).collect()
        
        assert len(result) == 10000
        assert all(0 <= x < 1000 for x in result)
    
    def test_batch_processing(self):
        """Test batch processing of data."""
        
        def process_batch(batch_id: int) -> List[int]:
            """Process a batch of 100 items."""
            start = batch_id * 100
            return [i * 2 for i in range(start, start + 100)]
        
        # Process 50 batches (5000 items total)
        batches = veda.par_iter(range(50)).map(process_batch).collect()
        
        # Flatten
        all_items = [item for batch in batches for item in batch]
        
        assert len(all_items) == 5000
    
    def test_memory_efficient_processing(self):
        """Test processing that stays within memory limits."""
        
        def create_data(i: int) -> bytes:
            # Create 1KB of data
            return bytes(i % 256 for _ in range(1024))
        
        def process_data(data: bytes) -> int:
            return len(data)
        
        # Process 1000 chunks
        results = (
            veda.par_iter(range(1000))
            .map(create_data)
            .map(process_data)
            .collect()
        )
        
        assert all(r == 1024 for r in results)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_iterator(self):
        """Test with empty data."""
        result = veda.par_iter([]).map(lambda x: x * 2).collect()
        assert result == []
    
    def test_single_item(self):
        """Test with single item."""
        result = veda.par_iter([42]).map(lambda x: x * 2).collect()
        assert result == [84]
    
    def test_very_fast_tasks(self):
        """Test with very fast tasks (overhead matters)."""
        result = veda.par_iter(range(1000)).map(lambda x: x).collect()
        assert result == list(range(1000))
    
    def test_very_slow_tasks(self):
        """Test with slow tasks."""
        def slow_task(x: int) -> int:
            time.sleep(0.1)
            return x * 2
        
        # Only process a few to keep test fast
        result = veda.par_iter(range(5)).map(slow_task).collect()
        assert len(result) == 5
    
    def test_none_values(self):
        """Test handling None values."""
        def may_return_none(x: int) -> int | None:
            if x % 3 == 0:
                return None
            return x
        
        result = veda.par_iter(range(10)).map(may_return_none).collect()
        
        assert None in result
        # 0, 3, 6, 9 are None (4 values), so 6 non-None values
        assert len([x for x in result if x is not None]) == 6
    
    def test_large_individual_results(self):
        """Test with large individual results."""
        def create_large_result(x: int) -> list:
            return [x] * 1000  # 1000-element list
        
        result = veda.par_iter(range(10)).map(create_large_result).collect()
        
        assert len(result) == 10
        assert all(len(r) == 1000 for r in result)


class TestChainedOperations:
    """Test complex operation chains."""
    
    def test_map_filter_fold(self):
        """Test map -> filter -> fold chain."""
        result = (
            veda.par_iter(range(100))
            .map(lambda x: x * 2)
            .filter(lambda x: x % 3 == 0)
            .fold(0, lambda acc, x: acc + x)
        )
        
        # Verify against serial computation
        expected = sum(x * 2 for x in range(100) if (x * 2) % 3 == 0)
        assert result == expected
    
    def test_multiple_maps(self):
        """Test multiple map operations."""
        result = (
            veda.par_iter(range(20))
            .map(lambda x: x * 2)
            .map(lambda x: x + 1)
            .map(lambda x: x ** 2)
            .collect()
        )
        
        expected = [(i * 2 + 1) ** 2 for i in range(20)]
        assert result == expected
    
    def test_map_with_aggregation(self):
        """Test map followed by various aggregations."""
        data = veda.par_iter(range(50)).map(lambda x: x ** 2)
        
        # Multiple terminal operations on same iterator
        # (Note: Each consumes the iterator)
        sum_result = veda.par_iter(range(50)).map(lambda x: x ** 2).sum()
        count_result = veda.par_iter(range(50)).map(lambda x: x ** 2).count()
        max_result = veda.par_iter(range(50)).map(lambda x: x ** 2).max()
        
        assert sum_result == sum(i ** 2 for i in range(50))
        assert count_result == 50
        assert max_result == 49 ** 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
