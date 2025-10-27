"""Stress tests for high load scenarios.

Tests system behavior under extreme conditions:
- High task counts
- Rapid task submission
- Resource exhaustion scenarios
- Long-running operations
"""

import threading
import time

import pytest

import vedart as veda


@pytest.mark.slow
class TestHighTaskCount:
    """Test with very large numbers of tasks."""

    def test_ten_thousand_tasks(self):
        """Process 10,000 tasks."""
        result = veda.par_iter(range(10000)).map(lambda x: x * 2).sum()
        expected = sum(i * 2 for i in range(10000))
        assert result == expected

    def test_hundred_thousand_fast_tasks(self):
        """Process 100,000 very fast tasks."""
        # Test task spawn overhead at scale
        result = veda.par_iter(range(100000)).map(lambda x: x).count()
        assert result == 100000

    def test_many_small_batches(self):
        """Process many small batches."""

        def process_batch(batch_id: int) -> int:
            return sum(range(batch_id, batch_id + 10))

        result = veda.par_iter(range(5000)).map(process_batch).collect()
        assert len(result) == 5000


@pytest.mark.slow
class TestRapidSubmission:
    """Test rapid task submission."""

    def test_concurrent_submission(self):
        """Test submitting tasks from multiple threads."""
        results = []
        errors = []

        def submit_tasks(start: int, count: int):
            try:
                result = (
                    veda.par_iter(range(start, start + count)).map(lambda x: x**2).sum()
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Submit from multiple threads
        threads = []
        for i in range(10):
            t = threading.Thread(target=submit_tasks, args=(i * 100, 100))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10

    def test_burst_submission(self):
        """Test burst of task submissions."""
        # Submit many tasks in quick succession
        results = []

        for _ in range(100):
            result = veda.par_iter(range(100)).map(lambda x: x).count()
            results.append(result)

        assert all(r == 100 for r in results)


@pytest.mark.slow
class TestResourceLimits:
    """Test behavior at resource limits."""

    def test_high_memory_usage(self):
        """Test with tasks that use significant memory."""

        def create_large_data(x: int) -> bytes:
            # Create 1MB of data
            return bytes(i % 256 for i in range(1024 * 1024))

        def process_large_data(data: bytes) -> int:
            return len(data)

        # Process 100MB total (100 * 1MB)
        # Should handle without OOM
        result = (
            veda.par_iter(range(100))
            .map(create_large_data)
            .map(process_large_data)
            .sum()
        )

        expected = 100 * 1024 * 1024
        assert result == expected

    def test_deep_recursion(self):
        """Test with deep task recursion."""

        def recursive_task(depth: int) -> int:
            if depth == 0:
                return 1

            # Not truly recursive in parallel, but tests deep nesting
            with veda.scope() as s:
                s.spawn(recursive_task, depth - 1)
                results = s.wait_all()

            return results[0] + 1

        # Test recursion depth of 10
        result = recursive_task(10)
        assert result == 11

    def test_queue_saturation(self):
        """Test with saturated task queues."""

        def slow_task(x: int) -> int:
            time.sleep(0.01)
            return x

        # Submit many slow tasks quickly
        # Queue should handle backpressure
        result = veda.par_iter(range(500)).map(slow_task).count()
        assert result == 500


@pytest.mark.slow
class TestLongRunningOperations:
    """Test long-running operations."""

    def test_long_running_tasks(self):
        """Test tasks that run for extended periods."""

        def long_task(x: int) -> int:
            time.sleep(0.5)  # 500ms per task
            return x * 2

        # Process 10 tasks (total ~5 seconds)
        start = time.perf_counter()
        result = veda.par_iter(range(10)).map(long_task).collect()
        duration = time.perf_counter() - start

        assert len(result) == 10
        # Should parallelize (not take 5 seconds)
        assert duration < 3.0  # Allow some overhead

    def test_mixed_duration_tasks(self):
        """Test with highly variable task durations."""

        def variable_task(x: int) -> int:
            # Duration varies from 1ms to 100ms
            time.sleep(0.001 * (x % 100 + 1))
            return x

        result = veda.par_iter(range(100)).map(variable_task).collect()
        assert len(result) == 100


@pytest.mark.slow
class TestFailureScenarios:
    """Test system behavior under failures."""

    def test_high_failure_rate(self):
        """Test with 50% task failure rate."""

        def unreliable_task(x: int) -> int | None:
            if x % 2 == 0:
                return None  # Simulate failure
            return x

        result = veda.par_iter(range(1000)).map(unreliable_task).collect()

        failures = len([r for r in result if r is None])
        successes = len([r for r in result if r is not None])

        assert failures == 500
        assert successes == 500

    def test_cascading_failures(self):
        """Test with failures that depend on previous failures."""
        state = {"failed_count": 0}

        def cascading_task(x: int) -> int | None:
            # Fail more as failures accumulate
            if state["failed_count"] > 5:
                state["failed_count"] += 1
                return None

            if x % 10 == 0:
                state["failed_count"] += 1
                return None

            return x

        result = veda.par_iter(range(100)).map(cascading_task).collect()

        # Some tasks should complete, some fail
        assert None in result
        assert any(r is not None for r in result)

    def test_recovery_after_failures(self):
        """Test system recovery after failures."""

        def task_with_recovery(x: int) -> int:
            try:
                if x == 50:
                    raise ValueError("Controlled failure")
                return x * 2
            except ValueError:
                # Recover with default value
                return -1

        result = veda.par_iter(range(100)).map(task_with_recovery).collect()

        assert len(result) == 100
        assert result[50] == -1  # Recovery value


@pytest.mark.slow
class TestConcurrentOperations:
    """Test concurrent parallel operations."""

    def test_multiple_parallel_operations(self):
        """Run multiple parallel operations simultaneously."""
        results = []

        def run_operation(op_id: int):
            result = veda.par_iter(range(100)).map(lambda x: x * op_id).sum()
            results.append(result)

        threads = []
        for i in range(10):
            t = threading.Thread(target=run_operation, args=(i,))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

        assert len(results) == 10

    def test_nested_parallel_operations(self):
        """Test parallel operations inside parallel operations."""

        def outer_task(x: int) -> int:
            # Inner parallel operation
            inner_result = veda.par_iter(range(10)).map(lambda y: x * y).sum()
            return inner_result

        result = veda.par_iter(range(20)).map(outer_task).collect()

        assert len(result) == 20

    def test_interleaved_operations(self):
        """Test interleaved parallel operations."""
        iter1 = veda.par_iter(range(100))
        iter2 = veda.par_iter(range(100, 200))

        # Interleave operations
        result1 = iter1.map(lambda x: x * 2)
        result2 = iter2.map(lambda x: x * 3)

        final1 = result1.collect()
        final2 = result2.collect()

        assert len(final1) == 100
        assert len(final2) == 100


@pytest.mark.slow
class TestEdgePerformance:
    """Test performance edge cases."""

    def test_tiny_tasks_high_count(self):
        """Test many tiny tasks (overhead dominant)."""
        start = time.perf_counter()
        result = veda.par_iter(range(50000)).map(lambda x: x).count()
        duration = time.perf_counter() - start

        assert result == 50000
        # Should complete reasonably fast despite overhead
        assert duration < 5.0

    def test_unbalanced_workload(self):
        """Test highly unbalanced workload distribution."""

        def unbalanced_task(x: int) -> int:
            # First task is very slow, others are fast
            if x == 0:
                time.sleep(1.0)
            return x

        start = time.perf_counter()
        result = veda.par_iter(range(100)).map(unbalanced_task).collect()
        duration = time.perf_counter() - start

        assert len(result) == 100
        # Should handle unbalanced load
        assert duration < 2.0  # 1s + overhead

    def test_bursty_workload(self):
        """Test bursty workload (alternating fast/slow)."""

        def bursty_task(x: int) -> int:
            if x % 10 == 0:
                time.sleep(0.1)  # Slow task
            return x

        result = veda.par_iter(range(100)).map(bursty_task).collect()
        assert len(result) == 100


@pytest.mark.slow
class TestSystemStability:
    """Test system stability under stress."""

    def test_repeated_initialization(self):
        """Test repeated init/shutdown cycles."""
        for _ in range(10):
            config = veda.Config.thread_only()
            veda.init(config)

            result = veda.par_iter(range(100)).map(lambda x: x).count()
            assert result == 100

            veda.shutdown()

    def test_sustained_load(self):
        """Test sustained load over time."""
        # Run for 100 iterations
        for _i in range(100):
            result = veda.par_iter(range(50)).map(lambda x: x * 2).sum()
            expected = sum(x * 2 for x in range(50))
            assert result == expected

    def test_stress_then_normal(self):
        """Test normal operation after stress."""
        # Stress phase
        for _ in range(10):
            veda.par_iter(range(1000)).map(lambda x: x).count()

        # Normal operation
        result = veda.par_iter(range(10)).map(lambda x: x**2).collect()
        expected = [i**2 for i in range(10)]
        assert result == expected


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "slow"])
