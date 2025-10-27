"""Tests for error handling and failure policies."""

import time

import pytest

import vedart as veda


class TestFailurePolicies:
    """Test different failure handling strategies."""

    def test_fail_fast_on_error(self):
        """Test that execution stops on first error in fail-fast mode."""

        def failing_task(x: int) -> int:
            if x == 5:
                raise ValueError("Task 5 failed")
            return x * 2

        with pytest.raises(ValueError, match="Task 5 failed"):
            veda.par_iter(range(10)).map(failing_task).collect()

    def test_continue_on_error_with_wrapper(self):
        """Test continuing execution with error wrapper."""

        def sometimes_fails(x: int) -> int:
            if x % 3 == 0 and x > 0:
                raise ValueError(f"Failed on {x}")
            return x**2

        def safe_task(x: int) -> int | None:
            try:
                return sometimes_fails(x)
            except ValueError:
                return None

        result = veda.par_iter(range(10)).map(safe_task).collect()

        # Check that we got some successes and some None values
        assert None in result
        assert any(x is not None for x in result)

    def test_error_in_all_tasks(self):
        """Test handling when all tasks fail."""

        def always_fails(x: int) -> int:
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError):
            veda.par_iter(range(5)).map(always_fails).collect()


class TestErrorAggregation:
    """Test error collection and aggregation."""

    def test_collect_multiple_error_types(self):
        """Test collecting different error types."""

        def multi_error_task(x: int) -> int:
            if x % 3 == 0:
                raise ValueError("Value error")
            elif x % 5 == 0:
                raise TypeError("Type error")
            return x

        errors = []

        def error_catching_wrapper(x: int) -> int | None:
            try:
                return multi_error_task(x)
            except Exception as e:
                errors.append((x, type(e).__name__))
                return None

        veda.par_iter(range(15)).map(error_catching_wrapper).collect()

        # Verify we caught errors
        assert len(errors) > 0
        assert any(err[1] == "ValueError" for err in errors)
        assert any(err[1] == "TypeError" for err in errors)

    def test_error_context_preservation(self):
        """Test that error context is preserved."""

        def task_with_context(x: int) -> int:
            if x == 7:
                raise ValueError(f"Failed at value {x}")
            return x

        with pytest.raises(ValueError) as exc_info:
            veda.par_iter(range(10)).map(task_with_context).collect()

        assert "7" in str(exc_info.value)


class TestRetryLogic:
    """Test retry functionality."""

    def test_retry_succeeds_eventually(self):
        """Test that retry eventually succeeds."""
        attempts = {}

        def flaky_task(x: int) -> int:
            if x not in attempts:
                attempts[x] = 0
            attempts[x] += 1

            # Succeed on 2nd attempt
            if attempts[x] < 2:
                raise ConnectionError("Temporary failure")
            return x * 2

        def retry_wrapper(x: int, max_retries: int = 3) -> int:
            for attempt in range(max_retries):
                try:
                    return flaky_task(x)
                except ConnectionError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.01)
            return 0  # Unreachable

        result = veda.par_iter(range(5)).map(retry_wrapper).collect()

        assert len(result) == 5
        assert all(r == i * 2 for i, r in enumerate(result))

    def test_retry_exhausted(self):
        """Test that retry eventually gives up."""

        def always_fails(x: int) -> int:
            raise ConnectionError("Permanent failure")

        def retry_wrapper(x: int, max_retries: int = 2) -> int | None:
            for attempt in range(max_retries):
                try:
                    return always_fails(x)
                except ConnectionError:
                    if attempt == max_retries - 1:
                        return None
            return None

        result = veda.par_iter(range(3)).map(retry_wrapper).collect()

        # All should fail after retries
        assert all(r is None for r in result)


class TestPartialResults:
    """Test partial result recovery."""

    def test_recover_partial_batch_results(self):
        """Test recovering partial results from batch."""

        def batch_processor(batch_id: int) -> list[int]:
            results = []
            for i in range(5):
                item = batch_id * 5 + i
                if item == 7:  # Simulate one failure
                    continue
                results.append(item)
            return results

        batches = veda.par_iter(range(4)).map(batch_processor).collect()

        # Flatten
        all_results = [item for batch in batches for item in batch]

        # Should have all items except 7
        expected_count = 20 - 1  # 4 batches * 5 items - 1 failure
        assert len(all_results) == expected_count
        assert 7 not in all_results

    def test_partial_results_with_mixed_success(self):
        """Test handling mixed success/failure scenarios."""

        def task(x: int) -> tuple[int, bool]:
            success = x % 2 == 0
            return (x if success else -1, success)

        results = veda.par_iter(range(10)).map(task).collect()

        successes = [r[0] for r in results if r[1]]
        failures = [r for r in results if not r[1]]

        assert len(successes) == 5  # Even numbers
        assert len(failures) == 5  # Odd numbers


class TestGracefulDegradation:
    """Test graceful degradation patterns."""

    def test_fallback_on_primary_failure(self):
        """Test automatic fallback to secondary service."""

        def primary_service(x: int) -> str:
            if x % 3 == 0:
                raise ConnectionError("Primary unavailable")
            return f"primary-{x}"

        def fallback_service(x: int) -> str:
            return f"fallback-{x}"

        def resilient_task(x: int) -> str:
            try:
                return primary_service(x)
            except ConnectionError:
                return fallback_service(x)

        results = veda.par_iter(range(9)).map(resilient_task).collect()

        # All should succeed (some via fallback)
        assert len(results) == 9
        assert any("primary" in r for r in results)
        assert any("fallback" in r for r in results)

    def test_quality_degradation(self):
        """Test degrading quality under load."""

        def high_quality_task(x: int) -> tuple[int, str]:
            # Simulates expensive operation
            time.sleep(0.01)
            return (x * 100, "high")

        def low_quality_task(x: int) -> tuple[int, str]:
            # Fast but lower quality
            return (x * 10, "low")

        # In practice, would check system load
        use_high_quality = False  # Simulate high load

        def adaptive_task(x: int) -> tuple[int, str]:
            if use_high_quality:
                return high_quality_task(x)
            else:
                return low_quality_task(x)

        results = veda.par_iter(range(5)).map(adaptive_task).collect()

        # All should use low quality due to "load"
        assert all(r[1] == "low" for r in results)


class TestScopeErrorHandling:
    """Test error handling in scoped execution."""

    def test_scope_error_propagation(self):
        """Test that errors in scope are propagated."""

        def failing_task(x: int) -> int:
            if x == 2:
                raise ValueError("Task 2 failed")
            return x * 2

        with pytest.raises(ValueError, match="Task 2 failed"):
            with veda.scope() as s:
                [s.spawn(failing_task, i) for i in range(5)]
                s.wait_all()

    def test_scope_partial_completion(self):
        """Test handling partial completion in scope."""

        def task(x: int) -> int:
            if x == 3:
                raise RuntimeError("Failed")
            time.sleep(0.01)
            return x

        with pytest.raises(RuntimeError):
            with veda.scope() as s:
                [s.spawn(task, i) for i in range(5)]
                s.wait_all()


class TestErrorPropagation:
    """Test error propagation through different operations."""

    def test_error_in_map_chain(self):
        """Test error propagation through map chain."""

        def step1(x: int) -> int:
            return x * 2

        def step2(x: int) -> int:
            if x == 10:
                raise ValueError("Error in step2")
            return x + 1

        with pytest.raises(ValueError, match="Error in step2"):
            (veda.par_iter(range(10)).map(step1).map(step2).collect())

    def test_error_in_filter(self):
        """Test error handling in filter operations."""

        def predicate(x: int) -> bool:
            if x == 5:
                raise ValueError("Filter error")
            return x % 2 == 0

        with pytest.raises(ValueError, match="Filter error"):
            veda.par_iter(range(10)).filter(predicate).collect()

    def test_error_in_fold(self):
        """Test error handling in fold/reduce."""

        def reducer(acc: int, x: int) -> int:
            if x == 7:
                raise ValueError("Fold error")
            return acc + x

        with pytest.raises(ValueError, match="Fold error"):
            veda.par_iter(range(10)).fold(0, reducer)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
