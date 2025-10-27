"""Stress tests for concurrent execution."""

import pytest

from vedart.core.scope import scope
from vedart.iter.parallel import par_iter


@pytest.mark.slow
def test_high_task_count(cleanup_runtime):
    """Test handling many tasks."""
    # Process many small tasks
    result = par_iter(range(10000)).map(lambda x: x * 2).sum()
    expected = sum(range(10000)) * 2
    assert result == expected


@pytest.mark.slow
def test_concurrent_scopes(cleanup_runtime):
    """Test multiple concurrent scopes."""

    def run_scope(n):
        with scope() as s:
            for i in range(10):
                s.spawn(lambda x=i: x * n)
            return s.wait_all()

    # Run multiple scopes concurrently
    with scope() as outer:
        [outer.spawn(run_scope, i) for i in range(10)]
        results = outer.wait_all()

    assert len(results) == 10


@pytest.mark.slow
def test_deep_nesting(cleanup_runtime):
    """Test deeply nested parallel operations."""

    def nested_compute(depth, value):
        if depth == 0:
            return value
        return (
            par_iter([value])
            .map(lambda x: nested_compute(depth - 1, x * 2))
            .collect()[0]
        )

    # Should not deadlock or crash
    result = nested_compute(3, 1)
    assert result == 8  # 1 * 2 * 2 * 2
