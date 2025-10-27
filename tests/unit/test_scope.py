"""Tests for scoped execution."""

import pytest

from pyveda.core.scope import scope, spawn


def test_scope_spawn(cleanup_runtime):
    """Test spawning tasks in scope."""
    with scope() as s:
        f1 = s.spawn(lambda: 1 + 1)
        f2 = s.spawn(lambda: 2 + 2)
        results = s.wait_all()
    
    assert results == [2, 4]


def test_scope_spawn_with_args(cleanup_runtime):
    """Test spawning with arguments."""
    def add(a, b):
        return a + b
    
    with scope() as s:
        f1 = s.spawn(add, 1, 2)
        f2 = s.spawn(add, 3, 4)
        results = s.wait_all()
    
    assert results == [3, 7]


def test_scope_exception_handling(cleanup_runtime):
    """Test exception handling in scope."""
    def failing_task():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        with scope() as s:
            s.spawn(failing_task)


def test_spawn_without_scope(cleanup_runtime):
    """Test spawn without scope."""
    future = spawn(lambda: 42)
    result = future.result(timeout=1.0)
    assert result == 42
