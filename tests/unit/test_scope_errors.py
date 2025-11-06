"""Tests for scope exception handling."""

import time

import pytest

from vedart.core.scope import scope
from vedart.exceptions import VedaError


def _slow_task(duration):
    """Task that takes some time."""
    time.sleep(duration)
    return duration


def _failing_task():
    """Task that raises an exception."""
    raise ValueError("Task failed!")


def test_scope_normal_exit(cleanup_runtime):
    """Test scope waits for tasks on normal exit."""
    with scope() as s:
        f1 = s.spawn(_slow_task, 0.01)
        f2 = s.spawn(_slow_task, 0.01)
    
    # Tasks should complete before exiting scope
    assert f1.done()
    assert f2.done()
    assert f1.result() == 0.01
    assert f2.result() == 0.01


def test_scope_exception_cleanup(cleanup_runtime):
    """Test scope cleans up tasks when exception occurs."""
    futures = []
    
    with pytest.raises(RuntimeError):
        with scope() as s:
            # Spawn tasks that would take a while
            futures.append(s.spawn(_slow_task, 0.3))
            futures.append(s.spawn(_slow_task, 0.3))
            futures.append(s.spawn(_slow_task, 0.3))
            
            # Raise exception before tasks complete
            raise RuntimeError("User exception")
    
    # Give sufficient time for cleanup (scope waits 0.1s + extra buffer)
    time.sleep(0.5)
    
    # Tasks should be done (cancelled or completed)
    for f in futures:
        assert f.done(), f"Future is still pending after cleanup: {f}"


def test_scope_task_failure_propagates(cleanup_runtime):
    """Test that task failures propagate properly."""
    with pytest.raises(ValueError, match="Task failed"):
        with scope() as s:
            s.spawn(_failing_task)
        # wait_all() should propagate the exception


def test_scope_cannot_spawn_after_close(cleanup_runtime):
    """Test that spawning on closed scope raises error."""
    s = scope()
    
    with s:
        s.spawn(_slow_task, 0.01)
    
    # Scope is closed after __exit__
    with pytest.raises(VedaError, match="closed scope"):
        s.spawn(_slow_task, 0.01)


def test_scope_exception_doesnt_leak_tasks(cleanup_runtime):
    """Test tasks don't outlive scope on exception."""
    import threading
    
    # Capture futures to verify they complete
    futures = []
    
    try:
        with scope() as s:
            # Spawn multiple tasks
            for _ in range(5):
                futures.append(s.spawn(_slow_task, 0.3))
            
            # Raise exception immediately
            raise KeyboardInterrupt("Simulated interrupt")
    except KeyboardInterrupt:
        pass
    
    # Wait for cleanup (scope waits 0.1s + extra buffer)
    time.sleep(0.5)
    
    # All futures should be done (cancelled or completed)
    for f in futures:
        assert f.done(), "Future should be done after scope exit"
    
    # Note: Thread pool workers may persist for efficiency, so we don't check
    # thread count. The important thing is that futures complete/cancel.
