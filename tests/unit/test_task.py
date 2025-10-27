"""Tests for task management."""

import pytest

from pyveda.core.task import Task, TaskPriority, TaskState


def test_task_creation():
    """Test task creation."""
    task = Task(
        func=lambda x: x * 2,
        args=(5,),
        priority=TaskPriority.HIGH,
    )
    assert task.priority == TaskPriority.HIGH
    assert task.state == TaskState.PENDING
    assert task.args == (5,)


def test_task_execution():
    """Test task execution."""
    task = Task(func=lambda x: x * 2, args=(5,))
    result = task.execute()
    assert result == 10
    assert task.state == TaskState.COMPLETED
    assert task.result == 10


def test_task_execution_failure():
    """Test task execution with error."""
    def failing_func():
        raise ValueError("Test error")
    
    task = Task(func=failing_func)
    with pytest.raises(ValueError):
        task.execute()
    
    assert task.state == TaskState.FAILED
    assert task.error is not None


def test_task_cancellation():
    """Test task cancellation."""
    task = Task(func=lambda: None)
    assert task.cancel() is True
    assert task.state == TaskState.CANCELLED
    
    # Cannot cancel running task
    task2 = Task(func=lambda: None)
    task2.state = TaskState.RUNNING
    assert task2.cancel() is False


def test_task_priority_ordering():
    """Test task priority comparison."""
    low = Task(priority=TaskPriority.LOW)
    high = Task(priority=TaskPriority.HIGH)
    
    # Higher priority should be "less than" for min heap
    assert high < low
