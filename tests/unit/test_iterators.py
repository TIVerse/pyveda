"""Tests for parallel iterators."""

import pytest

from pyveda.iter.parallel import par_iter


def test_par_iter_map(cleanup_runtime):
    """Test parallel map operation."""
    result = par_iter([1, 2, 3, 4]).map(lambda x: x * 2).collect()
    assert result == [2, 4, 6, 8]


def test_par_iter_filter(cleanup_runtime):
    """Test parallel filter operation."""
    result = par_iter(range(10)).filter(lambda x: x % 2 == 0).collect()
    assert result == [0, 2, 4, 6, 8]


def test_par_iter_map_filter_chain(cleanup_runtime):
    """Test chaining map and filter."""
    result = (
        par_iter(range(10))
        .map(lambda x: x * 2)
        .filter(lambda x: x > 10)
        .collect()
    )
    assert result == [12, 14, 16, 18]


def test_par_iter_sum(cleanup_runtime):
    """Test parallel sum."""
    result = par_iter(range(100)).sum()
    assert result == sum(range(100))


def test_par_iter_count(cleanup_runtime):
    """Test parallel count."""
    result = par_iter(range(50)).filter(lambda x: x % 2 == 0).count()
    assert result == 25


def test_par_iter_fold(cleanup_runtime):
    """Test parallel fold."""
    result = par_iter([1, 2, 3, 4]).fold(0, lambda acc, x: acc + x)
    assert result == 10


def test_par_iter_reduce(cleanup_runtime):
    """Test parallel reduce."""
    result = par_iter([1, 2, 3, 4]).reduce(lambda a, b: a + b)
    assert result == 10


def test_par_iter_empty(cleanup_runtime):
    """Test parallel operations on empty iterable."""
    result = par_iter([]).collect()
    assert result == []
    
    result = par_iter([]).sum()
    assert result == 0


def test_par_iter_preserves_order(cleanup_runtime):
    """Test that order is preserved."""
    result = par_iter(range(100)).map(lambda x: x).collect()
    assert result == list(range(100))


def test_par_iter_to_dict(cleanup_runtime):
    """Test conversion to dictionary."""
    result = par_iter([1, 2, 3]).to_dict(lambda x: str(x))
    assert result == {"1": 1, "2": 2, "3": 3}
