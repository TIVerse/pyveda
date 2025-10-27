"""Tests for parallel iterators."""

from vedart.iter.parallel import par_iter


# Helper functions (module-level for picklability)
def _double(x):
    return x * 2


def _is_even(x):
    return x % 2 == 0


def _greater_than_10(x):
    return x > 10


def _add(acc, x):
    return acc + x


def _identity(x):
    return x


def _to_string_key(x):
    return str(x)


def test_par_iter_map(cleanup_runtime):
    """Test parallel map operation."""
    result = par_iter([1, 2, 3, 4]).map(_double).collect()
    assert result == [2, 4, 6, 8]


def test_par_iter_filter(cleanup_runtime):
    """Test parallel filter operation."""
    result = par_iter(range(10)).filter(_is_even).collect()
    assert result == [0, 2, 4, 6, 8]


def test_par_iter_map_filter_chain(cleanup_runtime):
    """Test chaining map and filter."""
    result = par_iter(range(10)).map(_double).filter(_greater_than_10).collect()
    assert result == [12, 14, 16, 18]


def test_par_iter_sum(cleanup_runtime):
    """Test parallel sum."""
    result = par_iter(range(100)).sum()
    assert result == sum(range(100))


def test_par_iter_count(cleanup_runtime):
    """Test parallel count."""
    result = par_iter(range(50)).filter(_is_even).count()
    assert result == 25


def test_par_iter_fold(cleanup_runtime):
    """Test parallel fold."""
    result = par_iter([1, 2, 3, 4]).fold(0, _add)
    assert result == 10


def test_par_iter_reduce(cleanup_runtime):
    """Test parallel reduce."""
    result = par_iter([1, 2, 3, 4]).reduce(_add)
    assert result == 10


def test_par_iter_empty(cleanup_runtime):
    """Test parallel operations on empty iterable."""
    result = par_iter([]).collect()
    assert result == []

    result = par_iter([]).sum()
    assert result == 0


def test_par_iter_preserves_order(cleanup_runtime):
    """Test that order is preserved."""
    result = par_iter(range(100)).map(_identity).collect()
    assert result == list(range(100))


def test_par_iter_to_dict(cleanup_runtime):
    """Test conversion to dictionary."""
    result = par_iter([1, 2, 3]).to_dict(_to_string_key)
    assert result == {"1": 1, "2": 2, "3": 3}
