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


def test_par_iter_enumerate(cleanup_runtime):
    """Test enumerate returns sequential indices, not IDs."""
    result = par_iter(["a", "b", "c"]).enumerate().collect()
    assert result == [(0, "a"), (1, "b"), (2, "c")]
    
    # Test with larger dataset
    result = par_iter(range(10)).enumerate().collect()
    expected = [(i, i) for i in range(10)]
    assert result == expected


def test_par_iter_chunk_preserves_boundaries(cleanup_runtime):
    """Test chunk() + collect() preserves chunk boundaries."""
    result = par_iter(range(10)).chunk(3).collect()
    assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    
    # Test with exact multiple
    result = par_iter(range(9)).chunk(3).collect()
    assert result == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    
    # Test with larger chunk size
    result = par_iter(range(5)).chunk(10).collect()
    assert result == [[0, 1, 2, 3, 4]]


def test_par_iter_chunk_with_operations(cleanup_runtime):
    """Test chunk() works with operations."""
    result = par_iter(range(10)).map(_double).chunk(3).collect()
    assert result == [[0, 2, 4], [6, 8, 10], [12, 14, 16], [18]]


def test_par_iter_async_map(cleanup_runtime):
    """Test async_map uses async executor properly."""
    import asyncio
    
    async def async_double(x):
        """Async function that doubles a value."""
        await asyncio.sleep(0.001)  # Small delay
        return x * 2
    
    # async_map should handle async functions
    result = par_iter([1, 2, 3, 4, 5]).async_map(async_double).collect()
    assert result == [2, 4, 6, 8, 10]
    
    # Test with larger dataset to ensure no per-item loop creation
    result = par_iter(range(20)).async_map(async_double).collect()
    assert result == [i * 2 for i in range(20)]


def test_par_iter_async_map_with_regular_function(cleanup_runtime):
    """Test async_map converts regular functions to async."""
    # Regular function should be wrapped as async
    result = par_iter([1, 2, 3]).async_map(_double).collect()
    assert result == [2, 4, 6]
