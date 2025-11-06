"""Tests for async_map functionality."""

import asyncio

import pytest

from vedart.iter.parallel import par_iter


async def _async_double(x):
    """Async function for testing."""
    await asyncio.sleep(0.001)  # Small delay to simulate async work
    return x * 2


async def _async_slow(x):
    """Slower async function."""
    await asyncio.sleep(0.01)
    return x + 1


def test_async_map_basic(cleanup_runtime):
    """Test async_map with async function."""
    result = par_iter([1, 2, 3, 4]).async_map(_async_double).collect()
    assert result == [2, 4, 6, 8]


def test_async_map_with_many_items(cleanup_runtime):
    """Test async_map scales with many items."""
    # Should use async executor, not create event loops per item
    result = par_iter(range(20)).async_map(_async_double).collect()
    assert result == [x * 2 for x in range(20)]


def test_async_map_preserves_order(cleanup_runtime):
    """Test async_map preserves ordering."""
    result = par_iter(range(10)).async_map(_async_slow).collect()
    assert result == [x + 1 for x in range(10)]


def test_async_map_with_sync_function(cleanup_runtime):
    """Test async_map converts sync functions to async."""
    def sync_func(x):
        return x * 3
    
    result = par_iter([1, 2, 3]).async_map(sync_func).collect()
    assert result == [3, 6, 9]
