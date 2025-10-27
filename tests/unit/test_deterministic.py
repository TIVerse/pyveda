"""Tests for deterministic execution."""

from vedart.deterministic.replay import deterministic
from vedart.iter.parallel import par_iter


def test_deterministic_mode(cleanup_runtime):
    """Test deterministic mode produces same results."""

    def computation():
        return par_iter(range(100)).map(lambda x: x * 2).sum()

    with deterministic(seed=42):
        result1 = computation()

    with deterministic(seed=42):
        result2 = computation()

    assert result1 == result2


def test_deterministic_different_seeds(cleanup_runtime):
    """Test different seeds may produce different schedules."""

    # Note: Results should still be correct, just potentially different order
    def computation():
        return par_iter(range(100)).map(lambda x: x * 2).sum()

    with deterministic(seed=42):
        result1 = computation()

    with deterministic(seed=99):
        result2 = computation()

    # Results should be same (sum is associative)
    # but scheduling may differ
    assert result1 == result2
