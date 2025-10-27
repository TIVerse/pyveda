"""Tests for configuration management."""

import pytest

from vedart.config import Config, SchedulingPolicy


def test_config_default():
    """Test default configuration."""
    config = Config.default()
    assert config.num_threads is None
    assert config.scheduling_policy == SchedulingPolicy.ADAPTIVE
    assert config.enable_telemetry is True


def test_config_builder():
    """Test fluent configuration builder."""
    config = Config.builder().threads(4).processes(2).gpu(True).telemetry(False).build()
    assert config.num_threads == 4
    assert config.num_processes == 2
    assert config.enable_gpu is True
    assert config.enable_telemetry is False


def test_config_deterministic():
    """Test deterministic configuration."""
    config = Config.builder().deterministic(seed=42).build()
    assert config.deterministic_seed == 42
    assert config.scheduling_policy == SchedulingPolicy.DETERMINISTIC


def test_config_validation():
    """Test configuration validation."""
    with pytest.raises(ValueError):
        Config(num_threads=0)

    with pytest.raises(ValueError):
        Config(adaptive_interval_ms=5)

    with pytest.raises(ValueError):
        Config(min_workers=0)


def test_config_worker_limits():
    """Test worker limit configuration."""
    config = Config.builder().worker_limits(min_workers=2, max_workers=8).build()
    assert config.min_workers == 2
    assert config.max_workers == 8
