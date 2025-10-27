"""Pytest configuration and fixtures."""

import pytest

from pyveda.config import Config
from pyveda.core.runtime import Runtime, shutdown


@pytest.fixture(scope="function")
def runtime():
    """Create a runtime instance for testing."""
    config = Config(
        num_threads=2,
        num_processes=2,
        enable_gpu=False,
        enable_telemetry=False,
    )
    rt = Runtime(config)
    yield rt
    rt.shutdown()


@pytest.fixture(scope="function")
def cleanup_runtime():
    """Cleanup runtime after test."""
    yield
    shutdown()


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return list(range(100))
