"""VedaRT - Parallel runtime for Python.

A unified parallel computing library that brings together threads,
processes, asyncio, and GPU under a single adaptive API.

Example:
    import vedart as veda

    # Parallel iteration
    result = veda.par_iter(range(1000)).map(lambda x: x**2).sum()

    # GPU acceleration
    @veda.gpu
    def matrix_multiply(A, B):
        return A @ B

    # Deterministic debugging
    with veda.deterministic(seed=42):
        result = flaky_computation()
"""

from vedart.__version__ import __version__
from vedart.config import Config, ConfigBuilder, SchedulingPolicy
from vedart.core.runtime import get_runtime, init, shutdown
from vedart.core.scope import Scope, scope, spawn
from vedart.deterministic.replay import deterministic
from vedart.gpu.decorators import gpu, gpu_kernel
from vedart.iter.parallel import ParallelIterator, par_iter

# Telemetry (optional)
try:
    from vedart.telemetry.metrics import TelemetrySystem
except ImportError:
    TelemetrySystem = None  # type: ignore

__all__ = [
    "__version__",
    "Config",
    "ConfigBuilder",
    "SchedulingPolicy",
    "get_runtime",
    "init",
    "shutdown",
    "Scope",
    "scope",
    "spawn",
    "deterministic",
    "gpu",
    "gpu_kernel",
    "ParallelIterator",
    "par_iter",
    "TelemetrySystem",
]

# Configure logging
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
