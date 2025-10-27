"""System utilities for resource monitoring."""

import os
from typing import Tuple

import psutil


def get_cpu_count(logical: bool = True) -> int:
    """Get CPU count.
    
    Args:
        logical: If True, return logical cores (with hyperthreading)
                If False, return physical cores only
    
    Returns:
        Number of CPUs
    """
    if logical:
        return os.cpu_count() or 4
    else:
        return psutil.cpu_count(logical=False) or 4


def get_memory_info() -> Tuple[int, int, float]:
    """Get memory information.
    
    Returns:
        Tuple of (used_bytes, total_bytes, percent_used)
    """
    mem = psutil.virtual_memory()
    return (mem.used, mem.total, mem.percent)


def get_cpu_percent(interval: float = 0.0) -> float:
    """Get CPU utilization percentage.
    
    Args:
        interval: Measurement interval in seconds (0 = instant)
    
    Returns:
        CPU usage percentage
    """
    return psutil.cpu_percent(interval=interval)
