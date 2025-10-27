"""Tests for telemetry system."""

import time

import pytest

from pyveda.telemetry.metrics import Counter, Gauge, Histogram, MetricsSnapshot


def test_counter():
    """Test counter metric."""
    counter = Counter("test_counter")
    counter.inc()
    counter.inc(5)
    assert counter.get() == 6
    
    counter.reset()
    assert counter.get() == 0


def test_gauge():
    """Test gauge metric."""
    gauge = Gauge("test_gauge")
    gauge.set(10.0)
    assert gauge.get() == 10.0
    
    gauge.inc(5.0)
    assert gauge.get() == 15.0
    
    gauge.dec(3.0)
    assert gauge.get() == 12.0


def test_histogram():
    """Test histogram metric."""
    hist = Histogram("test_histogram")
    
    for i in range(100):
        hist.observe(float(i))
    
    assert hist.get_count() == 100
    assert 48 <= hist.get_percentile(50) <= 52  # p50 around 50
    assert 97 <= hist.get_percentile(99) <= 100  # p99 around 99


def test_metrics_snapshot_prometheus():
    """Test Prometheus export format."""
    snapshot = MetricsSnapshot(
        timestamp=time.time(),
        tasks_executed=100,
        tasks_pending=5,
        avg_latency_ms=10.5,
        p50_latency_ms=9.0,
        p99_latency_ms=25.0,
        throughput_tasks_per_sec=50.0,
        cpu_utilization_percent=65.0,
        memory_used_mb=512.0,
    )
    
    prom_output = snapshot.export_prometheus()
    
    assert "veda_tasks_executed_total 100" in prom_output
    assert "veda_tasks_pending 5" in prom_output
    assert 'veda_latency_seconds{quantile="0.5"}' in prom_output
    assert "veda_cpu_utilization_percent 65.0" in prom_output


def test_metrics_snapshot_json():
    """Test JSON export."""
    snapshot = MetricsSnapshot(
        timestamp=time.time(),
        tasks_executed=100,
        tasks_pending=5,
        avg_latency_ms=10.5,
        p50_latency_ms=9.0,
        p99_latency_ms=25.0,
        throughput_tasks_per_sec=50.0,
        cpu_utilization_percent=65.0,
        memory_used_mb=512.0,
    )
    
    json_output = snapshot.export_json()
    
    assert json_output["tasks"]["executed"] == 100
    assert json_output["latency_ms"]["avg"] == 10.5
    assert json_output["system"]["cpu_percent"] == 65.0
