#!/usr/bin/env python3
"""Verification script for VedaRT gap completion.

This script tests all implemented features to ensure they work correctly.
Run this after gap completion to validate the implementation.
"""

import sys
import time
from typing import List


def test_section(name: str) -> None:
    """Print a test section header."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print('=' * 70)


def success(message: str) -> None:
    """Print success message."""
    print(f"✓ {message}")


def failure(message: str, error: Exception) -> None:
    """Print failure message."""
    print(f"✗ {message}")
    print(f"  Error: {error}")


def main() -> int:
    """Run all verification tests."""
    print("VedaRT Gap Completion Verification")
    print("=" * 70)
    
    failures: List[str] = []
    
    # Test 1: Basic imports
    test_section("Basic Imports")
    try:
        import pyveda as veda
        success("Import pyveda")
    except Exception as e:
        failure("Import pyveda", e)
        failures.append("imports")
        return 1
    
    # Test 2: Configuration
    test_section("Configuration")
    try:
        config = veda.Config.builder()\
            .threads(4)\
            .telemetry(True)\
            .gpu(False)\
            .build()
        success("Config builder API")
        
        veda.init(config)
        success("Runtime initialization")
    except Exception as e:
        failure("Configuration", e)
        failures.append("config")
    
    # Test 3: Parallel Iterator Operations
    test_section("Parallel Iterator Operations")
    try:
        # Basic operations
        result = veda.par_iter(range(100)).map(lambda x: x * 2).sum()
        assert result == sum(range(100)) * 2
        success("map() and sum()")
        
        result = veda.par_iter(range(100)).filter(lambda x: x % 2 == 0).count()
        assert result == 50
        success("filter() and count()")
        
        # New operations
        result = veda.par_iter([1, 2, 3]).flat_map(lambda x: [x, x * 2]).collect()
        assert len(result) == 6
        success("flat_map()")
        
        result = veda.par_iter(range(10)).take(5).collect()
        assert len(result) == 5
        success("take()")
        
        result = veda.par_iter(range(10)).skip(5).collect()
        assert len(result) == 5
        success("skip()")
        
        result = veda.par_iter(range(10)).chunk(3).collect()
        assert len(result) == 4  # 3, 3, 3, 1
        success("chunk()")
        
        result = veda.par_iter([1, 2, 3]).zip([4, 5, 6]).collect()
        assert len(result) == 3
        success("zip()")
        
        result = veda.par_iter([3, 1, 4, 1, 5]).min()
        assert result == 1
        success("min()")
        
        result = veda.par_iter([3, 1, 4, 1, 5]).max()
        assert result == 5
        success("max()")
        
        result = veda.par_iter([1, 2, 3]).any(lambda x: x > 2)
        assert result is True
        success("any()")
        
        result = veda.par_iter([2, 4, 6]).all(lambda x: x % 2 == 0)
        assert result is True
        success("all()")
        
    except Exception as e:
        failure("Parallel Iterator Operations", e)
        failures.append("iterators")
    
    # Test 4: Async Detection and Routing
    test_section("Async Detection and Routing")
    try:
        import asyncio
        
        async def async_func(x):
            await asyncio.sleep(0.001)
            return x * 2
        
        from pyveda.core.task import Task
        task = Task(func=async_func, args=(5,))
        assert task.is_async is True
        success("Async function auto-detection")
        
        # Test async_map
        results = veda.par_iter([1, 2, 3]).async_map(async_func).collect()
        assert results == [2, 4, 6]
        success("async_map() operation")
        
    except Exception as e:
        failure("Async Detection and Routing", e)
        failures.append("async")
    
    # Test 5: Telemetry
    test_section("Telemetry")
    try:
        runtime = veda.get_runtime()
        
        if runtime.telemetry:
            # Run some tasks
            veda.par_iter(range(100)).map(lambda x: x ** 2).collect()
            
            # Get snapshot
            snapshot = runtime.telemetry.snapshot()
            
            assert snapshot.tasks_executed >= 0
            success("Telemetry snapshot()")
            
            # Test exports
            prom_text = snapshot.export_prometheus()
            assert 'pyveda_tasks_executed_total' in prom_text
            success("Prometheus export")
            
            json_data = snapshot.export_json()
            assert 'tasks' in json_data
            success("JSON export")
            
            # New snapshot should have updated values
            assert snapshot.tasks_executed > 0
            success("Task metrics recording")
            
    except Exception as e:
        failure("Telemetry", e)
        failures.append("telemetry")
    
    # Test 6: Deterministic Execution
    test_section("Deterministic Execution")
    try:
        from pyveda.deterministic.replay import deterministic, ExecutionTrace
        
        # Test with recording
        with deterministic(seed=42, record=True) as trace:
            result1 = veda.par_iter(range(10)).map(lambda x: x * 2).sum()
        
        assert trace is not None
        assert len(trace.get_events()) > 0
        success("Deterministic execution with recording")
        
        # Test trace I/O
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            trace_file = f.name
        
        try:
            trace.save_to_file(trace_file)
            success("Trace save_to_file()")
            
            trace2 = ExecutionTrace()
            trace2.load_from_file(trace_file)
            assert len(trace2.get_events()) == len(trace.get_events())
            success("Trace load_from_file()")
        finally:
            os.unlink(trace_file)
        
    except Exception as e:
        failure("Deterministic Execution", e)
        failures.append("deterministic")
    
    # Test 7: GPU Runtime
    test_section("GPU Runtime")
    try:
        runtime = veda.get_runtime()
        
        if runtime.gpu and runtime.gpu.is_available():
            # Test GPU stats
            stats = runtime.gpu.get_memory_stats()
            assert 'used_mb' in stats
            success("GPU memory stats")
            
            util = runtime.gpu.get_utilization()
            assert 0 <= util <= 100
            success("GPU utilization")
            
            print(f"  GPU Backend: {runtime.gpu.backend}")
            print(f"  GPU Devices: {runtime.gpu.device_count}")
        else:
            print("  ⓘ GPU not available (skipping GPU-specific tests)")
            success("GPU runtime initialized (no hardware)")
        
    except Exception as e:
        failure("GPU Runtime", e)
        failures.append("gpu")
    
    # Test 8: Tracing System
    test_section("Tracing System")
    try:
        from pyveda.telemetry import TracingSystem
        
        tracing = TracingSystem()
        
        with tracing.span('test_operation', attributes={'test': 'value'}) as span:
            time.sleep(0.01)
            assert span.name == 'test_operation'
        
        spans = tracing.get_spans()
        assert len(spans) == 1
        assert spans[0].duration_ms > 0
        success("Tracing spans")
        
        # Test exports
        jaeger = tracing.export_jaeger()
        assert len(jaeger) == 1
        success("Jaeger export")
        
        otlp = tracing.export_otlp()
        assert 'resourceSpans' in otlp
        success("OTLP export")
        
    except Exception as e:
        failure("Tracing System", e)
        failures.append("tracing")
    
    # Test 9: Exporters
    test_section("Exporters")
    try:
        from pyveda.telemetry import create_exporter
        
        # Create different exporters
        prom_exporter = create_exporter('prometheus', namespace='test')
        success("PrometheusExporter creation")
        
        json_exporter = create_exporter('json', pretty=True)
        success("JSONExporter creation")
        
        otlp_exporter = create_exporter('otlp')
        success("OpenTelemetryExporter creation")
        
    except Exception as e:
        failure("Exporters", e)
        failures.append("exporters")
    
    # Test 10: Scope Execution
    test_section("Scope Execution")
    try:
        def task_fn(x):
            return x ** 2
        
        with veda.scope() as s:
            futures = [s.spawn(task_fn, i) for i in range(5)]
        
        results = [f.result() for f in futures]
        assert results == [0, 1, 4, 9, 16]
        success("Scoped execution")
        
    except Exception as e:
        failure("Scope Execution", e)
        failures.append("scope")
    
    # Cleanup
    test_section("Cleanup")
    try:
        veda.shutdown()
        success("Runtime shutdown")
    except Exception as e:
        failure("Runtime shutdown", e)
        failures.append("shutdown")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    if not failures:
        print("✓ All tests passed!")
        print("\n🎉 Gap completion verification successful!")
        return 0
    else:
        print(f"✗ {len(failures)} test section(s) failed:")
        for fail in failures:
            print(f"  - {fail}")
        print("\n⚠ Some features may need attention.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
