"""Telemetry and monitoring example."""

import time

import vedart as veda


def main():
    """Demonstrate telemetry features."""
    print("VedaRT - Telemetry & Monitoring\n")
    
    # Initialize with telemetry enabled
    config = veda.Config.builder().telemetry(True).build()
    veda.init(config)
    
    runtime = veda.get_runtime()
    
    if runtime.telemetry:
        print("Running workload with telemetry...\n")
        
        # Run some parallel work
        for i in range(5):
            result = veda.par_iter(range(1000)).map(lambda x: x * 2).sum()
            time.sleep(0.1)
        
        # Get metrics snapshot
        snapshot = runtime.telemetry.snapshot()
        
        print("Metrics Snapshot:")
        print(f"  Tasks executed: {snapshot.tasks_executed}")
        print(f"  Avg latency: {snapshot.avg_latency_ms:.2f}ms")
        print(f"  P50 latency: {snapshot.p50_latency_ms:.2f}ms")
        print(f"  P99 latency: {snapshot.p99_latency_ms:.2f}ms")
        print(f"  CPU usage: {snapshot.cpu_utilization_percent:.1f}%")
        print(f"  Memory: {snapshot.memory_used_mb:.0f}MB\n")
        
        # Export Prometheus format
        print("Prometheus Format:")
        print(snapshot.export_prometheus()[:300] + "...\n")
        
        # Export JSON format
        print("JSON Format:")
        import json
        print(json.dumps(snapshot.export_json(), indent=2)[:300] + "...")
    else:
        print("Telemetry not enabled")


if __name__ == "__main__":
    main()
    veda.shutdown()
