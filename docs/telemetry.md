# Telemetry and Monitoring Guide

Complete guide to observability in PyVeda.

## Overview

PyVeda includes a comprehensive telemetry system for monitoring task execution, performance metrics, and system health.

## Enabling Telemetry

```python
import pyveda as veda

# Enable telemetry during initialization
config = veda.Config.builder()\
    .telemetry(True)\
    .build()

veda.init(config)
```

## Basic Usage

### Getting Metrics Snapshot

```python
runtime = veda.get_runtime()

if runtime.telemetry:
    # Get current metrics
    snapshot = runtime.telemetry.snapshot()
    
    print(f"Tasks executed: {snapshot.tasks_executed}")
    print(f"Tasks failed: {snapshot.tasks_failed}")
    print(f"Average latency: {snapshot.avg_latency_ms:.2f}ms")
    print(f"Throughput: {snapshot.throughput_tasks_per_sec:.1f} tasks/sec")
```

## Metrics Available

### Task Metrics

- **tasks_executed** - Total number of tasks completed
- **tasks_failed** - Total number of failed tasks
- **tasks_pending** - Currently queued tasks

### Latency Metrics

- **avg_latency_ms** - Mean task latency
- **p50_latency_ms** - Median latency (50th percentile)
- **p95_latency_ms** - 95th percentile latency
- **p99_latency_ms** - 99th percentile latency

### System Metrics

- **cpu_utilization_percent** - CPU usage (0-100)
- **memory_used_mb** - System memory used in MB
- **throughput_tasks_per_sec** - Task completion rate

### GPU Metrics (when enabled)

- **gpu_utilization_percent** - GPU usage (0-100)
- **gpu_memory_used_mb** - GPU memory used in MB

## Export Formats

### Prometheus

Export metrics for Prometheus scraping:

```python
snapshot = runtime.telemetry.snapshot()
prometheus_text = snapshot.export_prometheus()

# Serve via HTTP endpoint
from http.server import BaseHTTPRequestHandler, HTTPServer

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            snapshot = runtime.telemetry.snapshot()
            metrics = snapshot.export_prometheus()
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(metrics.encode())

# Run server
server = HTTPServer(('0.0.0.0', 9090), MetricsHandler)
server.serve_forever()
```

Example Prometheus output:

```
# HELP pyveda_tasks_executed_total Total tasks executed
# TYPE pyveda_tasks_executed_total counter
pyveda_tasks_executed_total 1523

# HELP pyveda_latency_ms Task latency in milliseconds
# TYPE pyveda_latency_ms summary
pyveda_latency_ms{quantile="0.5"} 12.5
pyveda_latency_ms{quantile="0.95"} 45.2
pyveda_latency_ms{quantile="0.99"} 89.1
```

### JSON

Export as JSON for custom integrations:

```python
snapshot = runtime.telemetry.snapshot()
json_data = snapshot.export_json()

import json
print(json.dumps(json_data, indent=2))
```

Output:

```json
{
  "timestamp": 1699564800.123,
  "tasks": {
    "executed": 1523,
    "failed": 3,
    "pending": 12
  },
  "latency": {
    "avg_ms": 15.4,
    "p50_ms": 12.5,
    "p95_ms": 45.2,
    "p99_ms": 89.1
  },
  "system": {
    "cpu_percent": 65.3,
    "memory_mb": 2048.5,
    "throughput_tasks_per_sec": 127.3
  }
}
```

### OpenTelemetry (OTLP)

Export to OpenTelemetry collectors:

```python
from pyveda.telemetry import create_exporter

# Create OTLP exporter
exporter = create_exporter('otlp', endpoint='http://localhost:4318')

snapshot = runtime.telemetry.snapshot()
otlp_data = exporter.export(snapshot)

# Send to collector
import requests
requests.post(
    'http://localhost:4318/v1/metrics',
    json=otlp_data,
    headers={'Content-Type': 'application/json'}
)
```

## Distributed Tracing

Track task execution with spans:

```python
from pyveda.telemetry import TracingSystem

tracing = TracingSystem()

# Create a traced operation
with tracing.span('data_processing', attributes={'batch_id': 123}) as span:
    # Process data
    data = load_data()
    
    with tracing.span('transform', parent_span_id=span.span_id) as child_span:
        transformed = transform(data)
    
    with tracing.span('save', parent_span_id=span.span_id) as child_span:
        save_results(transformed)

# Export traces
spans = tracing.get_spans()
jaeger_format = tracing.export_jaeger()
otlp_format = tracing.export_otlp()
```

## Custom Metrics

Create custom counters, gauges, and histograms:

```python
from pyveda.telemetry import Counter, Gauge, Histogram

# Counter for custom events
requests_total = Counter('http_requests_total', 'Total HTTP requests')
requests_total.inc()
requests_total.inc(5)

# Gauge for current values
active_connections = Gauge('active_connections', 'Active connections')
active_connections.set(42)
active_connections.inc()
active_connections.dec()

# Histogram for distributions
request_duration = Histogram('request_duration_ms', 'Request duration')
request_duration.observe(12.5)
request_duration.observe(45.2)

# Get statistics
p95 = request_duration.get_percentile(95)
mean = request_duration.get_mean()
```

## Real-Time Monitoring

### Continuous Collection

Telemetry runs in a background thread:

```python
# Start automatic collection (1 second interval)
runtime.telemetry.start()

# Run workload
for i in range(1000):
    veda.spawn(my_task, i)

# Stop collection
runtime.telemetry.stop()

# Get historical snapshots
snapshots = runtime.telemetry.get_snapshots()
for snapshot in snapshots[-10:]:  # Last 10 snapshots
    print(f"Time: {snapshot.timestamp}, Throughput: {snapshot.throughput_tasks_per_sec}")
```

### Live Dashboard

Simple terminal dashboard:

```python
import time
import os

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def display_dashboard():
    runtime = veda.get_runtime()
    
    while True:
        clear_screen()
        snapshot = runtime.telemetry.snapshot()
        
        print("=" * 60)
        print("PyVeda Live Dashboard")
        print("=" * 60)
        print(f"Tasks Executed: {snapshot.tasks_executed}")
        print(f"Tasks Failed:   {snapshot.tasks_failed}")
        print(f"Tasks Pending:  {snapshot.tasks_pending}")
        print(f"Throughput:     {snapshot.throughput_tasks_per_sec:.1f} tasks/sec")
        print(f"Avg Latency:    {snapshot.avg_latency_ms:.2f}ms")
        print(f"P99 Latency:    {snapshot.p99_latency_ms:.2f}ms")
        print(f"CPU Usage:      {snapshot.cpu_utilization_percent:.1f}%")
        print(f"Memory:         {snapshot.memory_used_mb:.0f}MB")
        
        if snapshot.gpu_utilization_percent:
            print(f"GPU Usage:      {snapshot.gpu_utilization_percent:.1f}%")
            print(f"GPU Memory:     {snapshot.gpu_memory_used_mb:.0f}MB")
        
        print("=" * 60)
        print("Press Ctrl+C to exit")
        
        time.sleep(1)

# Run dashboard in separate thread
import threading
dashboard_thread = threading.Thread(target=display_dashboard, daemon=True)
dashboard_thread.start()
```

## Integration Examples

### Grafana Dashboard

1. Set up Prometheus to scrape PyVeda metrics
2. Configure Grafana data source
3. Create dashboard with panels:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'pyveda'
    static_configs:
      - targets: ['localhost:9090']
```

### CloudWatch

Send metrics to AWS CloudWatch:

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

snapshot = runtime.telemetry.snapshot()

cloudwatch.put_metric_data(
    Namespace='PyVeda',
    MetricData=[
        {
            'MetricName': 'TasksExecuted',
            'Value': snapshot.tasks_executed,
            'Unit': 'Count'
        },
        {
            'MetricName': 'Throughput',
            'Value': snapshot.throughput_tasks_per_sec,
            'Unit': 'Count/Second'
        },
        {
            'MetricName': 'Latency',
            'Value': snapshot.avg_latency_ms,
            'Unit': 'Milliseconds'
        }
    ]
)
```

### Datadog

Export to Datadog:

```python
from datadog import initialize, statsd

initialize(statsd_host='localhost', statsd_port=8125)

snapshot = runtime.telemetry.snapshot()

statsd.gauge('pyveda.tasks.executed', snapshot.tasks_executed)
statsd.gauge('pyveda.throughput', snapshot.throughput_tasks_per_sec)
statsd.histogram('pyveda.latency', snapshot.avg_latency_ms)
```

## Performance Impact

Telemetry overhead is minimal:

- **Memory**: ~10 MB for metrics storage
- **CPU**: < 1% for background collection
- **Per-task**: ~1 μs overhead for metric recording

To disable:

```python
config = veda.Config.builder().telemetry(False).build()
veda.init(config)
```

## Best Practices

1. **Enable Telemetry** - Telemetry helps debug issues
2. **Monitor Latency** - Watch P99 for performance degradation
3. **Track Failures** - Alert on increased failure rate
4. **Export Regularly** - Send to external monitoring systems
5. **Set Thresholds** - Alert on anomalies
6. **Retain History** - Keep snapshots for trend analysis

## Alerting Examples

### High Failure Rate

```python
def check_failure_rate():
    snapshot = runtime.telemetry.snapshot()
    
    if snapshot.tasks_executed > 0:
        failure_rate = snapshot.tasks_failed / snapshot.tasks_executed
        
        if failure_rate > 0.05:  # 5% threshold
            send_alert(f"High failure rate: {failure_rate:.1%}")
```

### High Latency

```python
def check_latency():
    snapshot = runtime.telemetry.snapshot()
    
    if snapshot.p99_latency_ms > 100:  # 100ms threshold
        send_alert(f"High P99 latency: {snapshot.p99_latency_ms:.1f}ms")
```

### Resource Exhaustion

```python
def check_resources():
    snapshot = runtime.telemetry.snapshot()
    
    if snapshot.cpu_utilization_percent > 90:
        send_alert(f"High CPU usage: {snapshot.cpu_utilization_percent:.1f}%")
    
    if snapshot.gpu_memory_used_mb and snapshot.gpu_memory_used_mb > 8000:
        send_alert(f"High GPU memory: {snapshot.gpu_memory_used_mb:.0f}MB")
```

## Troubleshooting

### Metrics Not Updating

1. Check telemetry is enabled: `config.enable_telemetry = True`
2. Verify collection started: `runtime.telemetry.start()`
3. Confirm tasks are running

### High Memory Usage

1. Reduce snapshot history: Adjust `_max_snapshots` in TelemetrySystem
2. Clear old snapshots periodically
3. Use sampling for high-frequency metrics

### Missing GPU Metrics

1. Enable GPU: `config.enable_gpu = True`
2. Check GPU availability: `runtime.gpu.is_available()`
3. Verify CUDA installation

## Further Reading

- [Prometheus Documentation](https://prometheus.io/docs/)
- [OpenTelemetry Specification](https://opentelemetry.io/docs/)
- [Grafana Dashboards](https://grafana.com/docs/)
