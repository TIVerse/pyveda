"""Telemetry exporters for various monitoring systems."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Exporter(ABC):
    """Base class for telemetry exporters."""

    @abstractmethod
    def export(self, snapshot: Any) -> str:
        """Export telemetry snapshot.
        
        Args:
            snapshot: MetricsSnapshot to export
            
        Returns:
            Exported string representation
        """
        pass


class PrometheusExporter(Exporter):
    """Exports metrics in Prometheus text format."""

    def __init__(self, namespace: str = "pyveda") -> None:
        """Initialize Prometheus exporter.
        
        Args:
            namespace: Metric namespace prefix
        """
        self.namespace = namespace

    def export(self, snapshot: Any) -> str:
        """Export in Prometheus format.
        
        Args:
            snapshot: MetricsSnapshot
            
        Returns:
            Prometheus text format
        """
        lines = []
        
        # HELP and TYPE declarations
        lines.append(f'# HELP {self.namespace}_tasks_executed_total Total tasks executed')
        lines.append(f'# TYPE {self.namespace}_tasks_executed_total counter')
        lines.append(f'{self.namespace}_tasks_executed_total {snapshot.tasks_executed}')
        lines.append('')
        
        lines.append(f'# HELP {self.namespace}_tasks_failed_total Total tasks failed')
        lines.append(f'# TYPE {self.namespace}_tasks_failed_total counter')
        lines.append(f'{self.namespace}_tasks_failed_total {snapshot.tasks_failed}')
        lines.append('')
        
        lines.append(f'# HELP {self.namespace}_tasks_pending Current pending tasks')
        lines.append(f'# TYPE {self.namespace}_tasks_pending gauge')
        lines.append(f'{self.namespace}_tasks_pending {snapshot.tasks_pending}')
        lines.append('')
        
        lines.append(f'# HELP {self.namespace}_latency_ms Task latency in milliseconds')
        lines.append(f'# TYPE {self.namespace}_latency_ms summary')
        lines.append(f'{self.namespace}_latency_ms{{quantile="0.5"}} {snapshot.p50_latency_ms}')
        lines.append(f'{self.namespace}_latency_ms{{quantile="0.95"}} {snapshot.p95_latency_ms}')
        lines.append(f'{self.namespace}_latency_ms{{quantile="0.99"}} {snapshot.p99_latency_ms}')
        lines.append('')
        
        lines.append(f'# HELP {self.namespace}_cpu_utilization_percent CPU utilization')
        lines.append(f'# TYPE {self.namespace}_cpu_utilization_percent gauge')
        lines.append(f'{self.namespace}_cpu_utilization_percent {snapshot.cpu_utilization_percent}')
        lines.append('')
        
        lines.append(f'# HELP {self.namespace}_memory_used_mb Memory used in MB')
        lines.append(f'# TYPE {self.namespace}_memory_used_mb gauge')
        lines.append(f'{self.namespace}_memory_used_mb {snapshot.memory_used_mb}')
        lines.append('')
        
        lines.append(f'# HELP {self.namespace}_throughput_tasks_per_sec Task throughput')
        lines.append(f'# TYPE {self.namespace}_throughput_tasks_per_sec gauge')
        lines.append(f'{self.namespace}_throughput_tasks_per_sec {snapshot.throughput_tasks_per_sec}')
        
        return '\n'.join(lines)


class JSONExporter(Exporter):
    """Exports metrics as JSON."""

    def __init__(self, pretty: bool = True) -> None:
        """Initialize JSON exporter.
        
        Args:
            pretty: Whether to pretty-print JSON
        """
        self.pretty = pretty

    def export(self, snapshot: Any) -> str:
        """Export as JSON.
        
        Args:
            snapshot: MetricsSnapshot
            
        Returns:
            JSON string
        """
        data = {
            'timestamp': snapshot.timestamp,
            'tasks': {
                'executed': snapshot.tasks_executed,
                'failed': snapshot.tasks_failed,
                'pending': snapshot.tasks_pending,
            },
            'latency': {
                'avg_ms': snapshot.avg_latency_ms,
                'p50_ms': snapshot.p50_latency_ms,
                'p95_ms': snapshot.p95_latency_ms,
                'p99_ms': snapshot.p99_latency_ms,
            },
            'system': {
                'cpu_percent': snapshot.cpu_utilization_percent,
                'memory_mb': snapshot.memory_used_mb,
                'throughput_tasks_per_sec': snapshot.throughput_tasks_per_sec,
            }
        }
        
        if hasattr(snapshot, 'gpu_utilization_percent'):
            data['gpu'] = {
                'utilization_percent': snapshot.gpu_utilization_percent,
                'memory_used_mb': getattr(snapshot, 'gpu_memory_used_mb', 0.0),
            }
        
        if self.pretty:
            return json.dumps(data, indent=2)
        return json.dumps(data)


class OpenTelemetryExporter(Exporter):
    """Exports metrics in OpenTelemetry format."""

    def __init__(self, endpoint: Optional[str] = None) -> None:
        """Initialize OpenTelemetry exporter.
        
        Args:
            endpoint: Optional OTLP endpoint URL
        """
        self.endpoint = endpoint

    def export(self, snapshot: Any) -> str:
        """Export in OTLP format.
        
        Args:
            snapshot: MetricsSnapshot
            
        Returns:
            OTLP JSON representation
        """
        # Basic OTLP metrics format
        otlp_data = {
            'resourceMetrics': [{
                'resource': {
                    'attributes': [
                        {'key': 'service.name', 'value': {'stringValue': 'pyveda'}}
                    ]
                },
                'scopeMetrics': [{
                    'scope': {'name': 'pyveda.telemetry'},
                    'metrics': [
                        {
                            'name': 'tasks.executed',
                            'unit': 'tasks',
                            'sum': {
                                'dataPoints': [{
                                    'asInt': str(snapshot.tasks_executed),
                                    'timeUnixNano': str(int(snapshot.timestamp * 1e9))
                                }],
                                'aggregationTemporality': 2,  # Cumulative
                                'isMonotonic': True
                            }
                        },
                        {
                            'name': 'latency.avg',
                            'unit': 'ms',
                            'gauge': {
                                'dataPoints': [{
                                    'asDouble': snapshot.avg_latency_ms,
                                    'timeUnixNano': str(int(snapshot.timestamp * 1e9))
                                }]
                            }
                        },
                        {
                            'name': 'cpu.utilization',
                            'unit': 'percent',
                            'gauge': {
                                'dataPoints': [{
                                    'asDouble': snapshot.cpu_utilization_percent,
                                    'timeUnixNano': str(int(snapshot.timestamp * 1e9))
                                }]
                            }
                        }
                    ]
                }]
            }]
        }
        
        return json.dumps(otlp_data, indent=2)


def create_exporter(format: str = "json", **kwargs: Any) -> Exporter:
    """Create an exporter instance.
    
    Args:
        format: Export format ('json', 'prometheus', 'otlp')
        **kwargs: Format-specific arguments
        
    Returns:
        Exporter instance
        
    Raises:
        ValueError: If format is unknown
    """
    if format == "json":
        return JSONExporter(**kwargs)
    elif format == "prometheus":
        return PrometheusExporter(**kwargs)
    elif format == "otlp":
        return OpenTelemetryExporter(**kwargs)
    else:
        raise ValueError(f"Unknown exporter format: {format}")
