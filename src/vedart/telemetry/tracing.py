"""Distributed tracing for task execution."""

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents a trace span for a task or operation.

    Tracks timing, metadata, and relationships between operations.
    """

    name: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None

    def finish(self, error: Exception | None = None) -> None:
        """Mark span as finished.

        Args:
            error: Optional error that occurred
        """
        self.end_time = time.perf_counter()
        if error:
            self.status = "error"
            self.error = str(error)

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "status": self.status,
            "error": self.error,
        }


class TracingSystem:
    """Manages distributed tracing for VedaRT tasks.

    Provides span creation, context propagation, and export.
    """

    def __init__(self) -> None:
        """Initialize tracing system."""
        self._spans: list[Span] = []
        self._active_spans: dict[str, Span] = {}
        self._enabled = True
        logger.info("Tracing system initialized")

    @contextmanager
    def span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Create a trace span context.

        Args:
            name: Span name
            trace_id: Trace ID (auto-generated if None)
            parent_span_id: Parent span ID if nested
            attributes: Additional attributes

        Yields:
            Span instance
        """
        if not self._enabled:
            # Create dummy span when disabled
            dummy = Span(name=name, trace_id="", span_id="")
            yield dummy
            return

        import uuid

        span_id = str(uuid.uuid4())
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        span = Span(
            name=name,
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            attributes=attributes or {},
        )

        self._active_spans[span_id] = span

        try:
            yield span
        except Exception as e:
            span.finish(error=e)
            raise
        else:
            span.finish()
        finally:
            self._active_spans.pop(span_id, None)
            self._spans.append(span)

    def get_spans(self) -> list[Span]:
        """Get all recorded spans.

        Returns:
            List of spans
        """
        return self._spans.copy()

    def clear(self) -> None:
        """Clear all recorded spans."""
        self._spans.clear()
        self._active_spans.clear()

    def export_jaeger(self) -> list[dict[str, Any]]:
        """Export spans in Jaeger format.

        Returns:
            List of span dictionaries
        """
        # Simplified Jaeger format
        return [span.to_dict() for span in self._spans]

    def export_otlp(self) -> dict[str, Any]:
        """Export spans in OpenTelemetry Protocol format.

        Returns:
            OTLP-compatible dictionary
        """
        # Basic OTLP structure
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": "vedart"}}
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "vedart.tracing"},
                            "spans": [
                                {
                                    "name": span.name,
                                    "traceId": span.trace_id,
                                    "spanId": span.span_id,
                                    "parentSpanId": span.parent_span_id or "",
                                    "startTimeUnixNano": int(span.start_time * 1e9),
                                    "endTimeUnixNano": int(
                                        (span.end_time or span.start_time) * 1e9
                                    ),
                                    "attributes": [
                                        {"key": k, "value": {"stringValue": str(v)}}
                                        for k, v in span.attributes.items()
                                    ],
                                    "status": {"code": 1 if span.status == "ok" else 2},
                                }
                                for span in self._spans
                            ],
                        }
                    ],
                }
            ]
        }

    def enable(self) -> None:
        """Enable tracing."""
        self._enabled = True

    def disable(self) -> None:
        """Disable tracing."""
        self._enabled = False
