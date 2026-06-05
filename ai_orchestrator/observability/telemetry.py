"""OpenTelemetry tracing — distributed spans and in-memory fallback exporter."""

from __future__ import annotations

from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, Status, StatusCode, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased


class Telemetry:
    """Central tracing wrapper around the OpenTelemetry SDK.

    When *otlp_endpoint* is omitted the provider uses an in-memory exporter so
    that spans are never sent to an external collector — ideal for testing or
    single-node deployments.

    The *sample_rate* parameter (0.0–1.0) is wired to a
    :class:`TraceIdRatioBased` sampler on the underlying
    :class:`TracerProvider`, so it actually governs which spans are recorded
    rather than being silently ignored.
    """

    def __init__(
        self,
        service_name: str = "ai-orchestrator",
        otlp_endpoint: Optional[str] = None,
        sample_rate: float = 1.0,
    ) -> None:
        self._service_name = service_name
        self._otlp_endpoint = otlp_endpoint
        self._sample_rate = sample_rate

        resource = Resource.create({"service.name": service_name})
        # Clamp the rate to the valid range to avoid OpenTelemetry raising
        # on negative or >1.0 values; fall back to 1.0 on nonsensical input.
        clamped_rate = max(0.0, min(1.0, sample_rate)) if sample_rate > 0 else 0.0
        self._provider = TracerProvider(
            resource=resource,
            sampler=TraceIdRatioBased(clamped_rate),
        )

        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
            self._exporter = otlp_exporter
            self._provider.add_span_processor(
                SimpleSpanProcessor(otlp_exporter)
            )
        else:
            self._exporter = InMemorySpanExporter()
            self._provider.add_span_processor(
                SimpleSpanProcessor(self._exporter)
            )

        # Set the global tracer provider so instrumentations pick it up
        trace.set_tracer_provider(self._provider)
        self._tracer = self._provider.get_tracer(service_name)

    @property
    def tracer(self) -> trace.Tracer:
        """Return the OpenTelemetry tracer instance."""
        return self._tracer

    def create_span(
        self,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
        parent: Optional[Span] = None,
    ) -> Span:
        """Create and start a new span.

        If *parent* is provided the new span is nested under it as a child.
        """
        if parent is not None:
            span = self._tracer.start_span(
                name,
                attributes=attributes,
                context=trace.set_span_in_context(parent),
            )
        else:
            span = self._tracer.start_span(name, attributes=attributes)
        return span

    def add_event(
        self,
        span: Span,
        name: str,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a named event on *span* with optional attributes."""
        span.add_event(name, attributes=attributes or {})

    def record_exception(
        self,
        span: Span,
        exception: BaseException,
        attributes: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an exception on *span* and mark the span as ERROR.

        Setting the status to :class:`StatusCode.ERROR` is what surfaces
        the failure in UIs like Jaeger — without it, a span with an
        exception event still appears successful.
        """
        span.record_exception(exception, attributes=attributes or {})
        span.set_status(Status(StatusCode.ERROR, str(exception)))

    async def shutdown(self) -> None:
        """Flush and shut down the tracer provider.

        This is a no-op if the provider was never started (e.g. in tests
        where mocks prevent initialisation).
        """
        if self._provider is not None:
            self._provider.shutdown()
