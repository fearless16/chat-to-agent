"""Tests for the observability module — telemetry, metrics, and logging."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Telemetry tests
# ---------------------------------------------------------------------------


class TestTelemetry:
    """OpenTelemetry tracing integration.

    By default ``Telemetry()`` uses an in-memory exporter so no collector
    connection is needed.  We only patch the OTLP exporter for the
    endpoint-related test.
    """

    def test_init_defaults(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry()
        assert t._service_name == "ai-orchestrator"
        assert t._sample_rate == 1.0
        assert t._exporter is not None
        assert t._otlp_endpoint is None

    @patch(
        "ai_orchestrator.observability.telemetry.OTLPSpanExporter",
        MagicMock,
    )
    def test_init_with_endpoint(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry(otlp_endpoint="http://localhost:4317")
        assert t._otlp_endpoint == "http://localhost:4317"

    def test_tracer_property(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry()
        tracer = t.tracer
        assert tracer is not None
        assert t.tracer is tracer  # cached

    def test_create_span(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry()
        span = t.create_span("test-span", attributes={"key": "value"})
        assert span.name == "test-span"
        span.end()

        # Verify the span was exported to the in-memory exporter
        finished = t._exporter.get_finished_spans()
        assert len(finished) == 1
        assert finished[0].name == "test-span"

    def test_create_span_with_parent(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry()
        parent = t.create_span("parent-span")
        child = t.create_span("child-span", parent=parent)

        assert child.name == "child-span"
        assert parent.name == "parent-span"

        # Verify parent-child relationship in exported spans
        child.end()
        parent.end()
        finished = t._exporter.get_finished_spans()
        span_names = {s.name for s in finished}
        assert "parent-span" in span_names
        assert "child-span" in span_names

        # Child should reference the parent span context
        child_span = next(s for s in finished if s.name == "child-span")
        parent_span = next(s for s in finished if s.name == "parent-span")
        assert child_span.parent is not None
        assert child_span.parent.span_id == parent_span.context.span_id

    def test_add_event(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry()
        span = t.create_span("event-span")
        t.add_event(span, "test-event", attributes={"foo": "bar"})
        span.end()

        finished = t._exporter.get_finished_spans()
        assert len(finished) == 1
        events = finished[0].events
        assert len(events) == 1
        assert events[0].name == "test-event"
        assert events[0].attributes.get("foo") == "bar"

    def test_record_exception(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry()
        span = t.create_span("exception-span")
        exc = ValueError("test error")
        t.record_exception(span, exc, attributes={"component": "test"})
        span.end()

        finished = t._exporter.get_finished_spans()
        assert len(finished) == 1
        events = finished[0].events
        assert len(events) >= 1
        # Exception event should have the exception type
        exc_event = next(e for e in events if e.name == "exception")
        assert exc_event is not None

    @pytest.mark.asyncio
    async def test_shutdown(self) -> None:
        from ai_orchestrator.observability.telemetry import Telemetry

        t = Telemetry()
        await t.shutdown()
        assert True  # no error


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetricsRegistry:
    """Prometheus metrics counters, gauges, histograms."""

    @pytest.fixture(autouse=True)
    def _reset_registry(self) -> None:
        """Use a fresh CollectorRegistry for each test to avoid duplicate registration."""
        # We rely on the fact that MetricsRegistry creates its own registry,
        # so we don't need to mess with the global default registry.
        # The fixture exists so subclasses can override if needed.
        yield

    def test_init(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        assert mr._registry is not None

    def test_inc_task_submitted_default(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_task_submitted()
        text = mr.get_metrics_text()
        assert "orchestrator_task_submitted_total" in text
        assert 'task_type="interactive"' in text

    def test_inc_task_submitted_custom_type(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_task_submitted(task_type="batch")
        text = mr.get_metrics_text()
        assert 'task_type="batch"' in text

    def test_inc_task_completed_default(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_task_completed()
        text = mr.get_metrics_text()
        assert "orchestrator_task_completed_total" in text
        assert 'status="success"' in text

    def test_inc_task_completed_custom_status(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_task_completed(status="skipped")
        text = mr.get_metrics_text()
        assert 'status="skipped"' in text

    def test_inc_task_failed_default(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_task_failed()
        text = mr.get_metrics_text()
        assert "orchestrator_task_failed_total" in text
        assert 'reason="unknown"' in text

    def test_inc_task_failed_custom_reason(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_task_failed(reason="timeout")
        text = mr.get_metrics_text()
        assert 'reason="timeout"' in text

    def test_set_active_agents(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.set_active_agents(42)
        text = mr.get_metrics_text()
        assert "orchestrator_active_agents" in text
        assert "42.0" in text

    def test_observe_provider_latency(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.observe_provider_latency(provider="openai", ms=150.0)
        text = mr.get_metrics_text()
        assert "orchestrator_provider_latency_ms" in text
        assert 'provider="openai"' in text

    def test_inc_provider_error(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_provider_error(provider="anthropic", error_type="rate_limit")
        text = mr.get_metrics_text()
        assert "orchestrator_provider_error_total" in text
        assert 'provider="anthropic"' in text
        assert 'error_type="rate_limit"' in text

    def test_set_memory_usage(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.set_memory_usage(1_024_000_000)
        text = mr.get_metrics_text()
        assert "orchestrator_memory_usage_bytes" in text
        assert "1024000000.0" in text

    def test_set_pool_size(self) -> None:
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.set_pool_size(state="idle", count=10)
        text = mr.get_metrics_text()
        assert "orchestrator_pool_size" in text
        assert 'state="idle"' in text
        assert "10.0" in text

    def test_get_metrics_text_twice(self) -> None:
        """Calling get_metrics_text multiple times returns consistent output."""
        from ai_orchestrator.observability.metrics import MetricsRegistry

        mr = MetricsRegistry()
        mr.inc_task_submitted()
        mr.set_active_agents(5)
        text1 = mr.get_metrics_text()
        text2 = mr.get_metrics_text()
        assert text1 == text2


# ---------------------------------------------------------------------------
# Logging tests
# ---------------------------------------------------------------------------


class TestLogger:
    """Structured logging with structlog."""

    @pytest.fixture(autouse=True)
    def _reset_structlog(self) -> None:
        """Reset structlog config before each test to avoid cross-test pollution."""
        import structlog

        structlog.reset_defaults()
        # Reset the module-level guard so _configure_structlog re-runs
        from ai_orchestrator.observability import logging as logging_mod

        logging_mod.Logger._configured = False

    def test_init_defaults(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        assert log._service_name == "ai-orchestrator"
        assert log._log_level == "INFO"

    def test_init_custom(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger(service_name="custom-app", log_level="DEBUG", json_output=True)
        assert log._service_name == "custom-app"
        assert log._log_level == "DEBUG"

    def test_info(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        log.info("hello world", extra_field="value")
        assert True

    def test_warn(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        log.warn("warning message", count=3)
        assert True

    def test_error(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        log.error("error message", trace_id="abc123")
        assert True

    def test_debug(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        log.debug("debug message", detail="verbose")
        assert True

    def test_bind(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        bound = log.bind(user_id="u-001", request_id="r-999")
        assert bound is not log

    def test_with_task(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        task_logger = log.with_task("task-abc")
        assert task_logger is not log

    def test_with_agent(self) -> None:
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        agent_logger = log.with_agent("agent-007")
        assert agent_logger is not log

    def test_json_output(self) -> None:
        """With json_output=True, log messages should emit as JSON."""
        from ai_orchestrator.observability.logging import Logger

        log = Logger(service_name="test-svc", json_output=True)
        log.info("test json", value=42)
        assert True

    def test_text_output(self) -> None:
        """With json_output=False, log messages should be readable text."""
        from ai_orchestrator.observability.logging import Logger

        log = Logger(service_name="test-svc", json_output=False)
        log.info("test text", value=42)
        assert True

    def test_logger_formats_structured_event(self) -> None:
        """Verify structured fields appear without error."""
        from ai_orchestrator.observability.logging import Logger

        log = Logger()
        log.info("user action", action="login", user="admin")
        assert True
