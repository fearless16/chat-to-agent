"""NetworkSensor — primary network intelligence orchestrator.

Owns protocol detector, SSE/WS/Fetch observers, and stream parser.
Every `sense()` call drains CDP events, routes them to the correct
observer, feeds the stream parser, and produces a unified
NetworkFeatures dataclass.

Network streams are the source of truth. DOM is a rendering artifact.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from ai_orchestrator.browser_intelligence.sensors.base import BaseSensor
from ai_orchestrator.browser_intelligence.sensors.network.fetch_observer import (
    FetchObserver,
)
from ai_orchestrator.browser_intelligence.sensors.network.protocol_detector import (
    ProtocolDetector,
    TransportProtocol,
)
from ai_orchestrator.browser_intelligence.sensors.network.sse_observer import SSEObserver
from ai_orchestrator.browser_intelligence.sensors.network.stream_parser import (
    StreamParser,
)
from ai_orchestrator.browser_intelligence.sensors.network.ws_observer import WSObserver

log = logging.getLogger(__name__)


@dataclass
class NetworkFeatures:
    # Transport detection
    transport_protocol: str = "unknown"
    transport_detected: bool = False
    transport_confidence: float = 0.0

    # Streaming state (THE MOST IMPORTANT SIGNALS)
    stream_active: bool = False
    tokens_per_second: float = 0.0
    total_chunks: int = 0
    bytes_received: int = 0
    stream_idle_time: float = 0.0
    last_chunk_timestamp: float = 0.0

    # Generation lifecycle
    generation_started: bool = False
    generation_completed: bool = False
    stream_closed: bool = False

    # Error signals
    error_codes: list[int] = field(default_factory=list)
    network_errors: int = 0

    # Diagnostic
    active_connections: int = 0
    request_rate: float = 0.0
    response_rate: float = 0.0

    # -- Legacy fields for backward compatibility --
    websocket_activity: bool = False
    sse_active: bool = False
    streaming_indicators: list[str] = field(default_factory=list)
    generation_event_detected: bool = False
    generation_stop_detected: bool = False


class NetworkSensor(BaseSensor):
    """Primary network intelligence sensor.

    Orchestrates protocol detection, per-transport observers, and
    the protocol-agnostic stream parser. Produces a rich
    NetworkFeatures dataclass every sense() call.

    No provider names are hardcoded. No fixed sleeps. Observes only.
    """

    def __init__(self):
        self._cdp_session = None
        self._protocol_detector = ProtocolDetector()
        self._sse_observer = SSEObserver()
        self._ws_observer = WSObserver()
        self._fetch_observer = FetchObserver()
        self._stream_parser = StreamParser()

        self._event_queue: list[dict] = []
        self._last_drain_time: float = time.monotonic()

        self._request_count: int = 0
        self._response_count: int = 0
        self._network_errors: int = 0
        self._error_codes: list[int] = []

        self._generation_started: bool = False
        self._generation_completed: bool = False
        self._prev_stream_idle: float = 0.0
        self._prev_observer_chunks: int = 0
        self._prev_observer_bytes: int = 0

    async def attach(self, page) -> None:
        try:
            cdp = await page.context.new_cdp_session(page)
            await cdp.send("Network.enable")

            cdp.on("Network.requestWillBeSent", self._on_request)
            cdp.on("Network.responseReceived", self._on_response)
            cdp.on("Network.dataReceived", self._on_data_received)
            cdp.on("Network.loadingFinished", self._on_loading_finished)
            cdp.on("Network.loadingFailed", self._on_loading_failed)
            cdp.on("Network.webSocketCreated", self._on_ws_created)
            cdp.on("Network.webSocketClosed", self._on_ws_closed)
            cdp.on("Network.webSocketFrameReceived", self._on_ws_frame)

            try:
                cdp.on("Network.eventSourceMessageReceived", self._on_event_source_message)
            except Exception:
                pass

            self._cdp_session = cdp
            log.debug("NetworkSensor attached, CDP Network domain enabled")
        except Exception as exc:
            log.debug("CDP attach failed (headless?): %s", exc)

    async def detach(self) -> None:
        try:
            if self._cdp_session:
                await self._cdp_session.detach()
                self._cdp_session = None
        except Exception:
            pass

    async def sense(self, page) -> NetworkFeatures:
        features = NetworkFeatures()
        try:
            self._drain_queued_events()
            now = time.monotonic()

            elapsed = now - self._last_drain_time
            self._last_drain_time = now

            features.request_rate = self._request_count / max(elapsed, 1.0)
            features.response_rate = self._response_count / max(elapsed, 1.0)
            self._request_count = 0
            self._response_count = 0

            features.error_codes = list(self._error_codes)
            features.network_errors = self._network_errors
            self._error_codes.clear()
            self._network_errors = 0

            proto, conf = self._protocol_detector.latest_detection
            features.transport_protocol = proto.value
            features.transport_detected = proto != TransportProtocol.UNKNOWN
            features.transport_confidence = conf

            self._feed_stream_parser(now)

            stream_state = self._stream_parser.evaluate(now)

            features.stream_active = stream_state.stream_active
            features.tokens_per_second = stream_state.tokens_per_second
            features.total_chunks = stream_state.total_chunks
            features.bytes_received = stream_state.bytes_received
            features.stream_idle_time = stream_state.stream_idle_time
            features.last_chunk_timestamp = stream_state.last_chunk_timestamp

            if stream_state.generation_started and not self._generation_started:
                self._generation_started = True
                features.generation_started = True
            elif self._generation_started and stream_state.stream_active:
                features.generation_started = True

            if stream_state.stream_closed and self._generation_started and not self._generation_completed:
                self._generation_completed = True
                features.generation_completed = True
                features.stream_closed = True

            if stream_state.stream_closed:
                features.stream_closed = True

            features.active_connections = (
                self._ws_observer.active_connection_count
                + (1 if self._sse_observer.stream_active else 0)
                + (1 if self._fetch_observer.stream_active else 0)
            )

            features.streaming_indicators = self._build_streaming_indicators(stream_state)
            features.websocket_activity = self._ws_observer.connection_open
            features.sse_active = self._sse_observer.stream_active
            features.generation_event_detected = stream_state.stream_active
            features.generation_stop_detected = stream_state.stream_closed

            self._prev_stream_idle = stream_state.stream_idle_time

        except Exception as exc:
            log.debug("NetworkSensor sense failed: %s", exc)

        return features

    def reset(self) -> None:
        self._event_queue.clear()
        self._request_count = 0
        self._response_count = 0
        self._network_errors = 0
        self._error_codes.clear()
        self._last_drain_time = time.monotonic()
        self._generation_started = False
        self._generation_completed = False
        self._prev_stream_idle = 0.0
        self._prev_observer_chunks = 0
        self._prev_observer_bytes = 0

        self._protocol_detector.reset()
        self._sse_observer.reset()
        self._ws_observer.reset()
        self._fetch_observer.reset()
        self._stream_parser.reset()

    # ------------------------------------------------------------------
    # CDP event handlers — queue then drain, never block
    # ------------------------------------------------------------------

    def _on_request(self, event: dict) -> None:
        self._event_queue.append(("request", event))

    def _on_response(self, event: dict) -> None:
        self._event_queue.append(("response", event))

    def _on_data_received(self, event: dict) -> None:
        self._event_queue.append(("data_received", event))

    def _on_loading_finished(self, event: dict) -> None:
        self._event_queue.append(("loading_finished", event))

    def _on_loading_failed(self, event: dict) -> None:
        self._event_queue.append(("loading_failed", event))

    def _on_ws_created(self, event: dict) -> None:
        self._event_queue.append(("ws_created", event))

    def _on_ws_closed(self, event: dict) -> None:
        self._event_queue.append(("ws_closed", event))

    def _on_ws_frame(self, event: dict) -> None:
        self._event_queue.append(("ws_frame", event))

    def _on_event_source_message(self, event: dict) -> None:
        self._event_queue.append(("event_source_message", event))

    # ------------------------------------------------------------------
    # Thread-safe event drain
    # ------------------------------------------------------------------

    def _drain_queued_events(self) -> None:
        events = self._event_queue[:]
        self._event_queue.clear()

        for kind, event in events:
            try:
                if kind == "request":
                    self._handle_request(event)
                elif kind == "response":
                    self._handle_response(event)
                elif kind == "data_received":
                    self._handle_data_received(event)
                elif kind == "loading_finished":
                    self._handle_loading_finished(event)
                elif kind == "loading_failed":
                    self._handle_loading_failed(event)
                elif kind == "ws_created":
                    self._handle_ws_created(event)
                elif kind == "ws_closed":
                    self._handle_ws_closed(event)
                elif kind == "ws_frame":
                    self._handle_ws_frame(event)
                elif kind == "event_source_message":
                    self._handle_event_source_message(event)
            except Exception as exc:
                log.debug("Error handling CDP event %s: %s", kind, exc)

    # ------------------------------------------------------------------
    # Per-event-type dispatchers
    # ------------------------------------------------------------------

    def _handle_request(self, event: dict) -> None:
        self._request_count += 1
        self._sse_observer.on_request_will_be_sent(event)

    def _handle_response(self, event: dict) -> None:
        self._response_count += 1

        response = event.get("response", {})
        status = response.get("status", 0)
        if status >= 400:
            self._error_codes.append(status)
            self._network_errors += 1

        url = response.get("url", "")
        content_type = (
            response.get("mimeType", "")
            or response.get("headers", {}).get("content-type", "")
            or response.get("headers", {}).get("Content-Type", "")
        )
        response_headers = response.get("headers", {}) or {}

        self._protocol_detector.feed_response(
            url=url,
            status=status,
            content_type=content_type,
            response_headers=response_headers,
        )

        self._sse_observer.on_response_received(event)
        self._fetch_observer.on_response_received(event)

    def _handle_data_received(self, event: dict) -> None:
        self._fetch_observer.on_data_received(event)

    def _handle_loading_finished(self, event: dict) -> None:
        self._sse_observer.on_loading_finished(event)
        self._fetch_observer.on_loading_finished(event)

    def _handle_loading_failed(self, event: dict) -> None:
        error_text = event.get("errorText", "")
        canceled = event.get("canceled", False)
        if not canceled:
            self._network_errors += 1
        self._sse_observer.on_loading_failed(event)
        self._fetch_observer.on_loading_failed(event)
        self._stream_parser.signal_transport_disconnected()

    def _handle_ws_created(self, event: dict) -> None:
        url = event.get("url", "")
        self._protocol_detector.feed_websocket_created(url=url)
        self._ws_observer.on_ws_created(event)
        self._stream_parser.signal_transport_connected()

    def _handle_ws_closed(self, event: dict) -> None:
        self._ws_observer.on_ws_closed(event)
        self._stream_parser.signal_transport_disconnected()

    def _handle_ws_frame(self, event: dict) -> None:
        self._ws_observer.on_ws_frame_received(event)

    def _handle_event_source_message(self, event: dict) -> None:
        self._sse_observer.on_event_source_message_received(event)

    # ------------------------------------------------------------------
    # Stream parser feeding
    # ------------------------------------------------------------------

    def _feed_stream_parser(self, now: float) -> None:
        ws_chunks = self._ws_observer.data_frame_count
        sse_chunks = self._sse_observer.data_chunk_count
        fetch_chunks = self._fetch_observer.chunk_count

        total_aggregated = ws_chunks + sse_chunks + fetch_chunks
        prev_aggregated = self._prev_observer_chunks

        if total_aggregated > prev_aggregated:
            new_chunks = total_aggregated - prev_aggregated
            for _ in range(min(new_chunks, 100)):
                self._stream_parser.push_event(data=None, timestamp=now)
            self._prev_observer_chunks = total_aggregated

        se_bytes = self._sse_observer.bytes_received
        ws_bytes = self._ws_observer.bytes_received
        fe_bytes = self._fetch_observer.bytes_received
        total_bytes = se_bytes + ws_bytes + fe_bytes
        prev_bytes = self._prev_observer_bytes
        if total_bytes > prev_bytes:
            delta = total_bytes - prev_bytes
            self._stream_parser.push_bytes(delta, timestamp=now)
            self._prev_observer_bytes = total_bytes

        sse_done = self._sse_observer.done_seen and not self._sse_observer.stream_active
        ws_done = self._ws_observer.done_seen and not self._ws_observer.connection_open
        if sse_done or ws_done:
            self._stream_parser.signal_transport_disconnected()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_streaming_indicators(self, stream_state) -> list[str]:
        indicators: list[str] = []
        proto, _ = self._protocol_detector.latest_detection

        if proto != TransportProtocol.UNKNOWN:
            indicators.append(f"protocol:{proto.value}")

        if stream_state.stream_active:
            indicators.append("stream:active")

        if stream_state.tokens_per_second > 0:
            indicators.append(f"tokens:{stream_state.tokens_per_second:.1f}/s")

        if self._sse_observer.stream_active:
            indicators.append("sse:active")

        if self._ws_observer.connection_open:
            indicators.append("ws:open")

        if self._fetch_observer.stream_active:
            indicators.append("fetch:streaming")

        if stream_state.stream_closed:
            indicators.append("stream:closed")

        if stream_state.generation_started:
            indicators.append("gen:started")

        if self._generation_completed:
            indicators.append("gen:completed")

        return indicators
