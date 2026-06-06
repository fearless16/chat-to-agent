"""ProtocolDetector — identifies transport protocol without hardcoding providers."""

from __future__ import annotations

import enum
import logging
from typing import Optional

log = logging.getLogger(__name__)


class TransportProtocol(enum.Enum):
    SSE = "sse"
    WEBSOCKET = "websocket"
    FETCH_STREAM = "fetch_stream"
    XHR_POLL = "xhr"
    UNKNOWN = "unknown"


SSE_CONTENT_TYPES: frozenset[str] = frozenset({
    "text/event-stream",
    "text/event-stream;charset=utf-8",
    "text/event-stream; charset=utf-8",
})

WS_CONTENT_TYPES: frozenset[str] = frozenset({
    "application/octet-stream",
})

STREAMING_CONTENT_TYPES: frozenset[str] = frozenset({
    "application/json",
    "application/x-ndjson",
    "text/plain",
    "application/x-json-stream",
})

STREAM_URL_PATTERNS: tuple[str, ...] = (
    "/conversation",
    "/chat",
    "/completion",
    "/message",
    "/stream",
    "/api/generate",
    "/v1/chat",
    "/v1/completions",
    "/rpc",
    "/graphql",
)


def _content_type_matches(content_type: Optional[str], candidates: frozenset[str]) -> bool:
    if not content_type:
        return False
    ct_lower = content_type.lower().strip()
    for candidate in candidates:
        if ct_lower == candidate or ct_lower.startswith(candidate):
            return True
    return False


def _url_matches(url: str) -> int:
    url_lower = url.lower()
    return sum(1 for p in STREAM_URL_PATTERNS if p in url_lower)


class ProtocolDetector:
    """Detects the transport protocol in use by an HTTP response.

    Uses multiple signals to compute a confidence score. No provider names
    are hardcoded — detection is purely based on protocol characteristics.
    """

    def __init__(self):
        self._last_content_type: Optional[str] = None
        self._last_url: str = ""
        self._last_status: int = 0
        self._detection_history: list[tuple[str, float]] = []

    def feed_response(
        self,
        url: str = "",
        status: int = 0,
        content_type: Optional[str] = None,
        response_headers: Optional[dict[str, str]] = None,
    ) -> tuple[TransportProtocol, float]:
        """Analyze a response and return (protocol, confidence).

        Confidence is a float 0.0–1.0 aggregated from multiple signals.
        """
        self._last_url = url
        self._last_status = status
        self._last_content_type = content_type

        scores: dict[TransportProtocol, float] = {}

        if status == 101:
            scores[TransportProtocol.WEBSOCKET] = scores.get(
                TransportProtocol.WEBSOCKET, 0.0
            ) + 0.90

        if _content_type_matches(content_type, SSE_CONTENT_TYPES):
            scores[TransportProtocol.SSE] = scores.get(
                TransportProtocol.SSE, 0.0
            ) + 0.85

        if content_type == "application/octet-stream":
            scores[TransportProtocol.WEBSOCKET] = scores.get(
                TransportProtocol.WEBSOCKET, 0.0
            ) + 0.40

        if _content_type_matches(content_type, STREAMING_CONTENT_TYPES):
            transfer_encoding = ""
            if response_headers:
                transfer_encoding = (
                    response_headers.get("transfer-encoding", "")
                    or response_headers.get("Transfer-Encoding", "")
                ).lower()
            if "chunked" in transfer_encoding:
                scores[TransportProtocol.FETCH_STREAM] = scores.get(
                    TransportProtocol.FETCH_STREAM, 0.0
                ) + 0.70
            else:
                scores[TransportProtocol.FETCH_STREAM] = scores.get(
                    TransportProtocol.FETCH_STREAM, 0.0
                ) + 0.25

        url_hits = _url_matches(url)
        if url_hits > 0:
            url_boost = min(0.05 * url_hits, 0.25)
            for proto in (TransportProtocol.SSE, TransportProtocol.FETCH_STREAM):
                scores[proto] = scores.get(proto, 0.0) + url_boost

        xhr_detected = ""
        if response_headers:
            xhr_detected = (
                response_headers.get("x-requested-with", "")
            ).lower()
        if xhr_detected == "xmlhttprequest":
            scores[TransportProtocol.XHR_POLL] = scores.get(
                TransportProtocol.XHR_POLL, 0.0
            ) + 0.30

        if not scores:
            if url_hits > 0:
                best = TransportProtocol.FETCH_STREAM
                conf = min(0.15 * url_hits, 0.35)
            else:
                best = TransportProtocol.UNKNOWN
                conf = 0.0
        else:
            best = max(scores, key=lambda k: scores[k])
            conf = min(scores[best], 1.0)

        self._detection_history.append((best.value, conf))
        if len(self._detection_history) > 20:
            self._detection_history.pop(0)

        return best, conf

    def feed_websocket_created(self, url: str = "") -> tuple[TransportProtocol, float]:
        """Called when Network.webSocketCreated fires."""
        self._last_url = url or self._last_url
        conf = 0.95
        self._detection_history.append((TransportProtocol.WEBSOCKET.value, conf))
        if len(self._detection_history) > 20:
            self._detection_history.pop(0)
        return TransportProtocol.WEBSOCKET, conf

    @property
    def latest_detection(self) -> tuple[TransportProtocol, float]:
        if not self._detection_history:
            return TransportProtocol.UNKNOWN, 0.0
        last_proto_str, last_conf = self._detection_history[-1]
        try:
            proto = TransportProtocol(last_proto_str)
        except ValueError:
            proto = TransportProtocol.UNKNOWN
        return proto, last_conf

    def smoothed_confidence(self, window: int = 5) -> float:
        recent = self._detection_history[-window:]
        if not recent:
            return 0.0
        return sum(c for _, c in recent) / len(recent)

    def reset(self) -> None:
        self._last_content_type = None
        self._last_url = ""
        self._last_status = 0
        self._detection_history.clear()
