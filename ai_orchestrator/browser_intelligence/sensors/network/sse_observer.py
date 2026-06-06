"""SSEObserver — intercepts Server-Sent Events via CDP."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

log = logging.getLogger(__name__)

EVENTSOURCE_CONTENT_TYPES: frozenset[str] = frozenset({
    "text/event-stream",
    "text/event-stream;charset=utf-8",
    "text/event-stream; charset=utf-8",
})


class SSEObserver:
    """Observes SSE streams via CDP network events.

    Tracks event count, data chunks, bytes received, and stream
    lifecycle. Detects [DONE] markers for completion.
    """

    def __init__(self):
        self.event_count: int = 0
        self.data_chunk_count: int = 0
        self.bytes_received: int = 0
        self.last_event_time: float = 0.0
        self.first_event_time: float = 0.0
        self.stream_active: bool = False
        self.stream_closed: bool = False
        self.done_seen: bool = False
        self._active_stream_ids: set[str] = set()
        self._event_buffer: list[dict] = []
        self._request_id_to_url: dict[str, str] = {}

    def on_request_will_be_sent(self, event: dict) -> None:
        request = event.get("request", {})
        url = request.get("url", "")
        request_id = event.get("requestId", "")
        if request_id and url:
            self._request_id_to_url[request_id] = url

    def on_response_received(self, event: dict) -> bool:
        response = event.get("response", {})
        content_type = (
            response.get("mimeType", "")
            or response.get("headers", {}).get("content-type", "")
            or response.get("headers", {}).get("Content-Type", "")
        ).lower()

        if not self._is_event_stream(content_type):
            return False

        request_id = event.get("requestId", "")
        if request_id:
            self._active_stream_ids.add(request_id)
            self.stream_active = True
            self.stream_closed = False
            self.done_seen = False

        return True

    def on_event_source_message_received(self, event: dict) -> None:
        request_id = event.get("requestId", "")
        if request_id and request_id not in self._active_stream_ids:
            return

        now = time.monotonic()
        data = event.get("data", "")
        event_name = event.get("eventName", "")

        if self.first_event_time == 0.0:
            self.first_event_time = now

        self.event_count += 1
        self.last_event_time = now
        self.stream_active = True
        self.stream_closed = False

        if data:
            try:
                data_bytes = data.encode("utf-8")
            except Exception:
                data_bytes = b""
            self.bytes_received += len(data_bytes)
            self.data_chunk_count += 1

            stripped = data.strip()
            if stripped == "[DONE]":
                self.done_seen = True
                self.stream_closed = True
                self.stream_active = False

            self._event_buffer.append({
                "timestamp": now,
                "event": event_name,
                "data_len": len(data_bytes),
                "done": stripped == "[DONE]",
            })
            if len(self._event_buffer) > 500:
                self._event_buffer.pop(0)

    def on_loading_finished(self, event: dict) -> None:
        request_id = event.get("requestId", "")
        if request_id in self._active_stream_ids:
            self._active_stream_ids.discard(request_id)
            encoded_length = event.get("encodedDataLength", 0)
            if encoded_length > 0:
                self.bytes_received += encoded_length
            if not self._active_stream_ids:
                self.stream_closed = True
                self.stream_active = False

    def on_loading_failed(self, event: dict) -> None:
        request_id = event.get("requestId", "")
        if request_id in self._active_stream_ids:
            self._active_stream_ids.discard(request_id)
            self.stream_closed = True
            self.stream_active = False

    def tokens_per_second(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.monotonic()
        if self.data_chunk_count == 0 or self.first_event_time == 0.0:
            return 0.0
        elapsed = now - self.first_event_time
        if elapsed <= 0.0:
            return 0.0
        return self.data_chunk_count / elapsed

    def stream_idle_time(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.monotonic()
        if self.last_event_time == 0.0 or not self.stream_active:
            return 0.0
        return max(0.0, now - self.last_event_time)

    def reset(self) -> None:
        self.event_count = 0
        self.data_chunk_count = 0
        self.bytes_received = 0
        self.last_event_time = 0.0
        self.first_event_time = 0.0
        self.stream_active = False
        self.stream_closed = False
        self.done_seen = False
        self._active_stream_ids.clear()
        self._event_buffer.clear()
        self._request_id_to_url.clear()

    @staticmethod
    def _is_event_stream(content_type: str) -> bool:
        if not content_type:
            return False
        for ct in EVENTSOURCE_CONTENT_TYPES:
            if content_type == ct or content_type.startswith(ct):
                return True
        return False
