"""WSObserver — intercepts WebSocket frames via CDP."""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

log = logging.getLogger(__name__)


class WSObserver:
    """Observes WebSocket frames via CDP.

    Tracks frame count, payload data, bytes received, and detects
    [DONE] markers for completion.
    """

    def __init__(self):
        self.frame_count: int = 0
        self.data_frame_count: int = 0
        self.bytes_received: int = 0
        self.last_frame_time: float = 0.0
        self.first_frame_time: float = 0.0
        self.stream_active: bool = False
        self.stream_closed: bool = False
        self.done_seen: bool = False
        self.connection_open: bool = False
        self._active_ws_urls: set[str] = set()
        self._frame_buffer: list[dict] = []

    def on_ws_created(self, event: dict) -> None:
        url = event.get("url", "")
        self._active_ws_urls.add(url)
        self.connection_open = True
        self.stream_closed = False
        self.stream_active = False

    def on_ws_closed(self, event: dict) -> None:
        url = event.get("url", "")
        self._active_ws_urls.discard(url)
        if not self._active_ws_urls:
            self.connection_open = False
            self.stream_closed = True
            self.stream_active = False

    def on_ws_frame_received(self, event: dict) -> None:
        now = time.monotonic()
        response = event.get("response", {})
        payload = response.get("payloadData", "")

        self.frame_count += 1

        if self.first_frame_time == 0.0:
            self.first_frame_time = now

        self.last_frame_time = now

        if payload:
            try:
                payload_bytes = payload.encode("utf-8")
            except Exception:
                payload_bytes = b""
            self.bytes_received += len(payload_bytes)

            payload_stripped = payload.strip()

            if payload_stripped:
                self.data_frame_count += 1
                self.stream_active = True
                self.stream_closed = False

            if "[DONE]" in payload:
                self.done_seen = True
                self.stream_closed = True
                self.stream_active = False

            if not payload_stripped:
                if self.data_frame_count > 0:
                    self.stream_closed = True
                    self.stream_active = False

            self._frame_buffer.append({
                "timestamp": now,
                "payload_len": len(payload_bytes),
                "done": "[DONE]" in payload,
                "is_json": self._is_json(payload),
            })
            if len(self._frame_buffer) > 500:
                self._frame_buffer.pop(0)

    def tokens_per_second(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.monotonic()
        if self.data_frame_count == 0 or self.first_frame_time == 0.0:
            return 0.0
        elapsed = now - self.first_frame_time
        if elapsed <= 0.0:
            return 0.0
        return self.data_frame_count / elapsed

    def stream_idle_time(self, now: Optional[float] = None) -> float:
        if now is None:
            now = time.monotonic()
        if self.last_frame_time == 0.0 or not self.stream_active:
            return 0.0
        return max(0.0, now - self.last_frame_time)

    def reset(self) -> None:
        self.frame_count = 0
        self.data_frame_count = 0
        self.bytes_received = 0
        self.last_frame_time = 0.0
        self.first_frame_time = 0.0
        self.stream_active = False
        self.stream_closed = False
        self.done_seen = False
        self.connection_open = False
        self._active_ws_urls.clear()
        self._frame_buffer.clear()

    @property
    def active_connection_count(self) -> int:
        return len(self._active_ws_urls)

    @staticmethod
    def _is_json(payload: str) -> bool:
        if not payload:
            return False
        stripped = payload.strip()
        if not stripped:
            return False
        if stripped[0] not in ("{", "[", '"'):
            return False
        try:
            json.loads(stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            return False
