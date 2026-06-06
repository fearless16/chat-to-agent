"""FetchObserver — detects Fetch/XHR streaming via CDP."""

from __future__ import annotations

import logging
import time

log = logging.getLogger(__name__)


class FetchObserver:
    """Observes Fetch / XHR requests and responses via CDP.

    Detects chunked transfer encoding, counts data-received events,
    and tracks stream lifecycle.
    """

    def __init__(self):
        self.chunk_count: int = 0
        self.bytes_received: int = 0
        self.last_data_time: float = 0.0
        self.first_data_time: float = 0.0
        self.stream_active: bool = False
        self.stream_closed: bool = False
        self._active_stream_request_ids: set[str] = set()
        self._data_buffer: list[dict] = []
        self._response_text: str = ""
        self._pending_body_requests: list[tuple] = []
        self._cdp_session = None

    def on_response_received(self, event: dict) -> bool:
        response = event.get("response", {})
        response_headers = response.get("headers", {})
        transfer_encoding = (
            response_headers.get("transfer-encoding", "")
            or response_headers.get("Transfer-Encoding", "")
        ).lower()

        content_type = (
            response.get("mimeType", "")
            or response_headers.get("content-type", "")
            or response_headers.get("Content-Type", "")
        ).lower()

        request_id = event.get("requestId", "")
        is_streaming = False

        is_chunked = "chunked" in transfer_encoding
        if request_id and (is_chunked or self._is_streaming_content(content_type)):
            self._active_stream_request_ids.add(request_id)
            self.stream_active = True
            self.stream_closed = False
            is_streaming = True

        return is_streaming

    def on_data_received(self, event: dict) -> None:
        request_id = event.get("requestId", "")
        if request_id and request_id not in self._active_stream_request_ids:
            return

        now = time.monotonic()
        data_length = event.get("dataLength", 0)
        encoded_length = event.get("encodedDataLength", 0)

        if data_length > 0 or encoded_length > 0:
            received = data_length or encoded_length
            self.bytes_received += received
            self.chunk_count += 1

            if self.first_data_time == 0.0:
                self.first_data_time = now
            self.last_data_time = now
            self.stream_active = True
            self.stream_closed = False

            self._data_buffer.append({
                "timestamp": now,
                "bytes": received,
            })
            if len(self._data_buffer) > 500:
                self._data_buffer.pop(0)

    def on_loading_finished(self, event: dict) -> None:
        request_id = event.get("requestId", "")
        if request_id in self._active_stream_request_ids:
            self._active_stream_request_ids.discard(request_id)
        if not self._active_stream_request_ids and self.chunk_count > 0:
            self.stream_closed = True
            self.stream_active = False
        if request_id and self._cdp_session is not None:
            self._pending_body_requests.append((request_id,))

    def on_loading_failed(self, event: dict) -> None:
        request_id = event.get("requestId", "")
        if request_id in self._active_stream_request_ids:
            self._active_stream_request_ids.discard(request_id)
        if not self._active_stream_request_ids and self.chunk_count > 0:
            self.stream_closed = True
            self.stream_active = False

    def set_cdp_session(self, cdp_session) -> None:
        """Inject the CDP session so we can call Network.getResponseBody
        on loading_finished to capture the full stream body."""
        self._cdp_session = cdp_session

    async def drain_pending_fetches(self) -> None:
        while self._pending_body_requests:
            request_id, = self._pending_body_requests.pop(0)
            try:
                resp = await self._cdp_session.send(
                    "Network.getResponseBody", {"requestId": request_id}
                )
                body = resp.get("body", "") if isinstance(resp, dict) else ""
                if body:
                    # Accept all response body content — SSE, JSON, plain text.
                    # Previously filtered out SSE (starting with "data:") but
                    # SSE bodies must be captured and parsed downstream.
                    if len(body) > 10:
                        self._response_text += body
                        if len(self._response_text) > 1_000_000:
                            self._response_text = self._response_text[-1_000_000:]
            except Exception:
                pass

    def get_response_text(self) -> str:
        """Return the full body text for chunked fetch responses (if captured)."""
        return self._response_text

    def tokens_per_second(self, now: float | None = None) -> float:
        if now is None:
            now = time.monotonic()
        if self.chunk_count == 0 or self.first_data_time == 0.0:
            return 0.0
        elapsed = now - self.first_data_time
        if elapsed <= 0.0:
            return 0.0
        return self.chunk_count / elapsed

    def stream_idle_time(self, now: float | None = None) -> float:
        if now is None:
            now = time.monotonic()
        if self.last_data_time == 0.0 or not self.stream_active:
            return 0.0
        return max(0.0, now - self.last_data_time)

    def reset(self) -> None:
        self.chunk_count = 0
        self.bytes_received = 0
        self.last_data_time = 0.0
        self.first_data_time = 0.0
        self.stream_active = False
        self.stream_closed = False
        self._active_stream_request_ids.clear()
        self._data_buffer.clear()
        self._response_text = ""

    @staticmethod
    def _is_streaming_content(content_type: str) -> bool:
        if not content_type:
            return False
        ct = content_type.lower()
        streaming_types = (
            "application/x-ndjson",
            "application/x-json-stream",
            "text/plain",
            "text/event-stream",
        )
        return any(ct.startswith(t) for t in streaming_types)
