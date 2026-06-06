"""ResponseCapture — transport-native body capture for chat responses.

The legacy `FetchObserver` could only retrieve the body AFTER the stream
closed, via `Network.getResponseBody`. That meant the body was opaque
during streaming and any non-chat fetch that finished first would win
the buffer.

This module gives the engine a real, transport-native body capture path:

1.  `Network.streamResourceContent` (CDP) — chunks delivered as the
    stream runs, no need to wait for completion.
2.  `Network.responseReceived` + body callback — completion-time body
    retrieval as a backstop when streaming is unavailable.
3.  `Network.eventSourceMessageReceived` — SSE body chunks as they arrive.
4.  `Network.webSocketFrameReceived` — WS payload frames.

The engine itself owns the extraction. The capture layer is router-only:
given a request_id and url, it tracks which body bytes belong to which
response, hands the assembled text to the engine when the response is
classified as CHAT_RESPONSE.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Iterable

from ai_orchestrator.browser_intelligence.intelligence.traffic_classifier import (
    ResponseClassifier,
    TrafficCategory,
    TrafficClassification,
)

log = logging.getLogger(__name__)

# Cap on body buffer per request (chars) — keeps memory bounded.
_MAX_BODY_CHARS: int = 2_000_000

# Cap on number of tracked requests (prevents unbounded growth).
_MAX_TRACKED_REQUESTS: int = 256


@dataclass
class CapturedResponse:
    """A single chat response assembled by the engine.

    Built incrementally from CDP body chunks. The engine can read
    `text` at any time to get the latest assembled plaintext; the
    final value is set when the stream closes.
    """

    request_id: str
    url: str
    method: str
    status: int
    content_type: str
    started_at: float
    last_chunk_at: float
    chunks: int = 0
    bytes: int = 0
    classification: TrafficClassification | None = None
    stream_active: bool = True
    stream_closed: bool = False
    text: str = ""
    _first_sample_score: float = 0.0

    def append(self, chunk: str, *, timestamp: float | None = None) -> None:
        if not chunk:
            return
        self.text += chunk
        if len(self.text) > _MAX_BODY_CHARS:
            # Keep the tail, which is what chat UIs need.
            self.text = self.text[-_MAX_BODY_CHARS:]
        self.chunks += 1
        self.bytes += len(chunk)
        self.last_chunk_at = timestamp if timestamp is not None else time.monotonic()
        self.stream_active = True

    def close(self, timestamp: float | None = None) -> None:
        self.stream_active = False
        self.stream_closed = True
        self.last_chunk_at = timestamp if timestamp is not None else time.monotonic()

    @property
    def age_seconds(self, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        return max(0.0, now - self.started_at)

    @property
    def idle_seconds(self, now: float | None = None) -> float:
        now = now if now is not None else time.monotonic()
        if not self.last_chunk_at:
            return 0.0
        return max(0.0, now - self.last_chunk_at)


class ResponseCapture:
    """Aggregates CDP body events into per-request chat responses.

    Thread-safe: all state is internal and only mutated from the
    single asyncio task that drains the event queue, so no locks.
    """

    def __init__(self, classifier: ResponseClassifier | None = None):
        self._classifier = classifier or ResponseClassifier()
        self._by_request_id: dict[str, CapturedResponse] = {}
        self._by_url: dict[str, list[str]] = {}  # url → request_ids
        self._closed: list[CapturedResponse] = []  # recent chat responses
        self._closed_max: int = 32

    # ── Lifecycle ────────────────────────────────────────────────

    def begin_response(
        self,
        *,
        request_id: str,
        url: str,
        method: str,
        status: int,
        content_type: str,
        headers: dict | None = None,
        timestamp: float | None = None,
    ) -> TrafficClassification:
        """Register a new response. Returns the classification so the
        caller can decide whether to keep buffering this body."""
        now = timestamp if timestamp is not None else time.monotonic()
        cap = CapturedResponse(
            request_id=request_id,
            url=url,
            method=method,
            status=status,
            content_type=content_type,
            started_at=now,
            last_chunk_at=now,
        )
        classification = self._classifier.classify(
            url=url,
            method=method,
            headers=headers or {},
            content_type=content_type,
            status=status,
        )
        cap.classification = classification
        self._by_request_id[request_id] = cap
        self._by_url.setdefault(url, []).append(request_id)
        self._trim_tracked()
        return classification

    def append_chunk(
        self,
        request_id: str,
        chunk: str,
        *,
        timestamp: float | None = None,
    ) -> bool:
        """Append body text to the captured response. Returns True
        if the chunk was accepted (i.e. the response is being tracked
        and is not pollution)."""
        cap = self._by_request_id.get(request_id)
        if cap is None:
            return False
        if cap.classification and not cap.classification.is_chat:
            return False
        if not chunk:
            return False
        cap.append(chunk, timestamp=timestamp)
        return True

    def close_response(
        self,
        request_id: str,
        *,
        timestamp: float | None = None,
    ) -> CapturedResponse | None:
        cap = self._by_request_id.get(request_id)
        if cap is None:
            return None
        cap.close(timestamp=timestamp)
        # Move to closed store if it was a chat response.
        if cap.classification and cap.classification.is_chat:
            self._closed.append(cap)
            if len(self._closed) > self._closed_max:
                self._closed = self._closed[-self._closed_max:]
        # Drop from active tracking to free memory.
        self._by_request_id.pop(request_id, None)
        # Don't drop url→request_ids; let it age out naturally.
        return cap

    def discard(self, request_id: str) -> None:
        self._by_request_id.pop(request_id, None)

    def reset(self) -> None:
        self._by_request_id.clear()
        self._by_url.clear()
        self._closed.clear()

    # ── Queries ──────────────────────────────────────────────────

    def active_responses(self) -> list[CapturedResponse]:
        return list(self._by_request_id.values())

    def get_response(self, request_id: str) -> CapturedResponse | None:
        return self._by_request_id.get(request_id)

    def latest_chat_response(self) -> CapturedResponse | None:
        # Active in-progress chat responses take priority — they
        # represent streaming output the engine should read now.
        active = [
            c for c in self.active_responses()
            if c.classification and c.classification.is_chat and c.text
        ]
        if active:
            return max(active, key=lambda c: c.started_at)
        # Otherwise, the most recently closed chat response.
        closed = [c for c in self._closed if c.text]
        if closed:
            return closed[-1]
        return None

    def get_response_text(self) -> str:
        """Return the text of the most recent chat response.

        This is the primary public method the engine calls when it
        needs the model's reply. Returns "" when nothing chat-like
        has been observed.
        """
        cap = self.latest_chat_response()
        if cap is None:
            return ""
        return cap.text

    def get_response_text_sse(self) -> str:
        """Same as `get_response_text`; kept for backward compat with
        the engine API that historically asked for SSE vs WS variants."""
        return self.get_response_text()

    def stats(self) -> dict:
        return {
            "active": len(self._by_request_id),
            "closed": len(self._closed),
            "classifier": self._classifier.stats,
        }

    # ── Helpers ──────────────────────────────────────────────────

    def _trim_tracked(self) -> None:
        if len(self._by_request_id) <= _MAX_TRACKED_REQUESTS:
            return
        # Drop the oldest non-chat responses first.
        victims: list[str] = []
        for rid, cap in self._by_request_id.items():
            if cap.classification and not cap.classification.is_chat:
                victims.append(rid)
            if len(victims) >= len(self._by_request_id) - _MAX_TRACKED_REQUESTS + 1:
                break
        for rid in victims:
            self._by_request_id.pop(rid, None)
        # If still over the cap, drop oldest chat (newest remain).
        if len(self._by_request_id) > _MAX_TRACKED_REQUESTS:
            ordered = sorted(
                self._by_request_id.items(),
                key=lambda kv: kv[1].started_at,
            )
            excess = len(self._by_request_id) - _MAX_TRACKED_REQUESTS
            for rid, _ in ordered[:excess]:
                self._by_request_id.pop(rid, None)


# ──────────────────────────────────────────────────────────────────────
# Body parsers — turn raw body bytes (often SSE / NDJSON / WS JSON
# frames) into plain text deltas. Pure functions, no state.
# ──────────────────────────────────────────────────────────────────────

_DELTA_PATHS: tuple[tuple, ...] = (
    ("data", "delta_content"),
    ("data", "content"),
    ("delta", "content"),
    ("message", "content"),
    ("choices", 0, "delta", "content"),
    ("choices", 0, "message", "content"),
    ("choices", 0, "text"),
    ("response", "choices", 0, "delta", "content"),
    ("response", "choices", 0, "message", "content"),
    ("text",),
    ("content",),
    ("message",),
)


def _walk_path(obj, path) -> object:
    cur = obj
    for key in path:
        try:
            cur = cur[key]
        except (KeyError, IndexError, TypeError):
            return None
    return cur


def parse_stream_chunk(raw: str) -> str:
    """Parse one raw chunk of body text (SSE, NDJSON, WS JSON, or
    plain) and return the assembled delta text.

    Handles:
    - SSE lines: ``data: {json}`` separated by ``\\n\\n``.
    - NDJSON: one JSON object per line.
    - Bare JSON object.
    - Plain text (returned verbatim, trimmed).
    """
    if not raw:
        return ""
    out: list[str] = []
    saw_delta = False
    for line in raw.splitlines():
        s = line.strip()
        if not s:
            continue
        payload = s[5:].strip() if s.startswith("data:") else s
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            out.append(payload)
            saw_delta = True
            continue
        if not isinstance(obj, dict):
            continue
        matched = False
        for path in _DELTA_PATHS:
            val = _walk_path(obj, path)
            if isinstance(val, str) and val:
                out.append(val)
                saw_delta = True
                matched = True
                break
        if not matched:
            # Last-resort: walk and grab any string under a delta-like key.
            for k, val in _walk_delta_strings(obj):
                if val:
                    out.append(val)
                    saw_delta = True
                    break
    if saw_delta:
        return "".join(out)
    return raw


def _walk_delta_strings(obj) -> Iterable[tuple[str, str]]:
    keys = {
        "delta_content", "content", "text", "delta_text",
        "message", "reasoning_content", "delta", "completion",
    }
    stack = [obj]
    seen: set[int] = set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(v, str) and v and k.lower() in keys:
                    yield (k, v)
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append(v)
