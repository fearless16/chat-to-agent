"""StreamParser — protocol-agnostic stream analysis engine."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

IDLE_THRESHOLD_ACTIVE: float = 2.0
IDLE_THRESHOLD_STREAM_END: float = 5.0
IDLE_THRESHOLD_HARD_TIMEOUT: float = 10.0
IDLE_THRESHOLD_GENERATION_START: float = 5.0
EMA_ALPHA: float = 0.3


@dataclass
class StreamState:
    tokens_per_second: float = 0.0
    stream_idle_time: float = 0.0
    total_chunks: int = 0
    bytes_received: int = 0
    last_chunk_timestamp: float = 0.0
    stream_active: bool = False
    generation_started: bool = False
    stream_closed: bool = False
    transport_disconnected: bool = False


class StreamParser:
    """Protocol-agnostic stream metrics engine.

    Accepts events from any observer (SSE, WS, Fetch), computes
    unified metrics: token rate (EMA-smoothed), idle time, chunk
    count, bytes, and detects lifecycle transitions.

    Survives rapid connection drops, mid-session protocol changes,
    and empty payloads. All time is wall-clock via time.monotonic().
    """

    def __init__(self):
        self.total_chunks: int = 0
        self.bytes_received: int = 0
        self.last_chunk_timestamp: float = 0.0
        self.first_chunk_timestamp: float = 0.0
        self._ema_token_rate: float = 0.0
        self._ema_alpha: float = EMA_ALPHA
        self._raw_chunk_timestamps: list[float] = []
        self._transport_disconnected: bool = False
        self._chunk_sizes: list[int] = []
        self._prev_chunk_count: int = 0
        self._last_eval_chunk_ts: float = -1.0

    def push_event(self, data: Optional[str] = None, timestamp: Optional[float] = None) -> None:
        now = timestamp if timestamp is not None else time.monotonic()

        byte_count = 0
        if data:
            try:
                byte_count = len(data.encode("utf-8"))
            except Exception:
                byte_count = len(data)

        self.total_chunks += 1
        self.bytes_received += byte_count
        self.last_chunk_timestamp = now
        if self.first_chunk_timestamp == 0.0:
            self.first_chunk_timestamp = now

        self._raw_chunk_timestamps.append(now)
        if len(self._raw_chunk_timestamps) > 200:
            self._raw_chunk_timestamps.pop(0)

        self._chunk_sizes.append(byte_count)
        if len(self._chunk_sizes) > 200:
            self._chunk_sizes.pop(0)

        current_rate = self._compute_instantaneous_rate(now)
        self._ema_token_rate = (
            self._ema_alpha * current_rate
            + (1.0 - self._ema_alpha) * self._ema_token_rate
        )

    def push_bytes(self, byte_count: int, timestamp: Optional[float] = None) -> None:
        if byte_count <= 0:
            return
        now = timestamp if timestamp is not None else time.monotonic()
        self.total_chunks += 1
        self.bytes_received += byte_count
        self.last_chunk_timestamp = now
        if self.first_chunk_timestamp == 0.0:
            self.first_chunk_timestamp = now

        self._raw_chunk_timestamps.append(now)
        if len(self._raw_chunk_timestamps) > 200:
            self._raw_chunk_timestamps.pop(0)

        self._chunk_sizes.append(byte_count)
        if len(self._chunk_sizes) > 200:
            self._chunk_sizes.pop(0)

        current_rate = self._compute_instantaneous_rate(now)
        self._ema_token_rate = (
            self._ema_alpha * current_rate
            + (1.0 - self._ema_alpha) * self._ema_token_rate
        )

    def signal_transport_disconnected(self) -> None:
        self._transport_disconnected = True

    def signal_transport_connected(self) -> None:
        self._transport_disconnected = False

    def evaluate(self, now: Optional[float] = None) -> StreamState:
        if now is None:
            now = time.monotonic()

        idle_time = 0.0
        if self.total_chunks > 0:
            idle_time = max(0.0, now - self.last_chunk_timestamp)

        stream_active = idle_time < IDLE_THRESHOLD_ACTIVE and self.total_chunks > 0

        generation_started = False
        if self.total_chunks > self._prev_chunk_count:
            if self._prev_chunk_count == 0:
                generation_started = True
            elif self._last_eval_chunk_ts >= 0.0:
                idle_before_new_chunk = self.last_chunk_timestamp - self._last_eval_chunk_ts
                if idle_before_new_chunk > IDLE_THRESHOLD_GENERATION_START:
                    generation_started = True

        stream_closed = False
        if self.total_chunks > 0:
            if idle_time > IDLE_THRESHOLD_HARD_TIMEOUT:
                stream_closed = True
            elif idle_time > IDLE_THRESHOLD_STREAM_END and self._transport_disconnected:
                stream_closed = True

        self._prev_chunk_count = self.total_chunks
        self._last_eval_chunk_ts = self.last_chunk_timestamp

        return StreamState(
            tokens_per_second=self._ema_token_rate,
            stream_idle_time=idle_time,
            total_chunks=self.total_chunks,
            bytes_received=self.bytes_received,
            last_chunk_timestamp=self.last_chunk_timestamp,
            stream_active=stream_active,
            generation_started=generation_started,
            stream_closed=stream_closed,
            transport_disconnected=self._transport_disconnected,
        )

    def tokens_per_second(self) -> float:
        return self._ema_token_rate

    def average_chunk_size(self) -> float:
        if not self._chunk_sizes:
            return 0.0
        return sum(self._chunk_sizes) / len(self._chunk_sizes)

    def chunk_rate_variance(self) -> float:
        if len(self._raw_chunk_timestamps) < 3:
            return 0.0
        intervals = [
            self._raw_chunk_timestamps[i] - self._raw_chunk_timestamps[i - 1]
            for i in range(1, len(self._raw_chunk_timestamps))
        ]
        if not intervals:
            return 0.0
        mean = sum(intervals) / len(intervals)
        return sum((d - mean) ** 2 for d in intervals) / len(intervals)

    def reset(self) -> None:
        self.total_chunks = 0
        self.bytes_received = 0
        self.last_chunk_timestamp = 0.0
        self.first_chunk_timestamp = 0.0
        self._ema_token_rate = 0.0
        self._transport_disconnected = False
        self._raw_chunk_timestamps.clear()
        self._chunk_sizes.clear()
        self._prev_chunk_count = 0
        self._last_eval_chunk_ts = -1.0

    def _compute_instantaneous_rate(self, now: float) -> float:
        recent = [t for t in self._raw_chunk_timestamps if now - t <= 5.0]
        if len(recent) < 2:
            if self.total_chunks > 0 and now > self.first_chunk_timestamp:
                return self.total_chunks / (now - self.first_chunk_timestamp)
            return 0.0
        elapsed = now - recent[0]
        if elapsed <= 0.0:
            return 0.0
        return len(recent) / elapsed
