"""Browser Intelligence — events subsystem.

A small, dependency-free event bus for the engine. Every meaningful
state transition in the engine publishes an `IntelligenceEvent`; every
subscriber (recovery, learning, scheduling, observability) consumes
the stream. The bus is sync-by-default but exposes an async dispatch
helper for I/O-bound subscribers.
"""

from __future__ import annotations

import inspect
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Union

log = logging.getLogger(__name__)


class EventType(str, Enum):
    """All known event types in the Browser Intelligence OS."""

    AUTH_SUCCESS = "AUTH_SUCCESS"
    AUTH_FAILURE = "AUTH_FAILURE"

    PROMPT_SENT = "PROMPT_SENT"

    GENERATION_STARTED = "GENERATION_STARTED"
    GENERATION_PROGRESS = "GENERATION_PROGRESS"
    GENERATION_COMPLETED = "GENERATION_COMPLETED"

    RESPONSE_CAPTURED = "RESPONSE_CAPTURED"

    RATE_LIMIT_DETECTED = "RATE_LIMIT_DETECTED"
    SHADOW_BAN_DETECTED = "SHADOW_BAN_DETECTED"

    RECOVERY_STARTED = "RECOVERY_STARTED"
    RECOVERY_COMPLETED = "RECOVERY_COMPLETED"

    WORKER_DEGRADED = "WORKER_DEGRADED"

    DRIFT_DETECTED = "DRIFT_DETECTED"

    GENERIC = "GENERIC"


@dataclass
class IntelligenceEvent:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "event_id": self.event_id,
            "source": self.source,
        }


Subscriber = Callable[[IntelligenceEvent], Any]
AsyncSubscriber = Callable[[IntelligenceEvent], Awaitable[None]]


class EventBus:
    """Synchronous in-process event bus with optional async fan-out.

    Bounded ring buffer of recent events for replay/diagnostics.
    """

    def __init__(self, history_limit: int = 1024):
        self._subscribers: list[tuple[Subscriber, bool]] = []
        self._history: deque[IntelligenceEvent] = deque(maxlen=history_limit)
        self._emitted: int = 0
        self._dropped: int = 0

    def subscribe(
        self,
        handler: Subscriber,
        *,
        is_async: bool | None = None,
    ) -> Callable[[], None]:
        is_async_handler = is_async if is_async is not None else _is_async_callable(handler)
        self._subscribers.append((handler, is_async_handler))
        return lambda: self._unsubscribe(handler)

    def _unsubscribe(self, handler) -> None:
        self._subscribers = [
            (h, a) for (h, a) in self._subscribers if h is not handler
        ]

    def publish(
        self,
        event_type: EventType,
        payload: dict | None = None,
        *,
        source: str = "",
    ) -> IntelligenceEvent:
        evt = IntelligenceEvent(
            type=event_type,
            payload=payload or {},
            source=source,
        )
        self._emitted += 1
        self._history.append(evt)
        for handler, is_async in list(self._subscribers):
            try:
                if is_async:
                    # Async subscribers are run by dispatch_async().
                    # We skip them here so the publish() call stays
                    # synchronous and fast.
                    continue
                handler(evt)
            except Exception as exc:
                log.warning("Subscriber raised on %s: %s", event_type, exc)
        return evt

    async def dispatch_async(self, evt: IntelligenceEvent) -> None:
        seen: set[int] = set()
        queued: list[AsyncSubscriber] = []
        for handler, is_async in self._subscribers:
            if not is_async:
                continue
            hid = id(handler)
            if hid in seen:
                continue
            seen.add(hid)
            queued.append(handler)  # type: ignore[arg-type]
        for handler in queued:
            try:
                await handler(evt)  # type: ignore[arg-type]
            except Exception as exc:
                log.warning("Async subscriber raised: %s", exc)

    def history(self, last_n: int | None = None) -> list[IntelligenceEvent]:
        items = list(self._history)
        if last_n is not None:
            return items[-last_n:]
        return items

    def stats(self) -> dict:
        return {
            "emitted": self._emitted,
            "history_size": len(self._history),
            "history_capacity": self._history.maxlen or 0,
            "subscribers": len(self._subscribers),
        }

    def clear(self) -> None:
        self._history.clear()
        self._dropped = 0
        self._emitted = 0


def _is_async_callable(fn) -> bool:
    return inspect.iscoroutinefunction(fn)


def make_event(
    type: EventType,
    **payload: Any,
) -> IntelligenceEvent:
    return IntelligenceEvent(type=type, payload=payload)


__all__ = [
    "EventBus",
    "EventType",
    "IntelligenceEvent",
    "make_event",
    "Subscriber",
    "AsyncSubscriber",
]
