"""Base sensor protocol — all sensors emit features, none makes decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSensor(ABC):
    """Abstract base for all sensors.

    Sensors are pure observers. They emit structured feature data.
    They never access system state, never make decisions, never
    interact with other sensors.
    """

    @abstractmethod
    async def sense(self, page) -> Any:
        """Observe the page and return structured features.

        Must not mutate any external state.
        Must not access other sensors.
        Must not make decisions.
        """
        ...

    def reset(self) -> None:  # noqa: B027
        """Reset sensor state between pages/sessions."""
        pass
