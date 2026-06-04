"""Resource-aware task scheduler — watermark-based admission control."""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from ai_orchestrator.models.task import TaskPriority


@dataclass(frozen=True)
class SystemResources:
    """Snapshot of current system resource utilization."""

    total_ram_gb: float
    available_ram_gb: float
    total_cores: int
    available_cores: float
    memory_usage_percent: float
    cpu_usage_percent: float


class WatermarkLevel(enum.IntEnum):
    """Escalating resource-pressure levels."""

    NORMAL = 0
    WARNING = 1
    CLEANUP = 2
    EMERGENCY = 3
    CRITICAL = 4


_WATERMARK_THRESHOLDS_GB: list[float] = [3.0, 2.5, 2.0, 1.5]

# Provider estimate table: upper-cased kind -> GB estimate
_PROVIDER_RAM_ESTIMATES: dict[str, float] = {
    "API": 0.5,
    "BROWSER": 1.5,
    "LOCAL": 3.0,
}

_DEFAULT_AVG_RAM_PER_AGENT: float = 1.5
_DEFAULT_BROWSER_MAX_CONTEXTS: int = 10
_DEFAULT_PROVIDER_MAX_CONCURRENT: int = 20
_DEFAULT_CONFIGURED_MAX_AGENTS: int = 20

# Minimum agents to always allow, even under resource pressure.
_MIN_AGENTS: int = 1

# Priority -> first watermark level at which tasks are rejected.
# Ordered so higher numeric priority (lower urgency) maps to stricter rejection.
_PRIORITY_REJECTION_MAP: dict[TaskPriority, WatermarkLevel] = {
    TaskPriority.CRITICAL: WatermarkLevel.CRITICAL,
    TaskPriority.HIGH: WatermarkLevel.EMERGENCY,
    TaskPriority.NORMAL: WatermarkLevel.EMERGENCY,
    TaskPriority.LOW: WatermarkLevel.CLEANUP,
    TaskPriority.BACKGROUND: WatermarkLevel.WARNING,
}

_ACTION_MAP: dict[WatermarkLevel, str] = {
    WatermarkLevel.NORMAL: "no action needed",
    WatermarkLevel.WARNING: "reduce low-priority agents",
    WatermarkLevel.CLEANUP: "suspend idle browsers, flush caches",
    WatermarkLevel.EMERGENCY: "pause all non-critical agents, trim memory",
    WatermarkLevel.CRITICAL: "freeze new tasks, kill lowest priority agents",
}


class ResourceScheduler:
    """Schedules agents and admits tasks based on watermark pressure levels.

    Watermarks descend from NORMAL (plentiful) through WARNING, CLEANUP,
    EMERGENCY to CRITICAL (severe pressure).  Each watermark triggers
    progressively stronger actions and admission restrictions.
    """

    def __init__(self, configured_max_agents: int = _DEFAULT_CONFIGURED_MAX_AGENTS) -> None:
        self._configured_max = configured_max_agents

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_max_agents(
        self,
        resources: SystemResources,
        avg_ram_per_agent: float = _DEFAULT_AVG_RAM_PER_AGENT,
        browser_max_contexts: int = _DEFAULT_BROWSER_MAX_CONTEXTS,
        provider_max_concurrent: int = _DEFAULT_PROVIDER_MAX_CONCURRENT,
    ) -> int:
        """Return the maximum number of agents that can run concurrently.

        Formula::

            MaxAgents = min(floor((AvailRAM - 2) / avg_ram_per_agent),
                            cores * 2,
                            browser_max_contexts,
                            provider_max_concurrent,
                            configured_max)

        Always returns at least **1**.
        """
        # Ram-based agent limit
        ram_available = max(resources.available_ram_gb, 0.0)
        ram_limit = math.floor((ram_available - 2.0) / avg_ram_per_agent)
        ram_limit = max(ram_limit, 0)

        # Core-based limit (use available_cores for finer granularity)
        core_limit = int(math.floor(resources.available_cores * 2))

        candidates = [
            ram_limit,
            core_limit,
            browser_max_contexts,
            provider_max_concurrent,
            self._configured_max,
        ]
        result = min(candidates)
        return max(result, _MIN_AGENTS)

    def get_watermark_level(self, available_ram_gb: float) -> WatermarkLevel:
        """Classify available RAM into a pressure watermark level."""
        # Check from most restrictive (1.5 GB) to least restrictive (3.0 GB)
        # so that e.g. 2.5 GB → CLEANUP, not WARNING.
        for level, threshold in reversed(
            list(
                zip(
                    [
                        WatermarkLevel.WARNING,
                        WatermarkLevel.CLEANUP,
                        WatermarkLevel.EMERGENCY,
                        WatermarkLevel.CRITICAL,
                    ],
                    _WATERMARK_THRESHOLDS_GB,
                )
            )
        ):
            if available_ram_gb <= threshold:
                return level
        return WatermarkLevel.NORMAL

    def can_accept_task(
        self, resources: SystemResources, task_priority: TaskPriority
    ) -> bool:
        """Decide whether a task with *task_priority* can be accepted now.

        Each priority maps to a watermark level at which it is rejected:

        * CRITICAL — rejected at CRITICAL
        * HIGH / NORMAL — rejected at EMERGENCY+
        * LOW — rejected at CLEANUP+
        * BACKGROUND — rejected at WARNING+
        """
        current_level = self.get_watermark_level(resources.available_ram_gb)
        reject_at = _PRIORITY_REJECTION_MAP.get(task_priority, WatermarkLevel.CRITICAL)
        return current_level < reject_at

    def should_throttle(self, resources: SystemResources) -> bool:
        """Return ``True`` when the system is under heavy pressure.

        Throttling engages at EMERGENCY or CRITICAL.
        """
        level = self.get_watermark_level(resources.available_ram_gb)
        return level >= WatermarkLevel.EMERGENCY

    def suggest_action(self, resources: SystemResources) -> str:
        """Return a human-readable suggested action based on the current watermark."""
        level = self.get_watermark_level(resources.available_ram_gb)
        return _ACTION_MAP.get(level, "no action needed")

    def estimate_agent_ram(self, provider_kind: str) -> float:
        """Estimate per-agent RAM (GB) for a given provider type.

        Lookup is case-insensitive upper-cased.  Unknown provider kinds
        default to 1.5 GB.
        """
        return _PROVIDER_RAM_ESTIMATES.get(provider_kind.upper(), 1.5)

    def get_active_agent_count(
        self, resources: SystemResources, avg_ram_per_agent: float
    ) -> int:
        """Estimate how many agents are currently active based on RAM usage.

        ``used_ram = total_ram - available_ram``.  Returns
        ``floor(used_ram / avg_ram_per_agent)``, clamped to 0.
        """
        used = resources.total_ram_gb - resources.available_ram_gb
        if used <= 0.0:
            return 0
        return int(math.floor(used / avg_ram_per_agent))
