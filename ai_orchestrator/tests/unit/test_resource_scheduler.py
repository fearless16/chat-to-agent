"""Tests for ResourceScheduler — resource-aware task scheduling."""

from ai_orchestrator.models.task import TaskPriority
from ai_orchestrator.orchestrator.resource_scheduler import (
    ResourceScheduler,
    SystemResources,
    WatermarkLevel,
)


class TestSystemResources:
    """SystemResources dataclass construction and defaults."""

    def test_minimal_construction(self):
        """All fields are required and positional."""
        sr = SystemResources(
            total_ram_gb=16.0,
            available_ram_gb=8.0,
            total_cores=8,
            available_cores=4.0,
            memory_usage_percent=50.0,
            cpu_usage_percent=25.0,
        )
        assert sr.total_ram_gb == 16.0
        assert sr.available_ram_gb == 8.0
        assert sr.total_cores == 8
        assert sr.available_cores == 4.0
        assert sr.memory_usage_percent == 50.0
        assert sr.cpu_usage_percent == 25.0


class TestWatermarkLevel:
    """WatermarkLevel enum values match the specification."""

    def test_values(self):
        assert WatermarkLevel.NORMAL == 0
        assert WatermarkLevel.WARNING == 1
        assert WatermarkLevel.CLEANUP == 2
        assert WatermarkLevel.EMERGENCY == 3
        assert WatermarkLevel.CRITICAL == 4

    def test_ordering(self):
        """Verified by the int values — higher = worse."""
        assert WatermarkLevel.NORMAL < WatermarkLevel.WARNING < WatermarkLevel.CLEANUP
        assert WatermarkLevel.CLEANUP < WatermarkLevel.EMERGENCY < WatermarkLevel.CRITICAL


class TestGetWatermarkLevel:
    """get_watermark_level returns the correct level for each threshold."""

    def setup_method(self):
        self.scheduler = ResourceScheduler()

    def test_normal_above_3gb(self):
        assert self.scheduler.get_watermark_level(10.0) == WatermarkLevel.NORMAL
        assert self.scheduler.get_watermark_level(3.5) == WatermarkLevel.NORMAL
        assert self.scheduler.get_watermark_level(3.01) == WatermarkLevel.NORMAL

    def test_warning_at_3gb(self):
        assert self.scheduler.get_watermark_level(3.0) == WatermarkLevel.WARNING
        assert self.scheduler.get_watermark_level(2.8) == WatermarkLevel.WARNING
        assert self.scheduler.get_watermark_level(2.51) == WatermarkLevel.WARNING

    def test_cleanup_at_2_5gb(self):
        assert self.scheduler.get_watermark_level(2.5) == WatermarkLevel.CLEANUP
        assert self.scheduler.get_watermark_level(2.3) == WatermarkLevel.CLEANUP
        assert self.scheduler.get_watermark_level(2.01) == WatermarkLevel.CLEANUP

    def test_emergency_at_2gb(self):
        assert self.scheduler.get_watermark_level(2.0) == WatermarkLevel.EMERGENCY
        assert self.scheduler.get_watermark_level(1.8) == WatermarkLevel.EMERGENCY
        assert self.scheduler.get_watermark_level(1.51) == WatermarkLevel.EMERGENCY

    def test_critical_at_1_5gb(self):
        assert self.scheduler.get_watermark_level(1.5) == WatermarkLevel.CRITICAL
        assert self.scheduler.get_watermark_level(1.0) == WatermarkLevel.CRITICAL
        assert self.scheduler.get_watermark_level(0.0) == WatermarkLevel.CRITICAL

    def test_negative_ram_is_critical(self):
        assert self.scheduler.get_watermark_level(-1.0) == WatermarkLevel.CRITICAL
        assert self.scheduler.get_watermark_level(-100.0) == WatermarkLevel.CRITICAL


class TestComputeMaxAgents:
    """compute_max_agents formula follows spec."""

    def setup_method(self):
        self.scheduler = ResourceScheduler(configured_max_agents=20)

    def _res(self, ram: float, cores: int = 8) -> SystemResources:
        return SystemResources(
            total_ram_gb=16.0,
            available_ram_gb=ram,
            total_cores=16,
            available_cores=float(cores),
            memory_usage_percent=50.0,
            cpu_usage_percent=25.0,
        )

    def test_basic_computation(self):
        """AvailRAM=8, avg_ram=1.5 => floor((8-2)/1.5)=4, cores*2=16, min=4."""
        res = self._res(ram=8.0, cores=8)
        assert self.scheduler.compute_max_agents(res) == 4

    def test_ram_limited(self):
        """AvailRAM=4 => floor((4-2)/1.5)=1."""
        res = self._res(ram=4.0, cores=16)
        assert self.scheduler.compute_max_agents(res) == 1

    def test_core_limited(self):
        """AvailRAM=20 => floor(18/1.5)=12, cores*2=4 => min=4."""
        res = self._res(ram=20.0, cores=2)
        assert self.scheduler.compute_max_agents(res) == 4

    def test_browser_max_contexts_limited(self):
        res = self._res(ram=20.0, cores=16)
        assert self.scheduler.compute_max_agents(res, browser_max_contexts=3) == 3

    def test_provider_max_concurrent_limited(self):
        res = self._res(ram=20.0, cores=16)
        assert self.scheduler.compute_max_agents(res, provider_max_concurrent=2) == 2

    def test_configured_max_limited(self):
        scheduler = ResourceScheduler(configured_max_agents=5)
        res = self._res(ram=20.0, cores=16)
        assert scheduler.compute_max_agents(res) == 5

    def test_minimum_agents_is_at_least_one(self):
        """Always allow at least 1 agent even when resources are depleted."""
        res = self._res(ram=2.0, cores=1)
        assert self.scheduler.compute_max_agents(res) >= 1

    def test_exactly_2gb_ram_yields_one_agent(self):
        """floor((2-2)/1.5)=0, min with 1 => 1."""
        res = self._res(ram=2.0, cores=16)
        assert self.scheduler.compute_max_agents(res) == 1

    def test_high_core_count(self):
        """Very high core count should not overflow."""
        res = self._res(ram=100.0, cores=999_999)
        result = self.scheduler.compute_max_agents(res)
        # floor((100-2)/1.5) = 65, cores*2 = 1,999,998, browser=10, provider=20, configured=20
        # min = 20
        assert result == 20

    def test_custom_avg_ram(self):
        """With avg_ram=3.0, floor((10-2)/3.0)=2."""
        res = self._res(ram=10.0, cores=16)
        assert self.scheduler.compute_max_agents(res, avg_ram_per_agent=3.0) == 2


class TestCanAcceptTask:
    """can_accept_task enforces correct priority-based admission."""

    def setup_method(self):
        self.scheduler = ResourceScheduler()

    def _res(self, ram: float) -> SystemResources:
        return SystemResources(
            total_ram_gb=16.0,
            available_ram_gb=ram,
            total_cores=8,
            available_cores=4.0,
            memory_usage_percent=50.0,
            cpu_usage_percent=25.0,
        )

    # --- CRITICAL priority: only blocked at CRITICAL watermark ---

    def test_critical_accepted_at_emergency(self):
        assert self.scheduler.can_accept_task(
            self._res(1.8), TaskPriority.CRITICAL
        ) is True

    def test_critical_rejected_at_critical(self):
        assert self.scheduler.can_accept_task(
            self._res(1.0), TaskPriority.CRITICAL
        ) is False

    def test_critical_accepted_at_normal(self):
        assert self.scheduler.can_accept_task(
            self._res(10.0), TaskPriority.CRITICAL
        ) is True

    # --- NORMAL priority: rejected at EMERGENCY+ ---

    def test_normal_accepted_at_cleanup(self):
        assert self.scheduler.can_accept_task(
            self._res(2.3), TaskPriority.NORMAL
        ) is True

    def test_normal_rejected_at_emergency(self):
        assert self.scheduler.can_accept_task(
            self._res(1.8), TaskPriority.NORMAL
        ) is False

    def test_normal_rejected_at_critical(self):
        assert self.scheduler.can_accept_task(
            self._res(1.0), TaskPriority.NORMAL
        ) is False

    def test_normal_accepted_at_warning(self):
        assert self.scheduler.can_accept_task(
            self._res(2.8), TaskPriority.NORMAL
        ) is True

    # --- LOW priority: rejected at CLEANUP+ ---

    def test_low_accepted_at_warning(self):
        assert self.scheduler.can_accept_task(
            self._res(2.8), TaskPriority.LOW
        ) is True

    def test_low_rejected_at_cleanup(self):
        assert self.scheduler.can_accept_task(
            self._res(2.3), TaskPriority.LOW
        ) is False

    def test_low_rejected_at_emergency(self):
        assert self.scheduler.can_accept_task(
            self._res(1.8), TaskPriority.LOW
        ) is False

    def test_low_rejected_at_critical(self):
        assert self.scheduler.can_accept_task(
            self._res(1.0), TaskPriority.LOW
        ) is False

    # --- BACKGROUND priority: rejected at WARNING+ ---

    def test_background_accepted_at_normal(self):
        assert self.scheduler.can_accept_task(
            self._res(10.0), TaskPriority.BACKGROUND
        ) is True

    def test_background_rejected_at_warning(self):
        assert self.scheduler.can_accept_task(
            self._res(2.8), TaskPriority.BACKGROUND
        ) is False

    def test_background_rejected_at_cleanup(self):
        assert self.scheduler.can_accept_task(
            self._res(2.3), TaskPriority.BACKGROUND
        ) is False

    def test_background_rejected_at_emergency(self):
        assert self.scheduler.can_accept_task(
            self._res(1.8), TaskPriority.BACKGROUND
        ) is False

    def test_background_rejected_at_critical(self):
        assert self.scheduler.can_accept_task(
            self._res(1.0), TaskPriority.BACKGROUND
        ) is False


class TestShouldThrottle:
    """should_throttle returns True at EMERGENCY or CRITICAL."""

    def setup_method(self):
        self.scheduler = ResourceScheduler()

    def _res(self, ram: float) -> SystemResources:
        return SystemResources(
            total_ram_gb=16.0,
            available_ram_gb=ram,
            total_cores=8,
            available_cores=4.0,
            memory_usage_percent=50.0,
            cpu_usage_percent=25.0,
        )

    def test_no_throttle_at_normal(self):
        assert self.scheduler.should_throttle(self._res(10.0)) is False

    def test_no_throttle_at_warning(self):
        assert self.scheduler.should_throttle(self._res(2.8)) is False

    def test_no_throttle_at_cleanup(self):
        assert self.scheduler.should_throttle(self._res(2.3)) is False

    def test_throttle_at_emergency(self):
        assert self.scheduler.should_throttle(self._res(1.8)) is True

    def test_throttle_at_critical(self):
        assert self.scheduler.should_throttle(self._res(1.0)) is True

    def test_throttle_at_zero_ram(self):
        assert self.scheduler.should_throttle(self._res(0.0)) is True

    def test_throttle_at_negative_ram(self):
        assert self.scheduler.should_throttle(self._res(-5.0)) is True


class TestSuggestAction:
    """suggest_action returns appropriate guidance per watermark."""

    def setup_method(self):
        self.scheduler = ResourceScheduler()

    def _res(self, ram: float) -> SystemResources:
        return SystemResources(
            total_ram_gb=16.0,
            available_ram_gb=ram,
            total_cores=8,
            available_cores=4.0,
            memory_usage_percent=50.0,
            cpu_usage_percent=25.0,
        )

    def test_normal(self):
        assert self.scheduler.suggest_action(self._res(10.0)) == "no action needed"

    def test_warning(self):
        assert (
            self.scheduler.suggest_action(self._res(2.8))
            == "reduce low-priority agents"
        )

    def test_cleanup(self):
        assert (
            self.scheduler.suggest_action(self._res(2.3))
            == "suspend idle browsers, flush caches"
        )

    def test_emergency(self):
        assert (
            self.scheduler.suggest_action(self._res(1.8))
            == "pause all non-critical agents, trim memory"
        )

    def test_critical(self):
        assert (
            self.scheduler.suggest_action(self._res(1.0))
            == "freeze new tasks, kill lowest priority agents"
        )


class TestEstimateAgentRam:
    """estimate_agent_ram returns provider-specific estimates."""

    def setup_method(self):
        self.scheduler = ResourceScheduler()

    def test_api_provider(self):
        assert self.scheduler.estimate_agent_ram("API") == 0.5

    def test_browser_provider(self):
        assert self.scheduler.estimate_agent_ram("BROWSER") == 1.5

    def test_local_provider(self):
        assert self.scheduler.estimate_agent_ram("LOCAL") == 3.0

    def test_unknown_provider_defaults_to_1_5(self):
        assert self.scheduler.estimate_agent_ram("UNKNOWN") == 1.5
        assert self.scheduler.estimate_agent_ram("FOO") == 1.5
        assert self.scheduler.estimate_agent_ram("") == 1.5

    def test_case_sensitivity(self):
        """Provider kind is compared case-insensitively."""
        assert self.scheduler.estimate_agent_ram("api") == 0.5
        assert self.scheduler.estimate_agent_ram("Api") == 0.5


class TestGetActiveAgentCount:
    """get_active_agent_count estimates active agents from RAM usage."""

    def setup_method(self):
        self.scheduler = ResourceScheduler()

    def _res(self, ram: float, total: float = 16.0) -> SystemResources:
        return SystemResources(
            total_ram_gb=total,
            available_ram_gb=ram,
            total_cores=8,
            available_cores=4.0,
            memory_usage_percent=(total - ram) / total * 100,
            cpu_usage_percent=25.0,
        )

    def test_estimate_with_0_5gb_per_agent(self):
        """Used RAM = 16-8 = 8GB, /0.5 = 16 agents."""
        res = self._res(ram=8.0, total=16.0)
        assert self.scheduler.get_active_agent_count(res, avg_ram_per_agent=0.5) == 16

    def test_estimate_with_1_5gb_per_agent(self):
        """Used RAM = 16-8 = 8GB, /1.5 = 5 (floor)."""
        res = self._res(ram=8.0, total=16.0)
        assert self.scheduler.get_active_agent_count(res, avg_ram_per_agent=1.5) == 5

    def test_zero_used_ram(self):
        """When all RAM is available, used = 0, count = 0."""
        res = self._res(ram=16.0, total=16.0)
        assert self.scheduler.get_active_agent_count(res, avg_ram_per_agent=1.5) == 0

    def test_negative_available_ram(self):
        """If available > total (shouldn't happen), clamp to 0."""
        res = self._res(ram=20.0, total=16.0)
        assert self.scheduler.get_active_agent_count(res, avg_ram_per_agent=1.5) == 0

    def test_high_usage(self):
        """Used = 15GB, /1.5 = 10."""
        res = self._res(ram=1.0, total=16.0)
        assert self.scheduler.get_active_agent_count(res, avg_ram_per_agent=1.5) == 10
