"""Tests for events bus, learning, recovery, scheduling, drift, shadow-ban, brain, stealth."""

from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

import pytest

from ai_orchestrator.browser_intelligence.events import EventBus, EventType
from ai_orchestrator.browser_intelligence.intelligence.drift_detector import (
    DriftDetector,
    DriftSignal,
)
from ai_orchestrator.browser_intelligence.intelligence.provider_brain import (
    ProviderBrain,
)
from ai_orchestrator.browser_intelligence.intelligence.shadow_ban_detector import (
    ShadowBanDetector,
    ShadowBanState,
)
from ai_orchestrator.browser_intelligence.intelligence.stealth import (
    apply_stealth,
    make_stealth_profile,
    stealth_context_options,
    stealth_init_script,
    stealth_launch_args,
)
from ai_orchestrator.browser_intelligence.learning import (
    BayesianReliability,
    ProviderReliabilityStore,
    beta_mean,
    beta_update,
    beta_variance,
)
from ai_orchestrator.browser_intelligence.recovery import (
    CASCADE_ORDER,
    RecoveryCascade,
    RecoveryStep,
    build_default_cascade,
)
from ai_orchestrator.browser_intelligence.scheduling import (
    AdaptiveScheduler,
    SchedulingInputs,
)


# ──────────────────────────────────────────────────────────────────────
# Event bus
# ──────────────────────────────────────────────────────────────────────

class TestEventBus:
    def test_publish_delivers_to_subscribers(self):
        bus = EventBus()
        received: list[EventType] = []
        bus.subscribe(lambda e: received.append(e.type))
        bus.publish(EventType.GENERATION_STARTED, {"x": 1})
        bus.publish(EventType.GENERATION_COMPLETED, {"y": 2})
        assert received == [
            EventType.GENERATION_STARTED,
            EventType.GENERATION_COMPLETED,
        ]

    def test_unsubscribe_stops_delivery(self):
        bus = EventBus()
        received: list[EventType] = []
        unsub = bus.subscribe(lambda e: received.append(e.type))
        bus.publish(EventType.AUTH_SUCCESS)
        unsub()
        bus.publish(EventType.AUTH_FAILURE)
        assert received == [EventType.AUTH_SUCCESS]

    def test_async_subscribers_run(self):
        bus = EventBus()
        received: list[EventType] = []
        async def handler(e):
            received.append(e.type)
        bus.subscribe(handler, is_async=True)
        evt = bus.publish(EventType.GENERATION_STARTED)
        asyncio.run(bus.dispatch_async(evt))
        assert received == [EventType.GENERATION_STARTED]

    def test_subscriber_exception_does_not_break_others(self):
        bus = EventBus()
        received: list[EventType] = []
        def bad(e):
            raise RuntimeError("nope")
        def good(e):
            received.append(e.type)
        bus.subscribe(bad)
        bus.subscribe(good)
        bus.publish(EventType.GENERATION_STARTED)
        assert received == [EventType.GENERATION_STARTED]

    def test_history_ring_buffer(self):
        bus = EventBus(history_limit=3)
        for i in range(5):
            bus.publish(EventType.GENERIC, {"i": i})
        h = bus.history()
        assert len(h) == 3
        assert h[0].payload["i"] == 2

    def test_event_has_unique_id(self):
        bus = EventBus()
        e1 = bus.publish(EventType.GENERIC)
        e2 = bus.publish(EventType.GENERIC)
        assert e1.event_id != e2.event_id

    def test_stats(self):
        bus = EventBus()
        bus.subscribe(lambda e: None)
        bus.publish(EventType.GENERIC)
        s = bus.stats()
        assert s["emitted"] == 1
        assert s["subscribers"] == 1


# ──────────────────────────────────────────────────────────────────────
# Bayesian learning
# ──────────────────────────────────────────────────────────────────────

class TestBetaMath:
    def test_beta_mean_uninformative_prior(self):
        # Beta(1,1) is uniform: mean = 0.5
        assert beta_mean(1.0, 1.0) == pytest.approx(0.5)

    def test_beta_mean_with_one_success(self):
        # Beta(2,1): mean = 2/3
        assert beta_mean(2.0, 1.0) == pytest.approx(2 / 3)

    def test_beta_variance_uninformative(self):
        # Beta(1,1): var = 1/12
        assert beta_variance(1.0, 1.0) == pytest.approx(1 / 12)

    def test_beta_update_success_increments_alpha(self):
        a, b = beta_update(1.0, 1.0, success=True)
        assert a == 2.0
        assert b == 1.0

    def test_beta_update_failure_increments_beta(self):
        a, b = beta_update(1.0, 1.0, success=False)
        assert a == 1.0
        assert b == 2.0


class TestBayesianReliability:
    def test_initial_state_uniform(self):
        r = BayesianReliability(key="x")
        assert r.posterior_mean == pytest.approx(0.5)
        assert r.total_observations == 0

    def test_update_moves_mean(self):
        r = BayesianReliability(key="x")
        for _ in range(9):
            r.update(True)
        r.update(False)
        # 9 successes, 1 failure on Beta(1,1) prior → 10/12
        assert r.posterior_mean == pytest.approx(10 / 12)

    def test_credible_interval_within_unit_interval(self):
        r = BayesianReliability(key="x")
        for _ in range(20):
            r.update(True)
        lo, hi = r.credible_interval()
        assert 0.0 <= lo <= hi <= 1.0
        # Mean is ~0.95, CI should contain it.
        assert lo <= r.posterior_mean <= hi

    def test_exponential_decay_reduces_effective_counts(self):
        import time as _time
        r = BayesianReliability(key="x")
        for _ in range(100):
            r.update(True)
        mean_before = r.posterior_mean
        # Configure slow decay — 0.1% per hour.
        r.configure_decay(rate=0.001, interval_seconds=60)
        # Artificially age the last_updated timestamp.
        r.last_updated = _time.time() - 6000  # 100 minutes ago
        r.apply_decay()
        mean_after = r.posterior_mean
        # Effective counts should have decayed slightly.
        assert r._effective_successes < 100.0
        assert mean_after < mean_before

    def test_decay_inactive_when_not_configured(self):
        r = BayesianReliability(key="x")
        for _ in range(50):
            r.update(True)
        mean_before = r.posterior_mean
        r.apply_decay()
        mean_after = r.posterior_mean
        assert mean_before == mean_after

    def test_decay_converges_toward_prior(self):
        import time as _time
        r = BayesianReliability(key="x")
        for _ in range(100):
            r.update(True)
        r.configure_decay(rate=10.0, interval_seconds=1)
        r.last_updated = _time.time() - 2  # 2 periods of heavy decay
        r.apply_decay()
        # Heavy decay should pull mean close to 0.5 (the Beta(1,1) prior).
        assert r.posterior_mean < 0.7


class TestProviderReliabilityStore:
    def test_record_and_get(self, tmp_path: Path):
        store = ProviderReliabilityStore(path=tmp_path / "rel.json")
        rel = store.record("provider", "chatgpt", success=True)
        rel2 = store.record("provider", "chatgpt", success=True)
        # Same key, accumulating Beta counts.
        assert rel is rel2
        assert rel.successes == 2
        assert rel.posterior_mean > 0.5

    def test_persistence_round_trip(self, tmp_path: Path):
        path = tmp_path / "rel.json"
        s1 = ProviderReliabilityStore(path=path)
        s1.record("provider", "chatgpt", success=True)
        s1.record("account", "acct-1", success=False)
        s1.save()

        s2 = ProviderReliabilityStore(path=path)
        # Look up the same keys — they should be present.
        assert s2.get("provider", "chatgpt").successes == 1
        assert s2.get("account", "acct-1").failures == 1

    def test_snapshot_returns_means(self):
        s = ProviderReliabilityStore()
        s.record("provider", "p1", success=True)
        s.record("provider", "p1", success=True)
        s.record("provider", "p1", success=False)
        snap = s.snapshot()
        assert "p1" in snap["providers"]
        # 2 success + 1 failure on Beta(1,1) prior → mean 3/5
        assert snap["providers"]["p1"] == pytest.approx(3 / 5)

    def test_missing_path_works(self, tmp_path: Path):
        # No file → no error.
        store = ProviderReliabilityStore(path=tmp_path / "missing.json")
        rel = store.get("provider", "x")
        assert rel.posterior_mean == 0.5

    def test_corrupt_file_does_not_crash(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        store = ProviderReliabilityStore(path=path)
        # Should fall back to empty store.
        assert store.get("provider", "x").posterior_mean == 0.5


# ──────────────────────────────────────────────────────────────────────
# Recovery cascade
# ──────────────────────────────────────────────────────────────────────

class TestRecoveryCascade:
    def test_default_cascade_registers_all_steps(self):
        c = build_default_cascade()
        for step in CASCADE_ORDER:
            assert step in c.handlers

    def test_cascade_picks_cheapest_valid_step(self):
        c = RecoveryCascade()
        # Cheapest step (selector cache) succeeds → cascade should stop.
        c.register(RecoveryStep.SELECTOR_CACHE, lambda ctx: _ok(RecoveryStep.SELECTOR_CACHE))
        c.register(RecoveryStep.A11Y, lambda ctx: _ok(RecoveryStep.A11Y))
        out = asyncio.run(c.run({"selector_cache_hit": True}))
        # First step succeeds → no further steps taken.
        assert len(out) == 1
        assert out[0].step == RecoveryStep.SELECTOR_CACHE

    def test_cascade_falls_through_to_more_expensive(self):
        c = RecoveryCascade()
        c.register(RecoveryStep.SELECTOR_CACHE, lambda ctx: _fail(RecoveryStep.SELECTOR_CACHE))
        c.register(RecoveryStep.A11Y, lambda ctx: _ok(RecoveryStep.A11Y))
        c.register(RecoveryStep.GRAPH, lambda ctx: _ok(RecoveryStep.GRAPH))
        out = asyncio.run(c.run({}))
        # First fails, second succeeds.
        assert len(out) == 2
        assert out[0].step == RecoveryStep.SELECTOR_CACHE
        assert out[1].step == RecoveryStep.A11Y
        assert out[1].success

    def test_cascade_low_confidence_continues(self):
        c = RecoveryCascade(min_confidence=0.95)
        c.register(RecoveryStep.SELECTOR_CACHE, lambda ctx: _ok(RecoveryStep.SELECTOR_CACHE, conf=0.3))
        c.register(RecoveryStep.A11Y, lambda ctx: _ok(RecoveryStep.A11Y, conf=0.99))
        out = asyncio.run(c.run({}))
        # First step's success is below threshold → cascade continues.
        assert out[-1].step == RecoveryStep.A11Y

    def test_cascade_records_outcome_history(self):
        c = build_default_cascade()
        out = asyncio.run(c.run({}))
        assert len(c.history) == len(out)
        for o in c.history:
            assert o.cost > 0

    def test_cascade_skips_unregistered_steps(self):
        c = RecoveryCascade()
        c.register(RecoveryStep.WORKER, lambda ctx: _ok(RecoveryStep.WORKER))
        out = asyncio.run(c.run({}))
        # WORKER is expensive, but only it is registered → cascade runs it.
        assert out[-1].step == RecoveryStep.WORKER

    def test_cascade_handles_async_handler(self):
        c = RecoveryCascade()
        async def ah(ctx):
            return _ok(RecoveryStep.SELECTOR_CACHE)
        c.register(RecoveryStep.SELECTOR_CACHE, ah)
        out = asyncio.run(c.run({}))
        assert out[0].success

    def test_cascade_handler_exception_continues(self):
        c = RecoveryCascade()
        def bad(ctx):
            raise RuntimeError("boom")
        c.register(RecoveryStep.SELECTOR_CACHE, bad)
        c.register(RecoveryStep.A11Y, lambda ctx: _ok(RecoveryStep.A11Y))
        out = asyncio.run(c.run({}))
        assert out[0].step == RecoveryStep.SELECTOR_CACHE
        assert out[0].success is False
        assert out[1].step == RecoveryStep.A11Y

    def test_cascade_cost_is_ascending(self):
        costs = [s.cost for s in CASCADE_ORDER]
        for i in range(1, len(costs)):
            assert costs[i] >= costs[i - 1], f"step {CASCADE_ORDER[i]} is cheaper than {CASCADE_ORDER[i-1]}"


def _ok(step, conf=0.99):
    from ai_orchestrator.browser_intelligence.recovery import RecoveryOutcome
    return RecoveryOutcome(step=step, success=True, confidence=conf, detail="ok")


def _fail(step, conf=0.0):
    from ai_orchestrator.browser_intelligence.recovery import RecoveryOutcome
    return RecoveryOutcome(step=step, success=False, confidence=conf, detail="fail")


# ──────────────────────────────────────────────────────────────────────
# Scheduling
# ──────────────────────────────────────────────────────────────────────

class TestAdaptiveScheduler:
    def test_idle_system_scales_to_target(self):
        s = AdaptiveScheduler()
        d = s.decide(SchedulingInputs(
            cpu_percent=10,
            ram_percent=20,
            queue_depth=0,
            target_browsers=4,
        ))
        assert d.worker_count == 4

    def test_loaded_system_scales_down(self):
        s = AdaptiveScheduler()
        d = s.decide(SchedulingInputs(
            cpu_percent=95,
            ram_percent=95,
            queue_depth=0,
            target_browsers=8,
        ))
        assert d.worker_count <= 2  # min_workers=1, so 1 or 2

    def test_queue_pressure_adds_workers(self):
        s = AdaptiveScheduler()
        d = s.decide(SchedulingInputs(
            cpu_percent=30,
            ram_percent=40,
            queue_depth=20,
            target_browsers=4,
        ))
        # Health 0.4*0.7 + 0.6*0.6 = 0.64 → health_med branch.
        # desired = max(1, 4//2) = 2; queue adds workers.
        assert d.worker_count >= 2

    def test_tick_interval_within_bounds(self):
        s = AdaptiveScheduler()
        for cpu, ram in [(0, 0), (50, 50), (100, 100)]:
            d = s.decide(SchedulingInputs(
                cpu_percent=cpu,
                ram_percent=ram,
                queue_depth=0,
                target_browsers=4,
            ))
            assert 0.25 <= d.tick_interval <= 4.0

    def test_high_reliability_provider_gets_more_concurrency(self):
        s = AdaptiveScheduler(default_concurrency=2)
        d = s.decide(SchedulingInputs(
            provider_reliability={"p1": 0.95, "p2": 0.30, "p3": 0.65},
            queue_depth=0,
            target_browsers=4,
        ))
        assert d.concurrency_limits["p1"] >= 3  # default + 2
        assert d.concurrency_limits["p2"] == 1
        assert d.concurrency_limits["p3"] == 2  # default

    def test_health_score_in_unit_interval(self):
        si = SchedulingInputs(cpu_percent=50, ram_percent=50)
        assert 0.0 <= si.health_score() <= 1.0

    def test_no_fixed_concurrency(self):
        # Concurrency is a function of inputs, not a constant.
        s = AdaptiveScheduler()
        a = s.decide(SchedulingInputs(target_browsers=4))
        b = s.decide(SchedulingInputs(target_browsers=8))
        # Different targets → different worker counts.
        assert a.worker_count != b.worker_count


# ──────────────────────────────────────────────────────────────────────
# Drift detector
# ──────────────────────────────────────────────────────────────────────

class TestDriftDetector:
    def test_no_data_zero_drift(self):
        d = DriftDetector()
        snap = d.snapshot()
        assert snap.drift_score == 0.0
        assert snap.should_relearn is False

    def test_repeated_signals_low_drift(self):
        d = DriftDetector()
        for _ in range(50):
            d.observe(DriftSignal(kind="dom", fingerprint="role:textbox"))
        snap = d.snapshot()
        assert snap.drift_score < 0.35
        assert snap.should_relearn is False

    def test_novel_signals_high_drift(self):
        d = DriftDetector()
        for i in range(50):
            d.observe(DriftSignal(kind="dom", fingerprint=f"newshape-{i}"))
        snap = d.snapshot()
        # 50 different fingerprints in 50 observations → very high novelty.
        assert snap.drift_score > 0.5
        assert snap.should_relearn is True

    def test_threshold_is_respected(self):
        # At threshold=1.5 the score can never trigger relearn, no
        # matter how novel the signals are.
        d = DriftDetector(threshold=1.5)
        for i in range(20):
            d.observe(DriftSignal(kind="dom", fingerprint=f"shape-{i}"))
        snap = d.snapshot()
        # Even with 20 novel signals, threshold=1.5 → no relearn.
        assert snap.should_relearn is False

    def test_export_contains_known_fingerprints(self):
        d = DriftDetector()
        d.observe(DriftSignal(kind="dom", fingerprint="x"))
        d.observe(DriftSignal(kind="network", fingerprint="y"))
        e = d.export()
        assert "x" in e["known"]["dom"]
        assert "y" in e["known"]["network"]

    def test_seed_history_prevents_false_novelty(self):
        d = DriftDetector()
        d.seed_history([
            DriftSignal(kind="dom", fingerprint="oldshape"),
        ])
        for _ in range(20):
            d.observe(DriftSignal(kind="dom", fingerprint="oldshape"))
        snap = d.snapshot()
        # Already-known signal → novelty contribution is low.
        assert snap.drift_score < 0.4


# ──────────────────────────────────────────────────────────────────────
# Shadow-ban detector
# ──────────────────────────────────────────────────────────────────────

class TestShadowBanDetector:
    def test_insufficient_data_returns_normal(self):
        s = ShadowBanDetector()
        v = s.observe(
            response_length=100,
            completion_rate=1.0,
            tokens_per_second=20.0,
        )
        assert v.state == ShadowBanState.NORMAL
        assert "insufficient_data" in v.reasons

    def test_short_truncated_response_drift_toward_shadow(self):
        s = ShadowBanDetector()
        # Build a baseline of "normal" responses.
        for _ in range(15):
            s.observe(
                response_length=2000,
                completion_rate=1.0,
                tokens_per_second=30.0,
            )
        # Now observe a much shorter one.
        v = s.observe(
            response_length=50,
            completion_rate=0.3,
            tokens_per_second=1.0,
        )
        # After many normal observations, a sudden short+stalled one
        # should bump the posterior but only one outlier isn't enough
        # to flip state.
        assert v.p_shadow_ban > 0.0

    def test_persistent_shadow_ban_pattern_triggers_state(self):
        s = ShadowBanDetector()
        for _ in range(15):
            s.observe(
                response_length=2000,
                completion_rate=1.0,
                tokens_per_second=30.0,
            )
        # Many degraded observations in a row.
        for _ in range(10):
            v = s.observe(
                response_length=20,
                completion_rate=0.2,
                tokens_per_second=1.0,
                error_count=3,
                quality_score=0.05,
            )
        assert v.state == ShadowBanState.SHADOW_BANNED
        assert v.p_shadow_ban > 0.5

    def test_degraded_state_with_moderate_evidence(self):
        s = ShadowBanDetector()
        for _ in range(20):
            s.observe(
                response_length=2000,
                completion_rate=1.0,
                tokens_per_second=30.0,
            )
        # Mildly degraded observations.
        for _ in range(5):
            v = s.observe(
                response_length=1500,
                completion_rate=0.85,
                tokens_per_second=20.0,
                quality_score=0.4,
            )
        # Not full ban, but some signal.
        assert v.state in (ShadowBanState.NORMAL, ShadowBanState.DEGRADED)

    def test_posterior_sums_to_one(self):
        s = ShadowBanDetector()
        for _ in range(15):
            s.observe(response_length=2000, completion_rate=1.0, tokens_per_second=30.0)
        for _ in range(5):
            v = s.observe(
                response_length=200,
                completion_rate=0.3,
                tokens_per_second=2.0,
                error_count=2,
            )
        s_sum = v.p_normal + v.p_degraded + v.p_shadow_ban
        assert s_sum == pytest.approx(1.0, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────
# Provider brain
# ──────────────────────────────────────────────────────────────────────

class TestProviderBrain:
    def test_load_empty_path_creates_fresh_brain(self, tmp_path: Path):
        brain = ProviderBrain("chatgpt", tmp_path / "chatgpt.brain.json")
        brain.load()
        assert brain.session_count == 0

    def test_session_count_increments(self, tmp_path: Path):
        brain = ProviderBrain("chatgpt", tmp_path / "chatgpt.brain.json")
        brain.load()
        brain.record_session_start()
        brain.record_session_start()
        assert brain.session_count == 2

    def test_emission_calibration_blends(self, tmp_path: Path):
        brain = ProviderBrain("chatgpt", tmp_path / "chatgpt.brain.json")
        brain.load()
        brain.record_emission_calibration({"READY": 0.8, "GENERATING": 0.6})
        brain.record_emission_calibration({"READY": 1.0, "GENERATING": 0.4})
        cal = brain.emission_calibration
        # EMA at α=0.3 → 0.7*0.8 + 0.3*1.0 = 0.86
        assert cal["READY"] == pytest.approx(0.86, abs=1e-6)
        assert cal["GENERATING"] == pytest.approx(0.54, abs=1e-6)

    def test_selector_reliability_bayesian(self, tmp_path: Path):
        brain = ProviderBrain("chatgpt", tmp_path / "chatgpt.brain.json")
        brain.load()
        for _ in range(9):
            brain.record_selector_outcome("button.send", True)
        brain.record_selector_outcome("button.send", False)
        mean = brain.state.selector_reliability["button.send"]["mean"]
        # Beta(10,2) posterior mean: 10/12
        assert mean == pytest.approx(10 / 12, abs=1e-6)

    def test_persistence_round_trip(self, tmp_path: Path):
        path = tmp_path / "chatgpt.brain.json"
        b1 = ProviderBrain("chatgpt", path)
        b1.load()
        b1.record_session_start()
        b1.record_emission_calibration({"READY": 0.9})
        b1.record_selector_outcome("button.send", True)
        b1.save()

        b2 = ProviderBrain("chatgpt", path)
        b2.load()
        assert b2.session_count == 1
        assert b2.emission_calibration["READY"] == 0.9
        assert b2.state.selector_reliability["button.send"]["mean"] > 0.5

    def test_state_priors_renormalize(self, tmp_path: Path):
        brain = ProviderBrain("chatgpt", tmp_path / "chatgpt.brain.json")
        brain.load()
        brain.record_state_priors({"READY": 3, "GENERATING": 1})
        priors = brain.state_priors
        s = sum(priors.values())
        assert s == pytest.approx(1.0, abs=1e-6)
        assert priors["READY"] == pytest.approx(0.75)

    def test_recovery_history_caps_at_100(self, tmp_path: Path):
        brain = ProviderBrain("chatgpt", tmp_path / "chatgpt.brain.json")
        brain.load()
        for i in range(150):
            brain.record_recovery_outcome("a11y", True, 0.9)
        assert len(brain.state.recovery_history) == 100

    def test_factory_helpers(self, tmp_path: Path):
        chat = ProviderBrain.chatgpt(tmp_path)
        qwen = ProviderBrain.qwen(tmp_path)
        kimi = ProviderBrain.kimi(tmp_path)
        ds = ProviderBrain.deepseek(tmp_path)
        zai = ProviderBrain.zai(tmp_path)
        assert (chat._path.name == "chatgpt.brain.json")
        assert (qwen._path.name == "qwen.brain.json")
        assert (kimi._path.name == "kimi.brain.json")
        assert (ds._path.name == "deepseek.brain.json")
        assert (zai._path.name == "zai.brain.json")


# ──────────────────────────────────────────────────────────────────────
# Stealth
# ──────────────────────────────────────────────────────────────────────

class TestStealth:
    def test_profile_is_deterministic_given_seed(self):
        a = make_stealth_profile(seed=42)
        b = make_stealth_profile(seed=42)
        assert a.fingerprint() == b.fingerprint()
        assert a.webgl_vendor == b.webgl_vendor
        assert a.webgl_renderer == b.webgl_renderer

    def test_different_seeds_give_different_profiles(self):
        a = make_stealth_profile(seed=1)
        b = make_stealth_profile(seed=2)
        assert a.fingerprint() != b.fingerprint()

    def test_launch_args_include_anti_automation(self):
        args = stealth_launch_args()
        assert any("AutomationControlled" in a for a in args)

    def test_init_script_contains_stealth_marker(self):
        profile = make_stealth_profile(seed=1)
        script = stealth_init_script(profile)
        assert "__bis_stealth_v1" in script
        assert "webdriver" in script
        assert "canvas" in script.lower() or "toDataURL" in script

    def test_context_options_contain_locale_timezone_ua(self):
        profile = make_stealth_profile(seed=1)
        opts = stealth_context_options(profile)
        assert "user_agent" in opts
        assert "timezone_id" in opts
        assert "locale" in opts
        assert "viewport" in opts
        assert opts["timezone_id"] == profile.timezone
        assert opts["locale"] == profile.languages[0]

    def test_apply_stealth_returns_application(self):
        app = apply_stealth(seed=99)
        assert app.profile.fingerprint()
        assert app.launch_args
        assert app.context_options
        assert app.init_script

    def test_hardware_concurrency_is_spoofed(self):
        profile = make_stealth_profile(seed=7)
        assert profile.hardware_concurrency in (4, 8, 12, 16)

    def test_languages_includes_primary_and_secondary(self):
        profile = make_stealth_profile(seed=1)
        assert len(profile.languages) >= 2
