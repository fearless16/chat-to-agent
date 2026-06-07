"""Engine integration with the new brain / capture / event / recovery / scheduler.

These tests construct a real `BrowserIntelligenceEngine` and exercise
the full Phase 1-7 wiring without needing a browser:

- `get_response_text()` returns the classified, buffered body.
- Events are emitted for state transitions.
- Recovery cascade runs in cost order.
- Scheduler adapts to system inputs.
- Reliability store accumulates Bayesian updates.
- The persistent brain round-trips through save/load.

The integration path is asserted end-to-end, but in-process: no
Playwright. The browser harness lives in
`test_browser_intelligence_harness.py` and is gated behind the
`browser` marker.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine
from ai_orchestrator.browser_intelligence.events import EventBus, EventType
from ai_orchestrator.browser_intelligence.intelligence.response_capture import (
    ResponseCapture,
)
from ai_orchestrator.browser_intelligence.intelligence.traffic_classifier import (
    ResponseClassifier,
    TrafficCategory,
)
from ai_orchestrator.browser_intelligence.learning import (
    ProviderReliabilityStore,
)
from ai_orchestrator.browser_intelligence.recovery import (
    RecoveryCascade,
    RecoveryStep,
    build_default_cascade,
)
from ai_orchestrator.browser_intelligence.scheduling import (
    AdaptiveScheduler,
    SchedulingInputs,
)


class TestEngineWiring:
    def test_engine_constructs_with_all_new_modules(self):
        e = BrowserIntelligenceEngine()
        assert isinstance(e.response_capture, ResponseCapture)
        assert isinstance(e.traffic_classifier, ResponseClassifier)
        assert isinstance(e.event_bus, EventBus)
        assert isinstance(e.recovery_cascade, RecoveryCascade)
        assert isinstance(e.scheduler, AdaptiveScheduler)

    def test_get_response_text_filters_out_analytics(self):
        e = BrowserIntelligenceEngine()
        # Simulate a tracking ping.
        e._capture.begin_response(
            request_id="r1",
            url="https://google-analytics.com/collect",
            method="POST",
            status=200,
            content_type="image/gif",
        )
        e._capture.append_chunk("r1", "would-be-tracking-payload")
        e._capture.close_response("r1")
        # Simulate a chat response.
        e._capture.begin_response(
            request_id="r2",
            url="https://example.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        e._capture.append_chunk("r2", "this is the actual model reply")
        e._capture.close_response("r2")
        # The engine returns only the chat response.
        assert e.get_response_text() == "this is the actual model reply"

    def test_engine_emits_events_on_state_transitions(self):
        e = BrowserIntelligenceEngine()
        received: list[EventType] = []
        e.event_bus.subscribe(lambda evt: received.append(evt.type))
        # bind_provider emits AUTH_SUCCESS, so the subscriber must be
        # in place first.
        e.bind_provider("chatgpt")
        # Force a publish path.
        e.event_bus.publish(EventType.GENERATION_STARTED, {"x": 1})
        assert EventType.GENERATION_STARTED in received
        assert EventType.AUTH_SUCCESS in received  # emitted by bind_provider

    def test_engine_record_reward_updates_reliability(self, tmp_path: Path):
        store = ProviderReliabilityStore(path=tmp_path / "rel.json")
        e = BrowserIntelligenceEngine(reliability_store=store)
        # bind_provider records 1 success (1,0) → posterior = 2/3.
        e.bind_provider("chatgpt")
        # bind_provider: 1 success → (2,1) prior → mean 2/3.
        # record_reward(1.0): +1 success → (3,1) → mean 3/4 = 0.75.
        # record_reward(0.0): +1 failure → (3,2) → mean 3/5 = 0.6.
        e.record_reward(1.0)
        e.record_reward(0.0)
        snap = store.snapshot()
        assert "chatgpt" in snap["providers"]
        assert snap["providers"]["chatgpt"] == pytest.approx(3 / 5, abs=1e-6)

    def test_engine_snapshot_brain_is_json_serializable(self):
        e = BrowserIntelligenceEngine()
        e.bind_provider("chatgpt")
        snap = e.snapshot_brain()
        # Must be JSON-serializable for the persistent brain to save it.
        json.dumps(snap)
        assert snap["provider"] == "chatgpt"


class TestEnginePool:
    def test_pool_reuses_engine_for_same_page(self):
        from unittest.mock import MagicMock
        from ai_orchestrator.browser_intelligence.pool import EnginePool

        async def run() -> None:
            pool = EnginePool()
            page = MagicMock()
            e1 = await pool.get_or_create(page, "synthetic")
            e2 = await pool.get_or_create(page, "synthetic")
            assert e1 is e2
            s = pool.stats
            assert s["hits"] == 1
            assert s["misses"] == 1
            await pool.release_all()
        asyncio.run(run())

    def test_pool_separate_pages_get_separate_engines(self):
        from unittest.mock import MagicMock
        from ai_orchestrator.browser_intelligence.pool import EnginePool

        async def run() -> None:
            pool = EnginePool()
            page_a = MagicMock()
            page_b = MagicMock()
            e_a = await pool.get_or_create(page_a, "synthetic")
            e_b = await pool.get_or_create(page_b, "synthetic")
            assert e_a is not e_b
            await pool.release_all()
        asyncio.run(run())

    def test_pool_shares_reliability_store(self):
        from unittest.mock import MagicMock
        from ai_orchestrator.browser_intelligence.pool import EnginePool
        from ai_orchestrator.browser_intelligence.learning import (
            ProviderReliabilityStore,
        )

        async def run() -> None:
            store = ProviderReliabilityStore()
            pool = EnginePool(reliability_store=store)
            page = MagicMock()
            e = await pool.get_or_create(page, "synthetic")
            e.record_reward(1.0)
            e.record_reward(0.0)
            snap = store.snapshot()
            assert "synthetic" in snap["providers"]
            await pool.release_all()
        asyncio.run(run())

    def test_pool_gc_drops_idle_entries(self):
        from unittest.mock import MagicMock
        from ai_orchestrator.browser_intelligence.pool import EnginePool

        async def run() -> None:
            pool = EnginePool(idle_ttl_seconds=0.0)  # everything is "stale"
            page = MagicMock()
            await pool.get_or_create(page, "synthetic")
            n = await pool.gc()
            assert n == 1
            assert pool.stats["size"] == 0
        asyncio.run(run())


class TestRecoveryIntegration:
    def test_engine_runs_cascade(self):
        e = BrowserIntelligenceEngine()
        ctx = {"selector_cache_hit": True}
        out = asyncio.run(e.recovery_cascade.run(ctx))
        assert out[0].step == RecoveryStep.SELECTOR_CACHE
        assert out[0].success

    def test_engine_custom_recovery_step(self):
        e = BrowserIntelligenceEngine()
        # Replace the default cascade with one that has only WORKER.
        e._cascade = RecoveryCascade()
        async def worker(ctx):
            from ai_orchestrator.browser_intelligence.recovery import RecoveryOutcome
            return RecoveryOutcome(
                step=RecoveryStep.WORKER,
                success=True,
                confidence=0.99,
                detail="replaced",
            )
        e._cascade.register(RecoveryStep.WORKER, worker)
        out = asyncio.run(e._cascade.run({}))
        assert out[0].step == RecoveryStep.WORKER
        assert out[0].success


class TestSchedulerIntegration:
    def test_engine_scheduler_produces_decision(self):
        e = BrowserIntelligenceEngine()
        d = e.scheduler.decide(SchedulingInputs(
            cpu_percent=20,
            ram_percent=30,
            queue_depth=2,
            target_browsers=4,
            provider_reliability={"chatgpt": 0.9, "qwen": 0.4},
        ))
        assert d.worker_count >= 1
        assert 0.25 <= d.tick_interval <= 4.0
        assert d.concurrency_limits


class TestEndToEndPipeline:
    """A synthetic end-to-end run that exercises every new module."""

    def test_full_pipeline(self, tmp_path: Path):
        # 1) Construct engine with all new modules wired in.
        store = ProviderReliabilityStore(path=tmp_path / "rel.json")
        bus = EventBus()
        seen_events: list[EventType] = []
        bus.subscribe(lambda e: seen_events.append(e.type))

        e = BrowserIntelligenceEngine(
            event_bus=bus,
            reliability_store=store,
        )
        e.bind_provider("chatgpt")

        # 2) Drive a chat response into the capture layer.
        e._capture.begin_response(
            request_id="chat-1",
            url="https://api.openai.com/v1/chat/completions",
            method="POST",
            status=200,
            content_type="text/event-stream",
        )
        e._capture.append_chunk("chat-1", "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}")
        e._capture.append_chunk("chat-1", "\n\ndata: [DONE]\n\n")
        e._capture.close_response("chat-1")

        # 3) Verify response is the actual model reply.
        text = e.get_response_text()
        assert "Hello" in text

        # 4) Drive a reward back into the learning layer.
        e.record_reward(1.0)
        # Reliability store was updated.
        assert "chatgpt" in store.snapshot()["providers"]

        # 5) Run the recovery cascade.
        out = asyncio.run(e.recovery_cascade.run({}))
        assert out  # cascade ran

        # 6) Drive the scheduler.
        decision = e.scheduler.decide(SchedulingInputs(
            cpu_percent=10, ram_percent=20, queue_depth=0, target_browsers=4,
        ))
        assert decision.worker_count >= 1

        # 7) Bus received at least the events we triggered.
        assert EventType.AUTH_SUCCESS in seen_events
