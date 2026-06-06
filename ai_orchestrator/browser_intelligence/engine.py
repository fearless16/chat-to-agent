"""BrowserIntelligenceEngine — top-level orchestrator for the Browser Intelligence OS.

Orchestrates the full pipeline:
    Sensor → Feature → Estimation → Decision → Action

Phase 1 hardening: response text is now sourced from
`intelligence.ResponseCapture`, a transport-native body capture
layer that:
  1. Subscribes to Network.streamResourceContent (CDP) when the
     browser supports it.
  2. Falls back to Network.responseReceived + body retrieval on
     loadingFinished.
  3. Falls back to Network.eventSourceMessageReceived (SSE).
  4. Falls back to Network.webSocketFrameReceived (WS).
Every chunk is classified by `intelligence.ResponseClassifier`
before being appended to the response buffer. Pollution
(analytics / telemetry / auth / conversation list) is rejected
at the door, not filtered post-hoc.
"""

from __future__ import annotations

import contextlib
import logging

from ai_orchestrator.browser_intelligence.decision.completion import CompletionEngine
from ai_orchestrator.browser_intelligence.decision.confidence import ConfidenceEngine
from ai_orchestrator.browser_intelligence.decision.entropy import EntropyEngine
from ai_orchestrator.browser_intelligence.decision.evidence_fusion import EvidenceFusion
from ai_orchestrator.browser_intelligence.decision.utility import UtilityEngine
from ai_orchestrator.browser_intelligence.estimation.belief_state import BeliefState, HiddenState
from ai_orchestrator.browser_intelligence.estimation.emission_model import EmissionModel
from ai_orchestrator.browser_intelligence.estimation.hmm_engine import HMMEngine
from ai_orchestrator.browser_intelligence.estimation.transition_matrix import TransitionMatrix
from ai_orchestrator.browser_intelligence.events import EventBus, EventType
from ai_orchestrator.browser_intelligence.features.feature_composer import FeatureComposer
from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureStore
from ai_orchestrator.browser_intelligence.intelligence.response_capture import (
    CapturedResponse,
    ResponseCapture,
    parse_stream_chunk,
)
from ai_orchestrator.browser_intelligence.intelligence.traffic_classifier import (
    ResponseClassifier,
    TrafficCategory,
)
from ai_orchestrator.browser_intelligence.learning import ProviderReliabilityStore
from ai_orchestrator.browser_intelligence.recovery import RecoveryCascade, build_default_cascade
from ai_orchestrator.browser_intelligence.scheduling import AdaptiveScheduler, SchedulingInputs
from ai_orchestrator.browser_intelligence.sensors.accessibility_sensor import AccessibilitySensor
from ai_orchestrator.browser_intelligence.sensors.dom_sensor import DOMSensor
from ai_orchestrator.browser_intelligence.sensors.mutation_sensor import MutationSensor
from ai_orchestrator.browser_intelligence.sensors.network_sensor import NetworkSensor
from ai_orchestrator.browser_intelligence.sensors.performance_sensor import PerformanceSensor
from ai_orchestrator.browser_intelligence.sensors.visual_sensor import VisualSensor

log = logging.getLogger(__name__)

TICK_INTERVAL: float = 1.0

STREAM_STALLED_IDLE_SECONDS: float = 5.0


def _coerce_sse_text(raw: str) -> str:
    """Turn a raw SSE stream (`data: {json}` lines) or a stream of bare
    JSON payloads into clean text.

    Parses each chunk and pulls out common delta fields. If no deltas
    are found, the raw text is returned verbatim.
    """
    if not raw:
        return ""
    import json as _json
    out: list[str] = []
    saw_delta = False
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            payload = line[5:].strip()
        else:
            payload = line
        if not payload or payload == "[DONE]":
            continue
        # Try to parse as JSON; if not JSON, treat the line as literal text.
        try:
            obj = _json.loads(payload)
        except Exception:
            # Non-JSON line: emit as-is (likely from a WebSocket frame).
            out.append(payload)
            continue
        if not isinstance(obj, dict):
            continue
        matched = False
        for v in _DELTA_KEYS_TO_TRY:
            val = _walk_path(obj, v)
            if isinstance(val, str) and val:
                out.append(val)
                matched = True
                saw_delta = True
                break
        if not matched:
            # Last resort: walk the object looking for any string under a
            # delta-like key.
            for k, val in _walk_delta_strings(obj):
                out.append(val)
                saw_delta = True
                break
    if saw_delta:
        return "".join(out)
    return raw


def _walk_delta_strings(obj) -> list[tuple[str, str]]:
    """Yield (key, value) pairs for any string whose key looks delta-ish."""
    delta_keys = {
        "delta_content", "content", "text", "delta_text",
        "message", "reasoning_content", "delta",
    }
    stack: list = [obj]
    seen: set[int] = set()
    out: list[tuple[str, str]] = []
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        if isinstance(cur, dict):
            for k, v in cur.items():
                if isinstance(v, str) and v and k.lower() in delta_keys:
                    out.append((k, v))
                elif isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for v in cur:
                if isinstance(v, (dict, list)):
                    stack.append(v)
    return out


_DELTA_KEYS_TO_TRY: tuple = (
    ("data", "delta_content"),
    ("data", "content"),
    ("delta", "content"),
    ("message", "content"),
    ("choices", 0, "delta", "content"),
    ("choices", 0, "message", "content"),
    ("response", "choices", 0, "delta", "content"),
    ("response", "choices", 0, "message", "content"),
    ("response", "created"),
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


class BrowserIntelligenceEngine:
    """The Browser Intelligence Operating System runtime.

    Replaces hardcoded state detection with probabilistic inference.
    Every tick: sense → compose → estimate → decide.

    Adaptive readiness: thresholds rise as the emission model learns.
    Evidence fusion: multiple sensor confidences combined.
    Stream stalled: detects stuck streams (active but no tokens).
    """

    def __init__(
        self,
        dom_sensor: DOMSensor | None = None,
        a11y_sensor: AccessibilitySensor | None = None,
        network_sensor: NetworkSensor | None = None,
        mutation_sensor: MutationSensor | None = None,
        visual_sensor: VisualSensor | None = None,
        perf_sensor: PerformanceSensor | None = None,
        transition: TransitionMatrix | None = None,
        emission: EmissionModel | None = None,
        completion: CompletionEngine | None = None,
        confidence: ConfidenceEngine | None = None,
        entropy: EntropyEngine | None = None,
        utility: UtilityEngine | None = None,
        evidence_fusion: EvidenceFusion | None = None,
        *,
        response_capture: ResponseCapture | None = None,
        traffic_classifier: ResponseClassifier | None = None,
        event_bus: EventBus | None = None,
        reliability_store: ProviderReliabilityStore | None = None,
        recovery_cascade: RecoveryCascade | None = None,
        scheduler: AdaptiveScheduler | None = None,
    ):
        self._composer = FeatureComposer(
            dom_sensor=dom_sensor,
            accessibility_sensor=a11y_sensor,
            network_sensor=network_sensor,
            mutation_sensor=mutation_sensor,
            visual_sensor=visual_sensor,
            performance_sensor=perf_sensor,
        )
        self._store = FeatureStore()
        self._hmm = HMMEngine(
            transition=transition or TransitionMatrix(),
            emission=emission or EmissionModel(),
        )
        self._completion = completion or CompletionEngine()
        self._confidence = confidence or ConfidenceEngine()
        self._entropy = entropy or EntropyEngine()
        self._utility = utility or UtilityEngine()
        self._fusion = evidence_fusion or EvidenceFusion()

        self._fusion.register_sensor("dom")
        self._fusion.register_sensor("a11y")
        self._fusion.register_sensor("network")
        self._fusion.register_sensor("mutation")
        self._fusion.register_sensor("visual")
        self._fusion.register_sensor("performance")

        # Phase 1: transport-native response capture, classified at the
        # door so non-chat traffic never enters the response buffer.
        self._classifier = traffic_classifier or ResponseClassifier()
        self._capture = response_capture or ResponseCapture(classifier=self._classifier)

        # Phase 4: events, learning, recovery, scheduling.
        self._events = event_bus or EventBus()
        self._reliability = reliability_store
        self._cascade = recovery_cascade or build_default_cascade()
        self._scheduler = scheduler or AdaptiveScheduler()

        self._belief: BeliefState | None = None
        self._ready_for_prompt: bool = False
        self._is_error: bool = False
        self._rate_limited: bool = False
        self._action: str = "wait"
        self._stream_stalled: bool = False
        self._readiness_ticks: int = 0
        self._last_adaptive_threshold: float = 0.50
        self._provider_id: str = ""
        self._cdp_session = None
        self._pending_capture_fetches: list[str] = []

    @property
    def network_sensor(self) -> NetworkSensor:
        return self._composer._network

    @property
    def belief(self) -> BeliefState | None:
        return self._belief

    @property
    def most_likely_state(self) -> HiddenState | None:
        return self._belief.most_likely if self._belief else None

    @property
    def confidence(self) -> float:
        return self._belief.confidence if self._belief else 0.0

    @property
    def entropy(self) -> float:
        return self._belief.entropy if self._belief else 10.0

    @property
    def is_ready_for_prompt(self) -> bool:
        return self._ready_for_prompt

    @property
    def is_generating(self) -> bool:
        if self._belief is None:
            return False
        return (self._belief.probabilities.get(HiddenState.GENERATING, 0) > 0.4
                or self._belief.probabilities.get(HiddenState.THINKING, 0) > 0.4)

    @property
    def is_error(self) -> bool:
        return self._is_error

    @property
    def is_rate_limited(self) -> bool:
        return self._rate_limited

    @property
    def stream_stalled(self) -> bool:
        return self._stream_stalled

    @property
    def recommended_action(self) -> str:
        return self._action

    @property
    def emission_calibration(self) -> dict[str, float]:
        return {
            s.value: self._hmm.emission.calibration_score(s)
            for s in HiddenState
        }

    @property
    def adaptive_threshold(self) -> float:
        return self._last_adaptive_threshold

    @property
    def sensor_confidences(self) -> dict[str, float]:
        return self._fusion.all_sensor_confidences()

    @property
    def event_bus(self) -> EventBus:
        return self._events

    @property
    def response_capture(self) -> ResponseCapture:
        return self._capture

    @property
    def traffic_classifier(self) -> ResponseClassifier:
        return self._classifier

    @property
    def recovery_cascade(self) -> RecoveryCascade:
        return self._cascade

    @property
    def scheduler(self) -> AdaptiveScheduler:
        return self._scheduler

    def bind_provider(
        self,
        provider_id: str,
        *,
        reliability_store: ProviderReliabilityStore | None = None,
    ) -> None:
        """Attach a provider identity and (optionally) a shared
        reliability store. The store is the Phase-3 hook for the
        persistent brain — it survives across `BrowserIntelligenceEngine`
        instances so a fresh engine on the same provider inherits
        the prior session's calibration."""
        self._provider_id = provider_id
        if reliability_store is not None:
            self._reliability = reliability_store
        if self._reliability is not None:
            self._reliability.record("provider", provider_id, True)
        self._events.publish(
            EventType.AUTH_SUCCESS,
            {"provider": provider_id},
            source=f"engine:{provider_id}",
        )

    async def attach(self, page) -> None:
        try:
            await self._composer._network.attach(page)
            self._fusion.record_sensor_success("network")
        except Exception as exc:
            self._fusion.record_sensor_failure("network")
            log.warning("CDP Network attach failed: %s — network intelligence disabled", exc)
        # Try to wire up real-time body streaming (Network.streamResourceContent).
        # This is best-effort; older browsers won't support it.
        try:
            await self._wire_streaming_capture(page)
        except Exception as exc:
            log.debug("Streaming capture attach failed (non-fatal): %s", exc)
        self._hmm.initialize()

    async def _wire_streaming_capture(self, page) -> None:
        """Open an additional CDP session and subscribe to the
        streaming body events:
        - Network.streamResourceContent   (real-time chunks)
        - Network.responseReceivedExtraInfo
        - Network.eventSourceMessageReceived (SSE)
        - Network.webSocketFrameReceived    (WS)
        - Network.dataReceived              (fallback)

        All chunks are funneled through `ResponseCapture`, which
        consults the `ResponseClassifier` and only buffers text from
        requests classified as CHAT_RESPONSE.
        """
        try:
            self._cdp_session = await page.context.new_cdp_session(page)
            await self._cdp_session.send("Network.enable")
        except Exception as exc:
            log.debug("Streaming CDP session failed: %s", exc)
            return
        cdp = self._cdp_session

        # Forward raw body events into the response capture layer.
        # All four events are no-ops if the request_id is unknown or
        # the response is not classified as chat traffic.
        def on_response(event: dict) -> None:
            try:
                response = event.get("response", {}) or {}
                url = response.get("url", "") or ""
                request_id = event.get("requestId", "") or ""
                status = int(response.get("status", 0) or 0)
                ct = (
                    response.get("mimeType", "")
                    or response.get("headers", {}).get("content-type", "")
                    or response.get("headers", {}).get("Content-Type", "")
                )
                method = response.get("headers", {}).get(":method", "GET") or "GET"
                if not method or method == "":
                    method = "GET"
                classification = self._capture.begin_response(
                    request_id=request_id,
                    url=url,
                    method=method,
                    status=status,
                    content_type=ct or "",
                    headers=response.get("headers", {}) or {},
                )
                if classification.is_chat:
                    self._events.publish(
                        EventType.GENERATION_STARTED,
                        {
                            "url": url,
                            "request_id": request_id,
                            "content_type": ct or "",
                        },
                        source="engine:capture",
                    )
            except Exception as exc:
                log.debug("on_response error: %s", exc)

        def on_chunk(event: dict) -> None:
            try:
                request_id = event.get("requestId", "") or ""
                chunk = ""
                # Network.streamResourceContent: content field (may be base64).
                if "content" in event:
                    chunk = event.get("content", "") or ""
                    if event.get("base64Encoded"):
                        import base64
                        try:
                            chunk = base64.b64decode(chunk).decode("utf-8", errors="replace")
                        except Exception:
                            pass
                # Network.dataReceived: may carry a `data` field with body fragment.
                if not chunk:
                    chunk = event.get("data", "") or ""
                # Fallback: some CDP events carry data_chunk.
                if not chunk:
                    chunk = event.get("data_chunk", "") or ""
                if not chunk:
                    return
                accepted = self._capture.append_chunk(request_id, chunk)
                if accepted:
                    self._fusion.record_sensor_success("network")
            except Exception as exc:
                log.debug("on_chunk error: %s", exc)

        def on_loading_finished(event: dict) -> None:
            try:
                request_id = event.get("requestId", "") or ""
                if request_id:
                    self._pending_capture_fetches.append(request_id)
            except Exception as exc:
                log.debug("on_loading_finished error: %s", exc)

        def on_loading_failed(event: dict) -> None:
            try:
                request_id = event.get("requestId", "") or ""
                self._capture.discard(request_id)
            except Exception as exc:
                log.debug("on_loading_failed error: %s", exc)

        def on_ws_frame(event: dict) -> None:
            try:
                payload = (event.get("response", {}) or {}).get("payloadData", "")
                if not payload:
                    return
                # WebSocket frames don't have a request_id. We use the
                # URL of the active connection — the capture layer
                # tracks by request_id though, so for WS we synthesize
                # a virtual id keyed off `event.timestamp`.
                # The cleanest path: classify URL via a synthetic call.
                ts = str(event.get("timestamp", "")) or "ws"
                ws_url = (event.get("response", {}) or {}).get("url", "") or "ws://unknown"
                request_id = f"ws:{ws_url}:{ts}"
                classification = self._capture.begin_response(
                    request_id=request_id,
                    url=ws_url,
                    method="WS",
                    status=101,
                    content_type="text/event-stream",
                )
                if classification.is_chat:
                    self._capture.append_chunk(request_id, payload)
                    if "[DONE]" in payload:
                        self._capture.close_response(request_id)
            except Exception as exc:
                log.debug("on_ws_frame error: %s", exc)

        def on_event_source_message(event: dict) -> None:
            try:
                request_id = event.get("requestId", "") or ""
                data = event.get("data", "")
                if data and request_id:
                    self._capture.append_chunk(request_id, data)
            except Exception as exc:
                log.debug("on_event_source_message error: %s", exc)

        try:
            cdp.on("Network.responseReceived", on_response)
            cdp.on("Network.dataReceived", on_chunk)
            cdp.on("Network.loadingFinished", on_loading_finished)
            cdp.on("Network.loadingFailed", on_loading_failed)
            cdp.on("Network.webSocketFrameReceived", on_ws_frame)
            cdp.on("Network.eventSourceMessageReceived", on_event_source_message)
            # streamResourceContent is the key Phase-1 deliverable: it
            # gives us body bytes in real time, before stream close.
            # Not all Chrome versions ship it; guard with try/except.
            try:
                cdp.on("Network.streamResourceContent", on_chunk)
            except Exception:
                pass
        except Exception as exc:
            log.debug("Streaming CDP handler attach failed: %s", exc)

    async def tick(self, page) -> FeatureStore:
        fv = await self._composer.tick(page)
        self._store.push(fv)

        await self._drain_capture_fetches()

        self._update_sensor_confidence(fv)

        self._belief = self._hmm.update(fv)

        self._detect_stream_stalled(fv)

        actions = self._compute_available_actions()
        best_action, _ = self._utility.best_action(actions, self._belief)
        self._action = best_action

        adaptive_thresh = self._hmm.adaptive_readiness_threshold(
            base_threshold=0.45, min_threshold=0.30, max_threshold=0.75
        )
        self._last_adaptive_threshold = adaptive_thresh

        latest = self._store.latest
        dominant = self._belief.most_likely

        prev_ready = self._ready_for_prompt
        prev_rate_limited = self._rate_limited

        self._ready_for_prompt = (
            self._belief.is_confident(adaptive_thresh)
            and dominant == HiddenState.READY
        )

        self._is_error = (
            self._belief.probabilities.get(HiddenState.ERROR, 0) > 0.5
            or self._belief.probabilities.get(HiddenState.RATE_LIMITED, 0) > 0.5
        )
        if latest and latest.stream_active:
            self._is_error = False
        self._rate_limited = (
            self._belief.probabilities.get(HiddenState.RATE_LIMITED, 0) > 0.4
        )

        if self._ready_for_prompt:
            self._readiness_ticks += 1
        else:
            self._readiness_ticks = 0

        # Phase 4: emit events for important state transitions.
        if prev_ready is False and self._ready_for_prompt:
            self._events.publish(
                EventType.AUTH_SUCCESS,
                {
                    "provider": self._provider_id,
                    "ticks_to_ready": self._readiness_ticks,
                },
                source=f"engine:{self._provider_id or 'unknown'}",
            )
        if self._rate_limited and not prev_rate_limited:
            self._events.publish(
                EventType.RATE_LIMIT_DETECTED,
                {"provider": self._provider_id},
                source=f"engine:{self._provider_id or 'unknown'}",
            )

        return self._store

    def _update_sensor_confidence(self, fv) -> None:
        if fv.input_visible:
            self._fusion.record_sensor_success("dom")
        else:
            self._fusion.record_sensor_failure("dom")
        
        if fv.a11y_extraction_success:
            self._fusion.record_sensor_success("a11y")
        else:
            self._fusion.record_sensor_failure("a11y")
        
        if fv.mutation_rate >= 0.0:
            self._fusion.record_sensor_success("mutation")
        else:
            self._fusion.record_sensor_failure("mutation")
            
        if fv.visual_stability > 0.0:
            self._fusion.record_sensor_success("visual")
        else:
            self._fusion.record_sensor_failure("visual")
            
        if fv.page_stability > 0.0:
            self._fusion.record_sensor_success("performance")
        else:
            self._fusion.record_sensor_failure("performance")

    def _detect_stream_stalled(self, fv) -> None:
        self._stream_stalled = (
            fv.stream_active
            and fv.total_chunks > 5
            and fv.tokens_per_second < 0.01
            and fv.stream_idle_time > STREAM_STALLED_IDLE_SECONDS
        )

    async def _drain_capture_fetches(self) -> None:
        while self._pending_capture_fetches:
            request_id = self._pending_capture_fetches.pop(0)
            try:
                if self._cdp_session is None:
                    self._capture.close_response(request_id)
                    continue
                resp = await self._cdp_session.send(
                    "Network.getResponseBody", {"requestId": request_id}
                )
                body = resp.get("body", "") if isinstance(resp, dict) else ""
                if body and len(body) > 10:
                    self._capture.append_chunk(request_id, body)
            except Exception:
                pass
            finally:
                cap = self._capture.close_response(request_id)
                if cap is not None and cap.classification and cap.classification.is_chat:
                    self._events.publish(
                        EventType.GENERATION_COMPLETED,
                        {
                            "url": cap.url,
                            "bytes": cap.bytes,
                            "chunks": cap.chunks,
                        },
                        source="engine:capture",
                    )
                    self._events.publish(
                        EventType.RESPONSE_CAPTURED,
                        {
                            "url": cap.url,
                            "len": len(cap.text),
                            "confidence": cap.classification.confidence,
                        },
                        source="engine:capture",
                    )

    def is_response_complete(self) -> tuple[bool, float]:
        return self._completion.is_complete(self._store)

    def response_completion_confidence(self) -> float:
        return self._completion.completion_confidence(self._store)

    def get_response_text(self) -> str:
        """Return the assembled response text from the engine-owned
        capture layer.

        Pipeline:
            CDP body event
                ↓
            ResponseClassifier.classify(url, ct, …)
                ↓
            ResponseCapture.append_chunk(request_id, body)   (only if CHAT_RESPONSE)
                ↓
            self.get_response_text() returns the assembled text

        Non-chat responses (analytics, telemetry, auth, conversation
        list) are rejected at the door, so `engine.get_response_text()`
        returns the actual model reply — not a tracking ping.

        For backward compat we still consult the legacy per-observer
        buffers as a backstop, but only after the new layer returns
        nothing.
        """
        # Phase 1: transport-native, classified capture.
        text = self._capture.get_response_text()
        if text:
            return _coerce_sse_text(text)

        # Backstop: legacy observer buffers. These are still useful
        # when the streaming CDP session fails to attach (some headless
        # configurations block Network.streamResourceContent).
        sse_text = self._composer._network._sse_observer.get_response_text()
        ws_text = self._composer._network._ws_observer.get_response_text()
        fetch_text = self._composer._network._fetch_observer.get_response_text()

        # Pick the longest text — but only after the new layer
        # produced nothing, so a recent classified chat response wins.
        candidates: list[tuple[str, str]] = []
        if sse_text:
            candidates.append(("sse", sse_text))
        if ws_text:
            candidates.append(("ws", ws_text))
        if fetch_text:
            candidates.append(("fetch", fetch_text))
        if not candidates:
            return ""
        # Sort by length, take the longest, then coerce SSE/JSON deltas.
        candidates.sort(key=lambda kv: len(kv[1]), reverse=True)
        _kind, raw = candidates[0]
        return _coerce_sse_text(raw)

    def get_response_text_sse(self) -> str:
        """Convenience: return the SSE response text without coercion.

        Reads the legacy per-observer buffer (used by the JS-init
        fallback path) and the new classified capture layer, taking
        whichever has text.
        """
        # Legacy buffer first — the test harness writes here directly.
        legacy = self._composer._network._sse_observer.get_response_text()
        if legacy:
            return legacy
        return self._capture.get_response_text_sse()

    def state_probabilities(self) -> dict[str, float]:
        if self._belief is None:
            return {}
        return {
            s.value: round(p, 4)
            for s, p in self._belief.probabilities.items()
        }

    def action_utilities(self) -> dict[str, float]:
        if self._belief is None:
            return {}
        actions = self._compute_available_actions()
        return {
            a: round(v, 2)
            for a, v in self._utility.all_utilities(actions, self._belief).items()
        }

    def _compute_available_actions(self) -> list[str]:
        if self._belief is None:
            return ["wait"]

        best = self._belief.most_likely
        latest = self._store.latest

        if best == HiddenState.READY:
            return ["type_prompt", "wait"]
        if best == HiddenState.PROMPT_SENT:
            return ["click_send", "wait"]
        if best == HiddenState.GENERATING:
            actions = ["wait"]
            if self._stream_stalled:
                actions.append("recover")
            if latest and latest.stream_active and latest.tokens_per_second > 0:
                actions.append("extract_response")
            return actions
        if best == HiddenState.THINKING:
            return ["wait"]
        if best == HiddenState.COMPLETE:
            return ["extract_response", "wait"]
        if best == HiddenState.ERROR:
            return ["recover", "refresh", "wait"]
        if best == HiddenState.RATE_LIMITED:
            return ["wait", "quarantine", "refresh"]
        if best == HiddenState.AUTH_REQUIRED:
            return ["relogin", "refresh", "wait"]
        if best == HiddenState.SHADOW_BANNED:
            return ["quarantine", "wait"]
        if best == HiddenState.BOOTING:
            return ["wait", "refresh"]
        return ["wait"]

    def record_reward(self, reward: float) -> None:
        """Feed a [0, 1] reward back into the learning layer.

        Phase 4: maps a positive outcome to a Bayesian update on the
        provider / account reliability posteriors, and persists the
        current emission calibration into the persistent brain.
        """
        if self._reliability is None:
            return
        if self._provider_id:
            self._reliability.record("provider", self._provider_id, reward >= 0.5)
        calibration = self.emission_calibration
        # Best-effort: log into events for downstream consumers.
        self._events.publish(
            EventType.GENERIC,
            {
                "kind": "reward",
                "provider": self._provider_id,
                "reward": float(reward),
                "calibration": calibration,
            },
            source=f"engine:{self._provider_id or 'unknown'}",
        )

    def snapshot_brain(self) -> dict:
        """Return a JSON-serializable snapshot of everything the
        engine has learned this session. Callers feed this to
        `ProviderBrain.record_*` for persistence across restarts."""
        return {
            "provider": self._provider_id,
            "emission_calibration": self.emission_calibration,
            "sensor_confidences": self.sensor_confidences,
            "session_count": 1,
            "reliability": (
                self._reliability.snapshot() if self._reliability is not None else {}
            ),
        }

    async def detach(self) -> None:
        with contextlib.suppress(Exception):
            await self._composer._network.detach()
        if self._cdp_session is not None:
            with contextlib.suppress(Exception):
                await self._cdp_session.detach()
            self._cdp_session = None

    def reset(self) -> None:
        self._store.clear()
        self._hmm.reset()
        self._completion.reset()
        self._composer.reset()
        self._fusion.reset()
        self._capture.reset()
        self._pending_capture_fetches.clear()
        self._belief = None
        self._ready_for_prompt = False
        self._is_error = False
        self._rate_limited = False
        self._action = "wait"
        self._stream_stalled = False
        self._readiness_ticks = 0
        self._last_adaptive_threshold = 0.50
