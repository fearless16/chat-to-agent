"""BrowserIntelligenceEngine — top-level orchestrator for the Browser Intelligence OS.

Orchestrates the full pipeline:
    Sensor → Feature → Estimation → Decision → Action
"""

from __future__ import annotations

import logging
from typing import Optional

from ai_orchestrator.browser_intelligence.decision.completion import CompletionEngine
from ai_orchestrator.browser_intelligence.decision.confidence import ConfidenceEngine
from ai_orchestrator.browser_intelligence.decision.entropy import EntropyEngine
from ai_orchestrator.browser_intelligence.decision.evidence_fusion import EvidenceFusion
from ai_orchestrator.browser_intelligence.decision.utility import UtilityEngine
from ai_orchestrator.browser_intelligence.estimation.belief_state import BeliefState, HiddenState
from ai_orchestrator.browser_intelligence.estimation.emission_model import EmissionModel
from ai_orchestrator.browser_intelligence.estimation.hmm_engine import HMMEngine
from ai_orchestrator.browser_intelligence.estimation.transition_matrix import TransitionMatrix
from ai_orchestrator.browser_intelligence.features.feature_composer import FeatureComposer
from ai_orchestrator.browser_intelligence.features.feature_vector import FeatureStore
from ai_orchestrator.browser_intelligence.sensors.accessibility_sensor import AccessibilitySensor
from ai_orchestrator.browser_intelligence.sensors.dom_sensor import DOMSensor
from ai_orchestrator.browser_intelligence.sensors.mutation_sensor import MutationSensor
from ai_orchestrator.browser_intelligence.sensors.network_sensor import NetworkSensor
from ai_orchestrator.browser_intelligence.sensors.performance_sensor import PerformanceSensor
from ai_orchestrator.browser_intelligence.sensors.visual_sensor import VisualSensor

log = logging.getLogger(__name__)

TICK_INTERVAL: float = 1.0

STREAM_STALLED_IDLE_SECONDS: float = 5.0


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
        dom_sensor: Optional[DOMSensor] = None,
        a11y_sensor: Optional[AccessibilitySensor] = None,
        network_sensor: Optional[NetworkSensor] = None,
        mutation_sensor: Optional[MutationSensor] = None,
        visual_sensor: Optional[VisualSensor] = None,
        perf_sensor: Optional[PerformanceSensor] = None,
        transition: Optional[TransitionMatrix] = None,
        emission: Optional[EmissionModel] = None,
        completion: Optional[CompletionEngine] = None,
        confidence: Optional[ConfidenceEngine] = None,
        entropy: Optional[EntropyEngine] = None,
        utility: Optional[UtilityEngine] = None,
        evidence_fusion: Optional[EvidenceFusion] = None,
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

        self._belief: Optional[BeliefState] = None
        self._ready_for_prompt: bool = False
        self._is_error: bool = False
        self._rate_limited: bool = False
        self._action: str = "wait"
        self._stream_stalled: bool = False
        self._readiness_ticks: int = 0
        self._last_adaptive_threshold: float = 0.50

    @property
    def network_sensor(self) -> NetworkSensor:
        return self._composer._network

    @property
    def belief(self) -> Optional[BeliefState]:
        return self._belief

    @property
    def most_likely_state(self) -> Optional[HiddenState]:
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

    async def attach(self, page) -> None:
        try:
            await self._composer._network.attach(page)
            self._fusion.record_sensor_success("network")
        except Exception as exc:
            self._fusion.record_sensor_failure("network")
            log.warning("CDP Network attach failed: %s — network intelligence disabled", exc)
        self._hmm.initialize()

    async def tick(self, page) -> FeatureStore:
        fv = await self._composer.tick(page)
        self._store.push(fv)

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

        return self._store

    def _update_sensor_confidence(self, fv) -> None:
        if fv.input_visible:
            self._fusion.record_sensor_success("dom")
        if fv.has_streaming_marker or fv.has_streaming_marker is False:
            self._fusion.record_sensor_success("a11y")
        self._fusion.record_sensor_success("mutation")
        if fv.visual_stability > 0.0:
            self._fusion.record_sensor_success("visual")
        if fv.page_stability > 0.0:
            self._fusion.record_sensor_success("performance")

    def _detect_stream_stalled(self, fv) -> None:
        self._stream_stalled = (
            fv.stream_active
            and fv.total_chunks > 5
            and fv.tokens_per_second < 0.01
            and fv.stream_idle_time > STREAM_STALLED_IDLE_SECONDS
        )

    def is_response_complete(self) -> tuple[bool, float]:
        return self._completion.is_complete(self._store)

    def response_completion_confidence(self) -> float:
        return self._completion.completion_confidence(self._store)

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
        pass

    async def detach(self) -> None:
        try:
            await self._composer._network.detach()
        except Exception:
            pass

    def reset(self) -> None:
        self._store.clear()
        self._hmm.reset()
        self._completion.reset()
        self._composer.reset()
        self._fusion.reset()
        self._belief = None
        self._ready_for_prompt = False
        self._is_error = False
        self._rate_limited = False
        self._action = "wait"
        self._stream_stalled = False
        self._readiness_ticks = 0
        self._last_adaptive_threshold = 0.50
