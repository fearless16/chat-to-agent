"""Browser Intelligence — intelligence subsystem.

This package holds higher-order modules that consume sensor data
and produce decisions: traffic classification, response capture,
provider-drift detection, shadow-ban detection, and the persistent
per-provider brain.
"""

from ai_orchestrator.browser_intelligence.intelligence.traffic_classifier import (
    ResponseClassifier,
    TrafficCategory,
    TrafficClassification,
)
from ai_orchestrator.browser_intelligence.intelligence.response_capture import (
    CapturedResponse,
    ResponseCapture,
    parse_stream_chunk,
)
from ai_orchestrator.browser_intelligence.intelligence.drift_detector import (
    DriftDetector,
    DriftSignal,
    DriftSnapshot,
)
from ai_orchestrator.browser_intelligence.intelligence.shadow_ban_detector import (
    ShadowBanDetector,
    ShadowBanState,
    ShadowBanVerdict,
)
from ai_orchestrator.browser_intelligence.intelligence.provider_brain import (
    ProviderBrain,
    ProviderBrainState,
    ProviderName,
)
from ai_orchestrator.browser_intelligence.intelligence.stealth import (
    StealthApplication,
    StealthProfile,
    apply_stealth,
    make_stealth_profile,
    stealth_context_options,
    stealth_init_script,
    stealth_launch_args,
)

__all__ = [
    "ResponseClassifier",
    "TrafficCategory",
    "TrafficClassification",
    "ResponseCapture",
    "CapturedResponse",
    "parse_stream_chunk",
    "DriftDetector",
    "DriftSignal",
    "DriftSnapshot",
    "ShadowBanDetector",
    "ShadowBanState",
    "ShadowBanVerdict",
    "ProviderBrain",
    "ProviderBrainState",
    "ProviderName",
    "StealthApplication",
    "StealthProfile",
    "apply_stealth",
    "make_stealth_profile",
    "stealth_context_options",
    "stealth_init_script",
    "stealth_launch_args",
]
