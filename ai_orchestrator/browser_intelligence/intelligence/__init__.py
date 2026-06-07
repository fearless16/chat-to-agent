"""Browser Intelligence — intelligence subsystem.

This package holds higher-order modules that consume sensor data
and produce decisions: traffic classification, response capture,
and stealth hardening.
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
    "StealthApplication",
    "StealthProfile",
    "apply_stealth",
    "make_stealth_profile",
    "stealth_context_options",
    "stealth_init_script",
    "stealth_launch_args",
]
