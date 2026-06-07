"""ResponseClassifier — decides whether a network request is the actual
chat response or pollution (analytics, telemetry, auth, conversation list).

Operates on signals the engine already collects from CDP:
URL, headers, content-type, request method, body sample, timing.
Produces a `TrafficCategory` plus a confidence score in [0, 1].

The classifier is intentionally provider-agnostic. It uses URL-shape
heuristics (path segments, query parameters), content-type shape, and
sample-based JSON inspection. No provider names are hardcoded.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

log = logging.getLogger(__name__)


class TrafficCategory(str, Enum):
    """High-level classification for a network request."""

    CHAT_RESPONSE = "CHAT_RESPONSE"
    CONVERSATION_LIST = "CONVERSATION_LIST"
    ANALYTICS = "ANALYTICS"
    TELEMETRY = "TELEMETRY"
    AUTH = "AUTH"
    STATIC = "STATIC"
    UNKNOWN = "UNKNOWN"


@dataclass
class TrafficClassification:
    category: TrafficCategory
    confidence: float
    reasons: list[str] = field(default_factory=list)

    @property
    def is_chat(self) -> bool:
        return self.category == TrafficCategory.CHAT_RESPONSE

    @property
    def is_pollution(self) -> bool:
        return self.category in (
            TrafficCategory.ANALYTICS,
            TrafficCategory.TELEMETRY,
            TrafficCategory.CONVERSATION_LIST,
            TrafficCategory.AUTH,
            TrafficCategory.STATIC,
        )


# ──────────────────────────────────────────────────────────────────────
# Signal tables — pure data, easy to extend without code edits to the
# scoring core. Each tuple is (compiled-regex, category, weight).
# ──────────────────────────────────────────────────────────────────────

# URLs that almost certainly carry model completions.
_CHAT_PATH_HINTS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"/chat[-_/]?completions?", re.I), 0.45),
    (re.compile(r"/completions?/?", re.I), 0.30),
    (re.compile(r"/v1/messages?", re.I), 0.35),
    (re.compile(r"/conversation/[^/]+/message", re.I), 0.30),
    (re.compile(r"/api/v\d+/chat", re.I), 0.40),
    (re.compile(r"/backend-api/conversation", re.I), 0.40),
    (re.compile(r"/backend-api/f/conversation", re.I), 0.40),
    (re.compile(r"/conversation/stream", re.I), 0.40),
    (re.compile(r"/chat/completions/stream", re.I), 0.40),
    (re.compile(r"/sendMessage", re.I), 0.30),
    (re.compile(r"/chat/send", re.I), 0.35),
    (re.compile(r"/api/chat/completions?", re.I), 0.40),
    (re.compile(r"/completion", re.I), 0.20),
    # Kimi (kimi.com)
    (re.compile(r"/api/chat/segment", re.I), 0.40),
    (re.compile(r"/kimiplus/chat", re.I), 0.40),
    (re.compile(r"/api/chat\b", re.I), 0.30),
    # MiniMax (agent.minimax.io)
    (re.compile(r"/v1/text/chatcompletion", re.I), 0.40),
    (re.compile(r"/agent/chat", re.I), 0.35),
    # MiMo / Xiaomi (aistudio.xiaomimimo.com)
    (re.compile(r"/api/v1/generate", re.I), 0.35),
    (re.compile(r"/api/chat/stream", re.I), 0.40),
    # Qwen (chat.qwen.ai)
    (re.compile(r"/api/v1/qwen", re.I), 0.35),
    (re.compile(r"/api/chat/qwen", re.I), 0.35),
)

# URLs that are clearly not chat responses.
_LIST_PATH_HINTS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"/conversations(?:\?|/|$)", re.I), 0.55),
    (re.compile(r"/threads?(?:/|\?|$)", re.I), 0.30),
    (re.compile(r"/history(?:/|\?|$)", re.I), 0.40),
    (re.compile(r"/messages?/?\?.*list", re.I), 0.40),
    (re.compile(r"/sessions?(?:/|\?|$)", re.I), 0.30),
    (re.compile(r"/inbox", re.I), 0.30),
    (re.compile(r"/api/v\d+/users?/", re.I), 0.25),
)

_ANALYTICS_PATH_HINTS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"/analytics", re.I), 0.85),
    (re.compile(r"/telemetry", re.I), 0.85),
    (re.compile(r"/tracking", re.I), 0.80),
    (re.compile(r"/metrics", re.I), 0.75),
    (re.compile(r"/beacon", re.I), 0.80),
    (re.compile(r"/collect", re.I), 0.70),
    (re.compile(r"/log/", re.I), 0.65),
    (re.compile(r"/pixel\.", re.I), 0.85),
    (re.compile(r"/p\b", re.I), 0.55),
    (re.compile(r"/events?/?\?", re.I), 0.55),
    (re.compile(r"google-analytics", re.I), 0.95),
    (re.compile(r"googletagmanager", re.I), 0.95),
    (re.compile(r"doubleclick\.net", re.I), 0.95),
    (re.compile(r"facebook\.net|fbcdn\.net", re.I), 0.90),
    (re.compile(r"sentry\.io|appsflyer|amplitude|mixpanel|segment\.io", re.I), 0.95),
)

_TELEMETRY_PATH_HINTS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"/resultobject", re.I), 0.80),
    (re.compile(r"/requestid", re.I), 0.70),
    (re.compile(r"/health", re.I), 0.70),
    (re.compile(r"/ping", re.I), 0.70),
    (re.compile(r"/readyz|readyz\?", re.I), 0.65),
    (re.compile(r"/livez", re.I), 0.65),
    (re.compile(r"/usage", re.I), 0.40),
    (re.compile(r"/quota", re.I), 0.40),
    (re.compile(r"/client/[a-z\-]+/track", re.I), 0.60),
)

_AUTH_PATH_HINTS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"/auth/?"), 0.85),
    (re.compile(r"/login|signin|sign-in", re.I), 0.85),
    (re.compile(r"/logout|signout|sign-out", re.I), 0.85),
    (re.compile(r"/register|signup|sign-up", re.I), 0.85),
    (re.compile(r"/csrf|xsrf", re.I), 0.85),
    (re.compile(r"/oauth", re.I), 0.85),
    (re.compile(r"/sso/?", re.I), 0.85),
    (re.compile(r"/token|refresh", re.I), 0.65),
    (re.compile(r"/verify", re.I), 0.50),
    (re.compile(r"/session/create", re.I), 0.70),
)

_STATIC_HINTS: tuple[tuple[re.Pattern, float], ...] = (
    (re.compile(r"\.(?:js|css|woff2?|ttf|otf|svg|png|jpg|jpeg|gif|webp|ico|map)($|\?)", re.I), 0.85),
    (re.compile(r"/static/", re.I), 0.65),
    (re.compile(r"/_next/", re.I), 0.55),
    (re.compile(r"/assets/", re.I), 0.65),
    (re.compile(r"/public/", re.I), 0.55),
)

_STREAMING_CT_HINTS: tuple[str, ...] = (
    "text/event-stream",
    "application/x-ndjson",
    "application/x-json-stream",
    "application/octet-stream",
)

_JSON_CT_HINTS: tuple[str, ...] = (
    "application/json",
    "text/json",
    "text/plain",
)

_DELTA_KEYS: frozenset[str] = frozenset({
    "delta_content", "content", "text", "delta_text",
    "message", "reasoning_content", "delta", "completion",
})


class ResponseClassifier:
    """Scores a single network request and assigns a TrafficCategory.

    Construction is parameterless and side-effect-free, so the same
    instance is safe to share across workers / pages / accounts.
    """

    def __init__(self, chat_min_confidence: float = 0.55):
        self._chat_min_confidence = float(chat_min_confidence)
        self._seen_count: int = 0
        self._chat_count: int = 0
        self._pollution_count: int = 0

    @property
    def stats(self) -> dict[str, int]:
        return {
            "seen": self._seen_count,
            "chat": self._chat_count,
            "pollution": self._pollution_count,
        }

    def classify(
        self,
        url: str,
        *,
        method: str = "GET",
        headers: dict | None = None,
        content_type: str = "",
        body_sample: str = "",
        request_duration_ms: float = 0.0,
        status: int = 0,
    ) -> TrafficClassification:
        """Classify a request.

        Args:
            url: full request URL.
            method: HTTP method.
            headers: request + response headers dict.
            content_type: response content-type (lowercased preferred).
            body_sample: first ~1KB of the response body, if available.
            request_duration_ms: total request latency in ms.
            status: HTTP status code.

        Returns:
            TrafficClassification with category, confidence, and reasons.
        """
        self._seen_count += 1
        reasons: list[str] = []
        url_lc = (url or "").lower()
        ct_lc = (content_type or "").lower()
        method = (method or "GET").upper()
        headers = headers or {}

        # ── Hard rejections (fast) ──────────────────────────────
        # Static asset, never chat.
        score, cat = 0.0, TrafficCategory.UNKNOWN
        for pat, w in _STATIC_HINTS:
            if pat.search(url_lc):
                score = max(score, w)
                cat = TrafficCategory.STATIC
                reasons.append(f"static:{pat.pattern}")

        # Analytics / tracking domains.
        for pat, w in _ANALYTICS_PATH_HINTS:
            if pat.search(url_lc):
                if w > score:
                    score, cat = w, TrafficCategory.ANALYTICS
                reasons.append(f"analytics:{pat.pattern}")

        # Telemetry / health pings.
        for pat, w in _TELEMETRY_PATH_HINTS:
            if pat.search(url_lc):
                if w > score:
                    score, cat = w, TrafficCategory.TELEMETRY
                reasons.append(f"telemetry:{pat.pattern}")

        # Auth endpoints.
        for pat, w in _AUTH_PATH_HINTS:
            if pat.search(url_lc):
                if w > score:
                    score, cat = w, TrafficCategory.AUTH
                reasons.append(f"auth:{pat.pattern}")

        # Conversation list / history.
        for pat, w in _LIST_PATH_HINTS:
            if pat.search(url_lc):
                if w > score:
                    score, cat = w, TrafficCategory.CONVERSATION_LIST
                reasons.append(f"list:{pat.pattern}")

        # ── Positive chat signal ────────────────────────────────
        chat_score = 0.0
        for pat, w in _CHAT_PATH_HINTS:
            if pat.search(url_lc):
                chat_score += w
                reasons.append(f"chat:{pat.pattern}")

        # Streaming content-type is a strong chat indicator.
        if any(s in ct_lc for s in _STREAMING_CT_HINTS):
            chat_score += 0.35
            reasons.append("ct:streaming")

        # Method / status: chat responses are POST returns with 200/201.
        if method == "POST" and 200 <= status < 300:
            chat_score += 0.05
        if status == 401:
            # Auth failure on chat path → still AUTH, not CHAT.
            if chat_score > 0:
                chat_score *= 0.2
            reasons.append("status:401")
        if status >= 400:
            chat_score *= 0.5
            reasons.append(f"status:>=400:{status}")

        # Body sample evidence: presence of delta keys strongly suggests chat.
        if body_sample:
            sample_score, sample_reasons = self._score_body_sample(body_sample)
            chat_score += sample_score
            reasons.extend(sample_reasons)

        # Long-running request strongly suggests streaming response.
        if request_duration_ms > 1500 and "stream" in ct_lc:
            chat_score += 0.10
            reasons.append("long_stream")

        # ── Decision ────────────────────────────────────────────
        if chat_score > score and chat_score >= self._chat_min_confidence:
            confidence = min(1.0, 0.5 + chat_score * 0.5)
            self._chat_count += 1
            return TrafficClassification(
                category=TrafficCategory.CHAT_RESPONSE,
                confidence=confidence,
                reasons=reasons,
            )

        if score > 0:
            self._pollution_count += 1
            confidence = min(1.0, 0.5 + score * 0.5)
            return TrafficClassification(
                category=cat,
                confidence=confidence,
                reasons=reasons,
            )

        self._pollution_count += 1
        return TrafficClassification(
            category=TrafficCategory.UNKNOWN,
            confidence=0.0,
            reasons=reasons or ["no_signals"],
        )

    @staticmethod
    def _score_body_sample(sample: str) -> tuple[float, list[str]]:
        """Look for streaming-delta shape inside the body sample.

        Returns (score, reasons). Score 0 means no signal.
        """
        if not sample:
            return 0.0, []
        reasons: list[str] = []
        score = 0.0
        s = sample.lstrip()
        if s.startswith("data:"):
            score += 0.30
            reasons.append("body:sse_data_prefix")
            payload = s[5:].strip()
            try:
                obj = json.loads(payload)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                if any(k in obj for k in _DELTA_KEYS):
                    score += 0.25
                    reasons.append("body:delta_keys")
            return min(0.55, score), reasons

        # Try direct JSON parse
        try:
            obj = json.loads(sample)
        except Exception:
            return 0.0, []

        if isinstance(obj, dict):
            if any(k in obj for k in _DELTA_KEYS):
                score += 0.20
                reasons.append("body:delta_keys")
            choices = obj.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict) and "delta" in first:
                    score += 0.20
                    reasons.append("body:choices_delta")
            if obj.get("object") in ("chat.completion.chunk", "text_completion"):
                score += 0.15
                reasons.append("body:openai_chunk_object")
        return min(0.55, score), reasons
