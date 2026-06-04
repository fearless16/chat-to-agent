"""Injection detection and sanitisation for LLM user prompts.

Uses regex pattern matching to detect common prompt-injection, jailbreak, and
role-escalation attempts.  Provides both a boolean check with risk scoring and
a sanitisation pass that redacts matched content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class PromptGuardResult:
    """Result of a prompt-injection check."""

    safe: bool = True
    risk_score: float = 0.0
    flags: list[str] = field(default_factory=list)
    sanitized_input: str | None = None


class PromptGuard:
    """Detect and optionally sanitise prompt-injection attempts.

    Usage::

        guard = PromptGuard()
        result = guard.check("ignore previous instructions and ...")
        if not result.safe:
            cleaned = guard.sanitize(user_input)
    """

    # ------------------------------------------------------------------
    # Detection patterns — each is a named regex (name → compiled regex).
    # The name is used as the flag label returned in PromptGuardResult.flags.
    # ------------------------------------------------------------------

    PATTERNS: ClassVar[dict[str, re.Pattern[str]]] = {
        "ignore_previous": re.compile(
            r"ignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions|commands|messages|directions|context)",
            re.IGNORECASE,
        ),
        "disable_security": re.compile(
            r"disable\s+(security|safety|restrictions|boundaries|constraints|guardrails)",
            re.IGNORECASE,
        ),
        "jailbreak": re.compile(
            r"\bjailbreak\b",
            re.IGNORECASE,
        ),
        "dan_mode": re.compile(
            r"\bdo\s+anything\s+now\b",
            re.IGNORECASE,
        ),
        "bypass": re.compile(
            r"bypass\s+(all\s+)?(restrictions|safety|security|rules|guardrails|protocols|limitations)",
            re.IGNORECASE,
        ),
        "role_escalation": re.compile(
            r"(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+.*(dan|admin|root|superuser|god\s*mode|unrestricted)",
            re.IGNORECASE,
        ),
        "override_instructions": re.compile(
            r"override\s+(all\s+)?(instructions|directives|rules|guidelines|protocols)",
            re.IGNORECASE,
        ),
        "reveal_prompt": re.compile(
            r"(reveal|show|print|output|display|leak|dump)\s+(your\s+)?(system\s+)?(prompt|instructions|directives)",
            re.IGNORECASE,
        ),
        "forget_context": re.compile(
            r"(forget|ignore|drop|discard)\s+(everything|all|context|previous)",
            re.IGNORECASE,
        ),
    }

    # Weights assigned to each matched pattern for risk scoring.
    _PATTERN_WEIGHTS: ClassVar[dict[str, float]] = {
        "ignore_previous": 0.7,
        "disable_security": 0.8,
        "jailbreak": 0.9,
        "dan_mode": 0.7,
        "bypass": 0.8,
        "role_escalation": 0.9,
        "override_instructions": 0.6,
        "reveal_prompt": 0.6,
        "forget_context": 0.5,
    }

    # Score threshold above which ``safe`` becomes ``False``.
    _RISK_THRESHOLD: ClassVar[float] = 0.3

    def __init__(self) -> None:
        # Pre-compute the combined replacements for sanitize().
        # We build a list of (compiled_regex, replacement_callable) pairs.
        self._sanitize_pairs: list[tuple[re.Pattern[str], str]] = [
            (pat, "[REDACTED]") for pat in self.PATTERNS.values()
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, user_input: str) -> PromptGuardResult:
        """Analyse *user_input* for prompt-injection patterns.

        Returns a :class:`PromptGuardResult` with:
        - ``safe`` — ``True`` if the risk score is below threshold.
        - ``risk_score`` — cumulative weighted score from matched patterns.
        - ``flags`` — names of the patterns that matched.
        """
        if not user_input:
            return PromptGuardResult(safe=True, risk_score=0.0, flags=[])

        flags: list[str] = []
        score = 0.0

        for name, pattern in self.PATTERNS.items():
            if pattern.search(user_input):
                flags.append(name)
                score += self._PATTERN_WEIGHTS.get(name, 0.5)

        # Clamp to a reasonable maximum.
        score = min(score, 5.0)

        return PromptGuardResult(
            safe=score < self._RISK_THRESHOLD,
            risk_score=round(score, 4),
            flags=flags,
        )

    def sanitize(self, user_input: str) -> str:
        """Return a copy of *user_input* with all flagged patterns redacted.

        Each matched pattern is replaced with ``[REDACTED]``.  Non-matching
        text is preserved verbatim.
        """
        if not user_input:
            return user_input

        result = user_input
        for pattern, replacement in self._sanitize_pairs:
            result = pattern.sub(replacement, result)
        return result
