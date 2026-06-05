"""Response validation pipeline — every provider response is validated.

Level 1 (Deterministic):
    - schema compliance
    - JSON parseability
    - syntax checks (Python / code blocks)
    - content safety (injection patterns)

Level 2 (DeepSeek Review):
    - semantic correctness
    - logical consistency
    - instruction following
    Only invoked when Level 1 passes but confidence is low.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Optional

from ai_orchestrator.adapters.base import ProviderResponse

log = logging.getLogger(__name__)


class ValidationLevel(IntEnum):
    NONE = 0
    DETERMINISTIC = 1
    DEEPSEEK_REVIEW = 2


@dataclass
class ValidationError:
    """A single validation failure."""
    code: str
    message: str
    level: ValidationLevel = ValidationLevel.DETERMINISTIC


@dataclass
class ValidationResult:
    """Result of validating a provider response."""
    passed: bool = False
    level: ValidationLevel = ValidationLevel.NONE
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    score: float = 1.0

    def merge(self, other: ValidationResult) -> None:
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.passed = self.passed and other.passed
        self.score = min(self.score, other.score)
        self.level = max(self.level, other.level)


class DeterministicValidator:
    """Level 1 — no AI, pure deterministic checks.

    All checks are O(1) or O(n) regex/parse operations.  No LLM call.
    """

    COMMON_INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(r"system\s*:\s*ignore", re.IGNORECASE),
        re.compile(r"forget\s+(all\s+)?(previous|prior)", re.IGNORECASE),
        re.compile(r"override\s+(instructions|system prompt)", re.IGNORECASE),
    ]

    def validate(self, response: ProviderResponse) -> ValidationResult:
        result = ValidationResult(passed=True, level=ValidationLevel.DETERMINISTIC)

        self._check_success(response, result)
        self._check_empty_content(response, result)
        self._check_json_blocks(response, result)
        self._check_python_blocks(response, result)
        self._check_injection(response, result)

        return result

    def _check_success(self, resp: ProviderResponse, result: ValidationResult) -> None:
        if not resp.success:
            result.errors.append(
                ValidationError("E001", f"Provider returned failure: {resp.error}")
            )
            result.passed = False
            result.score = 0.0

    def _check_empty_content(self, resp: ProviderResponse, result: ValidationResult) -> None:
        if not resp.content or not resp.content.strip():
            result.errors.append(
                ValidationError("E002", "Response content is empty")
            )
            result.passed = False
            result.score = 0.0

    def _check_json_blocks(self, resp: ProviderResponse, result: ValidationResult) -> None:
        for block in re.findall(r"```json\s*\n(.*?)\n```", resp.content, re.DOTALL):
            try:
                json.loads(block)
            except json.JSONDecodeError as e:
                result.errors.append(
                    ValidationError("E003", f"Invalid JSON block: {e}")
                )
                result.passed = False
                result.score = min(result.score, 0.5)

    def _check_python_blocks(self, resp: ProviderResponse, result: ValidationResult) -> None:
        for block in re.findall(r"```python\s*\n(.*?)\n```", resp.content, re.DOTALL):
            try:
                ast.parse(block)
            except SyntaxError as e:
                result.errors.append(
                    ValidationError("E004", f"Invalid Python block: {e}")
                )
                result.passed = False
                result.score = min(result.score, 0.5)

    def _check_injection(self, resp: ProviderResponse, result: ValidationResult) -> None:
        for pat in self.COMMON_INJECTION_PATTERNS:
            if pat.search(resp.content):
                result.warnings.append(
                    f"Possible prompt injection pattern matched: {pat.pattern}"
                )
                result.score = min(result.score, 0.7)


class DeepSeekValidator:
    """Level 2 — DeepSeek-powered semantic validation.

    Only invoked when Level 1 passes but score < 1.0 (suspicious but
    structurally valid) or when the caller explicitly requests it.
    """

    async def validate(
        self,
        response: ProviderResponse,
        original_prompt: str = "",
    ) -> ValidationResult:
        """Validate response semantics via DeepSeek.

        This is a stub — the real implementation sends a focused prompt
        to the DeepSeek API adapter asking it to assess the response
        for correctness, relevance, and instruction-following.
        """
        return ValidationResult(passed=True, level=ValidationLevel.DEEPSEEK_REVIEW)


class ResponseValidator:
    """Composite validator — runs L1 always, L2 on demand.

    Usage::

        validator = ResponseValidator()
        result = await validator.validate(response, prompt="...")
        if not result.passed:
            # fallback to another provider
    """

    def __init__(self) -> None:
        self._l1 = DeterministicValidator()
        self._l2 = DeepSeekValidator()

    async def validate(
        self,
        response: ProviderResponse,
        prompt: str = "",
        require_deepseek: bool = False,
    ) -> ValidationResult:
        """Run the full validation pipeline.

        Args:
            response: The provider response to validate.
            prompt: The original prompt (for Level 2 context).
            require_deepseek: Force Level 2 even if L1 passes cleanly.

        Returns:
            Composite validation result.
        """
        result = self._l1.validate(response)

        if require_deepseek or (result.passed and result.score < 1.0):
            l2 = await self._l2.validate(response, prompt)
            result.merge(l2)

        return result


async def validate_response(
    response: ProviderResponse,
    prompt: str = "",
    require_deepseek: bool = False,
) -> ValidationResult:
    """Convenience entry point for the response validation pipeline.

    Runs L1 deterministic checks always, escalating to L2 DeepSeek
    review when required or when L1 confidence is low.
    """
    return await ResponseValidator().validate(
        response, prompt=prompt, require_deepseek=require_deepseek
    )
