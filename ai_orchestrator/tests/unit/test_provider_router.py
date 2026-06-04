"""Tests for ProviderRouter — capability scoring, penalty application, and account selection."""

import pytest

from ai_orchestrator.models.account import Account, AccountState, ProviderKind
from ai_orchestrator.models.capabilities import TaskRequirements
from ai_orchestrator.orchestrator.provider_router import (
    ProviderRouter,
    ScoredAccount,
    DEFAULT_WEIGHTS,
)


def make_account(
    provider: str = "deepseek",
    context_limit: int = 8_192,
    health_score: float = 1.0,
    current_rate_usage: float = 0.0,
    avg_latency_ms: float = 0.0,
    consecutive_failures: int = 0,
    state: AccountState = AccountState.IDLE,
) -> Account:
    return Account(
        id=f"{provider}-test-001",
        provider=provider,
        provider_kind=ProviderKind.API,
        state=state,
        health_score=health_score,
        consecutive_failures=consecutive_failures,
        context_limit=context_limit,
        current_rate_usage=current_rate_usage,
        avg_latency_ms=avg_latency_ms,
    )


class TestScoredAccount:
    """ScoredAccount dataclass construction."""

    def test_construction(self):
        acct = make_account()
        sa = ScoredAccount(account=acct, score=0.85, reason="solid_match")
        assert sa.account is acct
        assert sa.score == 0.85
        assert sa.reason == "solid_match"


class TestDefaultWeights:
    """DEFAULT_WEIGHTS has the expected structure."""

    def test_keys_present(self):
        assert "reasoning" in DEFAULT_WEIGHTS
        assert "coding" in DEFAULT_WEIGHTS
        assert "translation" in DEFAULT_WEIGHTS
        assert "multimodality" in DEFAULT_WEIGHTS
        assert "health_penalty" in DEFAULT_WEIGHTS
        assert "rate_penalty" in DEFAULT_WEIGHTS
        assert "latency_penalty" in DEFAULT_WEIGHTS
        assert "failures_penalty" in DEFAULT_WEIGHTS

    def test_values_positive(self):
        for v in DEFAULT_WEIGHTS.values():
            assert v > 0


class TestScoreAccountContextExceed:
    """score_account returns 0.0 when context is exceeded."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_context_exceeded_returns_zero(self):
        acct = make_account(context_limit=4_096)
        req = TaskRequirements(context_length=8_192)
        result = self.router.score_account(acct, req)
        assert result.score == 0.0
        assert result.reason == "context_exceeded"

    def test_context_exact_match(self):
        acct = make_account(context_limit=8_192)
        req = TaskRequirements(context_length=8_192)
        result = self.router.score_account(acct, req)
        assert result.score > 0.0

    def test_context_sufficient(self):
        acct = make_account(context_limit=32_768)
        req = TaskRequirements(context_length=4_096)
        result = self.router.score_account(acct, req)
        assert result.score > 0.0


class TestScoreAccountCapabilityScoring:
    """Capability scoring with requirements."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_requires_reasoning_desired(self):
        """Deepseek has reasoning=0.95; weight=0.5 => contribution=0.475."""
        acct = make_account(provider="deepseek")
        req = TaskRequirements(
            requires_reasoning=True,
            priority={"reasoning": 0.5, "coding": 0.0, "translation": 0.0, "multimodality": 0.0},
        )
        result = self.router.score_account(acct, req)
        # 0.5 * 0.95 = 0.475, no penalties => score ~0.475
        assert result.score == pytest.approx(0.475, abs=0.01)

    def test_requires_coding(self):
        """Deepseek has coding=0.95."""
        acct = make_account(provider="deepseek")
        req = TaskRequirements(
            requires_coding=True,
            priority={"reasoning": 0.0, "coding": 0.7, "translation": 0.0, "multimodality": 0.0},
        )
        result = self.router.score_account(acct, req)
        assert result.score == pytest.approx(0.7 * 0.95, abs=0.01)

    def test_requires_translation(self):
        """Qwen has translation=0.85."""
        acct = make_account(provider="qwen")
        req = TaskRequirements(
            requires_translation=True,
            priority={"reasoning": 0.0, "coding": 0.0, "translation": 0.6, "multimodality": 0.0},
        )
        result = self.router.score_account(acct, req)
        assert result.score == pytest.approx(0.6 * 0.85, abs=0.01)

    def test_requires_multimodality(self):
        """ChatGPT has multimodality=0.9."""
        acct = make_account(provider="chatgpt")
        req = TaskRequirements(
            requires_multimodality=True,
            priority={"reasoning": 0.0, "coding": 0.0, "translation": 0.0, "multimodality": 0.8},
        )
        result = self.router.score_account(acct, req)
        assert result.score == pytest.approx(0.8 * 0.9, abs=0.01)

    def test_no_active_requirements_uses_default_weights(self):
        """When no requires_* flags are set, uses DEFAULT_WEIGHTS across all capabilities."""
        acct = make_account(provider="deepseek")
        req = TaskRequirements(
            requires_reasoning=False, requires_coding=False,
            requires_translation=False, requires_multimodality=False,
        )
        result = self.router.score_account(acct, req)
        # Default weights: 0.3*0.95 + 0.3*0.95 + 0.2*0.7 + 0.2*0.3
        # = 0.285 + 0.285 + 0.14 + 0.06 = 0.77
        assert result.score == pytest.approx(0.77, abs=0.01)


class TestScoreAccountHealthPenalty:
    """Health penalty reduces score."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_low_health_reduces_score(self):
        acct = make_account(provider="deepseek", health_score=0.5)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        healthy = make_account(provider="deepseek", health_score=1.0)
        result_healthy = self.router.score_account(healthy, req)
        result = self.router.score_account(acct, req)
        # With health_score=0.5, penalty = (1-0.5)*2.0 = 1.0
        assert result.score < result_healthy.score
        assert result_healthy.score - result.score == pytest.approx(1.0, abs=0.01)

    def test_zero_health_max_penalty(self):
        acct = make_account(provider="deepseek", health_score=0.0)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.score_account(acct, req)
        # Health penalty = (1.0-0.0)*2.0 = 2.0
        # Capability = 1.0*0.95 = 0.95
        # Final = max(0, 0.95 - 2.0) = 0.0
        assert result.score == 0.0


class TestScoreAccountRatePenalty:
    """Rate usage penalty."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_high_rate_usage_reduces_score(self):
        acct = make_account(provider="deepseek", current_rate_usage=0.8)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.score_account(acct, req)
        # Rate penalty = 0.8 * 1.5 = 1.2
        # Capability = 0.95
        # Final = max(0, 0.95 - 1.2) = 0.0
        assert result.score == 0.0

    def test_zero_rate_usage_no_penalty(self):
        acct = make_account(provider="deepseek", current_rate_usage=0.0)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.score_account(acct, req)
        assert result.score == pytest.approx(0.95, abs=0.01)


class TestScoreAccountLatencyPenalty:
    """Latency penalty."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_high_latency_reduces_score(self):
        acct = make_account(provider="deepseek", avg_latency_ms=2000.0)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.score_account(acct, req)
        # Latency penalty = min(2000/1000, 5) * 1.0 = 2.0
        # Capability = 0.95
        # Final = max(0, 0.95 - 2.0) = 0.0
        assert result.score == 0.0

    def test_low_latency_minimal_penalty(self):
        acct = make_account(provider="deepseek", avg_latency_ms=50.0)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.score_account(acct, req)
        # Latency penalty = min(50/1000, 5) * 1.0 = 0.05
        # Capability = 0.95
        # Final = 0.95 - 0.05 = 0.9
        assert result.score == pytest.approx(0.90, abs=0.01)


class TestScoreAccountFailuresPenalty:
    """Consecutive failures penalty."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_failures_reduce_score(self):
        acct = make_account(provider="deepseek", consecutive_failures=3)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.score_account(acct, req)
        # Failures penalty = 3 * 0.5 = 1.5
        # Capability = 0.95
        # Final = max(0, 0.95 - 1.5) = 0.0
        assert result.score == 0.0

    def test_no_failures_no_penalty(self):
        acct = make_account(provider="deepseek", consecutive_failures=0)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.score_account(acct, req)
        assert result.score == pytest.approx(0.95, abs=0.01)


class TestScoreAccountUnknownProvider:
    """Unknown provider returns 0.0."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_unknown_provider_returns_zero(self):
        acct = make_account(provider="nonexistent_provider")
        req = TaskRequirements()
        result = self.router.score_account(acct, req)
        assert result.score == 0.0
        assert result.reason == "unknown_provider"


class TestRankAccounts:
    """rank_accounts orders accounts by descending score."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_returns_empty_for_empty_input(self):
        assert self.router.rank_accounts([], TaskRequirements()) == []

    def test_returns_sorted_descending(self):
        acct_a = make_account(provider="deepseek", health_score=1.0)
        acct_b = make_account(provider="deepseek", health_score=0.3)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        ranked = self.router.rank_accounts([acct_a, acct_b], req)
        assert len(ranked) == 2
        assert ranked[0].score >= ranked[1].score
        assert ranked[0].account is acct_a

    def test_scored_accounts_have_reason(self):
        acct = make_account(provider="deepseek")
        req = TaskRequirements()
        ranked = self.router.rank_accounts([acct], req)
        assert len(ranked) == 1
        assert ranked[0].reason != ""


class TestSelectAccount:
    """select_account returns highest-scoring account or None."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_returns_none_for_empty_list(self):
        assert self.router.select_account([], TaskRequirements()) is None

    def test_returns_best_account(self):
        acct_a = make_account(provider="deepseek", health_score=0.4)
        acct_b = make_account(provider="deepseek", health_score=1.0)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        selected = self.router.select_account([acct_a, acct_b], req)
        assert selected is acct_b

    def test_returns_only_account(self):
        acct = make_account(provider="deepseek")
        selected = self.router.select_account([acct], TaskRequirements())
        assert selected is acct


class TestSelectProvider:
    """select_provider picks the best (provider_name, account) pair."""

    def setup_method(self):
        self.router = ProviderRouter()

    def test_returns_none_for_empty_pool(self):
        assert self.router.select_provider(TaskRequirements(), {}) is None
        assert self.router.select_provider(TaskRequirements(), {"provider_a": []}) is None

    def test_selects_best_provider(self):
        pool = {
            "deepseek": [
                make_account(provider="deepseek", health_score=1.0),
            ],
            "chatgpt": [
                make_account(provider="chatgpt", health_score=0.3),
            ],
        }
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.select_provider(req, pool)
        assert result is not None
        provider, account = result
        assert provider == "deepseek"
        assert account.provider == "deepseek"

    def test_prefers_healthy_over_unhealthy(self):
        pool = {
            "provider_a": [
                make_account(provider="deepseek", health_score=0.2),
            ],
            "provider_b": [
                make_account(provider="qwen", health_score=0.9),
            ],
        }
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = self.router.select_provider(req, pool)
        assert result is not None
        provider, account = result
        # deepseek: reasoning=0.95, health_penalty=(0.8)*2=1.6 => 0.95-1.6 <=0
        # qwen: reasoning=0.85, health_penalty=(0.1)*2=0.2 => 0.85-0.2=0.65
        assert provider == "provider_b"

    def test_returns_typed_tuple(self):
        pool = {
            "deepseek": [make_account(provider="deepseek")],
        }
        result = self.router.select_provider(TaskRequirements(), pool)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], Account)


class TestCustomWeights:
    """Custom weights override defaults."""

    def test_custom_health_penalty(self):
        router = ProviderRouter(weights={"health_penalty": 5.0})
        acct = make_account(provider="deepseek", health_score=0.8)
        req = TaskRequirements(requires_reasoning=True, priority={"reasoning": 1.0})
        result = router.score_account(acct, req)
        # Health penalty = (1-0.8)*5.0 = 1.0
        # Capability = 0.95
        # Final = max(0, 0.95-1.0) = 0.0
        assert result.score == 0.0

    def test_partial_override(self):
        router = ProviderRouter(weights={"reasoning": 0.5})
        assert router.weights["reasoning"] == 0.5
        assert router.weights["coding"] == 0.3  # unchanged default
