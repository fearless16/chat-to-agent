"""Tests for the Provider capability model."""

import pytest

from ai_orchestrator.models.capabilities import (
    CapabilityVector,
    PROVIDER_PROFILES,
    ProviderCapabilities,
    TaskRequirements,
)


class TestCapabilityVector:
    """CapabilityVector validation and defaults."""

    def test_default_values(self):
        """Default vector has reasonable midpoint values."""
        v = CapabilityVector()
        assert v.reasoning == 0.5
        assert v.multimodality == 0.0
        assert v.long_context == 0.3

    def test_values_out_of_range_rejected(self):
        """Values outside [0, 1] are rejected by Pydantic."""
        with pytest.raises(Exception):
            CapabilityVector(reasoning=1.5)

    def test_high_capability_provider(self):
        """DeepSeek has highest reasoning and coding scores."""
        ds = PROVIDER_PROFILES["deepseek_api"]
        assert ds.capabilities.reasoning >= 0.9
        assert ds.capabilities.coding >= 0.9
        assert ds.capabilities.long_context >= 0.8

    def test_local_llm_cost_efficiency(self):
        """Local LLM has best cost efficiency."""
        local = PROVIDER_PROFILES["local_llm"]
        assert local.capabilities.cost_efficiency == 1.0

    def test_chatgpt_api_high_reliability(self):
        """ChatGPT API has high speed and reliability."""
        gpt = PROVIDER_PROFILES["chatgpt_api"]
        assert gpt.capabilities.speed >= 0.8
        assert gpt.capabilities.reliability >= 0.8

    def test_qwen_large_context(self):
        """Qwen has the largest context window among API providers."""
        qwen = PROVIDER_PROFILES["qwen_api"]
        assert qwen.context_limit >= 128_000

    def test_deepseek_million_context(self):
        """DeepSeek supports 1M context."""
        ds = PROVIDER_PROFILES["deepseek_api"]
        assert ds.context_limit == 1_000_000

    def test_task_requirements_defaults(self):
        """TaskRequirements has sensible defaults."""
        req = TaskRequirements()
        assert req.context_length == 4_096
        assert req.requires_reasoning is False
        assert "reasoning" in req.priority

    def test_all_provider_profiles_have_unique_names(self):
        """All provider profiles have provider_name set."""
        names = [p.provider_name for p in PROVIDER_PROFILES.values()]
        assert len(names) == len(set(names))
