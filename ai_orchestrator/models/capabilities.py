"""Capability matrix — provider capability vector definitions and scoring."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CapabilityVector(BaseModel):
    """A provider's capability profile on key dimensions.

    Values range from 0.0 (none) to 1.0 (best-in-class).
    """
    reasoning: float = Field(default=0.5, ge=0.0, le=1.0)
    coding: float = Field(default=0.5, ge=0.0, le=1.0)
    translation: float = Field(default=0.5, ge=0.0, le=1.0)
    multimodality: float = Field(default=0.0, ge=0.0, le=1.0)
    speed: float = Field(default=0.5, ge=0.0, le=1.0)
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    cost_efficiency: float = Field(default=0.5, ge=0.0, le=1.0)
    long_context: float = Field(default=0.3, ge=0.0, le=1.0)


class TaskRequirements(BaseModel):
    """What a task needs from a provider."""
    context_length: int = Field(default=4_096, ge=1)
    requires_reasoning: bool = False
    requires_coding: bool = False
    requires_translation: bool = False
    requires_multimodality: bool = False
    priority: dict[str, float] = Field(default_factory=lambda: {
        "reasoning": 0.5,
        "coding": 0.5,
        "translation": 0.0,
        "multimodality": 0.0,
    })


class ProviderCapabilities(BaseModel):
    """Full capability record for a provider."""
    provider_name: str
    transport: str = Field(default="API")
    capabilities: CapabilityVector = Field(default_factory=CapabilityVector)
    context_limit: int = Field(default=8_192)
    max_concurrent: int = Field(default=5)
    supports_streaming: bool = False
    supports_tools: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


# Pre-defined capability profiles matching the architecture doc's capability table
PROVIDER_PROFILES: dict[str, ProviderCapabilities] = {
    "chatgpt_api": ProviderCapabilities(
        provider_name="chatgpt",
        transport="API",
        capabilities=CapabilityVector(
            reasoning=0.9, coding=0.85, translation=0.8,
            multimodality=0.9, speed=0.9, reliability=0.9,
            cost_efficiency=0.2, long_context=0.3,
        ),
        context_limit=32_768,
        max_concurrent=10,
        supports_streaming=True,
        supports_tools=True,
    ),
    "chatgpt_ui": ProviderCapabilities(
        provider_name="chatgpt",
        transport="BROWSER",
        capabilities=CapabilityVector(
            reasoning=0.85, coding=0.8, translation=0.75,
            multimodality=0.9, speed=0.3, reliability=0.3,
            cost_efficiency=0.9, long_context=0.3,
        ),
        context_limit=32_768,
        max_concurrent=3,
        supports_streaming=False,
        supports_tools=False,
    ),
    "qwen_api": ProviderCapabilities(
        provider_name="qwen",
        transport="API",
        capabilities=CapabilityVector(
            reasoning=0.85, coding=0.8, translation=0.85,
            multimodality=0.7, speed=0.7, reliability=0.7,
            cost_efficiency=0.6, long_context=0.95,
        ),
        context_limit=131_072,  # Qwen3.5: 128K+
        max_concurrent=10,
        supports_streaming=True,
        supports_tools=True,
    ),
    "deepseek_api": ProviderCapabilities(
        provider_name="deepseek",
        transport="API",
        capabilities=CapabilityVector(
            reasoning=0.95, coding=0.95, translation=0.7,
            multimodality=0.3, speed=0.8, reliability=0.8,
            cost_efficiency=0.85, long_context=0.9,
        ),
        context_limit=1_000_000,  # 1M context
        max_concurrent=10,
        supports_streaming=True,
        supports_tools=True,
    ),
    "kimi_api": ProviderCapabilities(
        provider_name="kimi",
        transport="API",
        capabilities=CapabilityVector(
            reasoning=0.7, coding=0.6, translation=0.85,
            multimodality=0.5, speed=0.6, reliability=0.6,
            cost_efficiency=0.7, long_context=0.85,
        ),
        context_limit=128_000,
        max_concurrent=5,
        supports_streaming=True,
        supports_tools=False,
    ),
    "local_llm": ProviderCapabilities(
        provider_name="local_llm",
        transport="LOCAL",
        capabilities=CapabilityVector(
            reasoning=0.5, coding=0.5, translation=0.5,
            multimodality=0.2, speed=0.3, reliability=0.6,
            cost_efficiency=1.0, long_context=0.9,
        ),
        context_limit=256_000,
        max_concurrent=2,
        supports_streaming=True,
        supports_tools=False,
    ),
}
