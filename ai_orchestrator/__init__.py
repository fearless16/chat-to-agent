"""
ai_orchestrator — Production-grade multi-provider AI orchestration platform.

Coordinates multiple AI providers (OpenAI, Qwen, DeepSeek, Kimi, local LLMs)
with pooled accounts, parallel agents, dynamic resource scheduling, and
full observability.
"""

__version__ = "0.1.0"

from ai_orchestrator.workspace import ArtifactStore, FileWorkspace, GitWorkspace
from ai_orchestrator.security import CredentialVault, PromptGuard, Sandbox
from ai_orchestrator.runtime import RuntimeLoop
from ai_orchestrator.testrunner import TestRunner
from ai_orchestrator.orchestrator import (
    LeaseManager,
    ProviderRouter,
    ResourceScheduler,
    WorkflowEngine,
)
from ai_orchestrator.models import Account, Lease, Task, TaskStatus
from ai_orchestrator.agents import (
    CoderAgent,
    ExecutorAgent,
    FixerAgent,
    PlannerAgent,
    ResearcherAgent,
    ReviewerAgent,
    TesterAgent,
)

__all__ = [
    "Account",
    "ArtifactStore",
    "CoderAgent",
    "CredentialVault",
    "ExecutorAgent",
    "FileWorkspace",
    "FixerAgent",
    "GitWorkspace",
    "Lease",
    "LeaseManager",
    "PlannerAgent",
    "PromptGuard",
    "ProviderRouter",
    "ResearcherAgent",
    "ResourceScheduler",
    "ReviewerAgent",
    "RuntimeLoop",
    "Sandbox",
    "Task",
    "TaskStatus",
    "TesterAgent",
    "TestRunner",
    "WorkflowEngine",
]
