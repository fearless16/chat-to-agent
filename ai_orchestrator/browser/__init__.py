"""Browser automation — UI Schema, Selector Cache, Accessibility Runtime, UI Intelligence."""

from ai_orchestrator.browser.schema import (
    UISchema,
    UIMessage,
    UISelector,
    UIRole,
    UISchemaEngine,
)
from ai_orchestrator.browser.selector_cache import SelectorCache
from ai_orchestrator.browser.accessibility import AccessibilityRuntime, A11ySnapshot, A11yNode
from ai_orchestrator.browser.intelligence import UIIntelligence, UIQueryResult

__all__ = [
    "UISchema",
    "UIMessage",
    "UISelector",
    "UIRole",
    "UISchemaEngine",
    "SelectorCache",
    "AccessibilityRuntime",
    "A11ySnapshot",
    "A11yNode",
    "UIIntelligence",
    "UIQueryResult",
]
