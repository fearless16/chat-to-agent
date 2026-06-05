"""UI Schema Engine — provider-independent UI representation.

Every browser provider normalises to this schema so the rest of the
system never touches provider-specific selectors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


class UIRole(StrEnum):
    """Semantic role of a UI element on a chat page."""
    INPUT = "input"
    SEND = "send"
    ASSISTANT_MESSAGE = "assistant_message"
    USER_MESSAGE = "user_message"
    STREAMING_INDICATOR = "streaming_indicator"
    APP_CONTAINER = "app_container"


@dataclass(frozen=True)
class UISelector:
    """A resolved selector for a UI element.

    ``value`` is the Playwright-compatible selector string (CSS, XPath,
    or text-based).  ``source`` records how it was discovered so the
    intelligence layer can prioritise paths.
    """
    role: UIRole
    value: str
    source: str = "cache"  # cache | a11y | dom | deepseek | vision
    confidence: float = 1.0
    fallback_values: tuple[str, ...] = ()


@dataclass
class UIMessage:
    """A single message in the chat view."""
    content: str
    role: str = "assistant"  # user | assistant | system
    selector: Optional[str] = None


@dataclass
class UISchema:
    """Normalised representation of any chat UI page.

    The entire system consumes only this schema.  Provider-specific
    adapters are responsible for populating it via the UI Intelligence
    Layer.
    """
    messages: list[UIMessage] = field(default_factory=list)
    input_box: Optional[UISelector] = None
    send_button: Optional[UISelector] = None
    streaming: bool = False
    title: str = ""
    url: str = ""

    def is_complete(self) -> bool:
        return self.input_box is not None and self.send_button is not None


class UISchemaEngine:
    """Builds and caches UISchema instances for browser pages.

    The engine does *not* discover selectors itself — it is a pure data
    container.  Discovery is handled by the UI Intelligence Layer.
    """

    def __init__(self) -> None:
        self._schema: dict[str, UISchema] = {}

    def register(self, provider: str, schema: UISchema) -> None:
        self._schema[provider] = schema

    def get(self, provider: str) -> Optional[UISchema]:
        return self._schema.get(provider)

    def build_schema(
        self,
        messages: list[UIMessage],
        input_selector: Optional[UISelector] = None,
        send_selector: Optional[UISelector] = None,
        streaming: bool = False,
        title: str = "",
        url: str = "",
    ) -> UISchema:
        return UISchema(
            messages=messages,
            input_box=input_selector,
            send_button=send_selector,
            streaming=streaming,
            title=title,
            url=url,
        )
