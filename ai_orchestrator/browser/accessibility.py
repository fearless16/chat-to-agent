"""Accessibility Runtime — Playwright accessibility snapshot wrapper.

Uses ``page.accessibility.snapshot()`` which is compact, semantic,
stable across UI changes, and orders of magnitude cheaper than full
HTML or Vision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class A11yNode:
    """A single node from the accessibility tree."""
    role: str
    name: str
    value: Optional[str] = None
    description: Optional[str] = None
    focused: bool = False
    disabled: bool = False
    checked: Optional[bool] = None
    children: list["A11yNode"] = field(default_factory=list)

    @property
    def is_interactive(self) -> bool:
        return self.role in (
            "button", "link", "textbox", "combobox",
            "searchbox", "menuitem", "checkbox", "radio",
            "slider", "tab", "treeitem",
        )

    @property
    def is_text_input(self) -> bool:
        return self.role in ("textbox", "searchbox", "combobox")

    @property
    def is_button(self) -> bool:
        return self.role == "button"


@dataclass
class A11ySnapshot:
    """The full accessibility tree as a flat iterable."""
    root: Optional[A11yNode] = None
    _flat: list[A11yNode] = field(default_factory=list)

    def all_nodes(self) -> list[A11yNode]:
        return self._flat

    def interactive_nodes(self) -> list[A11yNode]:
        return [n for n in self._flat if n.is_interactive]

    def text_inputs(self) -> list[A11yNode]:
        return [n for n in self._flat if n.is_text_input]

    def buttons(self) -> list[A11yNode]:
        return [n for n in self._flat if n.is_button]


class AccessibilityRuntime:
    """Wraps Playwright ``page.accessibility.snapshot()``.

    Usage::

        rt = AccessibilityRuntime()
        snap = await rt.snapshot(page)
        for inp in snap.text_inputs():
            print(inp.name, inp.value)
    """

    async def snapshot(self, page) -> A11ySnapshot:
        """Take accessibility snapshot.

        Uses Playwright page.aria_snapshot() which returns a YAML-like
        accessibility tree. This is the stable, supported API in
        Playwright >=1.30. Falls back to the legacy
        page.accessibility.snapshot() CDP path for older versions.
        """
        raw = None
        try:
            raw = await page.aria_snapshot()
        except AttributeError:
            try:
                raw = await page.accessibility.snapshot()
            except AttributeError:
                return A11ySnapshot()
        except Exception:
            return A11ySnapshot()

        if raw is None:
            return A11ySnapshot()

        root = self._parse_node(raw)
        flat: list[A11yNode] = []
        self._flatten(root, flat)
        return A11ySnapshot(root=root, _flat=flat)

    async def find_input(
        self, page,
        hint: Optional[str] = None,
    ) -> Optional[A11yNode]:
        """Find the most likely text-input element.

        With *hint* (e.g. ``"message"``) prefers inputs whose name
        contains the hint.
        """
        snap = await self.snapshot(page)
        inputs = snap.text_inputs()
        if not inputs:
            return None
        if hint:
            for inp in inputs:
                if hint.lower() in (inp.name or "").lower():
                    return inp
        return inputs[0]

    async def find_send_button(
        self, page,
        hint: Optional[str] = None,
    ) -> Optional[A11yNode]:
        """Find the most likely send/submit button.

        With *hint* (e.g. ``"send"``) prefers buttons whose name
        contains the hint.
        """
        snap = await self.snapshot(page)
        buttons = snap.buttons()
        if not buttons:
            return None
        if hint:
            for btn in buttons:
                if hint.lower() in (btn.name or "").lower():
                    return btn
        return buttons[0]

    async def find_message_container(
        self, page,
        hint: Optional[str] = None,
    ) -> Optional[A11yNode]:
        snap = await self.snapshot(page)
        candidates: list[A11yNode] = []
        for node in snap.all_nodes():
            if node.role in ("article", "region", "section", "group"):
                name_lower = (node.name or "").lower()
                if "message" in name_lower or "assistant" in name_lower or "response" in name_lower or "conversation" in name_lower:
                    candidates.append(node)
        if not candidates:
            for node in snap.all_nodes():
                if node.role in ("article", "list", "listitem", "region", "section", "group"):
                    name_lower = (node.name or "").lower()
                    if hint and hint.lower() in name_lower:
                        candidates.append(node)
        if not candidates:
            return None
        candidates.sort(key=lambda n: len(n.name or ""), reverse=True)
        return candidates[0]

    # ── internal ────────────────────────────────────────────────────

    def _parse_node(self, raw: dict | str) -> A11yNode:
        """Parse a dict-based accessibility snapshot into an A11yNode tree.

        Handles both Playwright's ``page.accessibility.snapshot()`` dict
        format and the string-based aria snapshot format.
        """
        if isinstance(raw, str):
            return self._parse_aria_yaml(raw)
        return self._parse_dict_node(raw)

    def _parse_dict_node(self, data: dict) -> A11yNode:
        node = A11yNode(
            role=data.get("role", "unknown"),
            name=data.get("name", ""),
            value=data.get("value"),
            description=data.get("description"),
            focused=data.get("focused", False),
            disabled=data.get("disabled", False),
            checked=data.get("checked"),
        )
        for child_data in data.get("children", []):
            node.children.append(self._parse_dict_node(child_data))
        return node

    def _parse_aria_yaml(self, text: str) -> A11yNode:
        lines = text.splitlines()
        root = A11yNode(role="WebArea", name="")
        stack: list[tuple[int, A11yNode]] = [(-1, root)]

        for line in lines:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            content = line.lstrip()

            content = content.removeprefix("- ")
            role = content.split(" ")[0].split(":")[0] if content else ""
            name = ""
            value = None
            description = ""

            remainder = content[len(role):].strip()

            if remainder.endswith(":"):
                pass
            elif ":" in remainder:
                colon_idx = remainder.index(":")
                value = remainder[colon_idx + 1:].strip().strip('"')
                remainder = remainder[:colon_idx].strip()

            while remainder:
                if remainder.startswith('"'):
                    quote_end = remainder.index('"', 1)
                    name = remainder[1:quote_end]
                    remainder = remainder[quote_end + 1:].strip()
                elif remainder.startswith("["):
                    bracket_end = remainder.index("]")
                    description = remainder[1:bracket_end]
                    remainder = remainder[bracket_end + 1:].strip()
                else:
                    break

            node = A11yNode(
                role=role,
                name=name or value or "",
                value=value if value else None,
                description=description or None,
            )

            while stack and stack[-1][0] >= indent:
                stack.pop()

            stack[-1][1].children.append(node)

            if content.rstrip().endswith(":"):
                stack.append((indent, node))

        return root

    def _flatten(self, node: A11yNode, acc: list[A11yNode]) -> None:
        acc.append(node)
        for c in node.children:
            self._flatten(c, acc)
