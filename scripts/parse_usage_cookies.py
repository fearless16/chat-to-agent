#!/usr/bin/env python3
"""Parse cookies from USAGE.md and save as Netscape cookie files.

Reads the JSON cookie arrays from USAGE.md and converts them to
Netscape HTTP Cookie File format for each provider.
"""

import json
import re
import time
from pathlib import Path


def json_cookies_to_netscape(cookies: list[dict]) -> str:
    """Convert browser-extension JSON cookies to Netscape format."""
    lines = [
        "# Netscape HTTP Cookie File",
        "# Auto-converted from USAGE.md",
        f"# Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    for c in cookies:
        domain = c.get("domain", "")
        flag = "TRUE" if domain.startswith(".") else "FALSE"
        path = c.get("path", "/")
        secure = "TRUE" if c.get("secure", False) else "FALSE"
        expiry = str(int(c.get("expirationDate", 0)))
        name = c.get("name", "")
        value = c.get("value", "")
        if not name:
            continue
        lines.append(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
    lines.append("")
    return "\n".join(lines)


def main():
    usage = Path("USAGE.md").read_text()

    # Define provider sections and their JSON cookie arrays
    sections = {
        "chatgpt_ui": "==> chat gpt cookies:",
        "deepseek_ui": "==> deepseek:",
        "qwen_ui": "==> qwen:",
        "minimax_ui": "==> minimax",
        "xiaomimimo_ui": "ximoi mimo->",
        "kimi_ui": "kimi->",
        "z_ai_ui": "Zai =>",
    }

    profiles_dir = Path("profiles")
    profiles_dir.mkdir(exist_ok=True)

    for provider, marker in sections.items():
        idx = usage.find(marker)
        if idx == -1:
            print(f"⚠️  {provider}: marker '{marker}' not found in USAGE.md")
            continue

        # Find the JSON array after the marker
        rest = usage[idx:]
        # Find first [ and matching ]
        start = rest.find("[")
        if start == -1:
            print(f"⚠️  {provider}: no JSON array found after marker")
            continue

        # Find matching closing bracket
        depth = 0
        end = -1
        for i, ch in enumerate(rest[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end == -1:
            print(f"⚠️  {provider}: no matching ] found")
            continue

        json_str = rest[start:end]
        try:
            cookies = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"⚠️  {provider}: JSON parse error: {e}")
            continue

        base = provider.removesuffix("_ui") if provider.endswith("_ui") else provider

        # Save Netscape format
        netscape = json_cookies_to_netscape(cookies)
        netscape_path = profiles_dir / f"{base}_cookies.txt"
        netscape_path.write_text(netscape)

        # Also save as _ui variant
        netscape_ui_path = profiles_dir / f"{provider}_cookies.txt"
        netscape_ui_path.write_text(netscape)

        # Save Playwright storage state JSON
        storage = {"cookies": cookies, "origins": []}
        auth_path = profiles_dir / f"{base}_auth.json"
        auth_path.write_text(json.dumps(storage, indent=2))

        print(f"✅ {provider}: saved {len(cookies)} cookies → {netscape_path}")


if __name__ == "__main__":
    main()
