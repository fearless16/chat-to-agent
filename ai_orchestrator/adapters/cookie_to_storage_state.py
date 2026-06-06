"""Convert Netscape cookie files to Playwright storage state dicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def netscape_cookies_to_storage_state(
    cookie_file: str | Path, domain_override: str | None = None
) -> dict[str, Any]:
    """Parse a Netscape-format cookie file and return a Playwright storage state dict.

    The output dict has ``cookies`` and ``origins`` keys suitable for
    passing as ``storage_state=`` to ``browser.new_context()``.

    Netscape format (one cookie per line):
        domain  flag  path  secure  expiry  name  value
    Lines starting with ``#`` are comments and skipped.
    """
    cookies: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    path = Path(cookie_file)
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 7:
            continue

        domain, _flag, cookie_path, secure_str, expiry_str, name, value = parts[:7]

        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]

        domain = domain.lstrip(".")
        secure = secure_str.upper() == "TRUE"

        expires = -1.0
        if expiry_str.isdigit() or (expiry_str.startswith("-") and expiry_str[1:].isdigit()):
            expiry_int = int(expiry_str)
            if expiry_int > 0:
                expires = float(expiry_int)

        key = (domain, name, cookie_path)
        if key in seen:
            continue
        seen.add(key)

        if domain_override:
            domain = domain_override

        cookies.append({
            "name": name,
            "value": value,
            "domain": domain,
            "path": cookie_path,
            "expires": expires,
            "httpOnly": False,
            "secure": secure,
            "sameSite": "Lax",
        })

    return {
        "cookies": cookies,
        "origins": [],
    }
