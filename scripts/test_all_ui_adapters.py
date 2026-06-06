"""Real browser test — all 7 UI adapters with cookie / auth profiles."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from ai_orchestrator.adapters.cookie_to_storage_state import netscape_cookies_to_storage_state

PROFILES_DIR = Path("profiles")
PROMPT = "What is 2+2? Reply in ONE word only."

# Netscape-format cookie files (profiles/<name>.txt)
COOKIE_MAP = {
    "deepseek_ui":   "deepseek_cookies.txt",
    "kimi_ui":       "kimi_cookies.txt",
    "z_ai_ui":       "zai_cookies.txt",
    "xiaomimimo_ui": "xiaomimimo_cookies.txt",
    "minimax_ui":    "minimax_cookies.txt",
}

# Playwright storage_state JSON files (in repo root)
AUTH_JSON_MAP = {
    "chatgpt_ui": "chatgpt_auth.json",
    "qwen_ui":    "qwen_auth.json",
}

ADAPTER_MAP = {
    "chatgpt_ui":   ("ai_orchestrator.adapters.chatgpt_ui",    "ChatGPTUIAdapter"),
    "deepseek_ui":  ("ai_orchestrator.adapters.deepseek_ui",   "DeepSeekUIAdapter"),
    "kimi_ui":      ("ai_orchestrator.adapters.kimi_ui",       "KimiUIAdapter"),
    "z_ai_ui":      ("ai_orchestrator.adapters.zai_ui",        "ZAIUIAdapter"),
    "xiaomimimo_ui":("ai_orchestrator.adapters.xiaomimimo_ui", "XiaomiMiMoUIAdapter"),
    "minimax_ui":   ("ai_orchestrator.adapters.minimax_ui",    "MiniMaxUIAdapter"),
    "qwen_ui":      ("ai_orchestrator.adapters.qwen_ui",       "QwenUIAdapter"),
}


def _load_storage_state(name: str) -> dict | None:
    if name in COOKIE_MAP:
        cookie_path = PROFILES_DIR / COOKIE_MAP[name]
        if cookie_path.exists():
            return netscape_cookies_to_storage_state(cookie_path)
    if name in AUTH_JSON_MAP:
        auth_path = Path(AUTH_JSON_MAP[name])
        if auth_path.exists():
            with auth_path.open() as fh:
                return json.load(fh)
    return None


async def test_provider(name: str) -> dict:
    mod_name, cls_name = ADAPTER_MAP[name]
    mod = __import__(mod_name, fromlist=[cls_name])
    adapter_cls = getattr(mod, cls_name)

    storage_state = _load_storage_state(name)

    adapter = adapter_cls(
        mock_mode=False,
        headless=True,
        stealth=True,
        timeout_ms=90_000,
        storage_state=storage_state,
        persistent_profile=None,
        channel="chromium",
    )

    result = {
        "provider": name,
        "success": False,
        "content": "",
        "error": "",
        "latency_ms": 0,
        "model": "",
        "has_storage_state": storage_state is not None,
    }

    try:
        resp = await adapter.send(PROMPT, context=None)
        result["success"] = resp.success
        result["latency_ms"] = resp.latency_ms
        result["model"] = resp.model
        if resp.success:
            result["content"] = resp.content[:500]
        else:
            result["error"] = resp.error or "unknown error"
    except Exception as exc:
        result["error"] = str(exc)
    finally:
        await adapter.close()

    return result


async def main():
    providers = list(ADAPTER_MAP)
    results = []

    for i, name in enumerate(providers):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(providers)}] Testing {name}...")
        print(f"{'='*60}")
        result = await test_provider(name)
        results.append(result)

        if result["success"]:
            print(f"  OK   ({result['latency_ms']:.0f}ms)  model={result['model']}")
            print(f"  >>>  {result['content'][:200]}")
        else:
            print(f"  FAIL ({result['latency_ms']:.0f}ms)  has_storage_state={result['has_storage_state']}")
            print(f"  ERR  {result['error'][:300]}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r["success"])
    print(f"  {passed}/{len(results)} providers responded successfully")
    for r in results:
        status = "OK  " if r["success"] else "FAIL"
        snippet = (r["content"] or r["error"])[:80].replace("\n", " ")
        print(f"  {status}  {r['provider']:<14}  {snippet}")
    print()

    # Dump structured JSON for downstream consumption.
    Path("ui_adapter_results.json").write_text(json.dumps(results, indent=2))
    print(f"  >> wrote ui_adapter_results.json")

    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
