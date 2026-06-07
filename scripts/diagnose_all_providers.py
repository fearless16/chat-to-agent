"""BLOCKER 1-5 — Production diagnostic for ALL providers.

Captures for each:
  1. Screenshot (.png)
  2. URL + title
  3. Accessibility snapshot (top 100 keys)
  4. Last belief state per tick (logged to JSON)
  5. All intercepted response bodies
  6. Root cause
"""

import asyncio
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter
from ai_orchestrator.adapters.cookie_to_storage_state import netscape_cookies_to_storage_state

OUT = Path("diagnostic_output")
OUT.mkdir(exist_ok=True)

PROVIDERS = [
    "deepseek_ui",
    "z_ai_ui",
    "kimi_ui",
    "chatgpt_ui",
    "xiaomimimo_ui",
    "minimax_ui",
    "qwen_ui",
]

PROVIDER_CLASS_MAP = {
    "chatgpt_ui": "ChatGPTUIAdapter",
    "deepseek_ui": "DeepSeekUIAdapter",
    "kimi_ui": "KimiUIAdapter",
    "qwen_ui": "QwenUIAdapter",
    "z_ai_ui": "ZAIUIAdapter",
    "xiaomimimo_ui": "XiaomiMiMoUIAdapter",
    "minimax_ui": "MiniMaxUIAdapter",
}


def get_cookie_for(provider: str) -> dict | None:
    """Map provider name to cookie file using actual file names."""
    base = Path("profiles")
    files = {
        "deepseek_ui": "deepseek_cookies.txt",
        "z_ai_ui": "zai_cookies.txt",
        "kimi_ui": "kimi_cookies.txt",
        "chatgpt_ui": None,
        "xiaomimimo_ui": "xiaomimimo_cookies.txt",
        "minimax_ui": "minimax_cookies.txt",
        "qwen_ui": None,
    }
    fname = files.get(provider)
    if fname and (base / fname).exists():
        return netscape_cookies_to_storage_state(base / fname)
    return None


async def diagnose_provider(name: str):
    print(f"\n{'#' * 60}")
    print(f"# BLOCKER DIAGNOSTIC: {name}")
    print(f"#{'#' * 60}")

    # Build adapter
    from ai_orchestrator.adapters.chatgpt_ui import ChatGPTUIAdapter
    from ai_orchestrator.adapters.deepseek_ui import DeepSeekUIAdapter
    from ai_orchestrator.adapters.kimi_ui import KimiUIAdapter
    from ai_orchestrator.adapters.qwen_ui import QwenUIAdapter
    from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter
    from ai_orchestrator.adapters.xiaomimimo_ui import XiaomiMiMoUIAdapter
    from ai_orchestrator.adapters.minimax_ui import MiniMaxUIAdapter

    cls_map = {
        "chatgpt_ui": ChatGPTUIAdapter,
        "deepseek_ui": DeepSeekUIAdapter,
        "kimi_ui": KimiUIAdapter,
        "qwen_ui": QwenUIAdapter,
        "z_ai_ui": ZAIUIAdapter,
        "xiaomimimo_ui": XiaomiMiMoUIAdapter,
        "minimax_ui": MiniMaxUIAdapter,
    }

    adapter_cls = cls_map[name]
    storage_state = get_cookie_for(name)

    report = {
        "provider": name,
        "cookie_file": "profiles/" + {
            "deepseek_ui": "deepseek_cookies.txt",
            "z_ai_ui": "zai_cookies.txt",
            "kimi_ui": "kimi_cookies.txt",
            "chatgpt_ui": None,
            "xiaomimimo_ui": "xiaomimimo_cookies.txt",
            "minimax_ui": "minimax_cookies.txt",
            "qwen_ui": None,
        }.get(name, "NONE"),
        "storage_state_loaded": storage_state is not None,
        "ticks": [],
        "intercepted_requests": [],
        "belief_states": [],
    }

    adapter = adapter_cls(
        mock_mode=False,
        headless=True,
        stealth=True,
        timeout_ms=90_000,
        storage_state=storage_state,
    )

    t0 = time.monotonic()
    try:
        page = await adapter._get_page()

        # Capture page state
        url = page.url
        title = await page.title()
        report["url"] = url
        report["title"] = title
        print(f"[PAGE]   url={url}")
        print(f"[PAGE]   title={title}")

        # Screenshot
        shot_path = OUT / f"{name}_screenshot.png"
        try:
            await page.screenshot(path=str(shot_path))
            report["screenshot"] = str(shot_path)
            print(f"[SHOT]   saved to {shot_path}")
        except Exception as e:
            report["screenshot_error"] = str(e)
            print(f"[SHOT]   ERROR: {e}")

        # Accessibility snapshot
        try:
            a11y = await page.aria_snapshot()
            report["a11y_snapshot"] = a11y[:2000] if a11y else "(empty)"
            print(f"[A11Y]   {len(a11y) if a11y else 0} chars")
            # Extract key elements
            key_els = []
            if a11y:
                for line in a11y.split("\n")[:30]:
                    key_els.append(line.strip()[:120])
            report["a11y_key_elements"] = key_els
            print(f"[A11Y]   top elements: {'; '.join(key_els[:5])}")
        except Exception as e:
            report["a11y_error"] = str(e)
            print(f"[A11Y]   ERROR: {e}")

        # Engine attach + readiness ticks
        from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine
        engine = BrowserIntelligenceEngine()
        await engine.attach(page)

        import asyncio as aio_mod
        ready = False
        for tick_idx in range(30):
            store = await engine.tick(page)
            latest = store.latest

            tick_data = {
                "tick": tick_idx,
                "hmm_state": engine.most_likely_state.value if engine.most_likely_state else "N/A",
                "belief": {s.name: round(v, 4) for s, v in engine._hmm.belief.items()} if hasattr(engine, '_hmm') and engine._hmm else {},
                "ready": engine.is_ready_for_prompt,
                "error": engine.is_error,
            }
            report["ticks"].append(tick_data)

            if tick_idx % 5 == 0 or engine.is_ready_for_prompt:
                belief_str = ", ".join(f"{k}={v:.3f}" for k, v in tick_data["belief"].items())
                print(f"[TICK {tick_idx:2d}] state={tick_data['hmm_state']}  ready={tick_data['ready']}  error={tick_data['error']}")
                print(f"          belief: {belief_str}")

            if engine.is_ready_for_prompt:
                ready = True
                break
            if engine.is_error:
                print(f"[TICK {tick_idx:2d}] ENGINE ERROR STATE — stopping ticks")
                break
            await aio_mod.sleep(1.0)

        report["ready_achieved"] = ready
        report["engine_error"] = engine.is_error
        report["most_likely_state"] = engine.most_likely_state

        # Type prompt
        entered_prompt = False
        if ready:
            from ai_orchestrator.adapters.engine_adapter import INPUT_SELECTORS
            input_el = None
            for sel in INPUT_SELECTORS:
                try:
                    loc = page.locator(sel).first
                    if await loc.is_visible():
                        input_el = loc
                        report["input_selector_used"] = sel
                        break
                except Exception:
                    continue

            if input_el:
                await input_el.click()
                await input_el.fill("Hello, what is 2+2?")
                from ai_orchestrator.adapters.engine_adapter import CLEAR_INTERCEPTED_BODIES_SCRIPT
                await page.evaluate(CLEAR_INTERCEPTED_BODIES_SCRIPT)
                await aio_mod.sleep(0.3)
                await page.keyboard.press("Enter")
                entered_prompt = True
                print(f"[PROMPT] typed and sent, selector={report['input_selector_used']}")
            else:
                report["input_error"] = "No visible input element found"
                print("[PROMPT] NO INPUT ELEMENT FOUND")

        report["prompt_entered"] = entered_prompt

        # Wait for response + collect intercepted bodies
        if entered_prompt:
            from ai_orchestrator.adapters.engine_adapter import GET_INTERCEPTED_BODIES_SCRIPT
            extracted = ""
            for tick_idx in range(60):
                store = await engine.tick(page)
                latest = store.latest

                tick_data = {
                    "tick": report["ticks"][-1]["tick"] + 1 if report["ticks"] else tick_idx,
                    "hmm_state": engine.most_likely_state.value if engine.most_likely_state else "N/A",
                    "generating": latest.is_generating if latest else False,
                    "stream_active": latest.stream_active if latest else False,
                    "tokens_per_second": latest.tokens_per_second if latest else 0,
                    "complete": latest.generation_completed if latest else False,
                }
                report["ticks"].append(tick_data)

                # Collect intercepted bodies
                try:
                    bodies = await page.evaluate(GET_INTERCEPTED_BODIES_SCRIPT)
                    if bodies:
                        for b in bodies:
                            entry = {
                                "tick": tick_idx,
                                "url": (b.get("url") or "")[:200],
                                "size": len(b.get("text") or ""),
                                "transport": b.get("transport", "fetch"),
                                "preview": (b.get("text") or "")[:100],
                            }
                            report["intercepted_requests"].append(entry)
                            print(f"[BODY t={tick_idx}] size={entry['size']}B url={entry['url']} preview={entry['preview']}")
                except Exception:
                    pass

                # Try get response text
                engine_text = engine.get_response_text()
                if engine_text and len(engine_text) > len(extracted):
                    extracted = engine_text
                    print(f"[ENGINE t={tick_idx}] get_response_text={extracted[:200]}")

                done, reason = engine.is_response_complete()
                if done:
                    print(f"[DONE t={tick_idx}] completion: {reason}")
                    break

                if latest and (latest.generation_completed or latest.stream_closed):
                    print(f"[END t={tick_idx}] generation_completed={latest.generation_completed} stream_closed={latest.stream_closed}")
                    engine_text = engine.get_response_text()
                    if engine_text and len(engine_text) > len(extracted):
                        extracted = engine_text
                    break

                await aio_mod.sleep(1.0)

            # Final extraction
            if not extracted:
                from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter
                try:
                    bodies = await page.evaluate(GET_INTERCEPTED_BODIES_SCRIPT)
                    if bodies:
                        from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter as E
                        for entry in reversed(bodies):
                            text = (entry.get("text") or "").strip()
                            url = (entry.get("url") or "").lower()
                            if not text or len(text) < 10:
                                continue
                            delta = E._sse_delta_to_text(text)
                            if delta and len(delta) > 1:
                                extracted = delta
                                print(f"[FALLBACK SSE] got delta ({len(delta)}B)")
                                break
                            if text.startswith("data:") or text.startswith("[") and text.endswith("]"):
                                continue
                            if len(text) > 20:
                                extracted = text
                                print(f"[FALLBACK RAW] got text ({len(text)}B) url={url}")
                                break
                        # DOM fallback
                        if not extracted:
                            from ai_orchestrator.adapters.engine_adapter import RESPONSE_EXTRACT_SCRIPT
                            dom_text = (await page.evaluate(RESPONSE_EXTRACT_SCRIPT)).strip()
                            if dom_text:
                                extracted = dom_text
                                print(f"[FALLBACK DOM] got text ({len(dom_text)}B)")
                except Exception:
                    pass

            report["final_extracted"] = extracted[:500]
            report["final_extracted_len"] = len(extracted)
            report["is_tracking_ping"] = extracted.strip().startswith('{"ResultObject"')
            report["is_captcha"] = '"Code":"Success"' in extracted or '"Code":"200"' in extracted

            print(f"\n[FINAL EXTRACTED] {len(extracted)}B")
            print(f"  is_tracking_ping: {report['is_tracking_ping']}")
            print(f"  is_captcha: {report['is_captcha']}")
            print(f"  text: {extracted[:300]}")

    except Exception as e:
        import traceback
        report["error"] = f"{type(e).__name__}: {e}"
        report["traceback"] = traceback.format_exc()
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        try:
            await adapter.close()
        except Exception:
            pass

    report["elapsed_ms"] = (time.monotonic() - t0) * 1000

    # Save report
    report_path = OUT / f"{name}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n[REPORT] saved to {report_path}")

    return report


async def main():
    import importlib

    results = {}
    for provider in PROVIDERS:
        results[provider] = await diagnose_provider(provider)
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: {provider}")
        r = results[provider]
        print(f"  url:               {r.get('url','N/A')}")
        print(f"  title:             {r.get('title','N/A')}")
        print(f"  storage_state:     {r.get('storage_state_loaded')}")
        print(f"  ready_achieved:    {r.get('ready_achieved')}")
        print(f"  engine_error:      {r.get('engine_error')}")
        print(f"  prompt_entered:    {r.get('prompt_entered')}")
        print(f"  final_extracted:   {r.get('final_extracted','')[:100]}")
        print(f"  final_len:         {r.get('final_extracted_len',0)}")
        print(f"  is_tracking_ping:  {r.get('is_tracking_ping')}")
        print(f"  error:             {r.get('error','none')}")
        print(f"  elapsed_ms:        {r.get('elapsed_ms',0):.0f}")

    # Master summary
    summary_path = OUT / "master_summary.json"
    summary = {
        "systematic_cookie_mismatch": {
            "fact": "ALL cookie files use short names (e.g. deepseek_cookies.txt) but _load_auth_for expects full names (e.g. deepseek_ui_cookies.txt)",
            "affected": "deepseek_ui, z_ai_ui, kimi_ui, xiaomimimo_ui, minimax_ui",
            "unaffected": "chatgpt_ui, qwen_ui (no cookie files exist)",
            "fix": "Rename cookie files OR update _load_auth_for to map provider names to file names",
        },
        "results": {},
    }
    for provider, r in results.items():
        summary["results"][provider] = {
            "url": r.get("url"),
            "title": r.get("title"),
            "storage_state_loaded": r.get("storage_state_loaded"),
            "ready_achieved": r.get("ready_achieved"),
            "engine_error": r.get("engine_error"),
            "prompt_entered": r.get("prompt_entered"),
            "final_extracted_len": r.get("final_extracted_len"),
            "final_text_preview": r.get("final_extracted","")[:150],
            "error": r.get("error"),
            "is_tracking_ping": r.get("is_tracking_ping"),
            "elapsed_ms": r.get("elapsed_ms"),
        }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n\n[MASTER SUMMARY] saved to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
