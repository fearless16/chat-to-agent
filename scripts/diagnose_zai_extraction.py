"""BLOCKER 1 — Diagnostic: Trace z_ai_ui extraction pipeline.

Run: python scripts/diagnose_zai_extraction.py

Traces every step:
  Network → Observer → Buffer → Classifier → Selection → Returned Text
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")
log = logging.getLogger("diagnose_zai")

# ── Monkey-patch the extraction path with instrumentation ──────────────

from ai_orchestrator.adapters.engine_adapter import (
    EngineUIAdapter,
    GET_INTERCEPTED_BODIES_SCRIPT,
    CLEAR_INTERCEPTED_BODIES_SCRIPT,
)

_original_extract_response_text = EngineUIAdapter._extract_response_text
_original_looks_like_chat = EngineUIAdapter._looks_like_chat_response
_original_sse_delta = EngineUIAdapter._sse_delta_to_text

@staticmethod
async def _instrumented_extract_response_text(page) -> str:
    print("\n" + "=" * 80)
    print("[STEP 1] _extract_response_text — entering")
    print("=" * 80)

    try:
        bodies = await page.evaluate(GET_INTERCEPTED_BODIES_SCRIPT)
        print(f"\n[OBSERVER] Intercepted bodies count: {len(bodies)}")
        for i, b in enumerate(bodies):
            url = (b.get("url") or "")[:120]
            text = (b.get("text") or "")
            transport = b.get("transport", "fetch")
            size = len(text)
            preview = text[:200].replace("\n", "\\n")
            print(f"  [{i}] url={url}")
            print(f"       transport={transport}  size={size}B")
            print(f"       preview={preview}")
    except Exception as e:
        print(f"\n[OBSERVER ERROR] {e}")

    result = await _original_extract_response_text(page)
    print(f"\n[RESULT] _extract_response_text returned ({len(result)}B):")
    print(f"  {result[:300]}")
    return result


# Patch
EngineUIAdapter._extract_response_text = _instrumented_extract_response_text


# ── Also instrument engine.get_response_text() ─────────────────────────

from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine

_original_get_response_text = BrowserIntelligenceEngine.get_response_text

def _instrumented_get_response_text(self):
    text = _original_get_response_text(self)
    if text:
        print(f"\n[ENGINE] get_response_text() returned ({len(text)}B):")
        print(f"  {text[:300]}")
    else:
        print(f"\n[ENGINE] get_response_text() returned empty")
    return text

BrowserIntelligenceEngine.get_response_text = _instrumented_get_response_text


# ── Also instrument the _wait_for_response loop ────────────────────────

from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter

_original_wait_for_response = EngineUIAdapter._wait_for_response

async def _instrumented_wait_for_response(self, engine, page):
    print("\n" + "=" * 80)
    print("[LOOP] _wait_for_response — entering tick loop")
    print("=" * 80)

    extracted = ""
    for tick_idx in range(90):
        store = await engine.tick(page)
        latest = store.latest

        # Engine text
        engine_text = engine.get_response_text()
        if engine_text and len(engine_text) > len(extracted):
            extracted = engine_text

        # DOM extraction attempt
        if latest and latest.stream_active and latest.tokens_per_second > 0:
            if not extracted:
                text = await self._extract_response_text(page)
                if text and len(text) > len(extracted):
                    extracted = text

        done, reason = engine.is_response_complete()
        if done and len(extracted) > 20:
            print(f"\n[LOOP] Tick {tick_idx}: COMPLETE — {reason}")
            print(f"  extracted={len(extracted)}B: {extracted[:200]}")
            break

        if latest and (latest.generation_completed or latest.stream_closed):
            text = engine.get_response_text()
            if text and len(text) > len(extracted):
                extracted = text
            if not extracted:
                text = await self._extract_response_text(page)
                if text and len(text) > len(extracted):
                    extracted = text
            if latest:
                print(f"\n[LOOP] Tick {tick_idx}: generation_completed={latest.generation_completed} stream_closed={latest.stream_closed}")
                print(f"  extracted={len(extracted)}B: {extracted[:200]}")
            break

        await asyncio.sleep(1.0)

    if not extracted:
        text = engine.get_response_text()
        if text:
            extracted = text
        else:
            extracted = await self._extract_response_text(page)

    print(f"\n[FINAL] extracted={len(extracted)}B: {extracted[:300]}")
    return extracted

EngineUIAdapter._wait_for_response = _instrumented_wait_for_response


# ── Run the actual adapter ─────────────────────────────────────────────

async def main():
    from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter
    from ai_orchestrator.orchestrator.main import _load_auth_for

    print("=" * 80)
    print("BLOCKER 1 — z_ai_ui FULL EXTRACTION PIPELINE TRACE")
    print("=" * 80)

    storage_state = _load_auth_for("z_ai_ui")
    print(f"\n[AUTH] storage_state loaded: {bool(storage_state)}")
    if storage_state:
        cookies = storage_state.get("cookies", [])
        print(f"  cookies count: {len(cookies)}")
        for c in cookies:
            print(f"    {c.get('name')}={c.get('value','')[:40]}  domain={c.get('domain')}  expires={c.get('expires','session')}")

    adapter = ZAIUIAdapter(
        mock_mode=False,
        headless=True,
        stealth=True,
        timeout_ms=90_000,
        storage_state=storage_state,
    )

    print("\n[SEND] Calling adapter.send('Hello, what is 2+2?')...")
    t0 = time.monotonic()

    try:
        response = await adapter.send("Hello, what is 2+2?")
        elapsed = time.monotonic() - t0

        print("\n" + "=" * 80)
        print("[FINAL PROVIDER RESPONSE]")
        print("=" * 80)
        print(f"  success:       {response.success}")
        print(f"  content:       {response.content[:500]}")
        print(f"  model:         {response.model}")
        print(f"  latency_ms:    {elapsed * 1000:.1f}")
        print(f"  error:         {response.error}")
        print(f"  content_len:   {len(response.content)}")
        print(f"  is_telemetry:  {response.content.strip().startswith('{\"ResultObject\"')}")

        if response.content.strip().startswith('{"ResultObject"'):
            print("\n[!!!] CONFIRMED: Telemetry returned as final response !!!")
            print(f"  The tracking ping was selected over assistant text.")
        elif len(response.content) > 20:
            print("\n[OK] Response looks like real assistant text.")
        else:
            print("\n[???] Response is too short or empty.")

    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        print(f"  latency: {elapsed * 1000:.1f}ms")
        import traceback
        traceback.print_exc()
    finally:
        await adapter.close()

    print("\n[DONE]")


if __name__ == "__main__":
    asyncio.run(main())
