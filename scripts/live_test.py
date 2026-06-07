"""Live-monitored non-headless browser test for a single provider."""
import asyncio
import sys
import time
import traceback
import os

sys.path.insert(0, ".")

PROVIDER = sys.argv[1] if len(sys.argv) > 1 else "deepseek_ui"


async def monitor_browser(page, adapter):
    """Poll the page every 3s and dump DOM state."""
    tick = 0
    while True:
        tick += 1
        await asyncio.sleep(3)
        try:
            url = page.url
            title = await page.title()
            # Get a small DOM snapshot
            body_text = await page.evaluate("""() => {
                const b = document.body;
                if (!b) return 'NO BODY';
                // Get visible text summary
                const walker = document.createTreeWalker(b, NodeFilter.SHOW_TEXT);
                let txt = '';
                let count = 0;
                while (walker.nextNode() && count < 50) {
                    const t = walker.currentNode.nodeValue.trim();
                    if (t && t.length > 2) { txt += t.substring(0, 80) + ' | '; count++; }
                }
                return txt.substring(0, 500) || '(empty body)';
            }""")
            # Check for error banners
            error_el = await page.evaluate("""() => {
                const es = document.querySelectorAll('[class*="error"], [class*="Error"], [role="alert"]');
                return Array.from(es).slice(0, 3).map(e => (e.textContent || '').substring(0, 100));
            }""")
            # Check for input element
            has_input = await page.evaluate("""() => {
                return !!document.querySelector('textarea, [contenteditable="true"], [role="textbox"]');
            }""")

            print(f"\n[TICK {tick}] url={url[:120]}")
            print(f"  title={title[:80]}")
            print(f"  has_input={has_input}")
            print(f"  body_preview={body_text[:300]}")
            if error_el and any(e for e in error_el if e.strip()):
                print(f"  ERRORS={error_el}")
            # Check if page is still alive
            alive = not page.is_closed()
            print(f"  page_alive={alive}")
            if not alive:
                print("  PAGE CLOSED - stopping monitor")
                break
        except Exception as e:
            print(f"\n[TICK {tick}] MONITOR ERROR: {type(e).__name__}: {str(e)[:200]}")
            break


async def main():
    # Import adapter class
    adapter_map = {
        "deepseek_ui": ("ai_orchestrator.adapters.deepseek_ui", "DeepSeekUIAdapter"),
        "chatgpt_ui": ("ai_orchestrator.adapters.chatgpt_ui", "ChatGPTUIAdapter"),
        "qwen_ui": ("ai_orchestrator.adapters.qwen_ui", "QwenUIAdapter"),
        "kimi_ui": ("ai_orchestrator.adapters.kimi_ui", "KimiUIAdapter"),
        "z_ai_ui": ("ai_orchestrator.adapters.zai_ui", "ZAIUIAdapter"),
        "minimax_ui": ("ai_orchestrator.adapters.minimax_ui", "MiniMaxUIAdapter"),
        "xiaomimimo_ui": ("ai_orchestrator.adapters.xiaomimimo_ui", "XiaomiMiMoUIAdapter"),
    }

    if PROVIDER not in adapter_map:
        print(f"Unknown provider: {PROVIDER}")
        print(f"Available: {list(adapter_map)}")
        return

    mod_name, cls_name = adapter_map[PROVIDER]
    mod = __import__(mod_name, fromlist=[cls_name])
    Cls = getattr(mod, cls_name)

    print(f"=== Testing {PROVIDER} in NON-HEADLESS mode with live monitoring ===")
    print(f"   PID: {os.getpid()}")
    sys.stdout.flush()

    adapter = Cls(
        mock_mode=False,
        headless=False,        # VISIBLE browser
        stealth=True,
        timeout_ms=180_000,
        persistent_profile=None,
        storage_state=None,
        channel="chromium",
    )

    monitor_task = None
    try:
        t0 = time.monotonic()

        # Instead of calling adapter.send() (which does everything),
        # let's manually control the flow to monitor at each step

        from playwright.async_api import async_playwright
        from ai_orchestrator.browser_intelligence.engine import BrowserIntelligenceEngine
        from ai_orchestrator.adapters.engine_adapter import (
            INPUT_SELECTORS, FETCH_INTERCEPT_SCRIPT, CLEAR_INTERCEPTED_BODIES_SCRIPT,
        )

        # Step 1: Launch browser directly with crashpad disabled
        print("\n[1] Starting Playwright and launching browser...")
        sys.stdout.flush()

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(
            headless=False,
            channel="chromium",
            args=[
                "--no-sandbox",
                "--disable-gpu",
                "--disable-features=Crashpad",
                "--disable-crashpad-for-testing",
                "--disable-breakpad",
                "--disable-field-trial-config",
            ]
        )
        print(f"   Browser launched OK: {browser}")

        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        )
        page = await context.new_page()

        # Add fetch interceptor
        await page.add_init_script(FETCH_INTERCEPT_SCRIPT)

        # Step 2: Navigate to provider
        site_url = adapter._site.url
        print(f"\n[2] Navigating to {site_url}...")
        sys.stdout.flush()

        try:
            await page.goto(site_url, timeout=60000, wait_until="load")
            print(f"   Navigation OK. Current URL: {page.url}")
        except Exception as e:
            print(f"   Navigation error: {type(e).__name__}: {e}")
            # Try to continue anyway

        # Step 3: Wait for hydration
        print("\n[3] Waiting for SPA to hydrate...")
        sys.stdout.flush()
        await asyncio.sleep(5)

        # Step 4: Start monitoring + engine
        print("\n[4] Starting engine + monitor...")
        sys.stdout.flush()

        engine = BrowserIntelligenceEngine()
        await engine.attach(page)

        # Start monitor in background
        monitor_task = asyncio.create_task(monitor_browser(page, adapter))

        # Step 5: Tick engine until ready
        print("\n[5] Ticking engine until READY...")
        sys.stdout.flush()
        ready = False
        MAX_TICKS = 30
        for i in range(MAX_TICKS):
            await engine.tick(page)
            state = engine.most_likely_state or "unknown"
            ready_flag = engine.is_ready_for_prompt
            print(f"   tick {i+1}: state={state}, ready={ready_flag}")
            sys.stdout.flush()
            if ready_flag:
                # Need consecutive ready ticks
                consecutive = sum(1 for _ in range(min(3, i+1)))
                ready = True
                break
            if engine.is_error:
                print(f"   ENGINE ERROR: {state}")
                break
            await asyncio.sleep(1)

        if not ready:
            print(f"\n   TIMEOUT: Engine not ready after {MAX_TICKS} ticks")
            # Take DOM snapshot
            dom = await page.evaluate("() => document.body ? document.body.innerHTML.substring(0, 2000) : 'no body'")
            print(f"   DOM: {dom[:1000]}")
        else:
            print(f"\n   ENGINE READY! State={engine.most_likely_state}")

            # Step 6: Type prompt
            print("\n[6] Typing prompt...")
            sys.stdout.flush()
            try:
                input_el = None
                for sel in INPUT_SELECTORS:
                    try:
                        loc = page.locator(sel).first
                        if await loc.is_visible():
                            input_el = loc
                            break
                    except:
                        continue
                if input_el:
                    await input_el.click()
                    await input_el.fill("Say exactly: 'The answer is 42.' Do not add anything else.")
                    await page.evaluate(CLEAR_INTERCEPTED_BODIES_SCRIPT)
                    await asyncio.sleep(0.3)
                    await page.keyboard.press("Enter")
                    print("   Prompt sent!")
                else:
                    print("   No input element found!")
            except Exception as e:
                print(f"   Type error: {e}")

            # Step 7: Wait for response
            print("\n[7] Waiting for response...")
            sys.stdout.flush()
            for i in range(60):
                await engine.tick(page)
                text = engine.get_response_text()
                done, _ = engine.is_response_complete()
                if done and text and len(text) > 20:
                    print(f"   Response complete! {len(text)} chars")
                    break
                await asyncio.sleep(1)
                if i % 5 == 0:
                    print(f"   tick {i+1}: streaming={engine._composer._network._sse_observer._response_text[:50] if hasattr(engine._composer._network, '_sse_observer') else 'n/a'}")

            # Step 8: Extract results
            print("\n[8] Extracting response...")
            sys.stdout.flush()
            from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter
            content = await EngineUIAdapter._extract_response_text(page)
            reasoning = await EngineUIAdapter._extract_reasoning_text(page)
            print(f"   content_len={len(content)}")
            print(f"   content={content[:300]}")
            print(f"   reasoning_len={len(reasoning)}")
            print(f"   reasoning={reasoning[:300]}")

        elapsed = time.monotonic() - t0
        print(f"\n[DONE] Total time: {elapsed:.1f}s")

    except Exception as e:
        print(f"\n[FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
    finally:
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
            try: await monitor_task
            except asyncio.CancelledError: pass
        try:
            if 'browser' in dir() and browser:
                await browser.close()
        except: pass
        try:
            if 'pw' in dir() and pw:
                await pw.stop()
        except: pass
        try:
            await adapter.close()
        except: pass


if __name__ == "__main__":
    asyncio.run(main())
