import asyncio
from ai_orchestrator.adapters.zai_ui import ZAIUIAdapter
from ai_orchestrator.orchestrator.main import _load_auth_for

async def main():
    storage_state = _load_auth_for("z_ai_ui")
    adapter = ZAIUIAdapter(
        mock_mode=False,
        headless=True,
        stealth=True,
        timeout_ms=30_000,
        storage_state=storage_state,
    )
    
    try:
        page = await adapter._get_page()
        # Wait a bit for page load assets
        await asyncio.sleep(5)
        
        bodies = await page.evaluate("() => window.__engine_response_bodies__ || []")
        print(f"Captured {len(bodies)} response bodies:")
        for idx, entry in enumerate(bodies):
            url = entry.get("url")
            transport = entry.get("transport")
            text = entry.get("text") or ""
            snippet = text[:100].replace('\n', ' ')
            looks_like = adapter._looks_like_chat_response(entry)
            print(f"[{idx}] URL={url} transport={transport} looks_like={looks_like} len={len(text)} snippet={snippet}")
            
    finally:
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(main())
