import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(message)s")

async def test_provider(provider_name, AdapterClass):
    from ai_orchestrator.orchestrator.main import _load_auth_for
    print("=" * 80)
    print(f"Testing provider: {provider_name}")
    print("=" * 80)

    storage_state = _load_auth_for(provider_name)
    print(f"\n[AUTH] storage_state loaded for {provider_name}: {bool(storage_state)}")
    if not storage_state:
        print(f"Skipping {provider_name} due to missing auth state.")
        return

    adapter = AdapterClass(
        mock_mode=False,
        headless=False,  # Set to False so we can see what's happening
        stealth=True,
        timeout_ms=90_000,
        storage_state=None,  # Force using the live Chrome profile
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
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await adapter.close()

async def main():
    from ai_orchestrator.adapters.chatgpt_ui import ChatGPTUIAdapter
    from ai_orchestrator.adapters.qwen_ui import QwenUIAdapter
    
    await test_provider("chatgpt_ui", ChatGPTUIAdapter)
    await test_provider("qwen_ui", QwenUIAdapter)

if __name__ == "__main__":
    asyncio.run(main())
