import asyncio
import sys
from pathlib import Path

# Add root folder to sys.path so we can import ai_orchestrator
sys.path.append(str(Path(__file__).parent.parent))

from ai_orchestrator.adapters.minimax_ui import MiniMaxUIAdapter

async def main():
    prompt = "What is 2+2? Reply in ONE word only."
    print("Initializing MiniMax UI adapter in REAL (non-mock) headful mode...")
    
    # Run with headless=False so we can see Google Chrome open and do its job
    adapter = MiniMaxUIAdapter(
        mock_mode=False,
        headless=False,
        stealth=True,
        timeout_ms=120_000,
    )
    
    try:
        print(f"Sending prompt: {prompt!r}")
        res = await adapter.send(prompt)
        print("\n--- RESULTS ---")
        print(f"Success: {res.success}")
        print(f"Content: {res.content!r}")
        print(f"Error: {res.error}")
        print(f"Latency: {res.latency_ms:.2f} ms")
    finally:
        print("Closing adapter...")
        await adapter.close()

if __name__ == "__main__":
    asyncio.run(main())
