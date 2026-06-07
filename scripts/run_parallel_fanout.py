import asyncio
import json
from pathlib import Path
from ai_orchestrator.orchestrator.main import _PROVIDER_CLASS_MAP, _load_auth_for
from ai_orchestrator.adapters.engine_adapter import EngineUIAdapter

async def main():
    prompt = "What is 2+2? Reply in ONE word only."
    adapters = []
    
    # Exclude local_llm as requested by user
    providers = [p for p in _PROVIDER_CLASS_MAP.keys() if p != "local_llm"]
    
    print(f"Initializing {len(providers)} providers in REAL mode...")
    for p in providers:
        cls = _PROVIDER_CLASS_MAP[p]
        storage_state = _load_auth_for(p)
        
        adapter = cls(
            mock_mode=False,
            headless=True,
            stealth=True,
            timeout_ms=90_000,
            storage_state=storage_state,
        )
        adapters.append(adapter)
        
    print("\nSending prompt parallel via fan_out (timeout 120s)...")
    results = await EngineUIAdapter.fan_out(
        adapters=adapters,
        prompt=prompt,
        return_when="ALL_COMPLETED",
        timeout=120.0,
    )
    
    print("\n--- RESULTS ---")
    output = []
    for r in results:
        res = {
            "provider": r.model,
            "success": r.success,
            "content": r.content,
            "error": r.error,
            "latency_ms": r.latency_ms,
        }
        output.append(res)
        print(json.dumps(res, indent=2))
        
if __name__ == "__main__":
    asyncio.run(main())
