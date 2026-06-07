"""Parallel provider test — all 7 at once, 2-min hard timeout, screenshots."""
from __future__ import annotations

import asyncio
import json
import sys
import time
import os
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCREENSHOT_DIR = Path(__file__).resolve().parent.parent / "diagnostic_screenshots"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

PROVIDERS = ["chatgpt_ui", "deepseek_ui", "kimi_ui", "qwen_ui", "z_ai_ui", "xiaomimimo_ui", "minimax_ui"]


async def run_server():
    """Start uvicorn, wait up to 30s."""
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "uvicorn",
        "ai_orchestrator.orchestrator.main:app",
        "--host", "127.0.0.1", "--port", "8766",
        cwd=str(Path(__file__).resolve().parent.parent),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={"PYTHONPATH": str(Path(__file__).resolve().parent.parent)},
    )
    for _ in range(30):
        try:
            r = httpx.get("http://127.0.0.1:8766/v1/models", timeout=2)
            if r.status_code == 200:
                print("Server UP")
                return proc
        except Exception:
            await asyncio.sleep(1)
    print("Server FAILED to start")
    return None


async def test_one(provider: str) -> dict:
    """Hit one provider NS + streaming, return structured result."""
    BASE = "http://127.0.0.1:8766"
    PROMPT = "Say exactly: 'The answer is 42.' Do not add anything else."
    result = {"provider": provider}

    # Non-streaming
    t0 = time.monotonic()
    try:
        r = httpx.post(f"{BASE}/v1/chat/completions",
            json={"model": provider, "messages": [{"role": "user", "content": PROMPT}]},
            timeout=115)
        lat = round(time.monotonic() - t0, 1)
        if r.status_code == 200:
            body = r.json()
            msg = body.get("choices", [{}])[0].get("message", {})
            result["ns"] = {
                "ok": True, "latency_s": lat, "status": 200,
                "content": msg.get("content", ""),
                "reasoning": msg.get("reasoning_content"),
            }
        else:
            result["ns"] = {"ok": False, "latency_s": lat, "status": r.status_code, "detail": r.text[:500]}
    except Exception as e:
        result["ns"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # Streaming
    t0 = time.monotonic()
    try:
        chunks, reasoning = [], []
        async with httpx.AsyncClient(timeout=115) as ac:
            async with ac.stream("POST",
                f"{BASE}/v1/chat/completions",
                json={"model": provider, "messages": [{"role": "user", "content": PROMPT}], "stream": True}
            ) as r:
                lat = round(time.monotonic() - t0, 1) if r.status_code != 200 else None
                if r.status_code != 200:
                    result["st"] = {"ok": False, "status": r.status_code, "detail": (await r.aread())[:500]}
                else:
                    async for line in r.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            obj = json.loads(data)
                            d = obj.get("choices", [{}])[0].get("delta", {})
                            if d.get("reasoning_content"):
                                reasoning.append(d["reasoning_content"])
                            if d.get("content"):
                                chunks.append(d["content"])
                        except json.JSONDecodeError:
                            continue
                    lat = round(time.monotonic() - t0, 1)
                    result["st"] = {
                        "ok": True, "latency_s": lat, "chunks": len(chunks),
                        "reasoning_chunks": len(reasoning),
                        "content": "".join(chunks)[:300] if chunks else None,
                        "reasoning": "".join(reasoning)[:300] if reasoning else None,
                    }
    except Exception as e:
        result["st"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return result


async def main():
    proc = await run_server()
    if proc is None:
        return

    print(f"\nFiring {len(PROVIDERS)} providers IN PARALLEL with 120s hard timeout\n")
    sys.stdout.flush()

    # FAN OUT — all at once
    tasks = {p: asyncio.create_task(test_one(p)) for p in PROVIDERS}
    completed, pending = await asyncio.wait(tasks.values(), timeout=120)

    # Cancel stragglers
    for t in pending:
        t.cancel()

    results = {}
    for p, t in tasks.items():
        try:
            results[p] = t.result()
        except asyncio.CancelledError:
            results[p] = {"provider": p, "ns": {"ok": False, "error": "TIMEOUT"}, "st": {"ok": False, "error": "TIMEOUT"}}

    # Print results
    print("\n" + "=" * 90)
    print(f"{'Provider':<18} {'NS':<6} {'NS Lat':<8} {'NS C':<6} {'NS R':<6} {'ST':<6} {'ST Ch':<6} {'ST Lat':<8} {'Error/Content':<30}")
    print("=" * 90)
    for p in PROVIDERS:
        r = results.get(p, {})
        ns = r.get("ns", {})
        st = r.get("st", {})
        ns_ok = "OK" if ns.get("ok") else "NO"
        ns_lat = f"{ns.get('latency_s','-')}s"
        ns_c = f"{len(ns.get('content',''))}" if ns.get("content") else "0"
        ns_r = f"{len(ns.get('reasoning',''))}" if ns.get("reasoning") else "0"
        st_ok = "OK" if st.get("ok") else "NO"
        st_ch = str(st.get("chunks", 0))
        st_lat = f"{st.get('latency_s','-')}s" if st.get("latency_s") else "-"
        detail = ns.get("content", ns.get("detail", ns.get("error", st.get("detail", st.get("error", "")))))[:30]
        print(f"{p:<18} {ns_ok:<6} {ns_lat:<8} {ns_c:<6} {ns_r:<6} {st_ok:<6} {st_ch:<6} {st_lat:<8} {detail:<30}")

    # Summary line
    ok_count = sum(1 for p in PROVIDERS if results.get(p, {}).get("ns", {}).get("ok") or results.get(p, {}).get("st", {}).get("ok"))
    print(f"\n{ok_count}/{len(PROVIDERS)} providers returned a result (NS or ST)")

    # Write full results
    out_path = "diagnostic_output/all_providers_parallel_test.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f"Results: {out_path}")

    proc.terminate()
    await proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
