"""Comprehensive provider test — real mode, direct adapter + HTTP streaming."""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))


PROMPT = "Say exactly: 'The answer is 42.' Do not add anything else."


async def test_adapter_direct(provider_name: str) -> dict:
    """Test non-streaming via direct adapter call."""
    from ai_orchestrator.orchestrator.main import _build_adapter

    result = {"provider": provider_name, "ns_success": False, "ns_error": None}
    adapter = None
    try:
        adapter = _build_adapter(provider_name, mock_mode=False)
        t0 = time.monotonic()
        resp = await adapter.send(PROMPT, [{"role": "user", "content": PROMPT}])
        latency = (time.monotonic() - t0) * 1000

        result["ns_success"] = resp.success
        result["ns_latency_ms"] = round(latency, 1)
        result["ns_model"] = resp.model
        result["ns_content"] = (resp.content or "")[:500]
        result["ns_content_len"] = len(resp.content or "")
        result["ns_reasoning"] = (resp.reasoning_content or "")[:500] if resp.reasoning_content else None
        result["ns_reasoning_len"] = len(resp.reasoning_content or "")
        result["ns_has_reasoning"] = bool(resp.reasoning_content and resp.reasoning_content.strip())
        if resp.error:
            result["ns_error"] = resp.error[:300]
    except Exception as exc:
        result["ns_error"] = f"{type(exc).__name__}: {exc}"
        result["ns_traceback"] = traceback.format_exc()[-1500:]
    finally:
        if adapter:
            with __import__("contextlib").suppress(Exception):
                await adapter.close()
    return result


async def test_http_streaming(provider_name: str, base_url: str) -> dict:
    """Test streaming via HTTP endpoint."""
    result = {}
    async with httpx.AsyncClient(timeout=180.0) as client:
        try:
            async with client.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json={
                    "model": provider_name,
                    "messages": [{"role": "user", "content": PROMPT}],
                    "stream": True,
                },
            ) as resp:
                result["http_status"] = resp.status_code
                if resp.status_code != 200:
                    body = await resp.aread()
                    result["error"] = body.decode()[:500]
                    return result

                chunks = []
                reasoning_chunks = []
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        obj = json.loads(data)
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            reasoning_chunks.append(delta["reasoning_content"])
                        if "content" in delta and delta["content"]:
                            chunks.append(delta["content"])
                    except json.JSONDecodeError:
                        continue

                result["stream_content"] = "".join(chunks)[:500]
                result["stream_content_len"] = sum(len(c) for c in chunks)
                result["stream_chunk_count"] = len(chunks)
                result["stream_reasoning"] = "".join(reasoning_chunks)[:500]
                result["stream_reasoning_len"] = sum(len(r) for r in reasoning_chunks)
                result["stream_reasoning_chunk_count"] = len(reasoning_chunks)
                result["stream_has_reasoning"] = bool(reasoning_chunks)
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def start_server() -> subprocess.Popen | None:
    """Start the uvicorn server."""
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "ai_orchestrator.orchestrator.main:app",
             "--host", "127.0.0.1", "--port", "8765"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        time.sleep(3)
        return proc
    except Exception as exc:
        print(f"Failed to start server: {exc}")
        return None


async def main():
    providers = [
        "chatgpt_ui",
        "deepseek_ui",
        "kimi_ui",
        "qwen_ui",
        "z_ai_ui",
        "xiaomimimo_ui",
        "minimax_ui",
    ]

    print("=" * 80)
    print("PHASE 1: DIRECT ADAPTER TESTS (non-streaming)")
    print("=" * 80)

    results = {}
    for p in providers:
        print(f"\n── {p} ──")
        sys.stdout.flush()
        r = await test_adapter_direct(p)
        results[p] = r

        ns = r
        print(f"  success:       {ns.get('ns_success')}")
        print(f"  latency_ms:    {ns.get('ns_latency_ms')}")
        print(f"  model:         {ns.get('ns_model')}")
        print(f"  content_len:   {ns.get('ns_content_len')}")
        print(f"  content_preview: {ns.get('ns_content', '')[:120]}")
        print(f"  reasoning_len: {ns.get('ns_reasoning_len')}")
        print(f"  has_reasoning: {ns.get('ns_has_reasoning')}")
        if ns.get('ns_reasoning'):
            print(f"  reasoning_preview: {ns['ns_reasoning'][:120]}")
        if ns.get('ns_error'):
            print(f"  error:         {ns['ns_error'][:200]}")

    # Start server for streaming tests
    print(f"\n\n{'=' * 80}")
    print("PHASE 2: HTTP STREAMING TESTS")
    print("=" * 80)

    server = start_server()
    base_url = "http://127.0.0.1:8765"

    if server is None:
        print("Could not start server, skipping streaming tests")
    else:
        try:
            for p in providers:
                print(f"\n── {p} (HTTP streaming) ──")
                sys.stdout.flush()
                sr = await test_http_streaming(p, base_url)
                results[p].update({f"http_{k}": v for k, v in sr.items()})

                print(f"  http_status:              {sr.get('http_status')}")
                print(f"  stream_chunk_count:       {sr.get('stream_chunk_count')}")
                print(f"  stream_content_len:       {sr.get('stream_content_len')}")
                print(f"  stream_content_preview:   {sr.get('stream_content', '')[:120]}")
                print(f"  stream_reasoning_len:     {sr.get('stream_reasoning_len')}")
                print(f"  stream_has_reasoning:     {sr.get('stream_has_reasoning')}")
                if sr.get('stream_reasoning'):
                    print(f"  stream_reasoning_preview: {sr['stream_reasoning'][:120]}")
                if sr.get('error'):
                    print(f"  error:                    {sr['error'][:200]}")
        finally:
            server.terminate()
            server.wait(timeout=5)

    # ── Summary ──
    print(f"\n\n{'=' * 80}")
    print("FINAL SUMMARY")
    print("=" * 80)
    header = f"{'Provider':<18} {'NS OK':<6} {'C Len':<7} {'R Len':<7} {'HasR':<5} {'Lat':<8} {'Strm OK':<8} {'Strm R':<5}"
    print(header)
    print("-" * len(header))

    for p in providers:
        r = results.get(p, {})
        ns_ok = "YES" if r.get("ns_success") else "NO"
        c_len = r.get("ns_content_len", 0)
        r_len = r.get("ns_reasoning_len", 0)
        has_r = "YES" if r.get("ns_has_reasoning") else "NO"
        lat = f"{r.get('ns_latency_ms', 0):.0f}ms"
        strm_ok = "YES" if r.get("http_http_status") == 200 else "NO"
        strm_r = "YES" if r.get("http_stream_has_reasoning") else "NO"
        print(f"{p:<18} {ns_ok:<6} {c_len:<7} {r_len:<7} {has_r:<5} {lat:<8} {strm_ok:<8} {strm_r:<5}")

    print("-" * len(header))

    # Write full results
    with open("diagnostic_output/full_provider_test.json", "w") as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)

    print("\nFull results written to diagnostic_output/full_provider_test.json")


if __name__ == "__main__":
    asyncio.run(main())
