"""Deep analysis of MiniMax SSE stream — capture every delta chunk."""
import httpx, json, sys, time

BASE = "http://127.0.0.1:8766"
PROMPT = "What is the capital of France? Answer in one word."

# Non-streaming first
print("=== MINIMAX non-streaming ===")
t0 = time.time()
r = httpx.post(f"{BASE}/v1/chat/completions",
    json={"model": "minimax_ui", "messages": [{"role": "user", "content": PROMPT}]},
    timeout=180)
lat = time.time() - t0
print(f"Status: {r.status_code}  Latency: {lat:.1f}s")
if r.status_code == 200:
    body = r.json()
    msg = body["choices"][0]["message"]
    print(f"Content: {msg['content']}")
    rc = msg.get("reasoning_content")
    if rc:
        print(f"Reasoning ({len(rc)} chars): {rc[:400]}")
    else:
        print("No reasoning_content in response")
else:
    print(f"Error: {r.text[:500]}")

# Deep streaming — capture EVERY SSE line and analyze
print(f"\n=== MINIMAX streaming — RAW SSE ANALYSIS ===")
all_chunks = []
try:
    with httpx.stream("POST", f"{BASE}/v1/chat/completions",
        json={"model": "minimax_ui", "messages": [{"role": "user", "content": PROMPT}], "stream": True},
        timeout=180
    ) as r:
        print(f"HTTP {r.status_code}")
        if r.status_code == 200:
            for line in r.iter_lines():
                if not line.startswith("data: "): continue
                data = line[6:]
                if data == "[DONE]":
                    print(f"[DONE] — total {len(all_chunks)} data chunks captured")
                    break
                try:
                    obj = json.loads(data)
                    all_chunks.append(obj)
                    delta = obj.get("choices", [{}])[0].get("delta", {})
                    finish = obj.get("choices", [{}])[0].get("finish_reason")
                    # Print unique key analysis per chunk
                    keys = set()
                    for k, v in (delta or {}).items():
                        if v:
                            keys.add(f"{k}({repr(str(v)[:30])})")
                    if finish:
                        keys.add(f"finish={finish}")
                    if keys:
                        print(f"  delta keys: {', '.join(keys)}")
                    else:
                        print(f"  (empty delta)")
                except json.JSONDecodeError:
                    print(f"  (parse error for: {data[:60]})")
except Exception as e:
    print(f"Stream error: {e}")

# Analyze all reasoning-related keys
print(f"\n=== ANALYSIS ===")
print(f"Total SSE chunks: {len(all_chunks)}")
reasoning_keys_found = set()
content_keys_found = set()
for chunk in all_chunks:
    choices = chunk.get("choices", [{}])
    for c in choices:
        delta = c.get("delta", {})
        for k in delta:
            if delta[k]:
                if k in ("reasoning_content", "reasoning", "thinking", "thought"):
                    reasoning_keys_found.add(k)
                elif k == "content":
                    content_keys_found.add(k)
                else:
                    reasoning_keys_found.add(k)

if reasoning_keys_found:
    print(f"Reasoning-like keys found: {reasoning_keys_found}")
else:
    print("No reasoning/thinking/thought keys found in any SSE chunk")
    print(f"Only content keys found: {content_keys_found}")
    print("MiniMax does not appear to expose separate reasoning content in its stream")
