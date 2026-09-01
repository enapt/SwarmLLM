#!/usr/bin/env python3
"""Remote-inference checks against the live node, using the real swarm.

Non-streaming requests carry the route headers (x-swarm-route / x-swarm-nodes /
Server-Timing), which is how we know WHICH machine answered. Streaming checks
look for exactly one terminal finish event and no duplicated reply (#414), and
a multi-byte (Chinese) reply must not be refused as truncated (#416).
"""
import json, os, sys, time, urllib.request

KEY = open(os.path.expanduser("~/.local/share/swarmllm/api_key")).read().strip()
BASE = "http://localhost:8800"


def chat(model, prompt, max_tokens=64, stream=False, timeout=900):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"},
    )
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:400]}", "wall": round(time.perf_counter() - t0, 2)}
    hdrs = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith(("x-swarm", "server-timing"))}
    if not stream:
        d = json.loads(resp.read())
        wall = time.perf_counter() - t0
        ch = d["choices"][0]
        u = d.get("usage", {})
        ct = u.get("completion_tokens")
        return {
            "wall": round(wall, 2),
            "tps": round(ct / wall, 2) if ct else None,
            "usage": u,
            "finish": ch.get("finish_reason"),
            "text": (ch["message"].get("content") or "")[:200].replace("\n", " "),
            "headers": hdrs,
        }
    # streaming
    text, finishes, usage, errs, chunks = [], [], None, [], 0
    t_first = None
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        p = line[5:].strip()
        if p == "[DONE]":
            break
        try:
            d = json.loads(p)
        except Exception:
            continue
        if "error" in d:
            errs.append(d["error"])
            continue
        if d.get("usage"):
            usage = d["usage"]
        for c in d.get("choices", []):
            delta = c.get("delta", {}).get("content")
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()
                text.append(delta)
                chunks += 1
            if c.get("finish_reason"):
                finishes.append(c["finish_reason"])
    wall = time.perf_counter() - t0
    return {
        "wall": round(wall, 2),
        "ttft": round(t_first - t0, 2) if t_first else None,
        "usage": usage,
        "finishes": finishes,
        "chunks": chunks,
        "errors": errs,
        "text": "".join(text)[:200].replace("\n", " "),
        "headers": hdrs,
    }


def show(label, r):
    print(f"[{label}] " + json.dumps(r, ensure_ascii=False), flush=True)


PROMPT = "Explain the theory of relativity in simple terms."
plan = sys.argv[1:] or ["tinyllama-1.1b-chat-v1.0.q4-k-m", "gemma-2-2b-it-q4-k-m", "llama-3.2-1b-instruct-q8-0"]
for m in plan:
    for i in range(3):
        show(f"{m} non-stream #{i+1}", chat(m, PROMPT, 64))
    show(f"{m} STREAM", chat(m, "Count from 1 to 10, digits separated by spaces, nothing else.", 40, stream=True))
    show(f"{m} CHINESE non-stream", chat(m, "Answer in one sentence in Chinese: what is the ocean?", 60))
    show(f"{m} CHINESE stream", chat(m, "Answer in one sentence in Chinese: what is the ocean?", 60, stream=True))
