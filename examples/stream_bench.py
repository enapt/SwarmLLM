#!/usr/bin/env python3
"""Streaming chat bench against a running SwarmLLM node.

Reports per rep: TTFT, whole-request tok/s (completion_tokens / wall, the
figure the .145 round quoted), decode tok/s ((n-1)/(t_last - t_first), a
client-side window — see gotcha #312), completion tokens, route headers,
and the card's memory before/after.

usage: bench.py MODEL [--reps N] [--max-tokens N] [--prompt TEXT] [--port P] [--label L]
"""
import argparse, json, os, subprocess, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--prompt", default="Explain the theory of relativity in simple terms.")
ap.add_argument("--port", type=int, default=8800)
ap.add_argument("--label", default="")
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--temperature", type=float, default=0.0)
args = ap.parse_args()

key_path = os.environ.get("SWARMLLM_API_KEY_FILE", os.path.expanduser("~/.local/share/swarmllm/api_key"))
KEY = open(key_path).read().strip()
URL = f"http://localhost:{args.port}/v1/chat/completions"


def gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return -1


def one(rep):
    body = json.dumps(
        {
            "model": args.model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    )
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    n_chunks = 0
    text = []
    usage = None
    finish = None
    err = None
    try:
        resp = urllib.request.urlopen(req, timeout=1800)
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"}
    hdrs = {k.lower(): v for k, v in resp.headers.items() if k.lower().startswith(("x-swarm", "server-timing"))}
    for raw in resp:
        line = raw.decode(errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except Exception:
            continue
        if "error" in d:
            err = d["error"]
            break
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            c = delta.get("content")
            if c:
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n_chunks += 1
                text.append(c)
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    t_end = time.perf_counter()
    wall = t_end - t0
    ct = usage.get("completion_tokens") if usage else None
    pt = usage.get("prompt_tokens") if usage else None
    ttft = (t_first - t0) if t_first else None
    decode = None
    if ct and ct > 1 and t_first and t_last and t_last > t_first:
        decode = (ct - 1) / (t_last - t_first)
    whole = (ct / wall) if ct else None
    return {
        "rep": rep,
        "wall_s": round(wall, 2),
        "ttft_s": round(ttft, 2) if ttft else None,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "chunks": n_chunks,
        "whole_tps": round(whole, 2) if whole else None,
        "decode_tps": round(decode, 2) if decode else None,
        "finish": finish,
        "error": err,
        "headers": hdrs,
        "text": "".join(text)[:160].replace("\n", " "),
    }


mem0 = gpu_mem()
print(f"== {args.label or args.model} | model={args.model} max_tokens={args.max_tokens} reps={args.reps} gpu_mem_before={mem0} MiB", flush=True)
results = []
for i in range(-args.warmup, args.reps):
    r = one(i)
    tag = "warmup" if i < 0 else f"rep {i+1}"
    if "error" in r and r["error"] and "rep" not in r:
        print(f"  {tag}: {r['error']}", flush=True)
        continue
    print(
        f"  {tag}: wall {r['wall_s']}s ttft {r['ttft_s']}s | {r['completion_tokens']} tok | whole {r['whole_tps']} tok/s | decode {r['decode_tps']} tok/s | finish={r['finish']} err={r['error']} | {r['headers']}",
        flush=True,
    )
    if i == -1 or i == 0:
        print(f"    text: {r['text']}", flush=True)
    if i >= 0:
        results.append(r)
mem1 = gpu_mem()
ok = [r for r in results if r.get("decode_tps")]
if ok:
    best = max(r["decode_tps"] for r in ok)
    bestw = max(r["whole_tps"] for r in ok if r["whole_tps"])
    med = sorted(r["decode_tps"] for r in ok)[len(ok) // 2]
    print(
        f"== SUMMARY {args.label or args.model}: decode best {best} median {med} tok/s | whole best {bestw} tok/s | ttft min {min(r['ttft_s'] for r in ok if r['ttft_s'])}s | gpu_mem after {mem1} MiB (before {mem0})",
        flush=True,
    )
else:
    print(f"== SUMMARY {args.label or args.model}: NO SUCCESSFUL REPS | gpu_mem after {mem1} MiB", flush=True)
