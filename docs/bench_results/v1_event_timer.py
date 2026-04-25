"""Precise V1 timing: time from request start to the first `data: ` line
in the SSE body. This is what V1 actually targets. Curl's TTFB (head
arrival) and the v1 bench's `(curl|grep -m1 "data:") time -p` both
measure proxies that miss the actual signal:
  - TTFB clocks the HTTP response head, which axum sends before
    consuming the body stream — so even pre-V1, head goes out fast
    even when response.created is gated on chat_completions.
  - The `time -p` of `(curl|grep -m1)` measures wall clock until the
    subshell exits. Curl doesn't notice SIGPIPE from grep until its
    next attempted write, which only happens when the next SSE chunk
    arrives — i.e., wall clock is bounded by inference completion
    time, not by when grep matched.
"""

import argparse
import http.client
import json
import os
import statistics
import sys
import time

def time_first_event(host: str, port: int, path: str, body: dict, api_key: str):
    body_bytes = json.dumps(body).encode("utf-8")
    conn = http.client.HTTPConnection(host, port, timeout=60)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream",
    }
    t0 = time.perf_counter()
    conn.request("POST", path, body=body_bytes, headers=headers)
    resp = conn.getresponse()
    if resp.status != 200:
        body = resp.read()
        raise RuntimeError(f"non-200 {resp.status}: {body[:200]!r}")
    # Read until we see the first complete SSE data line.
    buf = b""
    deadline = t0 + 30.0
    while time.perf_counter() < deadline:
        chunk = resp.read(1)
        if not chunk:
            break
        buf += chunk
        # Find first complete "data: ..." line.
        while b"\n" in buf:
            line, _, rest = buf.partition(b"\n")
            if line.startswith(b"data:"):
                t1 = time.perf_counter()
                conn.close()
                return (t1 - t0) * 1000.0
            buf = rest
    conn.close()
    raise RuntimeError("no data: line within 30s")


def median_first_event_ms(host, port, path, body, api_key, iters=10):
    samples = []
    for _ in range(iters):
        for attempt in range(5):
            try:
                ms = time_first_event(host, port, path, body, api_key)
                samples.append(ms)
                break
            except RuntimeError as e:
                if "429" in str(e) and attempt < 4:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
        # Small gap between requests to avoid the daemon's per-IP rate
        # limiter rejecting us with 429 mid-run.
        time.sleep(1.5)
    return statistics.median(samples), statistics.mean(samples), max(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8830)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--label", default="unknown")
    ap.add_argument(
        "--model", default="tinyllama-1.1b-chat-v1.0.q4-k-m"
    )
    args = ap.parse_args()

    api_key = open("/tmp/resp_final/api_key").read().strip()
    chat_body = {
        "model": args.model,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "stream": True,
        "temperature": 0.1,
    }
    resp_body = {
        "model": args.model,
        "input": "hi",
        "max_output_tokens": 5,
        "stream": True,
        "store": False,
        "temperature": 0.1,
    }

    # Warm up.
    time_first_event("localhost", args.port, "/v1/chat/completions", chat_body, api_key)
    time_first_event("localhost", args.port, "/v1/responses", resp_body, api_key)

    chat_med, chat_mean, chat_max = median_first_event_ms(
        "localhost", args.port, "/v1/chat/completions", chat_body, api_key, args.iters
    )
    resp_med, resp_mean, resp_max = median_first_event_ms(
        "localhost", args.port, "/v1/responses", resp_body, api_key, args.iters
    )
    gap = abs(resp_med - chat_med)
    print(
        f"[{args.label}] first data: line — "
        f"chat median={chat_med:.1f}ms (mean {chat_mean:.1f}, max {chat_max:.1f})  "
        f"resp median={resp_med:.1f}ms (mean {resp_mean:.1f}, max {resp_max:.1f})  "
        f"gap={gap:.1f}ms"
    )


if __name__ == "__main__":
    main()
