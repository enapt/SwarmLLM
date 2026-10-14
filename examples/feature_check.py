#!/usr/bin/env python3
"""Every surface a user reaches, asserted against a RUNNING node.

The release gate (`release_gate.sh`) runs a throwaway node and checks the
chat surfaces it needs for a regression verdict. This checks what a person
using the live node actually reaches — every API surface (OpenAI chat with
tools, Anthropic messages, the Responses API, embeddings, MCP), the dashboard
pages and their scripts, the dashboard WebSocket, the read-only admin
endpoints, the error contract, and a split across the real swarm — with an
assertion on each, not a printout.

    examples/feature_check.py                       # http://localhost:8800
    examples/feature_check.py --base http://host:8800 --only openai,swarm

Requests run ONE AT A TIME: concurrent chats to a node are a GPU stress test
on this machine (CLAUDE.md). The swarm section loads the split model on the
peers it routes through (testing_techniques.md § "LIVE swarm's routes").
"""
import argparse, asyncio, json, os, re, sys, time, urllib.error, urllib.request

KEY = open(os.path.expanduser("~/.local/share/swarmllm/api_key")).read().strip()
RESULTS = []


def record(section, name, ok, detail=""):
    RESULTS.append((section, name, bool(ok), detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  {mark}  {name}" + (f" — {detail}" if detail else ""), flush=True)


def http(base, method, path, body=None, auth=True, headers=None, timeout=600, raw=False):
    h = {"Content-Type": "application/json"}
    if auth:
        h["Authorization"] = f"Bearer {KEY}"
    h.update(headers or {})
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, headers=h, method=method)
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        status, hdrs, payload = resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as e:
        status, hdrs, payload = e.code, {k.lower(): v for k, v in e.headers.items()}, e.read()
    wall = time.perf_counter() - t0
    if raw:
        return status, hdrs, payload, wall
    try:
        parsed = json.loads(payload) if payload else None
    except ValueError:
        parsed = payload.decode(errors="replace")
    return status, hdrs, parsed, wall


def sse(base, path, body, timeout=600):
    """POST and read an SSE stream; returns (status, [(event, data)])."""
    h = {"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers=h)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, [("error", e.read().decode(errors="replace"))]
    events, event = [], None
    for line in resp:
        line = line.decode("utf-8", "replace").rstrip("\r\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
            if data == "[DONE]":
                events.append((event, "[DONE]"))
            else:
                try:
                    events.append((event, json.loads(data)))
                except ValueError:
                    events.append((event, data))
            event = None
    return resp.status, events


def first_choice(d):
    return ((d or {}).get("choices") or [{}])[0] if isinstance(d, dict) else {}


# ---------------------------------------------------------------- sections


def check_core(base, a):
    print("core")
    s, _, d, _ = http(base, "GET", "/health", auth=False)
    record("core", "/health answers 200", s == 200, str(s))
    s, _, d, _ = http(base, "GET", "/health/ready", auth=False)
    down = [k for k, v in (d or {}).get("subsystems", {}).items() if not v]
    record("core", "/health/ready: every subsystem up", s == 200 and d.get("ready") and not down, f"down={down}")
    s, _, d, _ = http(base, "GET", "/v1/status")
    record("core", "/v1/status names the version", s == 200 and bool((d or {}).get("version")), str((d or {}).get("version")))
    s, _, d, _ = http(base, "GET", "/metrics", auth=False)
    record("core", "/metrics exposes peers_connected", s == 200 and "swarmllm_peers_connected" in str(d))
    s, _, d, _ = http(base, "GET", "/v1/models")
    ids = [m["id"] for m in (d or {}).get("data", [])]
    record("core", "/v1/models lists models with a context length",
           s == 200 and ids and all("context_length" in m for m in d["data"]), f"{len(ids)} models")
    for m in (a.model, a.tool_model, a.split_model):
        record("core", f"/v1/models offers {m}", m in ids)
    s, _, d, _ = http(base, "GET", "/v1/providers")
    record("core", "/v1/providers lists providers", s == 200 and isinstance((d or {}).get("providers"), list))


ADMIN_GETS = [
    "/api/admin/stats", "/api/admin/diagnostics", "/api/admin/peers", "/api/admin/models",
    "/api/admin/network-map", "/api/admin/performance", "/api/admin/swarm/capacity",
    "/api/admin/shard-storage", "/api/admin/storage/breakdown", "/api/admin/downloads",
    "/api/admin/wishlist", "/api/admin/credits", "/api/admin/credits/transactions",
    "/api/admin/version", "/api/admin/provider-health", "/api/admin/providers",
    "/api/admin/provider-models", "/api/admin/prune-history", "/api/admin/reference-models",
    "/api/admin/responses", "/api/admin/config", "/api/admin/schedule", "/api/admin/adapters",
    "/api/admin/quant-recommendations", "/api/admin/network-code", "/api/admin/foreign-pool-catalog",
    "/api/admin/claude-subscription/status", "/api/admin/hf/trending", "/api/admin/hf/search?q=qwen",
    "/api/identity/leaderboard", "/api/identity/peers", "/api/pool/state", "/api/pool/coverage",
    "/api/pool/leaderboard", "/api/pool/invitations", "/api/pool/pins",
]
# Answered as plain text, not JSON.
TEXT_GETS = ["/api/admin/diagnostics"]


def check_admin(base, a):
    print("admin (read-only)")
    for path in ADMIN_GETS:
        s, _, d, wall = http(base, "GET", path, timeout=60)
        ok = s == 200 and d is not None and not isinstance(d, str)
        record("admin", f"GET {path}", ok, f"{s} {wall:.1f}s" + ("" if ok else f" {str(d)[:160]}"))
    for path in TEXT_GETS:
        s, _, d, wall = http(base, "GET", path, timeout=60)
        record("admin", f"GET {path}", s == 200 and isinstance(d, str) and d.strip(), f"{s} {wall:.1f}s")
    s, _, d, _ = http(base, "GET", f"/api/admin/models/{a.split_model}/pipeline-plan", timeout=60)
    record("admin", "pipeline-plan preview answers", s == 200, f"{s} {str(d)[:160] if s != 200 else ''}")
    s, _, d, _ = http(base, "GET", "/api/admin/stats", auth=False)
    record("admin", "an admin endpoint refuses a request with no key", s in (401, 403), str(s))


def check_dashboard(base, a):
    print("dashboard")
    scripts = set()
    for page in ("/", "/chat", "/setup", "/admin"):
        s, h, body, _ = http(base, "GET", page, auth=False, raw=True)
        text = body.decode(errors="replace")
        record("dashboard", f"GET {page} serves HTML", s == 200 and "<html" in text.lower(), str(s))
        scripts.update(re.findall(r'(?:src|href)="(/static/[^"?#]+)', text))
    missing = []
    for path in sorted(scripts):
        s, _, _, _ = http(base, "GET", path, auth=False, raw=True)
        if s != 200:
            missing.append(f"{path}={s}")
    record("dashboard", f"every /static asset the pages name loads ({len(scripts)})", not missing, ", ".join(missing[:5]))
    # ES modules import each other; follow the imports breadth-first.
    seen, queue, bad = set(scripts), [p for p in scripts if p.endswith(".js")], []
    while queue:
        path = queue.pop()
        s, _, body, _ = http(base, "GET", path, auth=False, raw=True)
        if s != 200:
            bad.append(f"{path}={s}")
            continue
        for rel in re.findall(r'''(?:import|from)\s*\(?\s*['"](\.{1,2}/[^'"]+)['"]''', body.decode(errors="replace")):
            target = os.path.normpath(os.path.join(os.path.dirname(path), rel))
            if target not in seen:
                seen.add(target)
                queue.append(target)
    record("dashboard", f"every module reachable by import loads ({len(seen)})", not bad, ", ".join(bad[:5]))

    async def ws():
        import websockets
        _, _, t, _ = http(base, "POST", "/api/admin/ws-ticket", body={})
        ticket = (t or {}).get("ticket") if isinstance(t, dict) else None
        if not ticket:
            return False, f"no ticket: {t}"
        url = base.replace("http", "ws", 1) + f"/api/admin/ws?t={ticket}"
        async with websockets.connect(url, origin=base, open_timeout=10) as sock:
            msg = await asyncio.wait_for(sock.recv(), timeout=20)
            return True, f"first message type={json.loads(msg).get('type')}"
    try:
        ok, detail = asyncio.run(ws())
    except Exception as e:  # noqa: BLE001 — any failure is the finding
        ok, detail = False, repr(e)[:200]
    record("dashboard", "dashboard WebSocket connects and delivers a message", ok, detail)


def openai_body(model, prompt, **kw):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 48, "temperature": 0}
    b.update(kw)
    return b


WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}


def check_openai(base, a):
    print(f"openai chat ({a.model}; tools on {a.tool_model})")
    s, h, d, wall = http(base, "POST", "/v1/chat/completions", openai_body(a.model, "Name three primary colours."))
    ch = first_choice(d) if s == 200 else {}
    record("openai", "non-streamed reply has content, finish_reason and usage",
           s == 200 and ch.get("message", {}).get("content") and ch.get("finish_reason") and d.get("usage", {}).get("completion_tokens"),
           f"{s} {wall:.1f}s route={h.get('x-swarm-route')}")
    s, _, d, _ = http(base, "POST", "/v1/chat/completions",
                      openai_body(a.model, "Write a long story about a lighthouse.", max_tokens=8))
    fr = first_choice(d).get("finish_reason") if s == 200 else None
    record("openai", "max_tokens cut gives finish_reason=length", fr == "length", str(fr))
    s, _, d, _ = http(base, "POST", "/v1/chat/completions",
                      openai_body(a.model, "Count from 1 to 20, separated by spaces.", stop=["7"], max_tokens=60))
    txt = first_choice(d).get("message", {}).get("content", "") if s == 200 else ""
    record("openai", "a stop sequence ends the reply before it", s == 200 and "7" not in txt and "6" in txt, repr(txt[:60]))
    s, ev = sse(base, "/v1/chat/completions", openai_body(a.model, "Count from 1 to 10.", stream=True,
                                                          stream_options={"include_usage": True}))
    datas = [d for _, d in ev]
    finishes = [c for d in datas if isinstance(d, dict) for c in d.get("choices", []) if c.get("finish_reason")]
    text = "".join(c.get("delta", {}).get("content") or "" for d in datas if isinstance(d, dict) for c in d.get("choices", []))
    usage = [d["usage"] for d in datas if isinstance(d, dict) and d.get("usage")]
    record("openai", "stream: one finish event, [DONE], usage chunk, text",
           s == 200 and len(finishes) == 1 and datas and datas[-1] == "[DONE]" and usage and text.strip(),
           f"finishes={len(finishes)} usage={bool(usage)} text={text[:40]!r}")
    s, ev = sse(base, "/v1/chat/completions",
                openai_body(a.model, "Answer in one sentence in Chinese: what is the ocean?", stream=True, max_tokens=60))
    text = "".join(c.get("delta", {}).get("content") or "" for _, d in ev if isinstance(d, dict) for c in d.get("choices", []))
    record("openai", "stream: a Chinese reply arrives whole (no U+FFFD)", s == 200 and text and "�" not in text, repr(text[:30]))
    s, _, d, _ = http(base, "POST", "/v1/chat/completions",
                      openai_body(a.model, "Reply with a JSON object with keys name and age for a fictional person.",
                                  response_format={"type": "json_object"}, max_tokens=80))
    txt = first_choice(d).get("message", {}).get("content", "") if s == 200 else str(d)[:200]
    try:
        json.loads(txt)
        ok = True
    except ValueError:
        ok = False
    record("openai", "response_format json_object returns parseable JSON", ok, repr(txt[:80]))
    # Local models return no logprobs, by decision (FUTURE_WORK § "Decided — not doing"):
    # the contract is a 400 that says so, never a reply silently missing them.
    s, _, d, _ = http(base, "POST", "/v1/chat/completions",
                      openai_body(a.model, "Say hi.", logprobs=True, top_logprobs=2, max_tokens=4))
    err = (d or {}).get("error", {}) if isinstance(d, dict) else {}
    record("openai", "logprobs on a local model: a 400 that says only cloud models return them",
           s == 400 and err.get("type") == "invalid_request_error" and "logprobs" in err.get("message", ""), f"{s}")

    body = openai_body(a.tool_model, "What is the weather in Paris right now?", tools=[WEATHER_TOOL], max_tokens=96)
    s, _, d, _ = http(base, "POST", "/v1/chat/completions", body)
    ch = first_choice(d) if s == 200 else {}
    calls = ch.get("message", {}).get("tool_calls") or []
    args_ok = False
    if calls:
        try:
            args_ok = "paris" in json.loads(calls[0]["function"]["arguments"]).get("city", "").lower()
        except (ValueError, KeyError, AttributeError):
            pass
    record("openai", "tools: a tool call with JSON arguments, finish_reason=tool_calls",
           calls and calls[0]["function"]["name"] == "get_weather" and args_ok and ch.get("finish_reason") == "tool_calls",
           f"{s} finish={ch.get('finish_reason')} calls={str(calls)[:120]}")
    if calls:
        follow = dict(body)
        follow["messages"] = body["messages"] + [ch["message"], {
            "role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "content": json.dumps({"temp_c": 18, "sky": "clear"})}]
        s, _, d, _ = http(base, "POST", "/v1/chat/completions", follow)
        txt = (first_choice(d).get("message", {}).get("content") or "") if s == 200 else ""
        record("openai", "tools: the tool result is used in the next reply", "18" in txt, repr(txt[:80]))
    s, ev = sse(base, "/v1/chat/completions", dict(body, stream=True))
    args, names = "", set()
    for _, d in ev:
        if not isinstance(d, dict):
            continue
        for c in d.get("choices", []):
            for tc in c.get("delta", {}).get("tool_calls") or []:
                fn = tc.get("function", {})
                if fn.get("name"):
                    names.add(fn["name"])
                args += fn.get("arguments") or ""
    try:
        sargs_ok = bool(json.loads(args))
    except ValueError:
        sargs_ok = False
    record("openai", "tools streamed: deltas assemble to the call and valid JSON", names == {"get_weather"} and sargs_ok,
           f"names={names} args={args[:80]!r}")


def check_anthropic(base, a):
    print(f"anthropic messages ({a.model}; tools on {a.tool_model})")
    body = {"model": a.model, "max_tokens": 48, "system": "Answer in exactly one word.",
            "messages": [{"role": "user", "content": "What colour is the sky on a clear day?"}], "temperature": 0}
    s, _, d, _ = http(base, "POST", "/v1/messages", body, headers={"anthropic-version": "2023-06-01"})
    text = "".join(b.get("text", "") for b in (d or {}).get("content", [])) if s == 200 else ""
    record("anthropic", "non-streamed message has text, stop_reason, usage",
           s == 200 and text and d.get("stop_reason") and d.get("usage", {}).get("output_tokens"), f"{s} {text[:40]!r}")
    s, ev = sse(base, "/v1/messages", dict(body, stream=True))
    kinds = [e for e, _ in ev]
    text = "".join(d.get("delta", {}).get("text", "") for e, d in ev if e == "content_block_delta" and isinstance(d, dict))
    want = ["message_start", "content_block_start", "content_block_delta", "content_block_stop", "message_delta", "message_stop"]
    order_ok = bool(kinds) and all(k in kinds for k in want) and kinds[0] == "message_start" and kinds[-1] == "message_stop"
    record("anthropic", "stream: the full event sequence, in order, with text", s == 200 and order_ok and text,
           f"{list(dict.fromkeys(kinds))}")
    tools = [{"name": "get_weather", "description": "Get the current weather for a city.",
              "input_schema": WEATHER_TOOL["function"]["parameters"]}]
    tbody = {"model": a.tool_model, "max_tokens": 128, "tools": tools, "temperature": 0,
             "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}]}
    s, _, d, _ = http(base, "POST", "/v1/messages", tbody)
    uses = [b for b in (d or {}).get("content", []) if b.get("type") == "tool_use"] if s == 200 else []
    record("anthropic", "tools: a tool_use block with input, stop_reason=tool_use",
           uses and uses[0].get("name") == "get_weather" and isinstance(uses[0].get("input"), dict) and d.get("stop_reason") == "tool_use",
           f"{s} stop={(d or {}).get('stop_reason') if isinstance(d, dict) else d} {str(uses)[:120]}")
    s, ev = sse(base, "/v1/messages", dict(tbody, stream=True))
    partial = "".join(d.get("delta", {}).get("partial_json", "") for e, d in ev if e == "content_block_delta" and isinstance(d, dict))
    try:
        ok = bool(json.loads(partial))
    except ValueError:
        ok = False
    record("anthropic", "tools streamed: input_json_delta assembles to JSON", ok, repr(partial[:80]))


def response_text(d):
    return "".join(c.get("text", "") for o in (d or {}).get("output", []) if o.get("type") == "message"
                   for c in o.get("content", [])) if isinstance(d, dict) else ""


def check_responses(base, a):
    print(f"responses api ({a.model}; tools on {a.tool_model})")
    s, _, d, _ = http(base, "POST", "/v1/responses",
                      {"model": a.model, "input": "My name is Ada. Say hello to me.", "max_output_tokens": 40, "store": True, "temperature": 0})
    rid = (d or {}).get("id") if s == 200 else None
    text = response_text(d) if s == 200 else ""
    record("responses", "non-streamed response completes with output text",
           s == 200 and d.get("status") == "completed" and text, f"{s} {text[:40]!r}")
    if rid:
        s, _, d2, _ = http(base, "GET", f"/v1/responses/{rid}")
        record("responses", "a stored response can be fetched by id", s == 200 and d2.get("id") == rid, str(s))
        s, _, d3, _ = http(base, "GET", f"/v1/responses/{rid}/input_items")
        record("responses", "its input items can be listed", s == 200 and (d3 or {}).get("data"), str(s))
        s, _, d4, _ = http(base, "POST", "/v1/responses",
                           {"model": a.model, "input": "What is my name? One word.", "previous_response_id": rid,
                            "max_output_tokens": 16, "temperature": 0})
        t4 = response_text(d4) if s == 200 else ""
        record("responses", "previous_response_id carries the conversation", "ada" in t4.lower(), repr(t4[:40]))
    s, ev = sse(base, "/v1/responses", {"model": a.model, "input": "Count from 1 to 5.", "stream": True, "max_output_tokens": 30})
    kinds = [(d.get("type") if isinstance(d, dict) else e) for e, d in ev]
    record("responses", "stream: created → output_text.delta → completed",
           s == 200 and kinds and kinds[0] == "response.created" and "response.output_text.delta" in kinds and kinds[-1] == "response.completed",
           f"{list(dict.fromkeys(kinds))[:8]}")
    tools = [{"type": "function", "name": "get_weather", "description": "Get the current weather for a city.",
              "parameters": WEATHER_TOOL["function"]["parameters"]}]
    s, _, d, _ = http(base, "POST", "/v1/responses", {"model": a.tool_model, "input": "What is the weather in Paris right now?",
                                                      "tools": tools, "max_output_tokens": 128, "temperature": 0})
    calls = [o for o in (d or {}).get("output", []) if o.get("type") == "function_call"] if s == 200 else []
    try:
        ok = calls and calls[0]["name"] == "get_weather" and json.loads(calls[0]["arguments"])
    except (ValueError, KeyError):
        ok = False
    record("responses", "tools: a function_call output item with JSON arguments", bool(ok), f"{s} {str(calls or d)[:120]}")


def check_embeddings(base, a):
    print(f"embeddings ({a.model})")
    # No embedding model is served (docs/ARCHITECTURE.md § HTTP API): the endpoint stays so an
    # SDK gets a TYPED error pointing at a cloud provider instead of a bare 404.
    s, _, d, _ = http(base, "POST", "/v1/embeddings", {"model": a.model, "input": ["the sky is blue", "grass is green"]})
    if s == 200:
        vecs = [e.get("embedding") for e in (d or {}).get("data", [])]
        record("embeddings", "two inputs give two equal-length vectors",
               len(vecs) == 2 and vecs[0] and len(vecs[0]) == len(vecs[1]), f"dims={len(vecs[0]) if vecs and vecs[0] else 0}")
    else:
        err = (d or {}).get("error", {}) if isinstance(d, dict) else {}
        record("embeddings", "not served here: a typed 501 naming the alternative",
               s == 501 and err.get("type") == "not_implemented_error", f"{s} {err.get('message', '')[:80]}")


def check_mcp(base, a):
    print("mcp")
    hdr = {"Accept": "application/json, text/event-stream"}

    def rpc(i, method, params=None):
        s, _, d, _ = http(base, "POST", "/mcp", {"jsonrpc": "2.0", "id": i, "method": method, "params": params or {}}, headers=hdr)
        return s, d if isinstance(d, dict) else {}
    s, d = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "feature_check", "version": "1"}})
    record("mcp", "initialize names the server", s == 200 and d.get("result", {}).get("serverInfo", {}).get("name") == "swarmllm")
    s, d = rpc(2, "tools/list")
    tools = {t["name"]: t for t in d.get("result", {}).get("tools", [])}
    record("mcp", "tools/list returns 7 tools", len(tools) == 7, ",".join(tools))
    s, d = rpc(3, "resources/list")
    res = [r["uri"] for r in d.get("result", {}).get("resources", [])]
    record("mcp", "resources/list returns status, models, peers",
           {"swarmllm://status", "swarmllm://models", "swarmllm://peers"} <= set(res), ",".join(res))
    s, d = rpc(4, "resources/read", {"uri": "swarmllm://status"})
    record("mcp", "resources/read swarmllm://status has contents", bool(d.get("result", {}).get("contents")))
    infer = next((n for n in tools if n in ("swarm_infer", "infer", "chat", "generate")), None) or \
        next((n for n in tools if "infer" in n or "chat" in n), None)
    if infer:
        props = tools[infer].get("inputSchema", {}).get("properties", {})
        args = {"model": a.model, "max_tokens": 8}
        args["prompt" if "prompt" in props else "messages"] = "Say the word fig." if "prompt" in props else \
            [{"role": "user", "content": "Say the word fig."}]
        args = {k: v for k, v in args.items() if k in props}
        s, d = rpc(5, "tools/call", {"name": infer, "arguments": args})
        r = d.get("result", {})
        record("mcp", f"tools/call {infer} returns content", s == 200 and r.get("content") and not r.get("isError"),
               str(r or d)[:160])


def check_errors(base, a):
    print("error contract")
    s, _, d, _ = http(base, "POST", "/v1/chat/completions", openai_body("no-such-model-xyz", "hi"))
    record("errors", "unknown model → 404 with an OpenAI error object", s == 404 and isinstance((d or {}).get("error"), dict), f"{s} {str(d)[:100]}")
    s, _, d, _ = http(base, "POST", "/v1/chat/completions", openai_body(a.model, "hi"), auth=False,
                      headers={"Authorization": "Bearer wrong-key"})
    record("errors", "a wrong key → 401", s == 401, str(s))
    s, _, d, _ = http(base, "POST", "/v1/chat/completions", {"model": a.model, "messages": "not a list"})
    record("errors", "a malformed body → 400 or 422", s in (400, 422), str(s))
    s, _, d, _ = http(base, "POST", "/v1/messages", {"model": "no-such-model-xyz", "max_tokens": 4,
                                                     "messages": [{"role": "user", "content": "hi"}]})
    record("errors", "anthropic unknown model → 404 with type=error", s == 404 and (d or {}).get("type") == "error", f"{s} {str(d)[:100]}")


def check_swarm(base, a):
    print(f"swarm splits ({a.split_model})")
    s, _, peers, _ = http(base, "GET", "/api/admin/peers")
    record("swarm", "connected to at least one peer", s == 200 and len(peers or []) > 0, f"{len(peers or [])} peers")
    prompt = "Explain in three sentences why the sky is blue."
    for label, route in (("every layer on peers", {"pretend_local_holds": "none"}),
                         ("first layers here, the rest on peers", {"pretend_local_holds": a.split_local})):
        s, h, d, wall = http(base, "POST", "/v1/chat/completions",
                             openai_body(a.split_model, prompt, max_tokens=64, swarm_route=route), timeout=900)
        txt = first_choice(d).get("message", {}).get("content", "") if s == 200 else str(d)[:160]
        u = (d or {}).get("usage", {}) if s == 200 else {}
        sane = any(w in txt.lower() for w in ("scatter", "rayleigh", "wavelength"))
        record("swarm", f"{label}: a sane reply through another machine",
               s == 200 and h.get("x-swarm-route") not in (None, "local") and sane,
               f"{s} {wall:.1f}s route={h.get('x-swarm-route')} nodes={h.get('x-swarm-nodes')} "
               f"tokens={u.get('completion_tokens')} {txt[:50]!r}")


SECTIONS = {
    "core": check_core, "admin": check_admin, "dashboard": check_dashboard, "openai": check_openai,
    "anthropic": check_anthropic, "responses": check_responses, "embeddings": check_embeddings,
    "mcp": check_mcp, "errors": check_errors, "swarm": check_swarm,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8800")
    p.add_argument("--model", default="llama-3.2-1b-instruct-q8-0", help="small local model for the API surfaces")
    p.add_argument("--tool-model", default="qwen2.5-coder-7b-instruct-q4-k-m", help="a model that calls tools")
    p.add_argument("--split-model", default="llama-3.2-3b-instruct-q4-k-m", help="a model the peers hold too")
    p.add_argument("--split-local", default="0", help="the SHARD range (not layers) this node pretends to hold for the partial split")
    p.add_argument("--only", default=",".join(SECTIONS), help="comma-separated sections")
    a = p.parse_args()
    for name in a.only.split(","):
        try:
            SECTIONS[name](a.base, a)
        except Exception as e:  # noqa: BLE001 — a crash in one section must not hide the rest
            record(name, "section ran to the end", False, repr(e)[:200])
    failed = [r for r in RESULTS if not r[2]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    for sec, name, _, detail in failed:
        print(f"  FAIL [{sec}] {name} — {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
