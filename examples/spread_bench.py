#!/usr/bin/env python3
"""Spread-inference benchmark: one model on this machine, on a named peer, and
split across both — measured through the live node's own API.

usage:
  spread_bench.py MODEL --arm local --arm 'peer:holds=none,only=bf7b' \
                        --arm 'split:holds=0-4,only=bf7b' [--kind decode|prefill]
                        [--reps N] [--max-tokens N] [--prompt-tokens N] [--out F]

An ARM is `label[:key=val,...]`, turned into the request's `swarm_route` field:
  holds=none|all|A-B   pretend this node holds only these shards (`pretend_local_holds`)
  only=P1+P2           exclude EVERY other node that holds a part of MODEL or is
                       connected, so the planner can use only this node and P1, P2
  exclude=P1+P2        exclude just these (prefixes of node ids, >= 4 hex chars)
  peer=P:A-B           plan as though peer P held only shards A-B (`pretend_peer_holds`)
                       — with holds=0-(A-1) this makes a two-machine split the ONLY plan
`swarm_route` only ever makes the candidate set smaller (inference/route_override.rs).
To measure a split between two machines that could each hold the model, restrict
BOTH: `holds=0-3,only=P,peer=P:4-7` leaves the split as the only plan (#738).

Method, and why each part is there:
- Arms are INTERLEAVED, rotating the order every rep, so drift in the swarm or on
  the box lands on every arm rather than on whichever ran last (gotcha #700).
- The ROUTE is recorded per request (x-swarm-nodes + the per-segment layer ranges
  in Server-Timing) and the summary groups by it: the planner re-prices on live
  latency, and averaging across two topologies silently averages two things
  (perf_baseline_0920: one override gave segs=2 once and segs=1 twice).
- decode: one prompt per rep, shared by every arm in that rep, so replies can be
  compared; `--kind prefill`: a long prompt UNIQUE per request (the nonce leads, so
  no prefix-cache block can hit) and a short answer, so TTFT is prompt reading.
  Sizes are `usage.prompt_tokens`, never word counts (#720).
- decode tok/s = (completion_tokens - 1) / (t_last - t_first), a client-side
  window (#312); ms/token is its inverse. Prompt tok/s = prompt_tokens / TTFT,
  which INCLUDES scheduling and every network crossing — it is what a user waits.
- This machine's card is sampled during every request (utilisation and memory)
  because a laptop's card is shared with the desktop and a browser; a busy card
  is a different configuration, not noise.
- WHICH PATH ran is read from this node's own log (`--log`), by the request's
  id: the n-gram loop (`try_ngram_only_distributed ELIGIBLE`, one round trip per
  step), the whole-model hand-off (`remote-generate fast path`), a local
  generate, or the streaming local path. A streamed reply carries no route
  headers and the HTTP response is identical whichever path ran (#644), so
  without this a number cannot be attributed.
- `--order blocks` runs each arm's reps together, UNLOADING the model here
  before each block: a split leaves a partial worker behind, and on
  v0.3.209 the next whole-model request against it is refused
  (2026-09-27, FUTURE_WORK — "a partial worker cannot grow"). `--rounds 2`
  repeats the block sequence so drift still lands on every arm.
Petals measured the same two quantities for geo-distributed inference —
single-batch generation steps/s and parallel-forward tokens/s — and found
generation "degrades with higher latency" rather than bandwidth (Borzunov et al.,
arXiv 2312.08361). This harness is that shape against our own swarm.
"""
import argparse, json, os, re, statistics, subprocess, sys, threading, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--arm", action="append", required=True)
ap.add_argument("--kind", choices=["decode", "prefill"], default="decode")
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--warmup", type=int, default=1, help="warm-up rounds (every arm), not recorded")
ap.add_argument("--max-tokens", type=int, default=None)
ap.add_argument("--prompt-tokens", type=int, default=2000, help="approximate, --kind prefill")
ap.add_argument("--gap", type=float, default=3.0, help="seconds between requests")
ap.add_argument("--port", type=int, default=8800)
ap.add_argument("--out", default=None, help="JSONL, one line per recorded request")
ap.add_argument("--order", choices=["interleave", "blocks"], default="interleave")
ap.add_argument("--rounds", type=int, default=1, help="--order blocks: repeat the block sequence")
ap.add_argument("--log", default=os.path.expanduser("~/.local/share/swarmllm/node.log"))
ap.add_argument("--unload-also", default="", help="--order blocks: other model ids to unload before each block")
args = ap.parse_args()

KEY = open(os.environ.get("SWARMLLM_API_KEY_FILE", os.path.expanduser("~/.local/share/swarmllm/api_key"))).read().strip()
BASE = f"http://localhost:{args.port}"
MAX_TOKENS = args.max_tokens or (128 if args.kind == "decode" else 8)


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": f"Bearer {KEY}"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def known_nodes():
    """Every node id this node knows in connection with MODEL: connected peers and
    every holder of any part. `only=` has to exclude all of them — a holder that
    is not connected right now can be by the time the request is planned."""
    peers = get("/api/admin/peers")
    ids = {p["node_id"] for p in peers}
    rtt = {p["node_id"]: p.get("measured_min_rtt_ms") for p in peers}
    gpu = {p["node_id"]: p.get("gpu") for p in peers}
    for m in get("/api/admin/models"):
        if m.get("id") == args.model:
            for s in m.get("shards") or []:
                ids.update(s.get("holder_ids") or [])
    return ids, rtt, gpu


def parse_arm(spec, nodes):
    label, _, rest = spec.partition(":")
    route = {}
    for kv in filter(None, rest.split(",")):
        k, _, v = kv.partition("=")
        if k == "holds":
            route["pretend_local_holds"] = v
        elif k == "only":
            keep = v.split("+")
            route["exclude_nodes"] = sorted(n[:16] for n in nodes if not any(n.startswith(p) for p in keep))
        elif k == "exclude":
            route["exclude_nodes"] = v.split("+")
        elif k == "peer":
            node, _, holds = v.partition(":")
            route.setdefault("pretend_peer_holds", {})[node] = holds
        else:
            sys.exit(f"arm {spec!r}: unknown key {k!r}")
    return label, route


TOPICS = [
    "how a city's water supply works, from the reservoir to the tap",
    "the history and science of bread baking",
    "how airplanes stay in the air and why turbulence happens",
    "what happens inside a computer when you press a key",
    "how vaccines teach the immune system to recognise a disease",
    "why the seasons change and how the calendar was built around them",
]
WORDS = ("river mountain lantern harvest copper signal meadow archive orbit "
         "whisper engine garden pilot quarry ribbon summit thunder velvet willow "
         "harbor canvas ember falcon glacier island jasmine kettle ledger marble "
         "nectar oyster pepper quartz saddle timber umbrella vessel walnut yonder").split()


def decode_prompt(rep, tag):
    return (f"[bench {tag}] Write a long, detailed explanation of "
            f"{TOPICS[rep % len(TOPICS)]}. Use several paragraphs.")


def prefill_prompt():
    # The nonce LEADS, so every prefix-cache block of this prompt is new.
    nonce = uuid.uuid4().hex
    rng = int(nonce[:8], 16)
    words = []
    # ~0.75 words per token for this vocabulary; the real size is read back from usage.
    for i in range(int(args.prompt_tokens * 0.75)):
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        words.append(WORDS[rng % len(WORDS)])
        if i % 12 == 11:
            words[-1] += "."
    return f"[{nonce}] " + " ".join(words) + "\n\nIn one sentence, what is the text above about?"


class GpuSampler:
    def __init__(self):
        self.samples = []
        self.proc = None

    def __enter__(self):
        try:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-lms", "250"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()
        except Exception:
            self.proc = None
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                u, m = (int(x) for x in line.strip().split(","))
                self.samples.append((u, m))
            except Exception:
                pass

    def __exit__(self, *a):
        if self.proc:
            self.proc.terminate()

    def summary(self):
        if not self.samples:
            return None, None
        return round(statistics.mean(u for u, _ in self.samples)), max(m for _, m in self.samples)


SEG_RE = re.compile(r'seg\d+;dur=(\d+);desc="([0-9a-f]+) L(\d+)-(\d+)"')
DONE_RE = re.compile(r"DIAG: request complete request_id=(\S+) route=(\S+) segments=(\d+) model=(\S+)")
PLAN_RE = re.compile(r"Pipeline segment request_id=(\S+) segment=(\d+) node=([0-9a-f]{8})[0-9a-f]* layer_start=(\d+) layer_end=(\d+)")
PATHS = [  # (marker, name) — first match wins, in this order
    ("try_ngram_only_distributed ELIGIBLE", "ngram-loop"),
    ("remote-generate fast path: request sent", "handoff"),
    ("running it as a local generate", "local-generate"),
    ("split stream decode loop complete", "stream-local"),
    ("local streaming generate failed", "stream-local"),
]


def log_size():
    try:
        return os.path.getsize(args.log)
    except OSError:
        return None


def log_window(start):
    """This request's own lines: the last `request complete` for MODEL written
    since `start`, then every line carrying its id."""
    if start is None:
        return {}
    time.sleep(0.3)  # the completion line is written just after the stream ends
    with open(args.log, "rb") as f:
        f.seek(start)
        lines = f.read().decode(errors="replace").splitlines()
    done = [m for m in (DONE_RE.search(l) for l in lines) if m and m.group(4) == args.model]
    if not done:
        return {}
    rid = done[-1].group(1)
    mine = [l for l in lines if rid in l]
    path = next((name for marker, name in PATHS if any(marker in l for l in mine)), "pipeline")
    plan = [f"{m.group(3)} L{m.group(4)}-{m.group(5)}" for m in (PLAN_RE.search(l) for l in mine) if m]
    done_line = next(l for l in mine if "DIAG: request complete" in l)
    kv = dict(re.findall(r"(\w+)=(\S+)", done_line))
    return {"request_id": rid, "path": path, "plan": plan, "node_route": done[-1].group(2),
            "node_segments": int(done[-1].group(3)), "node_tpot_ms": kv.get("tpot_ms"),
            "node_ttft_ms": kv.get("ttft_ms"), "outcome": kv.get("outcome"),
            "retries": sum("retrying with fresh pipeline" in l for l in mine)}


def unload_here():
    """Retire this node's workers for MODEL and every `--unload-also` model, so
    the next arm starts from a card those models are not on. Unloading only
    MODEL was not enough — an idle worker of the model benchmarked just before
    keeps its graphics memory for the swap floor, and a 14B then got 1 of its 48
    layers on the card (2026-09-27, the first run of this harness). NEVER every
    model: a node on the swarm is serving peers, and their workers are theirs."""
    freed = []
    for mid in [args.model] + [m for m in args.unload_also.split(",") if m]:
        req = urllib.request.Request(f"{BASE}/api/admin/models/{mid}/unload", data=b"", method="POST",
                                     headers={"Authorization": f"Bearer {KEY}"})
        try:
            d = json.loads(urllib.request.urlopen(req, timeout=60).read())
            if d.get("segments_removed"):
                freed.append(f"{mid}:{d.get('estimated_freed_mb')}MB")
        except urllib.error.HTTPError:
            pass
    return ", ".join(freed) or "nothing resident"


def one(route, prompt):
    start = log_size()
    body = {"model": args.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    if route:
        body["swarm_route"] = route
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    rec = {"error": None}
    t0 = time.perf_counter()
    t_first = t_last = None
    text, usage, finish, chunks = [], None, None, 0
    with GpuSampler() as g:
        try:
            resp = urllib.request.urlopen(req, timeout=1800)
        except urllib.error.HTTPError as e:
            rec["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            resp = None
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            resp = None
        if resp is not None:
            h = {k.lower(): v for k, v in resp.headers.items()}
            rec["nodes"] = h.get("x-swarm-nodes", "")
            rec["route"] = h.get("x-swarm-route", "")
            rec["segments"] = [f"{n} L{a}-{b}" for _, n, a, b in SEG_RE.findall(h.get("server-timing", ""))]
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
                    rec["error"] = json.dumps(d["error"])[:300]
                    break
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    c = ch.get("delta", {}).get("content")
                    if c:
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        chunks += 1
                        text.append(c)
                    finish = ch.get("finish_reason") or finish
    wall = time.perf_counter() - t0
    util, mem = g.summary()
    ct = (usage or {}).get("completion_tokens")
    pt = (usage or {}).get("prompt_tokens")
    ttft = (t_first - t0) if t_first else None
    decode = (ct - 1) / (t_last - t_first) if ct and ct > 1 and t_last and t_last > t_first else None
    rec.update({
        "wall_s": round(wall, 3), "ttft_s": round(ttft, 3) if ttft else None,
        "prompt_tokens": pt, "completion_tokens": ct, "chunks": chunks, "finish": finish,
        "decode_tps": round(decode, 3) if decode else None,
        "ms_per_token": round(1000 / decode, 1) if decode else None,
        "prompt_tps": round(pt / ttft, 1) if pt and ttft else None,
        "gpu_util_mean": util, "gpu_mem_max_mib": mem, "text": "".join(text),
    })
    rec.update(log_window(start))
    return rec


def route_sig(r):
    where = " | ".join(r.get("plan") or r.get("segments") or []) or r.get("nodes") or "this node"
    return f"{r.get('path', '?')}: {where}"


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


nodes, rtt, gpu = known_nodes()
arms = [parse_arm(a, nodes) for a in args.arm]
print(f"== {args.model} kind={args.kind} max_tokens={MAX_TOKENS} reps={args.reps} warmup={args.warmup} "
      f"at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)
for label, route in arms:
    kept = sorted(n for n in nodes if not any(n.startswith(e) for e in route.get("exclude_nodes", [])))
    print(f"   arm {label}: swarm_route={json.dumps(route) if route else '(none)'}", flush=True)
    for n in kept:
        print(f"      may use {n[:16]} min_rtt_ms={rtt.get(n)} gpu={gpu.get(n)}", flush=True)
out = open(args.out, "a") if args.out else None
RUN = uuid.uuid4().hex[:6]
# (round, rep, arms in the order they run, unload-before?) — one entry per step.
schedule = []
if args.order == "interleave":
    for rep in range(-args.warmup, args.reps):
        k = rep % len(arms)
        schedule.append((0, rep, arms[k:] + arms[:k], False))
else:
    for rnd in range(args.rounds):
        for arm in arms:
            for rep in range(-args.warmup, args.reps):
                schedule.append((rnd, rep, [arm], rep == -args.warmup))
results = []
texts = {}
for rnd, rep, order, unload_first in schedule:
    if unload_first:
        print(f"  -- round {rnd + 1}, arm {order[0][0]}: unload here -> {unload_here()}", flush=True)
        time.sleep(2)
    # One prompt per (round, rep), whichever arm asks, so replies are comparable.
    shared = decode_prompt(rep, f"{RUN}{rnd}{rep}") if args.kind == "decode" else None
    for label, route in order:
        prompt = shared or prefill_prompt()
        r = one(route, prompt)
        r.update({"arm": label, "rep": rep, "round": rnd, "model": args.model, "kind": args.kind,
                  "swarm_route": route, "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        tag = "warmup" if rep < 0 else f"r{rnd + 1} rep {rep + 1}"
        print(f"  {tag:10} {label:10} ttft {r['ttft_s']}s | {r['prompt_tokens']}+{r['completion_tokens']} tok | "
              f"decode {r['decode_tps']} tok/s ({r['ms_per_token']} ms/tok) | prompt {r['prompt_tps']} tok/s | "
              f"gpu {r['gpu_util_mean']}% {r['gpu_mem_max_mib']}MiB | {route_sig(r)} retries={r.get('retries')} "
              f"| err={r['error']}", flush=True)
        if rep >= 0:
            results.append(r)
            texts.setdefault((rnd, rep), {})[label] = r["text"]
            if out:
                out.write(json.dumps(r) + "\n")
                out.flush()
        time.sleep(args.gap)

if args.kind == "decode" and len(arms) > 1:
    ref_label = arms[0][0]
    print(f"\n== reply agreement with {ref_label}, per prompt (common-prefix chars / reply length)")
    for key, by_arm in sorted(texts.items()):
        ref = by_arm.get(ref_label)
        if ref is None:
            continue
        print(f"  r{key[0] + 1} rep {key[1] + 1}: " + ", ".join(
            f"{l} {common_prefix(ref, t)}/{len(t)}" for l, t in by_arm.items() if l != ref_label))

print("\n== SUMMARY (grouped by arm AND the route it actually got)")
groups = {}
for r in results:
    groups.setdefault((r["arm"], route_sig(r)), []).append(r)
for (label, sig), rs in groups.items():
    ok = [r for r in rs if not r["error"]]
    def med(key):
        v = [r[key] for r in ok if r.get(key) is not None]
        return (round(statistics.median(v), 2), round(min(v), 2), round(max(v), 2)) if v else None
    print(f"  {label:10} n={len(rs)} ok={len(ok)} route: {sig}")
    print(f"             decode tok/s median/min/max {med('decode_tps')}  ms/token {med('ms_per_token')}")
    print(f"             ttft s {med('ttft_s')}  prompt tok/s {med('prompt_tps')}  prompt tokens {med('prompt_tokens')}")
    print(f"             this card: util% {med('gpu_util_mean')}  mem MiB {med('gpu_mem_max_mib')}")
    for r in rs:
        if r["error"]:
            print(f"             ERROR rep {r['rep'] + 1}: {r['error'][:200]}")
