#!/usr/bin/env python3
"""What bounds decode on this machine: the GPU, or one CPU thread?

WHY THIS EXISTS
---------------
"Decode is slow" was assumed to mean the card was working hard. It did not:
measured 2026-09-22 on the release node, the worker burned 22.7-26.0 ms of CPU
per token against 21.0-24.1 ms of wall per token — 1.08 cores busy — while the
GPU sat at 52% utilization. The token was being spent submitting work, not
doing it.
→ `docs/invariants/inference.md` § "A decode token is bound by GPU submission
  COUNT, not bandwidth".

This is the cheap first reading. It needs no profiler and does not restart the
node. Run it before theorising about a decode number that will not move.

HOW TO READ IT
--------------
  cpu/token ~= wall/token   ->  a CPU THREAD is the bottleneck. The GPU is
                                waiting for work. Look at submissions per token
                                (`examples/decode_submissions.sh`), op fusion,
                                and anything allocating per op.
  cpu/token << wall/token   ->  the CPU submits and then waits. The cost is
                                GPU-side: real kernel time or launch latency.
                                Bigger/fewer kernels, or a faster card.
  many cores busy           ->  this model is running on the CPU backend, not
                                the GPU. Check placement before reading further;
                                a CPU forward uses rayon across all cores.

WHAT IT CANNOT TELL YOU
-----------------------
  * WHERE in the CPU thread the time goes. A saturated thread could be in the
    driver, in allocation, or in framework overhead — that needs nsys.
  * Anything, if the request was served by a PEER. It reports the route so you
    can see that; a remote reply leaves the local worker idle and the numbers
    meaningless.
  * Clean numbers on a busy box. Check `uptime` first; measurement wants idle
    (#367).

Usage: decode_bound_by.py [MODEL] [MAX_TOKENS] [REPS]
"""
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

MODEL = sys.argv[1] if len(sys.argv) > 1 else "llama-3.2-3b-instruct-q4-k-m"
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
PORT = int(os.environ.get("PORT", "8800"))
KEY_FILE = os.environ.get(
    "KEY_FILE", os.path.expanduser("~/.local/share/swarmllm/api_key"))
CLK = os.sysconf("SC_CLK_TCK")
PROMPT = "Write a long detailed essay about the history of bridges."

try:
    KEY = open(KEY_FILE).read().strip()
except OSError as e:
    sys.exit(f"no api key at {KEY_FILE}: {e}")


def post(path, body, timeout=900):
    req = urllib.request.Request(
        f"http://localhost:{PORT}{path}", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()), dict(r.headers)


def gen(max_tokens):
    return post("/v1/chat/completions", {
        "model": MODEL, "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0})


def swarmllm_pids():
    out = subprocess.run(["pgrep", "-x", "swarmllm"],
                         capture_output=True, text=True)
    pids = []
    for p in out.stdout.split():
        # By /proc/PID/exe, never a cmdline match.
        try:
            if os.readlink(f"/proc/{p}/exe").endswith("swarmllm"):
                pids.append(int(p))
        except OSError:
            pass
    return pids


def stat_fields(pid):
    with open(f"/proc/{pid}/stat") as f:
        s = f.read()
    return s[s.rindex(")") + 2:].split()


def cputime(pid):
    f = stat_fields(pid)
    return (int(f[11]) + int(f[12])) / CLK   # utime + stime, all threads


def ppid(pid):
    return int(stat_fields(pid)[1])


def gpu_util(stop, samples):
    """Sample GPU utilization until `stop` is set."""
    while not stop.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            g, m = r.stdout.strip().splitlines()[0].split(",")
            samples.append((int(g), int(m)))
        except Exception:
            pass
        time.sleep(0.2)


print(f"model {MODEL}, {MAX_TOKENS} tokens x {REPS} reps, port {PORT}")
print(f"load average {open('/proc/loadavg').read().split()[0]} "
      f"(measurement wants an idle box)\n")

_, hdrs = gen(4)                               # warm: load the model
route = hdrs.get("x-swarm-route") or hdrs.get("x-swarm-nodes") or "(none)"
time.sleep(1)

pids = swarmllm_pids()
if not pids:
    sys.exit("no swarmllm process found")
daemon = next((p for p in pids if ppid(p) not in pids), pids[0])
workers = [p for p in pids if p != daemon]
print(f"daemon {daemon}, worker(s) {workers or '-'}, route header: {route}")
if not workers:
    sys.exit("no worker process — is this model served by a peer?")
w = workers[0]

rows = []
for rep in range(REPS):
    stop, samples = threading.Event(), []
    t = threading.Thread(target=gpu_util, args=(stop, samples), daemon=True)
    t.start()
    c0, t0 = cputime(w), time.time()
    body, _ = gen(MAX_TOKENS)
    wall, c1 = time.time() - t0, cputime(w)
    stop.set()
    t.join(timeout=2)
    toks = body["usage"]["completion_tokens"]
    cpu = c1 - c0
    guse = [g for g, _ in samples] or [-1]
    rows.append((toks, wall, cpu, statistics.median(guse)))
    print(f"  rep {rep+1}: {toks:>4} tok  wall {wall/toks*1000:>6.2f} ms/tok  "
          f"cpu {cpu/toks*1000:>6.2f} ms/tok  "
          f"{cpu/wall:>4.2f} cores  gpu {statistics.median(guse):>3.0f}%")
    time.sleep(1)

cores = statistics.median([c / wl for _, wl, c, _ in rows])
wall_ms = statistics.median([wl / tk * 1000 for tk, wl, _, _ in rows])
cpu_ms = statistics.median([c / tk * 1000 for tk, _, c, _ in rows])
gpu_med = statistics.median([g for *_, g in rows])

print(f"\nmedian: {wall_ms:.2f} ms/token wall, {cpu_ms:.2f} ms/token worker CPU,"
      f" {cores:.2f} cores busy, GPU {gpu_med:.0f}%")
if cores > 4:
    verdict = ("MANY CORES BUSY — this looks like the CPU backend, not the GPU."
               " Check placement before reading anything else into it.")
elif cpu_ms > 0.8 * wall_ms:
    verdict = ("ONE CPU THREAD IS THE BOTTLENECK. The GPU is waiting for work."
               " Next: examples/decode_submissions.sh for submissions/token.")
else:
    verdict = ("The CPU submits and waits — cost is GPU-side (kernel time or"
               " launch latency), not framework overhead.")
print(f"VERDICT: {verdict}")
