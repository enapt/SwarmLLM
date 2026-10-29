#!/usr/bin/env python3
"""How wide the continuous stream's chunks should be (`pipeline::dsd_stream::StreamWidth`, #374).

`tree_schedule_sim.py` assumes positions independent. This replays REAL ones: each trace is a
reply the target wrote greedily and, at every position, whether the drafter's first choice was the
target's token given the true prefix (`drafter_ranks.py reduce ... ids`: `rank == 0`). With greedy
decoding that decides every guess, so a fixed width's chunks kept whole and restarts come out of
the trace alone — the 70B traces gave 46/16/29, 34/26/8 and 38/21/17 at 3 guesses against the
stream's own 46/16/29, 34/27/7 and 38/21/17 (2026-10-30).

The stream as `dsd_stream` runs it:
- this node drafts a chunk (`d` ms a guess), runs its own layers (`n0 + n1*k`), sends it, keeps at
  most `W` out, and reads an answer only between sends (`run_with` awaits `send_chunk`);
- a chunk at position q checks q..q+k-1 (k rows, the look-ahead last); a refusal at j gains j+1
  and ends every chunk out after it, the next restarting at the true position; none gains k;
- the far end runs chunks in arrival order, `a + b*k` each, and skips one built on a guess it has
  already refused (`forward_streams`, `StoodOnARefusal`); an answer is `rtt/2` away each way.

Calibrated on the 70B split (this node + an RTX 3060 in Australia running most of its 64 layers on
its processor; 5 vs 3 guesses, arms alternating, 6 replies): rtt 420, d 61, n0 250, n1 15, a 600,
b 40 reproduce all six decode times within 5% RMS. The other profiles are plausible shapes, not
measurements — they show where ONE width fails, which is the point.

usage: python3 -P examples/stream_width_sim.py TRACE.npz [TRACE.npz ...] [--tokens N]
"""
import sys


def simulate(h, width, P, n_tokens, W=3):
    """`width(state) -> k`. Returns (ms, chunks answered, kept whole, restarts)."""
    rtt, d, n0, n1, a, b = P["rtt"], P["d"], P["n0"], P["n1"], P["a"], P["b"]
    N = min(len(h), n_tokens)
    pos, t, far_free, epoch, front = 0, 0.0, 0.0, 0, 0
    out, hist = [], []
    chunks = kept = restarts = 0
    while pos < N:
        while out and out[0]["answer_at"] <= t:
            c = out.pop(0)
            if c["epoch"] != epoch or c["skipped"]:
                continue
            chunks += 1
            q, k = c["q"], c["k"]
            j = next((i for i in range(k) if q + i >= N or not h[q + i]), None)
            if j is None:
                pos = q + k
                kept += 1
                hist.append((k, k))
            else:
                pos = q + j + 1
                restarts += 1
                hist.append((j, k))
                epoch += 1
                out = [o for o in out if o["epoch"] == epoch]
                front = pos
            if pos >= N:
                return t, chunks, kept, restarts
        if len(out) < W:
            k = width(hist)
            t += k * d + n0 + n1 * k
            start = max(t + rtt / 2, far_free)
            refused_end = next((o["far_end"] for o in reversed(out)
                                if o["epoch"] == epoch and o["misses"]), None)
            c = dict(q=front, k=k, epoch=epoch, skipped=False)
            if refused_end is not None and refused_end <= start:
                c["skipped"], c["far_end"] = True, start
            else:
                c["far_end"] = far_free = start + a + b * k
            c["misses"] = any(front + i >= N or not h[front + i] for i in range(k))
            c["answer_at"] = c["far_end"] + rtt / 2
            out.append(c)
            front += k
        else:
            t = max(t, out[0]["answer_at"])
    return t, chunks, kept, restarts


def recent_alpha(hist):
    """`AcceptanceEstimate::recent()`: each chunk keeps 0.9 of what came before; prior 0.7 over 2."""
    acc = trials = 0.0
    for kept, k in hist:
        acc, trials = acc * 0.9 + kept, trials * 0.9 + kept + (1 if kept < k else 0)
    return (1.4 + acc) / (2.0 + trials)


def chunk_rate(k, alpha, check_ms, rtt, near_ms, d, W=3):
    """`dsd_stream::chunk_rate`, line for line."""
    alpha = min(max(alpha, 0.0), 0.99)
    build = d * k + near_ms
    far = max(check_ms - near_ms - rtt, 0.0)
    trip = d * k + check_ms
    steady = max(build, far, trip / W)
    refused = 1 - alpha ** k
    gain = refused / (1 - alpha)
    return gain / max(steady + refused * (trip - steady + build / 2), 1.0)


def chosen(P, narrow=3, wide=5):
    """The stream's choice, given the costs it would have measured (exploration left out)."""
    def width(hist):
        al = recent_alpha(hist)
        rate = {}
        for k in (narrow, wide):
            near = P["n0"] + P["n1"] * k
            check = near + P["rtt"] + P["a"] + P["b"] * k
            rate[k] = chunk_rate(k, al, check, P["rtt"], near, P["d"])
        return wide if rate[wide] > rate[narrow] else narrow
    return width


PROFILES = {
    "70B, AU 3060 mostly processor (fitted)": dict(rtt=420, d=61, n0=250, n1=15, a=600, b=40),
    "7B, far card, drafter on card, 440 ms": dict(rtt=440, d=10, n0=15, n1=2, a=40, b=3),
    "7B, far card, drafter on processor": dict(rtt=440, d=30, n0=15, n1=2, a=40, b=3),
    "7B, far processor, 270 ms": dict(rtt=270, d=20, n0=20, n1=2, a=150, b=40),
    "7B, far processor, dear rows": dict(rtt=270, d=20, n0=20, n1=2, a=100, b=80),
    "LAN, far card": dict(rtt=2, d=10, n0=15, n1=2, a=40, b=3),
    "LAN, far processor": dict(rtt=2, d=20, n0=20, n1=3, a=150, b=40),
}


def main(argv):
    import numpy as np
    tokens = 200
    if "--tokens" in argv:
        i = argv.index("--tokens")
        tokens = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        sys.exit(__doc__)
    # rank[0] is the reply's first token, which the prompt pass writes: the stream guesses from 1.
    traces = [[bool(r == 0) for r in np.load(p)["rank"]][1:] for p in argv]
    print(f"{len(traces)} traces, {tokens} tokens a reply — % faster than 3 guesses a chunk")
    for name, P in PROFILES.items():
        def total(width):
            return sum(simulate(h, width, P, tokens)[0] for h in traces)
        base = total(lambda _: 3)
        cells = [f"{lab} {100 * (base / total(w) - 1):+5.1f}%"
                 for lab, w in (("4", lambda _: 4), ("5", lambda _: 5), ("chosen", chosen(P)))]
        print(f"  {name:42s} " + "  ".join(cells))


if __name__ == "__main__":
    main(sys.argv[1:])
