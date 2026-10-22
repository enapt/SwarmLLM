#!/usr/bin/env python3
"""How a split's checks are SCHEDULED, priced over a long and a short link (#134).

`spec_best_tree.py` and `spec_projection.py` price ONE round. This prices the
schedule around it: tree rounds as `dsd` runs them (the next tree grown beside
the check, a missed seed regrown from nothing before the next check), the same
rounds with the regrow sized by what it buys (`adaptive`, #134 (a)), several
self-contained tree checks in flight (`stream`, #134 (b): each rooted at the
confirmed reply, every one cutting the caches back), the chain stream that
ships (`spec_stream` with no branches: chunks rooted at the chunk before's
predicted end, a restart at every miss), and a tree that FLOWS (`flow`,
2026-10-22: rows sent as they are grown, the far end reading them as they
arrive and stalling only where no row holds the reply's token —
`docs/plans/large_model_splits.md` § "A tree that flows").

The acceptance model is Sequoia's positional one, as in `spec_best_tree.py`: at
each position the reply's token is the drafter's rank-r guess with probability
P[r] (or none of them), and a tree of M rows grown best-first is the top-M
paths by probability. Re-rooting keeps the rows under the reply's path. It
assumes positions independent — `spec_best_tree.py` replays real ones.

Calibrated 2026-10-21 against the real TH<->AU link (FUTURE_WORK #134): a regrow
from nothing on the card 155-211 ms (8 steps), on the processor 289-416 ms; a
round ~760-820 ms of which the network ~425. It reproduces what was measured
there within ~10%: processor-drafter tree rounds 5.8 (measured mean 5.4 over
400 tokens), the chain stream 6.6 (6.1), card-drafter tree rounds 7.2 (7.1-8.8
over 160 tokens). It is optimistic for the chain stream on a long link: it
leaves out each chunk's fixed cost and the work a restart throws away.
usage: tree_schedule_sim.py [seeds]
"""
import heapq
import random
import statistics
import sys

P = [0.745, 0.10, 0.04, 0.02]  # rank 1..4; the rest (0.095) no guess holds
NODES = 4096


def enumerate_tree():
    """Every node's path probability, best first (the infinite tree's top NODES)."""
    out = []
    heap = [(-q, (r,)) for r, q in enumerate(P)]
    heapq.heapify(heap)
    while heap and len(out) < NODES:
        negp, path = heapq.heappop(heap)
        out.append(-negp)
        for r, q in enumerate(P):
            heapq.heappush(heap, (negp * q, path + (r,)))
    return out


PROBS = enumerate_tree()


def thr(m):
    return PROBS[m - 1] if m > 0 else float("inf")

def count_at_least(x):
    # nodes with path probability >= x
    lo, hi = 0, len(PROBS)
    while lo < hi:
        mid = (lo + hi) // 2
        if PROBS[mid] >= x:
            lo = mid + 1
        else:
            hi = mid
    return lo

class Truth:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.ranks = []

    def rank(self, i):
        while len(self.ranks) <= i:
            u = self.rng.random()
            acc, r = 0.0, None
            for k, p in enumerate(P):
                acc += p
                if u < acc:
                    r = k
                    break
            self.ranks.append(r)  # None = a miss
        return self.ranks[i]

    def prefix_prob(self, start, length):
        """Probability of the true path from root `start` (the token AT start is
        the root) over the next `length` tokens; 0 if any is a miss."""
        p = 1.0
        for i in range(start + 1, start + 1 + length):
            r = self.rank(i)
            if r is None:
                return 0.0
            p *= P[r]
        return p

    def covered(self, root, m):
        """Depth of the true path a top-m tree rooted at `root` holds."""
        t, d, p = thr(m), 0, 1.0
        while True:
            r = self.rank(root + 1 + d)
            if r is None:
                return d
            p *= P[r]
            if p < t:
                return d
            d += 1

def reroot(truth, root, m, new_root):
    """Rows a top-m tree at `root` keeps under the true path to `new_root`."""
    if m == 0 or new_root == root:
        return m
    pp = truth.prefix_prob(root, new_root - root)
    if pp <= 0:
        return 0
    return min(count_at_least(thr(m) / pp), NODES - 1)

def check_ms(rows, link):
    return link["rtt"] + link["far_fixed"] + link["far_row"] * rows + link["near_row"] * rows + link["wire_row"] * rows

def grow_ms(rows_now, rows_add, drafter):
    # a call re-reads its seed, then grows in steps of `step` rows
    steps = -(-rows_add // drafter["step"])
    return drafter["call"] + drafter["reread_row"] * rows_now + drafter["step_ms"] * steps

STEPS = 8  # a tree grown from nothing today: 8 forwards, `per_step` rows each
PER_STEP = 8


def covered(truth, root, rows, depth):
    """A best-first tree of `rows` rows grown in `depth` forwards holds the true
    path to at most `depth` levels."""
    return min(truth.covered(root, rows), depth)


def level_acceptance(kept, levels=STEPS):
    """The `a` with a + a^2 + ... + a^levels = kept: how often a tree's next
    level holds the reply's token, fitted to what trees of `levels` levels kept."""
    lo, hi = 0.0, 0.999
    for _ in range(50):
        mid = (lo + hi) / 2
        if sum(mid ** i for i in range(1, levels + 1)) < kept:
            lo = mid
        else:
            hi = mid
    return lo


def steps_by_estimate(recent_kept, link, drafter):
    """What the code does: the steps that buy the most tokens per ms of the
    round, the tokens from the geometric fit to what checks recently kept."""
    a = level_acceptance(recent_kept)
    def value(s):
        e = sum(a ** i for i in range(1, s + 1))
        return (1 + e) / (check_ms(PER_STEP * s, link) + grow_ms(0, PER_STEP * s, drafter))
    return max((1, 2, 4, STEPS), key=value)


E_TRUE = {}


def steps_by_truth(link, drafter):
    """The bound: the same choice with each size's true expected depth."""
    def value(s):
        if s not in E_TRUE:
            rng = Truth(12345)
            E_TRUE[s] = statistics.mean(covered(rng, i * 50, PER_STEP * s, s) for i in range(80))
        return (1 + E_TRUE[s]) / (check_ms(PER_STEP * s, link) + grow_ms(0, PER_STEP * s, drafter))
    return max((1, 2, 4, STEPS), key=value)


def rounds(truth, tokens, link, drafter, regrow="fixed", send_most=64, seed_most=96, beside_most=128,
           beside_step=None):
    """`dsd`'s tree rounds: the next tree grown beside the check (up to 85% of
    its time), cut to the part under the outcome; a missed seed is regrown
    from nothing before the next check — 8 steps (`fixed`, today), or the
    steps that pay (`estimate`: the code's rule; `truth`: its bound).
    `beside_step` rows a forward beside the check (default the drafter's
    `step`; 32 with `beside_most=416` is `dsd_tree::beside_growth` since
    2026-10-22, a forward of 32 costing `step32_ms`)."""
    t, f, m, depth = 0.0, 0, 0, 0
    recent_kept = 3.0
    beside_step = beside_step or drafter["step"]
    beside_ms = drafter["step_ms"] if beside_step == drafter["step"] else drafter["step32_ms"]
    while f < tokens:
        if m == 0:
            s = {"fixed": lambda: STEPS,
                 "estimate": lambda: steps_by_estimate(recent_kept, link, drafter),
                 "truth": lambda: steps_by_truth(link, drafter)}[regrow]()
            t += grow_ms(0, PER_STEP * s, drafter)
            m, depth = PER_STEP * s, s
        n = min(m, send_most)
        ms = check_ms(n, link)
        budget, grown, steps = 0.85 * ms, 0, 0
        while grown < beside_most:
            step = drafter["call"] + drafter["reread_row"] * m + beside_ms if grown == 0 else beside_ms
            if budget < step:
                break
            budget -= step
            grown += beside_step
            steps += 1
        d = covered(truth, f, n, depth)
        recent_kept = 0.7 * recent_kept + 0.3 * d
        t += ms
        new_f = f + d + 1
        m = min(reroot(truth, f, m + grown, new_f), seed_most)
        depth = max(depth + steps - (new_f - f), 1) if m else 0
        f = new_f
    return tokens / t * 1000.0


def flow(truth, tokens, link, drafter, per_step=32, tree_most=1024, head=(13.0, 0.33), tail=(13.0, 0.33)):
    """A continuous tree stream, the tail walking (FlowSpec's shape over a long
    link): the drafter grows best-first under the root it last HEARD of, each
    forward's rows go through the head and over the link as soon as they are
    grown, the tail walks as far as the rows it holds reach — never waiting
    for a round to end — and each step forward of its walk comes back as a
    verdict that re-roots the growth. The drafter and the head share one card
    (one stream), so a step costs both. A token no guess holds (`P`'s miss)
    stalls the walk until the requester hears of it and grows under it: one
    round trip, where a round pays one for every check. `head` / `tail` are a
    forward's (fixed, per row) ms on each segment's card (the 7B's 14-layer
    halves on an RTX 3070 Laptop / 3060, `prefill_bench`
    `SWARM_BENCH_SPEC_TREE=1`: 23.4 ms at 32 rows over 16 layers)."""
    one_way = link["rtt"] / 2
    t, R, m, depth = 0.0, 0, 0, 0
    tail_f, tail_free = 0, 0.0
    verdicts = []  # (heard at the requester, confirmed position)
    pending = []  # (arrives at the tail, rows, the position its rows let the walk reach)

    def advance_tail(until):
        # The tail reads everything that has arrived by the time it is free in
        # ONE forward (one fixed cost), as a worker reads its queue.
        nonlocal tail_free, tail_f
        while pending:
            start = max(tail_free, pending[0][0])
            if start > until:
                return
            rows, reach = 0, 0
            while pending and pending[0][0] <= start:
                _, add, f = pending.pop(0)
                rows, reach = rows + add, max(reach, f)
            tail_free = start + tail[0] + tail[1] * rows
            if reach > tail_f:
                tail_f = reach
                heapq.heappush(verdicts, (tail_free + one_way, reach))

    while True:
        advance_tail(pending[-1][0] if pending else t)
        while verdicts and verdicts[0][0] <= t:
            heard, f = heapq.heappop(verdicts)
            if f >= tokens:
                return tokens / heard * 1000.0
            if f > R:
                m = min(reroot(truth, R, m, f), tree_most)
                depth = max(depth - (f - R), 1) if m else 0
                R = f
        if m >= tree_most:
            advance_tail(float("inf"))
            t = verdicts[0][0] if verdicts else t + 1.0
            continue
        add = min(per_step, tree_most - m)
        t += drafter["call"] + drafter["step32_ms"] * add / 32 + head[0] + head[1] * add
        m += add
        depth += 1
        pending.append((t + one_way + link["wire_row"] * add, add, R + min(truth.covered(R, m), depth) + 1))


def stream(truth, tokens, link, drafter, window=6, send_most=64, tree_most=224, seed_most=224, pace=True):
    """Self-contained tree checks, several in flight, each rooted at the
    confirmed reply with the tree as grown so far; the drafter grows two steps
    at a time between sends."""
    t, f, m, depth = 0.0, 0, 0, 0
    out = []  # (arrive, root, rows, depth)
    last_sent = (None, None)
    free_link_at = 0.0
    while f < tokens:
        while out and out[0][0] <= t:
            arrive, root, rows, dep = out.pop(0)
            new_f = root + covered(truth, root, rows, dep) + 1
            if new_f > f:
                m = min(reroot(truth, f, m, new_f), seed_most)
                depth = max(depth - (new_f - f), 1) if m else 0
                f = new_f
                last_sent = (None, None)
        if f >= tokens:
            break
        n = min(m, send_most)
        if len(out) < window and (f, n) != last_sent and (n > 0 or not out) and (not pace or t >= free_link_at):
            ms = check_ms(n, link)
            out.append((t + ms, f, n, depth))
            out.sort()
            last_sent = (f, n)
            free_link_at = t + (link["wire_row"] + link["far_row"]) * n
            continue
        if m < tree_most:
            add = min(drafter["step"] * 2, tree_most - m)
            t += grow_ms(m, add, drafter)
            m += add
            depth += 2
        else:
            t = out[0][0] if out else t + 1.0
    return tokens / t * 1000.0


def spec_stream(truth, tokens, link, drafter, trunk=3, window=3):
    """The chain stream that ships: chunks of `trunk` guesses along the
    drafter's likeliest path, each rooted at the PREDICTED end of the chunk
    before; a chunk not kept whole (or whose look-ahead the tail's sample
    refutes) restarts every chunk after it."""
    t, f = 0.0, 0
    out = []  # (arrive, root)
    nxt = 0
    while f < tokens:
        while out and out[0][0] <= t:
            arrive, root = out.pop(0)
            if root != f:
                continue  # dropped by a restart
            d = 0
            while d < trunk and truth.rank(root + 1 + d) == 0:
                d += 1
            whole = d == trunk and truth.rank(root + trunk + 1) == 0
            f = root + trunk + 1 if whole else root + d + 1
            if not whole:
                out = []
                nxt = f
        if f >= tokens:
            break
        if len(out) < window:
            t += drafter["call"] + drafter["step_ms"] * trunk / 2
            out.append((t + check_ms(trunk, link), nxt))
            nxt += trunk + 1
            continue
        t = out[0][0]
    return tokens / t * 1000.0

LINKS = {
    "far 425 ms": dict(rtt=425, far_fixed=40, far_row=0.3, near_row=0.5, wire_row=0.3),
    "near 20 ms": dict(rtt=20, far_fixed=10, far_row=0.3, near_row=0.5, wire_row=0.05),
}
DRAFTERS = {
    "card": dict(call=5.0, reread_row=0.55, step=8, step_ms=19.0, step32_ms=21.0),
    "processor": dict(call=8.0, reread_row=2.0, step=8, step_ms=45.0, step32_ms=120.0),
}


def main():
    tokens = 400
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
    print(f"tok/s over {tokens} tokens, mean of {len(seeds)} replies")
    for ln, link in LINKS.items():
        for dn, drafter in DRAFTERS.items():
            def mean(fn, **kw):
                return statistics.mean(fn(Truth(s), tokens, link, drafter, **kw) for s in seeds)
            print(
                f"{ln:11s} {dn:9s} tree rounds {mean(rounds):6.2f}"
                f"  regrow by estimate {mean(rounds, regrow='estimate'):6.2f}"
                f"  (bound {mean(rounds, regrow='truth'):6.2f})"
                f"  stream of trees {mean(stream):6.2f}"
                f"  chain stream {mean(spec_stream):6.2f}"
            )
            print(
                f"{'':21s} wide beside (32/416) {mean(rounds, regrow='estimate', beside_step=32, beside_most=416):6.2f}"
                f"  flow (tail walks) {mean(flow):6.2f}"
                f"  flow 16/step {mean(flow, per_step=16):6.2f}  flow 64/step {mean(flow, per_step=64):6.2f}"
                f"  flow ≤512 rows {mean(flow, tree_most=512):6.2f}"
            )


if __name__ == "__main__":
    main()
