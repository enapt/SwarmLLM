#!/usr/bin/env python3
"""Row-level replay of the flowing tree (dsd_flow.rs) to compare how a level's rows are ranked.

Model (Sequoia's positional acceptance, positions independent): at depth i below the tree's start the
far end draws the drafter's rank-r child with probability P[r] (or none of the listed ones).
A drafter call grows LEVELS levels; each level proposes up to ROWS rows, best by `score`, each a
child of a row already READ (a row is read in the level after it is proposed, or the next call's
first level), while live rows (the walk's row's descendants) < LIVE. The call's rows then go through
this node's half and over the link in ONE forward; the far half reads forwards in order; every row's
draw comes back one way later. The walk moves to the child carrying the draw once that draw is
here; no such child = a miss: the draw is pushed as a row and sent with the next forward.
Costs (measured 2026-10-23 on the real link, 3B drafter, 7B halves): a call 3 + 10 ms a level of
<= 8 rows; this node's half 9.5 + 0.18 ms a row; the far half 20 + 0.5 ms a row; one way 197 ms.

`tree_schedule_sim.py`'s `flow` prices a tree as its top-M paths; this one grows it row by row, so a
ranking rule (`policy`) can be compared. Calibration (6 replies of 400 each): at P[0] 0.82 it walks
0.90 tokens a forward (measured 0.90) at 14.2 tok/s, at 0.86 16.4 (measured 15.4-18.3) — it waits
longer than the real flow, so read it for ORDER, not level. It reproduces the order measured
(`docs/invariants/scheduling.md` § "With the 3B the flow keeps three levels"): 4 rows < 8, 2 levels
≈ 3, 16 rows < 8. What it ruled out (2026-10-23): holding rows past the walk's round-trip reach back
(`policy="horizon"`, every horizon 6-12 and weight lost 1-9%), a bonus for depth (≤ +0.3%), live
192 (+0.4-0.9%). What moves it is the drafter's first-choice rate: P[0] 0.82 → 0.86 → 0.90 gave
14.2 → 16.4 → 19.3 tok/s.
usage: flow_rows_sim.py [replies]
"""
import heapq
import math
import random
import statistics
import sys

ONE_WAY = 197.0
FAR_FIXED, FAR_ROW = 20.0, 0.5
NEAR_FIXED, NEAR_ROW = 9.5, 0.18
CALL_FIXED, LEVEL_MS = 3.0, 10.0
WINDOW = 12
CHILDREN = 8  # followers a read row lists


def run(P, seed, tokens=400, levels=3, rows=8, live_most=128, policy="best", horizon=8, lam=1.0):
    rng = random.Random(seed)
    truth = []

    def true_rank(depth):  # rank of the far end's draw at `depth` (1 = the root's child)
        while len(truth) < depth:
            u, acc, r = rng.random(), 0.0, None
            for k, p in enumerate(P):
                acc += p
                if u < acc:
                    r = k
                    break
            truth.append(r)
        return truth[depth - 1]

    logp = [math.log(p) for p in P]
    # node: parent, rank, depth, score(logprob), read, sent_forward, answer_at
    par, rnk, dep, lp, read = [-1], [None], [0], [0.0], [False]
    kids = [dict()]
    answer_at = [None]  # time the far end's draw for this row is HERE
    on_truth = [True]
    walk = 0  # node id the walk is at
    unread = [0]  # proposed or pushed rows not yet read by the drafter
    unsent = [0]
    cand = []  # heap of (-score, node_parent, rank)
    t, far_free = 0.0, 0.0
    in_flight = []  # arrival times of forwards' answers
    forwards = drafted_ms = near_ms = waiting = 0.0
    nfw = 0

    def live_rows():
        # descendants of the walk node
        n, stack = 0, [walk]
        while stack:
            x = stack.pop()
            for c in kids[x].values():
                n += 1
                stack.append(c)
        return n

    def score(parent, r):
        s = lp[parent] + logp[r]
        if policy == "horizon":
            below = dep[parent] + 1 - dep[walk]
            if below > horizon:
                s -= lam * (below - horizon)
        return s

    def add_row(parent, r):
        nid = len(par)
        par.append(parent); rnk.append(r); dep.append(dep[parent] + 1)
        lp.append(lp[parent] + (logp[r] if r is not None else -9.0)); read.append(False)
        kids.append(dict()); answer_at.append(None)
        on_truth.append(on_truth[parent] and r == true_rank(dep[parent] + 1))
        kids[parent][r] = nid
        unsent.append(nid)
        return nid

    def is_live(x):
        while x != -1:
            if x == walk:
                return True
            x = par[x]
        return False

    def advance_walk():
        nonlocal walk
        moved = 0
        while answer_at[walk] is not None and answer_at[walk] <= t:
            r = true_rank(dep[walk] + 1)
            c = kids[walk].get(r)
            if c is None:  # a miss: push the draw
                c = add_row(walk, r)
                unread.append(c)
                walk = c
                moved += 1
                break
            walk = c
            moved += 1
        return moved

    def grow():
        nonlocal t, drafted_ms
        if live_rows() >= live_most:
            return False
        proposed = 0
        for _ in range(levels):
            # read every unread row: its followers become candidates
            for x in unread:
                if not read[x]:
                    read[x] = True
                    for r in range(min(CHILDREN, len(P))):
                        if r not in kids[x]:
                            heapq.heappush(cand, (-0.0, x, r))
            unread.clear()
            # rescore lazily: rebuild the heap with current scores (the walk moved)
            items = [(-(score(x, r)), x, r) for (_, x, r) in cand if is_live(x) and r not in kids[x]]
            heapq.heapify(items)
            cand[:] = items
            n = 0
            live = live_rows()
            while n < rows and live < live_most and cand:
                _, x, r = heapq.heappop(cand)
                if not is_live(x) or r in kids[x]:
                    continue
                c = add_row(x, r)
                unread.append(c)
                n += 1
                live += 1
            proposed += n
        cost = CALL_FIXED + LEVEL_MS * levels
        t += cost
        drafted_ms += cost
        return proposed > 0

    def send():
        nonlocal t, far_free, near_ms, nfw
        batch = [x for x in unsent if answer_at[x] is None]
        unsent.clear()
        if not batch:
            return False
        c = NEAR_FIXED + NEAR_ROW * len(batch)
        t += c
        near_ms += c
        arrive = t + ONE_WAY
        start = max(arrive, far_free)
        far_free = start + FAR_FIXED + FAR_ROW * len(batch)
        back = far_free + ONE_WAY
        for x in batch:
            answer_at[x] = back
        heapq.heappush(in_flight, back)
        nfw += 1
        return True

    # the root's own draw comes with the prompt: known at t=0
    answer_at[0] = 0.0
    unsent.clear()
    while dep[walk] < tokens:
        while in_flight and in_flight[0] <= t:
            heapq.heappop(in_flight)
        advance_walk()
        if dep[walk] >= tokens:
            break
        grew = grow()
        if unsent and len(in_flight) < WINDOW:
            send()
            continue
        if grew:
            continue
        # wait for the next answer
        if in_flight:
            nxt = heapq.heappop(in_flight)
            if nxt > t:
                waiting += nxt - t
                t = nxt
        else:
            t += 1.0
    return dict(tps=tokens / t * 1000.0, tok_fw=tokens / max(nfw, 1), wait=waiting / 1000.0,
                fw=nfw, draft=drafted_ms / max(nfw, 1), near=near_ms / max(nfw, 1))


def mean(P, seeds, **kw):
    rs = [run(P, s, **kw) for s in seeds]
    return {k: statistics.mean(r[k] for r in rs) for k in rs[0]}


if __name__ == "__main__":
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    for p0 in (0.82, 0.86, 0.90):
        rest = 1 - p0
        P = [p0, rest * 0.45, rest * 0.2, rest * 0.1, rest * 0.06, rest * 0.04, rest * 0.03, rest * 0.02]
        base = mean(P, seeds)
        print(f"P0 {p0}: best-first 3x8x128 {base['tps']:.2f} tok/s  tok/fw {base['tok_fw']:.2f}  wait {base['wait']:.2f} s")
        for kw in (dict(levels=2), dict(rows=4), dict(rows=16)):
            r = mean(P, seeds, **kw)
            print(f"   {kw}: {r['tps']:.2f}  tok/fw {r['tok_fw']:.2f}  wait {r['wait']:.2f}")
        for h, lam in ((6, 0.5), (8, 0.5), (8, 1.0), (12, 0.5), (12, 1.0)):
            r = mean(P, seeds, policy="horizon", horizon=h, lam=lam)
            print(f"   horizon {h} lam {lam}: {r['tps']:.2f}  tok/fw {r['tok_fw']:.2f}  wait {r['wait']:.2f}")
