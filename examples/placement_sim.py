#!/usr/bin/env python3
"""Convergence simulator for regional chain-completeness placement.

This covers `docs/plans/regional_pipelines.md` § Stage 3.

The question it answers: if every node decides on its own which parts of a
large model to fetch, from its own view of the swarm, do the nodes near one
another end up holding a COMPLETE chain of the model between them?

It also asks three follow-ups:
- Does that happen without two neighbours fetching the same gap (herding)?
- Do nodes avoid fetching a part, shedding it and fetching it again (gotcha
  #795 / #876's loop) when a neighbour drops off for a few minutes?
- Do nodes avoid fetching anything at all where no near chain is possible?

It models only the decision being designed. Shard transfer, the registry and
prune are reduced to the inputs that decision reads:
- each node's held parts as gossiped, after a lag;
- in-flight downloads as gossiped;
- each node's ceiling (the parts it can run at once);
- its disk soft limit;
- measured round trips to its peers.
None of the existing replica, carrier or completion rules run here. What
changes is the new term alone.

Policies (`--policy`):
  none      no chain term: chains form only from the initial holdings.
  greedy    every node fetches whatever closes its neighbourhood's gap, up to
            its room. This is the herding strawman.
  greedy+inflight
            greedy, with peers' in-flight downloads counted as held: does
            counting what is on its way already damp the herd without an election?
  carrier   #231's carrier scoped to a latency neighbourhood (the first
            candidate; it cannot finish a chain that needs two machines to move):
            - one machine at a time per (model, neighbourhood), chosen by
              rendezvous hash among those whose plan closes part of the gap;
            - it fetches one contiguous window, at most its ceiling, chosen to
              leave the fewest parts uncovered and then the fewest segments;
            - a lease is renewed by progress, with patience before the
              machine is passed over.
  carrier+inflight
            carrier, with each peer's in-flight downloads counted as held when
            measuring the gap.
  plan      THE RULE STAGE 3 CHOSE: every machine computes ONE chain for its neighbourhood from what it
            sees: fewest segments (within one of the least the ceilings allow),
            then fewest parts fetched, ties broken by a fixed rendezvous order so
            machines seeing the same holdings agree. Each fetches the parts of its
            own segment that it lacks. Prune keeps a machine's planned segment. Only
            for models no near machine can run whole, and a complete chain is
            never rebalanced.

  plan+groups
            plan, over a GROUP every machine derives the same way from the near
            lists machines publish, not over each machine's own neighbourhood, so
            machines whose neighbourhoods only overlap still agree on one plan.

The `flap` scenario runs each policy four ways, for the two fixes of the loop:
  grace30   a disconnected peer's holdings still count for 30 minutes (a blip
            is not a gap);
  hold24h   a copy needed in the last 24 hours is kept (prune held at the
            window's maximum, #876);
  and both, and neither.

Usage:  python3 examples/placement_sim.py [--scenario city,fleet,lonely,line,metro,flap]
            [--policy none,greedy,greedy+inflight,carrier,carrier+inflight,plan,plan+groups]
            [--seeds N] [--radius MS]
A scenario takes seconds per policy and seed; `metro` (60 machines) ~15-35 s.
"""

import argparse
import hashlib
import math
import random
import statistics
import sys

STEP_MIN = 1  # one tick = one minute


def rendezvous(model, node_id):
    h = hashlib.blake2b(f"{model}|{node_id}".encode(), digest_size=8).digest()
    return int.from_bytes(h, "big")


class Node:
    def __init__(self, nid, pos, access_ms, ceiling, disk, holds=(), demand=True):
        self.id = nid
        self.pos = pos  # (x, y) in "ms of one-way fibre"
        self.access_ms = access_ms  # last-mile round trip this node adds
        self.ceiling = ceiling  # parts it can run at once
        self.disk = disk  # soft limit in parts; prune sheds above it
        self.held = set(holds)
        self.fetching = {}  # part -> minute it lands
        self.demand = demand  # someone near this node asks for the model
        self.period = 5
        self.phase = rendezvous("phase", nid) % self.period
        self.online = True
        # Gossip: what peers see, delivered after a lag.
        self.seen_held = set(holds)
        self.seen_fetching = set()
        self.pending = []  # (deliver_at, held, fetching)
        self.offline_since = None
        # Counters
        self.fetched = 0
        self.refetched = 0
        self.shed = 0
        self.ever_held = set(holds)
        self.shed_parts = set()
        # Local state of the election, as each node keeps it
        self.leases = {}  # (model, node) -> (lapses_at, held_count)
        self.passed_over = {}  # (model, node) -> until
        self.needed_until = {}  # part -> minute until which prune keeps it
        self.planned = None  # (start, end) of this node's segment in its neighbourhood plan


def rtt(a, b):
    if a is b:
        return 0.0
    d = math.dist(a.pos, b.pos)
    return 2 * d + a.access_ms + b.access_ms


def reach_of(n, held, ceiling):
    """reach[x] = where a segment this holder starts at part x can end: the end of its contiguous
    run from x, at most `ceiling` parts on. reach[x] == x where it does not hold x."""
    reach = [0] * n
    run_end = n
    for x in range(n - 1, -1, -1):
        if x in held:
            if x + 1 >= n or (x + 1) not in held:
                run_end = x + 1
            reach[x] = min(run_end, x + ceiling)
        else:
            reach[x] = x
    return reach


def cover_reaches(n, reaches):
    """Greedy interval cover of [0, n) with each holder used at most once: at each part, the
    unused holder that reaches furthest. Where nobody holds a part it is counted uncovered and
    skipped. Returns (uncovered_parts, segments, uncovered_positions)."""
    x, segs, gaps = 0, 0, []
    used = [False] * len(reaches)
    while x < n:
        best_end, best_i = x, -1
        for i, r in enumerate(reaches):
            if not used[i] and r[x] > best_end:
                best_end, best_i = r[x], i
        if best_i < 0:
            gaps.append(x)
            x += 1
            continue
        used[best_i] = True
        segs += 1
        x = best_end
    return len(gaps), segs, gaps


def chain_cover(n, holders):
    """`cover_reaches` over (held_set, ceiling) pairs. Returns (uncovered_parts, segments)."""
    unc, segs, _ = cover_reaches(n, [reach_of(n, h, c) for (h, c) in holders])
    return unc, segs


def plan_chain(n, hood, model, max_extra=1):
    """hood: list of (node_id, held_set, ceiling). Returns {node_id: (start, end)} for a chain
    covering [0, n) with each machine used once, or None when the ceilings cannot cover it.
    Objective, in order: fewest segments, then fewest parts to fetch."""
    hood = sorted(hood, key=lambda h: rendezvous(model, h[0]), reverse=True)
    k = len(hood)
    caps = sorted((c for (_, _, c) in hood), reverse=True)
    total, min_segs = 0, None
    for i, c in enumerate(caps):
        total += c
        if total >= n:
            min_segs = i + 1
            break
    if min_segs is None:
        return None
    max_segs = min(k, min_segs + max_extra)
    # missing[i][x] = parts in [0, x) machine i lacks: a segment [x, e) costs missing[i][e] - missing[i][x].
    missing = []
    for (_, held, _) in hood:
        m = [0] * (n + 1)
        for x in range(n):
            m[x + 1] = m[x] + (0 if x in held else 1)
        missing.append(m)
    memo = {}

    def best(x, mask, used):
        if x >= n:
            return (0, 0, ())
        if used >= max_segs:
            return None
        key = (x, mask)
        if key in memo:
            return memo[key]
        out = None
        for i, (nid, held, c) in enumerate(hood):
            if mask >> i & 1:
                continue
            for e in range(min(n, x + c), x, -1):
                rest = best(e, mask | (1 << i), used + 1)
                if rest is None:
                    continue
                cand = (rest[0] + 1, rest[1] + missing[i][e] - missing[i][x], ((nid, x, e),) + rest[2])
                if out is None or cand[:2] < out[:2]:
                    out = cand
        memo[key] = out
        return out

    r = best(0, 0, 0)
    if r is None:
        return None
    return {nid: (s, e) for (nid, s, e) in r[2]}


class Sim:
    def __init__(self, nodes, n_parts, model="M", radius_ms=50.0, policy="carrier",
                 gossip_lag=1, dl_minutes=(2, 6), per_cycle=2, patience=20,
                 passed_over=120, absent_grace=0, hold_hours=0.0, prune=False,
                 flaps=(), seed=1):
        self.nodes = nodes
        self.n = n_parts
        self.model = model
        self.R = radius_ms
        self.policy = policy
        self.lag = gossip_lag
        self.dl = dl_minutes
        self.per_cycle = per_cycle
        self.patience = patience
        self.passed_over_for = passed_over
        self.absent_grace = absent_grace
        self.hold_min = int(hold_hours * 60)
        self.prune = prune
        self.flaps = flaps  # list of (node_id, start_min, every_min, down_min)
        self.rng = random.Random(seed)
        self.t = 0
        self.near = {a.id: [b for b in nodes if b is not a and rtt(a, b) <= self.R] for a in nodes}
        self.first_complete = {}
        self.version = 0
        self.plan_cache = {}
        self.chain_plans = {}

    # ---- views -------------------------------------------------------------
    def visible(self, me, peer):
        if peer.online:
            return True
        return self.absent_grace > 0 and peer.offline_since is not None and \
            self.t - peer.offline_since < self.absent_grace

    def view_holders(self, me, inflight, override=None):
        """(held, ceiling) for me and each near peer, as me sees them.
        `override` maps a node id to an assumed held set."""
        out = []
        for p in [me] + self.near[me.id]:
            if p is not me and not self.visible(me, p):
                continue
            if override and p.id in override:
                h = set(override[p.id])
            elif p is me:
                h = set(me.held) | (set(me.fetching) if inflight else set())
            else:
                h = set(p.seen_held) | (set(p.seen_fetching) if inflight else set())
            out.append((p, h, p.ceiling))
        return out

    def hood_capacity(self, me):
        return me.ceiling + sum(p.ceiling for p in self.near[me.id] if self.visible(me, p))

    def hood_wants(self, me):
        return me.demand or any(p.demand for p in self.near[me.id])

    # ---- the decision being designed ----------------------------------------
    def best_window(self, me, holders, held_now):
        """The parts `me` should fetch to close its neighbourhood's gap: a contiguous window of at
        most me.ceiling parts that leaves the fewest parts uncovered, then needs the fewest parts
        fetched (only the shortfall, as #231's carrier), then the fewest segments, then leftmost.
        Every window is tried, not only those touching a gap: a gap left by a ceiling is closed by
        extending another run. A complete chain is never rebalanced.
        Returns (uncovered_after, segs_after, parts_to_fetch) or None."""
        n = self.n
        others = [reach_of(n, h, c) for (p, h, c) in holders if p is not me]
        base_unc, _, _ = cover_reaches(n, others + [reach_of(n, held_now, me.ceiling)])
        if base_unc == 0:
            return None
        L = min(me.ceiling, n)
        best = None
        for s in range(n):
            for e in range(s + 1, min(n, s + L) + 1):
                w = range(s, e)
                fetch = {x for x in w if x not in held_now}
                if not fetch or len(held_now) + len(fetch) > 3 * me.disk:
                    continue
                unc, segs, _ = cover_reaches(n, others + [reach_of(n, held_now | fetch, me.ceiling)])
                key = (unc, len(fetch), segs, s)
                if best is None or key < best[0]:
                    best = (key, fetch)
        if best is None or best[0][0] >= base_unc:
            return None
        (unc, _, segs, _), fetch = best
        return unc, segs, fetch

    def plan_for(self, viewer, cand, inflight):
        key = (viewer.id, cand.id, inflight, self.version)
        if key not in self.plan_cache:
            if len(self.plan_cache) > 200000:
                self.plan_cache.clear()
            self.plan_cache[key] = self._plan_for(viewer, cand, inflight)
        return self.plan_cache[key]

    def _plan_for(self, viewer, cand, inflight):
        holders = self.view_holders(viewer, inflight)
        # Judge the candidate's window inside the VIEWER's neighbourhood. That is all the viewer
        # can see, and the cause of disagreement in a partial mesh.
        if cand is viewer:
            held_now = set(cand.held) | (set(cand.fetching) if inflight else set())
        else:
            held_now = set(cand.seen_held) | (set(cand.seen_fetching) if inflight else set())
        if all(p is not cand for (p, _, _) in holders):
            holders = holders + [(cand, held_now, cand.ceiling)]
        r = self.best_window(cand, holders, held_now)
        return None if r is None else r[2]

    def elected(self, me, inflight):
        """The near carrier as `me` sees it, keeping `me`'s own lease book."""
        cands = [me] + [p for p in self.near[me.id] if self.visible(me, p)]
        ranked = sorted(cands, key=lambda c: rendezvous(self.model, c.id), reverse=True)
        for c in ranked:
            key = (self.model, c.id)
            if me.passed_over.get(key, -1) > self.t:
                continue
            plan = self.plan_for(me, c, inflight)
            if not plan:
                continue
            held_c = len(c.held) if c is me else len(c.seen_held)
            fetching_c = bool(c.fetching) if c is me else bool(c.seen_fetching)
            lease = me.leases.get(key)
            if lease is None or held_c > lease[1] or fetching_c:
                me.leases[key] = (self.t + self.patience, held_c)
                return c, plan
            if self.t >= lease[0]:
                me.passed_over[key] = self.t + self.passed_over_for
                del me.leases[key]
                continue
            return c, plan
        return None, None

    def decide(self, me):
        if not me.online or not self.hood_wants(me):
            return set()
        if self.hood_capacity(me) < self.n:  # no near chain is possible: fetch nothing
            return set()
        inflight = self.policy.endswith("+inflight")
        if self.policy in ("plan", "plan+groups"):
            # Stage 3 is for models no near machine can run whole; the existing rules handle the rest.
            grouped = self.policy == "plan+groups"
            hood = self.group_holders(me, True) if grouped else self.view_holders(me, True)
            now = self.group_holders(me, False) if grouped else self.view_holders(me, False)
            if any(c >= self.n for (_, _, c) in hood) or sum(c for (_, _, c) in hood) < self.n:
                return set()
            if chain_cover(self.n, [(h, c) for (_, h, c) in now])[0] == 0:
                return set()  # a complete chain is never rebalanced
            plan = self.plan_cached(me, hood)
            if not plan or me.id not in plan:
                return set()
            s, e = plan[me.id]
            me.planned = (s, e)
            return {x for x in range(s, e) if x not in me.held and x not in me.fetching}
        if self.policy.startswith("greedy"):
            holders = self.view_holders(me, inflight)
            r = self.best_window(me, holders, set(me.held) | (set(me.fetching) if inflight else set()))
            return set() if r is None else r[2]
        if self.policy.startswith("carrier"):
            c, plan = self.elected(me, inflight)
            if c is me and plan:
                return plan
        return set()

    def groups(self):
        """Planning groups every machine derives identically from the published near lists: in
        rendezvous order for the model, each machine not yet placed starts a group with every
        unplaced, visible machine on ITS near list. A member is within R of the group's first
        machine, so two members are within about 2R of each other."""
        key = ("groups", self.version)
        if key not in self.plan_cache:
            placed, out = set(), {}
            order = sorted(self.nodes, key=lambda n: rendezvous(self.model, n.id), reverse=True)
            for u in order:
                if u.id in placed or not u.online:
                    continue
                members = [u] + [v for v in self.near[u.id]
                                 if v.id not in placed and self.visible(u, v)]
                for v in members:
                    placed.add(v.id)
                    out[v.id] = members
            self.plan_cache[key] = out
        return self.plan_cache[key]

    def group_holders(self, me, inflight):
        out = []
        for p in self.groups().get(me.id, [me]):
            if p is me:
                h = set(me.held) | (set(me.fetching) if inflight else set())
            else:
                h = set(p.seen_held) | (set(p.seen_fetching) if inflight else set())
            out.append((p, h, p.ceiling))
        return out

    def plan_cached(self, me, hood):
        key = tuple(sorted((p.id, frozenset(h), c) for (p, h, c) in hood))
        if key not in self.chain_plans:
            if len(self.chain_plans) > 50000:
                self.chain_plans.clear()
            self.chain_plans[key] = plan_chain(
                self.n, [(p.id, h, c) for (p, h, c) in hood], self.model)
        return self.chain_plans[key]

    def keep(self, me, part):
        """Prune's question, the same predicate as the fetch side. Without my copy, would the
        near chain leave a part uncovered — or, under `plan`, is it inside my planned segment?"""
        if self.hold_min and me.needed_until.get(part, -1) > self.t:
            return True
        if self.policy.startswith("plan") and me.planned and me.planned[0] <= part < me.planned[1]:
            if self.hold_min:
                me.needed_until[part] = self.t + self.hold_min
            return True
        holders = self.view_holders(me, False)
        with_unc, _ = chain_cover(self.n, [(h, c) for (_, h, c) in holders])
        without = [(h - {part} if p is me else h, c) for (p, h, c) in holders]
        wo_unc, _ = chain_cover(self.n, without)
        needed = wo_unc > with_unc
        if needed and self.hold_min:
            me.needed_until[part] = self.t + self.hold_min
        return needed

    # ---- world -------------------------------------------------------------
    def announce(self, n):
        n.pending.append((self.t + self.lag, set(n.held), set(n.fetching)))

    def step(self):
        t = self.t
        for (nid, start, every, down) in self.flaps:
            n = self.by_id[nid]
            if t >= start and (t - start) % every == 0:
                n.online, n.offline_since = False, t
                self.version += 1
            if t >= start and (t - start) % every == down:
                n.online, n.offline_since = True, None
                self.version += 1
            if self.absent_grace and n.offline_since is not None and \
                    t - n.offline_since == self.absent_grace:
                self.version += 1
        for n in self.nodes:
            landed = [p for p, eta in n.fetching.items() if eta <= t]
            for p in landed:
                del n.fetching[p]
                if p in n.shed_parts:
                    n.refetched += 1
                n.held.add(p)
                n.ever_held.add(p)
                n.fetched += 1
            if landed:
                self.announce(n)
                self.version += 1
            while n.pending and n.pending[0][0] <= t:
                _, h, f = n.pending.pop(0)
                n.seen_held, n.seen_fetching = h, f
                self.version += 1
        for n in self.nodes:
            if not n.online or (t - n.phase) % n.period != 0:
                continue
            plan = self.decide(n)
            started = 0
            for p in sorted(plan):
                if started >= self.per_cycle:
                    break
                if p in n.held or p in n.fetching:
                    continue
                n.fetching[p] = t + self.rng.randint(*self.dl)
                started += 1
            if started:
                self.announce(n)
                self.version += 1
            if self.prune and len(n.held) > n.disk:
                for p in sorted(n.held, key=lambda q: rendezvous("shed", f"{n.id}{q}")):
                    if len(n.held) <= n.disk:
                        break
                    if not self.keep(n, p):
                        n.held.discard(p)
                        n.shed += 1
                        n.shed_parts.add(p)
                        self.announce(n)
                        self.version += 1
        # Completeness, from TRUE state: online near holders and what they actually hold.
        for n in self.nodes:
            if t % 5 or n.id in self.first_complete or not n.online:
                continue
            hs = [(n.held, n.ceiling)] + [(p.held, p.ceiling) for p in self.near[n.id] if p.online]
            if chain_cover(self.n, hs)[0] == 0:
                self.first_complete[n.id] = t
        self.t += STEP_MIN

    def run(self, minutes):
        self.by_id = {n.id: n for n in self.nodes}
        for _ in range(minutes):
            self.step()
        return self.report()

    def report(self):
        feasible, complete, segs = [], [], []
        for n in self.nodes:
            if not n.demand:
                continue
            cap = n.ceiling + sum(p.ceiling for p in self.near[n.id])
            if cap < self.n:
                continue
            feasible.append(n.id)
            hs = [(n.held, n.ceiling)] + [(p.held, p.ceiling) for p in self.near[n.id]]
            unc, s = chain_cover(self.n, hs)
            if unc == 0:
                complete.append(n.id)
                segs.append(s)
        infeasible_fetch = sum(
            n.fetched for n in self.nodes
            if n.ceiling + sum(p.ceiling for p in self.near[n.id]) < self.n)
        times = [self.first_complete[i] for i in feasible if i in self.first_complete]
        groups = self.groups()
        g_complete, widths = 0, []
        for gid in {id(g): g for g in groups.values()}.values():
            if sum(m.ceiling for m in gid) < self.n or not any(m.demand for m in gid):
                continue
            if chain_cover(self.n, [(m.held, m.ceiling) for m in gid])[0] == 0:
                g_complete += 1
            widths.append(max((rtt(a, b) for a in gid for b in gid), default=0.0))
        return {
            "feasible": len(feasible),
            "complete": len(complete),
            "t_all_complete": max(times) if len(times) == len(feasible) and times else None,
            "segments_mean": round(statistics.mean(segs), 2) if segs else None,
            "fetched": sum(n.fetched for n in self.nodes),
            "refetched": sum(n.refetched for n in self.nodes),
            "shed": sum(n.shed for n in self.nodes),
            "fetched_where_infeasible": infeasible_fetch,
            "groups_complete": f"{g_complete}/{len(widths)}",
            "group_width_ms": round(max(widths), 1) if widths else None,
        }


# ---- scenarios ------------------------------------------------------------
def city(prefix, cx, cy, k, rng, ceilings, n_parts, start_parts, demand=True, spread=4.0):
    out = []
    for i in range(k):
        pos = (cx + rng.uniform(-spread, spread), cy + rng.uniform(-spread, spread))
        holds = rng.sample(range(n_parts), rng.randint(*start_parts)) if start_parts[1] else []
        c = ceilings[i % len(ceilings)]
        out.append(Node(f"{prefix}{i}", pos, rng.uniform(2, 12), c, c + 2, holds, demand))
    return out


def scenario(name, policy, seed, **kw):
    rng = random.Random(seed)
    n_parts = 16
    flaps = ()
    sim_kw = {}
    if name == "city":
        # Six machines in one metro area (RTT ~8-35 ms), each able to run 3-8 of 16 parts,
        # holding 0-3 random parts from ordinary replication. Six far machines hold the rest.
        nodes = city("c", 0, 0, 6, rng, [4, 6, 8, 3, 5, 4], n_parts, (0, 3))
        far = [Node(f"f{i}", (rng.uniform(150, 400), rng.uniform(-200, 200)), 10, 8, 18,
                    range(16), demand=False) for i in range(6)]
        nodes += far
    elif name == "fleet":
        # Today's fleet: every machine >= 200 ms from every other, except one LAN pair.
        n_parts = 16
        nodes = [Node("us", (0, 0), 1, 7, 12, [0, 1, 2, 3]),
                 Node("lan", (0.2, 0), 1, 10, 30, [5, 6]),
                 Node("au", (100, 30), 8, 8, 16, range(16), demand=False)]
        for i in range(4):
            nodes.append(Node(f"far{i}", (rng.uniform(-300, 300), rng.uniform(100, 300)), 10,
                              rng.choice([4, 6, 12]), 20, rng.sample(range(16), 8), demand=False))
    elif name == "lonely":
        # Machines too few or too small for a near chain: nothing must be fetched.
        nodes = city("s", 0, 0, 2, rng, [4, 5], n_parts, (0, 3))
        nodes += [Node(f"f{i}", (300 + 50 * i, 0), 10, 8, 18, range(16), demand=False)
                  for i in range(4)]
    elif name == "line":
        # A chain of towns 20 ms apart: neighbourhoods overlap without being transitive.
        nodes = [Node(f"l{i}", (10 * i, 0), 3, 5, 7,
                      rng.sample(range(n_parts), rng.randint(0, 2))) for i in range(10)]
    elif name == "metro":
        # 60 machines in 6 cities, one 34-part model, demand in four of the cities.
        n_parts = 34
        nodes = []
        for ci, (cx, cy) in enumerate([(0, 0), (120, 10), (-90, 140), (300, -50),
                                       (40, 260), (-250, -120)]):
            nodes += city(f"m{ci}_", cx, cy, 10, rng, [4, 6, 8, 3, 10, 5, 4, 6, 2, 8], n_parts,
                          (0, 4), demand=ci < 4)
    elif name == "flap":
        # The city, with one neighbour holding 6 parts that drops off for 15 min every 3 h, and
        # prune on. This is #876's shape: a moving input read by fetch and by prune.
        nodes = [Node("a", (0, 0), 4, 8, 6, [8, 9, 10, 11]),
                 Node("b", (3, 1), 4, 6, 6, [0, 1, 2, 3, 4, 5]),
                 Node("c", (1, 3), 4, 6, 6, [12, 13, 14, 15, 6, 7]),
                 Node("d", (2, 2), 4, 4, 4, [])]
        nodes += [Node(f"f{i}", (300 + 20 * i, 0), 10, 8, 18, range(16), demand=False)
                  for i in range(4)]
        flaps = (("b", 60, 180, 15),)
        sim_kw["prune"] = True
    else:
        raise SystemExit(f"unknown scenario {name}")
    sim_kw.update(kw)
    sim = Sim(nodes, n_parts, policy=policy, flaps=flaps, seed=seed, **sim_kw)
    return sim.run(48 * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all")
    ap.add_argument("--policy", default="all")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--radius", type=float, default=50.0)
    args = ap.parse_args()
    scenarios = ["city", "fleet", "lonely", "line", "metro", "flap"] \
        if args.scenario == "all" else args.scenario.split(",")
    policies = ["none", "greedy", "greedy+inflight", "carrier", "carrier+inflight", "plan",
                "plan+groups"] \
        if args.policy == "all" else args.policy.split(",")
    variants = [("", {})]
    for sc in scenarios:
        vs = variants
        if sc == "flap":
            vs = [("", {}), (" grace30", {"absent_grace": 30}),
                  (" hold24h", {"hold_hours": 24}),
                  (" grace30+hold24h", {"absent_grace": 30, "hold_hours": 24})]
        for pol in policies:
            for vname, vkw in vs:
                rows = [scenario(sc, pol, s, radius_ms=args.radius, **vkw)
                        for s in range(1, args.seeds + 1)]
                agg = {}
                for k in rows[0]:
                    vals = [r[k] for r in rows if r[k] is not None]
                    if k == "t_all_complete":
                        done = len(vals)
                        agg[k] = (f"{statistics.median(vals):.0f}min ({done}/{len(rows)} runs)"
                                  if vals else f"never (0/{len(rows)})")
                    elif vals and isinstance(vals[0], str):
                        agg[k] = ",".join(vals)
                    elif vals:
                        agg[k] = round(statistics.mean(vals), 2)
                    else:
                        agg[k] = None
                print(f"{sc:7s} {pol + vname:28s} " +
                      " ".join(f"{k}={v}" for k, v in agg.items()))
                sys.stdout.flush()


if __name__ == "__main__":
    main()
