# standing_ring_sim.py -- the standing ring design's numbers (docs/plans/standing_ring.md).  python3 -I examples/standing_ring_sim.py
# Law used everywhere:  time a token = d + m(W, n) * L
#   d = ms the drafter needs a level, W = rows a level, L = ms from drafting a level to its verdict,
#   n = L/d levels in flight, m = chance a level loses the target's token from the beam (a miss voids n levels).
# m(W, n) is REPLAYED on the real per-position records in ~/swarmllm-ref/spec/gumbeldump_*.npz (no numpy here).
import zipfile, struct, math, os, sys
SPEC = os.path.expanduser('~/swarmllm-ref/spec/')

def npy(z, name):
    b = z.read(name + '.npy'); hl = struct.unpack('<H', b[8:10])[0]; hdr = eval(b[10:10 + hl].decode())
    shape = hdr['shape']; n = 1
    for s in shape: n *= s
    flat = struct.unpack('<%d%s' % (n, 'q' if 'i8' in hdr['descr'] else 'd'), b[10 + hl:10 + hl + 8 * n])
    if len(shape) == 1: return list(flat)
    return [flat[i * shape[1]:(i + 1) * shape[1]] for i in range(shape[0])]

def load(label, T=0.7, C=16):
    """per position: (reply, rank of the target's draw in the drafter's CUR order or None, child log-probs, CUR token ids)"""
    z = zipfile.ZipFile(SPEC + 'gumbeldump_%s.npz' % label)
    rep, ids, y, g, d, ns = (npy(z, k) for k in ('reply', 'ids', 'y', 'g', 'd', 'nsurv'))
    out = []
    for j in range(len(rep)):
        order = sorted(range(len(ids[j])), key=lambda i: -y[j][i])
        surv = order[:ns[j]]
        c = max(surv, key=lambda i: y[j][i] / T + g[j][i])            # the drafter's own coupled draw goes first
        cur = [c] + [i for i in order if i != c]
        cur = cur[:C]
        toks = [ids[j][i] for i in cur]
        lps = [y[j][order[0]]] + [y[j][i] for i in cur[1:]]            # the coupled draw is rated as the likeliest
        rk = toks.index(d[j]) if d[j] in toks else None
        out.append((rep[j], rk, lps, toks, d[j]))
    return out

def replay(pos, W, n, C=8):
    """beam of W rows a level over the REAL rank sequence; verdicts n levels old prune dead paths.  -> misses a token"""
    miss = tok = 0; beam = [(0.0, None)]; lvl = 0; last = None
    for rep, rk, lps, _t, _d in pos:
        if rep != last: beam = [(0.0, None)]; lvl = 0; last = rep
        lvl += 1
        beam = [x for x in beam if x[1] is None or x[1] > lvl - n]
        ch = []
        for lp, dv in beam:
            for i in range(min(C, len(lps))):
                ch.append((lp + lps[i], None if (dv is None and i == rk) else (dv if dv is not None else lvl)))
        ch.sort(key=lambda x: -x[0]); beam = ch[:W]; tok += 1
        if not any(x[1] is None for x in beam): miss += 1; beam = [(0.0, None)]; lvl = 0
    return miss / tok

WS = (1, 2, 4, 8, 16, 32, 64, 128); NS = (2, 4, 8, 16, 32, 64)
def table(pos): return {(W, n): replay(pos, W, n) for W in WS for n in NS}
def interp(tab, W, n):
    W = min(max(W, 1), 128); n = min(max(n, 2), 64)
    def br(grid, v):
        for a, b in zip(grid, grid[1:]):
            if v <= b: return a, b, (math.log(v) - math.log(a)) / (math.log(b) - math.log(a))
        return grid[-2], grid[-1], 1.0
    w0, w1, fw = br(WS, W); n0, n1, fn = br(NS, n)
    f = lambda w: tab[(w, n0)] * (1 - fn) + tab[(w, n1)] * fn
    return f(w0) * (1 - fw) + f(w1) * fw

P15 = load('c15-T07')
PROSE = [p for p in P15 if p[0] != 1]; CODE = [p for p in P15 if p[0] == 1]      # prompt 1 is the Python function
h = [sum(1 for p in P15 if p[1] == r) / len(P15) for r in range(8)]
print("A. real records: %d positions (prose %d, code %d); rank histogram %s  (file's own 'cur': 0.733 0.107 0.046 0.025)"
      % (len(P15), len(PROSE), len(CODE), ' '.join('%.3f' % x for x in h[:4])))
print("   top-1 prose %.3f  code %.3f" % (sum(p[1] == 0 for p in PROSE) / len(PROSE), sum(p[1] == 0 for p in CODE) / len(CODE)))
MP, MC = table(PROSE), table(CODE)
print("B. misses a token m(W rows a level, n levels in flight), REAL sequences  [prose | code]")
for n in NS:
    print("   n=%2d  " % n + '  '.join('W%d: %.3f|%.3f' % (W, MP[(W, n)], MC[(W, n)]) for W in (2, 4, 8, 16, 32, 128)))

# ---- C. guild check: do two drafters' guesses cover more than one drafter's, row for row? ------------------
try:
    P05, P3 = load('c05-T07'), load('c3-T07')
    same = all(a[4] == b[4] == c[4] for a, b, c in zip(P05, P15, P3)) and len(P05) == len(P15) == len(P3)
    def cover(k, *sets):
        hit = 0
        for row in zip(*sets):
            u = []
            for i in range(k):
                for p in row:
                    if i < len(p[3]) and p[3][i] not in u: u.append(p[3][i])
            hit += row[0][4] in u[:k]
        return hit / len(sets[0])
    print("C. guild (same replies in all three dumps: %s). coverage of the target's draw with k rows:" % same)
    for k in (2, 4, 8, 16):
        print("   k=%2d  1.5B alone %.3f | 3B alone %.3f | 1.5B+0.5B interleaved %.3f | 1.5B+3B %.3f | all three %.3f"
              % (k, cover(k, P15), cover(k, P3), cover(k, P15, P05), cover(k, P15, P3), cover(k, P15, P3, P05)))
except Exception as e:
    print("C. guild check skipped:", e)

# ---- D. calibration on the MEASURED 14B pair at 123-145 ms (docs/invariants/scheduling.md, 2026-11-15) -----
# arm: (rows a level, levels a call, ms a forward here = drafting + layers, far end ms, prose tok/s, code tok/s, tokens a forward prose, code)
ARMS = {'8x3 (default)': (8, 3, 14.5 + 26.5, 30.5, (17.61, 19.93), (40.94, 45.43), (0.72, 0.82), (1.65, 1.82)),
        '4 rows':        (4, 3, 12.5 + 23.5, 24.0, (15.06, 15.33), (40.48, 42.43), (0.56, 0.57), (1.51, 1.52)),
        '16 rows':       (16, 3, 24.0 + 34.0, 46.5, (15.72, 17.89), (33.57, 34.72), (1.02, 1.13), (1.95, 2.00)),
        '4 levels':      (8, 4, 19.0 + 29.5, 36.5, (16.85, 19.38), (40.23, 41.53), (0.88, 1.00), (1.98, 2.03)),
        '2 levels':      (8, 2, 10.0 + 24.0, 25.5, (18.45, 19.43), (37.45, 38.93), (0.65, 0.68), (1.25, 1.29))}
def today(arm, Lw, tab):
    W, lv, fwd, far, *_ = ARMS[arm]
    dl = fwd / lv; L = Lw + (fwd - lv * 4.7) + far            # wire part + this node's layers + the far end's forward
    dl = max(dl, L / (128.0 / W))                             # the 128 live-row budget: the flow waits past it
    m = interp(tab, W, L / dl); tpf = lv / (1 + m * L / dl)
    return 1000.0 / (dl * (1 + m * L / dl)), tpf, L, m
lo, hi = 50.0, 900.0                                          # ONE fitted value: the wire part of the verdict's delay
for _ in range(40):
    mid = (lo + hi) / 2
    if today('8x3 (default)', mid, MP)[0] > 18.8: lo = mid
    else: hi = mid
LW = (lo + hi) / 2
print("D. calibration: one fitted value, the wire part of a verdict's delay = %.0f ms (loop 123-145 + sending + waiting);" % LW)
print("   fitted on the default arm's prose only; every other cell is a prediction")
errs = []
for arm, a in ARMS.items():
    p = today(arm, LW, MP); c = today(arm, LW, MC)
    mp, mc = sum(a[4]) / 2, sum(a[5]) / 2
    if arm != '8x3 (default)': errs.append(p[0] / mp - 1)
    errs.append(c[0] / mc - 1)
    print("   %-14s prose %5.1f (measured %.1f-%.1f, %+3.0f%%)  code %5.1f (measured %.1f-%.1f, %+3.0f%%)  tokens a forward %.2f / %.2f (measured %.2f-%.2f / %.2f-%.2f)  L %.0f ms, m %.3f / %.3f"
          % (arm, p[0], a[4][0], a[4][1], 100 * (p[0] / mp - 1), c[0], a[5][0], a[5][1], 100 * (c[0] / mc - 1), p[1], c[1], a[6][0], a[6][1], a[7][0], a[7][1], p[2], p[3], c[3]))
print("   mean |error| over the 9 predicted cells: %.0f%%" % (100 * sum(abs(e) for e in errs) / len(errs)))
p = today('8x3 (default)', LW, MP)
print("   where today's %.1f ms a token goes: drafting+layers in series %.1f ms, voided levels (m*L) %.1f ms" % (1000 / p[0], 41 / 3, 1000 / p[0] - 41 / 3))

# ---- E. the fused design: a standing ring -------------------------------------------------------------------
# model: (GB read a pass, hidden dims, L1 prose, L1 code, L2 tok/s)       GB: 4-bit files; MoE = bytes of the experts a token uses
MODELS = {'14B dense':  dict(gb=9.0, dim=5120, L1=(11.2, 14.3), L2=65, hold=9.0),
          '27B hybrid': dict(gb=16.5, dim=5120, L1=(5.3, 5.8), L2=40, hold=16.5),
          '35B MoE':    dict(gb=2.0, dim=2048, L1=(35.5, 35.5), L2=100, hold=20.0),
          '70B dense':  dict(gb=42.0, dim=8192, L1=(None, None), L2=17.5, hold=42.0)}
# swarm: card memory bandwidth GB/s, GB a member lends a ring, uplink Mbit/s, lanes (rings sharing the beam), drafter ms a level
SWARMS = {'30':   dict(bw=448, lend=7.0, up=40, lanes=1, d=4.7),
          '300':  dict(bw=700, lend=12.0, up=80, lanes=4, d=4.7),
          '3000': dict(bw=1000, lend=20.0, up=150, lanes=16, d=3.5)}
ROWKB = 120.0 / 23                                            # MEASURED: ~120 KB a forward of 23 rows at 5120 dims
def ring(model, sw, loop, busy=False, tab=MP, d=None, mscale=1.0):
    M, S = MODELS[model], SWARMS[sw]
    K = max(2, math.ceil(M['hold'] / S['lend']))             # fewest stages that hold it; never one (no whole model)
    u = 4 if busy else 1                                      # conversations sharing every stage and uplink when busy
    lanes = 1 if busy else S['lanes']; d0 = (d or S['d']) * (2 if busy else 1)
    fixed = M['gb'] / K / (S['bw'] / 1000.0) + 7.5 + 3.6 * (u - 1)   # ms a stage forward: weights once + submissions (+3.6 a merged user, MEASURED)
    rowms = (0.15 * (M['gb'] / K) / 4.5 if model != '35B MoE' else 0.05) * 448.0 / S['bw']   # ms a row a stage: MEASURED 0.15 on a 4.5 GB slice of a 448 GB/s card
    kbit = ROWKB * 8 * M['dim'] / 5120.0
    best = None
    for dd in [d0 * (1 + 0.1 * i) for i in range(0, 40)]:      # the drafter may pace itself slower to fit the uplink and the stages
        cap = min(8.0, (S['up'] / u) * dd / kbit)              # rows a level one lane's uplink carries (8 = the one-kernel level)
        for Wl in (1, 2, 3, 4, 6, 8):
            if Wl > cap or rowms * Wl / dd > 0.6: continue     # a stage must keep up with the rows drafted while it works
            B = fixed / (1 - rowms * Wl / dd)                  # back-to-back forwards, each carrying every level drafted meanwhile
            L = loop * K / 2.0 + K * 1.5 * B + 5               # K one-way legs (a "loop" is two) + a forward + half a forward queued, a stage
            m = interp(tab, Wl * lanes, L / dd) * mscale
            t = dd + m * L
            if best is None or t < best[0]: best = (t, K, Wl * lanes, L, m, dd, K * fixed)
    return best
def rounds27(sw, loop, busy=False, one=False):
    """Qwen3.8-27B as shipped: recurrent, one check in flight, the file's own guess head (2.6 tokens a round MEASURED)."""
    S = SWARMS[sw]; K = max(2, math.ceil(16.5 / S['lend'])); u = 4 if busy else 1
    work = 16.5 / (S['bw'] / 1000.0) + K * (7.5 + 3.6 * (u - 1)) + 8      # every layer once + the guess head
    worlds = 1 if (busy or one) else S['lanes']
    E = 2.6 + 0.45 * math.log2(worlds)                         # ASSUMED: +0.45 tokens a doubling of linear worlds (the 1.5B's tree slope x 0.8)
    return 1000.0 * E / (loop * K / 2.0 + work), K, E
LOOPS = (20, 60, 130, 290)
def rat(v, M, kind=0):
    l1 = M['L1'][kind]
    return "%sx L1, %.1fx L2" % ('%.1f' % (v / l1) if l1 else 'no', v / M['L2'])
print("E. one reply on a standing ring, tok/s, at ring loops 20 / 60 / 130 / 290 ms (a loop = two one-way legs; a K-stage ring pays K legs)")
for model, M in MODELS.items():
    print("  %s   L1 %s  L2 ~%s (ASSUMED)" % (model, M['L1'][0] or 'none', M['L2']))
    for sw in SWARMS:
        for busy in (False, True):
            rows = [ring(model, sw, P, busy) for P in LOOPS]; crow = [ring(model, sw, P, busy, tab=MC) for P in LOOPS]
            r = rows[0]
            print("    swarm %-4s %-4s K=%d W=%-4.0f L=%3.0f ms m=%.3f | prose %s | code %s | at 20 ms: prose %s; code %s" % (
                sw, 'busy' if busy else 'idle', r[1], r[2], r[3], r[4],
                ' / '.join('%3.0f' % (1000 / x[0]) for x in rows), ' / '.join('%3.0f' % (1000 / x[0]) for x in crow),
                rat(1000 / rows[0][0], M), rat(1000 / crow[0][0], M, 1)))
            if model == '27B hybrid':
                rr = [rounds27(sw, P, busy) for P in LOOPS]
                r1 = [rounds27(sw, P, busy, True) for P in LOOPS]
                print("         as SHIPPED, rounds + guess head: one world (E 2.6 MEASURED) %s | %d worlds (E %.1f ASSUMED) %s | at 20 ms: %s ; %s" % (
                    ' / '.join('%3.0f' % x[0] for x in r1), 1 if busy else SWARMS[sw]['lanes'], rr[0][2], ' / '.join('%3.0f' % x[0] for x in rr), rat(r1[0][0], M), rat(rr[0][0], M)))
print("   (the 27B's flow rows need a tree drafter as good as the 14B's AND a recurrent state carried a row: both ASSUMED)")
print("F. today's computers: the 14B pair with the drafter free-running on the far card (MEASURED 4.7 ms a level, 40 Mbit/s ASSUMED), if the pair's loop were 20 / 60 / 130 / 290 ms:")
for nm, tab in (('prose', MP), ('code', MC)):
    o = []
    for P in LOOPS:
        best = None
        for dd in [4.7 * (1 + 0.1 * i) for i in range(40)]:
            for Wl in (1, 2, 3, 4, 6, 8):
                if Wl > 40 * dd / (ROWKB * 8) or 0.17 * Wl / dd > 0.6: continue
                L = P + (LW - 130) + (22.5 + 26.5) * 1.5 / (1 - 0.17 * Wl / dd)
                t = dd + interp(tab, Wl, L / dd) * L
                if best is None or t < best[0]: best = (t, Wl, L, dd)
        o.append('%.0f' % (1000 / best[0]))
    print("   %s %s tok/s; as shipped at 130 ms the model reads %.1f" % (nm, ' / '.join(o), today('8x3 (default)', LW, tab)[0]))
print("   27B, 35B and 70B today: layers on processors bind; MEASURED 5.0-7.2, 7.8-16 and 1.5-3.0 tok/s stand until every slice is on a card")
import heapq
def D(N, Q=h + [0.0]):
    hp = [(-q, 1) for q in Q if q > 0]; heapq.heapify(hp); tot = 0.0
    for _ in range(N):
        pp, _x = heapq.heappop(hp); pp = -pp; tot += pp
        for q in Q:
            if q > 0: heapq.heappush(hp, (-(pp * q), 0))
    return tot + 1
print("   check against the tree law D(N)/L (no lossless shape may commit more a traversal than the best tree of the rows in flight):")
for model in MODELS:
    r = ring(model, '3000', 20); N = int(r[2] * r[3] / r[5])
    print("     %-11s commits %.1f tokens a traversal from %d rows in flight; D(%d) = %.1f" % (model, r[3] / r[0], N, N, D(N)))
print("G. what a better guesser is worth (m x 0.5 at every width = the feature-fed block head's ASSUMED effect), swarm 3000 idle, 20 ms:")
for model in MODELS:
    a = ring(model, '3000', 20); b = ring(model, '3000', 20, mscale=0.5)
    print("   %-11s %3.0f -> %3.0f tok/s (%s)" % (model, 1000 / a[0], 1000 / b[0], rat(1000 / b[0], MODELS[model])))
print("H. the identity: swarm/local on equal cards = T/(d + m(T+wire)); d 4.7, m at W 8:")
for T in (17, 40, 70, 155, 400):
    L = T * 1.5 + 25; print("   T %3d ms: %.1fx" % (T, T / (4.7 + interp(MP, 8, L / 4.7) * L)), end='')
print()
