# standing_ring_check.py -- the judge's independent re-check of standing_ring_sim.py (docs/plans/standing_ring.md).  python3 -I examples/standing_ring_check.py
import io, sys, math, contextlib, os
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'standing_ring_sim.py')).read()
g = {}
with contextlib.redirect_stdout(io.StringIO()): exec(compile(src, 'standing_ring_sim.py', 'exec'), g)
PROSE, CODE, MP, MC, interp, today, LW, replay = (g[k] for k in ('PROSE', 'CODE', 'MP', 'MC', 'interp', 'today', 'LW', 'replay'))

print("1. OUT-OF-SAMPLE checks of the law  time a token = d + m*L  (nothing below was fitted)")
# (a) slope of ms-a-token against the loop should equal m at 8 rows, deep flight (replay: %.3f)
print("   replayed m(W 8, deep) prose = %.3f, code = %.3f" % (MP[(8, 16)], MC[(8, 16)]))
rig = {100: (19.30 + 19.86) / 2, 80: (19.89 + 21.86) / 2, 60: (21.82 + 24.23) / 2}
print("   one-card rig (MEASURED 11-15, flow, prose): ms a token %s -> slope %.3f ms per ms of loop" % (
    ', '.join('%d ms: %.1f' % (k, 1000 / v) for k, v in rig.items()), (1000 / rig[100] - 1000 / rig[60]) / 40))
print("   pair 130 ms -> US card 308 ms (MEASURED): prose (%.1f -> %.1f ms a token) slope %.3f ; code (%.1f -> %.1f) slope %.3f" % (
    1000 / 18.8, 1000 / 11.75, (1000 / 11.75 - 1000 / 18.8) / 178, 1000 / 42, 1000 / 27, (1000 / 27 - 1000 / 42) / 178))
# (b) the 308 ms run predicted by the calibrated model (wire part shifted by the loop difference)
for nm, tab, meas in (('prose', MP, 11.75), ('code', MC, 27.0)):
    p = today('8x3 (default)', LW + 308 - 130, tab)
    print("   308 ms %s: model %.1f, measured %.1f (%+.0f%%); the repo's tau/e0+m*P model read %s" % (nm, p[0], meas, 100 * (p[0] / meas - 1), '14.8' if nm == 'prose' else '39.9'))
p0 = 1000 / 18.8 - MP[(8, 16)] * 130
print("   today's pair at a ZERO loop by this law: %.1f ms a token = %.0f tok/s (d 13.7 + m x ~92 ms of stage work and queues)" % (p0, 1000 / p0))

# 2. lanes as a PARTITION (no row migrates between lanes; a dead lane adopts the best dropped candidates)
def replay_lanes(pos, J, Wl, n, C=8):
    miss = tok = 0; last = None
    def fresh(): return [[(0.0, None)]] + [[] for _ in range(J - 1)]
    lanes = fresh(); lvl = 0
    for rep, rk, lps, _t, _d in pos:
        if rep != last: lanes = fresh(); lvl = 0; last = rep
        lvl += 1; pool = []
        for j in range(J):
            b = [x for x in lanes[j] if x[1] is None or x[1] > lvl - n]
            ch = []
            for lp, dv in b:
                for i in range(min(C, len(lps))):
                    ch.append((lp + lps[i], None if (dv is None and i == rk) else (dv if dv is not None else lvl)))
            ch.sort(key=lambda x: -x[0]); lanes[j] = ch[:Wl]; pool += ch[Wl:]
        pool.sort(key=lambda x: -x[0])
        for j in range(J):
            if not lanes[j] and pool: lanes[j] = pool[:Wl]; pool = pool[Wl:]
        tok += 1
        if not any(x[1] is None for l in lanes for x in l): miss += 1; lanes = fresh(); lvl = 0
    return miss / tok
print("2. lanes as a partition vs one global beam of the same rows (prose m):")
LT = {}
for J in (1, 4, 16):
    for Wl in (1, 2, 3, 4, 6, 8):
        for n in (2, 4, 8, 16, 32, 64):
            LT[(J, Wl, n)] = replay_lanes(PROSE, J, Wl, n)
LTC = {(J, Wl, n): replay_lanes(CODE, J, Wl, n) for J in (1, 4, 16) for Wl in (1, 2, 3, 4, 6, 8) for n in (2, 4, 8, 16, 32, 64)}
for n in (4, 8, 16, 32):
    print("   n=%2d  1 lane x8: %.3f | 4 lanes x8: %.3f (global W32 %.3f) | 16 lanes x8: %.3f (global W128 %.3f)" % (
        n, LT[(1, 8, n)], LT[(4, 8, n)], MP[(32, n)], LT[(16, 8, n)], MP[(128, n)]))
def mlane(tabl, J, Wl, n):
    NS = (2, 4, 8, 16, 32, 64); n = min(max(n, 2), 64)
    for a, b in zip(NS, NS[1:]):
        if n <= b:
            f = (math.log(n) - math.log(a)) / (math.log(b) - math.log(a)); return tabl[(J, Wl, a)] * (1 - f) + tabl[(J, Wl, b)] * f
    return tabl[(J, Wl, 64)]

# 3. the ring, corrected: uplink never above 100 Mbit/s (the stated home range), lanes priced as a partition,
#    an MoE row priced by the experts it reads, card class shown separately from swarm size
MODELS = {'14B':  dict(gb=9.0, dim=5120, L1=(11.2, 14.3), L2=65, hold=9.0, moe=False),
          '27B':  dict(gb=16.5, dim=5120, L1=(5.3, 5.8), L2=40, hold=16.5, moe=False),
          '35B MoE': dict(gb=2.0, dim=2048, L1=(35.5, 35.5), L2=100, hold=20.0, moe=True),
          '70B':  dict(gb=42.0, dim=8192, L1=(None, None), L2=17.5, hold=42.0, moe=False)}
SW = {'ring of 8-12 GB cards (30)':  dict(bw=448, lend=7.0, up=40, lanes=1, d=4.7),
      'mixed cards, 4 lanes (300)':  dict(bw=700, lend=12.0, up=80, lanes=4, d=4.7),
      '24 GB cards, 16 lanes (3000)': dict(bw=1000, lend=20.0, up=100, lanes=16, d=3.5),
      '8-12 GB cards, 16 lanes (3000, small cards)': dict(bw=448, lend=7.0, up=100, lanes=16, d=4.7),
      '24 GB cards, 1 lane': dict(bw=1000, lend=20.0, up=100, lanes=1, d=3.5)}
ROWKB = 120.0 / 23
def ring(model, S, loop, busy=False, code=False, mscale=1.0, K=None, rowkb=ROWKB, moe_opt=False):
    M = MODELS[model]; tabl = LTC if code else LT
    K = K or max(2, math.ceil(M['hold'] / S['lend'])); u = 4 if busy else 1
    lanes = 1 if busy else S['lanes']; d0 = S['d'] * (2 if busy else 1)
    kbit = rowkb * 8 * M['dim'] / 5120.0; best = None
    for dd in [d0 * (1 + 0.1 * i) for i in range(40)]:
        cap = min(8.0, (S['up'] / u) * dd / kbit)
        for Wl in (1, 2, 3, 4, 6, 8):
            if Wl > cap: continue
            if M['moe'] and not moe_opt:      # a forward reads the union of the experts its rows choose (8 of 128 a row, ASSUMED independent)
                B = 12.0
                for _ in range(50):
                    r = max(1.0, Wl * B / dd); frac = 0.04 + 0.96 * (1 - (1 - 1 / 16.0) ** r)
                    B = 7.5 + 3.6 * (u - 1) + M['hold'] * frac / K / (S['bw'] / 1000.0)
                if Wl * B / dd > 128: continue
            else:
                fixed = M['gb'] / K / (S['bw'] / 1000.0) + 7.5 + 3.6 * (u - 1)
                rowms = (0.05 if M['moe'] else 0.15 * (M['gb'] / K) / 4.5) * 448.0 / S['bw']
                if rowms * Wl / dd > 0.6: continue
                B = fixed / (1 - rowms * Wl / dd)
            L = loop * K / 2.0 + K * 1.5 * B + 5
            m = mlane(tabl, lanes, Wl, L / dd) * mscale; t = dd + m * L
            if best is None or t < best[0]: best = (t, K, Wl, L, m, dd, lanes)
    return best
def rounds27(S, loop, busy=False, worlds=1):
    K = max(2, math.ceil(16.5 / S['lend'])); u = 4 if busy else 1
    work = 16.5 / (S['bw'] / 1000.0) + K * (7.5 + 3.6 * (u - 1)) + 8
    return 1000.0 * 2.6 / (loop * K / 2.0 + work)
LOOPS = (20, 60, 130, 290)
def cell(v, M, k=0):
    return "%3.0f (%s, %.1fx)" % (v, ('%.1fx' % (v / M['L1'][k])) if M['L1'][k] else '-', v / M['L2'])
print("3. corrected ring, tok/s (xL1, xL2) at ring loops 20 / 60 / 130 / 290 ms")
for model, M in MODELS.items():
    print(" %s  L1 %s  L2 ~%s" % (model, M['L1'][0], M['L2']))
    for sw, S in SW.items():
        for busy in (False, True):
            if busy and 'lane' in sw and '1 lane' in sw: continue
            if model == '27B':
                print("   %-44s %-4s rounds, 1 world: %s" % (sw, 'busy' if busy else 'idle', ' / '.join(cell(rounds27(S, P, busy), M) for P in LOOPS)))
                continue
            r = [ring(model, S, P, busy) for P in LOOPS]; c = [ring(model, S, P, busy, code=True) for P in LOOPS]
            print("   %-44s %-4s K=%d Wl=%d L=%3.0f m=%.3f d=%.1f | prose %s | code %s" % (sw, 'busy' if busy else 'idle', r[0][1], r[0][2], r[0][3], r[0][4], r[0][5],
                  ' / '.join(cell(1000 / x[0], M) for x in r), ' / '.join(cell(1000 / x[0], M, 1) for x in c)))
    if M['moe']:
        S = SW['24 GB cards, 16 lanes (3000)']
        print("   (MoE with the architect's optimistic 0.05 ms a row: %s)" % ' / '.join('%.0f' % (1000 / ring(model, S, P, moe_opt=True)[0]) for P in LOOPS))
print("4. sensitivities, 20 ms ring, idle, prose")
S3 = SW['24 GB cards, 16 lanes (3000)']; S30 = SW['ring of 8-12 GB cards (30)']
for model in ('14B', '70B'):
    a = ring(model, S3, 20); print("   %s 24 GB x16 lanes: %.0f ; rows 10 KB not 5.2: %.0f ; m halved: %.0f ; one stage more: %.0f ; two more: %.0f" % (
        model, 1000 / a[0], 1000 / ring(model, S3, 20, rowkb=10.0)[0], 1000 / ring(model, S3, 20, mscale=0.5)[0],
        1000 / ring(model, S3, 20, K=a[1] + 1)[0], 1000 / ring(model, S3, 20, K=a[1] + 2)[0]))
    a = ring(model, S30, 20); print("   %s 8-12 GB ring: %.0f ; rows 10 KB: %.0f ; 10 Mbit/s uplinks: %.0f ; one stage more: %.0f ; m halved: %.0f" % (
        model, 1000 / a[0], 1000 / ring(model, S30, 20, rowkb=10.0)[0], 1000 / ring(model, dict(S30, up=10), 20)[0], 1000 / ring(model, S30, 20, K=a[1] + 1)[0], 1000 / ring(model, S30, 20, mscale=0.5)[0]))
print("5. a local user who ALSO speculates (L2+): same drafter, whole model on 24 GB-class cards, no wire")
for model, T in (('14B', 9.0 + 7.5), ('27B', 16.5 + 7.5), ('70B', 42.0 + 15.0)):
    a, d = 0.70, 3.5; best = max(((1 - a ** (gm + 1)) / (1 - a)) / (gm * d + T * 1.15) for gm in range(1, 9))   # Leviathan rounds, chain, top-1 0.70 prose
    flow = 1000 / (d + mlane(LT, 1, 8, (1.5 * T + 5) / d) * (1.5 * T + 5))                                       # the same flow law with no legs
    print("   %s: plain %.0f ; rounds with a chain drafter %.0f ; the flow law with no wire %.0f tok/s" % (model, 1000 / T, 1000 * best, flow))
print("6. linear worlds (a lane of ONE row, the hybrid's shape): prose m at 4 levels deep: 1 world %.3f, 4 worlds %.3f, 16 worlds %.3f -> tokens a miss x%.2f, x%.2f" % (
    LT[(1, 1, 4)], LT[(4, 1, 4)], LT[(16, 1, 4)], LT[(1, 1, 4)] / LT[(4, 1, 4)], LT[(1, 1, 4)] / LT[(16, 1, 4)]))
S = SW['ring of 8-12 GB cards (30)']
print("7. where it stops, 14B on 8-12 GB cards, loop 0: %.0f tok/s ; on 24 GB cards: %.0f ; 70B on 24 GB cards: %.0f" % (
    1000 / ring('14B', dict(S, up=100), 0)[0], 1000 / ring('14B', SW['24 GB cards, 1 lane'], 0)[0], 1000 / ring('70B', SW['24 GB cards, 1 lane'], 0)[0]))
