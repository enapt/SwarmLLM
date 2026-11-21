# Kill ladder step 1 (standing ring): from flow reply lines already logged,
# levels drafted, voided, misses, and voided levels a miss -> implied verdict delay L = n*d.
import re, glob, os, sys, statistics as st
base = os.path.expanduser('~/swarmllm-gate/deep/fr')
pat = re.compile(r'DSD: a tree that flowed is complete')
def field(line, k):
    m = re.search(r'\b' + k + r'=([0-9.]+)', line)
    return float(m.group(1)) if m else None
rows = []
for tag in sys.argv[1:]:
    for f in sorted(glob.glob(f'{base}/real_link_{tag}/H_*.log')):
        for line in open(f, errors='replace'):
            if not pat.search(line): continue
            g = {k: field(line, k) for k in ['forwards','levels','rows_a_level','emitted','misses','rebases',
                 'flowed_ms','tokens_per_sec','drafting_ms','waiting_ms','least_answer_ms','copied_rows','copied_walked',
                 'earlier_walked','stale_answers','window','far_least_ms']}
            if not g['forwards'] or g['emitted'] is None or g['emitted'] < 40: continue
            g['tag'] = tag; g['file'] = os.path.basename(f)
            rows.append(g)
print(f'{len(rows)} flow replies with >= 40 tokens')
def show(sel, name):
    if not sel: return
    def med(k, fn=lambda g, k: g[k]):
        v = [fn(g, k) for g in sel if fn(g, k) is not None]
        return st.median(v) if v else float('nan')
    tok_fwd = med('x', lambda g, k: g['emitted'] / g['forwards'])
    drafted = med('x', lambda g, k: g['forwards'] * g['levels'])
    voided_frac = med('x', lambda g, k: 1 - g['emitted'] / (g['forwards'] * g['levels']))
    per_miss = med('x', lambda g, k: (g['forwards'] * g['levels'] - g['emitted']) / max(g['misses'], 1))
    d_ms = med('x', lambda g, k: g['flowed_ms'] / (g['forwards'] * g['levels']))  # wall ms a drafted level
    L_impl = med('x', lambda g, k: (g['forwards'] * g['levels'] - g['emitted']) / max(g['misses'], 1) * g['flowed_ms'] / (g['forwards'] * g['levels']))
    m_lvl = med('x', lambda g, k: g['misses'] / max(g['emitted'], 1))
    print(f'{name:28s} n={len(sel):3d} tok/s {med("tokens_per_sec"):5.1f} tok/fwd {tok_fwd:4.2f} levels {med("levels"):.0f} '
          f'voided {voided_frac*100:4.0f}% misses/token {m_lvl:4.2f} voided/miss {per_miss:5.1f} '
          f'ms/level {d_ms:5.1f} implied L {L_impl:5.0f} ms  least_answer {med("least_answer_ms"):4.0f} far_least {med("far_least_ms"):4.0f} '
          f'copied_walked {med("copied_walked"):.0f}')
for tag in sys.argv[1:]:
    for kind in ['prose', 'code']:
        pass
    show([g for g in rows if g['tag'] == tag], tag)
show(rows, 'ALL')
