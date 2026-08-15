#!/bin/bash
# Summarise a soak_test.sh CSV: first vs last, and the trend.
#
# A soak only answers a question if you compare the end against the start AND
# check the shape in between — a metric that rises and settles is not a leak,
# one that rises linearly is. This prints both so the difference is visible.
#
# Usage: examples/soak_report.sh [/tmp/swarm_soak/soak.csv]

set -u
CSV="${1:-/tmp/swarm_soak/soak.csv}"
[ -f "$CSV" ] || { echo "No CSV at $CSV" >&2; exit 1; }

python3 - "$CSV" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1])))
if len(rows)<2:
    print(f"only {len(rows)} sample(s) — nothing to compare yet"); sys.exit(0)

def num(r,k):
    try: return float(r[k])
    except Exception: return None

first,last=rows[0],rows[-1]
mins=(num(last,'elapsed_s') or 0)/60
print(f"samples: {len(rows)} over {mins:.0f} min\n")

print(f"{'metric':16} {'first':>10} {'last':>10} {'change':>12}   shape")
for key,label,unit in [('rss_kb','daemon rss','MB'),('worker_rss_kb','worker rss','MB'),
                       ('threads','threads',''),('fds','open fds',''),
                       ('workers','workers',''),('kv_used_mb','kv used','MB'),
                       ('log_lines','log lines','')]:
    vals=[num(r,key) for r in rows]
    vals=[v for v in vals if v is not None]
    if len(vals)<2: continue
    a,b=vals[0],vals[-1]
    if unit=='MB' and key.endswith('_kb'): a,b=a/1024,b/1024
    delta=b-a
    pct=(delta/a*100) if a else 0
    # Shape: compare the growth in the first half against the second. A leak
    # keeps growing; a warm-up grows then flattens.
    mid=len(vals)//2
    g1=vals[mid]-vals[0]; g2=vals[-1]-vals[mid]
    if abs(delta)<1e-9: shape="flat"
    elif g2>g1*0.6 and g2>0: shape="STILL RISING  <-- look"
    elif g2>0: shape="rising, decelerating"
    elif g2<0: shape="settled/decreasing"
    else: shape="flat"
    print(f"{label:16} {a:10.1f} {b:10.1f} {delta:+9.1f}{unit:<3} {pct:+6.1f}%  {shape}")

ok=num(last,'ok') or 0; fail=num(last,'fail') or 0
tot=ok+fail
print(f"\nrequests: {ok:.0f} ok, {fail:.0f} failed" + (f"  ({fail/tot*100:.1f}% failure)" if tot else ""))
if fail: print("  ^ investigate: a soak that stops serving is the failure this looks for")
PY
