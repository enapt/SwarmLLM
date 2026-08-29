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

print(f"{'metric':16} {'first':>10} {'last':>10} {'trend':>12}   shape")
print(f"{'':16} {'':>10} {'':>10} {'(Q4-Q1 mean)':>12}\n")
for key,label,unit in [('rss_kb','daemon rss','MB'),('worker_rss_kb','worker rss','MB'),
                       ('threads','threads',''),('fds','open fds',''),
                       ('workers','workers',''),('kv_used_mb','kv used','MB'),
                       ('log_lines','log lines','')]:
    vals=[num(r,key) for r in rows]
    vals=[v for v in vals if v is not None]
    if len(vals)<2: continue
    a,b=vals[0],vals[-1]
    if unit=='MB' and key.endswith('_kb'): a,b=a/1024,b/1024
    # Trend from QUARTILE MEANS, not from three single samples.
    #
    # It used to be `vals[mid]-vals[0]` against `vals[-1]-vals[mid]`, so three
    # individual readings decided the verdict. Worker RSS oscillates by
    # hundreds of MB between samples, and the first sample is taken while the
    # model is still loading — so a clean 2 h run whose quartile means fell
    # monotonically (2873 -> 2820 -> 2674 -> 2652 MB) was reported as
    # "+224.8MB STILL RISING", purely because the first reading was low and the
    # last happened to be high (2026-08-29). A leak detector that cries leak on
    # a clean run gets ignored, and then it cannot report a real one.
    q=max(1,len(vals)//4)
    qm=[sum(seg)/len(seg) for seg in
        (vals[:q], vals[q:2*q], vals[2*q:3*q], vals[3*q:]) if seg]
    trend=qm[-1]-qm[0]          # direction across the whole run
    late=qm[-1]-qm[-2] if len(qm)>1 else 0.0   # still moving at the end?
    spread=max(vals)-min(vals)  # what the sample noise can hide
    if unit=='MB' and key.endswith('_kb'):
        trend/=1024; late/=1024; spread/=1024
    # Percentage OF THE TREND, not of first-vs-last: showing "-221MB" beside
    # "+9.0%" is two numbers contradicting each other in one row.
    base=qm[0]/1024 if (unit=='MB' and key.endswith('_kb')) else qm[0]
    pct=(trend/base*100) if base else 0
    # Cumulative counters only ever rise; a rate is the informative number.
    # `log_lines` in particular ALWAYS rises here because soak_test.sh runs the
    # node at -v deliberately, so a rise verdict on it says nothing at all.
    if key=='log_lines':
        shape=f"{(b-a)/mins:.0f} lines/min (node runs at -v)" if mins else "n/a"
    elif abs(trend)<=spread*0.10: shape="flat (within sample noise)"
    elif trend>0 and late>0:      shape="STILL RISING  <-- look"
    elif trend>0:                 shape="rose then settled"
    else:                         shape="settled/decreasing"
    print(f"{label:16} {a:10.1f} {b:10.1f} {trend:+9.1f}{unit:<3} {pct:+6.1f}%  {shape}")

ok=num(last,'ok') or 0; fail=num(last,'fail') or 0
tot=ok+fail
print(f"\nrequests: {ok:.0f} ok, {fail:.0f} failed" + (f"  ({fail/tot*100:.1f}% failure)" if tot else ""))
if fail: print("  ^ investigate: a soak that stops serving is the failure this looks for")
PY
