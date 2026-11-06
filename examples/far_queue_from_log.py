#!/usr/bin/env python3
"""Per forward of a flowing reply: its answer's delay against the forwards still out when it was sent.

FIFO pairing per request id (a lone far answers in the order sent), read off a node log at debug
(`DIAG: sent tensor forward via send_request`, `Decoded tensor LayerResult, dispatching`). A delay
that rises by one service time per forward ahead is a serial queue of the flow's own forwards at
the far (FUTURE_WORK #412, `docs/plans/faster_than_local.md` § 13).

Usage: python3 -I examples/far_queue_from_log.py <node log>...   prints ahead:count x median ms
"""
import sys, re, datetime, statistics as st, collections
def ts(l): return datetime.datetime.fromisoformat(l[:26]).timestamp()
for f in sys.argv[1:]:
    sends=collections.defaultdict(list); recvs=collections.defaultdict(list)
    for l in open(f, errors='replace'):
        if 'sent tensor forward via send_request' in l:
            m=re.search(r'request_id=(\S+)', l); sends[m.group(1)].append(ts(l))
        elif 'Decoded tensor LayerResult, dispatching' in l:
            m=re.search(r'request_id=(\S+)', l); recvs[m.group(1)].append(ts(l))
    by=collections.defaultdict(list)
    for rid, s in sends.items():
        r=recvs.get(rid, [])
        if len(s) < 50: continue          # warm-ups
        n=min(len(s), len(r))
        for i in range(n):
            ahead=sum(1 for j in range(i) if r[j] > s[i])
            by[ahead].append((r[i]-s[i])*1000)
    rows=[(k, len(v), st.median(v)) for k, v in sorted(by.items())]
    print(f.split('/')[-1], ' '.join(f'{k}:{n}x{m:.0f}' for k, n, m in rows))
