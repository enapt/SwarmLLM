#!/usr/bin/env python3
"""How well the routing cost model priced the splits a node ran (FUTURE_WORK #3).

Reads a node log's `DIAG: request complete` lines that carry `predicted_terms=`
(logged since v0.3.234) and reports actual ÷ predicted time by route shape and by
pricing regime — warm or cold, a measured per-visit network term (`net>0`) or the
hop folded into the peer's per-layer figure (`net:0`), and replies fast enough to
have been sped up by a drafter. FUTURE_WORK #3's rule: collect, then fit per
term; never tune a constant from one peer's traffic.

usage: python3 examples/split_cost_terms.py ~/.local/share/swarmllm/node.log [--rows]
"""
import collections
import re
import statistics as st
import sys

# A reply faster than this per token was almost certainly sped up by guessing
# ahead (the 2026-10-14 rigs' drafter arms ran ~180-220 ms a token, plain ~400).
FAST_MS_PER_TOKEN = 260


def rows(path):
    for line in open(path, errors="replace"):
        if "request complete" not in line or "predicted_terms=" not in line:
            continue
        f = dict(re.findall(r"(\w+)=(\S+)", line))
        terms = dict(t.split(":") for t in f["predicted_terms"].split("/"))
        yield dict(
            ts=line[:16],
            seg=int(f.get("segments", 0)),
            model=f.get("model", "")[:28],
            regions=f.get("regions", ""),
            prompt_tokens=int(f.get("prompt_tokens", 0)),
            tokens=int(f.get("tokens", 0)),
            total=int(f.get("total_ms", 0)),
            pred=int(f.get("predicted_ms", 0)),
            outcome=f.get("outcome"),
            **{k: int(v) for k, v in terms.items()},
        )


def main():
    all_rows = list(rows(sys.argv[1]))
    if not all_rows:
        print("no priced requests in this log")
        return
    ok = [r for r in all_rows if r["outcome"] == "ok" and r["tokens"] > 0]
    print(f"{len(all_rows)} priced, {len(ok)} completed with tokens, {all_rows[0]['ts']} .. {all_rows[-1]['ts']}")
    groups = collections.defaultdict(list)
    for r in ok:
        fast = r["total"] / r["tokens"] < FAST_MS_PER_TOKEN
        key = (
            f"{r['seg']} segment(s)",
            "cold" if r["cold"] > 0 else "warm",
            "net>0" if r["net"] > 0 else "net:0",
            "fast (drafter?)" if fast else "plain",
        )
        groups[key].append(r)
    for key, group in sorted(groups.items()):
        ratio = sorted(r["total"] / max(r["pred"], 1) for r in group)
        peers = collections.Counter(r["regions"] for r in group).most_common(2)
        line = f"{' / '.join(key)}: n={len(group)} actual/predicted median {st.median(ratio):.2f} [{ratio[0]:.2f}..{ratio[-1]:.2f}] regions {peers}"
        if key[1] == "cold":
            line += f" | cold term median {st.median(r['cold'] for r in group):.0f} ms"
        print(line)
    if "--rows" in sys.argv:
        for r in ok:
            print(
                f"{r['ts']} seg={r['seg']} {r['model']:<28} {r['regions']:<10} pt={r['prompt_tokens']:<5} "
                f"tok={r['tokens']:<4} total={r['total']:<7} pred={r['pred']:<7} x{r['total'] / max(r['pred'], 1):.2f} "
                f"net={r['net']} compute={r['compute']} prompt={r['prompt']} cold={r['cold']}"
            )


if __name__ == "__main__":
    main()
