#!/usr/bin/env python3
"""Two arms' speeds per prompt, overall and grouped by the reply each arm actually wrote.

A greedy reply parts at a near-tie that the batch shape flips (a stream's checks and a tree's are
different shapes on the far card), and the drafter guesses one text better than another. So two arms
of ONE prompt can compare different replies as much as different shapes: on 2026-11-06 (#411) the
stream arms of one prose prompt wrote two stories, 11.0-11.9 tok/s on one and 14.0-14.1 on the
other, while every tree arm wrote a third — read whole, "+26% for trees". Read arms within one text
first, and run several prompts (`real_link.sh` takes `PROMPTS="prose prose2 prose3 code"`).

Reads a rig's out directory: `results.jsonl` (one JSON line a reply, `label` = `<arm>_<n>_<prompt>`,
`decode_tok_s`) and `<label>.txt` (the reply's text), as `real_link.sh` writes them.

usage: python3 -I examples/pair_speeds_by_reply.py <rig out dir> [prefix chars, default 120]

Prints, per prompt: each arm's range and median; then the replies grouped by their first
<prefix chars> (BOTH = both arms wrote that opening); then each exact reply's speeds by arm.
"""
import hashlib
import json
import os
import statistics as st
import sys


def rng(v):
    return f"{min(v):.2f}-{max(v):.2f} (median {st.median(v):.2f}, n={len(v)})" if v else "-"


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    out = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    by = {}
    for line in open(os.path.join(out, "results.jsonl")):
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error"):
            continue
        arm, _, prompt = r["label"].split("_")
        text = open(os.path.join(out, r["label"] + ".txt")).read()
        by.setdefault(prompt, []).append((arm, r["decode_tok_s"], text))

    for prompt, rs in by.items():
        arms = sorted({a for a, _, _ in rs})
        print(f"== {prompt}")
        for a in arms:
            print(f"   {a:12s} all: {rng([s for x, s, _ in rs if x == a])}")
        openings = {}
        for a, s, t in rs:
            openings.setdefault(t[:k], []).append((a, s, t))
        for i, (head, g) in enumerate(sorted(openings.items(), key=lambda kv: -len(kv[1]))):
            both = "BOTH" if len({a for a, _, _ in g}) > 1 else "    "
            print(f"   opening {i + 1} {both} {head[:50]!r}")
            print("      " + "  ".join(f"{a} {rng([s for x, s, _ in g if x == a])}" for a in arms
                                     if any(x == a for x, _, _ in g)))
        exact = {}
        for a, s, t in rs:
            exact.setdefault(hashlib.md5(t.encode()).hexdigest()[:8], []).append((a, s))
        print("   exact replies: " + "; ".join(
            f"{h}{' BOTH' if len({a for a, _ in v}) > 1 else ''} "
            + " ".join(f"{a}:{s}" for a, s in sorted(v)) for h, v in exact.items()))


if __name__ == "__main__":
    main()
