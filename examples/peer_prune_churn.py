#!/usr/bin/env python3
"""Each peer's part drops and fetch-backs, per model, read from THIS node's log.

A peer's prune is invisible to it in aggregate but visible to every node it
announces to: each announcement that omits a part logs
`Peer retracted shards it no longer hosts … dropped=N retained=M`, and a claim
put back afterwards logs `a holder claim this peer had withdrawn was reinstated`.
A (peer, model) row with many drops and its `keep` count climbing back between
them is a node deleting and re-fetching the same parts (FUTURE_WORK #286,
gotcha #795) — the cost is the re-fetch, hundreds of MB a part.

Usage: examples/peer_prune_churn.py LOG [LOG ...]   (oldest first)
"""
import collections
import re
import sys

RETRACTED = re.compile(
    r"^(\S+).*Peer retracted shards it no longer hosts node_id=(\w+) model=(\S+) "
    r"dropped=(\d+) retained=(\d+)"
)
REINSTATED = re.compile(r"^(\S+).*withdrawn was reinstated model=(\S+) shard=(\d+) node=(\w+)")


def main(paths):
    events = collections.defaultdict(list)
    for path in paths:
        with open(path, errors="replace") as log:
            for line in log:
                m = RETRACTED.match(line)
                if m:
                    events[(m[2][:8], m[3])].append((m[1][5:19], f"drop{m[4]}/keep{m[5]}"))
                    continue
                m = REINSTATED.match(line)
                if m:
                    events[(m[4][:8], m[2])].append((m[1][5:19], f"back#{m[3]}"))
    for (peer, model), seen in sorted(events.items(), key=lambda kv: -len(kv[1])):
        drops = sum(1 for _, e in seen if e.startswith("drop"))
        print(f"{peer} {model}: {drops} drops, {len(seen) - drops} reinstated")
        print("   ", " ".join(f"{t}:{e}" for t, e in sorted(seen)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
