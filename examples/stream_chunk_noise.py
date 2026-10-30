"""Per-chunk timing of a split's stream of guesses (#375).

Reads the DEBUG line 'DSD: a streamed check answered' (run the node with
SWARMLLM_LOGGING_LEVEL=debug) and reports, per width (rows):
  - checks sent alone, timed at arrival: spread
  - how long answers sat before the loop took them
  - dispersion: the far end's own time for a chunk, from the gap between two
    consecutive answers, where the later chunk reached the far end before the
    earlier one finished there (packet-pair dispersion, Keshav 1991; on a
    loaded path the true service time may be a LOCAL mode, Dovrolis et al.
    2001 - so a histogram, not one number).
RTT_MS: the round trip to the far end; NEAR_MS: this node's own layers' time
for a chunk (only decides which gaps count as "waited there").
usage: python3 -I examples/stream_chunk_noise.py RTT_MS NEAR_MS node.log [...]
Evidence it produced: docs/invariants/scheduling.md § "A chunk is as wide as
it pays", the #375 paragraph.
"""
import re
import sys
from collections import defaultdict

LINE = re.compile(r"DSD: a streamed check answered(.*)$")
KV = re.compile(r"(\w+)=([^\s]+)")


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


def summary(xs):
    if not xs:
        return "n=0"
    return "n=%3d min=%5.0f p25=%5.0f med=%5.0f mean=%5.0f p75=%5.0f max=%5.0f" % (
        len(xs), min(xs), q(xs, .25), q(xs, .5), sum(xs) / len(xs), q(xs, .75), max(xs))


def hist(xs, width=50):
    h = defaultdict(int)
    for x in xs:
        h[int(x // width) * width] += 1
    return " ".join("%d:%d" % (k, h[k]) for k in sorted(h))


def main():
    rtt, near = float(sys.argv[1]), float(sys.argv[2])
    chunks = defaultdict(list)
    for path in sys.argv[3:]:
        for line in open(path, errors="replace"):
            m = LINE.search(line)
            if not m:
                continue
            f = dict(KV.findall(m.group(1)))
            chunks[f.get("request_id", "?")].append(f)
    alone = defaultdict(list)
    queued = defaultdict(list)
    waited = defaultdict(list)
    disp = defaultdict(list)
    for rid, cs in chunks.items():
        for c in cs:
            r = int(c["rows"])
            (alone if c["alone"] == "true" else queued)[r].append(float(c["check_ms"]))
            waited[r].append(float(c["waited_ms"]))
        for a, b in zip(cs, cs[1:]):
            if int(b["turn"]) != int(a["turn"]) + 1:
                continue
            # b reached the far end before a finished there
            if float(b["started_at_ms"]) + near + rtt < float(a["arrived_at_ms"]):
                disp[int(b["rows"])].append(float(b["arrived_at_ms"]) - float(a["arrived_at_ms"]))
    print("requests:", len(chunks), " chunks:", sum(len(c) for c in chunks.values()))
    for r in sorted(set(alone) | set(queued)):
        print("rows %d alone   check %s" % (r, summary(alone[r])))
        print("rows %d queued  check %s" % (r, summary(queued[r])))
        print("rows %d waited for the loop %s  total %.0f" % (r, summary(waited[r]), sum(waited[r])))
        print("rows %d far by dispersion %s" % (r, summary(disp[r])))
        print("        histogram(50ms) %s" % hist(disp[r]))
        print("        alone histogram(50ms) %s" % hist(alone[r]))


main()
