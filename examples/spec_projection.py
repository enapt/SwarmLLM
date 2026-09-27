#!/usr/bin/env python3
"""Projected decode tok/s for a two-machine split under speculation, from measured parts.

round_ms = rtt + draft_levels*draft_ms + (head_fix + N*head_per) + N*bytes/bw + (tail_fix + N*tail_per) + logits_back
tokens/round = the replayed best-first tree figure (spec_best_tree.py) for the pair.
Plan and the numbers it produced: docs/plans/split_speculation.md
"""
QWEN = {1: 1.73, 4: 2.81, 8: 3.44, 16: 4.09, 32: 4.59, 64: 5.18, 128: 5.71, 256: 6.36}
LLAMA = {1: 1.81, 4: 3.42, 8: 4.46, 16: 5.38, 32: 6.36, 64: 7.10, 128: 8.12, 256: 8.91}
DEPTH = {1: 1, 4: 4, 8: 7, 16: 9, 32: 11, 64: 13, 128: 16, 256: 18}  # qwen best-first depth
def rnd(N, rtt, bw_mbps, draft_ms=4.0, per=0.25, fix=12.0, tail_mult=1.3, kb=3.8, logits_mb=0.0):
    xfer = N * kb * 1024 * 8 / (bw_mbps * 1e6) * 1000
    back = logits_mb * 8 / bw_mbps * 1000
    return rtt + DEPTH[N] * draft_ms + (fix + N * per) + xfer + (fix + N * per) * tail_mult + back
print("today (TH<->BE, stream): 2.85 tok/s measured = 351 ms/token; rtt_app taken as 300 ms")
for name, rtt in (("TH<->BE", 300), ("same continent", 60), ("same country", 25)):
    for bw in (20, 50):
        base = 1000 / (rtt + 51)
        row = [f"{name:15s} {bw:3d} Mbit/s  plain {base:5.1f}"]
        # DSD as shipped: chain of 4, full-vocab f32 logits back (5 x 152064 x 4 B = 3.0 MB)
        row.append(f"DSD-as-is {QWEN[4] * 1000 / rnd(4, rtt, bw, logits_mb=3.04):5.1f}")
        row.append(f"chain4@tail {QWEN[4] * 1000 / rnd(4, rtt, bw):5.1f}")
        for N in (16, 32, 64, 128):
            row.append(f"tree{N} {QWEN[N] * 1000 / rnd(N, rtt, bw):5.1f}/{LLAMA[N] * 1000 / rnd(N, rtt, bw):4.1f}")
        print("  ".join(row))
