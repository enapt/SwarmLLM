#!/usr/bin/env python3
"""Best-first draft trees of N nodes (Sequoia's positional-acceptance model), replayed on real positions.

The tree SHAPE is chosen from the rank histogram: a node is a path of drafter ranks
(r1, r2, ...), valued at prod p[r_i] where p[k] = P(the true token is the drafter's k-th guess).
The top-N paths by value form an optimal tree under that model (value falls along a path, so
the top-N set is closed under prefixes). The SHAPE is then replayed over the real reply: from
each round's start, walk while the true rank-path stays inside the tree — so correlation
between neighbouring positions is measured, not assumed away.
usage: spec_best_tree.py qwen|llama   (reads the dump spec_coverage.py writes)
"""
import heapq, sys
import numpy as np

which = sys.argv[1]
z = np.load(f"/home/user/swarmllm-ref/spec/dump_{which}.npz")
rank, reply = z["rank"], z["reply"]
K = 16
p = np.array([(rank == k).mean() for k in range(K)])

def best_tree(N):
    heap = [(-p[k], (k,)) for k in range(K)]
    heapq.heapify(heap)
    tree, total = set(), 0.0
    while len(tree) < N and heap:
        v, path = heapq.heappop(heap)
        tree.add(path); total += -v
        for k in range(K):
            heapq.heappush(heap, (v * p[k], path + (k,)))
    return tree, total

def replay(tree):
    toks = rnds = 0
    for r in np.unique(reply):
        rk = rank[reply == r]
        pos, n = 0, len(rk)
        while pos < n:
            path, run = (), 0
            while pos + run < n and path + (int(rk[pos + run]),) in tree:
                path += (int(rk[pos + run]),); run += 1
            got = min(run + 1, n - pos); toks += got; rnds += 1; pos += got
    return toks / rnds

print(f"{which}: P(rank=k) k<8: {np.round(p[:8], 3).tolist()}")
print(f"{'nodes':>6} {'model':>6} {'replay':>7} {'depth':>6}")
for N in (1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048):
    t, tot = best_tree(N)
    print(f"{N:6d} {1 + tot:6.2f} {replay(t):7.2f} {max(len(x) for x in t):6d}")
