#!/usr/bin/env python3
"""How much of a mixture-of-experts reply would a card holding experts BY USE serve,
against one holding whole layers — read off a `SWARMLLM_TRACE_EXPERTS` trace.

The trace has one line per MoE layer per decode step, in layer order: the experts that
layer chose for that token. `examples/skip_agreement_probe.rs` with `PROBE_SKIP=0`
writes one (one token a forward), for its prompts in order, `PROBE_N` tokens each.

    python3 examples/expert_locality.py TRACE --layers 48 --experts 128 --tokens 128 --prompts 4

For each card capacity (a fraction of all experts) it prints the share of the reply's
expert activations found on the card under:
  layers     the first layers' experts whole (today's split): the fraction itself
  lru        each layer keeps its most recently used experts (per prompt, from empty)
  static-own each layer's most used experts IN THIS PROMPT (an upper bound for a fixed set)
  static-cv  each layer's most used experts in the OTHER prompts (a fixed set chosen
             before the prompt is seen — what a node could ship with)
and the chance a layer reuses an expert the previous token used there.
"""
import argparse
from collections import Counter, OrderedDict


def read(path, layers, tokens, prompts):
    rows = [list(map(int, l.split())) for l in open(path) if l.strip()]
    need = layers * tokens * prompts
    if len(rows) != need:
        raise SystemExit(f"{len(rows)} lines, expected {layers} x {tokens} x {prompts} = {need}")
    # out[p][t][l] = set of experts
    out = []
    i = 0
    for _ in range(prompts):
        steps = []
        for _ in range(tokens):
            steps.append([set(rows[i + l]) for l in range(layers)])
            i += layers
        out.append(steps)
    return out


def lru_hits(steps, layers, slots):
    hits = total = 0
    for l in range(layers):
        cache = OrderedDict()
        for st in steps:
            for e in st[l]:
                total += 1
                if e in cache:
                    hits += 1
                    cache.move_to_end(e)
            for e in st[l]:
                cache[e] = True
                cache.move_to_end(e)
            while len(cache) > slots:
                cache.popitem(last=False)
    return hits, total


def static_hits(steps, layers, slots, chosen):
    hits = total = 0
    for st in steps:
        for l in range(layers):
            for e in st[l]:
                total += 1
                hits += e in chosen[l]
    return hits, total


def top(counts, slots):
    return {e for e, _ in counts.most_common(slots)}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("trace")
    a.add_argument("--layers", type=int, required=True)
    a.add_argument("--experts", type=int, required=True)
    a.add_argument("--tokens", type=int, required=True)
    a.add_argument("--prompts", type=int, required=True)
    a.add_argument("--fractions", default="0.13,0.25,0.46,0.6")
    args = a.parse_args()
    data = read(args.trace, args.layers, args.tokens, args.prompts)
    L, E = args.layers, args.experts

    reuse = n = 0
    for steps in data:
        for prev, cur in zip(steps, steps[1:]):
            for l in range(L):
                reuse += len(prev[l] & cur[l])
                n += len(cur[l])
    k = len(data[0][0][0])
    print(f"{L} layers, {E} experts, top-{k}; reuse of the previous token's experts in a layer: "
          f"{reuse / n:.3f} (chance {k / E:.3f})")

    counts = [[Counter() for _ in range(L)] for _ in data]
    for p, steps in enumerate(data):
        for st in steps:
            for l in range(L):
                counts[p][l].update(st[l])

    print(f"{'card holds':>10} {'layers':>7} {'lru':>7} {'static-own':>10} {'static-cv':>9}")
    for f in map(float, args.fractions.split(",")):
        slots = max(1, round(f * E))
        lru = own = cv = 0
        tot = 0
        for p, steps in enumerate(data):
            h, t = lru_hits(steps, L, slots)
            lru += h
            tot += t
            own += static_hits(steps, L, slots, [top(counts[p][l], slots) for l in range(L)])[0]
            others = [sum((counts[q][l] for q in range(len(data)) if q != p), Counter()) for l in range(L)]
            cv += static_hits(steps, L, slots, [top(others[l], slots) for l in range(L)])[0]
        print(f"{f:>9.0%} {f:>7.1%} {lru / tot:>7.1%} {own / tot:>10.1%} {cv / tot:>9.1%}")
    print(f"per prompt, LRU at 46%: " + ", ".join(
        f"{lru_hits(s, L, round(0.46 * E))[0] / lru_hits(s, L, round(0.46 * E))[1]:.1%}" for s in data))


if __name__ == "__main__":
    main()
