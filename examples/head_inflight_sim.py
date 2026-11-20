#!/usr/bin/env python3
"""Several rounds of guesses in flight for a Qwen 3.5-family split (FUTURE_WORK #486, docs/plans/small_slices.md
step 2b) — a zero-code check before any build.

The family's guesser is the file's next-token head beside the split's FIRST segment (#425 (a), #485). It guesses a
chain: guess k+1 is made from the head's own output for guess k, so the further a guess is from the last row the
trunk confirmed, the less often it is kept. Today one round is out at a time (`ngram_only_spec`): the head guesses
`g`, this computer's slice forwards them, the far slice checks, the verdict (and the last layer's rows) come back.
With K rounds in flight, round j+1 is sent from the PREDICTED end of round j — the head's own guess of the token the
check will add — so it is kept only if every guess before it in the chain was right.

Model (a Monte Carlo over the chain):
- a chain started from a confirmed row: guess k is right with probability `acc(k)`; a round covers the next `g`
  guesses (and, for a round after the first, the guess before them as its leading input);
- the head makes guesses one at a time (`guess_ms` each, the processor here); this computer's slice runs a round's
  forward in `here_ms`, then the far slice in `far_ms`, one forward at a time on each; the link is `rtt/2` each way;
- a verdict confirms up to the first wrong guess plus the check's own token, drops every round sent past a miss,
  and the head restarts its chain from the new confirmed row (the rows arrive with the verdict).

Calibrated on the rig of 2026-11-20 (`~/swarmllm-gate/split27_head`, Qwen3.8-27B, here [0,21) → the US laptop
[21,64), 272 ms): chains of 4, 37-40% of guesses kept, 11.6-12.4 ms a guess, 437 ms a round → here + far forward
~117 ms a round. The two acceptance shapes both fit 38% over 4: i.i.d. 0.65 a guess (optimistic for deep guesses)
and 0.75 x 0.8^(k-1) (a chained head losing a fifth a step, pessimistic).

Prior art: PipeInfer (arXiv 2407.11798, continuous asynchronous speculation: drafts sent ahead of the verdict,
cancelled on a miss), PEARL (drafting during the check), SpecPipe (arXiv 2504.04104). usage:
python3 -I examples/head_inflight_sim.py [seeds]
"""
import random
import statistics
import sys


def iid(a):
    return lambda k: a


def decaying(first, keep):
    return lambda k: first * keep ** (k - 1)


def run(seed, acc, g, k_most, rtt, guess_ms, here_ms, far_ms, tokens=400):
    rng = random.Random(seed)
    t = 0.0  # the head's clock: when it is free to guess again
    confirmed = 0
    here_free = far_free = 0.0
    chain_from, chain = 0, []  # chain[k-1] = is guess k (counted from chain_from) right
    flight = []  # (verdict heard, chain_from, first guess k, guesses)
    next_k = 1  # the chain index the next round's guesses start at

    def right(k):
        while len(chain) < k:
            chain.append(rng.random() < acc(len(chain) + 1))
        return chain[k - 1]

    while confirmed < tokens:
        if len(flight) < k_most:
            # Guess g more, send the round: its leading input is guess next_k - 1 (a round after the first).
            t += guess_ms * g
            start = max(here_free, t)
            here_free = start + here_ms
            far_start = max(far_free, here_free + rtt / 2)
            far_free = far_start + far_ms
            flight.append((far_free + rtt / 2, chain_from, next_k, g))
            next_k += g + 1
            continue
        flight.sort()
        heard, origin, k0, n = flight.pop(0)
        t = max(t, heard)
        if origin != chain_from:
            continue  # sent from a chain a miss already ended
        # A round still on the live chain had every guess before it right (a break restarts the chain below).
        kept = 0
        while kept < n and right(k0 + kept):
            kept += 1
        confirmed = chain_from + (k0 - 1) + kept + 1  # through the first wrong guess, the check's token
        # The next round in flight took the head's guess of the check's own token as its leading input.
        goes_on = kept == n and right(k0 + n) and any(f[1] == origin for f in flight)
        if not goes_on:
            # A miss, a wrong guess of the check's token, or nothing after it: every round after this one is
            # void, and the chain restarts from the confirmed row with the rows the verdict brought.
            chain_from, chain, next_k = confirmed, [], 1
            flight = [f for f in flight if f[1] != origin]
    return tokens / t * 1000.0


def main():
    seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
    shapes = {"i.i.d. 0.65": iid(0.65), "0.75 x 0.8^(k-1)": decaying(0.75, 0.8)}
    here_ms, far_ms = 40.0, 77.0  # ~117 ms a round between them (the rig's 437 − 272 − 4 × 12)

    def mean(**kw):
        return statistics.mean(run(s, **kw) for s in seeds)

    # The check: one round out, chains of 4, 272 ms, the processor's 12 ms a guess → the rig's 5.0-6.2 tok/s.
    for name, acc in shapes.items():
        v = mean(acc=acc, g=4, k_most=1, rtt=272, guess_ms=12, here_ms=here_ms, far_ms=far_ms)
        print(f"check ({name}): one round of 4 at 272 ms = {v:.2f} tok/s (rig: prose 5.0-5.9, code 5.95-6.2)")
    # What reaches 20-30 tok/s: the forward work a round, not the rounds in flight. Slices wholly on good cards
    # (a 27B forward of 5 rows ~30 ms on one, split 12 + 18 between two), the head on a card.
    print("\nwhat the bar needs: one round out (K=1) vs two, the head on a card (2 ms a guess)")
    for name, acc in shapes.items():
        for stages, label in (((40.0, 77.0), "slices as measured (117 ms a round)"),
                              ((12.0, 18.0), "slices wholly on good cards (30 ms)")):
            cells = []
            for rtt in (272, 130, 60, 20):
                one = max((mean(acc=acc, g=g, k_most=1, rtt=rtt, guess_ms=2.0, here_ms=stages[0],
                                far_ms=stages[1]), g) for g in (2, 3, 4, 6))
                two = max((mean(acc=acc, g=g, k_most=2, rtt=rtt, guess_ms=2.0, here_ms=stages[0],
                                far_ms=stages[1]), g) for g in (2, 3, 4, 6))
                cells.append(f"{rtt:3d} ms {one[0]:5.1f} / {two[0]:5.1f}")
            print(f"  {name:17s} {label:36s} " + "  ".join(cells))
    for guess_ms, where in ((12.0, "head on the processor, 12 ms a guess"), (2.0, "head on a card, ~2 ms a guess")):
        print(f"\n{where}")
        for name, acc in shapes.items():
            for rtt in (272, 130, 60):
                cells = []
                for k_most in (1, 2, 3, 4, 6):
                    best = max((mean(acc=acc, g=g, k_most=k_most, rtt=rtt, guess_ms=guess_ms, here_ms=here_ms,
                                     far_ms=far_ms), g) for g in (1, 2, 3, 4, 6))
                    cells.append(f"K={k_most} {best[0]:5.1f} (g={best[1]})")
                print(f"  {name:17s} {rtt:3d} ms: " + "  ".join(cells))


if __name__ == "__main__":
    main()
