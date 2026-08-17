# Credits: what exists, why it is switched off, and how to build it properly

**Status (2026-08-17): the credit economy is dormant by deliberate choice.** The
accounting still runs and is still recorded, but it gates nothing and the
dashboard no longer presents a balance as though it were money. This document
says exactly what was wrong, what the real design is, and what has to be true
before any of it is switched back on.

Nothing here is a criticism of the code that exists. The cryptography is sound
and most of the pieces are correct in isolation. What was missing is the part
that makes them add up.

---

## 1. What is actually true today

### 1.1 No credit has ever moved between two nodes *as payment for work*

`credit::transaction` builds a `CreditTransaction`, signs it with the serving
node's key, counter-signs with the requester's, verifies both signatures, and
rejects replays against `TREE_TRANSACTIONS`. The inbound handler in
`daemon/dispatch` accepts one and applies it. `swarmllm_types` carries the wire
type. It reads like a working payment system.

It has never run. `create_transaction` and `cosign_transaction` are both
`#[cfg(test)]` — they do not exist in a release binary. This is the verifying
half of a protocol whose sending half was never written.

**One exception, and it does not rescue the model.** Pool credit *forwarding*
(`pool::forward::forward_credits_to_owner` → `PoolCommand::ProcessCreditForward`
→ `pool::manager::handle_credit_forward`) is complete on both sides and really
does apply to the pool owner's balance, with signature checks, replay dedup,
freshness bounds and rate limits. So a balance genuinely moves between two
machines here.

It changes nothing about the problem, because **what moves is a number the
member minted for itself**. Forwarding concentrates self-issued credit in the
owner's account; it does not make the books add up, and a pool of N machines
that only ever served themselves still ends up with an owner balance that
corresponds to no work done for anybody else. It is a correct mechanism
operating on an unsound quantity — which is worth knowing when it is reused,
because the mechanism itself is a reasonable template for §3's settlement.

### 1.2 What happens instead

Measured across two machines, 2026-08-09:

- The **requester** reserves credits into escrow and then settles against its
  own balance (`escrow_reserve` → `escrow_settle_adjust`). `release_escrow`
  records a `to_node` field and logs it — and transfers nothing. That field
  makes it read like a payment. It is a memo.
- The **serving node** separately mints its own fee locally, via
  `pending_credit_earn` → `inference_serve_earning`.

So one request debits the requester ~430 and credits the server ~440, **from
nothing**, on two ledgers that never reconcile with each other. Each number is
individually plausible. The system-wide total is meaningless.

A node's balance therefore measures **its own activity**, not value received
from anyone. Nothing stops a node inflating its balance by serving itself;
`anti_gaming` guards rate and pattern, not provenance.

### 1.3 It was not inert — it gated two things

The previous version of this note said credits "gate nothing". That was wrong,
and the error mattered, because it is what made the gap urgent rather than
theoretical. Two live gates read that self-minted number:

1. **`MIN_BALANCE_FOR_INFERENCE = -1000`** (`router/mod.rs`) — a *remote*
   requester below the floor was refused inference outright, with
   `InsufficientCredits` → HTTP 402.
2. **Per-tier concurrency cap** (`priority::max_concurrent_for_tier`) — Bronze
   got ¼ of `max_concurrent_requests`, Silver ½, Gold/Platinum the lot. Tier
   came from balance plus a gossiped network percentile.

Both meant a node that had minted itself a good number got measurably better
service than one that had not, and a node driven negative by a bug or a burst of
failed requests got worse — permanently, since there was no recovery path.

Both are switched off as of 2026-08-17. See §4.

---

## 2. Why the obvious fix is wrong

The tempting move is to wire `create_transaction` into the serving path and
call it done. That produces a worse system than the one we have, because the
hard parts are not the signatures.

- **Who initiates.** The server knows what it did; the requester knows what it
  received. A transaction signed by the server alone is a self-assessed
  invoice — exactly the provenance problem we already have, now with a
  signature on it. The dual signature exists precisely so settlement happens
  where both parties agree on the amount.
- **Settlement failure is not an edge case.** The work is already done when
  settlement is attempted. If the requester has vanished, does the server eat
  it or retry? A retry queue is a new persistent structure with its own bounds,
  sweep, and failure modes.
- **Double-spend across peers.** A balance is a local integer. Nothing stops a
  node promising the same balance to five peers concurrently. Replay protection
  covers a transaction being applied *twice*; it does not cover a balance being
  *promised* twice. There is no global consensus here to appeal to, and adding
  one is out of scope for a P2P daemon with no chain.
- **Migration.** Every existing balance was minted locally by its own node.
  Whatever the first real transfer is, it starts from books that do not add up.

Related: gotcha **#280** (money leaving the requester is not evidence of money
reaching the server) and **#278** (the books reconcile by construction, so
reconciliation proves nothing).

---

## 3. The design

### 3.1 Prior art, and what transfers

Surveyed 2026-08-17 (diagnosis rule 0). Three families:

- **Global-consensus token emission** — Bittensor: validators score miners via
  Yuma Consensus and emissions are distributed on-chain. Requires a blockchain,
  a token, and a validator set. **Does not transfer**: SwarmLLM has no chain and
  should not grow one.
- **Refereed dispute resolution** — Gensyn: bitwise-deterministic primitives
  plus a dispute layer, with `Judge` (2026-04) as the evaluation half. Aimed at
  *training* verification. Interesting later for "did this node actually run the
  model", **orthogonal to** "did anyone get paid".
- **Bilateral micropayment channels with bounded exposure** — the micropayment
  literature. **This is the one that fits.** The findings that matter: you
  cannot prevent unilateral default without custody or consensus, so instead you
  *bound exposure* and make defaulting *visible*; and a party "gets few chances
  to default before credit is withdrawn". Channels are explicitly not perfectly
  trustless — uncooperative parties simply stop being served, so the expected
  value at risk stays small.

The key insight we get for free: **the problem does not need a global answer.**
Two nodes only ever need to agree with *each other*. That is a bilateral
problem, and a dual-signed receipt is already the right primitive for it.

### 3.2 Shape

**A running tab per peer pair, settled periodically, with a hard exposure cap.**

- Each node keeps, per peer, a signed running total of work done in each
  direction. This is bilateral state: A's view of (A,B) and B's view of (A,B)
  must converge, and nobody else needs to care.
- Work accrues to the tab as it completes. **No settlement on the hot path** —
  this is the constraint that rules out per-request receipts. Distributed
  inference latency is the product's core property and a settlement round trip
  per request would tax every token.
- Settlement fires on a threshold (N requests, T seconds, or an amount) and is
  one dual-signed `CreditTransaction` covering the accumulated delta. The
  existing type, signing and replay protection are reused as-is.
- **Exposure cap**: a node serves a given peer only while that peer's unsettled
  tab is under a limit. Past it, service to that peer pauses until they settle.
  This is the whole defence against cross-peer double-spend: it is not
  prevented, it is *bounded per peer*, and the bound is enforced by the only
  party at risk.
- A peer that does not settle is not punished globally — it simply stops being
  served by the node it owes, which needs no consensus and cannot be gamed by a
  third party making false accusations.

### 3.3 How each hard part is answered

| Hard part | Answer |
|---|---|
| Who initiates | The **requester** co-signs at settlement, not per request. Both parties have the same tab; settlement is agreement on a delta both computed independently. A disagreement is a refusal to sign, not a dispute needing a referee. |
| Settlement failure | The tab stays owed and service pauses at the cap. **No retry queue** — the unpaid amount is bounded by the cap, and the server's remedy is to stop working, not to chase. If the peer returns and settles, service resumes. |
| Double-spend across peers | Not prevented; **bounded**. Maximum loss to any node is one exposure cap. A node whose signed obligations exceed its balance is *provably* delinquent from its own signatures — which is the ingredient for reputation later, without needing it now. |
| Migration | See §5 — needs a product decision, not a technical one. |

### 3.4 What this deliberately does not do

- No blockchain, no token, no global ledger, no consensus.
- No attempt to make the swarm-wide total meaningful. It is a set of bilateral
  relationships; there is no "total" to be correct about.
- No proof that the serving node actually ran the model. That is a separate,
  harder problem (Gensyn's `Judge` territory) and conflating it with payment is
  how both end up unbuilt.

---

## 4. What was switched off, and why now

Credits were **contentious when explaining the system to people**, and on
inspection the objection was correct: the product showed users a number,
labelled it as earnings, and let it affect the service they got — while that
number was self-minted and reconciled with nobody.

Showing a meaningless number is a smaller problem than *acting* on one. So:

- `MIN_BALANCE_FOR_INFERENCE` → `0`, which the existing `!= 0` guard already
  treats as "no floor". No remote requester is refused over credits.
- `calculate_tier` no longer reads the balance. Every requester gets the same
  tier, so per-requester concurrency isolation is preserved (one peer still
  cannot monopolise the queue) while the *advantage* from a minted number is
  gone.
- The dashboard no longer presents a balance, a leaderboard, or "earn credits"
  copy as a headline feature.

**The accounting itself keeps running.** It is harmless, it costs nothing, and
the recorded figures are the only real data about what the traffic patterns
actually look like — which is what §5's parameters have to be chosen from.

---

## 5. Open decisions before this is switched back on

These are product calls, not implementation details, and they are deliberately
left open:

1. **Migration of existing balances.** Every one was locally minted. Options:
   keep them as an opening position (nobody's number changes, books stay
   historically wrong); reset everyone to zero (clean books, users lose a number
   some of them earned honestly); or freeze them as non-transferable legacy
   credit (honest, but two balance kinds to explain). No option is free.
2. **Whether credits should gate anything at all, ever.** A swarm that simply
   serves whoever asks is simpler, easier to explain, and has no gaming surface.
   The economy is worth building only if free-riding turns out to be a real
   problem in a real network — which we have no evidence of yet, having never
   run one at a size where it could be.
3. **Exposure cap and settlement thresholds.** Should come from measured traffic
   once there is enough of it, not from a guess.

## 6. Exit criteria

Credits do not become visible or enforcing again until all of these hold:

- [ ] A credit provably moves between two machines, verified end-to-end with the
      receiving node's balance rising by exactly what the sender's fell.
- [ ] A node cannot increase its own balance by serving itself.
- [ ] Exposure to a defaulting peer is bounded and the bound is tested.
- [ ] Settlement is off the hot path, with a measurement showing per-token
      latency is unchanged.
- [ ] Migration decision (§5.1) made and implemented.
- [ ] The dashboard explains what the number means in one sentence a
      non-technical user can act on.

---

## References

- `docs/FUTURE_WORK.md` § "Credits never move between nodes" — the original finding.
- Gotchas #278, #280.
- [The Nuts and Bolts of Micropayments: A Survey](https://arxiv.org/pdf/1710.02964)
- [Credit Limits beyond Full Collateralization in Decentralized Micropayments](https://arxiv.org/pdf/2604.25913)
- [Decentralized AI Inference Markets: Bittensor, Gensyn, and Cuckoo AI](https://blockeden.xyz/blog/2025/07/28/decentralized-ai-inference-markets/)
