# Making split inference fast: regional pipelines

**Written 2026-09-20 against v0.3.192-alpha, from measurements on the live
swarm.** Numbers and traps: `memory/perf_baseline_0920_post192.md`, gotchas
#656-#661.

## The goal, stated as a number

Published results for a 7B over a real WAN: **8.7-9.3 tok/s at ~80 ms RTT**,
projected **15-19 tok/s at 20 ms**
([internet-scale distributed inference](https://arxiv.org/html/2604.21072v2)).
That is the target for a model split across machines none of which could hold
it. It is not "as fast as one local GPU", and it does not need to be.

Measured here today: **0.35 tok/s** on a 4-segment chain whose peers were
105-1043 ms away. The gap is almost entirely RTT.

## Why distance costs so much (the thing that is easy to get wrong)

A split does **not** reduce the compute for one token — it relocates it. The
8B is 32 layers ≈ 28 ms/token on this card whether one machine does it or four.

What a split adds is a network circuit, and that circuit is walked **once per
token**, because token *n+1* cannot start until token *n* has been sampled at
the tail and fed back to the head. The node's own trace, 32-token reply:

```
seg0 9684263580c6660f  forwards=32  total=92119ms  mean=2879ms
```

32 traversals for 32 tokens. So:

| peers are | RTT | network/token | compute | tok/s |
|---|---|---|---|---|
| same LAN | 1 ms | ~3 ms | 28 ms | **32** |
| same city | 10 ms | ~25 ms | 28 ms | **19** |
| same region | 50 ms | ~125 ms | 28 ms | **6.5** |
| intercontinental | 600 ms | ~1500 ms | 28 ms | **0.65** |

**Faster nodes make the ratio worse, not better** — GPUs shrink the 28 ms and
leave the network term untouched. Locality is the only lever that moves the
dominant term.

Crossover against not splitting at all: this node runs the 8B on its CPU at
3.96 tok/s (252 ms/token), so a 4-way GPU split beats local CPU only below
**~90 ms RTT**. Inside a country or a region like SE Asia that is comfortable.
Across continents it is worse than doing nothing.

## What is actually missing

### 1. A bug: regional placement cannot fire on a normal node

Region is auto-detected by IP geolocation into `SharedState::detected_region`.
`config.identity.region` is `None` unless an operator hand-edited config.toml.
`SharedState::effective_region_sync()` is the accessor that resolves the two,
and its own doc warns that reading the config field directly "reports 'no
region' on the common auto-detected node".

Three production sites read the config field directly:

- **`model/auto_manage/wishlist.rs:168`** — the scorer deciding which shards to
  fetch. Every `if let Some(ref my_region)` branch below it is dead on an
  auto-detected node.
- **`daemon/state/capacity.rs:127`** — the capacity announcement, which the
  accessor's doc names explicitly as a path that MUST use it.
- **`api/admin_models/listing.rs:1226,1301`** — the listing's `local_region`.

`auto_manage/manager.rs::our_region()` does it correctly, which is why
shard-level regional rarity works while the wishlist's regional logic does not.

**This is the cheapest item here and it gates everything else**: there is no
point tuning regional placement while the input is `None`.

### 2. The router cannot see the cost of the chain it is building

`NodeCandidate` carries `latency_ms` — **our** RTT to that peer — and a coarse
`region_score` (1.0 same / 0.5 adjacent / 0.2 distant / 0.7 unknown). There is
no peer-to-peer latency anywhere in the scheduler.

Segments chain peer-to-peer (seg0's elapsed covers the whole request), so a
chain's real cost includes the A→B, B→C legs. Those are invisible. The DP
prices each node in isolation and sums, which cannot distinguish "three peers
in one city" from "three peers on three continents".

### 3. Placement optimises per-shard replication, not per-chain completeness

Parallax — the paper `scheduler/parallax.rs` is named after — has **two**
phases: region-constrained placement, then per-request chain selection. Its
phase 1 "force[s] scheduling to operate on a per-region basis, constraining
layer allocation within regional boundaries to minimise cross-region data
transfer" ([Parallax](https://arxiv.org/pdf/2509.26182)).

**SwarmLLM implements phase 2 and not phase 1.** What placement it has is
per-SHARD ("each shard wants ≥2 holders in our region"), which is a different
property from per-MODEL chain completeness: every shard can be regionally
replicated while no short regional chain exists, because nothing bounds how
many distinct holders a region's coverage is spread across.

## The plan

Ordered so each stage is independently shippable and measurable, and so no
stage depends on a later one.

### Stage 0 — Measure whether the fleet can do this at all (no code)

**Decision-relevant unknown**: are there ≥2 peers within ~80 ms of *each other*
holding complementary shards? If the fleet is 9 peers in 9 places, regional
pods are impossible today and Stages 2-3 change shape.

We cannot answer it yet — we only measure RTT to peers, never between them.
Cheapest probe: take RTT vectors from the two vantage points we control (local
+ Proxmox) and from any tester willing to paste theirs.

**Exit criterion**: an inter-peer RTT matrix, and per-region model coverage.
Everything below is sized by that answer.

### Stage 1 — Fix the region bug (hours)

Point the three sites at `effective_region_sync()` / `effective_region()`. Add
a `tests/repo_consistency.rs` guard forbidding a direct read of
`config.identity.region` outside the accessor and the precedence logic in
`background.rs` — the same shape as
`prompt_privacy_is_never_re_derived_from_the_per_model_map`, with a planted
violation.

**Verify by**: regional holder counts becoming non-zero on this node, and the
capacity announcement carrying a region. Not by reading the diff.

### Stage 2 — Give the scheduler inter-peer latency (the key enabler)

Adopt **network coordinates** ([Vivaldi](https://dl.acm.org/doi/10.1145/1030194.1015471)):
each node maintains a low-dimensional coordinate from the RTTs it already
measures, and publishes it. Any node can then estimate RTT(A,B) as the distance
between coordinates, with no extra probing — which is exactly the fact the cost
model is missing.

- Carry it in `NodeCapability` as a small `#[serde(default)]` vector behind a
  new `features` bit, per the additive-protocol rule. No `PROTOCOL_VERSION`
  bump. Nodes that do not publish one are priced as today.
- Then `parallax::vertex_cost` can charge the **edge** (A→B) rather than only
  the vertex, and `region_score` becomes a fallback rather than the only signal.

⚠ **Known limitation, decided deliberately**: the Azureus study found Vivaldi
poor at picking the single *closest* node; [Pharos](https://en.wikipedia.org/wiki/Pharos_network_coordinates)
adds a local-cluster overlay for that. We do not need fine-grained ordering —
we need to tell 20 ms from 600 ms, which Vivaldi does well. If fine ordering is
ever needed, Pharos's two-tier scheme is the documented answer.

**This is the highest-leverage change**, because it is what makes every later
decision decidable rather than guessed.

### Stage 3 — Placement: regional chain-completeness

Change the acquisition objective from "this shard is rare in my region" to
**"acquiring this shard shortens or completes a chain for a model in demand,
within my latency neighbourhood"**.

Parallax does this with a global scheduler. **SwarmLLM has none, so the
swarm-native form is a self-interested acquisition heuristic**: each node scores
a candidate shard by how much it reduces the expected number of
cross-neighbourhood hops for the models its neighbourhood wants. The machinery
exists — `wishlist`, `foreign_wishlist` (cross-pool demand), `region_shard_summaries`,
`swarm_capacity`. What is new is the objective.

I did not find this published. Centralised region-constrained placement is
established; a decentralised acquisition rule that converges to it is the part
worth building carefully and worth writing up.

⚠ Convergence is the risk: every node independently chasing the same gap can
oscillate or herd. The existing `shard_download_claims` and backoff machinery
are the obvious damping, and this needs a simulation before it touches the
fleet.

### Stage 4 — Speculation, to amortise whatever distance remains

Only meaningful once Stage 2-3 have brought RTT down: the DSD paper's own
operating envelope is `3t₀ < t₁ < 10t₀`, and with t₀ ≈ 7 ms per segment that
means t₁ ≈ 20-70 ms. Today's fleet sits at 40-85× t₀, far outside it.

Order within the stage, by deployability:

1. **Self-speculative first — no draft model, no training.**
   [SWIFT](https://arxiv.org/pdf/2410.06916) (on-the-fly adaptive layer
   skipping, **1.3-1.6×**) and [CLaSp](https://arxiv.org/pdf/2505.24196)
   (plug-and-play DP layer skip, **1.3-1.7×** on LLaMA-3). The worker already
   takes `--swift-self-speculative` and it is `false`. **This is the only
   variant that can ship on by default**, because it asks the user for nothing.
   LayerSkip needs finetuning and EAGLE/Medusa need trained heads — neither
   survives "arbitrary GGUF off HuggingFace".
2. **Then tree/pipelined speculation.** `pipeline/dsd.rs` already implements
   DSD (**2.1-2.6×**) citing arXiv 2511.11733;
   [FlowSpec](https://arxiv.org/abs/2507.02620) reports **1.37-1.73×** and is
   aimed precisely at *sparse* edge requests — the single interactive user, where
   there is no other traffic to fill the pipeline.
3. **Unblock the gates.** DSD is unreachable behind four: `speculative_decoding`
   off, `decentralized_spec_decoding` off, a required `draft_model_path`, and
   `temperature == 0` via `speculative_common_eligible`. That last one the
   n-gram path already diagnosed and removed for itself — its comment notes no
   client sends greedy by default (0.7 OpenAI, 1.0 Anthropic). DSD still
   inherits it.

### Stage 5 — Fewer crossings per token

- **Contiguity as a rebalancing goal**, so N peers cost N−1 crossings. Two peers
  produced *four* segments in the measured chain purely because their holdings
  interleaved.
- **A minimum segment size**, so the DP stops manufacturing 4-layer slices. With
  `parallax_partial_ranges` on it gave a third peer 4 layers and the request
  never returned in 580 s.
- **Tail→head directly** for the next token, with the coordinator notified
  asynchronously for streaming, if it is currently in the per-token path.

## Expected stacking, honestly

| lever | factor | confidence |
|---|---|---|
| locality, 600 ms → 50 ms | ~10× | high — it is arithmetic |
| self-speculative (SWIFT/CLaSp) | 1.3-1.7× | published, unmeasured here |
| tree/DSD speculation | 1.4-2.6× | published, unmeasured here |
| fewer crossings | ~1.5× | inferred from the 4-vs-2 segment case |

Consistent with the 8.7-9.3 tok/s at 80 ms the literature reports. **Locality
dominates everything else combined**, which is why Stages 1-3 come before the
speculation work however attractive the latter is.

## What NOT to do

- **Do not enable `parallax_partial_ranges`.** Measured worse, not predicted
  worse: a third peer recruited for four layers, request never returned.
- **Do not tune `ASSUMED_FORWARD_PASSES` or `transfer_ms` against each other** —
  FUTURE_WORK already warns they act on the same decision.
- **Do not let a slow-peer signal downgrade prompt privacy.** Already decided
  and researched; a hostile peer that is slow on purpose would win the plaintext.

## Open unknowns

- The inter-peer RTT matrix (Stage 0). Everything is sized by it.
- SWIFT/CLaSp acceptance rates on **quantised GGUF** models — the papers use
  full-precision HF checkpoints. Layer-skip drafting on Q4_K_M is unvalidated.
- Whether `region_score`'s 1.0/0.5/0.2/0.7 buckets survive contact with real
  coordinates, or should be replaced by them outright.
- Whether the coordinator currently sits in the per-token return path
  (Stage 5, third bullet) — not yet checked.
