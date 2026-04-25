# Items 14 / 17 / 18 — Research Write-up

> Companion to `distributed_inference_speedup.md` "Deferred" and
> `next_steps.md` § 3. Written 2026-04-26 to give the user a concrete
> decision basis after the per-item details were trimmed out of the
> main plan in commit `1ea95be` (2026-04-20). Restores the full
> original definitions plus an updated complexity / fit / sequencing
> assessment against current SwarmLLM state (post-v0.1.0, after Items
> 1-8 + 12 + 13 + 16 landed).
>
> **TL;DR ranking by signal-to-effort for SwarmLLM today:**
> Item 17 (disaggregated prefill/decode) > Item 18 (per-token
> early-exit) > Item 14 (Mirror Spec). All three are large
> investments. None should land before the WAN bench (next_steps.md
> § 2) tells us whether wire size or latency is the binding
> constraint — the answer flips which item is highest-leverage.

## Item 14 — Mirror Speculative Decoding (Apple, 2025)

**What.** Bidirectional concurrency: draft speculates forward, target
*simultaneously* speculates correction paths for the draft. Two
parallel pipelines hide inter-peer RTT. Stacks naturally with SWIFT
(Item 6) which provides the early-exit signal Mirror needs.

**Source.** [arxiv 2510.13161](https://arxiv.org/abs/2510.13161),
[Apple ML blog](https://machinelearning.apple.com/research/mirror).

**Reported.** 2.8–5.8× wall-time on 14B–66B; 30% over EAGLE-3.

**Fit for SwarmLLM today.**
- Stacks on Item 6 (SWIFT) — currently shelved (slower than baseline
  on candle CPU until flash-attn-with-mask lands, see MEMORY.md
  § "Distributed inference speedup arc"). Mirror inherits SWIFT's
  blocker.
- Already have draft-target plumbing from Item 2 (distributed spec
  decoding) and Item 12 (DSD multi-segment) — both flag-gated, both
  awaiting their own WAN benches. Mirror would be a third spec-decoding
  variant; the question is whether the user wants three separately
  flag-gated paths or to consolidate.
- Apple's reported 2.8–5.8× is single-machine GPU. The P2P / WAN
  story isn't validated by the paper — Mirror's win comes from
  hiding RTT, but in our setup RTT is already hidden by Item 4
  (remote-generate fastpath, single-segment) and Item 12 (DSD,
  multi-segment), both of which keep the per-token loop on the
  serving peer.

**Complexity.** Large. Requires SWIFT to be working first; new draft
correction protocol (peer-to-peer); multi-pipeline coordinator.

**Recommendation.** **Defer.** Don't pull Mirror in until SWIFT
(Item 6) ships measurable wins on candle, and until WAN bench shows
RTT-hiding is the binding constraint. With Items 4 + 12 already
hiding most of the RTT pain, Mirror's headroom is small.

## Item 17 — Disaggregated prefill / decode (Mooncake-style for P2P)

**What.** Bandwidth-rich peer runs prefill (compute-bound), streams
chunked KV cache to a latency-good peer for decode (memory-bound).
Item 4's "remote-generate fast path" is a degenerate same-peer
version; generalizing to two different peers unlocks cases where no
single peer can do both.

**Source.** [Mooncake](https://kvcache-ai.github.io/Mooncake/),
[DistServe arxiv 2401.09670](https://arxiv.org/abs/2401.09670),
[LMCache tech report](https://lmcache.ai/tech_report.pdf).

**Reported.** DistServe: 7.4× request rate, 12.6× tighter SLO.
Together.ai: 40% on long-context.

**Fit for SwarmLLM today.**
- Direct extension of Item 8 (cross-node prefix-KV sharing).
  Item 8 already implements the chunked-KV transfer wire frame
  (`WIRE_TAG_PREFIX_KV`, now zstd-capable as of 2026-04-25), the
  trust gate, and the BLAKE3-verified hydrate path. Item 17 reuses
  the same plumbing for a different access pattern — write KV at
  prefill peer, read at decode peer, instead of write-and-read at
  decode peer.
- Falls out naturally from `Parallax` Item 16 Phase A: the DP cost
  function already understands per-segment compute_ms vs network_ms;
  extending vertices to (peer, prefill_or_decode) is a finite
  enumeration, not a new data structure.
- The Item 8 corner case "TinyLlama on GPU loopback ~100 ms slower
  due to wire-vs-prefill cost" is exactly the case Item 17 helps
  with: by SPLITTING prefill and decode peers, the wire becomes
  productive (decode never had to do prefill in the first place)
  rather than a tax.

**Complexity.** Large but lower than Mirror — the wire format
already exists. Need: (a) Parallax DP extension to enumerate
(peer, phase) vertices, (b) decode-resume semantics on the receiving
peer (non-trivial — the decode peer needs to load the KV cache as
its own cache, not just consume it), (c) prefill-result handoff
protocol.

**Recommendation.** **Highest priority of the three** if WAN bench
shows wire size is acceptable. Item 17 is the natural Item 8 +
Item 16 sequel — same primitives, larger leverage. The "no single
peer can do both" case is real on heterogeneous swarms (a 24 GB
GPU peer can prefill 7B but can't decode it long-context; a 8 GB
GPU peer can decode short-context but can't prefill). Building
this turns the swarm into a pool of capabilities rather than a
pool of identical workers.

## Item 18 — Per-token early-exit with adaptive depth (HELIOS / TIDE / DREX)

**What.** Some tokens exit at layer 12, others at layer 32 — natural
extension of SWIFT (Item 6). In P2P, early-exiting tokens skip the
trailing pipeline hops entirely, killing tail latency on
multi-segment routes.

**Source.** [HELIOS arxiv 2504.10724](https://arxiv.org/abs/2504.10724),
[DREX arxiv 2512.15705](https://arxiv.org/abs/2512.15705).

**Fit for SwarmLLM today.**
- The pipeline framing already supports per-token early termination
  via `LayerForward.draft_tokens` + `spec_logits_requested` — those
  channels ship the per-token decision back to the coordinator, which
  is the same shape as "this token exited early at hop N, broadcast
  to drop the trailing hops".
- Per-token routers either trained per-model (TIDE) or
  speculative-exit signaled (SpecEE). Either approach requires either
  a per-model trained classifier (HF dependency, model-specific
  artifacts in the registry) or runtime statistics (less effective,
  but no per-model training).
- For SwarmLLM specifically, the multi-segment case is where this
  pays off most: a 4-hop pipeline where 30% of tokens exit at hop 1
  saves 70% of the per-token RTT on those tokens. On single-segment
  inference (Items 4 / 5 fastpath) the win evaporates.

**Complexity.** Large. Per-token routers are the hard part — either
need a research-grade training pipeline (TIDE) or a SpecEE-style
runtime classifier with measurable variance. The wire piece
(broadcast-to-drop) is small.

**Recommendation.** **Defer pending compelling deployment shape.**
The win requires multi-segment pipelines (which we have, but most
single-user deployments hit the Item 4 fastpath). Wait until WAN
bench tells us how often pipelines are >2 hops in practice.
If >50% of requests are multi-segment, this becomes the highest-
ROI item. If <10%, skip.

## Sequencing Recommendation (refresh of Round 3)

Round 3 sequencing was: ~~13~~ → 12 → 16 → 14/17/18. Item 13 + 16
have landed. Items 12 + 13 are flag-gated awaiting WAN bench. So the
next active sequencing question is which of 14/17/18 to start.

**Updated order with WAN bench as gate:**

1. **Run the WAN bench first (`next_steps.md` § 2).** Two daemons on
   different cloud regions. The bench answers: (a) is wire size or
   latency the binding constraint at WAN-class RTT, (b) what fraction
   of requests assemble multi-segment pipelines in practice, and
   (c) does Item 12 (DSD) actually outperform Item 4 (remote-generate)
   on real RTTs.

2. **If wire-bound (heterogeneous-peer case dominates):** Item 17
   (disaggregated prefill/decode). Reuses Item 8 plumbing, extends
   Item 16 DP, unlocks heterogeneous-peer assembly.

3. **If multi-segment-tail-bound (>50% of requests are 3+ hops):**
   Item 18 (per-token early exit). Highest payoff on long pipelines,
   evaporates on single-segment fastpath.

4. **If RTT-bound and SWIFT lands first:** Item 14 (Mirror Spec).
   Still the lowest of the three for SwarmLLM today because Items 4
   and 12 already hide the bulk of the RTT pain.

## Anti-goals

- **Don't enable any of these by default before WAN bench.** All
  three target conditions that may not apply to typical SwarmLLM
  deployments.
- **Don't build all three in sequence.** WAN bench should pick one
  to invest in. Building 17 + 18 + 14 sequentially is 2-3 weeks of
  work for diminishing returns vs a focused single-item investment.
- **Don't skip Item 6 (SWIFT) revisit.** It blocks Item 14 and
  underpins Item 18. The candle flash-attn-with-mask gap that
  shelved SWIFT is a generic blocker, not item-specific — fixing
  it unlocks both research candidates AND any future per-token
  early-exit work.
