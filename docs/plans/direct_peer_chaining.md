# Direct peer chaining for distributed inference

**Status:** research complete, not implemented. This document is the case for
doing it, the prior art it rests on, and the design that follows.

**Why it matters:** the per-token network cost of a distributed request
currently grows *linearly with the number of shards*. That is backwards for a
system whose reason to exist is running models too large for one machine, since
those are exactly the models that need many segments. Chaining makes the cost
roughly constant in the coordinator's own round trip, whatever the chain length.

---

## 1. What we do today, and what it costs

`inference::pipeline::distributed::forward_through_segments` loops
`for idx in 0..num_segments`, sending activations to each segment and awaiting
its result before sending them on to the next. **Every hop returns to the
coordinator.** For an N-segment pipeline that is N round trips per token.

`scheduler/parallax.rs`'s module doc states this outright and names it a
deliberate departure from the paper it is adapted from:

> Parallax was designed for peer-to-peer pipeline data flow where transition
> edge cost is `rtt(peer_A, peer_B)`. SwarmLLM pipelines are coordinator-relayed:
> every hop routes through the local node.

| segments | today | chained |
|---|---|---|
| 2 | 4 × RTT | 2 × RTT + 1 inter-peer hop |
| 4 | 8 × RTT | 2 × RTT + 3 inter-peer hops |
| 8 | 16 × RTT | 2 × RTT + 7 inter-peer hops |

Measured live: a tester's request split four ways across peers in Australia and
Belgium took **30.4 s**, of which the segments themselves accounted for about
**2 s** (601 + 741 + 612 + 56 ms). Nearly all the rest is round trips, and three
quarters of those exist only because the activations came home between hops.

## 2. Prior art — this is a solved problem, done in production

**[Petals](https://alphaxiv.org/paper/2209.01188)** (BigScience) runs BLOOM-176B
over volunteer hardware on the open internet. Its client "coordinates the forward
pass by sending activations through a dynamically selected chain of servers. Each
server processes its assigned layers and **forwards intermediate activations to
the next server**." Measured: **0.83 steps/s single-batch across 14
geographically distributed servers**, an order of magnitude better than
parameter offloading.

The decisive number for us is their latency sensitivity: raising client latency
**from under 5 ms to 100 ms cost 1.66 → 1.23 steps/s**, i.e. about **211 ms per
step**. Under coordinator relay a 100 ms increase would cost `N × 200 ms`; 211 ms
is two round trips' worth, *independent of chain length*. That is the property we
are missing, visible in someone else's measurement.

Petals also answers the objection that looks fatal: **failover without the
coordinator holding every boundary.** "Clients automatically re-route requests
and restore attention caches by re-sending previous inputs to replacement
servers." The client keeps what it sent; that is enough to rebuild a replacement
server's KV cache.

**[Privacy-aware split inference over WANs](https://arxiv.org/pdf/2602.16760)**
independently arrives at the same shape and adds the piece we already have: "the
client keeps the first and last layers locally", which is precisely our
`encrypted_pipeline` boomerang. Combined with speculative decoding they reach
**~1-2 network round trips per token** and a measured **2-3x end-to-end
speedup**, assuming 50-100 ms latency and 10-100 Mbps — conditions comparable to
or worse than our LAN peers and better than our intercontinental ones.

So the target state is well established: **~2 round trips per token regardless of
how many machines hold the model**, and below that with speculation.

## 3. What must change

### 3.1 The wire

`LayerForward` gains an optional next-hop descriptor. Additive and
`#[serde(default)]`, per the protocol-evolution rule, and gated on a new
`features::PIPELINE_CHAIN` bit so a sender never hands it to a node that would
ignore it:

```rust
/// Where this segment's OUTPUT should go. `None` = back to the coordinator,
/// which is the existing behaviour and the behaviour any older node has.
pub next_hop: Option<ChainHop>,   // { node_id, peer_bytes, layer_range, is_last }
```

A node that does not advertise the bit is simply never given a `next_hop`, so it
keeps returning results to the coordinator and the request still works.

### 3.2 The serving side

`daemon/dispatch/layer_forward.rs` currently always replies with a `LayerResult`.
With a `next_hop` it instead builds the next `LayerForward` and sends it onward,
sealed for that peer. It must still report *something* to the coordinator so the
coordinator can tell a working chain from a silent one — a cheap
`ChainProgress` ack (request id, segment index) rather than the activations.

### 3.3 The coordinator

Sends one forward into the head of the chain and awaits one result from the tail,
instead of looping. `resolve_pending_layer_result` already keys waiters by
request id with an `awaiting` node, so the waiter simply expects the *last*
segment's node rather than each in turn.

### 3.4 Encryption

Today each hop is sealed for the coordinator via a per-session X25519 key. Chained
hops must be sealed peer-to-peer. **The machinery already exists**: the tensor
relay seals "ephemeral-sealed for the recipient's static key", so a sender can
seal for a peer it has no prior session with. `build_layer_forward_aad` remains
the single source of truth and gains the next-hop fields, exactly as it gained
the chunk-meta trailer in R139.

Prompt privacy is unaffected and is the reason both published designs keep the
ends at the client: with `encrypted_pipeline` on, the coordinator still holds the
first and last segments, and the chain carries only intermediate hidden states.

### 3.5 NAT, and the honest fallback

Chaining requires peer *i* to reach peer *i+1*, where today every peer needs a
route only to us. Two mitigations, and the second is what makes this safe:

1. The scheduler prefers chains of mutually reachable peers. We already track
   `reach: ReachTier` and `region_score`; a chain of peers in one region is both
   more reachable and cheaper per hop.
2. **A hop that cannot be made directly falls back to relaying through the
   coordinator** — i.e. to exactly today's cost, for that hop only. Chaining is
   therefore an optimisation over the current design rather than a replacement
   for it, and a partially-chainable swarm still gets most of the benefit.

The same fallback is the first-cut failover story: on any chain timeout, retry
the request coordinator-relayed. No new failover machinery, and never worse than
today. Petals' cache-restoring re-route is the better answer later.

### 3.6 Most of the transport already exists

Checked against the code rather than assumed, and this is the main reason the
change is smaller than it looks:

- **`NetworkCommand::SendTensor { target_peer_bytes, forward }` is generic.** It
  is not coordinator-only; any part of the daemon holding `network_tx` can send a
  `LayerForward` to any peer, and the serving path in
  `daemon/dispatch/layer_forward.rs` already holds one. No new command is needed
  to make a serving node forward onward.
- **Sealing is already per-target.** `handle_send_tensor` resolves the target and
  seals for it, so a hop sealed for peer *i+1* rather than for the coordinator
  needs no new crypto.
- **The NAT fallback is already implemented.** `try_relay_tensor` exists for
  precisely the case where "the target is unreachable", and sends an
  ephemeral-sealed relay instead. That is mitigation (2) of §3.5, already built
  and already exercised.

What is genuinely new is therefore small: the `next_hop` field and its feature
bit, the branch in the serving handler, the coordinator awaiting the tail instead
of looping, extending `build_layer_forward_aad`, and the scheduler preference
below.

### 3.7 The scheduler

This is the part with real unknowns. Chaining makes the paper's original edge cost
— `rtt(peer_A, peer_B)` — the correct model, and **we do not measure inter-peer
latency at all**. Options, cheapest first:

- Use `region_score` as a proxy: same region implies a cheap hop. Available now,
  crude.
- Have peers gossip measured RTT to their own peers. `NodeCapability.
  observed_latencies` already carries a per-peer figure, so the shape exists;
  it currently carries ms-per-layer rather than RTT. Note gotcha #341's lesson
  before trusting it: a figure another node measured describes *its* path.
- Measure the realised chain and learn, the way `peer_speed` does for segments.

Until inter-peer latency is real, restrict chaining to peers the scheduler
already believes are close to each other (same region), and keep everything else
relayed.

## 4. What NOT to build instead

Both alternatives were researched and priced; the arithmetic is in
`docs/FUTURE_WORK.md`.

- **Tensor parallelism** parallelises one token's compute but needs two
  all-reduces per layer as sequential barriers — 56 for a 28-layer model.
  Australia to Belgium is ~16,700 km, so 167 ms is the round-trip floor in fibre:
  **9.4 s per token at the theoretical limit**, with infinitely fast GPUs and
  infinite bandwidth. It belongs on NVLink at ~1-2 µs, which is why
  `tp_max_latency_ms` is 10.
- **Sequence parallelism / ring attention** parallelises prefill and is the right
  shape for a "process chunks in parallel" intuition, but it is **bandwidth**-bound,
  so moving nodes closer together does not help. Circulating K/V costs, per node
  per layer, `(N-1)/N × prompt_tokens × kv_bytes_per_token`, and against prefill
  compute of `24 d² L / N` the prompt length cancels entirely — leaving a pure
  hardware ratio of roughly **(N-1) × 2.2 Gbps** for a 3B model with GQA before
  two nodes beat one. Residential upload is 20-50 Mbps. It would work on 10-gigabit
  LAN. Published measurement agrees on the direction: ring attention's compute per
  step falls quadratically with node count while its communication falls only
  linearly.

## 5. The cheaper win that should come first

`speculative_distributed` is **off by default**, and `speculative_gamma` is 4 — so
five tokens are verified per round trip. That divides per-token network cost by
five **on any topology, with no wire change at all**, and it is the mechanism the
WAN paper credits for reaching 1-2 round trips per token.

Turning it on and measuring is a fraction of the work of chaining and is not
mutually exclusive with it: chaining reduces the *number* of round trips per
token, speculation reduces how many tokens *need* one. They multiply.

**Recommended order:** measure speculative-distributed first, then chain.

## 6. How to know it worked

- `DIAG: request complete` already reports `segments` and `total_ms`; add hops
  and realised per-hop latency.
- The falsifiable prediction: **per-token cost stops scaling with segment count.**
  Compare a 2-segment and a 4-segment assembly of the same model across the same
  peers. Today the 4-segment case costs about twice the 2-segment one; chained it
  should cost about the same.
- Verify the mechanism fired, not just that the number improved: the chained path
  must log the inter-peer hops. A faster number with no hop lines means something
  else changed — see `.claude/rules/diagnosis.md` rule 4.
