# Direct peer chaining for distributed inference

**Status: implemented, validated end to end on two machines (2026-08-21), ON by
default since v0.3.109** (`inference.pipeline_chaining`; set it to `false` to
run every segment through the coordinator as before). This document is the case for doing it, the
prior art it rests on, the design, and what is still unproven.

### Six defects, all found by review rather than by running it (2026-08-21)

Worth listing, because the pattern is consistent: every one is silent. None
produces an error at the point it goes wrong; each either hangs the request or
returns a plausible-looking wrong answer.

1. **The tail replied to its predecessor, not the coordinator.** A segment
   answered whoever sent it the activations. Mid-chain that node has handed the
   work on and is not waiting. Every chained request would have hung.
2. **`is_last` was asked of the head.** A run whose tail is the final segment was
   not recognised as finishing the pipeline, so the answer arrived and was walked
   past.
3. **Chaining was gated on `generated_ids.is_empty()`** — a value that
   accumulates the completion, so it is true only before the prompt pass. The
   feature was disabled for the entire per-token phase, which is where the round
   trips are and the only reason it exists. The right question is whether the
   sampler will NEED those ids: `frequency_penalty != 0 || presence_penalty != 0`.
4. **A hop that could not forward returned its own activations.** They cover only
   its layers, and the coordinator has already skipped the rest of the run — so a
   partial tensor of entirely plausible size would be fed onward as though the
   whole chain had computed it. It now sends an error.
5. **A rejected result could skip work nothing had done.** The run was marked
   complete when a reply arrived, before the check that can still reject it for a
   wrong activation shape; that failover replaces one holder, so the loop resumed
   past hops nothing had computed.
6. **Any hop's failure report was discarded.** The waiter is pinned to the tail,
   so a report from anywhere else — including the head, which the pin no longer
   covers once a chain is planned — was dropped as belonging to a request no
   longer waiting on it, and the request sat until its deadline. Every node of
   the run may now report.

A seventh, in the terminal-frame repeat added alongside: a stale copy from an
abandoned attempt could land in a retry of the same request id and kill it. The
reply stream now records which peer the current attempt is being served by. See
gotchas #349 and #350.

### Five more, found by RUNNING it on two machines (2026-08-21)

Every one of the six above was found by review; these five each needed a real
chained request between a WSL host and a Debian LXC to show themselves. Same
pattern — silent, plausible, a hang or a wrong addressee rather than an error.

8. **The tail answered its predecessor — again, one layer down.** Defect 1 was
   fixed in `dispatch/layer_forward.rs` with `reply_target`. The network
   manager's "answer on the substream the forward arrived on" shortcut is keyed
   by REQUEST ID and knew nothing about chains, so with the dispatch fix in
   place the result still went down the previous hop's substream. The stored
   channel now records the peer it came from and the shortcut is taken only
   towards that peer; a mismatch ACKs the old substream and sends the result as
   its own request. Gotcha #354.
9. **The coordinator's identity was never on the wire.** Every decoder set
   `requester_node_id: None`, so `reply_target` fell back to the SENDER — right
   for one hop, wrong for a chain. The in-process unit tests built forwards with
   the field set and could not see it. Now a `0x07` reply-to trailer,
   AAD-bound, emitted whenever the forward names its requester; the coordinator
   names itself ONLY on a chained send and every hop copies it onward, so the
   one-hop frame every released node expects is byte-identical. Behind
   `features::PIPELINE_CHAIN_V2` — a v1 peer is never chained to. The first cut
   tied 0x07 to the chain trailer and dropped it on the one hop with an empty
   remaining chain: the tail. Gotcha #354 addendum.
10. **A hop that handed the forward on never answered the request it
    received.** The coordinator's rr to the head stayed open until the 600 s
    libp2p timeout — one dangling request per chained TOKEN on that
    connection, which also feeds the "fewest pending" connection selection.
    `handle_send_tensor` now ACKs the inbound substream on hand-off, in the
    manager, where no dispatch path can forget it.
11. **"Retrying the request unchained" retried nothing.** The log line promised
    it and the code returned `PeerUnresponsive`, expecting the router to
    re-plan; the router's transient check matches other wording, and a re-plan
    would have chained again anyway. Every chained failure was a hard 503 after
    the full deadline. The pipeline now re-runs the segment unchained itself,
    carrying `truncate_kv_to` for every segment the chain touched so the same
    positions are not appended twice — verified: the re-run answered
    byte-identically to the unchained control.
12. **The re-run died 1 ms after it was sent.** Re-sending a forward for a
    request id replaced the remote's stored channel, the old substream reset,
    and the coordinator turned that `OutboundFailure` into an error result the
    NEW waiter consumed — gotcha #229's shape, unreachable by the awaiting-node
    check because both forwards went to the same node. A failure (or stale
    sweep) for a forward with a NEWER pending forward for the same request is
    now dropped as superseded, without disconnecting the peer the new attempt
    is using.

Observed on the way, not caused by chaining: the pipeline seal has never run
for a remote segment (gotcha #355, `docs/FUTURE_WORK.md`); the first tensor
forward to a freshly connected peer can vanish on one dead-in-one-direction
connection and wait out the whole deadline (gotcha #353); and a "private"
`gossip_network_id` does not isolate routing on a machine with a live node —
the DHT and the loopback probe leak public holders; a pool + private mode does
(gotcha #352).

### Validation as it stands (2026-08-21) — end to end on two machines, PASS

Topology: coordinator C and tail B are two daemons in a Debian 12 LXC
(Proxmox, i5-10500T, CPU); head A is a daemon on the WSL2 host (CPU binary).
llama-3.2-3b-instruct-q4-k-m split at layer 12: A holds shards 0-1 (layers
0-12), B holds shards 2-3 (12-28), C holds nothing. The three are a pool with
`private_mode` on C and `private_mode_allow_lan = false`, which is what actually
keeps the two LIVE nodes on the same hosts out of C's candidate set (a private
`gossip_network_id` does not — gotcha #352). `inference.pipeline_chaining = true`
on C only; the serving side needs nothing but the feature bit.

What was observed, by request id in the three logs:

- **The chain completes.** Request `1a7bd758` (32 tokens, temperature 0): C sent
  ONE forward (segment 0, chain=[B]); A computed layers 0-12, logged "chaining
  activations to the next segment" and "handing a forward onward — released the
  substream it arrived on"; B computed 12-28 and logged "result is for a
  different node than the forward came from — released that substream, sending
  the result as a new request"; C received the tail's token directly. HTTP 200
  in 8.9 s, reply `Red\nBlue\nYellow` — byte-identical to the unchained control
  (`721eeb97`).
- **Per token, not just the prompt.** Request `e5b34033` (64 tokens, temperature
  0.7 — which keeps decode on the main loop; at temperature 0 the n-gram
  speculative path runs decode with its own per-segment loop, which does NOT
  chain yet): 64 coordinator sends, all to the head with a chain, **zero** sends
  to the tail, 64 tail results received directly; A 64 hand-offs, B 64 "new
  request" replies. 12.4 s total, coherent text.
- **A chained failure costs one deadline, not the request.** With the old
  binaries on A and B (tail still answering the head), C re-ran the segment
  unchained after its 296 s deadline and answered byte-identically to the control
  (`37c05883`); before that fix every chained failure was a 503.
- **Coordinator-side A/B, same nodes and connections** (chaining forced off per
  request via a non-zero `frequency_penalty`, which disables chaining because the
  sampler then needs the generated ids): see the table below.

| arm (64 tokens, temp 0.7, 2-segment LAN split) | trial 1 | trial 2 | trial 3 | min | sends to tail |
|---|---|---|---|---|---|
| chained   | 12 832 ms | 12 788 ms | 13 443 ms | **12 788** | 0 / 64 |
| unchained | 13 716 ms | 13 393 ms | 13 659 ms | **13 393** | 64 / 64 |

Min-of-3: 4.5 % less wall time (median 6 %), ≈9 ms per token — one
coordinator round trip per token on a ~1 ms LAN with one remote hop, which is
what the design predicts. The saving is proportional to RTT × (segments − 1);
a 2-segment LAN split is the least favourable case that still exercises the
mechanism.

**Re-measured on v0.3.112 (2026-08-22), 6 trials per arm instead of 3, same
three nodes and the same in-binary switch — and the speed claim above does not
survive it.** Chained min 11.79 s / median 11.98 s; unchained min 11.76 s /
median 12.22 s. The median still favours chaining (+2.0 %, ≈3.7 ms per token,
the right sign and order for one saved LAN round trip) but the minima are a
dead heat (−0.3 %), and the spread *within* each arm is 7.9 % — larger than the
gap between them. So the honest statement is: **on a 2-segment LAN split this
setup cannot resolve the saving**, and the 4.5 % figure sits inside its own
noise. Two corrections to the paragraph above, both found by re-running it:
"every chained trial came in under every unchained one" is contradicted by its
own table (chained trial 3, 13 443 ms, is slower than unchained trial 2,
13 393 ms), and three trials was too few to say anything at this effect size.

What the re-run *does* establish, and what matters more than the milliseconds,
is that the mechanism is correct on the shipped binary: 384 hand-offs across 6
chained runs — exactly 64 per run, one per token — and **zero** across the 6
unchained runs, with both arms returning complete, coherent 64-token replies.
The flag does what it says, on every token, and does nothing when it is off.

**N>2 has now been carried on the wire (2026-08-22, v0.3.112).** A model split
three ways across two machines — head A (layers 0-12, WSL), middle D (12-21,
Proxmox), tail B (21-28, Proxmox), coordinator holding nothing — produced
`segments=3` and a two-hop chain: the head logged `next=<D> remaining=1`, the
middle `next=<B> remaining=0`, and the tail answered the coordinator. It is
per-token, not prompt-only: across two 64-token chained runs at temperature 0.7
the MIDDLE node logged exactly 128 hand-offs and the unchained control logged
none. The timing question is still open — with 2 and 3 samples the arms sat ~1%
apart, and a LAN cannot resolve even two saved round trips per token against
run-to-run noise (see the re-measurement note above). Running that split also
surfaced a routing defect that made it fail entirely at first; see gotcha #364
and the v0.3.113 changelog entry.

What is NOT covered yet, in order of value: (1) the SWARM-SPEC n-gram verify
path (`ngram_only_spec.rs`, `speculative.rs`) builds its own per-segment
forwards and does not chain — at temperature 0 that is the default decode path,
so today the flag chains the PROMPT and nothing else for a greedy request;
(2) chains longer than 3 segments, and any chain over a link slow enough to
measure the saving on (`tc netem`, needs sudo);
(3) a chain over the internet rather than a LAN; (4) the prompt-privacy
(boomerang) shape with a chain in the middle.

The 2026-08-20 single-host attempt, for the record: the coordinator planned the
split and the head handed off, but the tail never received the activations —
the same host failed the unchained control at an earlier hop too, with 112
connection closures. That turned out to be gotcha #353 (a freshly connected
peer's newest connection dead in one direction, and the rr layer picking it
first), which two machines reproduced as well, once.


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
- Have peers gossip measured RTT to their own peers. `LatencyObservation` is
  already `{ peer, ms_per_layer }` and already travels in every capability
  announcement, so adding `rtt_ms: Option<u32>` beside it is additive in the
  strict sense the protocol rule requires — an older node omits it, a newer one
  reads `None` and falls back. Every node would then hold a gossiped view of the
  inter-peer latency graph, which is exactly the edge weight the Parallax cost
  model wants.

  **And this is the one case where an inherited figure is the right figure**,
  which is worth stating because it looks like a direct contradiction of the rule
  added on 2026-08-20 (gotcha #341): only a figure THIS node measured may rank a
  peer. That rule is about a quantity that describes *our* path — a stranger's
  measurement of their route to a peer says nothing about ours, and acting on it
  demoted a GPU behind a laptop CPU. Here the quantity being asked about **is**
  their path: when deciding whether to ask peer A to hand its activations to peer
  B, A's own measurement of A→B is the only honest source, and ours is
  irrelevant. The rule is not "never trust a gossiped number", it is "never let a
  gossiped number answer a question it is not about".
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
