---
paths:
  - "src/inference/scheduler/**"
  - "src/inference/router/**"
  - "src/inference/pipeline/**"
  - "src/inference/hedging.rs"
  - "src/inference/prefetch.rs"
  - "src/inference/dsd_controller.rs"
  - "src/inference/trace.rs"
  - "src/inference/ngram_lookup.rs"
  - "src/inference/cancel.rs"
  - "src/inference/prefill_pacer.rs"
  - "src/inference/thermal.rs"
---

# Scheduling, routing and failover

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## The component that will refuse must be asked while the plan can still change

Two halves of one rule, both learned from a 16 GB processor-only Mac mini that
was assigned 36 of a 48-layer 14B, refused them at load, retried, and produced
the identical plan (gotcha #452).

→ `docs/invariants/scheduling.md`

## Room for more layers is not room for the layers already held

**`NodeCandidate::held_ranges` + `layers_it_would_add`** charge a local segment
only the layers it would ADD to what this node's live worker holds, by the
worker's own exact-key rules (`process_pool::layers_added_by`) — never its
width, and never a resident COUNT. `max_local_hostable_layers` is room for NEW
layers; reading it as "layers this node can run" 503'd the second request for
every split this node had just served (#95). The DP prices a local RUN (what
`merge_contiguous` will hand the loader), held ranges are split points, and a
re-plan after the loader's refusal never plans past the local bound again.

**A peer that publishes its ranges (`ResidentModelLayers::ranges`) is priced the
same way** — `NodeCandidate::capacity_charge` against `PublishedRoom`'s room for
NEW layers, KV of reused layers included (#99). ⚠ **`max_hostable_layers` keeps
meaning TOTAL for peers** — four consumers read it so; only the search gets the
finer figure. A peer publishing no ranges is priced as before.

→ `docs/invariants/scheduling.md`

## A plan that names this node for the whole model is a local generation

**`pipeline::local_generate::try_local_generate_fastpath`** runs a single-segment
plan assigned to this node through `ModelProcessPool::generate`, not as a
`LayerForward` per token to our own worker — which is the one path the prefix
cache, continuous batching and n-gram speculation are absent from. The span is
checked against the complete model, never assumed from the segment count.

**A missing `split_models` entry does not disqualify it.** That map's one writer
refuses to register past a ceiling on what the node OFFERS, so absence means the
ceiling was full when it scanned — not "not ours". Reading it as a disqualifier
made this path fire ZERO times in a full day's log (gotcha #638). The fallback
reads the MANIFEST, the same source `scan.rs` derives `is_first`/`is_last` from,
and requires shard 0 and the last shard so the worker is never handed a
whole-model range with no embedding table (#187). **A budget on what to OFFER
must not decide how work already assigned here is EXECUTED.**

→ `docs/invariants/scheduling.md`

## The hand-off gate proposes; the priced search decides

`assemble_pipeline_for` no longer RETURNS the whole-model hand-off. When the
priced search is going to run — this node on its processor, `parallax_routing`
on, more than one candidate — the gate's plan is held in `hand_off` and the
search chooses; the plan is taken only where the search declined to price
anything (`ProcessorRouteVerdict::NoComparison`, or the search failing to route
at all), and never where it made a real comparison and this node won.

→ `docs/invariants/scheduling.md`

## A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it

`scheduler::trusted_with_the_plaintext_prompt` is the one bar. **Three paths can
put a node on layer 0, and it is read by all three**: `delegation_target` (the
hand-off), `route_shortest_path`'s source filter, and `greedy_assign_inner`'s
first-segment narrowing. `standby_may_take` applies it to the fourth place the
assignment can happen — a standby, which is handed the segment's input on
failover.

→ `docs/invariants/scheduling.md`

## Trust is paid for work that was checked, and the check runs whenever the payment would

`router::spot_check::check_distributed_result` returns a verdict and
`settle_participant_trust` applies it — in that order, on **every** distributed
result. A peer earns `InferenceSuccess` only from a result judged well-formed,
once per REQUEST rather than once per segment, and a malformed result pays
nobody. The penalty is applied only where a single peer served the request,
because with several the fault is not attributable to any of them and docking
all of them is a way to demote honest competitors.

Sampling a check that gates a reward inverts the reward: crediting every
participant and then checking one result in twenty gave a peer returning
degenerate output **+0.005 per request**, and it climbed to the 1.0 ceiling.

→ `docs/invariants/scheduling.md`

## An unmeasured candidate is priced pessimistically, never excluded

`priced_from_a_measurement` is one predicate with one meaning — "does the cost
model have anything real about this candidate" — and both consumers must read
it the same way. They did not.

→ `docs/invariants/scheduling.md`

## A gate named for a comparison must make it, and against a route that exists

Two reports from one machine, one knot (2026-09-07, reports #017/#018).

**`pipeline_may_replace_processor_route` now takes `RoutePrices`** — the local
figure, the chain figure, and whether running the whole model here is a route
this node's memory can actually offer. It refuses a chain priced at or above
the local processor, and the caller logs the reason the gate returned rather
than a fixed sentence.

→ `docs/invariants/scheduling.md`

## A re-plan is warranted by a changed fact, never by a failed attempt

**`SwarmError::LocalMemoryUnavailable`** is what this node's own loader returns
when its memory budget refuses a model, and it is the one local failure
`should_retry_after` re-plans with no remote segment involved. Before the retry,
the router records `SharedState::note_local_memory_refusal(request_id)`;
`local_can_hold_every_layer` lets that outrank both of its estimates, so the
second plan **cannot** hand this node the whole model.

→ `docs/invariants/scheduling.md`

## The relaxation is scoped to the figures that are actually unreliable

**`parallax::CapacityBound`** says whose `max_hostable_layers` a routing pass
honours, in four rungs: `Everyone`, `PeersAtFaceValue`, `PeersUnbounded`,
`LocalUnbounded`. `assemble_pipeline_for` walks them in that order, and the
local layer budget is enforced INSIDE the DP — carried along the best path,
exactly as the capped-peer bitmask is — as well as by the exact summed check
after reconstruction.

**A capacity ceiling is a split POINT, not only a cap.** `route_shortest_path`
builds boundaries from `available_ranges` (disk) plus the model's and the
boomerang's ends; `max_hostable_layers` now also contributes the furthest a
candidate reaches from each range's start and the earliest it can start and
still reach the end. A cap can reject a proposed range but cannot propose the
one that fits, so without these a capacity-respecting route is not passed over —
it is not expressible, and every rung refuses down to the one binding nobody.

**A relaxation spends the safety margin before it spends the peer's own
number.** `max_hostable_layers_at_face_value` is the peer taken at its word with
`DELEGATE_VRAM_MARGIN` spent, and `PeersAtFaceValue` sits above
`PeersUnbounded` so a route that respects what peers actually claimed is always
preferred to one that does not. Unknown is unbounded on every rung and always
was — `max_hostable_layers` answers `None` for an absent capability, a gossiped
zero, or an uncomputable per-layer size — so a relaxation can only ever act on a
figure that is present and real.

→ `docs/invariants/scheduling.md`

## A result the peer sent and a result we made up are not the same delivery

`LayerResult::locally_constructed` is the discriminator, and `#[serde(skip)]`
plus an explicit `false` in the binary decoder is the whole mechanism: the field
cannot survive either codec, so **anything that arrived over the network reads
false by construction**. `pipeline::local::wait_for_result` reads it to choose
between `SegmentOutcome::Returned` and `SegmentOutcome::AbandonedLocally`.

→ `docs/invariants/scheduling.md`

## Latency wants an average; capacity wants a maximum

`AckRttEstimator` (RFC 6298 smoothing) and `GoodputEstimator` (a windowed max)
are the two halves of "how good is our path to this peer", and they are
deliberately opposite in every respect. **`ACK_OBSERVE_MAX_BYTES` and
`GOODPUT_SAMPLE_MIN_BYTES` are the same number**: the round-trip figure is taken
only from SMALL forwards, where the time is the peer's, and throughput only from
LARGE ones, where the time is the payload's. A sample is dominated by one or the
other and cannot measure both.

→ `docs/invariants/scheduling.md`

## What a peer costs per VISIT is not what it costs per layer

`PeerSpeed::decode_terms` fits `segment_ms ≈ fixed + slope × layers` per peer,
and `NodeCandidate::observed_fixed_ms_per_visit` carries the fixed half into
`vertex_cost` as the per-visit term — the one already multiplied by
`ASSUMED_FORWARD_PASSES` when a segment is entered per token.

**Sizing a segment small does not make it cheap.** A peer given 2 of 32 layers
took a chain from 152 ms/token to 3768: the other 30 layers cost less than those
2 (gotcha #659). A purely proportional coefficient prices those 2 layers at a
sixteenth of the peer, which is precisely why the router put them there.

⚠ **The fixed cost is NOT the round trip and must never be derived from one.**
On this fleet a peer 1043 ms away is five times faster than one at 643. Ping
measures the path; this measures what the peer spends per visit.

**The fallback is the safety argument, not a detail.** The two terms are only
separable when a peer's samples span differing segment widths — a peer holding
one shard of one model never does — and `None` prices it exactly as before.
`observed_latency_ms_per_layer` and `observed_fixed_ms_per_visit` are read
TOGETHER: the slope beside a zero fixed cost prices a peer lower than either
model alone.

→ `docs/invariants/scheduling.md`

## A reply under way is never moved to a machine that cannot continue it

`distributed::failover_can_restore_state(sequence_num)` — true only on the
PROMPT PASS — is asked by `failover_segment` BEFORE it looks for a stand-in. A
reply already under way ends with `SegmentFailoverExhausted` carrying
`cannot_resume_message`, and the machines that just failed are barred for that
request id.

**Unless the stand-in can be given the state.** `state.retained_activations`
keeps what this node sent to each segment that a standby covers, and
`assemble_replay` concatenates that history with the takeover step into ONE
forward at position 0 — a prompt pass to a machine holding no cache, so no new
message type and nothing an older peer refuses. It is keyed by LAYER RANGE, never by segment index — a segment is its range and
ranges do not shift, so nothing that inserts a segment can hand one segment's
history to another. `restorable_history` answers
`None` unless it holds positions `0..index_pos` CONTIGUOUSLY, and every way of
losing a step marks the segment unrestorable rather than shortening the replay:
a partial replay rebuilds a cache that is plausible and wrong, which is the same
invisible failure the refusal exists to prevent. The replayed payload and
position 0 move together — sending the replay at the current position rotates
every position wrongly and is equally silent.

→ `docs/invariants/scheduling.md`

## A failed request hands back the work it had already done

`SharedState::salvaged_replies` holds what a request had generated when it
died, and `router::salvaged_reply_if_lost` is the single place it is handed to
the caller — called at the one point per dispatch path where the attempt is
definitively over, which is after the retry in `dispatch_single` and after the
sole attempt on the batched path.

**Recording it must not be something a path remembers.** The salvage shipped
inside `execute_distributed`'s own decode loop, and the FIVE alternative paths
it tries first — DSD, draft-model speculative, n-gram-only, local-generate,
remote-generate — were each called with `?`, so a failure part-way through a
reply propagated straight past it: ~220 tokens generated, request failed,
nothing kept (FUTURE_WORK #88). `grep -c may_salvage` was 0 in all three
speculative files. This entry is "a helper nobody is obliged to call will
eventually not be called" firing on the salvage itself.

So it is a READ, not a call: **`pipeline::PartialReply` on the executor is
filled by the shared emit helpers** — the one place those coordinators turn
accepted tokens into reply text, and a place
`streamed_reply_text_goes_through_the_shared_emit_helpers` already obliges a new
coordinator to use — and **`keeping_the_partial` is the one choke point that
reads it**, wrapping all five calls. Recording is what emitting IS, and a sixth
path inherits both. `emit_streaming_batch` records BEFORE its
`token_tx`-is-`None` early return: the non-streamed request is the only one a
salvage is for, and it was the one leaving there having done nothing.

A salvaged reply is finalised like any other (§ "A reply a PEER generated…") —
the accumulated text has been through no scrub at all.

→ `docs/invariants/scheduling.md`

## What a peer HOLDS on disk and what it has LOADED are different facts

`NodeCapability::hosted_shards` is disk; `NodeCapability::resident_layers` is
memory. **`inference::scheduler::PeerResidency` is the single reading of the
second**, in three states: `Layers(n)` exempts those layers' weights and charges
full price beyond them, `Cold` charges everything, and `WarmAmountUnknown` — a
peer that has published nothing — keeps the older, more generous pricing.

The third state is not politeness. Reading silence as "holding nothing" charges
full weights to every node on an older build and routes around the machines best
placed to answer, which is the additive-protocol rule's exact failure. A bare
"is it warm" boolean has the opposite fault: it exempts every layer under
consideration, so a peer warm for part of a model is credited with the whole of
it.

→ `docs/invariants/scheduling.md`

## A peer advertises the memory it will HONOUR, not the memory it has

**`NodeCapability::memory_for_model_layers_mb` is the single answer to "how much
memory can this peer give a model's layers"**, and `ram_model_budget_mb` is the
figure a node without a graphics card puts behind it.

→ `docs/invariants/scheduling.md`

## A hand-off is priced as the shape it will be given, not as the whole model

**`inference::scheduler::delegated_shape_cost_ms`** is the one answer to "what
does this request cost if that peer is given `layers_to_assign` of it". The
price gate (`costs_more_than_staying_here`), the line that logs the gate's
verdict, and `privacy_cost_ms` all go through it, so none of them can price a
peer differently from the others.

→ `docs/invariants/scheduling.md`

## Delegation asks the same capacity bound routing does, and the retry it promises must exist

Three defects reported from one live node on v0.3.153, all in the path that
hands a whole model — or a boomerang's middle — to a single peer.

→ `docs/invariants/scheduling.md`

## A cap sized in units of the WORK is a ceiling on the product

**`inference::tensor_util::bytes_to_tensor` bounds its allocation by the
PAYLOAD, and deliberately has no ceiling on the element count.** The declared
shape is compared against the bytes actually present — `num_elements * 4` for
f32, `quant::q8_0_byte_len_checked` for Q8_0 — before `Vec::with_capacity` is
reached. That caps the allocation at roughly one message the transport already
accepted (`MAX_ACTIVATION_SIZE` 128 MB on the wire, `MAX_PAYLOAD` 512 MB over
worker IPC) and it is exact.

→ `docs/invariants/scheduling.md`

## A model's geometry is learned in one place, and unknown must not be silent

**`SharedState::gguf_meta_for` is the only read of `gguf_meta`**, and it learns
the geometry from the local `gguf_header.bin` on a miss.
`the_model_geometry_is_read_through_one_accessor` in
`tests/repo_consistency.rs` fails the build on a bare `gguf_meta.get(`.

⚠ **"Local" is the catch: a coordinator routing a model it holds no part of has
no geometry, so `max_hostable_layers` charged peers NOTHING for the prompt's KV
cache** — weights-only, and on the `WarmAmountUnknown` branch no bound at all.
That is gotcha #447's mechanism (a warm 6 GB card handed 24 layers of an
8,111-token prompt, dead in attention 22 s in with no standby), and the
second half of the 2026-09-21 field report.

**`SharedState::ensure_model_geometry` closes it, called from
`assemble_awaiting_dht` BEFORE the plan** — `assemble_pipeline_for` is
synchronous and this is a fetch. Guard:
`a_route_learns_the_model_geometry_before_it_prices_peer_memory`, because both
orderings compile and the wrong one is silent.

- **It adds no fetch that was not already happening.** Every distributed
  coordinator's `extract_model_cache` pulls the same header into the same
  directory, so geometry already self-healed after one request — **only the
  FIRST request per model was ever exposed.** This moves that fetch ahead of
  the plan that needed it.
- **Best effort, bounded by a TOTAL timeout** — the exception to § Timeouts,
  justified because `probe_gguf_file` retries on `NETWORK_RETRY_DELAYS`
  (~155 s), no router may wait that long, and the work has a known small size
  (6-9 MB, tokenizer-dominated). **Timing out costs nothing beyond the old
  behaviour.** A failure sets a cooldown: "HuggingFace has no such file" does
  not change between two requests a second apart.
- ⚠ **The admin `pipeline_plan` preview must NOT warm.** It passes
  `prompt_tokens: None`, so its KV term is `(0, 0)` whatever the geometry, and
  it fires once per visible model card on every shard-total change — warming
  there would put a HuggingFace fetch on a dashboard refresh.

→ `docs/invariants/scheduling.md`

## Three consumers have now read `standbys.len()` as an answer it cannot give

`scheduler::standby_covers` (does this standby hold that range),
`standby_has_room` (could it run it), and `peer_segment_has_standby` (can
anything take over what THIS peer is serving) are the three questions a plan's
standby list is actually asked. The bare count answers none of them, and each
consumer that used it was wrong in its own way:

- **#451** — a plan logged `standbys=1` beside "NO standby available for failed
  segment"; both true, and a tester lost an hour to the contradiction. Fixed by
  reporting `segments_without_standby`.
- **#464** — a count is not a statement about CAPACITY. One node was named
  standby for four segments it could run one of.
- **#465** — a count is not a statement about the SEGMENT in front of you.
  `request_has_standby` gated the ACK fast-fail on the whole request, so the
  four segments of a five-segment plan that had no backup were abandoned early
  anyway. Abandoning without a replica can only turn a slow success into a 503
  (Dean & Barroso, *The Tail at Scale*; measured in #386, where the result
  arrived 1.6 s after we gave up).

The three are related in the direction that matters: #464 made #465 WORSE, since
segments that used to carry a fictional standby now honestly carry none.

**Before using a collection's length as an answer, check the granularity the
question is asked at.** `standbys.len()` is a fact about a plan; every consumer
so far has wanted a fact about one segment.

## A standby is a capacity commitment, not just a coverage claim

`scheduler::standby_has_room(max_hostable_layers, already_committed,
segment_layers)` is asked of every standby candidate, beside
`standby_covers`. The two are the same pair of questions #452 and #454 each had
to separate: `standby_covers` asks whether a node HOLDS the range,
`standby_has_room` whether it could RUN it.

→ `docs/invariants/scheduling.md`

## A stand-in may be SEVERAL nodes, and the segment count is then read live

**`scheduler::standby_cover_for` is the single answer to "what could take this
layer range over"** — one standby holding all of it (always preferred, returned
as a one-element cover) or several that tile it between them. A cover with a
HOLE, or one short of the end, answers `None`: a partial tiling runs the reply
through layers nobody executed, which nothing downstream can detect. Ask through
it, not `standby_covers`, or a plan reports a composite-backed segment as bare.

**Offered only on the prompt pass.** Mid-reply a stand-in must be replayed the
segment's retained input history, and part 2's input is part 1's OUTPUT, which
never passed through the coordinator and was never retained. Lifting that needs
a retention scheme that does not exist.

**`forward_through_segments_inner` therefore reads `segments.len()` LIVE**, in
the loop bound and in `is_last`. A takeover splices one segment into several, so
a count cached before the loop leaves the spliced tail unrun and — worse
silently — makes `is_last` name a middle segment, and `is_last` decides WHICH
SEGMENT SAMPLES. `install_takeover` splices only AFTER the first part answers,
so an unreachable cover leaves the assignment untouched. Guard:
`the_pipelines_segment_count_is_never_cached_across_the_forward_loop`.

→ `docs/invariants/scheduling.md`

## A count and an outcome that disagree are two different questions

`scheduler::standby_covers` is the one predicate for "could this standby take
that segment over", used by `segments_without_standby` (reported with the plan)
and by `pipeline::distributed::failover_segment`'s search.

Standbys are chosen per segment, so a plan can log `standbys=1` and still have
none for the range that fails — the failure then logs `total_standbys=1` beside
"NO standby available for failed segment", and both lines are true. A tester
read the pair as a contradiction and lost an hour to it. The plan now names
`segments_without_standby`, and the failure reports
`standbys_covering_this_segment` next to the total. When a summary count cannot
answer the question a reader will ask of it, print the answer, not the count.

## The units decide whether a forward is a prefill, not the byte count

**`inference::pipeline::local::PipelineExecutor::forward_is_prefill(activation_bytes, units)`**
is the single answer to "is this forward doing a prefill?", for the deadline
(`compute_segment_timeout`) and for the DIAG that reports it. `SegmentBudget`
carries the resolved verdict (`is_prefill()`) so the log cannot contradict the
budget it is describing.

→ `docs/invariants/scheduling.md`

## Inference Router Queue

`drain_queue` only fires on `RouterCommand::Submit` / `StreamSubmit` or
`queue_notify.notified()`. **Every code path that calls
`active_count.fetch_sub(1)` on completion MUST also call
`queue_notify.notify_one()`** — otherwise queued requests beyond the
per-tier cap (Bronze=¼ of `max_concurrent_requests`) sit indefinitely
until a new Submit arrives. Four enforced sites:

- `ActivePipelineGuard::drop` (panic path) in `router/mod.rs`
- normal-completion arm in `dispatch_single`
- `execute_distributed_batch` spawn body + join-loop panic arm in
  `router/distributed_exec.rs`
- `BatchCleanup::complete_one` + `Drop` in `router/local_exec.rs`

Adding a new dispatch path that fetches active_count down without
notifying is a silent stall under load (gotcha #85).

## Scheduler Liveness Oracle

`inference/scheduler/mod.rs::gather_candidates` filters shard holders
against `connected_node_ids.contains(&node_id)` (skipping the local
node). `peer_registry` is intentionally preserved across mid-pipeline
disconnects (for reconnect attempts in `handle_connection_closed`'s
`in_active_pipeline` branch) so it's NOT the right oracle. Tests that
populate peers via `state.peer_registry.insert(...)` MUST also call
`state.connected_node_ids.insert(node_id.clone())` or the test peer
will be filtered out (gotcha #86).

## A peer's stated reason is the answer; do not substitute one of your own

Two helpers in `inference::pipeline` own what happens when a serving node
refuses a forward. Both exist because the correct handling was implemented on
the **verify** hops and missing on the **prefill** hops, in the same files, with
a comment on one of them explaining exactly why it mattered.

→ `docs/invariants/scheduling.md`

## A reply a PEER generated is finalised here, not taken as it arrives

`inference::finalize_reply_text` is the single place reply text is finalised —
control-token scrub, leading `<think>` reasoning block, stop truncation, the
newlines that step strands. It has now been missed TWICE, and each time on the
commonest distributed shape of the day:

- **`pipeline::remote_generate`**, the path taken whenever ONE peer holds the
  whole model. A reasoning model asked over the swarm answered with its raw
  scratchpad while the same request answered locally came back clean (#634).
- **All three speculative coordinators**, which share one finaliser —
  `finish_speculative` — and gave it no stops at all. Filtering EOS *ids* is
  not finalising, so a control marker the tokenizer never declared as EOS
  reached the user as text, a `<think>` block came back as the answer, and a
  caller's `stop` was ignored outright. The n-gram one is the DEFAULT
  distributed path: a node holding nothing takes it for every request
  (2026-09-18, gotcha #643).

**`PipelineExecutor::reply_stops` is the single answer to "what stops end this
reply"** — the caller's own plus the template's, warmed by
`build_prompt_with_header` so the stops always describe the template the prompt
was built from. It is a value on the executor rather than a parameter because a
parameter is something seven call sites can get wrong, and the standard
distributed loop derived only the template half and never read
`sampling_params.stop` at all.

Finalise on the COORDINATOR, never by trusting the serving node: only the
coordinator covers peers on builds that never learned to strip anything, and the
helper is documented idempotent. **`remote_generate` is the one caller that
passes an EMPTY stop set** — there the peer ran the decode, applied the caller's
stops and its own template's, and reports what matched in `matched_stop_seq`, so
re-deciding it here against stops derived for a model this node may not hold can
truncate a reply the peer correctly kept. A coordinator that sampled the tokens
ITSELF has no such peer and must pass `reply_stops`.

An UNCLOSED `<think>` is still shown, and must stay that way: nothing knows
where an unfinished thought ends.

→ `docs/invariants/api-surfaces.md` § "A reply is finalised on the coordinator"

## Single-source-of-truth helpers — Scheduling, routing and failover

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/scheduling.md`

- **`ModelProcessPool::serves_on_cpu` is the whole-model delegation precondition** — "would this request run on our processor": no usable card, told to use the processor, a build without CUDA, or a card the model does not fit.
- **A node holding every layer that would run the model on its processor lets the priced search compete with its fast path** — `assemble_pipeline_for` answers `serves_on_cpu` ONCE (a lazy `OnceCell`) and threads it into `gather_candidates`, which PRICES the local candidate rather than excluding it.
- **A peer's capacity for a prompt is weights PLUS that prompt's KV cache** — `scheduler::max_hostable_layers` takes `prompt_kv_bytes_per_layer` — the same arithmetic the worker charges at admission, f16 mirror included.
- **`inference::scheduler::delegation_target`** — the single decision to hand a WHOLE model to a peer rather than run it on this node's CPU.
- **`inference::router::distributed_exec::failure_is_penalty_worthy`** — gates `penalty_serve_failure` on (a) the assignment actually having had a remote segment and (b) the error not being locally attributable.
- **A peer that went silent is barred from THIS request's retry, and the retry happens** — every producer of `SwarmError::PeerUnresponsive` (the per-segment deadline in `pipeline/local.rs`, both fast-path arms in `remote_generate.rs`) calls `blacklist_holder_for_request` before returning, and `router::peer_went_silent` retries the variant by TYPE under `used_remote_segment`. Envoy's `previous_hosts` rule: a host that just failed is likely to keep failing, so the retry goes elsewhere or nowhere. A producer that bars nobody makes the retry wait the same deadline twice; a retry keyed on prose misses the producer whose words differ. → `docs/invariants/scheduling.md` § "A peer that went silent is barred from the retry"
