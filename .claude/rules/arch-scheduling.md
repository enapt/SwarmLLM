---
paths:
  - "src/inference/scheduler/**"
  - "src/inference/router/**"
  - "src/inference/pipeline/**"
  - "src/inference/segment_latency.rs"
  - "src/inference/prefetch.rs"
  - "src/inference/dsd_controller.rs"
  - "src/inference/trace.rs"
  - "src/inference/ngram_lookup.rs"
  - "src/inference/cancel.rs"
  - "src/inference/prefill_pacer.rs"
  - "src/inference/thermal.rs"
  - "src/inference/route_override.rs"
---

# Scheduling, routing and failover

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## The component that will refuse must be asked while the plan can still change

Admission runs at load time, too late to reshape a plan (#452). **`ModelProcessPool::max_hostable_layers_for_planning`** (+ `held_layer_ranges_for_planning`) bounds the LOCAL node from the loader's own estimator and budgets, weighed on the device the LOADER would use — never on `serves_on_cpu`, which is the SPEED answer (#444 vs #129) (`None` = unknowable, never "no room"); **`process_pool::segment_shape`** prices the segment the worker will actually map, never the whole model. On a card that bound is what a RUNNING worker can ADD — the card alone; the card/processor split is a fresh worker's, for ONE run of layers: **`fresh_run_layers_for_planning`** → `NodeCandidate::fresh_run_layers`, held to one run by the search (`parallax::local_fits`) and by `local_can_hold_every_layer` (#234).

→ `docs/invariants/scheduling.md` § "The component that will refuse must be asked while the plan can still change"

## A whole model on one peer is handed over, never driven token by token (2026-09-27)

**`remote_generate::eligible` is the single answer to "does the hand-off take this plan"**; the n-gram loop stands aside for every plan it accepts. Ask privacy with `encrypted_pipeline_for_request`.

→ `docs/invariants/scheduling.md` § "A whole model on one peer is handed over, never driven token by token (2026-09-27)"

## A node is ranked on warm work only — including this node (2026-09-27)

`PeerSpeed::ranking_ms_per_layer` ranks on the WARM coefficient only (the cold-inclusive one sizes timeouts). Keyed by node id, ours too: every path that times a segment, local ones included, records it.

→ `docs/invariants/scheduling.md` § "A node is ranked on warm work only — including this node (2026-09-27)"

## Room for more layers is not room for the layers already held

**`NodeCandidate::held_ranges` + `layers_it_would_add`** charge a local segment only the layers it would ADD (`process_pool::layers_added_by`), never its width. Peers publishing `ResidentModelLayers::ranges` go through `NodeCandidate::capacity_charge`. ⚠ `max_hostable_layers` stays TOTAL for peers.

→ `docs/invariants/scheduling.md` § "Room for more layers is not room for the layers already held (2026-09-24, #95)"

## A plan that names this node for the whole model is a local generation

**`pipeline::local_generate::try_local_generate_fastpath`** runs a single-segment plan assigned here through `ModelProcessPool::generate`, never a `LayerForward` per token. A missing `split_models` entry does not disqualify it — read the MANIFEST. A budget on what to OFFER must not decide how assigned work is EXECUTED.

→ `docs/invariants/scheduling.md` § "A plan that names this node for the whole model is a local generation"

## A warm prompt is priced warm (2026-09-26)

**`scheduler::cached_prefix::plan_prompt`** prices the prompt warm for a LOCAL whole-model vertex only (`NodeCandidate::cached_prefix_tokens`), tokenized via `pipeline::render_prompt_from_header`. **`NodeCapability::models_run_on_card`**, never `gpu.is_some()`.

→ `docs/invariants/scheduling.md` § "A warm prompt is priced warm (2026-09-26)"

## The hand-off gate proposes; the priced search decides

The whole-model hand-off waits in `hand_off` when the priced search will run; it is taken only on `ProcessorRouteVerdict::NoComparison` or a failed route, never where the search compared and this node won.

→ `docs/invariants/scheduling.md` § "The hand-off gate proposes; the priced search decides"

## A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it

`scheduler::trusted_with_the_plaintext_prompt` is the one bar, read by `delegation_target`, `route_shortest_path`'s source filter, `greedy_assign_inner`'s first-segment narrowing and `standby_may_take`.

→ `docs/invariants/scheduling.md` § "A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it"

## Trust is paid for work that was checked, and the check runs whenever the payment would

`router::spot_check::check_distributed_result` gives a verdict, `settle_participant_trust` applies it, on **every** distributed result: `InferenceSuccess` only for a well-formed result, once per REQUEST; penalty only where a single peer served it. Never sample a check that gates a reward.

→ `docs/invariants/scheduling.md` § "Trust is paid for work that was checked, and the check runs whenever the payment would"

## An unmeasured candidate is priced pessimistically, never excluded

`priced_from_a_measurement` is one predicate with one meaning, read the same way by `delegation_target` and `pipeline_may_replace_processor_route`: an unmeasured peer competes on the pessimistic `UNKNOWN_COMPUTE_MS` prior, never vetoed (#479). The BASELINE must be priced — with no price for "here", stay home. A routing test pins the local speed (`with_local_processor_speed`).

→ `docs/invariants/scheduling.md` § "An unmeasured candidate is priced pessimistically, never excluded"

## A gate named for a comparison must make it, and against a route that exists

**`pipeline_may_replace_processor_route` takes `RoutePrices`** and refuses a chain priced at or above the local processor; the caller logs the gate's own reason.

→ `docs/invariants/scheduling.md` § "A gate named for a comparison must make it, and against a route that exists"

## A re-plan is warranted by a changed fact, never by a failed attempt

`SwarmError::LocalMemoryUnavailable` is the one local failure `should_retry_after` re-plans; the router first calls `SharedState::note_local_memory_refusal(request_id)` so `local_can_hold_every_layer` stops the second plan handing this node the whole model — and, where a LOAD refused (`ModelProcessPool::note_load_refusal`), gives it fewer new layers than it refused to add and no fresh-worker split (`local_layers_refused_for_request`), or a node holding PART of a model is re-planned the ranges it just refused (#234).

→ `docs/invariants/scheduling.md` § "A re-plan is warranted by a changed fact, never by a failed attempt"

## A holder's refusal is reported as what it was (2026-10-04, #218)

A re-plan with the refusing holder barred can only describe its own search ("has gone"). **`router::report_after_a_replan`** reports the refusal: `SwarmShortOfMemory` when the refused plan went past what the holders OFFER (`plan_exceeds_offered_memory`, recorded per request at the planner's exit), else `HoldersDeclined` in their own words. Ask `a_refusal_the_caller_should_hear`, never re-derive it.

→ `docs/invariants/scheduling.md` § "A holder's refusal is reported as what it was"

## The relaxation is scoped to the figures that are actually unreliable

**`parallax::CapacityBound`** (`Everyone`, `PeersAtFaceValue`, `PeersAtCeiling`, `LocalUnbounded`) is walked in that order by `assemble_pipeline_for`; a ceiling is a split POINT too; a relaxation spends `DELEGATE_VRAM_MARGIN` (`max_hostable_layers_at_face_value`) before the peer's own number. **No rung, and no greedy pass, goes past what a peer could EVER hold** — `NodeCapability::model_memory_ceiling_mb` weighed by its own admission arithmetic (`process_pool::processor_cost_curve_for`, over the shards IT holds), clamped ONCE where candidates are built (`clamp_to_ceiling` in `gather_candidates`: `max_hostable_layers`, the face value and a published room), so delegation, the whole-model disqualifier and the standbys inherit it with the rungs; `NodeCandidate::within_ceiling` only for a figure a rung computes itself. A plan nobody's ceiling fits is `SwarmShortOfMemory` from the planner (`scheduler::short_of_memory`); greedy's second pass runs whenever a ceiling is known. Unknown is unbounded on every rung.

→ `docs/invariants/scheduling.md` § "The relaxation is scoped to the figures that are actually unreliable"

## A result the peer sent and a result we made up are not the same delivery

`LayerResult::locally_constructed` (`#[serde(skip)]`) reads false for anything that arrived over the network; `pipeline::local::wait_for_result` chooses `SegmentOutcome::Returned` or `AbandonedLocally` by it.

→ `docs/invariants/scheduling.md` § "A result the peer sent and a result we made up are not the same delivery"

## Latency wants an average; capacity wants a maximum

`AckRttEstimator` (average) and `GoodputEstimator` (max) are opposite by design; **`ACK_OBSERVE_MAX_BYTES` and `GOODPUT_SAMPLE_MIN_BYTES` are the same number** — RTT from SMALL forwards, throughput from LARGE ones.

→ `docs/invariants/scheduling.md` § "Latency wants an average; capacity wants a maximum"

## What a peer costs per VISIT is not what it costs per layer

`PeerSpeed::decode_terms` fits `fixed + slope × layers`; `NodeCandidate::observed_fixed_ms_per_visit` feeds `vertex_cost`. ⚠ The fixed cost is NOT the round trip. `observed_latency_ms_per_layer` and `observed_fixed_ms_per_visit` are read TOGETHER; `None` prices as before.

→ `docs/invariants/scheduling.md` § "What a peer costs per VISIT is not what it costs per layer"

## A forward the peer could not OPEN goes to it again, once the link is re-keyed

**`pipeline::local::wait_for_result` takes a REQUIRED `ResendOnRefusal`**: on `ForwardRefusal::Undecryptable` (a TYPE) it waits for `SessionManager::rekeyed_since` and resends once to the same node. A chained forward passes `Never`; add no second resend.

→ `docs/invariants/scheduling.md` § "A forward the peer could not open goes to it again, once the link is re-keyed"

## A reply under way is never moved to a machine that cannot continue it

`distributed::failover_can_restore_state(sequence_num)` is asked by `failover_segment` BEFORE looking for a stand-in; else `SegmentFailoverExhausted`. A stand-in is given state only by `assemble_replay` over `state.retained_activations` (keyed by LAYER RANGE; `restorable_history` must be contiguous). Never a partial replay.

→ `docs/invariants/scheduling.md` § "A reply under way is never moved to a machine that cannot continue it"

## State that belongs to an ATTEMPT is keyed by the attempt, never by the request id (2026-09-28)

State scoped to one attempt (e.g. `engine_drafter::draft_key`) is keyed by the attempt, not the request id: **a release keyed by request id reaches every worker** (#749).

→ `docs/invariants/scheduling.md` § "State that belongs to an attempt is keyed by the attempt, never by the request id (2026-09-28)"

## Speculation that does not pay steps aside — γ = 0 is an answer (2026-09-29)

**`dsd_controller::best_gamma_for_check` may choose zero guesses** (SmartSpec); `recall` / `remember` carry it to the next request. A new speculating path needs a way to stop guessing; the continuous stream remembers `best_gamma_overall`.

→ `docs/invariants/scheduling.md` § "Speculation that does not pay steps aside — γ = 0 is an answer (2026-09-29)"

## A failed request hands back the work it had already done

`SharedState::salvaged_replies` is handed back only by **`router::salvaged_reply_if_lost`**. Recording is a READ: `pipeline::PartialReply` is filled by the shared emit helpers (`streamed_reply_text_goes_through_the_shared_emit_helpers`) and `keeping_the_partial` is the one choke point.

→ `docs/invariants/scheduling.md` § "A failed request hands back the work it had already done"

## What a peer HOLDS on disk and what it has LOADED are different facts

`hosted_shards` is disk, `resident_layers` memory. **`inference::scheduler::PeerResidency` is the single reading**: `Layers(n)`, `Cold`, `WarmAmountUnknown` — silence is never "holding nothing".

→ `docs/invariants/scheduling.md` § "What a peer HOLDS on disk and what it has LOADED are different facts"

## A peer advertises the memory it will HONOUR, not the memory it has

**`NodeCapability::memory_for_model_layers_mb` is the single answer to "how much
memory can this peer give a model's layers"**, and `ram_model_budget_mb` is the
figure a node without a graphics card puts behind it. That figure (`vram::live_headroom_mb`) is never more
than the worker's own admission honours (`kv_budget::device_free_margin_bytes`; #219).

→ `docs/invariants/scheduling.md`

## A hand-off is priced as the shape it will be given, not as the whole model

**`inference::scheduler::delegated_shape_cost_ms`** is the one answer to "what
does this request cost if that peer is given `layers_to_assign` of it". The
price gate (`costs_more_than_staying_here`), the line that logs the gate's
verdict, and `privacy_cost_ms` all go through it, so none of them can price a
peer differently from the others.

→ `docs/invariants/scheduling.md`

## Delegation asks the same capacity bound routing does, and the retry it promises must exist

**`scheduler::delegation_target` gates on `max_hostable_layers`** — the bound `route_shortest_path` uses — over `delegated_layer_span(num_layers, encrypted)`, the same span `boomerang_assignment` hands over (#454), and prices the prompt through `costs_more_than_staying_here` (`parallax::vertex_cost`, the search's own function; #455).

→ `docs/invariants/scheduling.md` § "Delegation asks the same capacity bound routing does, and the retry it promises must exist"

## A cap sized in units of the WORK is a ceiling on the product

**`inference::tensor_util::bytes_to_tensor` bounds its allocation by the PAYLOAD** and deliberately has no element-count ceiling.

→ `docs/invariants/scheduling.md` § "A cap sized in units of the WORK is a ceiling on the product"

## A model's geometry is learned in one place, and unknown must not be silent

**`SharedState::gguf_meta_for` is the only read of `gguf_meta`** (`the_model_geometry_is_read_through_one_accessor`). ⚠ **`SharedState::ensure_model_geometry`** runs from `assemble_awaiting_dht` BEFORE the plan (`a_route_learns_the_model_geometry_before_it_prices_peer_memory`); the admin `pipeline_plan` preview must NOT warm.

→ `docs/invariants/scheduling.md` § "A model's geometry is learned in one place, and unknown must not be silent"

## Three consumers have now read `standbys.len()` as an answer it cannot give

Never read `standbys.len()` as an answer: ask `scheduler::standby_covers`, `standby_has_room` or `peer_segment_has_standby`, at the granularity of the question.

→ `docs/invariants/scheduling.md` § "Three consumers have now read `standbys.len()` as an answer it cannot give"

## A standby is a capacity commitment, not just a coverage claim

`scheduler::standby_has_room(max_hostable_layers, already_committed,
segment_layers)` is asked of every standby candidate, beside
`standby_covers`. The two are the same pair of questions #452 and #454 each had
to separate: `standby_covers` asks whether a node HOLDS the range,
`standby_has_room` whether it could RUN it.

→ `docs/invariants/scheduling.md`

## A stand-in may be SEVERAL nodes, and the segment count is then read live

**`scheduler::standby_cover_for` is the single answer to "what could take this range over"** (a hole answers `None`; prompt pass only). **`forward_through_segments_inner` reads `segments.len()` LIVE** (`the_pipelines_segment_count_is_never_cached_across_the_forward_loop`); `failover_segment`'s `Takeover::{Finished, Continue}` says whether the pipeline finished.

→ `docs/invariants/scheduling.md` § "A stand-in may be SEVERAL nodes, and the segment count is then read live"

## A count and an outcome that disagree are two different questions

`scheduler::standby_covers` is the one predicate behind `segments_without_standby` and `failover_segment`'s search. Print the answer, not the count.

→ `docs/invariants/scheduling.md` § "A count and an outcome that disagree are two different questions"

## A wait on a peer's first answer allows for its load, on every path (2026-10-07)

**`pipeline::LoadAllowance` is the single answer to "may this peer first have to load the model"**: a REQUIRED argument of `remote_generate::first_token_timeout` (hand-off, HTTP pool forward), asked by `SegmentBudget::for_forward`; outside tests it is built only by asking (`for_peer` / `for_segments`). The first-token wait ends when the peer's last connection closes. Rig: `examples/cold_load_test.sh`.

→ `docs/invariants/scheduling.md` § "A wait on a peer's first answer allows for its load, on every path"

## A candidate that must first load the model is CHARGED for it (2026-10-07)

**`NodeCandidate::cold_load_ms_per_layer` × `layers_it_would_add`** is `parallax::vertex_cost`'s `cold_load_ms` — once per request, never × `expected_attempts` — set in `gather_candidates` from the SAME residency reading as the memory bound (`WarmAmountUnknown` → 0). The rate is the node's own (`process_pool::LoadRate` → `NodeCapability::model_load_ms_per_gib`), else `UNMEASURED_LOAD_MS_PER_GIB`; this node is charged its own too. Rig: `cold_load_test.sh … price`.

→ `docs/invariants/scheduling.md` § "A candidate that must first load the model is charged for it"

## The units decide whether a forward is a prefill, not the byte count

**`inference::pipeline::local::PipelineExecutor::forward_is_prefill(activation_bytes, units)`**
is the single answer to "is this forward doing a prefill?", for the deadline
(`compute_segment_timeout`) and for the DIAG that reports it. `SegmentBudget`
carries the resolved verdict (`is_prefill()`) so the log cannot contradict the
budget it is describing.

→ `docs/invariants/scheduling.md`

## Inference Router Queue

**Every `active_count.fetch_sub(1)` on completion MUST also `queue_notify.notify_one()`** (four sites in `router/`); a path that skips it stalls the queue (gotcha #85).

→ `docs/invariants/scheduling.md` § "Inference Router Queue"

## Scheduler Liveness Oracle

`gather_candidates` filters on `connected_node_ids`, not `peer_registry`; tests inserting into `peer_registry` MUST also insert into `connected_node_ids` (gotcha #86).

→ `docs/invariants/scheduling.md` § "Scheduler Liveness Oracle"

## A peer's stated reason is the answer; do not substitute one of your own

**`pipeline::peer_error_from_result`** recovers a refusing peer's stated reason from its `LayerResult` (`finish_reason`, not `token_ids.is_empty()`) at the `forward_through_segments` choke point; **`failover_segment` loops over standbys** and a standby's error is that standby's failure, never the segment's output (#435); **`every_holder_would_refuse`** (only `Validation`) is asked BEFORE failing over. Every consumer of a `LayerResult` checks `finish_reason` before the payload.

→ `docs/invariants/scheduling.md` § "A peer's stated reason is the answer; do not substitute one of your own"

## A reply a PEER generated is finalised here, not taken as it arrives

**`inference::finalize_reply_text` is the single place reply text is finalised**, on the COORDINATOR; **`PipelineExecutor::reply_stops`** is the single answer to "what stops end this reply". `remote_generate` alone passes an EMPTY stop set.

→ `docs/invariants/scheduling.md` § "A reply a PEER generated is finalised here, not taken as it arrives"

## A prompt longer than ONE peer serves is that peer's limit, not the request's (2026-09-25)

**`error::longer_than_served` writes the refusal and `error::served_context_refusal` reads it** (wire format); **`SharedState::peer_served_context`** reads `NodeCapability::context_ceiling_tokens`. Below the model's limit the coordinator bars the peer and fails over (`every_holder_would_refuse`, `SwarmError::LongerThanPeerServes`).

→ `docs/invariants/scheduling.md` § "A prompt longer than ONE peer serves is that peer's limit, not the request's (2026-09-25)"

## Single-source-of-truth helpers — Scheduling, routing and failover

Each names the ONE place a decision is made; a second implementation of any is this codebase's most-repeated defect (`.claude/rules/architecture.md` § "One invariant, N paths").

- **`ModelProcessPool::serves_on_cpu` is the whole-model delegation precondition**
- **A node holding every layer that would run the model on its processor lets the priced search compete with its fast path**
- **A peer's capacity for a prompt is weights PLUS that prompt's KV cache**
- **`inference::scheduler::delegation_target`**
- **`inference::router::distributed_exec::failure_is_penalty_worthy`**
- **A peer that went silent is barred from THIS request's retry, and the retry happens**

→ `docs/invariants/scheduling.md` § "Single-source-of-truth helpers — Scheduling, routing and failover"

## A peer that fails a model on every request is ejected from its plans (2026-10-02)

**`daemon::state::peer_outliers`**: two failures in a row (fed ONLY by `record_peer_delivery`) eject a peer per model for 2 min, doubling to 30; `note_peer_completed_request` forgives. Refusals are not failures.

→ `docs/invariants/scheduling.md` § "A peer that fails a model on every request is left out of its plans for a while (2026-10-02)"

## A split none of which is ours is LED by its head (2026-10-02)

**`remote_generate::delegation_eligible` is the single answer**; `try_delegated_split` runs FIRST in `execute_distributed`, under `RoutePlanOverride::lead_here`. Local work uses **`pipeline::worker_requester`**, never a hardcoded `Requester::Owner`.

→ `docs/invariants/scheduling.md` § "A split none of which is ours is LED by its head, never driven from here (2026-10-02)"
