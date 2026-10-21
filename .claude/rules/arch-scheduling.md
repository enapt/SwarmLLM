---
paths:
  - "src/inference/scheduler/**"
  - "src/inference/router/**"
  - "src/inference/pipeline/**"
  - "src/inference/segment_latency.rs"
  - "src/inference/prefetch.rs"
  - "src/inference/dsd_controller.rs"
  - "src/inference/draft_tree.rs"
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

Admission runs at load time, too late to reshape a plan (#452). **`ModelProcessPool::max_hostable_layers_for_planning`** (+ `held_layer_ranges_for_planning`) bounds the LOCAL node from the loader's own estimator and budgets, weighed on the device the LOADER would use — never on `serves_on_cpu`, which is the SPEED answer (#444 vs #129) (`None` = unknowable, never "no room"); **`process_pool::segment_shape`** prices the segment the worker will actually map, never the whole model. On a card that bound is what a RUNNING worker can ADD — the card alone; the card/processor split is a fresh worker's, for ONE run of layers: **`fresh_run_layers_for_planning`** → `NodeCandidate::fresh_run_layers`, held to one run by the search (`parallax::local_fits`) and by `local_can_hold_every_layer` (#234), offered only where the spawn would admit its processor share for the same requester (#250).

→ `docs/invariants/scheduling.md` § "The component that will refuse must be asked while the plan can still change"

## A split leaves room for the model that guesses for it (2026-10-14, #275)

Where this node cannot hold every layer, each planner FIRST plans with its bound cut to leave the drafter room (**`PipelineScheduler::candidates_leaving_room_for_the_drafter`** → `ModelProcessPool::max_hostable_layers_beside_guest`) — ONLY where every peer is held to what it offers (parallax at `CapacityBound::Everyone`, greedy's strict pass in `greedy_assign_beside`). ⚠ Never on a rung or pass that relaxes a PEER's bound: the cut moves layers onto peers, and a relaxation is a gamble justified only by there being no other plan (tried at `PeersAtCeiling` live, the peer refused 34 layers and the reply was lost). No route that way, plan as before. The drafter is chosen only if it would load beside the layers the PLAN loads here (`guest_would_be_admitted_beside`, `engine_drafter::TargetHere` REQUIRED), never only those loaded now; planner and request choose through ONE `engine_drafter::chosen`. The room, the choice and the spawn all price the REPLY's cache (`model_worker::drafter_cache_positions` → `vram::KvCharge::Conversation`), never a typical conversation's. A/B `SWARMLLM_DRAFTER_ROOM=0`, `SWARMLLM_DRAFTER_ADMISSION=0`, `SWARMLLM_DRAFTER_KV_CHARGE=typical`, `SWARMLLM_DRAFTER_BESIDE=loaded`.

→ `docs/invariants/scheduling.md` § "A split leaves room for the model that guesses for it"

## A layer is priced as the model it belongs to (2026-10-12, #250)

**`NodeCandidate::prior_ms_per_layer`** is the one prior from a stated speed (`vertex_cost`, greedy): the 7B-reference layer time × `layer_work`, the model's active bytes per layer over the reference's (`scheduler::layer_work_for` ← `GgufTensorMeta::active_bytes_per_layer`, experts at `expert_used_count / expert_count`). `est_tokens_per_sec` stays in 7B units for ratios and sorts. A/B `SWARMLLM_LAYER_PRICE=reference`. **A measured figure is kept per (peer, model)** (`peer_speed::ModelSpeed`): every pricing or deadline read goes through `SharedState::peer_model_speed`, and a model the peer has not run for us is unmeasured — never scaled from another model's figure. Gossip carries `layer_work` beside the raw figure (`LatencyObservation`). **A peer's OWN timing is a floor under that prior, never a discount** (#3 (f)): each worker times the decode steps it runs FOR PEERS (never the owner's — they would tell the swarm when the owner used a model) in `ModelProcessPool::forward_for_request` — a step is ONE position (`forward_is_one_decode_step`, counted by `tensor_util::positions_carried`), an n-gram miss round's one-position check included — into `OwnDecodeSpeed` (the FASTEST of 32 — waiting only adds, so a peer cannot queue a node into publishing a slow figure; published after 4, withdrawn an hour after the last), gossiped as `ResidentModelLayers::decode_us_per_layer`, read as `NodeCandidate::stated_ms_per_layer` and taken as `max(it, stated speed)` — it may price a peer slower than its stated speed, never faster, and anything THIS node measured outranks it. Rig: `split_rig.sh owntiming` (`RIG_NETNS=1` beside a live node).

→ `docs/invariants/scheduling.md` § "A layer is priced as the model it belongs to"

## A whole model on one peer is handed over, never driven token by token (2026-09-27)

**`remote_generate::eligible` is the single answer to "does the hand-off take this plan"**; the n-gram loop stands aside for every plan it accepts. Ask privacy with `encrypted_pipeline_for_request`.

→ `docs/invariants/scheduling.md` § "A whole model on one peer is handed over, never driven token by token (2026-09-27)"

## A node is ranked on warm work only — including this node (2026-09-27)

`PeerSpeed::ranking_ms_per_layer` ranks on the WARM coefficient only (the cold-inclusive one sizes timeouts). Keyed by node id, ours too: every path that times a segment, local ones included, records it. **A prompt is PRICED by a third figure, `ModelSpeed::prompt_ms_per_layer_byte`** — warm forwards of at least `PROMPT_RATE_MIN_BYTES`, a smaller one may only lower it (BBR's app-limited rule): a load or a round trip scaled to a long prompt priced one route at 17.6 h (#287). This node reads it too.

→ `docs/invariants/scheduling.md` § "A node is ranked on warm work only — including this node (2026-09-27)", § "A prompt is priced from warm forwards that carried enough to measure a rate"

## Room for more layers is not room for the layers already held

**`NodeCandidate::held_ranges` + `layers_it_would_add`** charge a local segment only the layers it would ADD (`process_pool::layers_added_by`), never its width. Peers publishing `ResidentModelLayers::ranges` go through `NodeCandidate::capacity_charge`. ⚠ `max_hostable_layers` stays TOTAL for peers. A candidate's OWN held range is a vertex on EVERY pass — it cuts nobody else's range (#3 (e): offered only on no route, two peers whose workers chained were planned a 29-layer re-read). A peer's room and a held range's charge are compared UNROUNDED (`PublishedRoom`: the held layers' KV is a fraction of one layer). Other candidates' ranges are cut at held endpoints only on a rung with no route (`parallax::route_over(.., meet_held)`, #275's rig), so no existing route is cut at another machine's held range.

→ `docs/invariants/scheduling.md` § "Room for more layers is not room for the layers already held (2026-09-24, #95)"

## A plan that names this node for the whole model is a local generation

**`pipeline::local_generate::try_local_generate_fastpath`** runs a single-segment plan assigned here through `ModelProcessPool::generate`, never a `LayerForward` per token. A missing `split_models` entry does not disqualify it — read the MANIFEST. A budget on what to OFFER must not decide how assigned work is EXECUTED.

→ `docs/invariants/scheduling.md` § "A plan that names this node for the whole model is a local generation"

## A warm prompt is priced warm (2026-09-26)

**`scheduler::cached_prefix::plan_prompt`** prices the prompt warm for a LOCAL whole-model vertex (`NodeCandidate::cached_prefix_tokens`), for a PEER's whole-model vertex from what this node handed it in the last 300 s (`PromptPlan::handed_off` ← `pipeline::handoff_prompt_cache`, only under `remote_generate::request_may_be_handed_off` and while the peer holds the whole model resident; never from its opt-in announcement; #3 (c)), and for a split SEGMENT this node believes holds its opening for exactly that range (`PromptPlan::split_openings` ← `pipeline::held_openings`, the beliefs a prompt pass resumes from; #10), tokenized via `pipeline::render_prompt_from_header`. Both routers read **`NodeCandidate::prompt_tokens_held`**; greedy prices the UNREAD prompt at `parallax::prefill_ms_per_layer`, never a bare credit. **`NodeCapability::models_run_on_card`**, never `gpu.is_some()`.

→ `docs/invariants/scheduling.md` § "A warm prompt is priced warm (2026-09-26)"

## The hand-off gate proposes; the priced search decides

The whole-model hand-off waits in `hand_off` when the priced search will run; it is taken only on `ProcessorRouteVerdict::NoComparison` or a failed route, never where the search compared and this node won.

→ `docs/invariants/scheduling.md` § "The hand-off gate proposes; the priced search decides"

## A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it

`scheduler::trusted_with_the_plaintext_prompt` is the one bar, read by `delegation_target`, `route_shortest_path`'s source filter, `greedy_assign_inner`'s first-segment narrowing and `standby_may_take`.

→ `docs/invariants/scheduling.md` § "A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it"

## Prompt privacy embeds with the first segment's own loader, or refuses (2026-10-20, #316)

Under `SharedState::local_embedding_privacy` (the one, live reading) token ids never reach another computer: **`PipelineExecutor::prompt_embedder`** embeds ONLY where **`segment_zero_may_leave_this_node`** (a first segment run here reads token ids — embedded, it skipped the adapter, tensor-parallel and prompt-cache paths), with the first segment's loader asked for no layers (`shard_loader::load_model_range`, `SplitModel::embed_ids`, loaded on demand by `local_embedder_for`), else refuses with `EmbeddingPrivacyUnavailable`. The fast paths stand aside in `fastpath_request_disqualified`; a stand-in for segment 0 needs `FORWARD_PRE_EMBEDDED`; no tensor-parallel group takes an embedded first segment (`tensor_parallel_group_for`). Every WHOLE hand-off — the hand-off, a led split, the HTTP forward to a pool member — asks **`SharedState::privacy_refuses_a_whole_hand_off`**. The prompt pass is read from position 0 on every path.

→ `docs/invariants/scheduling.md` § "Prompt privacy embeds the prompt with the first segment's own loader, or refuses"

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

`SwarmError::LocalMemoryUnavailable` is the one local failure `should_retry_after` re-plans; the router first calls `SharedState::note_local_memory_refusal(request_id)` so `local_can_hold_every_layer` stops the second plan handing this node the whole model — and, where a LOAD refused (`ModelProcessPool::note_load_refusal`), gives it fewer new layers than it refused to add and no fresh-worker split (`local_layers_refused_for_request`), or a node holding PART of a model is re-planned the ranges it just refused (#234). Where its conversation memory refused the PROMPT (`ModelProcessPool::admit_prompt`), the re-plan holds ALL its layers, held ones included, to the layers that fit (`local_layers_fitting_for_request` → `NodeCandidate::prompt_memory_refused`, under which `capacity_charge` is the width; #263). After either refusal, no planner's relaxed pass lifts this node's bound (`NodeCandidate::memory_refused_this_request`, greedy's `cap_of`). **A PEER that refused the prompt when asked before the pass** (#252 (b)) gets the same treatment in `gather_candidates` — `max_hostable_layers_at_ceiling` held to its `layers_that_fit` (`SharedState::peer_layers_fitting_for_request`), both flags set — and is never barred.

→ `docs/invariants/scheduling.md` § "A re-plan is warranted by a changed fact, never by a failed attempt"

## A holder's refusal is reported as what it was (2026-10-04, #218)

A re-plan with the refusing holder barred can only describe its own search ("has gone"). **`router::report_after_a_replan`** reports the refusal: `SwarmShortOfMemory` when the refused plan went past what the holders OFFER (`plan_exceeds_offered_memory`, recorded per request at the planner's exit), else `HoldersDeclined` in their own words. Ask `a_refusal_the_caller_should_hear`, never re-derive it.

→ `docs/invariants/scheduling.md` § "A holder's refusal is reported as what it was"

## The relaxation is scoped to the figures that are actually unreliable

**`parallax::CapacityBound`** (`Everyone`, `PeersAtFaceValue`, `PeersAtCeiling`, `LocalUnbounded`) is walked in that order by `assemble_pipeline_for`; a ceiling is a split POINT too; a relaxation spends `DELEGATE_VRAM_MARGIN` (`max_hostable_layers_at_face_value`) before the peer's own number. **No rung, and no greedy pass, goes past what a peer could EVER hold** — `NodeCapability::model_memory_ceiling_mb` weighed by its own admission arithmetic (`process_pool::ModelProcessPool::segment_cost_curve`, over the shards IT holds), clamped ONCE where candidates are built (`clamp_to_ceiling` in `gather_candidates`: `max_hostable_layers`, the face value and a published room), so delegation, the whole-model disqualifier and the standbys inherit it with the rungs; `NodeCandidate::within_ceiling` only for a figure a rung computes itself. A plan nobody's ceiling fits is `SwarmShortOfMemory` from the planner (`scheduler::short_of_memory`); greedy's second pass runs whenever a ceiling is known. Unknown is unbounded on every rung.

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

## A reply is priced for no more passes than it may take (2026-10-18, #3)

**`NodeCandidate::forward_passes` is the single answer to "how many passes are the per-token costs charged for"**: this model's mean recent reply length (`ReplyLengths::expected`, written ONLY by `publish_request_trace` from finished replies, #3 (b)), else `ASSUMED_FORWARD_PASSES`; then lowered — never raised — by the request's `max_tokens` (`PromptPlan::with_reply_of_at_most`, set once in `distributed_exec`). `vertex_cost`, greedy's spread of the once-per-request terms and the recorded `assumed_forward_passes` all read it; never the bare constant. `None` (preview, tests) prices as before.

→ `docs/invariants/scheduling.md` § "A reply is priced for no more passes than it may take"

## A split is priced for the rounds it takes (2026-10-19, #3 (d))

**`NodeCandidate::rounds_entered` is the single answer to "how many rounds are a split segment's per-token costs charged for"**: `forward_passes()` ÷ the tokens a round of this model's recent splits yielded (`SplitRounds::tokens_per_round`, never below 1), for a SEGMENT of a chain driven from here; a whole model and a led chain keep the passes. Rounds are counted where each LEAVES (`RequestTrace::note_decode_round`: `forward_through_segments` past the prompt pass, `forward_verify_through_segments`, `dsd_stream`'s chunk) and recorded ONLY by `PipelineExecutor::execute`. **`pipeline::request_may_guess_ahead` is the one request-side question** — the router asks it into `PromptPlan::with_guessing_ahead`, the executor before recording. A new decode path sends through the counted functions or counts its own rounds.

→ `docs/invariants/scheduling.md` § "A split is priced for the rounds it takes, not the tokens it writes"

## A forward the peer could not OPEN goes to it again, once the link is re-keyed

**`pipeline::local::wait_for_result` takes a REQUIRED `ResendOnRefusal`**: on `ForwardRefusal::Undecryptable` (a TYPE) it waits for `SessionManager::rekeyed_since` and resends once to the same node. A chained forward passes `Never`; add no second resend.

→ `docs/invariants/scheduling.md` § "A forward the peer could not open goes to it again, once the link is re-keyed"

## A reply whose machine failed mid-stream is CONTINUED from what its reader received (2026-10-08)

**`router::continuation_after` is the single answer** to "may a reply that already reached its reader go on after a failure": the re-plan's failure classes plus `ReplyTruncated`, at most `MAX_CONTINUATIONS`, from `RequestTrace::streamed_so_far` — text recorded by `StreamingTokenTx` only after a send SUCCEEDED. The continued request carries `InferenceRequest::continuation`, honoured ONLY in `PipelineExecutor::build_prompt_with_header` (every path's prompt); a path rendering elsewhere stands aside (`delegation_eligible`, `local_executor_serves`). Rig: `split_rig.sh continue` (#236).

→ `docs/invariants/scheduling.md` § "A reply whose machine failed mid-stream is continued"

## A reply under way is never moved to a machine that cannot continue it

`distributed::failover_can_restore_state(sequence_num)` is asked by `failover_segment` BEFORE looking for a stand-in; else `SegmentFailoverExhausted`. A stand-in is given state only by `assemble_replay` over `state.retained_activations` (keyed by LAYER RANGE; `restorable_history` must be contiguous). Never a partial replay.

→ `docs/invariants/scheduling.md` § "A reply under way is never moved to a machine that cannot continue it"

## State that belongs to an ATTEMPT is keyed by the attempt, never by the request id (2026-09-28)

State scoped to one attempt (e.g. `engine_drafter::draft_key`) is keyed by the attempt, not the request id: **a release keyed by request id reaches every worker** (#749).

→ `docs/invariants/scheduling.md` § "State that belongs to an attempt is keyed by the attempt, never by the request id (2026-09-28)"

## Speculation that does not pay steps aside — γ = 0 is an answer (2026-09-29)

**`dsd_controller::best_gamma_for_check` may choose zero guesses** (SmartSpec); `recall` / `remember` carry it to the next request. A new speculating path needs a way to stop guessing; the continuous stream remembers `best_gamma_overall`. A remembered zero must not come from ONE stalled check: every check cost goes through `CheckCost::record`, which holds an outlier back a round (#244).

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

**`NodeCandidate::cold_load_ms(range)`** (`cold_load_ms_per_layer` × `layers_it_would_add`) is the one formula: `parallax::vertex_cost`'s `cold_load_ms` — once per request, never × `expected_attempts` — AND the greedy fallback's `estimated_cost_per_layer` (÷ covered ÷ `ASSUMED_FORWARD_PASSES`; greedy plans every split whose holders overlap, #10). Set in `gather_candidates` from the SAME residency reading as the memory bound (`WarmAmountUnknown` → 0). The rate is the node's own (`process_pool::LoadRate` → `NodeCapability::model_load_ms_per_gib`), kept across restarts (`daemon::state::load_rate`: an update put every node back on the prior), else `UNMEASURED_LOAD_MS_PER_GIB`; this node is charged its own too. Rigs: `cold_load_test.sh … price`, `split_rig.sh splitcache` with `SPARE=1`.

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

- **`PipelineScheduler::local_speed_off_the_card` is this node's speed for a model its card cannot hold whole** — the processor's, blended with the card's for the layers a card/processor split puts there, fresh or resident (`ModelProcessPool::card_share_for_planning` → `split_tokens_per_sec`, harmonic by layer share). The search's local candidate AND the hand-off gate's baseline (`DelegationInput::local_cpu_tokens_per_sec`) read it (#129).
- **`ModelProcessPool::serves_on_cpu` is the whole-model delegation precondition**
- **A node holding every layer that would run the model on its processor lets the priced search compete with its fast path**
- **A peer's capacity for a prompt is weights PLUS that prompt's KV cache** — the weights at **`scheduler::weight_bytes_per_layer`**: the file over its layers, less the tensors its loader keeps in host memory where the peer's layers run on its card (`GgufTensorMeta::host_resident_bytes`, #296), in the bound, the published room, the booking and `allocate_offline`; a load's price keeps the file's figure.
- **`inference::scheduler::delegation_target`**
- **`inference::router::distributed_exec::failure_is_penalty_worthy`**
- **A peer that went silent is barred from THIS request's retry, and the retry happens**
- **`RequestTrace::finish_from`** — the one reading of a request's result as its outcome, on every router completion path: a reader that left is `Cancelled`, and a failure counts the tokens its reader was shown. **`router::failure_worth_listing`** — what the diagnostics list as a failure (never a reader who left; the first PEER segment as the server, never this node). → § "A request its reader stopped is cancelled, not failed"

→ `docs/invariants/scheduling.md` § "Single-source-of-truth helpers — Scheduling, routing and failover"

## A peer that fails a model on every request is ejected from its plans (2026-10-02)

**`daemon::state::peer_outliers`**: two failures in a row (fed ONLY by `record_peer_delivery`) eject a peer per model for 2 min, doubling to 30; `note_peer_completed_request` forgives. Refusals are not failures.

→ `docs/invariants/scheduling.md` § "A peer that fails a model on every request is left out of its plans for a while (2026-10-02)"

## A split none of which is ours is LED by its head (2026-10-02)

**`remote_generate::delegation_eligible` is the single answer**; its conditions on the REQUEST are **`request_may_be_led_by_its_head`**, which the planner asks too. `try_delegated_split` runs FIRST in `execute_distributed`, under `RoutePlanOverride::lead_here`. Local work uses **`pipeline::worker_requester`**, never a hardcoded `Requester::Owner`.

→ `docs/invariants/scheduling.md` § "A split none of which is ours is LED by its head, never driven from here (2026-10-02)"

**A requester holding a drafter guesses for it, and the head waits for a window only where it pays (2026-10-20/21, #318).** `pipeline::requester_guesses::start` offers guesses (`DelegatedSplit::guesses`) only to a head advertising `features::REQUESTER_GUESSES` (bit 28) — NOT in `ALL`: advertised only where `requester_guesses::takes_guesses` says the head would open a slot, the one answer both ask. The head's slot (`daemon::state::requester_guesses`) is opened ONLY by `handle_delegated_split` and offered to ONLY by dispatch's `RequesterGuesses` arm, for the authenticated peer that handed the reply over. The n-gram loop checks the newest window before prompt lookup and records every token BEFORE emitting it (the stream carries the ids, `StreamingToken::reply_ids`). A continued reply takes none. The head checks only as many guesses as its own fitted check cost says pay (`ngram_only_spec::choose_guess_gamma` → `dsd_controller::best_gamma_for_check`; zero is an answer). The stream is HANDED its slot (`OpenSlot::slot`), never finds one by the requester-chosen wire id. **A wait is priced, never assumed** (#318 (a)): only where nothing else would be checked, only for a window `GuessSlot::window_due_in` says is due (the requester's measured turnaround), and only while **`ngram_only_spec::worth_waiting_ms`** says it beats a one-token pass — `W < E·(P_1 + α·P_g)/(1 + α·E) − P_g`, never the plan's `P·(E − 1)`, which forgot the window still counts after a one-token pass. **A `swarm_route` instruction goes WITH the hand-off** (#318 (b)): only `lead_here` keeps a request here; `DelegatedSplit::route` goes only to a head advertising `features::CARRIED_ROUTE` (bit 30), which reads it from its own side (`route_override::led_override` — every part only shrinks; a route it could not have been planned under is refused before a token). **`remote_generate::features_a_head_needs` is the one answer to what a head must advertise**, asked by `delegation_eligible` and by the planner (`PromptPlan::head_features` → `RingFacts::leads_splits`). A/B `SWARMLLM_REQUESTER_GUESSES=0` (requester), `SWARMLLM_HEAD_GUESS_GAMMA=fixed`, `SWARMLLM_HEAD_GUESS_WAIT=0` (head).

→ `docs/invariants/scheduling.md` § "A requester guesses for the split its head leads"

## A split can check a TREE of guesses, and each row is checked only while it pays (2026-10-20, #134)

**`pipeline::tree_check_possible` is the one answer to "may this check carry a tree"** — every peer segment advertises `features::DRAFT_TREES` (bit 29, in `ALL`) and the tail walks; the round loop asks it before growing one, `forward_verify_through_segments` before sending one. The families come first (`ModelArch::reads_draft_trees`, the target's AND the drafter's); each worker's layers have the last word (`SplitModel::draft_tree_refusal` → `Validation` before anything runs; the reply goes on in chains, possible only while nothing waits to be read again). **`draft_tree::worth_checking`** chooses the rows: the drafter's path probability × the calibration (`dsd_tree::TreeRounds`, kept ÷ rated) against the check's fitted cost per row (`CheckCost::fit`) — never a fixed size; on a long link a row is the hidden state crossing it. **A kept path is cut and read again, never gathered** (`TreeOutcome::rows_in_place` → `TreeRounds::reread`, the next check's first rows). The noise is keyed by depth (`index_pos + 1 + depth`), as the drafter grew it. **The next tree grows while the check is on its way**: the round joins the check with a drafter call growing ON the tree it carries (`TreeAsk::seed`, the queue one call would have had), for `TreeRounds::beside_budget_ms` of the check's predicted time; the part under what the round emitted (`GrownTree::after`) is the next tree, grown further first only where `grow_before_the_check` says it pays. `CheckCost` is fed the check's own time, never the round's. **On only where every segment AND the drafter run whole on a card** (`pipeline::every_segment_runs_on_a_card` + the drafter's own worker, → `dsd_tree::trees_wanted`; this node by `ModelProcessPool::runs_whole_on_the_card`, a peer by `NodeCapability::models_run_on_card`; logged `drafter_on_card`): with the drafter on the processor a missed seed waits ~300 ms for a fresh tree, and on the real TH↔AU link trees won at 160 tokens (5.33-5.74 vs 5.01-5.05) but lost at 400 (mean 5.42 vs 6.06) and 28 (2026-10-21). The card-drafter case is the default by projection and owes a rig before the release that carries it. A card head's drafter is on the processor whenever the target fills the card — a card plan is not cut for it (#275 (4)). Elsewhere the continuous stream, which also won with the head's part on a processor (rows 10-19 ms; 4.75-5.06 vs 5.03-5.52). `SWARMLLM_DRAFT_TREES=1` / `=0` forces either; every node reads a tree since v0.3.246, so no peer needs to update. `SWARMLLM_TREE_GROW_BESIDE=0` grows each only before its check.

→ `docs/invariants/scheduling.md` § "A split checks a tree of guesses"

## A chain its head will lead is priced as its head leading it (2026-10-18, #3 (a))

**`parallax::chain_is_led_by_its_head` is the single answer to "is this chain priced as led by its head"**: `delegation_eligible`'s shape, for a request the hand-off takes (`PromptPlan::with_head_leading` ← `request_may_be_led_by_its_head`, set once in `distributed_exec`), every peer forwarding (`PIPELINE_CHAIN`) with a usable coordinate (`NodeCandidate::ring`). The search runs one led pass per head; each hop is `parallax::led_hop_ms` (half the coordinates' round trip × passes), the tail's back to the head included. `chain_cost_terms` prices a chain the same way, so every gate and the logged prediction read the price it was chosen at. Anything unknown prices it driven, as before — and the driven pass still starts from a head that may lead, or a chain the led pass cannot price has no route.

→ `docs/invariants/scheduling.md` § "A chain its head will lead is priced as its head leading it"
