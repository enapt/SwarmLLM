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

Admission runs at load time, too late to reshape a plan (#452). **`ModelProcessPool::max_hostable_layers_for_planning`** (+ `held_layer_ranges_for_planning`) bounds the LOCAL node from the loader's own estimator, on the device the LOADER would use — never `serves_on_cpu` (#444 vs #129); `None` = unknowable, never "no room". **`process_pool::segment_weights`** prices the segment the worker will map (#342). A fresh worker's card/processor split is for ONE run: **`fresh_run_layers_for_planning`** → `NodeCandidate::fresh_run_layers`, held to one run by `parallax::local_fits` and `local_can_hold_every_layer` (#234), proposed at both ends of each held range (`route_over`'s `fresh_run_of`, #354 (b)), offered only where the spawn would admit its processor share (#250). An idle worker of the model gives its memory to that run (`a_fresh_spawn_fits_in_place_of_the_worker`, #350).

→ `docs/invariants/scheduling.md` § "The component that will refuse must be asked while the plan can still change"

## A split leaves room for the model that guesses for it (2026-10-14, #275)

Where this node cannot hold every layer, or its card is short and the search may cut its range (`scheduler::plans_room_for_the_drafter`, #3 (j1)), each planner FIRST plans with its bound cut to leave the drafter room (**`PipelineScheduler::candidates_leaving_room_for_the_drafter`** → `ModelProcessPool::max_hostable_layers_beside_guest`; `scheduler::leaving_room_here`): on the CARD where planned there (#275 (4)), else system memory — ONLY where every peer is held to what it offers (`CapacityBound::Everyone`, greedy's strict pass in `greedy_assign_beside`). ⚠ Never on a rung or pass that relaxes a PEER's bound. No route that way, plan as before; a route that way is PRICED against the plan without the cut on the same rung (**`scheduler::the_plan_beside_the_drafter`**, #463; A/B `SWARMLLM_DRAFTER_ROOM_PRICED=0`). The drafter is chosen only if it would load beside the layers the PLAN loads here (`guest_fit_beside`, `engine_drafter::TargetHere` REQUIRED), through ONE `engine_drafter::chosen`; room, choice and spawn price the REPLY's cache (`model_worker::drafter_cache_positions` → `vram::KvCharge::Conversation`). A/B `SWARMLLM_DRAFTER_ROOM=0`, `SWARMLLM_DRAFTER_ADMISSION=0`, `SWARMLLM_DRAFTER_KV_CHARGE=typical`, `SWARMLLM_DRAFTER_BESIDE=loaded`.

→ `docs/invariants/scheduling.md` § "A split leaves room for the model that guesses for it"

## A layer is priced as the model it belongs to (2026-10-12, #250)

**`NodeCandidate::prior_ms_per_layer`** is the one prior from a stated speed (`vertex_cost`, greedy): 7B-reference layer time × `layer_work` (`scheduler::layer_work_for` ← `GgufTensorMeta::active_bytes_per_layer`, experts at `expert_used_count / expert_count`); `est_tokens_per_sec` stays in 7B units. A/B `SWARMLLM_LAYER_PRICE=reference`. **A measured figure is kept per (peer, model)** (`peer_speed::ModelSpeed`, read through `SharedState::peer_model_speed`); never scale another model's. **A peer's OWN timing is a floor under the prior, never a discount** (#3 (f)): `OwnDecodeSpeed` (the fastest of 32 decode steps run FOR PEERS, never the owner's; published after 4, withdrawn an hour after the last) gossiped as `ResidentModelLayers::decode_us_per_layer`, read as `NodeCandidate::stated_ms_per_layer`, taken as `max(it, stated speed)`; what THIS node measured outranks it. Rig: `split_rig.sh owntiming` (`RIG_NETNS=1`).

→ `docs/invariants/scheduling.md` § "A layer is priced as the model it belongs to"

## A whole model on one peer is handed over, never driven token by token (2026-09-27)

**`remote_generate::eligible` is the single answer to "does the hand-off take this plan"**; the n-gram loop stands aside for every plan it accepts. Ask privacy with `encrypted_pipeline_for_request`.

→ `docs/invariants/scheduling.md` § "A whole model on one peer is handed over, never driven token by token (2026-09-27)"

## A node is ranked on warm work only — including this node (2026-09-27)

`PeerSpeed::ranking_ms_per_layer` ranks on the WARM coefficient only (the cold-inclusive one sizes timeouts), keyed by node id, ours too. **A prompt is PRICED by `ModelSpeed::prompt_ms_per_layer_token`** — warm forwards of at least `PROMPT_RATE_MIN_BYTES` (a smaller one may only lower it, #287), **per TOKEN, never per byte** (`tensor_util::positions_borne_out`); the step after a promised prompt copy (#433) is cold (`SharedState::note_prompt_copy_promised`). **A chain hop's own report prices it, never faster than it states** (#427): **`SharedState::note_hop_report`** believes a `HopTimings` only from a peer of the reply's last plan, once, within `hop_reports::WAIT_FOR` (120 s), over ≥ `LEAST_STEPS` (4); it is `TimedOn::HeldSteps`, floored at the stated speed (`NodeCandidate::floor_a_peers_own_word`), NEVER in `ranking_ms_per_layer`. A/B `SWARMLLM_HOP_TIMINGS=0`. **A WHOLE hand-off's prompt is floored by its own first-token waits** (`ModelSpeed::whole_prompt_floor_ms_per_layer_token`, #430): `parallax::prefill_ms_per_layer`'s `whole_model` takes `max(floor × tokens, the price)`, never for a segment. **This node's own WHOLE run is priced from what its whole replies delivered** (#3 (j3)): **`SharedState::note_own_whole_run`** (the one writer, in `publish_request_trace`) records `WorkKind::Delegated`, read ONLY for this node's whole-model vertex (`parallax::vertex_cost_as`'s `own_whole_run`); its PROMPT keeps `prompt_decode_price`. A/B `SWARMLLM_OWN_WHOLE_RUN_PRICE=0`.

→ `docs/invariants/scheduling.md` § "A node is ranked on warm work only — including this node (2026-09-27)", § "A prompt is priced from warm forwards that carried enough to measure a rate", § "This node's whole run is priced from what it delivered"

## Room for more layers is not room for the layers already held

**`NodeCandidate::held_ranges` + `layers_it_would_add`** charge a local segment only the layers it would ADD (`process_pool::layers_added_by`), never its width; a run INSIDE a held range adds none, and no cold load, where its holder serves slices (`HeldRange::serves_slices`, #449: this node by `split::view::elastic_spans_on`, a peer only by `features::SERVES_SLICES`; `arch-worker-memory.md`). Peers publishing `ResidentModelLayers::ranges` go through `NodeCandidate::capacity_charge`. ⚠ `max_hostable_layers` stays TOTAL for peers. A candidate's OWN held range is a vertex on EVERY pass and cuts nobody else's (#3 (e)). A peer's room and a held range's charge are compared UNROUNDED (`PublishedRoom`). Other candidates' ranges are cut at held endpoints only on a rung with no route (`parallax::route_over(.., meet_held)`).

→ `docs/invariants/scheduling.md` § "Room for more layers is not room for the layers already held (2026-09-24, #95)"

## A peer's range is weighed as what it loads (2026-10-26, #353)

Never count a peer's layers at the file's AVERAGE. **`process_pool::RangeWeights::excess_layers`** puts the output head with the model's end and the token table with its start, in `NodeCandidate::capacity_charge`, `width_in_bound` (split points, greedy, both standby checks via `charge_in_bound`, `plan_exceeds_offered_memory` — beside the ranges the peer HOLDS, a REQUIRED argument; only a cold ceiling passes `&[]`), the booking, and `near_chain::Planner::reach`. Never compare a peer's bound with a bare width, nor hold it to a bare count measured over other layers. A/B `SWARMLLM_PEER_RANGE_WEIGHTS=average`.

→ `docs/invariants/scheduling.md` § "A peer's range is weighed as what it loads"

## A plan that names this node for the whole model is a local generation

**`pipeline::local_generate::try_local_generate_fastpath`** runs a single-segment plan assigned here through `ModelProcessPool::generate`, never a `LayerForward` per token. A missing `split_models` entry does not disqualify it — read the MANIFEST. A budget on what to OFFER must not decide how assigned work is EXECUTED.

→ `docs/invariants/scheduling.md` § "A plan that names this node for the whole model is a local generation"

## A warm prompt is priced warm (2026-09-26)

**`scheduler::cached_prefix::plan_prompt`** prices the prompt warm for a LOCAL whole-model vertex (`NodeCandidate::cached_prefix_tokens`), for a PEER's whole-model vertex from what this node handed it in the last 300 s (`PromptPlan::handed_off` ← `pipeline::handoff_prompt_cache`, only under `remote_generate::request_may_be_handed_off` while the peer holds the model resident; #3 (c)), and for a split SEGMENT believed to hold its opening for exactly that range (`PromptPlan::split_openings` ← `pipeline::held_openings`; #10), via `pipeline::render_prompt_from_header`. Both routers read **`NodeCandidate::prompt_tokens_held`**; greedy prices the UNREAD prompt at `parallax::prefill_ms_per_layer`. **`NodeCapability::models_run_on_card`**, never `gpu.is_some()`.

→ `docs/invariants/scheduling.md` § "A warm prompt is priced warm (2026-09-26)"

## The hand-off gate proposes; the priced search decides

The whole-model hand-off waits in `hand_off` when the priced search will run; it is taken only on `ProcessorRouteVerdict::NoComparison` or a failed route, never where the search compared and this node won.

→ `docs/invariants/scheduling.md` § "The hand-off gate proposes; the priced search decides"

## A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it

`scheduler::trusted_with_the_plaintext_prompt` is the one bar, read by `delegation_target`, `route_shortest_path`'s source filter, `greedy_assign_inner`'s first-segment narrowing and `standby_may_take`.

→ `docs/invariants/scheduling.md` § "A peer that will read the plaintext prompt clears a trust bar, on every path that can assign it"

## Prompt privacy embeds with the first segment's own loader, or refuses (2026-10-20, #316)

Under `SharedState::local_embedding_privacy` token ids never reach another computer: **`PipelineExecutor::prompt_embedder`** embeds ONLY where **`segment_zero_may_leave_this_node`**, with the first segment's loader asked for no layers (`shard_loader::load_model_range`, `SplitModel::embed_ids`, `local_embedder_for`), else refuses with `EmbeddingPrivacyUnavailable`. Fast paths stand aside in `fastpath_request_disqualified`; a stand-in for segment 0 needs `FORWARD_PRE_EMBEDDED`; no tensor-parallel group takes an embedded first segment (`tensor_parallel_group_for`). Every WHOLE hand-off asks **`SharedState::privacy_refuses_a_whole_hand_off`**. The prompt pass is read from position 0 on every path.

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

`SwarmError::LocalMemoryUnavailable` is the one local failure `should_retry_after` re-plans; the router first calls `SharedState::note_local_memory_refusal(request_id)` so `local_can_hold_every_layer` stops handing this node the whole model, and for a refused LOAD (`ModelProcessPool::note_load_refusal`) gives it fewer new layers and no fresh-worker split (`local_layers_refused_for_request`; #234). A PROMPT refused by conversation memory (`ModelProcessPool::admit_prompt`) holds ALL its layers to what fits (`local_layers_fitting_for_request` → `NodeCandidate::prompt_memory_refused`; #263). After either, no relaxed pass lifts this node's bound (`NodeCandidate::memory_refused_this_request`). **A PEER that refused the prompt before the pass** (#252 (b)) is treated the same in `gather_candidates` (`SharedState::peer_prompt_fits_for_request`), never barred; ⚠ its count is weighed in its own unit (`LayersCounted`) → § "A peer's count of what fits is weighed in its own unit". **A peer handed the WHOLE model that answers it cannot serve is held one layer fewer, never barred** (`remote_generate::after_the_whole_model_was_refused`, #378); A/B `SWARMLLM_WHOLE_REFUSAL=bar`. **The fast path hands its own refusal to the router** (`streaming::router_to_replan` → `RouterCommand::Replan`). **`router::another_replan_is_due` is the one answer to "may this request be planned again"**: the first unconditional, later ones up to `MAX_REPLANS` (5) only for a fast refusal from a remote segment (`a_further_replan_is_earned`, #446).

→ `docs/invariants/scheduling.md` § "A re-plan is warranted by a changed fact, never by a failed attempt"

## A holder's refusal is reported as what it was (2026-10-04, #218)

**`router::report_after_a_replan`** reports `SwarmShortOfMemory` when the refused plan went past what holders OFFER (`plan_exceeds_offered_memory`), else `HoldersDeclined` in their words; ask `a_refusal_the_caller_should_hear`, never re-derive. **The re-plan's own `SwarmShortOfMemory` is "no route" there** (it weighs at `layers_that_fit`, gotcha #931). **A peer that went SILENT and was the only route is reported as silent** (`PeerUnresponsive`; #430 (c)), never "has gone".

→ `docs/invariants/scheduling.md` § "A holder's refusal is reported as what it was"

## The relaxation is scoped to the figures that are actually unreliable

**`parallax::CapacityBound`** (`Everyone`, `PeersAtFaceValue`, `PeersAtCeiling`, `LocalUnbounded`) is walked in that order by `assemble_pipeline_for`; **each rung cuts split POINTS at the bound IT applies** (`route_over`'s `vertex_cap` against `capacity_charge`, #350); a relaxation spends `DELEGATE_VRAM_MARGIN` (`max_hostable_layers_at_face_value`) before the peer's own number. **No rung, no greedy pass, goes past what a peer could EVER hold** — `NodeCapability::model_memory_ceiling_mb` (`ModelProcessPool::segment_cost_curve`), clamped ONCE in `gather_candidates` (`clamp_to_ceiling`); `NodeCandidate::within_ceiling` only for a figure a rung computes itself. A plan nobody's ceiling fits is `SwarmShortOfMemory` (`scheduler::short_of_memory`); greedy's second pass runs whenever a ceiling is known. Unknown is unbounded on every rung.

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

**`NodeCandidate::forward_passes` is the single answer to "how many passes are the per-token costs charged for"**: this model's mean recent reply length (`ReplyLengths::expected`, written ONLY by `publish_request_trace` from finished replies, #3 (b)) — the prior counted as ONE more reply until four have finished (`PRIOR_REPLY_TOKENS`) — else `ASSUMED_FORWARD_PASSES`; then lowered, never raised, by the request's `max_tokens` (`PromptPlan::with_reply_of_at_most`, set once in `distributed_exec`). `vertex_cost`, greedy's spread of the once-per-request terms and the recorded `assumed_forward_passes` all read it; never the bare constant. `None` (preview, tests) prices as before.

→ `docs/invariants/scheduling.md` § "A reply is priced for no more passes than it may take"

## A split is priced for the rounds it takes (2026-10-19, #3 (d))

**`NodeCandidate::rounds_entered` is the single answer to "how many rounds are a split segment's per-token costs charged for"**: `forward_passes()` ÷ the tokens a round of this model's recent splits yielded (`SplitRounds::tokens_per_round`, never below 1), for a SEGMENT of a chain driven from here; a whole model and a led chain keep the passes. Rounds are counted where each LEAVES (`RequestTrace::note_decode_round`), recorded ONLY by `PipelineExecutor::execute`; a path whose forwards OVERLAP counts the round trips it WAITED (**`split_rounds::round_trips_waited`**). **`pipeline::request_may_guess_ahead` is the one request-side question** (→ `PromptPlan::with_guessing_ahead`). A new decode path sends through the counted functions or counts its own rounds. **A peer's `latency_ms` is a ROUND trip** (`PeerInfo::round_trip_ms`), charged ONCE per pass (#3 (h)) → `docs/invariants/scheduling.md` § "An unmeasured peer is charged one round trip a pass". **Bump `counted_as`** (`daemon::state::recent_by_model`) where what a round or reply length counts changes.

**Before four splits have run, one a drafter would guess for starts from a PRIOR** (#3 (j)): `split_rounds::YieldPrior::Drafter` (`GAMMA_STEP`, `BEFORE_TRUSTED`), chosen ONLY by **`pipeline::split_yield_prior`** (a drafter may guess AND `drafter_to_leave_room_for` names one held; or the file's own guess head would — **`pipeline::split_guess_head_here`**), REQUIRED in `SplitRounds::tokens_per_round`. A/B `SWARMLLM_SPLIT_YIELD_PRIOR=none`. **Now and then it is OPTIMISTIC so the split runs and is measured** (`YieldPrior::Explore` at `EXPLORED_TOKENS_A_ROUND`, 8): **`SplitRounds::prior_for_request`** is the one schedule; A/B `SWARMLLM_SPLIT_EXPLORE=0`. **Every figure is kept per TAIL** (#3 (j5)): `split_rounds::SplitTail` REQUIRED in `record` and `tokens_per_round`; exploration takes ONE tail (`SplitTail::BOTH`), restarted only by a split of THAT tail (`SplitRounds::tail_to_explore`, #3 (j9); `Exploring::fruitless`). **`parallax::route_each_tail`** searches once per tail where `split_tokens_per_round` and `split_tokens_per_round_far_tail` differ (**`parallax::priced_for_tail`** is the one reader of the far figure), keeping the cheaper — `route_shortest_path`, its held-range fallback and `cheapest_split` all through it; `chain_cost_terms` / `chain_tokens_per_round` price a plan by its own tail; greedy reads the near figure. **And per KIND of reply** (#469): **`pipeline::reply_kind`** is the one scoring, asked at plan time (`with_guessing_ahead`'s REQUIRED `kind`) and at record time; `ReplyKind` REQUIRED in `record` and `tokens_per_round`; an `Explore` request reads the tail's figure whatever its kind.

→ `docs/invariants/scheduling.md` § "A split is priced for the rounds it takes, not the tokens it writes"

## A forward the peer could not OPEN goes to it again, once the link is re-keyed

**`pipeline::local::wait_for_result` takes a REQUIRED `ResendOnRefusal`**: on `ForwardRefusal::Undecryptable` (a TYPE) it waits for `SessionManager::rekeyed_since` and resends once to the same node. A chained forward passes `Never`; add no second resend.

→ `docs/invariants/scheduling.md` § "A forward the peer could not open goes to it again, once the link is re-keyed"

## A reply whose machine failed mid-stream is CONTINUED from what its reader received (2026-10-08)

**`router::continuation_after` is the single answer** to "may a reply that already reached its reader go on after a failure": the re-plan's failure classes plus `ReplyTruncated`, at most `MAX_CONTINUATIONS`, from `RequestTrace::streamed_so_far` — text recorded by `StreamingTokenTx` only after a send SUCCEEDED. The continued request carries `InferenceRequest::continuation`, honoured ONLY in `PipelineExecutor::build_prompt_with_header` (every path's prompt); a path rendering elsewhere stands aside (`delegation_eligible`, `local_executor_serves`). Rig: `split_rig.sh continue` (#236).

→ `docs/invariants/scheduling.md` § "A reply whose machine failed mid-stream is continued"

## A reply under way is never moved to a machine that cannot continue it

`distributed::failover_can_restore_state(sequence_num)` is asked by `failover_segment` BEFORE looking for a stand-in; else `SegmentFailoverExhausted`. State is given only by `assemble_replay` over `state.retained_activations` (keyed by LAYER RANGE; `restorable_history` must be contiguous). Never a partial replay.

→ `docs/invariants/scheduling.md` § "A reply under way is never moved to a machine that cannot continue it"

## State that belongs to an ATTEMPT is keyed by the attempt, never by the request id (2026-09-28)

State scoped to one attempt (e.g. `engine_drafter::draft_key`) is keyed by the attempt, not the request id: **a release keyed by request id reaches every worker** (#749).

→ `docs/invariants/scheduling.md` § "State that belongs to an attempt is keyed by the attempt, never by the request id (2026-09-28)"

## Speculation that does not pay steps aside — γ = 0 is an answer (2026-09-29)

**`dsd_controller::best_gamma_for_check` may choose zero guesses**; `recall` / `remember` carry it on. A new speculating path needs a way to stop guessing (the continuous stream remembers `best_gamma_overall`). A remembered zero must not come from ONE stalled check: every check cost goes through `CheckCost::record` (#244). **The line's fixed part is never under the round trip to the plan's farthest peer** (`CheckCost::floor_ms`): a fit is built only by `CheckCost::for_a_plan(pipeline::check_reach(..))` or a recalled one's `on_a_plan`. **A peer not yet measured is still a peer** (`dsd_controller::CheckReach::Peer { round_trip_ms: None }`), never "this computer".

**On one computer** (`model_worker::LocalSpec`): **`dsd_controller::best_gamma_beside_a_step`**; A/B `SWARMLLM_LOCAL_GUESS_GAMMA=fixed`. **A split's block drafter is priced the same way** (`pipeline::block_draft::SplitBlockDrafter`, #428 stage 3; `dsd_controller::Step::OverTheChecksRoute`) and **JUDGED by the reply, never probed** (#425 (a); `dsd_controller::BlockShadow`, `DepthAcceptance`); `SET_ASIDE_AFTER_JUDGED` (8) judgements under a plain step or the head's price set it aside (`SplitBlockDrafter::note_judgement`, `PipelineExecutor::block_drafter_set_aside`). A/B `SWARMLLM_SPLIT_GUESS_PRICE=geometric`. **On one computer a guesser is set aside by what it WRITES** (#428): `dsd_controller::Delivered` against a plain step's ONLY (`delivers_less`); every guesser declines and probes by ONE rule (`model_worker::Probing`). **Where a block check costs two plain steps or more** (`model_worker::SHADOW_FROM_STEPS`) **a block probe is drafted, NOT checked**; A/B `SWARMLLM_BLOCK_PROBE=check`. A reply's first block is checked **NARROW** — **`dsd_controller::first_block_guesses`** (3) (#428 (b)); A/B `SWARMLLM_BLOCK_FIRST=whole`. The plain step is the MODEL's (`SplitModel::plain_steps`, `PlainSteps::RETIME_AFTER`); a step or check under 0.75× its figure starts it over (`RecentMedian::record_trusting_a_faster`, `CheckCost::record_trusting_a_faster`). **A check on one computer is priced by its WIDTH** (`CheckCost::width_price`; a peer's keeps the line). A block round is charged its WHOLE time (`BlockGuesses::rest_ms`) against a plain round's (`LocalSpec::plain_round_per_ms`); A/B `SWARMLLM_BLOCK_CHARGE=draft`.

→ `docs/invariants/scheduling.md` § "Speculation that does not pay steps aside — γ = 0 is an answer (2026-09-29)"

## A failed request hands back the work it had already done

`SharedState::salvaged_replies` is handed back only by **`router::salvaged_reply_if_lost`**. Recording is a READ: `pipeline::PartialReply` is filled by the shared emit helpers (`streamed_reply_text_goes_through_the_shared_emit_helpers`) and `keeping_the_partial` is the one choke point.

→ `docs/invariants/scheduling.md` § "A failed request hands back the work it had already done"

## What a peer HOLDS on disk and what it has LOADED are different facts

A peer's parts lists (`ShardAnnounce`) are disk, `resident_layers` memory. **`inference::scheduler::PeerResidency` is the single reading**: `Layers(n)`, `Cold`, `WarmAmountUnknown` — silence is never "holding nothing".

→ `docs/invariants/scheduling.md` § "What a peer HOLDS on disk and what it has LOADED are different facts"

## A peer advertises the memory it will HONOUR, not the memory it has

**`NodeCapability::memory_for_model_layers_mb` is the single answer to "how much memory can this peer give a model's layers"**; `ram_model_budget_mb` is a card-less node's figure. It (`vram::live_headroom_mb`) is never more than the worker's admission honours (`kv_budget::device_free_margin_bytes`; #219) and is asked FOR a model (REQUIRED `model_id`, #447): stated room plus what idle workers of OTHER models give way (`ResidentModelLayers::gives_way`, `NodeCapability::given_way_for`), never the model's own worker.

**A card node's machine memory is offered BESIDE its card**: read only through **`NodeCapability::processor_beside_card`** (`machine_room_beside_card_mb` + `processor_tokens_per_sec_7b`), added by `scheduler::layers_beside_card`, recorded as `NodeCandidate::card_layers` — for a warm peer the layers its card RUNS (`ResidentModelLayers::card_layers` ← **`WorkerHandle::layers_on_card`**; #3 (j11)), never every layer it holds, never more than its card (#361). A range is priced PER DEVICE — **`NodeCandidate::layers_by_device`**: `card_layers` of the run at the card's speed (a held copy's placement for a run it serves without a load: `HeldRange::covers_without_a_load` — its LAST layers where the holder loads a far range processor-first, `NodeCandidate::processor_run_first`, #461 (4); a fresh run is counted from its start in either order, since a piece cannot know where its run ends and the run's total holds; A/B `SWARMLLM_SLICE_PLACEMENT=start`), the rest at `processor_tokens_per_sec`, never one blended speed. A MEASURED decode figure is already that blend, so its prompt is divided by the per-device ratio (`parallax::prompt_speedup_over`; #430 (a)). **A machine's back-to-back segments are ONE run** (`merge_contiguous`): each priced with the run before it ahead on the card, **`parallax::layers_ahead_in_run`** → `vertex_cost_as`'s REQUIRED `ahead`. **A chain that is ONE machine from layer 0 to the last is its whole-model vertex, never pieces** (`route_over`).

→ `docs/invariants/scheduling.md` § "A range that runs partly on a card is priced per device", § "A node holding every layer that would run the model on its processor lets the priced search compete with its fast path" (the back-to-back pieces)

## A hand-off is priced as the shape it will be given, not as the whole model

**`inference::scheduler::delegated_shape_cost_ms`** is the one answer to "what does this request cost if that peer is given `layers_to_assign` of it". The price gate (`costs_more_than_staying_here`), the line that logs the gate's verdict, and `privacy_cost_ms` all go through it.

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

`scheduler::standby_has_room(max_hostable_layers, already_committed, segment_layers)` is asked of every standby candidate, beside `standby_covers`: `standby_covers` asks whether a node HOLDS the range, `standby_has_room` whether it could RUN it (#452, #454). **Every range — a standby's, each part of a cover, and the primaries that seed the tally (`primary_layer_commitments`) — is charged by `NodeCandidate::charge_in_bound`**, in the unit of the node's own bound: what it loads for a peer (a total, `width_in_bound`), what it ADDS for this node (room, `layers_charged`; its width once its prompt memory refused) (#354 (a)).

→ `docs/invariants/scheduling.md` § "A standby is a capacity commitment, not just a coverage claim"

## A stand-in may be SEVERAL nodes, and the segment count is then read live

**`scheduler::standby_cover_for` is the single answer to "what could take this range over"** (a hole answers `None`; prompt pass only). **`forward_through_segments_inner` reads `segments.len()` LIVE** (`the_pipelines_segment_count_is_never_cached_across_the_forward_loop`); `failover_segment`'s `Takeover::{Finished, Continue}` says whether the pipeline finished.

→ `docs/invariants/scheduling.md` § "A stand-in may be SEVERAL nodes, and the segment count is then read live"

## A count and an outcome that disagree are two different questions

`scheduler::standby_covers` is the one predicate behind `segments_without_standby` and `failover_segment`'s search. Print the answer, not the count.

→ `docs/invariants/scheduling.md` § "A count and an outcome that disagree are two different questions"

## A wait on a peer's first answer allows for its load, on every path (2026-10-07)

**`pipeline::LoadAllowance` is the single answer to "may this peer first have to load the model"**: REQUIRED in `remote_generate::first_token_timeout`, asked by `SegmentBudget::for_forward`; outside tests built only by asking (`for_peer` / `for_segments`). The wait ends when the peer's last connection closes. Rig: `examples/cold_load_test.sh`. **Its third REQUIRED argument, `remote_generate::MeasuredRead`** (#430): a whole hand-off to a peer with a believed prompt floor (`MeasuredRead::of_whole_model`) waits `2 × floor × layers × tokens` past the 600 s cap, up to an hour; a led split and the HTTP forward pass `Unknown`. **"Warm" is set by ANY answer** — `SharedState::note_peer_model_resident` is the one writer (every timing sample; a whole hand-off's reply of any length, `note_a_whole_reply`).

→ `docs/invariants/scheduling.md` § "A wait on a peer's first answer allows for its load, on every path"

## A candidate that must first load the model is CHARGED for it (2026-10-07)

**`NodeCandidate::cold_load_ms(range)`** (`cold_load_ms_per_layer` × `layers_it_would_add`) is the one formula — 0 on a request that explores (`YieldPrior::Explore`): `parallax::vertex_cost`'s `cold_load_ms` (once per request, never × `expected_attempts`) AND greedy's `estimated_cost_per_layer` (÷ covered ÷ `ASSUMED_FORWARD_PASSES`; #10). Set in `gather_candidates` from the SAME residency reading as the memory bound (`WarmAmountUnknown` → 0). The rate is the node's own (`process_pool::LoadRate` → `NodeCapability::model_load_ms_per_gib`, kept by `daemon::state::load_rate`), else `UNMEASURED_LOAD_MS_PER_GIB`; this node is charged its own too. Rigs: `cold_load_test.sh … price`, `split_rig.sh splitcache` with `SPARE=1`.

→ `docs/invariants/scheduling.md` § "A candidate that must first load the model is charged for it"

## The units decide whether a forward is a prefill, not the byte count

**`inference::pipeline::local::PipelineExecutor::forward_is_prefill(activation_bytes, units)`** is the single answer to "is this forward doing a prefill?", for the deadline (`compute_segment_timeout`) and for the DIAG that reports it. `SegmentBudget` carries the resolved verdict (`is_prefill()`) so the log cannot contradict the budget.

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

- **`PipelineScheduler::local_speed_off_the_card`** — this node's speed for a model its card cannot hold whole (processor blended with the card by layer share: `ModelProcessPool::card_share_for_planning` → `split_tokens_per_sec`); the hand-off gate's baseline (`DelegationInput::local_cpu_tokens_per_sec`) and the search's local candidate where the loader would not split (#129) read it.
- **`ModelProcessPool::serves_on_cpu`** — the whole-model delegation precondition, and the fast path's: "fits the card" only with no layer on the processor (`whole_model_on_card`: `holds_gpu_memory` AND `!short_of_the_card`). **Where it answers yes and the card runs only part, the search may cut ranges** (`scheduler::shares_what_its_card_cannot_hold`, #3 (j1); partial ranges are otherwise off outside the boomerang, #180), also where each candidate's CARD stops (`card_layers`, `MAX_SUBRANGE_VERTICES`). A/B `SWARMLLM_SPLIT_BESIDE_CARD=0`. → § "A computer short of its card shares the rest"
- **A node holding every layer that would run the model on its processor lets the priced search compete with its fast path**
- **A peer's capacity for a prompt is weights PLUS that prompt's KV cache** — weights at **`scheduler::weight_bytes_per_layer`** (less tensors its loader keeps in host memory where the peer runs on its card, `GgufTensorMeta::host_resident_bytes`, #296), in the bound, the published room, the booking and `allocate_offline`; a load's price keeps the file's figure.
- **`inference::scheduler::delegation_target`**
- **`inference::router::distributed_exec::failure_is_penalty_worthy`**
- **A peer that went silent is barred from THIS request's retry, and the retry happens**
- **`RequestTrace::finish_from`** — the one reading of a result as its outcome on every router completion path (a reader that left is `Cancelled`). **`router::failure_worth_listing`** — what diagnostics list as a failure (never a reader who left). → § "A request its reader stopped is cancelled, not failed"

→ `docs/invariants/scheduling.md` § "Single-source-of-truth helpers — Scheduling, routing and failover"

## A peer that fails a model on every request is ejected from its plans (2026-10-02)

**`daemon::state::peer_outliers`**: two failures in a row (fed ONLY by `record_peer_delivery`) eject a peer per model for 2 min, doubling to 30; `note_peer_completed_request` forgives. Refusals are not failures.

→ `docs/invariants/scheduling.md` § "A peer that fails a model on every request is left out of its plans for a while (2026-10-02)"

## A split none of which is ours is LED by its head (2026-10-02)

**`remote_generate::delegation_eligible` is the single answer**; its REQUEST conditions are **`request_may_be_led_by_its_head`**, which the planner asks too. `try_delegated_split` runs FIRST in `execute_distributed`, under `RoutePlanOverride::lead_here`. Local work uses **`pipeline::worker_requester`**, never a hardcoded `Requester::Owner`. **A head's decline is a fact about the head** (#372): **`SharedState::peer_may_lead`** (`daemon::state::declined_leads`: 5 min, doubling to 60, forgotten once it leads) is asked by `delegation_eligible` AND the planner's led pricing (`RingFacts::leads_splits`); noted before any token, never for a reader who left, only for a refusal about the HEAD (**`declined_leads::is_a_decline_to_lead`**, required by the one writer `note_head_declined_to_lead`). A/B `SWARMLLM_DECLINED_LEADS=0`.

→ `docs/invariants/scheduling.md` § "A split none of which is ours is LED by its head, never driven from here (2026-10-02)"

**A requester holding a drafter guesses for it; the head waits for a window only where it pays (#318).** `pipeline::requester_guesses::start` offers guesses (`DelegatedSplit::guesses`) only to a head advertising `features::REQUESTER_GUESSES` (bit 28) — NOT in `ALL`: advertised only where `requester_guesses::takes_guesses` says the head would open a slot, the one answer both ask. The head's slot (`daemon::state::requester_guesses`) is opened ONLY by `handle_delegated_split` and offered to ONLY by dispatch's `RequesterGuesses` arm, for the authenticated peer that handed the reply over. The n-gram loop records every token BEFORE emitting it (`StreamingToken::reply_ids`); a continued reply takes none. The head checks only as many guesses as its fitted check cost says pay (`ngram_only_spec::choose_guess_gamma` → `dsd_controller::best_gamma_for_check`; zero is an answer). The stream is HANDED its slot (`OpenSlot::slot`), never finds one by the wire id. **A wait is priced, never assumed** (#318 (a)): only where nothing else would be checked, only for a window `GuessSlot::window_due_in` says is due, and only while **`ngram_only_spec::worth_waiting_ms`** says it beats a one-token pass — `W < E·(P_1 + α·P_g)/(1 + α·E) − P_g`, never `P·(E − 1)`. **A `swarm_route` instruction goes WITH the hand-off** (#318 (b)): only `lead_here` keeps a request here; `DelegatedSplit::route` goes only to a head advertising `features::CARRIED_ROUTE` (bit 30), which reads it via `route_override::led_override` (every part only shrinks; an impossible route is refused before a token). **`remote_generate::features_a_head_needs` is the one answer to what a head must advertise** (`delegation_eligible`, and the planner's `PromptPlan::head_features` → `RingFacts::leads_splits`); template variables need `TEMPLATE_VARS` (bit 37, #391). A/B `SWARMLLM_REQUESTER_GUESSES=0` (requester), `SWARMLLM_HEAD_GUESS_GAMMA=fixed`, `SWARMLLM_HEAD_GUESS_WAIT=0` (head).

→ `docs/invariants/scheduling.md` § "A requester guesses for the split its head leads"

## A split can check a TREE of guesses, and each row is checked only while it pays (2026-10-20, #134)

**`pipeline::tree_check_possible` is the one answer to "may this check carry a tree"** — every peer segment advertises `features::DRAFT_TREES` (bit 29, in `ALL`) and the tail walks; asked before growing one and by `forward_verify_through_segments` before sending one. Families first (`ModelArch::reads_draft_trees`, target's AND drafter's); each worker has the last word (`SplitModel::draft_tree_refusal`). **`draft_tree::worth_checking`** chooses rows by path probability × `dsd_tree::TreeRounds`' calibration against the fitted cost per row (`CheckCost::fit`), never a fixed size. **A kept path is cut and read again, never gathered** (`TreeOutcome::rows_in_place`). **The next tree grows while the check is on its way** (`dsd_tree::beside_growth`; A/B `SWARMLLM_TREE_BESIDE=8,128`, `SWARMLLM_TREE_GROW_BESIDE=0`); **a growth before it is SIZED** (#134 (a), `TreeRounds::grow_size`; A/B `SWARMLLM_TREE_REGROW=fixed`). `CheckCost` is fed the check's own time, never the round's. **On only where every segment AND the drafter run whole on a card AND the farthest peer segment is a measured `TREES_LEAST_ROUND_TRIP_MS` (300) or more away — `FLOW_LEAST_ROUND_TRIP_MS` (110, #411) where the tree would FLOW from the far end**; a tree drawn HERE keeps 300 (`dsd_tree::FlowDraws`, `dsd_tree::least_round_trip_ms`; the loop asks `flow_possible` BEFORE the line) via `pipeline::every_segment_runs_on_a_card` + the drafter's worker + `pipeline::farthest_segment_round_trip_ms` (least measured, never smoothed; unknown is not far) → `dsd_tree::trees_wanted`; this node by `ModelProcessPool::runs_whole_on_the_card` (LIVE; `EngineDrafter::read_up_to_the_last` reads the prompt first, #3 (j1); A/B `SWARMLLM_DRAFTER_READ_FIRST=0`), a peer by `NodeCapability::models_run_on_card`; `examples/tree_schedule_sim.py`. A plan filling the head's card leaves its drafter room (#275 (4)). Elsewhere the continuous stream. `SWARMLLM_DRAFT_TREES=1` / `=0` forces either.

→ `docs/invariants/scheduling.md` § "A split checks a tree of guesses"

**A split's BLOCK drafter sends a tree too (#437)**: the n-gram loop grows it from the block's marginals (`BlockDrafter::draft_ranked` → `GrownTree::from_marginals`, the block's own picks the trunk) over the chain's count and ONE level past it (`model_worker::block_tree_positions`; a whole block drafted 2-4x slower on the processor and lost on prose; A/B `SWARMLLM_BLOCK_TREE_DEPTH=block`) where **`ModelArch::block_tree_features`** names what every peer segment needs (`features::BLOCK_TREES`, bit 46, beside `DRAFT_TREES` for EVERY family: an older peer answers a tree with no rows) — `tree_check_possible`'s `needs` is REQUIRED. Its rounds are DSD's `TreeRounds` (`check` / `settle` / `reread`); a round with kept tokens to read again ALWAYS goes through the tree path, whatever its guesses. The drafter reads only the kept path's rows, as the reply's positions: a forward holding its tree whole catches its rows (`model_worker::rows_are_the_replys`), `sampling::rows_a_walk_kept` at the walking tail, `BlockRowsAsk::keep_the_path` for segments before it. A/B `SWARMLLM_BLOCK_TREE=0`.

→ `docs/invariants/inference.md` § "A recurrent layer reads a tree from each row's parent's state" (**The coordinator half**)

## A tree flows where trees are checked from the start (2026-10-22, #134 (b))

**`pipeline::dsd_flow::flow_possible`** is the one answer to "may this reply's tree flow": the run's tail draws, or this node's own layers after the run do (#369; A/B `SWARMLLM_FLOW_FINISH_HERE=0`, `SWARMLLM_FLOW_TAIL_JOIN=0`, `SWARMLLM_FLOW_DEAD_WAIT=0`); the sampler reads no history; every peer advertises `features::DRAFT_TREE_FLOW` (bit 31) beside `DRAFT_TREES`. One tree, `levels_a_call` (3) levels of `rows_a_level` (8) a call; its copies (`FlowTree`, every segment's cache, the drafter's `FlowGrowth`) change only by ordered `FlowOp`s; every forward carries the WHOLE shape; rebase, never cut per verdict. **`FlowGuard` holds the rows that DIE down on a thin uplink**; an answer's delay is its ARRIVAL (`Arrival`); A/B `SWARMLLM_FLOW_GUARD=0`. **A far computer's own time is not the link's** (#388): read only through **`dsd_flow::believed_served_ms`**; a far cut is a TRIAL (`FarTrial`, gotcha #193); A/B `SWARMLLM_FLOW_SERVED_TIME=0`. **A copied run is depth the drafter does not pay for** (`FlowRun::copy_ahead`, `ngram_lookup::copied_run` ≤ 32, under 67% agreement ends it, #338; A/B `SWARMLLM_COPY_AGREEMENT=0`, `SWARMLLM_FLOW_COPY=0`). **With no match in the conversation, the OWNER's earlier replies are looked in** (`inference::owner_replies`, #337): only through `OwnerReplies::for_the_owner` (`InferenceRequest::requested_by` REQUIRED in `::local`, never off the wire; AND `worker_requester`), copied only from a reply EVERY peer of the split served (`owner_replies::Exposure`, `dsd_flow::flow_exposure`, `PipelineExecutor::exposure_of_a_check`, `speculative::ngram_lookup_drafts`'s REQUIRED `earlier`). **On one computer a model process keeps its own** (`OwnerReplies::in_this_process`; **`owner_replies::the_owner_asked`** is the predicate both ask; `requested_by` REQUIRED in `ModelProcessPool::generate`); A/B `SWARMLLM_FLOW_COPY_STORE=0`. **A failed flow finishes in tree rounds** (`rounds_may_finish`). A/B `SWARMLLM_TREE_FLOW=0`; knobs `SWARMLLM_FLOW_ROWS` / `_LEVELS` / `_LIVE` / `_WINDOW`; ⛔ never the drafter's call beside this node's layers while they are two processes on one card (gotcha #759); faults `SWARMLLM_FAULT_SLOW_TURNS=<from>:<count>:<ms>`, `SWARMLLM_FAULT_FLOW_FAIL=<n>`, `SWARMLLM_FAULT_DRAFT_DELAY_MS=<ms>`.

**A flow's speed is E, the tokens its rows get right** — only fewer misses gain. **A small drafter may be HALF its target** (`engine_drafter::within_reach`, `drafter_max_fraction()` 2, `SMALL_DRAFTER_READS` 2 GiB, `small_drafter_fraction`; A/B `SWARMLLM_DRAFTER_MAX_FRACTION=4`); **a drafter in SYSTEM MEMORY reads at most 2 GiB a guess** (`engine_drafter::may_guess_there`, #373; A/B `SWARMLLM_PROCESSOR_DRAFTER_READS=any`); **of two the same size, the target's own release line drafts** (`engine_drafter::release_line`, #370). Judge a flow change by misses and tok/s on a real or delayed link; **name the drafter in every flow figure** (#308). The flow pays from `FLOW_LEAST_ROUND_TRIP_MS` where the far end draws (#411; `docs/plans/faster_than_local.md` § 10 item 10).

→ `docs/invariants/scheduling.md` § "A tree that flows" (On the REAL link: the knob sweep and the drafter table)

## A stream's chunk is as wide as it pays (2026-10-30, #374, #375)

**`dsd_stream::StreamWidth` chooses each chunk's width** of two — `SWARMLLM_SPEC_STREAM_GUESSES` (3) or two more — by `chunk_rate`, with **`dsd_controller::FarPace`** (the gap between two answers where the later chunk was already waiting at the far end; median of ≥ 3, #375). Never price a row from whole checks sent alone where chunks queue at the far end. A width with no gaps is tried on every other queued chunk; every 16th goes the width not chosen; only where no chunk waits does the lone-check fit (`CheckCost::slope_is_measured`) price both. Each chunk's wait is its own task (`stamp_on_arrival`). Never one fixed width; at most two (a new shape costs a card a capture). A/B `SWARMLLM_STREAM_WIDTH=fixed`; `examples/stream_width_sim.py`.

→ `docs/invariants/scheduling.md` § "A chunk is as wide as it pays"

## A chain its head will lead is priced as its head leading it (2026-10-18, #3 (a))

**`parallax::chain_is_led_by_its_head` is the single answer to "is this chain priced as led by its head"**: `delegation_eligible`'s shape, for a request the hand-off takes (`PromptPlan::with_head_leading` ← `request_may_be_led_by_its_head`, set once in `distributed_exec`), every peer forwarding (`PIPELINE_CHAIN`) with a usable coordinate (`NodeCandidate::ring`). One led pass per head; each hop is `parallax::led_hop_ms` (half the coordinates' round trip × passes), the tail's back to the head included. `chain_cost_terms` prices the same way so gates and the logged prediction read the chosen price. Unknown prices it driven — and the driven pass still starts from a head that may lead.

→ `docs/invariants/scheduling.md` § "A chain its head will lead is priced as its head leading it"
