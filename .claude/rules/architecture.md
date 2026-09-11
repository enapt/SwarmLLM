# Architecture Rules

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. **The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`,** one
file per topic, linked from each rule.

Read the linked file before changing the code a rule names. What is here is
enough to know a rule exists and applies; it is deliberately not enough to
judge an exception to it.

## SharedState Sub-Structs

SharedState is organized into 4 sub-structs. Always use the correct accessor:

- `state.events.activity_tx` — NOT `state.activity_tx`
- `state.events.dashboard_tx` — NOT `state.dashboard_tx`
- `state.credits.credit_balance` — NOT `state.credit_balance`
- `state.credits.pool_state` — NOT `state.pool_state`
- `state.models.acquisition_progress` — NOT `state.acquisition_progress`
- `state.models.hf_sources` — NOT `state.hf_sources`
- `state.models.wishlist` — R111. ArcSwap<Wishlist>; refresh via `crate::model::auto_manage::refresh_wishlist(state)`.
- `state.models.hf_trending_cache` — R112. ArcSwap<HfTrendingSnapshot>; written by `HfWatcher` only.
- `state.models.foreign_wishlist` — R130. `DashMap<(NodeId, ModelId), (score_0_100, received_at_ms)>`; capped at `MAX_FOREIGN_WISHLIST_ENTRIES = 10_000` with oldest-first eviction, 2h freshness window enforced on read. Written by `apply_wishlist_announcement` on inbound `SwarmMessage::WishlistAnnouncement`; read by `compute_wishlist` for the 0..10 cross-pool demand boost.
- `state.models.quant_recommendations` — R133. `ArcSwap<QuantRecommendations>`; refreshed via `crate::model::auto_manage::quant::refresh_quant_recommendations(state)` on every auto-manage tick AND on every WS stats build. Read by `GET /api/admin/quant-recommendations` and the swarm-tab tips tile.
- `state.models.shard_download_backoff` — external report 2026-07-23. `DashMap<ShardId, ShardDownloadBackoff { fail_count, retry_after: Instant }>`. Exponential per-shard download cooldown (30→60→120→240→300s cap, via the pure `shard_backoff_delay_secs`). Recorded via `record_shard_download_failure` at every terminal *transient* download-failure site (HF `download_shard` error + GGUF-probe failure in `model/auto_manage/download.rs`, P2P give-up-with-no-HF-source in `network/manager/shard_transfer.rs`, and stall-reconciliation in `health/monitor.rs::cleanup_acquisition_progress`). Checked by `shard_in_backoff` in `scoring.rs::gather_candidates` (skips the shard while cooling down). Cleared via `clear_shard_download_backoff` on success (HF success arm + P2P completion in `requests.rs`). Distinct from `shard_p2p_failed`, which only *forces* the HF path without throttling re-selection — the two solve different problems and a new failure site should touch whichever it needs. Do NOT record backoff on the P2P→HF fallback branch: that path wants an *immediate* HF retry. Entries self-evict from `shard_in_backoff` once idle past `SHARD_BACKOFF_FORGET_SECS` (1h), so the map stays bounded without a dedicated sweep.
- `state.models.removed_by_user` — 2026-08-21 (gotcha #360). `DashMap<ShardId, bool>`, persisted in DB tree `removed_shards`, loaded in `SharedState::new` like `locked_shards`. A shard the USER deleted (`delete_shard`, `delete_model` — every manifest shard) is an instruction, not a gap: `gather_candidates` skips it unless `in_configured_range || pinned_to_us`; an explicit request clears it (`hf_download_shards` for the named shards, `download_shard`, `pool_add_pin` naming this node). Helpers live in `daemon/state/removed_shards.rs` (`mark_shard_removed_by_user`, `shard_removed_by_user`, `clear_shard_removed_by_user`, `clear_removed_by_user_for_model`); the shard listing emits `removed_by_user` (only when not local) and the dashboard shows a "Removed" badge. Never write the map or the tree directly.
- `state.models.shards_needing_repair` — see `docs/invariants/state-and-config.md`
- `state.models.shards_pending_verification` — see `docs/invariants/state-and-config.md`
- `ModelRegistry::origin_verified` — see `docs/invariants/state-and-config.md`
- `network::manager::tensors::AckRttEstimator` — see `docs/invariants/state-and-config.md`
- A KV-budget refusal MUST release what the request already took — see `docs/invariants/state-and-config.md`
- `SharedState::can_fetch_shard_from_origin` — see `docs/invariants/state-and-config.md`
- `state.credits.foreign_pool_catalog` — R134. `DashMap<(PoolId, ModelId), received_at_ms>`; capped at 5000 with oldest-first eviction, 2h freshness window. Written by inbound `SwarmMessage::PoolModelAvailability` handler. Read by `GET /api/admin/foreign-pool-catalog` and by `pool::scope::cross_pool_extras` (R134.7) when `pool.allow_cross_pool_inference` AND `private_mode` are both on.
- `state.local_memory_refusals` — 2026-09-07. `DashSet<Uuid>` on the ROOT
  SharedState, beside `request_holder_blacklist` and released by
  `release_request_state` with it. Written ONLY by
  `note_local_memory_refusal`, read only by
  `local_memory_refused_for_request`. It is the re-plan's queueing hint — see
  "A re-plan is warranted by a changed fact, never by a failed attempt".
- `state.metrics.node_stats` — NOT `state.node_stats`
- `state.metrics.providers_config` — NOT `state.providers_config`
- `state.metrics.swarm_capacity` — R110. ArcSwap<SwarmCapacity>; refresh via `crate::daemon::state::refresh_swarm_capacity(state)`. Eagerly refreshed on peer connect (`network/manager/identify.rs`) and disconnect (`network/manager/connections.rs`) so the dashboard banner stays consistent with the peer-list panel under churn — the WS stats-cache 1.5s coalesce alone is too lazy.
- `state.metrics.hedge_tracker` — R136 Layer 2. `Arc<HedgeTracker>` with per-(model, segment, holder) EWMA latency + rate-budget counters. Always present. Observation via `state.record_hedge_observation(...)` from the forward-success path in `pipeline/distributed.rs` (post-hoc dry-run metrics). Real race-then-discard duplicate dispatch for speculative-verify hops ships in `pipeline/hedge_dispatch.rs::forward_verify_with_hedge` — single-segment only; multi-segment hedging remains deferred (`docs/FUTURE_WORK.md`). R142.6 added `last_observed_at_ms` to `HedgeStats` and `HedgeTracker::evict_stale` wired to the HealthMonitor tick to bound the (model × segment × holder) map.
- `state.metrics.prefetch_orchestrator` — R136 Layer 3. `PrefetchHandle` (Arc<PrefetchOrchestrator>) with per-session first-token histogram + idle-time learner + throttling. Observation via `observe_user_turn(session, first_token)` + `record_response_completion(session, now_ms)` at the router success site. R142.6 wired `evict_idle` to the HealthMonitor tick to bound the histories map. K-layer prefetch dispatch is the remaining integration; data-collection and orchestration are complete.
- `state.standalone_tokenizers` — R136 Layer 1/3 follow-on. `DashMap<ModelId, Arc<SplitTokenizer>>` on the ROOT SharedState (not a sub-struct — used by both `state.metrics`-derived L3 prefetch AND the `pipeline/ngram_only_spec.rs` L1 path, so cross-cutting). Lazy-loaded from `gguf_header.bin` via `state.standalone_tokenizer(&model_id)` accessor. Returns `None` when the header isn't on disk; caller falls through gracefully.
- `state.pending_activation_chunks` — R139 Tier 4K. `DashMap<Uuid, ChunkAssemblyState>` on the ROOT SharedState (cross-cuts the RR-decrypt path in `network/manager/tensors.rs` and the persistent-stream reader in `network/pipeline_stream.rs`). Receiver-side assembly for STREAM-chunked activation forwards. Entry-locked insert via `state.try_assemble_chunked_forward(forward, sender_peer_bytes)`. Periodic stale-entry sweep wired to the HealthMonitor tick via `state.sweep_stale_chunk_assemblies(ttl_secs)`. Chunk-meta is bound into AAD via `build_layer_forward_aad`, so reorder/truncation/cross-transfer-substitution fail Poly1305 before reaching the assembly.
- `state.listen_multiaddrs` — R140. `arc_swap::ArcSwap<Vec<String>>` on the ROOT SharedState (cross-cuts NetworkManager-writes and PoolManager-reads). Live snapshot of the swarm's reachable addresses, each terminated with `/p2p/<local_peer_id>`. Written by `NetworkManager::refresh_listen_multiaddrs()` (events.rs) on `NewListenAddr` / `ExpiredListenAddr` / `ListenerClosed` / `ExternalAddrConfirmed` / UPnP `NewExternalAddr` / `ExpiredExternalAddr`, plus once at startup after `listen_on()` (and after the `network.external_addresses` config override is added). **R143: the snapshot is the UNION of `swarm.listeners()` (bound sockets — private LAN on a NAT'd node) AND `swarm.external_addresses()` (UPnP-mapped / AutoNAT-confirmed / relay-circuit / manually-declared public addrs).** Without the union a NAT'd node's invite code silently shipped a LAN-only address. Built via the extracted, unit-tested `build_reachable_multiaddr_list(candidates, peer_id)` + `ensure_p2p_suffix` helpers; filtered through `addr_is_remotely_reachable` — keeps LAN + Tailscale CGN (100.64.0.0/10) + public, drops loopback / unspecified / link-local / IMDS. Read by `PoolManager::handle_generate_invite_code` when minting v2 `swarmpool://` codes; empty list → `SwarmError::ServiceUnavailable`. When the list has entries but NONE pass the stricter `pool::invite::any_internet_reachable` (public IP / DNS / relay-circuit — excludes LAN + CGN), invite generation still succeeds but emits a `pool`/`invite_lan_only` warning ActivityEvent so the user isn't handed a LAN-only code that dies over the internet.
- `config.api.dashboard_trust_lan` — read via `SharedState::cfg()` (see below), never re-derived with `addr.ip().is_loopback()`. `api::dashboard_trust::classify` is the single answer to "may this request be handed the API key automatically?"; the sibling `dashboard_trust_overlay` is read the same way. Was a private `AtomicBool` mirror until 2026-08-09, folded into the live config when that became general.
- `state.observed_inbound_connection` — see `docs/invariants/state-and-config.md`
- `SharedState::model_is_in_use` is the answer to "may I delete this… — see `docs/invariants/state-and-config.md`
- **`api::dashboard_trust::classify` is the single answer to "may this request be handed the API key automatically?"** Do NOT re-derive it with `addr.ip().is_loopback()`. That predicate means "the last TCP hop began inside this daemon's network namespace", which is simultaneously broader than intended (a same-host reverse proxy such as `tailscale serve` satisfies it on behalf of a fully remote client) and narrower (a container publish, a NAT, or a Tailscale subnet router never satisfies it — not even from the host's own `localhost` — because subnet routers SNAT by default). Same-origin checks belong on `Origin` vs the request's own `Host` (`websocket.rs::ws_origin_allowed`), never on a hardcoded loopback allowlist: that mistake independently cost every non-loopback dashboard its live WebSocket updates. See gotcha #195.
- `state.relay_proven_features` — `DashMap<NodeId, RelayProvenFeatures { features: u64, proven_at: Instant }>` on the ROOT SharedState (`daemon/state/relay.rs`). Records relay features a peer has *demonstrably* used by relaying a message addressed to us: `handle_relayed_tensor` records `features::TENSOR_RELAY`, `handle_relayed_envelope` records `features::RELAY` (via `record_relay_proven_features`, which ORs bits + refreshes `proven_at`). The relay send path's feature gates (`target_supports_{relay,tensor_relay}` in `network/manager/relay.rs`) consult `relay_feature_proven(peer, bit)` FIRST, before the gossiped `NodeCapability.features`. **This is the cold-start return-path fix**: a serving node reaches a coordinator known only via `ensure_relayed_origin_known` (whose `peer_registry` entry has `capability: None`, because the capability-gossip handler at `daemon/dispatch/mod.rs` is update-only and can't populate a not-yet-existing entry). Without the proof, the return relay of a computed `LayerResult` was refused until a capability-gossip round landed (≤30s), dropping the first result. Freshness = `RELAY_ROUTE_TTL_SECS` (re-proven on every inbound relayed message, so an active session never goes stale); swept alongside `relay_routes` in `sweep_stale_relay_state`. New relay send paths that gate on a peer's relay capability MUST consult this proof, not just the gossiped capability.

When adding new fields to SharedState, put them in the appropriate sub-struct unless they're accessed by 10+ files across 3+ subsystem boundaries.

When adding new fields to SharedState, put them in the appropriate sub-struct unless they're accessed by 10+ files across 3+ subsystem boundaries.

→ `docs/invariants/state-and-config.md`

## "Is the empty chat state showing?" is a question about the DOM

**`App.chat.refreshEmptyState()`** rebuilds `#chat-empty` in place when that is
what is on screen, and no-ops otherwise. Every caller that needs the empty state
to reflect changed data goes through it: the `stats_update` tick, the model list
loading, a model being picked, and entering the Chat tab.

→ `docs/invariants/frontend.md`

## Every surface that shows a model's reply renders it the same way

**`utils.renderReplyInto(el, text, opts)`** is the one place a reply becomes
rendered HTML: it adds `md-body`, runs `renderMarkdown`, and keeps the source on
`el._rawText` so Copy hands back what the model actually wrote rather than the
markup stripped of its markdown. `chat.js::_renderReply` is now a thin wrapper
over it.

→ `docs/invariants/frontend.md`

## A reasoning model's scratchpad is not the reply

`inference::take_leading_reasoning_block` removes a leading `<think>…</think>`
in `finalize_reply_text` (the non-streaming choke point), and
`tool_parse::StreamingToolText` withholds the same block while streaming — the
buffer both encoders already share, so all four API paths inherit it.

→ `docs/invariants/api-surfaces.md`

## A rendered prompt that lost the question is a FAILED render

`chat_template::render_kept_the_last_question` is a post-condition on
`apply_chat_template`, inside `build_prompt_inner`: a render that does not
contain the last user message's text is discarded, logged, and replaced by the
fallback chain.

Asserting on the FRAME cannot see this. The Qwen3 render test asserted the
prompt ends on `<|im_start|>assistant`, which a prompt that dropped every
message also does, and it was green while every Qwen3 request in the field
arrived with no question in it.

→ `docs/invariants/api-surfaces.md`

## A model is told about its tools the way it was trained to be

**`chat_template::build_prompt` is the ONE place that decides how a model learns
what tools it has** — it needs the tool definitions AND the model's own
template, and nothing else holds both. `template_renders_tools` chooses: a
template that reads `tools` renders them itself; one that never mentions them
gets `describe_tools_in_prose`.

Flattening tools into a system message at the API edge is what left Qwen3's own
`{%- if tools %}` branch unreachable on every request ever made. `tools` is a
REQUIRED parameter of `build_prompt` and `InferenceRequest::local`, and rides on
the request beside `messages` — the router builds its prompt long after the API
surface is gone. The Anthropic surface translates its `input_schema` shape into
the `{"type": "function", "function": {...}}` one templates are written against.

**`tojson` is a minijinja feature (`json`), and an unknown filter fails the
WHOLE render.** Every real tool-rendering template calls it.

**And the `tojson` a template gets is OURS, not minijinja's** —
`chat_template::tojson` implements the signature `transformers` defines
(`ensure_ascii`, `indent`, `separators`, `sort_keys`, Python's separator
defaults) and does not escape HTML. minijinja's builtin does both wrong for this
use: it rewrites `<`, `>`, `&` and `'` for a web page, and it rejects every
keyword but `indent` — which fails the whole render.

→ `docs/invariants/api-surfaces.md`

## A template that refuses a system role is still told what the system turn said

Gemma and Mistral `raise_exception` on a system turn, which fails the whole
render — and the tool description IS a system message, so every such request
rendered through a FALLBACK instead of the model's own template.
**`chat_template::fold_system_into_first_user`** moves the system text into the
first user turn, as a RETRY after the render has already declined, so a template
that renders a system turn today is untouched.

→ `docs/invariants/api-surfaces.md`

## Chat templates render on minijinja, and its settings are part of the contract

Rendering is `minijinja` + `minijinja-contrib`'s `pycompat` — the engine
HuggingFace's TGI and SGLang use — NOT a subset of our own. A thousand lines of
hand-rolled Jinja were deleted on 2026-09-10 because a subset does not decline
on a template past its edge, it HALF-renders.

Four settings are load-bearing and must not be dropped: `trim_blocks`,
`lstrip_blocks` and `keep_trailing_newline` (what `transformers` renders with,
so a template's own indentation is not part of the prompt), and the `pycompat`
unknown-method callback (templates call `split` / `lstrip` / `startswith`, which
minijinja does not implement natively). `raise_exception` must fail the render;
a bad `strftime_now` specifier must not.

A template arrives inside a downloaded GGUF, so it is untrusted input AND a
program: output size, an instruction budget, and the template source are all
bounded.

→ `docs/invariants/api-surfaces.md`

## A context that will not fit is shrunk, not refused

`inference::executor::context_retry_ladder` is the answer to "what context size
will this card actually accept": halve from the capped figure to a floor, serve
the first size accepted, and log what was granted. `effective_llama_context`
caps by a CONSTANT, and whether that constant fits is not constant — an 8B's
weights can leave less room than its 8192-token KV cache needs, which failed
every request on a 6 GB card.

llama.cpp will not say how much it needs and free memory read beforehand is
evidence rather than proof, so the size is asked for rather than predicted.
**Called only from `llama`-gated code, so every default build reports it dead**
(gotcha #264).

→ `docs/FUTURE_WORK.md` § "A context that does not fit is refused instead of shrunk"

## A prompt that closed someone else's turn is finished for the model

`chat_template::open_the_models_turn_if_the_prompt_closed_it` runs in
`build_prompt_with_model` — the choke point every prompt passes through — and
appends the family's own generation prompt when the rendered prompt ends on a
turn-CLOSING marker.

→ `docs/invariants/api-surfaces.md`

## What part of a tool-carrying reply is content is decided in one place

**`tool_parse::leading_content`** is the single answer for both non-streaming
surfaces, which each computed `text[..content_prefix_len(text)].trim()`
themselves. It adds the one rule that only makes sense once a call has been
found: **a reasoning block ended by a tool call rather than by `</think>` is
still a reasoning block.** `inference::take_leading_reasoning_block` requires
the closing tag and is right to — without one it cannot know where the
scratchpad stops. Here the call answers that.

→ `docs/invariants/api-surfaces.md`

## A tool-carrying reply streams the part that cannot be a tool call

**`tool_parse::content_prefix_len`** is the single answer to "how much of this
reply so far is certainly ordinary content", and **`tool_parse::StreamingToolText`**
is the buffer all four API paths share — OpenAI streaming and not, Anthropic
streaming and not.

→ `docs/invariants/api-surfaces.md`

## The component that will refuse must be asked while the plan can still change

Two halves of one rule, both learned from a 16 GB processor-only Mac mini that
was assigned 36 of a 48-layer 14B, refused them at load, retried, and produced
the identical plan (gotcha #452).

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

→ `docs/invariants/scheduling.md`

## A settle that cannot be written down has not happened

`credit::escrow` has three paths that change an escrow's status and then move a
balance — `release_escrow`, `refund_escrow`, `cleanup_expired`. **All three
leave the entry `Pending` and the balance untouched when the status write
fails**, and return `Err`. Two of them did; `release_escrow` warned and
reconciled anyway, which mints credits — `EscrowManager::new` re-inserts every
`Pending` entry at startup, so the settled escrow comes back claimable and the
expiry sweep refunds the full reservation. 500 in, 40 consumed, 560 out.

The trade-off direction is fixed: **lost-or-refunded, never double-paid.**

`Database::set_write_failure(Some(tree))` is how a persist failure is caused in
a test, armed at `with_write_table`. It is scoped to a TREE deliberately —
failing every write hides this class of bug, because the balance move reverts
itself and the books come out even.

→ `docs/invariants/state-and-config.md`

## A disconnect retires a session key; it must not destroy it

`SessionManager::remove_session` moves the live key into `retired` — openable,
never sealable, for `PREVIOUS_KEY_GRACE`, carrying its own replay window — and
`open` falls back to it after the current and superseded keys, including when
there is no session at all.

→ `docs/invariants/network.md`

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

## A holder record names a BUILD, not just a shard

**`ModelRegistry::shard_holders` filters out holders that positively claim a
different GGUF build**, against `expected_build_tag` — this node's own manifest
hash for that shard. It is the single read accessor for the holder map (~60
consumers), which is why the filter lives there and not at the call sites.

→ `docs/invariants/network.md`

## Additive Protocol Evolution (NETWORKING_PLAN cross-cutting)

Version-breaking network changes were a top adoption blocker: a node on vN
couldn't talk to vN±1 because a new/repurposed `SwarmMessage` variant failed to
deserialize on the other side. The rule that fixes this:

- **Never repurpose or remove a `SwarmMessage` variant** across a release, and
  never change an existing variant's wire shape incompatibly. Add a NEW variant
  instead and keep handling the old one.
- **Gate every new/optional message type on a negotiated feature.** Each node
  advertises the features it implements in `NodeCapability::features` (a `u64`
  bitfield, `swarmllm_types::features`). A sender MUST check the recipient
  advertises the matching bit before sending the new variant — an older node
  advertises `0` and is correctly skipped, so it is never handed something it
  can't decode. `features::supports(advertised, needed)` is the check;
  `features::ALL` is what this build advertises (set in `health/monitor.rs`).
  The Phase-1 relay (`features::RELAY`) is the reference example: see
  `network/manager/relay.rs::target_supports_relay`.
- **`PROTOCOL_VERSION`** (`swarmllm_types`) is the wire epoch — bump ONLY on a
  genuinely breaking change (which the first rule forbids without a fallback),
  NOT for additive feature bits. Adding a `features` bit does not bump it.
- New `NodeCapability` fields MUST be `#[serde(default)]` so older nodes'
  announcements still deserialize.

## The units decide whether a forward is a prefill, not the byte count

**`inference::pipeline::local::PipelineExecutor::forward_is_prefill(activation_bytes, units)`**
is the single answer to "is this forward doing a prefill?", for the deadline
(`compute_segment_timeout`) and for the DIAG that reports it. `SegmentBudget`
carries the resolved verdict (`is_prefill()`) so the log cannot contradict the
budget it is describing.

→ `docs/invariants/scheduling.md`

## A ticker merged into a response stream is a termination condition

`api::sse::progress_ticker` is the ONE keep-alive/progress ticker for both SSE
encoders. Its wait is cancellable — `tokio::select!` on the interval versus a
`tokio::sync::watch` finish signal, which is why the signal is a `watch` and not
an `AtomicBool`: the ticker has to *wait* on it, not merely read it.

→ `docs/invariants/api-surfaces.md`

## Event System

All events flow through `state.events.activity_tx` (ActivityEvent). Use the builder:
```rust
state.emit_activity(
    ActivityEvent::new("category", "kind", format!("message"))
        .with_model(model_id)
        .with_toast("info", 4000)
);
```

For dashboard refresh signals, use `state.events.dashboard_tx`:
- `DashboardSignal::ModelsChanged` — after shard download/load/prune/delete
- `DashboardSignal::PeersChanged` — after peer connect/disconnect
- `DashboardSignal::UpdateAvailable(info)` — after update check

There are ONLY 2 broadcast channels. Do NOT add new ones.

## Frontend Event Handling

All WS events are handled by `_handleActivityEvent()` in notifications.js. Do NOT:
- Add new WS message types (use activity_event with a new `kind`)
- Add direct `showToast()` calls for backend events (set `toast_level` on the ActivityEvent instead)
- Add direct `logActivity()` calls from WS handlers (everything goes through `_handleActivityEvent`)

## Frontend Storage

All storage keys are registered as constants on `App` in state.js (e.g., `App.MODEL_SORT_KEY`). Do NOT use raw string literals for localStorage/sessionStorage keys.

## Frontend Data Fetching

Use `App.data.loadModels()` and `App.data.loadStats()` for model/stats data. Do NOT make independent `authFetch('/api/admin/models')` calls from components — this bypasses the dedup cache.

## Frontend Component IIFE Boilerplate

Every `frontend/js/components/*.js` file opens with the same boilerplate
inside its IIFE:

```js
(function () {
  if (!window.App) return;
  var U = App.utils;   // <-- mandatory if the component calls escapeHtml / formatBytes / etc.
  // ...
})();
```

`U.escapeHtml`, `U.formatBytes`, `U.formatMB`, etc. are pulled off
`App.utils`, which is populated by `core/utils.js`. Components that
reference `U.*` without declaring `var U = App.utils` first will throw
`ReferenceError: U is not defined` at the call site — the R111
swarm-tab regression hid behind this until the Capacity Plan view
rendered for the first time. When adding a new component, copy the
existing boilerplate from a sibling file (e.g. `chat.js`).

## Active-Pipeline Guard on Manual Mutations

Anything that removes a shard file or model from a node MUST first
check whether `active_pipelines` references it, and refuse with
`SwarmError::ServiceUnavailable(...)` (mapped to HTTP 503) if so —
yanking a shard file out from under an in-flight token loop surfaces
as `ShardNotFound` mid-stream, which is unrecoverable. The
auto-manage prune path already does this via `active_pipeline_shards`
in `model/auto_manage/prune.rs`. The same guard MUST live in:

- `api/admin_models/shards.rs::delete_shard` — checks
  `seg.shard_id.model_id == mid && seg.shard_id.index == shard_index`.
- `api/admin_models/lifecycle.rs::delete_model` — checks
  `seg.shard_id.model_id == mid`.

New "delete" or "evict-from-disk" admin handlers MUST add the guard
before the destructive operation. Note that `unload_model` /
`unload_shard` (memory-only eviction) are NOT in scope — the worker
will simply re-load on next request.

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

## ACK-Timeout Fast-Fail for rr Sends

`SendDirectMessage` carries `delivery_request_id: Option<uuid::Uuid>`.
When `Some(uuid)`, the `pending_rr_observability` entry inserted in
`handle_send_rr_message` carries that uuid; the 10s
`RR_ACK_TIMEOUT_SECS` sweep in the network manager closes
`streaming_token_txs[uuid]` if no Response or OutboundFailure event
fires (libp2p rr can silently drop sends under load — observed and
documented). Caller sees Err in ~10–20s instead of `FIRST_TOKEN_TIMEOUT`
(120s). Fire-and-forget rr paths use `None`; streaming paths
(`remote-generate` fast path) MUST set `Some(uuid)`. Pair with the
`is_transient_remote_failure` retry in `dispatch_single` so a single
silent-drop transparently re-routes to a different holder.

## A tensor forward is acknowledged on receipt; a result is always its own request (2026-08-21)

`requests.rs` answers an inbound `LayerForward` with `SwarmResponse::Ack` the
moment it decodes; `tensors::handle_send_tensor_result` always sends the result
via `send_tensor_result_as_request`. There is no held `ResponseChannel` any
more — the "single substream per token" path and its map are gone.

Why: with the request held open until the result was ready, a coordinator
could not tell "computing" from "never received it", and a forward that landed
on a peer that had gone quiet cost the WHOLE segment deadline (300 s, twice in
one request on the live swarm). Now the serving node advertises
`features::FORWARD_ACK`; the coordinator's stale sweep fails an unacknowledged
forward to such a peer at `forward_ack_deadline_secs` (RTT-scaled, 10-90 s) so
the pipeline fails over in seconds. **The compute deadline stays with the
pipeline** — the sweep never reaps a slow answer, only a missing receipt; the
comment at the sweep recounts how an earlier reaper synthesised timeouts for
exactly the slow peers the measured deadlines were built for.

Rules: do not gate the fast-fail on anything but the peer's `FORWARD_ACK` bit
(an older server answers only with the result, which may take minutes); the
ACK must be sent BEFORE any work, from the network manager, so no dispatch
path can forget it; and both halves are compatible with older peers in both
directions — an old coordinator accepts an ACK response and a result arriving
as a request, an old server is never held to the ACK deadline. This also
retired the chain-specific addressing branches from earlier the same day
(gotcha #354): one rule now covers chained and unchained results.

## "Is this connection direct?" — `network::relay::addr_is_direct_transport`

The single answer, for BOTH layers that choose a connection. A relay-carried
connection looks different from each end: the dialer sees
`/…/p2p-circuit/p2p/<peer>`; the LISTENER sees a bare `/p2p/<peer>` — no
transport hop at all. `is_relay_circuit_addr` catches only the first form, and
the vendored request-response layer recorded NO address for inbound connections
(upstream behaviour), classifying `None` as direct. So a relay-carried inbound
connection was "direct", and — being the newest with nothing pending — it won
every send: two LAN nodes 0.7 ms apart measured 60-800 ms to each other and
forwarded every tensor through the anchor in Europe (gotcha #356, 2026-08-21;
gotcha #179 had fixed the classifier but inbound never had an address to classify).

Rules that follow: the vendored inbound handler records the send-back address
(`vendor/libp2p-request-response/src/lib.rs`, SwarmLLM patch); the manager's
`peer_direct_conns` is gated on `addr_is_direct_transport` (a real `/ip4`,
`/ip6` or `/dns*` hop AND no circuit), not on the circuit check alone. Any new
"prefer direct" logic uses the same predicate. `latency_ms` in the peer
registry is measured over whichever connection the rr layer picks, so a wrong
pick here is not a cosmetic error — it is added to every number routing uses.

The loop-stall tripwire in `NetworkManager::run` (per-arm, ≥100 ms → `DIAG:
network event loop stalled`) exists because this investigation's first
hypothesis was the loop; a shadow node cleared it in four minutes. Keep it.

## Completing libp2p Identify does not make a peer one of ours

**`network::manager::identify::peer_speaks_swarmllm`** is the single answer to
"is this libp2p node a SwarmLLM node?", and it is gated at the one point where
Identify turns a connection into a peer — BEFORE the Kademlia insert, so a
foreign node never enters the routing table either.

→ `docs/invariants/network.md`

## One dial per PEER, never one per address

**`NetworkManager::dial_bootstrap_peers` + `discovery::plan_bootstrap_dials`**
are how bootstrap and cached addresses are dialled. Group by target peer, one
`DialOpts::peer_id(..).addresses(all)` each, through `dial_checked`, with
`PeerCondition::DisconnectedAndNotDialing`.

→ `docs/invariants/network.md`

## Peer Cache: storable vs dialable

`network/peer_cache.rs` answers two different questions and they must not be
conflated:

→ `docs/invariants/network.md`

## ModelRegistry Holder Counts

**`merge_dht_providers` is the one writer of `shard_holders` that cannot remove
a holder** — it loops `record_shard_holder` over a DHT `GetProviders` result.
That matters because a provider record outlives the fact it asserts: libp2p-kad
keeps one for 24 h, republishes at 12 h, and other peers serve it, so a node that
deleted or lost a shard is still advertised as holding it for hours. An
add-only writer wins every disagreement with a writer that removes, and its
cadence decides how fast.

→ `docs/invariants/network.md`

## Cross-feature compile checks

`cargo check` with default features does NOT compile any `cfg`-gated
path. Nothing local sees them: `cargo fmt`, `cargo clippy --all-targets`,
the whole test suite and the pre-push hook are all default-features, and
so is the per-push CI run. **The only signal for the GPU paths is the
cache-warm workflow**, which does not run on every push.

Two gates matter here:

- **`llama`** — `pipeline/dsd.rs` and the spec/llama-gated code in
  `pipeline/speculative.rs`. Verify with `cargo check --features llama`,
  which is cheap. R91 caught a regression R90 had let through.
- **`flash-attn` / `cuda`** — the CUDA arm of
  `inference::layers::run_attention` and anything else under
  `#[cfg(feature = "flash-attn")]`. `cargo check --features flash-attn`
  works locally when `nvcc` is present (set `CUDA_COMPUTE_CAP=80` to match
  the release build) but compiles the kernels, so budget tens of minutes.

  **It must be `--all-targets`, and that is not a detail.** CI runs
  `cargo check --locked --features flash-attn --all-targets`; a plain
  `cargo build --features cuda` does NOT compile test code, so a gated
  `#[cfg(feature = "flash-attn")]` **test** is invisible to it. That is
  precisely how main went red on 2026-08-10: `run_attention` gained a
  parameter, every production caller was updated, and one caller inside a
  flash-attn-gated benchmark test was not — through a release `--features
  cuda` build, a default `--all-targets` clippy, 1819 passing tests and a
  green pre-push hook. **Changing the signature of anything callable from
  gated code means running the gated check with `--all-targets` before
  pushing.** Grep for the symbol first: `grep -rn "the_fn(" src/` shows the
  gated callers that no default build will compile.

  A debug-profile `cargo check --features flash-attn` rebuilds the kernels
  (tens of minutes) even though the release profile may already have them.
  Adding `--release` reuses them and is much faster, but then the
  **integration-test targets fail spuriously**: `Database::open_temp` is
  `#[cfg(any(test, debug_assertions))]`, and `--release` turns
  `debug_assertions` off, so `tests/integration/*` stop compiling with a
  wall of `no associated function named open_temp`. That is the profile,
  not a regression. Read which TARGET failed — `lib test` is the one that
  carries the gated unit tests and the one CI reports.

**The specific trap, which has now fired (gotcha #264): an import used only
inside a `cfg`-gated arm is reported UNUSED by every local build.** Acting on
that advice — which clippy gives confidently, and which is correct for the
configuration being compiled — deletes a symbol the GPU build needs, and
nothing local goes red. `DType` in `layers/mod.rs` is annotated
`#[cfg_attr(not(feature = "flash-attn"), allow(unused_imports))]` for exactly
this reason.

So: **before removing anything an unused-warning points at, grep the file for
`#[cfg(`.** If the file has gated arms, the warning is only telling you about
one configuration. And after pushing a change that touches gated code, check
the cache-warm run rather than assuming a green CI means the GPU builds work —
`gh run list --workflow="Cache warm"`.

### The CUTLASS kernels live OUTSIDE `target/` (2026-08-17)

`candle-flash-attn`'s 19 kernels are built into `.flash-attn-build` (via
`CANDLE_FLASH_ATTN_BUILD_DIR`, set by `.github/actions/gpu-build-env`) and cached
separately from the Rust build cache.

**Why, and the trap to remember**: `Swatinem/rust-cache` deletes everything in
`target/` belonging to a package whose manifest is inside the repo — which every
crate under `vendor/` is. So both GPU jobs restored a cache reporting
`full match: true` and then recompiled all 19 kernels anyway, ~39 min of the
Windows GPU build and ~27 of the Linux one, on every release, for months. Nothing
ever went red; the only symptom was 39 minutes of silence in the log between the
last `Compiling` line and the build script's output. **"Cache hit" is not "the
slow thing was cached" — read the compile lines, not the restore line**
(gotcha #318).

It works because `cudaforge`'s own `BuildCache` skips up-to-date kernels by
CONTENT HASH rather than mtime, so a directory restored from a tarball is
accepted. A warm run logs `All kernels up-to-date, skipping compilation`; that
line, plus `Cache restored from key: flash-attn-kernels-*`, is how you confirm
the mechanism fired rather than inferring it from a faster wall clock.

Two things a change here must preserve, both learned the hard way:

- **Create the directory.** Upstream panics `Directory doesn't exists` unless the
  override path already exists — i.e. on the very first run after introducing it.
- **An empty env var is not an unset one.** `std::env::var` returns `Ok("")` for a
  variable that is set but empty, and a matrix expression like
  `${{ matrix.x && '…' || '' }}` yields exactly that for every cell that does not
  want the override. The vendored build script filters empty explicitly.

The CI `flash-attn` compile-check cell points the build script at a
non-existent temp dir with `CANDLE_FLASH_ATTN_CHECK_ONLY=1`, which exercises this
patch in ~50 s on every push — nothing else in CI compiles that crate, because
compiling it is the cost being avoided.

## A source-scanning guard is only as good as the spellings it knows (2026-08-30)

`tests/repo_consistency.rs` is where this project stops known past mistakes from
coming back. Five of those guards were tested by **planting the violation each
one exists to catch. Four did not notice** — all had been reporting success for
months (gotcha #413).

Three rules follow, and they are cheap.

**Scan statements, not lines.** `statements(text)` joins continuation lines and
closes the gap a wrapped chain leaves before its `.`, so
`s.metrics\n    .node_stats` reads back as `s.metrics.node_stats`. Every guard
matching a dotted path or a field-plus-operation must use it.
`self.shared_state.metrics.node_stats.requests_served_atomic.fetch_add(1,
Ordering::Relaxed)` is 99 characters at two levels of indentation, so **one more
nesting level and rustfmt splits it across four lines** — which is the ordinary
shape, not an edge case. It blinded the serving-accounting guard, the
per-request-state guard, the live-config guard (the one against #281's fourth
recurrence), the update-reporting guard, the advertised-load guard and the dial
guard.

**Take the whole body, never a character window.** The VRAM guard read
`src[start..start + 1600]` of a function 1698 characters long and was blind to
its last 100, where a planted boot-snapshot read passed. It was brittle in both
directions: the `cfg()` call its positive assertion depended on sat at offset
1568, **twenty characters inside the cap**, so twenty characters of unrelated
growth would have failed it on correct code. `fn_body(src, signature)` takes to
the closing brace in column zero. For the same reason, never match a literal
carrying indentation (`"shared\n        .config\n        .resources"`) — that is
pinned to whatever rustfmt produced the day it was written.

**A file-level `contains` is not a site-level claim.** The prompt-privacy guard
asserted each FILE mentions the send somewhere, the setting somewhere and
`cfg()` somewhere. Both files make unrelated `cfg()` calls, so the third
assertion was satisfied unconditionally and none of the three established the
gate was on the path; a second ungated send in either file kept it green. Use a
proximity window sized from the real code.

**And give every scan a self-test that plants the violation.** A repo-wide scan
that finds nothing is indistinguishable from one that *cannot* find anything, so
the scanner's reach has to be pinned the way any other behaviour is —
`the_statement_scanner_sees_a_chain_rustfmt_has_wrapped`,
`the_unbuffered_gguf_guard_catches_every_form_of_the_defect`,
`the_boot_snapshot_check_is_not_pinned_to_one_formatting`,
`the_prefix_sharing_guard_catches_an_ungated_send`.

**How to test one by hand**: a stray `.rs` under `src/` that no `mod`
references. The scanner walks the directory and finds it; the compiler never
sees it, so it need not even compile. Delete it afterwards.

**A guard too weak to fire is also too weak to be checked for correctness.**
Strengthening the #281 guard is what surfaced a real contradiction in its own
field list — it forbade all seventeen `.config.auto_manage.` fields while its
own comment stated the principle that only the four the Settings panel exposes
are live-settable, the rest being config-file/CLI where the boot value is
CORRECT.

## A report built to be handed to a stranger is a publishing surface (2026-09-01)

**`network::redact::redact_addresses`** hides every host in the diagnostics
report — an IP literal or a DNS name, in a multiaddr or in free prose — and is
called at the ONE point `api::admin::diagnostics` returns. `?full=1` is the
deliberate opt-in for an operator debugging their own machine;
`swarmllm diagnostics --full` and `examples/two_node_test.sh` are the two
callers that ask for it. `the_diagnostics_report_hides_addresses_unless_full_is_asked_for`
in `tests/repo_consistency.rs` fails the build if the default changes, and
checks the two surfaces that hand the report to a person still ask for the safe
form.

→ `docs/invariants/network.md`

## One invariant, N paths — the recurring bug of this codebase

The single most repeated defect here is a **shared invariant implemented per
path**, where fixing the path in the bug report leaves the others broken. It
recurred *seven times* on 2026-07-25/26 alone: stop-string application, tool-call
buffering (twice), `include_usage` emission (twice), control-token scrubbing, and
`strip_provider_prefix`. In every case a correct helper already existed and one
consumer didn't call it.

**Before fixing anything in the request/response path, enumerate the paths.**
There are more than you expect:

- **Inference text sources (THREE)** — `inference/executor.rs` (in-process),
  `inference/process_pool.rs` (worker subprocess), `inference/pipeline/
  distributed.rs` (assembled from remote segments). A reply-content rule belongs
  at all three. Note the cold-start request takes the *distributed* path while
  later ones take the split path, so a per-path bug can look fixed five times
  and leak on the sixth.
- **OpenAI response paths** — `router_inference` + `split_non_stream_response`
  (non-streaming), `router_inference_stream` + `split_stream_response`
  (streaming).
- **Anthropic response paths** — `anthropic_non_stream` +
  `anthropic_split_non_stream`, `anthropic_stream` + `anthropic_split_stream`.
  The `_split_` variants are the local-complete fast path; the others go via the
  router.
- **Responses API** — `run_streaming` (foreground) and the background task's own
  chat request in `responses/background.rs`. They share the event loop but build
  their chat requests separately, so an opt-in set on one is absent on the other.

**A shared helper is not enough — put it where the caller cannot skip it.**
This was the standing advice here, and it kept failing: `with_template_stops`,
`emit_openai_tool_calls`, `emit_anthropic_tool_blocks` and
`strip_control_token_artifacts` all existed, were documented, and were still
missed by a sibling path. A helper nobody is *obliged* to call will eventually
not be called. Three escalating ways to make it obligatory, best first:

1. **Do it at the choke point, not in the callers.** Find the single place the
   value crosses the boundary and transform it there.
   `providers::strip_prefix_in_body` now runs inside `try_proxy_openai`,
   `proxy_to_anthropic` and `proxy_via_subprocess_anthropic` — the three
   functions that actually send — so a new proxy path is correct with no
   author action. Same shape for `inference::finalize_reply_text`: the three
   reply-text sources call one finaliser that owns the whole ordered sequence
   (scrub → truncate → trim → newline cleanup), instead of each composing those
   steps itself, which is how they silently diverged.
2. **Make the wrong call unrepresentable.** If context is needed to be correct,
   make it a required parameter rather than an `Option` with a convenience
   wrapper that passes `None` — that wrapper is how `build_prompt` disabled the
   template fallback on 6 of 7 paths (gotcha #171).
3. **Assert the property on the shared helper**, not once per path, so a new
   path inherits the coverage instead of needing its own test.

Only when none of those fit should you fall back to a doc comment saying
forgetting it is the bug.

**Verify by running the request, not by reading the diff.** Every one of the
seven passed review. The ones caught early were caught by executing the actual
path — and where a report names a specific model, that model is part of the
reproduction (gotcha #168).

**Bad reply content is evidence about the PROMPT first, the output second.**
The `<|im_end|>` leak was chased across four releases as an output-scrubbing
problem. It was a prompt problem: `apply_chat_template` returned `None` for
every official Llama-3.x template, and the fallback chain reached ChatML, so a
Llama-3 model was asked a ChatML question and answered in ChatML (gotcha #169).
Before touching `strip_control_token_artifacts` or the stop-string list, check
`grep "chat template failed" node.log` — that WARN names the real fault and had
been firing on every request for several releases. `build_prompt_with_model`
falling back at all is a bug report, not a safety net: the fallbacks
(gemma/vicuna/llava/ChatML) exist for models that ship no template, and any
model that DOES ship one should be rendering it.

## API errors must be readable by the caller

Every failure the API can produce has to come back as
`{"error": {"message", "type", "param", "code"}}`. Two ways to break that, both
of which shipped:

- **Using axum's `Json<T>` as a request extractor.** Its rejection is raw text
  with a 422. Nine handlers used the `JsonBody<T>` wrapper and 27 did not, so
  most admin, model and pool endpoints returned something the dashboard could
  not read. `getApiErrorMessage` does `await resp.json()` inside a try/catch, so
  raw text throws, the catch swallows it, and the user gets the generic fallback
  with the real reason discarded — every one of those endpoints could only ever
  say "action failed". Use `JsonBody<T>` in the request-body position.
- **No `.fallback()` on the router.** An unrouted path returned a bare 404 with
  an empty body. `/v1/completions` is the case that matters: OpenAI deprecated it
  but plenty of tooling still calls it, and an empty 404 gives no hint that
  `/v1/chat/completions` exists. `unknown_route` now answers in the envelope and
  names the replacement.

Choose the STATUS from the cause, not from where the error came from.
`probe_failure_is_user_fixable` is the pattern: a mistyped HuggingFace repo is a
404 the caller can act on, while a rate limit or an upstream outage stays a 502.
Reporting a typo as `502 Bad Gateway` says this server is broken about something
in the caller's own input.

## Timeouts: bound what actually varies

A fixed deadline is only correct when the work behind it has a fixed size.
Where it does not, the constant silently becomes a **minimum-capability
requirement for the user** that nobody chose deliberately. Five instances were
found in one night (2026-07-27, gotcha #190):

- `UPDATE_DOWNLOAD_TIMEOUT_SECS = 300` against a ~933 MB GPU build required a
  sustained ~3.1 MB/s. Anyone slower could **never** complete an update.
- `HF_DOWNLOAD_TIMEOUT_SECS = 3600` required ~145 KB/s for a 512 MB shard.
- `INFERENCE_FORWARD_TIMEOUT_SECS = 120` capped a question forwarded to a peer
  regardless of prompt length.
- `PROVIDER_PROXY_TIMEOUT_SECS = 300` was documented as being about time to the
  first token but enforced on the whole exchange, cutting off cloud replies that
  were still streaming.
- `REQUEST_TIMEOUT_SECS = 300` capped every HTTP request, generation included —
  and so silently capped the prompt-scaled first-token budget at 300s no matter
  what it was raised to.

Rules that follow:

1. **Prefer an inactivity timeout to a total one.** `reqwest`'s `read_timeout`
   (0.12+) catches a stalled transfer just as fast while leaving a slow healthy
   one alone, and requires no guess about size or bandwidth. Use it for every
   download and every streamed proxy response.
2. **Where inactivity does not apply, scale the budget by the input and cap it** —
   `pipeline::remote_generate::first_token_timeout(prompt_tokens)` is the shared
   helper; call it rather than inventing another rule. Prefill is linear in
   prompt length and is ~99% of a long request.
3. **Generation gets no blanket deadline.** Routes that can run a model are
   merged into the router OUTSIDE the `TimeoutLayer` (`generation_routes` in
   `api/server.rs`). The merge MUST stay before the auth layer or those
   endpoints answer without a key — pinned by
   `generation_routes_still_require_a_key`.
4. **When you change a limit, grep the whole path for other limits.** A budget
   is only as generous as the tightest ceiling above it, and that ceiling is
   usually in another file, in middleware, behind a comment that went stale
   before the code did.
5. **Read the comment against the code.** In four of the five, the comment
   reasoned about one quantity ("before the first token") while the constant
   bounded another (the total). A stale comment asserting an invariant reads as
   verification and stops anyone re-deriving it.
6. **The frontend is part of "the whole path".** The five instances above were
   all in Rust, and the tightest ceiling on a comparison request turned out to
   be a hardcoded 45 s `AbortController` in `frontend/js/components/compare.js`
   — on the very requests the daemon deliberately serves outside its own
   `TimeoutLayer`. It discarded replies the daemon had finished computing
   (report #009: `execute_ms=44886`, `finish_reason=stop`, aborted a fraction of
   a second earlier), and the duration was baked into the translated string in
   all 21 locales, so it was not even greppable as a number. A generation
   request from the browser gets no client-invented deadline either; where one
   is unavoidable it is derived from what the daemon permits and says something
   true when it fires — the node may still be working, and the reply was not
   necessarily lost.
   **Streaming is what makes rule 1 available to a browser**: a non-streaming
   `fetch` has no intermediate bytes, so it cannot have an inactivity timeout.
   The chat tab streams, which is why its 30 s `authFetch` default bounds only
   time-to-headers and is harmless there.

## Config defaults must stay live

The daemon must write **only values that differ from the compiled default**.
`config::to_minimal_toml` is the one serializer for the config file; do not call
`toml::to_string_pretty(&config)` directly.

→ `docs/invariants/state-and-config.md`

## Attention kernel choice and the query-length cliff (2026-08-23)

Four helpers now own decisions that used to be spread across call sites. All
four exist because a predicate that *reads* obviously correct was answering a
different question than the one that mattered.

→ `docs/invariants/inference.md`

## Local speculative decoding — `inference::model_worker::ngram_spec_eligible`

The single answer to "may this request be speculated?", consulted by BOTH the
slot-admission gate and the decode loop. Two copies would eventually disagree,
and the failure is silent: the gate diverts a request off the batched path and
the loop then declines to speculate it, so it loses batching and gains nothing.

→ `docs/invariants/inference.md`

## A prompt's length in tokens is a POSITION, not a statistic

**`inference::pipeline::prompt::prompt_positions`** is the single answer to "how
many positions does this prompt occupy?", for all five sites in that module.
Never open-code it, and never estimate it.

→ `docs/invariants/inference.md`

## A peer's stated reason is the answer; do not substitute one of your own

Two helpers in `inference::pipeline` own what happens when a serving node
refuses a forward. Both exist because the correct handling was implemented on
the **verify** hops and missing on the **prefill** hops, in the same files, with
a comment on one of them explaining exactly why it mattered.

→ `docs/invariants/scheduling.md`

## A generation loop that blocks its thread must be told to, and a full buffer is not a departed client

**`inference::executor::without_starving_the_runtime`** wraps every in-process
`executor.generate*` call: that loop never yields, and on a Tokio worker it stops
the runtime draining the response, so a streamed reply does not stream at all.
**`api::sse_send_live_blocking`** is what a generation callback sends with —
`try_send(..).is_ok()` reads a FULL channel as a departed client and ends the
reply at the buffer's capacity, reported as a natural `stop`. Terminal
`finish_reason` events go through it too.

→ `docs/invariants/api-surfaces.md`

## A streaming path must announce that it finished

`api/openai/streaming.rs` treats "no finish event arrived" as "this path never
streamed" and falls back to emitting the whole `InferenceOutput.content` as one
delta. So a coordinator that streams tokens and then returns without sending a
terminal `finish_reason: Some(..)` does not merely omit a marker — it
**duplicates the entire reply**.

→ `docs/invariants/api-surfaces.md`

## A stream that fails must say so (2026-09-01)

`finish_reason` carries only what the OpenAI spec defines — `stop`, `length`,
`tool_calls` — and none of them means "something went wrong". A failure on the
OpenAI streaming surface is `StreamEvent::Error`, typed through
`classify_error`, and no finish delta at all; the Anthropic sibling is
`AnthropicSseEvent::Error`. `a_stream_that_fails_never_pretends_the_model_chose_to_stop`
in `tests/repo_consistency.rs` fails the build on an `Err` arm that produces the
literal `"stop"` within a few lines.

→ `docs/invariants/api-surfaces.md`

## Two counters both called "tokens" — write down which event each one counts

`StreamReassembler::truncated()` is the single answer to "did tokens the peer
SENT fail to arrive?". It is deliberately NOT `usage.completion_tokens >
emitted()`.

→ `docs/invariants/api-surfaces.md`

## A model's turn-ender is found in its vocabulary, not taken from its declared EOS

**`GgufTokenizerMeta::end_of_generation_ids_from_vocab`** searches the
vocabulary BY NAME for the tokens that end a reply, and every path that resolves
EOS ids merges it in — `eos_tokens_with_arch_fallback` and
`split::entry::SplitModelEntry::from_header`. A declared EOS is trusted but
never assumed COMPLETE: the per-family id lists only ever ran when a GGUF
declared nothing, so a model that declares one token and ends its turns with
another got no help from them at all.

Phi-3/3.5/4 are that model. They declare `<|endoftext|>` and close every turn
with `<|end|>`, and nothing stopped the reply there — it ran to `max_tokens`
inventing further turns, visibly on a GPT-2-BPE vocabulary and invisibly on a
SentencePiece one.

**`<|end|>` is conditional, and the condition is the whole reason to read
upstream first.** For harmony (gpt-oss) and solar-open it separates messages
inside one reply, so stopping on it truncates every such reply at its first
message. Both the EOS search and `chat_template::extract_stop_strings` carry the
same exclusion, keyed on the same neighbours llama.cpp keys it on.

→ `docs/invariants/inference.md`

## Partial RoPE has one implementation, and it answers with a tensor the KV cache can write

**`inference::layers::rope_over_heads`** is the single implementation of "rotate
the leading `rope_dim` of each head, pass the rest through". Its result is
contiguous, and the pass-through half is made contiguous BEFORE the `cat`, not
the whole head after it.

`Tensor::cat` answers with a transposed VIEW rather than a fresh buffer when any
argument is non-contiguous and `dim != 0`. `slice_set` refuses a non-contiguous
source and is how the KV cache writes K, so two copies of this branch — one in
`LayerWeights`, one in `Qwen35AttnWeights`, both leaving the pass-through as a
`narrow` view — killed every request on every partial-RoPE model: Phi-4-mini,
GLM-4, Qwen 3.5. The discriminator is `rope_dim < head_dim`, not GQA.

`SeqCache::append` makes its source contiguous too, so a new producer of K or V
cannot bring the class back.

→ `docs/invariants/inference.md`

## A vocabulary piece becomes token ids in exactly one place

**`inference::tokenizer::BpeTokenizer::push_piece_ids`** is the only way a merged
piece is turned into ids on the BPE path. There are two sites that need it — the
single-character early return and the output walk — and **both were
`.unwrap_or(0)`**, i.e. `<unk>`.

→ `docs/invariants/inference.md`

## Centralised Wire-Format Helpers


These helpers exist as the single source of truth for invariants that
silently break at the wire if duplicated:

- **`network::protocol::build_layer_forward_aad`** — encryption AAD
  bytes for `LayerForward` envelopes. Both encrypt
  (`network/manager/tensors.rs`, `network/pipeline_stream.rs`) and
  decrypt (`decode_layer_forward_encrypted`) MUST go through it.
  Adding a new authenticated field to `LayerForward` means extending
  this helper, not appending bytes on the encrypt side. Post-R100,
  the helper covers the cleartext header AND the spec/kv-truncate
  trailer fields; the decoder reconstructs AAD via the helper after
  parsing trailers (since trailer bytes don't appear contiguously
  on the wire — sealed payload sits between header and trailers).
  Post-R139, also covers the chunk-meta trailer (0x05) so chunked
  STREAM frames can't be reordered / truncated / substituted across
  transfers without Poly1305 rejection.
- **`network::pipeline_stream::chunk_layer_forward`** (R139) — splits
  a `LayerForward` at byte-offset boundaries into K chunks for
  STREAM-style chunked send. Returns the input verbatim wrapped in a
  single-element Vec when `activations.len() ≤ chunk_size_bytes`
  (single-chunk implicit fallthrough — no chunk_meta on the wire).
  Sender call sites that opt into chunked send MUST go through this
  helper rather than re-implementing the split; the chunk_meta
  values it sets are the contract the receiver's
  `try_assemble_chunked_forward` and `build_layer_forward_aad` both
  rely on.
- **`SharedState::resolve_pending_layer_result`** — the ONLY way to deliver a
  `LayerResult` into `pending_layer_results`. Never `remove(&request_id)` +
  `tx.send(...)` from a network path. The map is keyed by `request_id`, but a
  request that has failed over has TWO forwards outstanding: the abandoned one
  and the standby's. Resolving by id alone lets the abandoned forward's late
  error (from `fail_tensor_forward`, `fail_pending_forward`, or the
  stale-forward sweep) consume the standby's waiter — which then discards the
  standby's genuine result and surfaces the empty payload downstream as
  `Internal: Tensor bytes too short`. Observed live 2026-08-01: a request that
  would have completed in ~10s via failover failed after 181s (gotcha #229).
  Waiters record the node they expect in `PendingLayerResult::awaiting`; the
  helper checks and takes in one atomic `remove_if`. A bare `remove` is only
  legitimate for owner-side cleanup — a coordinator dropping its OWN waiter on
  an error path, or the health monitor's stale sweep.
- **`daemon::dispatch::timestamp_fresh_one_sided`** — generic
  one-sided staleness check (R94). Time units must be consistent
  across `ts`/`now`/`max_age`/`skew`. Use directly for any new
  timestamp gate; gossip and pre-signed-message helpers below are
  thin wrappers around it.
- **`daemon::dispatch::gossip_timestamp_fresh`** — private `u64`-ms
  wrapper used inside `daemon/dispatch/mod.rs` itself for the four
  inbound gossip handlers (`RegionShardSummary`, `ModelDemandGossip`,
  `WishlistAnnouncement`, `PoolModelAvailability`). The
  `network::manager::events.rs` GossipSub pre-filter uses
  `timestamp_fresh_one_sided` directly via an inline closure — both
  sites share the same one-sided invariant, but via the underlying
  primitive rather than this wrapper.
- **`credit::ledger::check_signed_freshness`** — one-sided staleness
  check for `chrono::DateTime<Utc>`-typed signed messages (balance
  reports, credit transactions, pool removals). Constants
  `CLOCK_SKEW_TOLERANCE_SECS` / `BALANCE_REPORT_MAX_AGE_SECS` are
  `pub(crate)` so all callers share the same window (gotcha #32). R94
  routed `pool/manager::handle_inbound_removal` through here.
- **`pipeline::pack_verify_tokens_to_le_bytes`** (R93) — packs `&[u32]`
  speculative-verify tokens as i64-LE bytes for the worker's
  multi-token decode branch. Shared by `speculative.rs::send_verify_batch`
  and `dsd.rs::forward_verify_through_segments`.
- **`pipeline::build_spec_verify_forward`** (R93) — constructs the
  18-field `LayerForward` envelope for spec verify (R139 added the
  18th field, `chunk_meta`). Adding a new field
  to `LayerForward` extends this helper, not the call sites.
- **`pipeline::build_kv_truncate_forward`** (R95) — sibling helper
  for stop-sequence KV-truncate signals (empty activations,
  `spec_logits_requested: false`).
- **`pipeline::register_pending_layer_result`** (R93) — cap-check +
  oneshot insert + `PendingLayerResultGuard` RAII (gotcha #45). Used
  by speculative prefill, speculative verify, and DSD verify;
  `distributed.rs` keeps two inline call sites that need `&mut self`
  or skip the cap during failover.
- **`storage::Database::with_write_table`** (R96) — opens a write
  transaction, runs a closure on the data table, commits on `Ok` or
  rolls back on `Err`. Used by `put_json`, `insert_raw`, `remove`,
  `clear_tree`, `replace_tree`. Read-side dedup deferred (lifetime
  constraints on `ReadOnlyTable`).
- **`swarmllm_types::ShardResponse::empty()`** (R97) — canonical
  empty/error response for refused requests, queue-full rejections,
  and disk read/seek/open failures. 8+ rejection sites across
  `network/manager/{requests,shard_transfer}` go through it.
- **`swarmllm_types::LayerResult::error(request_id, reason)`** (R106)
  — canonical empty/error LayerResult for failed pipeline forwards.
  Five rejection sites (`network/manager/{tensors,requests,mod}.rs`,
  `network/pipeline_stream.rs`, `daemon/dispatch/layer_forward.rs`)
  go through it. Adding a new field to `LayerResult` only requires
  updating this constructor — mirrors `ShardResponse::empty()`.
- **`network/manager/connections::try_enqueue_redial`** (R97) —
  dedup + cap + push for `pending_redial`. Used by both the
  active-pipeline and unregistered-peer reconnect paths.
- **`responses::types::raw_tool_kind_or_unknown`** (R93) — extracts
  the `type` field from a `ToolDef::Raw` JSON value with `<unknown>`
  fallback. Used by both Chat and Anthropic `translate_tools` error
  arms.
- **`cli::bail_if_no_api_key` / `cli::exit_daemon_unreachable`** (R96)
  — the canonical "daemon not running" / "daemon unreachable"
  messages. Used by `cli::{bench, chat, peers, status}`.
- **`model::auto_manage::spawn_check_and_load`** — canonical
  "shard landed → reload model → refresh dashboard" spawn. Always
  performs the three steps together: compute_vram_budget →
  check_and_load_model → signal_dashboard(ModelsChanged). Used by
  `api/admin_models/shards.rs::delete_shard`,
  `network/manager/requests.rs` shard-download landing, and
  `model/acquisition.rs::register_model`. New paths that complete a
  shard or shard-set acquisition MUST go through this helper rather
  than open-coding the three-step sequence.
- **`pool::invite::{encode_invite_code, decode_invite_code}`** (R140) —
  canonical `swarmpool://` v2 invite code codec. Encode JSON-serializes
  `InviteCodePayload` → ChaCha20-Poly1305 seals with a random embedded
  key → base64url; decode reverses with version + expiry + token-length
  validation. The decoder normalizes ANY user-pasted error to
  `SwarmError::Validation` (clean UX message) rather than `Internal` —
  the most likely failure cause is a truncated/mistyped paste, not a
  daemon bug. New entry points that accept v2 codes (CLI, MCP tool, web
  API) MUST go through `decode_invite_code` rather than parsing the
  blob manually. Adding a field to `InviteCodePayload` requires bumping
  `INVITE_VERSION` AND updating the decoder's mismatch error to point
  users at a daemon upgrade. `pool::invite::looks_like_v2` is the
  prefix-sniff helper used by API + frontend to route between v2 and
  the legacy 8-char path.

## The single-source-of-truth helpers, by topic

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see “One invariant, N paths”.
**Read the topic file before changing one.**

### API surfaces, errors and streaming → `docs/invariants/api-surfaces.md`

- **`api::mcp::dispatch::spawn_model_call_task`** — the single place that decides whether a fan-out model call actually **answered**, as opposed to merely not erroring.
- **`crate::error::reclassify_flattened_error`** — recovers an error's CLASS from a message that crossed a boundary carrying no types.
- **`crate::error::classify_error`** — the single answer to "what is this failure, to a caller": `(StatusCode, client-safe message, error type)`.
- **`crate::error::failure_log_level` + the `log_failure!` macro** — the single answer to "how loudly should this failure be recorded in THIS node's log".
- **`crate::error::error_hint_with_key`** — returns the actionable hint as a stable `(key, english)` pair, from ONE match arm.
- **`AnthropicSseEvent::Error`** — the ONLY way the Anthropic streaming surface reports a failure.

### Inference kernels, caches and the tokenizer → `docs/invariants/inference.md`

- **Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul** — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`.
- **`inference::decode_attn::gqa_decode_attention_cpu`** — single-position attention straight over the KV cache in its stored `[b, kvh, S, d]` layout, one rayon task per (batch, kv head).
- **`inference::fast_math`** — eight-lane AVX2 `expf` (`exp_inplace`, Cephes polynomial, ~2 ulp vs libm, pinned by `vectorised_exp_tracks_libm` over [-80, 80]) and the fused `silu_mul` CustomOp2. A new elementwise pass that calls `f32::exp` in a loop routes through here instead.
- **`inference::cpu_pools::in_phase_pool`** — binds a forward pass to the CPU thread pool that suits its phase, at ONE choke point: `SplitModel::forward_inner_impl` and `forward_batch`.
- **`inference::layers::new_kv_cache`** — the only way to construct a KV cache.
- **`inference::split::kv_cache::LayerKv`** — one layer's KV cache: the f32 BHSD cache every path reads, plus an optional f16 BSHD mirror for the CUDA flash kernel.
- **`inference::split::kv_cache::SeqCache` / `KvPair` + `LayerKv::truncate`** — the KV cache buffer is this project's own, not candle's, for ONE reason: candle's `Cache` keeps its length private, so the only way to keep the first `n` positions was snapshot + `reset()` + `append()` — two full copies per layer on every rejected speculative draft. **Never re-introduce a copy on the rollback path.**
- **`inference::attn_softmax::scaled_masked_softmax`** — the single expression of attention's tail: scale, optional Gemma-2 logit soft-cap, additive mask, softmax.
- **`inference::layers::standard_attention` grouped GQA decode** — (c4cc3b16, 2026-08-16) for `q_len == 1` with `n_kv_head < n_head`, standard attention no longer expands the KV cache with `repeat_kv`; it reshapes the query heads into matmul rows against the UNEXPANDED cache.
- **`inference::layers::cuda_decode_prefers_standard`** — on CUDA, `q_len == 1` takes standard for EVERY head geometry, prefill always flash. The GQA exclusion was retired on 2026-08-23 once `grouped_gqa_decode_attention` deleted the `repeat_kv` cost it existed to route around; `SWARMLLM_GQA_DECODE_FLASH=1` restores the old rule for an A/B inside one binary.
- **`inference::mem_bandwidth::measured_gbps`** — what this machine's memory actually delivers, measured once and cached.
- **`inference::cancel::unless_cancelled` — every wait that can run for minutes watches the request's cancel flag** — `InferenceRequest::cancel` is the ONE cancellation signal — set by `CancelOnDisconnect`, by both SSE surfaces on `sse_tx.closed()`, and by `/cancel`; read around every WAIT, never around a send.
- **A prompt pass asks between layers whether its request was cancelled** — `KvCacheStore::set_cancel_oracle` is probed once per layer by `forward_inner_impl`, which returns `CANCELLED_MID_FORWARD`; `forward_was_cancelled` is the one reader of that message.
- **`inference::split::token_embedding::rows_on_demand_eligible`** — the single answer to "is this model's `token_embd.weight` held quantized with its rows dequantized on lookup, or dequantized whole at load?".
- **`inference::split::read_gguf_header`** — the single way to parse a GGUF header off a PATH, and the buffering is the entire reason it exists.
- **`inference::split::GgufTensorMeta::tied_output_location`** — the single definition of "is this model weight-tied", i.e. does it reuse `token_embd.weight` as the LM head instead of shipping an `output.weight`. Both sidecar writers and the reader go through it.

### Worker memory: graphics, RAM and the KV cache → `docs/invariants/memory.md`

- **`inference::worker_ipc::worker_error_is_fatal`** — the single source of truth for "did this worker error destroy the worker's device state, or just this request?".
- **`daemon::shard_loader::force_cpu_for`** — the single mapping from `inference.gpu_layers` (`-1` auto / `0` CPU only / `>0` GPU) to the loader's `force_cpu` flag.
- **`daemon::gpu_support::MIN_COMPUTE_CAP` + `local_gpu_is_supported`** — the single answer to "can this card run OUR kernels?".
- **`model::auto_manage::vram::ADMISSION_KV_CONTEXT`** — the context length admission charges KV cache for, on either device, whatever the user configured.
- **`ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`** — reclaim graphics memory from models nothing is using rather than demoting the requested one to the processor.
- **`should_return_to_gpu` + `ModelProcessPool::worker_should_return_to_gpu`** — the single answer to "is this resident worker still in the right place?", asked on the request path in `get_or_spawn` rather than on a timer.
- **Graphics memory has ONE owner: `ModelProcessPool`** — it admits (`admit_to_gpu`), charges (`vram_reserved_mb`) and reclaims (`free_vram_for_admission`, `try_idle_vram_unload`). Nothing else may take memory away from a loaded model.
- **`model::auto_manage::storage_budget` is the ONE answer to "how much shard storage may this node hold?", and `held_shard_bytes` the one answer to "how much does it hold?"** — `storage_budget_now(&state)` gives both, live; the download pass, prune's disk pressure, the settings storage bar, the pool page and the diagnostics report all read it instead of re-deriving one.
- **`model::auto_manage::prune::effective_idle_secs` — residency is a hard UPPER BOUND on "idle since", and the worker's own `last_used` is the signal that moves** — NOT `model_trust.last_request_at`, which nothing in the current code writes — a stale persisted value once unloaded a model five seconds after it answered.
- **An admitted prompt is RECORDED, not just decided** — `KvCacheStore::record_prompt_admission` / `outstanding_admission_bytes`; `ensure_room_for_prompt` adds the outstanding total to the live figure before `admit_prompt`, and records its own claim once admitted.
- **`inference::split::kv_budget::admit_prompt` + `PrefixCache::release`** — ONE decision for a whole prompt, before prefill, charging live caches PLUS the prefix cache's snapshots (the same device memory, previously charged nowhere): fit → evict cached prompts, oldest hit first → refuse with a 503 at token 0. **A budget must see every tenant of the memory it bounds.**
- **`inference::split::kv_budget`** — the KV memory budget and the admission check against it.
- **`inference::process_pool::worker_socket_path`** — the worker IPC socket path, and the ONLY place it is built.

### Network protocol, peers and the model registry → `docs/invariants/network.md`

- **`inference::pipeline::remote_generate::StreamReassembler`** — the single place a remote reply's token stream is put back in order.
- **A hole in a peer-served reply is FILLED, not waited out** — `RetainedReplies` keeps each fast-path reply this node streams, and `SwarmMessage::ResendTokens` is answered from it — only to the peer the reply was for.
- **`NodeCapability.cpu`** — a processor described the way a graphics card always has been.
- **`PeerInfo::ack_srtt_ms` is what routing prices a peer by** — written on every acknowledged tensor forward from `AckRttEstimator::srtt_ms`, capped at `ACK_SRTT_ROUTING_CAP_MS` (10 s) because the estimator DOUBLES on a miss.
- **`mem_bandwidth::remeasure_keeping_the_best`** — the memory-bandwidth figure a processor-only node advertises may RISE over its run and never fall.
- **A peer's advertised version may bring the update check FORWARD and may do nothing else** — `update::PeerVersionWatch` on `state.events.peer_versions` wakes `UpdateChecker` through `state.events.update_nudge` (a `Notify`, not a third broadcast channel); it never decides the outcome.
- **`update::SelfUpdateBlocker` — "this node cannot update itself" carries WHY** — `UpdateChecker::self_update_blocker` returns the reason; `key()` is the stable string the dashboard translates across 21 locales, `advice()` the English one the daemon log and `swarmllm update` print.
- **`ModelRegistry::manifests_to_gossip`** — the single answer to "which manifests should this node re-broadcast?": ones it published **and ones it holds a shard of**.
- **`model::manifest::merge_known_shard_hashes`** — the rule that a shard hash may go from unknown to known but never back.
- **`types::slugify_model_name`** — the single derivation of a model id from a human display name.
- **`model::huggingface::is_trusted_publisher`** — canonical curator-allowlist check for an HF `repo_id`.
- **`SharedState::resolve_connected_peer_id_bytes`** — the resolver to use for any message that `network::manager::relay::is_relay_eligible` refuses, i.e. everything except `RemoteGenerateRequest` / `StreamingToken` / `CancelInference`. For those, "reachable" means "connected".
- **`ModelRegistry::describes_a_different_build`** — is this manifest the same FILE as ours, or another build wearing the same name? A model id comes from a display name (`slugify_model_name`), so every independent GGUF build collapses into one identity. Compares SHAPE, never hashes, and is gated on `has_origin_knowledge`.
- **`model::manifest::is_backup_artifact_id`** — canonical check for a model id that is a copied-folder backup (`<model>.FULLBACKUP`, `<model>.old`, `<model>~`, `… copy`) rather than a real model identity. Netted at `ModelRegistry::register_manifest`, the one point every adoption path funnels through.

### Scheduling, routing and failover → `docs/invariants/scheduling.md`

- **`ModelProcessPool::serves_on_cpu` is the whole-model delegation precondition** — "would this request run on our processor": no usable card, told to use the processor, a build without CUDA, or a card the model does not fit.
- **A node holding every layer that would run the model on its processor lets the priced search compete with its fast path** — `assemble_pipeline_for` answers `serves_on_cpu` ONCE (a lazy `OnceCell`) and threads it into `gather_candidates`, which PRICES the local candidate rather than excluding it.
- **A peer's capacity for a prompt is weights PLUS that prompt's KV cache** — `scheduler::max_hostable_layers` takes `prompt_kv_bytes_per_layer` — the same arithmetic the worker charges at admission, f16 mirror included.
- **`inference::scheduler::delegation_target`** — the single decision to hand a WHOLE model to a peer rather than run it on this node's CPU.
- **`inference::router::distributed_exec::failure_is_penalty_worthy`** — gates `penalty_serve_failure` on (a) the assignment actually having had a remote segment and (b) the error not being locally attributable.

### SharedState, live config and credits → `docs/invariants/state-and-config.md`

- **`SharedState::release_request_state`** — clears the maps a finished request leaves behind: `active_pipelines`, `active_traces`, `request_holder_blacklist`, `peer_vram_commitments` and `local_memory_refusals` — keyed by request id, sharing one lifetime. Deliberately does NOT touch `active_count` or `queue_notify`.
- **Credits are DORMANT — nothing may publish or act on a balance** — `MIN_BALANCE_FOR_INFERENCE = 0` and `calculate_tier` returns `DORMANT_TIER` whatever it is given, so no balance affects who is served or how fast, and the leaderboard neither ranks by credits nor publishes them.
- **`SharedState::cfg()`** — the live config, and the single answer to "what is this setting **now**".
- **`SharedState::record_peer_serve`** — the single answer to "this node did inference work for a peer", counting it AND billing for it.
- **`config::InferenceConfig::claims_shard`** — the single answer to "does this node claim shard N?", i.e. how `inference.shard_range` is read. **Never read `shard_range` directly.**
- **`SharedState::local_fast_path_for` is the single answer to "may this request take the local split fast path?"** — both API surfaces used to compose it themselves (`has_complete_split_model && !should_offer_work_to_the_swarm`), and the fast path skips the router — which is where `delegation_target` lives.
