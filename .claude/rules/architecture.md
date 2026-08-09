# Architecture Rules

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
- `state.models.contribution_auto` — R121. AtomicBool runtime mirror of `config.node.contribution_auto`. `state.config` is startup-frozen; this atomic exists so the Auto/Manual toggle on the Settings panel takes effect on the next prune tick without a daemon restart. Read by `model/auto_manage/prune.rs`; written by `PUT /api/admin/config` in `api/admin.rs`. New auto-manage paths that depend on the toggle MUST read this atomic, NOT `state.config.node.contribution_auto`.
- `state.models.foreign_wishlist` — R130. `DashMap<(NodeId, ModelId), (score_0_100, received_at_ms)>`; capped at `MAX_FOREIGN_WISHLIST_ENTRIES = 10_000` with oldest-first eviction, 2h freshness window enforced on read. Written by `apply_wishlist_announcement` on inbound `SwarmMessage::WishlistAnnouncement`; read by `compute_wishlist` for the 0..10 cross-pool demand boost.
- `state.models.quant_recommendations` — R133. `ArcSwap<QuantRecommendations>`; refreshed via `crate::model::auto_manage::quant::refresh_quant_recommendations(state)` on every auto-manage tick AND on every WS stats build. Read by `GET /api/admin/quant-recommendations` and the swarm-tab tips tile.
- `state.models.shard_download_backoff` — external report 2026-07-23. `DashMap<ShardId, ShardDownloadBackoff { fail_count, retry_after: Instant }>`. Exponential per-shard download cooldown (30→60→120→240→300s cap, via the pure `shard_backoff_delay_secs`). Recorded via `record_shard_download_failure` at every terminal *transient* download-failure site (HF `download_shard` error + GGUF-probe failure in `model/auto_manage/download.rs`, P2P give-up-with-no-HF-source in `network/manager/shard_transfer.rs`, and stall-reconciliation in `health/monitor.rs::cleanup_acquisition_progress`). Checked by `shard_in_backoff` in `scoring.rs::gather_candidates` (skips the shard while cooling down). Cleared via `clear_shard_download_backoff` on success (HF success arm + P2P completion in `requests.rs`). Distinct from `shard_p2p_failed`, which only *forces* the HF path without throttling re-selection — the two solve different problems and a new failure site should touch whichever it needs. Do NOT record backoff on the P2P→HF fallback branch: that path wants an *immediate* HF retry. Entries self-evict from `shard_in_backoff` once idle past `SHARD_BACKOFF_FORGET_SECS` (1h), so the map stays bounded without a dedicated sweep.
- `state.credits.foreign_pool_catalog` — R134. `DashMap<(PoolId, ModelId), received_at_ms>`; capped at 5000 with oldest-first eviction, 2h freshness window. Written by inbound `SwarmMessage::PoolModelAvailability` handler. Read by `GET /api/admin/foreign-pool-catalog` and by `pool::scope::cross_pool_extras` (R134.7) when `pool.allow_cross_pool_inference` AND `private_mode` are both on.
- `state.metrics.node_stats` — NOT `state.node_stats`
- `state.metrics.providers_config` — NOT `state.providers_config`
- `state.metrics.swarm_capacity` — R110. ArcSwap<SwarmCapacity>; refresh via `crate::daemon::state::refresh_swarm_capacity(state)`. Eagerly refreshed on peer connect (`network/manager/identify.rs`) and disconnect (`network/manager/connections.rs`) so the dashboard banner stays consistent with the peer-list panel under churn — the WS stats-cache 1.5s coalesce alone is too lazy.
- `state.metrics.hedge_tracker` — R136 Layer 2. `Arc<HedgeTracker>` with per-(model, segment, holder) EWMA latency + rate-budget counters. Always present. Observation via `state.record_hedge_observation(...)` from the forward-success path in `pipeline/distributed.rs` (post-hoc dry-run metrics). Real race-then-discard duplicate dispatch for speculative-verify hops ships in `pipeline/hedge_dispatch.rs::forward_verify_with_hedge` — single-segment only; multi-segment hedging remains deferred (`docs/FUTURE_WORK.md`). R142.6 added `last_observed_at_ms` to `HedgeStats` and `HedgeTracker::evict_stale` wired to the HealthMonitor tick to bound the (model × segment × holder) map.
- `state.metrics.prefetch_orchestrator` — R136 Layer 3. `PrefetchHandle` (Arc<PrefetchOrchestrator>) with per-session first-token histogram + idle-time learner + throttling. Observation via `observe_user_turn(session, first_token)` + `record_response_completion(session, now_ms)` at the router success site. R142.6 wired `evict_idle` to the HealthMonitor tick to bound the histories map. K-layer prefetch dispatch is the remaining integration; data-collection and orchestration are complete.
- `state.standalone_tokenizers` — R136 Layer 1/3 follow-on. `DashMap<ModelId, Arc<SplitTokenizer>>` on the ROOT SharedState (not a sub-struct — used by both `state.metrics`-derived L3 prefetch AND the `pipeline/ngram_only_spec.rs` L1 path, so cross-cutting). Lazy-loaded from `gguf_header.bin` via `state.standalone_tokenizer(&model_id)` accessor. Returns `None` when the header isn't on disk; caller falls through gracefully.
- `state.pending_activation_chunks` — R139 Tier 4K. `DashMap<Uuid, ChunkAssemblyState>` on the ROOT SharedState (cross-cuts the RR-decrypt path in `network/manager/tensors.rs` and the persistent-stream reader in `network/pipeline_stream.rs`). Receiver-side assembly for STREAM-chunked activation forwards. Entry-locked insert via `state.try_assemble_chunked_forward(forward, sender_peer_bytes)`. Periodic stale-entry sweep wired to the HealthMonitor tick via `state.sweep_stale_chunk_assemblies(ttl_secs)`. Chunk-meta is bound into AAD via `build_layer_forward_aad`, so reorder/truncation/cross-transfer-substitution fail Poly1305 before reaching the assembly.
- `state.listen_multiaddrs` — R140. `arc_swap::ArcSwap<Vec<String>>` on the ROOT SharedState (cross-cuts NetworkManager-writes and PoolManager-reads). Live snapshot of the swarm's reachable addresses, each terminated with `/p2p/<local_peer_id>`. Written by `NetworkManager::refresh_listen_multiaddrs()` (events.rs) on `NewListenAddr` / `ExpiredListenAddr` / `ListenerClosed` / `ExternalAddrConfirmed` / UPnP `NewExternalAddr` / `ExpiredExternalAddr`, plus once at startup after `listen_on()` (and after the `network.external_addresses` config override is added). **R143: the snapshot is the UNION of `swarm.listeners()` (bound sockets — private LAN on a NAT'd node) AND `swarm.external_addresses()` (UPnP-mapped / AutoNAT-confirmed / relay-circuit / manually-declared public addrs).** Without the union a NAT'd node's invite code silently shipped a LAN-only address. Built via the extracted, unit-tested `build_reachable_multiaddr_list(candidates, peer_id)` + `ensure_p2p_suffix` helpers; filtered through `addr_is_remotely_reachable` — keeps LAN + Tailscale CGN (100.64.0.0/10) + public, drops loopback / unspecified / link-local / IMDS. Read by `PoolManager::handle_generate_invite_code` when minting v2 `swarmpool://` codes; empty list → `SwarmError::ServiceUnavailable`. When the list has entries but NONE pass the stricter `pool::invite::any_internet_reachable` (public IP / DNS / relay-circuit — excludes LAN + CGN), invite generation still succeeds but emits a `pool`/`invite_lan_only` warning ActivityEvent so the user isn't handed a LAN-only code that dies over the internet.
- `state.dashboard_trust_lan` — `AtomicBool` on the ROOT SharedState. Runtime mirror of `config.api.dashboard_trust_lan`, the opt-in that lets a browser on a private/LAN address be handed the dashboard's API key. `state.config` is startup-frozen and this is a setting the user flips *because* their dashboard is unreachable, so a restart requirement would defeat it — same reasoning as R121's `contribution_auto`. Written by `PUT /api/admin/config` (`api/admin.rs`), read by `api::dashboard_trust::classify`. New code gating on LAN dashboard trust MUST read this atomic, NOT the config field. The sibling `config.api.dashboard_trust_overlay` is read straight from config: it is not runtime-toggleable because the tailnet case works by default and turning it *off* is a deliberate hardening step, not a recovery action.
- `state.observed_inbound_connection` — `AtomicBool` on the ROOT SharedState.
  Set once by `handle_connection_established` for the first non-loopback
  connection where we are the LISTENER. **The only direct evidence that inbound
  reaches this node**: outbound succeeds from behind almost anything, so a node
  with peers, a real LAN address and clean logs can still be silently dropping
  every inbound packet and look perfectly healthy from the inside. Read by the
  WSL2 firewall check in `health::monitor`, which previously warned every
  mirrored-mode node on every start whether or not anything was blocked. Any
  future "are we reachable" question should use this rather than inferring from
  peer count — having peers proves only that WE dialled successfully.
- **`SharedState::model_is_in_use` is the answer to "may I delete this model's
  files?"** — NOT `active_pipelines` on its own. That is the COORDINATOR's map of
  DISTRIBUTED assignments (gotcha #194) and holds nothing for peer-served work or
  for a reply the local model is producing through the split fast path, which
  bypasses the router entirely. Deleting during a local reply therefore returned
  `200 files_removed: 8` and killed the worker mid-stream (measured 2026-08-05) —
  the exact outcome the guard below exists to prevent, on the most common
  single-node path. The helper asks `active_traces` first, because every
  in-flight request registers one for progress reporting whichever path serves
  it; `serving_models` and `active_pipelines` remain as belt-and-braces. Both
  `delete_model` and `delete_shard` go through it.
- **`api::dashboard_trust::classify` is the single answer to "may this request be handed the API key automatically?"** Do NOT re-derive it with `addr.ip().is_loopback()`. That predicate means "the last TCP hop began inside this daemon's network namespace", which is simultaneously broader than intended (a same-host reverse proxy such as `tailscale serve` satisfies it on behalf of a fully remote client) and narrower (a container publish, a NAT, or a Tailscale subnet router never satisfies it — not even from the host's own `localhost` — because subnet routers SNAT by default). Same-origin checks belong on `Origin` vs the request's own `Host` (`websocket.rs::ws_origin_allowed`), never on a hardcoded loopback allowlist: that mistake independently cost every non-loopback dashboard its live WebSocket updates. See gotcha #195.
- `state.relay_proven_features` — `DashMap<NodeId, RelayProvenFeatures { features: u64, proven_at: Instant }>` on the ROOT SharedState (`daemon/state/relay.rs`). Records relay features a peer has *demonstrably* used by relaying a message addressed to us: `handle_relayed_tensor` records `features::TENSOR_RELAY`, `handle_relayed_envelope` records `features::RELAY` (via `record_relay_proven_features`, which ORs bits + refreshes `proven_at`). The relay send path's feature gates (`target_supports_{relay,tensor_relay}` in `network/manager/relay.rs`) consult `relay_feature_proven(peer, bit)` FIRST, before the gossiped `NodeCapability.features`. **This is the cold-start return-path fix**: a serving node reaches a coordinator known only via `ensure_relayed_origin_known` (whose `peer_registry` entry has `capability: None`, because the capability-gossip handler at `daemon/dispatch/mod.rs` is update-only and can't populate a not-yet-existing entry). Without the proof, the return relay of a computed `LayerResult` was refused until a capability-gossip round landed (≤30s), dropping the first result. Freshness = `RELAY_ROUTE_TTL_SECS` (re-proven on every inbound relayed message, so an active session never goes stale); swept alongside `relay_routes` in `sweep_stale_relay_state`. New relay send paths that gate on a peer's relay capability MUST consult this proof, not just the gossiped capability.

When adding new fields to SharedState, put them in the appropriate sub-struct unless they're accessed by 10+ files across 3+ subsystem boundaries.

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
- **`inference::worker_ipc::worker_error_is_fatal`** (R146) — the single
  source of truth for "did this worker error destroy the worker's device
  state, or just this request?". Used by the worker to stamp
  `WorkerMsg::Error.fatal` AND by the daemon's
  `ModelProcessPool::classify_worker_error` to re-derive the verdict from
  the message text (the field is `#[serde(default)]`, so a worker binary
  older than the field always reports `false`). A `true` verdict evicts
  the worker from the pool, which drops the last `Arc<WorkerHandle>` and
  lets `Drop` kill the child — the only thing that actually returns VRAM
  to the OS. New fatal-error classes go in the pattern list, not into a
  caller-side special case; divergence between the two sides means a
  stranded worker holding its whole allocation for the daemon's lifetime.
  Lean inclusive: a needless respawn costs one model reload.
- **`daemon::shard_loader::force_cpu_for`** (R146) — the single mapping
  from `inference.gpu_layers` (`-1` auto / `0` CPU only / `>0` GPU) to the
  loader's `force_cpu` flag. Every device-placement decision goes through
  it: `ModelProcessPool::effective_gpu_layers` → `--gpu-layers` spawn arg
  → `model_worker::set_worker_force_cpu` → `ShardLoadParams.force_cpu` /
  `SplitModel::load_from_gguf(force_cpu)`. Do NOT re-derive placement by
  calling `Device::cuda_if_available` directly in a new load path — that
  is exactly how `gpu_layers` came to be silently ignored for every
  sharded model. Partial offload is not expressible (see
  `docs/FUTURE_WORK.md`); a fractional value logs a warning rather than
  being quietly rounded.
- **`daemon::gpu_support::MIN_COMPUTE_CAP` + `local_gpu_is_supported`**
  (2026-08-07) — the single answer to "can this card run OUR kernels?".
  `MIN_COMPUTE_CAP` is a property of the BUILD and MUST equal
  `CUDA_COMPUTE_CAP` in `release.yml` / `cache-warm.yml` / `ci.yml`;
  `compute_cap_matches_release_workflow` fails the build if they drift, and
  `flash_attn_and_the_compute_cap_floor_agree` ties the floor to the feature
  in BOTH directions (8.0 is only worth paying for because of flash-attn).
  Do NOT ask `Device::cuda_if_available` whether the GPU is usable — it
  SUCCEEDS on a pre-Ampere card and only module load fails, per request, so
  the node starts cleanly, logs "GPU detected", advertises itself to the
  swarm as a GPU node, and then fails everything with
  `CUDA_ERROR_NO_BINARY_FOR_GPU`. An unreadable capability is **unknown,
  never unsupported**: sending a working card to the CPU because nvidia-smi
  misbehaved is a worse bug than the one this prevents. Enforcement is at
  `ModelProcessPool::effective_gpu_layers` (the same choke point as
  `gpu_layers` and OOM CPU-pinning), with
  `worker_ipc::permanent_gpu_failure` as the backstop for when the probe
  returned unknown.
- **`inference::split::kv_budget`** (2026-08-08) — the KV memory budget and the
  admission check against it. The loader records `kv_headroom_bytes` on the
  model; `forward_inner_impl` checks `quantum_exceeds_headroom` before a forward
  claims another growth quantum, and refuses with `ServiceUnavailable` (503,
  so a coordinator re-routes to a peer). **Do NOT re-introduce a load-time
  context clamp** — one existed, it shrank every user's context so a single
  full-length conversation would fit, and it did not bound concurrency at all.
  Three invariants a new caller must preserve: the check runs ONLY when
  `positions_claimed` is non-zero (otherwise it walks the whole store per
  generated token for an answer that is almost always "no"); it charges the
  POSITIONS claimed, not one quantum, because a prefill jumps many quanta in a
  single forward and charging one under-counted the largest claim a request
  ever makes by 10x; and `kv_budget_bytes: None` means UNKNOWN, never zero — every CPU node
  and any GPU node whose free VRAM could not be read records `None`, and reading
  that as a zero budget refuses everything.
- **`config::InferenceConfig::claims_shard`** (2026-08-09) — the single answer to
  "does this node claim shard N?", i.e. how `inference.shard_range` is read.
  **Never read `shard_range` directly.** Five places asked the question with
  their own copy of the comparison and THREE never asked at all: the startup
  disk scan, the periodic rescan, and one manifest path. The rescan is the one
  that mattered — startup applied the range correctly and then, minutes later,
  the rescan found the remaining files still on disk and re-registered them, so
  a node configured for shards 0-1 of a four-shard model came up serving
  `layers=[0..12)` and was serving `[0..28)` on its own five minutes later.
  The feature then fails twice over: the node stops being half of a split model
  AND loads the whole thing into memory, which is the saving being asked for.
  Silent — no error, no warning, and the config key parses.
  **A new shard-registration path MUST call this**; that is the whole reason it
  is a method on the config that owns the field rather than a free function
  someone can forget. Verified on two machines: the restriction held for 10
  minutes against the 4m47s it previously took to lose it, and a genuine
  two-segment pipeline then answered correctly across both.

- **`inference::cpu_pools::in_phase_pool`** (2026-08-07) — binds a forward pass
  to the CPU thread pool that suits its phase, at ONE choke point:
  `SplitModel::forward_inner_impl` and `forward_batch`. Every entry point —
  LoRA, speculative verify, pre-embedded segment, SWIFT skip-mask, batched
  prefill — funnels through those, so a new one inherits it and cannot forget.
  Do NOT call `install` at a call site instead.
  **Reading a prompt and writing a reply want different thread counts**: decode
  is bandwidth-bound (69% of roofline), so past the point that saturates memory
  the extra threads only contend. **The cap is PHYSICAL CORES and must not
  become a fraction of them.** A fraction was tried — `max(4, physical/2)`,
  measured correctly on an 8-core Ryzen — and a second machine (6-core Intel
  i5-10500T) showed decode climbing monotonically to all six, where that rule
  would have cost 23%. Peak threads is bandwidth divided by per-core draw, which
  core count cannot predict; physical-vs-SMT is the only part both machines and
  the mechanism agree on. Prefill keeps the global pool untouched; decode is
  capped only ever downward, and every contribution level is already at or below
  physical, so the common path builds no pool and pays nothing.
  `SWARMLLM_DECODE_THREADS` overrides, and `=0` restores the single-pool
  behaviour for A/B measurement inside one binary — the same discipline as
  `SWARMLLM_FORCE_STANDARD_ATTN`.
- **`inference::layers::new_kv_cache`** (2026-08-07) — the only way to construct
  a KV cache. **Never call `KvCache::new(2, max_seq_len)`**, which is what every
  site did and which reads as obviously correct — the parameter is even called
  `max_seq_len`. candle's `Cache::new(dim, n)` sets `grow_by` AND `max_seq_len`
  to `n`, and `append` allocates the full buffer on the FIRST append, so passing
  a model's context length reserved the whole context window from token one: a
  100-token chat held 940 MB at 3% utilisation on llama-3.2-3b. The helper
  passes `KV_CACHE_GROWTH_TOKENS` instead and lets `append` grow on demand; the
  conversation's real ceiling is enforced separately by the
  `total_seq > max_seq_len` guard in `forward_inner_impl`, so this value cannot
  shorten a conversation. `kv_cache_reservation(positions)` is the sibling for a
  cache that must hold N tokens immediately — prefix-cache hydration — and it
  deliberately ignores the snapshot's recorded `max_seq_len`, because snapshots
  cross the network and a peer on an older build recorded a whole-context value.
  **Reason about KV memory from `KvCacheStore::occupancy()`, never from process
  RSS**: the reservation is lazily-faulted zero pages, so a 4-8x change in
  reserved bytes moved RSS ~5% and in both directions. Two conclusions drawn
  from RSS about this cache were wrong before the counter existed.
- **`inference::attn_softmax::scaled_masked_softmax`** (2026-08-07) — the single
  expression of attention's tail: scale, optional Gemma-2 logit soft-cap,
  additive mask, softmax. Do NOT re-express those as separate candle ops in a
  new attention path. Each one materialises a whole
  `[batch, heads, q_len, kv_len]` score tensor — 11 MB at llama-3.2-3b prefill
  shapes — so writing them out cost 34.6 ms where one fused pass costs 11.4,
  and attention fell from 22.4% of a prompt chunk to 9.5% when they were folded
  together. The fused CPU kernel declines anything it cannot index (non-CPU,
  non-f32, strided, or a mask that is not a shared `[q_len, kv_len]` block) and
  falls through to `composed`, which is the original expression and the
  reference its tests compare against — so a new caller is always correct,
  just possibly not fast.
  **The mask is ADDITIVE f32 everywhere: `0.0` visible, `-inf` masked.** There
  used to be two representations — a `u8` predicate for the standard path and a
  float copy the flash arm rebuilt on every call — and a new attention backend
  had to know which it was being handed. `SplitModel::causal_mask` is the only
  producer. It also returns a CONTIGUOUS tensor deliberately: a `narrow()` view
  costs 2.1x in `broadcast_add` and is refused by the fused kernel outright, so
  any path that slices a mask (the query-blocking loop in `standard_attention`
  does) must `.contiguous()` it before passing it on.
  Changing the scale means changing `scale_from_head_dim`, which reproduces
  candle's `tensor / f64` (an `affine(1/rhs)`, i.e. already a multiply) exactly.
  `scale_matches_candle_division` pins that against candle itself rather than
  against the helper — an equivalence test where both sides call the same
  helper passes happily with the scale inverted.
- **`inference::layers::cuda_decode_prefers_standard`** (2026-08-08) — MHA
  decode takes standard, GQA decode takes flash **at every context length**;
  prefill always flash. Same rule as the CPU path, for the same reason:
  `standard_attention` rebuilds the `repeat_kv` expansion every token, free when
  `n_head == n_kv_head` and growing with context otherwise.
  **There is no crossover, and re-introducing one needs a FORWARD measurement,
  not a per-call one.** A 1024-token threshold shipped on 2026-08-07 from timing
  the attention call in isolation; measured end to end the next day it was wrong
  at every length (1.13x at kv~272, 1.42x at ~528, 1.61x at ~912 in flash's
  favour). Isolated, `repeat_kv`'s allocation and bandwidth cost is amortised
  against warm buffers and no competing traffic. **Third occurrence of gotcha
  #255.** Controls that make the change attributable: at 2048 KV both arms were
  identical (both already flash) and MHA identical to the decimal.
- **`inference::layers::cuda_decode_prefers_standard` (superseded note, 2026-08-07)** — the
  measured CUDA attention routing rule, extracted so it is testable without
  a GPU. **The right kernel is opposite for prefill and decode, and it turns
  on GQA** — the same lesson as the CPU crossover above it (gotcha #255) on
  a different device. Flash unconditionally costs up to **25x per attention
  call** on MHA decode, because candle-flash-attn ships no split-KV kernel
  and one query row cannot fill the card; GQA reverses above ~1k context
  because `standard_attention` rebuilds the `repeat_kv` expansion every
  token. Changing the constant means re-running
  `flash_vs_standard_attention_on_cuda` — the measured table lives in the
  dispatch's comment and in `docs/FUTURE_WORK.md`, and the benchmark
  asserts the dispatch never picks a kernel materially slower than
  always-standard.
- **`inference::router::distributed_exec::failure_is_penalty_worthy`**
  (R146) — gates `penalty_serve_failure` on (a) the assignment actually
  having had a remote segment and (b) the error not being locally
  attributable. Any new automatic credit or reputation penalty MUST route
  through an equivalent attribution check. `ServiceUnavailable` means
  "THIS server can't serve" and `Internal` means our own bug — neither can
  ever justify charging a peer.
- **`model::huggingface::is_trusted_publisher`** (R141) — canonical
  curator-allowlist check for an HF `repo_id`. Splits on the first `/`
  and case-insensitively matches the prefix against
  `TRUSTED_HF_PUBLISHERS` (in `huggingface/watcher.rs`). Used by BOTH
  the watcher's trust-promotion path (`promote_trust_for_trending` →
  `min_downloads_for_repo` consumes the tiered 10k/100k threshold) AND
  the wishlist scorer (`compute_wishlist` Candidate-row pass — flat +10
  score bonus + `wishlist.why.trusted_publisher` why-tag). Any new
  surface that needs to gate on "is this from a known-good curator"
  MUST go through this helper rather than re-creating the allowlist —
  the allowlist is a trust delegation and divergence creates a security
  / consistency gap. Adding a curator: append to
  `TRUSTED_HF_PUBLISHERS` (one place); both consumers pick it up
  automatically. Removing a curator (compromise, abandoned account,
  loss of trust) requires the same one-place edit; do not soft-disable
  via wrappers because the trust delta is a real security event worth
  surfacing in the diff.
- **`inference::split::GgufTensorMeta::tied_output_location`** — the single
  definition of "is this model weight-tied", i.e. does it reuse
  `token_embd.weight` as the LM head instead of shipping an `output.weight`.
  Consumed by BOTH sidecar writers (`daemon::manifest::extract_tied_output_weight`,
  `huggingface::probe::download_tied_output_weight`) AND the reader
  (`inference::split::resolve_tied_output` → `ShardReader`). Producer and
  consumer MUST agree on which tensor the sidecar holds; a new surface that
  needs the predicate goes through this method rather than re-deriving
  `contains_key("output.weight")`. The sidecar filename is
  `inference::split::TIED_OUTPUT_FILENAME`, never a literal.
  **Why this exists**: a node serving the LAST pipeline segment needs the output
  head, but on a weight-tied model that tensor physically lives in shard 0 —
  which that node frequently does not hold. The sidecar carries the raw bytes;
  `ShardReader::new` maps them over the tensor's gguf byte range so
  `ct.tensor(&mut reader, "token_embd.weight", …)` resolves unchanged. It maps
  the sidecar ONLY when no local shard already covers that offset, since a
  duplicate `gguf_offset` would make `find_shard`'s binary search ambiguous.
  `tied_output` is a REQUIRED parameter on `ShardReader::new` with no
  convenience wrapper — for three releases the sidecar had three writers and
  zero readers, and every weight-tied model was unservable on any node lacking
  shard 0 (gotcha #178).
- **`SharedState::resolve_connected_peer_id_bytes`** — the resolver to use for
  any message that `network::manager::relay::is_relay_eligible` refuses, i.e.
  everything except `RemoteGenerateRequest` / `StreamingToken` /
  `CancelInference`. For those direct-only messages "reachable" means
  "connected", so the ungated `resolve_peer_id_bytes` hands back a target the
  send path can only drop. **Gossipsub reachability is NOT request_response
  reachability**: a peer relayed to us through the mesh is frequently
  undialable, and `peer_id_map` is deliberately persistent across disconnects
  (its only eviction is gated behind an 8,000-entry soft cap that never trips on
  a small swarm). Replying to gossip via `peer_id_map` alone produced an
  unbounded 30s loop of undeliverable sends — one departed peer, 45% of a
  night's log volume (gotcha #220). `connected_node_ids` is the liveness oracle;
  `peer_registry` is explicitly NOT, being preserved across disconnects for
  reconnect. **When adding such a gate, re-check any `else`/fallback arm below
  it** — the health-pong site had a `Broadcast` fallback that a naive `None`
  would have turned into mesh-wide traffic every 30s, worse than the bug.
- **`model::manifest::is_backup_artifact_id`** — canonical check for a
  model id that is a copied-folder backup (`<model>.FULLBACKUP`,
  `<model>.old`, `<model>~`, `… copy`) rather than a real model identity.
  A model's identity must come from the model, not from whatever a
  directory was called. `ModelRegistry::register_manifest` nets this at
  the single point EVERY adoption path funnels through (gossip ingress,
  DB reload on startup, local disk scan, acquisition) — so a backup name
  can neither be stored, persisted, nor re-gossiped regardless of how it
  arrived. Belt-and-suspenders explicit guards also sit at the network
  boundary (`daemon/dispatch` ModelManifest handler emits a
  `security`/`manifest_rejected` activity event + skips the auto-manage
  wake; ShardAnnounce + RegionShardSummary skip backup ids so holder /
  region counts stay clean without a manifest) and the local disk scan
  (`daemon/startup.rs`, so an artifact is never persisted). New surfaces
  that accept a model id from disk or the network MUST reject via this
  helper rather than re-deriving the keyword list. The keyword list is
  matched against the LAST dotted segment only, so a legit id carrying
  dots from its source filename (`tinyllama-1.1b-chat-v1.0.q4-k-m`) is
  never caught. The v0.3.10 disk-scan-only guard was insufficient because
  a peer on an older build re-gossips the name straight back in.

## Peer Cache: storable vs dialable

`network/peer_cache.rs` answers two different questions and they must not be
conflated:

- **`filter_storable(addrs, local_peer_id)`** — what is worth *keeping*. Drops
  only what is junk under any circumstances: not remotely reachable
  (`addr_is_remotely_reachable`), or routing through our own peer id in ANY
  `/p2p/` hop (the relay position of a `/p2p-circuit`, not just the target).
  **Keeps private addresses regardless of where this node currently is** — a
  laptop saving its cache on a hotspot must not permanently lose the LAN peers
  it had at home. Used by `save_peer_cache`.
- **`filter_dialable(addrs, local_peer_id, local_addrs)`** — what is worth
  dialling *from here*. Everything `filter_storable` does, plus a peer's
  RFC1918 / CGNAT / IPv6-ULA addresses are dropped when EITHER: (a) our own
  reachable addresses contain no private address (`local_is_public_only` — we
  can't route to anyone else's private network), OR (b) **the peer itself
  advertises a publicly-reachable address** (`peer_has_public` — then its
  private addresses are its own LAN/Docker bridge and we reach it publicly
  instead). Used by every dial path and by `GET /api/admin/diagnostics`.

  The `peer_has_public` clause is the **Docker fix** (2026-07-23): a Docker
  node advertises its container bridge `172.17.0.1` alongside its real public
  IP, and `172.17.0.1` is not globally unique — it is the Docker gateway of
  *whichever* host dials it, so a dial loops back to the dialer's own node
  rather than failing cleanly (confirmed live). A peer with a public address is
  reached there; its private noise is dropped even when we are on a LAN too.
  A peer with ONLY private addresses (no public) is still kept, so the home
  two-machine / pool case is untouched — those peers are additionally found via
  mDNS regardless.

**`local_addrs` empty means "not bound yet", NOT "public."** `listen_multiaddrs`
is empty until the swarm finishes binding; a node seconds into starting that
concluded it was a public server would discard every LAN peer it had, breaking
the home two-machine and pool cases the cache exists for. So the
`local_is_public_only` clause treats empty as unknown → keep. The
`peer_has_public` clause is independent of local context: it keys on the peer's
own addresses, so it correctly drops a public-capable peer's Docker/LAN noise
even at startup.

Nothing in `src/pool/` reads this cache — pools route through `pool_state` /
`allowed_node_set` — and mDNS discovers LAN peers independently, so a LAN pool
has a second route back regardless.

Retraction of a peer's *shard* claims is a different mechanism entirely; see
`ShardAnnounce.complete_for_models` below.

## ModelRegistry Holder Counts

`ModelRegistry::shard_holders` caches at most `MAX_HOLDERS_PER_SHARD = 50`
holders per shard (LRU-evicted, local node never evicted). This is the
**routing oracle** — pipeline scheduler, region eviction, busy-holder
check etc. all read this map.

`ModelRegistry::global_holder_count` holds the **uncapped swarm-wide
count** from the most recent DHT `GetProviders` response, written by
`network/manager/dht.rs::handle_dht_providers_found` with the raw
`providers.len()` (PeerId count, not the resolved NodeId count — some
PeerIds may fail to resolve but they're still distinct providers in the
DHT's view). This is the **prune-score oracle** — `model/auto_manage/
prune.rs` uses `max(cached_holder_count, global_holder_count)` for the
`redundancy_ratio` numerator and the severe-saturation bonus check.

Don't:
- Read `global_holder_count` for routing decisions — DHT staleness is
  fine for an O(hundreds of seconds) prune cadence but unacceptable for
  scheduling.
- Read `shard_holders().len()` alone for `redundancy_ratio` — at 1000-
  node scale the cache pegs at 50 and the prune score saturates.
- Forget to clear `global_holder_count` when a model is removed —
  `remove_all_model_shards` retains over both maps; new code paths that
  evict a model must do the same or stale figures will inflate future
  ShardId-reuse scores.

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

## Config defaults must stay live

The daemon must write **only values that differ from the compiled default**.
`config::to_minimal_toml` is the one serializer for the config file; do not call
`toml::to_string_pretty(&config)` directly.

**Why.** A `#[serde(default)]` fills a key that is *missing*. Once a key is
written to disk it wins forever, so any later change to that default can never
reach that install — and the file looks like a deliberate user choice, which is
indistinguishable from one. `PUT /api/admin/config` is called by the setup
wizard on "Start SwarmLLM", so in practice every field landed on disk on first
run. This produced three separate user-visible faults before it was fixed:

- `bootstrap_peers = []` stranded every node set up before 2026-07-21 with no
  bootstrap peer, no DHT route and no log line (gotcha #198).
- A default-on dashboard-trust flag shipped *off* to exactly the fresh installs
  it was written for (gotcha #196).
- `check_interval_hours = 6` kept nodes on a six-hour update check after the
  default became hourly — found live on 2026-07-29 while watching a node fail
  to notice a release.

Rules that follow:

1. **Never serialize the whole `Config` to disk.** Use `to_minimal_toml`.
2. **A section's `impl Default` MUST agree with its fields' `#[serde(default)]`.**
   These are different code paths: a *missing* section uses `impl Default`, a
   *present but empty* section uses each field's serde default. They disagreed
   for `updates.mode` (`Some(Notify)` vs `None`), which made the effective
   update mode depend on whether the `[updates]` header happened to exist.
   Pinned by `empty_section_matches_missing_section`, which checks every
   section, so a new one inherits the coverage.
3. **Every field needs a serde default**, or a pruned file will not reload.
   Pinned by `empty_toml_parses_to_full_default`.
4. **Changing a default does not reach existing installs.** If the old value is
   already on disk it stays. When a default changes in a way that matters, add
   an entry to `migrate_superseded_defaults` — and only when the old value was
   the daemon's, never something a user could plausibly have chosen, because
   silently overriding a deliberate setting is worse than a stale default.
5. **Unknown keys warn, they do not fail.** `deny_unknown_fields` would refuse
   to start on a config mentioning a later release's key. `warn_unknown_keys_in`
   names the key and continues.
