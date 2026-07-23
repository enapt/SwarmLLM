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

When adding new fields to SharedState, put them in the appropriate sub-struct unless they're accessed by 10+ files across 3+ subsystem boundaries.

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
  dialling *from here*. Everything `filter_storable` does, plus: a node whose
  own reachable addresses contain no private address cannot route to anyone
  else's RFC1918 / CGNAT / IPv6-ULA, so those are dropped. Used by every dial
  path and by `GET /api/admin/diagnostics`.

**`local_addrs` empty means "not bound yet", NOT "public."** `listen_multiaddrs`
is empty until the swarm finishes binding; a node seconds into starting that
concluded it was a public server would discard every LAN peer it had, breaking
the home two-machine and pool cases the cache exists for. Unknown context keeps
everything.

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

`cargo check` with default features does NOT compile the `llama` cfg
path. Visibility-tightening or cross-file refactors that touch
`pipeline/dsd.rs` or any spec/llama-gated code in
`pipeline/speculative.rs` MUST verify with
`cargo check --features llama` before push. Pre-push hook only runs
default-features `cargo check`. R91 caught a regression introduced
in R90 that default-features had silently let through.
