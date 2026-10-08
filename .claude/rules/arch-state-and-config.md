---
paths:
  - "src/daemon/state/**"
  - "src/config/**"
  - "src/credit/**"
  - "src/storage/**"
  - "src/health/**"
  - "src/lib.rs"
  - "src/types.rs"
  - "src/daemon/mod.rs"
  - "src/daemon/startup.rs"
  - "src/daemon/supervisor.rs"
  - "src/daemon/background.rs"
  - "src/daemon/helpers.rs"
  - "src/daemon/manifest.rs"
  - "src/main.rs"
  - "src/update_restart.rs"
  - "src/bin/launcher.rs"
---

# SharedState, live config and credits

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## SharedState Sub-Structs

The 4 sub-structs and the `cfg()` / `classify` accessors are in `architecture.md`
(always loaded). Per-field rules:

- `state.events.activity_tx` — NOT `state.activity_tx`
- `state.events.dashboard_tx` — NOT `state.dashboard_tx`
- `state.credits.credit_balance` — NOT `state.credit_balance`
- `state.credits.pool_state` — NOT `state.pool_state`
- `state.models.acquisition_progress` — NOT `state.acquisition_progress`
- `state.models.wishlist` — R111. ArcSwap<Wishlist>; refresh via `crate::model::auto_manage::refresh_wishlist(state)`.
- `state.models.hf_trending_cache` — R112. ArcSwap<HfTrendingSnapshot>; written by `HfWatcher` only.
- `state.models.foreign_wishlist` — R130. `DashMap<(NodeId, ModelId), (score, received_at_ms)>`, capped (`MAX_FOREIGN_WISHLIST_ENTRIES`), 2h freshness; written only by `apply_wishlist_announcement`, read by `compute_wishlist`.
- `state.models.quant_recommendations` — R133. `ArcSwap<QuantRecommendations>`; refresh via `crate::model::auto_manage::quant::refresh_quant_recommendations(state)`.
- `state.models.shard_download_claims` — shards being WRITTEN now, one RAII `ShardDownloadClaim` from `claim_shard_download`; read `is_shard_in_progress`. Never rest writer exclusion on `acquisition_progress`. → `docs/invariants/network.md`
- `state.models.shard_download_backoff` — per-shard exponential download cooldown; record via `record_shard_download_failure`, check `shard_in_backoff`, clear via `clear_shard_download_backoff`. Not on the P2P→HF fallback branch.
- `state.models.manifest_heard` — manifest VERSIONS gossiped swarm-wide, per version not per model; written only by `note_manifest_heard` (takes the `MessageTransport`, ignores `Direct`), read by `manifest_heard_within`. → `docs/invariants/network.md` § "The repetition, not the size"
- `state.models.hf_sources` / `origin_claims` / `canonical_builds` / `origin_refusals` / `canonical_holding` — one upload per model id (#151). ⛔ `hf_sources` is written ONLY by `SharedState::write_hf_source`; guards `a_models_source_is_written_only_through_the_canonical_choice`, `a_models_header_is_fetched_only_from_the_upload_its_parts_are`; downloads ask `canonical_allows_acquisition`. → `docs/invariants/network.md` § "One upload per model id"
- `state.metrics.peer_outliers` — `PeerOutliers`; fed ONLY by `record_peer_delivery`, reset by `note_peer_completed_request`. → `docs/invariants/scheduling.md` § "A peer that fails a model on every request"
- `state.models.auto_model` — `Mutex<Option<ModelId>>`; read/written ONLY by `api::openai::resolver::auto_model_for`. Never make `auto` read `loaded_model_info`.
- `state.models.removed_by_user` — `DashMap<ShardId, bool>` of user-deleted shards (#360); only via the helpers in `daemon/state/removed_shards.rs` (`mark_shard_removed_by_user`, `shard_removed_by_user`, `clear_shard_removed_by_user`, `clear_removed_by_user_for_model`); never write the map or tree directly.
- `state.models.shards_needing_repair` — see `docs/invariants/state-and-config.md`
- `state.models.shards_pending_verification` — see `docs/invariants/state-and-config.md`
- `ModelRegistry::bytes_disputed` (was `state.models.disputed_shards`, moved 2026-10-03) — kept-but-disagreeing shards WITH the hash their bytes have, which is what the node announces for them (`announced_build_tag`); write only `SharedState::note_shard_disputed(shard, &verdict)` / `clear_shard_dispute`, read only `SharedState::disputed_shards_now` / `shard_is_disputed`; guard `every_path_that_keeps_disagreeing_bytes_records_the_dispute`. NOT `shards_needing_repair`. → `docs/invariants/network.md` § "One upload per model id"
- `state.models.canonical_holding` + `ModelRegistry::withheld_models` — written together ONLY by `SharedState::note_canonical_holding`; a copy of another upload is withheld from the swarm. → `docs/invariants/network.md` § "One upload per model id"
- `ModelRegistry::origin_verified` — see `docs/invariants/state-and-config.md`
- `network::manager::tensors::AckRttEstimator` — see `docs/invariants/state-and-config.md`
- A KV-budget refusal MUST release what the request already took — see `docs/invariants/state-and-config.md`
- `SharedState::can_fetch_shard_from_origin` — see `docs/invariants/state-and-config.md`
- `state.credits.foreign_pool_catalog` — R134. `DashMap<(PoolId, ModelId), received_at_ms>`, cap 5000, 2h; written by the `PoolModelAvailability` handler, read via `pool::scope::cross_pool_extras`.
- `state.local_memory_refusals` — `DashMap<Uuid, Option<u32>>` on the root (the fewest layers a LOAD refused to add, taken from `ModelProcessPool::take_layers_refused`); write ONLY `note_local_memory_refusal`, read only `local_memory_refused_for_request` / `local_layers_refused_for_request`; released by `release_request_state`, which also drops a refusal the router never took.
- `state.models.heal_verdicts` / `heal_pass_times` — the copy repair's account (#217): write ONLY `canonical::note_verdict` and `canonical::pass`; read only by diagnostics (`-- copy repair --`). Every quiet return in `canonical::settle` records why.
- `state.request_peers` — `DashMap<Uuid, HashSet<NodeId>>` on the root: the peers each request this node led ran segments on, every attempt; write ONLY `note_request_peers` (`PipelineExecutor::execute`), take ONLY `take_request_peers` (`router::release_request_on_peers`, once, at the request's end); released by `release_request_state` (#238).
- `state.planned_past_offered_memory` — `DashMap<Uuid, OfferedMemory>` on the root; write ONLY `note_planned_past_offered_memory` (the scheduler, real requests only), read only `planned_past_offered_memory` (the router, at the refusal); released by `release_request_state`.
- `state.encrypted_pipeline_models` — **never read directly**: use `SharedState::encrypted_pipeline_for` (`privacy_explicitly_enabled_for` only for a deliberate user choice); guard `prompt_privacy_is_never_re_derived_from_the_per_model_map`.
- `state.region_demand` / `state.local_region_demand` — merged vs own-region demand. ⚠ Only the LOCAL map may be GOSSIPED; guard `the_demand_we_gossip_is_the_demand_we_measured`. → `docs/invariants/network.md` § "Gossip volume"
- `state.metrics.gossip` also yields per-topic MESSAGE counts (`GossipTopicTotals`: `published`, `sent`, `recv`, `recv_unfiltered`) beside the bytes; `state.metrics.gossip_by_kind` (`GossipKindMeter`) splits a topic by message variant. ⚠ `sent` counts attempts, so never assert `sum(parts) <= whole`. Gotcha #674.
- `state.metrics.lifetime_served` — serving totals since the node FIRST started (#226): the redb record as this run found it; read ONLY `SharedState::served_lifetime`, written ONLY `persist_served_lifetime` (monitor tick + shutdown), as one absolute figure. Not a balance — credits are dormant.
- `state.metrics.bandwidth` — `Arc<BandwidthMeter>`; `totals()` answers `None`, never `0`; `refresh()` only from the health-monitor tick, everything else reads `current()`.
- `state.metrics.gossip` — `Arc<GossipMeter>`; per-topic GossipSub bytes, answers `None` never `0`, warns about nothing. ⚠ `libp2p-gossipsub` is a DIRECT dep only for its `metrics` feature; guard `the_gossip_counters_come_from_the_same_crate_libp2p_uses`.
- `state.metrics.inference` — `Arc<InferenceTraffic>`; inference bytes counted in the CODEC (`network/protocol/mod.rs`) and nowhere else — never at the send sites.
- `state.metrics.last_dispatch_at_ms` + `last_dispatch_kind` — dispatcher liveness, written by `metrics.note_dispatch` before the `match`, read by `HealthMonitor::report_dispatcher_stall`; stall = message WAITING (`dispatch_stalled_for`); withdraws inference via `SharedState::inference_outage`. Guards `the_dispatcher_liveness_marker_is_written_before_the_match_and_watched_elsewhere`, `both_inference_outages_withdraw_through_one_predicate`.
- `state.metrics.node_stats` — NOT `state.node_stats`
- `state.metrics.providers_config` — NOT `state.providers_config`
- `state.metrics.swarm_capacity` — R110. ArcSwap<SwarmCapacity>; refresh via `crate::daemon::state::refresh_swarm_capacity(state)`.
- `state.metrics.segment_latency` — `Arc<SegmentLatencyTracker>`; write `SharedState::record_segment_latency`, read `peer_performance_rows`. Do not re-add hedging of a stateful step (#94).
- `state.metrics.prefetch_orchestrator` — R136. `PrefetchHandle`; `observe_user_turn` / `record_response_completion`, `evict_idle` on the HealthMonitor tick.
- `state.standalone_tokenizers` — `DashMap<ModelId, Arc<SplitTokenizer>>` on the root; read via `state.standalone_tokenizer(&model_id)` (`None` without `gguf_header.bin`).
- `state.pending_activation_chunks` — `DashMap<Uuid, ChunkAssemblyState>` on the root; insert via `state.try_assemble_chunked_forward`, sweep via `state.sweep_stale_chunk_assemblies(ttl_secs)`.
- `state.listen_multiaddrs` — `ArcSwap<Vec<String>>` on the root; written ONLY by `NetworkManager::refresh_listen_multiaddrs()` (listeners ∪ external addrs, built by `build_reachable_multiaddr_list`); read by `PoolManager::handle_generate_invite_code`.
- `config.api.dashboard_trust_lan` — read via `SharedState::cfg()`; `api::dashboard_trust::classify` decides API-key hand-out, never `addr.ip().is_loopback()`.
- `state.observed_inbound_connection` — see `docs/invariants/state-and-config.md`
- `SharedState::model_is_in_use` is the answer to "may I delete this model's files?" — see `docs/invariants/state-and-config.md`
- Same-origin checks use `Origin` vs `Host` (`websocket.rs::ws_origin_allowed`), beside `classify` (gotcha #195).
- `state.relay_proven_features` — `DashMap<NodeId, RelayProvenFeatures>` on the root; recorded by `record_relay_proven_features`, read by `relay_feature_proven(peer, bit)`; any new relay send gate MUST consult it before the gossiped `NodeCapability.features`.

→ `docs/invariants/state-and-config.md`

## A settle that cannot be written down has not happened

`credit::escrow`'s `release_escrow`, `refund_escrow` and `cleanup_expired` MUST leave the entry `Pending` and the balance untouched, and return `Err`, when the status write fails. Direction is fixed: **lost-or-refunded, never double-paid.** Test a persist failure with `Database::set_write_failure(Some(tree))`, scoped to a TREE.

→ `docs/invariants/state-and-config.md` § "A settle that cannot be written down has not happened"

## A platform predicate answers "what kernel is this", not "where am I running"

**`config::network::wsl_network_adaptation(is_wsl2, in_container, mirrored)` is the one decision** about WSL2 network overrides (`None` / `Mirrored` / `NatSafeDefaults`). `is_wsl2()` is true inside a Docker container on Windows (#640; the SECOND too-broad predicate after #161), so keep `running_in_container()` strict and ask what else inherits a signal before a platform predicate chooses settings.

→ `docs/invariants/state-and-config.md` § "A platform predicate answers "what kernel is this", not "where am I running""

## Config defaults must stay live

The daemon must write **only values that differ from the compiled default**.
`config::to_minimal_toml` is the one serializer for the config file; do not call
`toml::to_string_pretty(&config)` directly.

→ `docs/invariants/state-and-config.md`

## A partial config update builds on the FILE, not on what the daemon remembers

`api::admin::base_for_partial_update` is the base for `PUT /api/admin/config`: the parsed `config.toml` when it parses, the live config otherwise — never `cfg()`, never a refusal. ⚠ `apply_live_config` has exactly one production caller besides `reload_config`; a second writer needs this rule revisited. Command-line overrides of a live setting (`--no-update-check`, `--anchor`) must be re-applied inside `apply_live_config` from the boot snapshot (FUTURE_WORK #107).

→ `docs/invariants/state-and-config.md` § "A partial config update builds on the FILE, not on what the daemon remembers"

## Single-source-of-truth helpers — SharedState, live config and credits

Each names the ONE place a decision is made; a second implementation is this codebase's most-repeated defect (`architecture.md` § "One invariant, N paths").

- `SharedState::release_request_state` (per-request maps), `SharedState::cfg()` (live config), `SharedState::record_peer_serve`, `config::InferenceConfig::claims_shard` (never read `shard_range` directly).
- `SharedState::local_executor_serves(&request)` — guard `the_singleton_executor_is_handed_a_request_only_through_one_predicate`; `SharedState::local_fast_path_for` (takes `swarm_route` as a REQUIRED argument; #633).
- ⚠ Credits are DORMANT — nothing may publish or act on a balance.

→ `docs/invariants/state-and-config.md` § "Single-source-of-truth helpers — SharedState, live config and credits"

