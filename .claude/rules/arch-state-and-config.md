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
- `state.models.shard_download_claims` — the shards a download task is WRITING right now, one RAII `ShardDownloadClaim` each, taken by `claim_shard_download` and released only by dropping it. `is_shard_in_progress` reads it; `shard_marked_in_progress` is the map-only sibling for a caller that already holds the claim. **Exclusion between writers must not rest on `acquisition_progress`** — that is a progress structure with several writers and a timer-driven deleter, and coupling the guard to it has failed in the field twice. → `docs/invariants/network.md`
- `state.models.shard_download_backoff` — external report 2026-07-23. `DashMap<ShardId, ShardDownloadBackoff { fail_count, retry_after: Instant }>`. Exponential per-shard download cooldown (30→60→120→240→300s cap, via the pure `shard_backoff_delay_secs`). Recorded via `record_shard_download_failure` at every terminal *transient* download-failure site (HF `download_shard` error + GGUF-probe failure in `model/auto_manage/download.rs`, P2P give-up-with-no-HF-source in `network/manager/shard_transfer.rs`, and stall-reconciliation in `health/monitor.rs::cleanup_acquisition_progress`). Checked by `shard_in_backoff` in `scoring.rs::gather_candidates` (skips the shard while cooling down). Cleared via `clear_shard_download_backoff` on success (HF success arm + P2P completion in `requests.rs`). Distinct from `shard_p2p_failed`, which only *forces* the HF path without throttling re-selection — the two solve different problems and a new failure site should touch whichever it needs. Do NOT record backoff on the P2P→HF fallback branch: that path wants an *immediate* HF retry. Entries self-evict from `shard_in_backoff` once idle past `SHARD_BACKOFF_FORGET_SECS` (1h), so the map stays bounded without a dedicated sweep.
- `state.models.removed_by_user` — 2026-08-21 (gotcha #360). `DashMap<ShardId, bool>`, persisted in DB tree `removed_shards`, loaded in `SharedState::new` like `locked_shards`. A shard the USER deleted (`delete_shard`, `delete_model` — every manifest shard) is an instruction, not a gap: `gather_candidates` skips it unless `in_configured_range || pinned_to_us`; an explicit request clears it (`hf_download_shards` for the named shards, `download_shard`, `pool_add_pin` naming this node). Helpers live in `daemon/state/removed_shards.rs` (`mark_shard_removed_by_user`, `shard_removed_by_user`, `clear_shard_removed_by_user`, `clear_removed_by_user_for_model`); the shard listing emits `removed_by_user` (only when not local) and the dashboard shows a "Removed" badge. Never write the map or the tree directly.
- `state.models.shards_needing_repair` — see `docs/invariants/state-and-config.md`
- `state.models.shards_pending_verification` — see `docs/invariants/state-and-config.md`
- `state.models.disputed_shards` — shards this node HOLDS whose bytes disagree with the swarm's hash and which are KEPT and served anyway, because that hash has no origin backing (`ModelRegistry::mismatch_policy`). Written and cleared only by `SharedState::note_shard_disputed` / `clear_shard_dispute`, from the **three** paths that compute `mismatch_policy` and can land on `KeepBytes`: the startup verification sweep, the auto-manage rescan, and the pending-verification drain (`auto_manage::manager::verify_pending_shards` — silent until 2026-09-14, and the path a PEER-PROVISIONED node actually reaches, so the report read `0` while the warning fired twice). All three clear on a later successful verify, which is the only thing that settles a dispute, and all three clear on EVERY success rather than only where a dispute is known — a clear that has to be predicted is a clear that gets forgotten. Passing `KeepBytes` as a CONSTANT to ask whether a file is already on disk is a question, not an acceptance, and records nothing. Guard: `every_path_that_keeps_disagreeing_bytes_records_the_dispute`. **Deliberately NOT `shards_needing_repair`** — that set's drain clears any mark whose file is on disk, which is every shard on this path. It exists so the disagreement is countable from OUTSIDE the log: the diagnostics report prints the section even at zero, because a pasted report saying `0` is a measurement and one that says nothing is not. **Read it through `SharedState::disputed_shards_now`, which self-evicts entries whose file has gone** — a dispute also ends when the shard is deleted or pruned, and those three call sites do not know about the set; a phantom would corrupt the very figure the set exists to measure. → `docs/invariants/network.md`
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
- `state.encrypted_pipeline_models` — **never read directly.**
  `SharedState::encrypted_pipeline_for` is the single answer to "is prompt
  privacy on for this model"; the map holds only the EXPLICIT per-model toggle
  and misses both automatic cases, one of which
  (`encrypted_pipeline_auto`, default ON wherever the node holds both ends) is
  how privacy is normally switched on at all. Use
  `privacy_explicitly_enabled_for` only where a DELIBERATE user choice is the
  question. `prompt_privacy_is_never_re_derived_from_the_per_model_map` in
  `tests/repo_consistency.rs` fails the build on a direct read.
- `state.region_demand` / `state.local_region_demand` — 2026-09-21. TWO maps, and the split IS the fix. `region_demand` is the MERGED view `(model, region) → rate`, written by the inbound `ModelDemandGossip` handler as well as locally, and read by auto-manage scoring, pruning and the wishlist, which all want the whole picture. `local_region_demand` is `model → rate` for THIS node's own region — the EMA of `models.model_request_counts`, maintained by `auto_manage::manager::decay_request_counts`, whose `old` value comes from this map so a same-region peer's gossip cannot inflate the figure we then publish as our own measurement. ⚠ **Only the local map may be GOSSIPED.** Publishing the merged one re-originated every peer's demand under this node's id every 30 s, refreshing the timestamps the staleness check relies on so entries could never age out — a node that had served zero requests published ~93 demand messages per tick about other people's traffic. The two have different key types, so the wrong one no longer compiles at the publish site, and `the_demand_we_gossip_is_the_demand_we_measured` guards the rest of the file. **GossipSub already propagates the originator's message; re-originating is not what makes it travel.** → `docs/invariants/network.md` § "Gossip volume"
- `state.metrics.inference_counts` — see `state.metrics.gossip`: the per-topic MESSAGE counts (`published`, `sent`, `recv`, `recv_unfiltered`) are parsed beside the byte counters, because bytes alone cannot say whether a topic is expensive from size, frequency or duplication. `published` vs `sent` separates speaking from relaying; `recv_unfiltered` vs `recv` is the duplicate factor.
- `state.metrics.bandwidth` — 2026-09-12. `Arc<BandwidthMeter>`; libp2p's transport counters, armed once when the swarm is built (`with_bandwidth_metrics`, the ONLY builder phase where a transport can be wrapped) and read back by encoding the registry. **`totals()` answers `None`, never `0`, when nothing is counting** — a figure that reads zero whether the node is silent or the counters were never wired is the reading this replaces. `refresh()` is called from the health-monitor tick and NOWHERE else, because a rate needs two readings at a known cadence; everything else reads `current()`. libp2p's transport counters, armed once when the swarm is built (`with_bandwidth_metrics`, the ONLY builder phase where a transport can be wrapped) and read back by encoding the registry. **`totals()` answers `None`, never `0`, when nothing is counting** — a figure that reads zero whether the node is silent or the counters were never wired is the reading this replaces. `refresh()` is called from the health-monitor tick and NOWHERE else, because a rate needs two readings at a known cadence; everything else reads `current()`.
- `state.metrics.gossip` — 2026-09-21. `Arc<GossipMeter>`; GossipSub's own per-topic byte counters, armed in `build_behaviour` (the ONE point `with_metrics` can attach, since it consumes and returns the behaviour). **Its own registry, not `bandwidth`'s** — the swarm builder holds a `&mut` borrow of that one while it runs, and the two are independently absent: the transport counters appear on the first byte of any protocol, these on the first gossip message SENT TO A PEER. An isolated node therefore reads `None` for ever and that is correct, not broken (`msg_sent` is recorded per recipient inside `send_message`). **Answers `None`, never `0`** — same reason as `bandwidth`. **Warns about nothing**, deliberately: every node reads `None` for its first seconds because `prometheus_client` writes a `Family`'s rows only once a label set exists, and a diagnostic on that fires on every start (gotcha #582). ⚠ **`libp2p-gossipsub` is a DIRECT dependency solely to enable its `metrics` feature** — the `libp2p` facade's `metrics` feature does not — and if that requirement ever drifts from libp2p's pin, cargo builds two copies and the split reads as silently absent. Guard: `the_gossip_counters_come_from_the_same_crate_libp2p_uses`.
- `state.metrics.inference` — 2026-09-21. `Arc<InferenceTraffic>`; the inference half of the traffic split, **counted in the CODEC** (`network/protocol/mod.rs`) and nowhere else. ⚠ **Not at the send sites, and that is the whole point**: `network.tensor_compression` defaults ON, so `dispatch_tensor_payload` holds the UNCOMPRESSED activation and a counter there reports bytes the interface never carried. `other_*` is the total minus the named categories, so an over-count corrupts the remainder as well as itself. `counts_as_inference` excludes shard transfers (counted at the cap's choke point — twice would break the sum) and `RelayedTensor` (somebody else's work, already `relay_bytes_forwarded`), and INCLUDES `StreamingToken`, which is most of what a whole-model node's inference costs. Counted as the full frame, header included, in both directions. Plain atomics, so zero genuinely means zero — unlike `bandwidth` and `gossip`, nothing here can be "not counting yet".
- `state.metrics.last_dispatch_at_ms` + `last_dispatch_kind` — 2026-09-19. Message-dispatcher liveness: epoch millis of the last message taken off `network_out`, and that message's `SwarmMessage` variant name. **Written by the dispatcher (`metrics.note_dispatch`, immediately after `recv()` returns and BEFORE the `match`, so it covers every arm including the `continue`s), read by `HealthMonitor::report_dispatcher_stall` on its own tick.** The split across two tasks is the point: `daemon::supervisor` reacts only when `JoinSet::join_next()` returns, i.e. to a panic or a clean exit, so a task parked for ever inside an `.await` produces no signal — 45 minutes of total silence produced not one supervisor line (`docs/FUTURE_WORK.md` #90). A heartbeat emitted from inside the dispatch loop would be just as silent, for the same reason the loop is stuck. `0` means nothing has been dispatched yet, which is not a stall. Guard: `the_dispatcher_liveness_marker_is_written_before_the_match_and_watched_elsewhere`. **Past `DISPATCH_STALL_AFTER` the node also WITHDRAWS inference from what it advertises**, through `SharedState::inference_outage` — the one predicate it shares with the dead-graphics-stack case, so two unrelated outages cannot advertise different things about the same node. Shard serving is untouched; `NodeCapability::can_serve_inference` carries it and defaults to `true` on the wire. Guard: `both_inference_outages_withdraw_through_one_predicate`.
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

## A platform predicate answers "what kernel is this", not "where am I running"

**`network::wsl_network_adaptation(is_wsl2, in_container, mirrored)` is the one
decision** about WSL2 network overrides, returning
`None` / `Mirrored` / `NatSafeDefaults` as a truth table rather than three
separate calls at the use site.

`is_wsl2()` reads `/proc/version`, so it is **true inside a Docker container on
Windows**, which inherits the host kernel's string and none of its networking.
That forced `listen_address = 127.0.0.1` on every containerised node — a
published container port cannot reach a loopback listener, so the node was
unreachable while looking healthy (report #003, gotcha #640). **This is the
SECOND time this predicate proved too broad**; gotcha #161 was the first.

`running_in_container()` ORs several signals because none survives every runtime
and cgroup version. Keep it strict: a missed container leaves the bug, but a
false positive on a real WSL2 shell undoes #161's fix. Before letting any
platform predicate choose settings, ask what else inherits its signal — a
container, a VM, a chroot, an emulator.

→ `docs/invariants/state-and-config.md`

## Config defaults must stay live

The daemon must write **only values that differ from the compiled default**.
`config::to_minimal_toml` is the one serializer for the config file; do not call
`toml::to_string_pretty(&config)` directly.

→ `docs/invariants/state-and-config.md`

## A partial config update builds on the FILE, not on what the daemon remembers

`PUT /api/admin/config` rewrites the whole document from a base, so the base
decides what survives. `api::admin::base_for_partial_update` is that choice: the
parsed `config.toml` when it parses, the live config when it is missing or
broken — never a refusal, because a file someone is mid-edit must not cost them
the change they just made in the dashboard.

Building it from `cfg()` destroyed any hand edit made while the daemon ran, which
is the documented way to set what the dashboard does not expose
(`bootstrap_peers`). Re-reading is correct rather than a trade-off because
**`apply_live_config` has exactly one production caller besides `reload_config`,
and it is this handler** — so no runtime state lives outside the document, and
the only way file and live config diverge is an edit worth keeping. A second
production writer would break that argument: add one and this rule needs
revisiting, not just extending.

→ `docs/invariants/state-and-config.md`

## Single-source-of-truth helpers — SharedState, live config and credits

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/state-and-config.md`

- **`SharedState::release_request_state`** — clears the maps a finished request leaves behind: `active_pipelines`, `active_traces`, `request_holder_blacklist`, `peer_vram_commitments` and `local_memory_refusals` — keyed by request id, sharing one lifetime. Deliberately does NOT touch `active_count` or `queue_notify`.
- **Credits are DORMANT — nothing may publish or act on a balance** — `MIN_BALANCE_FOR_INFERENCE = 0` and `calculate_tier` returns `DORMANT_TIER` whatever it is given, so no balance affects who is served or how fast, and the leaderboard neither ranks by credits nor publishes them.
- **`SharedState::cfg()`** — the live config, and the single answer to "what is this setting **now**".
- **`SharedState::record_peer_serve`** — the single answer to "this node did inference work for a peer", counting it AND billing for it.
- **`config::InferenceConfig::claims_shard`** — the single answer to "does this node claim shard N?", i.e. how `inference.shard_range` is read. **Never read `shard_range` directly.**
- **`SharedState::local_fast_path_for` is the single answer to "may this request take the local split fast path?"** — both API surfaces used to compose it themselves (`has_complete_split_model && !should_offer_work_to_the_swarm`), and the fast path skips the router — which is where `delegation_target` lives. **It takes the request's `swarm_route` override as a REQUIRED argument**, because it decides whether the request ever reaches the router and the router is the only reader of that override: without it the knob was inert on every node holding a whole model, which auto-manage deliberately converges nodes on being (gotcha #633). The two dispatch callers pass what the caller asked for; the admin LISTING passes `None`, because a listing answers for the node and not for a request. `override_keeps_whole_model` is the pure decision, a truth table beside `local_fast_path_allowed`. **This fast path has now eaten a feature three times** (#187, #443, #633) — when adding anything request-scoped, grep for every early return between the API edge and the code that reads it.
