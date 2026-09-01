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
- `state.models.foreign_wishlist` — R130. `DashMap<(NodeId, ModelId), (score_0_100, received_at_ms)>`; capped at `MAX_FOREIGN_WISHLIST_ENTRIES = 10_000` with oldest-first eviction, 2h freshness window enforced on read. Written by `apply_wishlist_announcement` on inbound `SwarmMessage::WishlistAnnouncement`; read by `compute_wishlist` for the 0..10 cross-pool demand boost.
- `state.models.quant_recommendations` — R133. `ArcSwap<QuantRecommendations>`; refreshed via `crate::model::auto_manage::quant::refresh_quant_recommendations(state)` on every auto-manage tick AND on every WS stats build. Read by `GET /api/admin/quant-recommendations` and the swarm-tab tips tile.
- `state.models.shard_download_backoff` — external report 2026-07-23. `DashMap<ShardId, ShardDownloadBackoff { fail_count, retry_after: Instant }>`. Exponential per-shard download cooldown (30→60→120→240→300s cap, via the pure `shard_backoff_delay_secs`). Recorded via `record_shard_download_failure` at every terminal *transient* download-failure site (HF `download_shard` error + GGUF-probe failure in `model/auto_manage/download.rs`, P2P give-up-with-no-HF-source in `network/manager/shard_transfer.rs`, and stall-reconciliation in `health/monitor.rs::cleanup_acquisition_progress`). Checked by `shard_in_backoff` in `scoring.rs::gather_candidates` (skips the shard while cooling down). Cleared via `clear_shard_download_backoff` on success (HF success arm + P2P completion in `requests.rs`). Distinct from `shard_p2p_failed`, which only *forces* the HF path without throttling re-selection — the two solve different problems and a new failure site should touch whichever it needs. Do NOT record backoff on the P2P→HF fallback branch: that path wants an *immediate* HF retry. Entries self-evict from `shard_in_backoff` once idle past `SHARD_BACKOFF_FORGET_SECS` (1h), so the map stays bounded without a dedicated sweep.
- `state.models.removed_by_user` — 2026-08-21 (gotcha #360). `DashMap<ShardId, bool>`, persisted in DB tree `removed_shards`, loaded in `SharedState::new` like `locked_shards`. A shard the USER deleted (`delete_shard`, `delete_model` — every manifest shard) is an instruction, not a gap: `gather_candidates` skips it unless `in_configured_range || pinned_to_us`; an explicit request clears it (`hf_download_shards` for the named shards, `download_shard`, `pool_add_pin` naming this node). Helpers live in `daemon/state/removed_shards.rs` (`mark_shard_removed_by_user`, `shard_removed_by_user`, `clear_shard_removed_by_user`, `clear_removed_by_user_for_model`); the shard listing emits `removed_by_user` (only when not local) and the dashboard shows a "Removed" badge. Never write the map or the tree directly.
- `state.models.shards_needing_repair` — 2026-08-25 (gotchas #381/#382). `DashSet<ShardId>`
  of shards whose bytes were found WRONG and which need a fresh, verified copy. Written
  ONLY by `SharedState::mark_shard_for_repair`; drained by
  `AutoShardManager::complete_pending_shard_fetches`; cleared by
  `clear_shard_repair` on a landed copy.
  **Why it exists**: three places can catch a bad shard — the P2P accept path, the
  background verification sweep, the auto-manage rescan — and all three removed the file
  and stopped. "Removed" was implemented three times and "and get a good one" nowhere, so
  repair happened only as a side effect of auto-manage noticing the gap: a node with
  auto-manage OFF kept a permanently incomplete model, and every rescan re-hashed the same
  bad file to reach the same conclusion. A new site that detects a bad shard calls the
  helper; it must not open-code the removal.
  **Deliberately NOT `shard_p2p_failed`**, which forces the HuggingFace path. Having
  DETECTED the corruption means we hold the real hash, so a peer copy is checked against
  it — and if that one is bad too the accept path quarantines it and docks the sender,
  which is the behaviour wanted. Repair therefore runs from peers OR the origin.
  **It runs outside the `auto_manage.enabled` gate** for the same reason
  `try_idle_vram_unload` does: a shard this node already held is not a new acquisition
  decision. And it refuses a shard in `removed_by_user` — a deletion is an instruction,
  not a gap.
- `state.models.shards_pending_verification` — 2026-08-25. `DashSet<ShardId>` of shards
  THIS NODE HOLDS whose expected hash just changed. Written by the registry's
  manifest-update hook (`set_persist_hook`, which now carries
  `(manifest, persist, recheck_shards)`); drained by
  `AutoShardManager::verify_pending_shards`, which re-hashes and, on mismatch,
  drops the holder claim and calls `mark_shard_for_repair`.
  **This is how a node learns from the SWARM that what it is serving is wrong.**
  The only other re-check of an already-held shard is the one-shot startup sweep,
  which runs ~2 s after boot against whatever the DB held — i.e. BEFORE a corrected
  hash can arrive by gossip — and the rescan explicitly skips shards already
  registered. So a node holding a corrupt shard could not discover the fact from its
  peers at all; it took a further restart, after the persist hook had written the
  corrected hash to the DB. Measured on the live swarm (gotcha #382).
  Three things to keep: only shards we ACTUALLY HOLD are queued (re-hashing one we do
  not have is hundreds of MB of I/O for no answer — pinned by
  `a_held_shard_is_rechecked_when_its_expected_hash_changes`); a shard with a download
  in flight is skipped, since re-hashing a partly-written file is a false alarm; and
  the drain runs OUTSIDE the `auto_manage.enabled` gate, because serving neighbours
  bad bytes is not a disk-management preference.
- **`ModelRegistry::origin_verified`** (2026-08-25) — a shard hash derived from bytes
  THIS node fetched from the model's origin, which outranks any gossiped claim. Applied
  in `register_manifest` BEFORE change-detection, so a claim the origin has already
  disproved does not even provoke a re-check. Persisted (`ORIGIN_VERIFIED_TREE`) and
  loaded FIRST in `load_from_db`, or a restart hands the argument back to gossip.
  Recorded by `SharedState::record_origin_downloaded_shard` from BOTH origin-download
  paths — the auto-manage downloader and the admin "download this part" handler; the
  second recorded nothing until it was added, which is the same one-invariant-N-paths
  trap as everything else in this file.
  **Why**: manifest registration is last-writer-wins, and a real hash replaces a real
  hash (deliberately, for re-publishes). A peer that had self-certified a corrupt shard
  gossiped its wrong hash, our node adopted it over one verified against the origin, and
  v0.3.121's new re-check then faithfully **quarantined the GOOD copy** and refetched
  against the same wrong reference — an unbounded ~500 MB loop, observed live within an
  hour of shipping (gotcha #384).
  **Deliberately local and NEVER gossiped**: provenance that travels the network is just
  another assertion, and forgeable. A node trusts only what IT fetched.
  **The general rule this encodes**: a repair mechanism is a destruction mechanism
  pointed at whatever it believes is wrong. Before adding one, ask what happens when the
  REFERENCE is the thing that is wrong.

- **`network::manager::tensors::AckRttEstimator`** (2026-08-25) — how long a peer gets to
  acknowledge a tensor forward before the pipeline gives up on it. **RFC 6298**, the same
  algorithm TCP uses to decide a packet is lost: `deadline = SRTT + 4*RTTVAR`, alpha=1/8,
  beta=1/4, clamped to `RR_ACK_TIMEOUT_SECS`..`RR_ACK_TIMEOUT_MAX_SECS`, over the
  observed send→ACK latency of THAT peer.
  **The variance term is the point.** The rule it replaces scaled a ping RTT, and a ping
  cannot see queueing delay on a loaded node — the ACK is emitted by the network event
  loop, so it is late exactly when that loop is busy. Measured: a peer at ~500 ms RTT (so
  the 10 s floor) failed EVERY distributed request for about an hour, then served the
  same request in 6.1 s once it settled. Its ACKs were arriving late, not missing —
  proved by running a node at `-v` and seeing `kind="ack"` from six peers including that
  one (gotcha #386).
  **`observe_timeout` (RFC 6298 §5.5) is not optional.** The estimator only ever sees
  acknowledgements that ARRIVE, and an abandoned forward is one whose ACK we stopped
  waiting for — so a peer needing longer than the current deadline can never produce the
  sample that would widen it. Without backing off on a miss the whole scheme is INERT
  exactly where it is needed. Bounded by the same ceiling; a peer that recovers returns
  to the floor.
  **The fast-fail also requires a standby** (`request_has_standby`). It is justified by
  the failover it enables — the premise of hedged requests in Dean & Barroso's *The Tail
  at Scale*, which send a second copy to a DIFFERENT replica. With none, abandoning
  cannot buy a failover and can only turn a slow success into a 503: measured, a reply
  arrived 1.6 s after we gave up and was discarded. Absence of the pipeline entry counts
  as "yes", so paths this check cannot see (a chain hop) keep the old behaviour.

- **A KV-budget refusal MUST release what the request already took** (2026-08-25) —
  `executor.rs` calls `kv_cache_store.clear_request` before returning the
  `ServiceUnavailable`. A chunked prefill claims a quantum per boundary, so a request
  refused at chunk N has already allocated N-1; leaving them made a refusal a RATCHET —
  the request failed, its cache survived the 10-minute session TTL, and the next attempt
  started from a higher floor. Measured in exact 1152 MB steps: 1152 → 2304 → 3456
  against a 1166 MB budget, card at 97%, decode 29 → 1.0 tok/s (gotcha #387).
  Safe because the request is over: an error is being returned, no later forward reuses
  the cache, and a retry rebuilds it from the prompt.
  **Generalise it**: a resource check that refuses but does not release is worse than no
  check under repetition, because the failed attempts are exactly the ones that
  accumulate. Ask of any admission check what happens to what the refused request took.

- **`SharedState::can_fetch_shard_from_origin`** — 2026-08-25 — the single answer to
  "will an origin fetch actually HAPPEN?", which is NOT "does an origin exist". Every
  caller about to discard local bytes in favour of an origin copy asks this first:
  **never throw away data you cannot replace.** Two conditions, found one at a time and
  each after concluding the other was the only one — a recorded `hf_source`, AND not
  offline mode (`trigger_download` skips the HuggingFace branch entirely when set, by
  design). Auto-manage being off is deliberately NOT one, since the drain runs outside
  that gate. Consumed by the P2P accept path (`classify_p2p_shard_acceptance`) and by the
  rescan's no-hash branch. A third condition belongs here, not at a call site.
- `state.credits.foreign_pool_catalog` — R134. `DashMap<(PoolId, ModelId), received_at_ms>`; capped at 5000 with oldest-first eviction, 2h freshness window. Written by inbound `SwarmMessage::PoolModelAvailability` handler. Read by `GET /api/admin/foreign-pool-catalog` and by `pool::scope::cross_pool_extras` (R134.7) when `pool.allow_cross_pool_inference` AND `private_mode` are both on.
- `state.metrics.node_stats` — NOT `state.node_stats`
- `state.metrics.providers_config` — NOT `state.providers_config`
- `state.metrics.swarm_capacity` — R110. ArcSwap<SwarmCapacity>; refresh via `crate::daemon::state::refresh_swarm_capacity(state)`. Eagerly refreshed on peer connect (`network/manager/identify.rs`) and disconnect (`network/manager/connections.rs`) so the dashboard banner stays consistent with the peer-list panel under churn — the WS stats-cache 1.5s coalesce alone is too lazy.
- `state.metrics.hedge_tracker` — R136 Layer 2. `Arc<HedgeTracker>` with per-(model, segment, holder) EWMA latency + rate-budget counters. Always present. Observation via `state.record_hedge_observation(...)` from the forward-success path in `pipeline/distributed.rs` (post-hoc dry-run metrics). Real race-then-discard duplicate dispatch for speculative-verify hops ships in `pipeline/hedge_dispatch.rs::forward_verify_with_hedge` — single-segment only; multi-segment hedging remains deferred (`docs/FUTURE_WORK.md`). R142.6 added `last_observed_at_ms` to `HedgeStats` and `HedgeTracker::evict_stale` wired to the HealthMonitor tick to bound the (model × segment × holder) map.
- `state.metrics.prefetch_orchestrator` — R136 Layer 3. `PrefetchHandle` (Arc<PrefetchOrchestrator>) with per-session first-token histogram + idle-time learner + throttling. Observation via `observe_user_turn(session, first_token)` + `record_response_completion(session, now_ms)` at the router success site. R142.6 wired `evict_idle` to the HealthMonitor tick to bound the histories map. K-layer prefetch dispatch is the remaining integration; data-collection and orchestration are complete.
- `state.standalone_tokenizers` — R136 Layer 1/3 follow-on. `DashMap<ModelId, Arc<SplitTokenizer>>` on the ROOT SharedState (not a sub-struct — used by both `state.metrics`-derived L3 prefetch AND the `pipeline/ngram_only_spec.rs` L1 path, so cross-cutting). Lazy-loaded from `gguf_header.bin` via `state.standalone_tokenizer(&model_id)` accessor. Returns `None` when the header isn't on disk; caller falls through gracefully.
- `state.pending_activation_chunks` — R139 Tier 4K. `DashMap<Uuid, ChunkAssemblyState>` on the ROOT SharedState (cross-cuts the RR-decrypt path in `network/manager/tensors.rs` and the persistent-stream reader in `network/pipeline_stream.rs`). Receiver-side assembly for STREAM-chunked activation forwards. Entry-locked insert via `state.try_assemble_chunked_forward(forward, sender_peer_bytes)`. Periodic stale-entry sweep wired to the HealthMonitor tick via `state.sweep_stale_chunk_assemblies(ttl_secs)`. Chunk-meta is bound into AAD via `build_layer_forward_aad`, so reorder/truncation/cross-transfer-substitution fail Poly1305 before reaching the assembly.
- `state.listen_multiaddrs` — R140. `arc_swap::ArcSwap<Vec<String>>` on the ROOT SharedState (cross-cuts NetworkManager-writes and PoolManager-reads). Live snapshot of the swarm's reachable addresses, each terminated with `/p2p/<local_peer_id>`. Written by `NetworkManager::refresh_listen_multiaddrs()` (events.rs) on `NewListenAddr` / `ExpiredListenAddr` / `ListenerClosed` / `ExternalAddrConfirmed` / UPnP `NewExternalAddr` / `ExpiredExternalAddr`, plus once at startup after `listen_on()` (and after the `network.external_addresses` config override is added). **R143: the snapshot is the UNION of `swarm.listeners()` (bound sockets — private LAN on a NAT'd node) AND `swarm.external_addresses()` (UPnP-mapped / AutoNAT-confirmed / relay-circuit / manually-declared public addrs).** Without the union a NAT'd node's invite code silently shipped a LAN-only address. Built via the extracted, unit-tested `build_reachable_multiaddr_list(candidates, peer_id)` + `ensure_p2p_suffix` helpers; filtered through `addr_is_remotely_reachable` — keeps LAN + Tailscale CGN (100.64.0.0/10) + public, drops loopback / unspecified / link-local / IMDS. Read by `PoolManager::handle_generate_invite_code` when minting v2 `swarmpool://` codes; empty list → `SwarmError::ServiceUnavailable`. When the list has entries but NONE pass the stricter `pool::invite::any_internet_reachable` (public IP / DNS / relay-circuit — excludes LAN + CGN), invite generation still succeeds but emits a `pool`/`invite_lan_only` warning ActivityEvent so the user isn't handed a LAN-only code that dies over the internet.
- `config.api.dashboard_trust_lan` — read via `SharedState::cfg()` (see below), never re-derived with `addr.ip().is_loopback()`. `api::dashboard_trust::classify` is the single answer to "may this request be handed the API key automatically?"; the sibling `dashboard_trust_overlay` is read the same way. Was a private `AtomicBool` mirror until 2026-08-09, folded into the live config when that became general.
- `state.observed_inbound_connection` — `AtomicBool` on the ROOT SharedState,
  **persisted, and written only via `SharedState::record_inbound_connection_observed`**.
  Set by `handle_connection_established` for a non-loopback connection where we
  are the LISTENER. **The only direct evidence that inbound reaches this node**:
  outbound succeeds from behind almost anything, so a node with peers, a real
  LAN address and clean logs can still be silently dropping every inbound packet
  and look perfectly healthy from the inside. Read by the WSL2 firewall check in
  `health::monitor`. Any future "are we reachable" question should use this
  rather than inferring from peer count — having peers proves only that WE
  dialled successfully.
  **It is seeded from the database at startup, and the persistence is the load-
  bearing part.** The fact being recorded is a property of the machine's
  network, and restarting the daemon does not reconfigure a firewall — so an
  in-memory-only observation made the check re-decide that question every start
  from whatever happened in the next few minutes. A reachable node routinely
  sees nothing in that window: it dials every peer it already knows in the first
  seconds of starting (10 connections inside 2 seconds, measured), so it is the
  dialer on every link and may never be dialled back at all. The result
  was a node telling its owner to run Administrator PowerShell firewall commands
  it did not need. Measured 2026-08-18 on this development machine: inbound TCP
  open and verified by hand from a peer on the same subnet, 181 inbound
  connections across the log's history, **zero in a 9-hour run**, and a run that
  warned at 06:47 contradicted by its own inbound connection at 07:41. Three of
  the four most recent runs warned; all three were wrong (gotcha #335).
  **There is no length of silence that proves unreachability**, so the grace
  period is a noise control, not the fix, and the message reports what was
  observed rather than naming a cause. A new check that wants to conclude
  something about this machine's network must persist its evidence the same way;
  an observation whose lifetime is shorter than the fact's cannot support the
  claim.
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

## The units decide whether a forward is a prefill, not the byte count

**`inference::pipeline::local::PipelineExecutor::forward_is_prefill(activation_bytes, units)`**
is the single answer to "is this forward doing a prefill?", for the deadline
(`compute_segment_timeout`) and for the DIAG that reports it. `SegmentBudget`
carries the resolved verdict (`is_prefill()`) so the log cannot contradict the
budget it is describing.

**`ActivationUnits::PromptBytes` means prefill, unconditionally.** Segment 0 of a
non-pre-embedded pipeline is handed the prompt itself; every later hop carries
hidden states. A forward carrying the prompt performs the whole prefill by
construction, however short the prompt — so the size test does not apply to it.
Only `HiddenStates` may be classified by size, because
`PREFILL_ACTIVATION_THRESHOLD_BYTES` (100_000) is a **hidden-state** scale: one
token of hidden state is thousands of bytes, one token of prompt is a few.

**Why it is a rule.** The units were already explicit, and already honoured on
the *measured* path — `for_forward` refuses to predict from the peer-speed
coefficient for `PromptBytes`, with a comment saying that feeding those into the
same average "would silently corrupt the estimate". It then fell through to a
fallback that made exactly that unit error: prompt bytes compared against the
hidden-state threshold answered "decode" for anything under ~100 KB of text
(~25k tokens), i.e. essentially every real prompt. The forward that does the
entire prefill got `DECODE_SECS_PER_LAYER = 2` rather than
`PREFILL_SECS_PER_LAYER = 15` — 32 s instead of 240 s at 16 layers. Measured on
the live swarm 2026-08-29 (gotcha #407): a 4728-token prompt, `activation_bytes
= 24045`, holder abandoned after 32 s of a job needing minutes; the standby
answered correctly and the request succeeded, so **nothing failed and nothing
alerted** — the whole cost was a wasted deadline inside a 176 s request.

The guard existed on the sophisticated path and was missing from the crude one
beneath it. When a value's units depend on which path produced it, every
consumer needs the units — the fallback included. A threshold is a comparison
against a scale, so it is a unit conversion wearing a constant's clothes.

**Do not re-derive the verdict at a call site**, and do not widen
`PREFILL_ACTIVATION_THRESHOLD_BYTES` to "cover" prompts: that would break the
`HiddenStates` classification it was chosen for. `SegmentBudget` remains
constructible only through `for_forward` for the same reason it always was.

## A ticker merged into a response stream is a termination condition

`api::sse::progress_ticker` is the ONE keep-alive/progress ticker for both SSE
encoders. Its wait is cancellable — `tokio::select!` on the interval versus a
`tokio::sync::watch` finish signal, which is why the signal is a `watch` and not
an `AtomicBool`: the ticker has to *wait* on it, not merely read it.

**Why it is shared.** OpenAI and Anthropic each had a byte-identical copy, and
both carried the same defect: sleep the whole interval, THEN check whether the
response had finished. `Stream::merge` ends only when both halves end, so every
streamed reply stayed open for the remainder of that sleep — measured, an
8-token reply delivered in 0.5 s held its connection to 15.0 s, while the same
request answered non-streaming in 0.56 s (gotcha #390). Clients that stop at
`[DONE]` never noticed; anything reading to end-of-stream waited, and the server
held a task and a connection per stream either way.

The comment above the old copy said *"the ticker MUST terminate"* and was right
about the hazard it had in mind — an unbounded ticker holds the response open
for ever. **Terminating late is the same bug with a bound on it.**

Three things a change here must keep:

- **A dropped sender ends the ticker.** The token stream is gone; there is
  nothing left to keep alive. Without this, `changed()` returning `Err` in a
  loop that ignored it would spin.
- **An unfinished response still gets keep-alives**, or a slow request looks dead
  to the client and to any intermediary. That is the hazard the original comment
  was written about and it is still covered by a test.
- **The interval is a `Duration`, not a count of seconds.** That is what lets a
  test drive it at millisecond scale and assert exactly, with no `tokio`
  `test-util` dev-dependency: an hour-long interval means a ticker that waits it
  out cannot possibly answer inside the timeout.

**A note on the observed period.** Keep-alive comments arrive at alternating
gaps of 12.52 s and 15.00 s, not a flat 15 s, and end-of-stream used to land on
whichever boundary came next — which is why the pre-fix measurements clustered at
those same two values and why they looked inexplicable until the ticker itself
was timed. The likely cause is two sources beating against each other: the merged
ticker on its own interval, and axum's `KeepAlive`, which resets on every write.
That explanation is *unverified* — it fits the numbers and nothing depends on it,
since the hold it produced is gone.

Ask of any periodic task merged into a response stream: when the thing it is
keeping alive finishes, how long until this notices?

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
- **`model::auto_manage::vram::ADMISSION_KV_CONTEXT`** (2026-08-18; CPU since
  2026-08-21) — the context length admission charges KV cache for, on either device,
  whatever the user configured.
  **Admission may charge less than the worst case exactly where a runtime check
  catches the difference, and nowhere else.** Both workers now have one:
  `kv_budget::claim_exceeds_headroom` in `forward_inner_impl`, a 503 that re-routes.
  The GPU worker derives its budget from free VRAM at load; the CPU worker is HANDED
  its budget by the daemon at spawn (`--kv-budget-bytes` →
  `inference::split::CPU_KV_BUDGET_BYTES`, computed by
  `ModelProcessPool::record_cpu_kv_budget` as the typical-context charge plus the RAM
  budget still uncommitted at admission), because only the daemon knows what else is
  resident. **`ModelProcessPool::charges_ram` decides whether a spawn is charged against
  RAM at all** — going to the CPU, OR no GPU detected, OR a build without CUDA. The
  first alone missed every CPU-only node: with no GPU there is no VRAM budget,
  `admit_to_gpu` admits everything, and the model landed in RAM uncharged and
  un-budgeted. It changes charging, never placement — the worker still falls back
  to the CPU on its own, so a working card whose probe failed is not sent to the CPU
  by it (unreadable is unknown, not absent). Until 2026-08-21 the CPU had no guard, so `estimate_worker_ram_mb` priced
  the WHOLE ceiling — correct for the mechanism that existed, and what turned a 2.3 GB
  phi-3.5 into a "needs 27125 MB" refusal at a 32k override (external report; MHA,
  0.75 MB/token). **Do not remove one side without the other**: admission at a typical
  context with no runtime guard means swapping, which degrades every request on the
  machine; a guard with ceiling-priced admission means refusing models that fit.
  `a_cpu_refusal_itemises_weights_kv_and_context` pins that the same model is now
  admitted and that the old ceiling figure is still what `resident_footprint` reports
  when asked for it.
  **The RAM budget itself is a LIVE snapshot, never a startup figure**:
  `vram::ram_budget_now` → `RamBudget { cap_mb (from `cfg()`), live_headroom_mb
  (max(70% of available NOW, total/4)) }`, installed into the pool as a provider
  closure (`set_ram_budget_provider`, `Weak<SharedState>`) and asked at EVERY
  admission, by `free_ram_for_admission`'s retry loop, and by `record_cpu_kv_budget`.
  The clamp used to be folded into the cap once at startup, so a daemon restarted
  while memory was busy carried the smaller figure for life — the same
  `max_ram_mb = 18000` answered "budget allows 13370 MB" one day and "10500 MB" the
  next with 14773 MB actually free (external report, gotcha #362). The refusal names
  whichever limit applied (`RamBudget::limiting_figure`). `set_ram_budget_mb` is the
  no-provider fallback (tests) and the startup log figure; do not read it for a
  decision.
  Why it exists: `inference.max_seq_len_override` is the only way to hold an agentic
  client's system prompt (~5000 tokens of tool schema before the user speaks), and
  raising it used to raise this charge in step, so the model stopped fitting the card
  and was loaded on the CPU — measured at 396 s of prompt processing and a thermal
  warning (external report 2026-08-17). Pre-paying at load bought nothing the runtime
  check was not already enforcing.
  **Deliberately NOT `DEFAULT_MAX_SEQ_LEN`.** That is a product default and moves with
  the audience — it went 4096 → 8192 the same day — while this is a statement about a
  typical working conversation. Tying them would mean raising the default silently
  re-broke the case above. `raising_the_context_no_longer_costs_a_model_its_place_on_the_gpu`
  and `the_cap_is_inert_at_the_context_it_was_derived_from` pin both halves.

- **`ModelProcessPool::free_vram_for_admission` + `plan_vram_reclaim`** (2026-08-25)
  — reclaim graphics memory from models nothing is using rather than demoting the
  requested one to the processor. Called from the admission-refusal branch in
  `get_or_spawn_worker`, BEFORE the CPU fallback is taken. **The exact sibling of
  `free_ram_for_admission`**, which has done reclaim-then-retry for the RAM budget
  since v0.3.111; the GPU side simply never had it, so a model that happened to
  load first kept the card for as long as it stayed resident and everything asked
  for afterwards ran on the CPU (gotcha #388).
  **Why the pre-existing LRU eviction did not cover it**: `evict_split_models_lru`
  frees against `estimate_vram_from_shard_dir` (shard bytes × layer fraction —
  weights ONLY) while `admit_to_gpu` weighs `estimate_gpu_footprint_mb` (weights +
  KV at `ADMISSION_KV_CONTEXT`). Two estimates of one quantity, and the SMALLER one
  decides how much to free — so eviction reaches its own stop condition while
  admission is still short, every time. Measured live: eviction freed 1202 MB, then
  admission refused against a 2449 MB shortfall and the model loaded on the CPU at
  ~7 tok/s with the card at 12%.
  **Neither timer can do this job.** `try_idle_vram_unload` runs on the auto-manage
  tick, so it cannot act on the request arriving *now*, and its regional-demand
  clause deliberately keeps a model the SWARM wants resident for up to
  `idle_hard_unload_secs` (1 hour) — precisely a popular 8B.
  Three properties a change here must keep. **Plan before destroying**: the whole
  plan is costed first and abandoned whole if it cannot succeed, because unloading
  models and still not fitting costs a cold start and buys nothing. The RAM sibling
  may unload opportunistically — its alternative is failing the request outright;
  here the alternative is a slower answer, so a wasted eviction is a real
  regression. **An idle floor** (`VRAM_MAKE_ROOM_MIN_IDLE_SECS`): two models
  alternating faster than they load would otherwise evict each other on every
  request; below the floor the previous behaviour is kept, so this can only improve
  placement. **LRU by real idle time, not residency**: `spawned_at` cannot tell a
  worker answering steadily for an hour from one loaded an hour ago and never used
  since, so `WorkerHandle::last_used` is stamped in `register_response` — the ONE
  place every execution path (local, distributed, peer-served) passes through.
  **Do not "correct" the estimate.** 6033 MB charged against a measured
  `vram_after_load_mb=4853` is not a 24% over-charge; the difference is the KV
  headroom doing its job, and trusting the measured figure would admit a model and
  then OOM it.

- **`should_return_to_gpu` + `ModelProcessPool::worker_should_return_to_gpu`**
  (2026-08-27) — the single answer to "is this resident worker still in the right
  place?", asked on the request path in `get_or_spawn` rather than on a timer.
  **A pin that clears buys nothing while the worker it produced is still running.**
  A momentarily-full card demotes a model and pins it; `unload_model` lifts the pin
  when a GPU tenant goes away and logs `clearing CPU pins`; and `get_or_spawn`'s
  fast path then returned the processor worker regardless of device. `clear_cpu_pin`
  is documented as letting "the next worker spawn" use the card, and there is no
  next spawn — the worker survives until `idle_unload_secs` (15 min) of *no requests
  at all*, which someone actively using the model never reaches. Reported by an
  external tester: gemma-2-2b-it re-requested against a card at 653 of 6141 MB
  reused its processor worker (gotcha #401).
  **The decision reads the WORKER, not the model.**
  `WorkerHandle::placed_on_cpu_because` records why the process actually went to
  the processor, at the moment it was spawned; `cpu_reason` (and so
  `effective_gpu_layers`) answers for a spawn happening *now*, off the pin that is
  exactly the thing being cleared. **A running worker is a fact; `cpu_reason` is a
  prediction**, and they differ precisely in the window this change creates. Three
  callers were asking the prediction about a resident worker and are now corrected
  to the fact: `unload_model` (whether killing this worker freed graphics memory —
  a model on its way back to the card would have lifted every other model's pin on
  the strength of memory it never held), `cpu_placement_reason` (the dashboard's
  "why is this not on my GPU?" — which would have dropped its explanation from a
  model still on the processor, and explained a model happily on the card as "you
  configured CPU-only"), and `would_fit_on_gpu`'s already-charged short-circuit,
  which would otherwise have re-created the 2026-08-18 contradiction its own
  comment describes. **A new caller asking about a model that has a worker should
  ask the worker.**
  `WorkerHandle::gpu_estimate_mb` caches what admission priced the model at,
  because `estimate_gpu_footprint_mb` re-reads `gguf_header.bin` and scans the
  model directory: fine once per spawn, not fine once per request.
  Four guards, three carried over from `plan_vram_reclaim` because this destroys
  something that works. Only a worker THIS NODE demoted (on a machine with no card
  `cpu_placed` is false, and there is nowhere to promote to). The demotion must have
  stopped applying — of `cpu_reason`'s three causes only the OOM pin ever clears, so
  this fires on the event that lifted it rather than polling. Never a busy worker,
  and not one used inside `VRAM_MAKE_ROOM_MIN_IDLE_SECS`, because unloading kills
  the subprocess and the idle floor makes the race window empty rather than merely
  unlikely (the residual — a model under continuous load waits for a gap — is in
  `docs/FUTURE_WORK.md`). And **cost the move before making it**: an unreadable
  footprint or an unset budget is not evidence, and leaves the model where it is.
  `admit_to_gpu` treats the same gap as "do not judge" and lets a spawn through;
  the question here is whether to destroy something working, and the answer on no
  information is no.
  **The outcome is announced after admission, not before it.** The retirement logs
  what it is doing; the user-facing `model_gpu_restored` event is emitted only once
  the worker is on the card, because admission prices the model again and has the
  last word — the same correction `admit_to_gpu`'s refusal log already carries.

- **Graphics memory has ONE owner: `ModelProcessPool`** (2026-08-27). It admits
  (`admit_to_gpu`), charges (`vram_reserved_mb`), and reclaims — on demand
  (`free_vram_for_admission`) and on the idle timer (`try_idle_vram_unload`,
  which runs outside the auto-manage gate). **Nothing else may take memory away
  from a loaded model.**
  **What this replaced.** `SharedState.split_models` — a cache of per-segment
  metadata read out of `gguf_header.bin` — was governed by its own VRAM budget
  that evicted entries *and unloaded their workers*. Two accountants for one
  card, and the second was the weaker: a different estimate (weights only,
  against the pool's weights + KV — gotcha #388's shape), a different in-flight
  oracle (`active_pipelines`, which per gotcha #194 cannot see peer-served work
  or the split fast path), and **no idle floor at all** — measured, it evicted a
  model that had answered a request three seconds earlier, which
  `free_vram_for_admission` refuses by design. Worse, it was placement-blind, so
  registering a metadata entry for a segment bound for the PROCESSOR killed a
  model running happily on the card (gotcha #402).
  **Researched, not guessed.** Ollama's scheduler is the direct analogue and
  keeps one owner: a single centralised free-space tracker, `runnerRef.vramSize`
  reported by the runner, victims chosen by refCount (in-flight) → keep-alive →
  `lastUsedAt`. Our pool already had the equivalent of all three.
  **The split-model map is now a metadata CACHE**, bounded by
  `MAX_SPLIT_MODEL_ENTRIES` via `inference::split::trim_split_model_cache` —
  count-capped, LRU, active-pipeline-protected, and structurally unable to
  unload anything (it takes no pool and returns only keys). **Do not restore the
  unload**: it existed (2026-07-21) because eviction was *supposed* to free
  graphics memory and did not, so the budget was enforced against a phantom.
  That premise is gone — this no longer claims to free anything. Trimming an
  entry still wanted now costs a header re-read, not a killed worker.
  **A registration budget survives, and may refuse but never take.**
  `SharedState::split_model_budget_with` + `split_models_committed_mb` + the
  `MemoryScope` enum answer "should this node advertise another segment as
  locally servable?" — `compute_vram_budget` (the card) or, on a node with no
  card, `inference.max_split_model_memory_mb`. `MemoryScope` exists because
  those two were reached through one `.or()` and describe different memory:
  filtering by graphics residency under the second would disable it on exactly
  the machines it is for. **`ModelProcessPool::model_uses_gpu_memory`** is the
  single answer to "does this model occupy graphics memory", resident from the
  worker's own `placed_on_cpu_because` and otherwise predicted from
  `cpu_reason`, read through `charges_ram` so "sent to the processor", "no card
  detected" and "a build without CUDA" give one answer rather than three.
  **The rule to carry**: a budget over a collection whose members live in
  different places must be told which place each one is in — and a component
  that does not own a resource must not be able to reclaim it.

  **Retirement DRAINS, it does not kill** (2026-08-29).
  `unload_model` waits for the worker's `responses` map to empty
  (`await_responses_drained`, bounded by `WORKER_DRAIN_WAIT`) before sending
  `DaemonMsg::Shutdown`. Every retirement funnels through `unload_model` —
  `free_vram_for_admission` against its victims, `worker_should_return_to_gpu`
  against the processor copy it replaces, the idle timer — so all of them
  inherit it, and a new displacement path gets it for free.
  **Two things worth knowing before touching it.** The "stop admitting" half is
  already done by `workers.remove`, because every forward path re-acquires the
  handle from that map; a request arriving after the remove spawns a fresh
  worker. And **dropping the handle is not what kills the in-flight request** —
  the map holds an `Arc` and the caller holds another, so the child outlives the
  drop. The explicit `Shutdown` is the entire race, which is why one wait in one
  place closes it rather than the cross-cutting change `docs/FUTURE_WORK.md`
  anticipated.
  The bound is not negotiable: a request that never completes must not hold that
  model's memory for the daemon's lifetime, since that would refuse every later
  load on the device — the same trade `WORKER_EXIT_WAIT` already makes, and a
  worse failure than the race. It reports what it stranded. Retirement only ever
  targets idle models, so the common path returns without sleeping at all, and a
  test pins that so no unload pays for a race that is not happening.

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
- **`SharedState::release_request_state`** (2026-08-09) — clears the three maps a
  finished request leaves behind: `active_pipelines`, `active_traces`,
  `request_holder_blacklist`. They are keyed by request id and share one
  lifetime. Five call sites removed all three by hand and the invariant was held
  by three adjacent lines plus a comment asserting it — the shape this codebase
  keeps getting caught by. Dropping one is silent and unbounded: `active_traces`
  is the oracle behind `model_is_in_use`, so a stranded entry refuses to delete
  that model for the rest of the daemon's life, and a stale blacklist entry keeps
  barring a peer that was only meant to be skipped once.
  It deliberately does NOT touch `active_count` or `queue_notify` — those belong
  to the dispatch path that owns the slot, and must move together (§ Inference
  Router Queue). `per_request_state_is_released_in_one_place` fails the build on
  a new direct removal; `TraceGuard` is allowlisted because it registers a trace
  for the split fast path and owns nothing else.

- **`inference::pipeline::remote_generate::StreamReassembler`** (2026-08-09) — the
  single place a remote reply's token stream is put back in order. Each token is
  an independent `request_response` send, so the transport orders nothing between
  them and the terminal token can overtake content still in flight. Emit through
  the reassembler, never straight from the receive loop, and **never treat a
  `finish_reason` as end-of-stream on its own** — that is precisely what
  truncated replies from distant peers (gotcha #282).
  The contract: content tokens carry `token_id` 0,1,2…; the done token carries
  the total sent. An all-zero stream means the peer is too old to sequence, and
  the reassembler degrades to arrival order so a mixed-version network keeps
  working — do not "simplify" that away. Only the consecutive run is released, so
  a lost token truncates rather than silently reordering the reply.
  Any new multi-message exchange should be asked the same question — what happens
  if these arrive backwards. R139's chunked activation forwards already answer it
  (slot table indexed by `chunk_idx` plus a filled count); this path did not.

- **`api::mcp::dispatch::spawn_model_call_task`** (2026-08-10) — the single place
  that decides whether a fan-out model call actually **answered**, as opposed to
  merely not erroring. Every real model call in `compare` / `research` /
  `batch_prompts` passes through it, so it stamps `"empty": true` onto any result
  whose call succeeded with blank text, and `count_answered` downstream only reads
  that flag. Do NOT re-derive blankness by inspecting the collected JSON.
  **The three tools deliberately name the answer field differently** — `content`
  for compare and batch, `response` for research — so a downstream check has to
  know every one of those names and silently mis-reports the moment a fourth tool
  picks a new one. That is not hypothetical: the first cut of this fix did exactly
  that, checked `content` only, and would have flagged every successful research
  answer as blank (gotcha #291). The verdict belongs where the text is, before any
  tool names it. A new fan-out tool inherits the flag with no author action.
  Note `status` is deliberately NOT changed for a blank answer — clients already
  branch on `"ok"`, and reclassifying a success to fix a reporting gap would break
  them.

- **`crate::error::reclassify_flattened_error`** (2026-08-12) — recovers an
  error's CLASS from a message that crossed a boundary carrying no types.
  `SwarmError` survives neither the worker IPC hop nor the network hop; both
  deliver a `String`, and whatever is left is re-wrapped as `Inference` → HTTP
  500. Call it at any such boundary before falling back to `Inference`.
  Two boundaries had the identical problem and only the worker one had a
  remedy (three private helpers in `process_pool.rs`, now folded into this).
  A prompt too long for a peer-held model answered `500 server_error` carrying
  the words "Validation error", while the same request on a local model
  answered `400 invalid_request_error` — so whose fault a mistake was depended
  on which machine held the model (gotcha #304). It also mis-attributed blame:
  `failure_is_penalty_worthy` exempts `Validation` but never saw one, so the
  peer was docked for the caller's mistake. **Matching on prose is #295's trap
  and this is the exception** — the markers are `SwarmError`'s own
  `#[error(...)]` Display prefixes, i.e. part of the type, not wording written
  for a human that gets rewritten. Adding a variant means adding its marker
  here; nothing else may re-derive a class from a message.

- **`crate::error::classify_error`** (2026-08-12) — the single answer to "what is
  this failure, to a caller": `(StatusCode, client-safe message, error type)`.
  `ApiError::into_response` is one caller; the SSE encoders are the others.
  **Never choose an error type at a call site.** It used to live inside
  `into_response`, so streaming could not reach it and both encoders hardcoded
  one: the same over-long prompt was a `400 invalid_request_error` when the
  client did not stream and a `"server_error"` inside a `200` when it did — the
  user's own mistake reported as this server breaking, and monitoring told this
  node has a bug (gotcha #301). Classify where the typed error still exists:
  `StreamFailure::from_error` does it at the site that previously discarded it
  with `e.to_string()`. Do NOT re-derive a type by matching on the message —
  that is #295's substring-matching-prose trap, and the wording is what changes.
  `a_streamed_error_names_the_same_failure_as_its_non_streaming_sibling` fails
  the build on a new literal.

- **`crate::error::failure_log_level` + the `log_failure!` macro** (2026-08-17) —
  the single answer to "how loudly should this failure be recorded in THIS
  node's log". **Never pick `error!` vs `warn!` at a site that logs a
  `SwarmError`.** The level is derived from the status `classify_error` already
  had to choose, because that status IS the answer to whose mistake it was: 4xx
  → Info, `501` → Info, other 5xx → Warn, 500 → Error. A new variant therefore
  inherits a sensible level with no second decision to forget.
  **Why it exists**: an over-long prompt produced three `ERROR` lines when the
  model happened to be peer-held and one `WARN` when it was local — the same
  user mistake at a different severity, decided by which machine held the model
  — and a `501` for embeddings (deliberate, documented, answered with what to
  use instead) logged `ERROR Server error`. `ERROR` means "this node is broken",
  so the product was reporting its users' typos as its own faults (gotcha #316).
  This is the logging-layer survivor of #300-#305: the HTTP surface had already
  been taught to classify, and every site that *logged* still hardcoded a level.
  **`classify_error` is pure and must stay pure** — it used to `tracing::error!`
  from its catch-all, which meant merely *asking* it what level to use emitted
  an ERROR of its own (gotcha #315). The full error is logged by whoever reports
  the failure, from the original `SwarmError` rather than the genericised
  message. Pin that behaviourally by counting emitted events, not by scanning
  source for `tracing::` — the first attempt did the latter and tripped over the
  comment explaining the removal.
- **`crate::error::error_hint_with_key`** (2026-08-17) — returns the actionable
  hint as a stable `(key, english)` pair, from ONE match arm. `error_hint` is a
  thin view over it. The envelope carries `hint_key` beside the unchanged
  English `hint`, and the dashboard looks up `error_hint.<key>`, falling back to
  the English it was sent so nothing can ever render as a raw key name.
  A separate `error_hint_key` function would be a second decision to keep in
  step — this codebase's most-repeated defect — so they cannot drift here.
  Adding a hint means adding its translation in all 21 locales;
  `every_backend_hint_key_has_a_translation` fails the build both ways (a key
  with no entry, and an entry no variant can emit).
- **Credits are DORMANT — nothing may publish or act on a balance** (2026-08-17).
  `MIN_BALANCE_FOR_INFERENCE = 0` and `credit::priority::calculate_tier` returns
  `DORMANT_TIER` regardless of its arguments, so no balance affects who is
  served or how fast; the leaderboard neither ranks by credits nor publishes
  them. The figure is self-minted — no credit has ever moved between nodes as
  payment for work — so acting on it meant rationing the product by a number
  nobody can stand behind. `credits_stay_dormant` in `tests/repo_consistency.rs`
  fails the build if that changes, and it scans the WHOLE of `api/identity.rs`
  rather than the lines that were fixed: the leaderboard's *self* entry is built
  by different code from its peer entries, so the first fix left the node still
  publishing its own (gotcha #317). **It also scans `frontend/js` for code that
  reads a credit figure** (2026-08-30) — the backend half was checked and the
  dashboard half was not, so an unreachable `sortKey === 'credits'` branch
  survived the cleanup that removed every element rendering the balance. It
  sorted on a field the peer payload has not carried for releases, so it read
  `undefined` and nobody noticed; one restored column header would have had the
  peer list ranking by a self-minted number with nothing to catch it. Comments
  are excluded, because one of them documents precisely why nothing renders the
  figure. Design and exit criteria in `docs/CREDITS_DESIGN.md`.
- **`AnthropicSseEvent::Error`** (2026-08-12) — the ONLY way the Anthropic
  streaming surface reports a failure. Emit `event: error`; never write the
  reason into assistant content, and never invent a `stop_reason` for it.
  Before it existed this surface could not say "that went wrong" at all, so each
  path improvised: the router arm reported every failure as `stop_reason:
  "end_turn"` with an empty body (a `PromptPrivacyUnavailable` refusal — the
  thing #295 exists to explain — reached the client as the model choosing to say
  nothing), and the split path wrote `[inference failed: …]` into the message,
  where a client cannot tell it from a real reply and it persists as an
  assistant turn, alongside `stop_reason: "error"`, which the API does not
  define (gotcha #300). Three invariants a new caller must keep: the frame is
  **terminal** (`build_anthropic_sse_response` ends its keepalive ticker on it,
  as it does on `message_stop` — a terminal frame that does not stop the ticker
  hangs the connection); close any open content block first; and translate the
  type through `anthropic_error_type`, because our canonical types are
  OpenAI-flavoured and Anthropic clients match on Anthropic's own set (#302).

- **`SharedState::cfg()`** (2026-08-09) — the live config, and the single answer
  to "what is this setting **now**". `state.config` is the boot-time snapshot:
  correct for what is decided once at startup (listen addresses, data dir, how
  the swarm was built), wrong for anything the Settings panel can change.
  **Reading a user-settable value from `state.config` is the recurring bug this
  ends**: the setting saves, answers `{"status":"ok"}`, shows its new value, and
  the running daemon carries on with the old one. Measured on the released
  v0.3.87 — `max_disk_mb` 50000 → 123456 still reported 50000; contribution →
  Maximum left the storage target at 6250 MB.
  `PUT /api/admin/config` stores the whole updated config here, so a new setting
  is live with no extra wiring. The `OperationalParams` watch channel remains,
  but ONLY to wake subsystems that must *react* rather than re-read — resizing
  the router's concurrency, retiming the auto-manage interval. A value that is
  merely read each tick needs nothing but `cfg()`.
  **Do not add another per-setting mirror.** Four already existed
  (`contribution_auto` from R121, `dashboard_trust_lan`, the two cross-pool
  toggles) because each was bolted on when someone noticed one setting doing
  nothing, which left the next one broken; `OperationalParams` meanwhile carried
  five fields nothing consumed while documenting itself as hot-reloadable.
  `user_settable_config_is_read_live_not_from_the_boot_snapshot` in
  `tests/repo_consistency.rs` fails the build on a new frozen read. It checks
  whole SECTIONS, not field names, because the frozen value is just as often
  reached through a method — `config.resources.shard_upload_mbps(..)` never
  mentions `max_bandwidth_mbps`, and that is how that one survived a first pass.
  **Some things genuinely cannot follow live and the UI must say so** rather than
  implying otherwise: libp2p connection limits are fixed when the swarm is built,
  and CPU thread counts are handed to a worker as it spawns (recycling a live
  worker would drop whatever it is answering). `settings.contribution_restart_note`
  is where that is said.

- **`SharedState::record_peer_serve`** (2026-08-09) — the single answer to "this
  node did inference work for a peer", counting it AND billing for it. Reached
  from exactly two places, the only two inbound paths that serve someone else:
  `dispatch/layer_forward.rs` (one segment of a pipeline) and
  `dispatch/remote_generate.rs` (the whole decode, the fast path).
  **Do not count or bill serving at a call site**, and do not write
  `requests_served_atomic`, `forwards_served_atomic` or `pending_credit_earn`
  anywhere else — `serving_is_counted_and_paid_in_exactly_one_place` in
  `tests/repo_consistency.rs` fails the build if you do.
  **Why it is enforced rather than documented**: the previous helper,
  `track_forward_participation`, had a doc comment saying exactly this and was
  still called by only one of the two paths — the *less* travelled one. The fast
  path is how a machine holding a whole model answers a peer, so in practice
  most serving recorded nothing and earned nothing while the requester was still
  debited (gotcha #279).
  **The converse is equally load-bearing**: work the node does for ITSELF must
  not come through here. The router's completion hook and the local-segment path
  both used to bump these counters, and `pipeline/distributed.rs` used to credit
  the node for its own segment, so a user whose only traffic was their own chat
  was told they had served the swarm and was paid for it. The product promises
  "earn credits by serving inference for others" and "inference across your own
  devices is free"; both directions have to hold for that to be true.
  Note that `release_escrow` transfers nothing to `to_node` despite recording it,
  and `credit::transaction::create_transaction` has no production callers — so
  this accumulator is the ONLY way a serving node is ever paid (gotcha #280).

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

- **Vendored `GgmlType::vec_dot_rows` + the row-blocked tiled matmul** (2026-08-21
  night) — `vendor/candle/candle-core/src/quantized/{k_quants,avx}.rs`. One weight
  column against `rows` activation rows in one call; the AVX2 Q4_K and Q6_K kernels
  (`dot_q4k_q8k_rows::<R>`, `dot_q6k_q8k_rows::<R>`) unpack the column once and
  share it across R rows. **Overrides MUST stay bit-identical to the per-row loop**
  (`vec_dot_rows_generic`): each row keeps its own accumulators and sees the
  single-row kernel's operations in the same order; `examples/qmatmul_bench`
  asserts exact equality against the upstream ordering for Q4_K and Q6_K at every
  m it prices — run it after touching either kernel. **R is a register-pressure
  knob, not a "bigger is better" one**: Q4_K at R=8 spilled the 16 ymm registers and
  R=4 was 1.2x faster at every m. `matmul` also runs the column-outer loop per
  `ROW_BLOCK = 128` rows so the quantized activations stay in L2 — a whole-prompt
  forward had streamed ~3 MB from L3 per column, which is why a per-row cost measured
  at m=128 never carried to large m. The `examples/prefill_bench` single-forward
  number is only representative of the production 128-token chunks because of this.
- **`inference::decode_attn::gqa_decode_attention_cpu`** (2026-08-21 night) — single-
  position attention straight over the KV cache in its stored `[b, kvh, S, d]` layout,
  one rayon task per (batch, kv head). Dispatched at the top of
  `standard_attention` for `q_len == 1` on the CPU (MHA and GQA). The two batched
  matmuls it replaces cost 1.3 ms/layer at ~920 KV for ~11 MFLOP — GEMM packing and
  dispatch, a quarter of every decoded token. **Returns `Ok(None)` for anything
  outside its scope** (non-CPU, non-f32, `q_len > 1`, K/V whose `(S, d)` plane is not
  dense, a mask it cannot reduce to one row) and the caller carries on unchanged —
  it is an accelerator, never a requirement; keep it that way. `SWARMLLM_DECODE_ATTN=
  standard` disables it (same discipline as `SWARMLLM_FORCE_STANDARD_ATTN`); that is
  how its +24% decode was attributed (A/B/A/B in one binary). Not bit-identical to
  the matmul path (different summation order); `decode_kernel_matches_the_matmul_
  path` bounds it (abs < 1e-5, rel < 1e-4 with a 0.05 floor — the first metric
  flagged fp32 noise on a near-zero output as a failure). The DRAM floor for the
  cache read at ~900 KV × 28 layers is ~7 ms/token on this box; the kernel sits at
  ~15 — the remainder is per-layer dispatch, not arithmetic.
- **`inference::fast_math`** (2026-08-21 night) — eight-lane AVX2 `expf`
  (`exp_inplace`, Cephes polynomial, ~2 ulp vs libm, pinned by
  `vectorised_exp_tracks_libm` over [-80, 80]) and the fused `silu_mul` CustomOp2.
  Every `exp` on the CPU path had been a scalar libm call — ~540 M in the softmax
  rows and ~205 M in SiLU for an 896-token llama-3.2-3b prompt. **Used by the fused
  softmax (`attn_softmax::softmax_row`), the three SiLU×up call sites in
  `layers/mod.rs`, and the decode kernel.** Not bit-identical; every consumer keeps
  its tolerance test against the composed candle reference (softmax 1e-6 rel, silu
  2e-6). A new elementwise pass that calls `f32::exp` in a loop is the thing to
  route through here instead. Inputs below ~-87.3 underflow to 0 (libm: denormal),
  above 88.37 saturate — right for softmax (shifted ≤ 0) and SiLU (limits).
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
- **`inference::split::kv_cache::LayerKv`** (2026-08-10) — one layer's KV cache:
  the f32 BHSD cache every path reads, plus an optional f16 BSHD mirror for the
  CUDA flash kernel. **Never touch the inner `KvCache` directly.** `append` and
  `reset` are INHERENT methods and so take priority over the `Deref`, which is
  what stops an existing call site reaching the inner versions and leaving the
  mirror behind; `KvCacheStore::truncate_to` (the speculative-decode path) got
  correct behaviour for free from that, since it truncates via reset+append.
  **Why a mirror rather than replacing the f32 cache**: rounding to f16 moves
  from every-read to once-at-write, and since the f32 source is never itself
  overwritten the flash kernel receives bitwise the same numbers — so the flash
  path is numerically unchanged, not merely close, while `standard_attention`
  keeps full precision. Published results on f16 KV divergence (arXiv 2604.15409)
  are worst under long context and GQA, which is exactly our case, so the f32
  copy stays.
  **Three things a new caller must respect.** (1) The mirror is GQA-only —
  `layers::model_wants_kv_mirror` gates it, because MHA decode reads the f32
  cache and an unread mirror cost 3-8% per token plus 50% more KV memory
  (measured on phi-3.5). (2) `set_mirror_wanted(true)` is deliberately INERT: a
  mirror started against a cache that already holds positions can never catch up
  and would be refused forever by the length guard while still costing memory.
  (3) The mirror is real VRAM and `kv_budget::kv_bytes_per_token` must charge for
  it — omitting it let a model be admitted and then OOM instead of returning the
  503 that reroutes to a peer.
  Worth 1.41x on GQA decode at ~2064 KV; the win is long-context only (~1.04x at
  256), which is what an O(history) cost predicts.
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
- **`inference::layers::standard_attention` grouped GQA decode** (c4cc3b16,
  2026-08-16) — for `q_len == 1` with `n_kv_head < n_head`, standard attention
  no longer expands the KV cache with `repeat_kv`; it reshapes the query heads
  into extra matmul rows against the unexpanded cache
  (`grouped_gqa_decode_attention`). Identical arithmetic — the reshape is valid
  ONLY because `repeat_kv` numbers heads group-major (query head `h` belongs to
  group `h / n_rep`); get that backwards and every head reads another group's
  cache while still producing plausible logits, which is why
  `grouping_query_heads_matches_expanding_kv_heads` compares against the
  expanded path rather than asserting shapes. MHA was pinned byte-identical
  (`mha_decode_matches_the_plain_path` — now within 1e-5, since the decode kernel
  serves MHA decode too, 2026-08-22). This flipped the CPU decode routing: GQA decode
  had been sent to the fused kernel precisely because of the `repeat_kv` cost,
  and with it gone the same benchmark reports the opposite at every length
  (3-9x) — so **all CPU decode now takes standard**, with the control run
  reproducing the old verdict on the reverted code. 1.41x end-to-end CPU decode
  on llama-3.2-3b; 4h-soak-validated (`soak_0816_cpu_speedup.md`).
- **`inference::layers::cuda_decode_prefers_standard`** (2026-08-08) — on CUDA:
  MHA decode takes standard, GQA decode takes flash **at every context length**;
  prefill always flash. The GQA side rested on the same reason the CPU rule did —
  `standard_attention` rebuilt the `repeat_kv` expansion every token — and that
  premise changed with the grouped path above, so it is a re-measure candidate
  (`docs/FUTURE_WORK.md`); it stands unchanged because GPUs already route GQA
  decode to a fused kernel and this box cannot resolve a small GPU delta (#267).
  The MHA side is not premise-dependent: flash has no split-KV kernel, one query
  row cannot fill the card, up to 25x per call.
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
- **`inference::mem_bandwidth::measured_gbps`** (2026-08-18) — what this machine's
  memory actually delivers, measured once and cached. **The figure a processor-only
  node advertises as its speed.** It was `estimate_tokens_per_sec_7b(50.0, false)` —
  a hardcoded bandwidth for every machine — so every CPU node in the swarm quoted
  the identical 1.70 tok/s whether it was an eight-channel server or a fanless
  mini-PC. Nothing could tell two of them apart, which is why a delegation gate
  comparing them would have been comparing a constant with itself. Measured 29.9
  GB/s on the 5800H laptop this was written on, against the 50 assumed.
  Buffer must exceed any last-level cache (256 MB) or it reports cache bandwidth;
  min-of-3 because every error source is additive; reads at decode width, not
  thread-per-core, so it ranks machines the way running a model does. Costs 254 ms
  once, on the health-monitor task rather than the startup path.
  **Adding a device class means giving it a real measurement, not a constant.**

- **`NodeCapability.cpu`** (2026-08-18) — a processor described the way a graphics
  card always has been. `GpuInfo` has existed since the beginning; the CPU had no
  representation, so a peer without a card rendered as the bare word "CPU" and every
  such machine looked identical. Additive and `#[serde(default)]`, per the
  additive-protocol rule — verified in BOTH directions against the released
  v0.3.101 binary: an older node ignores the new field with no deserialisation
  failure, and a newer node reads `cpu: None` from an older one and falls back to
  the old label. **A new capability field is not done until that pair has been run**;
  the swarm is always mixed-version during a rollout.
  Deliberately carries no more than the GPU already does — the `os` field's refusal
  to send a build string is about identifying the INSTALL, whereas a processor model
  identifies the hardware doing the work, which is what a peer needs to judge.

- **`inference::scheduler::delegation_target`** (2026-08-18) — the single decision
  to hand a WHOLE model to a peer rather than run it on this node's CPU. Fires only
  when `ModelProcessPool::is_cpu_bound_for_lack_of_vram` says we have a working GPU
  this model does not fit, and only for a peer that holds every layer, is directly
  reachable within `DELEGATE_MAX_LATENCY_MS`, is trusted at least as much as an
  ordinary peer, and advertises GPU room with margin.
  **It returns a peer or nothing, and never falls through to the routing search.**
  That is the whole difference from the version reverted in `cbbed678`: that one
  priced a full local node at 10,000/layer and let the DP decide, which made local
  layers unusable, priced out the good split (some layers here, rest on a peer 5 ms
  away) and picked a node in another country. Both outcomes here are a single
  segment. Do NOT reintroduce a penalty term — it distorts every other route.
  **Prompt privacy changes the SHAPE, it does not disqualify the peer.** With privacy
  off the peer gets the whole model. With privacy on, `boomerang_assignment` keeps
  layer 0 and the final layer local — the embedding and the sampling, which is what
  the guarantee actually is — and gives the peer everything between, as encrypted
  activations. Since `encrypted_pipeline_auto` is on by default for any model whose
  ends this node holds, that is the COMMON path, not an edge case: treating privacy
  as a veto stranded the default configuration on its CPU for no privacy gain.
  **The boomerang is constructed, not searched, for the same reason.** Asked to route
  it, the general search answers "all of it locally" — that satisfies the encrypted
  constraint at zero network cost and nothing in its cost model knows this node is
  about to fall back to its CPU. Verified 2026-08-18: merely standing the fast path
  aside produced `segments=1 node=<local> layer_start=0 layer_end=28`. Teaching the
  search that local compute is expensive here is what `cbbed678` did, and it
  distorted every other route.
  Three things made it inert until it was run on real machines, all now fixed and
  all worth knowing before touching this: gotcha #329 (`would_fit_on_gpu` said yes
  for a model resident on the CPU), #330 (every node gossips zero free VRAM), #331
  (the latency bound was calibrated against network intuition, not against what
  `peer_registry.latency_ms` actually measures).

- **`inference::router::distributed_exec::failure_is_penalty_worthy`**
  (R146) — gates `penalty_serve_failure` on (a) the assignment actually
  having had a remote segment and (b) the error not being locally
  attributable. Any new automatic credit or reputation penalty MUST route
  through an equivalent attribution check. `ServiceUnavailable` means
  "THIS server can't serve" and `Internal` means our own bug — neither can
  ever justify charging a peer.
- **`ModelRegistry::manifests_to_gossip`** (2026-08-11) — the single answer to
  "which manifests should this node re-broadcast?": ones it published **and ones
  it holds a shard of**. Both the one-shot startup announcement
  (`daemon/background.rs`) and the 30s periodic broadcast
  (`health/monitor.rs::broadcast_manifests`) go through it.
  **Never filter on `publisher` alone.** Doing so broke model discovery
  swarm-wide: every holder used to rewrite `publisher` to itself at startup to
  earn broadcast rights, and `register_manifest` overwrites unconditionally, so
  holders erased each other's claim until none of them broadcast. Since there is
  **no on-demand manifest fetch**, a node that joined later could never learn a
  model in full — `all_shards_available` stayed false and every request answered
  "No model loaded" while the dashboard listed the model as available. Measured:
  `phi-3.5-mini` registered 81 times under 50 distinct publishers (gotcha #296).
  The correct predicate already existed in the startup path and was missing from
  the timer, so discovery worked only for peers connected during someone's boot.
  Holding a shard is the honest signal, which is why the gossip handler
  deliberately does NOT require `sender == publisher`. `publisher` means who
  published it — do not reintroduce a self-claim to grant broadcast rights.
- **`model::manifest::merge_known_shard_hashes`** (2026-08-24) — the rule that a
  shard hash may go from unknown to known but never back. Called from
  `ModelRegistry::register_manifest`, the single funnel every adoption path uses
  (gossip ingress, DB reload, disk scan, acquisition), so no caller can skip it.
  **Why**: a shard's BLAKE3 hash is a property of the MODEL, but a manifest is
  built from what its author holds on disk — `build_shard_infos_from_layouts`
  hashes a shard file only when it exists and writes all-zero otherwise. So every
  partial holder publishes real hashes for its own shards and placeholders for
  the rest, and the registry's blind `insert` let a placeholder destroy a hash we
  already had. That matters because `network/manager/requests.rs` verifies a
  completed P2P transfer ONLY when the manifest carries a non-zero hash: lose the
  hash and the bytes are taken on trust, recorded as held, and re-served to other
  peers unchecked. Measured on the live node — five shards fetched against a
  manifest carrying placeholders for exactly those five, one corrupt (gotcha
  #381).
  Three things a change here must keep. The merge is **one-directional**: a real
  incoming hash still replaces a real stored one (a genuine re-publish), and only
  unknown is treated as no information — so this cannot be used to pin a stale
  hash. `manifest_hash` is **recomputed** when anything was recovered, because the
  stored manifest is then a local composite rather than what the publisher sent
  and `load_from_dir` re-derives that hash to validate a saved copy; recomputing
  also keeps the changed-detection quiet, since the merge is deterministic and an
  unchanged re-gossip lands on the same bytes. The accept path is the consumer
  that matters: `classify_p2p_shard_acceptance` (same module) turns "do we have a
  hash?" into a three-way policy rather than a yes/no gate — verify against the
  hash; or, with no hash but a reachable origin, discard the peer's copy and
  fetch that shard from the ORIGIN; or accept-unchecked only when neither is
  possible. Enforcing verification unconditionally was shipped once and
  soak-caught, which is why the third case survives — but it is now *reported* as
  unchecked rather than counted as verified. The second case is self-limiting,
  not a retreat from P2P: the origin download hashes what it writes, so the
  manifest gains the real hash and this merge then spreads it by gossip.
  Three things it must keep. The peer is NOT penalised — it may have served
  perfect bytes, and "cannot tell" is neither fine nor the peer's fault. The
  fetch must actually **happen** before a copy is discarded, which is why
  `AutoShardManager::complete_pending_origin_fetches` sits OUTSIDE the
  `auto_manage.enabled` gate — the same distinction already drawn for
  `try_idle_vram_unload`: that switch means "do not decide what to fetch on my
  behalf", not "abandon a shard this node already asked for". **Never throw away
  data you cannot replace.** And a recovered hash is PERSISTED — via
  `ModelRegistry::set_persist_hook`, installed at startup with a `Weak`, in the
  same shape as `set_ram_budget_provider` — because `load_from_db` is what
  repopulates the registry at boot and the disk copy is the thing carrying
  placeholders. Persisting is gated on the merge having changed something, or a
  peer re-gossiping placeholders writes to the DB every 30s for every model.
  The sibling half is `daemon/background.rs`: a shard with no hash is counted as
  `unchecked`, never as `verified`. It had been counted as verified, so the sweep
  reported "all shards OK verified=21" over five shards it had never hashed. **A
  check that cannot run must be reported, not rounded up into the success line.**

- **`types::slugify_model_name`** (2026-08-15) — the single derivation of a model
  id from a human display name. It is what
  `daemon::manifest::generate_and_register_local_manifest` registers, persists
  and gossips, so resolving a name a user typed, building the model's directory
  path, and announcing which models this node hosts must all arrive at the same
  string or they are looking for a model nobody published.
  There were **three** derivations and no two agreed. Two were near-identical
  slugifiers differing on any character that is neither alphanumeric nor `-`/`.`
  — one DELETED it, the other REPLACED it with `-` — so `Model (Q4_K_M)`
  registered as `model-q4-k-m` and resolved as `model-q4km`; quant suffixes carry
  underscores, so that is an ordinary GGUF name. The third was no derivation at
  all: `health::monitor`'s capability announcement sent the RAW display name, so
  a node that loaded a model with `-m` advertised holdings under an id no peer
  could match to a manifest — **invisible as a holder of a model it was sitting
  on**, and a phantom `shard_count: 0` entry in every peer's list (gotcha #310).
  **The replace-and-collapse semantics are canonical because they made the ids
  already on disk and in the DHT.** `the_shared_helper_still_produces_the_ids_
  already_on_disk` reproduces the old manifest algorithm verbatim and asserts
  agreement, so changing the semantics renames every user's models and goes red.
  A new surface that turns a name into an id calls this; it must never grow a
  second copy, and a raw display name is never a `ModelId`.
- **`inference::split::token_embedding::rows_on_demand_eligible`** (2026-08-18) — the
  single answer to "is this model's `token_embd.weight` held quantized with its rows
  dequantized on lookup, or dequantized whole at load?". **Two places must agree**: the
  loader, which allocates, and the footprint estimators, which decide whether the model
  is admitted at all. A disagreement is invisible until a node either refuses a model
  that would have fitted or is admitted and then runs out of memory — the same trap
  `EMBEDDING_DTYPE` already carries a test for. `table_supports_row_gather` is the
  device-independent half, for the estimators, which are built once and consulted for
  both a CPU and a CUDA worker; the `SWARMLLM_DENSE_EMBEDDING` override lives in THAT
  inner predicate so both callers inherit it, because putting it one level up left the
  estimator pricing a gather the loader was not doing.
  **The gather must stay on the device holding the table.** `QTensor::data()` is a
  zero-copy borrow on CPU and a full device-to-host copy on CUDA, so the CPU
  implementation reused on a GPU would move the whole table across PCIe every decode
  step — llama.cpp measured that shape at 6.18 ms/token against 1.72 before
  `k_get_rows_kq`. Both devices therefore go through the vendored
  `QTensor::gather_rows`: CPU slices rows out of the borrow, CUDA runs `index_select`
  over a `[vocab, row_bytes]` byte view (no new kernel — `is_u32_u8` is already in
  `candle-kernels`, and the quantized buffer's padding is only ever trailing, so rows
  are contiguous). Metal has none and keeps the dense table.
  Measured 754 MB on CPU and 736 MB on an RTX 3070, both llama-3.2-3b against a 751 MB
  prediction. Weight-tied models gain most because the loader used to load that tensor
  TWICE — once dequantized for the lookup, once quantized for the LM head — and now
  shares one `Arc<QTensor>` via `QMatMul::from_arc`.
  **Verify a change here with DECODE RATE, not memory**: the failure mode above frees
  exactly as much memory while being far slower. The check that rules it out is
  PREFILL — gathering 512 rows costs no more than gathering 1 would if each row made a
  host trip, so unchanged prefill is positive evidence the gather stayed on-device.
  A new embedding path goes through `TokenEmbedding`, whose two variants both return
  `EMBEDDING_DTYPE` so no call site can tell them apart.

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
- **`inference::split::read_gguf_header`** (2026-08-29) — the single way to parse
  a GGUF header off a PATH, and the buffering is the entire reason it exists.
  `gguf_file::Content::read` walks the metadata with many tiny reads — for every
  string a length, then its bytes — so handing it a bare `std::fs::File` turns
  each one into a syscall. A 7.8 MB header carrying a 128k-token vocabulary and
  280k merges is roughly 820k of them.
  **Measured on the live node** (gotcha #410): `GET /api/admin/models` took a
  stable 11.2 s, of which **9.6 s was KERNEL time** — it parses every local
  model's header and seven call sites were passing an unbuffered handle.
  Optimisation cannot touch syscall count, which is why the release binary was
  no faster than a debug one on that path. Direct A/B on one header: 980 ms
  unbuffered against 98 ms buffered.
  **Two sites deliberately do NOT use it** — `local_embedder` and `vision` keep
  their own `BufReader`, because the same handle goes on to read tensors and the
  helper's handle dies with it. They still buffer; that is the invariant, not the
  helper. A `Cursor` over a slice or an mmap already holds the bytes and needs
  neither.
  **`ModelProcessPool::footprint_inputs` also parsed the same file twice** — once
  through `GgufTokenizerMeta::from_gguf_file`, which materialises the whole
  vocabulary and merge list as owned `String`s, purely to reach `vocab.len()` as
  a fallback, and once through its own reader for everything else. The count was
  already in the parsed `Content`; counting the array borrows it.
  `a_gguf_header_is_never_parsed_straight_off_an_unbuffered_file` in
  `tests/repo_consistency.rs` fails the build on a new unbuffered site. It checks
  PROXIMITY — a `File::open` within a few lines of a `Content::read` with no
  `BufReader` or `Cursor` between them — not naming. A naming rule was the first
  cut and it silently missed the `match File::open { Ok(mut f) => Content::read(
  &mut f)` form, which is the shape one of the seven sites actually had.
  **The general rule**: before theorising about why something is slow, split
  user from system time (`utime`/`stime` in `/proc/<pid>/stat`). It is two
  numbers and it partitions the hypothesis space in one step — kernel-dominated
  means syscalls or waiting, and no amount of reading the code distinguishes
  "parses a lot of metadata" from "makes 820k read calls". A **stable** duration
  is a fixed amount of work, not contention; go and find the count.
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
- **`ModelRegistry::describes_a_different_build`** (2026-08-29) — is this
  manifest the same FILE as ours, or another build wearing the same name? A
  model id comes from a display name (`slugify_model_name`), so every
  independent GGUF build of one model collapses into one identity: three Q4_K_M
  builds of Qwen2.5-Coder-7B-Instruct were live on the swarm at once —
  4,683,073,536 / 4,683,074,144 / 4,683,074,336 bytes — sharing not one shard
  hash (gotcha #406).
  **Compares SHAPE, never hashes**: a manifest carries a real `size_bytes` for
  every shard whether or not its author holds it, but a hash only for the ones
  it does, so every partial holder would read as a different build under a hash
  comparison. **Gated on `has_origin_knowledge`** — our own origin download is
  the only evidence that is not just another node's assertion, the same
  adjudicator `origin_verified` uses — so a genuine re-publish (new bytes, new
  shape, no origin copy of ours to weigh against it) still lands.
  **Why it matters**: adoption is `manifests.insert`, last-writer-wins, so
  before this the other build's `shard_count`, `total_size_bytes` and per-shard
  sizes overwrote ours while `origin_verified` kept our hashes — a manifest
  describing one file and authenticating another, and `size_bytes` is what
  decides byte-range requests.
  **Still open**: `record_shard_holder` keys on `ShardId` alone, so holders of
  different builds are pooled and the scheduler will route to either. Bounded —
  verification catches it, so it costs a wasted transfer, never a wrong answer.
  Closing it needs a build discriminator on `ShardAnnounce`; see
  `docs/FUTURE_WORK.md`.

  **Both rejection messages share ONE rate limiter**
  (`manifest::note_manifest_rejection` over `manifest::RejectionKey`), because
  they are the same event at two granularities and a peer re-gossips on a timer
  for as long as it is up. The manifest arm was rate-limited on 2026-08-26
  (4709 WARN lines, 14% of a month's warnings); **the per-shard arm was missed,
  and it is the worse of the two — it fires once per SHARD**, so one publisher
  with an 8-shard model on a 30 s cadence produced 16-28 lines a minute
  indefinitely, ~10% of the whole log, measured live 2026-08-29. The key
  distinguishes the two kinds so neither can silence the other, and each shard
  is its own key so eight genuine disagreements still get eight lines. A new
  "we are ignoring what this peer keeps telling us" warning belongs behind this
  limiter, not beside it: **anything a peer repeats on a timer will be repeated
  at you for ever, so the first question about such a log line is what silences
  it.**
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
#179 had fixed the classifier but inbound never had an address to classify).

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

**The trap.** libp2p's `identify` protocol is `/ipfs/id/1.0.0` — universal to
every libp2p node on the internet, IPFS and every other rust-libp2p project
included. `handle_identify_received`
registered a peer on the strength of it alone: minting a `NodeId` from the
peer's Ed25519 key, establishing an encryption session, and counting it in
`connected_node_ids`, the number the dashboard shows. The field that carries
the peer's own statement of what it speaks — `info.protocols` — was never read
anywhere in the codebase.

Measured on the live swarm 2026-08-25/26 (gotcha #396): five foreign nodes on
Linode, port 4001, relaying through one another, appeared in every node's peer
list with no version, no shards and no latency but `healthy=true`. They
identify as `openhydra/0.1.0` on `rust-libp2p/0.45.0` — port 4001 is the
libp2p convention and identifies nobody, so reading IPFS into it was a guess;
the rejection log line reports `agent_version` precisely so the next one does
not have to be guessed at. Each one's
first real SwarmLLM request answered `OutboundFailure … The remote supports
none of the requested protocols` — the peer saying so in our own log, one
second after we adopted it. They arrive via **PEX**, which dials an address
without asking whose it is, so it spreads: a brand-new node with an empty data
directory acquired all five within 90 seconds.

**Both halves are load-bearing.** Declining to register is not sufficient:
`handle_connection_closed` re-dials any peer with no registry entry, on the
assumption it disconnected before Identify ran. So skipping registration alone
trades a wrong peer-list entry for an endless dial loop — worse, and silent.
`NetworkManager::foreign_peers` (bounded by `MAX_FOREIGN_PEERS`) records the
verdict, and BOTH that re-dial branch and the PEX dial loop consult it. A new
path that dials a peer learned from an untrusted source must do the same.

**Safe to gate** because `/swarmllm/id/1.0.0` has been the identify
`protocol_version` since the first P2P commit, so no released node fails it;
the protocol-list arm is the belt-and-braces for a future build that changes it.
Match the namespace as a PREFIX, never a substring — the peer controls those
strings.

**Declining to register is still not sufficient, and the second half needed a
third.** Not re-dialling was covered by `foreign_peers`, and every one of the
dial sites now routes through `NetworkManager::dial_checked`
(`every_dial_goes_through_the_foreign_peer_gate` in `tests/repo_consistency.rs`
keeps it that way) — **though "every" was wrong when this was written: the test
matched `self.swarm.dial(` and `discovery::bootstrap_peers` was a free function
taking `&mut Swarm`, so the one site that mattered was invisible to it for
another release (gotcha #405). The pattern is now both forms, comment lines
excluded, with the loopback probe named as the single exception.** And the node
*still* opened 5-6 connections per foreign peer
in seven minutes, each closed 43 ms later by this gate and then re-established.
`dial_checked` refused none of them across two runs, which is the measurement
that matters: **the dials come from inside libp2p, not from us.** Which behaviour
was never pinned down and no longer needs to be — `SwarmBehaviour::blocked_peers`
(`libp2p::allow_block_list`) refuses both directions at the swarm level, and
`handle_identify_received` blocks the peer before disconnecting it. Measured
5/6/6 connections → **1/1/1**, one unavoidable first contact each, healthy peers
unaffected.

**The general rule**: completing a handshake that everyone speaks proves nothing
about who you are talking to. Before treating a successful negotiation as
identity, ask which population could also complete it.

## One dial per PEER, never one per address

**`NetworkManager::dial_bootstrap_peers` + `discovery::plan_bootstrap_dials`**
are how bootstrap and cached addresses are dialled. Group by target peer, one
`DialOpts::peer_id(..).addresses(all)` each, through `dial_checked`, with
`PeerCondition::DisconnectedAndNotDialing`.

**Why per-address dialling is wrong, not merely wasteful.** A bare
`swarm.dial(addr)` carries no `PeerCondition`, so libp2p's per-peer dedup cannot
see it, and it does not reach `dial_checked`, so the foreign-peer gate does not
apply either. `discovery::bootstrap_peers` did exactly that, once per address,
on every discovery tick. A peer cached at two addresses (TCP + QUIC) therefore
got two simultaneous dials whenever it was momentarily disconnected; with
request_response's own dial that reaches `max_connections_per_peer = 3`.
The vendored rr layer then spreads sends across all three (ranked, but ranked
among connections that should not all exist), and one that has quietly died
swallows its share until the 8-failure rule closes the peer entirely. Measured
paired against an unpatched node, same swarm, same 43 minutes: **13 connection
establishments to one peer against 3** (gotcha #405).

**It costs nothing.** `libp2p_swarm::connection::pool::concurrent_dial` is a
`FuturesUnordered` — one dial attempt RACES every address it was given, bounded
by `dial_concurrency_factor`, and yields exactly one connection. Handing one
attempt every address is the same parallelism; it just stops keeping the losers.
Do not "restore" per-address dialling for latency.

Three rules a new dial site must follow:

- **Read the peer from the LAST `/p2p/` component** (`target_peer_from_address`).
  A relay circuit is `…/p2p/<relay>/p2p-circuit/p2p/<target>`; the first names
  the RELAY, so every gate then asks about the wrong node and the dial asks
  libp2p to reach the relay at an address belonging to someone else. This was
  wrong in two independent places.
- **Do not weaken the condition.** `PeerCondition::Disconnected` asks only
  whether a connection is ESTABLISHED, so two dials issued before either
  completes both pass. `DisconnectedAndNotDialing` is libp2p's own default and
  our code had explicitly opted out of it at three sites. **The mDNS site keeps
  the weaker condition deliberately** — a LAN rediscovery must be able to
  override a stale bootstrap attempt; that one has a documented reason and the
  WAN paths did not.
- **Never dial yourself.** `dial_checked` refuses it for every source. A third
  party hands your own address back — PEX relays whatever its registry holds,
  and a circuit terminating at you names you in its last hop — and the
  sender-side self-filter cannot help, because the sender is someone else.
  libp2p refuses these, so nothing broke; it was 223 wasted dials in 45 minutes
  that nothing surfaced.

**Dial attribution is logged at DEBUG** (`site` field in `dial_checked`), not
TRACE. Two investigations have turned on "was that dial ours?" and both had to
rebuild to answer it; the second only found the self-dialling because the
instrument was finally there to say so.

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

**`merge_dht_providers` is the one writer of `shard_holders` that cannot remove
a holder** — it loops `record_shard_holder` over a DHT `GetProviders` result.
That matters because a provider record outlives the fact it asserts: libp2p-kad
keeps one for 24 h, republishes at 12 h, and other peers serve it, so a node that
deleted or lost a shard is still advertised as holding it for hours. An
add-only writer wins every disagreement with a writer that removes, and its
cadence decides how fast.

Measured live 2026-08-22 on a three-way split (gotcha #364): the holder
retracted shard 2 correctly and re-announced its reduced holding every 5 minutes
(`Peer retracted shards it no longer hosts … dropped=1`, six times), the
coordinator re-merged the stale DHT record every few seconds, and every request
was then scheduled onto a node without those weights — `503 Segment failover
exhausted`, indefinitely, while a healthy node holding exactly that shard was
never considered. **Retraction alone is futile when something re-adds the claim
faster than it is withdrawn** (same shape as #163).

So `ModelRegistry::retracted_claims` records what a holder has withdrawn, and
`merge_dht_providers` skips a (shard, node) pair found there. Two halves that
must stay together: `record_shard_holder` CLEARS the entry, so a node that
genuinely re-acquires the shard is believed the moment it announces that itself;
the DHT path must NEVER clear it, which is the entire point. Honoured for
`RETRACTION_HONOURED_SECS` (26 h — deliberately longer than the provider
record's own life, or the record simply wins again at the end of the window).

A new writer of `shard_holders` fed by anything other than the holder's own word
must answer the same question first: can it remove, and if not, what stops it
resurrecting something already withdrawn?

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

**Why it exists.** `GET /api/admin/diagnostics` has been the documented thing to
attach to a bug report for many releases; the dashboard has a one-click **Copy
diagnostics** button whose whole point is that a non-engineer does not have to
read what it copied; and its hint said *"No keys or invite codes are included"*,
which reads as *safe to post*. It also copied this machine's own addresses and
the first ten entries of the peer cache — on a live node, **other people's home
IP addresses**, verified against the real swarm (gotcha #426). A comment in
`reference-models.js` asserted the daemon had "already redacted" it, which is
the #419 shape: a stale assurance reads as verification and stops the next
person checking.

Four properties a change here must keep.

- **One pass over the finished text, not a rule per section.** Addresses reach
  the report from at least four places — the listen-address list, the peer
  cache, relay circuits, and the prose of whatever a failed dial produced — so a
  per-section rule is the "One invariant, N paths" trap below. A section added
  later inherits the redaction with no author action, pinned by a test that
  plants an address in free prose.
- **Keep the KIND, replace only the host.** Transport, port, peer id and the
  `/p2p-circuit` structure all survive, so "this node has no public address" and
  "that hop is relayed" stay legible. Deleting the lines would make `?full=1`
  the form people paste instead, which is worse than not redacting.
- **Salt the tag per report.** Two entries for one host share a tag, which is
  what keeps "ten cache entries, all one machine" readable. **IPv4 is 2^32**, so
  an unsalted digest is a reversible encoding of the thing being hidden,
  recovered by enumeration in seconds.
- **Exempt what identifies nobody.** Loopback and the unspecified address, and
  the project's own bootstrap anchor — the anchor ships in every binary, so
  hiding it protects no one and costs the most useful reading in the report. The
  exemption is derived from `default_bootstrap_peers()`, never restated, so
  moving the anchor moves the exemption.

**Peer ids and node ids are deliberately NOT redacted.** They are the swarm's
public identities, they appear throughout the rest of the report, and they are
not a coordinate anyone can dial. The address is.

**A query flag must have no invalid spelling.** `DiagnosticsQuery::full` is a
`String`, not a `bool`, because serde deserializes a `bool` from a query string
only for the literal words `true` and `false` — so `?full=1`, the form this
project's own README, `docs/DIAGNOSTICS.md` and `two_node_test.sh` all use, was
refused by the extractor with a bare-text 400 that never reached the JSON error
envelope (§ "API errors must be readable by the caller"). `query_flag_is_on` is
the shared predicate.

**The general rule.** Before writing that something is safe to share, enumerate
what is actually in it — and treat any surface built for a non-technical user to
hand to a stranger as a publishing surface, not a debug dump.

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

## Attention kernel choice and the query-length cliff (2026-08-23)

Four helpers now own decisions that used to be spread across call sites. All
four exist because a predicate that *reads* obviously correct was answering a
different question than the one that mattered.

- **`inference::layers::flash_handles_offset_causal`** — may flash attention take
  a query block that lands on a warm prefix (`k_len > q_len && q_len > 1`)? Yes,
  and it is not a judgement call: the vendored kernel's
  `col_idx_limit_right = row_idx + 1 + max_seqlen_k - max_seqlen_q`
  (`vendor/candle-flash-attn/kernels/mask.h`) is bottom-right aligned causal,
  the same predicate `SplitExecutor::causal_mask` builds by hand. The dispatch
  used to divert that shape to `standard_attention` on the stated grounds that
  flash could not express the mask; since `prefill_chunk_tokens` is a CEILING
  that always applies, that meant EVERY prompt chunk after the first took the
  slower kernel on every GQA model (gotcha #368). A benchmark that prefills in
  one call cannot see it. `SWARMLLM_FLASH_OFFSET_CAUSAL=0` restores the old
  behaviour for A/B.

- **`inference::layers::cuda_decode_prefers_standard`** — now `q_len == 1` for
  EVERY head geometry, not just MHA. The GQA exclusion existed because
  `standard_attention` rebuilt the `repeat_kv` expansion every token;
  `grouped_gqa_decode_attention` deleted that work in August and the rule
  outlived its premise by a week. Re-measured: standard wins at every context
  length and the margin grows with it. **The isolated table overstates the
  penalty** — its flash arm runs without the f16 KV mirror that production
  always has — so end to end this is ~6% at 4120 KV and unresolvable at ~900.
  Both numbers are recorded at the benchmark; the forward is the one that
  describes a reply (#266). `SWARMLLM_GQA_DECODE_FLASH=1` restores the old rule.

- **`inference::layers::grouping_applies` + `grouped_gqa_attention`** — read the
  KV cache at its stored width instead of expanding it, for ANY query length
  whose score matrix fits in one pass (not just `q_len == 1`). Blocked calls keep
  the expanded path: the blocking loop slices the query axis, and under grouping
  that axis carries repeats and positions interleaved, so a block boundary would
  cut one query position's rows across two passes. The mask must be TILED to
  match — row `r * q_len + t` sees mask row `t` — and getting that transposed
  computes plausible garbage, which is why the test compares against the expanded
  path with a REAL causal mask rather than a permissive one.

- **`inference::cpu_pools::DECODE_SHAPED_MAX_TOKENS`** — the phase predicate is
  no longer `seq_len == 1`. What the pool choice turns on is whether the matmuls
  are bandwidth-bound, and a 4-token verify re-reads exactly the weights one
  token does. Measured with `examples/qmatmul_bench`, the narrow pool wins at
  every width to 32 and draws level at 128. **A short block must not feed the
  decode-width calibration** — the calibration compares candidate widths by the
  cost of one token, and a sample from a different query length is not that
  measurement (same class of error as #367).

Together these were one defect: every CPU decode fast path was gated on
`q_len == 1` exactly, so a 2-token forward lost the grouping, the single-position
kernel and the narrow pool at once and cost 7.8x a 1-token forward — for one
extra token (gotcha #369).

**Validated on a SECOND machine** (2026-08-23), because #367's lesson is that a
fix measured on one is not measured: an Intel i5-10500T, 6 cores, no GPU — the
same box whose first run found the .114 calibration defect. It holds there and by
more, and the cost of a wide forward is LOWER than on the 8-core Ryzen, which is
what fewer cores predicts (more bandwidth-bound, so extra rows cost less):

```text
  width   old (i5)   new (i5)   break-even accept
      2   373/362     148/149    1.10-1.16 of 2
      4   383/377     171/182    1.47-1.58 of 4
      8   462/453     242/254    2.10-2.20 of 8
```

Against 2.5 of 4 on the Ryzen. Decode was unchanged on the i5 (8.08/8.05 against
7.88/7.75 tok/s, arms overlapping) and prefill gained 1.5-2.9%, consistent in
direction across both pairings.

**The GPU changes above are still single-machine.** They were measured only on
the RTX 3070 here; no second card was available. Treat their magnitudes as
one machine's numbers until a second one confirms them.

## Local speculative decoding — `inference::model_worker::ngram_spec_eligible`

The single answer to "may this request be speculated?", consulted by BOTH the
slot-admission gate and the decode loop. Two copies would eventually disagree,
and the failure is silent: the gate diverts a request off the batched path and
the loop then declines to speculate it, so it loses batching and gains nothing.

Clauses: `!logprobs` (accepted tokens carry none back out), SWIFT off (it is
already speculating), and a non-zero draft width — which is how
`inference.ngram_lookup_enabled` arrives, so the switch cannot disagree with the
shape.

**Temperature is deliberately NOT one of them, and briefly was.** The argument
for excluding sampled requests — that comparing a draft against what the sampler
returned "is a verification only while the sampler is deterministic" — is wrong,
and it left the feature inert for essentially all traffic: the OpenAI surface
defaults to 0.7 and the Anthropic one to 1.0, so Claude Code and MCP tool use,
the workload `ngram_lookup` names as its reason for existing, never speculated.

With a deterministic draft `x` (`q = δ_x`) the speculative-sampling rule accepts
with probability `min(1, p(x)/q(x)) = p(x)` and otherwise draws from
`norm((p − q)₊)` — `p` with `x` removed, renormalised. "Draw `t ~ p`; keep the
draft iff `t == x`" has exactly those two branches, so sampling each position
with the real sampler and keeping a match IS that rule, at any temperature.
`accepting_only_on_a_match_preserves_the_sampled_distribution` pins it, with a
control that fails if the metric could not detect a bias.

Measured at temperature 0.7 on an RTX 3070: a copying reply speculated at 8.83
tokens per round — the same as greedy, because a copied token's distribution is
sharply peaked — and ran 1.79-1.90 s -> 0.58-0.69 s. An open-ended reply at the
same temperature accepted nothing and `SpecBackoff` suppressed 62 of 80 rounds,
which is the correct outcome rather than a failure.

**The DISTRIBUTED n-gram path carried the same gate for the same wrong reason**,
and it was fixed the same way — but note the mechanism differed. It took the
target's raw `argmax` (`greedy_accept_reject`), which ignores temperature, top-k,
top-p AND the repetition penalties, so there the gate was correct for the
implementation. `speculative::sampled_accept_reject` samples every position
through the real sampler instead; at temperature 0 with no penalties it
reproduces the argmax decision exactly (pinned by
`at_temperature_zero_it_agrees_with_the_argmax_it_replaces`), so nothing already
using it changes. It also closed a routing-dependent difference: penalties
always applied on the local worker and never across peers, so the same request
got a different answer depending on where it ran.

**The peer-supplied non-finite guard is not optional and must survive any change
of sampler.** These logits come from another node, and NaN comparisons are
non-deterministic in `argmax`, so a malicious segment could otherwise steer which
drafts are accepted. Both helpers reject the whole round, and a test asserts they
agree on that verdict so the two cannot drift.

**Draft-MODEL paths (`speculative.rs`'s main loop, `dsd.rs`) stay greedy-only on
purpose.** A draft model has a real distribution `q`, so doing this properly
needs `min(1, p/q)` and a residual built from both — a different algorithm, not
this one. The rule here works only because an n-gram draft is a point mass.

Three things a change here must keep:

- **It runs on the sequential loop only, when the worker is otherwise idle AND
  speculation has been paying.** `slot_admission_eligible` diverts a solo
  speculatable request there; one arriving while others decode joins the batch
  instead, because that loop owns the worker for its whole duration and
  diverting would stall everyone in flight.
  **The third condition was added after measuring, and matters as much as the
  other two.** The trade was originally justified with this project's ~3%
  batching figure (#348) — a PROCESSOR measurement. On a graphics card batching
  amortises kernel launches across requests, the same launch-bound property
  speculation exploits, so it is worth far more: 8 concurrent open-ended
  requests took 29.07 s diverted against 12.48 s batched, aggregate throughput
  77 -> 33 tok/s (gotcha #373). `spec_payoff_justifies_diverting` therefore gates
  it on the tokens-per-round speculation has actually been achieving; unknown
  lets one request find out, and the answer steers the rest. With it, both
  workloads beat either fixed policy — open-ended 8-way 11.03 s against 12.48 s
  batched, copy-heavy 8-way 3.37 s against 5.52 s.
- **It is not bit-identical** (gotcha #370). Do not describe it as such.
- **A miss is not free** (gotcha #371): the forward the draft provokes costs
  even when nothing is accepted, which is what `SpecBackoff` exists for. Measure
  the workload speculation CANNOT help, not just the one it can.

## A prompt's length in tokens is a POSITION, not a statistic

**`inference::pipeline::prompt::prompt_positions`** is the single answer to "how
many positions does this prompt occupy?", for all five sites in that module.
Never open-code it, and never estimate it.

`distributed.rs` sets `index_pos = ptc + vision_expand` after the prompt pass,
and ships `index_pos` to the worker as the **rotary position of the next token
and the offset it attends from**. It must equal the number of KV positions the
prefill actually wrote. It is not a report; it is a coordinate.

It was `(prompt.chars().count() / 4).max(1)`, commented *"approximate ... no
tokenizer in-process"* — true when written, and left in place after
`standalone_tokenizer` began lazily building one from `gguf_header.bin` three
lines below it. On a 24 KB tool-calling prompt the estimate came out **6053
against a true 5529**, so the first generated token was computed 524 positions
past the end of the cache. The logits are noise, and the model answers with
end-of-turn or with filler repeated to the token limit (gotcha #400).

Three properties a change here must keep:

- **It only ever bit the pipeline path**, which is the one every request takes
  while the model is not loaded — so the same request failed cold and succeeded
  warm. Anything that makes a placement or path decision differently for a cold
  request inherits this shape: *test it cold*.
- **The error scales with prompt length.** A short prompt misses by a position
  or two and still reads correctly. That is why four releases of `curl`
  reproduction attempts failed — a retry is warm and a minimal repro is short —
  and why the fix's test uses a prompt the old estimate demonstrably got wrong,
  with a control asserting exactly that.
- **The estimate survives only where there is genuinely no tokenizer** (no
  `gguf_header.bin`), and warns when it does. Refusing the request would be
  worse than a reply that may drift, but a node navigating by a guess must say
  so, because nothing else can tell it it is lost.

**The rule this encodes.** A value that is *reported* may be approximate; a
value that is *acted on* may not. `ptc` fed both `usage.prompt_tokens` and
`index_pos`, and the comment justifying the approximation was written for the
first while the second silently depended on it being exact. When one number
serves two purposes it inherits the stricter requirement — so a comment
explaining why an estimate is good enough is a place to go and check what else
reads it.

**And the diagnostic lesson**, which cost more than the fix: an earlier pass
cleared position bookkeeping by observing that `index_pos` "jumps correctly to
the prompt length and then increments by one". That compared the number against
ITSELF. The discriminator is `index_pos` against the worker's `kv_offset` on the
same forward — two values that must be equal, printed in adjacent log lines the
whole time. Ask what a number is supposed to EQUAL, not whether it looks
plausible.

## A peer's stated reason is the answer; do not substitute one of your own

Two helpers in `inference::pipeline` own what happens when a serving node
refuses a forward. Both exist because the correct handling was implemented on
the **verify** hops and missing on the **prefill** hops, in the same files, with
a comment on one of them explaining exactly why it mattered.

- **`peer_error_from_result`** — recovers the peer's failure from a completed
  `LayerResult`. A serving node reports why it refused in
  `finish_reason: NetworkFinishReason::Error(msg)`, and `LayerResult::error`
  leaves `token_ids` empty when it does, so a caller testing only
  `token_ids.is_empty()` throws the reason away. It is applied at the single
  choke point: `forward_through_segments` is now a thin wrapper over
  `forward_through_segments_inner`, because the inner function has SIX `Ok`
  return sites and checking each one is the mistake being fixed. The one path
  that does not go through it — `speculative.rs`'s prefill, which awaits
  `wait_for_result` directly — carries its own call and says so.
- **`every_holder_would_refuse`** — asked BEFORE failing over. Deliberately
  narrow: only `Validation`, because that describes the REQUEST and every holder
  reproduces it identically. A missing shard or a dead worker says nothing about
  the next holder and must still fail over; narrowing this predicate too far
  would disable failover itself, which is why the negative control test lists
  five such errors by name.

**What was measured (2026-08-30, released v0.3.135).** One over-long prompt gave
three different answers depending on topology — `500 server_error` naming an
internal mechanism, `503 Segment failover exhausted` advising
`swarmllm get-model` for a prompt that is simply too long, and `400` after a
wasted round trip per peer — while the same prompt on a LOCAL model was always a
clean `400` naming the exact number of tokens to cut. Whose fault a mistake was
depended on which machine held the model: gotcha #304's shape, on a path #304's
fix did not reach. And because the class was flattened to `Inference`,
`failure_is_penalty_worthy` (which exempts `Validation`) docked the serving peer
for the caller's mistake. Gotcha #415.

**The rule to carry.** Before failing over, ask whether the next holder could
possibly answer differently — and never replace a reason you were given with one
you invented.

## A streaming path must announce that it finished

`api/openai/streaming.rs` treats "no finish event arrived" as "this path never
streamed" and falls back to emitting the whole `InferenceOutput.content` as one
delta. So a coordinator that streams tokens and then returns without sending a
terminal `finish_reason: Some(..)` does not merely omit a marker — it
**duplicates the entire reply**.

`ngram_only_spec.rs` was the only coordinator missing it;
`distributed.rs`, `remote_generate.rs`, `dsd.rs` and `speculative.rs` all had
it, which is why nothing else showed the fault. Measured on the released
v0.3.135: "Count 1 to 3, digits only" came back as `1\n2\n3<|eot_id|>1\n2\n3`
from every peer-held model on default settings (gotcha #414).

The same file also streamed the EOS token as reply text at two sites, while
`finish_speculative` filters it out of the non-streaming content — so one reply
differed by transport. **End-of-turn is a control token: it ends the reply, it
is not part of it.** Keep it in the accumulator so the loop still stops on it,
and exclude it from what is streamed.

`a_streaming_pipeline_path_sends_its_terminal_finish_event` in
`tests/repo_consistency.rs` fails the build on a streaming coordinator with no
terminal send. `pipeline/mod.rs` is excluded because it holds the shared emit
helpers — ending the stream is the coordinator's job, since only it knows why
generation stopped.

Ask of any "did this happen?" flag what its consumer does when it stays false.

## Two counters both called "tokens" — write down which event each one counts

`StreamReassembler::truncated()` is the single answer to "did tokens the peer
SENT fail to arrive?". It is deliberately NOT `usage.completion_tokens >
emitted()`.

`decode_token` accumulates bytes in a `carry` buffer and returns EMPTY until a
multi-byte codepoint completes, and the serving node skips forwarding an
empty-text event without numbering it. So `streamed_count` — what the done token
carries and what `token_id` densely numbers — counts NETWORK SENDS, while
`usage.completion_tokens` counts MODEL STEPS. They are equal only for pure
ASCII, which is what everything got tested with.

Comparing the two therefore refused correct replies: on the released v0.3.135,
"one sentence in Chinese" answered `503 Reply truncated in transit: 1 of 3
tokens arrived` and five emoji answered `9 of 12`, both having arrived complete,
with a hint telling the user to try a different machine (gotcha #416). The
product ships 21 locales and most are multi-byte.

Three properties a change here must keep. **`missing()` is derived from the done
token**, which is the only figure comparable to what arrived. **An unsequenced
peer is never judged truncated** — it cannot say what to expect, the same
degradation `is_complete` makes for mixed-version swarms. And **a genuine loss
must still be caught**: `a_genuinely_lost_token_is_still_truncated` is the
control, because "never report truncation" would silently reintroduce the
truncated replies of gotcha #282.

The clamp of `completion_tokens` down to `delivered` now happens only on a real
truncation, so usage stops under-reporting every multi-byte reply.

## A vocabulary piece becomes token ids in exactly one place

**`inference::tokenizer::BpeTokenizer::push_piece_ids`** is the only way a merged
piece is turned into ids on the BPE path. There are two sites that need it — the
single-character early return and the output walk — and **both were
`.unwrap_or(0)`**, i.e. `<unk>`.

**What that cost.** A SentencePiece vocabulary deliberately contains no bare tab
or newline; it carries `<0x09>` / `<0x0A>` and expects byte fallback, which
`spm_encode` has always done. The BPE path did not. So on any GGUF taking that
branch — `tokenizer_model == "llama"` that ALSO ships merges: TinyLlama,
Llama-2, Mistral, Vicuna — **every newline in every prompt was handed to the
model as `<unk>`**, chat-template newlines included, so essentially every
multi-line prompt was structurally corrupted (gotcha #421).

**Byte fallback is gated on `is_sentencepiece` and must stay that way.** A GPT-2
vocabulary maps every byte through `byte_encoder` into a character it does
contain and carries no `<0xNN>` tokens, so a miss there is a genuine vocabulary
problem and falling back would invent tokens. Pinned in both directions by
`a_character_with_no_vocabulary_entry_falls_back_to_its_byte_token` (with a
control: a character having neither a piece nor a byte token must still land on
`<unk>`) and `a_gpt2_vocabulary_does_not_get_byte_fallback`.

**The general rule, and why this one is worth writing down.** Nothing internal
could have found it: the tokenizer round-trips against itself perfectly, and the
equivalence tests written the same day compare it to an *earlier version of
itself* — both were equally wrong. It took HuggingFace `tokenizers` as an
outside reference, and nine minutes. **A component that is only ever checked
against its own past cannot be shown to be correct, only unchanged.**
`examples/tokenizer_scaling` with `SWARM_TOK_TEXT` is the harness; 17/17 samples
now agree on the fixed path and 6/6 on the untouched GPT-2 one.

**Do not add a third site that maps a piece to an id.** Call the helper.
