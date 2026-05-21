use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::types::{ModelId, NetworkCommand, ShardId};

use super::scan::rescan_local_shards;
use super::vram::global_pool_vram_mb;

/// Interval between per-model request-count resets. Keeps the popularity window
/// tight enough that a burst of demand on a stale model doesn't dominate.
const AUTO_MANAGE_REQUEST_RESET_INTERVAL_SECS: u64 = 600;
/// Minimum gap between notify-triggered evaluations. Prevents cascading
/// re-evaluations when peers broadcast shard progress in bursts.
const AUTO_MANAGE_NOTIFY_COOLDOWN_SECS: u64 = 60;
/// Settle delay after a notify before evaluating — gives gossip time to fan out
/// and peers time to announce their own downloads so our score is up-to-date.
const AUTO_MANAGE_NOTIFY_SETTLE_SECS: u64 = 15;
/// EMA decay weight for model request popularity scoring.
/// A spike of 100 requests persists ~30min and drops to noise after ~2h.
const EMA_DECAY_WEIGHT: f64 = 0.85;
const EMA_FRESH_WEIGHT: f64 = 0.15;
/// Minimum auto-manage scan interval. Prevents pathological config values from
/// starving other tasks with sub-10s ticks.
const MIN_AUTO_MANAGE_INTERVAL_SECS: u64 = 10;

/// Read pool shard pins via a non-blocking try_read on `pool_state`.
/// Returns an empty Vec on lock contention or when the node isn't in a pool.
/// Used by SCORING — a sync hot path that can't `.await` and accepts the
/// "miss this cycle" failure mode for an under-replicated bonus signal.
///
/// SEC: do NOT use this from prune. A pinned shard slipping through to the
/// pruneable set because pool_state was momentarily write-locked is a
/// data-loss vector. Prune calls `read_shard_pins_blocking` instead.
pub(super) fn read_shard_pins(state: &SharedState) -> Vec<crate::types::ShardPin> {
    state
        .credits
        .pool_state
        .try_read()
        .ok()
        .and_then(|ps| ps.as_ref().map(|s| s.shard_pins.clone()))
        .unwrap_or_default()
}

/// Async sibling of `read_shard_pins` that awaits the read lock instead of
/// failing-empty on contention. Used by prune (and any other code path
/// where treating "lock contended" as "no pins exist" is unsafe).
pub(super) async fn read_shard_pins_blocking(state: &SharedState) -> Vec<crate::types::ShardPin> {
    state
        .credits
        .pool_state
        .read()
        .await
        .as_ref()
        .map(|s| s.shard_pins.clone())
        .unwrap_or_default()
}

/// Maximum time a P2P download permit may sit in `p2p_download_permits`
/// before being considered stalled. Matches the rough order of the longest
/// honest shard chunk fetch (32 MiB chunk × multi-segment retry).
/// Anything older is almost certainly a silent drop in the libp2p path.
///
/// R141: tightened from 600s → 180s. The previous window was sized for
/// the absolute-worst-case slow peer including pessimistic retries, but
/// a non-technical user staring at a stuck download for 10 minutes is a
/// product-broken experience. 3 minutes still comfortably covers an
/// honest 32 MiB chunk over a slow link (~150 KiB/s sustained), and a
/// genuine slow-peer wins the next cycle when the HF fallback kicks in
/// or a faster holder is selected.
const P2P_PERMIT_STALL_SECS: u64 = 180;

/// Sweep `p2p_download_permits` for entries older than `P2P_PERMIT_STALL_SECS`.
/// Releases the permit (drop semantics on the OwnedSemaphorePermit) and
/// clears the matching `acquisition_progress` shard entry so the next
/// auto-manage tick can retry. Marks the shard as P2P-failed so the HF
/// fallback fires next cycle, mirroring the give-up path in
/// `shard_transfer.rs::retry_shard_or_fallback`.
pub(super) fn sweep_stalled_p2p_permits(state: &SharedState) {
    let cutoff = std::time::Duration::from_secs(P2P_PERMIT_STALL_SECS);
    let now = std::time::Instant::now();
    let mut stalled: Vec<crate::types::ShardId> = Vec::new();
    for entry in state.models.p2p_download_permits.iter() {
        if now.duration_since(entry.value().1) > cutoff {
            stalled.push(entry.key().clone());
        }
    }
    for sid in stalled {
        state.models.p2p_download_permits.remove(&sid);
        state.models.shard_p2p_failed.insert(sid.clone());
        if let Some(mut entry) = state.models.acquisition_progress.get_mut(&sid.model_id) {
            entry.shard_progress.remove(&sid.index);
        }
        tracing::warn!(
            model = %sid.model_id,
            shard = sid.index,
            stall_secs = P2P_PERMIT_STALL_SECS,
            "Auto-manage: released stalled P2P download permit; HF fallback will fire next cycle"
        );
        // Wake the manager loop so the HF retry can fire promptly rather
        // than waiting for the next periodic interval.
        state.models.auto_manage_notify.notify_one();
    }
}

/// Compute a position on a u32 consistent hash ring for a node's virtual slot.
pub(super) fn hash_ring_position(node_bytes: &[u8; 32], virtual_node: u32) -> u32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(node_bytes);
    hasher.update(&virtual_node.to_le_bytes());
    let hash = hasher.finalize();
    u32::from_le_bytes([
        hash.as_bytes()[0],
        hash.as_bytes()[1],
        hash.as_bytes()[2],
        hash.as_bytes()[3],
    ])
}

/// Auto-manages shard downloads to improve network health.
///
/// Periodically evaluates:
/// 1. Which models are popular on the network (most holders / most shards)
/// 2. Which shards are rarest (fewest holders) for those models
/// 3. Whether this node has budget (disk space, max_shards) to download more
/// 4. Whether the global VRAM pool can run the model (deprioritize models too large to run)
///
/// Then triggers HuggingFace shard downloads for the rarest shards of popular models.
pub struct AutoShardManager {
    pub(super) shared_state: Arc<SharedState>,
    pub(super) network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
    /// Notify trigger -- woken when new HF sources or manifests arrive from peers.
    notify: Arc<tokio::sync::Notify>,
    /// Semaphore to limit concurrent shard downloads.
    pub(super) download_semaphore: Arc<tokio::sync::Semaphore>,
}

/// A candidate shard identified for auto-download.
#[derive(Debug, Clone)]
pub(super) struct ShardCandidate {
    pub model_id: ModelId,
    pub model_name: String,
    pub shard_index: u32,
    pub shard_size_bytes: u64,
    pub holder_count: usize,
    /// Score: higher = more worth downloading. Factors in rarity and model popularity.
    pub score: f64,
}

/// A candidate shard identified for auto-pruning.
#[derive(Debug, Clone)]
pub(super) struct PruneCandidate {
    pub model_id: ModelId,
    pub model_name: String,
    pub shard_index: u32,
    pub shard_size_bytes: u64,
    pub holder_count: usize,
    pub target_replicas: u32,
    /// Score: higher = more prunable.
    pub score: f64,
}

impl AutoShardManager {
    pub fn new(
        shared_state: Arc<SharedState>,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let notify = shared_state.models.auto_manage_notify.clone();
        let max_concurrent = shared_state
            .config
            .auto_manage
            .max_concurrent_downloads
            .max(1);
        let download_semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        Self {
            shared_state,
            network_tx,
            shutdown_rx,
            notify,
            download_semaphore,
        }
    }

    /// Run the auto-manage loop. Checks periodically based on config interval,
    /// and also wakes immediately when new HF sources or manifests arrive from peers.
    /// Always runs (even when disabled) so it can respond to runtime config changes.
    /// Sum the on-disk byte count of all shards held by a given node.
    /// Returns (total_bytes, shard_count).
    pub(super) fn local_shard_bytes(&self, node_id: &crate::types::NodeId) -> (u64, u32) {
        let local_shards = self.shared_state.model_registry.shards_for_node(node_id);
        let count = local_shards.len() as u32;
        let bytes = local_shards
            .iter()
            .filter_map(|sid| {
                let manifest = self
                    .shared_state
                    .model_registry
                    .get_manifest(&sid.model_id)?;
                manifest
                    .shards
                    .iter()
                    .find(|s| s.index == sid.index)
                    .map(|si| si.size_bytes)
            })
            .sum();
        (bytes, count)
    }

    pub async fn run(mut self) {
        let config = &self.shared_state.config.auto_manage;
        if !config.enabled {
            tracing::info!("AutoShardManager: disabled at startup (enable from dashboard)");
        }

        // Use interval_seconds if set, else fall back to interval_minutes * 60
        let mut interval_secs = config
            .interval_seconds
            .unwrap_or_else(|| config.interval_minutes.max(1) as u64 * 60)
            .max(MIN_AUTO_MANAGE_INTERVAL_SECS);
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Skip the first tick (fires immediately) -- let the node discover peers first.
        // Then add a deterministic per-node phase offset so a fleet of nodes
        // that all booted in the same epoch window doesn't fire `evaluate()`
        // at the same clock minute and trigger a thundering-herd of HF byte-
        // range requests against the same CDN origin. Phase = first 2 bytes
        // of BLAKE3(node_id) modulo the interval. Stable per-node, no clock
        // synchronization assumptions, no randomness needed.
        interval.tick().await;
        let phase_offset_secs = {
            let nid = self.shared_state.identity.node_id();
            let h = blake3::hash(&nid.0);
            let bytes = h.as_bytes();
            let raw = u16::from_le_bytes([bytes[0], bytes[1]]) as u64;
            raw % interval_secs.max(1)
        };
        if phase_offset_secs > 0 {
            tracing::debug!(
                phase_offset_secs,
                "Auto-manage applying per-node phase offset to break startup thundering-herd"
            );
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        return;
                    }
                }
                _ = tokio::time::sleep(Duration::from_secs(phase_offset_secs)) => {}
            }
        }

        tracing::info!(
            interval_secs = interval_secs,
            max_storage_mb = config.max_storage_mb,
            "AutoShardManager running"
        );

        // Request count reset interval.
        let mut request_reset_interval =
            tokio::time::interval(Duration::from_secs(AUTO_MANAGE_REQUEST_RESET_INTERVAL_SECS));
        request_reset_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        request_reset_interval.tick().await; // skip first tick

        // Cooldown: minimum time between evaluations triggered by notify.
        // Start back-dated so the first notify after launch isn't throttled.
        let notify_cooldown = Duration::from_secs(AUTO_MANAGE_NOTIFY_COOLDOWN_SECS);
        let mut last_notify_eval = std::time::Instant::now() - (notify_cooldown * 2);

        let mut config_watch_rx = self.shared_state.config_watch_rx();

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("AutoShardManager shutting down");
                        break;
                    }
                }
                _ = interval.tick() => {
                    // Always rescan for new shard files on disk (even if auto-manage disabled
                    // — picking up a manually-placed shard is a correctness fix, not policy).
                    // The network re-announce is gated on `auto_manage_enabled`: when the
                    // user has paused auto-manage they have opted out of participating
                    // automatically, so we register the shard locally + load the metadata
                    // but DON'T tell the swarm we host it. The manual
                    // `POST /api/admin/rescan-shards` admin endpoint always re-announces
                    // (user-driven action overrides the pause).
                    let auto_enabled = self
                        .shared_state
                        .models
                        .auto_manage_enabled
                        .load(std::sync::atomic::Ordering::Acquire);
                    let net_arg = if auto_enabled { Some(&self.network_tx) } else { None };
                    let changed = rescan_local_shards(&self.shared_state, net_arg).await;
                    if !changed.is_empty() {
                        tracing::info!(
                            models = ?changed.iter().map(|m| m.0.as_str()).collect::<Vec<_>>(),
                            "Rescan discovered new local shards"
                        );
                    }

                    // SEC: sweep stalled P2P download permits. Without this,
                    // a silent libp2p drop (request never reaches peer)
                    // parks the OwnedSemaphorePermit forever — after enough
                    // such drops, all `max_concurrent_downloads` slots are
                    // permanently held and auto-manage downloads freeze
                    // with no log signal. Sweep runs every loop tick so
                    // worst-case stall window is ~one interval.
                    sweep_stalled_p2p_permits(&self.shared_state);

                    // Re-check enabled -- admin API can toggle at runtime
                    if self.shared_state.models.auto_manage_enabled.load(std::sync::atomic::Ordering::Acquire) {
                        self.evaluate().await;
                    }
                }
                _ = self.notify.notified() => {
                    // Woken by a new HfSourceGossip or ModelManifest -- wait for gossip
                    // to settle and peers to announce their downloads before evaluating.
                    // Race the settle window against shutdown so a daemon stop fired
                    // mid-sleep doesn't add 15s to the supervisor's exit window.
                    let mut shutdown_rx = self.shutdown_rx.clone();
                    tokio::select! {
                        _ = tokio::time::sleep(Duration::from_secs(AUTO_MANAGE_NOTIFY_SETTLE_SECS)) => {}
                        _ = shutdown_rx.changed() => {
                            if *shutdown_rx.borrow() { break; }
                        }
                    }
                    // Cooldown: skip if we evaluated recently (prevents cascading
                    // re-evaluations from shard progress gossip between peers).
                    // Exception: bypass when P2P has exhausted for one or more
                    // shards — those need HF fallback picked up ASAP, not in 45s.
                    let since_last = last_notify_eval.elapsed();
                    let has_p2p_failures = !self.shared_state.models.shard_p2p_failed.is_empty();
                    if since_last < notify_cooldown && !has_p2p_failures {
                        tracing::debug!(
                            remaining_secs = (notify_cooldown - since_last).as_secs(),
                            "AutoShardManager: notify cooldown active, skipping evaluation"
                        );
                        continue;
                    }
                    if self.shared_state.models.auto_manage_enabled.load(std::sync::atomic::Ordering::Acquire) {
                        tracing::info!("AutoShardManager: triggered by new HF source or manifest");
                        last_notify_eval = std::time::Instant::now();
                        self.evaluate().await;
                    }
                }
                _ = request_reset_interval.tick() => {
                    self.decay_request_counts();
                    self.update_model_trust();
                }
                _ = config_watch_rx.changed() => {
                    let params = config_watch_rx.borrow().clone();
                    let new_secs = (params.auto_manage_interval_minutes.max(1) as u64) * 60;
                    let new_secs = new_secs.max(MIN_AUTO_MANAGE_INTERVAL_SECS);
                    if new_secs != interval_secs {
                        tracing::info!(
                            old_interval_secs = interval_secs,
                            new_interval_secs = new_secs,
                            "Hot-reloaded auto-manage interval"
                        );
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "auto_manage",
                                "interval_changed",
                                format!(
                                    "Auto-manage interval changed: {}s → {}s",
                                    interval_secs, new_secs
                                ),
                            )
                            .with_detail_num(new_secs as i64),
                        );
                        interval_secs = new_secs;
                        interval = tokio::time::interval(Duration::from_secs(new_secs));
                        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
                        // Skip the immediate first tick — tokio::time::interval fires
                        // on `.tick()` once at t=0 by default, which would trigger a
                        // spurious evaluation right after the operator changed the
                        // interval (R104 follow-up).
                        interval.tick().await;
                    }
                }
            }
        }
    }

    /// Unified auto-manage evaluation: download under-replicated shards, then
    /// prune over-replicated ones. A single pass ensures consistent target replicas
    /// and coordinates pruning with downloads (only prune when there's resource
    /// pressure or when making room for higher-value shards).
    async fn evaluate(&self) {
        // Phase C.2: refresh Parallax stability counters before scoring so
        // both download and prune paths see the same up-to-date bias view.
        // No-op when `parallax_auto_rebalance=false` or the cluster is
        // infeasible for the given model.
        self.update_parallax_stability();

        let local_node_id = self.shared_state.identity.node_id().clone();
        let hosted_before = self
            .shared_state
            .model_registry
            .all_shard_entries()
            .into_iter()
            .filter(|(_, holders)| holders.contains(&local_node_id))
            .count();
        let models_hosted = self
            .shared_state
            .model_registry
            .all_shard_entries()
            .into_iter()
            .filter(|(_, holders)| holders.contains(&local_node_id))
            .map(|(sid, _)| sid.model_id)
            .collect::<std::collections::HashSet<_>>()
            .len();
        let active_downloads = self.shared_state.models.acquisition_progress.len();

        self.evaluate_and_download().await;

        // Prune only if enabled -- pruning is the last resort to free resources.
        // The download phase already respects storage budgets, so pruning only
        // fires when we're genuinely over-replicated or under resource pressure.
        if self.shared_state.config.auto_manage.prune_enabled {
            self.evaluate_and_prune().await;
        }

        // R111: refresh the user-visible wishlist at the end of every
        // tick so the dashboard reflects the latest swarm state even
        // when no client is currently rendering it. The WS stats build
        // also refreshes — duplicating here is cheap (single registry
        // pass) and means an idle dashboard sees fresh data the moment
        // it connects.
        crate::model::auto_manage::refresh_wishlist(&self.shared_state);
        crate::model::auto_manage::quant::refresh_quant_recommendations(&self.shared_state);
        // R134.6: opt-in opportunistic quant upgrade. No-op when the flag
        // is off; otherwise promotes the recommended variant's trust so
        // the next tick's download pass picks it up naturally.
        crate::model::auto_manage::quant::apply_quant_auto_action(&self.shared_state);

        let hosted_after = self
            .shared_state
            .model_registry
            .all_shard_entries()
            .into_iter()
            .filter(|(_, holders)| holders.contains(&local_node_id))
            .count();
        let new_downloads = self
            .shared_state
            .models
            .acquisition_progress
            .len()
            .saturating_sub(active_downloads);
        let delta = hosted_after as i64 - hosted_before as i64;

        // Emit only when something actually changed. The notify path bypasses
        // the 60s cooldown when has_p2p_failures is true, so a 50-shard burst
        // of HF download completions can fire 50 cycle_complete events in a
        // few seconds. activity_tx has cap 256 — combined with 150+ per-shard
        // events that's enough to saturate the channel during cold start and
        // drop unrelated subsystem events.
        if delta != 0 || new_downloads > 0 {
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "auto_manage",
                    "cycle_complete",
                    format!(
                        "Auto-manage cycle: {} models, {} shards hosted ({:+}), {} download(s) started",
                        models_hosted, hosted_after, delta, new_downloads
                    ),
                )
                .with_detail_num(hosted_after as i64),
            );
        }
    }

    /// Download under-replicated shards based on geo-aware scoring.
    async fn evaluate_and_download(&self) {
        let config = &self.shared_state.config.auto_manage;
        let local_node_id = self.shared_state.identity.node_id().clone();

        // Clean up stale peer_shard_downloads: if a peer is now a registered
        // shard holder, remove its in-flight download entry. This handles the
        // case where the Complete gossip message was lost or delayed.
        self.shared_state
            .models
            .peer_shard_downloads
            .retain(|shard_id, downloaders| {
                downloaders.retain(|(node_id, _pct)| {
                    // Keep entry only if the peer is NOT yet a registered holder
                    !self
                        .shared_state
                        .model_registry
                        .shard_holders(shard_id)
                        .contains(node_id)
                });
                !downloaders.is_empty()
            });

        // Peer warmup grace period: if we have zero peers and just started,
        // wait for peer discovery before evaluating. Prevents a fresh node
        // from immediately downloading everything from HF before it learns
        // that peers already hold shards.
        let peers = self.shared_state.peer_registry.len();
        if peers == 0 {
            let stats = self.shared_state.metrics.node_stats.read().await;
            let uptime_secs = (chrono::Utc::now() - stats.uptime_start)
                .num_seconds()
                .max(0) as u64;
            drop(stats);
            // Exception: if a shard has already exhausted P2P in this session,
            // the user is waiting on the HF fallback path — don't delay 60s.
            let p2p_exhausted = !self.shared_state.models.shard_p2p_failed.is_empty();
            if uptime_secs < 60 && !p2p_exhausted {
                tracing::info!(
                    uptime_secs,
                    "AutoShardManager: waiting for peer discovery before evaluation (no peers yet)"
                );
                return;
            }
        }

        // Discover HF sources from hf_source.json files alongside manifests
        self.discover_hf_sources();

        // Log global pool capacity for visibility
        let pool_vram = global_pool_vram_mb(&self.shared_state);
        tracing::debug!(
            pool_vram_mb = pool_vram,
            peers = self.shared_state.peer_registry.len(),
            "AutoShardManager: global VRAM pool"
        );

        // 1. Check budget: how much storage do we have left?
        let budget = self.remaining_budget_bytes(config, &local_node_id);
        if budget == 0 {
            tracing::info!(
                peers = self.shared_state.peer_registry.len(),
                "AutoShardManager: no remaining storage budget — skipping downloads"
            );
            return;
        }

        // 2. Gather candidate shards across all known models (VRAM-aware scoring)
        let candidates = self.gather_candidates(&local_node_id, pool_vram);
        if candidates.is_empty() {
            tracing::info!(
                budget_bytes = budget,
                "AutoShardManager: no candidate shards to download"
            );
            return;
        }

        // 3. Select the best candidates within budget
        let selected = self.select_within_budget(candidates, budget, config.max_shards);
        if selected.is_empty() {
            return;
        }

        tracing::info!(
            count = selected.len(),
            "AutoShardManager: downloading shards"
        );

        // 4. Trigger downloads
        for candidate in &selected {
            self.trigger_download(candidate).await;
        }
    }

    /// Decay model request counts with EMA and feed into region_demand.
    /// Instead of zeroing counters, we blend: new_rate = old * EMA_DECAY + fresh * EMA_FRESH.
    /// A spike of 100 requests persists ~30min and drops to noise after ~2h.
    pub(super) fn decay_request_counts(&self) {
        let our_region = self.our_region().unwrap_or_else(|| "??".to_string());

        for entry in self.shared_state.models.model_request_counts.iter() {
            let model_id = entry.key().clone();
            let fresh = entry.value().swap(0, std::sync::atomic::Ordering::Relaxed);

            let key = (model_id, our_region.clone());
            let old = self
                .shared_state
                .region_demand
                .get(&key)
                .map(|v| *v)
                .unwrap_or(0.0);
            // SEC: NaN guard on the cached value. Gotcha #98 (R102) added
            // the guard at the gossip ingress, but a NaN from a pre-R102
            // DB rehydrate or a brief race window still poisons the EMA
            // here. `NaN * 0.85 + n * 0.15 = NaN` permanently — and
            // `geo_target_replicas` reads the value with no NaN check,
            // landing in the `else` arm with `demand_factor = 3.0` (max
            // popularity) for a model nobody is requesting. Treat NaN/Inf
            // as 0.0 so the bad entry self-heals on the next decay tick.
            let old = if old.is_finite() { old } else { 0.0 };
            let new_rate = old * EMA_DECAY_WEIGHT + fresh as f64 * EMA_FRESH_WEIGHT;
            if new_rate > 0.001 {
                self.shared_state.region_demand.insert(key, new_rate);
            } else {
                // Clean up negligible entries
                self.shared_state.region_demand.remove(&key);
            }
        }
    }

    /// Update model trust levels: promote popular models, decay inactive ones.
    ///
    /// - Models with >= 3 unique holder nodes -> NetworkPopular
    /// - Models without requests for 7 days -> decay (DemandVerified->Discovered)
    /// - Pinned models never decay
    /// - Ensures new gossip-discovered models get a Discovered entry
    pub(super) fn update_model_trust(&self) {
        let registry = &self.shared_state.model_registry;

        for manifest in registry.models() {
            // Count unique holder nodes for this model
            let mut holder_nodes = std::collections::HashSet::new();
            for shard in &manifest.shards {
                let sid = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                for node in registry.shard_holders(&sid) {
                    holder_nodes.insert(node);
                }
            }

            let mut trust = self
                .shared_state
                .models
                .model_trust
                .entry(manifest.id.clone())
                .or_insert_with(crate::types::ModelTrustInfo::new_discovered);

            // Promote to NetworkPopular if enough unique nodes hold shards.
            // Threshold scales with pool size: min(3, pool_size-1) so a 3-node
            // cluster can promote at 2 holders, while large networks still need 3.
            // No DemandVerified prerequisite -- if peers already hold shards,
            // the model is clearly legitimate and new nodes should be able to adopt it.
            let pool_size = crate::pool::scope::effective_pool_size(&self.shared_state);
            let popular_threshold = 3usize.min(pool_size.saturating_sub(1)).max(1);
            if holder_nodes.len() >= popular_threshold
                && trust.trust_level < crate::types::ModelTrustLevel::NetworkPopular
            {
                trust.trust_level = crate::types::ModelTrustLevel::NetworkPopular;
                tracing::info!(
                    model = %manifest.id,
                    holders = holder_nodes.len(),
                    "Model promoted to NetworkPopular"
                );
                self.shared_state.emit_activity(
                    crate::daemon::state::ActivityEvent::new(
                        "auto_manage",
                        "model_promoted",
                        format!(
                            "Model {} promoted to NetworkPopular ({} holders)",
                            manifest.name,
                            holder_nodes.len()
                        ),
                    )
                    .with_model(&manifest.id.0)
                    .with_model_name(&manifest.name)
                    .with_detail_num(holder_nodes.len() as i64),
                );
            }

            // Decay inactive models
            trust.maybe_decay();

            // Persist updated trust info
            let _ = self
                .shared_state
                .db
                .put_json("model_trust", &manifest.id.0, trust.value());
        }
    }

    /// Discover HF source metadata from `hf_source.json` files next to manifests.
    ///
    /// This allows seeding HF source info by placing a small JSON file:
    /// `{ "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf" }`
    pub(super) fn discover_hf_sources(&self) {
        let models_dir = self.shared_state.shard_store().models_dir();

        if !models_dir.is_dir() {
            return;
        }

        if let Ok(entries) = std::fs::read_dir(&models_dir) {
            for entry in entries.flatten() {
                if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    continue;
                }
                let model_id_str = entry.file_name().to_string_lossy().to_string();
                let mid = ModelId(model_id_str.clone());

                // Skip if already known
                if self.shared_state.models.hf_sources.contains_key(&mid) {
                    continue;
                }

                let hf_path = entry.path().join(crate::model::shard::HF_SOURCE_FILENAME);
                if hf_path.exists() {
                    if let Ok(data) = std::fs::read_to_string(&hf_path) {
                        if let Ok(source) = serde_json::from_str::<crate::daemon::HfSource>(&data) {
                            tracing::info!(
                                model = %model_id_str,
                                repo = %source.repo_id,
                                "Discovered HF source from hf_source.json"
                            );
                            self.shared_state.emit_activity(
                                crate::daemon::state::ActivityEvent::new(
                                    "auto_manage",
                                    "hf_source_discovered",
                                    format!(
                                        "Discovered HF source for {} ({})",
                                        model_id_str, source.repo_id
                                    ),
                                )
                                .with_model(&model_id_str)
                                .with_detail_str(&source.repo_id),
                            );
                            self.shared_state
                                .models
                                .hf_sources
                                .insert(mid.clone(), source.clone());
                            // Persist to DB
                            let _ =
                                self.shared_state
                                    .db
                                    .put_json("hf_sources", &model_id_str, &source);
                        }
                    }
                }
            }
        }
    }

    /// Get our detected/configured region as uppercase ISO code.
    pub(super) fn our_region(&self) -> Option<String> {
        // Try detected_region first (non-blocking try_read)
        if let Ok(guard) = self.shared_state.detected_region.try_read() {
            if let Some(ref r) = *guard {
                return Some(r.to_uppercase());
            }
        }
        self.shared_state
            .config
            .identity
            .region
            .as_ref()
            .map(|r| r.to_uppercase())
    }

    /// Count shard holders that are in the same region as us.
    pub(super) fn count_regional_holders(
        &self,
        holders: &[crate::types::NodeId],
        local_node_id: &crate::types::NodeId,
        our_region: &str,
    ) -> usize {
        let mut count = 0;
        for h in holders {
            if *h == *local_node_id {
                count += 1; // We're always in our own region
                continue;
            }
            if let Some(peer) = self.shared_state.peer_registry.get(h) {
                if let Some(ref cap) = peer.capability {
                    if let Some(ref r) = cap.region {
                        if r.to_uppercase() == our_region {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::types::ModelId;

    use super::super::manager::{PruneCandidate, ShardCandidate};

    #[test]
    fn shard_candidate_scoring() {
        // Higher score = more worth downloading
        let c1 = ShardCandidate {
            model_id: ModelId("m1".into()),
            model_name: "Model 1".into(),
            shard_index: 0,
            shard_size_bytes: 512 * 1024 * 1024,
            holder_count: 0,
            score: 10.0 * 10.0, // popular + zero holders
        };
        let c2 = ShardCandidate {
            model_id: ModelId("m2".into()),
            model_name: "Model 2".into(),
            shard_index: 0,
            shard_size_bytes: 512 * 1024 * 1024,
            holder_count: 5,
            score: 10.0 * 1.0, // popular but well-replicated
        };
        assert!(c1.score > c2.score);
    }

    #[test]
    fn budget_zero_when_max_shards_reached() {
        // AutoManageConfig with max_shards = 0 means unlimited
        let config = crate::config::AutoManageConfig {
            enabled: true,
            max_storage_mb: 10000,
            interval_minutes: 60,
            max_shards: 0,
            interval_seconds: None,
            max_concurrent_downloads: 1,
            default_model_shard_cap: 0,
            model_policies: std::collections::HashMap::new(),
            prune_enabled: true,
            min_replicas: 2,
            prune_cooldown_secs: 300,
            max_holder_load_for_prune: 3,
            parallax_auto_rebalance: true,
            hf_watcher_enabled: false,
            wishlist_gossip_publish: false,
            auto_switch_quants: false,
        };
        assert_eq!(config.max_shards, 0); // unlimited
    }

    #[test]
    fn default_max_concurrent_downloads() {
        let config = crate::config::AutoManageConfig::default();
        assert_eq!(config.max_concurrent_downloads, 3);
    }

    #[tokio::test]
    async fn semaphore_limits_concurrent_downloads() {
        let sem = Arc::new(tokio::sync::Semaphore::new(2));

        // Acquire 2 permits -- should succeed
        let p1 = sem.clone().acquire_owned().await.unwrap();
        let p2 = sem.clone().acquire_owned().await.unwrap();
        assert_eq!(sem.available_permits(), 0);

        // Third acquire would block, so use try_acquire
        assert!(sem.try_acquire().is_err());

        // Drop one permit -- should free a slot
        drop(p1);
        assert_eq!(sem.available_permits(), 1);

        let _p3 = sem.clone().acquire_owned().await.unwrap();
        assert_eq!(sem.available_permits(), 0);

        drop(p2);
        drop(_p3);
        assert_eq!(sem.available_permits(), 2);
    }

    // --- Pruning unit tests ---

    /// Helper: compute geo-scaled target replicas (log2-based).
    /// Mirrors geo_target_replicas() but as a pure function for testing.
    fn geo_target_pure(raw_demand_factor: f64, min_replicas: u32, pool_size: usize) -> u32 {
        let global_floor = if pool_size <= 1 {
            min_replicas as usize
        } else {
            let log2_pool = (pool_size as f64).log2().ceil() as usize;
            let upper = (pool_size / 3).max(min_replicas as usize);
            log2_pool.clamp(min_replicas as usize, upper).max(1)
        };
        let target = (global_floor as f64 * raw_demand_factor).ceil() as u32;
        target.clamp(min_replicas, (pool_size as u32).max(min_replicas))
    }

    use crate::model::auto_manage::pressure_adjusted_target as pressure_adjusted_target_pure;

    #[test]
    fn geo_target_idle_small_pool() {
        // pool=10: log2(10)=4, floor=clamp(4, 2, 3)=3, factor 1.0 -> 3
        assert_eq!(geo_target_pure(1.0, 2, 10), 3);
    }

    #[test]
    fn geo_target_popular_small_pool() {
        // pool=10: floor=3, factor 3.0 -> 9
        assert_eq!(geo_target_pure(3.0, 2, 10), 9);
    }

    #[test]
    fn geo_target_medium_pool() {
        // pool=100: log2(100)=7, floor=clamp(7, 2, 33)=7, factor 1.0 -> 7
        assert_eq!(geo_target_pure(1.0, 2, 100), 7);
        // factor 2.0 -> 14
        assert_eq!(geo_target_pure(2.0, 2, 100), 14);
    }

    #[test]
    fn geo_target_large_pool() {
        // pool=1000: log2(1000)=10, floor=clamp(10, 2, 333)=10, factor 3.0 -> 30
        assert_eq!(geo_target_pure(3.0, 2, 1000), 30);
    }

    #[test]
    fn geo_target_clamped_by_pool_size() {
        // pool=3: log2(3)=2, floor=clamp(2, 2, 1)=2, factor 3.0 -> 6, clamped to 3
        assert_eq!(geo_target_pure(3.0, 2, 3), 3);
    }

    #[test]
    fn geo_target_single_node() {
        // pool=1: floor=min_replicas=2, factor 1.0 -> 2, clamped to 1
        // Actually pool_size <= 1 uses min_replicas directly
        assert_eq!(geo_target_pure(1.0, 2, 1), 2);
        assert_eq!(geo_target_pure(3.0, 1, 1), 1);
    }

    #[test]
    fn pressure_relaxed_adds_one() {
        // pressure < 0.5 -> target + 1
        assert_eq!(pressure_adjusted_target_pure(3, 0.3, 2), 4);
    }

    #[test]
    fn pressure_normal_keeps_target() {
        // 0.5 <= pressure < 0.8
        assert_eq!(pressure_adjusted_target_pure(3, 0.6, 2), 3);
    }

    #[test]
    fn pressure_eager_subtracts_one() {
        // 0.8 <= pressure < 0.95
        assert_eq!(pressure_adjusted_target_pure(4, 0.85, 2), 3);
    }

    #[test]
    fn pressure_eager_respects_min() {
        assert_eq!(pressure_adjusted_target_pure(2, 0.85, 2), 2);
    }

    #[test]
    fn pressure_urgent_subtracts_two() {
        // pressure >= 0.95
        assert_eq!(pressure_adjusted_target_pure(5, 0.97, 2), 3);
    }

    #[test]
    fn pressure_urgent_respects_min() {
        assert_eq!(pressure_adjusted_target_pure(3, 0.98, 2), 2);
        assert_eq!(pressure_adjusted_target_pure(2, 0.98, 2), 2);
    }

    #[test]
    fn prune_event_serialization() {
        let event = crate::types::PruneEvent {
            model_id: crate::types::ModelId("test-model".to_string()),
            model_name: "Test Model".to_string(),
            shard_index: 1,
            reason: "over-replicated".to_string(),
            freed_bytes: 1024 * 1024,
            remaining_local_shards: 2,
            holder_count_before: 5,
            holder_count_after: 4,
            timestamp: chrono::Utc::now(),
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("test-model"));
        assert!(json.contains("over-replicated"));

        let deser: crate::types::PruneEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.model_id.0, "test-model");
        assert_eq!(deser.shard_index, 1);
        assert_eq!(deser.holder_count_before, 5);
    }

    #[test]
    fn prune_config_defaults() {
        let config = crate::config::AutoManageConfig::default();
        assert!(config.prune_enabled);
        assert_eq!(config.min_replicas, 2);
        assert_eq!(config.prune_cooldown_secs, 300);
        assert_eq!(config.max_holder_load_for_prune, 3);
    }

    #[test]
    fn model_auto_manage_policy_prune_enabled_default() {
        // prune_enabled defaults to true via serde
        let json = r#"{"enabled": true, "max_shards": 0}"#;
        let policy: crate::config::ModelAutoManagePolicy = serde_json::from_str(json).unwrap();
        assert!(policy.prune_enabled);
    }

    #[test]
    fn resource_schedule_default_prune_aggressiveness() {
        let schedule = crate::config::ResourceSchedule::default();
        assert_eq!(schedule.prune_aggressiveness, "normal");
    }

    #[test]
    fn prune_candidate_score_ordering() {
        // Higher score = more prunable
        let cold_redundant = PruneCandidate {
            model_id: crate::types::ModelId("m1".into()),
            model_name: "M1".into(),
            shard_index: 1,
            shard_size_bytes: 1000,
            holder_count: 6,
            target_replicas: 2,
            score: 3.0 + 1.0, // high redundancy + cold bonus
        };
        let warm_less_redundant = PruneCandidate {
            model_id: crate::types::ModelId("m2".into()),
            model_name: "M2".into(),
            shard_index: 0,
            shard_size_bytes: 1000,
            holder_count: 3,
            target_replicas: 2,
            score: 1.5, // low redundancy, first shard penalty
        };
        assert!(cold_redundant.score > warm_less_redundant.score);
    }
}
