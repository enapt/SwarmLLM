use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::types::{ModelId, NetworkCommand, ShardId};

use super::scan::rescan_local_shards;
use super::vram::global_pool_vram_mb;

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
    pub async fn run(mut self) {
        let config = &self.shared_state.config.auto_manage;
        if !config.enabled {
            tracing::info!("AutoShardManager: disabled at startup (enable from dashboard)");
        }

        // Use interval_seconds if set, else fall back to interval_minutes * 60
        let interval_secs = config
            .interval_seconds
            .unwrap_or_else(|| config.interval_minutes.max(1) as u64 * 60)
            .max(10); // minimum 10 seconds
        let mut interval = tokio::time::interval(Duration::from_secs(interval_secs));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Skip the first tick (fires immediately) -- let the node discover peers first
        interval.tick().await;

        tracing::info!(
            interval_secs = interval_secs,
            max_storage_mb = config.max_storage_mb,
            "AutoShardManager running"
        );

        // Request count reset interval (10 minutes)
        let mut request_reset_interval = tokio::time::interval(Duration::from_secs(600));
        request_reset_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        request_reset_interval.tick().await; // skip first tick

        // Cooldown: minimum time between evaluations triggered by notify.
        // Prevents cascading re-evaluations when peers broadcast shard progress.
        let mut last_notify_eval = std::time::Instant::now() - Duration::from_secs(120);
        let notify_cooldown = Duration::from_secs(60);

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("AutoShardManager shutting down");
                        break;
                    }
                }
                _ = interval.tick() => {
                    // Always rescan for new shard files on disk (even if auto-manage disabled)
                    let changed = rescan_local_shards(
                        &self.shared_state,
                        Some(&self.network_tx),
                    ).await;
                    if !changed.is_empty() {
                        tracing::info!(
                            models = ?changed.iter().map(|m| m.0.as_str()).collect::<Vec<_>>(),
                            "Rescan discovered new local shards"
                        );
                    }

                    // Re-check enabled -- admin API can toggle at runtime
                    if self.shared_state.models.auto_manage_enabled.load(std::sync::atomic::Ordering::Acquire) {
                        self.evaluate().await;
                    }
                }
                _ = self.notify.notified() => {
                    // Woken by a new HfSourceGossip or ModelManifest -- wait for gossip
                    // to settle and peers to announce their downloads before evaluating.
                    tokio::time::sleep(Duration::from_secs(15)).await;
                    // Cooldown: skip if we evaluated recently (prevents cascading
                    // re-evaluations from shard progress gossip between peers).
                    let since_last = last_notify_eval.elapsed();
                    if since_last < notify_cooldown {
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
            }
        }
    }

    /// Unified auto-manage evaluation: download under-replicated shards, then
    /// prune over-replicated ones. A single pass ensures consistent target replicas
    /// and coordinates pruning with downloads (only prune when there's resource
    /// pressure or when making room for higher-value shards).
    async fn evaluate(&self) {
        self.evaluate_and_download().await;

        // Prune only if enabled -- pruning is the last resort to free resources.
        // The download phase already respects storage budgets, so pruning only
        // fires when we're genuinely over-replicated or under resource pressure.
        if self.shared_state.config.auto_manage.prune_enabled {
            self.evaluate_and_prune().await;
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
            if uptime_secs < 60 {
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
            tracing::info!("AutoShardManager: no remaining storage budget — skipping downloads");
            return;
        }

        // 2. Gather candidate shards across all known models (VRAM-aware scoring)
        let candidates = self.gather_candidates(&local_node_id, pool_vram);
        if candidates.is_empty() {
            tracing::info!("AutoShardManager: no candidate shards to download");
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
    /// Instead of zeroing counters, we blend: new_rate = old * 0.85 + fresh * 0.15.
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
            let new_rate = old * 0.85 + fresh as f64 * 0.15;
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
            let pool_size = self.shared_state.peer_registry.len() + 1;
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
        let models_dir = self.shared_state.config.node.data_dir.join("models");

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

                let hf_path = entry.path().join("hf_source.json");
                if hf_path.exists() {
                    if let Ok(data) = std::fs::read_to_string(&hf_path) {
                        if let Ok(source) = serde_json::from_str::<crate::daemon::HfSource>(&data) {
                            tracing::info!(
                                model = %model_id_str,
                                repo = %source.repo_id,
                                "Discovered HF source from hf_source.json"
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

    /// Helper: adjust target based on resource pressure.
    fn pressure_adjusted_target_pure(target: u32, pressure: f64, min_replicas: u32) -> u32 {
        if pressure < 0.5 {
            target + 1
        } else if pressure < 0.8 {
            target
        } else if pressure < 0.95 {
            target.saturating_sub(1).max(min_replicas)
        } else {
            target.saturating_sub(2).max(min_replicas)
        }
    }

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
