use std::sync::Arc;
use std::time::Duration;

use chrono::Timelike;
use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
use crate::model::manifest::ModelManifestExt;
use crate::types::{ModelId, NetworkCommand, NodeId, ShardId};

/// Estimate VRAM required to run a model based on size and quantization.
///
/// Rule of thumb for quantized GGUF models:
/// - The model weights need ~model_size_bytes of VRAM (already quantized)
/// - KV cache overhead adds ~10-20% on top depending on context length
/// - We use 1.15x multiplier as a conservative estimate
pub fn estimate_model_vram_mb(total_size_bytes: u64) -> u64 {
    // Quantized model weights are already compressed; VRAM ≈ file size + ~15% overhead
    (total_size_bytes as f64 * 1.15 / (1024.0 * 1024.0)) as u64
}

/// Compute the total VRAM available across the entire network (all peers + local node).
pub fn global_pool_vram_mb(shared: &SharedState) -> u64 {
    let mut total = 0u64;

    // Local GPU — use gpu_info if available, fallback to nvidia-smi
    total += local_vram_mb(shared);

    // All known peers
    for peer in shared.peer_registry.iter() {
        if let Some(ref cap) = peer.capability {
            if let Some(ref gpu) = cap.gpu {
                total += gpu.vram_total_mb;
            }
        }
    }

    total
}

/// Get local VRAM in MB, with nvidia-smi fallback when gpu_info is None.
pub fn local_vram_mb(shared: &SharedState) -> u64 {
    if let Some(ref gpu) = shared.gpu_info {
        return gpu.vram_total_mb;
    }
    // Fallback: detect via nvidia-smi
    detect_vram_nvidia_smi().unwrap_or(0)
}

/// Fallback GPU VRAM detection via nvidia-smi.
fn detect_vram_nvidia_smi() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().ok()
}

/// Query live GPU VRAM usage in MB via nvidia-smi.
///
/// Called on each auto-manage tick (~5 min) for accurate VRAM pressure.
/// Returns None if nvidia-smi is unavailable or fails.
fn query_gpu_vram_used() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().ok()
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
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
    /// Notify trigger — woken when new HF sources or manifests arrive from peers.
    notify: Arc<tokio::sync::Notify>,
    /// Semaphore to limit concurrent shard downloads.
    download_semaphore: Arc<tokio::sync::Semaphore>,
}

/// A candidate shard identified for auto-download.
#[derive(Debug, Clone)]
struct ShardCandidate {
    model_id: ModelId,
    model_name: String,
    shard_index: u32,
    shard_size_bytes: u64,
    holder_count: usize,
    /// Score: higher = more worth downloading. Factors in rarity and model popularity.
    score: f64,
}

/// A candidate shard identified for auto-pruning.
#[derive(Debug, Clone)]
struct PruneCandidate {
    model_id: ModelId,
    model_name: String,
    shard_index: u32,
    shard_size_bytes: u64,
    holder_count: usize,
    target_replicas: u32,
    /// Score: higher = more prunable.
    score: f64,
}

impl AutoShardManager {
    pub fn new(
        shared_state: Arc<SharedState>,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let notify = shared_state.auto_manage_notify.clone();
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

        // Skip the first tick (fires immediately) — let the node discover peers first
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

                    // Re-check enabled — admin API can toggle at runtime
                    if self.shared_state.auto_manage_enabled.load(std::sync::atomic::Ordering::Acquire) {
                        self.evaluate_and_download().await;
                        self.evaluate_and_prune().await;
                    }
                }
                _ = self.notify.notified() => {
                    // Woken by a new HfSourceGossip or ModelManifest — wait for gossip
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
                    if self.shared_state.auto_manage_enabled.load(std::sync::atomic::Ordering::Acquire) {
                        tracing::info!("AutoShardManager: triggered by new HF source or manifest");
                        last_notify_eval = std::time::Instant::now();
                        self.evaluate_and_download().await;
                        self.evaluate_and_prune().await;
                    }
                }
                _ = request_reset_interval.tick() => {
                    self.reset_request_counts();
                    self.update_model_trust();
                }
            }
        }
    }

    /// Core logic: evaluate network state and download the best candidate shards.
    async fn evaluate_and_download(&self) {
        let config = &self.shared_state.config.auto_manage;
        let local_node_id = self.shared_state.identity.node_id().clone();

        // Peer warmup grace period: if we have zero peers and just started,
        // wait for peer discovery before evaluating. Prevents a fresh node
        // from immediately downloading everything from HF before it learns
        // that peers already hold shards.
        let peers = self.shared_state.peer_registry.len();
        if peers == 0 {
            let stats = self.shared_state.node_stats.read().await;
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
            tracing::debug!("AutoShardManager: no remaining storage budget");
            return;
        }

        // 2. Gather candidate shards across all known models (VRAM-aware scoring)
        let candidates = self.gather_candidates(&local_node_id, pool_vram);
        if candidates.is_empty() {
            tracing::debug!("AutoShardManager: no candidate shards to download");
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

    /// Compute remaining download budget in bytes.
    fn remaining_budget_bytes(
        &self,
        config: &crate::config::AutoManageConfig,
        local_node_id: &NodeId,
    ) -> u64 {
        let max_bytes = if config.max_storage_mb > 0 {
            config
                .max_storage_mb
                .saturating_mul(1024)
                .saturating_mul(1024)
        } else {
            // Fall back to global max_disk_mb, using 50% for auto-manage
            self.shared_state
                .config
                .resources
                .max_disk_mb
                .saturating_mul(1024)
                .saturating_mul(1024)
                / 2
        };

        // Sum up bytes of shards we already hold
        let mut current_bytes = 0u64;
        let mut current_shard_count = 0u32;
        for manifest in self.shared_state.model_registry.models() {
            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let holders = self.shared_state.model_registry.shard_holders(&shard_id);
                if holders.contains(local_node_id) {
                    current_bytes += shard.size_bytes;
                    current_shard_count += 1;
                }
            }
        }

        // Check max_shards limit
        if config.max_shards > 0 && current_shard_count >= config.max_shards {
            return 0;
        }

        // Scale budget by ContributionMode
        let effective_max = match self.shared_state.config.node.contribution {
            swarmllm_types::ContributionMode::Minimal => max_bytes / 4,
            swarmllm_types::ContributionMode::Moderate => max_bytes,
            swarmllm_types::ContributionMode::Maximum => max_bytes.saturating_mul(3) / 2,
        };

        effective_max.saturating_sub(current_bytes)
    }

    /// Gather all candidate shards we don't already hold, scored by value.
    ///
    /// Scoring factors:
    /// - **configured_bonus** (100x): shards in our `--shards` range missing from disk
    /// - **rarity_bonus** (1-10x): fewer holders → higher priority
    /// - **popularity**: more unique holders across model → higher value
    /// - **vram_fitness** (0.1-1.0x): models that fit in global VRAM pool score higher
    /// - **spread_bonus** (0.05-1.0x): deprioritizes models we already have many shards of
    fn gather_candidates(&self, local_node_id: &NodeId, pool_vram_mb: u64) -> Vec<ShardCandidate> {
        let mut candidates = Vec::new();
        let registry = &self.shared_state.model_registry;
        let shard_store =
            crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
        let configured_range = self.shared_state.config.inference.shard_range;
        let default_cap = self
            .shared_state
            .auto_manage_default_model_cap
            .load(std::sync::atomic::Ordering::Relaxed);

        for manifest in registry.models() {
            // ── Policy gate: skip models excluded from auto-manage ──
            if let Some(policy) = self
                .shared_state
                .model_auto_manage_policies
                .get(&manifest.id)
            {
                if !policy.enabled {
                    tracing::debug!(
                        model = %manifest.id,
                        "Skipping model — auto-manage disabled by policy"
                    );
                    continue;
                }
            }

            // ── Trust gate: skip models not yet verified for auto-propagation ──
            // Exception: if this node already hosts at least one shard, always
            // allow gap-filling regardless of trust level. Only new-model adoption
            // (zero local shards) requires explicit trust / user pinning.
            {
                let trust = self.shared_state.model_trust.get(&manifest.id);
                let trust_level = trust
                    .as_ref()
                    .map(|t| &t.trust_level)
                    .unwrap_or(&crate::types::ModelTrustLevel::Discovered);
                let is_pinned = trust.as_ref().map(|t| t.pinned_by_user).unwrap_or(false);
                let already_hosting = manifest.shards.iter().any(|s| {
                    let sid = ShardId {
                        model_id: manifest.id.clone(),
                        index: s.index,
                    };
                    self.shared_state
                        .model_registry
                        .shard_holders(&sid)
                        .contains(local_node_id)
                });
                if *trust_level < crate::types::ModelTrustLevel::DemandVerified
                    && !is_pinned
                    && !already_hosting
                {
                    tracing::debug!(
                        model = %manifest.id,
                        trust = %trust_level,
                        "Skipping model — insufficient trust for auto-manage"
                    );
                    continue;
                }
            }

            // ── Per-model cap: count local shards, skip if at cap ──
            let local_shard_count = manifest
                .shards
                .iter()
                .filter(|s| {
                    let sid = ShardId {
                        model_id: manifest.id.clone(),
                        index: s.index,
                    };
                    registry.shard_holders(&sid).contains(local_node_id)
                })
                .count() as u32;

            let effective_cap = self
                .shared_state
                .model_auto_manage_policies
                .get(&manifest.id)
                .and_then(|p| {
                    if p.max_shards > 0 {
                        Some(p.max_shards)
                    } else {
                        None
                    }
                })
                .or(if default_cap > 0 {
                    Some(default_cap)
                } else {
                    None
                });

            if let Some(cap) = effective_cap {
                if local_shard_count >= cap {
                    tracing::debug!(
                        model = %manifest.id,
                        local = local_shard_count,
                        cap = cap,
                        "Skipping model — at per-model shard cap"
                    );
                    continue;
                }
            }

            // ── Spread bonus: deprioritize models we already have many shards of ──
            let local_fraction = if manifest.shard_count > 0 {
                local_shard_count as f64 / manifest.shard_count as f64
            } else {
                0.0
            };
            let spread_bonus = (1.0 - local_fraction).max(0.05);

            // Model popularity: count total unique holders across all shards
            let mut all_holders = std::collections::HashSet::new();
            let mut shard_holder_counts: Vec<(u32, usize)> = Vec::new();

            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let holders = registry.shard_holders(&shard_id);
                shard_holder_counts.push((shard.index, holders.len()));
                for h in &holders {
                    all_holders.insert(h.clone());
                }
            }

            // Popularity = number of unique nodes holding any shard of this model.
            // A value of 0 means no one has shards yet (manifest just arrived via gossip).
            // We still want to acquire shards for it if we have an HF source — this is
            // the "complete the set" flow where one node downloads and others follow.
            let model_popularity = (all_holders.len() as f64).max(1.0);

            // VRAM fitness: does the global pool have enough VRAM to actually run this model?
            // Don't block downloads, but deprioritize models the pool can't run yet.
            let model_vram_needed = estimate_model_vram_mb(manifest.total_size_bytes);
            let vram_fitness = if pool_vram_mb == 0 {
                0.5 // No GPU info available, neutral score
            } else if model_vram_needed <= pool_vram_mb {
                1.0 // Model fits in pool — full priority
            } else {
                // Model too large for current pool: scale down but don't zero out
                // ratio < 1.0 → the bigger the gap, the lower the score
                let ratio = pool_vram_mb as f64 / model_vram_needed as f64;
                ratio.max(0.1) // Floor at 0.1x so it's still possible, just deprioritized
            };

            if vram_fitness < 1.0 {
                tracing::debug!(
                    model = %manifest.id,
                    model_vram_mb = model_vram_needed,
                    pool_vram_mb = pool_vram_mb,
                    vram_fitness = vram_fitness,
                    "Model exceeds current pool VRAM — deprioritizing"
                );
            }

            // Average holder count across shards
            let avg_holders = shard_holder_counts
                .iter()
                .map(|(_, c)| *c as f64)
                .sum::<f64>()
                / manifest.shard_count.max(1) as f64;

            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                let holders = registry.shard_holders(&shard_id);

                // Skip if we already hold it (both in registry AND on disk)
                if holders.contains(local_node_id)
                    && shard_store.shard_path(&manifest.id, shard.index).exists()
                {
                    continue;
                }

                // Count peers actively downloading this shard — treat them as
                // near-holders so we don't duplicate their work.
                let peer_dl_count = self
                    .shared_state
                    .peer_shard_downloads
                    .get(&shard_id)
                    .map(|v| v.len())
                    .unwrap_or(0);
                let holder_count = holders.len() + peer_dl_count;

                // Check if this shard is in our configured --shards range but missing
                let in_configured_range = match configured_range {
                    Some((start, end)) => shard.index >= start && shard.index <= end,
                    None => false,
                };

                // Skip shards already held by peers — distributed inference handles
                // cross-node shards. Only download from HF to seed the network
                // (holder_count == 0) or to fill our configured --shards range.
                if holder_count > 0 && !in_configured_range {
                    tracing::debug!(
                        model = %manifest.id,
                        shard = shard.index,
                        holders = holder_count,
                        "Skipping shard — already held by peers"
                    );
                    continue;
                }

                // Small-network deduplication: when there are peers online, use a
                // deterministic assignment so multiple nodes don't all race to download
                // the same unheld shard. Each node "owns" a subset of shard indices
                // based on hash(node_id || model_id || shard_index).
                let peers = self.shared_state.peer_registry.len();
                if holder_count == 0 && peers > 0 && !in_configured_range {
                    let node_count = (peers + 1) as u32; // include self
                                                         // Use hash to assign this shard to a specific node slot
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(manifest.id.0.as_bytes());
                    hasher.update(&shard.index.to_le_bytes());
                    let hash = hasher.finalize();
                    let assigned_slot = u32::from_le_bytes([
                        hash.as_bytes()[0],
                        hash.as_bytes()[1],
                        hash.as_bytes()[2],
                        hash.as_bytes()[3],
                    ]) % node_count;
                    // Our slot: hash(node_id || model_id) % node_count
                    let mut my_hasher = blake3::Hasher::new();
                    my_hasher.update(&local_node_id.0);
                    my_hasher.update(manifest.id.0.as_bytes());
                    let my_hash = my_hasher.finalize();
                    let my_slot = u32::from_le_bytes([
                        my_hash.as_bytes()[0],
                        my_hash.as_bytes()[1],
                        my_hash.as_bytes()[2],
                        my_hash.as_bytes()[3],
                    ]) % node_count;
                    if assigned_slot != my_slot {
                        tracing::debug!(
                            model = %manifest.id,
                            shard = shard.index,
                            assigned_slot,
                            my_slot,
                            node_count,
                            "Skipping shard — assigned to different node slot"
                        );
                        continue;
                    }
                }

                // Skip shards already being downloaded on THIS node (explicit or
                // auto-manage). Prevents racing with an in-flight download.
                if let Some(acq) = self.shared_state.acquisition_progress.get(&manifest.id) {
                    if let Some(sp) = acq.shard_progress.get(&shard.index) {
                        if matches!(
                            sp.state,
                            crate::model::acquisition::ShardState::Downloading
                                | crate::model::acquisition::ShardState::Pending
                                | crate::model::acquisition::ShardState::Verifying
                        ) {
                            tracing::debug!(
                                model = %manifest.id,
                                shard = shard.index,
                                "Skipping shard — already downloading on this node"
                            );
                            continue;
                        }
                    }
                }

                // Score = popularity * rarity_bonus * configured_bonus * vram_fitness
                // - configured_bonus: 100x for shards we're supposed to serve (highest priority)
                // - rarity_bonus: higher when this shard has fewer holders than average
                // - vram_fitness: 0.1-1.0x based on whether pool can run the model
                let rarity_bonus = if holder_count == 0 {
                    10.0 // Very high priority for zero-holder shards
                } else {
                    (avg_holders + 1.0) / (holder_count as f64 + 1.0)
                };

                let configured_bonus = if in_configured_range { 100.0 } else { 1.0 };

                // Node-specific jitter (0.0–0.1) so nodes with identical views
                // of the network don't all pick the same shard to download.
                // BLAKE3(node_id || model_id || shard_index) → deterministic per-node tiebreaker.
                let jitter = {
                    let mut hasher = blake3::Hasher::new();
                    hasher.update(&local_node_id.0);
                    hasher.update(manifest.id.0.as_bytes());
                    hasher.update(&shard.index.to_le_bytes());
                    let hash = hasher.finalize();
                    hash.as_bytes()[0] as f64 / 2550.0 // 0.0–0.1 range
                };

                let score = model_popularity
                    * rarity_bonus
                    * configured_bonus
                    * vram_fitness
                    * spread_bonus
                    + jitter;

                candidates.push(ShardCandidate {
                    model_id: manifest.id.clone(),
                    model_name: manifest.name.clone(),
                    shard_index: shard.index,
                    shard_size_bytes: shard.size_bytes,
                    holder_count,
                    score,
                });
            }

            // ── T7: mmproj as download candidate for VLM models ──
            if let Some(ref mmproj_info) = manifest.mmproj {
                let mmproj_shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: crate::types::MMPROJ_SHARD_INDEX,
                };
                let mmproj_holders = registry.shard_holders(&mmproj_shard_id);
                let mmproj_path = self
                    .shared_state
                    .config
                    .node
                    .data_dir
                    .join("models")
                    .join(&manifest.id.0)
                    .join("mmproj.gguf");

                if !mmproj_holders.contains(local_node_id) || !mmproj_path.exists() {
                    let holder_count = mmproj_holders.len();
                    // mmproj gets a high priority bonus — every VLM node benefits from having it
                    let rarity_bonus = if holder_count == 0 {
                        10.0
                    } else {
                        3.0 / (holder_count as f64 + 1.0)
                    };
                    let mmproj_score = model_popularity * rarity_bonus * vram_fitness * 5.0; // 5x bonus for mmproj

                    candidates.push(ShardCandidate {
                        model_id: manifest.id.clone(),
                        model_name: manifest.name.clone(),
                        shard_index: crate::types::MMPROJ_SHARD_INDEX,
                        shard_size_bytes: mmproj_info.size_bytes,
                        holder_count,
                        score: mmproj_score,
                    });
                }
            }
        }

        // Sort by score descending (best candidates first)
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // If any configured-range shards are missing for the primary model,
        // focus exclusively on those first. Don't download extra shards until
        // our assigned range is complete.
        if let Some((start, end)) = configured_range {
            // Find the model that matches our configured shard range
            // (the one we were started with via --model + --shards)
            let configured_model: Option<ModelId> = registry
                .models()
                .iter()
                .find(|m| m.shards.iter().any(|s| s.index >= start && s.index <= end))
                .map(|m| m.id.clone());

            if let Some(ref mid) = configured_model {
                let has_configured_missing = candidates
                    .iter()
                    .any(|c| c.model_id == *mid && c.shard_index >= start && c.shard_index <= end);
                if has_configured_missing {
                    candidates.retain(|c| {
                        c.model_id == *mid && c.shard_index >= start && c.shard_index <= end
                    });
                }
            }
        }

        candidates
    }

    /// Select candidates that fit within the remaining budget.
    fn select_within_budget(
        &self,
        candidates: Vec<ShardCandidate>,
        mut budget_bytes: u64,
        max_shards: u32,
    ) -> Vec<ShardCandidate> {
        let mut selected = Vec::new();
        let max = if max_shards > 0 {
            max_shards as usize
        } else {
            usize::MAX
        };

        // Per-cycle cap scaled by ContributionMode:
        // Minimal: 1 shard/cycle regardless of network size
        // Moderate: 1-2 based on peer count (original behavior)
        // Maximum: 2-4 for aggressive seeding
        let peers = self.shared_state.peer_registry.len();
        let per_cycle_cap = match self.shared_state.config.node.contribution {
            swarmllm_types::ContributionMode::Minimal => 1,
            swarmllm_types::ContributionMode::Moderate => {
                if peers < 5 {
                    1
                } else {
                    2
                }
            }
            swarmllm_types::ContributionMode::Maximum => {
                if peers < 5 {
                    2
                } else {
                    4
                }
            }
        };

        // Track which specific shards are currently downloading so we don't
        // start a duplicate. We check per-shard progress, NOT per-model — otherwise
        // downloading shard 0 would block acquisition of shard 1.
        let downloading_shards: std::collections::HashSet<(String, u32)> = {
            let mut set = std::collections::HashSet::new();
            for entry in self.shared_state.acquisition_progress.iter() {
                if !matches!(
                    entry.value().state,
                    crate::model::acquisition::AcquisitionState::Downloading
                ) {
                    continue;
                }
                let mid = entry.key().0.clone();
                for (&idx, sp) in &entry.value().shard_progress {
                    if matches!(sp.state, crate::model::acquisition::ShardState::Downloading) {
                        set.insert((mid.clone(), idx));
                    }
                }
            }
            set
        };

        // Round-robin interleaving: group candidates by model, take one per model
        // in rotation instead of pure score-descending. This ensures shards from
        // different models are downloaded in the same cycle when possible.
        let mut by_model: std::collections::HashMap<String, Vec<ShardCandidate>> =
            std::collections::HashMap::new();
        let mut model_order: Vec<String> = Vec::new();
        for candidate in candidates {
            // Skip this specific shard if it's already being downloaded
            if downloading_shards.contains(&(candidate.model_id.0.clone(), candidate.shard_index)) {
                continue;
            }
            if !by_model.contains_key(&candidate.model_id.0) {
                model_order.push(candidate.model_id.0.clone());
            }
            by_model
                .entry(candidate.model_id.0.clone())
                .or_default()
                .push(candidate);
        }

        // Round-robin: take one candidate from each model in order
        let mut model_indices: Vec<usize> = vec![0; model_order.len()];
        'outer: loop {
            let mut any_taken = false;
            for (mi, model_key) in model_order.iter().enumerate() {
                if selected.len() >= max || selected.len() >= per_cycle_cap {
                    break 'outer;
                }
                let candidates_for_model = &by_model[model_key];
                while model_indices[mi] < candidates_for_model.len() {
                    let candidate = &candidates_for_model[model_indices[mi]];
                    model_indices[mi] += 1;
                    if candidate.shard_size_bytes <= budget_bytes {
                        budget_bytes -= candidate.shard_size_bytes;
                        selected.push(candidate.clone());
                        any_taken = true;
                        break; // move to next model
                    }
                }
            }
            if !any_taken {
                break;
            }
        }

        selected
    }

    /// Trigger download of a single shard.
    ///
    /// Strategy: try peers first if any hold the shard, fall back to HuggingFace.
    /// After download, register the shard and check if the model is now complete.
    /// Acquires a semaphore permit to limit concurrent downloads.
    async fn trigger_download(&self, candidate: &ShardCandidate) {
        // Acquire semaphore permit to limit concurrent downloads.
        // The permit is moved into the spawned task and dropped on completion.
        let permit = match self.download_semaphore.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => {
                tracing::warn!("Download semaphore closed, skipping download");
                return;
            }
        };

        tracing::info!(
            model = %candidate.model_id,
            shard = candidate.shard_index,
            holders = candidate.holder_count,
            score = candidate.score,
            "AutoShardManager: requesting shard download"
        );

        let model_dir = self.shared_state.config.node.data_dir.join("models").join(
            crate::model::shard::sanitize_path_component(&candidate.model_id.0),
        );

        // ── T8: mmproj full-file download (not byte-range) ──
        if candidate.shard_index == crate::types::MMPROJ_SHARD_INDEX {
            self.trigger_mmproj_download(candidate, model_dir, permit)
                .await;
            return;
        }

        // Check if we already have the shard file locally.
        // Guard: only treat it as complete if there is NO active download for this
        // shard AND the file size matches expected.  A partially-downloaded file
        // will exist on disk but be smaller than `shard_size_bytes`.
        let shard_path = model_dir.join(format!("shard_{:03}.bin", candidate.shard_index));
        if shard_path.exists() {
            // Check if this shard is currently being downloaded (by API handler or another cycle)
            let is_downloading = self
                .shared_state
                .acquisition_progress
                .get(&candidate.model_id)
                .map(|entry| {
                    entry
                        .shard_progress
                        .get(&candidate.shard_index)
                        .map(|sp| sp.state == crate::model::acquisition::ShardState::Downloading)
                        .unwrap_or(false)
                })
                .unwrap_or(false);

            if is_downloading {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Shard file exists but download is in progress, skipping"
                );
                return;
            }

            // Verify shard integrity: try BLAKE3 hash if available, fall back to size check
            let shard_store =
                crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
            let file_ok = if let Some(manifest) = self
                .shared_state
                .model_registry
                .get_manifest(&candidate.model_id)
            {
                if let Some(shard_info) = manifest
                    .shards
                    .iter()
                    .find(|s| s.index == candidate.shard_index)
                {
                    if shard_info.hash != [0u8; 32] {
                        // Hash available — verify properly
                        shard_store
                            .verify_shard(&candidate.model_id, shard_info)
                            .is_ok()
                    } else if candidate.shard_size_bytes > 0 {
                        // Zero-hash placeholder — fall back to size check
                        std::fs::metadata(&shard_path)
                            .map(|m| m.len() >= candidate.shard_size_bytes * 9 / 10)
                            .unwrap_or(false)
                    } else {
                        false // unknown expected size — needs re-download
                    }
                } else if candidate.shard_size_bytes > 0 {
                    std::fs::metadata(&shard_path)
                        .map(|m| m.len() >= candidate.shard_size_bytes * 9 / 10)
                        .unwrap_or(false)
                } else {
                    false
                }
            } else if candidate.shard_size_bytes > 0 {
                std::fs::metadata(&shard_path)
                    .map(|m| m.len() >= candidate.shard_size_bytes * 9 / 10)
                    .unwrap_or(false)
            } else {
                false
            };

            if file_ok {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Shard file already exists on disk, registering"
                );
                self.register_local_shard(candidate);
                self.check_model_complete(&candidate.model_id).await;
                return;
            } else {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Shard file exists but is too small (partial download?), re-downloading"
                );
                // Fall through to download
            }
        }

        let mid = candidate.model_id.clone();

        // NOTE: We do NOT send a ShardAnnounce before the download starts.
        // Premature announces cause peers to register us as a holder before
        // the shard is actually on disk, making the UI show "peer-held" instead
        // of "peer-downloading".  The ShardDownloadProgress gossip broadcasts
        // our progress, and the completion message triggers holder registration
        // on remote nodes.

        // Download from HuggingFace if source is known
        if let Some(hf_source) = self.shared_state.hf_sources.get(&candidate.model_id) {
            // Create progress entry with per-shard tracking for the specific shard
            let mut shard_progress = std::collections::HashMap::new();
            shard_progress.insert(
                candidate.shard_index,
                crate::model::acquisition::ShardProgress {
                    index: candidate.shard_index,
                    total_bytes: candidate.shard_size_bytes,
                    downloaded_bytes: 0,
                    state: crate::model::acquisition::ShardState::Downloading,
                },
            );
            // Merge with existing progress entry rather than overwriting.
            // Multiple shards of the same model may be downloading concurrently
            // and each needs its own shard_progress entry tracked.
            if let Some(mut entry) = self.shared_state.acquisition_progress.get_mut(&mid) {
                entry.state = crate::model::acquisition::AcquisitionState::Downloading;
                // Set total_shards from the manifest, not by incrementing
                // (incrementing causes inflated counts when merging progress entries)
                if let Some(manifest) = self.shared_state.model_registry.get_manifest(&mid) {
                    entry.total_shards = manifest.shard_count;
                    entry.total_bytes = manifest.total_size_bytes;
                }
                // Only add this shard's progress if not already tracked
                entry
                    .shard_progress
                    .entry(candidate.shard_index)
                    .or_insert_with(|| crate::model::acquisition::ShardProgress {
                        index: candidate.shard_index,
                        total_bytes: candidate.shard_size_bytes,
                        downloaded_bytes: 0,
                        state: crate::model::acquisition::ShardState::Downloading,
                    });
                entry.log.push(format!(
                    "Auto-manage: downloading shard {} (score: {:.1})",
                    candidate.shard_index, candidate.score
                ));
            } else {
                let total_shards = self
                    .shared_state
                    .model_registry
                    .get_manifest(&mid)
                    .map(|m| m.shard_count)
                    .unwrap_or(1);
                let status = crate::model::acquisition::AcquisitionStatus {
                    model_id: mid.clone(),
                    state: crate::model::acquisition::AcquisitionState::Downloading,
                    total_shards,
                    downloaded_shards: 0,
                    verified_shards: 0,
                    failed_shards: 0,
                    total_bytes: candidate.shard_size_bytes,
                    downloaded_bytes: 0,
                    shard_progress,
                    speed_bytes_per_sec: 0,
                    started_at: Some(chrono::Utc::now()),
                    log: vec![format!(
                        "Auto-manage: downloading shard {} of {} (score: {:.1})",
                        candidate.shard_index, candidate.model_name, candidate.score
                    )],
                };
                self.shared_state
                    .acquisition_progress
                    .insert(mid.clone(), status);
            }
            let repo_id = hf_source.repo_id.clone();
            let filename = hf_source.filename.clone();
            drop(hf_source); // release DashMap ref

            let shared = self.shared_state.clone();
            let model_id = candidate.model_id.clone();
            let shard_idx = candidate.shard_index;
            let dest = model_dir.clone();

            tracing::info!(
                model = %model_id,
                shard = shard_idx,
                repo = %repo_id,
                "AutoShardManager: downloading shard from HuggingFace"
            );

            let net_tx = self.network_tx.clone();

            // Spawn the download so we don't block the evaluation loop.
            // The semaphore permit is moved into the task and dropped on completion,
            // releasing the slot for the next download.
            tokio::spawn(async move {
                let _permit = permit; // Hold permit for duration of download
                let (ptx, mut prx) =
                    tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(32);

                // Progress updater — updates per-shard progress + broadcasts to network
                let prog_mid = model_id.clone();
                let prog_shared = shared.clone();
                let prog_net_tx = net_tx.clone();
                tokio::spawn(async move {
                    let mut last_broadcast_pct: u32 = 0;
                    while let Some(prog) = prx.recv().await {
                        let pct = if prog.total_bytes > 0 {
                            (prog.downloaded_bytes as f64 / prog.total_bytes as f64 * 100.0) as u32
                        } else {
                            0
                        };

                        if let Some(mut entry) = prog_shared.acquisition_progress.get_mut(&prog_mid)
                        {
                            entry.downloaded_bytes = prog.downloaded_bytes;
                            entry.total_bytes = prog.total_bytes;
                            // Update per-shard progress
                            if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                                sp.downloaded_bytes = prog.downloaded_bytes;
                                sp.total_bytes = prog.total_bytes;
                            }
                        }

                        // Broadcast progress every 5% to avoid gossip flood
                        if pct >= last_broadcast_pct + 5 || pct == 100 {
                            last_broadcast_pct = pct;
                            let progress_msg = crate::types::SwarmMessage::ShardDownloadProgress(
                                crate::types::ShardDownloadProgress {
                                    node_id: prog_shared.identity.node_id().clone(),
                                    shard_id: crate::types::ShardId {
                                        model_id: prog_mid.clone(),
                                        index: shard_idx,
                                    },
                                    progress_pct: pct,
                                    state: crate::types::DownloadState::Downloading,
                                },
                            );
                            let _ = prog_net_tx
                                .try_send(crate::types::NetworkCommand::Broadcast(progress_msg));
                        }
                    }
                });

                // Probe to get v2 layouts, then download the specific shard
                let configured_shard_size = shared.config.model.shard_size_bytes();
                let probe_result = crate::model::huggingface::probe_gguf_file(
                    &repo_id,
                    &filename,
                    configured_shard_size,
                )
                .await;
                let info = match probe_result {
                    Ok(info) => info,
                    Err(e) => {
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_idx,
                            error = %e,
                            "AutoShardManager: GGUF probe failed"
                        );
                        if let Some(mut entry) = shared.acquisition_progress.get_mut(&model_id) {
                            entry.state = crate::model::acquisition::AcquisitionState::Failed {
                                reason: format!("GGUF probe failed: {}", e),
                            };
                        }
                        return;
                    }
                };
                // Check architecture support before downloading
                let arch_str = &info.tensor_meta.architecture;
                let arch = crate::inference::split::ModelArch::from_gguf_arch(arch_str);
                if !arch.is_supported() {
                    tracing::warn!(
                        model = %model_id,
                        arch = %arch_str,
                        "AutoShardManager: skipping unsupported architecture"
                    );
                    return;
                }

                let layout = match info.layouts.get(shard_idx as usize) {
                    Some(l) => l,
                    None => {
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_idx,
                            total_shards = info.shard_count(),
                            "AutoShardManager: shard index out of range"
                        );
                        return;
                    }
                };

                // Download header + tied output weight (if weight-tied) + shard
                if let Err(e) = crate::model::huggingface::download_gguf_header(
                    &repo_id,
                    &filename,
                    &dest,
                    info.header_size,
                )
                .await
                {
                    tracing::warn!(
                        model = %model_id,
                        shard = shard_idx,
                        error = %e,
                        "AutoShardManager: gguf_header.bin download failed — shard registered but first-segment local inference will be unavailable until header is re-downloaded"
                    );
                }

                // Download tied output weight for weight-tied models
                if let Err(e) = crate::model::huggingface::download_tied_output_weight(
                    &repo_id,
                    &filename,
                    &dest,
                    &info.tensor_meta,
                )
                .await
                {
                    tracing::warn!(error = %e, "Tied output weight download failed (non-fatal)");
                }

                match crate::model::huggingface::download_shard_v2(
                    &repo_id,
                    &filename,
                    &dest,
                    layout,
                    Some(ptx),
                    None,
                )
                .await
                {
                    Ok(_shard_path) => {
                        tracing::info!(
                            model = %model_id,
                            shard = shard_idx,
                            "AutoShardManager: shard downloaded from HF"
                        );

                        // Verify the downloaded shard before registering
                        let shard_store =
                            crate::model::shard::ShardStore::new(&shared.config.node.data_dir);
                        if let Some(manifest) = shared.model_registry.get_manifest(&model_id) {
                            if let Some(shard_info) =
                                manifest.shards.iter().find(|s| s.index == shard_idx)
                            {
                                // Use allow_zero_hash=true since HF downloads may have placeholder hashes
                                if let Err(e) = shard_store
                                    .verify_shard_with_options(&model_id, shard_info, true)
                                {
                                    tracing::warn!(
                                        model = %model_id,
                                        shard = shard_idx,
                                        error = %e,
                                        "AutoShardManager: HF shard failed verification — not registering"
                                    );
                                    if let Some(mut entry) =
                                        shared.acquisition_progress.get_mut(&model_id)
                                    {
                                        entry.state =
                                            crate::model::acquisition::AcquisitionState::Failed {
                                                reason: format!(
                                                    "Shard {} verification failed: {}",
                                                    shard_idx, e
                                                ),
                                            };
                                        entry.log.push(format!(
                                            "Shard {} failed verification",
                                            shard_idx
                                        ));
                                    }
                                    return;
                                }
                            }
                        }

                        // Compute BLAKE3 hash of the downloaded shard and update the manifest
                        // so startup verification passes on restart.
                        // block_in_place: reads up to 1GB shard + CPU-intensive hash
                        let shard_path = dest.join(format!("shard_{:03}.bin", shard_idx));
                        let hash_result: Option<[u8; 32]> = tokio::task::block_in_place(|| {
                            std::fs::read(&shard_path)
                                .ok()
                                .map(|data| *blake3::hash(&data).as_bytes())
                        });
                        if let Some(hash) = hash_result {
                            if let Some(mut manifest) =
                                shared.model_registry.get_manifest(&model_id)
                            {
                                if let Some(si) =
                                    manifest.shards.iter_mut().find(|s| s.index == shard_idx)
                                {
                                    si.hash = hash;
                                }
                                manifest.manifest_hash = manifest.compute_hash();
                                let model_dir =
                                    shared.config.node.data_dir.join("models").join(&model_id.0);
                                if let Err(e) = manifest.save_to_dir(&model_dir) {
                                    tracing::warn!(
                                        model = %model_id,
                                        error = %e,
                                        "AutoShardManager: failed to persist manifest after shard hash update — hash in memory only"
                                    );
                                }
                                shared.model_registry.register_manifest(manifest);
                            }
                        }

                        // Register the shard
                        let node_id = shared.identity.node_id().clone();
                        let sid = crate::types::ShardId {
                            model_id: model_id.clone(),
                            index: shard_idx,
                        };
                        shared
                            .model_registry
                            .record_shard_holder(sid.clone(), node_id.clone());

                        // Update progress
                        if let Some(mut entry) = shared.acquisition_progress.get_mut(&model_id) {
                            entry.state = crate::model::acquisition::AcquisitionState::Complete;
                            entry.downloaded_shards = 1;
                            entry.verified_shards = 1;
                            if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                                sp.state = crate::model::acquisition::ShardState::Complete;
                                sp.downloaded_bytes = sp.total_bytes;
                            }
                            entry.log.push("Shard downloaded and registered".into());
                        }

                        // Broadcast completion to network
                        let complete_msg = crate::types::SwarmMessage::ShardDownloadProgress(
                            crate::types::ShardDownloadProgress {
                                node_id: node_id.clone(),
                                shard_id: sid.clone(),
                                progress_pct: 100,
                                state: crate::types::DownloadState::Complete,
                            },
                        );
                        let _ =
                            net_tx.try_send(crate::types::NetworkCommand::Broadcast(complete_msg));

                        // Now that the shard is on disk, announce it to the network
                        // so peers register us as a holder.
                        let announce = crate::types::SwarmMessage::ShardAnnounce(
                            crate::types::ShardAnnounce {
                                node_id,
                                shards: vec![sid],
                                timestamp: chrono::Utc::now(),
                            },
                        );
                        let _ = net_tx.try_send(crate::types::NetworkCommand::Broadcast(announce));

                        // Load whatever shards are now available for inference
                        let vram_budget = compute_vram_budget(&shared);
                        check_and_load_model(&shared, &model_id, vram_budget).await;

                        // Notify dashboard that models have changed
                        let _ = shared.models_changed_tx.send(());

                        // Self-wake so we immediately re-evaluate and download
                        // more shards (libp2p gossipsub doesn't deliver our own
                        // broadcasts back to us, so we must notify ourselves).
                        shared.auto_manage_notify.notify_one();

                        // Clean up acquisition_progress after a delay so the
                        // frontend sees "complete" before we remove it.
                        let cleanup_shared = shared.clone();
                        let cleanup_mid = model_id.clone();
                        tokio::spawn(async move {
                            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                            cleanup_shared.acquisition_progress.remove(&cleanup_mid);
                        });
                    }
                    Err(e) => {
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_idx,
                            error = %e,
                            "AutoShardManager: HF shard download failed"
                        );
                        if let Some(mut entry) = shared.acquisition_progress.get_mut(&model_id) {
                            entry.state =
                                crate::model::acquisition::AcquisitionState::Failed { reason: e };
                            entry.log.push("HF download failed".into());
                        }
                    }
                }
            });
        } else {
            tracing::debug!(
                model = %candidate.model_id,
                shard = candidate.shard_index,
                "No HF source known — relying on peer shard transfer"
            );
        }
    }

    /// Download mmproj (vision encoder) as a full file from HuggingFace.
    /// Unlike text shards which use byte-range downloads, mmproj is a separate GGUF file.
    async fn trigger_mmproj_download(
        &self,
        candidate: &ShardCandidate,
        model_dir: std::path::PathBuf,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let mmproj_path = model_dir.join("mmproj.gguf");
        if mmproj_path.exists() {
            // Already on disk — just register the sentinel shard
            let node_id = self.shared_state.identity.node_id().clone();
            let shard_id = ShardId {
                model_id: candidate.model_id.clone(),
                index: crate::types::MMPROJ_SHARD_INDEX,
            };
            self.shared_state
                .model_registry
                .record_shard_holder(shard_id, node_id);
            tracing::info!(model = %candidate.model_id, "mmproj already on disk, registered sentinel shard");
            return;
        }

        // Look up mmproj_filename from HfSource
        let mmproj_filename = self
            .shared_state
            .hf_sources
            .get(&candidate.model_id)
            .and_then(|s| s.mmproj_filename.clone());

        let Some(filename) = mmproj_filename else {
            tracing::debug!(
                model = %candidate.model_id,
                "No mmproj_filename in HfSource — cannot download mmproj"
            );
            return;
        };

        let repo_id = self
            .shared_state
            .hf_sources
            .get(&candidate.model_id)
            .map(|s| s.repo_id.clone());

        let Some(repo_id) = repo_id else {
            return;
        };

        let shared = self.shared_state.clone();
        let model_id = candidate.model_id.clone();
        let net_tx = self.network_tx.clone();

        tracing::info!(
            model = %model_id,
            repo = %repo_id,
            filename = %filename,
            "AutoShardManager: downloading mmproj from HuggingFace"
        );

        tokio::spawn(async move {
            let _permit = permit;
            let (ptx, _prx) =
                tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(32);

            match crate::model::huggingface::download_model(
                &repo_id,
                &filename,
                &model_dir,
                Some(ptx),
            )
            .await
            {
                Ok(_path) => {
                    tracing::info!(
                        model = %model_id,
                        "AutoShardManager: mmproj downloaded from HF"
                    );
                    // Register sentinel shard
                    let node_id = shared.identity.node_id().clone();
                    let sid = crate::types::ShardId {
                        model_id: model_id.clone(),
                        index: crate::types::MMPROJ_SHARD_INDEX,
                    };
                    shared
                        .model_registry
                        .record_shard_holder(sid.clone(), node_id.clone());

                    // Broadcast shard announce so peers know we hold mmproj
                    let announce =
                        crate::types::SwarmMessage::ShardAnnounce(crate::types::ShardAnnounce {
                            node_id,
                            shards: vec![sid],
                            timestamp: chrono::Utc::now(),
                        });
                    let _ = net_tx.try_send(crate::types::NetworkCommand::Broadcast(announce));

                    // Notify dashboard
                    let _ = shared.models_changed_tx.send(());
                }
                Err(e) => {
                    tracing::warn!(
                        model = %model_id,
                        error = %e,
                        "AutoShardManager: mmproj download failed"
                    );
                }
            }
        });
    }

    /// Register a shard file that already exists on disk.
    fn register_local_shard(&self, candidate: &ShardCandidate) {
        tracing::debug!(
            model = %candidate.model_id,
            shard = candidate.shard_index,
            "DIAG: register_local_shard"
        );
        let node_id = self.shared_state.identity.node_id().clone();
        let shard_id = ShardId {
            model_id: candidate.model_id.clone(),
            index: candidate.shard_index,
        };
        self.shared_state
            .model_registry
            .record_shard_holder(shard_id, node_id);
    }

    /// Check if any local shards are available for this model and load them.
    /// A node does NOT need all shards — it loads whatever it has and participates
    /// in distributed inference for the layers it covers.
    async fn check_model_complete(&self, model_id: &ModelId) {
        let vram_budget = compute_vram_budget(&self.shared_state);
        check_and_load_model(&self.shared_state, model_id, vram_budget).await;
        let _ = self.shared_state.models_changed_tx.send(());
    }

    /// Evaluate and prune over-replicated shards. Called after downloads in each cycle.
    async fn evaluate_and_prune(&self) {
        let config = &self.shared_state.config.auto_manage;
        if !config.prune_enabled {
            return;
        }

        let local_node_id = self.shared_state.identity.node_id().clone();
        let registry = &self.shared_state.model_registry;
        let shard_store =
            crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);

        // Compute resource pressure
        let resource_pressure = self.compute_resource_pressure();
        let pressure_urgent = resource_pressure > 0.95;
        tracing::info!(
            resource_pressure = format!("{:.2}", resource_pressure),
            pressure_urgent,
            "DIAG: evaluate_and_prune starting"
        );

        // Check if we're in reduced hours
        let schedule_pressure = self.schedule_pressure_bonus().await;

        // Gather per-model request counts for popularity
        let request_counts: std::collections::HashMap<ModelId, u64> = self
            .shared_state
            .model_request_counts
            .iter()
            .map(|e| {
                (
                    e.key().clone(),
                    e.value().load(std::sync::atomic::Ordering::Relaxed),
                )
            })
            .collect();

        let pool_size = self.shared_state.peer_registry.len() + 1; // +1 for us

        // Track how many shards pruned per model in this cycle
        let mut pruned_per_model: std::collections::HashMap<ModelId, u32> =
            std::collections::HashMap::new();

        // Collect prune candidates across all models
        let mut prune_candidates: Vec<PruneCandidate> = Vec::new();

        for manifest in registry.models() {
            // Check per-model prune policy
            if let Some(policy) = self
                .shared_state
                .model_auto_manage_policies
                .get(&manifest.id)
            {
                if !policy.prune_enabled {
                    continue;
                }
            }

            // Check cooldown
            if let Some(last_prune) = self.last_prune_time(&manifest.id) {
                let elapsed = chrono::Utc::now()
                    .signed_duration_since(last_prune)
                    .num_seconds()
                    .max(0) as u64;
                if elapsed < config.prune_cooldown_secs {
                    continue;
                }
            }

            // Compute target replicas for this model
            let request_count = request_counts.get(&manifest.id).copied().unwrap_or(0);
            let target = self.target_replicas(request_count, config.min_replicas, pool_size);

            // Adjust target for resource pressure
            let adjusted_target =
                self.pressure_adjusted_target(target, resource_pressure, config.min_replicas);

            for shard in &manifest.shards {
                let shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };

                // Only consider shards we hold locally
                let holders = registry.shard_holders(&shard_id);
                if !holders.contains(&local_node_id) {
                    continue;
                }

                // Skip locked/pinned shards
                if self.shared_state.locked_shards.contains_key(&shard_id) {
                    continue;
                }

                // Skip if in configured --shards range
                if let Some((start, end)) = self.shared_state.config.inference.shard_range {
                    if shard.index >= start && shard.index <= end {
                        continue;
                    }
                }

                let holder_count = holders.len();

                // Skip if at or below target
                if holder_count <= adjusted_target as usize {
                    continue;
                }

                // Region-aware: block if we'd eliminate last holder in our region
                if self.would_eliminate_region(&shard_id, &local_node_id, &holders) {
                    continue;
                }

                // Load-aware: block if remaining holders are busy
                if self.remaining_holders_busy(
                    &holders,
                    &local_node_id,
                    config.max_holder_load_for_prune,
                ) {
                    continue;
                }

                // Skip if model actively loaded and used recently (unless pressure-urgent)
                if !pressure_urgent && self.shard_recently_used(&manifest.id) {
                    continue;
                }

                // Re-acquisition check: can we get it back?
                if !self.can_reacquire(&manifest.id, &shard_id, &holders, &local_node_id) {
                    continue;
                }

                // Compute prune score (higher = more prunable)
                let redundancy_ratio = holder_count as f64 / adjusted_target.max(1) as f64;
                let mut score = redundancy_ratio;

                // Cold shard bonus (not loaded in VRAM)
                let is_loaded = self
                    .shared_state
                    .split_models
                    .iter()
                    .any(|entry| entry.key().0 == manifest.id);
                if !is_loaded {
                    score += 1.0;
                }

                // Resource pressure bonus
                score += 0.5 * (resource_pressure + schedule_pressure);

                // Pipeline completeness penalty
                if shard.index == 0 || shard.index == manifest.shard_count.saturating_sub(1) {
                    score -= 0.5;
                }

                // Rarest shard penalty
                let min_holders = manifest
                    .shards
                    .iter()
                    .map(|s| {
                        let sid = ShardId {
                            model_id: manifest.id.clone(),
                            index: s.index,
                        };
                        registry.shard_holders(&sid).len()
                    })
                    .min()
                    .unwrap_or(0);
                if holder_count == min_holders {
                    score -= 0.3;
                }

                // Recently acquired penalty (< 30 min)
                // Use file modified time as proxy
                let shard_path = shard_store.shard_path(&manifest.id, shard.index);
                if let Ok(meta) = std::fs::metadata(&shard_path) {
                    if let Ok(modified) = meta.modified() {
                        let age = modified.elapsed().unwrap_or_default();
                        if age < Duration::from_secs(1800) {
                            score -= 0.2;
                        }
                    }
                }

                prune_candidates.push(PruneCandidate {
                    model_id: manifest.id.clone(),
                    model_name: manifest.name.clone(),
                    shard_index: shard.index,
                    shard_size_bytes: shard.size_bytes,
                    holder_count,
                    target_replicas: adjusted_target,
                    score,
                });
            }

            // ── T9: mmproj pruning with higher min_replicas floor ──
            // mmproj needs wider availability for VLM requests, so use a higher floor.
            if manifest.mmproj.is_some() {
                let mmproj_shard_id = ShardId {
                    model_id: manifest.id.clone(),
                    index: crate::types::MMPROJ_SHARD_INDEX,
                };
                let mmproj_holders = registry.shard_holders(&mmproj_shard_id);
                if mmproj_holders.contains(&local_node_id) {
                    let mmproj_path = self
                        .shared_state
                        .config
                        .node
                        .data_dir
                        .join("models")
                        .join(&manifest.id.0)
                        .join("mmproj.gguf");
                    if mmproj_path.exists() {
                        // Higher floor: at least 3 replicas (or pool_size, whichever is smaller)
                        let mmproj_min = (config.min_replicas + 1).min(pool_size as u32).max(3);
                        let mmproj_holder_count = mmproj_holders.len();
                        if mmproj_holder_count > mmproj_min as usize && pressure_urgent {
                            let mmproj_size =
                                manifest.mmproj.as_ref().map(|m| m.size_bytes).unwrap_or(0);
                            prune_candidates.push(PruneCandidate {
                                model_id: manifest.id.clone(),
                                model_name: manifest.name.clone(),
                                shard_index: crate::types::MMPROJ_SHARD_INDEX,
                                shard_size_bytes: mmproj_size,
                                holder_count: mmproj_holder_count,
                                target_replicas: mmproj_min,
                                score: 0.1, // Very low score — only prune under extreme pressure
                            });
                        }
                    }
                }
            }
        }

        // Sort by score descending (most prunable first)
        prune_candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Execute pruning with per-model limits
        let max_per_model = if pressure_urgent { 2u32 } else { 1u32 };

        for candidate in &prune_candidates {
            let count = pruned_per_model
                .get(&candidate.model_id)
                .copied()
                .unwrap_or(0);
            if count >= max_per_model {
                continue;
            }

            // Actually delete the shard file (or mmproj.gguf for sentinel)
            let shard_path = if candidate.shard_index == crate::types::MMPROJ_SHARD_INDEX {
                self.shared_state
                    .config
                    .node
                    .data_dir
                    .join("models")
                    .join(crate::model::shard::sanitize_path_component(
                        &candidate.model_id.0,
                    ))
                    .join("mmproj.gguf")
            } else {
                shard_store.shard_path(&candidate.model_id, candidate.shard_index)
            };
            if shard_path.exists() {
                if let Err(e) = std::fs::remove_file(&shard_path) {
                    tracing::warn!(
                        model = %candidate.model_id,
                        shard = candidate.shard_index,
                        error = %e,
                        "Failed to delete shard file during pruning"
                    );
                    continue;
                }
            }

            // Unregister from shard registry
            let shard_id = ShardId {
                model_id: candidate.model_id.clone(),
                index: candidate.shard_index,
            };
            registry.remove_shard_holder(&shard_id, &local_node_id);

            // Count remaining local shards for this model
            let remaining_local = registry
                .models()
                .iter()
                .find(|m| m.id == candidate.model_id)
                .map(|m| {
                    m.shards
                        .iter()
                        .filter(|s| {
                            let sid = ShardId {
                                model_id: m.id.clone(),
                                index: s.index,
                            };
                            registry.shard_holders(&sid).contains(&local_node_id)
                        })
                        .count() as u32
                })
                .unwrap_or(0);

            let event = crate::types::PruneEvent {
                model_id: candidate.model_id.clone(),
                model_name: candidate.model_name.clone(),
                shard_index: candidate.shard_index,
                reason: format!(
                    "Over-replicated ({} holders, target {})",
                    candidate.holder_count, candidate.target_replicas
                ),
                freed_bytes: candidate.shard_size_bytes,
                remaining_local_shards: remaining_local,
                holder_count_before: candidate.holder_count,
                holder_count_after: candidate.holder_count.saturating_sub(1),
                timestamp: chrono::Utc::now(),
            };

            tracing::info!(
                model = %candidate.model_id,
                shard = candidate.shard_index,
                holders = candidate.holder_count,
                target = candidate.target_replicas,
                freed_mb = candidate.shard_size_bytes / (1024 * 1024),
                "Pruning over-replicated shard"
            );

            // Emit prune event
            let _ = self.shared_state.prune_events_tx.send(event.clone());
            let _ = self.shared_state.models_changed_tx.send(());

            // Add to history
            {
                let mut history = self.shared_state.prune_history.write().await;
                if history.len() >= 100 {
                    history.pop_front();
                }
                history.push_back(event);
            }

            *pruned_per_model
                .entry(candidate.model_id.clone())
                .or_insert(0) += 1;
        }
    }

    /// Compute target replicas based on popularity (request count in last window).
    fn target_replicas(&self, request_count: u64, min_replicas: u32, pool_size: usize) -> u32 {
        let base = min_replicas as f64;
        let factor = match request_count {
            0 => 1.0,
            1..=10 => 1.5,
            11..=50 => 2.0,
            _ => 3.0,
        };
        let target = (base * factor).ceil() as u32;
        target.clamp(min_replicas, (pool_size as u32).max(min_replicas))
    }

    /// Adjust target based on resource pressure.
    fn pressure_adjusted_target(&self, target: u32, pressure: f64, min_replicas: u32) -> u32 {
        if pressure < 0.5 {
            // Relaxed: keep extras
            target.saturating_add(1)
        } else if pressure < 0.8 {
            target
        } else if pressure < 0.95 {
            // Eager
            target.saturating_sub(1).max(min_replicas)
        } else {
            // Urgent
            target.saturating_sub(2).max(min_replicas)
        }
    }

    /// Compute resource pressure (0.0–1.0) based on VRAM and disk usage.
    fn compute_resource_pressure(&self) -> f64 {
        let config = &self.shared_state.config;
        let local_node_id = self.shared_state.identity.node_id().clone();

        // Disk pressure
        let budget_mb = if config.auto_manage.max_storage_mb > 0 {
            config.auto_manage.max_storage_mb
        } else {
            config.resources.max_disk_mb / 2
        };
        let mut local_bytes = 0u64;
        for manifest in self.shared_state.model_registry.models() {
            for shard in &manifest.shards {
                let sid = ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                };
                if self
                    .shared_state
                    .model_registry
                    .shard_holders(&sid)
                    .contains(&local_node_id)
                {
                    local_bytes += shard.size_bytes;
                }
            }
        }
        let disk_pressure = if budget_mb > 0 {
            local_bytes as f64 / (budget_mb as f64 * 1024.0 * 1024.0)
        } else {
            0.0
        };

        // VRAM pressure — prefer live nvidia-smi data over internal model tracking
        let vram_pressure = if let Some(ref gpu) = self.shared_state.gpu_info {
            if gpu.vram_total_mb > 0 {
                let used_mb = query_gpu_vram_used().unwrap_or_else(|| {
                    // Fallback: sum estimated VRAM of loaded models
                    self.shared_state
                        .split_models
                        .iter()
                        .map(|e| e.value().estimated_vram_mb)
                        .sum()
                });
                used_mb as f64 / gpu.vram_total_mb as f64
            } else {
                0.0
            }
        } else {
            0.0
        };

        disk_pressure.max(vram_pressure).min(1.0)
    }

    /// Compute schedule-based pressure bonus during reduced hours.
    async fn schedule_pressure_bonus(&self) -> f64 {
        let schedule = self.shared_state.resource_schedule.read().await;
        if !schedule.enabled {
            return 0.0;
        }

        let now_hour = chrono::Utc::now().hour();
        let in_reduced = if schedule.reduced_hours_start <= schedule.reduced_hours_end {
            now_hour >= schedule.reduced_hours_start && now_hour < schedule.reduced_hours_end
        } else {
            // Wraps midnight (e.g., 22-8)
            now_hour >= schedule.reduced_hours_start || now_hour < schedule.reduced_hours_end
        };

        if !in_reduced {
            return 0.0;
        }

        match schedule.prune_aggressiveness.as_str() {
            "aggressive" => 0.3,
            "normal" => 0.15,
            _ => 0.0, // "conservative"
        }
    }

    /// Check if removing us as holder would eliminate the last holder in our region.
    fn would_eliminate_region(
        &self,
        _shard_id: &ShardId,
        local_node_id: &NodeId,
        holders: &[NodeId],
    ) -> bool {
        let our_region = self
            .shared_state
            .config
            .identity
            .region
            .as_deref()
            .unwrap_or("");

        if our_region.is_empty() {
            // No region data — fallback: ensure at least 2 holders with low latency
            let low_latency_remaining = holders
                .iter()
                .filter(|h| {
                    *h != local_node_id
                        && self
                            .shared_state
                            .peer_registry
                            .get(*h)
                            .map(|p| p.latency_ms.unwrap_or(9999) < 200)
                            .unwrap_or(false)
                })
                .count();
            return low_latency_remaining < 2;
        }

        // Count remaining holders in our region (excluding us)
        let same_region_remaining = holders
            .iter()
            .filter(|h| {
                if *h == local_node_id {
                    return false;
                }
                if let Some(peer) = self.shared_state.peer_registry.get(*h) {
                    peer.capability
                        .as_ref()
                        .and_then(|c| c.region.as_deref())
                        .map(|r| r.eq_ignore_ascii_case(our_region))
                        .unwrap_or(false)
                } else {
                    false
                }
            })
            .count();

        same_region_remaining == 0
    }

    /// Check if remaining holders (excluding us) are too busy.
    fn remaining_holders_busy(
        &self,
        holders: &[NodeId],
        local_node_id: &NodeId,
        max_load: u32,
    ) -> bool {
        let remaining: Vec<u32> = holders
            .iter()
            .filter_map(|h| {
                if h == local_node_id {
                    return None;
                }
                self.shared_state
                    .peer_registry
                    .get(h)
                    .map(|p| p.active_request_count)
            })
            .collect();

        if remaining.is_empty() {
            return true; // No peers to offload to
        }

        let avg_load = remaining.iter().sum::<u32>() as f64 / remaining.len() as f64;
        avg_load > max_load as f64
    }

    /// Check if this model's shards were used recently (last 5 min).
    fn shard_recently_used(&self, model_id: &ModelId) -> bool {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        self.shared_state.split_models.iter().any(|entry| {
            if entry.key().0 != *model_id {
                return false;
            }
            let last = entry.value().last_used_secs();
            now_secs.saturating_sub(last) < 300
        })
    }

    /// Check if we can re-acquire this shard if needed later.
    fn can_reacquire(
        &self,
        model_id: &ModelId,
        _shard_id: &ShardId,
        holders: &[NodeId],
        local_node_id: &NodeId,
    ) -> bool {
        // Check HF source
        if self.shared_state.hf_sources.contains_key(model_id) {
            return true;
        }

        // Check if healthy peers hold it (excluding us)
        let peer_holders = holders.iter().any(|h| {
            h != local_node_id
                && self
                    .shared_state
                    .peer_registry
                    .get(h)
                    .map(|p| p.latency_ms.unwrap_or(9999) < 5000)
                    .unwrap_or(false)
        });
        peer_holders
    }

    /// Get last prune time for a model from prune history.
    fn last_prune_time(&self, model_id: &ModelId) -> Option<chrono::DateTime<chrono::Utc>> {
        // Check prune history (we need a sync read, so try_read)
        if let Ok(history) = self.shared_state.prune_history.try_read() {
            history.iter().rev().find_map(|e| {
                if e.model_id == *model_id {
                    Some(e.timestamp)
                } else {
                    None
                }
            })
        } else {
            None
        }
    }

    /// Reset model request counts (called periodically, e.g. every 10 min).
    fn reset_request_counts(&self) {
        for entry in self.shared_state.model_request_counts.iter() {
            entry.value().store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Update model trust levels: promote popular models, decay inactive ones.
    ///
    /// - Models with >= 3 unique holder nodes → NetworkPopular
    /// - Models without requests for 7 days → decay (DemandVerified→Discovered)
    /// - Pinned models never decay
    /// - Ensures new gossip-discovered models get a Discovered entry
    fn update_model_trust(&self) {
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
                .model_trust
                .entry(manifest.id.clone())
                .or_insert_with(crate::types::ModelTrustInfo::new_discovered);

            // Promote to NetworkPopular if >= 3 unique holder nodes
            if holder_nodes.len() >= 3
                && trust.trust_level < crate::types::ModelTrustLevel::NetworkPopular
                && trust.trust_level >= crate::types::ModelTrustLevel::DemandVerified
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
    fn discover_hf_sources(&self) {
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
                if self.shared_state.hf_sources.contains_key(&mid) {
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
}

/// Estimate VRAM for a segment (layer range) by scaling the full-model estimate
/// by the fraction of layers covered.
fn estimate_segment_vram_mb(
    manifest: &crate::types::ModelManifest,
    layer_start: usize,
    layer_end: usize,
) -> u64 {
    let total_layers = manifest.num_layers as usize;
    if total_layers == 0 {
        return estimate_model_vram_mb(manifest.total_size_bytes);
    }
    let fraction = (layer_end - layer_start) as f64 / total_layers as f64;
    let full_vram = estimate_model_vram_mb(manifest.total_size_bytes);
    (full_vram as f64 * fraction).ceil() as u64
}

/// Compute the VRAM budget from SharedState for passing to `check_and_load_model`.
pub fn compute_vram_budget(shared: &crate::daemon::SharedState) -> Option<u64> {
    let gpu_total = shared
        .gpu_info
        .as_ref()
        .map(|g| g.vram_total_mb)
        .unwrap_or(0);
    shared.config.resources.inference_vram_budget_mb(gpu_total)
}

/// Scan the local models directory for shard files that exist on disk but are
/// not yet registered in the model registry. For any newly discovered shards,
/// register the local node as a holder, re-announce to the network, and trigger
/// model (re)loading so the node can use the new shards without a restart.
///
/// Returns the list of model IDs that had new shards discovered.
pub async fn rescan_local_shards(
    shared: &Arc<crate::daemon::SharedState>,
    network_tx: Option<&mpsc::Sender<NetworkCommand>>,
) -> Vec<ModelId> {
    let models_dir = shared.config.node.data_dir.join("models");
    if !models_dir.is_dir() {
        return vec![];
    }

    let local_node_id = shared.identity.node_id().clone();
    let shard_store = crate::model::shard::ShardStore::new(&shared.config.node.data_dir);
    let mut changed_models = Vec::new();

    let entries = match std::fs::read_dir(&models_dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };

    for entry in entries.flatten() {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let model_id_str = entry.file_name().to_string_lossy().to_string();
        let model_id = ModelId(model_id_str.clone());

        let manifest = match shared.model_registry.get_manifest(&model_id) {
            Some(m) => m,
            None => continue, // No manifest = can't register shards
        };

        let mut new_shards = 0u32;
        for shard_info in &manifest.shards {
            let shard_id = ShardId {
                model_id: model_id.clone(),
                index: shard_info.index,
            };

            // Already registered?
            if shared
                .model_registry
                .shard_holders(&shard_id)
                .contains(&local_node_id)
            {
                continue;
            }

            // Check if file exists on disk with reasonable size
            let path = shard_store.shard_path(&model_id, shard_info.index);
            if !path.exists() {
                continue;
            }
            let size_ok = std::fs::metadata(&path)
                .map(|m| m.len() >= shard_info.size_bytes * 9 / 10)
                .unwrap_or(false);
            if !size_ok {
                continue;
            }

            // Verify shard hash
            if shard_info.hash.len() == 32 {
                if let Err(e) = shard_store.verify_shard(&model_id, shard_info) {
                    tracing::warn!(
                        model = %model_id_str,
                        shard = shard_info.index,
                        error = %e,
                        "Rescan: shard verification failed, skipping"
                    );
                    continue;
                }
            }

            // Register as holder
            shared
                .model_registry
                .record_shard_holder(shard_id, local_node_id.clone());
            new_shards += 1;

            tracing::info!(
                model = %model_id_str,
                shard = shard_info.index,
                "Rescan: discovered new local shard"
            );
        }

        if new_shards > 0 {
            changed_models.push(model_id.clone());
            tracing::info!(
                model = %model_id_str,
                new_shards,
                "Rescan: registered new local shards"
            );
        }
    }

    // For models with new shards: reload the model and re-announce
    if !changed_models.is_empty() {
        let vram_budget = compute_vram_budget(shared);
        for model_id in &changed_models {
            // Evict old model segments so they reload with updated layer ranges
            let keys_to_remove: Vec<_> = shared
                .split_models
                .iter()
                .filter(|e| e.key().0 == *model_id)
                .map(|e| e.key().clone())
                .collect();
            for key in keys_to_remove {
                shared.split_models.remove(&key);
                tracing::info!(
                    model = %model_id,
                    range = format!("[{}..{})", key.1, key.2),
                    "Rescan: evicted old model segment for reload"
                );
            }

            check_and_load_model(shared, model_id, vram_budget).await;
        }

        // Re-announce shards to the network
        if let Some(tx) = network_tx {
            let local_node_id = shared.identity.node_id().clone();
            let mut hosted_shards = Vec::new();
            for entry in shared.model_registry.all_shard_entries() {
                let (shard_id, holders) = entry;
                if holders.contains(&local_node_id) {
                    hosted_shards.push(shard_id);
                }
            }
            if !hosted_shards.is_empty() {
                let announce = crate::types::ShardAnnounce {
                    node_id: local_node_id,
                    shards: hosted_shards,
                    timestamp: chrono::Utc::now(),
                };
                let _ = tx
                    .send(NetworkCommand::Broadcast(
                        crate::types::SwarmMessage::ShardAnnounce(announce),
                    ))
                    .await;
            }
        }
    }

    changed_models
}

/// Load whatever local shards are available for inference.
///
/// Called after each shard download completes (both auto-manage and manual).
/// A node does NOT need all shards — it loads whatever it has:
/// - All shards local: loads the full layer range (is_first=true, is_last=true)
/// - Partial shards: loads the covered layers for distributed inference
///   (this node handles its segment, other nodes handle theirs)
pub async fn check_and_load_model(
    shared: &std::sync::Arc<crate::daemon::SharedState>,
    model_id: &ModelId,
    vram_budget_mb: Option<u64>,
) {
    let manifest = match shared.model_registry.get_manifest(model_id) {
        Some(m) => m,
        None => return,
    };

    let local_node_id = shared.identity.node_id().clone();
    let model_dir = shared.config.node.data_dir.join("models").join(&model_id.0);

    // Find which shards we actually have on disk and are fully downloaded.
    // A shard is considered ready only when:
    //  1. It's in the shard registry for our node
    //  2. The file exists on disk
    //  3. Its size is at least 90% of the manifest's expected size (handles last-shard)
    //  4. There's no active download in progress for it
    let shard_store = crate::model::shard::ShardStore::new(&shared.config.node.data_dir);
    let mut local_shard_indices: Vec<u32> = manifest
        .shards
        .iter()
        .filter(|s| {
            let sid = ShardId {
                model_id: model_id.clone(),
                index: s.index,
            };
            let in_registry = shared
                .model_registry
                .shard_holders(&sid)
                .contains(&local_node_id);
            let path = shard_store.shard_path(model_id, s.index);
            let on_disk = path.exists();
            if !in_registry || !on_disk {
                return false;
            }
            // Check file is fully downloaded (not a partial write)
            let size_ok = std::fs::metadata(&path)
                .map(|m| m.len() >= s.size_bytes * 9 / 10)
                .unwrap_or(false);
            if !size_ok {
                return false;
            }
            // Check no active download for this shard
            let is_downloading = shared
                .acquisition_progress
                .get(model_id)
                .map(|entry| {
                    entry
                        .shard_progress
                        .get(&s.index)
                        .map(|sp| sp.state == crate::model::acquisition::ShardState::Downloading)
                        .unwrap_or(false)
                })
                .unwrap_or(false);
            !is_downloading
        })
        .map(|s| s.index)
        .collect();
    local_shard_indices.sort();

    if local_shard_indices.is_empty() {
        return;
    }

    // Note: we don't short-circuit here even if some segments are already loaded,
    // because the node may have additional non-contiguous ranges to load.

    let has_all = local_shard_indices.len() == manifest.shard_count as usize;

    // Determine ALL layer ranges covered by our local shards using manifest
    // tensor metadata.  V2 manifests carry per-shard tensor entries with
    // accurate layer_range data, so we always use that.
    let ranges: Vec<(usize, usize)> = if has_all {
        vec![(0, manifest.num_layers as usize)]
    } else {
        crate::inference::split::available_layer_ranges_from_manifest(
            &manifest,
            &local_shard_indices,
        )
    };

    if ranges.is_empty() {
        tracing::warn!(
            model = %model_id,
            local_shards = ?local_shard_indices,
            "No complete layers available in local shards"
        );
        return;
    }

    let missing_shards = manifest.shard_count as usize - local_shard_indices.len();
    tracing::info!(
        model = %model_id,
        available_shards = local_shard_indices.len(),
        missing_shards,
        total_shards = manifest.shard_count,
        ranges = ?ranges,
        ready = missing_shards == 0,
        local_shard_indices = ?local_shard_indices,
        "DIAG: check_and_load_model"
    );

    let _shard_store = crate::model::shard::ShardStore::new(&shared.config.node.data_dir);
    let mut any_loaded = false;

    // TOCTOU guard: use loading_models to prevent concurrent duplicate loads.
    // If another task is already loading this model, skip silently.
    let _loading_guard = {
        use dashmap::mapref::entry::Entry;
        match shared.loading_models.entry(model_id.clone()) {
            Entry::Vacant(e) => {
                e.insert(std::sync::Arc::new(tokio::sync::Notify::new()));
                Some(model_id.clone()) // We hold the guard
            }
            Entry::Occupied(_) => {
                tracing::debug!(model = %model_id, "check_and_load_model: another load in progress, skipping");
                return;
            }
        }
    };
    // Ensure we remove the guard when done (RAII via scope + defer pattern)
    struct LoadGuard<'a> {
        shared: &'a std::sync::Arc<crate::daemon::SharedState>,
        model_id: Option<ModelId>,
    }
    impl<'a> Drop for LoadGuard<'a> {
        fn drop(&mut self) {
            if let Some(ref mid) = self.model_id {
                if let Some((_, notify)) = self.shared.loading_models.remove(mid) {
                    notify.notify_waiters();
                }
            }
        }
    }
    let _guard = LoadGuard {
        shared,
        model_id: _loading_guard,
    };

    for &(layer_start, layer_end) in &ranges {
        if layer_start >= layer_end {
            continue;
        }

        let split_key = (model_id.clone(), layer_start, layer_end);
        if shared.split_models.contains_key(&split_key) {
            any_loaded = true;
            continue; // Already loaded this segment
        }

        // VRAM budget pre-check: skip loading if budget is full (shards stay on disk for P2P)
        if let Some(budget) = vram_budget_mb {
            let estimated = estimate_segment_vram_mb(&manifest, layer_start, layer_end);
            let total_loaded: u64 = shared
                .split_models
                .iter()
                .map(|e| e.value().estimated_vram_mb)
                .sum();
            if total_loaded + estimated > budget {
                // Try LRU eviction first
                crate::inference::split::evict_split_models_lru(
                    &shared.split_models,
                    &shared.active_pipelines,
                    budget,
                    estimated,
                );
                let total_after: u64 = shared
                    .split_models
                    .iter()
                    .map(|e| e.value().estimated_vram_mb)
                    .sum();
                if total_after + estimated > budget {
                    tracing::info!(
                        model = %model_id,
                        layers = format!("[{layer_start}..{layer_end})"),
                        estimated_mb = estimated,
                        loaded_mb = total_after,
                        budget_mb = budget,
                        "VRAM budget full — skipping auto-load (shards remain on disk for P2P)"
                    );
                    continue;
                }
            }
        }

        // is_first requires shard 0 (token_embd.weight is always at tensor offset 0)
        // is_last requires the final shard (output.weight spans to the end of the file)
        let has_shard_0 = local_shard_indices.contains(&0);
        let last_shard_idx = manifest.shard_count.saturating_sub(1);
        let has_last_shard = local_shard_indices.contains(&last_shard_idx);
        let is_first = layer_start == 0 && has_shard_0;
        let is_last = layer_end >= manifest.num_layers as usize && has_last_shard;

        // Create metadata entry from GGUF header (no GPU loading in main process).
        // The worker subprocess will load the model on first inference request.
        let header_path = model_dir.join("gguf_header.bin");
        let vram_estimate = crate::daemon::estimate_vram_from_shard_dir(
            &model_dir,
            layer_start,
            layer_end,
            manifest.num_layers as usize,
        );
        let new_entry = crate::inference::split::SplitModelEntry::from_header(
            &header_path,
            layer_start,
            layer_end,
            is_first,
            is_last,
            vram_estimate,
        );

        // Update loaded_model_info from the entry metadata
        let eos_tokens = new_entry.eos_tokens.clone();
        let chat_template = new_entry.cached_chat_template.clone();
        let bos_token = new_entry.bos_token.clone();
        let eos_token = new_entry.eos_token_str.clone();

        // Safety-net eviction: use VRAM budget (falls back to max_split_model_memory_mb)
        let eviction_budget = vram_budget_mb.or(shared.config.inference.max_split_model_memory_mb);
        if let Some(budget) = eviction_budget {
            crate::inference::split::evict_split_models_lru(
                &shared.split_models,
                &shared.active_pipelines,
                budget,
                new_entry.estimated_vram_mb,
            );
        }
        shared.split_models.insert(split_key, new_entry);

        // Update loaded_model_info so the API knows the model is available
        if !any_loaded {
            *shared.loaded_model_info.write().await = Some(crate::daemon::LoadedModelInfo {
                name: manifest.name.clone(),
                size_bytes: manifest.total_size_bytes,
                eos_tokens,
                chat_template,
                bos_token,
                eos_token,
            });
        }
        any_loaded = true;

        tracing::info!(
            model = %model_id,
            name = %manifest.name,
            layers = format!("[{}..{})", layer_start, layer_end),
            "Auto-manage: model metadata loaded (subprocess will load on first inference)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        // Acquire 2 permits — should succeed
        let p1 = sem.clone().acquire_owned().await.unwrap();
        let p2 = sem.clone().acquire_owned().await.unwrap();
        assert_eq!(sem.available_permits(), 0);

        // Third acquire would block, so use try_acquire
        assert!(sem.try_acquire().is_err());

        // Drop one permit — should free a slot
        drop(p1);
        assert_eq!(sem.available_permits(), 1);

        let _p3 = sem.clone().acquire_owned().await.unwrap();
        assert_eq!(sem.available_permits(), 0);

        drop(p2);
        drop(_p3);
        assert_eq!(sem.available_permits(), 2);
    }

    // --- Pruning unit tests ---

    /// Helper: compute target replicas using the same formula as AutoShardManager.
    fn target_replicas_pure(request_count: u64, min_replicas: u32, pool_size: usize) -> u32 {
        let base = min_replicas as f64;
        let factor = match request_count {
            0 => 1.0,
            1..=10 => 1.5,
            11..=50 => 2.0,
            _ => 3.0,
        };
        let target = (base * factor).ceil() as u32;
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
    fn popularity_tiers_zero_requests() {
        assert_eq!(target_replicas_pure(0, 2, 10), 2);
    }

    #[test]
    fn popularity_tiers_low_requests() {
        // 1-10 requests → factor 1.5 → ceil(2*1.5) = 3
        assert_eq!(target_replicas_pure(1, 2, 10), 3);
        assert_eq!(target_replicas_pure(5, 2, 10), 3);
        assert_eq!(target_replicas_pure(10, 2, 10), 3);
    }

    #[test]
    fn popularity_tiers_medium_requests() {
        // 11-50 requests → factor 2.0 → ceil(2*2.0) = 4
        assert_eq!(target_replicas_pure(11, 2, 10), 4);
        assert_eq!(target_replicas_pure(50, 2, 10), 4);
    }

    #[test]
    fn popularity_tiers_high_requests() {
        // 51+ requests → factor 3.0 → ceil(2*3.0) = 6
        assert_eq!(target_replicas_pure(51, 2, 10), 6);
        assert_eq!(target_replicas_pure(1000, 2, 10), 6);
    }

    #[test]
    fn popularity_clamped_by_pool_size() {
        // pool_size=3, 51+ requests → ceil(2*3.0)=6, clamped to 3
        assert_eq!(target_replicas_pure(100, 2, 3), 3);
        // pool_size=4, 0 requests → base=2, factor=1.0, target=2
        assert_eq!(target_replicas_pure(0, 2, 4), 2);
        // pool_size=2, 51+ requests → ceil(2*3.0)=6, clamped to 2
        assert_eq!(target_replicas_pure(100, 2, 2), 2);
    }

    #[test]
    fn popularity_pool_size_zero() {
        // Single node, no peers → pool_size=0, should not panic
        assert_eq!(target_replicas_pure(0, 1, 0), 1);
        assert_eq!(target_replicas_pure(100, 2, 0), 2);
        assert_eq!(target_replicas_pure(0, 1, 1), 1);
    }

    #[test]
    fn pressure_relaxed_adds_one() {
        // pressure < 0.5 → target + 1
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
