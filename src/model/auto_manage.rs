use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, watch};

use crate::daemon::SharedState;
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

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("AutoShardManager shutting down");
                        break;
                    }
                }
                _ = interval.tick() => {
                    // Re-check enabled — admin API can toggle at runtime
                    if self.shared_state.auto_manage_enabled.load(std::sync::atomic::Ordering::Acquire) {
                        self.evaluate_and_download().await;
                    }
                }
                _ = self.notify.notified() => {
                    // Woken by a new HfSourceGossip or ModelManifest — wait briefly
                    // for additional gossip to settle, then evaluate.
                    tokio::time::sleep(Duration::from_secs(2)).await;
                    if self.shared_state.auto_manage_enabled.load(std::sync::atomic::Ordering::Acquire) {
                        tracing::info!("AutoShardManager: triggered by new HF source or manifest");
                        self.evaluate_and_download().await;
                    }
                }
            }
        }
    }

    /// Core logic: evaluate network state and download the best candidate shards.
    async fn evaluate_and_download(&self) {
        let config = &self.shared_state.config.auto_manage;
        let local_node_id = self.shared_state.identity.node_id().clone();

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
            config.max_storage_mb * 1024 * 1024
        } else {
            // Fall back to global max_disk_mb, using 50% for auto-manage
            (self.shared_state.config.resources.max_disk_mb * 1024 * 1024) / 2
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

        max_bytes.saturating_sub(current_bytes)
    }

    /// Gather all candidate shards we don't already hold, scored by value.
    ///
    /// Scoring factors:
    /// - **configured_bonus** (100x): shards in our `--shards` range missing from disk
    /// - **rarity_bonus** (1-10x): fewer holders → higher priority
    /// - **popularity**: more unique holders across model → higher value
    /// - **vram_fitness** (0.1-1.0x): models that fit in global VRAM pool score higher
    fn gather_candidates(&self, local_node_id: &NodeId, pool_vram_mb: u64) -> Vec<ShardCandidate> {
        let mut candidates = Vec::new();
        let registry = &self.shared_state.model_registry;
        let shard_store =
            crate::model::shard::ShardStore::new(&self.shared_state.config.node.data_dir);
        let configured_range = self.shared_state.config.inference.shard_range;

        for manifest in registry.models() {
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

                let score =
                    model_popularity * rarity_bonus * configured_bonus * vram_fitness + jitter;

                candidates.push(ShardCandidate {
                    model_id: manifest.id.clone(),
                    model_name: manifest.name.clone(),
                    shard_index: shard.index,
                    shard_size_bytes: shard.size_bytes,
                    holder_count,
                    score,
                });
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

        for candidate in candidates {
            if selected.len() >= max {
                break;
            }
            if candidate.shard_size_bytes > budget_bytes {
                continue;
            }
            // Skip this specific shard if it's already being downloaded
            if downloading_shards.contains(&(candidate.model_id.0.clone(), candidate.shard_index)) {
                continue;
            }

            budget_bytes -= candidate.shard_size_bytes;
            selected.push(candidate);

            // Only download 1-2 shards per evaluation cycle to spread load
            if selected.len() >= 2 {
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

        let model_dir = self
            .shared_state
            .config
            .node
            .data_dir
            .join("models")
            .join(&candidate.model_id.0);

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
                    } else {
                        // Zero-hash placeholder — fall back to size check
                        std::fs::metadata(&shard_path)
                            .map(|m| m.len() >= candidate.shard_size_bytes * 9 / 10)
                            .unwrap_or(false)
                    }
                } else {
                    std::fs::metadata(&shard_path)
                        .map(|m| m.len() >= candidate.shard_size_bytes * 9 / 10)
                        .unwrap_or(false)
                }
            } else {
                std::fs::metadata(&shard_path)
                    .map(|m| m.len() >= candidate.shard_size_bytes * 9 / 10)
                    .unwrap_or(false)
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
                let status = crate::model::acquisition::AcquisitionStatus {
                    model_id: mid.clone(),
                    state: crate::model::acquisition::AcquisitionState::Downloading,
                    total_shards: 1,
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

                // Download header + shard
                crate::model::huggingface::download_gguf_header(
                    &repo_id,
                    &filename,
                    &dest,
                    info.header_size,
                )
                .await
                .ok();

                match crate::model::huggingface::download_shard_v2(
                    &repo_id,
                    &filename,
                    &dest,
                    layout,
                    Some(ptx),
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

                        // Register the shard
                        let node_id = shared.identity.node_id().clone();
                        let sid = crate::types::ShardId {
                            model_id: model_id.clone(),
                            index: shard_idx,
                        };
                        shared
                            .model_registry
                            .record_shard_holder(sid.clone(), node_id.clone());
                        {
                            let mut holders = shared.shard_registry.entry(sid.clone()).or_default();
                            if !holders.contains(&node_id) {
                                holders.push(node_id.clone());
                            }
                        }

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
                        check_and_load_model(&shared, &model_id).await;

                        // Self-wake so we immediately re-evaluate and download
                        // more shards (libp2p gossipsub doesn't deliver our own
                        // broadcasts back to us, so we must notify ourselves).
                        shared.auto_manage_notify.notify_one();
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

    /// Register a shard file that already exists on disk.
    fn register_local_shard(&self, candidate: &ShardCandidate) {
        tracing::debug!(
            model = %candidate.model_id,
            shard = candidate.shard_index,
            "Shard file already exists on disk, registering"
        );
        let node_id = self.shared_state.identity.node_id().clone();
        let shard_id = ShardId {
            model_id: candidate.model_id.clone(),
            index: candidate.shard_index,
        };
        self.shared_state
            .model_registry
            .record_shard_holder(shard_id.clone(), node_id.clone());
        let mut holders = self
            .shared_state
            .shard_registry
            .entry(shard_id)
            .or_default();
        if !holders.contains(&node_id) {
            holders.push(node_id);
        }
    }

    /// Check if any local shards are available for this model and load them.
    /// A node does NOT need all shards — it loads whatever it has and participates
    /// in distributed inference for the layers it covers.
    async fn check_model_complete(&self, model_id: &ModelId) {
        check_and_load_model(&self.shared_state, model_id).await;
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
                            // Persist to sled
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

    tracing::info!(
        model = %model_id,
        local_shards = local_shard_indices.len(),
        total_shards = manifest.shard_count,
        ranges = ?ranges,
        "Loading model segments for inference"
    );

    let shard_store = crate::model::shard::ShardStore::new(&shared.config.node.data_dir);
    let mut any_loaded = false;

    for &(layer_start, layer_end) in &ranges {
        if layer_start >= layer_end {
            continue;
        }

        let split_key = (model_id.clone(), layer_start, layer_end);
        if shared.split_models.contains_key(&split_key) {
            any_loaded = true;
            continue; // Already loaded this segment
        }

        // is_first requires shard 0 (token_embd.weight is always at tensor offset 0)
        // is_last requires the final shard (output.weight spans to the end of the file)
        let has_shard_0 = local_shard_indices.contains(&0);
        let last_shard_idx = manifest.shard_count.saturating_sub(1);
        let has_last_shard = local_shard_indices.contains(&last_shard_idx);
        let is_first = layer_start == 0 && has_shard_0;
        let is_last = layer_end >= manifest.num_layers as usize && has_last_shard;

        // Try loading: model.gguf → source_path → shard files
        let gguf_path = model_dir.join("model.gguf");
        let source_path_file = model_dir.join("source_path");

        let load_result = if gguf_path.exists() {
            tracing::info!(
                model = %model_id,
                layers = format!("[{layer_start}..{layer_end})"),
                "Loading split model from reconstructed GGUF"
            );
            crate::inference::split::SplitModel::load_from_gguf(
                &gguf_path,
                layer_start,
                layer_end,
                is_first,
                is_last,
            )
        } else if source_path_file.exists() {
            match std::fs::read_to_string(&source_path_file) {
                Ok(p) => {
                    let p = std::path::PathBuf::from(p.trim());
                    if p.exists() {
                        tracing::info!(
                            model = %model_id,
                            layers = format!("[{layer_start}..{layer_end})"),
                            "Loading split model from source GGUF"
                        );
                        crate::inference::split::SplitModel::load_from_gguf(
                            &p,
                            layer_start,
                            layer_end,
                            is_first,
                            is_last,
                        )
                    } else {
                        crate::daemon::try_load_from_shards(&crate::daemon::ShardLoadParams {
                            model_dir: &model_dir,
                            shard_store: &shard_store,
                            model_id,
                            layer_start,
                            layer_end,
                            is_first,
                            is_last,
                            manifest: &manifest,
                        })
                    }
                }
                Err(e) => Err(crate::error::SwarmError::Io(e)),
            }
        } else {
            crate::daemon::try_load_from_shards(&crate::daemon::ShardLoadParams {
                model_dir: &model_dir,
                shard_store: &shard_store,
                model_id,
                layer_start,
                layer_end,
                is_first,
                is_last,
                manifest: &manifest,
            })
        };

        match load_result {
            Ok(split_model) => {
                let eos_tokens = split_model.eos_tokens().to_vec();
                let chat_template = split_model.chat_template().map(|s| s.to_string());
                let bos_token = split_model.bos_token().to_string();
                let eos_token = split_model.eos_token_str().to_string();
                // VRAM-aware eviction before inserting new model
                let max_batch = shared.config.inference.max_batch_size as usize;
                let new_entry = if max_batch > 1 {
                    crate::inference::split::SplitModelEntry::new_with_batching(
                        split_model,
                        shared.kv_cache_store.clone(),
                        max_batch,
                    )
                } else {
                    crate::inference::split::SplitModelEntry::new(split_model)
                };
                if let Some(budget_mb) = shared.config.inference.max_split_model_memory_mb {
                    crate::inference::split::evict_split_models_lru(
                        &shared.split_models,
                        &shared.active_pipelines,
                        budget_mb,
                        new_entry.estimated_vram_mb,
                    );
                }
                shared.split_models.insert(split_key, new_entry);

                // Update loaded_model_info so the API knows the model is available
                if !any_loaded {
                    *shared.loaded_model_info.write().await =
                        Some(crate::daemon::LoadedModelInfo {
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
                    "Auto-manage: model segment loaded and ready for inference"
                );
            }
            Err(e) => {
                tracing::error!(
                    model = %model_id,
                    layers = format!("[{}..{})", layer_start, layer_end),
                    error = %e,
                    "Auto-manage: failed to load model segment from shards"
                );
            }
        }
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
            max_concurrent_downloads: 3,
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
}
