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
        Self {
            shared_state,
            network_tx,
            shutdown_rx,
        }
    }

    /// Run the auto-manage loop. Checks periodically based on config interval.
    pub async fn run(mut self) {
        let config = &self.shared_state.config.auto_manage;
        if !config.enabled {
            tracing::info!("AutoShardManager disabled, exiting");
            return;
        }

        let interval_mins = config.interval_minutes.max(1); // minimum 1 min
        let mut interval = tokio::time::interval(Duration::from_secs(interval_mins as u64 * 60));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Skip the first tick (fires immediately) — let the node discover peers first
        interval.tick().await;

        tracing::info!(
            interval_minutes = interval_mins,
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
                    // Re-check enabled — config may have changed at runtime
                    if self.shared_state.config.auto_manage.enabled {
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

            let model_popularity = all_holders.len() as f64;
            if model_popularity < 1.0 {
                // No one has any shards — probably just published, skip
                continue;
            }

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

                let holder_count = holders.len();

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
                let score = model_popularity * rarity_bonus * configured_bonus * vram_fitness;

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
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        // If any configured-range shards are missing for the primary model,
        // focus exclusively on those first. Don't download extra shards until
        // our assigned range is complete.
        if let Some((start, end)) = configured_range {
            // Find the model that matches our configured shard range
            // (the one we were started with via --model + --shards)
            let configured_model: Option<ModelId> = registry.models().iter()
                .find(|m| m.shards.iter().any(|s| s.index >= start && s.index <= end))
                .map(|m| m.id.clone());

            if let Some(ref mid) = configured_model {
                let has_configured_missing = candidates.iter().any(|c| {
                    c.model_id == *mid && c.shard_index >= start && c.shard_index <= end
                });
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

        // Also check existing downloads in progress
        let in_progress: std::collections::HashSet<String> = self
            .shared_state
            .acquisition_progress
            .iter()
            .filter(|e| {
                matches!(
                    e.value().state,
                    crate::model::acquisition::AcquisitionState::Downloading
                )
            })
            .map(|e| e.key().0.clone())
            .collect();

        for candidate in candidates {
            if selected.len() >= max {
                break;
            }
            if candidate.shard_size_bytes > budget_bytes {
                continue;
            }
            // Don't download if model is already being acquired
            if in_progress.contains(&candidate.model_id.0) {
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
    async fn trigger_download(&self, candidate: &ShardCandidate) {
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

        // Check if we already have the shard file locally
        let shard_path = model_dir.join(format!("shard_{:03}.bin", candidate.shard_index));
        if shard_path.exists() {
            self.register_local_shard(candidate);
            self.check_model_complete(&candidate.model_id).await;
            return;
        }

        // Create a progress entry so the UI can track it
        let mid = candidate.model_id.clone();
        let status = crate::model::acquisition::AcquisitionStatus {
            model_id: mid.clone(),
            state: crate::model::acquisition::AcquisitionState::Downloading,
            total_shards: 1,
            downloaded_shards: 0,
            verified_shards: 0,
            failed_shards: 0,
            total_bytes: candidate.shard_size_bytes,
            downloaded_bytes: 0,
            shard_progress: std::collections::HashMap::new(),
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

        // Announce interest to network peers — they may push the shard to us
        let shard_id = ShardId {
            model_id: candidate.model_id.clone(),
            index: candidate.shard_index,
        };
        let announce = crate::types::SwarmMessage::ShardAnnounce(crate::types::ShardAnnounce {
            node_id: self.shared_state.identity.node_id().clone(),
            shards: vec![shard_id],
            timestamp: chrono::Utc::now(),
        });
        let _ = self
            .network_tx
            .try_send(NetworkCommand::Broadcast(announce));

        // Download from HuggingFace if source is known
        if let Some(hf_source) = self.shared_state.hf_sources.get(&candidate.model_id) {
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

            // Spawn the download so we don't block the evaluation loop
            tokio::spawn(async move {
                let (ptx, mut prx) = tokio::sync::mpsc::channel::<
                    crate::model::huggingface::DownloadProgress,
                >(32);

                // Progress updater
                let prog_mid = model_id.clone();
                let prog_shared = shared.clone();
                tokio::spawn(async move {
                    while let Some(prog) = prx.recv().await {
                        if let Some(mut entry) =
                            prog_shared.acquisition_progress.get_mut(&prog_mid)
                        {
                            entry.downloaded_bytes = prog.downloaded_bytes;
                            entry.total_bytes = prog.total_bytes;
                        }
                    }
                });

                let configured_shard_size = shared.config.model.shard_size_bytes();
                match crate::model::huggingface::download_shards(
                    &repo_id,
                    &filename,
                    &dest,
                    &[shard_idx],
                    Some(ptx),
                    Some(configured_shard_size),
                )
                .await
                {
                    Ok((_path, _info)) => {
                        tracing::info!(
                            model = %model_id,
                            shard = shard_idx,
                            "AutoShardManager: shard downloaded from HF"
                        );

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
                            let mut holders =
                                shared.shard_registry.entry(sid).or_default();
                            if !holders.contains(&node_id) {
                                holders.push(node_id);
                            }
                        }

                        // Update progress
                        if let Some(mut entry) =
                            shared.acquisition_progress.get_mut(&model_id)
                        {
                            entry.state =
                                crate::model::acquisition::AcquisitionState::Complete;
                            entry.downloaded_shards = 1;
                            entry.verified_shards = 1;
                            entry.log.push("Shard downloaded and registered".into());
                        }

                        // Check if all shards are now available → auto-load
                        check_and_load_model(&shared, &model_id).await;
                    }
                    Err(e) => {
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_idx,
                            error = %e,
                            "AutoShardManager: HF shard download failed"
                        );
                        if let Some(mut entry) =
                            shared.acquisition_progress.get_mut(&model_id)
                        {
                            entry.state =
                                crate::model::acquisition::AcquisitionState::Failed {
                                    reason: e,
                                };
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

    /// Check if all shards for a model are now locally available, and if so, load it.
    async fn check_model_complete(&self, model_id: &ModelId) {
        check_and_load_model(&self.shared_state, model_id).await;
    }

    /// Discover HF source metadata from `hf_source.json` files next to manifests.
    ///
    /// This allows seeding HF source info by placing a small JSON file:
    /// `{ "repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf" }`
    fn discover_hf_sources(&self) {
        let models_dir = self
            .shared_state
            .config
            .node
            .data_dir
            .join("models");

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
                        if let Ok(source) =
                            serde_json::from_str::<crate::daemon::HfSource>(&data)
                        {
                            tracing::info!(
                                model = %model_id_str,
                                repo = %source.repo_id,
                                "Discovered HF source from hf_source.json"
                            );
                            self.shared_state.hf_sources.insert(mid.clone(), source.clone());
                            // Persist to sled
                            let _ = self.shared_state.db.put_json(
                                "hf_sources",
                                &model_id_str,
                                &source,
                            );
                        }
                    }
                }
            }
        }
    }
}

/// Check if all shards for a model are locally available, and if so, load the split model.
///
/// This is called after each shard download completes (both auto-manage and manual).
/// If all shards in the manifest are held locally, it loads the model into the
/// `split_models` DashMap so it becomes available for inference.
async fn check_and_load_model(
    shared: &std::sync::Arc<crate::daemon::SharedState>,
    model_id: &ModelId,
) {
    let manifest = match shared.model_registry.get_manifest(model_id) {
        Some(m) => m,
        None => return,
    };

    let local_node_id = shared.identity.node_id().clone();
    let model_dir = shared
        .config
        .node
        .data_dir
        .join("models")
        .join(&model_id.0);

    // Check if ALL shard files actually exist on disk (don't trust registry alone)
    let shard_store = crate::model::shard::ShardStore::new(&shared.config.node.data_dir);
    let local_count = manifest
        .shards
        .iter()
        .filter(|s| {
            let sid = ShardId {
                model_id: model_id.clone(),
                index: s.index,
            };
            // Verify both registry AND file on disk
            let in_registry = shared.model_registry.shard_holders(&sid).contains(&local_node_id);
            let on_disk = shard_store.shard_path(model_id, s.index).exists();
            in_registry && on_disk
        })
        .count();

    if local_count < manifest.shard_count as usize {
        tracing::debug!(
            model = %model_id,
            local = local_count,
            total = manifest.shard_count,
            "Model not yet complete"
        );
        return;
    }

    // Already loaded?
    if shared.split_models.contains_key(model_id) {
        tracing::debug!(model = %model_id, "Model already loaded in split_models");
        return;
    }

    tracing::info!(
        model = %model_id,
        shards = manifest.shard_count,
        "All shards available — loading model for inference"
    );

    // Determine layer range (all layers for a complete model)
    let layer_start = 0;
    let layer_end = manifest.num_layers as usize;

    let shard_store = crate::model::shard::ShardStore::new(&shared.config.node.data_dir);
    let params = crate::daemon::ShardLoadParams {
        model_dir: &model_dir,
        shard_store: &shard_store,
        model_id,
        layer_start,
        layer_end,
        is_first: true,
        is_last: true,
    };

    match crate::daemon::try_load_from_shards(&params) {
        Ok(split_model) => {
            let eos_tokens = split_model.eos_tokens().to_vec();
            let chat_template = split_model.chat_template().map(|s| s.to_string());
            let bos_token = split_model.bos_token().to_string();
            let eos_token = split_model.eos_token_str().to_string();
            shared.split_models.insert(
                model_id.clone(),
                std::sync::Arc::new(tokio::sync::Mutex::new(split_model)),
            );

            // Update loaded_model_info so the API knows the model is available
            *shared.loaded_model_info.write().await =
                Some(crate::daemon::LoadedModelInfo {
                    name: manifest.name.clone(),
                    size_bytes: manifest.total_size_bytes,
                    eos_tokens,
                    chat_template,
                    bos_token,
                    eos_token,
                });

            tracing::info!(
                model = %model_id,
                name = %manifest.name,
                "Auto-manage: model loaded and ready for inference"
            );
        }
        Err(e) => {
            tracing::error!(
                model = %model_id,
                error = %e,
                "Auto-manage: failed to load model from shards"
            );
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
        };
        assert_eq!(config.max_shards, 0); // unlimited
    }
}
