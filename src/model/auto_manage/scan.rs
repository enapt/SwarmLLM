use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::types::{ModelId, NetworkCommand, ShardId};

use super::vram::{compute_vram_budget, estimate_segment_vram_mb};

/// Scan the local models directory for shard files that exist on disk but are
/// not yet registered in the model registry. For any newly discovered shards,
/// register the local node as a holder, re-announce to the network, and trigger
/// model (re)loading so the node can use the new shards without a restart.
///
/// Returns the list of model IDs that had new shards discovered.
pub async fn rescan_local_shards(
    shared: &Arc<SharedState>,
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
            // Skip size check when manifest has no size info (size_bytes == 0) —
            // an empty file must not pass as a valid shard.
            let size_ok = shard_info.size_bytes > 0
                && std::fs::metadata(&path)
                    .map(|m| m.len() >= shard_info.size_bytes * 9 / 10)
                    .unwrap_or(false);
            if !size_ok {
                continue;
            }

            // Verify shard hash (skip zero-hash placeholders)
            if shard_info.hash != [0u8; 32] {
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
            let mname = shared
                .model_registry
                .get_manifest(&model_id)
                .map(|m| m.name.clone());
            shared.emit_activity(crate::daemon::state::ActivityEvent {
                category: "model",
                kind: "shard_scan_found",
                message: format!(
                    "Found {} new shard{} of {} on disk",
                    new_shards,
                    if new_shards != 1 { "s" } else { "" },
                    mname.as_deref().unwrap_or(&model_id_str)
                ),
                model_id: Some(model_id.0.clone()),
                model_name: mname,
                node_id: None,
                detail_num: Some(new_shards as i64),
                detail_str: None,
            });
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
                // S5: Register as DHT provider for rescanned shards
                let _ = tx
                    .send(NetworkCommand::StartProviding(hosted_shards.clone()))
                    .await;
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
/// A node does NOT need all shards -- it loads whatever it has:
/// - All shards local: loads the full layer range (is_first=true, is_last=true)
/// - Partial shards: loads the covered layers for distributed inference
///   (this node handles its segment, other nodes handle theirs)
pub async fn check_and_load_model(
    shared: &Arc<SharedState>,
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

    let mut any_loaded = false;

    // TOCTOU guard: use loading_models to prevent concurrent duplicate loads.
    // If another task is already loading this model, skip silently.
    let _loading_guard = {
        use dashmap::mapref::entry::Entry;
        match shared.loading_models.entry(model_id.clone()) {
            Entry::Vacant(e) => {
                e.insert(Arc::new(tokio::sync::Notify::new()));
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
        shared: &'a Arc<SharedState>,
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
                    shared.emit_activity(crate::daemon::state::ActivityEvent {
                        category: "model",
                        kind: "model_load_skipped",
                        message: format!(
                            "Not loading {} — {estimated}MB needed but only {}MB free of {budget}MB budget",
                            manifest.name, budget - total_after
                        ),
                        model_id: Some(model_id.0.clone()),
                        model_name: Some(manifest.name.clone()),
                        node_id: None,
                        detail_num: Some(estimated as i64),
                        detail_str: Some("vram_budget".to_string()),
                    });
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

        shared.emit_activity(crate::daemon::state::ActivityEvent {
            category: "model",
            kind: "model_loaded",
            message: format!(
                "Loaded {} into memory — layers [{}, {}) ready for inference",
                manifest.name, layer_start, layer_end
            ),
            model_id: Some(model_id.0.clone()),
            model_name: Some(manifest.name.clone()),
            node_id: None,
            detail_num: Some((layer_end - layer_start) as i64),
            detail_str: Some(format!("[{}..{})", layer_start, layer_end)),
        });
    }
}
