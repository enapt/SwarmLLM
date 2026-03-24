use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

pub async fn list_models(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let mut models: Vec<serde_json::Value> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let local_node_id = state.shared_state.identity.node_id().clone();

    // Collect peer info for each model from model_registry shard_holders
    let mut model_peers: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for (shard_id, holders) in state.shared_state.model_registry.all_shard_entries() {
        let model_name = shard_id.model_id.0.clone();
        for holder in &holders {
            if *holder != local_node_id {
                model_peers
                    .entry(model_name.clone())
                    .or_default()
                    .insert(format!("{}", holder));
            }
        }
    }

    // Also gather peer node_ids that host each model from capability data
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        if let Some(ref cap) = peer.capability {
            for shard in &cap.hosted_shards {
                model_peers
                    .entry(shard.model_id.0.clone())
                    .or_default()
                    .insert(format!("{}", peer.node_id));
            }
        }
    }

    // Helper: build per-shard detail for a manifest, including download state
    let build_shard_detail = |m: &crate::types::ModelManifest,
                              state: &AppState|
     -> Vec<serde_json::Value> {
        let acq = state.shared_state.acquisition_progress.get(&m.id);
        m.shards
            .iter()
            .map(|s| {
                let shard_id = crate::types::ShardId {
                    model_id: m.id.clone(),
                    index: s.index,
                };
                let holders = state.shared_state.model_registry.shard_holders(&shard_id);
                let local = holders.contains(&local_node_id);

                // Check if this shard is currently loaded in memory (VRAM or RAM).
                // Check subprocess pool and split_models DashMap.
                // Also check legacy executor — but only if its loaded model matches this model.
                let legacy_loaded = state
                    .shared_state
                    .model_loaded
                    .load(std::sync::atomic::Ordering::Relaxed)
                    && state
                        .shared_state
                        .loaded_model_info
                        .try_read()
                        .map(|info| {
                            info.as_ref().is_some_and(|i| {
                                m.id.0
                                    .contains(&i.name.to_lowercase().replace([' ', '_'], "-"))
                            })
                        })
                        .unwrap_or(false);
                // Determine if this shard is loaded in memory.
                // If a shard window exists, it's the authority on what's loaded
                // (even between worker restarts). Otherwise check if model is loaded.
                let shard_window = state
                    .shared_state
                    .model_process_pool
                    .get_shard_window(&m.id);
                let is_model_loaded = state.shared_state.model_process_pool.is_loaded(&m.id)
                    || state
                        .shared_state
                        .split_models
                        .iter()
                        .any(|e| e.key().0 == m.id)
                    || legacy_loaded;
                let in_vram = if local {
                    match &shard_window {
                        // Explicit window = shard is loaded only if in the window
                        Some(w) => w.contains(&s.index),
                        // No window = all local shards loaded if model is active
                        None => is_model_loaded,
                    }
                } else {
                    false
                };

                let locked = state
                    .shared_state
                    .locked_shards
                    .get(&shard_id)
                    .map(|v| *v)
                    .unwrap_or(false);

                let mut shard_json = serde_json::json!({
                    "index": s.index,
                    "size_bytes": s.size_bytes,
                    "local": local,
                    "holders": holders.len(),
                    "in_vram": in_vram,
                    "locked": locked,
                });

                // Attach per-shard download state if downloading
                if let Some(ref p) = acq {
                    if let Some(sp) = p.shard_progress.get(&s.index) {
                        let dl_state = match sp.state {
                            crate::model::acquisition::ShardState::Downloading => {
                                Some("Downloading")
                            }
                            crate::model::acquisition::ShardState::Verifying => Some("Verifying"),
                            crate::model::acquisition::ShardState::Pending => Some("Queued"),
                            _ => None,
                        };
                        if let Some(state_str) = dl_state {
                            let pct = if sp.total_bytes > 0 {
                                (sp.downloaded_bytes as f64 / sp.total_bytes as f64 * 100.0) as u32
                            } else {
                                0
                            };
                            if let Some(obj) = shard_json.as_object_mut() {
                                obj.insert(
                                    "download".to_string(),
                                    serde_json::json!({
                                        "state": state_str,
                                        "progress_pct": pct,
                                        "downloaded_bytes": sp.downloaded_bytes,
                                        "total_bytes": sp.total_bytes,
                                    }),
                                );
                            }
                        }
                    }
                }

                // Attach peer download state
                if let Some(peer_dl) = state.shared_state.peer_shard_downloads.get(&shard_id) {
                    let peers: Vec<serde_json::Value> = peer_dl
                        .value()
                        .iter()
                        .map(|(nid, pct)| {
                            serde_json::json!({
                                "node_id": format!("{}", nid),
                                "progress_pct": pct,
                            })
                        })
                        .collect();
                    if !peers.is_empty() {
                        if let Some(obj) = shard_json.as_object_mut() {
                            obj.insert("peer_downloads".to_string(), serde_json::json!(peers));
                        }
                    }
                }

                shard_json
            })
            .collect()
    };

    // Helper: check encrypted pipeline status for a model.
    // Returns (effective_flag, has_first_shard, has_last_shard).
    // effective_flag is true only when the DB flag is set AND node holds first+last.
    let encrypted_pipeline_info = |model_id: &str| -> (bool, bool, bool) {
        let mid = crate::types::ModelId(model_id.to_string());
        let flag = state
            .shared_state
            .encrypted_pipeline_models
            .get(&mid)
            .map(|r| *r.value())
            .unwrap_or(state.shared_state.config.inference.encrypted_pipeline);
        let local_node_id = state.shared_state.identity.node_id();
        let has_first = state
            .shared_state
            .model_registry
            .shard_holders(&crate::types::ShardId {
                model_id: mid.clone(),
                index: 0,
            })
            .contains(local_node_id);
        let has_last = if let Some(manifest) = state.shared_state.model_registry.get_manifest(&mid)
        {
            let last_idx = manifest.shard_count.saturating_sub(1);
            state
                .shared_state
                .model_registry
                .shard_holders(&crate::types::ShardId {
                    model_id: mid,
                    index: last_idx,
                })
                .contains(local_node_id)
        } else {
            false
        };
        let effective = flag && has_first && has_last;
        (effective, has_first, has_last)
    };

    // 1. Locally loaded model (full model via --model flag)
    // Even though it's locally loaded, get the real manifest to show shard info
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        // Staleness check: verify model directory still exists on disk.
        // If files were deleted while the process is running, skip this entry
        // to avoid confusing the UI with a "loaded" model that can't run.
        let loaded_model_dir = state.config.node.data_dir.join("models").join(
            info.name
                .to_lowercase()
                .replace(' ', "-")
                .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', ""),
        );
        let has_shard_files = loaded_model_dir.exists()
            && std::fs::read_dir(&loaded_model_dir)
                .ok()
                .map(|rd| {
                    rd.flatten().any(|e| {
                        let name = e.file_name();
                        let n = name.to_string_lossy();
                        n.starts_with("shard_") && n.ends_with(".bin")
                    })
                })
                .unwrap_or(false);

        // Only show loaded model if files exist OR if it was loaded via --model (no shards)
        if !has_shard_files && info.size_bytes > 0 {
            // Stale entry — files deleted while running. Skip.
            // The model will still appear from registry/peers if applicable.
        } else {
            let peer_count = model_peers.get(&info.name).map_or(0, |s| s.len());
            seen_ids.insert(info.name.clone());

            // Try both the display name and the slugified ID to avoid duplicates.
            // The registry may use a slug like "qwen2.5-coder-7b-instruct" while
            // loaded_model_info.name is "Qwen2.5 Coder 7B Instruct".
            let slug = info
                .name
                .to_lowercase()
                .replace(' ', "-")
                .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "");
            seen_ids.insert(slug.clone());

            let mid = crate::types::ModelId(info.name.clone());
            let manifest = state
                .shared_state
                .model_registry
                .get_manifest(&mid)
                .or_else(|| {
                    // Try slug form (registry may use "qwen2.5-coder-7b-instruct" not display name)
                    let slug_id = crate::types::ModelId(slug.clone());
                    state.shared_state.model_registry.get_manifest(&slug_id)
                })
                .or_else(|| {
                    // Try matching by manifest `name` field (auto-manage sets loaded_model_info.name
                    // from manifest.name, but the registry key is manifest.id which may differ)
                    state
                        .shared_state
                        .model_registry
                        .models()
                        .into_iter()
                        .find(|m| m.name == info.name)
                });

            // Mark the manifest's actual registry ID as seen to prevent duplicates
            // in section 2 (which iterates by manifest.id)
            if let Some(ref m) = manifest {
                seen_ids.insert(m.id.0.clone());
            }
            let local_node_id = &state.shared_state.identity.node_id().clone();
            let (shard_count, hosted_local, global_available, shard_detail) = match manifest {
                Some(ref m) => {
                    let detail = build_shard_detail(m, &state);
                    let local_count = (0..m.shard_count)
                        .filter(|&idx| {
                            let sid = crate::types::ShardId {
                                model_id: m.id.clone(),
                                index: idx,
                            };
                            state
                                .shared_state
                                .model_registry
                                .shard_holders(&sid)
                                .contains(local_node_id)
                        })
                        .count();
                    let global = (0..m.shard_count)
                        .filter(|&idx| {
                            let sid = crate::types::ShardId {
                                model_id: m.id.clone(),
                                index: idx,
                            };
                            !state
                                .shared_state
                                .model_registry
                                .shard_holders(&sid)
                                .is_empty()
                        })
                        .count();
                    (m.shard_count, local_count, global, detail)
                }
                None => (1, 1, 1, vec![]),
            };

            // A model is "ready" for inference if all layers are covered across the
            // network — no single node needs all shards. Local shard count doesn't
            // determine readiness; network-wide coverage does.
            let all_covered = global_available == shard_count as usize;
            let status = if all_covered { "loaded" } else { "partial" };

            // Use the manifest's canonical ID when available (matches what /v1/models returns)
            let model_id = manifest
                .as_ref()
                .map(|m| m.id.0.clone())
                .unwrap_or_else(|| slug.clone());
            // Check for manifest.json and gguf_header.bin on disk
            let model_dir = state.config.node.data_dir.join("models").join(&model_id);
            let has_manifest = model_dir.join("manifest.json").exists();
            let has_header = model_dir.join("gguf_header.bin").exists();

            let probed = {
                let mid_check = crate::types::ModelId(model_id.clone());
                has_header
                    || state.shared_state.hf_sources.contains_key(&mid_check)
                    || state.shared_state.hf_probe_cache.contains_key(&mid_check)
            };
            let mmproj_info = {
                let mid_mmproj = crate::types::ModelId(info.name.clone());
                let holders = state
                    .shared_state
                    .model_registry
                    .mmproj_holders(&mid_mmproj);
                let local_has = holders.contains(local_node_id);
                serde_json::json!({
                    "available": !holders.is_empty(),
                    "local": local_has,
                    "holders": holders.len(),
                })
            };
            let trust_level = state
                .shared_state
                .model_trust
                .get(&crate::types::ModelId(model_id.clone()))
                .map(|t| t.trust_level.to_string())
                .unwrap_or_else(|| "pinned".to_string()); // loaded models are at least pinned
            let enc_info = encrypted_pipeline_info(&model_id);
            models.push(serde_json::json!({
                "id": model_id,
                "name": info.name,
                "total_size_bytes": info.size_bytes,
                "shard_count": shard_count,
                "hosted_shards": hosted_local,
                "global_available": global_available,
                "healthy": all_covered,
                "status": status,
                "mode": if hosted_local == shard_count as usize { "full" } else { "distributed" },
                "source": "local",
                "local": true,
                "peers_hosting": peer_count,
                "shards": shard_detail,
                "has_manifest": has_manifest,
                "has_header": has_header,
                "probed": probed,
                "mmproj": mmproj_info,
                "trust_level": trust_level,
                "encrypted_pipeline": enc_info.0,
                "has_first_shard": enc_info.1,
                "has_last_shard": enc_info.2,
            }));
        } // else: stale loaded model, files deleted
    }

    // 2. Models from the P2P manifest registry
    let registry = &state.shared_state.model_registry;
    let manifests = registry.models();

    for m in &manifests {
        if seen_ids.contains(&m.id.0) {
            continue;
        }
        seen_ids.insert(m.id.0.clone());

        let hosted_count = (0..m.shard_count)
            .filter(|&idx| {
                let shard_id = crate::types::ShardId {
                    model_id: m.id.clone(),
                    index: idx,
                };
                let holders = state.shared_state.model_registry.shard_holders(&shard_id);
                holders.contains(&local_node_id)
            })
            .count();

        let peer_count = model_peers.get(&m.id.0).map_or(0, |s| s.len());
        let shard_detail = build_shard_detail(m, &state);

        let (source, mode) = if hosted_count == m.shard_count as usize {
            ("local", "full")
        } else if hosted_count > 0 {
            ("hybrid", "distributed")
        } else {
            ("network", "distributed")
        };

        // Compute global shard availability (any holder, not just local)
        let global_available = (0..m.shard_count)
            .filter(|&idx| {
                let shard_id = crate::types::ShardId {
                    model_id: m.id.clone(),
                    index: idx,
                };
                !state
                    .shared_state
                    .model_registry
                    .shard_holders(&shard_id)
                    .is_empty()
            })
            .count();

        // Check if the model is loaded and ready for inference.
        // A model is "ready" when all layers are covered across the network —
        // no single node needs all shards. Nodes participate with whatever
        // shards they have; the pipeline scheduler assembles the full pipeline.
        let is_loaded = if hosted_count > 0 {
            state
                .shared_state
                .split_models
                .iter()
                .any(|e| e.key().0 == m.id)
                || state.shared_state.model_process_pool.is_loaded(&m.id)
        } else {
            false
        };
        let all_covered = global_available == m.shard_count as usize;

        let status = if is_loaded && all_covered {
            // Local segments loaded AND all layers covered network-wide
            "loaded"
        } else if all_covered {
            // All layers covered network-wide — ready for distributed inference
            "ready"
        } else if hosted_count > 0 {
            // This node hosts some shards but network doesn't cover all layers yet
            "partial"
        } else {
            // Known model but no shards held locally
            "available"
        };

        // Check acquisition progress — clean up completed entries
        let acq_state = state
            .shared_state
            .acquisition_progress
            .get(&m.id)
            .and_then(|entry| {
                let s = &entry.state;
                match s {
                    crate::model::acquisition::AcquisitionState::Downloading => Some("downloading"),
                    crate::model::acquisition::AcquisitionState::Failed { .. } => Some("failed"),
                    // Don't report "complete" — let the status field handle readiness
                    _ => None,
                }
            });
        let acq_progress = if acq_state == Some("downloading") {
            state
                .shared_state
                .acquisition_progress
                .get(&m.id)
                .map(|entry| {
                    serde_json::json!({
                        "downloaded_bytes": entry.downloaded_bytes,
                        "total_bytes": entry.total_bytes,
                        "downloaded_shards": entry.downloaded_shards,
                        "speed_bytes_per_sec": entry.speed_bytes_per_sec,
                        "source": entry.source,
                        "trigger": entry.trigger,
                    })
                })
        } else {
            None
        };

        let estimated_vram = crate::model::auto_manage::estimate_model_vram_mb(m.total_size_bytes);

        // Check for manifest.json and gguf_header.bin on disk
        let model_dir = state.config.node.data_dir.join("models").join(&m.id.0);
        let has_manifest = model_dir.join("manifest.json").exists();
        let has_header = model_dir.join("gguf_header.bin").exists();

        let probed = has_header
            || state.shared_state.hf_sources.contains_key(&m.id)
            || state.shared_state.hf_probe_cache.contains_key(&m.id);
        let mmproj_info_reg = {
            let holders = state.shared_state.model_registry.mmproj_holders(&m.id);
            let local_has = holders.contains(&local_node_id);
            serde_json::json!({
                "available": !holders.is_empty(),
                "local": local_has,
                "holders": holders.len(),
            })
        };
        let trust_level = state
            .shared_state
            .model_trust
            .get(&m.id)
            .map(|t| t.trust_level.to_string())
            .unwrap_or_else(|| "discovered".to_string());
        let enc_info_reg = encrypted_pipeline_info(&m.id.0);
        models.push(serde_json::json!({
            "id": m.id.0,
            "name": m.name,
            "total_size_bytes": m.total_size_bytes,
            "shard_count": m.shard_count,
            "hosted_shards": hosted_count,
            "global_available": global_available,
            "healthy": global_available == m.shard_count as usize,
            "status": status,
            "mode": mode,
            "source": source,
            "local": hosted_count > 0,
            "peers_hosting": peer_count,
            "shards": shard_detail,
            "estimated_vram_mb": estimated_vram,
            "has_manifest": has_manifest,
            "has_header": has_header,
            "probed": probed,
            "acquisition": acq_state,
            "acquisition_progress": acq_progress,
            "mmproj": mmproj_info_reg,
            "trust_level": trust_level,
            "encrypted_pipeline": enc_info_reg.0,
                "has_first_shard": enc_info_reg.1,
                "has_last_shard": enc_info_reg.2,
        }));
    }

    // 3. Models discovered from peer announcements (not in our registry or loaded)
    for (model_name, peers) in &model_peers {
        if seen_ids.contains(model_name) {
            continue;
        }
        seen_ids.insert(model_name.clone());
        let trust_level = state
            .shared_state
            .model_trust
            .get(&crate::types::ModelId(model_name.clone()))
            .map(|t| t.trust_level.to_string())
            .unwrap_or_else(|| "discovered".to_string());
        let enc_info_net = encrypted_pipeline_info(model_name);
        models.push(serde_json::json!({
            "id": model_name,
            "name": model_name,
            "total_size_bytes": 0,
            "shard_count": 0,
            "hosted_shards": 0,
            "healthy": true,
            "status": "available",
            "mode": "full",
            "source": "network",
            "local": false,
            "peers_hosting": peers.len(),
            "shards": [],
            "trust_level": trust_level,
            "encrypted_pipeline": enc_info_net.0,
                "has_first_shard": enc_info_net.1,
                "has_last_shard": enc_info_net.2,
        }));
    }

    Json(models)
}

/// POST /api/admin/models/:id/add — Express interest in a model (trigger download).
pub async fn add_model_interest(
    State(state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!(model_id = %model_id, "Model acquisition requested");

    let mid = crate::types::ModelId(model_id.clone());

    // Send acquisition command if the channel is wired up
    if let Some(ref tx) = state.acquisition_tx {
        tx.send(crate::model::acquisition::AcquisitionCommand::Acquire { model_id: mid })
            .await
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to send acquisition command: {e}"
                )))
            })?;

        Ok(Json(serde_json::json!({
            "status": "acquiring",
            "model_id": model_id,
        })))
    } else {
        // Standalone mode — no acquisition manager
        Ok(Json(serde_json::json!({
            "status": "unavailable",
            "model_id": model_id,
            "message": "Model acquisition requires daemon mode with P2P networking",
        })))
    }
}

/// GET /api/admin/models/:id/status — Query model acquisition progress.
///
/// Reads directly from the shared `acquisition_progress` DashMap for low-latency,
/// lock-free progress reporting without going through the AcquisitionManager channel.
pub async fn model_acquisition_status(
    State(state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());

    // Fast path: read from shared state (no channel round-trip)
    if let Some(status) = state.shared_state.acquisition_progress.get(&mid) {
        return Ok(Json(
            serde_json::to_value(status.value()).unwrap_or_default(),
        ));
    }

    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "state": "unknown",
        "message": "No active acquisition for this model",
    })))
}

/// GET /api/admin/shard-storage — Show per-model shard storage usage.
///
/// Returns a list of all models with storage breakdown per shard,
/// plus a total storage summary. Used by the auto-manage UI.
pub async fn shard_storage(State(state): State<AppState>) -> Json<serde_json::Value> {
    let local_node_id = state.shared_state.identity.node_id().clone();
    let models_dir = state.config.node.data_dir.join("models");

    let mut model_storage: Vec<serde_json::Value> = Vec::new();
    let mut total_local_bytes: u64 = 0;

    for manifest in state.shared_state.model_registry.models() {
        let mut local_shards = 0u32;
        let mut local_bytes = 0u64;
        let mut shard_details: Vec<serde_json::Value> = Vec::new();

        // Check if any shards are currently being downloaded
        let mut any_downloading = false;

        // Get per-shard download progress for this model (if any)
        let acq_progress = state.shared_state.acquisition_progress.get(&manifest.id);

        for shard in &manifest.shards {
            let shard_id = crate::types::ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            };
            let holders = state.shared_state.model_registry.shard_holders(&shard_id);
            let is_local = holders.contains(&local_node_id);

            if is_local {
                local_shards += 1;
                local_bytes += shard.size_bytes;
            }

            // Only attach download state to the SPECIFIC shard being downloaded
            let download_state = acq_progress.as_ref().and_then(|p| {
                let p = p.value();
                // Check per-shard progress first (populated by auto-manage)
                if let Some(sp) = p.shard_progress.get(&shard.index) {
                    if matches!(sp.state, crate::model::acquisition::ShardState::Downloading) {
                        any_downloading = true;
                        return Some(serde_json::json!({
                            "state": "Downloading",
                            "progress_pct": if sp.total_bytes > 0 {
                                (sp.downloaded_bytes as f64 / sp.total_bytes as f64 * 100.0) as u32
                            } else { 0 },
                            "downloaded_bytes": sp.downloaded_bytes,
                            "total_bytes": sp.total_bytes,
                        }));
                    }
                }
                None
            });

            // Also check peer download states (from gossip)
            let peer_downloading =
                state
                    .shared_state
                    .peer_shard_downloads
                    .get(&shard_id)
                    .map(|entry| {
                        let peers: Vec<serde_json::Value> = entry.value().iter().map(|(nid, pct)| {
                        serde_json::json!({ "node_id": format!("{}", nid), "progress_pct": pct })
                    }).collect();
                        peers
                    });

            let locked = state.shared_state.locked_shards.contains_key(&shard_id);
            let mut shard_json = serde_json::json!({
                "index": shard.index,
                "size_bytes": shard.size_bytes,
                "local": is_local,
                "holders": holders.len(),
                "locked": locked,
            });
            if let Some(dl) = download_state {
                if let Some(obj) = shard_json.as_object_mut() {
                    obj.insert("download".to_string(), dl);
                }
            }
            if let Some(peers_dl) = peer_downloading {
                if !peers_dl.is_empty() {
                    if let Some(obj) = shard_json.as_object_mut() {
                        obj.insert("peer_downloads".to_string(), serde_json::json!(peers_dl));
                    }
                    any_downloading = true;
                }
            }
            shard_details.push(shard_json);
        }

        total_local_bytes += local_bytes;

        let estimated_vram =
            crate::model::auto_manage::estimate_model_vram_mb(manifest.total_size_bytes);

        // Determine model readiness
        let all_shards_available = shard_details.iter().all(|s| {
            s.get("local").and_then(|v| v.as_bool()).unwrap_or(false)
                || s.get("holders").and_then(|v| v.as_u64()).unwrap_or(0) > 0
        });
        let all_local = local_shards == manifest.shard_count;
        let ready_status = if all_local {
            "ready"
        } else if any_downloading {
            "downloading"
        } else if all_shards_available {
            "available"
        } else {
            "incomplete"
        };

        model_storage.push(serde_json::json!({
            "id": manifest.id.0,
            "name": manifest.name,
            "total_size_bytes": manifest.total_size_bytes,
            "shard_count": manifest.shard_count,
            "local_shards": local_shards,
            "local_bytes": local_bytes,
            "estimated_vram_mb": estimated_vram,
            "shards": shard_details,
            "ready": ready_status,
            "all_shards_available": all_shards_available,
        }));
    }

    // Get actual disk usage of models dir
    let disk_usage_bytes = dir_size(&models_dir).unwrap_or(0);

    // Compute global VRAM pool
    let pool_vram_mb = crate::model::auto_manage::global_pool_vram_mb(&state.shared_state);
    let local_vram_mb = crate::model::auto_manage::local_vram_mb(&state.shared_state);

    Json(serde_json::json!({
        "models": model_storage,
        "total_local_bytes": total_local_bytes,
        "disk_usage_bytes": disk_usage_bytes,
        "auto_manage_enabled": state.config.auto_manage.enabled,
        "auto_manage_max_storage_mb": state.config.auto_manage.max_storage_mb,
        "max_disk_mb": state.config.resources.max_disk_mb,
        "pool_vram_mb": pool_vram_mb,
        "local_vram_mb": local_vram_mb,
        "peer_count": state.shared_state.peer_registry.len(),
    }))
}

/// Recursively compute total size of a directory.
fn dir_size(path: &std::path::Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let ft = entry.file_type()?;
            // Skip symlinks to avoid cycles and traversal attacks
            if ft.is_symlink() {
                continue;
            }
            if ft.is_file() {
                total += entry.metadata()?.len();
            } else if ft.is_dir() {
                total += dir_size(&entry.path())?;
            }
        }
    }
    Ok(total)
}
/// DELETE /api/admin/models/:model_id — Remove a model and all its shard files.
///
/// Removes shard files from disk, clears manifest from DB, removes from SharedState
/// registries. Returns 200 on success, 404 if model not found.
pub async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Sanitize model_id to prevent path traversal
    let safe_model_id = crate::model::shard::sanitize_path_component(&model_id);
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Verify the model exists
    if shared.model_registry.get_manifest(&mid).is_none() {
        return Err(ApiError(crate::error::SwarmError::Config(format!(
            "Model '{}' not found",
            model_id
        ))));
    }

    let node_id = shared.identity.node_id().clone();

    // Remove shard files from disk
    let model_dir = state
        .config
        .node
        .data_dir
        .join("models")
        .join(&safe_model_id);
    let mut files_removed = 0u32;
    if model_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    if let Err(e) = std::fs::remove_file(&path) {
                        tracing::warn!(path = %path.display(), error = %e, "Failed to remove shard file");
                    } else {
                        files_removed += 1;
                    }
                }
            }
        }
        // Remove the model directory itself
        let _ = std::fs::remove_dir(&model_dir);
    }

    // Remove manifest from DB
    let _ = shared.db.remove("model_meta", &model_id);

    // Remove HF source from DB
    let _ = shared.db.remove("hf_sources", &model_id);

    // S5: Collect local shards before removal for DHT stop_providing
    let local_shards: Vec<_> = shared
        .model_registry
        .shards_for_node(&node_id)
        .into_iter()
        .filter(|s| s.model_id == mid)
        .collect();

    // Remove from SharedState registries
    shared.model_registry.remove_manifest(&mid);
    shared.model_registry.remove_all_model_shards(&mid);

    // S5: Stop providing deleted shards via DHT
    if !local_shards.is_empty() {
        if let Some(ref ntx) = state.network_tx {
            let _ = ntx.try_send(crate::types::NetworkCommand::StopProviding(local_shards));
        }
    }

    // Remove from acquisition_progress
    shared.acquisition_progress.remove(&mid);

    // Remove from gguf_meta
    shared.gguf_meta.remove(&mid);

    // Remove from hf_sources
    shared.hf_sources.remove(&mid);

    // Remove split models for this model
    shared.split_models.retain(|key, _| key.0 != mid);

    // Kill the worker subprocess to free GPU memory
    shared.model_process_pool.unload_model(&mid).await;
    shared.model_process_pool.clear_shard_window(&mid);

    // Broadcast shard removal via GossipSub
    if let Some(ref ntx) = state.network_tx {
        let announce = crate::types::SwarmMessage::ShardAnnounce(crate::types::ShardAnnounce {
            node_id: node_id.clone(),
            shards: vec![], // Empty shards = we no longer host anything for this model
            timestamp: chrono::Utc::now(),
        });
        let _ = ntx
            .send(crate::types::NetworkCommand::Broadcast(announce))
            .await;
    }

    tracing::info!(model = %model_id, files = files_removed, "Model deleted");

    Ok(Json(serde_json::json!({
        "status": "deleted",
        "model_id": model_id,
        "files_removed": files_removed,
    })))
}

/// POST /api/admin/models/:model_id/unload — Unload a model from memory without deleting files.
///
/// Clears split models and loaded model info for the given model, freeing VRAM/memory.
/// Shard files, manifests, and registry entries remain intact for future re-loading.
pub async fn unload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Get model name and estimated size before removing
    let model_display_name = shared
        .model_registry
        .get_manifest(&mid)
        .map(|m| m.name.clone())
        .unwrap_or_else(|| model_id.clone());
    let estimated_mb = shared
        .model_registry
        .get_manifest(&mid)
        .map(|m| crate::model::auto_manage::estimate_model_vram_mb(m.total_size_bytes))
        .unwrap_or(0);

    // Remove split models for this model
    let mut segments_removed = 0u32;
    shared.split_models.retain(|key, _| {
        if key.0 == mid {
            segments_removed += 1;
            false
        } else {
            true
        }
    });

    // Kill the worker subprocess to free GPU memory
    shared.model_process_pool.unload_model(&mid).await;
    shared.model_process_pool.clear_shard_window(&mid);

    // Clear loaded model info if it matches this model
    {
        let mut info = shared.loaded_model_info.write().await;
        if info.as_ref().map(|i| i.name == model_id).unwrap_or(false) {
            *info = None;
            shared
                .model_loaded
                .store(false, std::sync::atomic::Ordering::SeqCst);
        }
    }

    // Clear GGUF metadata cache for this model
    shared.gguf_meta.remove(&mid);

    // Notify dashboard
    let _ = shared.models_changed_tx.send(());

    // Emit activity event
    shared.emit_activity(crate::daemon::state::ActivityEvent {
        category: "model",
        kind: "model_unloaded",
        message: format!(
            "Unloaded {} from memory (~{}MB freed)",
            model_display_name, estimated_mb
        ),
        model_id: Some(model_id.clone()),
        model_name: Some(model_display_name.clone()),
        node_id: None,
        detail_num: Some(estimated_mb as i64),
        detail_str: None,
    });

    tracing::info!(model = %model_id, segments = segments_removed, "Model unloaded from memory");

    Ok(Json(serde_json::json!({
        "status": "unloaded",
        "model_id": model_id,
        "model_name": model_display_name,
        "segments_removed": segments_removed,
        "estimated_freed_mb": estimated_mb,
    })))
}

/// POST /api/admin/models/:model_id/shards/:shard_index/unload — Unload a single shard from memory.
///
/// Narrows the shard window to exclude this shard, then restarts the model worker.
/// The shard file stays on disk. The worker reloads with the remaining shards.
pub async fn unload_shard(
    State(state): State<AppState>,
    Path((model_id, shard_index)): Path<(String, u32)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // SEC: Reject mmproj sentinel index
    if shard_index == u32::MAX {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Reserved shard index".into(),
        )));
    }
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Validate shard is local
    let local_node_id = shared.identity.node_id().clone();
    let shard_id = crate::types::ShardId {
        model_id: mid.clone(),
        index: shard_index,
    };
    if !shared
        .model_registry
        .shard_holders(&shard_id)
        .contains(&local_node_id)
    {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Shard is not held locally".into(),
        )));
    }

    // Get current shard window (or all local shard indices if no window set)
    let current_window = shared.model_process_pool.get_shard_window(&mid);
    let all_local: Vec<u32> = shared
        .model_registry
        .get_manifest(&mid)
        .map(|m| {
            let local_node_id = shared.identity.node_id().clone();
            m.shards
                .iter()
                .filter(|s| {
                    let sid = crate::types::ShardId {
                        model_id: mid.clone(),
                        index: s.index,
                    };
                    shared
                        .model_registry
                        .shard_holders(&sid)
                        .contains(&local_node_id)
                })
                .map(|s| s.index)
                .collect()
        })
        .unwrap_or_default();

    let window = current_window.unwrap_or(all_local);
    let new_window: Vec<u32> = window.into_iter().filter(|&i| i != shard_index).collect();

    if new_window.is_empty() {
        // Unloading the last shard = unload the model entirely
        shared.model_process_pool.unload_model(&mid).await;
        shared.split_models.retain(|key, _| key.0 != mid);
    } else {
        // Narrow window and restart worker — next inference request
        // respawns loading only the remaining shards
        shared
            .model_process_pool
            .restart_with_window(&mid, new_window.clone())
            .await;
        // Remove split model entries so they reload with new window
        shared.split_models.retain(|key, _| key.0 != mid);
    }

    let _ = shared.models_changed_tx.send(());

    {
        let mname = shared
            .model_registry
            .get_manifest(&mid)
            .map(|m| m.name.clone());
        let display = mname.as_deref().unwrap_or(&model_id);
        let remaining = if new_window.is_empty() {
            "model fully unloaded".to_string()
        } else {
            let nums: Vec<_> = new_window.iter().map(|i| (i + 1).to_string()).collect();
            format!("shards {} remain", nums.join(", "))
        };
        shared.emit_activity(crate::daemon::state::ActivityEvent {
            category: "model",
            kind: "shard_unloaded_memory",
            message: format!(
                "Unloaded shard {} of {} — {}",
                shard_index + 1,
                display,
                remaining
            ),
            model_id: Some(model_id.clone()),
            model_name: mname,
            node_id: None,
            detail_num: Some(shard_index as i64),
            detail_str: None,
        });
    }

    tracing::info!(
        model = %model_id,
        shard = shard_index,
        remaining = ?new_window,
        "Shard unloaded from memory"
    );

    Ok(Json(serde_json::json!({
        "status": "unloaded",
        "model_id": model_id,
        "shard_index": shard_index,
        "remaining_loaded": new_window,
    })))
}

/// POST /api/admin/models/:model_id/shards/:shard_index/load — Load a single shard into memory.
///
/// Adds the shard to the shard window and restarts the worker so it picks up the new shard.
/// The shard must already exist on disk (local).
pub async fn load_shard(
    State(state): State<AppState>,
    Path((model_id, shard_index)): Path<(String, u32)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // SEC: Reject mmproj sentinel index
    if shard_index == u32::MAX {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Reserved shard index".into(),
        )));
    }
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Verify the shard exists on disk
    let shard_id = crate::types::ShardId {
        model_id: mid.clone(),
        index: shard_index,
    };
    let local_node_id = shared.identity.node_id().clone();
    if !shared
        .model_registry
        .shard_holders(&shard_id)
        .contains(&local_node_id)
    {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Shard is not on disk — download it first".into(),
        )));
    }

    // Expand the shard window to include this shard.
    // If no window exists, start from all local shards (same as unload_shard).
    let current_window = shared.model_process_pool.get_shard_window(&mid);
    let mut new_window = current_window.unwrap_or_else(|| {
        shared
            .model_registry
            .get_manifest(&mid)
            .map(|m| {
                m.shards
                    .iter()
                    .filter(|s| {
                        let sid = crate::types::ShardId {
                            model_id: mid.clone(),
                            index: s.index,
                        };
                        shared
                            .model_registry
                            .shard_holders(&sid)
                            .contains(&local_node_id)
                    })
                    .map(|s| s.index)
                    .collect()
            })
            .unwrap_or_default()
    });
    if !new_window.contains(&shard_index) {
        new_window.push(shard_index);
        new_window.sort();
    }

    // Restart worker with expanded window
    shared
        .model_process_pool
        .restart_with_window(&mid, new_window.clone())
        .await;
    // Clear split models so they reload with new window
    shared.split_models.retain(|key, _| key.0 != mid);

    let _ = shared.models_changed_tx.send(());

    {
        let mname = shared
            .model_registry
            .get_manifest(&mid)
            .map(|m| m.name.clone());
        let display = mname.as_deref().unwrap_or(&model_id);
        let window_label = if new_window.len() == 1 {
            format!("shard {}", new_window[0] + 1)
        } else {
            let nums: Vec<_> = new_window.iter().map(|i| (i + 1).to_string()).collect();
            format!("shards {}", nums.join(", "))
        };
        shared.emit_activity(crate::daemon::state::ActivityEvent {
            category: "model",
            kind: "shard_loaded_memory",
            message: format!("Loaded {} — {} now in memory", display, window_label),
            model_id: Some(model_id.clone()),
            model_name: mname,
            node_id: None,
            detail_num: Some(shard_index as i64),
            detail_str: None,
        });
    }

    tracing::info!(
        model = %model_id,
        shard = shard_index,
        loaded = ?new_window,
        "Shard loaded into memory"
    );

    Ok(Json(serde_json::json!({
        "status": "loaded",
        "model_id": model_id,
        "shard_index": shard_index,
        "loaded_shards": new_window,
    })))
}

/// DELETE /api/admin/models/:model_id/shards/:shard_index — Remove a single shard.
///
/// Deletes the shard file from disk, removes self from shard_holders in model_registry,
/// and broadcasts updated ShardAnnounce. Keeps manifest, header, and other shards intact.
pub async fn delete_shard(
    State(state): State<AppState>,
    Path((model_id, shard_index)): Path<(String, u32)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let safe_model_id = crate::model::shard::sanitize_path_component(&model_id);
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;
    let local_node_id = shared.identity.node_id().clone();

    // Verify shard exists in registry
    let shard_id = crate::types::ShardId {
        model_id: mid.clone(),
        index: shard_index,
    };
    let holders = shared.model_registry.shard_holders(&shard_id);
    if !holders.contains(&local_node_id) {
        return Err(ApiError(crate::error::SwarmError::Config(format!(
            "Shard {} of model '{}' is not held locally",
            shard_index, model_id
        ))));
    }

    // Delete shard file from disk
    let shard_path = state
        .config
        .node
        .data_dir
        .join("models")
        .join(&safe_model_id)
        .join(format!("shard_{:03}.bin", shard_index));

    if shard_path.exists() {
        std::fs::remove_file(&shard_path).map_err(|e| ApiError(crate::error::SwarmError::Io(e)))?;
    }

    // Remove self from shard_holders
    shared
        .model_registry
        .remove_shard_holder(&shard_id, &local_node_id);
    // S5: Stop providing this shard via DHT
    if let Some(ref ntx) = state.network_tx {
        let _ = ntx.try_send(crate::types::NetworkCommand::StopProviding(vec![
            shard_id.clone()
        ]));
    }

    // Evict any cached split model segments that included this shard
    shared.split_models.retain(|key, _| key.0 != mid);

    // Kill the worker subprocess to free GPU memory and clear the shard window
    // so next spawn doesn't try to load the deleted shard
    shared.model_process_pool.unload_model(&mid).await;
    shared.model_process_pool.clear_shard_window(&mid);

    // Broadcast updated ShardAnnounce with remaining held shards
    if let Some(ref ntx) = state.network_tx {
        let remaining_shards: Vec<crate::types::ShardId> = shared
            .model_registry
            .all_shard_entries()
            .iter()
            .filter(|(sid, holders)| sid.model_id == mid && holders.contains(&local_node_id))
            .map(|(sid, _)| sid.clone())
            .collect();

        let announce = crate::types::SwarmMessage::ShardAnnounce(crate::types::ShardAnnounce {
            node_id: local_node_id,
            shards: remaining_shards,
            timestamp: chrono::Utc::now(),
        });
        let _ = ntx
            .send(crate::types::NetworkCommand::Broadcast(announce))
            .await;
    }

    {
        let mname = shared
            .model_registry
            .get_manifest(&mid)
            .map(|m| m.name.clone());
        shared.emit_activity(crate::daemon::state::ActivityEvent {
            category: "model",
            kind: "shard_deleted",
            message: format!(
                "Deleted shard {} of {} from disk",
                shard_index + 1,
                mname.as_deref().unwrap_or(&model_id)
            ),
            model_id: Some(model_id.clone()),
            model_name: mname,
            node_id: None,
            detail_num: Some(shard_index as i64),
            detail_str: None,
        });
    }

    tracing::info!(model = %model_id, shard = shard_index, "Shard removed");

    Ok(Json(serde_json::json!({
        "status": "deleted",
        "model_id": model_id,
        "shard_index": shard_index,
    })))
}

/// POST /api/admin/models/:model_id/shards/:shard_index/download — Download a single shard.
///
/// Tries P2P first (if peers hold the shard), falls back to HuggingFace.
/// Creates an acquisition_progress entry for the download bar.
pub async fn download_shard(
    State(state): State<AppState>,
    Path((model_id, shard_index)): Path<(String, u32)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    let shard_id = crate::types::ShardId {
        model_id: mid.clone(),
        index: shard_index,
    };

    // Check if we already have it locally
    let local_node_id = shared.identity.node_id().clone();
    if shared
        .model_registry
        .shard_holders(&shard_id)
        .contains(&local_node_id)
    {
        return Ok(Json(serde_json::json!({
            "status": "already_local",
            "model_id": model_id,
            "shard_index": shard_index,
        })));
    }

    // Get shard size from manifest
    let shard_size = shared
        .model_registry
        .get_manifest(&mid)
        .and_then(|m| {
            m.shards
                .iter()
                .find(|s| s.index == shard_index)
                .map(|s| s.size_bytes)
        })
        .unwrap_or(0);

    // Try P2P: find peers who hold this shard
    let holders: Vec<_> = shared
        .model_registry
        .shard_holders(&shard_id)
        .into_iter()
        .filter(|n| n != &local_node_id)
        .collect();

    if !holders.is_empty() {
        // Pick the best peer: LAN first, then lowest latency, then highest trust
        let target = {
            let mut scored: Vec<_> = holders
                .iter()
                .filter_map(|nid| {
                    shared.peer_registry.get(nid).map(|p| {
                        let is_lan = if p.is_lan_peer { 0u64 } else { 1 };
                        let latency = p.latency_ms.unwrap_or(9999) as u64;
                        let trust = (10000.0 - p.trust_score * 100.0) as u64;
                        (nid.clone(), is_lan * 100_000 + latency * 100 + trust)
                    })
                })
                .collect();
            scored.sort_by_key(|(_, score)| *score);
            scored
                .first()
                .map(|(nid, _)| nid.clone())
                .unwrap_or_else(|| holders[0].clone())
        };

        let peer_id_bytes = shared
            .peer_registry
            .get(&target)
            .and_then(|p| p.peer_id_bytes.clone());

        if let Some(bytes) = peer_id_bytes {
            // Create acquisition_progress for the download bar
            let mut shard_progress = std::collections::HashMap::new();
            shard_progress.insert(
                shard_index,
                crate::model::acquisition::ShardProgress {
                    index: shard_index,
                    total_bytes: shard_size,
                    downloaded_bytes: 0,
                    state: crate::model::acquisition::ShardState::Downloading,
                },
            );
            shared.acquisition_progress.insert(
                mid.clone(),
                crate::model::acquisition::AcquisitionStatus {
                    model_id: mid.clone(),
                    state: crate::model::acquisition::AcquisitionState::Downloading,
                    total_shards: 1,
                    downloaded_shards: 0,
                    verified_shards: 0,
                    failed_shards: 0,
                    total_bytes: shard_size,
                    downloaded_bytes: 0,
                    shard_progress,
                    speed_bytes_per_sec: 0,
                    started_at: Some(chrono::Utc::now()),
                    log: vec![format!("Downloading shard {} from peer", shard_index + 1)],
                    source: "peers".to_string(),
                    trigger: "user".to_string(),
                },
            );

            let request = crate::types::ShardRequest {
                shard_id,
                chunk_offset: 0,
                chunk_size: 32 * 1024 * 1024,
            };
            if let Some(ref tx) = state.network_tx {
                let _ = tx
                    .send(crate::types::NetworkCommand::SendShardRequest {
                        target_peer_bytes: bytes,
                        request,
                    })
                    .await;
            }

            let mname = shared
                .model_registry
                .get_manifest(&mid)
                .map(|m| m.name.clone());
            let peer_label = shared
                .nickname_registry
                .get(&target)
                .map(|r| r.nickname.clone())
                .unwrap_or_else(|| format!("{}", target).chars().take(8).collect());
            shared.emit_activity(crate::daemon::state::ActivityEvent {
                category: "download",
                kind: "shard_download_p2p",
                message: format!(
                    "Downloading shard {} of {} from peer {}",
                    shard_index + 1,
                    mname.as_deref().unwrap_or(&model_id),
                    peer_label
                ),
                model_id: Some(model_id.clone()),
                model_name: mname,
                node_id: Some(format!("{}", target)),
                detail_num: Some(shard_index as i64),
                detail_str: Some("p2p".to_string()),
            });

            return Ok(Json(serde_json::json!({
                "status": "downloading",
                "source": "p2p",
                "model_id": model_id,
                "shard_index": shard_index,
                "peer": format!("{}", target).chars().take(16).collect::<String>(),
            })));
        }
    }

    // Fallback: HuggingFace (if source exists)
    if let Some(hf) = shared.hf_sources.get(&mid) {
        return Ok(Json(serde_json::json!({
            "status": "use_hf",
            "source": "huggingface",
            "model_id": model_id,
            "shard_index": shard_index,
            "repo_id": hf.repo_id.clone(),
            "filename": hf.filename.clone(),
        })));
    }

    Err(ApiError(crate::error::SwarmError::Validation(
        "No peers hold this shard and no HuggingFace source available".into(),
    )))
}

/// GET /api/admin/models/:model_id/auto-manage — Get per-model auto-manage policy.
pub async fn get_model_auto_manage(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Json<serde_json::Value> {
    let mid = crate::types::ModelId(model_id.clone());
    let default_cap = state
        .shared_state
        .auto_manage_default_model_cap
        .load(std::sync::atomic::Ordering::Relaxed);

    match state.shared_state.model_auto_manage_policies.get(&mid) {
        Some(policy) => Json(serde_json::json!({
            "model_id": model_id,
            "enabled": policy.enabled,
            "max_shards": policy.max_shards,
            "prune_enabled": policy.prune_enabled,
        })),
        None => Json(serde_json::json!({
            "model_id": model_id,
            "enabled": true,
            "max_shards": default_cap,
            "prune_enabled": true,
        })),
    }
}

/// PUT /api/admin/models/:model_id/auto-manage — Set per-model auto-manage policy.
#[derive(Debug, Deserialize)]
pub struct ModelAutoManageUpdate {
    pub enabled: Option<bool>,
    pub max_shards: Option<u32>,
    pub prune_enabled: Option<bool>,
}

pub async fn set_model_auto_manage(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Json(body): Json<ModelAutoManageUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());

    let policy = crate::config::ModelAutoManagePolicy {
        enabled: body.enabled.unwrap_or(true),
        max_shards: body.max_shards.unwrap_or(0),
        prune_enabled: body.prune_enabled.unwrap_or(true),
    };

    // Update in-memory
    state
        .shared_state
        .model_auto_manage_policies
        .insert(mid.clone(), policy.clone());

    // Persist to database
    let _ = state
        .shared_state
        .db
        .put_json("model_auto_manage_policies", &model_id, &policy);

    // Wake auto-manage to re-evaluate
    state.shared_state.auto_manage_notify.notify_one();

    tracing::info!(
        model = %model_id,
        enabled = policy.enabled,
        max_shards = policy.max_shards,
        prune_enabled = policy.prune_enabled,
        "Per-model auto-manage policy updated"
    );

    Ok(Json(serde_json::json!({
        "status": "ok",
        "model_id": model_id,
        "enabled": policy.enabled,
        "max_shards": policy.max_shards,
        "prune_enabled": policy.prune_enabled,
    })))
}

// ---- Encrypted Pipeline API ----

/// GET /api/admin/models/:model_id/encrypted-pipeline — Get encrypted pipeline status.
pub async fn get_model_encrypted_pipeline(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Json<serde_json::Value> {
    let mid = crate::types::ModelId(model_id.clone());
    let global_default = state.shared_state.config.inference.encrypted_pipeline;
    let per_model = state
        .shared_state
        .encrypted_pipeline_models
        .get(&mid)
        .map(|r| *r.value());
    let effective = per_model.unwrap_or(global_default);

    // Check if the local node has the required shards (first + last)
    let manifest = state.shared_state.model_registry.get_manifest(&mid);
    let local_node_id = state.shared_state.identity.node_id();
    let (has_first, has_last, shard_count) = if let Some(m) = &manifest {
        let has_first = state
            .shared_state
            .model_registry
            .shard_holders(&crate::types::ShardId {
                model_id: mid.clone(),
                index: 0,
            })
            .contains(local_node_id);
        let last_idx = m.shard_count.saturating_sub(1);
        let has_last = state
            .shared_state
            .model_registry
            .shard_holders(&crate::types::ShardId {
                model_id: mid.clone(),
                index: last_idx,
            })
            .contains(local_node_id);
        (has_first, has_last, m.shard_count)
    } else {
        (false, false, 0)
    };

    Json(serde_json::json!({
        "model_id": model_id,
        "encrypted_pipeline": effective,
        "per_model_override": per_model,
        "global_default": global_default,
        "ready": has_first && has_last,
        "has_first_shard": has_first,
        "has_last_shard": has_last,
        "shard_count": shard_count,
        "overhead_note": "Encrypted pipeline adds 1 extra network hop per token (activations return to requester for final decoding). Latency increases ~2x RTT to the last remote segment. Only useful for 3+ shard models.",
    }))
}

/// PUT /api/admin/models/:model_id/encrypted-pipeline — Toggle encrypted pipeline for a model.
#[derive(Debug, Deserialize)]
pub struct EncryptedPipelineUpdate {
    pub enabled: bool,
}

pub async fn set_model_encrypted_pipeline(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
    Json(body): Json<EncryptedPipelineUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());

    // Validate: check if the local node has the required shards
    if body.enabled {
        let manifest = state.shared_state.model_registry.get_manifest(&mid);
        if let Some(m) = &manifest {
            let local_node_id = state.shared_state.identity.node_id();
            let has_first = state
                .shared_state
                .model_registry
                .shard_holders(&crate::types::ShardId {
                    model_id: mid.clone(),
                    index: 0,
                })
                .contains(local_node_id);
            let last_idx = m.shard_count.saturating_sub(1);
            let has_last = state
                .shared_state
                .model_registry
                .shard_holders(&crate::types::ShardId {
                    model_id: mid.clone(),
                    index: last_idx,
                })
                .contains(local_node_id);

            if !has_first || !has_last {
                let mut missing = Vec::new();
                if !has_first {
                    missing.push("first shard (shard 0, embedding table)".to_string());
                }
                if !has_last {
                    missing.push(format!("last shard (shard {last_idx}, output head)"));
                }
                return Err(ApiError(crate::error::SwarmError::Config(format!(
                    "Cannot enable encrypted pipeline: this node is missing {}. \
                     Download the missing shard(s) first.",
                    missing.join(" and ")
                ))));
            }

            if m.shard_count <= 2 {
                tracing::warn!(
                    model = %model_id,
                    shard_count = m.shard_count,
                    "Encrypted pipeline on a {}-shard model means fully local inference \
                     (no distributed offloading). Consider disabling if you want to share work.",
                    m.shard_count,
                );
            }
        }
    }

    // Update in-memory
    state
        .shared_state
        .encrypted_pipeline_models
        .insert(mid.clone(), body.enabled);

    // Persist to database
    let _ = state
        .shared_state
        .db
        .put_json("encrypted_pipeline_models", &model_id, &body.enabled);

    tracing::info!(
        model = %model_id,
        enabled = body.enabled,
        "Per-model encrypted pipeline updated"
    );

    Ok(Json(serde_json::json!({
        "status": "ok",
        "model_id": model_id,
        "encrypted_pipeline": body.enabled,
        "note": if body.enabled {
            "Encrypted pipeline active. Both embedding (first shard) and sampling (last shard) \
             run locally. Remote nodes only see intermediate activations. \
             Adds ~1 extra RTT per token for the return hop."
        } else {
            "Encrypted pipeline disabled. Normal pipeline scheduling applies."
        },
    })))
}

// ---- GGUF Metadata Browser API ----

/// GET /api/admin/models/:id/metadata — Return parsed GGUF metadata for a model.
///
/// Reads the gguf_header.bin file from the model directory and returns structured
/// metadata including architecture, context length, quantization info, vocab size,
/// layer count, and other hyperparameters.
pub async fn model_metadata(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let safe_id = crate::model::shard::sanitize_path_component(&model_id);
    let model_dir = state.config.node.data_dir.join("models").join(&safe_id);
    let header_path = model_dir.join("gguf_header.bin");

    if !header_path.exists() {
        return Err(ApiError(crate::error::SwarmError::Config(format!(
            "No GGUF header found for model '{}'",
            model_id
        ))));
    }

    let header_bytes =
        std::fs::read(&header_path).map_err(|e| ApiError(crate::error::SwarmError::Io(e)))?;
    let mut cursor = std::io::Cursor::new(&header_bytes);
    let ct = candle_core::quantized::gguf_file::Content::read(&mut cursor).map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to parse GGUF header: {e}"
        )))
    })?;

    let get_str = |key: &str| -> Option<String> {
        ct.metadata
            .get(key)
            .and_then(|v| v.to_string().ok().cloned())
    };
    let get_u32 = |key: &str| -> Option<u32> { ct.metadata.get(key).and_then(|v| v.to_u32().ok()) };
    let get_f32 = |key: &str| -> Option<f32> { ct.metadata.get(key).and_then(|v| v.to_f32().ok()) };

    let arch = get_str("general.architecture").unwrap_or_default();
    let arch_get_u32 = |suffix: &str| -> Option<u32> { get_u32(&format!("{arch}.{suffix}")) };
    let arch_get_f32 = |suffix: &str| -> Option<f32> { get_f32(&format!("{arch}.{suffix}")) };

    let vocab_size = ct.metadata.get("tokenizer.ggml.tokens").and_then(|v| {
        if let candle_core::quantized::gguf_file::Value::Array(arr) = v {
            Some(arr.len() as u32)
        } else {
            None
        }
    });

    let tensor_count = ct.tensor_infos.len();

    let file_type = get_u32("general.file_type");
    let quant_str = file_type.map(|ft| match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q8_1",
        10 => "Q4_K_S",
        11 => "Q4_K_M",
        12 => "Q5_K_S",
        13 => "Q5_K_M",
        14 => "Q6_K",
        15 => "Q2_K",
        16 => "Q3_K_S",
        17 => "Q3_K_M",
        18 => "Q3_K_L",
        _ => "Unknown",
    });

    // Collect all metadata keys for the raw section (exclude large tokenizer arrays)
    let mut raw_metadata: Vec<serde_json::Value> = ct
        .metadata
        .iter()
        .filter(|(k, _)| {
            !k.starts_with("tokenizer.ggml.tokens")
                && !k.starts_with("tokenizer.ggml.merges")
                && !k.starts_with("tokenizer.ggml.scores")
                && !k.starts_with("tokenizer.ggml.token_type")
        })
        .map(|(k, v)| {
            let val_str = match v {
                candle_core::quantized::gguf_file::Value::U8(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::I8(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::U16(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::I16(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::U32(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::I32(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::U64(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::I64(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::F32(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::F64(n) => format!("{n}"),
                candle_core::quantized::gguf_file::Value::Bool(b) => format!("{b}"),
                candle_core::quantized::gguf_file::Value::String(s) => {
                    if s.len() > 200 {
                        format!("{}...", &s[..200])
                    } else {
                        s.clone()
                    }
                }
                candle_core::quantized::gguf_file::Value::Array(arr) => {
                    format!("[array of {} items]", arr.len())
                }
            };
            serde_json::json!({ "key": k, "value": val_str })
        })
        .collect();
    raw_metadata.sort_by(|a, b| {
        a["key"]
            .as_str()
            .unwrap_or("")
            .cmp(b["key"].as_str().unwrap_or(""))
    });

    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "general": {
            "name": get_str("general.name"),
            "architecture": &arch,
            "architecture_supported": crate::inference::split::ModelArch::from_gguf_arch(&arch).is_supported(),
            "file_type": file_type,
            "quantization": quant_str,
        },
        "model": {
            "context_length": arch_get_u32("context_length"),
            "block_count": arch_get_u32("block_count"),
            "embedding_length": arch_get_u32("embedding_length"),
            "head_count": arch_get_u32("attention.head_count"),
            "head_count_kv": arch_get_u32("attention.head_count_kv"),
            "rope_dimension_count": arch_get_u32("rope.dimension_count"),
            "rope_freq_base": arch_get_f32("rope.freq_base"),
            "layer_norm_rms_epsilon": arch_get_f32("attention.layer_norm_rms_epsilon"),
            "vocab_size": vocab_size,
        },
        "tokenizer": {
            "model": get_str("tokenizer.ggml.model"),
            "pre": get_str("tokenizer.ggml.pre"),
            "eos_token_id": get_u32("tokenizer.ggml.eos_token_id"),
            "bos_token_id": get_u32("tokenizer.ggml.bos_token_id"),
            "padding_token_id": get_u32("tokenizer.ggml.padding_token_id"),
        },
        "tensors": {
            "count": tensor_count,
            "data_offset": ct.tensor_data_offset,
        },
        "raw": raw_metadata,
    })))
}

// ---- Download Queue API ----

/// GET /api/admin/downloads — Return all active and recent downloads with full detail.
pub async fn download_queue(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut downloads: Vec<serde_json::Value> = Vec::new();

    for entry in state.shared_state.acquisition_progress.iter() {
        let status = entry.value();
        let model_id = &status.model_id;

        let source = if state.shared_state.hf_sources.contains_key(model_id) {
            "huggingface"
        } else {
            "network"
        };

        let eta_secs =
            if status.speed_bytes_per_sec > 0 && status.total_bytes > status.downloaded_bytes {
                Some((status.total_bytes - status.downloaded_bytes) / status.speed_bytes_per_sec)
            } else {
                None
            };

        let shard_details: Vec<serde_json::Value> = status
            .shard_progress
            .iter()
            .map(|(idx, sp)| {
                let pct = if sp.total_bytes > 0 {
                    ((sp.downloaded_bytes as f64 / sp.total_bytes as f64) * 100.0) as u32
                } else {
                    0
                };
                serde_json::json!({
                    "index": idx,
                    "state": serde_json::to_value(&sp.state).unwrap_or_default(),
                    "progress_pct": pct,
                    "downloaded_bytes": sp.downloaded_bytes,
                    "total_bytes": sp.total_bytes,
                })
            })
            .collect();

        let overall_pct = if status.total_bytes > 0 {
            ((status.downloaded_bytes as f64 / status.total_bytes as f64) * 100.0) as u32
        } else {
            0
        };

        let model_name = state
            .shared_state
            .model_registry
            .get_manifest(model_id)
            .map(|m| m.name.clone())
            .unwrap_or_else(|| model_id.0.clone());

        let cancellable = matches!(
            status.state,
            crate::model::acquisition::AcquisitionState::Downloading
                | crate::model::acquisition::AcquisitionState::AwaitingManifest
        );

        downloads.push(serde_json::json!({
            "model_id": model_id.0,
            "model_name": model_name,
            "state": serde_json::to_value(&status.state).unwrap_or_default(),
            "source": source,
            "total_shards": status.total_shards,
            "downloaded_shards": status.downloaded_shards,
            "verified_shards": status.verified_shards,
            "failed_shards": status.failed_shards,
            "total_bytes": status.total_bytes,
            "downloaded_bytes": status.downloaded_bytes,
            "overall_pct": overall_pct,
            "speed_bytes_per_sec": status.speed_bytes_per_sec,
            "eta_secs": eta_secs,
            "started_at": status.started_at,
            "shard_details": shard_details,
            "cancellable": cancellable,
            "log": status.log.iter().rev().take(10).collect::<Vec<_>>(),
        }));
    }

    // Sort: downloading first, then awaiting, then failed, then complete
    downloads.sort_by(|a, b| {
        let state_order = |v: &serde_json::Value| -> u8 {
            let s = v["state"].as_str().unwrap_or("");
            match s {
                "downloading" => 0,
                "awaiting_manifest" => 1,
                _ if s.contains("failed") || v["state"].is_object() => 3,
                "complete" => 4,
                _ => 2,
            }
        };
        state_order(a).cmp(&state_order(b))
    });

    Json(serde_json::json!({
        "downloads": downloads,
        "total": downloads.len(),
    }))
}
// ---- Prune History API ----

/// GET /api/admin/prune-history — Recent prune events.
pub async fn prune_history(State(state): State<AppState>) -> Json<serde_json::Value> {
    let history = state.shared_state.prune_history.read().await;
    let events: Vec<serde_json::Value> = history
        .iter()
        .rev()
        .map(|e| {
            serde_json::json!({
                "model_id": e.model_id.0,
                "model_name": e.model_name,
                "shard_index": e.shard_index,
                "reason": e.reason,
                "freed_bytes": e.freed_bytes,
                "remaining_local_shards": e.remaining_local_shards,
                "holder_count_before": e.holder_count_before,
                "holder_count_after": e.holder_count_after,
                "timestamp": e.timestamp.to_rfc3339(),
            })
        })
        .collect();

    Json(serde_json::json!({
        "events": events,
        "total": events.len(),
    }))
}

// ---- Shard Lock API ----

#[derive(Debug, Deserialize)]
pub struct ShardLockUpdate {
    pub locked: bool,
}

/// PUT /api/admin/models/:model_id/shards/:index/lock — Lock or unlock a shard.
pub async fn lock_shard(
    State(state): State<AppState>,
    Path((model_id, index)): Path<(String, u32)>,
    Json(body): Json<ShardLockUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let shard_id = crate::types::ShardId {
        model_id: crate::types::ModelId(model_id.clone()),
        index,
    };

    if body.locked {
        state
            .shared_state
            .locked_shards
            .insert(shard_id.clone(), true);
        // Persist to database
        if let Ok(key_str) = serde_json::to_string(&shard_id) {
            let _ = state
                .shared_state
                .db
                .insert_raw("locked_shards", &key_str, b"1");
        }
    } else {
        state.shared_state.locked_shards.remove(&shard_id);
        if let Ok(key_str) = serde_json::to_string(&shard_id) {
            let _ = state.shared_state.db.remove("locked_shards", &key_str);
        }
    }

    tracing::info!(
        model = %model_id,
        shard = index,
        locked = body.locked,
        "DIAG: shard lock state updated"
    );

    Ok(Json(serde_json::json!({
        "status": "ok",
        "model_id": model_id,
        "shard_index": index,
        "locked": body.locked,
    })))
}

// ── LoRA Adapter Management ──

#[derive(Deserialize)]
pub struct RegisterAdapterRequest {
    pub id: Option<String>,
    pub name: String,
    pub base_model: String,
    pub rank: usize,
    pub alpha: f32,
    /// Path to the safetensors file (relative to data_dir/adapters or absolute).
    pub path: String,
}

/// POST /api/admin/adapters — Register a LoRA adapter.
pub async fn register_adapter(
    State(state): State<AppState>,
    Json(body): Json<RegisterAdapterRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let adapter_id = body.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let path = std::path::PathBuf::from(&body.path);
    let resolved = if path.is_absolute() {
        path
    } else {
        state
            .shared_state
            .adapter_registry
            .adapter_dir()
            .join(&path)
    };

    // Reject path traversal attempts (e.g. "../../../etc/passwd")
    for component in resolved.components() {
        if matches!(component, std::path::Component::ParentDir) {
            return Err(ApiError(crate::error::SwarmError::Internal(
                "Path traversal not allowed in adapter path".to_string(),
            )));
        }
    }

    if !resolved.exists() {
        return Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Adapter file not found: {}",
            resolved.display()
        ))));
    }

    let device = candle_core::Device::Cpu;
    let metadata = state.shared_state.adapter_registry.register(
        &adapter_id,
        &body.name,
        &body.base_model,
        body.rank,
        body.alpha,
        &resolved,
        &device,
    )?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "adapter": metadata,
    })))
}

/// GET /api/admin/adapters — List all registered adapters.
pub async fn list_adapters(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let adapters = state.shared_state.adapter_registry.list();
    Ok(Json(serde_json::json!({
        "adapters": adapters,
    })))
}

/// DELETE /api/admin/adapters/:id — Remove a registered adapter.
pub async fn delete_adapter(
    State(state): State<AppState>,
    Path(adapter_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if state.shared_state.adapter_registry.remove(&adapter_id) {
        Ok(Json(serde_json::json!({
            "status": "ok",
            "message": format!("Adapter '{adapter_id}' removed"),
        })))
    } else {
        Err(ApiError(crate::error::SwarmError::Internal(format!(
            "Adapter '{adapter_id}' not found"
        ))))
    }
}

// ── Cloud Provider Management ──
