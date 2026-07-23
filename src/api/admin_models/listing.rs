use axum::extract::{Path, State};
use axum::Json;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::helpers::*;
use super::validate_model_id;

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

    // Helper: count local and global shard availability for a manifest
    let count_shard_availability =
        |m: &crate::types::ModelManifest, state: &AppState| -> (usize, usize) {
            let mut local_count = 0usize;
            let mut global_count = 0usize;
            for idx in 0..m.shard_count {
                let sid = crate::types::ShardId {
                    model_id: m.id.clone(),
                    index: idx,
                };
                let holders = state.shared_state.model_registry.shard_holders(&sid);
                if holders.contains(&local_node_id) {
                    local_count += 1;
                }
                // A shard only counts toward network-wide "ready" if a holder can
                // actually serve it now — the local node or a connected peer.
                // Matching the scheduler's oracle stops a disconnected peer's
                // stale announce from making a model look ready it can't serve.
                if state.shared_state.any_holder_reachable(&holders) {
                    global_count += 1;
                }
            }
            (local_count, global_count)
        };

    // Helper: build per-shard detail for a manifest, including download state + in_vram
    let build_shard_detail =
        |m: &crate::types::ModelManifest, state: &AppState| -> Vec<serde_json::Value> {
            let acq = state.shared_state.models.acquisition_progress.get(&m.id);
            m.shards
                .iter()
                .map(|s| {
                    let (mut shard_json, is_local, _) = build_shard_json(
                        s,
                        &state.shared_state,
                        &m.id,
                        &local_node_id,
                        acq.as_deref(),
                    );
                    // Add in_vram field (only for model detail, not storage view)
                    let in_vram = if is_local {
                        state.shared_state.is_shard_in_vram(&m.id, s.index)
                    } else {
                        false
                    };
                    if let Some(obj) = shard_json.as_object_mut() {
                        obj.insert("in_vram".to_string(), serde_json::json!(in_vram));
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

    // Helper: build the common model JSON object shared by all 3 listing paths.
    let build_model_json = |id: &str,
                            name: &str,
                            total_size_bytes: u64,
                            shard_count: u32,
                            hosted_shards: usize,
                            global_available: usize,
                            status: &str,
                            mode: &str,
                            source: &str,
                            peers_hosting: usize,
                            shards: Vec<serde_json::Value>|
     -> serde_json::Value {
        let enc_info = encrypted_pipeline_info(id);
        let trust_level = state
            .shared_state
            .models
            .model_trust
            .get(&crate::types::ModelId(id.to_string()))
            .map(|t| t.trust_level.to_string())
            .unwrap_or_else(|| {
                if hosted_shards > 0 {
                    "pinned".to_string()
                } else {
                    "discovered".to_string()
                }
            });
        let hf_source = state
            .shared_state
            .models
            .hf_sources
            .get(&crate::types::ModelId(id.to_string()))
            .map(|entry| {
                serde_json::json!({
                    "repo_id": entry.repo_id,
                    "filename": entry.filename,
                })
            });
        serde_json::json!({
            "id": id,
            "name": name,
            "total_size_bytes": total_size_bytes,
            "shard_count": shard_count,
            "hosted_shards": hosted_shards,
            "global_available": global_available,
            "healthy": global_available == shard_count as usize,
            "status": status,
            "mode": mode,
            "source": source,
            "local": hosted_shards > 0,
            "peers_hosting": peers_hosting,
            "shards": shards,
            "trust_level": trust_level,
            "encrypted_pipeline": enc_info.0,
            "has_first_shard": enc_info.1,
            "has_last_shard": enc_info.2,
            "hf_source": hf_source,
        })
    };

    // Helper: compute disk-level metadata (manifest, header, probed, mmproj) for a model.
    let disk_metadata = |model_id: &str, model_display_name: &str| -> serde_json::Value {
        let model_dir = state.model_dir(model_id);
        let has_manifest = model_dir
            .join(crate::model::shard::MANIFEST_FILENAME)
            .exists();
        let has_header = model_dir
            .join(crate::model::shard::HEADER_FILENAME)
            .exists();
        let mid_check = crate::types::ModelId(model_id.to_string());
        let probed = has_header
            || state
                .shared_state
                .models
                .hf_sources
                .contains_key(&mid_check)
            || state
                .shared_state
                .models
                .hf_probe_cache
                .contains_key(&mid_check);
        let mid_mmproj = crate::types::ModelId(model_display_name.to_string());
        let holders = state
            .shared_state
            .model_registry
            .mmproj_holders(&mid_mmproj);
        let local_has = holders.contains(&local_node_id);
        serde_json::json!({
            "has_manifest": has_manifest,
            "has_header": has_header,
            "probed": probed,
            "mmproj": {
                "available": state.shared_state.any_holder_reachable(&holders),
                "local": local_has,
                "holders": holders.len(),
                "holder_ids": holders
                    .iter()
                    .filter(|h| *h != &local_node_id)
                    .take(32)
                    .map(|h| format!("{}", h))
                    .collect::<Vec<_>>(),
            },
        })
    };

    // 1. Locally loaded model (full model via --model flag)
    // Even though it's locally loaded, get the real manifest to show shard info
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        // Staleness check: verify model directory still exists on disk.
        // If files were deleted while the process is running, skip this entry
        // to avoid confusing the UI with a "loaded" model that can't run.
        let loaded_model_dir = state
            .config
            .node
            .data_dir
            .join("models")
            .join(crate::types::slugify_model_name(&info.name));
        let dir_check = loaded_model_dir.clone();
        let has_shard_files = tokio::task::spawn_blocking(move || {
            dir_check.exists()
                && std::fs::read_dir(&dir_check)
                    .ok()
                    .map(|rd| {
                        rd.flatten().any(|e| {
                            let name = e.file_name();
                            let n = name.to_string_lossy();
                            n.starts_with("shard_") && n.ends_with(".bin")
                        })
                    })
                    .unwrap_or(false)
        })
        .await
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
            let slug = crate::types::slugify_model_name(&info.name);
            seen_ids.insert(slug.clone());

            let manifest = state
                .shared_state
                .model_registry
                .resolve_manifest_by_name(&info.name);

            // Mark the manifest's actual registry ID as seen to prevent duplicates
            // in section 2 (which iterates by manifest.id)
            if let Some(ref m) = manifest {
                seen_ids.insert(m.id.0.clone());
            }
            let (shard_count, hosted_local, global_available, shard_detail) = match manifest {
                Some(ref m) => {
                    let detail = build_shard_detail(m, &state);
                    let (local_count, global) = count_shard_availability(m, &state);
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

            let mode = if hosted_local == shard_count as usize {
                "full"
            } else {
                "distributed"
            };
            let mut entry = build_model_json(
                &model_id,
                &info.name,
                info.size_bytes,
                shard_count,
                hosted_local,
                global_available,
                status,
                mode,
                "local",
                peer_count,
                shard_detail,
            );
            // Merge disk metadata (manifest, header, probed, mmproj)
            let dm = disk_metadata(&model_id, &info.name);
            if let (Some(entry_obj), Some(dm_obj)) = (entry.as_object_mut(), dm.as_object()) {
                for (k, v) in dm_obj {
                    entry_obj.insert(k.clone(), v.clone());
                }
            }
            models.push(entry);
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

        let (hosted_count, global_available) = count_shard_availability(m, &state);

        let peer_count = model_peers.get(&m.id.0).map_or(0, |s| s.len());
        let shard_detail = build_shard_detail(m, &state);

        let (source, mode) = if hosted_count == m.shard_count as usize {
            ("local", "full")
        } else if hosted_count > 0 {
            ("hybrid", "distributed")
        } else {
            ("network", "distributed")
        };

        // Check if the model is loaded and ready for inference.
        // A model is "ready" when all layers are covered across the network —
        // no single node needs all shards. Nodes participate with whatever
        // shards they have; the pipeline scheduler assembles the full pipeline.
        let is_loaded = if hosted_count > 0 {
            state.shared_state.has_split_model(&m.id)
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
            .models
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
                .models
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

        let mut entry = build_model_json(
            &m.id.0,
            &m.name,
            m.total_size_bytes,
            m.shard_count,
            hosted_count,
            global_available,
            status,
            mode,
            source,
            peer_count,
            shard_detail,
        );
        // Merge disk metadata + registry-specific fields
        let dm = disk_metadata(&m.id.0, &m.name);
        if let Some(obj) = entry.as_object_mut() {
            if let Some(dm_obj) = dm.as_object() {
                for (k, v) in dm_obj {
                    obj.insert(k.clone(), v.clone());
                }
            }
            obj.insert(
                "estimated_vram_mb".to_string(),
                serde_json::json!(estimated_vram),
            );
            obj.insert("acquisition".to_string(), serde_json::json!(acq_state));
            obj.insert(
                "acquisition_progress".to_string(),
                serde_json::json!(acq_progress),
            );
        }
        models.push(entry);
    }

    // 3. Models discovered from peer announcements (not in our registry or loaded)
    for (model_name, peers) in &model_peers {
        if seen_ids.contains(model_name) {
            continue;
        }
        seen_ids.insert(model_name.clone());
        models.push(build_model_json(
            model_name,
            model_name,
            0,
            0,
            0,
            0,
            "available",
            "full",
            "network",
            peers.len(),
            vec![],
        ));
    }

    // Final display net: never surface a backup-copy model name in the UI,
    // whatever path it arrived via. A peer on an older build still gossips
    // `<model>.FULLBACKUP`; the registry, DB-load and gossip guards should
    // already keep it out, but this guarantees the models list stays clean.
    models.retain(|m| {
        m.get("id")
            .and_then(|v| v.as_str())
            .map(|id| !crate::model::manifest::is_backup_artifact_id(id))
            .unwrap_or(true)
    });

    Json(models)
}

/// POST /api/admin/models/:id/add — Express interest in a model (trigger download).
pub async fn add_model_interest(
    State(state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_model_id(&model_id)?;
    tracing::info!(model_id = %model_id, "Model acquisition requested");

    let mid = crate::types::ModelId(model_id.clone());

    // User-initiated acquisition pins the model for auto-manage: without this,
    // auto-manage would skip a "discovered" model even if P2P exhausts and our
    // HF-fallback path (src/network/manager.rs::retry_shard_or_fallback) notifies
    // it. Pinning satisfies the trust gate in auto_manage/scoring.rs.
    state
        .shared_state
        .models
        .model_trust
        .entry(mid.clone())
        .and_modify(|t| t.pinned_by_user = true)
        .or_insert_with(|| {
            let mut t = crate::types::ModelTrustInfo::new_discovered();
            t.pinned_by_user = true;
            t
        });

    // Send acquisition command if the channel is wired up
    if let Some(ref tx) = state.acquisition_tx {
        tx.send(crate::model::acquisition::AcquisitionCommand::Acquire { model_id: mid })
            .await
            .map_err(|e| {
                ApiError(crate::error::SwarmError::ServiceUnavailable(format!(
                    "Acquisition manager unavailable: {e}"
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
    validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());

    // Fast path: read from shared state (no channel round-trip)
    if let Some(status) = state.shared_state.models.acquisition_progress.get(&mid) {
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
    let models_dir = state.shared_state.shard_store().models_dir();

    let mut model_storage: Vec<serde_json::Value> = Vec::new();
    let mut total_local_bytes: u64 = 0;

    for manifest in state.shared_state.model_registry.models() {
        let mut local_shards = 0u32;
        let mut local_bytes = 0u64;
        let mut shard_details: Vec<serde_json::Value> = Vec::new();

        let mut any_downloading = false;
        let acq_progress = state
            .shared_state
            .models
            .acquisition_progress
            .get(&manifest.id);

        for shard in &manifest.shards {
            let (shard_json, is_local, downloading) = build_shard_json(
                shard,
                &state.shared_state,
                &manifest.id,
                &local_node_id,
                acq_progress.as_deref(),
            );
            if is_local {
                local_shards += 1;
                local_bytes += shard.size_bytes;
            }
            if downloading {
                any_downloading = true;
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
    let models_dir_clone = models_dir.clone();
    let disk_usage_bytes =
        tokio::task::spawn_blocking(move || dir_size(&models_dir_clone).unwrap_or(0))
            .await
            .unwrap_or(0);

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
/// GET /api/admin/models/{model_id}/metadata — Return parsed GGUF metadata
/// for a locally downloaded model. Reads the header file, no shard scan.
pub async fn model_metadata(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_model_id(&model_id)?;
    let model_dir = state.model_dir(&model_id);
    let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);

    if !header_path.exists() {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "No GGUF header found for model '{}'",
            model_id
        ))));
    }

    let hp = header_path.clone();
    let header_bytes = tokio::task::spawn_blocking(move || std::fs::read(&hp))
        .await
        .map_err(|e| {
            ApiError(crate::error::SwarmError::Internal(format!(
                "spawn_blocking join: {e}"
            )))
        })?
        .map_err(|e| ApiError(crate::error::SwarmError::Io(e)))?;
    let mut cursor = std::io::Cursor::new(&header_bytes);
    let ct = candle_core::quantized::gguf_file::Content::read(&mut cursor).map_err(|e| {
        ApiError(crate::error::SwarmError::Validation(format!(
            "Failed to parse GGUF header (file may be corrupt): {e}"
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

    for entry in state.shared_state.models.acquisition_progress.iter() {
        let status = entry.value();
        let mut obj = serialize_acquisition_to_json(status, &state.shared_state);
        // REST-only fields
        if let Some(o) = obj.as_object_mut() {
            o.insert(
                "failed_shards".into(),
                serde_json::json!(status.failed_shards),
            );
            o.insert("started_at".into(), serde_json::json!(status.started_at));
        }
        downloads.push(obj);
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

pub async fn prune_history(State(state): State<AppState>) -> Json<serde_json::Value> {
    let history = state.shared_state.models.prune_history.read().await;
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

// ── Pipeline Plan (read-only preview for UI) ──

/// GET /api/admin/models/:id/pipeline-plan — Return the pipeline the scheduler
/// would currently assemble for this model. Read-only: no execution, no side
/// effects. Used by the frontend to render the inference path on the shard
/// matrix and network map. Fails with 404 if the model isn't registered or if
/// the current peer set can't cover all layers (same conditions that would
/// fail a real inference request).
pub async fn pipeline_plan(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());
    let local_node_id = state.shared_state.identity.node_id().clone();
    let scheduler = crate::inference::scheduler::PipelineScheduler::new(state.shared_state.clone());
    let assignment = scheduler
        .assemble_pipeline_for(&mid, &local_node_id, uuid::Uuid::new_v4())
        .map_err(ApiError)?;

    // Map segment layer range → full list of shard indices so the UI can
    // highlight every cell the segment covers, not just the anchor shard.
    let manifest = state.shared_state.model_registry.get_manifest(&mid);
    let shard_indices_for = |lr: (u32, u32)| -> Vec<u32> {
        let (s, e) = lr;
        manifest
            .as_ref()
            .map(|m| {
                let mut v: Vec<u32> = m
                    .shards
                    .iter()
                    .filter(|sh| {
                        let (ss, se) = sh.layer_range;
                        se > s && ss < e
                    })
                    .map(|sh| sh.index)
                    .collect();
                v.sort_unstable();
                v
            })
            .unwrap_or_default()
    };

    let segments: Vec<serde_json::Value> = assignment
        .segments
        .iter()
        .map(|seg| {
            let peer = state.shared_state.peer_registry.get(&seg.node_id);
            let peer_latency = peer.as_ref().and_then(|p| p.latency_ms);
            let region = peer
                .as_ref()
                .and_then(|p| p.capability.as_ref())
                .and_then(|c| c.region.clone());
            let is_local = seg.node_id == local_node_id;
            let nickname = state
                .shared_state
                .nickname_registry
                .get(&seg.node_id)
                .map(|r| r.nickname.clone());
            let shard_indices = shard_indices_for(seg.layer_range);
            serde_json::json!({
                "node_id": format!("{}", seg.node_id),
                "nickname": nickname,
                "region": region,
                "shard_index": seg.shard_id.index,
                "shard_indices": shard_indices,
                "layer_range": [seg.layer_range.0, seg.layer_range.1],
                "latency_ms": if is_local { Some(0) } else { peer_latency },
                "is_local": is_local,
            })
        })
        .collect();

    let standbys: Vec<serde_json::Value> = assignment
        .standbys
        .iter()
        .map(|seg| {
            serde_json::json!({
                "node_id": format!("{}", seg.node_id),
                "shard_index": seg.shard_id.index,
                "layer_range": [seg.layer_range.0, seg.layer_range.1],
            })
        })
        .collect();

    let local_region = state.shared_state.config.identity.region.clone();
    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "local_node_id": format!("{}", local_node_id),
        "local_region": local_region,
        "segments": segments,
        "standbys": standbys,
    })))
}

// ── Cloud Provider Management ──
