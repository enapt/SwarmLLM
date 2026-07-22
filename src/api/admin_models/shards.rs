use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::helpers::*;
use super::validate_shard_params;

pub async fn unload_shard(
    State(state): State<AppState>,
    Path((model_id, shard_index)): Path<(String, u32)>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_shard_params(&model_id, shard_index)?;
    let shared = &state.shared_state;
    let (mid, _shard_id, local_node_id) = resolve_local_shard(shared, &model_id, shard_index)?;

    // Get current shard window (or all local shard indices if no window set)
    let current_window = shared.model_process_pool.get_shard_window(&mid);
    let window = current_window.unwrap_or_else(|| {
        shared
            .model_registry
            .local_shard_indices(&mid, &local_node_id)
    });
    let new_window: Vec<u32> = window.into_iter().filter(|&i| i != shard_index).collect();

    apply_shard_window_change(shared, &model_id, &mid, &new_window).await;

    {
        let display = shared.model_registry.display_name(&mid);
        let remaining = if new_window.is_empty() {
            "model fully unloaded".to_string()
        } else {
            let nums: Vec<_> = new_window.iter().map(|i| (i + 1).to_string()).collect();
            format!("shards {} remain", nums.join(", "))
        };
        shared.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "model",
                "shard_unloaded_memory",
                format!(
                    "Unloaded shard {} of {} from {} — {}",
                    shard_index + 1,
                    display,
                    shared.memory_type_label(),
                    remaining
                ),
            )
            .with_model(model_id.clone())
            .with_detail_num(shard_index as i64),
        );
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
    validate_shard_params(&model_id, shard_index)?;
    let shared = &state.shared_state;
    let (mid, _shard_id, local_node_id) = resolve_local_shard(shared, &model_id, shard_index)?;

    // Expand the shard window to include this shard.
    // If no window exists, start from all local shards (same as unload_shard).
    let current_window = shared.model_process_pool.get_shard_window(&mid);
    let mut new_window = current_window.unwrap_or_else(|| {
        shared
            .model_registry
            .local_shard_indices(&mid, &local_node_id)
    });
    if !new_window.contains(&shard_index) {
        new_window.push(shard_index);
        new_window.sort();
    }

    apply_shard_window_change(shared, &model_id, &mid, &new_window).await;

    {
        let display = shared.model_registry.display_name(&mid);
        let window_label = if new_window.len() == 1 {
            format!("shard {}", new_window[0] + 1)
        } else {
            let nums: Vec<_> = new_window.iter().map(|i| (i + 1).to_string()).collect();
            format!("shards {}", nums.join(", "))
        };
        shared.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "model",
                "shard_loaded_memory",
                format!(
                    "Loaded {} — {} now in {}",
                    display,
                    window_label,
                    shared.memory_type_label()
                ),
            )
            .with_model(model_id.clone())
            .with_detail_num(shard_index as i64),
        );
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
    validate_shard_params(&model_id, shard_index)?;
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
        return Err(ApiError(crate::error::SwarmError::ShardNotFound(shard_id)));
    }

    // R110 follow-up: refuse mid-pipeline deletion. The auto-manage prune
    // path already protects in-use shards via active_pipeline_shards;
    // applying the same gate here means a "Delete shard" click during an
    // active token loop returns 503 instead of yanking the file out from
    // under the in-flight inference (which would surface as a confusing
    // mid-stream error to the client).
    let in_use = shared.active_pipelines.iter().any(|entry| {
        entry
            .value()
            .segments
            .iter()
            .any(|seg| seg.shard_id.model_id == mid && seg.shard_id.index == shard_index)
    });
    if in_use {
        return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
            format!(
                "Shard {}/{} is currently serving an active inference; retry shortly",
                model_id, shard_index
            ),
        )));
    }

    // Delete shard file from disk
    let shard_store = state.shared_state.shard_store();
    let shard_path = shard_store.shard_path(&mid, shard_index);

    if shard_path.exists() {
        let sp = shard_path.clone();
        tokio::task::spawn_blocking(move || std::fs::remove_file(&sp))
            .await
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "spawn_blocking join: {e}"
                )))
            })?
            .map_err(|e| ApiError(crate::error::SwarmError::Io(e)))?;
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

    // Evict cached segments and kill worker subprocess
    shared.evict_and_unload(&mid).await;

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
            // Complete for this model: whatever is absent was just deleted.
            complete_for_models: vec![mid.clone()],
        });
        let _ = ntx
            .send(crate::types::NetworkCommand::Broadcast(announce))
            .await;
    }

    // Clear stale model_loaded history entries so the reload emits a fresh event
    // (the layer range may have changed after shard deletion)
    shared.events.clear_model_load_history(&model_id);

    // Reload remaining shards so the model stays available for inference
    // (check_and_load_model will re-compute layer ranges from remaining shards)
    crate::model::auto_manage::spawn_check_and_load(shared.clone(), mid.clone());

    // Notify dashboard so shard grid and model state update immediately
    shared.signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);

    {
        let display = shared.model_registry.display_name(&mid);
        // Check what shards remain loaded
        let remaining_local: Vec<u32> = shared
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
                            .contains(&shared.identity.node_id().clone())
                    })
                    .filter(|s| s.index != crate::types::MMPROJ_SHARD_INDEX)
                    .map(|s| s.index + 1)
                    .collect()
            })
            .unwrap_or_default();
        let status = if remaining_local.is_empty() {
            "model fully removed".to_string()
        } else {
            let nums: Vec<_> = remaining_local.iter().map(|i| i.to_string()).collect();
            format!("shards {} remain on disk", nums.join(", "))
        };
        shared.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "model",
                "shard_deleted",
                format!(
                    "Deleted shard {} of {} — inference unloaded, {}",
                    shard_index + 1,
                    display,
                    status
                ),
            )
            .with_model(model_id.clone())
            .with_detail_num(shard_index as i64)
            .with_toast("info", 4000),
        );
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
    validate_shard_params(&model_id, shard_index)?;
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

    // Pre-flight disk space check (returns 507 Insufficient Storage with hint)
    if shard_size > 0 {
        let dest_dir = shared.model_dir(&mid.0);
        crate::model::check_disk_space(&dest_dir, shard_size)?;
    }

    // Try P2P: find peers who hold this shard
    let holders: Vec<_> = shared
        .model_registry
        .shard_holders(&shard_id)
        .into_iter()
        .filter(|n| n != &local_node_id)
        .collect();

    if !holders.is_empty() {
        // Pick the best peer: LAN first, then lowest latency, then highest trust
        let target = shared.select_best_peer(&holders);

        let peer_id_bytes = shared
            .peer_registry
            .get(&target)
            .and_then(|p| p.peer_id_bytes.clone());

        if let Some(bytes) = peer_id_bytes {
            // Create acquisition_progress for the download bar
            let mut shard_progress = std::collections::HashMap::new();
            shard_progress.insert(
                shard_index,
                crate::model::acquisition::ShardProgress::new_downloading(shard_index, shard_size),
            );
            let mut dl_status = crate::model::acquisition::AcquisitionStatus::new_downloading(
                mid.clone(),
                1,
                shard_size,
                "peers",
                "user",
                format!("Downloading shard {} from peer", shard_index + 1),
            );
            dl_status.shard_progress = shard_progress;
            shared
                .models
                .acquisition_progress
                .insert(mid.clone(), dl_status);

            let request = crate::types::ShardRequest {
                shard_id,
                chunk_offset: 0,
                chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE,
            };
            if let Some(ref tx) = state.network_tx {
                let _ = tx
                    .send(crate::types::NetworkCommand::SendShardRequest {
                        target_peer_bytes: bytes,
                        request,
                    })
                    .await;
            }

            let display = shared.model_registry.display_name(&mid);
            let peer_label = shared
                .nickname_registry
                .get(&target)
                .map(|r| r.nickname.clone())
                .unwrap_or_else(|| format!("{}", target).chars().take(8).collect());
            shared.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "shard_download_p2p",
                    format!(
                        "Downloading shard {} of {} from peer {}",
                        shard_index + 1,
                        display,
                        peer_label
                    ),
                )
                .with_model(model_id.clone())
                .with_node(format!("{}", target))
                .with_detail_num(shard_index as i64)
                .with_detail_str("p2p".to_string()),
            );

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
    if let Some(hf) = shared.models.hf_sources.get(&mid) {
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
    // validate_shard_params (not validate_model_id) so the mmproj sentinel
    // index (u32::MAX) is rejected — locked_shards iterators downstream
    // assume regular indices and would mishandle the sentinel.
    validate_shard_params(&model_id, index)?;
    let shard_id = crate::types::ShardId {
        model_id: crate::types::ModelId(model_id.clone()),
        index,
    };

    // Persist BEFORE updating the in-memory map — if the DB write fails,
    // we want to surface the error to the operator with the in-memory
    // state still matching disk. A silent discard (`let _ = ...`) here
    // would cause a divergence on restart: the auto-manage pruner could
    // remove a shard the operator believed was pinned. The same pattern
    // was already corrected for pool pins, auto-manage policy, encrypted
    // pipeline, and HF trust pin in earlier sweeps.
    // ShardId = { model_id: String, index: u32 } — both are infallibly
    // serializable, so .expect() avoids a dead error path that would
    // otherwise sit in the SwarmError::Internal bucket.
    let key_str = serde_json::to_string(&shard_id).expect("ShardId is always JSON-serializable");
    if body.locked {
        state
            .shared_state
            .db
            .insert_raw("locked_shards", &key_str, b"1")
            .map_err(crate::error::ApiError)?;
        state
            .shared_state
            .models
            .locked_shards
            .insert(shard_id.clone(), true);
    } else {
        state
            .shared_state
            .db
            .remove("locked_shards", &key_str)
            .map_err(crate::error::ApiError)?;
        state.shared_state.models.locked_shards.remove(&shard_id);
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
