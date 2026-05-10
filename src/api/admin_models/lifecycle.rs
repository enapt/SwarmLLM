use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;

use super::validate_model_id;

pub async fn delete_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Verify the model exists
    if shared.model_registry.get_manifest(&mid).is_none() {
        return Err(ApiError(crate::error::SwarmError::ModelNotAvailable(mid)));
    }

    // R110 follow-up: refuse to delete a model whose shards are mid-pipeline.
    // The auto-manage prune path already checks `active_pipeline_shards`;
    // applying the same gate here means a "Delete model" click during an
    // in-flight inference returns 503 instead of yanking shards out from
    // under the running token loop.
    let in_use = shared.active_pipelines.iter().any(|entry| {
        entry
            .value()
            .segments
            .iter()
            .any(|seg| seg.shard_id.model_id == mid)
    });
    if in_use {
        return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
            format!(
                "Model {} is currently serving an active inference; retry shortly",
                model_id
            ),
        )));
    }

    let node_id = shared.identity.node_id().clone();

    // Remove shard files from disk (in spawn_blocking to avoid blocking Tokio)
    let model_dir = state.model_dir(&model_id);
    let model_dir_clone = model_dir.clone();
    let files_removed = tokio::task::spawn_blocking(move || {
        let mut count = 0u32;
        if model_dir_clone.exists() {
            if let Ok(entries) = std::fs::read_dir(&model_dir_clone) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_file() {
                        if let Err(e) = std::fs::remove_file(&path) {
                            tracing::warn!(path = %path.display(), error = %e, "Failed to remove shard file");
                        } else {
                            count += 1;
                        }
                    }
                }
            }
            let _ = std::fs::remove_dir(&model_dir_clone);
        }
        count
    })
    .await
    .map_err(|e| {
        // Surface a panic in the file-removal task instead of silently
        // returning files_removed=0, which would tell the caller the delete
        // succeeded when blocking I/O actually crashed.
        ApiError(crate::error::SwarmError::Internal(format!(
            "Model file removal task panicked: {e}"
        )))
    })?;

    // Remove manifest from DB. Log on failure — silently dropping the error
    // would leave the row on disk while the in-memory registry is updated
    // below, causing the model to reappear after restart.
    if let Err(e) = shared.db.remove("model_meta", &model_id) {
        tracing::warn!(model = %model_id, error = %e, "Failed to remove model_meta from DB");
    }
    if let Err(e) = shared.db.remove("hf_sources", &model_id) {
        tracing::warn!(model = %model_id, error = %e, "Failed to remove hf_sources from DB");
    }

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

    // Remove from acquisition_progress and request counts
    shared.models.acquisition_progress.remove(&mid);
    shared.models.model_request_counts.remove(&mid);

    // Remove from gguf_meta
    shared.gguf_meta.remove(&mid);

    // Remove from hf_sources
    shared.models.hf_sources.remove(&mid);

    // Free vision encoder (mmproj) and local embedder caches
    shared.vision_modules.remove(&mid);
    shared.local_embedders.remove(&mid);

    // Evict cached segments and kill worker subprocess
    shared.evict_and_unload(&mid).await;

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

    // Notify dashboard
    shared.signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);

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
    validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Get model name and estimated size before removing
    let model_display_name = shared.model_registry.display_name(&mid);
    let estimated_mb = shared
        .model_registry
        .get_manifest(&mid)
        .map(|m| crate::model::auto_manage::estimate_model_vram_mb(m.total_size_bytes))
        .unwrap_or(0);

    // Remove split models for this model
    let segments_removed = shared
        .split_models
        .iter()
        .filter(|e| e.key().0 == mid)
        .count() as u32;
    shared.evict_and_unload(&mid).await;

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

    // Free vision encoder (mmproj) and local embedder caches
    shared.vision_modules.remove(&mid);
    shared.local_embedders.remove(&mid);

    // Notify dashboard
    shared.signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);

    // Emit activity event
    shared.emit_activity(
        crate::daemon::state::ActivityEvent::new(
            "model",
            "model_unloaded",
            format!(
                "Unloaded {} — ~{}MB {} freed",
                model_display_name,
                estimated_mb,
                shared.memory_type_label()
            ),
        )
        .with_model(model_id.clone())
        .with_model_name(model_display_name.clone())
        .with_detail_num(estimated_mb as i64),
    );

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
pub async fn get_model_auto_manage(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Json<serde_json::Value> {
    let mid = crate::types::ModelId(model_id.clone());
    let default_cap = state
        .shared_state
        .models
        .auto_manage_default_model_cap
        .load(std::sync::atomic::Ordering::Relaxed);

    match state
        .shared_state
        .models
        .model_auto_manage_policies
        .get(&mid)
    {
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
    validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());

    let policy = crate::config::ModelAutoManagePolicy {
        enabled: body.enabled.unwrap_or(true),
        max_shards: body.max_shards.unwrap_or(0),
        prune_enabled: body.prune_enabled.unwrap_or(true),
    };

    // Update in-memory
    state
        .shared_state
        .models
        .model_auto_manage_policies
        .insert(mid.clone(), policy.clone());

    // Persist to database
    if let Err(e) = state
        .shared_state
        .db
        .put_json("model_auto_manage_policies", &model_id, &policy)
    {
        tracing::warn!(error = %e, model = %model_id, "Failed to persist auto-manage policy — may be lost on restart");
    }

    // Wake auto-manage to re-evaluate
    state.shared_state.models.auto_manage_notify.notify_one();

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
    validate_model_id(&model_id)?;
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
                return Err(ApiError(crate::error::SwarmError::Validation(format!(
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
    if let Err(e) =
        state
            .shared_state
            .db
            .put_json("encrypted_pipeline_models", &model_id, &body.enabled)
    {
        tracing::warn!(error = %e, model = %model_id, "Failed to persist encrypted pipeline toggle — may be lost on restart");
    }

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
