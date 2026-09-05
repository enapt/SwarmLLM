use crate::api::server::JsonBody;
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
    // `active_pipelines` alone missed the common case — see
    // `SharedState::model_is_in_use`. Deleting while the local model was
    // answering returned 200, removed all 8 shard files and killed the worker
    // mid-reply.
    let in_use = shared.model_is_in_use(&mid);
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
    if let Err(e) = shared.db.remove("model_trust", &model_id) {
        tracing::warn!(model = %model_id, error = %e, "Failed to remove model_trust from DB");
    }

    // S5: Collect local shards before removal for DHT stop_providing
    let local_shards: Vec<_> = shared
        .model_registry
        .shards_for_node(&node_id)
        .into_iter()
        .filter(|s| s.model_id == mid)
        .collect();

    // The user removed the whole model: remember EVERY shard of it (not just
    // the ones that were local), so auto-manage does not quietly re-acquire
    // any of it when the manifest arrives again by gossip.
    if let Some(manifest) = shared.model_registry.get_manifest(&mid) {
        for s in &manifest.shards {
            shared.mark_shard_removed_by_user(&crate::types::ShardId {
                model_id: mid.clone(),
                index: s.index,
            });
        }
    }

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
    shared.models.model_trust.remove(&mid);

    // Parallax stability counters are keyed by ShardId, so they need a
    // predicate rather than a point remove. Without this every shard of every
    // model the node ever evaluated leaves a permanent entry — the map is
    // written each auto-manage tick via `entry().or_insert()` and had no
    // removal path anywhere, so it grew monotonically with model churn.
    shared
        .models
        .parallax_stability
        .retain(|shard_id, _| shard_id.model_id != mid);

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
        let announce = crate::types::SwarmMessage::ShardAnnounce(
            // Empty + complete-for-this-model = "I host none of it any more".
            // Before `complete_for_models` existed the receiver looped over the
            // empty vec and did nothing, so this broadcast was a no-op.
            crate::model::manifest::shard_announce(
                &shared.model_registry,
                node_id.clone(),
                vec![],
                vec![mid.clone()],
            ),
        );
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

    // Same existence check `delete_model` makes, for the same reason: without
    // it an unknown id answered 200 with `estimated_freed_mb: 0`, so a typo in
    // the dashboard or a script reported "unloaded, freed 0 MB" — success for
    // work on something that does not exist, and inconsistent with delete,
    // which 404s the identical id.
    //
    // This keeps unload idempotent where that actually means something: a model
    // the node KNOWS but has not loaded still returns 200, because "make sure
    // this is not resident" is genuinely already true.
    if shared.model_registry.get_manifest(&mid).is_none() {
        return Err(ApiError(crate::error::SwarmError::ModelNotAvailable(mid)));
    }

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
    JsonBody(body): JsonBody<ModelAutoManageUpdate>,
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

/// POST /api/admin/models/{id}/enable-privacy — one action to make prompt
/// privacy possible for a model.
///
/// Prompt privacy needs the first and last piece of a model on this machine,
/// which previously meant reading which pieces were missing and downloading them
/// by hand. This fetches exactly those pieces and nothing else.
///
/// It deliberately does NOT set a per-model flag. Privacy turns itself on once
/// both ends are present (`encrypted_pipeline_for`), so there is no window where
/// a flag is on but the shards have not arrived — which would fail every request
/// for the model with "No node available" until the download finished. It does
/// clear an explicit OFF, since that would otherwise keep suppressing it.
pub async fn enable_model_privacy(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());
    let manifest = state
        .shared_state
        .model_registry
        .get_manifest(&mid)
        .ok_or_else(|| ApiError(crate::error::SwarmError::ModelNotAvailable(mid.clone())))?;

    // A previous explicit "off" would keep overriding the automatic behaviour
    // even once the shards land, so asking for privacy clears it.
    let had_explicit_off = state
        .shared_state
        .encrypted_pipeline_models
        .get(&mid)
        .map(|r| !*r.value())
        .unwrap_or(false);
    if had_explicit_off {
        state.shared_state.encrypted_pipeline_models.remove(&mid);
        let _ = state
            .shared_state
            .db
            .remove("encrypted_pipeline_models", &model_id);
    }

    let last_index = manifest.shard_count.saturating_sub(1);
    let me = state.shared_state.identity.node_id();
    // Confirm on disk, not just in the registry.
    //
    // This answer becomes the sentence "Prompt privacy is ON — prompts and
    // answers stay here", which a user reasonably reads as a guarantee about
    // where their text goes. The registry is a snapshot built at startup and
    // updated by events, so after a model folder was deleted by hand — the only
    // way to free one, there being no remove command — it still listed the
    // shards and this claimed privacy was ready for a model that could not
    // answer at all (reported 2026-08-02). The health monitor now reconciles the
    // registry against disk each announce cycle, but that is up to a cycle
    // late; a claim of this kind should be true when it is made, and two stat
    // calls on a hand-run command cost nothing.
    let store = state.shared_state.shard_store();
    let holds = |index: u32| {
        let shard_id = crate::types::ShardId {
            model_id: mid.clone(),
            index,
        };
        state
            .shared_state
            .model_registry
            .shard_holders(&shard_id)
            .contains(me)
            && store.shard_file_present(&shard_id)
    };
    let mut needed: Vec<u32> = Vec::new();
    for index in [0u32, last_index] {
        if !holds(index) && !needed.contains(&index) {
            needed.push(index);
        }
    }

    if needed.is_empty() {
        return Ok(Json(serde_json::json!({
            "model_id": model_id,
            "status": "already_available",
            "encrypted_pipeline": state.shared_state.encrypted_pipeline_for(&mid),
            "cleared_explicit_off": had_explicit_off,
            "downloading": [],
        })));
    }

    // Needs a HuggingFace source to fetch from. Without one the shards can still
    // arrive over the network from peers, so say so rather than failing.
    let source = state
        .shared_state
        .models
        .hf_sources
        .get(&mid)
        .map(|r| r.value().clone());
    let Some(source) = source else {
        return Ok(Json(serde_json::json!({
            "model_id": model_id,
            "status": "no_download_source",
            "message": "No HuggingFace source recorded for this model, so the missing pieces \
                        cannot be fetched directly. They may still arrive from peers; privacy \
                        turns on by itself once both ends are present.",
            "needed_shards": needed,
            "cleared_explicit_off": had_explicit_off,
        })));
    };

    let resp = crate::api::admin_hf::hf_download_shards(
        State(state.clone()),
        JsonBody(crate::api::admin_hf::HfShardDownloadRequest {
            repo_id: source.repo_id.clone(),
            filename: source.filename.clone(),
            shards: needed.clone(),
            // Merge into the existing model directory rather than deriving a new
            // model id from the filename — these are pieces of a model we already
            // have, not a new download.
            model_id: Some(model_id.clone()),
            peer_fair_share: false,
        }),
    )
    .await?;

    state.shared_state.emit_activity(
        crate::daemon::state::ActivityEvent::new(
            "models",
            "privacy_shards_requested",
            format!(
                "Fetching the pieces needed to keep prompts private for {} ({} of {})",
                model_id,
                needed.len(),
                manifest.shard_count
            ),
        )
        .with_model(model_id.clone())
        .with_toast("info", 6000),
    );

    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "status": "downloading",
        "needed_shards": needed,
        "cleared_explicit_off": had_explicit_off,
        "note": "Prompt privacy turns on by itself once both ends are present.",
        "download": resp.0,
    })))
}

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
    // The effective answer the scheduler will act on, so the UI reflects reality
    // rather than only what was explicitly stored — the automatic-on case has no
    // stored flag at all.
    let effective = state.shared_state.encrypted_pipeline_for(&mid);

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
    JsonBody(body): JsonBody<EncryptedPipelineUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    validate_model_id(&model_id)?;
    let mid = crate::types::ModelId(model_id.clone());

    // Set when privacy is switched on for a model whose ends this node does not
    // hold: the preference is recorded, but requests will fail closed until the
    // shards are here. See the branch that sets it.
    let mut not_ready_note: Option<String> = None;

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
                // Record the preference anyway, and say it is not yet
                // deliverable.
                //
                // Refusing here made prompt privacy a ONE-WAY setting: turning
                // it off succeeds and removes the override, and turning it back
                // on is then rejected until the shards are fetched — so a state
                // the system itself stores could not be restored through the API
                // that stores it. Found by disabling it to test something and
                // being unable to put it back.
                //
                // **Turning privacy ON should never be the blocked direction.**
                // The consequence of enabling it early is that requests for this
                // model are refused with `PromptPrivacyUnavailable`, whose hint
                // already names the fix — that is failing CLOSED, which is what
                // a privacy switch should do when it cannot be honoured.
                not_ready_note = Some(format!(
                    "Recorded, but not in effect yet: this node is missing {}. \
                     Requests for this model will be refused rather than sent out \
                     unprotected. Fetch the missing shard(s) to make it usable.",
                    missing.join(" and ")
                ));
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
        "ready": not_ready_note.is_none(),
        "note": match (&not_ready_note, body.enabled) {
            (Some(warning), _) => warning.as_str(),
            (None, true) => {
                "Encrypted pipeline active. Both embedding (first shard) and sampling (last shard) \
                 run locally. Remote nodes only see intermediate activations. \
                 Adds ~1 extra RTT per token for the return hop."
            }
            (None, false) => "Encrypted pipeline disabled. Normal pipeline scheduling applies.",
        },
    })))
}
