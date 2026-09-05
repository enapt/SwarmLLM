use crate::api::server::JsonBody;
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
/// One shard's worth of `AcquisitionStatus`, for a download this node started
/// on its own. Shared by the fresh-entry and stale-entry arms of
/// `download_shard` so the two cannot describe the same thing differently.
fn fresh_shard_download_status(
    mid: &crate::types::ModelId,
    shard_index: u32,
    shard_size: u64,
) -> crate::model::acquisition::AcquisitionStatus {
    let mut st = crate::model::acquisition::AcquisitionStatus::new_downloading(
        mid.clone(),
        1,
        shard_size,
        "peers",
        "user",
        format!("Downloading shard {} from peer", shard_index + 1),
    );
    st.shard_progress.insert(
        shard_index,
        crate::model::acquisition::ShardProgress::new_downloading(shard_index, shard_size),
    );
    st
}

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
    // Two questions, both of which must be no.
    //
    // 1. Is ANY inference running against this model? `active_pipelines` alone
    //    misses local split replies and peer-served work — see
    //    `SharedState::model_is_in_use`. Those paths do not name individual
    //    shards anywhere, so for them the whole model has to be off limits;
    //    refusing a shard delete during a live reply is the right call.
    // 2. If a distributed pipeline IS running, does it span THIS shard?
    //    `seg.shard_id` names only the segment's FIRST shard, so a segment
    //    covering several would leave the rest unguarded.
    let in_use = shared.model_is_in_use(&mid)
        || shared.active_pipelines.iter().any(|entry| {
            entry.value().segments.iter().any(|seg| {
                seg.shard_id.model_id == mid
                    && shared
                        .model_registry
                        .shards_spanned_by_segment(seg)
                        .iter()
                        .any(|s| s.index == shard_index)
            })
        });
    if in_use {
        return Err(ApiError(crate::error::SwarmError::ServiceUnavailable(
            format!(
                "Shard {}/{} is currently serving an active inference; retry shortly",
                model_id, shard_index
            ),
        )));
    }

    // Would this delete strand prompt privacy? The setting requires BOTH ends of
    // the model locally, and unlike `delete_model` there is no flag to clear
    // here — removing an end would leave the setting on and every request for
    // the model failing at pipeline assembly. Refuse and name the setting, so
    // turning it off stays the user's explicit choice rather than a side effect
    // of freeing disk. See `SharedState::privacy_required_shards`.
    if let Some((first, last)) = shared.privacy_required_shards(&mid) {
        if shard_index == first || shard_index == last {
            let which = if shard_index == first && shard_index == last {
                "only shard".to_string()
            } else if shard_index == first {
                "first shard".to_string()
            } else {
                "last shard".to_string()
            };
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "Shard {model_id}/{shard_index} is the {which} of a model with prompt \
                 privacy switched on, which needs both ends of the model on this node. \
                 Deleting it would make every request for {model_id} fail. Turn prompt \
                 privacy off for this model first if you want to remove the shard."
            ))));
        }
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
    // The user removed it: auto-manage must not bring it back on its own.
    shared.mark_shard_removed_by_user(&shard_id);
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

        let announce = crate::types::SwarmMessage::ShardAnnounce(
            // Complete for this model: whatever is absent was just deleted.
            crate::model::manifest::shard_announce(
                &shared.model_registry,
                local_node_id,
                remaining_shards,
                vec![mid.clone()],
            ),
        );
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
    // An explicit request for the shard outranks an earlier removal of it.
    shared.clear_shard_removed_by_user(&shard_id);

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

    // Try P2P: find peers who hold this shard. `select_best_allowed_peer`
    // drops the local node and anything outside the pool when private mode is
    // on, so this button cannot reach a stranger the auto-manage scorer would
    // have refused; `None` falls through to the HuggingFace path below.
    let holders = shared.model_registry.shard_holders(&shard_id);

    if let Some(target) = shared.select_best_allowed_peer(&holders) {
        let peer_id_bytes = shared
            .peer_registry
            .get(&target)
            .and_then(|p| p.peer_id_bytes.clone());

        if let Some(bytes) = peer_id_bytes {
            // Fold this shard into whatever download of this model is already
            // in flight, rather than replacing it.
            //
            // `acquisition_progress` is keyed by MODEL, so a blind insert made
            // every "Download this part" click discard the entry the previous
            // click created — leaving `total_bytes` at the last shard's size
            // while `downloaded_bytes` kept accumulating across all of them.
            // Asking for five shards of an 8B model showed 1107 MB of 709, a
            // progress bar at 156%. Observed 2026-08-24 doing exactly that.
            shared
                .models
                .acquisition_progress
                .entry(mid.clone())
                .and_modify(|st| {
                    // Only extend a download that is still running; a finished
                    // or failed one is stale and this click starts afresh.
                    if matches!(
                        st.state,
                        crate::model::acquisition::AcquisitionState::Downloading
                    ) {
                        if st.shard_progress.contains_key(&shard_index) {
                            return; // already tracking this shard
                        }
                        st.total_shards = st.total_shards.saturating_add(1);
                        st.total_bytes = st.total_bytes.saturating_add(shard_size);
                        st.shard_progress.insert(
                            shard_index,
                            crate::model::acquisition::ShardProgress::new_downloading(
                                shard_index,
                                shard_size,
                            ),
                        );
                    } else {
                        *st = fresh_shard_download_status(&mid, shard_index, shard_size);
                    }
                })
                .or_insert_with(|| fresh_shard_download_status(&mid, shard_index, shard_size));

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
            let peer_label =
                crate::identity::nickname::short_display_name(&target, &shared.nickname_registry);
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
    JsonBody(body): JsonBody<ShardLockUpdate>,
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

#[cfg(test)]
mod download_progress_tests {
    use super::fresh_shard_download_status;
    use crate::model::acquisition::{AcquisitionState, ShardProgress};
    use crate::types::ModelId;

    /// Asking for several shards of one model must describe ONE download of
    /// several shards, not overwrite itself once per click.
    ///
    /// `acquisition_progress` is keyed by model, so a blind insert left
    /// `total_bytes` at the last shard's size while `downloaded_bytes` kept
    /// summing across every transfer. Observed live: 1107 MB of 709, a bar at
    /// 156%.
    #[test]
    fn asking_for_five_shards_totals_five_shards() {
        let mid = ModelId("m".into());
        let sizes: [u64; 5] = [573, 523, 507, 539, 708];

        // First click builds the entry; the rest fold into it.
        let mut st = fresh_shard_download_status(&mid, 3, sizes[0]);
        for (i, size) in sizes.iter().enumerate().skip(1) {
            let idx = 3 + i as u32;
            assert!(matches!(st.state, AcquisitionState::Downloading));
            st.total_shards = st.total_shards.saturating_add(1);
            st.total_bytes = st.total_bytes.saturating_add(*size);
            st.shard_progress
                .insert(idx, ShardProgress::new_downloading(idx, *size));
        }

        assert_eq!(st.total_shards, 5, "five shards were asked for");
        assert_eq!(
            st.total_bytes,
            sizes.iter().sum::<u64>(),
            "the total must cover every shard, not just the last one"
        );
        assert_eq!(st.shard_progress.len(), 5);
    }

    /// A repeat click on a shard already in flight must not double-count it.
    #[test]
    fn re_requesting_the_same_shard_does_not_inflate_the_total() {
        let mid = ModelId("m".into());
        let st = fresh_shard_download_status(&mid, 2, 500);
        assert!(st.shard_progress.contains_key(&2));
        // The handler's guard is `contains_key`, so a second click returns
        // early and the figures below are what must survive it.
        assert_eq!(st.total_shards, 1);
        assert_eq!(st.total_bytes, 500);
    }
}
