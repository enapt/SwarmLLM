use crate::error::ApiError;

/// Apply a recomputed shard-window to the model process pool and refresh the
/// surrounding state (split-model cache, model-load history, dashboard
/// signal). Shared between `unload_shard` and `load_shard` — the window
/// arithmetic is path-specific (subtract vs add) but the after-effects are
/// identical.
///
/// If `new_window` is empty the model is fully unloaded; otherwise the worker
/// is restarted with the narrowed/expanded window.
pub(super) async fn apply_shard_window_change(
    shared: &crate::daemon::SharedState,
    model_id: &str,
    mid: &crate::types::ModelId,
    new_window: &[u32],
) {
    if new_window.is_empty() {
        shared.evict_and_unload(mid).await;
    } else {
        shared
            .model_process_pool
            .restart_with_window(mid, new_window.to_vec())
            .await;
        shared.evict_split_models(mid);
    }
    shared.events.clear_model_load_history(model_id);
    shared.signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);
}

pub(super) fn resolve_local_shard(
    shared: &crate::daemon::SharedState,
    model_id: &str,
    shard_index: u32,
) -> Result<
    (
        crate::types::ModelId,
        crate::types::ShardId,
        crate::types::NodeId,
    ),
    ApiError,
> {
    let mid = crate::types::ModelId(model_id.to_string());
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
        // Structurally valid request for an absent resource → 404.
        // Matches the pattern used by delete_shard in the same module.
        return Err(ApiError(crate::error::SwarmError::ShardNotFound(shard_id)));
    }
    Ok((mid, shard_id, local_node_id))
}

/// Build a shard download progress JSON value from a ShardProgress entry.
/// Returns None if the shard is in a terminal state (Complete/Failed).
pub(super) fn shard_download_json(
    sp: &crate::model::acquisition::ShardProgress,
) -> Option<serde_json::Value> {
    let state_str = match sp.state {
        crate::model::acquisition::ShardState::Downloading => "Downloading",
        crate::model::acquisition::ShardState::Verifying => "Verifying",
        crate::model::acquisition::ShardState::Pending => "Queued",
        _ => return None,
    };
    let pct = crate::model::acquisition::shard_pct(sp.downloaded_bytes, sp.total_bytes);
    Some(serde_json::json!({
        "state": state_str,
        "progress_pct": pct,
        "downloaded_bytes": sp.downloaded_bytes,
        "total_bytes": sp.total_bytes,
    }))
}

/// Serialize peer download progress for a shard into JSON values.
pub(super) fn peer_downloads_json(peers: &[(crate::types::NodeId, u32)]) -> Vec<serde_json::Value> {
    peers
        .iter()
        .map(|(nid, pct)| {
            serde_json::json!({
                "node_id": format!("{}", nid),
                "progress_pct": pct,
            })
        })
        .collect()
}

/// Build per-shard JSON with common fields (index, size, local, holders, locked, download, peer_downloads).
/// Used by both `build_shard_detail` and `shard_storage` to avoid duplicating ~30 lines.
pub(super) fn build_shard_json(
    shard: &crate::types::ShardInfo,
    shared: &crate::daemon::SharedState,
    model_id: &crate::types::ModelId,
    local_node_id: &crate::types::NodeId,
    acq: Option<&crate::model::acquisition::AcquisitionStatus>,
) -> (serde_json::Value, bool, bool) {
    let shard_id = crate::types::ShardId {
        model_id: model_id.clone(),
        index: shard.index,
    };
    let holders = shared.model_registry.shard_holders(&shard_id);
    let is_local = holders.contains(local_node_id);
    let locked = shared
        .models
        .locked_shards
        .get(&shard_id)
        .map(|v| *v)
        .unwrap_or(false);
    let holder_ids: Vec<String> = holders
        .iter()
        .filter(|h| *h != local_node_id)
        .take(32)
        .map(|h| format!("{}", h))
        .collect();

    let mut shard_json = serde_json::json!({
        "index": shard.index,
        "size_bytes": shard.size_bytes,
        "local": is_local,
        "holders": holders.len(),
        "holder_ids": holder_ids,
        "locked": locked,
    });

    let mut any_downloading = false;

    // Attach per-shard download state if downloading
    if let Some(p) = acq {
        if let Some(sp) = p.shard_progress.get(&shard.index) {
            if let Some(dl) = shard_download_json(sp) {
                any_downloading = true;
                if let Some(obj) = shard_json.as_object_mut() {
                    obj.insert("download".to_string(), dl);
                }
            }
        }
    }

    // Attach peer download state
    if let Some(peer_dl) = shared.models.peer_shard_downloads.get(&shard_id) {
        let peers = peer_downloads_json(peer_dl.value());
        if !peers.is_empty() {
            any_downloading = true;
            if let Some(obj) = shard_json.as_object_mut() {
                obj.insert("peer_downloads".to_string(), serde_json::json!(peers));
            }
        }
    }

    (shard_json, is_local, any_downloading)
}

/// Serialize an acquisition progress entry to JSON. Used by both REST download_queue
/// and WebSocket build_stats_message. Caller can extend with extra fields.
pub fn serialize_acquisition_to_json(
    status: &crate::model::acquisition::AcquisitionStatus,
    shared: &crate::daemon::state::SharedState,
) -> serde_json::Value {
    let model_id = &status.model_id;
    let source = if shared.models.hf_sources.contains_key(model_id) {
        "huggingface"
    } else {
        "network"
    };
    let eta_secs = if status.speed_bytes_per_sec > 0 && status.total_bytes > status.downloaded_bytes
    {
        Some((status.total_bytes - status.downloaded_bytes) / status.speed_bytes_per_sec)
    } else {
        None
    };
    let shard_details: Vec<serde_json::Value> = status
        .shard_progress
        .iter()
        .map(|(idx, sp)| {
            let pct = crate::model::acquisition::shard_pct(sp.downloaded_bytes, sp.total_bytes);
            serde_json::json!({
                "index": idx,
                "state": serde_json::to_value(&sp.state).unwrap_or_default(),
                "progress_pct": pct,
                "downloaded_bytes": sp.downloaded_bytes,
                "total_bytes": sp.total_bytes,
            })
        })
        .collect();
    let overall_pct =
        crate::model::acquisition::shard_pct(status.downloaded_bytes, status.total_bytes);
    let model_name = shared
        .model_registry
        .get_manifest(model_id)
        .map(|m| m.name.clone())
        .unwrap_or_else(|| model_id.0.clone());
    let cancellable = matches!(
        status.state,
        crate::model::acquisition::AcquisitionState::Downloading
            | crate::model::acquisition::AcquisitionState::AwaitingManifest
    );
    serde_json::json!({
        "model_id": model_id.0,
        "model_name": model_name,
        "state": serde_json::to_value(&status.state).unwrap_or_default(),
        "source": source,
        "total_shards": status.total_shards,
        "downloaded_shards": status.downloaded_shards,
        "verified_shards": status.verified_shards,
        "total_bytes": status.total_bytes,
        "downloaded_bytes": status.downloaded_bytes,
        "overall_pct": overall_pct,
        "speed_bytes_per_sec": status.speed_bytes_per_sec,
        "eta_secs": eta_secs,
        "cancellable": cancellable,
        "log": status.log.iter().rev().take(10).collect::<Vec<_>>(),
        "shard_details": shard_details,
        // R110: surface what kicked off this download so the UI can show
        // a non-technical badge ("hosted by your swarm" / "added by you" /
        // "swarm pipeline pull"). Empty string if older internal call
        // sites didn't set it — frontend renders no badge in that case.
        "trigger": status.trigger,
    })
}
