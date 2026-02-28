use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::config::ContributionMode;
use crate::error::ApiError;

/// Extract EOS token IDs from a GGUF file, with architecture-specific fallbacks.
/// Mirrors the logic in inference/split.rs for consistency.
fn extract_eos_token_ids(path: &std::path::Path) -> Vec<u32> {
    let mut eos_tokens = Vec::new();
    let Ok(mut file) = std::fs::File::open(path) else {
        return vec![2];
    };
    let Ok(ct) = candle_core::quantized::gguf_file::Content::read(&mut file) else {
        return vec![2];
    };
    if let Some(eos_id) = ct
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|v| v.to_u32().ok())
    {
        eos_tokens.push(eos_id);
    }
    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_default();
    match arch.as_str() {
        "qwen2" => {
            for &id in &[151643u32, 151645] {
                if !eos_tokens.contains(&id) {
                    eos_tokens.push(id);
                }
            }
        }
        _ => {
            if !eos_tokens.contains(&2) {
                eos_tokens.push(2);
            }
        }
    }
    if eos_tokens.is_empty() {
        eos_tokens.push(2);
    }
    eos_tokens
}

/// GET /api/admin/stats — Full dashboard stats snapshot.
pub async fn stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let node_id = format!("{}", state.shared_state.identity.node_id());
    let stats = state.shared_state.node_stats.read().await;
    let credit = state.shared_state.credit_balance.read().await;

    let uptime_seconds = (chrono::Utc::now() - stats.uptime_start)
        .num_seconds()
        .max(0) as u64;

    let tier = crate::credit::priority::PriorityCalculator::tier_name(credit.balance);

    // Count only shards held locally (not all tracked shards network-wide)
    let hosted_shards = {
        let local_nid = state.shared_state.identity.node_id();
        state
            .shared_state
            .model_registry
            .all_shard_entries()
            .iter()
            .filter(|(_, holders)| holders.contains(local_nid))
            .count()
    };

    // Hardware detection
    let hardware = detect_hardware(&state.shared_state);

    Json(serde_json::json!({
        "node_id": node_id,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime_seconds,
        "tier": tier,
        "peers": stats.peers_connected,
        "requests_served": stats.requests_served,
        "forwards_served": stats.forwards_served,
        "requests_made": stats.requests_made,
        "active_requests": state.shared_state.active_pipelines.len(),
        "hosted_shards": hosted_shards,
        "credits": {
            "balance": credit.balance,
            "lifetime_earned": credit.lifetime_earned,
            "lifetime_spent": credit.lifetime_spent,
        },
        "hardware": hardware,
    }))
}

/// GET /api/admin/config — Return current configuration.
pub async fn get_config(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = &state.config;
    let contribution = match config.node.contribution {
        ContributionMode::Minimal => "minimal",
        ContributionMode::Moderate => "moderate",
        ContributionMode::Maximum => "maximum",
    };
    Json(serde_json::json!({
        "contribution": contribution,
        "max_concurrent_requests": config.inference.max_concurrent_requests,
        "max_bandwidth_mbps": config.resources.max_bandwidth_mbps,
        "max_disk_mb": config.resources.max_disk_mb,
        "listen_port": config.node.listen_port,
        "session_timeout_seconds": config.inference.session_timeout_seconds,
        "auto_manage_shards": state.shared_state.auto_manage_enabled.load(std::sync::atomic::Ordering::Relaxed),
        "auto_manage_max_storage_mb": config.auto_manage.max_storage_mb,
        "shard_size_mb": config.model.shard_size_mb,
        "max_batch_size": config.inference.max_batch_size,
        "batch_timeout_ms": config.inference.batch_timeout_ms,
    }))
}

/// PUT /api/admin/config — Update configuration at runtime.
pub async fn update_config(
    State(state): State<AppState>,
    Json(body): Json<ConfigUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Persist the updated config to the config TOML file.
    // For now, acknowledge the update — runtime config changes require daemon restart.
    let config_path = state.config.node.data_dir.join("config.toml");

    // Build a partial config update
    let mut config = state.config.clone();

    if let Some(contribution) = &body.contribution {
        config.node.contribution = match contribution.as_str() {
            "minimal" => ContributionMode::Minimal,
            "maximum" => ContributionMode::Maximum,
            _ => ContributionMode::Moderate,
        };
    }
    if let Some(max_reqs) = body.max_concurrent_requests {
        config.inference.max_concurrent_requests = max_reqs;
    }
    if let Some(bw) = body.max_bandwidth_mbps {
        config.resources.max_bandwidth_mbps = bw;
    }
    if let Some(disk) = body.max_disk_mb {
        config.resources.max_disk_mb = disk;
    }
    if let Some(auto_manage) = body.auto_manage_shards {
        config.auto_manage.enabled = auto_manage;
        // Update the runtime atomic so AutoShardManager picks it up immediately
        state
            .shared_state
            .auto_manage_enabled
            .store(auto_manage, std::sync::atomic::Ordering::Release);
        if auto_manage {
            // Wake the AutoShardManager so it evaluates promptly
            state.shared_state.auto_manage_notify.notify_one();
        }
    }
    if let Some(max_storage) = body.auto_manage_max_storage_mb {
        config.auto_manage.max_storage_mb = max_storage;
    }
    if let Some(shard_size) = body.shard_size_mb {
        if !(crate::config::SHARD_SIZE_MIN_MB..=crate::config::SHARD_SIZE_MAX_MB)
            .contains(&shard_size)
        {
            return Err(ApiError(crate::error::SwarmError::Config(format!(
                "shard_size_mb must be between {} and {} (got {})",
                crate::config::SHARD_SIZE_MIN_MB,
                crate::config::SHARD_SIZE_MAX_MB,
                shard_size
            ))));
        }
        config.model.shard_size_mb = shard_size;
    }
    if let Some(batch_size) = body.max_batch_size {
        config.inference.max_batch_size = batch_size.max(1);
    }
    if let Some(timeout) = body.batch_timeout_ms {
        config.inference.batch_timeout_ms = timeout;
    }

    // Write updated config to disk
    let toml_str = toml::to_string_pretty(&config)
        .map_err(|e| ApiError(crate::error::SwarmError::Config(e.to_string())))?;

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&config_path, toml_str)
        .map_err(|e| ApiError(crate::error::SwarmError::Io(e)))?;

    tracing::info!(path = %config_path.display(), "Configuration saved");

    Ok(Json(serde_json::json!({ "status": "ok" })))
}

/// GET /api/admin/models — List known models and their status.
///
/// Returns all models: locally loaded, from the P2P registry, and discovered
/// on the network from peer announcements. Each model includes its source,
/// availability info, and which peers host it.
pub async fn list_models(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let mut models: Vec<serde_json::Value> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let local_node_id = state.shared_state.identity.node_id().clone();

    // Collect peer info for each model from shard_registry
    let mut model_peers: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for entry in state.shared_state.shard_registry.iter() {
        let shard_id = entry.key();
        let holders = entry.value();
        let model_name = shard_id.model_id.0.clone();
        for holder in holders.iter() {
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
    let build_shard_detail =
        |m: &crate::types::ModelManifest, state: &AppState| -> Vec<serde_json::Value> {
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

                    let mut shard_json = serde_json::json!({
                        "index": s.index,
                        "size_bytes": s.size_bytes,
                        "local": local,
                        "holders": holders.len(),
                    });

                    // Attach per-shard download state if downloading
                    if let Some(ref p) = acq {
                        if let Some(sp) = p.shard_progress.get(&s.index) {
                            if matches!(sp.state, crate::model::acquisition::ShardState::Downloading) {
                                let pct = if sp.total_bytes > 0 {
                                    (sp.downloaded_bytes as f64 / sp.total_bytes as f64 * 100.0) as u32
                                } else {
                                    0
                                };
                                shard_json.as_object_mut().unwrap().insert(
                                    "download".to_string(),
                                    serde_json::json!({
                                        "state": "Downloading",
                                        "progress_pct": pct,
                                        "downloaded_bytes": sp.downloaded_bytes,
                                        "total_bytes": sp.total_bytes,
                                    }),
                                );
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
                            shard_json
                                .as_object_mut()
                                .unwrap()
                                .insert("peer_downloads".to_string(), serde_json::json!(peers));
                        }
                    }

                    shard_json
                })
                .collect()
        };

    // 1. Locally loaded model (full model via --model flag)
    // Even though it's locally loaded, get the real manifest to show shard info
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
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
                state
                    .shared_state
                    .model_registry
                    .get_manifest(&slug_id)
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
        }));
    }

    // 2. Models from the P2P manifest registry
    let registry = &state.shared_state.model_registry;
    let manifests = registry.list_models();

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
        let is_loaded = state
            .shared_state
            .split_models
            .iter()
            .any(|e| e.key().0 == m.id);
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
                    })
                })
        } else {
            None
        };

        let estimated_vram = crate::model::auto_manage::estimate_model_vram_mb(m.total_size_bytes);

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
            "acquisition": acq_state,
            "acquisition_progress": acq_progress,
        }));
    }

    // 3. Models discovered from peer announcements (not in our registry or loaded)
    for (model_name, peers) in &model_peers {
        if seen_ids.contains(model_name) {
            continue;
        }
        seen_ids.insert(model_name.clone());
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

/// GET /api/admin/peers — List connected peers.
pub async fn list_peers(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let timeout = chrono::Duration::seconds(90); // 3 missed pings
    let now = chrono::Utc::now();

    let peers: Vec<serde_json::Value> = state
        .shared_state
        .peer_registry
        .iter()
        .map(|entry| {
            let peer = entry.value();
            let healthy = now.signed_duration_since(peer.last_seen) < timeout;
            let hosted_models: Vec<String> = peer
                .capability
                .as_ref()
                .map(|c| {
                    c.hosted_shards
                        .iter()
                        .map(|s| s.model_id.0.clone())
                        .collect()
                })
                .unwrap_or_default();

            serde_json::json!({
                "node_id": format!("{}", peer.node_id),
                "addresses": peer.addresses,
                "last_seen": peer.last_seen.to_rfc3339(),
                "latency_ms": peer.latency_ms,
                "trust_score": peer.trust_score,
                "healthy": healthy,
                "gpu": peer.capability.as_ref().and_then(|c| c.gpu.as_ref().map(|g| &g.name)),
                "hosted_models": hosted_models,
            })
        })
        .collect();

    Json(peers)
}

/// GET /api/admin/credits — Credit details.
pub async fn credit_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let credit = state.shared_state.credit_balance.read().await;
    let tier = crate::credit::priority::PriorityCalculator::tier_name(credit.balance);

    Json(serde_json::json!({
        "balance": credit.balance,
        "lifetime_earned": credit.lifetime_earned,
        "lifetime_spent": credit.lifetime_spent,
        "tier": tier,
        "last_updated": credit.last_updated.to_rfc3339(),
    }))
}

/// GET /api/admin/api-key — Return the current API key.
/// This endpoint requires authentication itself (Bearer token).
pub async fn get_api_key(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "api_key": state.shared_state.api_key,
    }))
}

// ---- Request types ----

#[derive(Debug, Deserialize)]
pub struct ConfigUpdate {
    pub contribution: Option<String>,
    pub max_concurrent_requests: Option<u32>,
    pub max_bandwidth_mbps: Option<u64>,
    pub max_disk_mb: Option<u64>,
    pub auto_manage_shards: Option<bool>,
    pub auto_manage_max_storage_mb: Option<u64>,
    pub shard_size_mb: Option<u64>,
    pub max_batch_size: Option<u32>,
    pub batch_timeout_ms: Option<u64>,
    #[serde(default)]
    pub models: Vec<String>,
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

            let mut shard_json = serde_json::json!({
                "index": shard.index,
                "size_bytes": shard.size_bytes,
                "local": is_local,
                "holders": holders.len(),
            });
            if let Some(dl) = download_state {
                shard_json
                    .as_object_mut()
                    .unwrap()
                    .insert("download".to_string(), dl);
            }
            if let Some(peers_dl) = peer_downloading {
                if !peers_dl.is_empty() {
                    shard_json
                        .as_object_mut()
                        .unwrap()
                        .insert("peer_downloads".to_string(), serde_json::json!(peers_dl));
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
            if ft.is_file() {
                total += entry.metadata()?.len();
            } else if ft.is_dir() {
                total += dir_size(&entry.path())?;
            }
        }
    }
    Ok(total)
}

// ---- HuggingFace Endpoints ----

/// GET /api/admin/hf/search?q=... — Search HuggingFace for GGUF models.
pub async fn hf_search(
    axum::extract::Query(params): axum::extract::Query<HfSearchParams>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let query = params.q.unwrap_or_default();
    if query.is_empty() {
        return Ok(Json(vec![]));
    }

    let results = crate::model::huggingface::search_gguf_models(&query)
        .await
        .map_err(|e| ApiError(crate::error::SwarmError::Internal(e)))?;

    let values: Vec<serde_json::Value> = results
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "repo_id": r.repo_id,
                "filename": r.filename,
                "size_bytes": r.size_bytes,
                "downloads": r.downloads,
            })
        })
        .collect();

    Ok(Json(values))
}

/// POST /api/admin/hf/download — Start downloading a GGUF model from HuggingFace.
pub async fn hf_download(
    State(state): State<AppState>,
    Json(body): Json<HfDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = body.repo_id;
    let filename = body.filename;

    if repo_id.is_empty() || filename.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "repo_id and filename are required".into(),
        )));
    }

    let dest_dir = state
        .config
        .node
        .data_dir
        .join("models")
        .join(repo_id.replace('/', "_"));

    tracing::info!(repo = %repo_id, file = %filename, "Starting HuggingFace download");

    // Spawn download in background
    let repo_clone = repo_id.clone();
    let file_clone = filename.clone();
    let shared = state.shared_state.clone();
    let model_id_str = format!("hf:{}/{}", repo_id, filename);
    let mid = crate::types::ModelId(model_id_str.clone());

    // Create initial acquisition progress entry
    let status = crate::model::acquisition::AcquisitionStatus {
        model_id: mid.clone(),
        state: crate::model::acquisition::AcquisitionState::Downloading,
        total_shards: 1,
        downloaded_shards: 0,
        verified_shards: 0,
        failed_shards: 0,
        total_bytes: 0,
        downloaded_bytes: 0,
        shard_progress: std::collections::HashMap::new(),
        speed_bytes_per_sec: 0,
        started_at: Some(chrono::Utc::now()),
        log: vec![format!("Downloading {} from HuggingFace...", filename)],
    };
    shared.acquisition_progress.insert(mid.clone(), status);

    tokio::spawn(async move {
        let (ptx, mut prx) =
            tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);

        let download_mid = mid.clone();
        let download_shared = shared.clone();

        // Spawn progress updater
        let progress_mid = mid.clone();
        let progress_shared = shared.clone();
        tokio::spawn(async move {
            let mut last_bytes = 0u64;
            let mut last_time = std::time::Instant::now();
            while let Some(prog) = prx.recv().await {
                if let Some(mut entry) = progress_shared.acquisition_progress.get_mut(&progress_mid)
                {
                    entry.downloaded_bytes = prog.downloaded_bytes;
                    entry.total_bytes = prog.total_bytes;
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(last_time).as_secs_f64();
                    if dt > 0.5 {
                        let speed = ((prog.downloaded_bytes - last_bytes) as f64 / dt) as u64;
                        entry.speed_bytes_per_sec = speed;
                        last_bytes = prog.downloaded_bytes;
                        last_time = now;
                    }
                }
            }
        });

        match crate::model::huggingface::download_model(
            &repo_clone,
            &file_clone,
            &dest_dir,
            Some(ptx),
        )
        .await
        {
            Ok(path) => {
                tracing::info!(path = %path.display(), "HuggingFace download complete");
                if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid)
                {
                    entry.state = crate::model::acquisition::AcquisitionState::Complete;
                    entry.downloaded_shards = 1;
                    entry.verified_shards = 1;
                    entry
                        .log
                        .push(format!("Download complete: {}", path.display()));
                }

                // Try to load the downloaded model
                let executor = download_shared.executor.clone();
                let gpu_layers = download_shared.config.inference.gpu_layers;
                let model_name = format!("{}/{}", repo_clone, file_clone);

                let mut exec = executor.lock().await;
                match exec.load_model(&path, gpu_layers) {
                    Ok(()) => {
                        let size = exec.model_size_bytes().unwrap_or(0);
                        let gguf_meta = crate::inference::executor::extract_gguf_metadata(&path);
                        let eos_tokens = extract_eos_token_ids(&path);
                        *download_shared.loaded_model_info.write().await =
                            Some(crate::daemon::LoadedModelInfo {
                                name: model_name.clone(),
                                size_bytes: size,
                                eos_tokens,
                                chat_template: gguf_meta
                                    .as_ref()
                                    .and_then(|m| m.chat_template.clone()),
                                bos_token: gguf_meta
                                    .as_ref()
                                    .map(|m| m.bos_token.clone())
                                    .unwrap_or_default(),
                                eos_token: gguf_meta
                                    .as_ref()
                                    .map(|m| m.eos_token.clone())
                                    .unwrap_or_default(),
                            });
                        download_shared
                            .model_loaded
                            .store(true, std::sync::atomic::Ordering::Release);
                        if let Some(mut entry) =
                            download_shared.acquisition_progress.get_mut(&download_mid)
                        {
                            entry.log.push(format!("Model loaded: {}", model_name));
                        }
                        tracing::info!(model = %model_name, "HF model loaded for inference");
                    }
                    Err(e) => {
                        if let Some(mut entry) =
                            download_shared.acquisition_progress.get_mut(&download_mid)
                        {
                            entry.log.push(format!("Model load failed: {}", e));
                        }
                        tracing::error!(error = %e, "Failed to load HF model");
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "HuggingFace download failed");
                if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid)
                {
                    entry.state =
                        crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                    entry.failed_shards = 1;
                    entry.log.push(format!("Download failed: {}", e));
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "model_id": model_id_str,
    })))
}

/// POST /api/admin/shutdown — Gracefully shut down the node.
/// Only accepts requests from localhost (127.0.0.1 or ::1) for safety.
pub async fn shutdown_node(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !addr.ip().is_loopback() {
        return Err(ApiError(crate::error::SwarmError::Internal(
            "Shutdown only allowed from localhost".into(),
        )));
    }
    tracing::info!("Shutdown requested via API from {}", addr);

    // Signal all subsystems to shut down
    state.shared_state.shutdown();

    // Flush the database before exiting to prevent corruption
    let db = state.shared_state.db.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        if let Err(e) = db.flush() {
            tracing::error!(error = %e, "Failed to flush database on shutdown");
        }
        std::process::exit(0);
    });

    Ok(Json(serde_json::json!({ "status": "shutting_down" })))
}

#[derive(Debug, Deserialize)]
pub struct HfSearchParams {
    pub q: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HfDownloadRequest {
    pub repo_id: String,
    pub filename: String,
    #[serde(default)]
    pub mode: String,
}

#[derive(Debug, Deserialize)]
pub struct HfShardDownloadRequest {
    pub repo_id: String,
    pub filename: String,
    /// Which shard indices to download (e.g. [0,1,2] for the first 3 shards).
    /// If empty, the server will probe the file and return shard info without downloading.
    #[serde(default)]
    pub shards: Vec<u32>,
    /// Optional: target an existing model_id so downloaded shards merge into its directory.
    /// If omitted, a new model_id is derived from the filename.
    #[serde(default)]
    pub model_id: Option<String>,
}

/// GET /api/admin/hf/probe — Probe a remote GGUF file to get shard info.
pub async fn hf_probe(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfProbeParams>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = params.repo_id.unwrap_or_default();
    let filename = params.filename.unwrap_or_default();

    if repo_id.is_empty() || filename.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "repo_id and filename query params are required".into(),
        )));
    }

    let shard_size = state.config.model.shard_size_bytes();
    match crate::model::huggingface::probe_gguf_file_with_shard_size(
        &repo_id, &filename, shard_size,
    )
    .await
    {
        Ok(info) => Ok(Json(serde_json::json!({
            "status": "ok",
            "total_size": info.total_size,
            "header_size": info.header_size,
            "shard_count": info.shard_count,
            "shard_size": info.shard_size,
        }))),
        Err(e) => Err(ApiError(crate::error::SwarmError::Internal(e))),
    }
}

#[derive(Debug, Deserialize)]
pub struct HfProbeParams {
    pub repo_id: Option<String>,
    pub filename: Option<String>,
}

/// POST /api/admin/hf/download-shards — Download specific shards of a GGUF from HuggingFace.
///
/// Instead of downloading the full multi-GB GGUF file, this downloads only the
/// GGUF header (~6MB) plus the requested shard byte ranges (~512MB each).
/// After download, it generates a manifest and registers the shards.
pub async fn hf_download_shards(
    State(state): State<AppState>,
    Json(body): Json<HfShardDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = body.repo_id;
    let filename = body.filename;
    let shard_indices = body.shards;

    if repo_id.is_empty() || filename.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "repo_id and filename are required".into(),
        )));
    }

    if shard_indices.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "shards array is required (e.g. [0, 1, 2])".into(),
        )));
    }

    // Use provided model_id if it matches an existing model, otherwise derive from filename
    let safe_name = if let Some(ref mid) = body.model_id {
        let candidate_dir = state.config.node.data_dir.join("models").join(mid);
        if candidate_dir.exists() {
            mid.clone()
        } else {
            // Fall back to filename-derived name
            filename
                .trim_end_matches(".gguf")
                .to_lowercase()
                .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
                .split('-')
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("-")
        }
    } else {
        filename
            .trim_end_matches(".gguf")
            .to_lowercase()
            .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    };

    let dest_dir = state.config.node.data_dir.join("models").join(&safe_name);

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        shards = ?shard_indices,
        dest = %dest_dir.display(),
        "Starting HuggingFace shard download"
    );

    let model_id_str = safe_name.clone();
    let mid = crate::types::ModelId(model_id_str.clone());

    // Create initial acquisition progress entry with per-shard progress so that
    // auto-manage can detect these downloads are already in flight and skip them.
    let mut initial_shard_progress = std::collections::HashMap::new();
    for &idx in &shard_indices {
        initial_shard_progress.insert(
            idx,
            crate::model::acquisition::ShardProgress {
                index: idx,
                total_bytes: 0,
                downloaded_bytes: 0,
                state: crate::model::acquisition::ShardState::Downloading,
            },
        );
    }
    let status = crate::model::acquisition::AcquisitionStatus {
        model_id: mid.clone(),
        state: crate::model::acquisition::AcquisitionState::Downloading,
        total_shards: shard_indices.len() as u32,
        downloaded_shards: 0,
        verified_shards: 0,
        failed_shards: 0,
        total_bytes: 0,
        downloaded_bytes: 0,
        shard_progress: initial_shard_progress,
        speed_bytes_per_sec: 0,
        started_at: Some(chrono::Utc::now()),
        log: vec![format!(
            "Downloading shards {:?} of {} from HuggingFace...",
            shard_indices, filename
        )],
    };
    let shared = state.shared_state.clone();
    shared.acquisition_progress.insert(mid.clone(), status);

    // Clone values needed both in the spawn and the response
    let response_model_id = model_id_str.clone();
    let response_shards = shard_indices.clone();

    // Capture network_tx for broadcasting HfSourceGossip + ModelManifest after download
    let network_tx = state.network_tx.clone();

    tokio::spawn(async move {
        let download_mid = mid.clone();
        let download_shared = shared.clone();
        let configured_shard_size = shared.config.model.shard_size_bytes();

        // ── Phase 1: Probe + header → broadcast manifest EARLY ──────────
        // This lets peers learn about the model and begin auto-acquiring
        // shards in parallel while this node is still downloading.

        let info = match crate::model::huggingface::probe_gguf_file_with_shard_size(
            &repo_id,
            &filename,
            configured_shard_size,
        )
        .await
        {
            Ok(info) => info,
            Err(e) => {
                tracing::error!(error = %e, "HuggingFace probe failed");
                if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                    entry.state = crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                    entry.log.push(format!("Probe failed: {}", e));
                }
                return;
            }
        };

        if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
            entry.total_bytes = info.total_size;
            entry.total_shards = info.shard_count;
            entry.log.push(format!(
                "Probed: {} shards, {:.1} MB total",
                info.shard_count,
                info.total_size as f64 / (1024.0 * 1024.0)
            ));
        }

        // Download GGUF header (~6MB) — needed for manifest generation
        if let Err(e) = crate::model::huggingface::download_gguf_header(
            &repo_id,
            &filename,
            &dest_dir,
            info.header_size,
        )
        .await
        {
            tracing::error!(error = %e, "GGUF header download failed");
            if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                entry.state = crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                entry.log.push(format!("Header download failed: {}", e));
            }
            return;
        }

        // Generate manifest from header BEFORE downloading shard data.
        // Pass empty shard_indices — no shards to register yet (they don't exist on disk).
        let header_path = dest_dir.join("gguf_header.bin");
        let manifest_result = generate_manifest_from_header(&ManifestGenParams {
            header_path: &header_path,
            model_id_str: &model_id_str,
            filename: &filename,
            total_size: info.total_size,
            shard_count: info.shard_count,
            shard_indices: &[],
            shared: &download_shared,
        });

        if let Err(e) = &manifest_result {
            tracing::error!(error = %e, "Manifest generation failed (early broadcast skipped)");
            if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                entry.log.push(format!("Manifest generation failed: {e}"));
            }
            // Continue with downloads anyway — manifest can be regenerated later
        }

        // Record HF source so auto-manager (and peers) know where to fetch shards
        let hf_source = crate::daemon::HfSource {
            repo_id: repo_id.clone(),
            filename: filename.clone(),
        };
        download_shared.hf_sources.insert(
            crate::types::ModelId(model_id_str.clone()),
            hf_source.clone(),
        );
        let _ = download_shared.db.put_json("hf_sources", &model_id_str, &hf_source);
        let hf_source_path = dest_dir.join("hf_source.json");
        let _ = std::fs::write(
            &hf_source_path,
            serde_json::to_string_pretty(&hf_source).unwrap_or_default(),
        );

        // Broadcast HfSourceGossip + ModelManifest EARLY so peers can start
        // auto-acquiring shards immediately (before our shard data downloads finish).
        if let Some(ref ntx) = network_tx {
            let gossip_msg = crate::types::SwarmMessage::HfSourceGossip(
                crate::types::HfSourceGossip {
                    model_id: crate::types::ModelId(model_id_str.clone()),
                    repo_id: repo_id.clone(),
                    filename: filename.clone(),
                    publisher: download_shared.identity.node_id().clone(),
                },
            );
            let _ = ntx
                .send(crate::types::NetworkCommand::Broadcast(gossip_msg))
                .await;

            if let Some(manifest) = download_shared
                .model_registry
                .get_manifest(&crate::types::ModelId(model_id_str.clone()))
            {
                let _ = ntx
                    .send(crate::types::NetworkCommand::Broadcast(
                        crate::types::SwarmMessage::ModelManifest(manifest),
                    ))
                    .await;
            }

            tracing::info!(model = %model_id_str, "Broadcast manifest + HF source EARLY (before shard downloads)");
        }

        // Wake auto-manage on this node too (gossipsub doesn't self-deliver)
        download_shared.auto_manage_notify.notify_one();

        // ── Phase 2: Download shard data ────────────────────────────────

        let (ptx, mut prx) =
            tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);

        // Spawn progress updater
        let progress_mid = mid.clone();
        let progress_shared = shared.clone();
        tokio::spawn(async move {
            let mut last_bytes = 0u64;
            let mut last_time = std::time::Instant::now();
            while let Some(prog) = prx.recv().await {
                if let Some(mut entry) = progress_shared.acquisition_progress.get_mut(&progress_mid)
                {
                    entry.downloaded_bytes = prog.downloaded_bytes;
                    entry.total_bytes = prog.total_bytes;
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(last_time).as_secs_f64();
                    if dt > 0.5 {
                        let speed = ((prog.downloaded_bytes - last_bytes) as f64 / dt) as u64;
                        entry.speed_bytes_per_sec = speed;
                        last_bytes = prog.downloaded_bytes;
                        last_time = now;
                    }
                }
            }
        });

        // Download individual shard byte ranges (header already downloaded above)
        let total_shard_bytes: u64 = shard_indices
            .iter()
            .map(|&idx| {
                let start = (idx as u64) * info.shard_size;
                let end = ((idx as u64 + 1) * info.shard_size).min(info.total_size);
                end - start
            })
            .sum();

        let mut cumulative_downloaded: u64 = 0;
        let mut failed = false;

        for &shard_idx in &shard_indices {
            if shard_idx >= info.shard_count {
                tracing::error!(shard_idx, max = info.shard_count - 1, "Shard index out of range");
                failed = true;
                break;
            }

            let (shard_tx, mut shard_rx) = tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);
            let progress_tx_clone = ptx.clone();
            let base_downloaded = cumulative_downloaded;
            let total = total_shard_bytes;
            let progress_task = tokio::spawn(async move {
                while let Some(prog) = shard_rx.recv().await {
                    let _ = progress_tx_clone.try_send(crate::model::huggingface::DownloadProgress {
                        downloaded_bytes: base_downloaded + prog.downloaded_bytes,
                        total_bytes: total,
                    });
                }
            });

            match crate::model::huggingface::download_shard(
                &repo_id,
                &filename,
                &dest_dir,
                shard_idx,
                info.total_size,
                info.shard_size,
                Some(shard_tx),
            )
            .await
            {
                Ok(_shard_path) => {
                    progress_task.abort();
                    let start = (shard_idx as u64) * info.shard_size;
                    let end = ((shard_idx as u64 + 1) * info.shard_size).min(info.total_size);
                    cumulative_downloaded += end - start;

                    if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                        entry.downloaded_shards += 1;
                        entry.log.push(format!("Shard {} downloaded", shard_idx));
                        // Mark this shard's progress as complete so check_and_load_model
                        // won't skip it as "still downloading"
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                            sp.state = crate::model::acquisition::ShardState::Complete;
                            sp.downloaded_bytes = sp.total_bytes;
                        }
                    }

                    // Register this shard locally so the node knows it has it
                    let shard_id = crate::types::ShardId {
                        model_id: crate::types::ModelId(model_id_str.clone()),
                        index: shard_idx,
                    };
                    let node_id = download_shared.identity.node_id().clone();
                    download_shared
                        .model_registry
                        .record_shard_holder(shard_id.clone(), node_id.clone());
                    let mut holders = download_shared.shard_registry.entry(shard_id.clone()).or_default();
                    if !holders.contains(&node_id) {
                        holders.push(node_id.clone());
                    }
                    drop(holders);

                    // Announce this individual shard to the network immediately
                    // so peers see partial progress and can start acquiring
                    if let Some(ref ntx) = network_tx {
                        let ann = crate::types::SwarmMessage::ShardAnnounce(
                            crate::types::ShardAnnounce {
                                node_id,
                                shards: vec![shard_id],
                                timestamp: chrono::Utc::now(),
                            },
                        );
                        let _ = ntx.send(crate::types::NetworkCommand::Broadcast(ann)).await;
                    }
                }
                Err(e) => {
                    progress_task.abort();
                    tracing::error!(error = %e, shard_idx, "Shard download failed");
                    if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                        entry.failed_shards += 1;
                        entry.log.push(format!("Shard {} failed: {}", shard_idx, e));
                    }
                    failed = true;
                    break;
                }
            }
        }

        // Drop the progress sender so the updater task exits
        drop(ptx);

        if failed {
            if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                entry.state = crate::model::acquisition::AcquisitionState::Failed {
                    reason: "One or more shard downloads failed".to_string(),
                };
            }
        } else {
            tracing::info!(
                model = %model_id_str,
                shards = ?shard_indices,
                "All shard downloads complete"
            );

            // Regenerate manifest with correct BLAKE3 hashes now that shard files
            // exist on disk. The early manifest had [0u8; 32] placeholders.
            let _ = generate_manifest_from_header(&ManifestGenParams {
                header_path: &header_path,
                model_id_str: &model_id_str,
                filename: &filename,
                total_size: info.total_size,
                shard_count: info.shard_count,
                shard_indices: &shard_indices,
                shared: &download_shared,
            });

            if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                entry.state = crate::model::acquisition::AcquisitionState::Complete;
                entry.verified_shards = shard_indices.len() as u32;
                entry.log.push("All shards downloaded and registered".to_string());
            }

            // Load available shards for inference (partial is fine)
            crate::model::auto_manage::check_and_load_model(
                &download_shared,
                &crate::types::ModelId(model_id_str.clone()),
            )
            .await;

            // Wake auto-manage again to re-evaluate (maybe download more shards)
            download_shared.auto_manage_notify.notify_one();
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "model_id": response_model_id,
        "shards": response_shards,
    })))
}

struct ManifestGenParams<'a> {
    header_path: &'a std::path::Path,
    model_id_str: &'a str,
    filename: &'a str,
    total_size: u64,
    shard_count: u32,
    shard_indices: &'a [u32],
    shared: &'a std::sync::Arc<crate::daemon::SharedState>,
}

/// Generate a manifest from a downloaded GGUF header and register shards.
fn generate_manifest_from_header(params: &ManifestGenParams<'_>) -> Result<(), String> {
    use crate::inference::split::GgufTensorMeta;

    let header_path = params.header_path;
    let total_size = params.total_size;
    let shard_count = params.shard_count;

    // Parse model metadata from the GGUF header
    let meta = GgufTensorMeta::from_gguf_file(header_path)
        .map_err(|e| format!("Failed to parse GGUF header: {e}"))?;

    let model_id = crate::types::ModelId(params.model_id_str.to_string());
    let num_layers = meta.block_count as u32;

    // Build a friendly model name from the GGUF metadata or filename
    let model_name = meta
        .model_name
        .clone()
        .unwrap_or_else(|| params.filename.trim_end_matches(".gguf").to_string());

    // Detect architecture from the GGUF header
    let architecture = {
        let header_bytes = std::fs::read(header_path).map_err(|e| e.to_string())?;
        let mut cursor = std::io::Cursor::new(&header_bytes);
        let ct = candle_core::quantized::gguf_file::Content::read(&mut cursor)
            .map_err(|e| format!("Failed to re-parse GGUF header: {e}"))?;
        let arch_str = ct
            .metadata
            .get("general.architecture")
            .and_then(|v| v.to_string().ok().cloned())
            .unwrap_or_else(|| "llama".to_string());
        match arch_str.as_str() {
            "qwen2" => crate::types::ModelArchitecture::Qwen2,
            "mistral" => crate::types::ModelArchitecture::Mistral,
            "phi" | "phi3" => crate::types::ModelArchitecture::Phi,
            _ => crate::types::ModelArchitecture::Llama,
        }
    };

    let configured_shard_size = params.shared.config.model.shard_size_bytes();

    // Build shard infos with accurate layer ranges computed from actual GGUF tensor
    // positions.  The naive approach (num_layers / shard_count) doesn't work because
    // layer tensors don't align to shard byte boundaries — a layer's data may start in
    // one shard and end in the next.
    let mut shards = Vec::with_capacity(shard_count as usize);

    let model_dir = header_path
        .parent()
        .ok_or_else(|| "GGUF header path has no parent directory".to_string())?;

    for idx in 0..shard_count {
        let shard_start = (idx as u64) * configured_shard_size;
        let shard_end = ((idx as u64 + 1) * configured_shard_size).min(total_size);
        let shard_size = shard_end - shard_start;

        // Compute accurate layer range: which layers have ALL tensors in this shard
        let (ls, le) = crate::inference::split::compute_local_layer_range(
            &meta,
            configured_shard_size,
            &[idx],
        );

        // Compute BLAKE3 hash for shards we actually have
        let hash = {
            let shard_path = model_dir.join(format!("shard_{idx:03}.bin"));
            if shard_path.exists() {
                let data = std::fs::read(&shard_path).map_err(|e| e.to_string())?;
                *blake3::hash(&data).as_bytes()
            } else {
                [0u8; 32] // Unknown hash for shards we don't have
            }
        };

        shards.push(crate::types::ShardInfo {
            index: idx,
            layer_range: (ls as u32, le as u32),
            size_bytes: shard_size,
            hash,
        });
    }

    let node_id = params.shared.identity.node_id().clone();

    let mut manifest = crate::types::ModelManifest {
        id: model_id.clone(),
        name: model_name,
        architecture,
        num_layers,
        num_params_billions: 0.0,
        quantization: crate::types::Quantization::Q4KM,
        total_size_bytes: total_size,
        shard_count,
        shards,
        tokenizer_hash: [0u8; 32],
        manifest_hash: [0u8; 32],
        publisher: node_id.clone(),
        publish_date: chrono::Utc::now(),
        license: "Unknown".to_string(),
    };
    manifest.manifest_hash = manifest.compute_hash();

    // Save manifest to disk
    manifest.save_to_dir(model_dir).map_err(|e| e.to_string())?;

    // Register manifest in the model registry
    params
        .shared
        .model_registry
        .register_manifest(manifest.clone());

    // Store GGUF metadata
    params.shared.gguf_meta.insert(model_id.clone(), meta);

    // Register this node as holder of the downloaded shards
    for &shard_idx in params.shard_indices {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_idx,
        };
        params
            .shared
            .model_registry
            .record_shard_holder(shard_id.clone(), node_id.clone());
        let mut holders = params.shared.shard_registry.entry(shard_id).or_default();
        if !holders.contains(&node_id) {
            holders.push(node_id.clone());
        }
    }

    tracing::info!(
        model = %model_id,
        shards_registered = params.shard_indices.len(),
        num_layers,
        "Generated manifest and registered shards from HF download"
    );

    Ok(())
}

// ---- Hardware detection ----

fn detect_hardware(shared_state: &crate::daemon::SharedState) -> serde_json::Value {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let used_ram_mb = sys.used_memory() / (1024 * 1024);

    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown".to_string());
    let cpu_cores = sys.cpus().len();

    // Disk info — use sysinfo disks
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let (mut total_disk_mb, mut available_disk_mb) = (0u64, 0u64);
    for disk in disks.list() {
        total_disk_mb += disk.total_space() / (1024 * 1024);
        available_disk_mb += disk.available_space() / (1024 * 1024);
    }
    let used_disk_mb = total_disk_mb.saturating_sub(available_disk_mb);

    // GPU info from llama.cpp device detection (set at startup)
    // Falls back to nvidia-smi when gpu_info is None (e.g. non-CUDA build)
    let (gpu_name, gpu_vram_mb) = match &shared_state.gpu_info {
        Some(gpu) => (Some(gpu.name.clone()), Some(gpu.vram_total_mb)),
        None => detect_gpu_nvidia_smi(),
    };

    serde_json::json!({
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "total_ram_mb": total_ram_mb,
        "used_ram_mb": used_ram_mb,
        "available_disk_mb": available_disk_mb,
        "total_disk_mb": total_disk_mb,
        "used_disk_mb": used_disk_mb,
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
    })
}

/// Fallback GPU detection via nvidia-smi when llama.cpp gpu_info is unavailable.
fn detect_gpu_nvidia_smi() -> (Option<String>, Option<u64>) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let line = text.trim();
            if let Some((name, vram_str)) = line.split_once(',') {
                let name = name.trim().to_string();
                let vram_mb = vram_str.trim().parse::<u64>().ok();
                (Some(name), vram_mb)
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    }
}

/// GET /api/admin/network-map — Aggregated region data for the world heatmap.
///
/// Returns `{ regions: { "US": { total: N, models: { "model-id": count } }, ... } }`
/// based on self-reported region in peer capabilities.
pub async fn network_map(State(state): State<AppState>) -> Json<serde_json::Value> {
    use std::collections::HashMap;

    let mut regions: HashMap<String, (u64, HashMap<String, u64>)> = HashMap::new();

    // Include our own node if it has a region configured
    if let Some(ref region) = state.shared_state.config.identity.region {
        let code = region.to_uppercase();
        let entry = regions.entry(code).or_insert_with(|| (0, HashMap::new()));
        entry.0 += 1;
        // Add our hosted models
        for item in state.shared_state.shard_registry.iter() {
            let model_id = &item.key().model_id.0;
            let node_id = state.shared_state.identity.node_id();
            if item.value().contains(node_id) {
                *entry.1.entry(model_id.clone()).or_insert(0) += 1;
            }
        }
    }

    // Aggregate peer regions from capabilities
    for peer in state.shared_state.peer_registry.iter() {
        if let Some(ref cap) = peer.value().capability {
            if let Some(ref region) = cap.region {
                let code = region.to_uppercase();
                let entry = regions.entry(code).or_insert_with(|| (0, HashMap::new()));
                entry.0 += 1;
                // Count distinct models this peer hosts
                let mut peer_models = std::collections::HashSet::new();
                for shard in &cap.hosted_shards {
                    peer_models.insert(shard.model_id.0.clone());
                }
                for model_id in peer_models {
                    *entry.1.entry(model_id).or_insert(0) += 1;
                }
            }
        }
    }

    // Build JSON
    let region_json: serde_json::Map<String, serde_json::Value> = regions
        .into_iter()
        .map(|(code, (total, models))| {
            let models_json: serde_json::Map<String, serde_json::Value> = models
                .into_iter()
                .map(|(k, v)| (k, serde_json::json!(v)))
                .collect();
            (
                code,
                serde_json::json!({
                    "total": total,
                    "models": models_json,
                }),
            )
        })
        .collect();

    Json(serde_json::json!({ "regions": region_json }))
}
