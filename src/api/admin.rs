use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use axum::extract::Path;

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

    // Inference performance metrics from latency samples
    let inference_perf = {
        let samples = state.shared_state.inference_latency_samples.read();
        match samples {
            Ok(s) if !s.is_empty() => {
                let count = s.len();
                let sum: f64 = s.iter().sum();
                let avg_ms = (sum / count as f64) * 1000.0;
                let min_ms = s.iter().cloned().fold(f64::INFINITY, f64::min) * 1000.0;
                let max_ms = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max) * 1000.0;
                // p50 / p95 / p99
                let mut sorted: Vec<f64> = s.iter().cloned().collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let p50_ms = sorted[count / 2] * 1000.0;
                let p95_ms = sorted[((count as f64 * 0.95) as usize).min(count - 1)] * 1000.0;
                let p99_ms = sorted[((count as f64 * 0.99) as usize).min(count - 1)] * 1000.0;
                serde_json::json!({
                    "total_requests": state.shared_state.inference_requests_total
                        .load(std::sync::atomic::Ordering::Relaxed),
                    "avg_latency_ms": (avg_ms * 10.0).round() / 10.0,
                    "min_latency_ms": (min_ms * 10.0).round() / 10.0,
                    "max_latency_ms": (max_ms * 10.0).round() / 10.0,
                    "p50_latency_ms": (p50_ms * 10.0).round() / 10.0,
                    "p95_latency_ms": (p95_ms * 10.0).round() / 10.0,
                    "p99_latency_ms": (p99_ms * 10.0).round() / 10.0,
                    "samples": count,
                })
            }
            _ => serde_json::json!({
                "total_requests": state.shared_state.inference_requests_total
                    .load(std::sync::atomic::Ordering::Relaxed),
                "avg_latency_ms": null,
                "samples": 0,
            }),
        }
    };

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
        "inference": inference_perf,
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
        config.inference.max_concurrent_requests = max_reqs.clamp(1, 256);
    }
    if let Some(bw) = body.max_bandwidth_mbps {
        config.resources.max_bandwidth_mbps = bw.clamp(1, 100_000);
    }
    if let Some(disk) = body.max_disk_mb {
        config.resources.max_disk_mb = disk.clamp(100, 10_000_000);
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

/// POST /api/admin/config/reload — Hot-reload operational parameters from config file.
///
/// Re-reads the config.toml and applies hot-reloadable parameters
/// (max_concurrent_requests, auto_manage interval, max_batch_size, max_peers,
/// session_timeout_secs) without requiring a daemon restart.
pub async fn reload_config(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config_path = state.config.node.data_dir.join("config.toml");
    tracing::info!(
        path = %config_path.display(),
        "Config reload requested via API"
    );

    let params = crate::config::reload_operational_params(&config_path).map_err(ApiError)?;

    let old = crate::config::OperationalParams::from_config(&state.config);
    let changed = params != old;

    state.shared_state.apply_config_reload(params.clone());

    if changed {
        tracing::info!(?params, "Config reloaded with changes via API");
    } else {
        tracing::info!("Config reloaded via API — no changes detected");
    }

    Ok(Json(serde_json::json!({
        "status": "ok",
        "changed": changed,
        "params": {
            "max_concurrent_requests": params.max_concurrent_requests,
            "auto_manage_interval_minutes": params.auto_manage_interval_minutes,
            "max_batch_size": params.max_batch_size,
            "max_peers": params.max_peers,
            "session_timeout_secs": params.session_timeout_secs,
        }
    })))
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
                state.shared_state.hf_sources.contains_key(&mid_check)
                    || state.shared_state.hf_probe_cache.contains_key(&mid_check)
            };
            let mmproj_info = {
                let mid_mmproj = crate::types::ModelId(info.name.clone());
                let holders = state.shared_state.model_registry.mmproj_holders(&mid_mmproj);
                let local_has = holders.contains(local_node_id);
                serde_json::json!({
                    "available": !holders.is_empty(),
                    "local": local_has,
                    "holders": holders.len(),
                })
            };
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
            }));
        } // else: stale loaded model, files deleted
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
        let is_loaded = if hosted_count > 0 {
            state
                .shared_state
                .split_models
                .iter()
                .any(|e| e.key().0 == m.id)
        } else {
            // No local shards on disk — can't be "loaded" even if split_models
            // has a stale entry from a previous load
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

        let probed = state.shared_state.hf_sources.contains_key(&m.id)
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
                "is_lan_peer": peer.is_lan_peer,
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

            let locked = state.shared_state.locked_shards.contains_key(&shard_id);
            let mut shard_json = serde_json::json!({
                "index": shard.index,
                "size_bytes": shard.size_bytes,
                "local": is_local,
                "holders": holders.len(),
                "locked": locked,
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

// ---- HuggingFace Endpoints ----

/// GET /api/admin/hf/search?q=... — Search HuggingFace for GGUF models.
pub async fn hf_search(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<HfSearchParams>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let query = params.query.unwrap_or_default();
    if query.is_empty() {
        return Ok(Json(vec![]));
    }

    let results = crate::model::huggingface::search_gguf_models(&query)
        .await
        .map_err(|e| ApiError(crate::error::SwarmError::Internal(e)))?;

    // Available VRAM for fits_vram check (pool VRAM or local GPU)
    let available_vram_bytes: u64 = state
        .shared_state
        .gpu_info
        .as_ref()
        .map(|g| g.vram_free_mb * 1024 * 1024)
        .unwrap_or(0);

    // Group results by repo_id with quant variants (preserve HF API order = by downloads)
    let mut repo_order: Vec<String> = Vec::new();
    let mut repo_map: std::collections::HashMap<
        String,
        Vec<crate::model::huggingface::HfModelResult>,
    > = std::collections::HashMap::new();
    for r in results {
        if !repo_map.contains_key(&r.repo_id) {
            repo_order.push(r.repo_id.clone());
        }
        repo_map.entry(r.repo_id.clone()).or_default().push(r);
    }

    let values: Vec<serde_json::Value> = repo_order
        .into_iter()
        .filter_map(|repo_id| {
            let files = repo_map.remove(&repo_id)?;
            Some((repo_id, files))
        })
        .map(|(repo_id, files)| {
            let downloads = files.first().map(|f| f.downloads).unwrap_or(0);
            let likes = files.first().map(|f| f.likes).unwrap_or(0);

            let variants: Vec<serde_json::Value> = files
                .iter()
                .map(|f| {
                    let quant = crate::model::huggingface::extract_quant_tag(&f.filename)
                        .unwrap_or_else(|| "unknown".into());
                    serde_json::json!({
                        "filename": f.filename,
                        "size_bytes": f.size_bytes,
                        "quant": quant,
                    })
                })
                .collect();

            // Recommended variant: prefer Q4_K_M, else smallest Q4+, else first
            let recommended = files
                .iter()
                .find(|f| {
                    crate::model::huggingface::extract_quant_tag(&f.filename)
                        .is_some_and(|q| q == "Q4_K_M")
                })
                .or_else(|| {
                    files
                        .iter()
                        .filter(|f| {
                            crate::model::huggingface::extract_quant_tag(&f.filename)
                                .is_some_and(|q| q.starts_with("Q4"))
                        })
                        .min_by_key(|f| f.size_bytes)
                })
                .or(files.first());

            let recommended_variant = recommended
                .and_then(|f| crate::model::huggingface::extract_quant_tag(&f.filename))
                .unwrap_or_else(|| "unknown".into());

            // fits_vram: check if smallest variant fits
            let smallest_size = files.iter().map(|f| f.size_bytes).min().unwrap_or(u64::MAX);
            let fits_vram = available_vram_bytes > 0 && smallest_size < available_vram_bytes;

            serde_json::json!({
                "repo_id": repo_id,
                "downloads": downloads,
                "likes": likes,
                "variants": variants,
                "recommended_variant": recommended_variant,
                "fits_vram": fits_vram,
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

    // Sanitize repo_id to prevent path traversal — reject ".." components
    let sanitized_repo = repo_id.replace('/', "_");
    if sanitized_repo.contains("..") || sanitized_repo.starts_with('.') {
        return Err(ApiError(crate::error::SwarmError::Config(
            "Invalid repo_id: path traversal detected".into(),
        )));
    }

    // Sanitize filename to prevent path traversal
    if filename.contains("..")
        || filename.starts_with('.')
        || filename.contains('/')
        || filename.contains('\\')
    {
        return Err(ApiError(crate::error::SwarmError::Config(
            "Invalid filename: path traversal detected".into(),
        )));
    }

    let dest_dir = state
        .config
        .node
        .data_dir
        .join("models")
        .join(&sanitized_repo);

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

    // Register cancellation flag for this download
    let hf_cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    shared
        .download_cancel_flags
        .insert(mid.clone(), hf_cancel_flag.clone());

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

        // Clean up cancel flag
        download_shared.download_cancel_flags.remove(&download_mid);

        // Clean up acquisition_progress after a delay so the frontend sees
        // the final state and triggers a re-render before we remove it.
        let cleanup_shared = download_shared.clone();
        let cleanup_mid = download_mid.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            cleanup_shared.acquisition_progress.remove(&cleanup_mid);
        });
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

    // Signal all subsystems to shut down via the watch channel.
    // The daemon.rs supervisor loop will handle graceful draining,
    // peer cache saving, DB flushing, and process exit.
    state.shared_state.shutdown();

    Ok(Json(serde_json::json!({ "status": "shutting_down" })))
}

#[derive(Debug, Deserialize)]
pub struct HfSearchParams {
    pub query: Option<String>,
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
    /// When true AND `shards` is empty: compute a deterministic fair share of shards
    /// based on the node's identity and peer count. Each node claims `ceil(shard_count / (peers + 1))`
    /// shards, with assignment determined by BLAKE3(node_id || model_id) for consistency.
    /// Peers with auto-manage enabled will auto-acquire the remaining shards.
    #[serde(default)]
    pub peer_fair_share: bool,
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
    match crate::model::huggingface::probe_gguf_file(&repo_id, &filename, shard_size).await {
        Ok(info) => {
            // Cache probe result so the frontend can look up HF source later
            let model_id_str = filename
                .trim_end_matches(".gguf")
                .to_lowercase()
                .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
                .split('-')
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("-");
            let mid = crate::types::ModelId(model_id_str);
            let probe_info = crate::daemon::HfProbeInfo {
                repo_id: repo_id.clone(),
                filename: filename.clone(),
                shard_count: info.shard_count(),
                total_size_bytes: info.total_size,
                probed_at: chrono::Utc::now(),
            };
            state.shared_state.hf_probe_cache.insert(mid, probe_info);

            let arch_str = &info.tensor_meta.architecture;
            let model_arch = crate::inference::split::ModelArch::from_gguf_arch(arch_str);
            Ok(Json(serde_json::json!({
                "status": "ok",
                "total_size": info.total_size,
                "header_size": info.header_size,
                "shard_count": info.shard_count(),
                "architecture": arch_str,
                "architecture_supported": model_arch.is_supported(),
            })))
        }
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
    let peer_fair_share = body.peer_fair_share;

    if repo_id.is_empty() || filename.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Config(
            "repo_id and filename are required".into(),
        )));
    }

    if shard_indices.is_empty() && !peer_fair_share {
        return Err(ApiError(crate::error::SwarmError::Config(
            "shards array is required (e.g. [0, 1, 2])".into(),
        )));
    }

    if shard_indices.len() > 256 {
        return Err(ApiError(crate::error::SwarmError::Config(
            "Too many shards requested (max 256)".into(),
        )));
    }

    tracing::info!(
        repo_id = %repo_id,
        filename = %filename,
        shard_count = shard_indices.len(),
        peer_fair_share,
        "DIAG: hf_download_shards handler"
    );

    // peer_fair_share: compute shard assignment deterministically.
    // Deferred until after probe (we need shard_count), so store the peer count now.
    let fair_share_peer_count = if peer_fair_share && shard_indices.is_empty() {
        Some(state.shared_state.peer_registry.len())
    } else {
        None
    };
    let fair_share_node_id = state.shared_state.identity.node_id().clone();

    // Use provided model_id if it matches an existing model, otherwise derive from filename.
    // Always sanitize to prevent path traversal.
    let safe_name = if let Some(ref mid) = body.model_id {
        let sanitized = crate::model::shard::sanitize_path_component(mid);
        let candidate_dir = state.config.node.data_dir.join("models").join(&sanitized);
        if candidate_dir.exists() {
            sanitized
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

    // ── Synchronous probe + architecture check ──────────────────────────
    // Probe before spawning the download task so we can return an immediate
    // HTTP error for unsupported architectures (fast: reads ~few KB header).
    let configured_shard_size = state.shared_state.config.model.shard_size_bytes();
    let info = crate::model::huggingface::probe_gguf_file(
        &repo_id,
        &filename,
        configured_shard_size,
    )
    .await
    .map_err(|e| {
        tracing::error!(error = %e, "HuggingFace probe failed");
        ApiError(crate::error::SwarmError::Internal(format!(
            "HuggingFace probe failed: {e}"
        )))
    })?;

    let arch_str = &info.tensor_meta.architecture;
    let model_arch = crate::inference::split::ModelArch::from_gguf_arch(arch_str);
    if !model_arch.is_supported() {
        let msg = format!(
            "Unsupported architecture '{}'. Supported: {}",
            arch_str,
            crate::inference::split::ModelArch::supported_list().join(", ")
        );
        tracing::warn!(%arch_str, "Refusing download: unsupported architecture");
        return Err(ApiError(crate::error::SwarmError::Internal(msg)));
    }

    // Create initial acquisition progress entry with per-shard progress so that
    // auto-manage can detect these downloads are already in flight and skip them.
    let log_msg = if peer_fair_share && shard_indices.is_empty() {
        format!("Computing fair share of {} from HuggingFace...", filename)
    } else {
        format!(
            "Downloading shards {:?} of {} from HuggingFace...",
            shard_indices, filename
        )
    };
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
        log: vec![log_msg],
    };
    let shared = state.shared_state.clone();
    shared.acquisition_progress.insert(mid.clone(), status);

    // Clone values needed both in the spawn and the response
    let response_model_id = model_id_str.clone();
    let response_shards = shard_indices.clone();

    // Register cancellation flag for this download
    let cancel_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    shared
        .download_cancel_flags
        .insert(mid.clone(), cancel_flag.clone());

    // Capture network_tx for broadcasting HfSourceGossip + ModelManifest after download
    let network_tx = state.network_tx.clone();

    tokio::spawn(async move {
        let download_mid = mid.clone();
        let download_shared = shared.clone();

        // peer_fair_share: download just ONE seed shard. Auto-manage handles the rest.
        // Each node picks a deterministic shard (based on node_id hash) so that
        // different nodes seed different shards when they add the same model.
        let shard_indices = if let Some(peer_count) = fair_share_peer_count {
            let total_shards = info.shard_count() as u32;

            // Deterministic shard selection: hash(node_id || model_id) → shard index
            let mut hasher = blake3::Hasher::new();
            hasher.update(fair_share_node_id.0.as_ref());
            hasher.update(model_id_str.as_bytes());
            let hash = hasher.finalize();
            let seed_shard = u32::from_le_bytes([
                hash.as_bytes()[0],
                hash.as_bytes()[1],
                hash.as_bytes()[2],
                hash.as_bytes()[3],
            ]) % total_shards;

            let assigned = vec![seed_shard];

            tracing::info!(
                total_shards,
                peers = peer_count,
                seed_shard,
                "peer_fair_share: seeding 1 shard (auto-manage will acquire more as needed)"
            );

            // Update acquisition progress with the single seed shard
            if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                entry.total_shards = 1;
                entry.log.push(format!(
                    "Seeding shard {seed_shard}/{total_shards} — auto-manage will acquire more as peers join"
                ));
                entry.shard_progress.insert(
                    seed_shard,
                    crate::model::acquisition::ShardProgress {
                        index: seed_shard,
                        total_bytes: 0,
                        downloaded_bytes: 0,
                        state: crate::model::acquisition::ShardState::Downloading,
                    },
                );
            }
            assigned
        } else {
            shard_indices
        };

        if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
            // Set total_bytes to the sum of requested shards only (not full model size)
            let requested_bytes: u64 = shard_indices
                .iter()
                .filter_map(|&idx| info.layouts.get(idx as usize))
                .map(|l| l.size_bytes)
                .sum();
            entry.total_bytes = requested_bytes;
            // Don't overwrite total_shards — keep as the requested count, not the full model count
            entry.log.push(format!(
                "Probed: {} shards, {:.1} MB total",
                info.shard_count(),
                info.total_size as f64 / (1024.0 * 1024.0)
            ));
            // Set per-shard total_bytes now that we know sizes from the probe
            for &idx in &shard_indices {
                if let Some(layout) = info.layouts.get(idx as usize) {
                    if let Some(sp) = entry.shard_progress.get_mut(&idx) {
                        sp.total_bytes = layout.size_bytes;
                    }
                }
            }
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
                entry.state =
                    crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                entry.log.push(format!("Header download failed: {}", e));
            }
            return;
        }

        // Download tied output weight if model is weight-tied (no output.weight tensor).
        // This is needed by the last node in distributed inference for logit projection.
        if let Err(e) = crate::model::huggingface::download_tied_output_weight(
            &repo_id,
            &filename,
            &dest_dir,
            &info.tensor_meta,
        )
        .await
        {
            tracing::warn!(error = %e, "Tied output weight download failed (non-fatal)");
        }

        // Generate manifest from header BEFORE downloading shard data.
        // Pass empty shard_indices — no shards to register yet (they don't exist on disk).
        let header_path = dest_dir.join("gguf_header.bin");
        let manifest_result = generate_manifest_from_header(&ManifestGenParams {
            header_path: &header_path,
            model_id_str: &model_id_str,
            filename: &filename,
            total_size: info.total_size,
            shard_count: info.shard_count(),
            shard_indices: &[],
            shared: &download_shared,
            precomputed_layouts: Some(&info.layouts),
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
            mmproj_filename: None,
        };
        download_shared.hf_sources.insert(
            crate::types::ModelId(model_id_str.clone()),
            hf_source.clone(),
        );
        let _ = download_shared
            .db
            .put_json("hf_sources", &model_id_str, &hf_source);
        let hf_source_path = dest_dir.join("hf_source.json");
        let _ = std::fs::write(
            &hf_source_path,
            serde_json::to_string_pretty(&hf_source).unwrap_or_default(),
        );

        // Broadcast HfSourceGossip + ModelManifest EARLY so peers can start
        // auto-acquiring shards immediately (before our shard data downloads finish).
        if let Some(ref ntx) = network_tx {
            let gossip_msg =
                crate::types::SwarmMessage::HfSourceGossip(crate::types::HfSourceGossip {
                    model_id: crate::types::ModelId(model_id_str.clone()),
                    repo_id: repo_id.clone(),
                    filename: filename.clone(),
                    publisher: download_shared.identity.node_id().clone(),
                    mmproj_filename: None,
                });
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

            // Broadcast download intent for each shard so peers know we're
            // working on them and auto-manage won't duplicate the download.
            let our_node_id = download_shared.identity.node_id().clone();
            for &idx in &shard_indices {
                let intent_msg = crate::types::SwarmMessage::ShardDownloadProgress(
                    crate::types::ShardDownloadProgress {
                        node_id: our_node_id.clone(),
                        shard_id: crate::types::ShardId {
                            model_id: crate::types::ModelId(model_id_str.clone()),
                            index: idx,
                        },
                        progress_pct: 0,
                        state: crate::types::DownloadState::Downloading,
                    },
                );
                let _ = ntx
                    .send(crate::types::NetworkCommand::Broadcast(intent_msg))
                    .await;
            }
        }

        // NOTE: Do NOT wake auto-manage here. Shards aren't downloaded yet,
        // so holder_count == 0 and auto-manage would race to download them
        // from HF. The notify happens AFTER downloads complete (line ~1666).

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

        // Download individual v2 layer-aligned shards
        let total_shard_bytes: u64 = shard_indices
            .iter()
            .filter_map(|&idx| info.layouts.get(idx as usize))
            .map(|layout| layout.size_bytes)
            .sum();

        let mut cumulative_downloaded: u64 = 0;
        let mut failed = false;

        for &shard_idx in &shard_indices {
            // Check cancellation flag before each shard download
            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                tracing::info!(model = %model_id_str, "Download cancelled by user");
                if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid)
                {
                    entry.state = crate::model::acquisition::AcquisitionState::Failed {
                        reason: "Cancelled by user".to_string(),
                    };
                    entry.log.push("Download cancelled by user".to_string());
                }
                // Clean up cancel flag
                download_shared.download_cancel_flags.remove(&download_mid);
                return;
            }

            let layout = match info.layouts.get(shard_idx as usize) {
                Some(l) => l,
                None => {
                    tracing::error!(
                        shard_idx,
                        max = info.layouts.len().saturating_sub(1),
                        "Shard index out of range"
                    );
                    failed = true;
                    break;
                }
            };

            let (shard_tx, mut shard_rx) =
                tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);
            let progress_tx_clone = ptx.clone();
            let base_downloaded = cumulative_downloaded;
            let total = total_shard_bytes;
            let shard_progress_shared = shared.clone();
            let shard_progress_mid = mid.clone();
            let gossip_ntx = network_tx.clone();
            let gossip_node_id = shared.identity.node_id().clone();
            let gossip_model_id = model_id_str.clone();
            let progress_task = tokio::spawn(async move {
                let mut last_broadcast_pct: u32 = 0;
                while let Some(prog) = shard_rx.recv().await {
                    // Forward cumulative bytes to the overall progress updater
                    let _ =
                        progress_tx_clone.try_send(crate::model::huggingface::DownloadProgress {
                            downloaded_bytes: base_downloaded + prog.downloaded_bytes,
                            total_bytes: total,
                        });
                    // Update per-shard progress directly
                    if let Some(mut entry) = shard_progress_shared
                        .acquisition_progress
                        .get_mut(&shard_progress_mid)
                    {
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                            sp.downloaded_bytes = prog.downloaded_bytes;
                            if sp.total_bytes == 0 {
                                sp.total_bytes = prog.total_bytes;
                            }
                        }
                    }
                    // Broadcast progress to peers every ~2% so they see near real-time updates
                    let pct = if prog.total_bytes > 0 {
                        ((prog.downloaded_bytes as f64 / prog.total_bytes as f64) * 100.0) as u32
                    } else {
                        0
                    };
                    if pct >= last_broadcast_pct + 2 {
                        last_broadcast_pct = pct;
                        if let Some(ref ntx) = gossip_ntx {
                            let msg = crate::types::SwarmMessage::ShardDownloadProgress(
                                crate::types::ShardDownloadProgress {
                                    node_id: gossip_node_id.clone(),
                                    shard_id: crate::types::ShardId {
                                        model_id: crate::types::ModelId(gossip_model_id.clone()),
                                        index: shard_idx,
                                    },
                                    progress_pct: pct,
                                    state: crate::types::DownloadState::Downloading,
                                },
                            );
                            let _ = ntx.send(crate::types::NetworkCommand::Broadcast(msg)).await;
                        }
                    }
                }
            });

            match crate::model::huggingface::download_shard_v2(
                &repo_id,
                &filename,
                &dest_dir,
                layout,
                Some(shard_tx),
            )
            .await
            {
                Ok(_shard_path) => {
                    progress_task.abort();
                    cumulative_downloaded += layout.size_bytes;

                    if let Some(mut entry) =
                        download_shared.acquisition_progress.get_mut(&download_mid)
                    {
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
                    if let Some(mut entry) =
                        download_shared.acquisition_progress.get_mut(&download_mid)
                    {
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

        // Clean up cancel flag
        download_shared.download_cancel_flags.remove(&download_mid);

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
                shard_count: info.shard_count(),
                shard_indices: &shard_indices,
                shared: &download_shared,
                precomputed_layouts: Some(&info.layouts),
            });

            if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid) {
                entry.state = crate::model::acquisition::AcquisitionState::Complete;
                entry.verified_shards = shard_indices.len() as u32;
                entry
                    .log
                    .push("All shards downloaded and registered".to_string());
            }

            // Load available shards for inference (partial is fine)
            let vram_budget = crate::model::auto_manage::compute_vram_budget(&download_shared);
            crate::model::auto_manage::check_and_load_model(
                &download_shared,
                &crate::types::ModelId(model_id_str.clone()),
                vram_budget,
            )
            .await;

            // Notify dashboard that models have changed
            let _ = download_shared.models_changed_tx.send(());

            // Wake auto-manage again to re-evaluate (maybe download more shards)
            download_shared.auto_manage_notify.notify_one();

            // Clean up acquisition_progress after a delay so the frontend sees
            // the "complete" state and triggers a re-render before we remove it.
            let cleanup_shared = download_shared.clone();
            let cleanup_mid = download_mid.clone();
            tokio::spawn(async move {
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
                cleanup_shared.acquisition_progress.remove(&cleanup_mid);
            });
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "model_id": response_model_id,
        "shards": response_shards,
        "peer_fair_share": peer_fair_share,
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
    /// Pre-computed layouts from probe. When provided, these are used directly
    /// instead of recomputing (avoids shard_count mismatch between probe and manifest).
    precomputed_layouts: Option<&'a [crate::inference::split::LayerShardLayout]>,
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
            "qwen2" | "qwen3" | "qwen2moe" => crate::types::ModelArchitecture::Qwen2,
            "mistral" => crate::types::ModelArchitecture::Mistral,
            "phi" | "phi3" => crate::types::ModelArchitecture::Phi,
            _ => crate::types::ModelArchitecture::Llama,
        }
    };

    let model_dir = header_path
        .parent()
        .ok_or_else(|| "GGUF header path has no parent directory".to_string())?;

    let computed_layouts;
    let layouts: &[crate::inference::split::LayerShardLayout] = if let Some(precomputed) =
        params.precomputed_layouts
    {
        precomputed
    } else {
        computed_layouts = crate::inference::split::compute_layer_shard_layouts(&meta, shard_count);
        &computed_layouts
    };
    let shards = crate::model::manifest::build_shard_infos_from_layouts(model_dir, layouts);

    let node_id = params.shared.identity.node_id().clone();

    let mut manifest = crate::types::ModelManifest {
        schema_version: 2,
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
        mmproj: None,
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
            .record_shard_holder(shard_id, node_id.clone());
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

    // Always include our own node on the map.
    // Use auto-detected region (IP geolocation), configured region, or "??" as fallback.
    {
        let detected = state.shared_state.detected_region.read().await;
        let code = detected.as_deref().unwrap_or("??").to_uppercase();
        let entry = regions.entry(code).or_insert_with(|| (0, HashMap::new()));
        entry.0 += 1;
        // Add our hosted models
        let node_id = state.shared_state.identity.node_id();
        for (shard_id, holders) in state.shared_state.model_registry.all_shard_entries() {
            if holders.contains(node_id) {
                *entry.1.entry(shard_id.model_id.0.clone()).or_insert(0) += 1;
            }
        }
    }

    // Aggregate peer regions from capabilities.
    // Peers without capability/region info are placed in our own region (most peers
    // on a LAN share the same region) or "??" as fallback.
    let self_region = {
        let detected = state.shared_state.detected_region.read().await;
        detected.as_deref().unwrap_or("??").to_uppercase()
    };
    for peer in state.shared_state.peer_registry.iter() {
        let (region_code, hosted_shards) = match peer.value().capability {
            Some(ref cap) => {
                let code = cap.region.as_deref().unwrap_or(&self_region).to_uppercase();
                (code, &cap.hosted_shards[..])
            }
            None => (self_region.clone(), &[][..]),
        };
        let entry = regions
            .entry(region_code)
            .or_insert_with(|| (0, HashMap::new()));
        entry.0 += 1;
        // Count distinct models this peer hosts
        let mut peer_models = std::collections::HashSet::new();
        for shard in hosted_shards {
            peer_models.insert(shard.model_id.0.clone());
        }
        for model_id in peer_models {
            *entry.1.entry(model_id).or_insert(0) += 1;
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

/// POST /api/admin/downloads/:model_id/cancel — Cancel an in-progress HF download.
///
/// Sets the cancellation flag so the download loop aborts. Cleans up partial .tmp files.
/// Returns 200 on success, 404 if no active download for that model.
pub async fn cancel_download(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());
    let shared = &state.shared_state;

    // Check if there's an active download for this model
    let has_active = shared
        .acquisition_progress
        .get(&mid)
        .map(|entry| {
            matches!(
                entry.state,
                crate::model::acquisition::AcquisitionState::Downloading
                    | crate::model::acquisition::AcquisitionState::AwaitingManifest
            )
        })
        .unwrap_or(false);

    if !has_active {
        return Err(ApiError(crate::error::SwarmError::Config(format!(
            "No active download found for model '{}'",
            model_id
        ))));
    }

    // Set the cancel flag (the download loop checks this)
    if let Some(flag) = shared.download_cancel_flags.get(&mid) {
        flag.store(true, std::sync::atomic::Ordering::Release);
    }

    // Mark the acquisition as failed/cancelled
    if let Some(mut entry) = shared.acquisition_progress.get_mut(&mid) {
        entry.state = crate::model::acquisition::AcquisitionState::Failed {
            reason: "Cancelled by user".to_string(),
        };
        entry.log.push("Download cancelled by user".to_string());
    }

    // Clean up partial .tmp files in the model directory
    let safe_id = crate::model::shard::sanitize_path_component(&model_id);
    let model_dir = state.config.node.data_dir.join("models").join(&safe_id);
    if model_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
                    tracing::info!(path = %path.display(), "Removing partial download file");
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
    }

    tracing::info!(model = %model_id, "Download cancelled");

    Ok(Json(serde_json::json!({
        "status": "cancelled",
        "model_id": model_id,
    })))
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

    // Remove manifest from sled DB
    let _ = shared.db.remove("model_meta", &model_id);

    // Remove HF source from DB
    let _ = shared.db.remove("hf_sources", &model_id);

    // Remove from SharedState registries
    shared.model_registry.remove_manifest(&mid);
    shared.model_registry.remove_all_model_shards(&mid);

    // Remove from acquisition_progress
    shared.acquisition_progress.remove(&mid);

    // Remove from gguf_meta
    shared.gguf_meta.remove(&mid);

    // Remove from hf_sources
    shared.hf_sources.remove(&mid);

    // Remove split models for this model
    shared.split_models.retain(|key, _| key.0 != mid);

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

    // Remove split models for this model (frees VRAM/memory)
    let mut segments_removed = 0u32;
    shared.split_models.retain(|key, _| {
        if key.0 == mid {
            segments_removed += 1;
            false
        } else {
            true
        }
    });

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

    tracing::info!(model = %model_id, segments = segments_removed, "Model unloaded from memory");

    Ok(Json(serde_json::json!({
        "status": "unloaded",
        "model_id": model_id,
        "segments_removed": segments_removed,
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

    // Evict any cached split model segments that included this shard
    shared.split_models.retain(|key, _| key.0 != mid);

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

    tracing::info!(model = %model_id, shard = shard_index, "Shard removed");

    Ok(Json(serde_json::json!({
        "status": "deleted",
        "model_id": model_id,
        "shard_index": shard_index,
    })))
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

/// GET /api/admin/hf/source/:model_id — Look up HuggingFace source for a model.
///
/// Returns the repo_id and filename needed to trigger per-shard downloads.
/// Checks both hf_sources (downloaded models) and hf_probe_cache (probed models).
pub async fn hf_source(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());

    if let Some(src) = state.shared_state.hf_sources.get(&mid) {
        return Ok(Json(serde_json::json!({
            "model_id": model_id,
            "repo_id": src.repo_id,
            "filename": src.filename,
        })));
    }

    if let Some(probe) = state.shared_state.hf_probe_cache.get(&mid) {
        return Ok(Json(serde_json::json!({
            "model_id": model_id,
            "repo_id": probe.repo_id,
            "filename": probe.filename,
        })));
    }

    Err(ApiError(crate::error::SwarmError::Config(format!(
        "No HuggingFace source found for model '{}'",
        model_id
    ))))
}

/// GET /api/admin/network-code — Return this node's network invite code.
///
/// Returns a shareable invite code that other nodes can use to connect.
/// The code encodes the node's QUIC listening address.
pub async fn network_code(State(state): State<AppState>) -> Json<serde_json::Value> {
    let port = state.config.node.listen_port;
    let peer_count = state.shared_state.peer_registry.len();

    // Build the QUIC listen address with the node's peer ID
    let signing_key_bytes = state.shared_state.identity.signing_key_bytes();
    let peer_id_str = match crate::network::transport::ed25519_to_libp2p_keypair(signing_key_bytes)
    {
        Ok(kp) => kp.public().to_peer_id().to_string(),
        Err(_) => {
            return Json(serde_json::json!({
                "error": "Failed to derive peer ID"
            }))
        }
    };

    // Pick a real IP by scanning peer addresses that other nodes see for us,
    // or fall back to detecting the local machine's non-loopback IP.
    let best_ip = {
        // Try to find a non-loopback IP from peers' addresses for our node
        let mut found_ip = None;
        for peer in state.shared_state.peer_registry.iter() {
            for addr in &peer.addresses {
                if addr.starts_with("/ip4/") {
                    let parts: Vec<&str> = addr.split('/').collect();
                    if parts.len() >= 3 {
                        let ip = parts[2];
                        if ip != "127.0.0.1" && ip != "0.0.0.0" && ip != "10.255.255.254" {
                            found_ip = Some(ip.to_string());
                            break;
                        }
                    }
                }
            }
            if found_ip.is_some() {
                break;
            }
        }
        found_ip.unwrap_or_else(|| {
            // Fall back: try to detect local non-loopback IP via UDP socket trick
            std::net::UdpSocket::bind("0.0.0.0:0")
                .and_then(|s| {
                    s.connect("8.8.8.8:80")?;
                    s.local_addr()
                })
                .map(|a| a.ip().to_string())
                .unwrap_or_else(|_| "127.0.0.1".to_string())
        })
    };

    let multiaddr_str = format!("/ip4/{best_ip}/udp/{port}/quic-v1/p2p/{peer_id_str}");
    let code = if let Ok(addr) = multiaddr_str.parse::<libp2p::Multiaddr>() {
        crate::network::discovery::encode_network_code(&addr)
    } else {
        multiaddr_str.clone()
    };

    // Determine visibility phase
    let phase = if peer_count == 0 {
        "seedling"
    } else if peer_count < 20 {
        "growing"
    } else {
        "established"
    };

    Json(serde_json::json!({
        "code": code,
        "multiaddr": multiaddr_str,
        "node_id": format!("{}", state.shared_state.identity.node_id()),
        "peer_id": peer_id_str,
        "port": port,
        "phase": phase,
        "peer_count": peer_count,
    }))
}

/// POST /api/admin/join-network — Join the network using an invite code.
///
/// Accepts a network invite code (swarm://...) or raw multiaddr and dials the peer.
pub async fn join_network(
    State(state): State<AppState>,
    Json(body): Json<JoinNetworkRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let addr_str = crate::network::discovery::decode_network_code(&body.code).map_err(ApiError)?;

    // Send the address to the network manager to dial
    if state.network_tx.is_some() {
        tracing::info!(addr = %addr_str, "Joining network via invite code");

        // Parse the multiaddr to validate
        let _addr: libp2p::Multiaddr =
            addr_str.parse().map_err(|e: libp2p::multiaddr::Error| {
                ApiError(crate::error::SwarmError::Network(format!(
                    "Invalid address: {e}"
                )))
            })?;

        // Save to peer cache so it persists across restarts
        let mut cached = crate::network::peer_cache::load_peer_cache(&state.shared_state.db);
        if !cached.contains(&addr_str) {
            cached.push(addr_str.clone());
            crate::network::peer_cache::save_peer_cache(&state.shared_state.db, &cached);
        }

        Ok(Json(serde_json::json!({
            "status": "ok",
            "address": addr_str,
            "message": "Peer address saved. Restart the node or wait for the next discovery cycle to connect."
        })))
    } else {
        Err(ApiError(crate::error::SwarmError::Network(
            "Network manager not available".to_string(),
        )))
    }
}

#[derive(Deserialize)]
pub struct JoinNetworkRequest {
    pub code: String,
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

// ---- Resource Schedule API ----

/// GET /api/admin/schedule — Get current resource schedule.
pub async fn get_schedule(State(state): State<AppState>) -> Json<serde_json::Value> {
    let schedule = state.shared_state.resource_schedule.read().await;
    Json(serde_json::json!({
        "enabled": schedule.enabled,
        "reduced_hours_start": schedule.reduced_hours_start,
        "reduced_hours_end": schedule.reduced_hours_end,
        "reduced_contribution": schedule.reduced_contribution,
        "prune_aggressiveness": schedule.prune_aggressiveness,
    }))
}

#[derive(Debug, Deserialize)]
pub struct ScheduleUpdate {
    pub enabled: Option<bool>,
    pub reduced_hours_start: Option<u32>,
    pub reduced_hours_end: Option<u32>,
    pub reduced_contribution: Option<String>,
    pub prune_aggressiveness: Option<String>,
}

/// PUT /api/admin/schedule — Update resource schedule at runtime (persisted to sled).
pub async fn update_schedule(
    State(state): State<AppState>,
    Json(body): Json<ScheduleUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut schedule = state.shared_state.resource_schedule.write().await;

    if let Some(enabled) = body.enabled {
        schedule.enabled = enabled;
    }
    if let Some(start) = body.reduced_hours_start {
        if start > 23 {
            return Err(ApiError(crate::error::SwarmError::Config(
                "reduced_hours_start must be 0-23".to_string(),
            )));
        }
        schedule.reduced_hours_start = start;
    }
    if let Some(end) = body.reduced_hours_end {
        if end > 23 {
            return Err(ApiError(crate::error::SwarmError::Config(
                "reduced_hours_end must be 0-23".to_string(),
            )));
        }
        schedule.reduced_hours_end = end;
    }
    if let Some(ref contribution) = body.reduced_contribution {
        schedule.reduced_contribution = contribution.clone();
    }
    if let Some(ref aggressiveness) = body.prune_aggressiveness {
        match aggressiveness.as_str() {
            "normal" | "aggressive" | "conservative" => {
                schedule.prune_aggressiveness = aggressiveness.clone();
            }
            _ => {
                return Err(ApiError(crate::error::SwarmError::Config(
                    "prune_aggressiveness must be 'normal', 'aggressive', or 'conservative'"
                        .to_string(),
                )));
            }
        }
    }

    // Persist to sled
    let _ = state
        .shared_state
        .db
        .put_json("resource_schedule", "current", &*schedule);

    tracing::debug!(
        enabled = schedule.enabled,
        prune_aggressiveness = %schedule.prune_aggressiveness,
        "DIAG: schedule updated"
    );

    let result = serde_json::json!({
        "status": "ok",
        "enabled": schedule.enabled,
        "reduced_hours_start": schedule.reduced_hours_start,
        "reduced_hours_end": schedule.reduced_hours_end,
        "reduced_contribution": schedule.reduced_contribution,
        "prune_aggressiveness": schedule.prune_aggressiveness,
    });

    Ok(Json(result))
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

/// GET /api/admin/providers — List configured provider status (no keys exposed).
pub async fn get_providers(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.shared_state.providers_config.read().await;

    let providers = vec![
        serde_json::json!({
            "name": "anthropic",
            "configured": config.anthropic.is_some(),
        }),
        serde_json::json!({
            "name": "openai",
            "configured": config.openai.is_some(),
        }),
        serde_json::json!({
            "name": "deepseek",
            "configured": config.deepseek.is_some(),
        }),
        serde_json::json!({
            "name": "mistral",
            "configured": config.mistral.is_some(),
        }),
        serde_json::json!({
            "name": "groq",
            "configured": config.groq.is_some(),
        }),
    ];

    Json(serde_json::json!({ "providers": providers }))
}

#[derive(Debug, Deserialize)]
pub struct ProvidersUpdate {
    #[serde(default)]
    pub anthropic_key: Option<String>,
    #[serde(default)]
    pub openai_key: Option<String>,
    #[serde(default)]
    pub deepseek_key: Option<String>,
    #[serde(default)]
    pub mistral_key: Option<String>,
    #[serde(default)]
    pub groq_key: Option<String>,
}

/// PUT /api/admin/providers — Update provider API keys. Empty string = remove key.
pub async fn update_providers(
    State(state): State<AppState>,
    Json(body): Json<ProvidersUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut config = state.shared_state.providers_config.write().await;

    fn update_entry(entry: &mut Option<crate::config::ProviderEntry>, key: Option<String>) {
        if let Some(k) = key {
            if k.is_empty() {
                *entry = None;
            } else {
                *entry = Some(crate::config::ProviderEntry {
                    api_key: k,
                    default_model: entry.as_ref().and_then(|e| e.default_model.clone()),
                });
            }
        }
    }

    update_entry(&mut config.anthropic, body.anthropic_key);
    update_entry(&mut config.openai, body.openai_key);
    update_entry(&mut config.deepseek, body.deepseek_key);
    update_entry(&mut config.mistral, body.mistral_key);
    update_entry(&mut config.groq, body.groq_key);

    // Persist to database
    let _ = state
        .shared_state
        .db
        .put_json("providers", "config", &*config);

    tracing::info!("Cloud provider configuration updated");

    // Notify WebSocket clients so model list and mode indicator refresh immediately
    let _ = state.shared_state.models_changed_tx.send(());

    Ok(Json(serde_json::json!({
        "status": "ok",
        "anthropic": config.anthropic.is_some(),
        "openai": config.openai.is_some(),
        "deepseek": config.deepseek.is_some(),
        "mistral": config.mistral.is_some(),
        "groq": config.groq.is_some(),
    })))
}

/// GET /api/admin/provider-models — List well-known models for configured providers.
///
/// Returns a flat list of popular models for each provider that has an API key configured.
/// The frontend uses this to populate the model selector with cloud models alongside local ones.
pub async fn list_provider_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.shared_state.providers_config.read().await;
    let mut models = Vec::new();

    if config.openai.is_some() {
        for (id, name) in [
            ("gpt-4o", "GPT-4o"),
            ("gpt-4o-mini", "GPT-4o Mini"),
            ("gpt-4.1", "GPT-4.1"),
            ("gpt-4.1-mini", "GPT-4.1 Mini"),
            ("gpt-4.1-nano", "GPT-4.1 Nano"),
            ("o3-mini", "o3 Mini"),
        ] {
            models.push(serde_json::json!({
                "id": id, "name": name, "provider": "openai"
            }));
        }
    }

    if config.anthropic.is_some() {
        for (id, name) in [
            ("claude-opus-4-6", "Claude Opus 4.6"),
            ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
            ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
        ] {
            models.push(serde_json::json!({
                "id": id, "name": name, "provider": "anthropic"
            }));
        }
    }

    if config.deepseek.is_some() {
        for (id, name) in [
            ("deepseek-chat", "DeepSeek Chat"),
            ("deepseek-reasoner", "DeepSeek Reasoner"),
        ] {
            models.push(serde_json::json!({
                "id": id, "name": name, "provider": "deepseek"
            }));
        }
    }

    if config.mistral.is_some() {
        for (id, name) in [
            ("mistral-large-latest", "Mistral Large"),
            ("mistral-small-latest", "Mistral Small"),
            ("codestral-latest", "Codestral"),
        ] {
            models.push(serde_json::json!({
                "id": id, "name": name, "provider": "mistral"
            }));
        }
    }

    if config.groq.is_some() {
        for (id, name) in [
            ("llama-3.3-70b-versatile", "Llama 3.3 70B"),
            ("llama-3.1-8b-instant", "Llama 3.1 8B"),
            ("gemma2-9b-it", "Gemma 2 9B"),
        ] {
            models.push(serde_json::json!({
                "id": id, "name": name, "provider": "groq"
            }));
        }
    }

    // Include custom providers with their default model if set
    for custom in &config.custom {
        if let Some(ref model) = custom.default_model {
            models.push(serde_json::json!({
                "id": format!("{}:{}", custom.name, model),
                "name": model,
                "provider": custom.name,
            }));
        }
    }

    Json(serde_json::json!({ "models": models }))
}

// ========================================================================
// Update / Version Endpoints
// ========================================================================

/// GET /api/admin/version — Current and latest version info.
pub async fn version_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let update_state = state.shared_state.update_state.read().await;
    let current_version = env!("CARGO_PKG_VERSION");

    let (latest_version, update_available, changelog) =
        if let Some(ref info) = update_state.update_available {
            (
                Some(info.latest_version.clone()),
                true,
                Some(info.changelog.clone()),
            )
        } else {
            (None, false, None)
        };

    let channel = match state.shared_state.config.updates.auto_update {
        crate::config::AutoUpdateMode::Disabled => "disabled",
        crate::config::AutoUpdateMode::Stable => "stable",
        crate::config::AutoUpdateMode::All => "all",
    };

    Json(serde_json::json!({
        "current_version": current_version,
        "latest_version": latest_version,
        "update_available": update_available,
        "channel": channel,
        "last_checked": update_state.last_checked,
        "last_error": update_state.last_error,
        "changelog": changelog,
    }))
}

/// POST /api/admin/update/check — Trigger an immediate update check.
pub async fn check_update(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config = state.shared_state.config.updates.clone();
    let update_state = state.shared_state.update_state.clone();
    let update_tx = state.shared_state.update_tx.clone();

    let checker = crate::update::UpdateChecker::new(
        config,
        "enapt/SwarmLLM".to_string(),
        update_state.clone(),
        update_tx,
    );

    match checker.check_for_update().await {
        Ok(Some(info)) => {
            // Auto-download
            let mut info = info;
            if let Ok(tmp_path) = checker.download_update(&info).await {
                info.downloaded = true;
                let _ = tmp_path; // path is known from binary location
            }
            let mut us = update_state.write().await;
            us.update_available = Some(info.clone());
            us.last_checked = Some(chrono::Utc::now().to_rfc3339());
            us.last_error = None;
            // Notify WebSocket
            let _ = state.shared_state.update_tx.send(info.clone());
            Ok(Json(serde_json::json!({
                "status": "update_available",
                "info": info,
            })))
        }
        Ok(None) => {
            let mut us = update_state.write().await;
            us.last_checked = Some(chrono::Utc::now().to_rfc3339());
            us.last_error = None;
            Ok(Json(serde_json::json!({
                "status": "up_to_date",
                "current_version": env!("CARGO_PKG_VERSION"),
            })))
        }
        Err(e) => {
            let mut us = update_state.write().await;
            us.last_checked = Some(chrono::Utc::now().to_rfc3339());
            us.last_error = Some(e.to_string());
            Err(ApiError(e))
        }
    }
}

/// POST /api/admin/update/apply — Apply a downloaded update (restart required).
pub async fn apply_update(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let update_state = state.shared_state.update_state.read().await;
    let info = match &update_state.update_available {
        Some(info) if info.downloaded => info.clone(),
        Some(_) => {
            return Err(ApiError(crate::error::SwarmError::Internal(
                "Update not yet downloaded — call POST /api/admin/update/check first".to_string(),
            )));
        }
        None => {
            return Err(ApiError(crate::error::SwarmError::Internal(
                "No update available".to_string(),
            )));
        }
    };
    drop(update_state);

    let config = state.shared_state.config.updates.clone();
    let (tx, _) = tokio::sync::broadcast::channel(1);
    let checker = crate::update::UpdateChecker::new(
        config,
        "enapt/SwarmLLM".to_string(),
        state.shared_state.update_state.clone(),
        tx,
    );

    let binary_path = std::env::current_exe().map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Cannot determine binary path: {e}"
        )))
    })?;
    let tmp_path = binary_path.with_extension("update.tmp");

    if !tmp_path.exists() {
        return Err(ApiError(crate::error::SwarmError::Internal(
            "Downloaded update file not found — re-run update check".to_string(),
        )));
    }

    checker.apply_update(&tmp_path).map_err(ApiError)?;

    Ok(Json(serde_json::json!({
        "status": "applied",
        "version": info.latest_version,
        "message": "Update applied. Restart the daemon to use the new version.",
    })))
}
