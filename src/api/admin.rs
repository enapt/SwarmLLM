use axum::extract::State;
use axum::Json;
use serde::Deserialize;

// Re-export sub-module handlers so server.rs routes continue to use `admin::handler_name`
pub use super::admin_hf::*;
pub use super::admin_models::*;
pub use super::admin_providers::*;

use crate::api::server::AppState;
use crate::config::ContributionMode;
use crate::error::ApiError;

/// GET /api/admin/stats — Full dashboard stats snapshot.
pub async fn stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let node_id = hex::encode(state.shared_state.identity.node_id().0);
    let stats = state.shared_state.metrics.node_stats.read().await;
    let credit = state.shared_state.credits.credit_balance.read().await;

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

    // Hardware detection — sysinfo does blocking filesystem reads (/proc/*)
    let ss = state.shared_state.clone();
    let hardware = tokio::task::spawn_blocking(move || detect_hardware(&ss))
        .await
        .unwrap_or_else(|_| serde_json::json!({}));

    // Inference performance metrics from latency samples
    let inference_perf = {
        let samples = state.shared_state.metrics.inference_latency_samples.read();
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
                    "total_requests": state.shared_state.metrics.inference_requests_total
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
                "total_requests": state.shared_state.metrics.inference_requests_total
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
        "peers": state.shared_state.peer_registry.len(),
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
        "auto_manage_shards": state.shared_state.models.auto_manage_enabled.load(std::sync::atomic::Ordering::Relaxed),
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
    // Note: most config changes take effect after daemon restart.
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
            .models
            .auto_manage_enabled
            .store(auto_manage, std::sync::atomic::Ordering::Release);
        if auto_manage {
            // Wake the AutoShardManager so it evaluates promptly
            state.shared_state.models.auto_manage_notify.notify_one();
        }
        state.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "system",
                "config_updated",
                format!(
                    "Auto-manage {}",
                    if auto_manage { "enabled" } else { "disabled" }
                ),
            )
            .with_toast("info", 4000),
        );
    }
    if let Some(max_storage) = body.auto_manage_max_storage_mb {
        config.auto_manage.max_storage_mb = max_storage;
    }
    if let Some(shard_size) = body.shard_size_mb {
        if !(crate::config::SHARD_SIZE_MIN_MB..=crate::config::SHARD_SIZE_MAX_MB)
            .contains(&shard_size)
        {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
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
        .map_err(|e| ApiError(crate::error::SwarmError::Validation(e.to_string())))?;

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

            let nickname = state
                .shared_state
                .nickname_registry
                .get(&peer.node_id)
                .map(|r| r.nickname.clone());

            serde_json::json!({
                "node_id": format!("{}", peer.node_id),
                "nickname": nickname,
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
    let credit = state.shared_state.credits.credit_balance.read().await;
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

/// POST /api/admin/shutdown — Gracefully shut down the node.
/// Only accepts requests from localhost (127.0.0.1 or ::1) for safety.
pub async fn shutdown_node(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !addr.ip().is_loopback() {
        return Err(ApiError(crate::error::SwarmError::Unauthorized(
            "Shutdown only allowed from localhost".into(),
        )));
    }
    tracing::info!(addr = %addr, "Shutdown requested via API");

    // Signal all subsystems to shut down via the watch channel.
    // The daemon.rs supervisor loop will handle graceful draining,
    // peer cache saving, DB flushing, and process exit.
    state.shared_state.shutdown();

    Ok(Json(serde_json::json!({ "status": "shutting_down" })))
}
// ---- Hardware detection ----

fn detect_hardware(shared_state: &crate::daemon::SharedState) -> serde_json::Value {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let used_ram_mb = sys.used_memory() / (1024 * 1024);

    // Per-process memory (RSS) — actual memory this node is using
    let process_rss_mb = {
        let pid = sysinfo::Pid::from_u32(std::process::id());
        sys.process(pid)
            .map(|p| p.memory() / (1024 * 1024))
            .unwrap_or(0)
    };

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
    let (gpu_name, gpu_vram_mb, gpu_vram_used_mb) = match &shared_state.gpu_info {
        Some(gpu) => {
            // Query live VRAM usage via nvidia-smi for an up-to-date reading
            let used = query_gpu_vram_used();
            (
                Some(gpu.name.clone()),
                Some(gpu.vram_total_mb),
                used.or(Some(gpu.vram_total_mb.saturating_sub(gpu.vram_free_mb))),
            )
        }
        None => {
            let (name, total) = detect_gpu_nvidia_smi();
            let used = query_gpu_vram_used();
            (name, total, used)
        }
    };

    // gpu_inference: true only when llama.cpp actually bound to the GPU device
    let gpu_inference = shared_state.gpu_info.is_some();
    let inference_backend = shared_state.gpu_info.as_ref().map(|g| g.backend.clone());

    let (memory_bandwidth_gbps, est_tokens_per_sec_7b) = match &shared_state.gpu_info {
        Some(gpu) => {
            let bw = crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&gpu.name);
            let tps = crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(bw, true);
            (Some(bw), Some(tps))
        }
        None => (None, None),
    };

    serde_json::json!({
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "gpu_vram_used_mb": gpu_vram_used_mb,
        "gpu_inference": gpu_inference,
        "inference_backend": inference_backend,
        "memory_bandwidth_gbps": memory_bandwidth_gbps,
        "est_tokens_per_sec_7b": est_tokens_per_sec_7b,
        "total_ram_mb": total_ram_mb,
        "used_ram_mb": used_ram_mb,
        "process_rss_mb": process_rss_mb,
        "available_disk_mb": available_disk_mb,
        "total_disk_mb": total_disk_mb,
        "used_disk_mb": used_disk_mb,
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
    })
}

/// Fallback GPU detection via nvidia-smi when llama.cpp gpu_info is unavailable.
pub(crate) fn detect_gpu_nvidia_smi() -> (Option<String>, Option<u64>) {
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

/// Query current GPU VRAM usage via nvidia-smi (memory.used in MB).
fn query_gpu_vram_used() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().ok()
}

/// GET /api/admin/network-map — Aggregated region data for the world heatmap.
///
/// POST /api/admin/rescan-shards — Scan the models directory for new shard files.
///
/// Discovers shard files that were added to disk since the last scan (e.g. by
/// manual copy), registers them in the model registry, reloads affected models,
/// and re-announces shards to the network. No restart needed.
pub async fn rescan_shards(State(state): State<AppState>) -> Json<serde_json::Value> {
    let network_tx = state.network_tx.clone();
    let changed =
        crate::model::auto_manage::rescan_local_shards(&state.shared_state, network_tx.as_ref())
            .await;
    Json(serde_json::json!({
        "status": "ok",
        "models_updated": changed.iter().map(|m| &m.0).collect::<Vec<_>>(),
        "count": changed.len(),
    }))
}

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

    // Collect all known model IDs for coverage gap detection
    let all_models: Vec<String> = state
        .shared_state
        .model_registry
        .models()
        .iter()
        .map(|m| m.id.0.clone())
        .collect();

    let pool_size = state.shared_state.peer_registry.len() + 1;
    let min_replicas = state.shared_state.config.auto_manage.min_replicas as usize;

    // Build JSON with regional demand, coverage gaps, and replication targets
    let region_json: serde_json::Map<String, serde_json::Value> = regions
        .into_iter()
        .map(|(code, (total, models))| {
            let models_json: serde_json::Map<String, serde_json::Value> = models
                .into_iter()
                .map(|(k, v)| (k, serde_json::json!(v)))
                .collect();

            // Per-model demand rates for this region
            let mut demand: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
            for entry in state.shared_state.region_demand.iter() {
                let (model_id, region) = entry.key();
                if region.eq_ignore_ascii_case(&code) {
                    demand.insert(model_id.0.clone(), serde_json::json!(*entry.value()));
                }
            }

            // Coverage gaps: models where this region has 0 holders
            let coverage_gaps: Vec<&str> = all_models
                .iter()
                .filter(|m| !models_json.contains_key(m.as_str()))
                .map(|m| m.as_str())
                .collect();

            // Per-model replication target for this region
            let mut replication_target: serde_json::Map<String, serde_json::Value> =
                serde_json::Map::new();
            for model_id_str in models_json.keys() {
                let model_id = crate::types::ModelId(model_id_str.clone());
                let request_count = state
                    .shared_state
                    .models
                    .model_request_counts
                    .get(&model_id)
                    .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0);
                let global_floor = if pool_size <= 1 {
                    min_replicas
                } else {
                    let log2_pool = (pool_size as f64).log2().ceil() as usize;
                    log2_pool.clamp(min_replicas, pool_size / 3).max(1)
                };
                let demand_factor = match request_count {
                    0 => 1.0,
                    1..=5 => 1.5,
                    6..=20 => 2.0,
                    21..=100 => 2.5,
                    _ => 3.0,
                };
                let target = (global_floor as f64 * demand_factor).ceil() as usize;
                replication_target.insert(
                    model_id_str.clone(),
                    serde_json::json!(target.min(pool_size).max(1)),
                );
            }

            (
                code,
                serde_json::json!({
                    "total": total,
                    "models": models_json,
                    "demand": demand,
                    "coverage_gaps": coverage_gaps,
                    "replication_target": replication_target,
                }),
            )
        })
        .collect();

    Json(serde_json::json!({ "regions": region_json }))
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

    // Use TCP address (port+10) — more reliable across environments (WSL2, Docker, NAT).
    // QUIC on WSL2 often fails with handshake timeouts on the virtual adapter IP.
    let tcp_port = port + 10;
    let multiaddr_str = format!("/ip4/{best_ip}/tcp/{tcp_port}/p2p/{peer_id_str}");
    let code = if let Ok(addr) = multiaddr_str.parse::<libp2p::Multiaddr>() {
        crate::network::discovery::encode_network_code(&addr)
    } else {
        multiaddr_str.clone()
    };

    // Determine network phase
    let phase = if peer_count == 0 {
        "seedling" // no peers — solo node
    } else {
        "established" // 1+ peers — connected to network
    };

    Json(serde_json::json!({
        "code": code,
        "node_id": format!("{}", state.shared_state.identity.node_id()),
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
    // SEC: Limit invite code length to prevent memory exhaustion during decode
    if body.code.len() > 4096 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invite code too long (max 4096 chars)".into(),
        )));
    }
    let addr_str = crate::network::discovery::decode_network_code(&body.code)
        .map_err(|e| ApiError(crate::error::SwarmError::Validation(e.to_string())))?;

    // Validate the multiaddr
    let _addr: libp2p::Multiaddr = addr_str.parse().map_err(|e: libp2p::multiaddr::Error| {
        ApiError(crate::error::SwarmError::Validation(format!(
            "Invalid address in invite code: {e}"
        )))
    })?;

    tracing::info!(addr = %addr_str, "Joining network via invite code");

    // Save to peer cache so it persists across restarts
    let mut cached = crate::network::peer_cache::load_peer_cache(&state.shared_state.db);
    if !cached.contains(&addr_str) {
        cached.push(addr_str.clone());
        crate::network::peer_cache::save_peer_cache(&state.shared_state.db, &cached);
    }

    // Dial immediately if network manager is available
    if let Some(ref tx) = state.network_tx {
        let _ = tx
            .send(crate::types::NetworkCommand::DialAddress(addr_str.clone()))
            .await;
    }

    Ok(Json(serde_json::json!({
        "status": "ok",
        "address": addr_str,
        "message": "Connecting to peer..."
    })))
}

#[derive(Deserialize)]
pub struct JoinNetworkRequest {
    pub code: String,
}
// ---- Resource Schedule API ----

/// GET /api/admin/schedule — Get current resource schedule.
pub async fn get_schedule(State(state): State<AppState>) -> Json<serde_json::Value> {
    let schedule = state.shared_state.models.resource_schedule.read().await;
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

/// PUT /api/admin/schedule — Update resource schedule at runtime (persisted to redb).
pub async fn update_schedule(
    State(state): State<AppState>,
    Json(body): Json<ScheduleUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mut schedule = state.shared_state.models.resource_schedule.write().await;

    if let Some(enabled) = body.enabled {
        schedule.enabled = enabled;
    }
    if let Some(start) = body.reduced_hours_start {
        if start > 23 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "reduced_hours_start must be 0-23".to_string(),
            )));
        }
        schedule.reduced_hours_start = start;
    }
    if let Some(end) = body.reduced_hours_end {
        if end > 23 {
            return Err(ApiError(crate::error::SwarmError::Validation(
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
                return Err(ApiError(crate::error::SwarmError::Validation(
                    "prune_aggressiveness must be 'normal', 'aggressive', or 'conservative'"
                        .to_string(),
                )));
            }
        }
    }

    // Persist to DB
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
