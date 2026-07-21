use axum::extract::State;
use axum::Json;
use serde::Deserialize;

/// Timeout for considering a peer "healthy" — peers not seen within this window
/// are marked as unhealthy in the admin dashboard.
const PEER_HEALTHY_TIMEOUT_SECS: i64 = 90;
/// Maximum length of an invite code string accepted by the join-network endpoint.
const MAX_INVITE_CODE_LEN: usize = 4096;

/// Upper bound for `max_concurrent_requests` accepted via the config API.
const MAX_CONCURRENT_REQUESTS_CAP: u32 = 256;
/// Upper bound for `max_bandwidth_mbps` accepted via the config API.
const MAX_BANDWIDTH_MBPS_CAP: u64 = 100_000;
/// Lower bound for `max_disk_mb` accepted via the config API.
const MIN_DISK_MB: u64 = 100;
/// Upper bound for `max_disk_mb` accepted via the config API.
const MAX_DISK_MB: u64 = 10_000_000;
/// Upper bound for `auto_manage_max_storage_mb` accepted via the config API.
const MAX_AUTO_MANAGE_STORAGE_MB: u64 = 10_000_000;
/// Upper bound for `batch_timeout_ms` accepted via the config API.
const MAX_BATCH_TIMEOUT_MS: u64 = 60_000;
/// Maximum bytes accepted for the `status` filter on `list_responses`.
const MAX_STATUS_FILTER_BYTES: usize = 256;

/// Serialize a peer registry entry to JSON. Used by both REST and WebSocket.
///
/// When `include_addresses` is true, includes `addresses` and `last_seen` fields
/// (REST API returns these; WebSocket omits them for bandwidth).
pub fn serialize_peer_to_json(
    peer: &crate::types::PeerInfo,
    state: &crate::daemon::state::SharedState,
    include_addresses: bool,
) -> serde_json::Value {
    let timeout = chrono::Duration::seconds(PEER_HEALTHY_TIMEOUT_SECS);
    let now = chrono::Utc::now();
    let healthy = now.signed_duration_since(peer.last_seen) < timeout;
    let hosted_shards_count = peer
        .capability
        .as_ref()
        .map(|c| c.hosted_shards.len())
        .unwrap_or(0);
    let hosted_models: Vec<String> = peer
        .capability
        .as_ref()
        .map(|c| {
            let mut models: Vec<String> = c
                .hosted_shards
                .iter()
                .map(|s| s.model_id.0.clone())
                .collect();
            models.sort_unstable();
            models.dedup();
            models
        })
        .unwrap_or_default();
    let nickname = state
        .nickname_registry
        .get(&peer.node_id)
        .map(|r| r.nickname.clone());
    let mut obj = serde_json::json!({
        "node_id": format!("{}", peer.node_id),
        "nickname": nickname,
        "latency_ms": peer.latency_ms,
        "trust_score": peer.trust_score,
        "healthy": healthy,
        "gpu": peer.capability.as_ref().and_then(|c| c.gpu.as_ref().map(|g| &g.name)),
        "hosted_models": hosted_models,
        "hosted_shards": hosted_shards_count,
        "is_lan_peer": peer.is_lan_peer,
    });
    if include_addresses {
        if let Some(o) = obj.as_object_mut() {
            o.insert("addresses".into(), serde_json::json!(peer.addresses));
            o.insert(
                "last_seen".into(),
                serde_json::json!(peer.last_seen.to_rfc3339()),
            );
        }
    }
    obj
}

// Re-export sub-module handlers so server.rs routes continue to use `admin::handler_name`
pub use super::admin_hf::*;
pub use super::admin_models::*;
pub use super::admin_providers::*;

use crate::api::server::AppState;
use crate::config::ContributionMode;
use crate::error::ApiError;

/// GET /api/admin/swarm/capacity — collective hardware + serveable-models snapshot.
///
/// Designed for the "what can my swarm run?" dashboard header. Refreshes
/// the snapshot inline so the response always reflects the current peer
/// set (cheap — single pass over the registries). Non-technical-friendly
/// fields: every value is human-renderable without further interpretation.
pub async fn swarm_capacity(State(state): State<AppState>) -> Json<serde_json::Value> {
    crate::daemon::state::refresh_swarm_capacity(&state.shared_state);
    let snap = state.shared_state.metrics.swarm_capacity.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/wishlist — ranked list of models the swarm wants.
///
/// R111. The wishlist is the user-visible face of auto-manage: instead of
/// the daemon downloading models in mysterious silence, the user sees a
/// ranked queue with status badges and human-readable "why" tags. Refreshed
/// on demand so manual browsing always sees fresh data.
pub async fn wishlist(State(state): State<AppState>) -> Json<serde_json::Value> {
    crate::model::auto_manage::refresh_wishlist(&state.shared_state);
    let snap = state.shared_state.models.wishlist.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/quant-recommendations — per-family quant choice
/// recommendations (R133).
///
/// Groups models in the local registry by inferred base name (model name
/// with the quant suffix stripped) and surfaces the highest-quality
/// variant that fits the swarm's aggregate VRAM with reasonable
/// replication. Read-only — the recommender does NOT auto-switch which
/// quant the auto-manage system downloads. Frontend can render
/// "We're hosting Q4_K_M because the swarm only has X TB; with N more
/// nodes we'd switch to Q5_K_M."
pub async fn quant_recommendations(State(state): State<AppState>) -> Json<serde_json::Value> {
    crate::model::auto_manage::quant::refresh_quant_recommendations(&state.shared_state);
    let snap = state.shared_state.models.quant_recommendations.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/foreign-pool-catalog — R134 — cross-pool model
/// availability discovery surface. Returns the cached signals from
/// `PoolModelAvailability` gossip, grouped by pool, with stale entries
/// trimmed against `FOREIGN_POOL_CATALOG_MAX_AGE_MS`. Pure discovery —
/// does NOT bind routing decisions. Useful for the admin UI's
/// "models the swarm knows about but this pool doesn't host yet" tile.
pub async fn foreign_pool_catalog(State(state): State<AppState>) -> Json<serde_json::Value> {
    use std::collections::BTreeMap;
    let now_ms = crate::types::unix_now_ms();
    state.shared_state.credits.trim_stale_foreign_pool_catalog(
        now_ms,
        crate::daemon::dispatch::FOREIGN_POOL_CATALOG_MAX_AGE_MS,
    );
    // Group by pool_id for the response shape.
    let mut by_pool: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
    for entry in state.shared_state.credits.foreign_pool_catalog.iter() {
        let (pool, model) = entry.key();
        by_pool
            .entry(format!("{pool}"))
            .or_default()
            .push(serde_json::json!({
                "model_id": model.0,
                "received_at_ms": *entry.value(),
            }));
    }
    let pools: Vec<serde_json::Value> = by_pool
        .into_iter()
        .map(|(pool_id, models)| serde_json::json!({"pool_id": pool_id, "models": models}))
        .collect();
    Json(serde_json::json!({
        "pools": pools,
        "computed_at_ms": now_ms,
    }))
}

/// GET /api/admin/hf/trending — latest HuggingFace trending-GGUF snapshot
/// captured by the background HfWatcher (R112). Surfaces the same data the
/// wishlist scorer consumes so the frontend can render a "trending now"
/// view without re-querying HF.
pub async fn hf_trending(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snap = state.shared_state.models.hf_trending_cache.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/swarm/capacity-plan — what-if scenarios.
///
/// R113. Drives the dashboard's "if N more contributors joined with X GB
/// each, you'd unlock Y" message — the educational layer that turns the
/// product's value prop ("contribute and run huge models together") into
/// a concrete next step. Three baked scenarios (small/medium/large) +
/// a headline_target showing the closest aspirational upgrade.
pub async fn swarm_capacity_plan(State(state): State<AppState>) -> Json<serde_json::Value> {
    let plan = crate::daemon::state::compute_capacity_plan(&state.shared_state);
    Json(serde_json::to_value(&plan).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/storage/breakdown — disk allocation summary for the
/// stacked-bar UI. Replaces the dual "Max Disk" / "Max Auto-Download Storage"
/// settings with a single bar showing total / used / auto-manage-budget /
/// free. R110.
///
/// Numbers are pre-converted to MB so the frontend doesn't have to handle
/// byte→MB rounding (avoids `49.99 GB` rendering when user typed `50 GB`).
pub async fn storage_breakdown(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.config.clone();
    let local_node_id = state.shared_state.identity.node_id().clone();
    let mgr = &state.shared_state.models;

    // Bytes currently held on disk by this node.
    let mut used_bytes: u64 = 0;
    let mut held_shards: u32 = 0;
    for entry in state.shared_state.model_registry.models() {
        for shard in &entry.shards {
            let sid = crate::types::ShardId {
                model_id: entry.id.clone(),
                index: shard.index,
            };
            let holders = state.shared_state.model_registry.shard_holders(&sid);
            if holders.contains(&local_node_id) {
                used_bytes = used_bytes.saturating_add(shard.size_bytes);
                held_shards += 1;
            }
        }
    }

    // What auto-manage will try to grow to. Shared with the scheduler
    // via `model::auto_manage::compute_budget_max_bytes` so the two
    // can't drift if the ContributionMode scaling changes.
    let auto_target_bytes = crate::model::auto_manage::compute_budget_max_bytes(
        config.auto_manage.max_storage_mb,
        config.resources.max_disk_mb,
        &config.node.contribution,
    );
    let total_bytes = config
        .resources
        .max_disk_mb
        .saturating_mul(1024)
        .saturating_mul(1024);
    let auto_target_capped = auto_target_bytes.min(total_bytes);
    // Free = max(0, total - used). When used > total (rare — happens if
    // user shrinks Max Disk after already having more on disk), we report
    // 0 free and let the UI show a "you're over your budget" hint.
    let free_bytes = total_bytes.saturating_sub(used_bytes);

    let auto_enabled = mgr
        .auto_manage_enabled
        .load(std::sync::atomic::Ordering::Relaxed);

    Json(serde_json::json!({
        // All values in MB to match the slider units; UI converts to GB
        // for display. Single-source-of-truth: never present "max_disk_mb"
        // and "auto_manage_max_storage_mb" as independent inputs again.
        "total_mb": total_bytes / (1024 * 1024),
        "used_mb": used_bytes / (1024 * 1024),
        "free_mb": free_bytes / (1024 * 1024),
        "auto_target_mb": auto_target_capped / (1024 * 1024),
        "held_shards": held_shards,
        "auto_manage_enabled": auto_enabled,
        "contribution": match config.node.contribution {
            ContributionMode::Minimal => "minimal",
            ContributionMode::Moderate => "moderate",
            ContributionMode::Maximum => "maximum",
        },
    }))
}

/// GET /api/admin/stats — Full dashboard stats snapshot.
pub async fn stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let node_id = hex::encode(state.shared_state.identity.node_id().0);

    // Snapshot the locked values into stack copies and drop the guards BEFORE
    // the sysinfo spawn_blocking await. Holding RwLock guards across the
    // blocking-call .await would otherwise park concurrent writers
    // (apply_credit on the inference hot path, the health monitor) for the
    // duration of the /proc/* scan.
    let (uptime_start, requests_made) = {
        let stats = state.shared_state.metrics.node_stats.read().await;
        (stats.uptime_start, stats.requests_made)
    };
    let (tier, credit_json) = {
        let credit = state.shared_state.credits.credit_balance.read().await;
        (
            crate::credit::priority::PriorityCalculator::tier_name(credit.balance),
            super::credit_summary_json(&credit),
        )
    };
    let uptime_seconds = (chrono::Utc::now() - uptime_start).num_seconds().max(0) as u64;

    // Count only shards held locally (not all tracked shards network-wide)
    let hosted_shards = crate::api::metrics::count_local_shards(&state.shared_state);

    // Hardware detection — sysinfo does blocking filesystem reads (/proc/*)
    let ss = state.shared_state.clone();
    let hardware = tokio::task::spawn_blocking(move || detect_hardware(&ss))
        .await
        .unwrap_or_else(|_| serde_json::json!({}));

    // Inference performance metrics from latency samples
    let inference_perf = match crate::api::metrics::compute_latency_stats(&state.shared_state) {
        Some(ls) => serde_json::json!({
            "total_requests": ls.total_requests,
            "avg_latency_ms": ls.avg_ms,
            "min_latency_ms": ls.min_ms,
            "max_latency_ms": ls.max_ms,
            "p50_latency_ms": ls.p50_ms,
            "p95_latency_ms": ls.p95_ms,
            "p99_latency_ms": ls.p99_ms,
            "samples": ls.count,
        }),
        None => serde_json::json!({
            "total_requests": state.shared_state.metrics.inference_requests_total
                .load(std::sync::atomic::Ordering::Relaxed),
            "avg_latency_ms": null,
            "samples": 0,
        }),
    };

    // SWARM-SPEC layer metrics (R136): hedge + prefetch tracker
    // snapshots. Empty / zero counters until those layers see real
    // traffic with their feature flags enabled.
    // R137: + L1 n-gram hit/miss lifetime counters so operators can
    // tell whether the cascade is actually firing on their workload mix.
    let ngram_hits = state
        .shared_state
        .metrics
        .ngram_hits
        .load(std::sync::atomic::Ordering::Relaxed);
    let ngram_misses = state
        .shared_state
        .metrics
        .ngram_misses
        .load(std::sync::atomic::Ordering::Relaxed);
    let ngram_total = ngram_hits + ngram_misses;
    let ngram_hit_rate = if ngram_total > 0 {
        ngram_hits as f64 / ngram_total as f64
    } else {
        0.0
    };
    let swarm_spec_metrics = serde_json::json!({
        "hedge": state.shared_state.metrics.hedge_tracker.metrics(),
        "prefetch": state.shared_state.metrics.prefetch_orchestrator.metrics(),
        "ngram": {
            "hits": ngram_hits,
            "misses": ngram_misses,
            "total": ngram_total,
            "hit_rate": (ngram_hit_rate * 10000.0).round() / 10000.0,
        },
    });

    Json(serde_json::json!({
        "node_id": node_id,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime_seconds,
        "tier": tier,
        "peers": state.shared_state.connected_node_ids.len(),
        "requests_served": state.shared_state.metrics.requests_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        "forwards_served": state.shared_state.metrics.forwards_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        "requests_made": requests_made,
        "active_requests": state.shared_state.active_pipelines.len(),
        "hosted_shards": hosted_shards,
        "credits": credit_json,
        "hardware": hardware,
        "inference": inference_perf,
        "swarm_spec": swarm_spec_metrics,
    }))
}

/// GET /api/admin/config — Return current configuration.
///
/// Reads the persisted config from disk so this always reflects the latest saved
/// values (including those applied by `PUT /api/admin/config`). Falls back to the
/// in-memory startup config if the file cannot be read.
pub async fn get_config(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config_path = state.config.node.data_dir.join("config.toml");
    // Hot-path: dashboard polls this. Read off the async runtime so the
    // sync read doesn't block a Tokio worker (R98). update_config at line
    // ~296 already wraps writes in spawn_blocking — match that pattern.
    let config = tokio::task::spawn_blocking(move || std::fs::read_to_string(&config_path))
        .await
        .ok()
        .and_then(|r| r.ok())
        .and_then(|s| toml::from_str::<crate::config::Config>(&s).ok())
        .unwrap_or_else(|| state.config.clone());
    let config = &config;
    let contribution = match config.node.contribution {
        ContributionMode::Minimal => "minimal",
        ContributionMode::Moderate => "moderate",
        ContributionMode::Maximum => "maximum",
    };
    // Include claude_subscription config if the feature is enabled
    #[cfg(feature = "claude-subscription")]
    let claude_sub = {
        let providers = state.shared_state.metrics.providers_config.try_read();
        providers.ok().and_then(|p| {
            p.claude_subscription.as_ref().map(|s| {
                serde_json::json!({
                    "enabled": s.enabled,
                })
            })
        })
    };
    #[cfg(not(feature = "claude-subscription"))]
    let claude_sub: Option<serde_json::Value> = None;

    let mut result = serde_json::json!({
        "contribution": contribution,
        "contribution_auto": config.node.contribution_auto,
        "max_concurrent_requests": config.inference.max_concurrent_requests,
        "max_bandwidth_mbps": config.resources.max_bandwidth_mbps,
        "max_disk_mb": config.resources.max_disk_mb,
        "max_gpu_vram_mb": config.resources.max_gpu_vram_mb,
        "listen_port": config.node.listen_port,
        "session_timeout_seconds": config.inference.session_timeout_seconds,
        "auto_manage_shards": state.shared_state.models.auto_manage_enabled.load(std::sync::atomic::Ordering::Relaxed),
        "auto_manage_max_storage_mb": config.auto_manage.max_storage_mb,
        "shard_size_mb": config.model.shard_size_mb,
        "max_batch_size": config.inference.max_batch_size,
        "batch_timeout_ms": config.inference.batch_timeout_ms,
        // R137: surface the runtime values (not the startup-frozen config)
        // so the dashboard reflects post-PUT state immediately.
        "allow_cross_pool_inference": state.shared_state.credits.allow_cross_pool_inference.load(std::sync::atomic::Ordering::Relaxed),
        "share_model_catalog": state.shared_state.credits.share_model_catalog.load(std::sync::atomic::Ordering::Relaxed),
    });
    if let Some(cs) = claude_sub {
        result["claude_subscription"] = cs;
    }
    Json(result)
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
            "moderate" => ContributionMode::Moderate,
            "maximum" => ContributionMode::Maximum,
            other => {
                return Err(ApiError(crate::error::SwarmError::Validation(format!(
                    "Unknown contribution mode '{other}' (expected: minimal, moderate, maximum)"
                ))));
            }
        };
    }
    if let Some(auto) = body.contribution_auto {
        config.node.contribution_auto = auto;
        // Mirror to the runtime atomic so prune.rs picks it up on the
        // next tick without a daemon restart. Persisted-config side is
        // for restart durability only.
        state
            .shared_state
            .models
            .contribution_auto
            .store(auto, std::sync::atomic::Ordering::Release);
    }
    if let Some(max_reqs) = body.max_concurrent_requests {
        config.inference.max_concurrent_requests = max_reqs.clamp(1, MAX_CONCURRENT_REQUESTS_CAP);
    }
    if let Some(bw) = body.max_bandwidth_mbps {
        config.resources.max_bandwidth_mbps = bw.clamp(1, MAX_BANDWIDTH_MBPS_CAP);
    }
    if let Some(disk) = body.max_disk_mb {
        config.resources.max_disk_mb = disk.clamp(MIN_DISK_MB, MAX_DISK_MB);
    }
    if let Some(vram) = body.max_gpu_vram_mb {
        // 0 = auto (80% of detected VRAM). Cap at 1 TB so a stray UI
        // value can't disable VRAM accounting entirely on the dashboard
        // side; the inference path will still honor whatever this is.
        config.resources.max_gpu_vram_mb = vram.min(1_048_576);
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
        config.auto_manage.max_storage_mb = max_storage.clamp(1, MAX_AUTO_MANAGE_STORAGE_MB);
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
        config.inference.batch_timeout_ms = timeout.clamp(1, MAX_BATCH_TIMEOUT_MS);
    }
    if let Some(allow) = body.allow_cross_pool_inference {
        // R137: persist to TOML for restart durability AND mirror to the
        // runtime atomic so cross_pool_extras picks it up on the next
        // request without a daemon restart.
        config.pool.allow_cross_pool_inference = allow;
        state
            .shared_state
            .credits
            .allow_cross_pool_inference
            .store(allow, std::sync::atomic::Ordering::Release);
    }
    if let Some(share) = body.share_model_catalog {
        // R137: same pattern as above for the model-catalog gossip flag.
        // Read by HealthMonitor::broadcast_pool_model_availability on the
        // next gossip tick (≤30s by default).
        config.pool.share_model_catalog = share;
        state
            .shared_state
            .credits
            .share_model_catalog
            .store(share, std::sync::atomic::Ordering::Release);
    }

    // Write updated config to disk
    let toml_str = toml::to_string_pretty(&config).map_err(|e| {
        ApiError(crate::error::SwarmError::Internal(format!(
            "Failed to serialize config to TOML: {e}"
        )))
    })?;

    let cp = config_path.clone();
    let cp_for_err = config_path.clone();
    tokio::task::spawn_blocking(move || {
        if let Some(parent) = cp.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&cp, toml_str)
    })
    .await
    .map_err(|e| {
        tracing::error!(error = %e, "Config save task panicked");
        ApiError(crate::error::SwarmError::Internal(
            "Failed to save configuration".into(),
        ))
    })?
    .map_err(|e| {
        // OS/disk failure (permission denied, disk full, ENOTDIR, etc)
        // — not a code bug. ServiceUnavailable (503) surfaces the right
        // semantics to the client (transient, retryable) and the
        // structured error log + response body carry the path so the
        // operator can triage.
        ApiError(crate::error::SwarmError::ServiceUnavailable(format!(
            "Failed to write config to {}: {e}",
            cp_for_err.display()
        )))
    })?;

    tracing::info!(path = %config_path.display(), "Configuration saved");

    // Hot-reload operational params so in-memory state reflects the saved config
    state
        .shared_state
        .apply_config_reload(crate::config::OperationalParams::from_config(&config));

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

    // Map config-file errors to HTTP-appropriate variants:
    //   missing file → 404 NotFound (the dashboard hasn't saved yet)
    //   parse / IO error → 400 Validation (broken file content)
    // SwarmError::Config is reserved for startup-only errors per the rule in
    // .claude/rules/completeness.md and would otherwise leak the unhelpful
    // "invalid_request_error" type for what is really a config-file issue.
    let params = crate::config::reload_operational_params(&config_path).map_err(|e| match e {
        crate::error::SwarmError::Config(msg) if msg.starts_with("Config file not found") => {
            ApiError(crate::error::SwarmError::NotFound(msg))
        }
        crate::error::SwarmError::Config(msg) => {
            ApiError(crate::error::SwarmError::Validation(msg))
        }
        crate::error::SwarmError::Io(io_err) => ApiError(crate::error::SwarmError::Validation(
            format!("Config file IO error: {io_err}"),
        )),
        other => ApiError(other),
    })?;

    let old = crate::config::OperationalParams::from_config(&state.config);
    let changed = params != old;

    state.shared_state.apply_config_reload(params.clone());

    if changed {
        tracing::info!(?params, "Config reloaded with changes via API");
    } else {
        tracing::info!(path = %config_path.display(), "Config reloaded via API — no changes detected");
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
            "contribution": params.contribution,
            "contribution_auto": params.contribution_auto,
            "max_gpu_vram_mb": params.max_gpu_vram_mb,
        }
    })))
}

/// GET /api/admin/peers — List connected peers.
pub async fn list_peers(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let peers: Vec<serde_json::Value> = state
        .shared_state
        .peer_registry
        .iter()
        .map(|entry| serialize_peer_to_json(entry.value(), &state.shared_state, true))
        .collect();

    Json(peers)
}

/// GET /api/admin/credits — Credit details.
pub async fn credit_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    // Snapshot the balance and drop the read lock before computing escrow
    // so the credit hot-path doesn't park behind us.
    let (balance, lifetime_earned, lifetime_spent, last_updated, tier) = {
        let credit = state.shared_state.credits.credit_balance.read().await;
        (
            credit.balance,
            credit.lifetime_earned,
            credit.lifetime_spent,
            credit.last_updated.to_rfc3339(),
            crate::credit::priority::PriorityCalculator::tier_name(credit.balance),
        )
    };
    let escrow_held = state.shared_state.credits.escrow_manager.pending_total();
    let escrow_pending = state.shared_state.credits.escrow_manager.pending_count();

    Json(serde_json::json!({
        "balance": balance,
        "lifetime_earned": lifetime_earned,
        "lifetime_spent": lifetime_spent,
        "tier": tier,
        "last_updated": last_updated,
        "escrow_held": escrow_held,
        "escrow_pending_count": escrow_pending,
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
    pub contribution_auto: Option<bool>,
    pub max_concurrent_requests: Option<u32>,
    pub max_bandwidth_mbps: Option<u64>,
    pub max_disk_mb: Option<u64>,
    pub max_gpu_vram_mb: Option<u64>,
    pub auto_manage_shards: Option<bool>,
    pub auto_manage_max_storage_mb: Option<u64>,
    pub shard_size_mb: Option<u64>,
    pub max_batch_size: Option<u32>,
    pub batch_timeout_ms: Option<u64>,
    /// R137: hot-reloadable cross-pool inference fallback toggle.
    /// Persisted to config TOML + mirrored to
    /// `state.credits.allow_cross_pool_inference`.
    pub allow_cross_pool_inference: Option<bool>,
    /// R137: hot-reloadable cross-pool model catalog gossip toggle.
    /// Persisted to config TOML + mirrored to
    /// `state.credits.share_model_catalog`.
    pub share_model_catalog: Option<bool>,
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
            let used = crate::model::auto_manage::vram::query_gpu_vram_used();
            (
                Some(gpu.name.clone()),
                Some(gpu.vram_total_mb),
                used.or(Some(gpu.vram_total_mb.saturating_sub(gpu.vram_free_mb))),
            )
        }
        None => {
            let (name, total) = detect_gpu_nvidia_smi();
            let used = crate::model::auto_manage::vram::query_gpu_vram_used();
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
pub(crate) use crate::model::auto_manage::vram::detect_gpu_nvidia_smi;

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
                    let max_replicas = (pool_size / 3).max(1);
                    log2_pool.clamp(min_replicas.min(max_replicas), max_replicas)
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
    // or fall back to detecting the local machine's non-loopback IP. Cap the
    // peer scan at NETWORK_CODE_PEER_SCAN_CAP — a public-facing IP is almost
    // always advertised by the first few peers, and at 10k-peer scale the
    // unbounded inner loop becomes a notable per-request hot path on the
    // dashboard's invite-code refresh.
    const NETWORK_CODE_PEER_SCAN_CAP: usize = 64;
    const NETWORK_CODE_ADDR_PER_PEER_CAP: usize = 16;
    let best_ip = {
        // Try to find a non-loopback IP from peers' addresses for our node
        let mut found_ip = None;
        for peer in state
            .shared_state
            .peer_registry
            .iter()
            .take(NETWORK_CODE_PEER_SCAN_CAP)
        {
            for addr in peer.addresses.iter().take(NETWORK_CODE_ADDR_PER_PEER_CAP) {
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
        "peer_id": peer_id_str,
        // The node's current reachable dial addresses (listeners ∪ confirmed
        // external addresses), each terminated with /p2p/<peer_id>. For a node
        // with `network.external_address` set (e.g. an anchor with a DuckDNS
        // host) this is the exact string to drop into other nodes'
        // `bootstrap_peers`. Empty until the swarm has bound + confirmed addrs.
        "listen_multiaddrs": state.shared_state.listen_multiaddrs.load().as_ref().clone(),
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
    if body.code.len() > MAX_INVITE_CODE_LEN {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Invite code too long (max {} chars)",
            MAX_INVITE_CODE_LEN
        ))));
    }
    let addr_str = crate::network::discovery::decode_network_code(&body.code)
        .map_err(|e| ApiError(crate::error::SwarmError::Validation(e.to_string())))?;

    // Validate the multiaddr
    let _addr: libp2p::Multiaddr = addr_str.parse().map_err(|e: libp2p::multiaddr::Error| {
        ApiError(crate::error::SwarmError::Validation(format!(
            "Invalid address in invite code: {e}"
        )))
    })?;

    // SEC: Reject private / loopback / link-local / cloud-metadata addresses.
    // Without this check, an attacker with the API key could supply a multiaddr
    // pointing at internal services (e.g. 169.254.169.254 IMDS, 127.0.0.1
    // services on the host) and the daemon would attempt P2P-layer dials —
    // P2P-layer SSRF — and persist the address to peer cache for re-dialing.
    if crate::network::helpers::is_non_public_addr(&addr_str) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Address resolves to a private/loopback/link-local IP — refusing to dial".into(),
        )));
    }

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
    // Clone current schedule, validate + apply updates without holding the write lock
    let mut new_schedule = state
        .shared_state
        .models
        .resource_schedule
        .read()
        .await
        .clone();

    if let Some(enabled) = body.enabled {
        new_schedule.enabled = enabled;
    }
    if let Some(start) = body.reduced_hours_start {
        if start > 23 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "reduced_hours_start must be 0-23".to_string(),
            )));
        }
        new_schedule.reduced_hours_start = start;
    }
    if let Some(end) = body.reduced_hours_end {
        if end > 23 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "reduced_hours_end must be 0-23".to_string(),
            )));
        }
        new_schedule.reduced_hours_end = end;
    }
    if let Some(ref contribution) = body.reduced_contribution {
        match contribution.as_str() {
            "minimal" | "moderate" | "maximum" => {
                new_schedule.reduced_contribution = contribution.clone();
            }
            _ => {
                return Err(ApiError(crate::error::SwarmError::Validation(
                    "reduced_contribution must be 'minimal', 'moderate', or 'maximum'".to_string(),
                )));
            }
        }
    }
    if let Some(ref aggressiveness) = body.prune_aggressiveness {
        match aggressiveness.as_str() {
            "normal" | "aggressive" | "conservative" => {
                new_schedule.prune_aggressiveness = aggressiveness.clone();
            }
            _ => {
                return Err(ApiError(crate::error::SwarmError::Validation(
                    "prune_aggressiveness must be 'normal', 'aggressive', or 'conservative'"
                        .to_string(),
                )));
            }
        }
    }

    // Persist to DB (no write lock held)
    if let Err(e) = state
        .shared_state
        .db
        .put_json("resource_schedule", "current", &new_schedule)
    {
        tracing::warn!(error = %e, "Failed to persist resource schedule — will revert on restart");
    }

    tracing::debug!(
        enabled = new_schedule.enabled,
        prune_aggressiveness = %new_schedule.prune_aggressiveness,
        "DIAG: schedule updated"
    );

    let result = serde_json::json!({
        "status": "ok",
        "enabled": new_schedule.enabled,
        "reduced_hours_start": new_schedule.reduced_hours_start,
        "reduced_hours_end": new_schedule.reduced_hours_end,
        "reduced_contribution": new_schedule.reduced_contribution,
        "prune_aggressiveness": new_schedule.prune_aggressiveness,
    });

    // Briefly acquire write lock to commit
    *state.shared_state.models.resource_schedule.write().await = new_schedule;

    Ok(Json(result))
}

// ============================================================================
// V6 (responses_api_v2): Responses dashboard endpoint
// ============================================================================

/// Query parameters for `GET /api/admin/responses`.
#[derive(Debug, Deserialize)]
pub struct AdminResponsesQuery {
    /// Filter by status (queued | in_progress | completed | failed |
    /// cancelled | incomplete). Repeatable via comma list. Empty / unset
    /// returns every status.
    #[serde(default)]
    pub status: Option<String>,
    /// Cap the number of records returned. Default 100, max 500.
    #[serde(default)]
    pub limit: Option<u32>,
}

/// `GET /api/admin/responses?status=...&limit=...` — list stored
/// `/v1/responses` records for the dashboard. Sorted newest first.
///
/// Streams the underlying redb tree so memory stays O(limit) rather
/// than O(total_records). The full preview JSON is only built for the
/// records that survive the bounded top-k pass.
pub async fn list_responses(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    State(state): State<crate::api::server::AppState>,
    axum::extract::Query(params): axum::extract::Query<AdminResponsesQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // SEC: input_preview shows a 100-char prefix of the user's prompt. With a
    // shared cluster API key, exposing it to non-loopback callers would leak
    // every other user's prompts. Loopback callers (local dashboard) get the
    // full preview; remote API-key holders see metadata only.
    let include_preview = addr.ip().is_loopback();
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    /// Heap entry that orders by `created_at` only — the record itself
    /// doesn't implement Ord. Sort order is REVERSED (`older` compares
    /// as greater) so a default max-heap behaves as a min-heap of
    /// newest survivors: `peek()` returns the oldest kept candidate,
    /// `pop()` evicts it.
    struct HeapEntry {
        created_at: i64,
        rec: crate::api::openai::responses::store::ResponsesRecord,
    }
    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.created_at == other.created_at
        }
    }
    impl Eq for HeapEntry {}
    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            other.created_at.cmp(&self.created_at)
        }
    }

    // Cap the raw query string before splitting so a caller can't pass a
    // megabyte-long `?status=` and force `.split(',')` to materialise an
    // arbitrarily large Vec — the only valid values are a handful of
    // short ASCII enum strings, 256 bytes is generous.
    let status_filter: Option<Vec<String>> = match params.status {
        Some(s) if s.len() > MAX_STATUS_FILTER_BYTES => {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "status filter too long ({} bytes, max {MAX_STATUS_FILTER_BYTES})",
                s.len()
            ))));
        }
        Some(s) => {
            let tokens: Vec<String> = s
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            // Reject typos: a `?status=complete` filter would silently match
            // zero records and look like an empty result instead of an error.
            const VALID: &[&str] = &[
                "queued",
                "in_progress",
                "completed",
                "failed",
                "cancelled",
                "incomplete",
            ];
            for tok in &tokens {
                if !VALID.contains(&tok.as_str()) {
                    return Err(ApiError(crate::error::SwarmError::Validation(format!(
                        "unknown status filter '{tok}': must be one of queued, in_progress, completed, failed, cancelled, incomplete"
                    ))));
                }
            }
            Some(tokens)
        }
        None => None,
    };
    let limit = params.limit.unwrap_or(100).clamp(1, 500) as usize;

    // Whether a live background-streaming task is in flight for this id.
    let live_ids: std::collections::HashSet<String> =
        crate::api::openai::responses::background::BACKGROUND_STATE
            .iter()
            .map(|e| e.key().clone())
            .collect();

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(limit + 1);

    state
        .db
        .for_each_json::<crate::api::openai::responses::store::ResponsesRecord, _>(
            crate::api::openai::responses::store::TREE,
            |_subkey, rec| {
                if let Some(filter) = &status_filter {
                    let s = serde_json::to_string(&rec.response.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    if !filter.iter().any(|f| f == &s) {
                        return;
                    }
                }
                if heap.len() < limit {
                    heap.push(HeapEntry {
                        created_at: rec.created_at,
                        rec,
                    });
                } else if let Some(top) = heap.peek() {
                    if rec.created_at > top.created_at {
                        heap.pop();
                        heap.push(HeapEntry {
                            created_at: rec.created_at,
                            rec,
                        });
                    }
                }
            },
        )
        .map_err(ApiError)?;

    // into_sorted_vec yields ascending by `Ord` — which we reversed —
    // so the result is oldest → newest. Reverse for newest-first.
    let mut kept: Vec<HeapEntry> = heap.into_sorted_vec();
    kept.reverse();

    let data: Vec<serde_json::Value> = kept
        .into_iter()
        .map(|HeapEntry { rec, .. }| {
            let live = live_ids.contains(&rec.id);
            let preview = if include_preview {
                match &rec.request.input {
                    crate::api::openai::responses::types::ResponsesInput::Text(s) => {
                        truncate_preview(s)
                    }
                    crate::api::openai::responses::types::ResponsesInput::Items(items) => items
                        .first()
                        .and_then(|item| match item {
                            crate::api::openai::responses::types::InputItem::Typed(
                                crate::api::openai::responses::types::TypedInputItem::Message(m),
                            ) => match &m.content {
                                crate::api::openai::responses::types::InputMessageContent::Text(
                                    t,
                                ) => Some(truncate_preview(t)),
                                _ => None,
                            },
                            _ => None,
                        })
                        .unwrap_or_default(),
                }
            } else {
                String::new()
            };
            let output_preview = if include_preview {
                rec.response.output_text.as_deref().map(truncate_preview)
            } else {
                None
            };
            serde_json::json!({
                "id": rec.id,
                "created_at": rec.created_at,
                "expires_at": rec.expires_at,
                "model": rec.response.model,
                "status": rec.response.status,
                "background": rec.response.background.unwrap_or(false),
                "live": live,
                "input_preview": preview,
                "output_text_preview": output_preview,
                "usage": {
                    "input_tokens": rec.response.usage.input_tokens,
                    "output_tokens": rec.response.usage.output_tokens,
                },
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
        "total": data.len(),
    })))
}

fn truncate_preview(s: &str) -> String {
    const MAX: usize = 120;
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let mut out = String::with_capacity(MAX);
    for (i, ch) in s.chars().enumerate() {
        if i >= MAX {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}
