use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures::{SinkExt, StreamExt};

use crate::api::server::AppState;
use crate::daemon::SharedState;
use crate::model::acquisition::ShardState;

/// GET /api/admin/ws — WebSocket handler for real-time dashboard updates.
///
/// Validates the Origin header to prevent cross-site WebSocket hijacking.
/// Only connections from the same host (localhost) are accepted.
pub async fn handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
) -> impl IntoResponse {
    // Validate Origin header for ALL connections (including loopback) to prevent
    // DNS rebinding attacks where a malicious page rebinds to 127.0.0.1 and opens
    // a WebSocket to exfiltrate dashboard data.
    // Missing Origin is allowed (non-browser clients like CLIs don't send it).
    if let Some(origin) = headers.get(axum::http::header::ORIGIN) {
        let origin_str = origin.to_str().unwrap_or("");
        let port = state.config.node.listen_port;
        let allowed = [
            format!("http://localhost:{port}"),
            format!("http://127.0.0.1:{port}"),
        ];
        if !allowed.iter().any(|a| a == origin_str) {
            tracing::warn!(
                origin = %origin_str,
                remote = %addr,
                "WebSocket connection rejected: origin not allowed"
            );
            return axum::http::StatusCode::FORBIDDEN.into_response();
        }
    }
    let shared = state.shared_state.clone();
    ws.on_upgrade(move |socket| handle_socket(socket, shared))
}

/// RAII guard to ensure ws_connection_count is decremented even on panic.
struct WsCountGuard(Arc<SharedState>);
impl Drop for WsCountGuard {
    fn drop(&mut self) {
        self.0
            .ws_connection_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

async fn handle_socket(socket: WebSocket, shared_state: Arc<SharedState>) {
    // Enforce a global WebSocket connection limit to prevent resource exhaustion.
    // Incrementing inside handle_socket (not before on_upgrade) prevents counter
    // leaks when the HTTP upgrade itself fails.
    const MAX_WS_CONNECTIONS: usize = 100;
    let current = shared_state
        .ws_connection_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if current >= MAX_WS_CONNECTIONS {
        shared_state
            .ws_connection_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        tracing::warn!(
            current = current,
            max = MAX_WS_CONNECTIONS,
            "WebSocket connection limit reached — dropping"
        );
        return;
    }
    // RAII guard ensures decrement on panic or any exit path
    let _ws_guard = WsCountGuard(shared_state.clone());
    tracing::debug!("DIAG: websocket client connected");
    let (mut sender, mut receiver) = socket.split();

    // Track last pong timestamp for dead connection detection
    let last_pong = std::sync::Arc::new(tokio::sync::Mutex::new(tokio::time::Instant::now()));
    let last_pong_push = last_pong.clone();

    // Spawn a task to push stats every 2 seconds + ping every 30 seconds + prune/LAN events
    let push_state = shared_state.clone();
    let mut prune_rx = shared_state.prune_events_tx.subscribe();
    let mut lan_rx = shared_state.lan_discovery_tx.subscribe();
    let mut update_rx = shared_state.update_tx.subscribe();
    let mut models_changed_rx = shared_state.models_changed_tx.subscribe();
    let mut system_rx = shared_state.system_notify_tx.subscribe();
    let mut peer_list_rx = shared_state.peer_list_changed_tx.subscribe();
    let mut activity_rx = shared_state.activity_tx.subscribe();
    let mut push_task = tokio::spawn(async move {
        // Send the current peer list immediately on connect so the dashboard
        // populates without waiting for the first peer_list_changed event.
        let initial_peers = build_peer_list_message(&push_state);
        let _ = sender
            .send(Message::Text(
                serde_json::to_string(&initial_peers).unwrap_or_default(),
            ))
            .await;

        let mut stats_interval = tokio::time::interval(Duration::from_secs(2));
        let mut ping_interval = tokio::time::interval(Duration::from_secs(30));
        // Track previous shard registry snapshot for change detection
        let mut prev_shard_snapshot: HashMap<String, Vec<ShardSnapshot>> = HashMap::new();
        let mut tick_count: u64 = 0;
        loop {
            tokio::select! {
                _ = stats_interval.tick() => {
                    let msg = build_stats_message(&push_state, &mut prev_shard_snapshot).await;
                    if sender.send(Message::Text(msg)).await.is_err() {
                        break;
                    }
                }
                _ = ping_interval.tick() => {
                    tick_count += 1;
                    // Skip only the first tick (fires immediately)
                    if tick_count == 1 {
                        continue;
                    }
                    // Check if last pong was received within a reasonable window.
                    // Ping fires every 30s; allow up to 35s since last pong to detect
                    // dead connections within ~1 ping cycle instead of ~2.
                    let last = *last_pong_push.lock().await;
                    if last.elapsed() > Duration::from_secs(35) {
                        tracing::debug!("WebSocket client failed pong check — closing");
                        break;
                    }
                    let ping_data = chrono::Utc::now().timestamp().to_le_bytes().to_vec();
                    if sender.send(Message::Ping(ping_data)).await.is_err() {
                        break;
                    }
                }
                event = prune_rx.recv() => {
                    if let Ok(event) = event {
                        let msg = serde_json::json!({
                            "type": "prune_event",
                            "data": {
                                "model_id": event.model_id.0,
                                "model_name": event.model_name,
                                "shard_index": event.shard_index,
                                "reason": event.reason,
                                "freed_bytes": event.freed_bytes,
                                "remaining_local_shards": event.remaining_local_shards,
                                "holder_count_before": event.holder_count_before,
                                "holder_count_after": event.holder_count_after,
                                "timestamp": event.timestamp.to_rfc3339(),
                            }
                        });
                        let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                        if sender.send(Message::Text(msg_str)).await.is_err() {
                            break;
                        }
                    }
                }
                count = lan_rx.recv() => {
                    if let Ok(count) = count {
                        let msg = serde_json::json!({
                            "type": "lan_peer_discovered",
                            "data": {
                                "peer_count": count,
                            }
                        });
                        let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                        if sender.send(Message::Text(msg_str)).await.is_err() {
                            break;
                        }
                    }
                }
                _ = models_changed_rx.recv() => {
                    let msg = serde_json::json!({
                        "type": "models_changed",
                    });
                    let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                    if sender.send(Message::Text(msg_str)).await.is_err() {
                        break;
                    }
                }
                notification = system_rx.recv() => {
                    if let Ok(notif) = notification {
                        let msg = serde_json::json!({
                            "type": "system_notification",
                            "data": notif,
                        });
                        let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                        if sender.send(Message::Text(msg_str)).await.is_err() {
                            break;
                        }
                    }
                }
                _ = peer_list_rx.recv() => {
                    let peers = build_peer_list_message(&push_state);
                    let msg_str = serde_json::to_string(&peers).unwrap_or_default();
                    if sender.send(Message::Text(msg_str)).await.is_err() {
                        break;
                    }
                }
                update_info = update_rx.recv() => {
                    if let Ok(info) = update_info {
                        let msg = serde_json::json!({
                            "type": "update_available",
                            "data": {
                                "current_version": info.current_version,
                                "latest_version": info.latest_version,
                                "changelog": info.changelog,
                                "published_at": info.published_at,
                                "downloaded": info.downloaded,
                            }
                        });
                        let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                        if sender.send(Message::Text(msg_str)).await.is_err() {
                            break;
                        }
                    }
                }
                activity = activity_rx.recv() => {
                    if let Ok(event) = activity {
                        let msg = serde_json::json!({
                            "type": "activity_event",
                            "data": event,
                        });
                        let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                        if sender.send(Message::Text(msg_str)).await.is_err() {
                            break;
                        }
                    }
                }
            }
        }
    });

    // Race push_task against receiver loop — either side exiting cleans up the other.
    // Without this, if push_task exits first (sender dropped), the receiver loop blocks
    // indefinitely waiting for the client to send a close frame.
    let recv_loop = async {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(Message::Pong(_)) => {
                    *last_pong.lock().await = tokio::time::Instant::now();
                }
                _ => {}
            }
        }
    };

    tokio::select! {
        _ = &mut push_task => {
            tracing::debug!("DIAG: websocket push_task exited first");
        }
        _ = recv_loop => {
            push_task.abort();
            tracing::debug!("DIAG: websocket receiver loop exited first");
        }
    }
    // _ws_guard drop handles decrement
    tracing::debug!("DIAG: websocket client disconnected");
}

/// Build a `peer_list` WS message with the current peer registry snapshot.
/// Same data shape as GET /api/admin/peers so the frontend can share one render path.
fn build_peer_list_message(state: &SharedState) -> serde_json::Value {
    let timeout = chrono::Duration::seconds(90);
    let now = chrono::Utc::now();

    let peers: Vec<serde_json::Value> = state
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
                .nickname_registry
                .get(&peer.node_id)
                .map(|r| r.nickname.clone());
            serde_json::json!({
                "node_id": format!("{}", peer.node_id),
                "nickname": nickname,
                "latency_ms": peer.latency_ms,
                "trust_score": peer.trust_score,
                "healthy": healthy,
                "gpu": peer.capability.as_ref().and_then(|c| c.gpu.as_ref().map(|g| &g.name)),
                "hosted_models": hosted_models,
                "is_lan_peer": peer.is_lan_peer,
            })
        })
        .collect();

    serde_json::json!({
        "type": "peer_list",
        "data": { "peers": peers }
    })
}

/// Lightweight snapshot of a shard's holder state for change detection.
#[derive(Clone, PartialEq)]
struct ShardSnapshot {
    index: u32,
    local: bool,
    holder_count: usize,
}

async fn build_stats_message(
    state: &SharedState,
    prev_shard_snapshot: &mut HashMap<String, Vec<ShardSnapshot>>,
) -> String {
    let stats = state.node_stats.read().await;
    let credit = state.credit_balance.read().await;
    let local_node_id = state.identity.node_id().clone();

    // Collect active acquisition progress with per-shard detail
    let acquisitions: Vec<serde_json::Value> = state
        .acquisition_progress
        .iter()
        .map(|entry| {
            let status = entry.value();
            let shard_details: Vec<serde_json::Value> = status
                .shard_progress
                .iter()
                .map(|(idx, sp)| {
                    let pct = if sp.total_bytes > 0 {
                        ((sp.downloaded_bytes as f64 / sp.total_bytes as f64) * 100.0) as u32
                    } else {
                        0
                    };
                    let state_str = match &sp.state {
                        ShardState::Pending => "pending",
                        ShardState::Downloading => "downloading",
                        ShardState::Verifying => "verifying",
                        ShardState::Complete => "complete",
                        ShardState::Failed => "failed",
                    };
                    serde_json::json!({
                        "index": idx,
                        "state": state_str,
                        "progress_pct": pct,
                        "downloaded_bytes": sp.downloaded_bytes,
                        "total_bytes": sp.total_bytes,
                    })
                })
                .collect();

            let model_name = state
                .model_registry
                .get_manifest(&status.model_id)
                .map(|m| m.name.clone())
                .unwrap_or_else(|| status.model_id.0.clone());
            let source = if state.hf_sources.contains_key(&status.model_id) {
                "huggingface"
            } else {
                "network"
            };
            let cancellable = matches!(
                status.state,
                crate::model::acquisition::AcquisitionState::Downloading
                    | crate::model::acquisition::AcquisitionState::AwaitingManifest
            );
            let overall_pct = if status.total_bytes > 0 {
                ((status.downloaded_bytes as f64 / status.total_bytes as f64) * 100.0) as u32
            } else {
                0
            };
            let eta_secs = if status.speed_bytes_per_sec > 0
                && status.total_bytes > status.downloaded_bytes
            {
                Some((status.total_bytes - status.downloaded_bytes) / status.speed_bytes_per_sec)
            } else {
                None
            };
            serde_json::json!({
                "model_id": status.model_id.0,
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
            })
        })
        .collect();

    // Build shard registry snapshot — only include if changed from previous tick
    let mut current_snapshot: HashMap<String, Vec<ShardSnapshot>> = HashMap::new();
    for (shard_id, holders) in state.model_registry.all_shard_entries() {
        let model_id = shard_id.model_id.0.clone();
        let local = holders.contains(&local_node_id);
        current_snapshot
            .entry(model_id)
            .or_default()
            .push(ShardSnapshot {
                index: shard_id.index,
                local,
                holder_count: holders.len(),
            });
    }

    // Only include shard_registry in the message if it changed
    let registry_changed = current_snapshot != *prev_shard_snapshot;
    let shard_registry_val = if registry_changed {
        let registry: serde_json::Value = current_snapshot
            .iter()
            .map(|(model_id, shards)| {
                let shard_arr: Vec<serde_json::Value> = shards
                    .iter()
                    .map(|s| {
                        // Compute in_vram: local AND model is loaded with this shard in the window
                        let mid = crate::types::ModelId(model_id.clone());
                        let in_vram = if s.local {
                            let window = state.model_process_pool.get_shard_window(&mid);
                            match window {
                                Some(w) => w.contains(&s.index),
                                None => {
                                    state.model_process_pool.is_loaded(&mid)
                                        || state.split_models.iter().any(|e| e.key().0 == mid)
                                }
                            }
                        } else {
                            false
                        };
                        serde_json::json!({
                            "index": s.index,
                            "local": s.local,
                            "holders": s.holder_count,
                            "in_vram": in_vram,
                        })
                    })
                    .collect();
                (model_id.clone(), serde_json::Value::Array(shard_arr))
            })
            .collect::<serde_json::Map<String, serde_json::Value>>()
            .into();
        *prev_shard_snapshot = current_snapshot;
        Some(registry)
    } else {
        None
    };

    let lan_peers = state
        .lan_peer_count
        .load(std::sync::atomic::Ordering::Relaxed);

    // Use peer_registry.len() as authoritative peer count — DashMap never
    // has contention issues, unlike node_stats.peers_connected which uses
    // try_write() and can silently skip updates during connection bursts.
    let peers_connected = state.peer_registry.len() as u32;

    let mut data = serde_json::json!({
        "peers": peers_connected,
        "lan_peers": lan_peers,
        "credits": {
            "balance": credit.balance,
            "lifetime_earned": credit.lifetime_earned,
            "lifetime_spent": credit.lifetime_spent,
        },
        "active_requests": state.active_pipelines.len(),
        "requests_served": stats.requests_served,
        "requests_made": stats.requests_made,
        "forwards_served": stats.forwards_served,
        "uptime_seconds": (chrono::Utc::now() - stats.uptime_start).num_seconds(),
        "acquisitions": acquisitions,
    });

    if let Some(registry) = shard_registry_val {
        data["shard_registry"] = registry;
    }

    // Peer shard download progress (from gossip)
    {
        let mut peer_dl: Vec<serde_json::Value> = Vec::new();
        for entry in state.peer_shard_downloads.iter() {
            let shard_id = entry.key();
            for (nid, pct) in entry.value().iter() {
                peer_dl.push(serde_json::json!({
                    "model_id": shard_id.model_id.0,
                    "shard_index": shard_id.index,
                    "node_id": format!("{}", nid),
                    "progress_pct": pct,
                }));
            }
        }
        if !peer_dl.is_empty() {
            data["peer_downloads"] = serde_json::json!(peer_dl);
        }
    }

    // Region summary for network map — includes demand rates and coverage gaps
    {
        let mut region_counts: HashMap<String, u64> = HashMap::new();
        if let Some(ref region) = state.config.identity.region {
            *region_counts.entry(region.to_uppercase()).or_insert(0) += 1;
        }
        for peer in state.peer_registry.iter() {
            if let Some(ref cap) = peer.value().capability {
                if let Some(ref region) = cap.region {
                    *region_counts.entry(region.to_uppercase()).or_insert(0) += 1;
                }
            }
        }
        if !region_counts.is_empty() {
            data["region_summary"] = serde_json::to_value(&region_counts).unwrap_or_default();
        }

        // Per-region demand rates
        let mut region_demand: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for entry in state.region_demand.iter() {
            let (model_id, region) = entry.key();
            region_demand
                .entry(region.clone())
                .or_default()
                .insert(model_id.0.clone(), *entry.value());
        }
        if !region_demand.is_empty() {
            data["region_demand"] = serde_json::to_value(&region_demand).unwrap_or_default();
        }

        // Coverage gaps: per-region list of models with 0 holders
        let all_model_ids: Vec<String> = state
            .model_registry
            .models()
            .iter()
            .map(|m| m.id.0.clone())
            .collect();
        if !all_model_ids.is_empty() && !region_counts.is_empty() {
            let mut gaps: HashMap<String, Vec<String>> = HashMap::new();
            for region_code in region_counts.keys() {
                let mut region_gaps = Vec::new();
                for model_id in &all_model_ids {
                    let key = (region_code.clone(), crate::types::ModelId(model_id.clone()));
                    let has_holders = state
                        .region_shard_summaries
                        .get(&key)
                        .map(|s| s.shard_counts.iter().any(|(_, c)| *c > 0))
                        .unwrap_or(false);
                    if !has_holders {
                        region_gaps.push(model_id.clone());
                    }
                }
                if !region_gaps.is_empty() {
                    gaps.insert(region_code.clone(), region_gaps);
                }
            }
            if !gaps.is_empty() {
                data["region_coverage_gaps"] = serde_json::to_value(&gaps).unwrap_or_default();
            }
        }
    }

    let msg = serde_json::json!({
        "type": "stats_update",
        "data": data,
    });

    serde_json::to_string(&msg).unwrap_or_default()
}
