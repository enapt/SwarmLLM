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
use crate::types::ShardId;

/// GET /api/admin/ws — WebSocket handler for real-time dashboard updates.
pub async fn handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state.shared_state.clone()))
}

async fn handle_socket(socket: WebSocket, shared_state: Arc<SharedState>) {
    let (mut sender, mut receiver) = socket.split();

    // Track last pong timestamp for dead connection detection
    let last_pong = std::sync::Arc::new(tokio::sync::Mutex::new(tokio::time::Instant::now()));
    let last_pong_push = last_pong.clone();

    // Spawn a task to push stats every 2 seconds + ping every 30 seconds
    let push_state = shared_state.clone();
    let push_task = tokio::spawn(async move {
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
                    // Check if last pong was within 10s of the last ping
                    let last = *last_pong_push.lock().await;
                    if last.elapsed() > Duration::from_secs(40) {
                        tracing::debug!("WebSocket client failed pong check — closing");
                        break;
                    }
                    let ping_data = chrono::Utc::now().timestamp().to_le_bytes().to_vec();
                    if sender.send(Message::Ping(ping_data)).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Read incoming messages (keep-alive + pong tracking)
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Close(_)) | Err(_) => break,
            Ok(Message::Pong(_)) => {
                *last_pong.lock().await = tokio::time::Instant::now();
            }
            _ => {} // Ignore other messages
        }
    }

    push_task.abort();
    tracing::debug!("WebSocket client disconnected");
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

            serde_json::json!({
                "model_id": status.model_id.0,
                "state": serde_json::to_value(&status.state).unwrap_or_default(),
                "total_shards": status.total_shards,
                "downloaded_shards": status.downloaded_shards,
                "verified_shards": status.verified_shards,
                "total_bytes": status.total_bytes,
                "downloaded_bytes": status.downloaded_bytes,
                "speed_bytes_per_sec": status.speed_bytes_per_sec,
                "shard_details": shard_details,
            })
        })
        .collect();

    // Build shard registry snapshot — only include if changed from previous tick
    let mut current_snapshot: HashMap<String, Vec<ShardSnapshot>> = HashMap::new();
    for entry in state.shard_registry.iter() {
        let shard_id: &ShardId = entry.key();
        let holders = entry.value();
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
                        serde_json::json!({
                            "index": s.index,
                            "local": s.local,
                            "holders": s.holder_count,
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

    let mut data = serde_json::json!({
        "peers": stats.peers_connected,
        "credits": {
            "balance": credit.balance,
            "lifetime_earned": credit.lifetime_earned,
            "lifetime_spent": credit.lifetime_spent,
        },
        "active_requests": state.active_pipelines.len(),
        "requests_served": stats.requests_served,
        "requests_made": stats.requests_made,
        "forwards_served": stats.forwards_served,
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

    // Region summary for network map
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
    }

    let msg = serde_json::json!({
        "type": "stats_update",
        "data": data,
    });

    serde_json::to_string(&msg).unwrap_or_default()
}
