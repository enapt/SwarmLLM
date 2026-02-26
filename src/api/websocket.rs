use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures::{SinkExt, StreamExt};

use crate::api::server::AppState;
use crate::daemon::SharedState;

/// GET /api/admin/ws — WebSocket handler for real-time dashboard updates.
pub async fn handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state.shared_state.clone()))
}

async fn handle_socket(socket: WebSocket, shared_state: Arc<SharedState>) {
    let (mut sender, mut receiver) = socket.split();

    // Spawn a task to push stats every 2 seconds
    let push_state = shared_state.clone();
    let push_task = tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        loop {
            interval.tick().await;
            let msg = build_stats_message(&push_state).await;
            if sender.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Read incoming messages (keep-alive; clients don't send much)
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Close(_)) | Err(_) => break,
            _ => {} // Ignore other messages
        }
    }

    push_task.abort();
    tracing::debug!("WebSocket client disconnected");
}

async fn build_stats_message(state: &SharedState) -> String {
    let stats = state.node_stats.read().await;
    let credit = state.credit_balance.read().await;

    // Collect active acquisition progress
    let acquisitions: Vec<serde_json::Value> = state
        .acquisition_progress
        .iter()
        .map(|entry| serde_json::to_value(entry.value()).unwrap_or_default())
        .collect();

    let msg = serde_json::json!({
        "type": "stats_update",
        "data": {
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
        }
    });

    serde_json::to_string(&msg).unwrap_or_default()
}
