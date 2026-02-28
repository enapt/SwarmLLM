use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::identity::nickname::{
    display_name, IdentityPrefs, NicknameRecord, NicknameStore, VisibilityMode,
};
use crate::types::{NetworkCommand, NicknameGossip, SwarmMessage};

/// GET /api/identity/nickname — get this node's nickname & prefs.
pub async fn get_nickname(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = NicknameStore::new(state.db.clone());
    let prefs = store.get_prefs().map_err(ApiError::from)?;
    let node_id = state.shared_state.identity.node_id();
    let name = display_name(node_id, &state.shared_state.nickname_registry);

    Ok(Json(serde_json::json!({
        "node_id": format!("{node_id}"),
        "nickname": prefs.nickname,
        "visibility": prefs.visibility,
        "display_name": name,
    })))
}

#[derive(Deserialize)]
pub struct SetNicknameRequest {
    pub nickname: String,
    #[serde(default)]
    pub visibility: Option<VisibilityMode>,
}

/// PUT /api/identity/nickname — set/update nickname (signs + gossips).
pub async fn set_nickname(
    State(state): State<AppState>,
    Json(body): Json<SetNicknameRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let record = NicknameRecord::new_signed(&state.shared_state.identity, body.nickname.clone())
        .map_err(ApiError::from)?;

    // Persist locally
    let store = NicknameStore::new(state.db.clone());
    store.put_record(&record).map_err(ApiError::from)?;

    let visibility = body.visibility.unwrap_or(VisibilityMode::Nickname);
    let prefs = IdentityPrefs {
        nickname: Some(body.nickname),
        visibility,
    };
    store.put_prefs(&prefs).map_err(ApiError::from)?;

    // Update in-memory registry
    state
        .shared_state
        .nickname_registry
        .insert(record.node_id.clone(), record.clone());

    // Gossip to network
    if let Some(ref ntx) = state.network_tx {
        let msg = SwarmMessage::NicknameGossip(NicknameGossip {
            record: record.clone(),
        });
        let _ = ntx.send(NetworkCommand::Broadcast(msg)).await;
    }

    let node_id = state.shared_state.identity.node_id();
    let name = display_name(node_id, &state.shared_state.nickname_registry);

    Ok(Json(serde_json::json!({
        "status": "ok",
        "display_name": name,
    })))
}

/// DELETE /api/identity/nickname — go anonymous.
pub async fn delete_nickname(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let store = NicknameStore::new(state.db.clone());
    let node_id = state.shared_state.identity.node_id();

    // Remove from DB and registry
    store.remove_record(node_id).map_err(ApiError::from)?;
    state.shared_state.nickname_registry.remove(node_id);

    // Reset prefs to anonymous
    let prefs = IdentityPrefs::default();
    store.put_prefs(&prefs).map_err(ApiError::from)?;

    Ok(Json(serde_json::json!({
        "status": "ok",
        "display_name": format!("{node_id}"),
    })))
}

#[derive(Deserialize)]
pub struct LeaderboardQuery {
    #[serde(default = "default_limit")]
    pub limit: usize,
}

fn default_limit() -> usize {
    50
}

/// GET /api/identity/leaderboard?limit=50 — top N peers by credits.
pub async fn leaderboard(
    State(state): State<AppState>,
    Query(query): Query<LeaderboardQuery>,
) -> Json<serde_json::Value> {
    let limit = query.limit.min(200);

    // Gather all known peers with their credit info
    let mut entries: Vec<serde_json::Value> = Vec::new();

    // Add self
    let self_id = state.shared_state.identity.node_id();
    let self_credit = state.shared_state.credit_balance.read().await;
    let self_name = display_name(self_id, &state.shared_state.nickname_registry);
    let self_tier = crate::credit::priority::PriorityCalculator::tier_name(self_credit.balance);
    entries.push(serde_json::json!({
        "node_id": format!("{self_id}"),
        "display_name": self_name,
        "credits": self_credit.balance,
        "tier": self_tier,
    }));

    // Add known peers — estimate tier from trust_score as a proxy for credit standing.
    // Per-peer credit balances aren't tracked in SharedState, so we derive a rough
    // estimate: high trust_score peers likely have positive balances.
    for peer in state.shared_state.peer_registry.iter() {
        let peer_name = display_name(&peer.node_id, &state.shared_state.nickname_registry);
        // Estimate credit bucket from trust_score (0.0-1.0 → mapped to balance range)
        let estimated_balance = (peer.trust_score * 5000.0) as i64;
        let peer_tier =
            crate::credit::priority::PriorityCalculator::tier_name(estimated_balance);
        entries.push(serde_json::json!({
            "node_id": format!("{}", peer.node_id),
            "display_name": peer_name,
            "credits": estimated_balance,
            "tier": peer_tier,
        }));
    }

    // Sort by credits descending
    entries.sort_by(|a, b| {
        let ca = a["credits"].as_i64().unwrap_or(0);
        let cb = b["credits"].as_i64().unwrap_or(0);
        cb.cmp(&ca)
    });
    entries.truncate(limit);

    // Add rank
    for (i, entry) in entries.iter_mut().enumerate() {
        if let Some(obj) = entry.as_object_mut() {
            obj.insert("rank".into(), serde_json::json!(i + 1));
        }
    }

    Json(serde_json::json!({ "leaderboard": entries }))
}

/// GET /api/identity/peers — all peers with display names.
pub async fn peers_with_names(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut peers: Vec<serde_json::Value> = Vec::new();

    for peer in state.shared_state.peer_registry.iter() {
        let name = display_name(&peer.node_id, &state.shared_state.nickname_registry);
        peers.push(serde_json::json!({
            "node_id": format!("{}", peer.node_id),
            "display_name": name,
            "last_seen": peer.last_seen.to_rfc3339(),
            "addresses": peer.addresses,
        }));
    }

    Json(serde_json::json!({ "peers": peers }))
}
