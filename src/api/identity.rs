use axum::extract::{Query, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::identity::nickname::{
    display_name, IdentityPrefs, NicknameRecord, NicknameRecordExt, NicknameStore, VisibilityMode,
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

    let broadcast = matches!(visibility, VisibilityMode::Nickname);

    if broadcast {
        // Update in-memory registry (peers view this node's nickname).
        state
            .shared_state
            .nickname_registry
            .insert(record.node_id.clone(), record.clone());
    } else {
        // Anonymous: retract any previously-broadcast self-record so the local
        // display also falls back to the node id.
        state.shared_state.nickname_registry.remove(&record.node_id);
    }

    tracing::debug!(
        nickname = %record.nickname,
        broadcast,
        "DIAG: set_nickname persisted"
    );

    // Gossip to network only when visibility allows it.
    if broadcast {
        if let Some(ref ntx) = state.network_tx {
            let msg = SwarmMessage::NicknameGossip(NicknameGossip {
                record: record.clone(),
            });
            let _ = ntx.send(NetworkCommand::Broadcast(msg)).await;
        }
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

/// Minimum age (days) for anti-spoofing on large networks (>20 peers).
const MIN_LIFETIME_DAYS: u64 = 7;
/// Minimum verified dual-signed transactions for anti-spoofing on large networks.
const MIN_VERIFIED_TRANSACTIONS: u32 = 10;
/// Below this peer count, all known peers appear on the leaderboard.
const SMALL_NETWORK_THRESHOLD: usize = 20;
/// Upper bound on `?limit=` for the leaderboard query. Prevents clients from
/// requesting oversized responses on large networks.
const MAX_LEADERBOARD_LIMIT: usize = 200;

/// Check if a peer is eligible for the leaderboard based on anti-spoofing rules.
/// On small networks (<20 peers), all peers are shown. On larger networks,
/// peers must meet age and transaction thresholds to prevent sybil manipulation.
fn is_leaderboard_eligible(first_seen: u64, verified_tx_count: u32, peer_count: usize) -> bool {
    if peer_count < SMALL_NETWORK_THRESHOLD {
        return true;
    }
    let now_ts = crate::types::unix_now_secs();
    const SECS_PER_DAY: u64 = 86_400;
    let age_days = if first_seen > 0 {
        (now_ts.saturating_sub(first_seen)) / SECS_PER_DAY
    } else {
        0
    };
    age_days >= MIN_LIFETIME_DAYS && verified_tx_count >= MIN_VERIFIED_TRANSACTIONS
}

/// GET /api/identity/leaderboard?limit=50 — top N peers by credits.
///
/// Shows all peers on small networks. On large networks (20+), applies anti-spoofing
/// filters (7-day age, 10 verified transactions). Uses gossiped credit balances.
pub async fn leaderboard(
    State(state): State<AppState>,
    Query(query): Query<LeaderboardQuery>,
) -> Json<serde_json::Value> {
    // Clamp to [1, MAX_LEADERBOARD_LIMIT]. Silent clamp rather than 400
    // so the dashboard doesn't have to special-case validation here; a
    // bottom of 1 prevents the surprising "limit=0 returns empty array".
    let limit = query.limit.clamp(1, MAX_LEADERBOARD_LIMIT);
    let peer_count = state.shared_state.peer_registry.len();

    tracing::debug!(peer_count, limit, "DIAG: leaderboard query");
    let mut entries: Vec<serde_json::Value> = Vec::new();

    // Add self (always included)
    let self_id = state.shared_state.identity.node_id();
    let (self_balance, self_tier) = {
        let self_credit = state.shared_state.credits.credit_balance.read().await;
        let balance = self_credit.balance;
        let tier = crate::credit::priority::PriorityCalculator::tier_name(balance);
        (balance, tier)
    };
    let self_name = display_name(self_id, &state.shared_state.nickname_registry);
    entries.push(serde_json::json!({
        "node_id": format!("{self_id}"),
        "display_name": self_name,
        "credits": self_balance,
        "tier": self_tier,
        "trust_score": 1.0,
        "eligible": true,
    }));

    // Add known peers — use gossiped credit balances when available
    for peer in state.shared_state.peer_registry.iter() {
        let eligible =
            is_leaderboard_eligible(peer.first_seen, peer.verified_transaction_count, peer_count);
        if !eligible {
            continue;
        }

        let peer_name = display_name(&peer.node_id, &state.shared_state.nickname_registry);
        // Only a gossiped balance is real. This used to fall back to
        // `trust_score * 5000.0` when no balance had been received — which,
        // at the DEFAULT_TRUST of 0.5, rendered a confident "+2500 credits"
        // for every peer we knew nothing about. The 2026-07-21 bug report
        // chased that number as a ledger inconsistency: one node showed
        // itself at -90 while its peer displayed +2500 for it. Neither
        // figure was wrong about the ledger; the +2500 was never a ledger
        // figure at all. A number the user cannot distinguish from a real
        // balance must not be invented.
        let balance = state
            .shared_state
            .credits
            .peer_credit_balances
            .get(&peer.node_id)
            .map(|v| *v);
        entries.push(serde_json::json!({
            "node_id": format!("{}", peer.node_id),
            "display_name": peer_name,
            // null = "we have not received this peer's balance gossip yet".
            // The dashboard renders it as an em dash rather than a number.
            "credits": balance,
            "balance_known": balance.is_some(),
            "tier": balance.map(crate::credit::priority::PriorityCalculator::tier_name),
            "trust_score": peer.trust_score,
            "eligible": true,
        }));
    }

    // Sort by credits descending. Peers with no known balance sort last
    // rather than being treated as zero — an unknown balance is not a claim
    // that the peer has nothing.
    entries.sort_by(
        |a, b| match (a["credits"].as_i64(), b["credits"].as_i64()) {
            (Some(ca), Some(cb)) => cb.cmp(&ca),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        },
    );
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
