use crate::api::server::JsonBody;
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
    JsonBody(body): JsonBody<SetNicknameRequest>,
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

/// Public, display-only summary of a node's hardware and standing.
///
/// Everything here is already broadcast to the whole network in
/// `NodeCapability` — this surfaces what peers are told anyway, it does not
/// widen what is shared. Deliberately omits anything that would narrow a node
/// to a person or machine: no addresses, no OS build strings, no disk layout.
///
/// Every field is optional and renders as "unknown" when absent rather than
/// being guessed, because a node that predates a field must not be shown a
/// fabricated value (the 2026-07-21 report chased an invented `+2500` balance
/// that was never a ledger figure).
fn capability_summary(cap: Option<&crate::types::NodeCapability>) -> serde_json::Value {
    let Some(c) = cap else {
        return serde_json::json!({ "known": false });
    };
    serde_json::json!({
        "known": true,
        "gpu": c.gpu.as_ref().map(|g| serde_json::json!({
            "name": g.name,
            "vram_mb": g.vram_total_mb,
        })),
        // The single field a "GPU or CPU?" filter keys on, so the frontend
        // never has to re-derive it from a nullable nested object.
        "accelerator": if c.gpu.is_some() { "gpu" } else { "cpu" },
        "os": c.os.clone(),
        "region": c.region.clone(),
        "ram_total_mb": c.ram_total_mb,
        "est_tokens_per_sec_7b": c.est_tokens_per_sec_7b,
        "uptime_seconds": c.uptime_seconds,
        "version": c.version,
        "shards_hosted": c.hosted_shards.len(),
        "relay_capable": c.relay_capable,
    })
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

    // Add self (always included).
    //
    // This entry is built here rather than in the peer loop above, which is why
    // dropping `credits`/`tier` from that loop was not enough on its own — the
    // node's OWN row kept publishing both. Caught by calling the endpoint, not
    // by reading the change.
    let self_id = state.shared_state.identity.node_id();
    let self_name = display_name(self_id, &state.shared_state.nickname_registry);
    entries.push(serde_json::json!({
        "node_id": format!("{self_id}"),
        "display_name": self_name,
        "trust_score": 1.0,
        "eligible": true,
        "is_self": true,
        "capability": capability_summary(state.shared_state.local_capability.load().as_deref()),
    }));

    // Add known peers — use gossiped credit balances when available
    for peer in state.shared_state.peer_registry.iter() {
        let eligible =
            is_leaderboard_eligible(peer.first_seen, peer.verified_transaction_count, peer_count);
        if !eligible {
            continue;
        }

        let peer_name = display_name(&peer.node_id, &state.shared_state.nickname_registry);
        // No balance is read here any more — the leaderboard neither ranks by
        // credits nor publishes them (`docs/CREDITS_DESIGN.md`).
        //
        // Worth keeping the reason the old lookup was written the way it was,
        // for whoever restores it: only a GOSSIPED balance was ever real. It
        // once fell back to `trust_score * 5000.0` when none had been received,
        // which at the DEFAULT_TRUST of 0.5 rendered a confident "+2500
        // credits" for every peer we knew nothing about. The 2026-07-21 bug
        // report chased that number as a ledger inconsistency: one node showed
        // itself at -90 while its peer displayed +2500 for it. Neither figure
        // was wrong about the ledger; the +2500 was never a ledger figure at
        // all. A number the user cannot distinguish from a real balance must
        // not be invented.
        entries.push(serde_json::json!({
            "node_id": format!("{}", peer.node_id),
            "display_name": peer_name,
            // `credits` and `tier` were published here until 2026-08-17.
            // They are gone rather than merely hidden in the UI: the figure is
            // self-minted, so publishing it invites anyone reading the API to
            // treat it as a contribution measure (`docs/CREDITS_DESIGN.md`).
            "trust_score": peer.trust_score,
            "eligible": true,
            "is_self": false,
            "capability": capability_summary(peer.capability.as_ref()),
        }));
    }

    // Sort by shards hosted, descending.
    //
    // This ranked by credits until 2026-08-17. That put a 1-2-3 podium on a
    // number every node mints for itself and reconciles with nobody — a node
    // that had only ever served ITSELF outranked one that had served the swarm
    // (`docs/CREDITS_DESIGN.md` § 1). Shards hosted is not a perfect measure of
    // contribution either, but it corresponds to something real: a node cannot
    // claim a shard it does not hold and then serve requests for it.
    //
    // Ties break on trust score, so an established peer sorts above a brand new
    // one holding the same count instead of the order being arbitrary.
    entries.sort_by(|a, b| {
        let shards = |e: &serde_json::Value| e["capability"]["shards_hosted"].as_u64().unwrap_or(0);
        shards(b).cmp(&shards(a)).then_with(|| {
            b["trust_score"]
                .as_f64()
                .unwrap_or(0.0)
                .partial_cmp(&a["trust_score"].as_f64().unwrap_or(0.0))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
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
