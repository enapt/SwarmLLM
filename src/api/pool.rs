use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::config::CreditRateConfig;
use crate::error::{ApiError, SwarmError};
use crate::pool::types::PoolCommand;
use crate::types::NodeId;

/// Await a raw oneshot from the pool manager. Channel drop = ServiceUnavailable.
async fn await_pool_recv<T>(rx: tokio::sync::oneshot::Receiver<T>) -> Result<T, ApiError> {
    rx.await.map_err(|_| {
        ApiError(SwarmError::ServiceUnavailable(
            "Pool manager unavailable".into(),
        ))
    })
}

/// Await a Result-bearing oneshot reply from the pool manager.
async fn await_pool_reply<T>(
    rx: tokio::sync::oneshot::Receiver<Result<T, SwarmError>>,
) -> Result<T, ApiError> {
    await_pool_recv(rx).await?.map_err(ApiError)
}

/// GET /api/pool/state — Get current pool state.
pub async fn pool_state(State(state): State<AppState>) -> Json<serde_json::Value> {
    let pool_state = state.shared_state.credits.pool_state.read().await;
    match pool_state.as_ref() {
        Some(ps) => Json(serde_json::json!({
            "in_pool": true,
            "pool_id": hex::encode(ps.pool_id.0),
            "name": ps.name,
            "members": ps.members.iter().map(|m| {
                let mut member = serde_json::json!({
                    "node_id": hex::encode(m.node_id.0),
                    "credits_contributed": m.credits_contributed,
                    "joined_at": m.joined_at.to_rfc3339(),
                    "online": m.online,
                    "contribution_level": m.contribution_level,
                });
                if let Some(ref name) = m.device_name {
                    member["device_name"] = serde_json::json!(name);
                }
                if let Some(ref last) = m.last_seen {
                    member["last_seen"] = serde_json::json!(last.to_rfc3339());
                }
                if let Some(ref stats) = m.device_stats {
                    member["stats"] = serde_json::json!({
                        "forwards_served": stats.forwards_served,
                        "requests_served": stats.requests_served,
                        "shards_hosted": stats.shards_hosted,
                        "vram_mb": stats.vram_mb,
                        "ram_mb": stats.ram_mb,
                        "uptime_secs": stats.uptime_secs,
                        "models_hosted": stats.models_hosted,
                    });
                }
                member
            }).collect::<Vec<_>>(),
            "created_at": ps.created_at.to_rfc3339(),
            "total_lifetime_credits": ps.total_lifetime_credits,
            "member_credit_split_pct": ps.member_credit_split_pct,
            "private_mode": state.shared_state.credits.private_mode.load(std::sync::atomic::Ordering::Relaxed),
        })),
        None => Json(serde_json::json!({
            "in_pool": false,
        })),
    }
}

/// POST /api/pool/create — Create a new device pool (this node becomes owner).
pub async fn pool_create(
    State(state): State<AppState>,
    Json(body): Json<PoolCreateRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::debug!(name = %body.name, "DIAG: pool_create request");
    // Validate pool name length
    if body.name.trim().is_empty() || body.name.len() > 64 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Pool name must be 1-64 characters".into(),
        )));
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(
        &state,
        PoolCommand::CreatePool {
            name: body.name,
            reply: tx,
        },
    )
    .await?;

    let ps = await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({
        "status": "created",
        "pool_id": hex::encode(ps.pool_id.0),
        "name": ps.name,
    })))
}

/// POST /api/pool/invite — Invite a node to the pool.
pub async fn pool_invite(
    State(state): State<AppState>,
    Json(body): Json<PoolInviteRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let node_id = parse_node_id(&body.node_id)?;
    tracing::debug!(invitee = %node_id, "DIAG: pool_invite request");
    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(
        &state,
        PoolCommand::CreateInvitation {
            invitee: node_id,
            reply: tx,
        },
    )
    .await?;

    let inv = await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({
        "status": "invited",
        "invitation_id": inv.id.to_string(),
        "expires_at": inv.expires_at.to_rfc3339(),
    })))
}

/// POST /api/pool/accept — Accept a pool invitation.
pub async fn pool_accept(
    State(state): State<AppState>,
    Json(body): Json<PoolAcceptRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let invitation_id = uuid::Uuid::parse_str(&body.invitation_id).map_err(|_| {
        ApiError(crate::error::SwarmError::Validation(
            "Invalid invitation ID".into(),
        ))
    })?;

    // Look up the invitation from the pool manager
    let (inv_tx, inv_rx) = tokio::sync::oneshot::channel();
    send_pool_command(&state, PoolCommand::GetInvitations { reply: inv_tx }).await?;

    let invitations = await_pool_recv(inv_rx).await?;

    let invitation = invitations
        .into_iter()
        .find(|i| i.id == invitation_id)
        .ok_or_else(|| {
            ApiError(crate::error::SwarmError::Validation(
                "Invitation not found or expired".into(),
            ))
        })?;

    // Check expiry in the API layer before forwarding to pool manager
    if invitation.expires_at < chrono::Utc::now() {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Invitation expired at {}",
            invitation.expires_at.to_rfc3339()
        ))));
    }

    let (tx, rx) = tokio::sync::oneshot::channel();
    send_pool_command(
        &state,
        PoolCommand::AcceptInvitation {
            invitation,
            reply: tx,
        },
    )
    .await?;

    await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({ "status": "accepted" })))
}

/// POST /api/pool/remove — Remove a member from the pool (owner only).
pub async fn pool_remove(
    State(state): State<AppState>,
    Json(body): Json<PoolRemoveRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let node_id = parse_node_id(&body.node_id)?;
    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(&state, PoolCommand::RemoveMember { node_id, reply: tx }).await?;

    await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({ "status": "removed" })))
}

/// POST /api/pool/leave — Leave the current pool.
pub async fn pool_leave(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(&state, PoolCommand::LeavePool { reply: tx }).await?;

    await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({ "status": "left" })))
}

/// GET /api/pool/invitations — List pending invitations for this node.
pub async fn pool_invitations(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(&state, PoolCommand::GetInvitations { reply: tx }).await?;

    let invitations = await_pool_recv(rx).await?;

    let values: Vec<serde_json::Value> = invitations
        .into_iter()
        .map(|inv| {
            serde_json::json!({
                "id": inv.id.to_string(),
                "pool_id": format!("{}", inv.pool_id),
                "invitee": hex::encode(inv.invitee_node_id.0),
                "expires_at": inv.expires_at.to_rfc3339(),
                "created_at": inv.created_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(Json(values))
}

/// GET /api/pool/leaderboard — Pool member credit contribution leaderboard.
pub async fn pool_leaderboard(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(&state, PoolCommand::GetLeaderboard { reply: tx }).await?;

    let entries = await_pool_recv(rx).await?;

    let values: Vec<serde_json::Value> = entries
        .into_iter()
        .map(|e| {
            serde_json::json!({
                "node_id": hex::encode(e.node_id.0),
                "credits_contributed": e.credits_contributed,
                "rank": e.rank,
            })
        })
        .collect();

    Ok(Json(values))
}

/// POST /api/pool/device-name — Set this device's nickname within the pool.
pub async fn pool_set_device_name(
    State(state): State<AppState>,
    Json(body): Json<PoolDeviceNameRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if body.name.trim().is_empty() {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Device name must not be empty".into(),
        )));
    }
    if body.name.trim().len() > 32 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Device name must be 32 characters or fewer".into(),
        )));
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    send_pool_command(
        &state,
        PoolCommand::SetDeviceName {
            name: body.name,
            reply: tx,
        },
    )
    .await?;
    await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// PUT /api/pool/credit-split — Set credit split percentage (owner only).
pub async fn pool_set_credit_split(
    State(state): State<AppState>,
    Json(body): Json<PoolCreditSplitRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if body.pct > 100 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Credit split percentage must be 0–100".into(),
        )));
    }
    let (tx, rx) = tokio::sync::oneshot::channel();
    send_pool_command(
        &state,
        PoolCommand::SetCreditSplit {
            pct: body.pct,
            reply: tx,
        },
    )
    .await?;
    await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// PUT /api/pool/contribution — Set contribution level for a member device (owner only).
pub async fn pool_set_contribution(
    State(state): State<AppState>,
    Json(body): Json<PoolContributionRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let node_id = parse_node_id(&body.node_id)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    send_pool_command(
        &state,
        PoolCommand::SetContributionLevel {
            node_id,
            level: body.level,
            reply: tx,
        },
    )
    .await?;
    await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({"status": "ok"})))
}

/// POST /api/pool/generate-code — Generate a short invite code (owner only).
pub async fn pool_generate_code(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(&state, PoolCommand::GenerateInviteCode { reply: tx }).await?;

    let code = await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "code": code,
    })))
}

/// POST /api/pool/join — Join a pool using an invite code.
pub async fn pool_join(
    State(state): State<AppState>,
    Json(body): Json<PoolJoinRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let code = body.code.trim().to_uppercase();
    if code.len() != 8 || !code.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invite code must be 8 uppercase alphanumeric characters".into(),
        )));
    }

    let (tx, rx) = tokio::sync::oneshot::channel();

    send_pool_command(&state, PoolCommand::JoinWithCode { code, reply: tx }).await?;

    await_pool_reply(rx).await?;
    Ok(Json(serde_json::json!({
        "status": "ok",
        "message": "Join request broadcast. You will be added to the pool once the owner's node processes the request.",
    })))
}

// ---- Helpers ----

async fn send_pool_command(state: &AppState, cmd: PoolCommand) -> Result<(), ApiError> {
    let tx_lock = state.shared_state.credits.pool_tx.read().await;
    let tx = tx_lock.as_ref().ok_or_else(|| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Pool manager not running".into(),
        ))
    })?;
    tx.send(cmd).await.map_err(|_| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Pool manager channel closed".into(),
        ))
    })
}

fn parse_node_id(hex: &str) -> Result<NodeId, ApiError> {
    let bytes = hex::decode(hex).map_err(|_| {
        ApiError(crate::error::SwarmError::Validation(
            "Invalid node_id hex".into(),
        ))
    })?;
    if bytes.len() != 32 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "node_id must be 32 bytes (64 hex chars)".into(),
        )));
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&bytes);
    Ok(NodeId(arr))
}

// ---- Request types ----

#[derive(Debug, Deserialize)]
pub struct PoolCreateRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct PoolInviteRequest {
    pub node_id: String,
}

#[derive(Debug, Deserialize)]
pub struct PoolAcceptRequest {
    pub invitation_id: String,
}

#[derive(Debug, Deserialize)]
pub struct PoolRemoveRequest {
    pub node_id: String,
}

#[derive(Debug, Deserialize)]
pub struct PoolJoinRequest {
    pub code: String,
}

#[derive(Debug, Deserialize)]
pub struct PoolDeviceNameRequest {
    pub name: String,
}

#[derive(Debug, Deserialize)]
pub struct PoolCreditSplitRequest {
    pub pct: u8,
}

#[derive(Debug, Deserialize)]
pub struct PoolContributionRequest {
    pub node_id: String,
    pub level: u8,
}

#[derive(Debug, Deserialize)]
pub struct PoolRatesRequest {
    pub inference_serve: Option<i64>,
    pub inference_consume: Option<i64>,
    pub shard_hosting: Option<i64>,
    pub shard_seeding: Option<i64>,
    pub relay_service: Option<i64>,
    pub penalty_serve_failure: Option<i64>,
}

/// GET /api/admin/pools/:id/rates — Get credit rates for a pool.
pub async fn pool_rates_get(
    State(state): State<AppState>,
    Path(pool_id_hex): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let pool_id = parse_node_id(&pool_id_hex)?;

    let rates = state
        .shared_state
        .credits
        .pool_credit_rates
        .get(&pool_id)
        .map(|r| r.value().clone())
        .unwrap_or_else(|| state.config.pool.credit_rates.clone());

    Ok(Json(serde_json::json!({
        "pool_id": pool_id_hex,
        "rates": {
            "inference_serve": rates.inference_serve,
            "inference_consume": rates.inference_consume,
            "shard_hosting": rates.shard_hosting,
            "shard_seeding": rates.shard_seeding,
            "relay_service": rates.relay_service,
            "penalty_serve_failure": rates.penalty_serve_failure,
        }
    })))
}

/// PUT /api/admin/pools/:id/rates — Set credit rates for a pool.
pub async fn pool_rates_set(
    State(state): State<AppState>,
    Path(pool_id_hex): Path<String>,
    Json(body): Json<PoolRatesRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let pool_id = parse_node_id(&pool_id_hex)?;
    let defaults = CreditRateConfig::default();

    // Merge: use provided values or fall back to current/defaults
    let current = state
        .shared_state
        .credits
        .pool_credit_rates
        .get(&pool_id)
        .map(|r| r.value().clone())
        .unwrap_or_else(|| state.config.pool.credit_rates.clone());

    tracing::debug!(pool_id = %pool_id_hex, "DIAG: pool_rates_set request");

    let new_rates = CreditRateConfig {
        inference_serve: body.inference_serve.unwrap_or(current.inference_serve),
        inference_consume: body.inference_consume.unwrap_or(current.inference_consume),
        shard_hosting: body.shard_hosting.unwrap_or(current.shard_hosting),
        shard_seeding: body.shard_seeding.unwrap_or(current.shard_seeding),
        relay_service: body.relay_service.unwrap_or(current.relay_service),
        penalty_serve_failure: body
            .penalty_serve_failure
            .unwrap_or(current.penalty_serve_failure),
    };

    // Validate: all rates must be positive (penalty is stored as a positive magnitude,
    // negated at the point of use in router.rs via `apply_credit_direct(..., -penalty, ...)`)
    let positive_rates = [
        ("inference_serve", new_rates.inference_serve),
        ("inference_consume", new_rates.inference_consume),
        ("shard_hosting", new_rates.shard_hosting),
        ("shard_seeding", new_rates.shard_seeding),
        ("relay_service", new_rates.relay_service),
    ];
    for (name, value) in &positive_rates {
        if *value <= 0 {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "{name} must be positive (got {value})"
            ))));
        }
    }
    if new_rates.penalty_serve_failure <= 0 {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "penalty_serve_failure must be positive (got {})",
            new_rates.penalty_serve_failure
        ))));
    }

    // Validate: no rate exceeds 10x the default
    let default_positive = [
        ("inference_serve", defaults.inference_serve),
        ("inference_consume", defaults.inference_consume),
        ("shard_hosting", defaults.shard_hosting),
        ("shard_seeding", defaults.shard_seeding),
        ("relay_service", defaults.relay_service),
    ];
    for ((name, value), (_, default_val)) in positive_rates.iter().zip(default_positive.iter()) {
        if *value > default_val * 10 {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "{name} cannot exceed 10x the default ({}) — got {value}",
                default_val * 10
            ))));
        }
    }
    if new_rates.penalty_serve_failure > defaults.penalty_serve_failure * 10 {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "penalty_serve_failure cannot exceed 10x the default ({}) — got {}",
            defaults.penalty_serve_failure * 10,
            new_rates.penalty_serve_failure
        ))));
    }

    state
        .shared_state
        .credits
        .pool_credit_rates
        .insert(pool_id, new_rates.clone());

    Ok(Json(serde_json::json!({
        "status": "updated",
        "pool_id": pool_id_hex,
        "rates": {
            "inference_serve": new_rates.inference_serve,
            "inference_consume": new_rates.inference_consume,
            "shard_hosting": new_rates.shard_hosting,
            "shard_seeding": new_rates.shard_seeding,
            "relay_service": new_rates.relay_service,
            "penalty_serve_failure": new_rates.penalty_serve_failure,
        }
    })))
}

// ---- Private Mode ----

/// GET /api/pool/private-mode — Get current private mode state + coverage summary.
pub async fn get_private_mode(State(state): State<AppState>) -> Json<serde_json::Value> {
    let enabled = state
        .shared_state
        .credits
        .private_mode
        .load(std::sync::atomic::Ordering::Relaxed);
    let allow_lan = state.shared_state.config.pool.private_mode_allow_lan;
    let coverage = compute_pool_coverage(&state.shared_state).await;

    Json(serde_json::json!({
        "enabled": enabled,
        "allow_lan": allow_lan,
        "offline_mode": state.shared_state.credits.offline_mode.load(std::sync::atomic::Ordering::Relaxed),
        "coverage": coverage,
    }))
}

#[derive(Deserialize)]
pub struct SetPrivateModeRequest {
    pub enabled: bool,
    #[serde(default)]
    pub offline_mode: Option<bool>,
}

/// PUT /api/pool/private-mode — Toggle private mode and/or offline mode.
pub async fn set_private_mode(
    State(state): State<AppState>,
    Json(body): Json<SetPrivateModeRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Must be in a pool to enable private mode
    if body.enabled {
        let pool_state = state.shared_state.credits.pool_state.read().await;
        if pool_state.is_none() {
            return Err(ApiError(SwarmError::Validation(
                "Must be in a device pool to enable private mode".into(),
            )));
        }
    }

    state
        .shared_state
        .credits
        .private_mode
        .store(body.enabled, std::sync::atomic::Ordering::Relaxed);
    if let Err(e) = state
        .shared_state
        .db
        .put_json("pool_state", "private_mode", &body.enabled)
    {
        tracing::warn!(error = %e, "Failed to persist private_mode — will revert on restart");
    }

    // Offline mode toggle (optional)
    if let Some(offline) = body.offline_mode {
        state
            .shared_state
            .credits
            .offline_mode
            .store(offline, std::sync::atomic::Ordering::Relaxed);
        if let Err(e) = state
            .shared_state
            .db
            .put_json("pool_state", "offline_mode", &offline)
        {
            tracing::warn!(error = %e, "Failed to persist offline_mode — will revert on restart");
        }
    }
    let offline = state
        .shared_state
        .credits
        .offline_mode
        .load(std::sync::atomic::Ordering::Relaxed);

    // Emit activity event so all WS clients update
    let msg = if body.enabled && offline {
        "Offline private mode enabled — air-gapped, LAN only".to_string()
    } else if body.enabled {
        "Private mode enabled — inference restricted to your devices".to_string()
    } else {
        "Private mode disabled — using full swarm network".to_string()
    };
    state.shared_state.emit_activity(
        crate::daemon::state::ActivityEvent::new("pool", "private_mode_changed", msg)
            .with_toast(if body.enabled { "info" } else { "success" }, 5000),
    );

    // Signal dashboard to refresh models (availability changes in private mode)
    state
        .shared_state
        .signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);

    // Trigger auto-manage re-evaluation so shard availability updates immediately
    state.shared_state.models.auto_manage_notify.notify_one();

    tracing::info!(
        private_mode = body.enabled,
        offline_mode = offline,
        "Private/offline mode toggled"
    );

    // Return coverage summary so UI can show trade-offs immediately
    let coverage = compute_pool_coverage(&state.shared_state).await;

    Ok(Json(serde_json::json!({
        "enabled": body.enabled,
        "offline_mode": offline,
        "coverage": coverage,
    })))
}

/// GET /api/pool/coverage — Detailed model coverage within the device pool.
pub async fn pool_coverage(State(state): State<AppState>) -> Json<serde_json::Value> {
    let coverage = compute_pool_coverage(&state.shared_state).await;
    Json(coverage)
}

// ---- Shard Pinning ----

/// GET /api/pool/pins — List current shard pins.
pub async fn pool_pins(State(state): State<AppState>) -> Json<serde_json::Value> {
    let pool_state = state.shared_state.credits.pool_state.read().await;
    let pins: Vec<serde_json::Value> = pool_state
        .as_ref()
        .map(|ps| {
            ps.shard_pins
                .iter()
                .map(|p| {
                    serde_json::json!({
                        "model_id": p.model_id,
                        "shard_indices": p.shard_indices,
                        "target_node_id": hex::encode(p.target_node_id.0),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Json(serde_json::json!({ "pins": pins }))
}

#[derive(Deserialize)]
pub struct PinRequest {
    pub model_id: String,
    #[serde(default)]
    pub shard_indices: Vec<u32>,
    pub target_node_id: String,
}

/// POST /api/pool/pin — Pin a model/shards to a specific device (owner only).
pub async fn pool_add_pin(
    State(state): State<AppState>,
    Json(body): Json<PinRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let target = parse_node_id(&body.target_node_id)?;
    let PinRequest {
        model_id,
        shard_indices,
        ..
    } = body;
    let model_id_resp = model_id.clone();

    // Hold the pool_state write lock across the DB persist so a concurrent
    // pin/unpin cannot write a stale snapshot.
    {
        let my_id = state.shared_state.identity.node_id().clone();
        let mut ps_guard = state.shared_state.credits.pool_state.write().await;
        let ps = ps_guard
            .as_mut()
            .ok_or_else(|| ApiError(SwarmError::Validation("Not in a pool".into())))?;
        if ps.pool_id != my_id {
            return Err(ApiError(SwarmError::Validation(
                "Only the pool owner can manage pins".into(),
            )));
        }
        if !ps.members.iter().any(|m| m.node_id == target) {
            return Err(ApiError(SwarmError::Validation(
                "Target device is not a pool member".into(),
            )));
        }
        let pin = crate::types::ShardPin {
            model_id,
            shard_indices,
            target_node_id: target,
        };
        ps.shard_pins
            .retain(|p| !(p.model_id == pin.model_id && p.target_node_id == pin.target_node_id));
        ps.shard_pins.push(pin);

        if let Err(e) = state.shared_state.db.put_json(
            crate::pool::manager::TREE_POOL_STATE,
            crate::pool::manager::KEY_MY_POOL,
            &*ps,
        ) {
            tracing::warn!(error = %e, "Failed to persist pool shard pin — may be lost on restart");
        }
    }

    // Notify auto-manage to re-evaluate
    state.shared_state.models.auto_manage_notify.notify_one();

    Ok(Json(
        serde_json::json!({ "status": "pinned", "model_id": model_id_resp }),
    ))
}

/// DELETE /api/pool/pin — Remove a shard pin (owner only).
pub async fn pool_remove_pin(
    State(state): State<AppState>,
    Json(body): Json<PinRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let target = parse_node_id(&body.target_node_id)?;
    let PinRequest { model_id, .. } = body;
    let model_id_resp = model_id.clone();

    {
        let my_id = state.shared_state.identity.node_id().clone();
        let mut ps_guard = state.shared_state.credits.pool_state.write().await;
        let ps = ps_guard
            .as_mut()
            .ok_or_else(|| ApiError(SwarmError::Validation("Not in a pool".into())))?;
        if ps.pool_id != my_id {
            return Err(ApiError(SwarmError::Validation(
                "Only the pool owner can manage pins".into(),
            )));
        }
        let before = ps.shard_pins.len();
        ps.shard_pins
            .retain(|p| !(p.model_id == model_id && p.target_node_id == target));
        if ps.shard_pins.len() == before {
            return Err(ApiError(SwarmError::Validation("Pin not found".into())));
        }
        if let Err(e) = state.shared_state.db.put_json(
            crate::pool::manager::TREE_POOL_STATE,
            crate::pool::manager::KEY_MY_POOL,
            &*ps,
        ) {
            tracing::warn!(error = %e, "Failed to persist pool shard unpin — may be lost on restart");
        }
    }

    Ok(Json(
        serde_json::json!({ "status": "unpinned", "model_id": model_id_resp }),
    ))
}

/// Compute which models are fully/partially covered by pool members.
async fn compute_pool_coverage(shared: &crate::daemon::SharedState) -> serde_json::Value {
    let pool_state = shared.credits.pool_state.read().await;
    let pool_members: std::collections::HashSet<NodeId> = pool_state
        .as_ref()
        .map(|ps| ps.members.iter().map(|m| m.node_id.clone()).collect())
        .unwrap_or_default();

    // Include LAN peers if configured
    let mut allowed = pool_members;
    allowed.insert(shared.identity.node_id().clone());
    if shared.config.pool.private_mode_allow_lan {
        for entry in shared.peer_registry.iter() {
            if entry.value().is_lan_peer {
                allowed.insert(entry.key().clone());
            }
        }
    }

    let mut models = Vec::new();
    let mut total_fully_covered = 0u32;
    let mut total_partially_covered = 0u32;
    let mut total_est_download_bytes: u64 = 0;

    for manifest in shared.model_registry.models() {
        let total_shards = manifest.shard_count;
        let mut pool_shards = 0u32;
        let mut missing_indices = Vec::new();
        let mut missing_bytes: u64 = 0;

        for shard in &manifest.shards {
            let sid = crate::types::ShardId {
                model_id: manifest.id.clone(),
                index: shard.index,
            };
            let holders = shared.model_registry.shard_holders(&sid);
            if holders.iter().any(|h| allowed.contains(h)) {
                pool_shards += 1;
            } else {
                missing_indices.push(shard.index);
                missing_bytes += shard.size_bytes;
            }
        }

        let coverage_pct = if total_shards > 0 {
            (pool_shards as f64 / total_shards as f64 * 100.0).round() as u32
        } else {
            0
        };

        if pool_shards == total_shards {
            total_fully_covered += 1;
        } else if pool_shards > 0 {
            total_partially_covered += 1;
        }
        total_est_download_bytes += missing_bytes;

        models.push(serde_json::json!({
            "id": manifest.id.0,
            "name": manifest.name,
            "total_shards": total_shards,
            "pool_shards": pool_shards,
            "coverage_pct": coverage_pct,
            "missing": missing_indices,
            "est_download_mb": missing_bytes / (1024 * 1024),
        }));
    }

    // Disk budget info
    let max_storage_mb = if shared.config.auto_manage.max_storage_mb > 0 {
        shared.config.auto_manage.max_storage_mb
    } else {
        shared.config.resources.max_disk_mb / 2
    };
    // Estimate current auto-managed disk usage from local shard files
    let shard_store = shared.shard_store();
    let mut used_bytes: u64 = 0;
    for manifest in shared.model_registry.models() {
        let my_id = shared.identity.node_id();
        for shard in &manifest.shards {
            let path = shard_store.shard_path(&manifest.id, shard.index);
            if shared
                .model_registry
                .shard_holders(&crate::types::ShardId {
                    model_id: manifest.id.clone(),
                    index: shard.index,
                })
                .contains(my_id)
            {
                if let Ok(meta) = std::fs::metadata(&path) {
                    used_bytes += meta.len();
                }
            }
        }
    }
    let used_mb = used_bytes / (1024 * 1024);

    serde_json::json!({
        "models": models,
        "total_models": models.len(),
        "fully_covered": total_fully_covered,
        "partially_covered": total_partially_covered,
        "not_covered": models.len() as u32 - total_fully_covered - total_partially_covered,
        "est_total_download_mb": total_est_download_bytes / (1024 * 1024),
        "pool_member_count": allowed.len(),
        "disk_budget_mb": max_storage_mb,
        "disk_used_mb": used_mb,
    })
}
