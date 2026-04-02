use axum::extract::{Path, State};
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::config::CreditRateConfig;
use crate::error::{ApiError, SwarmError};
use crate::pool::types::PoolCommand;
use crate::types::NodeId;

/// Await a oneshot reply from the pool manager, converting channel errors to ServiceUnavailable.
async fn await_pool_reply<T>(
    rx: tokio::sync::oneshot::Receiver<Result<T, SwarmError>>,
) -> Result<T, ApiError> {
    rx.await
        .map_err(|_| {
            ApiError(SwarmError::ServiceUnavailable(
                "Pool manager unavailable".into(),
            ))
        })?
        .map_err(ApiError)
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
    tracing::debug!("DIAG: pool_invite request");
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

    let invitations = inv_rx.await.map_err(|_| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Pool manager unavailable".into(),
        ))
    })?;

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

    let invitations = rx.await.map_err(|_| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Pool manager unavailable".into(),
        ))
    })?;

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

    let entries = rx.await.map_err(|_| {
        ApiError(crate::error::SwarmError::ServiceUnavailable(
            "Pool manager unavailable".into(),
        ))
    })?;

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
    if body.name.len() > 64 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Device name must be 64 characters or fewer".into(),
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
    if code.is_empty() || code.len() > 16 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invite code must be 8 characters".into(),
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

    // Validate: reward rates must be positive, penalty must be negative
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
    if new_rates.penalty_serve_failure >= 0 {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "penalty_serve_failure must be negative (got {})",
            new_rates.penalty_serve_failure
        ))));
    }

    // Validate: no reward rate exceeds 10x the default, penalty within 10x magnitude
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
    if new_rates.penalty_serve_failure < defaults.penalty_serve_failure * 10 {
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
