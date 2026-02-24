use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::config::ContributionMode;
use crate::error::ApiError;

/// GET /api/admin/stats — Full dashboard stats snapshot.
pub async fn stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let node_id = format!("{}", state.shared_state.identity.node_id());
    let stats = state.shared_state.node_stats.read().await;
    let credit = state.shared_state.credit_balance.read().await;

    let uptime_seconds = (chrono::Utc::now() - stats.uptime_start)
        .num_seconds()
        .max(0) as u64;

    let tier = crate::credit::priority::PriorityCalculator::tier_name(credit.balance);

    let hosted_shards = state.shared_state.model_registry.shard_count();

    // Hardware detection
    let hardware = detect_hardware();

    Json(serde_json::json!({
        "node_id": node_id,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime_seconds,
        "tier": tier,
        "peers": stats.peers_connected,
        "requests_served": stats.requests_served,
        "requests_made": stats.requests_made,
        "active_requests": state.shared_state.active_pipelines.len(),
        "hosted_shards": hosted_shards,
        "credits": {
            "balance": credit.balance,
            "lifetime_earned": credit.lifetime_earned,
            "lifetime_spent": credit.lifetime_spent,
        },
        "hardware": hardware,
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
    }))
}

/// PUT /api/admin/config — Update configuration at runtime.
pub async fn update_config(
    State(state): State<AppState>,
    Json(body): Json<ConfigUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Persist the updated config to the config TOML file.
    // For now, acknowledge the update — runtime config changes require daemon restart.
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
        config.inference.max_concurrent_requests = max_reqs;
    }
    if let Some(bw) = body.max_bandwidth_mbps {
        config.resources.max_bandwidth_mbps = bw;
    }
    if let Some(disk) = body.max_disk_mb {
        config.resources.max_disk_mb = disk;
    }

    // Write updated config to disk
    let toml_str = toml::to_string_pretty(&config)
        .map_err(|e| ApiError(crate::error::SwarmError::Config(e.to_string())))?;

    if let Some(parent) = config_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    std::fs::write(&config_path, toml_str)
        .map_err(|e| ApiError(crate::error::SwarmError::Io(e)))?;

    tracing::info!(path = %config_path.display(), "Configuration saved");

    Ok(Json(serde_json::json!({ "status": "ok" })))
}

/// GET /api/admin/models — List known models and their status.
pub async fn list_models(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let registry = &state.shared_state.model_registry;
    let manifests = registry.list_models();

    let models: Vec<serde_json::Value> = manifests
        .iter()
        .map(|m| {
            let hosted_count = (0..m.shard_count)
                .filter(|&idx| {
                    let shard_id = crate::types::ShardId {
                        model_id: m.id.clone(),
                        index: idx,
                    };
                    state.shared_state.model_registry.has_shard(&shard_id)
                })
                .count();

            serde_json::json!({
                "id": m.id.0,
                "name": m.name,
                "total_size_bytes": m.total_size_bytes,
                "shard_count": m.shard_count,
                "hosted_shards": hosted_count,
                "healthy": hosted_count == m.shard_count as usize,
                "status": if hosted_count == m.shard_count as usize { "complete" } else { "partial" },
            })
        })
        .collect();

    Json(models)
}

/// POST /api/admin/models/:id/add — Express interest in a model (trigger download).
pub async fn add_model_interest(
    State(_state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    tracing::info!(model_id = %model_id, "Model interest registered");
    // In a complete implementation, this would trigger shard discovery and download.
    // For now, acknowledge the intent.
    Json(serde_json::json!({
        "status": "queued",
        "model_id": model_id,
    }))
}

/// GET /api/admin/peers — List connected peers.
pub async fn list_peers(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let peers: Vec<serde_json::Value> = state
        .shared_state
        .peer_registry
        .iter()
        .map(|entry| {
            let peer = entry.value();
            serde_json::json!({
                "node_id": format!("{}", peer.node_id),
                "addresses": peer.addresses,
                "last_seen": peer.last_seen.to_rfc3339(),
                "latency_ms": peer.latency_ms,
                "trust_score": peer.trust_score,
                "gpu": peer.capability.as_ref().and_then(|c| c.gpu.as_ref().map(|g| &g.name)),
            })
        })
        .collect();

    Json(peers)
}

/// GET /api/admin/credits — Credit details.
pub async fn credit_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    let credit = state.shared_state.credit_balance.read().await;
    let tier = crate::credit::priority::PriorityCalculator::tier_name(credit.balance);

    Json(serde_json::json!({
        "balance": credit.balance,
        "lifetime_earned": credit.lifetime_earned,
        "lifetime_spent": credit.lifetime_spent,
        "tier": tier,
        "last_updated": credit.last_updated.to_rfc3339(),
    }))
}

// ---- Governance Endpoints ----

/// GET /api/admin/issues — List issues (paginated, filterable).
pub async fn list_issues(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let issues = crate::governance::issues::list_issues(&state.db).map_err(ApiError)?;

    let result: Vec<serde_json::Value> = issues
        .iter()
        .map(|i| {
            serde_json::json!({
                "hash": hex::encode(i.hash),
                "title": i.title,
                "author": format!("{}", i.author),
                "category": i.category,
                "severity": i.severity,
                "status": i.status,
                "created_at": i.created_at.to_rfc3339(),
                "upvotes": i.upvotes,
                "tags": i.tags,
            })
        })
        .collect();

    Ok(Json(result))
}

/// POST /api/admin/issues — Create a new issue.
pub async fn create_issue(
    State(state): State<AppState>,
    Json(body): Json<CreateIssueRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let node_id = state.shared_state.identity.node_id().clone();
    let credit = state.shared_state.credit_balance.read().await;
    let params = state.shared_state.governance_params.read().await;

    let issue = crate::governance::issues::create_issue(
        &state.db,
        crate::governance::issues::CreateIssueParams {
            author: node_id,
            title: body.title,
            body: body.body,
            category: body.category,
            severity: body.severity,
            tags: body.tags.unwrap_or_default(),
            credit_balance: credit.balance,
        },
        &params,
    )
    .map_err(ApiError)?;

    Ok(Json(serde_json::json!({
        "hash": hex::encode(issue.hash),
        "status": "created",
    })))
}

/// GET /api/admin/issues/:hash — Get issue details + comments.
pub async fn get_issue(
    State(state): State<AppState>,
    axum::extract::Path(hash_hex): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hash = parse_hash(&hash_hex)?;
    let issue = crate::governance::issues::get_issue(&state.db, &hash)
        .map_err(ApiError)?
        .ok_or_else(|| ApiError(crate::error::SwarmError::IssueNotFound(hash_hex)))?;

    let comments = crate::governance::issues::get_comments(&state.db, &hash).map_err(ApiError)?;

    let comment_values: Vec<serde_json::Value> = comments
        .iter()
        .map(|c| {
            serde_json::json!({
                "author": format!("{}", c.author),
                "body": c.body,
                "created_at": c.created_at.to_rfc3339(),
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "hash": hex::encode(issue.hash),
        "title": issue.title,
        "body": issue.body,
        "author": format!("{}", issue.author),
        "category": issue.category,
        "severity": issue.severity,
        "status": issue.status,
        "created_at": issue.created_at.to_rfc3339(),
        "upvotes": issue.upvotes,
        "tags": issue.tags,
        "comments": comment_values,
    })))
}

/// POST /api/admin/issues/:hash/comment — Add comment.
pub async fn add_issue_comment(
    State(state): State<AppState>,
    axum::extract::Path(hash_hex): axum::extract::Path<String>,
    Json(body): Json<CommentRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hash = parse_hash(&hash_hex)?;
    let comment = crate::types::IssueComment {
        issue_hash: hash,
        author: state.shared_state.identity.node_id().clone(),
        body: body.body,
        created_at: chrono::Utc::now(),
        signature: vec![],
    };

    crate::governance::issues::add_comment(&state.db, &comment).map_err(ApiError)?;

    Ok(Json(serde_json::json!({ "status": "ok" })))
}

/// POST /api/admin/issues/:hash/upvote — Upvote issue.
pub async fn upvote_issue(
    State(state): State<AppState>,
    axum::extract::Path(hash_hex): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hash = parse_hash(&hash_hex)?;
    let credit = state.shared_state.credit_balance.read().await;

    let upvote = crate::types::IssueUpvote {
        issue_hash: hash,
        voter: state.shared_state.identity.node_id().clone(),
        weight: credit.lifetime_earned,
        signature: vec![],
    };

    let updated = crate::governance::issues::upvote_issue(&state.db, &upvote).map_err(ApiError)?;

    Ok(Json(serde_json::json!({
        "upvotes": updated.upvotes,
    })))
}

/// GET /api/admin/proposals — List proposals.
pub async fn list_proposals(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let proposals = crate::governance::proposals::list_proposals(&state.db).map_err(ApiError)?;

    let result: Vec<serde_json::Value> = proposals
        .iter()
        .map(|p| {
            serde_json::json!({
                "hash": hex::encode(p.hash),
                "title": p.title,
                "author": format!("{}", p.author),
                "category": p.category,
                "status": p.status,
                "created_at": p.created_at.to_rfc3339(),
                "voting_deadline": p.voting_deadline.to_rfc3339(),
                "linked_issues": p.linked_issues.iter().map(hex::encode).collect::<Vec<_>>(),
            })
        })
        .collect();

    Ok(Json(result))
}

/// POST /api/admin/proposals — Create a new proposal.
pub async fn create_proposal(
    State(state): State<AppState>,
    Json(body): Json<CreateProposalRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let node_id = state.shared_state.identity.node_id().clone();
    let params = state.shared_state.governance_params.read().await;

    // Determine role (simplified — in production would use full GovernanceNodeStats)
    let role = crate::types::GovernanceRole::Contributor; // Default for API users

    let proposal = crate::governance::proposals::create_proposal(
        &state.db,
        crate::governance::proposals::CreateProposalParams {
            author: node_id,
            title: body.title,
            body: body.body,
            category: body.category,
            linked_issues: body.linked_issues.unwrap_or_default(),
            patch: body.patch,
            role,
        },
        &params,
    )
    .map_err(ApiError)?;

    Ok(Json(serde_json::json!({
        "hash": hex::encode(proposal.hash),
        "status": "draft",
    })))
}

/// GET /api/admin/proposals/:hash — Get proposal details + votes.
pub async fn get_proposal(
    State(state): State<AppState>,
    axum::extract::Path(hash_hex): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hash = parse_hash(&hash_hex)?;
    let proposal = crate::governance::proposals::get_proposal(&state.db, &hash)
        .map_err(ApiError)?
        .ok_or_else(|| ApiError(crate::error::SwarmError::ProposalNotFound(hash_hex)))?;

    let votes = crate::governance::voting::get_votes(&state.db, &hash).map_err(ApiError)?;

    let network_stats = state.shared_state.network_stats.read().await;
    let params = state.shared_state.governance_params.read().await;
    let tally = crate::governance::voting::tally_votes(&votes, &proposal, &network_stats, &params);

    Ok(Json(serde_json::json!({
        "hash": hex::encode(proposal.hash),
        "title": proposal.title,
        "body": proposal.body,
        "author": format!("{}", proposal.author),
        "category": proposal.category,
        "status": proposal.status,
        "created_at": proposal.created_at.to_rfc3339(),
        "voting_deadline": proposal.voting_deadline.to_rfc3339(),
        "linked_issues": proposal.linked_issues.iter().map(hex::encode).collect::<Vec<_>>(),
        "has_patch": proposal.patch.is_some(),
        "votes": {
            "approve": tally.approve_weight,
            "reject": tally.reject_weight,
            "abstain": tally.abstain_weight,
            "total": tally.total_weight,
            "voters": tally.unique_voters,
            "quorum_met": tally.quorum_met,
            "approved": tally.approved,
        },
    })))
}

/// POST /api/admin/proposals/:hash/vote — Cast vote.
pub async fn vote_proposal(
    State(state): State<AppState>,
    axum::extract::Path(hash_hex): axum::extract::Path<String>,
    Json(body): Json<VoteRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let hash = parse_hash(&hash_hex)?;
    let proposal = crate::governance::proposals::get_proposal(&state.db, &hash)
        .map_err(ApiError)?
        .ok_or_else(|| ApiError(crate::error::SwarmError::ProposalNotFound(hash_hex)))?;

    let credit = state.shared_state.credit_balance.read().await;
    let vote = crate::types::ProposalVote {
        proposal_hash: hash,
        voter: state.shared_state.identity.node_id().clone(),
        vote: body.vote,
        weight: credit.lifetime_earned,
        role: crate::types::GovernanceRole::Member,
        timestamp: chrono::Utc::now(),
        signature: vec![],
    };

    crate::governance::voting::cast_vote(&state.db, &vote, &proposal).map_err(ApiError)?;

    Ok(Json(serde_json::json!({ "status": "ok" })))
}

/// GET /api/admin/releases — List releases.
pub async fn list_releases(
    State(state): State<AppState>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let releases = crate::governance::releases::list_releases(&state.db).map_err(ApiError)?;

    let params = state.shared_state.governance_params.read().await;
    let mut result = Vec::new();
    for rc in &releases {
        let approvals =
            crate::governance::releases::count_approvals(&state.db, &rc.version).unwrap_or(0);
        let reports = crate::governance::releases::get_test_reports(&state.db, &rc.version)
            .unwrap_or_default();
        let canary = crate::governance::releases::determine_canary_phase(rc, &reports, &params);

        result.push(serde_json::json!({
            "version": format!("{}", rc.version),
            "builder": format!("{}", rc.builder),
            "created_at": rc.created_at.to_rfc3339(),
            "changelog": rc.changelog,
            "proposals": rc.included_proposals.len(),
            "binaries": rc.binaries.len(),
            "approvals": approvals,
            "threshold": params.release_approval_threshold,
            "approved": approvals >= params.release_approval_threshold,
            "test_reports": reports.len(),
            "canary_phase": canary,
        }));
    }

    Ok(Json(result))
}

/// GET /api/admin/releases/latest — Get latest stable release info.
pub async fn get_latest_release(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let releases = crate::governance::releases::list_releases(&state.db).map_err(ApiError)?;

    // Find latest non-prerelease
    let latest = releases
        .iter()
        .filter(|r| !r.version.is_prerelease())
        .max_by_key(|r| (r.version.major, r.version.minor, r.version.patch));

    match latest {
        Some(rc) => Ok(Json(serde_json::json!({
            "version": format!("{}", rc.version),
            "changelog": rc.changelog,
            "created_at": rc.created_at.to_rfc3339(),
        }))),
        None => Ok(Json(serde_json::json!({
            "version": null,
            "message": "No stable releases found",
        }))),
    }
}

/// GET /api/admin/governance/role — Get your current role.
pub async fn governance_role(State(state): State<AppState>) -> Json<serde_json::Value> {
    let params = state.shared_state.governance_params.read().await;
    let network_stats = state.shared_state.network_stats.read().await;

    // For now, determine role based on credit balance and uptime
    let credit = state.shared_state.credit_balance.read().await;
    let node_stats = state.shared_state.node_stats.read().await;
    let uptime_days = (chrono::Utc::now() - node_stats.uptime_start)
        .num_days()
        .max(0) as u32;

    let gov_stats = crate::types::GovernanceNodeStats {
        lifetime_earned_percentile: 0.5, // Simplified — would need real network data
        uptime_days,
        accepted_proposals: 0,
        is_council_member: false,
    };

    let role = crate::types::GovernanceRole::from_node_governance_stats(&gov_stats, &params);
    let is_genesis = crate::governance::releases::is_genesis_period(&network_stats);

    Json(serde_json::json!({
        "role": role,
        "uptime_days": uptime_days,
        "lifetime_earned": credit.lifetime_earned,
        "is_genesis_period": is_genesis,
        "can_create_proposals": role.can_create_proposals(),
        "can_approve_releases": role.can_approve_releases(),
    }))
}

/// GET /api/admin/governance/params — Get governance parameters.
pub async fn governance_params(State(state): State<AppState>) -> Json<serde_json::Value> {
    let params = state.shared_state.governance_params.read().await;
    Json(serde_json::to_value(&*params).unwrap_or_default())
}

// ---- Helper functions ----

fn parse_hash(hex_str: &str) -> Result<crate::types::Blake3Hash, ApiError> {
    let bytes = hex::decode(hex_str).map_err(|_| {
        ApiError(crate::error::SwarmError::Governance(
            "Invalid hash format".into(),
        ))
    })?;
    if bytes.len() != 32 {
        return Err(ApiError(crate::error::SwarmError::Governance(
            "Hash must be 32 bytes".into(),
        )));
    }
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&bytes);
    Ok(hash)
}

// ---- Request types ----

#[derive(Debug, Deserialize)]
pub struct CreateIssueRequest {
    pub title: String,
    pub body: String,
    pub category: crate::types::IssueCategory,
    pub severity: Option<crate::types::IssueSeverity>,
    pub tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub struct CommentRequest {
    pub body: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateProposalRequest {
    pub title: String,
    pub body: String,
    pub category: crate::types::ProposalCategory,
    pub linked_issues: Option<Vec<crate::types::Blake3Hash>>,
    pub patch: Option<crate::types::ProposalPatch>,
}

#[derive(Debug, Deserialize)]
pub struct VoteRequest {
    pub vote: crate::types::VoteChoice,
}

#[derive(Debug, Deserialize)]
pub struct ConfigUpdate {
    pub contribution: Option<String>,
    pub max_concurrent_requests: Option<u32>,
    pub max_bandwidth_mbps: Option<u64>,
    pub max_disk_mb: Option<u64>,
    #[serde(default)]
    pub models: Vec<String>,
}

// ---- Hardware detection ----

fn detect_hardware() -> serde_json::Value {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let used_ram_mb = sys.used_memory() / (1024 * 1024);

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

    // GPU detection — check for common env hints
    // Real GPU detection would require CUDA/ROCm libraries; keep it simple
    let gpu_name = std::env::var("SWARMLLM_GPU_NAME").ok();
    let gpu_vram_mb: Option<u64> = std::env::var("SWARMLLM_GPU_VRAM_MB")
        .ok()
        .and_then(|v| v.parse().ok());

    serde_json::json!({
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "total_ram_mb": total_ram_mb,
        "used_ram_mb": used_ram_mb,
        "available_disk_mb": available_disk_mb,
        "total_disk_mb": total_disk_mb,
        "used_disk_mb": used_disk_mb,
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
    })
}
