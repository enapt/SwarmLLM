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
    let hardware = detect_hardware(&state.shared_state);

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
///
/// Returns all models: locally loaded, from the P2P registry, and discovered
/// on the network from peer announcements. Each model includes its source,
/// availability info, and which peers host it.
pub async fn list_models(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let mut models: Vec<serde_json::Value> = Vec::new();
    let mut seen_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    let local_node_id = state.shared_state.identity.node_id().clone();

    // Collect peer info for each model from shard_registry
    let mut model_peers: std::collections::HashMap<String, std::collections::HashSet<String>> =
        std::collections::HashMap::new();
    for entry in state.shared_state.shard_registry.iter() {
        let shard_id = entry.key();
        let holders = entry.value();
        let model_name = shard_id.model_id.0.clone();
        for holder in holders.iter() {
            if *holder != local_node_id {
                model_peers
                    .entry(model_name.clone())
                    .or_default()
                    .insert(format!("{}", holder));
            }
        }
    }

    // Also gather peer node_ids that host each model from capability data
    for entry in state.shared_state.peer_registry.iter() {
        let peer = entry.value();
        if let Some(ref cap) = peer.capability {
            for shard in &cap.hosted_shards {
                model_peers
                    .entry(shard.model_id.0.clone())
                    .or_default()
                    .insert(format!("{}", peer.node_id));
            }
        }
    }

    // Helper: build per-shard detail for a manifest
    let build_shard_detail =
        |m: &crate::types::ModelManifest, state: &AppState| -> Vec<serde_json::Value> {
            m.shards
                .iter()
                .map(|s| {
                    let shard_id = crate::types::ShardId {
                        model_id: m.id.clone(),
                        index: s.index,
                    };
                    let holders = state.shared_state.model_registry.shard_holders(&shard_id);
                    let local = holders.contains(&local_node_id);
                    serde_json::json!({
                        "index": s.index,
                        "size_bytes": s.size_bytes,
                        "local": local,
                        "holders": holders.len(),
                    })
                })
                .collect()
        };

    // 1. Locally loaded model (full model via --model flag)
    // Even though it's locally loaded, get the real manifest to show shard info
    if let Some(info) = state.shared_state.loaded_model_info.read().await.as_ref() {
        let peer_count = model_peers.get(&info.name).map_or(0, |s| s.len());
        seen_ids.insert(info.name.clone());

        let mid = crate::types::ModelId(info.name.clone());
        let manifest = state.shared_state.model_registry.get_manifest(&mid);
        let (shard_count, hosted_shards, shard_detail) = match manifest {
            Some(ref m) => {
                let detail = build_shard_detail(m, &state);
                (m.shard_count, m.shard_count, detail)
            }
            None => (1, 1, vec![]),
        };

        models.push(serde_json::json!({
            "id": info.name,
            "name": info.name,
            "total_size_bytes": info.size_bytes,
            "shard_count": shard_count,
            "hosted_shards": hosted_shards,
            "healthy": true,
            "status": "loaded",
            "mode": "full",
            "source": "local",
            "local": true,
            "peers_hosting": peer_count,
            "shards": shard_detail,
        }));
    }

    // 2. Models from the P2P manifest registry
    let registry = &state.shared_state.model_registry;
    let manifests = registry.list_models();

    for m in &manifests {
        if seen_ids.contains(&m.id.0) {
            continue;
        }
        seen_ids.insert(m.id.0.clone());

        let hosted_count = (0..m.shard_count)
            .filter(|&idx| {
                let shard_id = crate::types::ShardId {
                    model_id: m.id.clone(),
                    index: idx,
                };
                let holders = state.shared_state.model_registry.shard_holders(&shard_id);
                holders.contains(&local_node_id)
            })
            .count();

        let peer_count = model_peers.get(&m.id.0).map_or(0, |s| s.len());
        let shard_detail = build_shard_detail(m, &state);

        let (source, mode) = if hosted_count == m.shard_count as usize {
            ("local", "full")
        } else if hosted_count > 0 {
            ("hybrid", "sharded")
        } else {
            ("network", "sharded")
        };

        let status = if hosted_count == m.shard_count as usize {
            "complete"
        } else if hosted_count > 0 {
            "partial"
        } else {
            "available"
        };

        models.push(serde_json::json!({
            "id": m.id.0,
            "name": m.name,
            "total_size_bytes": m.total_size_bytes,
            "shard_count": m.shard_count,
            "hosted_shards": hosted_count,
            "healthy": hosted_count == m.shard_count as usize,
            "status": status,
            "mode": mode,
            "source": source,
            "local": hosted_count > 0,
            "peers_hosting": peer_count,
            "shards": shard_detail,
        }));
    }

    // 3. Models discovered from peer announcements (not in our registry or loaded)
    for (model_name, peers) in &model_peers {
        if seen_ids.contains(model_name) {
            continue;
        }
        seen_ids.insert(model_name.clone());
        models.push(serde_json::json!({
            "id": model_name,
            "name": model_name,
            "total_size_bytes": 0,
            "shard_count": 0,
            "hosted_shards": 0,
            "healthy": true,
            "status": "available",
            "mode": "full",
            "source": "network",
            "local": false,
            "peers_hosting": peers.len(),
            "shards": [],
        }));
    }

    Json(models)
}

/// POST /api/admin/models/:id/add — Express interest in a model (trigger download).
pub async fn add_model_interest(
    State(state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    tracing::info!(model_id = %model_id, "Model acquisition requested");

    let mid = crate::types::ModelId(model_id.clone());

    // Send acquisition command if the channel is wired up
    if let Some(ref tx) = state.acquisition_tx {
        tx.send(crate::model::acquisition::AcquisitionCommand::Acquire { model_id: mid })
            .await
            .map_err(|e| {
                ApiError(crate::error::SwarmError::Internal(format!(
                    "Failed to send acquisition command: {e}"
                )))
            })?;

        Ok(Json(serde_json::json!({
            "status": "acquiring",
            "model_id": model_id,
        })))
    } else {
        // Standalone mode — no acquisition manager
        Ok(Json(serde_json::json!({
            "status": "unavailable",
            "model_id": model_id,
            "message": "Model acquisition requires daemon mode with P2P networking",
        })))
    }
}

/// GET /api/admin/models/:id/status — Query model acquisition progress.
///
/// Reads directly from the shared `acquisition_progress` DashMap for low-latency,
/// lock-free progress reporting without going through the AcquisitionManager channel.
pub async fn model_acquisition_status(
    State(state): State<AppState>,
    axum::extract::Path(model_id): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let mid = crate::types::ModelId(model_id.clone());

    // Fast path: read from shared state (no channel round-trip)
    if let Some(status) = state.shared_state.acquisition_progress.get(&mid) {
        return Ok(Json(
            serde_json::to_value(status.value()).unwrap_or_default(),
        ));
    }

    Ok(Json(serde_json::json!({
        "model_id": model_id,
        "state": "unknown",
        "message": "No active acquisition for this model",
    })))
}

/// GET /api/admin/peers — List connected peers.
pub async fn list_peers(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let timeout = chrono::Duration::seconds(90); // 3 missed pings
    let now = chrono::Utc::now();

    let peers: Vec<serde_json::Value> = state
        .shared_state
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

            serde_json::json!({
                "node_id": format!("{}", peer.node_id),
                "addresses": peer.addresses,
                "last_seen": peer.last_seen.to_rfc3339(),
                "latency_ms": peer.latency_ms,
                "trust_score": peer.trust_score,
                "healthy": healthy,
                "gpu": peer.capability.as_ref().and_then(|c| c.gpu.as_ref().map(|g| &g.name)),
                "hosted_models": hosted_models,
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

// ---- HuggingFace Endpoints ----

/// GET /api/admin/hf/search?q=... — Search HuggingFace for GGUF models.
pub async fn hf_search(
    axum::extract::Query(params): axum::extract::Query<HfSearchParams>,
) -> Result<Json<Vec<serde_json::Value>>, ApiError> {
    let query = params.q.unwrap_or_default();
    if query.is_empty() {
        return Ok(Json(vec![]));
    }

    let results = crate::model::huggingface::search_gguf_models(&query)
        .await
        .map_err(|e| ApiError(crate::error::SwarmError::Internal(e)))?;

    let values: Vec<serde_json::Value> = results
        .into_iter()
        .map(|r| {
            serde_json::json!({
                "repo_id": r.repo_id,
                "filename": r.filename,
                "size_bytes": r.size_bytes,
                "downloads": r.downloads,
            })
        })
        .collect();

    Ok(Json(values))
}

/// POST /api/admin/hf/download — Start downloading a GGUF model from HuggingFace.
pub async fn hf_download(
    State(state): State<AppState>,
    Json(body): Json<HfDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = body.repo_id;
    let filename = body.filename;

    if repo_id.is_empty() || filename.is_empty() {
        return Ok(Json(serde_json::json!({
            "status": "error",
            "message": "repo_id and filename are required",
        })));
    }

    let dest_dir = state
        .config
        .node
        .data_dir
        .join("models")
        .join(repo_id.replace('/', "_"));

    tracing::info!(repo = %repo_id, file = %filename, "Starting HuggingFace download");

    // Spawn download in background
    let repo_clone = repo_id.clone();
    let file_clone = filename.clone();
    let shared = state.shared_state.clone();
    let model_id_str = format!("hf:{}/{}", repo_id, filename);
    let mid = crate::types::ModelId(model_id_str.clone());

    // Create initial acquisition progress entry
    let status = crate::model::acquisition::AcquisitionStatus {
        model_id: mid.clone(),
        state: crate::model::acquisition::AcquisitionState::Downloading,
        total_shards: 1,
        downloaded_shards: 0,
        verified_shards: 0,
        failed_shards: 0,
        total_bytes: 0,
        downloaded_bytes: 0,
        shard_progress: std::collections::HashMap::new(),
        speed_bytes_per_sec: 0,
        started_at: Some(chrono::Utc::now()),
        log: vec![format!("Downloading {} from HuggingFace...", filename)],
    };
    shared.acquisition_progress.insert(mid.clone(), status);

    tokio::spawn(async move {
        let (ptx, mut prx) =
            tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);

        let download_mid = mid.clone();
        let download_shared = shared.clone();

        // Spawn progress updater
        let progress_mid = mid.clone();
        let progress_shared = shared.clone();
        tokio::spawn(async move {
            let mut last_bytes = 0u64;
            let mut last_time = std::time::Instant::now();
            while let Some(prog) = prx.recv().await {
                if let Some(mut entry) = progress_shared.acquisition_progress.get_mut(&progress_mid)
                {
                    entry.downloaded_bytes = prog.downloaded_bytes;
                    entry.total_bytes = prog.total_bytes;
                    let now = std::time::Instant::now();
                    let dt = now.duration_since(last_time).as_secs_f64();
                    if dt > 0.5 {
                        let speed = ((prog.downloaded_bytes - last_bytes) as f64 / dt) as u64;
                        entry.speed_bytes_per_sec = speed;
                        last_bytes = prog.downloaded_bytes;
                        last_time = now;
                    }
                }
            }
        });

        match crate::model::huggingface::download_model(
            &repo_clone,
            &file_clone,
            &dest_dir,
            Some(ptx),
        )
        .await
        {
            Ok(path) => {
                tracing::info!(path = %path.display(), "HuggingFace download complete");
                if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid)
                {
                    entry.state = crate::model::acquisition::AcquisitionState::Complete;
                    entry.downloaded_shards = 1;
                    entry.verified_shards = 1;
                    entry
                        .log
                        .push(format!("Download complete: {}", path.display()));
                }

                // Try to load the downloaded model
                let executor = download_shared.executor.clone();
                let gpu_layers = download_shared.config.inference.gpu_layers;
                let model_name = format!("{}/{}", repo_clone, file_clone);

                let mut exec = executor.lock().await;
                match exec.load_model(&path, gpu_layers) {
                    Ok(()) => {
                        let size = exec.model_size_bytes().unwrap_or(0);
                        *download_shared.loaded_model_info.write().await =
                            Some(crate::daemon::LoadedModelInfo {
                                name: model_name.clone(),
                                size_bytes: size,
                            });
                        if let Some(mut entry) =
                            download_shared.acquisition_progress.get_mut(&download_mid)
                        {
                            entry.log.push(format!("Model loaded: {}", model_name));
                        }
                        tracing::info!(model = %model_name, "HF model loaded for inference");
                    }
                    Err(e) => {
                        if let Some(mut entry) =
                            download_shared.acquisition_progress.get_mut(&download_mid)
                        {
                            entry.log.push(format!("Model load failed: {}", e));
                        }
                        tracing::error!(error = %e, "Failed to load HF model");
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "HuggingFace download failed");
                if let Some(mut entry) = download_shared.acquisition_progress.get_mut(&download_mid)
                {
                    entry.state =
                        crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                    entry.failed_shards = 1;
                    entry.log.push(format!("Download failed: {}", e));
                }
            }
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "model_id": model_id_str,
    })))
}

/// POST /api/admin/shutdown — Gracefully shut down the node.
pub async fn shutdown_node(State(state): State<AppState>) -> Json<serde_json::Value> {
    tracing::info!("Shutdown requested via API");

    // Signal all subsystems to shut down
    state.shared_state.shutdown();

    // Give a moment for the response to send, then exit
    tokio::spawn(async {
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        std::process::exit(0);
    });

    Json(serde_json::json!({ "status": "shutting_down" }))
}

#[derive(Debug, Deserialize)]
pub struct HfSearchParams {
    pub q: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct HfDownloadRequest {
    pub repo_id: String,
    pub filename: String,
    #[serde(default)]
    pub mode: String,
}

// ---- Hardware detection ----

fn detect_hardware(shared_state: &crate::daemon::SharedState) -> serde_json::Value {
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

    // GPU info from llama.cpp device detection (set at startup)
    let (gpu_name, gpu_vram_mb) = match &shared_state.gpu_info {
        Some(gpu) => (Some(gpu.name.clone()), Some(gpu.vram_total_mb)),
        None => (None, None),
    };

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
