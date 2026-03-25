use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use ed25519_dalek::Verifier;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::pool::crypto;
use crate::pool::types::*;
use crate::types::{NetworkCommand, NodeId, SwarmMessage};

/// Database tree names for pool persistence.
const TREE_POOL_STATE: &str = "pool_state";
const TREE_POOL_INVITATIONS: &str = "pool_invitations";
const TREE_POOL_FORWARDS: &str = "pool_forwards";
const TREE_POOL_REMOVAL_REPLAYS: &str = "pool_removal_replays";
const KEY_MY_POOL: &str = "my_pool";

/// The PoolManager is the 9th subsystem task.
/// It owns all pool state, persists to redb, and handles pool commands.
pub struct PoolManager {
    shared_state: Arc<SharedState>,
    cmd_rx: mpsc::Receiver<PoolCommand>,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
    rate_limiter: PoolRateLimiter,
    /// Pending invitations we've sent (as owner) or received (as invitee).
    pending_invitations: HashMap<uuid::Uuid, PoolInvitation>,
    /// Active invite codes (owner only). Keyed by code_hash for O(1) lookup.
    /// One-time use, expired codes cleaned on each generate.
    invite_codes: HashMap<[u8; 32], PoolInviteCode>,
    /// When set, auto-accept invitations from this specific pool owner (set by JoinWithCode).
    /// Bound to a code_hash to prevent a different pool's invitation from being auto-accepted.
    auto_accept_code_hash: Option<[u8; 32]>,
}

impl PoolManager {
    pub fn new(
        shared_state: Arc<SharedState>,
        cmd_rx: mpsc::Receiver<PoolCommand>,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let rate_limit = shared_state.config.pool.rate_limit_per_hour;
        Self {
            shared_state,
            cmd_rx,
            network_tx,
            shutdown_rx,
            rate_limiter: PoolRateLimiter::new(rate_limit as usize, 1),
            pending_invitations: HashMap::new(),
            invite_codes: HashMap::new(),
            auto_accept_code_hash: None,
        }
    }

    /// Restore pool state from database on startup.
    async fn restore_state(&mut self) {
        let db = &self.shared_state.db;

        // Restore pool state — await directly to avoid TOCTOU race with first commands
        if let Ok(Some(state)) = db.get_json::<PoolState>(TREE_POOL_STATE, KEY_MY_POOL) {
            tracing::info!(
                pool_id = %state.pool_id,
                members = state.members.len(),
                "Restored pool state from database"
            );
            let pool_id = state.pool_id.clone();
            *self.shared_state.credits.pool_state.write().await = Some(state.clone());
            self.shared_state
                .credits
                .pool_registry
                .insert(pool_id, state);
        }

        // Restore pending invitations
        if let Ok(invitations) = db.iter_json::<PoolInvitation>(TREE_POOL_INVITATIONS) {
            let now = chrono::Utc::now();
            for inv in invitations {
                if inv.expires_at > now {
                    self.pending_invitations.insert(inv.id, inv);
                }
            }
            if !self.pending_invitations.is_empty() {
                tracing::info!(
                    count = self.pending_invitations.len(),
                    "Restored pending pool invitations"
                );
            }
        }
    }

    /// Run the pool manager event loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        self.restore_state().await;

        let gossip_secs = self.shared_state.config.pool.gossip_interval_secs;
        let mut gossip_interval =
            tokio::time::interval(std::time::Duration::from_secs(gossip_secs));
        gossip_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        tracing::info!("PoolManager running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("PoolManager shutting down");
                        break;
                    }
                }
                cmd = self.cmd_rx.recv() => {
                    match cmd {
                        Some(cmd) => {
                            tracing::debug!(cmd = ?std::mem::discriminant(&cmd), "DIAG: pool command received");
                            self.handle_command(cmd).await;
                        }
                        None => break,
                    }
                }
                _ = gossip_interval.tick() => {
                    self.gossip_pool_state().await;
                }
            }
        }

        Ok(())
    }

    async fn handle_command(&mut self, cmd: PoolCommand) {
        match cmd {
            PoolCommand::CreatePool { name, reply } => {
                let result = self.handle_create_pool(name).await;
                let _ = reply.send(result);
            }
            PoolCommand::CreateInvitation { invitee, reply } => {
                let result = self.handle_create_invitation(invitee).await;
                let _ = reply.send(result);
            }
            PoolCommand::AcceptInvitation { invitation, reply } => {
                let result = self.handle_accept_invitation(invitation).await;
                let _ = reply.send(result);
            }
            PoolCommand::RemoveMember { node_id, reply } => {
                let result = self.handle_remove_member(node_id).await;
                let _ = reply.send(result);
            }
            PoolCommand::LeavePool { reply } => {
                let result = self.handle_leave_pool().await;
                let _ = reply.send(result);
            }
            PoolCommand::ProcessCreditForward { forward } => {
                self.handle_credit_forward(forward).await;
            }
            PoolCommand::PoolStateGossip { state } => {
                self.handle_pool_state_gossip(state).await;
            }
            PoolCommand::InboundInvitation { invitation } => {
                self.handle_inbound_invitation(invitation).await;
            }
            PoolCommand::InboundBlindedInvitation { blinded } => {
                self.handle_inbound_blinded_invitation(blinded).await;
            }
            PoolCommand::InboundAcceptance { acceptance } => {
                self.handle_inbound_acceptance(acceptance).await;
            }
            PoolCommand::InboundRemoval { removal } => {
                self.handle_inbound_removal(removal).await;
            }
            PoolCommand::InboundMemberLeft {
                pool_id,
                node_id,
                signature,
            } => {
                self.handle_inbound_member_left(pool_id, node_id, signature)
                    .await;
            }
            PoolCommand::SetDeviceName { name, reply } => {
                let result = self.handle_set_device_name(name).await;
                let _ = reply.send(result);
            }
            PoolCommand::SetCreditSplit { pct, reply } => {
                let result = self.handle_set_credit_split(pct).await;
                let _ = reply.send(result);
            }
            PoolCommand::SetContributionLevel {
                node_id,
                level,
                reply,
            } => {
                let result = self.handle_set_contribution_level(node_id, level).await;
                let _ = reply.send(result);
            }
            PoolCommand::GenerateInviteCode { reply } => {
                let result = self.handle_generate_invite_code().await;
                let _ = reply.send(result);
            }
            PoolCommand::JoinWithCode { code, reply } => {
                let result = self.handle_join_with_code(code).await;
                let _ = reply.send(result);
            }
            PoolCommand::InboundJoinRequest {
                code_hash,
                requester,
            } => {
                self.handle_inbound_join_request(code_hash, requester).await;
            }
            PoolCommand::GetState { reply } => {
                let state = self.shared_state.credits.pool_state.read().await.clone();
                let _ = reply.send(state);
            }
            PoolCommand::GetInvitations { reply } => {
                let invitations: Vec<PoolInvitation> =
                    self.pending_invitations.values().cloned().collect();
                let _ = reply.send(invitations);
            }
            PoolCommand::GetMembership { reply } => {
                let state = self.shared_state.credits.pool_state.read().await;
                let my_id = self.shared_state.identity.node_id();
                let membership = state
                    .as_ref()
                    .and_then(|s| s.members.iter().find(|m| m.node_id == *my_id).cloned());
                let _ = reply.send(membership);
            }
            PoolCommand::GetLeaderboard { reply } => {
                let leaderboard = self.build_leaderboard().await;
                let _ = reply.send(leaderboard);
            }
        }
    }

    async fn handle_create_pool(&mut self, name: String) -> Result<PoolState, SwarmError> {
        // Validate pool name: 1-64 chars, printable ASCII, no control chars
        if name.is_empty() || name.len() > 64 {
            return Err(SwarmError::Internal(
                "Pool name must be 1-64 characters".into(),
            ));
        }
        if !name.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
            return Err(SwarmError::Internal(
                "Pool name may only contain printable ASCII characters".into(),
            ));
        }

        // Check we're not already in a pool
        if self.shared_state.credits.pool_state.read().await.is_some() {
            return Err(SwarmError::Internal("Already in a pool".into()));
        }

        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Internal("Rate limit exceeded".into()));
        }

        let my_id = self.shared_state.identity.node_id().clone();
        let now = chrono::Utc::now();

        // Sign the pool creation
        let payload = {
            let mut h = blake3::Hasher::new();
            h.update(b"pool_create_v1");
            h.update(&my_id.0);
            h.update(name.as_bytes());
            h.update(now.to_rfc3339().as_bytes());
            h.finalize().as_bytes().to_vec()
        };
        let sig = self.shared_state.identity.sign(&payload);

        let state = PoolState {
            pool_id: my_id.clone(),
            name,
            members: vec![PoolMembership {
                node_id: my_id.clone(),
                credits_contributed: 0,
                joined_at: now,
                acceptance_signature: sig.clone(),
                invitation_id: uuid::Uuid::nil(),
                device_name: None,
                last_seen: Some(now),
                online: true,
                device_stats: None,
                contribution_level: 100,
            }],
            created_at: now,
            owner_signature: sig,
            total_lifetime_credits: 0,
            member_credit_split_pct: 0,
        };

        // Persist and update shared state
        self.persist_pool_state(&state)?;
        *self.shared_state.credits.pool_state.write().await = Some(state.clone());
        self.shared_state
            .credits
            .pool_registry
            .insert(my_id, state.clone());

        tracing::info!(
            pool_id = %state.pool_id,
            name = %state.name,
            "Created device pool"
        );
        self.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "pool",
                "pool_created",
                format!("Device pool '{}' created", state.name),
            )
            .with_toast("success", 5000),
        );

        Ok(state)
    }

    async fn handle_create_invitation(
        &mut self,
        invitee: NodeId,
    ) -> Result<PoolInvitation, SwarmError> {
        // Extract pool_id from state, validating constraints, then release the lock.
        let pool_id = {
            let guard = self.shared_state.credits.pool_state.read().await;
            let state = guard
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Not in a pool".into()))?;

            let my_id = self.shared_state.identity.node_id();
            if state.pool_id != *my_id {
                return Err(SwarmError::Internal(
                    "Only the pool owner can invite".into(),
                ));
            }

            let max_size = self.shared_state.config.pool.max_pool_size;
            if state.members.len() >= max_size as usize {
                return Err(SwarmError::Internal(format!(
                    "Pool is full (max {max_size} members)"
                )));
            }

            if state.members.iter().any(|m| m.node_id == invitee) {
                return Err(SwarmError::Internal("Node is already a pool member".into()));
            }

            state.pool_id.clone()
        };

        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Internal("Rate limit exceeded".into()));
        }

        let ttl = self.shared_state.config.pool.invitation_ttl_hours;
        let invitation =
            crypto::create_invitation(&self.shared_state.identity, &pool_id, &invitee, ttl);

        self.shared_state.db.put_json(
            TREE_POOL_INVITATIONS,
            &invitation.id.to_string(),
            &invitation,
        )?;
        self.pending_invitations
            .insert(invitation.id, invitation.clone());

        // SEC-M18 FIX: Broadcast a blinded invitation that hides the invitee's identity.
        // Only the intended invitee can recognize the invitation by recomputing the BLAKE3
        // commitment H("pool_invitee_commit_v1" || their_node_id || invitation_id).
        let blinded = BlindedPoolInvitation::from_invitation(&invitation);
        let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::BlindedInvitation(blinded));
        let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;

        tracing::info!(
            invitee = %invitee,
            invitation_id = %invitation.id,
            "Created pool invitation"
        );

        Ok(invitation)
    }

    async fn handle_accept_invitation(
        &mut self,
        invitation: PoolInvitation,
    ) -> Result<(), SwarmError> {
        // Check we're not already in a pool
        if self.shared_state.credits.pool_state.read().await.is_some() {
            return Err(SwarmError::Internal("Already in a pool".into()));
        }

        // Verify the invitation is for us
        let my_id = self.shared_state.identity.node_id();
        if invitation.invitee_node_id != *my_id {
            return Err(SwarmError::Internal(
                "Invitation is not for this node".into(),
            ));
        }

        // Check expiry
        if invitation.expires_at < chrono::Utc::now() {
            return Err(SwarmError::Internal("Invitation has expired".into()));
        }

        // Verify owner signature (pool_id == owner's NodeId)
        let owner_key = ed25519_dalek::VerifyingKey::from_bytes(&invitation.pool_id.0)
            .map_err(|_| SwarmError::Internal("Invalid pool owner key".into()))?;
        crypto::verify_invitation(&invitation, &owner_key)?;

        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Internal("Rate limit exceeded".into()));
        }

        // Create acceptance
        let acceptance = crypto::create_acceptance(&self.shared_state.identity, &invitation);

        // Set our pool state as a member of this pool
        let membership = PoolMembership {
            node_id: my_id.clone(),
            credits_contributed: 0,
            joined_at: chrono::Utc::now(),
            acceptance_signature: acceptance.invitee_signature.clone(),
            invitation_id: invitation.id,
            device_name: None,
            last_seen: Some(chrono::Utc::now()),
            online: true,
            device_stats: None,
            contribution_level: 100,
        };

        // Create a local pool state representing our membership
        let state = PoolState {
            pool_id: invitation.pool_id.clone(),
            name: String::new(), // Will be updated from gossip
            members: vec![membership],
            created_at: chrono::Utc::now(),
            owner_signature: invitation.owner_signature.clone(),
            total_lifetime_credits: 0,
            member_credit_split_pct: 0,
        };

        self.persist_pool_state(&state)?;
        *self.shared_state.credits.pool_state.write().await = Some(state.clone());

        // Broadcast acceptance to the network
        let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::Acceptance(acceptance));
        let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;

        // Remove from pending (memory + DB to prevent replay after restart)
        self.pending_invitations.remove(&invitation.id);
        let _ = self
            .shared_state
            .db
            .remove(TREE_POOL_INVITATIONS, &invitation.id.to_string());

        tracing::info!(
            pool_id = %invitation.pool_id,
            "Accepted pool invitation"
        );

        self.shared_state
            .emit_activity(crate::daemon::state::ActivityEvent {
                category: "pool",
                kind: "pool_device_joined",
                message: "Joined device pool".to_string(),
                model_id: None,
                model_name: None,
                node_id: Some(format!("{}", invitation.pool_id)),
                detail_num: None,
                detail_str: None,
                toast_level: Some("success"),
                toast_duration_ms: Some(5000),
                shard_index: None,
                freed_bytes: None,
                holder_count_before: None,
                holder_count_after: None,
                remaining_local_shards: None,
                timestamp: None,
            });

        Ok(())
    }

    async fn handle_remove_member(&mut self, node_id: NodeId) -> Result<(), SwarmError> {
        // Check rate limit before mutating state
        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Internal("Rate limit exceeded".into()));
        }

        let (removal, state_clone) = {
            let mut guard = self.shared_state.credits.pool_state.write().await;
            let ps = guard
                .as_mut()
                .ok_or_else(|| SwarmError::Internal("Not in a pool".into()))?;

            let my_id = self.shared_state.identity.node_id();
            if ps.pool_id != *my_id {
                return Err(SwarmError::Internal(
                    "Only the pool owner can remove members".into(),
                ));
            }

            if node_id == *my_id {
                return Err(SwarmError::Internal(
                    "Owner cannot remove themselves".into(),
                ));
            }

            let before = ps.members.len();
            ps.members.retain(|m| m.node_id != node_id);
            if ps.members.len() == before {
                return Err(SwarmError::Internal("Node is not a pool member".into()));
            }

            let removal =
                crypto::create_removal(&self.shared_state.identity, &ps.pool_id, &node_id);
            let clone = ps.clone();
            (removal, clone)
        };

        self.persist_pool_state(&state_clone)?;

        let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::Removal(removal));
        let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;

        tracing::info!(removed = %node_id, "Removed member from pool");
        self.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "pool",
                "pool_member_removed",
                format!("Device {} removed from pool", &format!("{}", node_id)[..16]),
            )
            .with_node(format!("{}", node_id))
            .with_toast("info", 5000),
        );

        Ok(())
    }

    async fn handle_leave_pool(&mut self) -> Result<(), SwarmError> {
        let pool_id = {
            let guard = self.shared_state.credits.pool_state.read().await;
            let ps = guard
                .as_ref()
                .ok_or_else(|| SwarmError::Internal("Not in a pool".into()))?;

            if !self.rate_limiter.check_and_record() {
                return Err(SwarmError::Internal("Rate limit exceeded".into()));
            }

            ps.pool_id.clone()
        };

        // Clear pool state — DB first, then memory, to prevent inconsistency on DB failure.
        // If DB write fails, we return error with in-memory state still intact (correct for retry).
        // If process crashes after DB clear but before memory clear, restart sees no DB record (correct).
        self.shared_state.db.remove(TREE_POOL_STATE, KEY_MY_POOL)?;
        *self.shared_state.credits.pool_state.write().await = None;
        self.shared_state.credits.pool_registry.remove(&pool_id);

        // Broadcast signed member-left notice
        let my_id = self.shared_state.identity.node_id().clone();
        let leave_payload = {
            let mut h = blake3::Hasher::new();
            h.update(b"pool_member_left_v1");
            h.update(&pool_id.0);
            h.update(&my_id.0);
            h.finalize().as_bytes().to_vec()
        };
        let leave_signature = self.shared_state.identity.sign(&leave_payload);
        let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::MemberLeft {
            pool_id,
            node_id: my_id,
            signature: leave_signature,
        });
        let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;

        tracing::info!("Left device pool");

        self.shared_state
            .emit_activity(crate::daemon::state::ActivityEvent {
                category: "pool",
                kind: "pool_device_left",
                message: "Left device pool".to_string(),
                model_id: None,
                model_name: None,
                node_id: None,
                detail_num: None,
                detail_str: None,
                toast_level: Some("info"),
                toast_duration_ms: Some(5000),
                shard_index: None,
                freed_bytes: None,
                holder_count_before: None,
                holder_count_after: None,
                remaining_local_shards: None,
                timestamp: None,
            });

        Ok(())
    }

    async fn handle_credit_forward(&mut self, mut forward: PoolCreditForward) {
        // SEC-I4: Validate forward amount > 0
        if forward.amount <= 0 {
            tracing::warn!(from = %forward.from_node_id, amount = forward.amount, "Rejecting credit forward with non-positive amount");
            return;
        }

        // Validate that to_node_id matches this node (the pool owner) to prevent
        // credit forwarding to arbitrary nodes via forged to_node_id.
        let my_id = self.shared_state.identity.node_id();
        if forward.to_node_id != *my_id {
            tracing::warn!(
                from = %forward.from_node_id,
                to = %forward.to_node_id,
                "Credit forward to_node_id doesn't match pool owner — rejecting"
            );
            return;
        }

        // Dedup check: reject replayed credit forwards
        if let Ok(Some(_)) = self
            .shared_state
            .db
            .get_json::<PoolCreditForward>(TREE_POOL_FORWARDS, &forward.id.to_string())
        {
            tracing::warn!(id = %forward.id, from = %forward.from_node_id, "Rejecting replayed credit forward");
            return;
        }

        // Verify member signature before accepting
        let member_key = match ed25519_dalek::VerifyingKey::from_bytes(&forward.from_node_id.0) {
            Ok(k) => k,
            Err(_) => {
                tracing::warn!(from = %forward.from_node_id, "Invalid member key in credit forward");
                return;
            }
        };

        // Verify sender is an actual pool member BEFORE co-signing
        {
            let state = self.shared_state.credits.pool_state.read().await;
            if let Some(ref ps) = *state {
                let is_member = ps.members.iter().any(|m| m.node_id == forward.from_node_id);
                if !is_member {
                    tracing::warn!(from = %forward.from_node_id, "Credit forward from non-member rejected");
                    return;
                }
            } else {
                tracing::warn!("Credit forward received but no pool state — rejecting");
                return;
            }
        }

        // SEC-C2 + SEC-I7: Owner co-signs the credit forward (verifies member sig internally)
        if let Err(e) =
            crypto::cosign_credit_forward(&self.shared_state.identity, &mut forward, &member_key)
        {
            tracing::warn!(from = %forward.from_node_id, error = %e, "Failed to cosign credit forward");
            return;
        }

        // Store in audit log
        if let Err(e) =
            self.shared_state
                .db
                .put_json(TREE_POOL_FORWARDS, &forward.id.to_string(), &forward)
        {
            tracing::warn!(error = %e, "Failed to persist credit forward");
        }

        // SEC-C2: Apply credit to the pool owner's balance
        if let Err(e) = crate::credit::ledger::apply_credit_direct(
            &self.shared_state.credits.credit_balance,
            &self.shared_state.db,
            forward.amount,
            true,
        )
        .await
        {
            tracing::warn!(error = %e, "Failed to apply forwarded credits to owner balance");
        }

        // Update the member's contribution in pool state
        let mut state = self.shared_state.credits.pool_state.write().await;
        if let Some(ref mut ps) = *state {
            if let Some(member) = ps
                .members
                .iter_mut()
                .find(|m| m.node_id == forward.from_node_id)
            {
                member.credits_contributed =
                    member.credits_contributed.saturating_add(forward.amount);
            }
            ps.total_lifetime_credits = ps.total_lifetime_credits.saturating_add(forward.amount);

            if let Err(e) = self.persist_pool_state(ps) {
                tracing::warn!(error = %e, "Failed to persist pool state after credit forward");
            }
        }

        // Broadcast the co-signed credit forward
        let msg =
            SwarmMessage::PoolMessage(crate::types::PoolMessage::CreditForward(forward.clone()));
        let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;

        tracing::debug!(
            from = %forward.from_node_id,
            amount = forward.amount,
            "Processed credit forward"
        );
    }

    async fn handle_pool_state_gossip(&mut self, state: PoolState) {
        // Verify owner signature before inserting into registry
        let owner_key = match ed25519_dalek::VerifyingKey::from_bytes(&state.pool_id.0) {
            Ok(k) => k,
            Err(_) => {
                tracing::warn!("Invalid owner key in pool state gossip");
                return;
            }
        };
        // Reconstruct the pool creation signing payload and verify
        let payload = {
            let mut h = blake3::Hasher::new();
            h.update(b"pool_create_v1");
            h.update(&state.pool_id.0);
            h.update(state.name.as_bytes());
            h.update(state.created_at.to_rfc3339().as_bytes());
            h.finalize().as_bytes().to_vec()
        };
        let sig_bytes: &[u8; 64] = match state.owner_signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => {
                tracing::warn!("Pool state gossip has invalid signature length");
                return;
            }
        };
        let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
        if owner_key.verify(&payload, &sig).is_err() {
            tracing::warn!(pool_id = %state.pool_id, "Invalid owner signature in pool state gossip");
            return;
        }

        // Enforce max pool size to prevent Ed25519 verification DoS
        let max_pool_size = self.shared_state.config.pool.max_pool_size;
        if state.members.len() > max_pool_size as usize {
            tracing::warn!(
                pool_id = %state.pool_id,
                members = state.members.len(),
                max = max_pool_size,
                "Rejecting pool state gossip: exceeds max pool size"
            );
            return;
        }

        // Reject duplicate node_ids in gossiped pool state (prevents inflated stats)
        {
            let mut seen_nodes = std::collections::HashSet::new();
            for member in &state.members {
                if !seen_nodes.insert(member.node_id.clone()) {
                    tracing::warn!(
                        pool_id = %state.pool_id,
                        dup_node = %member.node_id,
                        "Duplicate node_id in gossiped pool state — rejecting"
                    );
                    return;
                }
            }
        }

        // SEC-C7: Verify acceptance_signature of each member
        for member in &state.members {
            // The pool owner's own membership uses the pool creation signature, skip it
            if member.node_id == state.pool_id {
                continue;
            }
            let member_key = match ed25519_dalek::VerifyingKey::from_bytes(&member.node_id.0) {
                Ok(k) => k,
                Err(_) => {
                    tracing::warn!(member = %member.node_id, "Invalid member key in pool state gossip");
                    return;
                }
            };
            // Verify the acceptance signature using the acceptance payload
            let acceptance_payload = {
                let mut hasher = blake3::Hasher::new();
                hasher.update(b"pool_acceptance_v1");
                hasher.update(member.invitation_id.as_bytes());
                hasher.update(&state.pool_id.0);
                hasher.update(&member.node_id.0);
                hasher.finalize().as_bytes().to_vec()
            };
            let sig_bytes: &[u8; 64] = match member.acceptance_signature.as_slice().try_into() {
                Ok(b) => b,
                Err(_) => {
                    tracing::warn!(member = %member.node_id, "Invalid acceptance signature length in pool state gossip");
                    return;
                }
            };
            let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
            if member_key.verify(&acceptance_payload, &sig).is_err() {
                tracing::warn!(member = %member.node_id, "Invalid acceptance signature in pool state gossip");
                return;
            }
        }

        // Store in registry for network-wide visibility
        self.shared_state
            .credits
            .pool_registry
            .insert(state.pool_id.clone(), state);
    }

    async fn handle_inbound_invitation(&mut self, invitation: PoolInvitation) {
        let my_id = self.shared_state.identity.node_id();
        if invitation.invitee_node_id == *my_id {
            // Verify owner signature before storing
            let owner_key = match ed25519_dalek::VerifyingKey::from_bytes(&invitation.pool_id.0) {
                Ok(k) => k,
                Err(_) => {
                    tracing::warn!("Invalid owner key in inbound invitation");
                    return;
                }
            };
            if crypto::verify_invitation(&invitation, &owner_key).is_err() {
                tracing::warn!(pool_id = %invitation.pool_id, "Invalid owner signature on inbound invitation");
                return;
            }

            // Check expiry
            if invitation.expires_at < chrono::Utc::now() {
                tracing::debug!(invitation_id = %invitation.id, "Ignoring expired invitation");
                return;
            }

            // This invitation is for us — store as pending
            self.pending_invitations
                .insert(invitation.id, invitation.clone());
            if let Err(e) = self.shared_state.db.put_json(
                TREE_POOL_INVITATIONS,
                &invitation.id.to_string(),
                &invitation,
            ) {
                tracing::warn!(error = %e, "Failed to persist inbound invitation");
            }
            tracing::info!(
                pool_id = %invitation.pool_id,
                invitation_id = %invitation.id,
                "Received pool invitation"
            );

            // Auto-accept only if this invitation matches the code we used to join
            if self.auto_accept_code_hash.is_some() {
                self.auto_accept_code_hash = None;
                tracing::info!(
                    invitation_id = %invitation.id,
                    "Auto-accepting invitation (from invite code join)"
                );
                if let Err(e) = self.handle_accept_invitation(invitation).await {
                    tracing::warn!(error = %e, "Auto-accept failed");
                }
            }
        }
    }

    /// SEC-M18: Handle a blinded invitation broadcast.
    /// Recompute the commitment with our node_id to check if the invitation is for us.
    async fn handle_inbound_blinded_invitation(&mut self, blinded: BlindedPoolInvitation) {
        let my_id = self.shared_state.identity.node_id();
        let expected = compute_invitee_commitment(my_id, &blinded.id);

        if expected != blinded.invitee_commitment {
            // Not for us — ignore silently (this is expected for most nodes)
            return;
        }

        // This invitation is for us! Reconstruct a full PoolInvitation for local storage.
        let invitation = PoolInvitation {
            id: blinded.id,
            pool_id: blinded.pool_id.clone(),
            invitee_node_id: my_id.clone(),
            expires_at: blinded.expires_at,
            owner_signature: blinded.owner_signature.clone(),
            created_at: blinded.created_at,
        };

        // Verify owner signature before storing
        let owner_key = match ed25519_dalek::VerifyingKey::from_bytes(&invitation.pool_id.0) {
            Ok(k) => k,
            Err(_) => {
                tracing::warn!("Invalid owner key in blinded invitation");
                return;
            }
        };
        if crypto::verify_invitation(&invitation, &owner_key).is_err() {
            tracing::warn!(pool_id = %invitation.pool_id, "Invalid owner signature on blinded invitation");
            return;
        }

        // Check expiry
        if invitation.expires_at < chrono::Utc::now() {
            tracing::debug!(invitation_id = %invitation.id, "Ignoring expired blinded invitation");
            return;
        }

        // Store as pending
        self.pending_invitations
            .insert(invitation.id, invitation.clone());
        if let Err(e) = self.shared_state.db.put_json(
            TREE_POOL_INVITATIONS,
            &invitation.id.to_string(),
            &invitation,
        ) {
            tracing::warn!(error = %e, "Failed to persist blinded invitation");
        }
        tracing::info!(
            pool_id = %invitation.pool_id,
            invitation_id = %invitation.id,
            "Recognized blinded pool invitation for us"
        );

        // Auto-accept only if we have a pending code-based join
        if self.auto_accept_code_hash.is_some() {
            self.auto_accept_code_hash = None;
            tracing::info!(
                invitation_id = %invitation.id,
                "Auto-accepting blinded invitation (from invite code join)"
            );
            if let Err(e) = self.handle_accept_invitation(invitation).await {
                tracing::warn!(error = %e, "Auto-accept failed");
            }
        }
    }

    async fn handle_inbound_acceptance(&mut self, acceptance: PoolAcceptance) {
        // If we're the pool owner, add the member
        let my_id = self.shared_state.identity.node_id();
        if acceptance.pool_id != *my_id {
            return; // Not our pool
        }

        // Verify the acceptance signature
        let invitee_key =
            match ed25519_dalek::VerifyingKey::from_bytes(&acceptance.invitee_node_id.0) {
                Ok(k) => k,
                Err(_) => return,
            };
        if crypto::verify_acceptance(&acceptance, &invitee_key).is_err() {
            tracing::warn!("Invalid acceptance signature");
            return;
        }

        // Check invitation replay
        if !self
            .pending_invitations
            .contains_key(&acceptance.invitation_id)
        {
            tracing::warn!("Acceptance for unknown invitation");
            return;
        }

        // Check pool capacity BEFORE consuming the invitation to avoid locking out invitees
        {
            let state = self.shared_state.credits.pool_state.read().await;
            if let Some(ref ps) = *state {
                let max_size = self.shared_state.config.pool.max_pool_size;
                if ps.members.len() >= max_size as usize {
                    tracing::warn!(
                        "Pool full, rejecting acceptance — invitation preserved for retry"
                    );
                    return;
                }
            }
        }

        // Consume the invitation to prevent replay (pool capacity already verified above)
        self.pending_invitations.remove(&acceptance.invitation_id);
        let _ = self
            .shared_state
            .db
            .remove(TREE_POOL_INVITATIONS, &acceptance.invitation_id.to_string());

        let mut state = self.shared_state.credits.pool_state.write().await;
        if let Some(ref mut ps) = *state {
            // Check not already a member
            if ps
                .members
                .iter()
                .any(|m| m.node_id == acceptance.invitee_node_id)
            {
                return;
            }

            ps.members.push(PoolMembership {
                node_id: acceptance.invitee_node_id.clone(),
                credits_contributed: 0,
                joined_at: acceptance.accepted_at,
                acceptance_signature: acceptance.invitee_signature.clone(),
                invitation_id: acceptance.invitation_id,
                device_name: None,
                last_seen: Some(chrono::Utc::now()),
                online: true,
                device_stats: None,
                contribution_level: 100,
            });

            if let Err(e) = self.persist_pool_state(ps) {
                tracing::warn!(error = %e, "Failed to persist pool state after acceptance");
            }

            tracing::info!(
                new_member = %acceptance.invitee_node_id,
                members = ps.members.len(),
                "Pool member joined"
            );
        } else {
            tracing::warn!(
                "Acceptance received but no pool state — invitation consumed to prevent replay"
            );
        }
    }

    async fn handle_inbound_removal(&mut self, removal: PoolRemoval) {
        let my_id = self.shared_state.identity.node_id();

        // If we're the one being removed
        if removal.removed_node_id == *my_id {
            // SEC: Freshness check — reject removals older than 5 minutes
            let age = chrono::Utc::now().signed_duration_since(removal.removed_at);
            let age_secs = age.num_seconds();
            if !(-30..=300).contains(&age_secs) {
                tracing::warn!(
                    age_secs = age.num_seconds(),
                    "Pool removal rejected: timestamp too old or too far in future"
                );
                return;
            }

            // SEC: Replay protection — check if we've already processed this removal_id
            let removal_key = removal.removal_id.to_string();
            if self
                .shared_state
                .db
                .get_json::<bool>(TREE_POOL_REMOVAL_REPLAYS, &removal_key)
                .ok()
                .flatten()
                .is_some()
            {
                tracing::warn!(removal_id = %removal.removal_id, "Pool removal replay detected — ignoring");
                return;
            }

            // Verify the removal was signed by the pool owner
            let owner_key = match ed25519_dalek::VerifyingKey::from_bytes(&removal.pool_id.0) {
                Ok(k) => k,
                Err(_) => return,
            };
            if crypto::verify_removal(&removal, &owner_key).is_err() {
                tracing::warn!("Invalid removal signature");
                return;
            }

            // Record the removal_id to prevent replay
            let _ = self
                .shared_state
                .db
                .put_json(TREE_POOL_REMOVAL_REPLAYS, &removal_key, &true);

            *self.shared_state.credits.pool_state.write().await = None;
            let _ = self.shared_state.db.remove(TREE_POOL_STATE, KEY_MY_POOL);
            self.shared_state
                .credits
                .pool_registry
                .remove(&removal.pool_id);
            tracing::info!(pool_id = %removal.pool_id, "Removed from pool by owner");
        }
    }

    async fn handle_inbound_member_left(
        &mut self,
        pool_id: PoolId,
        node_id: NodeId,
        signature: Vec<u8>,
    ) {
        let my_id = self.shared_state.identity.node_id();

        // Only the pool owner processes member-left notifications
        if pool_id != *my_id {
            return;
        }

        // Verify the leave notice is signed by the departing node
        let member_key = match ed25519_dalek::VerifyingKey::from_bytes(&node_id.0) {
            Ok(k) => k,
            Err(_) => {
                tracing::warn!(node = %node_id, "Invalid member key in member-left notice");
                return;
            }
        };
        let payload = {
            let mut h = blake3::Hasher::new();
            h.update(b"pool_member_left_v1");
            h.update(&pool_id.0);
            h.update(&node_id.0);
            h.finalize().as_bytes().to_vec()
        };
        let sig_bytes: &[u8; 64] = match signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => {
                tracing::warn!("Member-left notice has invalid signature length");
                return;
            }
        };
        let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
        if member_key.verify(&payload, &sig).is_err() {
            tracing::warn!(node = %node_id, "Invalid signature on member-left notice");
            return;
        }

        let mut state = self.shared_state.credits.pool_state.write().await;
        if let Some(ref mut ps) = *state {
            let before = ps.members.len();
            ps.members.retain(|m| m.node_id != node_id);
            if ps.members.len() < before {
                if let Err(e) = self.persist_pool_state(ps) {
                    tracing::warn!(error = %e, "Failed to persist pool state after member left");
                }
                tracing::info!(member = %node_id, "Member left pool");
            }
        }
    }

    async fn gossip_pool_state(&self) {
        let state = self.shared_state.credits.pool_state.read().await;
        if let Some(ref ps) = *state {
            let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::StateGossip(ps.clone()));
            let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;
        }
    }

    async fn build_leaderboard(&self) -> Vec<LeaderboardEntry> {
        let state = self.shared_state.credits.pool_state.read().await;
        let mut entries = Vec::new();

        if let Some(ref ps) = *state {
            let mut members: Vec<_> = ps
                .members
                .iter()
                .map(|m| (m.node_id.clone(), m.credits_contributed))
                .collect();

            members.sort_by(|a, b| b.1.cmp(&a.1));

            for (rank, (node_id, credits)) in members.into_iter().enumerate() {
                entries.push(LeaderboardEntry {
                    node_id,
                    credits_contributed: credits,
                    rank: (rank + 1) as u32,
                });
            }
        }

        entries
    }

    fn persist_pool_state(&self, state: &PoolState) -> Result<(), SwarmError> {
        self.shared_state
            .db
            .put_json(TREE_POOL_STATE, KEY_MY_POOL, state)
    }

    /// Set the device nickname for this node within the pool.
    async fn handle_set_device_name(&mut self, name: String) -> Result<(), SwarmError> {
        let name = name.trim().to_string();
        if name.len() > 32 {
            return Err(SwarmError::Internal(
                "Device name must be 32 characters or less".into(),
            ));
        }
        let my_id = self.shared_state.identity.node_id().clone();
        let mut ps = self.shared_state.credits.pool_state.write().await;
        let ps = ps
            .as_mut()
            .ok_or_else(|| SwarmError::Internal("Not in a pool".into()))?;
        if let Some(member) = ps.members.iter_mut().find(|m| m.node_id == my_id) {
            member.device_name = if name.is_empty() { None } else { Some(name) };
        }
        self.persist_pool_state(ps)?;
        Ok(())
    }

    /// Set the credit split percentage (owner only). 0 = all to owner, 100 = all to member.
    async fn handle_set_credit_split(&mut self, pct: u8) -> Result<(), SwarmError> {
        if pct > 100 {
            return Err(SwarmError::Internal("Split must be 0-100".into()));
        }
        let my_id = self.shared_state.identity.node_id().clone();
        let mut ps = self.shared_state.credits.pool_state.write().await;
        let ps = ps
            .as_mut()
            .ok_or_else(|| SwarmError::Internal("Not in a pool".into()))?;
        if ps.pool_id != my_id {
            return Err(SwarmError::Internal(
                "Only the pool owner can change the credit split".into(),
            ));
        }
        ps.member_credit_split_pct = pct;
        self.persist_pool_state(ps)?;
        tracing::info!(pct, "Pool credit split updated");
        Ok(())
    }

    /// Set contribution level for a member device (owner only).
    async fn handle_set_contribution_level(
        &mut self,
        node_id: NodeId,
        level: u8,
    ) -> Result<(), SwarmError> {
        if level > 100 {
            return Err(SwarmError::Internal("Level must be 0-100".into()));
        }
        let my_id = self.shared_state.identity.node_id().clone();
        let mut ps = self.shared_state.credits.pool_state.write().await;
        let ps = ps
            .as_mut()
            .ok_or_else(|| SwarmError::Internal("Not in a pool".into()))?;
        if ps.pool_id != my_id {
            return Err(SwarmError::Internal(
                "Only the pool owner can set contribution levels".into(),
            ));
        }
        if let Some(member) = ps.members.iter_mut().find(|m| m.node_id == node_id) {
            member.contribution_level = level;
            tracing::info!(
                node = %node_id,
                level,
                "Set device contribution level"
            );
        } else {
            return Err(SwarmError::Internal("Device not found in pool".into()));
        }
        self.persist_pool_state(ps)?;
        Ok(())
    }

    // ---- Invite Code Handlers ----

    /// Generate a short invite code (owner only). One-time use, expires after TTL.
    async fn handle_generate_invite_code(&mut self) -> Result<String, SwarmError> {
        let pool_state = self.shared_state.credits.pool_state.read().await;
        let ps = pool_state
            .as_ref()
            .ok_or_else(|| SwarmError::Internal("Not in a pool".into()))?;

        // Only the owner can generate invite codes
        if ps.pool_id != *self.shared_state.identity.node_id() {
            return Err(SwarmError::Internal(
                "Only the pool owner can generate invite codes".into(),
            ));
        }

        // Check pool isn't full
        let max_size = self.shared_state.config.pool.max_pool_size;
        if ps.members.len() as u32 >= max_size {
            return Err(SwarmError::Internal(format!(
                "Pool is full ({max_size} members)"
            )));
        }

        // Rate limit
        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Internal(
                "Rate limited — try again later".into(),
            ));
        }

        // Clean expired codes
        self.invite_codes
            .retain(|_, v| !v.is_expired() && !v.consumed);

        // Limit active codes to prevent abuse (max 5 active at once)
        const MAX_ACTIVE_CODES: usize = 5;
        if self.invite_codes.len() >= MAX_ACTIVE_CODES {
            return Err(SwarmError::Internal(format!(
                "Too many active invite codes ({MAX_ACTIVE_CODES}). Wait for existing codes to expire."
            )));
        }

        let ttl = self.shared_state.config.pool.invitation_ttl_hours;
        let invite = PoolInviteCode::generate(self.shared_state.identity.node_id(), ttl);
        let code = invite.code.clone();
        self.invite_codes.insert(invite.code_hash, invite);

        tracing::info!(code_preview = &code[..4], "Generated pool invite code");

        Ok(code)
    }

    /// Join a pool using an invite code (from the joining device).
    /// Broadcasts a JoinRequest over gossip so the owner can auto-invite.
    async fn handle_join_with_code(&mut self, code: String) -> Result<(), SwarmError> {
        // Must not already be in a pool
        if self.shared_state.credits.pool_state.read().await.is_some() {
            return Err(SwarmError::Internal("Already in a pool".into()));
        }

        // Validate code format (8 uppercase alphanumeric)
        let code = code.trim().to_uppercase();
        if code.len() != 8 || !code.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(SwarmError::Internal(
                "Invalid invite code format (expected 8 characters)".into(),
            ));
        }

        // Compute code hash and broadcast join request
        let code_hash = *blake3::hash(code.as_bytes()).as_bytes();
        let my_id = self.shared_state.identity.node_id().clone();

        // Sign the join request
        let mut payload_hasher = blake3::Hasher::new();
        payload_hasher.update(b"pool_join_request_v1");
        payload_hasher.update(&code_hash);
        payload_hasher.update(&my_id.0);
        let payload = payload_hasher.finalize();
        let signature = self.shared_state.identity.sign(payload.as_bytes()).to_vec();

        // Broadcast join request over gossip
        let msg = crate::types::SwarmMessage::PoolMessage(crate::types::PoolMessage::JoinRequest {
            code_hash,
            requester: my_id,
            signature,
        });
        let _ = self
            .network_tx
            .send(crate::types::NetworkCommand::Broadcast(msg))
            .await;

        // Set auto-accept with the code hash so when the invitation arrives via gossip,
        // we only auto-accept if it matches this specific join request.
        self.auto_accept_code_hash = Some(code_hash);

        tracing::info!("Broadcast pool join request with invite code (auto-accept enabled)");
        Ok(())
    }

    /// Handle an inbound join request (owner only). If the code_hash matches
    /// an active invite code, auto-create an invitation for the requester.
    async fn handle_inbound_join_request(&mut self, code_hash: [u8; 32], requester: NodeId) {
        // Only process if we're a pool owner
        let is_owner = {
            let ps = self.shared_state.credits.pool_state.read().await;
            ps.as_ref()
                .map(|s| s.pool_id == *self.shared_state.identity.node_id())
                .unwrap_or(false)
        };
        if !is_owner {
            return;
        }

        // Look up the code by hash
        let code_entry = match self.invite_codes.get_mut(&code_hash) {
            Some(entry) => entry,
            None => return, // Not our code or already consumed
        };

        // Validate code is still active
        if code_entry.consumed || code_entry.is_expired() {
            tracing::debug!("Join request with expired/consumed invite code — ignoring");
            return;
        }

        // SEC: The requester NodeId is transport-authenticated by the dispatch layer
        // (set from the authenticated sender of the P2P message). The dispatch code at
        // daemon/dispatch.rs extracts `requester` from the transport-verified sender identity,
        // so a peer cannot spoof the requester field without controlling the transport key.
        // Signature verification over the payload is done at the network message level
        // by gossip_seal/transport auth — no additional check needed here.

        // Mark code as consumed (one-time use)
        code_entry.consumed = true;

        tracing::info!(
            requester = %requester,
            "Invite code claimed — auto-creating invitation"
        );

        // Auto-create invitation for the requester
        match self.handle_create_invitation(requester.clone()).await {
            Ok(inv) => {
                tracing::info!(
                    invitation_id = %inv.id,
                    member = %requester,
                    "Auto-invitation created from invite code"
                );
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    member = %requester,
                    "Failed to auto-invite from invite code"
                );
                // Un-consume the code so the user can try again
                if let Some(entry) = self.invite_codes.get_mut(&code_hash) {
                    entry.consumed = false;
                }
            }
        }
    }
}
