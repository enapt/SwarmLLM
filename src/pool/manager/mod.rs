use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use ed25519_dalek::Verifier;

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::pool::crypto;
use crate::pool::types::*;
use crate::types::{NetworkCommand, NodeId, SwarmMessage};

mod gossip;

/// Database tree names for pool persistence.
pub(crate) const TREE_POOL_STATE: &str = "pool_state";
const TREE_POOL_INVITATIONS: &str = "pool_invitations";
const TREE_POOL_FORWARDS: &str = "pool_forwards";
const TREE_POOL_REMOVAL_REPLAYS: &str = "pool_removal_replays";
pub(crate) const KEY_MY_POOL: &str = "my_pool";

/// Max lifetime of a pending auto-accept intent created by a code-based join.
/// Used both by the periodic expiry sweep and the inbound invitation handler.
const AUTO_ACCEPT_TIMEOUT_SECS: u64 = 300;

/// Maximum pending invitations (outbound + inbound blinded) before rejecting new ones.
const MAX_PENDING_INVITATIONS: usize = 50;
/// Maximum active invite codes before refusing to generate more.
const MAX_ACTIVE_CODES: usize = 5;

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
    /// When set, auto-accept the next invitation (set by JoinWithCode).
    /// Includes a timestamp so the intent expires after 5 minutes.
    auto_accept_code_hash: Option<([u8; 32], std::time::Instant)>,
    /// Per-member sliding window of recent credit-forward timestamps.
    /// Caps a single member at CREDIT_FORWARD_MAX_PER_WINDOW forwards per
    /// CREDIT_FORWARD_WINDOW_SECS — defeats the UUID-replay-with-fresh-id
    /// vector (each forward gets a fresh UUID so the DB dedup key alone
    /// cannot detect repeated claims).
    credit_forward_rl: HashMap<NodeId, std::collections::VecDeque<std::time::Instant>>,
}

const CREDIT_FORWARD_WINDOW_SECS: u64 = 60;
const CREDIT_FORWARD_MAX_PER_WINDOW: usize = 60;

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
            credit_forward_rl: HashMap::new(),
        }
    }

    /// Returns true if this member is under the credit-forward rate limit.
    fn check_credit_forward_rate(&mut self, member: &NodeId) -> bool {
        let window = std::time::Duration::from_secs(CREDIT_FORWARD_WINDOW_SECS);
        let now = std::time::Instant::now();
        let entry = self.credit_forward_rl.entry(member.clone()).or_default();
        while entry
            .front()
            .is_some_and(|t| now.duration_since(*t) > window)
        {
            entry.pop_front();
        }
        if entry.len() >= CREDIT_FORWARD_MAX_PER_WINDOW {
            return false;
        }
        entry.push_back(now);
        true
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

        // Clear stale removal replay entries — the timestamp freshness check (±5min)
        // already prevents replays, so persisted entries serve no purpose after restart.
        let _ = db.clear_tree(TREE_POOL_REMOVAL_REPLAYS);

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

        tracing::info!(target: "swarmllm::pool::manager", "PoolManager running");

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!(target: "swarmllm::pool::manager", "PoolManager shutting down");
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
                    self.send_device_stats_report().await;
                    // Expire stale auto-accept intent
                    if let Some((_, created_at)) = self.auto_accept_code_hash {
                        if created_at.elapsed().as_secs() >= AUTO_ACCEPT_TIMEOUT_SECS {
                            tracing::debug!("Clearing expired auto-accept code hash");
                            self.auto_accept_code_hash = None;
                        }
                    }
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
                left_at,
                nonce,
                signature,
            } => {
                self.handle_inbound_member_left(pool_id, node_id, left_at, nonce, signature)
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
            PoolCommand::GetLeaderboard { reply } => {
                let leaderboard = self.build_leaderboard().await;
                let _ = reply.send(leaderboard);
            }
            PoolCommand::InboundDeviceStatsReport {
                pool_id,
                node_id,
                device_name,
                stats,
            } => {
                self.handle_inbound_device_stats_report(pool_id, node_id, device_name, stats)
                    .await;
            }
        }
    }

    async fn handle_create_pool(&mut self, name: String) -> Result<PoolState, SwarmError> {
        // Validate pool name: 1-64 chars, printable ASCII, no control chars
        if name.is_empty() || name.len() > 64 {
            return Err(SwarmError::Validation(
                "Pool name must be 1-64 characters".into(),
            ));
        }
        if !name.chars().all(|c| c.is_ascii_graphic() || c == ' ') {
            return Err(SwarmError::Validation(
                "Pool name may only contain printable ASCII characters".into(),
            ));
        }

        // Check we're not already in a pool
        if self.shared_state.credits.pool_state.read().await.is_some() {
            return Err(SwarmError::Validation("Already in a pool".into()));
        }

        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Validation("Rate limit exceeded".into()));
        }

        let my_id = self.shared_state.identity.node_id().clone();
        let now = chrono::Utc::now();

        // Sign the pool creation
        let payload = super::crypto::pool_create_payload(&my_id, &name, &now);
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
            shard_pins: Vec::new(),
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
                .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;

            let my_id = self.shared_state.identity.node_id();
            if state.pool_id != *my_id {
                return Err(SwarmError::Validation(
                    "Only the pool owner can invite".into(),
                ));
            }

            let max_size = self.shared_state.config.pool.max_pool_size;
            if state.members.len() >= max_size as usize {
                return Err(SwarmError::Validation(format!(
                    "Pool is full (max {max_size} members)"
                )));
            }

            if state.members.iter().any(|m| m.node_id == invitee) {
                return Err(SwarmError::Validation(
                    "Node is already a pool member".into(),
                ));
            }

            state.pool_id.clone()
        };

        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Validation("Rate limit exceeded".into()));
        }

        // Prune expired before checking cap
        let now = chrono::Utc::now();
        self.pending_invitations
            .retain(|_, inv| inv.expires_at > now);
        if self.pending_invitations.len() >= MAX_PENDING_INVITATIONS {
            return Err(SwarmError::Validation(format!(
                "Too many pending invitations ({MAX_PENDING_INVITATIONS}). Wait for existing ones to expire or be accepted."
            )));
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
            return Err(SwarmError::Validation("Already in a pool".into()));
        }

        // Verify the invitation is for us
        let my_id = self.shared_state.identity.node_id();
        if invitation.invitee_node_id != *my_id {
            return Err(SwarmError::Validation(
                "Invitation is not for this node".into(),
            ));
        }

        // Check expiry
        if invitation.expires_at < chrono::Utc::now() {
            return Err(SwarmError::Validation("Invitation has expired".into()));
        }

        // Verify owner signature (pool_id == owner's NodeId)
        let owner_key = ed25519_dalek::VerifyingKey::from_bytes(&invitation.pool_id.0)
            .map_err(|_| SwarmError::Validation("Invalid pool owner key".into()))?;
        crypto::verify_invitation(&invitation, &owner_key)?;

        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Validation("Rate limit exceeded".into()));
        }

        // Create acceptance
        let acceptance = crypto::create_acceptance(&self.shared_state.identity, &invitation);

        // Auto-set device_name from identity nickname if available
        let device_name = {
            let store = crate::identity::nickname::NicknameStore::new(self.shared_state.db.clone());
            store.get_prefs().ok().and_then(|p| p.nickname)
        };

        // Set our pool state as a member of this pool
        let membership = PoolMembership {
            node_id: my_id.clone(),
            credits_contributed: 0,
            joined_at: chrono::Utc::now(),
            acceptance_signature: acceptance.invitee_signature.clone(),
            invitation_id: invitation.id,
            device_name,
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
            shard_pins: Vec::new(),
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

        self.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "pool",
                "pool_device_joined",
                "Joined device pool".to_string(),
            )
            .with_node(format!("{}", invitation.pool_id))
            .with_toast("success", 5000),
        );

        // Immediately send our nickname + stats to the leader
        self.send_device_stats_report().await;

        Ok(())
    }

    async fn handle_remove_member(&mut self, node_id: NodeId) -> Result<(), SwarmError> {
        // Check rate limit before mutating state
        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Validation("Rate limit exceeded".into()));
        }

        let (removal, state_clone) = {
            let mut guard = self.shared_state.credits.pool_state.write().await;
            let ps = guard
                .as_mut()
                .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;

            let my_id = self.shared_state.identity.node_id();
            if ps.pool_id != *my_id {
                return Err(SwarmError::Validation(
                    "Only the pool owner can remove members".into(),
                ));
            }

            if node_id == *my_id {
                return Err(SwarmError::Validation(
                    "Owner cannot remove themselves".into(),
                ));
            }

            let before = ps.members.len();
            ps.members.retain(|m| m.node_id != node_id);
            if ps.members.len() == before {
                return Err(SwarmError::Validation("Node is not a pool member".into()));
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
        // Extract pool_id under a read guard, then drop the guard before
        // the rate-limit check. Holding the read lock across
        // check_and_record() blocks concurrent writers for no reason
        // (the actual mutation happens later under a write lock, where
        // any state change between then and now is naturally observed).
        // Mirrors the lock-then-extract-then-rate-limit pattern in
        // handle_create_invitation / handle_accept_invitation.
        let pool_id = {
            let guard = self.shared_state.credits.pool_state.read().await;
            let ps = guard
                .as_ref()
                .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;
            ps.pool_id.clone()
        };

        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Validation("Rate limit exceeded".into()));
        }

        // Clear pool state — DB first, then memory, to prevent inconsistency on DB failure.
        // If DB write fails, we return error with in-memory state still intact (correct for retry).
        // If process crashes after DB clear but before memory clear, restart sees no DB record (correct).
        self.shared_state.db.remove(TREE_POOL_STATE, KEY_MY_POOL)?;
        *self.shared_state.credits.pool_state.write().await = None;
        self.shared_state.credits.pool_registry.remove(&pool_id);

        // Broadcast signed member-left notice
        let my_id = self.shared_state.identity.node_id().clone();
        let left_at = chrono::Utc::now().timestamp();
        let nonce = uuid::Uuid::new_v4();
        let leave_payload = super::crypto::member_left_payload(&pool_id, &my_id, left_at, &nonce);
        let leave_signature = self.shared_state.identity.sign(&leave_payload);
        let pool_id_short = hex::encode(&pool_id.0[..8]);
        let msg = SwarmMessage::PoolMessage(crate::types::PoolMessage::MemberLeft {
            pool_id,
            node_id: my_id,
            left_at,
            nonce,
            signature: leave_signature,
        });
        let _ = self.network_tx.send(NetworkCommand::Broadcast(msg)).await;

        tracing::info!(pool_id = %pool_id_short, "Left device pool");

        self.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "pool",
                "pool_device_left",
                "Left device pool".to_string(),
            )
            .with_toast("info", 5000),
        );

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

        // Rate-limit per-member forwards. The UUID is member-generated, so the DB
        // dedup above only blocks exact-UUID replays — a fresh UUID with identical
        // amount/timestamp would bypass it. Rate-limiting bounds the exploitability.
        if !self.check_credit_forward_rate(&forward.from_node_id) {
            tracing::warn!(
                from = %forward.from_node_id,
                window_secs = CREDIT_FORWARD_WINDOW_SECS,
                max = CREDIT_FORWARD_MAX_PER_WINDOW,
                "Credit forward rate limit exceeded — dropping"
            );
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
                tracing::warn!(from = %forward.from_node_id, "Credit forward received but no pool state — rejecting");
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

        // SEC-C2: Apply credit to the pool owner's balance FIRST, then persist
        // the audit-log dedup entry. If balance apply fails (transient redb
        // error, lock contention) we return early — the dedup table stays
        // empty so the same forward UUID can be retried by the member next
        // tick. The reverse ordering would permanently lose the credit:
        // dedup entry written → balance apply fails → retry hits dedup,
        // silently drops, owner never credited. Same pattern as escrow.rs:
        // 114 (apply_credit_direct) → 124 (put_json audit log).
        //
        // The remaining race is a process crash between balance apply and
        // audit-log write: balance applied, dedup empty, next retry double-
        // credits. That window is microseconds and crashes are user-driven,
        // so manual reconciliation is acceptable.
        if let Err(e) = crate::credit::ledger::apply_credit_direct(
            &self.shared_state.credits.credit_balance,
            &self.shared_state.db,
            forward.amount,
            crate::credit::ledger::CreditDelta::Earning,
        )
        .await
        {
            tracing::warn!(
                error = %e,
                from = %forward.from_node_id,
                id = %forward.id,
                "Failed to apply forwarded credits to owner balance — leaving dedup empty so member can retry"
            );
            return;
        }

        // Store in audit log (dedup table). Errors here mean the balance
        // already moved but we couldn't persist the dedup entry — log a
        // clear error so the operator can manually reconcile.
        if let Err(e) =
            self.shared_state
                .db
                .put_json(TREE_POOL_FORWARDS, &forward.id.to_string(), &forward)
        {
            tracing::error!(
                error = %e,
                from = %forward.from_node_id,
                id = %forward.id,
                amount = forward.amount,
                "Credit forward applied to balance but FAILED to persist dedup entry — \
                 a retry of this forward will double-credit. Manual reconciliation required."
            );
        }

        // Update the member's contribution in pool state.
        // Clone+drop before DB write to avoid holding write lock across I/O.
        let snapshot = {
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
                ps.total_lifetime_credits =
                    ps.total_lifetime_credits.saturating_add(forward.amount);
                Some(ps.clone())
            } else {
                None
            }
        };
        if let Some(ref ps) = snapshot {
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
                tracing::warn!(pool_id = %hex::encode(&invitation.pool_id.0[..8]), "Invalid owner key in blinded invitation");
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

        // Prune expired invitations before inserting to prevent unbounded growth
        let now = chrono::Utc::now();
        self.pending_invitations
            .retain(|_, inv| inv.expires_at > now);

        if self.pending_invitations.len() >= MAX_PENDING_INVITATIONS {
            tracing::debug!("Pending invitations at capacity ({MAX_PENDING_INVITATIONS}), dropping inbound blinded invitation");
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

        // Auto-accept only if we have a pending code-based join that hasn't expired
        if let Some((_, created_at)) = self.auto_accept_code_hash {
            if created_at.elapsed().as_secs() < AUTO_ACCEPT_TIMEOUT_SECS {
                self.auto_accept_code_hash = None;
                tracing::info!(
                    invitation_id = %invitation.id,
                    pool_id = %invitation.pool_id,
                    "Auto-accepting blinded invitation (from invite code join)"
                );
                if let Err(e) = self.handle_accept_invitation(invitation).await {
                    tracing::warn!(error = %e, "Auto-accept failed");
                }
            } else {
                // Expired — clear the stale auto-accept intent
                self.auto_accept_code_hash = None;
                tracing::info!(
                    "Auto-accept expired (>5min) — invitation stored but not auto-accepted"
                );
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
            tracing::warn!(invitee = %acceptance.invitee_node_id, "Invalid acceptance signature");
            return;
        }

        // Check invitation replay
        if !self
            .pending_invitations
            .contains_key(&acceptance.invitation_id)
        {
            tracing::warn!(invitation_id = %acceptance.invitation_id, invitee = %acceptance.invitee_node_id, "Acceptance for unknown invitation");
            return;
        }

        // Check pool capacity BEFORE consuming the invitation to avoid locking out invitees
        {
            let state = self.shared_state.credits.pool_state.read().await;
            if let Some(ref ps) = *state {
                let max_size = self.shared_state.config.pool.max_pool_size;
                if ps.members.len() >= max_size as usize {
                    tracing::warn!(
                        invitee = %acceptance.invitee_node_id,
                        invitation_id = %acceptance.invitation_id,
                        current_members = ps.members.len(),
                        max_size,
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

        // Clone+drop before DB write to avoid holding write lock across I/O.
        let (snapshot, member_count) = {
            let mut state = self.shared_state.credits.pool_state.write().await;
            if let Some(ref mut ps) = *state {
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
                (Some(ps.clone()), ps.members.len())
            } else {
                tracing::warn!(
                    "Acceptance received but no pool state — invitation consumed to prevent replay"
                );
                (None, 0)
            }
        };
        if let Some(ref ps) = snapshot {
            if let Err(e) = self.persist_pool_state(ps) {
                tracing::warn!(error = %e, "Failed to persist pool state after acceptance");
            }
            tracing::info!(
                new_member = %acceptance.invitee_node_id,
                members = member_count,
                "Pool member joined"
            );
            // Immediately gossip updated pool state so the new member sees full membership
            self.gossip_pool_state().await;
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
                tracing::warn!(pool_id = %removal.pool_id, "Invalid removal signature");
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
        left_at: i64,
        nonce: uuid::Uuid,
        signature: Vec<u8>,
    ) {
        let my_id = self.shared_state.identity.node_id();

        // Only the pool owner processes member-left notifications
        if pool_id != *my_id {
            return;
        }

        // Freshness check: reject notices more than 5 minutes out of range.
        let now = chrono::Utc::now().timestamp();
        if (now - left_at).abs() > 300 {
            tracing::warn!(node = %node_id, left_at, now, "Stale member-left notice — rejecting");
            return;
        }

        // Replay protection: same tree as pool removals, keyed by nonce.
        let replay_key = format!("ml-{}", nonce);
        if self
            .shared_state
            .db
            .get_json::<bool>(TREE_POOL_REMOVAL_REPLAYS, &replay_key)
            .ok()
            .flatten()
            .is_some()
        {
            tracing::warn!(node = %node_id, nonce = %nonce, "Member-left replay detected — ignoring");
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
        let payload = super::crypto::member_left_payload(&pool_id, &node_id, left_at, &nonce);
        let sig_bytes: &[u8; 64] = match signature.as_slice().try_into() {
            Ok(b) => b,
            Err(_) => {
                tracing::warn!(node = %node_id, "Member-left notice has invalid signature length");
                return;
            }
        };
        let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
        if member_key.verify(&payload, &sig).is_err() {
            tracing::warn!(node = %node_id, "Invalid signature on member-left notice");
            return;
        }

        // Record nonce to prevent replay.
        if let Err(e) = self
            .shared_state
            .db
            .put_json(TREE_POOL_REMOVAL_REPLAYS, &replay_key, &true)
        {
            tracing::warn!(error = %e, "Failed to persist member-left replay key");
        }

        // Clone+drop before DB write to avoid holding write lock across I/O.
        let snapshot = {
            let mut state = self.shared_state.credits.pool_state.write().await;
            if let Some(ref mut ps) = *state {
                let before = ps.members.len();
                ps.members.retain(|m| m.node_id != node_id);
                if ps.members.len() < before {
                    Some(ps.clone())
                } else {
                    None
                }
            } else {
                None
            }
        };
        if let Some(ref ps) = snapshot {
            if let Err(e) = self.persist_pool_state(ps) {
                tracing::warn!(error = %e, "Failed to persist pool state after member left");
            }
            tracing::info!(member = %node_id, "Member left pool");
        }
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
            return Err(SwarmError::Validation(
                "Device name must be 32 characters or less".into(),
            ));
        }
        let my_id = self.shared_state.identity.node_id().clone();
        let snapshot = {
            let mut ps = self.shared_state.credits.pool_state.write().await;
            let ps = ps
                .as_mut()
                .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;
            if let Some(member) = ps.members.iter_mut().find(|m| m.node_id == my_id) {
                member.device_name = if name.is_empty() { None } else { Some(name) };
            }
            ps.clone()
        };
        self.persist_pool_state(&snapshot)?;
        Ok(())
    }

    /// Set the credit split percentage (owner only). 0 = all to owner, 100 = all to member.
    async fn handle_set_credit_split(&mut self, pct: u8) -> Result<(), SwarmError> {
        if pct > 100 {
            return Err(SwarmError::Validation("Split must be 0-100".into()));
        }
        let my_id = self.shared_state.identity.node_id().clone();
        let snapshot = {
            let mut ps = self.shared_state.credits.pool_state.write().await;
            let ps = ps
                .as_mut()
                .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;
            if ps.pool_id != my_id {
                return Err(SwarmError::Validation(
                    "Only the pool owner can change the credit split".into(),
                ));
            }
            ps.member_credit_split_pct = pct;
            ps.clone()
        };
        self.persist_pool_state(&snapshot)?;
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
            return Err(SwarmError::Validation("Level must be 0-100".into()));
        }
        let my_id = self.shared_state.identity.node_id().clone();
        let snapshot = {
            let mut ps = self.shared_state.credits.pool_state.write().await;
            let ps = ps
                .as_mut()
                .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;
            if ps.pool_id != my_id {
                return Err(SwarmError::Validation(
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
                return Err(SwarmError::Validation("Device not found in pool".into()));
            }
            ps.clone()
        };
        self.persist_pool_state(&snapshot)?;
        Ok(())
    }

    // ---- Invite Code Handlers ----

    /// Generate a short invite code (owner only). One-time use, expires after TTL.
    async fn handle_generate_invite_code(&mut self) -> Result<String, SwarmError> {
        let pool_state = self.shared_state.credits.pool_state.read().await;
        let ps = pool_state
            .as_ref()
            .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;

        // Only the owner can generate invite codes
        if ps.pool_id != *self.shared_state.identity.node_id() {
            return Err(SwarmError::Validation(
                "Only the pool owner can generate invite codes".into(),
            ));
        }

        // Check pool isn't full
        let max_size = self.shared_state.config.pool.max_pool_size;
        if ps.members.len() as u32 >= max_size {
            return Err(SwarmError::Validation(format!(
                "Pool is full ({max_size} members)"
            )));
        }

        // Rate limit
        if !self.rate_limiter.check_and_record() {
            return Err(SwarmError::Validation(
                "Rate limited — try again later".into(),
            ));
        }

        // Clean expired codes
        self.invite_codes
            .retain(|_, v| !v.is_expired() && !v.consumed);

        if self.invite_codes.len() >= MAX_ACTIVE_CODES {
            return Err(SwarmError::Validation(format!(
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
            return Err(SwarmError::Validation("Already in a pool".into()));
        }

        // Validate code format (8 uppercase alphanumeric)
        let code = code.trim().to_uppercase();
        if code.len() != 8 || !code.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Err(SwarmError::Validation(
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

        // Set auto-accept with the code hash and a timestamp so it expires after 5 minutes.
        self.auto_accept_code_hash = Some((code_hash, std::time::Instant::now()));

        tracing::info!(code_hash = %hex::encode(code_hash), "Broadcast pool join request with invite code (auto-accept enabled)");
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
