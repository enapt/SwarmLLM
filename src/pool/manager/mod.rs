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
/// SEC: invite codes persistence. Without this, a pool owner restart between
/// `JoinWithCode` arriving and the invitation being sent loses the code from
/// memory; the joiner's `auto_accept_code_hash` expires silently and the
/// join always requires manual intervention. Keyed on the hex-encoded
/// `code_hash` (32 bytes → 64 chars).
const TREE_POOL_INVITE_CODES: &str = "pool_invite_codes";
pub(crate) const KEY_MY_POOL: &str = "my_pool";

/// Tree for persisted per-node mode bools (`private_mode`, `offline_mode`).
///
/// **Why a separate tree from `pool_state`** (R105 deferral closure):
/// these flags used to share the `pool_state` tree with `PoolState` JSON
/// records keyed by `my_pool`. The risk was a future reader iterating
/// `pool_state` with `iter_json::<PoolState>` (e.g. an audit tool, or a
/// stricter `check_integrity` validator) hitting the `bool` payloads under
/// `private_mode`/`offline_mode` keys and reporting them as corrupt. Moving
/// the mode flags out of `pool_state` removes the namespace collision and
/// keeps each tree single-typed.
pub(crate) const TREE_NODE_MODES: &str = "node_modes";
pub(crate) const KEY_PRIVATE_MODE: &str = "private_mode";
pub(crate) const KEY_OFFLINE_MODE: &str = "offline_mode";

/// Restore a per-node mode flag (`private_mode` / `offline_mode`) from
/// persistent storage, migrating any legacy entry left behind in the
/// `pool_state` tree by daemons predating R138.
///
/// Lookup order:
/// 1. `node_modes/{key}` — the canonical home (post-R138).
/// 2. `pool_state/{key}` — legacy path. If found, the value is **moved**
///    (written into `node_modes`, removed from `pool_state`) so a single
///    restart finishes the migration and the namespace-collision risk
///    that motivated this split (R105) goes away permanently for that
///    node.
/// 3. `None` — caller falls back to the config default.
///
/// Errors on the write/remove half of the migration are intentionally
/// swallowed with a warn-level log: the in-memory value is still correct
/// for this run, and the next restart will retry the migration.
pub(crate) fn restore_node_mode(db: &crate::storage::db::Database, key: &str) -> Option<bool> {
    if let Ok(Some(v)) = db.get_json::<bool>(TREE_NODE_MODES, key) {
        return Some(v);
    }
    if let Ok(Some(v)) = db.get_json::<bool>(TREE_POOL_STATE, key) {
        if let Err(e) = db.put_json(TREE_NODE_MODES, key, &v) {
            tracing::warn!(
                error = %e,
                key,
                "Failed to migrate legacy pool_state mode flag into node_modes — will retry next restart"
            );
            // Still return v: the runtime read succeeded, only the migration write failed.
            return Some(v);
        }
        if let Err(e) = db.remove(TREE_POOL_STATE, key) {
            tracing::warn!(
                error = %e,
                key,
                "Migrated mode flag into node_modes but failed to delete legacy pool_state entry; benign — next restart will skip migration"
            );
        }
        return Some(v);
    }
    None
}

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
    /// Per-member sliding window of recent credit-forward `(timestamp, amount)`
    /// pairs. Caps a single member at `CREDIT_FORWARD_MAX_PER_WINDOW` forwards
    /// AND at `CREDIT_FORWARD_MAX_VALUE_PER_WINDOW` cumulative credits per
    /// `CREDIT_FORWARD_WINDOW_SECS` — the count cap defeats the
    /// UUID-replay-with-fresh-id vector, the value cap (R138, closes R102
    /// deferral) bounds the worst-case TOTAL credit transfer a single member
    /// can attempt per window even if they stay under the count cap.
    credit_forward_rl: HashMap<NodeId, std::collections::VecDeque<(std::time::Instant, i64)>>,
    /// R131: timestamp of the most recent successful PoolState gossip
    /// broadcast. Used by `maybe_gossip_pool_state` to debounce bursty
    /// re-broadcasts (the "flood under 50-member rotation" case from
    /// FUTURE_WORK.md). `None` until the first broadcast.
    last_pool_gossip_at: Option<std::time::Instant>,
    /// R131: set by `maybe_gossip_pool_state` when a broadcast is
    /// suppressed by the debounce window. The pool-coalesce timer in
    /// the run loop drains this flag on its next tick after the
    /// `POOL_GOSSIP_MIN_INTERVAL` cooldown, ensuring a single trailing
    /// broadcast catches up bursty changes.
    pool_gossip_dirty: bool,
    /// R134: snapshot of the `PoolState` as it was at the last broadcast
    /// (full OR diff). Used as the baseline for computing the next
    /// outgoing diff — its `generation` is the `parent_generation` field
    /// the receiver expects. `None` until the first broadcast.
    last_broadcast_state: Option<PoolState>,
    /// R134: number of diff broadcasts emitted since the last full
    /// broadcast. When this hits `MAX_DIFFS_BEFORE_FULL`, the next
    /// broadcast is forced full so receivers that missed earlier diffs
    /// recover bounded-time.
    diffs_since_full: u32,
}

const CREDIT_FORWARD_WINDOW_SECS: u64 = 60;
const CREDIT_FORWARD_MAX_PER_WINDOW: usize = 60;

/// R138 (closes R102 deferral) — cumulative credit-forward value a single
/// member can attempt within `CREDIT_FORWARD_WINDOW_SECS`. Sized at
/// `2 * max_transaction_amount` (200k) — defense-in-depth above the
/// existing per-tx 100k cap. A legitimate slave device forwarding for
/// per-token credit spend at OpenAI-level rates never approaches this;
/// a member sustaining forwards near the per-tx max gets a hint to slow
/// down. Independent of the count cap (60 forwards/window) — either limit
/// alone is sufficient to reject.
const CREDIT_FORWARD_MAX_VALUE_PER_WINDOW: i64 = 200_000;

/// R131: minimum interval between unsolicited PoolState broadcasts. Bursty
/// member-change events within this window collapse into a single trailing
/// broadcast scheduled by the pool-coalesce timer. New-member visibility
/// degrades by at most `POOL_GOSSIP_MIN_INTERVAL` — well under the
/// existing default `gossip_interval_secs = 600` and acceptable for
/// non-critical state. Receivers always have the option to request
/// state again via the existing `PoolGet` command if they fall behind.
pub(crate) const POOL_GOSSIP_MIN_INTERVAL: std::time::Duration = std::time::Duration::from_secs(15);

/// R134: maximum number of diff broadcasts emitted between full broadcasts.
/// After this many diffs the next gossip is forced full so receivers that
/// missed a diff (network blip, GossipSub propagation gap) recover within
/// bounded time without waiting for the next periodic full broadcast.
pub(crate) const MAX_DIFFS_BEFORE_FULL: u32 = 4;

/// R131: poll cadence for the pool-coalesce timer. Fires regularly so
/// the trailing broadcast after a debounced burst lands within
/// `POOL_GOSSIP_MIN_INTERVAL + POOL_GOSSIP_COALESCE_TICK` of the original
/// event. Cheap when nothing's dirty.
const POOL_GOSSIP_COALESCE_TICK: std::time::Duration = std::time::Duration::from_secs(3);

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
            last_pool_gossip_at: None,
            pool_gossip_dirty: false,
            last_broadcast_state: None,
            diffs_since_full: 0,
        }
    }

    /// Returns true if this member is under the credit-forward rate limits.
    ///
    /// Two independent checks (R138 closes R102 deferral):
    /// 1. **Count cap** — at most `CREDIT_FORWARD_MAX_PER_WINDOW` forwards
    ///    per `CREDIT_FORWARD_WINDOW_SECS`. Defeats the UUID-replay-with-
    ///    fresh-id vector.
    /// 2. **Value cap** — cumulative `amount` summed across forwards in the
    ///    same window cannot exceed `CREDIT_FORWARD_MAX_VALUE_PER_WINDOW`.
    ///    Defense-in-depth above the existing per-tx 100k cap; bounds the
    ///    worst-case TOTAL credit transfer per window.
    ///
    /// The pair is appended atomically only if BOTH checks pass.
    fn check_credit_forward_rate(&mut self, member: &NodeId, amount: i64) -> bool {
        let window = std::time::Duration::from_secs(CREDIT_FORWARD_WINDOW_SECS);
        let now = std::time::Instant::now();
        let entry = self.credit_forward_rl.entry(member.clone()).or_default();
        while entry
            .front()
            .is_some_and(|(t, _)| now.duration_since(*t) > window)
        {
            entry.pop_front();
        }
        if entry.len() >= CREDIT_FORWARD_MAX_PER_WINDOW {
            return false;
        }
        // Value cap: project the window total post-insert and reject if over.
        // saturating_add so a malicious overflow attempt can't bypass via
        // negative-amount wrap (the caller's amount > 0 check at
        // handle_credit_forward already blocks non-positive amounts, but
        // saturate defensively in case the call surface changes).
        let projected: i64 = entry
            .iter()
            .map(|(_, a)| *a)
            .fold(0_i64, i64::saturating_add)
            .saturating_add(amount);
        if projected > CREDIT_FORWARD_MAX_VALUE_PER_WINDOW {
            return false;
        }
        entry.push_back((now, amount));

        // Bound the outer HashMap by sweeping NodeIds whose VecDeque emptied
        // out (member left the pool, restart, etc.). Without this sweep the
        // map accumulates one stale entry per ever-seen sender forever.
        // Cheap guard — only sweeps when the map exceeds a soft cap.
        const CREDIT_FORWARD_RL_SOFT_CAP: usize = 256;
        if self.credit_forward_rl.len() > CREDIT_FORWARD_RL_SOFT_CAP {
            self.credit_forward_rl.retain(|_, deque| {
                !deque.is_empty() && now.duration_since(deque.back().unwrap().0) <= window
            });
        }
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

        // SEC: Do NOT clear TREE_POOL_REMOVAL_REPLAYS on restart. The 5-minute
        // freshness window IS exactly the replay window — a saved PoolRemoval
        // packet replayed within that window after a restart would re-evict the
        // member. Evict only entries older than the freshness window. Entries
        // are stored as the i64 unix timestamp at which the message was processed.
        // Legacy `true` (bool) entries from older builds are evicted unconditionally
        // on first restart — they're stale by definition since their age is unknown.
        let cutoff = chrono::Utc::now().timestamp() - 300;
        let mut to_remove: Vec<String> = Vec::new();
        let _ =
            db.for_each_json::<serde_json::Value, _>(TREE_POOL_REMOVAL_REPLAYS, |subkey, val| {
                let keep = val.as_i64().is_some_and(|ts| ts >= cutoff);
                if !keep {
                    to_remove.push(subkey.to_string());
                }
            });
        for key in to_remove {
            let _ = db.remove(TREE_POOL_REMOVAL_REPLAYS, &key);
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

        // SEC: rehydrate invite codes (owner side). Without this, a pool
        // owner restart between code generation and the joiner sending
        // JoinRequest silently breaks the join flow.
        if let Ok(codes) = db.iter_json::<PoolInviteCode>(TREE_POOL_INVITE_CODES) {
            let now = chrono::Utc::now();
            for code in codes {
                if !code.consumed && code.expires_at > now {
                    self.invite_codes.insert(code.code_hash, code);
                } else {
                    // Best-effort cleanup of expired/consumed codes from disk.
                    let key = hex::encode(code.code_hash);
                    let _ = db.remove(TREE_POOL_INVITE_CODES, &key);
                }
            }
            if !self.invite_codes.is_empty() {
                tracing::info!(
                    count = self.invite_codes.len(),
                    "Restored active pool invite codes"
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

        // R131: pool-state gossip coalescer. Drains the `pool_gossip_dirty`
        // flag whenever the debounce window has elapsed, so a burst of
        // member-change events broadcasts at most once per
        // `POOL_GOSSIP_MIN_INTERVAL` instead of once per event.
        let mut pool_coalesce_interval = tokio::time::interval(POOL_GOSSIP_COALESCE_TICK);
        pool_coalesce_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        pool_coalesce_interval.tick().await; // skip the immediate first tick

        // R107: periodically prune stale TREE_POOL_REMOVAL_REPLAYS entries.
        // The startup-only sweep at `restore_state` was insufficient under
        // sustained pool churn — entries accumulated continuously between
        // restarts (the table is unbounded), letting redb grow without
        // bound. Sweep every 5 minutes (matches the freshness window).
        let mut replay_sweep_interval = tokio::time::interval(std::time::Duration::from_secs(300));
        replay_sweep_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        // First tick fires immediately; skip it since restore_state already swept.
        replay_sweep_interval.tick().await;

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
                    // Periodic full broadcast: always fires (anti-poison +
                    // recovery for late joiners), bypassing the debounce
                    // window. Equivalent to forcing through the rate-limit.
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
                _ = pool_coalesce_interval.tick() => {
                    // R131: drain a debounced broadcast if one is pending
                    // AND enough time has passed since the last broadcast.
                    if self.pool_gossip_dirty
                        && self.last_pool_gossip_at
                            .map(|t| t.elapsed() >= POOL_GOSSIP_MIN_INTERVAL)
                            .unwrap_or(true)
                    {
                        self.gossip_pool_state().await;
                    }
                }
                _ = replay_sweep_interval.tick() => {
                    self.sweep_replay_table();
                }
            }
        }

        Ok(())
    }

    /// Prune `TREE_POOL_REMOVAL_REPLAYS` entries older than the freshness
    /// window. Mirrors the startup logic in `restore_state` — keeps the
    /// table bounded under sustained pool churn between restarts.
    fn sweep_replay_table(&self) {
        let cutoff = chrono::Utc::now().timestamp() - 300;
        let mut to_remove: Vec<String> = Vec::new();
        let _ = self.shared_state.db.for_each_json::<serde_json::Value, _>(
            TREE_POOL_REMOVAL_REPLAYS,
            |subkey, val| {
                let keep = val.as_i64().is_some_and(|ts| ts >= cutoff);
                if !keep {
                    to_remove.push(subkey.to_string());
                }
            },
        );
        if !to_remove.is_empty() {
            let count = to_remove.len();
            for key in to_remove {
                let _ = self.shared_state.db.remove(TREE_POOL_REMOVAL_REPLAYS, &key);
            }
            tracing::debug!(count, "Pruned stale pool replay entries");
        }
    }

    async fn handle_command(&mut self, cmd: PoolCommand) {
        match cmd {
            PoolCommand::CreatePool { name, reply } => {
                let result = self.handle_create_pool(name).await;
                let _ = reply.send(result);
            }
            PoolCommand::CreateInvitation { invitee, reply } => {
                // Direct owner-initiated invite — not bound to an invite code.
                let result = self.handle_create_invitation(invitee, None).await;
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
            PoolCommand::PoolStateDiffGossip { diff } => {
                self.handle_pool_state_diff_gossip(diff).await;
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
                // The owner's own membership is signed with the pool-creation
                // signature, not an acceptance — gossip verifiers skip it
                // (`member.node_id == state.pool_id`), so this is unused.
                invitation_expires_at: now,
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
            generation: 0,
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
        code_hash: Option<[u8; 32]>,
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
        // SEC: code_hash binds this invitation to the specific JoinRequest that triggered it
        // — prevents an attacker who observes a gossiped JoinRequest from issuing their own
        // pool's invitation that the requester would auto-accept.
        let blinded = BlindedPoolInvitation::from_invitation(&invitation, code_hash);
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
            invitation_expires_at: invitation.expires_at,
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
            generation: 0,
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
                format!(
                    "Device {} removed from pool",
                    crate::identity::nickname::short_display_name(
                        &node_id,
                        &self.shared_state.nickname_registry
                    )
                ),
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

        // SEC: reject forwards with stale/future timestamps. Without this, a
        // member can pre-sign a batch of forwards with fresh UUIDs and drip
        // them in months later (UUID dedup only blocks exact-id replays). The
        // signed `forward.timestamp` becomes the freshness anchor; reuses the
        // same window/skew constants as `verify_balance_report` and
        // `handle_inbound_removal` (gotcha #32, #44).
        if let Err(e) = crate::credit::ledger::check_signed_freshness(
            forward.timestamp,
            crate::credit::ledger::CLOCK_SKEW_TOLERANCE_SECS,
            crate::credit::ledger::BALANCE_REPORT_MAX_AGE_SECS,
            "pool_credit_forward",
        ) {
            tracing::warn!(
                error = %e,
                from = %forward.from_node_id,
                id = %forward.id,
                "Rejecting credit forward with stale/future timestamp"
            );
            return;
        }

        // Rate-limit per-member forwards. The UUID is member-generated, so the DB
        // dedup above only blocks exact-UUID replays — a fresh UUID with identical
        // amount/timestamp would bypass it. Rate-limiting bounds the exploitability.
        // R138: also enforces a cumulative-value cap per window
        // (CREDIT_FORWARD_MAX_VALUE_PER_WINDOW = 200k credits) on top of the
        // count cap (60 forwards/window). Either limit rejects.
        if !self.check_credit_forward_rate(&forward.from_node_id, forward.amount) {
            tracing::warn!(
                from = %forward.from_node_id,
                amount = forward.amount,
                window_secs = CREDIT_FORWARD_WINDOW_SECS,
                max_count = CREDIT_FORWARD_MAX_PER_WINDOW,
                max_value = CREDIT_FORWARD_MAX_VALUE_PER_WINDOW,
                "Credit forward rate limit exceeded (count or cumulative value) — dropping"
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
        if let Err(e) = crate::credit::ledger::apply_credit_direct_noted(
            &self.shared_state.credits.credit_balance,
            &self.shared_state.db,
            forward.amount,
            crate::credit::ledger::CreditDelta::Earning,
            "pool_forward_earning",
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
        // AND the invitation is bound to the same code_hash we redeemed. Without
        // this code_hash binding, a network-adjacent attacker who observes a
        // gossiped JoinRequest could issue an invitation under a pool they
        // control and the auto-accept window would route the requester there.
        if let Some((stored_hash, created_at)) = self.auto_accept_code_hash {
            if created_at.elapsed().as_secs() >= AUTO_ACCEPT_TIMEOUT_SECS {
                // Expired — clear the stale auto-accept intent
                self.auto_accept_code_hash = None;
                tracing::info!(
                    "Auto-accept expired (>5min) — invitation stored but not auto-accepted"
                );
                return;
            }
            match blinded.code_hash {
                Some(hash) if hash == stored_hash => {
                    self.auto_accept_code_hash = None;
                    tracing::info!(
                        invitation_id = %invitation.id,
                        pool_id = %invitation.pool_id,
                        "Auto-accepting blinded invitation (from invite code join)"
                    );
                    if let Err(e) = self.handle_accept_invitation(invitation).await {
                        tracing::warn!(error = %e, "Auto-accept failed");
                    }
                }
                Some(_) => {
                    tracing::warn!(
                        invitation_id = %invitation.id,
                        pool_id = %invitation.pool_id,
                        "Refusing to auto-accept: invitation code_hash does not match the one we redeemed (possible hijack attempt)"
                    );
                }
                None => {
                    tracing::debug!(
                        invitation_id = %invitation.id,
                        pool_id = %invitation.pool_id,
                        "Stored invitation but not auto-accepting: invitation has no code_hash binding"
                    );
                }
            }
        }
    }

    async fn handle_inbound_acceptance(&mut self, acceptance: PoolAcceptance) {
        // If we're the pool owner, add the member
        let my_id = self.shared_state.identity.node_id();
        if acceptance.pool_id != *my_id {
            return; // Not our pool
        }

        // Look the invitation up FIRST. Its `expires_at` is an input to the
        // signature check below (R147), so we need our own copy of the
        // invitation before we can verify anything — deliberately ours, not
        // the acceptance's, so the attacker doesn't choose the value the
        // signature is verified against.
        let (expected_invitee, invitation_expires_at) = match self
            .pending_invitations
            .get(&acceptance.invitation_id)
        {
            Some(inv) => (inv.invitee_node_id.clone(), inv.expires_at),
            None => {
                tracing::warn!(invitation_id = %acceptance.invitation_id, invitee = %acceptance.invitee_node_id, "Acceptance for unknown invitation");
                return;
            }
        };

        // SEC: Verify the acceptance comes from the invitee we actually invited.
        // Without this, anyone who learns the invitation_id (e.g. via leaked
        // PoolMembership broadcast or log scrape) could craft an acceptance with
        // their own NodeId + own valid signature, consume the slot, and lock out
        // the real invitee.
        if expected_invitee != acceptance.invitee_node_id {
            tracing::warn!(
                invitation_id = %acceptance.invitation_id,
                expected = %expected_invitee,
                got = %acceptance.invitee_node_id,
                "Acceptance invitee does not match original invitation — rejecting"
            );
            return;
        }

        // SEC: an expired invitation cannot be accepted. The signature binds
        // `expires_at`, so this is the enforcement half of that binding —
        // together they mean a captured acceptance stops being useful once the
        // invitation it names has lapsed, whether or not the owner still holds
        // it in `pending_invitations`.
        if chrono::Utc::now() > invitation_expires_at {
            tracing::warn!(
                invitation_id = %acceptance.invitation_id,
                invitee = %acceptance.invitee_node_id,
                expired_at = %invitation_expires_at,
                "Acceptance for an expired invitation — rejecting"
            );
            return;
        }

        // Verify the acceptance signature, bound to our invitation's expiry.
        let invitee_key =
            match ed25519_dalek::VerifyingKey::from_bytes(&acceptance.invitee_node_id.0) {
                Ok(k) => k,
                Err(_) => return,
            };
        if crypto::verify_acceptance(&acceptance, &invitee_key, &invitation_expires_at).is_err() {
            tracing::warn!(invitee = %acceptance.invitee_node_id, "Invalid acceptance signature");
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
                    // Our stored invitation's expiry — the same value the
                    // signature was verified against above, never a value
                    // taken from the acceptance.
                    invitation_expires_at,
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
            // Trigger updated pool state broadcast — debounced so a burst
            // of acceptances (or a periodic re-broadcast that lands in the
            // same window) doesn't flood the network with N copies of an
            // N-member roster.
            self.maybe_gossip_pool_state().await;
        }
    }

    async fn handle_inbound_removal(&mut self, removal: PoolRemoval) {
        let my_id = self.shared_state.identity.node_id();

        // If we're the one being removed
        if removal.removed_node_id == *my_id {
            // SEC: Freshness check — reject removals older than 5 minutes,
            // routed through the centralised one-sided helper so we can't
            // accidentally re-introduce the .abs() replay-window bug
            // (gotcha #32 / #44).
            if let Err(e) = crate::credit::ledger::check_signed_freshness(
                removal.removed_at,
                crate::credit::ledger::CLOCK_SKEW_TOLERANCE_SECS,
                crate::credit::ledger::BALANCE_REPORT_MAX_AGE_SECS,
                "pool_removal",
            ) {
                tracing::warn!(error = %e, "Pool removal rejected: stale or future-dated");
                return;
            }

            // SEC: reject nil removal_id explicitly. The wire field defaults
            // to `Uuid::nil()` for backwards-compat with pre-id messages,
            // but the dedup table is keyed on the UUID — a nil-keyed entry
            // would block ALL future legacy nil-UUID removals from being
            // processed (one-shot suppression DoS). Real pool removals
            // emitted by current builds always carry a fresh UUID.
            if removal.removal_id.is_nil() {
                tracing::warn!(
                    pool = %removal.pool_id,
                    removed = %removal.removed_node_id,
                    "Pool removal rejected: missing removal_id"
                );
                return;
            }

            // SEC: Replay protection — check if we've already processed this removal_id.
            // Type-agnostic check (bool from old builds OR i64 timestamp going forward)
            // so legacy entries still block replays during the upgrade.
            let removal_key = removal.removal_id.to_string();
            if self
                .shared_state
                .db
                .get_json::<serde_json::Value>(TREE_POOL_REMOVAL_REPLAYS, &removal_key)
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

            // Record the removal_id (with timestamp) to prevent replay; the
            // timestamp lets startup sweep keep only entries within the 5-min
            // freshness window so a planned restart can't re-open the replay door.
            // SEC (R105): order matters. Previously the replay-key write
            // happened BEFORE the pool_state delete; a crash between them
            // permanently blocked the same removal from re-processing
            // (replay key persisted) while pool_state still claimed
            // membership — node stuck in a pool the owner had ejected
            // them from, with no way to recover except manual DB edit.
            //
            // Inverted order: remove pool_state first, then write the
            // replay key. Crash between is now benign in both directions:
            //   - crash after pool_state delete, before replay write:
            //     same removal re-arrives → re-removes already-removed
            //     pool_state (idempotent), then writes the replay key.
            //   - crash before pool_state delete: the dedup key wasn't
            //     written, so the next delivery processes from scratch.
            *self.shared_state.credits.pool_state.write().await = None;
            let _ = self.shared_state.db.remove(TREE_POOL_STATE, KEY_MY_POOL);
            self.shared_state
                .credits
                .pool_registry
                .remove(&removal.pool_id);

            let _ = self.shared_state.db.put_json(
                TREE_POOL_REMOVAL_REPLAYS,
                &removal_key,
                &chrono::Utc::now().timestamp(),
            );
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

        // Freshness check: reject notices that are too old or pre-signed in the
        // future. Use a one-sided staleness bound (NOT .abs()) so an attacker
        // can't pre-sign a notice timestamped 5 minutes in the future and replay
        // it for a full 10-minute window. Mirrors `verify_balance_report` in
        // `credit/ledger.rs`. 30s future tolerance for honest cross-node clock
        // skew, 300s past tolerance for staleness.
        let now = chrono::Utc::now().timestamp();
        if left_at > now + 30 {
            tracing::warn!(node = %node_id, left_at, now, "Future-dated member-left notice — rejecting");
            return;
        }
        if left_at < now - 300 {
            tracing::warn!(node = %node_id, left_at, now, "Stale member-left notice — rejecting");
            return;
        }

        // Replay protection: same tree as pool removals, keyed by nonce.
        // Type-agnostic check (bool from old builds OR i64 timestamp).
        let replay_key = format!("ml-{}", nonce);
        if self
            .shared_state
            .db
            .get_json::<serde_json::Value>(TREE_POOL_REMOVAL_REPLAYS, &replay_key)
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

        // R107: Order matters — mirror the R105 fix in
        // `handle_inbound_removal`. Previously the replay-key write
        // happened BEFORE the in-memory `members.retain` and the
        // `persist_pool_state` write; a crash in that window left a
        // permanent replay key blocking redelivery while the departed
        // member still appeared in pool state on disk — the member was
        // stuck-in-pool with no way to recover. New order: mutate
        // pool_state and persist first, then write the replay key.
        // Crash between is now benign in both directions:
        //   - crash after persist, before replay write: same notice
        //     re-arrives → idempotent retain (already removed), then
        //     writes the replay key.
        //   - crash before persist: replay key not written, so the
        //     next delivery processes from scratch.
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

        // Record nonce (with timestamp) to prevent replay — written AFTER
        // pool_state mutation per the ordering invariant above.
        if let Err(e) = self.shared_state.db.put_json(
            TREE_POOL_REMOVAL_REPLAYS,
            &replay_key,
            &chrono::Utc::now().timestamp(),
        ) {
            tracing::warn!(error = %e, "Failed to persist member-left replay key");
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
        if name.chars().count() > 32 || name.len() > 64 {
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
        // Snapshot the values we need from pool_state and drop the read
        // lock before the rate-limit / code-generation work below. Holding
        // the lock across rate_limiter.check_and_record() and key generation
        // would block a concurrent pool_state writer (handle_remove_member,
        // handle_accept_invitation) for no reason.
        let max_size = self.shared_state.config.pool.max_pool_size;
        let (is_owner, member_count, pool_name) = {
            let pool_state = self.shared_state.credits.pool_state.read().await;
            let ps = pool_state
                .as_ref()
                .ok_or_else(|| SwarmError::Validation("Not in a pool".into()))?;
            (
                ps.pool_id == *self.shared_state.identity.node_id(),
                ps.members.len() as u32,
                ps.name.clone(),
            )
        };

        if !is_owner {
            return Err(SwarmError::Validation(
                "Only the pool owner can generate invite codes".into(),
            ));
        }
        if member_count >= max_size {
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

        // Snapshot the swarm's current reachable listen addresses BEFORE
        // committing the code, so we can return a clear error if there are
        // none. An invite code without any addresses would technically work
        // for any joiner already on the swarm, but the v2 promise is
        // "bootstrap-before-decentralization": if we can't deliver that, fail
        // loudly rather than silently regress to legacy behavior.
        let multiaddrs: Vec<String> = self.shared_state.listen_multiaddrs.load().as_ref().clone();
        if multiaddrs.is_empty() {
            return Err(SwarmError::ServiceUnavailable(
                "No reachable network addresses yet — wait a few seconds after startup and try again".into(),
            ));
        }

        let ttl = self.shared_state.config.pool.invitation_ttl_hours;
        let invite = PoolInviteCode::generate(self.shared_state.identity.node_id(), ttl);
        let short_code = invite.code.clone();
        let code_hash_hex = hex::encode(invite.code_hash);
        // SEC: persist alongside the in-memory map so a restart between
        // generate-code and the joiner sending JoinRequest doesn't lose the
        // code (without this, every owner crash silently breaks join flows
        // that were in flight). Keyed on hex(code_hash) — DB keys are
        // strings.
        if let Err(e) =
            self.shared_state
                .db
                .put_json(TREE_POOL_INVITE_CODES, &code_hash_hex, &invite)
        {
            tracing::warn!(error = %e, "Failed to persist invite code");
        }
        self.invite_codes.insert(invite.code_hash, invite);

        // Build the v2 wire payload.
        let payload = crate::pool::invite::InviteCodePayload {
            version: crate::pool::invite::INVITE_VERSION,
            pool_id: self.shared_state.identity.node_id().clone(),
            pool_name,
            multiaddrs,
            code: short_code.clone(),
            expires_at_unix: chrono::Utc::now().timestamp() + (ttl as i64) * 3600,
        };
        let encoded = crate::pool::invite::encode_invite_code(&payload)?;

        // Kill the silent LAN-only failure: the code is valid, but if none of
        // our addresses are reachable from the open internet it will only work
        // for a joiner on the same LAN/overlay. Warn the user loudly rather
        // than let them share a code that silently dies over the internet.
        let internet_reachable = crate::pool::invite::any_internet_reachable(&payload.multiaddrs);
        if !internet_reachable {
            tracing::warn!(
                multiaddr_count = payload.multiaddrs.len(),
                "Generated invite code has only local-network addresses — it will not work over the internet"
            );
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "pool",
                    "invite_lan_only",
                    "This invite code only works on your local network. To invite someone over the internet, enable UPnP on your router, forward the P2P port, or set up a relay/anchor node (see the networking docs)."
                        .to_string(),
                )
                .with_toast("warning", 9000),
            );
        }

        tracing::info!(
            code_preview = &short_code[..4],
            multiaddr_count = payload.multiaddrs.len(),
            internet_reachable,
            "Generated v2 pool invite code"
        );

        Ok(encoded)
    }

    /// Join a pool using an invite code (from the joining device).
    /// Broadcasts a JoinRequest over gossip so the owner can auto-invite.
    async fn handle_join_with_code(&mut self, code: String) -> Result<(), SwarmError> {
        // Must not already be in a pool
        if self.shared_state.credits.pool_state.read().await.is_some() {
            return Err(SwarmError::Validation("Already in a pool".into()));
        }

        // Two acceptance paths:
        //  - v2 `swarmpool://...` blob with multiaddrs. We dial first so the
        //    joiner doesn't depend on already being on the same swarm. This is
        //    the "bootstrap-before-decentralization" mode.
        //  - Legacy 8-char `A3F7K2M9` code. Kept for shared-swarm cases (LAN
        //    via mDNS, joiner already bootstrapped via DHT). Surfaces a
        //    clearer error if the joiner ISN'T already on the swarm — they'll
        //    see "join request broadcast" with no follow-up.
        let trimmed = code.trim();
        let short_code = if crate::pool::invite::looks_like_v2(trimmed) {
            let payload = crate::pool::invite::decode_invite_code(trimmed)?;
            self.dial_invite_multiaddrs(&payload.multiaddrs).await;
            payload.code.to_uppercase()
        } else {
            let upper = trimmed.to_uppercase();
            if upper.len() != 8 || !upper.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Err(SwarmError::Validation(
                    "Invite code must be a swarmpool:// link or 8 letters/digits".into(),
                ));
            }
            upper
        };

        // Compute code hash and broadcast join request
        let code_hash = *blake3::hash(short_code.as_bytes()).as_bytes();
        let my_id = self.shared_state.identity.node_id().clone();

        // Sign the join request
        let mut payload_hasher = blake3::Hasher::new();
        payload_hasher.update(b"pool_join_request_v1");
        payload_hasher.update(&code_hash);
        payload_hasher.update(&my_id.0);
        let signed_payload = payload_hasher.finalize();
        let signature = self
            .shared_state
            .identity
            .sign(signed_payload.as_bytes())
            .to_vec();

        // Broadcast join request over gossip — owner picks it up either via
        // the direct dial we just made (v2 path) or via the existing swarm
        // (legacy path / mature DHT).
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

        tracing::info!(
            code_hash = %hex::encode(code_hash),
            "Broadcast pool join request with invite code (auto-accept enabled)"
        );
        Ok(())
    }

    /// Dial each multiaddr from a v2 invite code. Fire-and-forget — the
    /// subsequent `JoinRequest` broadcast will land via whichever dial
    /// succeeds. We don't `await` connection establishment here because (a)
    /// libp2p's `dial` returns immediately and the connection lands later via
    /// `ConnectionEstablished`, and (b) we don't want to block the pool
    /// manager event loop on network round trips. The owner's GossipSub
    /// subscription will pick up the JoinRequest as soon as any one peering
    /// completes — usually within a few hundred ms of this call.
    async fn dial_invite_multiaddrs(&self, multiaddrs: &[String]) {
        for addr in multiaddrs {
            if let Err(e) = self
                .network_tx
                .send(crate::types::NetworkCommand::DialAddress(addr.clone()))
                .await
            {
                tracing::warn!(error = %e, addr = %addr, "Failed to send DialAddress to network manager");
            }
        }
        tracing::info!(count = multiaddrs.len(), "Dialing invite code multiaddrs");
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

        // Mark code as consumed (one-time use). Also drop the persisted
        // copy so a future restart doesn't see a "fresh" non-expired code.
        code_entry.consumed = true;
        let key = hex::encode(code_hash);
        if let Err(e) = self.shared_state.db.remove(TREE_POOL_INVITE_CODES, &key) {
            tracing::debug!(error = %e, "Failed to remove consumed invite code from db");
        }

        tracing::info!(
            requester = %requester,
            "Invite code claimed — auto-creating invitation"
        );

        // Auto-create invitation for the requester. Bind the resulting blinded
        // invitation to this code_hash so the requester's auto-accept gate can
        // verify the invitation came from the pool whose code they redeemed.
        match self
            .handle_create_invitation(requester.clone(), Some(code_hash))
            .await
        {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::identity::Identity;
    use crate::storage::db::Database;
    use tokio::sync::Mutex;

    /// Build a fully-wired PoolManager backed by a temp database. Returns
    /// (manager, shared_state, identity) so the caller can introspect /
    /// mutate pool state directly to set up scenarios.
    async fn build_test_pool_manager() -> (PoolManager, Arc<SharedState>, Identity) {
        let (pm, state, id, _rx) = build_test_pool_manager_with_rx().await;
        (pm, state, id)
    }

    /// As above but also returns the network receiver so callers can
    /// assert on outbound broadcasts.
    async fn build_test_pool_manager_with_rx() -> (
        PoolManager,
        Arc<SharedState>,
        Identity,
        mpsc::Receiver<NetworkCommand>,
    ) {
        let config = Config::default();
        let identity = Identity::generate();
        let db = Database::open_temp().expect("temp db");
        let executor = Arc::new(Mutex::new(crate::inference::executor::ModelExecutor::new()));

        let (shared_state, _shutdown_rx, _dht_rx) =
            SharedState::new(config, identity.clone(), db, executor, None);

        let (_cmd_tx, cmd_rx) = mpsc::channel(16);
        let (network_tx, network_rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let pm = PoolManager::new(shared_state.clone(), cmd_rx, network_tx, shutdown_rx);
        (pm, shared_state, identity, network_rx)
    }

    /// Drain the network channel and count the number of pending
    /// `PoolMessage::StateGossip` broadcasts. Used to assert the
    /// debounce behaviour without depending on `Instant` arithmetic on
    /// the spawned task side.
    fn count_state_gossips(rx: &mut mpsc::Receiver<NetworkCommand>) -> usize {
        let mut count = 0;
        while let Ok(cmd) = rx.try_recv() {
            if let NetworkCommand::Broadcast(SwarmMessage::PoolMessage(
                crate::types::PoolMessage::StateGossip(_),
            )) = cmd
            {
                count += 1;
            }
        }
        count
    }

    /// R134: synthesize a membership with a real `acceptance_signature`
    /// signed by `identity` for `pool_id` + `invitation_id`. Used by the
    /// diff-gossip tests so the receiver's per-member signature check
    /// passes (the legacy `membership_for` helper leaves the field empty).
    fn signed_membership_for(
        identity: &Identity,
        pool_id: &PoolId,
        invitation_id: uuid::Uuid,
    ) -> PoolMembership {
        let expires_at = chrono::Utc::now() + chrono::Duration::hours(24);
        let payload = crate::pool::crypto::acceptance_payload(
            &invitation_id,
            pool_id,
            identity.node_id(),
            &expires_at,
        );
        let sig = identity.sign(&payload);
        let now = chrono::Utc::now();
        PoolMembership {
            node_id: identity.node_id().clone(),
            credits_contributed: 0,
            joined_at: now,
            acceptance_signature: sig,
            invitation_id,
            invitation_expires_at: expires_at,
            device_name: None,
            last_seen: Some(now),
            online: true,
            device_stats: None,
            contribution_level: 100,
        }
    }

    /// Synthesize a pool membership record for the given node — only
    /// `node_id` matters for the handler under test.
    fn membership_for(node_id: NodeId) -> PoolMembership {
        let now = chrono::Utc::now();
        PoolMembership {
            node_id,
            credits_contributed: 0,
            joined_at: now,
            acceptance_signature: Vec::new(),
            invitation_id: uuid::Uuid::nil(),
            invitation_expires_at: now,
            device_name: None,
            last_seen: Some(now),
            online: true,
            device_stats: None,
            contribution_level: 100,
        }
    }

    /// A pool gossip containing one member whose acceptance signature cannot be
    /// verified (e.g. a stale pre-R147 record) must NOT be rejected wholesale:
    /// the unverifiable member is dropped, the owner + verified members are kept,
    /// and the pool still lands in the registry.
    #[tokio::test]
    async fn pool_gossip_drops_unverifiable_member_keeps_valid_ones() {
        // Receiver C — neither the owner nor a member of this pool.
        let (mut recv_pm, recv_state, _c) = build_test_pool_manager().await;

        let owner = Identity::generate();
        let pool_id = owner.node_id().clone();
        let name = "cross-version-pool".to_string();
        let now = chrono::Utc::now();
        let owner_sig = owner.sign(&crate::pool::crypto::pool_create_payload(
            &pool_id, &name, &now,
        ));

        // Verified member M1 (real acceptance signature).
        let m1 = Identity::generate();
        let m1_membership = signed_membership_for(&m1, &pool_id, uuid::Uuid::new_v4());
        // Unverifiable member M2 — empty acceptance signature (stale/legacy record).
        let m2 = Identity::generate();
        let m2_membership = membership_for(m2.node_id().clone());

        let owner_membership = PoolMembership {
            node_id: pool_id.clone(),
            credits_contributed: 0,
            joined_at: now,
            acceptance_signature: owner_sig.clone(),
            invitation_id: uuid::Uuid::nil(),
            invitation_expires_at: now,
            device_name: None,
            last_seen: Some(now),
            online: true,
            device_stats: None,
            contribution_level: 100,
        };

        let state = PoolState {
            pool_id: pool_id.clone(),
            name,
            members: vec![owner_membership, m1_membership, m2_membership],
            created_at: now,
            owner_signature: owner_sig,
            total_lifetime_credits: 0,
            member_credit_split_pct: 0,
            shard_pins: Vec::new(),
            generation: 0,
        };

        recv_pm.handle_pool_state_gossip(state).await;

        // Not rejected wholesale — the pool is stored...
        let stored = recv_state
            .credits
            .pool_registry
            .get(&pool_id)
            .expect("pool should be stored, not dropped on one unverifiable member");
        // ...with the unverifiable member removed and the rest kept.
        let ids: Vec<_> = stored.members.iter().map(|m| m.node_id.clone()).collect();
        assert!(ids.contains(&pool_id), "owner kept");
        assert!(ids.contains(m1.node_id()), "verified member kept");
        assert!(!ids.contains(m2.node_id()), "unverifiable member dropped");
        assert_eq!(stored.members.len(), 2);
    }

    #[tokio::test]
    async fn member_left_replay_protection_blocks_second_remove() {
        let (mut pm, state, owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        // Add a member and produce a signed leave notice.
        let member = Identity::generate();
        {
            let mut ps = state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(member.node_id().clone()));
            }
        }

        let pool_id = owner.node_id().clone();
        let left_at = chrono::Utc::now().timestamp();
        let nonce = uuid::Uuid::new_v4();
        let payload =
            crate::pool::crypto::member_left_payload(&pool_id, member.node_id(), left_at, &nonce);
        let signature = member.sign(&payload);

        // First call: removes the member.
        pm.handle_inbound_member_left(
            pool_id.clone(),
            member.node_id().clone(),
            left_at,
            nonce,
            signature.clone(),
        )
        .await;
        assert_eq!(
            state
                .credits
                .pool_state
                .read()
                .await
                .as_ref()
                .unwrap()
                .members
                .len(),
            1,
            "first leave notice must remove the member (only owner remains)"
        );

        // Re-add the member to set up the replay test.
        {
            let mut ps = state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(member.node_id().clone()));
            }
        }

        // Replay: same nonce, same signature. Must NOT remove again — the
        // replay key is already persisted from the first call.
        pm.handle_inbound_member_left(pool_id, member.node_id().clone(), left_at, nonce, signature)
            .await;
        assert_eq!(
            state
                .credits
                .pool_state
                .read()
                .await
                .as_ref()
                .unwrap()
                .members
                .len(),
            2,
            "replay (same nonce) must be rejected — member stays in the pool"
        );
    }

    #[tokio::test]
    async fn member_left_with_invalid_signature_rejected() {
        let (mut pm, state, owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        let member = Identity::generate();
        let imposter = Identity::generate();
        {
            let mut ps = state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(member.node_id().clone()));
            }
        }

        let left_at = chrono::Utc::now().timestamp();
        let nonce = uuid::Uuid::new_v4();
        // Imposter signs the payload with their own key but claims to be `member`.
        let payload = crate::pool::crypto::member_left_payload(
            owner.node_id(),
            member.node_id(),
            left_at,
            &nonce,
        );
        let bad_signature = imposter.sign(&payload);

        pm.handle_inbound_member_left(
            owner.node_id().clone(),
            member.node_id().clone(),
            left_at,
            nonce,
            bad_signature,
        )
        .await;
        assert_eq!(
            state
                .credits
                .pool_state
                .read()
                .await
                .as_ref()
                .unwrap()
                .members
                .len(),
            2,
            "invalid signature must NOT remove the member"
        );
    }

    #[tokio::test]
    async fn member_left_with_stale_timestamp_rejected() {
        let (mut pm, state, owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        let member = Identity::generate();
        {
            let mut ps = state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(member.node_id().clone()));
            }
        }

        // Timestamp 10 minutes ago — beyond the ±5min freshness window.
        let stale = chrono::Utc::now().timestamp() - 600;
        let nonce = uuid::Uuid::new_v4();
        let payload = crate::pool::crypto::member_left_payload(
            owner.node_id(),
            member.node_id(),
            stale,
            &nonce,
        );
        let signature = member.sign(&payload);

        pm.handle_inbound_member_left(
            owner.node_id().clone(),
            member.node_id().clone(),
            stale,
            nonce,
            signature,
        )
        .await;
        assert_eq!(
            state
                .credits
                .pool_state
                .read()
                .await
                .as_ref()
                .unwrap()
                .members
                .len(),
            2,
            "stale timestamp must NOT remove the member"
        );
    }

    #[tokio::test]
    async fn member_left_with_future_timestamp_rejected() {
        // Pre-signed future-dated notices must be rejected even within the .abs()
        // window that previously doubled the effective replay window. This is the
        // companion test to verify_balance_report's one-sided staleness check.
        let (mut pm, state, owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        let member = Identity::generate();
        {
            let mut ps = state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(member.node_id().clone()));
            }
        }

        // 250s in the future — within the (now ±5min .abs()) window but well
        // beyond the 30s honest-skew tolerance. Must be rejected.
        let future = chrono::Utc::now().timestamp() + 250;
        let nonce = uuid::Uuid::new_v4();
        let payload = crate::pool::crypto::member_left_payload(
            owner.node_id(),
            member.node_id(),
            future,
            &nonce,
        );
        let signature = member.sign(&payload);

        pm.handle_inbound_member_left(
            owner.node_id().clone(),
            member.node_id().clone(),
            future,
            nonce,
            signature,
        )
        .await;
        assert_eq!(
            state
                .credits
                .pool_state
                .read()
                .await
                .as_ref()
                .unwrap()
                .members
                .len(),
            2,
            "future-dated timestamp must NOT remove the member"
        );
    }

    #[tokio::test]
    async fn member_left_for_other_pool_id_ignored() {
        // Notices for a different pool_id (not us) must short-circuit.
        let (mut pm, state, _owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        let other_owner = Identity::generate();
        let member = Identity::generate();
        {
            let mut ps = state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(member.node_id().clone()));
            }
        }

        let left_at = chrono::Utc::now().timestamp();
        let nonce = uuid::Uuid::new_v4();
        let payload = crate::pool::crypto::member_left_payload(
            other_owner.node_id(),
            member.node_id(),
            left_at,
            &nonce,
        );
        let signature = member.sign(&payload);

        pm.handle_inbound_member_left(
            other_owner.node_id().clone(),
            member.node_id().clone(),
            left_at,
            nonce,
            signature,
        )
        .await;
        assert_eq!(
            state
                .credits
                .pool_state
                .read()
                .await
                .as_ref()
                .unwrap()
                .members
                .len(),
            2,
            "leave notice for someone else's pool must NOT touch our state"
        );
    }

    #[tokio::test]
    async fn join_request_ignored_when_not_pool_owner() {
        // No pool created — this node is not an owner. Inbound join
        // requests must short-circuit without panicking.
        let (mut pm, _state, _owner) = build_test_pool_manager().await;
        let requester = Identity::generate();
        let code_hash = [0u8; 32];

        pm.handle_inbound_join_request(code_hash, requester.node_id().clone())
            .await;
        // No panic, no auto-invitation created.
        assert_eq!(pm.pending_invitations.len(), 0);
    }

    #[tokio::test]
    async fn join_request_with_unknown_code_ignored() {
        let (mut pm, _state, _owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        let requester = Identity::generate();
        // Code hash that's not in our invite_codes map.
        let unknown_hash = [0u8; 32];

        pm.handle_inbound_join_request(unknown_hash, requester.node_id().clone())
            .await;
        assert_eq!(
            pm.pending_invitations.len(),
            0,
            "unknown code hash must not auto-invite"
        );
    }

    #[tokio::test]
    async fn join_request_with_consumed_code_ignored() {
        let (mut pm, _state, _owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        // Insert a code marked as already consumed.
        let pool_id = pm.shared_state.identity.node_id().clone();
        let mut code = PoolInviteCode::generate(&pool_id, 24);
        code.consumed = true;
        let hash = code.code_hash;
        pm.invite_codes.insert(hash, code);

        let requester = Identity::generate();
        pm.handle_inbound_join_request(hash, requester.node_id().clone())
            .await;
        assert_eq!(
            pm.pending_invitations.len(),
            0,
            "consumed code must not auto-invite"
        );
    }

    #[tokio::test]
    async fn join_request_with_valid_code_consumes_and_invites() {
        let (mut pm, _state, _owner) = build_test_pool_manager().await;
        pm.handle_create_pool("test-pool".into()).await.unwrap();

        let pool_id = pm.shared_state.identity.node_id().clone();
        let code = PoolInviteCode::generate(&pool_id, 24);
        let hash = code.code_hash;
        pm.invite_codes.insert(hash, code);

        let requester = Identity::generate();
        pm.handle_inbound_join_request(hash, requester.node_id().clone())
            .await;

        // Code was consumed (one-time use).
        assert!(
            pm.invite_codes.get(&hash).unwrap().consumed,
            "valid join request must consume the code"
        );
        // Auto-invitation was created.
        assert_eq!(
            pm.pending_invitations.len(),
            1,
            "valid join request must create an auto-invitation"
        );
    }

    #[tokio::test]
    async fn auto_accept_rejects_invitation_with_mismatched_code_hash() {
        // SEC: A network-adjacent attacker who observes a JoinRequest could
        // issue an invitation under a pool they control. Without the code_hash
        // binding, the requester's auto-accept window would silently route them
        // to the attacker's pool. Verify the gate refuses to auto-accept when
        // blinded.code_hash doesn't match the requested code's hash.
        let (mut pm, state, _owner) = build_test_pool_manager().await;

        // Pretend we just sent a JoinWithCode that hashes to `our_hash`.
        let our_hash = *blake3::hash(b"OUR-CODE").as_bytes();
        pm.auto_accept_code_hash = Some((our_hash, std::time::Instant::now()));

        // Attacker's pool (different identity). PoolId is a type alias for NodeId.
        let attacker = Identity::generate();
        let me = state.identity.node_id().clone();

        // Build a valid PoolInvitation signed by the attacker (we delegate to
        // the production signing helper so the verify path inside the manager
        // accepts it).
        let attacker_inv = crate::pool::crypto::create_invitation(
            &attacker,
            attacker.node_id(),
            &me,
            1, // 1-hour TTL
        );

        // Attacker's invitation has no code_hash → auto-accept must refuse.
        let blinded_no_hash = BlindedPoolInvitation::from_invitation(&attacker_inv, None);
        pm.handle_inbound_blinded_invitation(blinded_no_hash).await;

        assert!(
            pm.auto_accept_code_hash.is_some(),
            "auto_accept_code_hash must NOT be cleared by an unbound invitation"
        );
        assert_eq!(
            pm.pending_invitations.len(),
            1,
            "invitation should still be stored as a pending invitation"
        );
        assert!(
            state.credits.pool_state.read().await.is_none(),
            "must not have joined attacker's pool"
        );

        // Now try with a wrong code_hash bound — still refuses.
        let wrong_hash = *blake3::hash(b"WRONG-CODE").as_bytes();
        let attacker_inv2 =
            crate::pool::crypto::create_invitation(&attacker, attacker.node_id(), &me, 1);
        let blinded_wrong =
            BlindedPoolInvitation::from_invitation(&attacker_inv2, Some(wrong_hash));
        pm.handle_inbound_blinded_invitation(blinded_wrong).await;
        assert!(
            pm.auto_accept_code_hash.is_some(),
            "auto_accept_code_hash must NOT be cleared by a wrong-hash invitation"
        );
        assert!(
            state.credits.pool_state.read().await.is_none(),
            "must not have joined attacker's pool with wrong code_hash"
        );
    }

    /// R131: first call to `maybe_gossip_pool_state` fires immediately —
    /// `last_pool_gossip_at` is `None`, so the cooldown isn't engaged.
    /// Multi-thread runtime needed because `collect_device_stats` uses
    /// `tokio::task::block_in_place` (sysinfo memory probe).
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r131_first_gossip_fires_immediately() {
        let (mut pm, _state, _id, mut rx) = build_test_pool_manager_with_rx().await;
        pm.handle_create_pool("test".into()).await.unwrap();
        let _initial = count_state_gossips(&mut rx); // drain the create-pool gossip

        pm.maybe_gossip_pool_state().await;
        assert_eq!(count_state_gossips(&mut rx), 1, "first call must broadcast");
        assert!(pm.last_pool_gossip_at.is_some());
        assert!(!pm.pool_gossip_dirty);
    }

    /// R131: second call within `POOL_GOSSIP_MIN_INTERVAL` is suppressed
    /// but sets `pool_gossip_dirty` so the coalesce timer catches up.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r131_second_gossip_within_window_debounces() {
        let (mut pm, _state, _id, mut rx) = build_test_pool_manager_with_rx().await;
        pm.handle_create_pool("test".into()).await.unwrap();
        let _drain = count_state_gossips(&mut rx);

        pm.maybe_gossip_pool_state().await;
        let after_first = pm.last_pool_gossip_at;
        assert_eq!(count_state_gossips(&mut rx), 1);

        // Second call within the window: no broadcast, dirty flag set,
        // last_pool_gossip_at unchanged.
        pm.maybe_gossip_pool_state().await;
        assert_eq!(
            count_state_gossips(&mut rx),
            0,
            "second call within window must not broadcast"
        );
        assert!(pm.pool_gossip_dirty);
        assert_eq!(pm.last_pool_gossip_at, after_first);
    }

    /// R131: after the cooldown elapses, a third call fires the broadcast
    /// AND clears the dirty flag. Simulates the coalesce timer's drain.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r131_gossip_after_cooldown_fires_and_clears_dirty() {
        let (mut pm, _state, _id, mut rx) = build_test_pool_manager_with_rx().await;
        pm.handle_create_pool("test".into()).await.unwrap();
        let _drain = count_state_gossips(&mut rx);

        pm.maybe_gossip_pool_state().await;
        pm.maybe_gossip_pool_state().await; // suppressed, sets dirty
        let _drain2 = count_state_gossips(&mut rx);
        assert!(pm.pool_gossip_dirty);

        // Simulate cooldown expiry by backdating the timestamp. Avoids a
        // multi-second sleep in the test.
        pm.last_pool_gossip_at = Some(
            std::time::Instant::now()
                - POOL_GOSSIP_MIN_INTERVAL
                - std::time::Duration::from_secs(1),
        );

        pm.maybe_gossip_pool_state().await;
        assert_eq!(
            count_state_gossips(&mut rx),
            1,
            "broadcast after cooldown must fire"
        );
        assert!(!pm.pool_gossip_dirty);
    }

    // ----------------- R134 diff-gossip tests -----------------

    fn count_state_diffs(rx: &mut mpsc::Receiver<NetworkCommand>) -> usize {
        let mut count = 0;
        while let Ok(cmd) = rx.try_recv() {
            if let NetworkCommand::Broadcast(SwarmMessage::PoolMessage(
                crate::types::PoolMessage::StateDiff(_),
            )) = cmd
            {
                count += 1;
            }
        }
        count
    }

    /// R134: when diff gossip is off (default) every broadcast stays full.
    /// Confirms the wire change is opt-in and the legacy path is preserved.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r134_diff_gossip_off_keeps_full_broadcasts() {
        let (mut pm, state, _id, mut rx) = build_test_pool_manager_with_rx().await;
        // Default config has `state_diff_gossip == false`.
        assert!(!state.config.pool.state_diff_gossip);
        pm.handle_create_pool("test".into()).await.unwrap();
        let _drain = count_state_gossips(&mut rx);

        // Add a member, force a broadcast, expect a full state gossip.
        {
            let mut ps = state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members
                    .push(membership_for(Identity::generate().node_id().clone()));
            }
        }
        pm.gossip_pool_state().await;
        assert_eq!(count_state_diffs(&mut rx), 0);
        let (mut pm2, state2, _id2, mut rx2) = build_test_pool_manager_with_rx().await;
        pm2.handle_create_pool("t2".into()).await.unwrap();
        let _drain2 = count_state_gossips(&mut rx2);
        // Sanity: full broadcasts still fire when the channel hasn't been drained.
        pm2.gossip_pool_state().await;
        assert!(count_state_gossips(&mut rx2) >= 1);
        drop(state2);
    }

    /// R134: with diff gossip on, the first post-create broadcast is full
    /// (no baseline yet); the second broadcast after a state change is a
    /// signed diff that the receiver can apply.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r134_diff_gossip_on_emits_diff_after_baseline() {
        let mut config = Config::default();
        config.pool.state_diff_gossip = true;
        let identity = Identity::generate();
        let db = Database::open_temp().expect("temp db");
        let executor = Arc::new(Mutex::new(crate::inference::executor::ModelExecutor::new()));
        let (shared_state, _s, _d) = SharedState::new(config, identity.clone(), db, executor, None);
        let (_cmd_tx, cmd_rx) = mpsc::channel(16);
        let (network_tx, mut network_rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut pm = PoolManager::new(shared_state.clone(), cmd_rx, network_tx, shutdown_rx);

        pm.handle_create_pool("test-diff".into()).await.unwrap();
        let _drain = count_state_gossips(&mut network_rx);

        // First gossip — no baseline yet → must be full.
        pm.gossip_pool_state().await;
        assert_eq!(
            count_state_gossips(&mut network_rx),
            1,
            "first gossip is full"
        );
        assert_eq!(count_state_diffs(&mut network_rx), 0);
        assert!(pm.last_broadcast_state.is_some());
        assert_eq!(pm.diffs_since_full, 0);

        // Mutate state — add a synthetic member — then gossip again.
        let new_member = Identity::generate();
        {
            let mut ps = shared_state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(new_member.node_id().clone()));
            }
        }

        pm.gossip_pool_state().await;
        let diffs = count_state_diffs(&mut network_rx);
        assert_eq!(diffs, 1, "second broadcast must be a diff");
        assert_eq!(pm.diffs_since_full, 1);
    }

    /// R134: after `MAX_DIFFS_BEFORE_FULL` consecutive diffs the next
    /// broadcast is forced full so late-joiners recover bounded-time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r134_diff_gossip_forces_full_after_cap() {
        let mut config = Config::default();
        config.pool.state_diff_gossip = true;
        let identity = Identity::generate();
        let db = Database::open_temp().expect("temp db");
        let executor = Arc::new(Mutex::new(crate::inference::executor::ModelExecutor::new()));
        let (shared_state, _s, _d) = SharedState::new(config, identity.clone(), db, executor, None);
        let (_cmd_tx, cmd_rx) = mpsc::channel(16);
        let (network_tx, mut network_rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let mut pm = PoolManager::new(shared_state.clone(), cmd_rx, network_tx, shutdown_rx);

        pm.handle_create_pool("test-cap".into()).await.unwrap();
        let _drain = count_state_gossips(&mut network_rx);
        pm.gossip_pool_state().await;
        let _ = count_state_gossips(&mut network_rx); // baseline full

        for i in 0..MAX_DIFFS_BEFORE_FULL {
            // Trigger a mutation each iteration so the diff isn't empty.
            let m = Identity::generate();
            {
                let mut ps = shared_state.credits.pool_state.write().await;
                if let Some(ref mut p) = *ps {
                    p.members.push(membership_for(m.node_id().clone()));
                }
            }
            pm.gossip_pool_state().await;
            assert_eq!(count_state_diffs(&mut network_rx), 1, "diff {i}");
        }
        assert_eq!(pm.diffs_since_full, MAX_DIFFS_BEFORE_FULL);

        // Next broadcast (with a fresh mutation) is forced full.
        let m = Identity::generate();
        {
            let mut ps = shared_state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(membership_for(m.node_id().clone()));
            }
        }
        pm.gossip_pool_state().await;
        // Count full broadcasts FIRST — the count_state_diffs helper drains
        // non-Diff messages in the same call, which would discard the full.
        let (fulls, diffs) = drain_pool_messages(&mut network_rx);
        assert_eq!(diffs, 0);
        assert_eq!(fulls, 1);
        assert_eq!(pm.diffs_since_full, 0);
    }

    /// R134: end-to-end — owner emits a diff, "receiver" PoolManager
    /// (a second instance with the owner's pool already cached) applies
    /// the diff and lands on the same checksum/generation as the owner.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r134_receiver_applies_diff_and_advances_generation() {
        // Owner side: produce a baseline + a diff.
        let mut owner_cfg = Config::default();
        owner_cfg.pool.state_diff_gossip = true;
        let owner_id = Identity::generate();
        let db_owner = Database::open_temp().unwrap();
        let executor = Arc::new(Mutex::new(crate::inference::executor::ModelExecutor::new()));
        let (owner_state, _o, _d) = SharedState::new(
            owner_cfg,
            owner_id.clone(),
            db_owner,
            executor.clone(),
            None,
        );
        let (_t1, owner_cmd_rx) = mpsc::channel(16);
        let (owner_net_tx, mut owner_net_rx) = mpsc::channel(16);
        let (_t2, owner_sd) = watch::channel(false);
        let mut owner_pm =
            PoolManager::new(owner_state.clone(), owner_cmd_rx, owner_net_tx, owner_sd);

        owner_pm.handle_create_pool("e2e".into()).await.unwrap();
        let _ = drain_pool_messages(&mut owner_net_rx);
        owner_pm.gossip_pool_state().await; // baseline full

        // Snapshot baseline that the receiver will start from.
        let baseline: PoolState = owner_state
            .credits
            .pool_state
            .read()
            .await
            .as_ref()
            .cloned()
            .unwrap();
        let _ = drain_pool_messages(&mut owner_net_rx);

        // Owner mutates state then gossips → diff message on the wire.
        // Use a properly-signed acceptance so the receiver's per-member
        // signature verification passes.
        let new_member = Identity::generate();
        let inv_id = uuid::Uuid::new_v4();
        let new_membership = signed_membership_for(&new_member, &baseline.pool_id, inv_id);
        {
            let mut ps = owner_state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members.push(new_membership);
            }
        }
        owner_pm.gossip_pool_state().await;
        let mut diff_msg: Option<swarmllm_types::PoolStateDiff> = None;
        while let Ok(cmd) = owner_net_rx.try_recv() {
            if let NetworkCommand::Broadcast(SwarmMessage::PoolMessage(
                crate::types::PoolMessage::StateDiff(d),
            )) = cmd
            {
                diff_msg = Some(d);
            }
        }
        let diff = diff_msg.expect("owner must emit a StateDiff");
        let expected_gen = owner_state
            .credits
            .pool_state
            .read()
            .await
            .as_ref()
            .map(|s| s.generation)
            .unwrap();
        assert_eq!(diff.new_generation, expected_gen);
        assert_eq!(diff.parent_generation, expected_gen - 1);

        // Receiver side: a second PoolManager belonging to a non-owner.
        // Seed `pool_registry` with the baseline so the diff has a parent.
        let receiver_id = Identity::generate();
        let db_recv = Database::open_temp().unwrap();
        let (recv_state, _r, _r2) = SharedState::new(
            Config::default(),
            receiver_id.clone(),
            db_recv,
            executor,
            None,
        );
        recv_state
            .credits
            .pool_registry
            .insert(baseline.pool_id.clone(), baseline.clone());
        let (_t3, recv_cmd_rx) = mpsc::channel(16);
        let (recv_net_tx, _recv_net_rx) = mpsc::channel(16);
        let (_t4, recv_sd) = watch::channel(false);
        let mut recv_pm = PoolManager::new(recv_state.clone(), recv_cmd_rx, recv_net_tx, recv_sd);

        recv_pm.handle_pool_state_diff_gossip(diff.clone()).await;

        let cached = recv_state
            .credits
            .pool_registry
            .get(&baseline.pool_id)
            .map(|e| e.value().clone())
            .unwrap();
        assert_eq!(cached.generation, expected_gen);
        assert!(cached
            .members
            .iter()
            .any(|m| m.node_id == *new_member.node_id()));
        assert_eq!(
            crate::pool::crypto::pool_state_checksum(&cached),
            diff.state_checksum
        );
    }

    /// R134: diff with non-matching parent_generation is dropped silently.
    /// Prevents an out-of-order delivery from corrupting cached state.
    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn r134_receiver_drops_mismatched_parent_generation() {
        // Build owner that produces a diff at gen=2 → 3.
        let mut owner_cfg = Config::default();
        owner_cfg.pool.state_diff_gossip = true;
        let owner_id = Identity::generate();
        let db_owner = Database::open_temp().unwrap();
        let executor = Arc::new(Mutex::new(crate::inference::executor::ModelExecutor::new()));
        let (owner_state, _o, _d) = SharedState::new(
            owner_cfg,
            owner_id.clone(),
            db_owner,
            executor.clone(),
            None,
        );
        let (_t1, ocrx) = mpsc::channel(16);
        let (otx, mut orx) = mpsc::channel(16);
        let (_t2, osd) = watch::channel(false);
        let mut owner_pm = PoolManager::new(owner_state.clone(), ocrx, otx, osd);
        owner_pm.handle_create_pool("oo".into()).await.unwrap();
        let _ = drain_pool_messages(&mut orx);
        owner_pm.gossip_pool_state().await; // full, gen=1
        let _ = drain_pool_messages(&mut orx);
        {
            let mut ps = owner_state.credits.pool_state.write().await;
            if let Some(ref mut p) = *ps {
                p.members
                    .push(membership_for(Identity::generate().node_id().clone()));
            }
        }
        owner_pm.gossip_pool_state().await; // diff gen=1→2
        let mut diff = None;
        while let Ok(cmd) = orx.try_recv() {
            if let NetworkCommand::Broadcast(SwarmMessage::PoolMessage(
                crate::types::PoolMessage::StateDiff(d),
            )) = cmd
            {
                diff = Some(d);
            }
        }
        let diff = diff.unwrap();

        // Receiver has the pool cached at generation 0 — diff's parent is 1.
        let receiver_id = Identity::generate();
        let (recv_state, _r, _r2) = SharedState::new(
            Config::default(),
            receiver_id.clone(),
            Database::open_temp().unwrap(),
            executor,
            None,
        );
        let mut stale_baseline = owner_state
            .credits
            .pool_state
            .read()
            .await
            .as_ref()
            .cloned()
            .unwrap();
        stale_baseline.generation = 0;
        // Reduce to the baseline membership (just the owner) at stale gen 0.
        stale_baseline
            .members
            .retain(|m| m.node_id == owner_id.node_id().clone());
        let pool_id = stale_baseline.pool_id.clone();
        recv_state
            .credits
            .pool_registry
            .insert(pool_id.clone(), stale_baseline.clone());
        let (_t3, rcrx) = mpsc::channel(16);
        let (rtx, _rrx) = mpsc::channel(16);
        let (_t4, rsd) = watch::channel(false);
        let mut recv_pm = PoolManager::new(recv_state.clone(), rcrx, rtx, rsd);
        recv_pm.handle_pool_state_diff_gossip(diff.clone()).await;

        let cached = recv_state
            .credits
            .pool_registry
            .get(&pool_id)
            .unwrap()
            .value()
            .clone();
        assert_eq!(cached.generation, 0, "diff dropped, generation unchanged");
        assert_eq!(
            cached.members.len(),
            1,
            "diff dropped, membership unchanged"
        );
    }

    /// Drain network_rx once, counting full broadcasts and diff broadcasts
    /// separately so neither categorisation discards the other.
    fn drain_pool_messages(rx: &mut mpsc::Receiver<NetworkCommand>) -> (usize, usize) {
        let mut fulls = 0;
        let mut diffs = 0;
        while let Ok(cmd) = rx.try_recv() {
            if let NetworkCommand::Broadcast(SwarmMessage::PoolMessage(msg)) = cmd {
                match msg {
                    crate::types::PoolMessage::StateGossip(_) => fulls += 1,
                    crate::types::PoolMessage::StateDiff(_) => diffs += 1,
                    _ => {}
                }
            }
        }
        (fulls, diffs)
    }

    /// R138 — `restore_node_mode` returns `None` and writes nothing when
    /// neither tree has the key. SharedState falls through to config default.
    #[test]
    fn restore_node_mode_empty_db_returns_none() {
        let db = Database::open_temp().expect("temp db");
        assert_eq!(restore_node_mode(&db, KEY_PRIVATE_MODE), None);
        assert_eq!(restore_node_mode(&db, KEY_OFFLINE_MODE), None);
        // No spurious writes.
        let v: Option<bool> = db
            .get_json::<bool>(TREE_NODE_MODES, KEY_PRIVATE_MODE)
            .unwrap();
        assert!(v.is_none());
    }

    /// R138 — canonical case: `node_modes` already holds the value; the
    /// helper returns it without touching `pool_state`.
    #[test]
    fn restore_node_mode_reads_new_tree_directly() {
        let db = Database::open_temp().expect("temp db");
        db.put_json(TREE_NODE_MODES, KEY_PRIVATE_MODE, &true)
            .unwrap();
        assert_eq!(restore_node_mode(&db, KEY_PRIVATE_MODE), Some(true));
        // Sanity: pool_state stays untouched.
        let legacy: Option<bool> = db
            .get_json::<bool>(TREE_POOL_STATE, KEY_PRIVATE_MODE)
            .unwrap();
        assert!(legacy.is_none());
    }

    /// R138 — migration path: a legacy `pool_state/{key}` bool from a
    /// pre-R138 daemon is copied to `node_modes` and removed from
    /// `pool_state` on first restart. Idempotent: a second call goes
    /// straight through the new-tree branch.
    #[test]
    fn restore_node_mode_migrates_legacy_pool_state_entry() {
        let db = Database::open_temp().expect("temp db");
        // Simulate a pre-R138 persisted entry.
        db.put_json(TREE_POOL_STATE, KEY_PRIVATE_MODE, &true)
            .unwrap();
        db.put_json(TREE_POOL_STATE, KEY_OFFLINE_MODE, &false)
            .unwrap();

        // First restore: returns the legacy value AND migrates.
        assert_eq!(restore_node_mode(&db, KEY_PRIVATE_MODE), Some(true));
        assert_eq!(restore_node_mode(&db, KEY_OFFLINE_MODE), Some(false));

        // node_modes now owns the values.
        assert_eq!(
            db.get_json::<bool>(TREE_NODE_MODES, KEY_PRIVATE_MODE)
                .unwrap(),
            Some(true)
        );
        assert_eq!(
            db.get_json::<bool>(TREE_NODE_MODES, KEY_OFFLINE_MODE)
                .unwrap(),
            Some(false)
        );

        // pool_state no longer contains the bools — the namespace
        // collision risk that motivated R138 (iter_json::<PoolState>
        // hitting bool payloads) is gone.
        assert_eq!(
            db.get_json::<bool>(TREE_POOL_STATE, KEY_PRIVATE_MODE)
                .unwrap(),
            None
        );
        assert_eq!(
            db.get_json::<bool>(TREE_POOL_STATE, KEY_OFFLINE_MODE)
                .unwrap(),
            None
        );

        // Idempotent: a second restore goes through the new-tree path.
        assert_eq!(restore_node_mode(&db, KEY_PRIVATE_MODE), Some(true));
    }

    /// R138 — when both trees hold the same key, `node_modes` wins
    /// (canonical home). The stale legacy entry is left untouched —
    /// the migration only fires on the new-tree-empty branch.
    #[test]
    fn restore_node_mode_prefers_new_tree_over_legacy() {
        let db = Database::open_temp().expect("temp db");
        db.put_json(TREE_NODE_MODES, KEY_PRIVATE_MODE, &true)
            .unwrap();
        // Hypothetical stale legacy entry (different value to make the
        // assertion meaningful).
        db.put_json(TREE_POOL_STATE, KEY_PRIVATE_MODE, &false)
            .unwrap();
        assert_eq!(restore_node_mode(&db, KEY_PRIVATE_MODE), Some(true));
    }

    /// R138 (closes R102 deferral) — value-cap path of
    /// `check_credit_forward_rate`. Verifies that the second forward is
    /// rejected when the cumulative amount would exceed
    /// `CREDIT_FORWARD_MAX_VALUE_PER_WINDOW`, even though the count is
    /// still under `CREDIT_FORWARD_MAX_PER_WINDOW`.
    #[tokio::test]
    async fn credit_forward_rate_limit_value_cap_rejects_over_total() {
        let (mut pm, _state, _id) = build_test_pool_manager().await;
        let member = NodeId([7u8; 32]);
        // First forward at exactly half the value cap — allowed.
        assert!(pm.check_credit_forward_rate(&member, CREDIT_FORWARD_MAX_VALUE_PER_WINDOW / 2));
        // Second forward at slightly more than half — would push total
        // over the cap. Rejected.
        assert!(
            !pm.check_credit_forward_rate(&member, CREDIT_FORWARD_MAX_VALUE_PER_WINDOW / 2 + 1),
            "value cap must reject when total would exceed limit"
        );
        // The rejected forward was NOT recorded — verify by trying a
        // small amount that fits in the remaining headroom.
        assert!(
            pm.check_credit_forward_rate(&member, CREDIT_FORWARD_MAX_VALUE_PER_WINDOW / 2),
            "rejected forward must not have consumed budget"
        );
    }

    /// R138 — count cap still fires before the value cap on tiny forwards.
    #[tokio::test]
    async fn credit_forward_rate_limit_count_cap_rejects_burst() {
        let (mut pm, _state, _id) = build_test_pool_manager().await;
        let member = NodeId([8u8; 32]);
        // Tiny forwards (value cap won't trip): hit count cap first.
        for _ in 0..CREDIT_FORWARD_MAX_PER_WINDOW {
            assert!(pm.check_credit_forward_rate(&member, 1));
        }
        // Next one is rejected by count even though value is far below cap.
        assert!(!pm.check_credit_forward_rate(&member, 1));
    }

    /// R138 — independent windows per member.
    #[tokio::test]
    async fn credit_forward_rate_limit_is_per_member() {
        let (mut pm, _state, _id) = build_test_pool_manager().await;
        let alice = NodeId([9u8; 32]);
        let bob = NodeId([0xAA; 32]);
        // Alice spends her full value budget.
        assert!(pm.check_credit_forward_rate(&alice, CREDIT_FORWARD_MAX_VALUE_PER_WINDOW));
        // Bob is unaffected.
        assert!(pm.check_credit_forward_rate(&bob, CREDIT_FORWARD_MAX_VALUE_PER_WINDOW));
        // Alice cannot forward more.
        assert!(!pm.check_credit_forward_rate(&alice, 1));
    }

    // ── Invite code v2: end-to-end through the PoolManager ─────────────

    /// Owner-side: with no reachable listen addresses snapshotted, code
    /// generation must surface a clean ServiceUnavailable. This guards
    /// against the regression where a fresh daemon (before the swarm has
    /// emitted NewListenAddr) silently hands out a code with an empty
    /// multiaddr list — which v2's whole purpose is to prevent.
    #[tokio::test]
    async fn v2_generate_fails_when_no_listen_addrs_yet() {
        let (mut pm, state, _id) = build_test_pool_manager().await;
        pm.handle_create_pool("test".into()).await.unwrap();
        // listen_multiaddrs is empty by default in tests.
        assert!(state.listen_multiaddrs.load().is_empty());
        let err = pm.handle_generate_invite_code().await.unwrap_err();
        match err {
            SwarmError::ServiceUnavailable(msg) => {
                assert!(
                    msg.to_lowercase().contains("network addresses"),
                    "msg: {msg}"
                );
            }
            other => panic!("expected ServiceUnavailable, got {other:?}"),
        }
    }

    /// Owner-side: with a listen address present, generation returns a
    /// well-formed `swarmpool://` blob whose decoded payload carries the
    /// owner's NodeId, pool name, multiaddrs, and a still-valid 8-char
    /// join token. This is the round-trip the joining node will see.
    #[tokio::test]
    async fn v2_generate_returns_decodable_blob() {
        let (mut pm, state, identity) = build_test_pool_manager().await;
        pm.handle_create_pool("Test Pool".into()).await.unwrap();
        state.listen_multiaddrs.store(std::sync::Arc::new(vec![
            "/ip4/100.64.0.5/tcp/8810/p2p/12D3KooWFakeXXX".into(),
            "/ip4/192.168.1.5/udp/8800/quic-v1/p2p/12D3KooWFakeXXX".into(),
        ]));

        let encoded = pm.handle_generate_invite_code().await.unwrap();
        assert!(encoded.starts_with("swarmpool://"));

        let payload = crate::pool::invite::decode_invite_code(&encoded).unwrap();
        assert_eq!(payload.pool_id, *identity.node_id());
        assert_eq!(payload.pool_name, "Test Pool");
        assert_eq!(payload.multiaddrs.len(), 2);
        assert!(payload.multiaddrs[0].contains("100.64.0.5"));
        assert_eq!(payload.code.len(), 8);
        assert!(payload.code.chars().all(|c| c.is_ascii_alphanumeric()));
        // The owner's in-memory invite_codes map now keys on hash(code).
        let hash = *blake3::hash(payload.code.as_bytes()).as_bytes();
        assert!(pm.invite_codes.contains_key(&hash));
    }

    /// Joiner-side: pasting a v2 code dials each multiaddr in the bundle
    /// AND broadcasts the JoinRequest with the hash of the inner 8-char
    /// token. The auto-accept hint is armed so a subsequent invitation
    /// from the owner is auto-accepted.
    #[tokio::test]
    async fn v2_join_dials_then_broadcasts() {
        // Owner side — mint a v2 code we can paste.
        let (mut owner_pm, owner_state, _owner_id) = build_test_pool_manager().await;
        owner_pm
            .handle_create_pool("From Owner".into())
            .await
            .unwrap();
        owner_state
            .listen_multiaddrs
            .store(std::sync::Arc::new(vec![
                "/ip4/100.64.0.5/tcp/8810/p2p/12D3KooWFakeXXX".into(),
                "/ip4/198.51.100.5/udp/8800/quic-v1/p2p/12D3KooWFakeXXX".into(),
            ]));
        let encoded = owner_pm.handle_generate_invite_code().await.unwrap();
        let payload = crate::pool::invite::decode_invite_code(&encoded).unwrap();
        let expected_hash = *blake3::hash(payload.code.as_bytes()).as_bytes();

        // Joiner side — paste the code.
        let (mut joiner_pm, _joiner_state, _joiner_id, mut net_rx) =
            build_test_pool_manager_with_rx().await;
        joiner_pm.handle_join_with_code(encoded).await.unwrap();

        // Collect outbound: expect two DialAddress (one per multiaddr) and
        // one Broadcast(JoinRequest) — in some order, but all present.
        let mut dialed = Vec::<String>::new();
        let mut broadcast_hash: Option<[u8; 32]> = None;
        while let Ok(cmd) = net_rx.try_recv() {
            match cmd {
                NetworkCommand::DialAddress(addr) => dialed.push(addr),
                NetworkCommand::Broadcast(SwarmMessage::PoolMessage(
                    crate::types::PoolMessage::JoinRequest { code_hash, .. },
                )) => broadcast_hash = Some(code_hash),
                _ => {}
            }
        }
        assert_eq!(dialed.len(), 2, "both multiaddrs must be dialed");
        assert!(dialed.iter().any(|a| a.contains("100.64.0.5")));
        assert!(dialed.iter().any(|a| a.contains("198.51.100.5")));
        assert_eq!(
            broadcast_hash,
            Some(expected_hash),
            "join request must carry hash of the inner 8-char code"
        );
        assert!(
            joiner_pm.auto_accept_code_hash.is_some(),
            "auto-accept must be armed"
        );
    }

    /// Joiner-side: legacy 8-char code path still works (broadcast-only,
    /// no dial) so existing on-swarm flows aren't regressed.
    #[tokio::test]
    async fn legacy_8char_join_skips_dial() {
        let (mut joiner_pm, _state, _id, mut net_rx) = build_test_pool_manager_with_rx().await;
        joiner_pm
            .handle_join_with_code("A3F7K2M9".into())
            .await
            .unwrap();
        let mut dialed = 0usize;
        let mut broadcast = 0usize;
        while let Ok(cmd) = net_rx.try_recv() {
            match cmd {
                NetworkCommand::DialAddress(_) => dialed += 1,
                NetworkCommand::Broadcast(SwarmMessage::PoolMessage(
                    crate::types::PoolMessage::JoinRequest { .. },
                )) => broadcast += 1,
                _ => {}
            }
        }
        assert_eq!(dialed, 0, "legacy path must not dial");
        assert_eq!(broadcast, 1, "legacy path must broadcast exactly once");
    }

    /// Joiner-side: malformed input (not v2, not 8 chars) is rejected
    /// cleanly with Validation — no panic, no half-armed auto-accept.
    #[tokio::test]
    async fn join_rejects_garbage_input() {
        let (mut pm, _state, _id) = build_test_pool_manager().await;
        let err = pm
            .handle_join_with_code("hello, world".into())
            .await
            .unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
        assert!(pm.auto_accept_code_hash.is_none());
    }
}
