use std::sync::Arc;

use arc_swap::ArcSwap;
use dashmap::DashMap;
use ed25519_dalek::VerifyingKey;
use tokio::sync::{mpsc, watch, RwLock};

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    CreditBalance, CreditGossip, NetworkCommand, NodeId, PriorityTier, ShardId, SwarmMessage,
};

/// Earning/spending rates (credits per unit).
/// IMPORTANT: Both earn and spend use `rate * tokens` (no layer multiplier) to prevent
/// credit inflation. A 22-layer model serving 100 tokens earns 10*100=1000 credits,
/// and the consumer spends 10*100=1000 credits — balanced.
/// These constants are default values. The `CreditLedger` methods (`earn_inference`,
/// `spend_inference`) resolve overrides from `[pool.credit_rates]` in config.toml,
/// but some callers (dispatch.rs, router.rs) reference these constants directly.
pub const RATE_INFERENCE_SERVE: i64 = 10; // per token served (not per layer)
pub const RATE_INFERENCE_CONSUME: i64 = 10; // per token consumed — balanced with serve

/// Minimum credit balance required to submit inference requests.
/// Nodes below this floor have their requests rejected (not just deprioritized).
/// Set to 0 to disable enforcement (permissive mode for small networks).
/// Nodes can earn credits by hosting shards, serving inference, or seeding data.
pub const MIN_BALANCE_FOR_INFERENCE: i64 = -1000;

/// Database tree name for credit data.
pub const TREE_CREDITS: &str = "credits";
pub const KEY_BALANCE: &str = "balance";
/// Tree name for credit transaction history (spec: `credit_txns`).
pub const TREE_TRANSACTIONS: &str = "credit_txns";

/// Interval between bucketed-balance gossip messages used for percentile estimation.
const LEDGER_GOSSIP_INTERVAL_SECS: u64 = 300;
/// Interval between on-disk balance persistence ticks.
const LEDGER_PERSIST_INTERVAL_SECS: u64 = 60;
/// Interval between shard-hosting credit payouts.
const LEDGER_HOSTING_INTERVAL_SECS: u64 = 3600;
/// Interval between expired-escrow cleanup sweeps.
const LEDGER_ESCROW_CLEANUP_INTERVAL_SECS: u64 = 300;

/// CreditLedger tracks the local node's credit balance and transaction history.
///
/// It persists balance and transactions to redb, and gossips balance buckets
/// for network-wide percentile estimation.
pub struct CreditLedger {
    node_id: NodeId,
    balance: Arc<RwLock<CreditBalance>>,
    db: Database,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
    /// Bucketed balances from other nodes, used for percentile estimation.
    ///
    /// `ArcSwap` gives lock-free reads on the inference credit-gate hot
    /// path: `estimate_percentile` → `.load_full()` is a single atomic Arc
    /// clone with no await, even while `process_balance_gossip` is storing
    /// a fresh snapshot on the dispatcher task. Previously this was an
    /// `Arc<RwLock<Vec<i64>>>` — writes (one per inbound `CreditGossip`
    /// in the dispatcher) and reads (inference credit-gate check) held
    /// the same RwLock, so a busy swarm could serialize the two.
    peer_balances: Arc<ArcSwap<Vec<i64>>>,
    /// Reference to SharedState for pool credit forwarding.
    shared_state: Option<std::sync::Arc<crate::daemon::SharedState>>,
    /// Node identity for signing balance reports (Sybil resistance).
    identity: Option<crate::identity::Identity>,
}

impl CreditLedger {
    pub fn new(
        node_id: NodeId,
        balance: Arc<RwLock<CreditBalance>>,
        db: Database,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
        peer_balances: Arc<ArcSwap<Vec<i64>>>,
    ) -> Self {
        // Restore persisted balance synchronously to avoid race condition.
        // redb reads are fast, so this is safe in constructor.
        let restored = match db.get_json::<CreditBalance>(TREE_CREDITS, KEY_BALANCE) {
            Ok(v) => v,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "DIAG: failed to read credit balance from database — starting at zero"
                );
                None
            }
        };

        if let Some(restored_balance) = restored {
            if restored_balance.node_id == node_id {
                // Use try_write — should always succeed at startup before concurrent access.
                if let Ok(mut bal) = balance.try_write() {
                    tracing::info!(
                        balance = restored_balance.balance,
                        lifetime_earned = restored_balance.lifetime_earned,
                        "Restored credit balance from database"
                    );
                    *bal = restored_balance;
                } else {
                    tracing::error!(node_id = %node_id, "CRITICAL: Failed to restore credit balance — lock unavailable at startup. Balance may be zero.");
                }
            }
        }

        Self {
            node_id,
            balance,
            db,
            network_tx,
            shutdown_rx,
            peer_balances,
            shared_state: None,
            identity: None,
        }
    }

    /// Set a shared state reference for pool credit forwarding.
    pub fn set_shared_state(&mut self, shared_state: std::sync::Arc<crate::daemon::SharedState>) {
        self.shared_state = Some(shared_state);
    }

    /// Set the node identity for signing balance reports (Sybil resistance).
    pub fn set_identity(&mut self, identity: crate::identity::Identity) {
        self.identity = Some(identity);
    }

    pub async fn get_balance(&self) -> CreditBalance {
        self.balance.read().await.clone()
    }

    /// Resolve effective credit rates: per-pool override > global config > compile-time defaults.
    fn credit_rates(&self) -> crate::config::CreditRateConfig {
        if let Some(ref ss) = self.shared_state {
            resolve_credit_rates(ss)
        } else {
            crate::config::CreditRateConfig::default()
        }
    }

    /// Earn credits for serving inference.
    ///
    /// Formula: `rate * tokens` — no layer multiplier to stay balanced with the
    /// spend side (`rate * tokens`). Each node that participates in a distributed
    /// pipeline earns for the tokens it helped produce, regardless of how many
    /// layers it processed. This prevents credit inflation on deep models.
    ///
    /// If this node is a pool member (not owner), forwards credits to the pool owner.
    pub async fn earn_inference(
        &self,
        request_id: uuid::Uuid,
        tokens: u32,
    ) -> Result<i64, SwarmError> {
        let rates = self.credit_rates();
        let amount = rates.inference_serve.saturating_mul(tokens as i64);

        tracing::info!(
            amount,
            tokens,
            request_id = %request_id,
            "Earned credits for inference serving"
        );

        // Forward credits to pool owner if we're a member (not owner).
        // Credits are only applied to the local balance if they are NOT forwarded.
        // This prevents credit inflation if forwarding fails.
        if let Some(ref ss) = self.shared_state {
            match crate::pool::forward::forward_credits_to_owner(ss, amount).await {
                Ok(true) => {
                    // Credits forwarded to pool owner — member retains nothing
                    tracing::info!(amount, "Forwarded earned credits to pool owner");
                    return Ok(0);
                }
                Ok(false) => {} // Not in a pool or is the owner — credit locally below
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        amount,
                        "Pool credit forwarding failed — crediting locally. Owner will not receive these credits."
                    );
                }
            }
        }

        // Credit locally (not in pool, or forwarding failed/unavailable)
        self.apply_credit(amount, true).await?;
        self.persist_balance().await?;

        Ok(amount)
    }

    /// Spend credits for consuming inference.
    /// Formula: `rate * tokens` — balanced with earn side.
    pub async fn spend_inference(
        &self,
        request_id: uuid::Uuid,
        tokens: u32,
    ) -> Result<i64, SwarmError> {
        let rates = self.credit_rates();
        let amount = rates.inference_consume.saturating_mul(tokens as i64);
        self.apply_credit(-amount, false).await?;
        self.persist_balance().await?;

        tracing::info!(
            amount,
            tokens,
            request_id = %request_id,
            "Spent credits for inference consumption"
        );

        Ok(amount)
    }

    /// Earn credits for hosting a shard.
    pub async fn earn_shard_hosting(
        &self,
        _shard_id: &ShardId,
        size_gb: f64,
        hours: f32,
    ) -> Result<i64, SwarmError> {
        let rates = self.credit_rates();
        let amount = (rates.shard_hosting as f64 * size_gb * hours as f64) as i64;
        if amount > 0 {
            self.apply_credit(amount, true).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Earn credits for seeding shard data.
    pub async fn earn_shard_seeding(
        &self,
        _shard_id: &ShardId,
        bytes_transferred: u64,
    ) -> Result<i64, SwarmError> {
        let gb = bytes_transferred as f64 / (1024.0 * 1024.0 * 1024.0);
        let rates = self.credit_rates();
        let amount = (rates.shard_seeding as f64 * gb) as i64;
        if amount > 0 {
            self.apply_credit(amount, true).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Earn credits for relay service.
    pub async fn earn_relay_service(&self, duration_seconds: u64) -> Result<i64, SwarmError> {
        let hours = duration_seconds as f64 / 3600.0;
        let rates = self.credit_rates();
        let amount = (rates.relay_service as f64 * hours) as i64;
        if amount > 0 {
            self.apply_credit(amount, true).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Calculate the current priority tier based on balance and network percentile.
    pub async fn calculate_tier(&self) -> PriorityTier {
        let bal = self.balance.read().await;
        let percentile = self.estimate_percentile(bal.balance);
        super::priority::calculate_tier(bal.balance, percentile)
    }

    /// Bucket the current balance for gossip (round to nearest 100 for privacy).
    pub async fn balance_bucket(&self) -> i64 {
        let bal = self.balance.read().await;
        bucket_balance(bal.balance)
    }

    #[cfg(test)]
    pub fn peer_balances(&self) -> &Arc<ArcSwap<Vec<i64>>> {
        &self.peer_balances
    }

    /// Estimate this node's percentile in the network.
    fn estimate_percentile(&self, balance: i64) -> f32 {
        let balances = self.peer_balances.load_full();
        if balances.is_empty() {
            // With no network data, use balance sign as a proxy
            return if balance > 0 { 0.5 } else { 0.1 };
        }

        let below = balances.iter().filter(|&&b| b < balance).count();
        below as f32 / balances.len() as f32
    }

    /// Apply a credit delta to the balance.
    /// SEC-I1: Uses saturating arithmetic to prevent overflow.
    async fn apply_credit(&self, delta: i64, is_earning: bool) -> Result<(), SwarmError> {
        let mut bal = self.balance.write().await;
        bal.balance = bal.balance.saturating_add(delta);
        bal.last_updated = chrono::Utc::now();

        if is_earning {
            bal.lifetime_earned = bal.lifetime_earned.saturating_add(delta.unsigned_abs());
        } else {
            bal.lifetime_spent = bal.lifetime_spent.saturating_add(delta.unsigned_abs());
        }

        tracing::debug!(
            balance = bal.balance,
            delta,
            lifetime_earned = bal.lifetime_earned,
            lifetime_spent = bal.lifetime_spent,
            "Credit balance updated"
        );

        Ok(())
    }

    /// Persist the current balance to the database.
    /// Snapshots out of the lock before the synchronous redb write so a
    /// queued writer (apply_credit on the inference hot path) doesn't park
    /// on the read lock for the disk write duration.
    async fn persist_balance(&self) -> Result<(), SwarmError> {
        let snapshot = self.balance.read().await.clone();
        self.db.put_json(TREE_CREDITS, KEY_BALANCE, &snapshot)?;
        Ok(())
    }

    /// Run the credit ledger background task.
    ///
    /// This periodically:
    /// - Persists the balance to disk
    /// - Gossips bucketed balance for percentile estimation
    /// - Calculates and logs the current tier
    pub async fn run(self) -> Result<(), SwarmError> {
        tracing::info!(target: "swarmllm::credit::ledger", "CreditLedger running");

        // Gossip bucketed balance for percentile estimation.
        let mut gossip_interval =
            tokio::time::interval(std::time::Duration::from_secs(LEDGER_GOSSIP_INTERVAL_SECS));
        gossip_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Persist balance to disk.
        let mut persist_interval =
            tokio::time::interval(std::time::Duration::from_secs(LEDGER_PERSIST_INTERVAL_SECS));
        persist_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Earn shard hosting credits.
        let mut hosting_interval =
            tokio::time::interval(std::time::Duration::from_secs(LEDGER_HOSTING_INTERVAL_SECS));
        hosting_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Clean up expired escrows.
        let mut escrow_cleanup_interval = tokio::time::interval(std::time::Duration::from_secs(
            LEDGER_ESCROW_CLEANUP_INTERVAL_SECS,
        ));
        escrow_cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut shutdown_rx = self.shutdown_rx.clone();

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        // Final persist on shutdown
                        let _ = self.persist_balance().await;
                        tracing::info!(target: "swarmllm::credit::ledger", "CreditLedger shutting down");
                        break;
                    }
                }
                _ = gossip_interval.tick() => {
                    // Read once, derive both. Two sequential read-locks let a
                    // concurrent apply_credit slip a write between them, so the
                    // gossiped (bucket, tier) pair could reflect different
                    // balance values.
                    let (bucket, tier) = {
                        let balance_now = {
                            let bal = self.balance.read().await;
                            bal.balance
                        };
                        let percentile = self.estimate_percentile(balance_now);
                        (
                            bucket_balance(balance_now),
                            super::priority::calculate_tier(balance_now, percentile),
                        )
                    };

                    tracing::info!(
                        balance_bucket = bucket,
                        tier = ?tier,
                        "Gossiping credit balance"
                    );

                    // Sign the balance report if identity is available
                    let timestamp = chrono::Utc::now();
                    let signature = if let Some(ref identity) = self.identity {
                        sign_balance_report(&self.node_id, bucket, timestamp, identity)
                    } else {
                        Vec::new()
                    };

                    let msg = SwarmMessage::CreditGossip(CreditGossip {
                        node_id: self.node_id.clone(),
                        balance_bucket: bucket,
                        timestamp,
                        signature,
                    });

                    if let Err(e) = self.network_tx.send(NetworkCommand::Broadcast(msg)).await {
                        tracing::warn!(error = %e, "Failed to send credit gossip");
                    }
                }
                _ = persist_interval.tick() => {
                    // Flush pending credit earn accumulator from forward participation.
                    // This prevents credit loss from try_write() contention in the hot path.
                    if let Some(ref ss) = self.shared_state {
                        let pending = ss.credits.pending_credit_earn.swap(0, std::sync::atomic::Ordering::AcqRel);
                        if pending != 0 {
                            if let Err(e) = apply_credit_direct(
                                &self.balance,
                                &self.db,
                                pending,
                                if pending > 0 { CreditDelta::Earning } else { CreditDelta::Spending },
                            ).await {
                                // Restore the credits so they aren't lost. Use
                                // compare_exchange so we don't double-count: if
                                // a concurrent earn_inference call landed
                                // increments while we were flushing, the field
                                // is no longer 0 and a naive fetch_add would
                                // add `pending` on top of those increments.
                                // On the contended path we fall back to
                                // fetch_add anyway — losing the credits is
                                // worse than the rare overcount, and the next
                                // flush tick will reconcile against the DB.
                                match ss.credits.pending_credit_earn.compare_exchange(
                                    0,
                                    pending,
                                    std::sync::atomic::Ordering::AcqRel,
                                    std::sync::atomic::Ordering::Acquire,
                                ) {
                                    Ok(_) => tracing::warn!(
                                        error = %e,
                                        pending,
                                        "Failed to flush pending credit earn — restored cleanly"
                                    ),
                                    Err(observed) => {
                                        ss.credits.pending_credit_earn.fetch_add(
                                            pending,
                                            std::sync::atomic::Ordering::AcqRel,
                                        );
                                        tracing::warn!(
                                            error = %e,
                                            pending,
                                            observed,
                                            "Failed to flush pending credit earn — concurrent earn detected, restored with potential overcount"
                                        );
                                    }
                                }
                            } else {
                                tracing::debug!(pending, "Flushed pending forward participation credits");
                            }
                        }
                    }
                    if let Err(e) = self.persist_balance().await {
                        tracing::warn!(error = %e, "Failed to persist credit balance");
                    }
                }
                _ = hosting_interval.tick() => {
                    // Earn credits for each shard we're hosting
                    if let Some(ref ss) = self.shared_state {
                        let mut total_earned: i64 = 0;
                        let mut shard_count: u32 = 0;
                        for manifest in ss.model_registry.models() {
                            for shard in &manifest.shards {
                                let shard_id = ShardId {
                                    model_id: manifest.id.clone(),
                                    index: shard.index,
                                };
                                let holders = ss.model_registry.shard_holders(&shard_id);
                                if holders.contains(&self.node_id) {
                                    let size_gb = shard.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                                    match self.earn_shard_hosting(&shard_id, size_gb, 1.0).await {
                                        Ok(earned) => {
                                            total_earned += earned;
                                            shard_count += 1;
                                        }
                                        Err(e) => {
                                            tracing::warn!(error = %e, "Failed to earn shard hosting credit");
                                        }
                                    }
                                }
                            }
                        }
                        if shard_count > 0 {
                            tracing::info!(
                                shards = shard_count,
                                credits_earned = total_earned,
                                "Earned shard hosting credits"
                            );
                        }

                        // Earn credits for seeding (serving shard data to peers)
                        let bytes_served = ss.shard_bytes_served.swap(
                            0,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if bytes_served > 0 {
                            let dummy_shard = ShardId {
                                model_id: crate::types::ModelId("__seeding__".into()),
                                index: 0,
                            };
                            match self.earn_shard_seeding(&dummy_shard, bytes_served).await {
                                Ok(earned) if earned > 0 => {
                                    tracing::info!(
                                        bytes_served,
                                        credits_earned = earned,
                                        "Earned shard seeding credits"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        "Failed to earn shard seeding credit"
                                    );
                                }
                                _ => {}
                            }
                        }

                        // Drain accumulated relay service seconds and earn credits
                        let relay_secs = ss.relay_seconds_served.swap(
                            0,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        if relay_secs > 0 {
                            match self.earn_relay_service(relay_secs).await {
                                Ok(earned) if earned > 0 => {
                                    tracing::info!(
                                        relay_seconds = relay_secs,
                                        credits_earned = earned,
                                        "Earned relay service credits"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = %e,
                                        "Failed to earn relay service credit"
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                }
                _ = escrow_cleanup_interval.tick() => {
                    // Clean up expired escrows (refund credits for timed-out requests)
                    if let Some(ref ss) = self.shared_state {
                        let cleaned = ss.credits.escrow_manager.cleanup_expired(&self.balance).await;
                        if cleaned > 0 {
                            tracing::info!(
                                cleaned,
                                "Cleaned up expired escrows"
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// Apply credit operations directly on shared state (for use outside the CreditLedger task).
///
/// Resolve effective credit rates from shared state: per-pool override > global config.
pub(crate) fn resolve_credit_rates(
    state: &crate::daemon::state::SharedState,
) -> crate::config::CreditRateConfig {
    if let Ok(pool_state) = state.credits.pool_state.try_read() {
        if let Some(ref ps) = *pool_state {
            if let Some(rates) = state.credits.pool_credit_rates.get(&ps.pool_id) {
                return rates.value().clone();
            }
        }
    }
    state.config.pool.credit_rates.clone()
}

/// Kind of credit movement applied by `apply_credit_direct`. Determines
/// which monotonic lifetime counter is updated.
///
/// - `Earning`  → `lifetime_earned += |delta|`
/// - `Spending` → `lifetime_spent  += |delta|`
/// - `Refund`   → neither counter is touched (used when reverting a
///   prior `Spending`, e.g. an escrow refund — `lifetime_spent` must stay
///   monotonic)
#[derive(Debug, Clone, Copy)]
pub enum CreditDelta {
    Earning,
    Spending,
    Refund,
}

/// This replicates what `CreditLedger::apply_credit` + `persist_balance` do, so that
/// callers like `InferenceRouter`, `PipelineExecutor`, and `EscrowManager`
/// don't bypass persistence and proper accounting.
pub async fn apply_credit_direct(
    balance: &Arc<RwLock<CreditBalance>>,
    db: &crate::storage::db::Database,
    delta: i64,
    kind: CreditDelta,
) -> Result<(), SwarmError> {
    // Update in-memory balance under write lock, then release before DB write.
    // The small crash window (memory updated, process dies before persist) is
    // acceptable — the same pattern is used by CreditLedger::apply_credit.
    let snapshot = {
        let mut bal = balance.write().await;
        // SEC-I1: saturating arithmetic to prevent overflow
        bal.balance = bal.balance.saturating_add(delta);
        bal.last_updated = chrono::Utc::now();

        match kind {
            CreditDelta::Earning => {
                bal.lifetime_earned = bal.lifetime_earned.saturating_add(delta.unsigned_abs());
            }
            CreditDelta::Spending => {
                bal.lifetime_spent = bal.lifetime_spent.saturating_add(delta.unsigned_abs());
            }
            CreditDelta::Refund => {
                // Reverting a prior spend — leave the monotonic counters alone.
            }
        }

        tracing::debug!(
            balance = bal.balance,
            delta,
            kind = ?kind,
            lifetime_earned = bal.lifetime_earned,
            lifetime_spent = bal.lifetime_spent,
            "Credit balance updated (direct)"
        );

        bal.clone()
    };
    // Persist outside write lock to avoid blocking inference hot path
    db.put_json(TREE_CREDITS, KEY_BALANCE, &snapshot)?;

    Ok(())
}

/// Maximum staleness for a signed balance report (5 minutes).
const BALANCE_REPORT_MAX_AGE_SECS: i64 = 300;

/// Allowable clock skew tolerance for a balance report timestamped in the
/// future. Honest cross-node clocks drift by single-digit seconds; anything
/// larger is rejected so an attacker can't pre-sign with a future timestamp
/// to extend the effective replay window.
const CLOCK_SKEW_TOLERANCE_SECS: i64 = 30;

/// Build the deterministic signing payload for a balance report.
/// Format: "swarmllm-balance-v1" || node_id(32) || balance_bucket(8) || timestamp_secs(8)
fn build_balance_report_payload(
    node_id: &NodeId,
    balance_bucket: i64,
    timestamp: chrono::DateTime<chrono::Utc>,
) -> Vec<u8> {
    let mut payload = Vec::with_capacity(56);
    payload.extend_from_slice(b"swarmllm-balance-v1");
    payload.extend_from_slice(&node_id.0);
    payload.extend_from_slice(&balance_bucket.to_le_bytes());
    payload.extend_from_slice(&timestamp.timestamp().to_le_bytes());
    payload
}

/// Sign a balance report with the node's Ed25519 identity.
pub fn sign_balance_report(
    node_id: &NodeId,
    balance_bucket: i64,
    timestamp: chrono::DateTime<chrono::Utc>,
    identity: &crate::identity::Identity,
) -> Vec<u8> {
    let payload = build_balance_report_payload(node_id, balance_bucket, timestamp);
    identity.sign(&payload)
}

/// Verify a signed balance report.
///
/// Checks:
/// 1. Signature is valid Ed25519 over the canonical payload
/// 2. The signing key matches the claimed node_id (node_id == verifying_key bytes)
/// 3. Timestamp is within `BALANCE_REPORT_MAX_AGE_SECS` of `now`
///
/// Returns `Ok(())` for valid signed reports, `Err` for invalid/unsigned/stale.
pub fn verify_balance_report(gossip: &CreditGossip) -> Result<(), SwarmError> {
    // Reject unsigned reports
    if gossip.signature.is_empty() {
        return Err(SwarmError::InvalidSignature);
    }

    // Timestamp freshness check. Use a one-sided staleness bound (NOT .abs())
    // so an attacker can't pre-sign a report with a timestamp 5 minutes in
    // the future and replay it for a full 10-minute window. A small negative
    // tolerance is allowed for honest cross-node clock skew.
    let now = chrono::Utc::now();
    let age_secs = (now - gossip.timestamp).num_seconds();
    if age_secs < -CLOCK_SKEW_TOLERANCE_SECS {
        return Err(SwarmError::CreditError(format!(
            "Future-dated balance report from {}: {}s ahead (skew tolerance {}s)",
            gossip.node_id, -age_secs, CLOCK_SKEW_TOLERANCE_SECS,
        )));
    }
    if age_secs > BALANCE_REPORT_MAX_AGE_SECS {
        return Err(SwarmError::CreditError(format!(
            "Stale balance report from {}: {}s old (max {}s)",
            gossip.node_id, age_secs, BALANCE_REPORT_MAX_AGE_SECS,
        )));
    }

    // Verify Ed25519 signature
    let verifying_key =
        VerifyingKey::from_bytes(&gossip.node_id.0).map_err(|_| SwarmError::InvalidSignature)?;

    let payload =
        build_balance_report_payload(&gossip.node_id, gossip.balance_bucket, gossip.timestamp);

    crate::crypto::verify_ed25519_sig(&gossip.signature, &payload, &verifying_key)?;

    Ok(())
}

/// Process a received balance gossip message with Sybil resistance.
///
/// Only signed reports are accepted. Invalid signatures, unsigned reports,
/// and stale reports are rejected entirely.
///
/// Deduplicates by node_id: each peer gets exactly one entry in the rolling window.
/// This prevents a single peer from dominating the percentile distribution by
/// re-gossiping frequently (Sybil percentile stuffing).
pub async fn process_balance_gossip(
    peer_balances: &Arc<ArcSwap<Vec<i64>>>,
    gossip: &CreditGossip,
    peer_balance_map: Option<&DashMap<NodeId, i64>>,
) {
    // Reject implausible balance buckets
    const MAX_PLAUSIBLE_BALANCE: i64 = 100_000_000;
    if gossip.balance_bucket.abs() > MAX_PLAUSIBLE_BALANCE {
        tracing::debug!(
            balance_bucket = gossip.balance_bucket,
            "Ignoring implausible peer balance gossip"
        );
        return;
    }

    match verify_balance_report(gossip) {
        Ok(()) => {
            // If a peer_balance_map is provided, use it for deduplication.
            // The Vec snapshot is rebuilt from the map's values for
            // percentile estimation and swapped atomically via ArcSwap —
            // readers on the inference hot path never block.
            if let Some(map) = peer_balance_map {
                // Cap the map to prevent unbounded growth from departed peers
                const MAX_PEERS: usize = 10_000;
                if map.len() < MAX_PEERS || map.contains_key(&gossip.node_id) {
                    map.insert(gossip.node_id.clone(), gossip.balance_bucket);
                }

                let new_values: Vec<i64> = map.iter().map(|e| *e.value()).collect();
                peer_balances.store(Arc::new(new_values));
            } else {
                // Fallback: raw push (used in tests without SharedState).
                // ArcSwap has no read-modify-write primitive, so do the
                // update by load + clone + store. Contention is a non-
                // issue in this test-only branch.
                const MAX_BALANCE_VEC_PEERS: usize = 1000;
                let prev = peer_balances.load_full();
                let mut balances: Vec<i64> = (*prev).clone();
                balances.push(gossip.balance_bucket);
                if balances.len() > MAX_BALANCE_VEC_PEERS {
                    let excess = balances.len() - MAX_BALANCE_VEC_PEERS;
                    balances.drain(..excess);
                }
                peer_balances.store(Arc::new(balances));
            }

            tracing::debug!(
                peer = %gossip.node_id,
                bucket = gossip.balance_bucket,
                "Processed signed credit gossip"
            );
        }
        Err(e) => {
            tracing::warn!(
                peer = %gossip.node_id,
                error = %e,
                "Rejected invalid balance report"
            );
        }
    }
}

/// Bucket a balance value for privacy-preserving gossip.
/// Rounds toward negative infinity (floor division) to nearest 100.
/// This ensures negative balances near zero are not confused with positive.
fn bucket_balance(balance: i64) -> i64 {
    balance.div_euclid(100) * 100
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bucket_balance_rounds_down() {
        assert_eq!(bucket_balance(0), 0);
        assert_eq!(bucket_balance(99), 0);
        assert_eq!(bucket_balance(100), 100);
        assert_eq!(bucket_balance(150), 100);
        assert_eq!(bucket_balance(250), 200);
        assert_eq!(bucket_balance(-150), -200);
        assert_eq!(bucket_balance(-50), -100);
    }

    #[tokio::test]
    async fn earn_inference_increases_balance() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([1u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 0,
            lifetime_earned: 0,
            lifetime_spent: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id,
            balance.clone(),
            db,
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        let earned = ledger
            .earn_inference(uuid::Uuid::new_v4(), 10)
            .await
            .unwrap();

        // 10 credits/token * 10 tokens = 100
        assert_eq!(earned, 100);

        let bal = balance.read().await;
        assert_eq!(bal.balance, 100);
        assert_eq!(bal.lifetime_earned, 100);
        assert_eq!(bal.lifetime_spent, 0);
    }

    #[tokio::test]
    async fn spend_inference_decreases_balance() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([2u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 1000,
            lifetime_earned: 1000,
            lifetime_spent: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id,
            balance.clone(),
            db,
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        let spent = ledger
            .spend_inference(uuid::Uuid::new_v4(), 5)
            .await
            .unwrap();

        // 10 credits/token * 5 tokens = 50
        assert_eq!(spent, 50);

        let bal = balance.read().await;
        assert_eq!(bal.balance, 950); // 1000 - 50
        assert_eq!(bal.lifetime_spent, 50);
    }

    #[tokio::test]
    async fn percentile_estimation() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([4u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 500,
            lifetime_earned: 500,
            lifetime_spent: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id,
            balance.clone(),
            db,
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        // No peers: default percentile
        let tier = ledger.calculate_tier().await;
        assert_eq!(tier, PriorityTier::Silver); // balance > 0, percentile 0.5

        // Add peer balances: our 500 should be above most
        ledger.peer_balances().store(Arc::new(vec![
            100, 200, 300, 150, 250, 50, 400, 350, 180, 220,
        ]));

        let tier = ledger.calculate_tier().await;
        // 500 > all 10 peers, percentile = 1.0, so Platinum
        assert_eq!(tier, PriorityTier::Platinum);
    }

    #[tokio::test]
    async fn balance_persists_to_db() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([5u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 0,
            lifetime_earned: 0,
            lifetime_spent: 0,
            last_updated: chrono::Utc::now(),
        }));
        let (network_tx, _rx) = mpsc::channel(16);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let ledger = CreditLedger::new(
            node_id.clone(),
            balance,
            db.clone(),
            network_tx,
            shutdown_rx,
            Arc::new(ArcSwap::from_pointee(Vec::new())),
        );

        ledger
            .earn_inference(uuid::Uuid::new_v4(), 10)
            .await
            .unwrap();

        // Check it was persisted
        let stored: CreditBalance = db.get_json(TREE_CREDITS, KEY_BALANCE).unwrap().unwrap();
        assert_eq!(stored.balance, 100); // 10 credits/token * 10 tokens
        assert_eq!(stored.node_id, node_id);
    }

    // ---- Signed balance report tests ----

    #[test]
    fn sign_and_verify_balance_report() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let bucket = 500;
        let timestamp = chrono::Utc::now();

        let signature = sign_balance_report(&node_id, bucket, timestamp, &identity);

        let gossip = CreditGossip {
            node_id,
            balance_bucket: bucket,
            timestamp,
            signature,
        };

        assert!(verify_balance_report(&gossip).is_ok());
    }

    #[test]
    fn unsigned_report_rejected() {
        let identity = crate::identity::Identity::generate();
        let gossip = CreditGossip {
            node_id: identity.node_id().clone(),
            balance_bucket: 300,
            timestamp: chrono::Utc::now(),
            signature: Vec::new(), // unsigned
        };

        assert!(verify_balance_report(&gossip).is_err());
    }

    #[test]
    fn wrong_signer_rejected() {
        let real_identity = crate::identity::Identity::generate();
        let imposter = crate::identity::Identity::generate();
        let bucket = 500;
        let timestamp = chrono::Utc::now();

        // Imposter signs but claims to be real_identity
        let signature = sign_balance_report(real_identity.node_id(), bucket, timestamp, &imposter);

        let gossip = CreditGossip {
            node_id: real_identity.node_id().clone(),
            balance_bucket: bucket,
            timestamp,
            signature,
        };

        let result = verify_balance_report(&gossip);
        assert!(result.is_err());
    }

    #[test]
    fn tampered_balance_rejected() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();

        let signature = sign_balance_report(&node_id, 500, timestamp, &identity);

        // Tamper: change the balance_bucket
        let gossip = CreditGossip {
            node_id,
            balance_bucket: 999,
            timestamp,
            signature,
        };

        assert!(verify_balance_report(&gossip).is_err());
    }

    #[test]
    fn stale_report_rejected() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        // Timestamp 10 minutes ago — beyond the 5 min window
        let old_timestamp = chrono::Utc::now() - chrono::Duration::seconds(600);

        let signature = sign_balance_report(&node_id, 500, old_timestamp, &identity);

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: old_timestamp,
            signature,
        };

        let result = verify_balance_report(&gossip);
        assert!(result.is_err());
    }

    #[test]
    fn fresh_report_accepted() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        // Timestamp 2 minutes ago — within the 5 min window
        let recent = chrono::Utc::now() - chrono::Duration::seconds(120);

        let signature = sign_balance_report(&node_id, 500, recent, &identity);

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: recent,
            signature,
        };

        assert!(verify_balance_report(&gossip).is_ok());
    }

    #[test]
    fn future_dated_report_rejected() {
        // Pre-signed reports with a timestamp in the future would extend the
        // effective replay window if `.abs()` was used on the staleness
        // check. R65 fixed this with a one-sided bound + 30s clock-skew
        // tolerance — verify a report 60s in the future (well beyond
        // tolerance) is rejected.
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let future = chrono::Utc::now() + chrono::Duration::seconds(60);
        let signature = sign_balance_report(&node_id, 500, future, &identity);
        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: future,
            signature,
        };
        assert!(
            verify_balance_report(&gossip).is_err(),
            "Future-dated balance report should be rejected"
        );
    }

    #[test]
    fn slightly_future_report_accepted_within_skew_tolerance() {
        // Honest cross-node clock skew of 5s should still accept the report.
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let slight_future = chrono::Utc::now() + chrono::Duration::seconds(5);
        let signature = sign_balance_report(&node_id, 500, slight_future, &identity);
        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp: slight_future,
            signature,
        };
        assert!(
            verify_balance_report(&gossip).is_ok(),
            "Report 5s in future (within 30s skew tolerance) should be accepted"
        );
    }

    #[tokio::test]
    async fn process_signed_gossip_adds_balance() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();
        let signature = sign_balance_report(&node_id, 500, timestamp, &identity);

        let peer_balances = Arc::new(ArcSwap::from_pointee(Vec::new()));

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        let balances = peer_balances.load_full();
        assert_eq!(balances.len(), 1);
        assert_eq!(balances[0], 500);
    }

    #[tokio::test]
    async fn process_invalid_gossip_rejected() {
        let identity = crate::identity::Identity::generate();
        let imposter = crate::identity::Identity::generate();
        let timestamp = chrono::Utc::now();

        // Imposter signs a report claiming to be identity
        let signature = sign_balance_report(identity.node_id(), 500, timestamp, &imposter);

        let peer_balances = Arc::new(ArcSwap::from_pointee(Vec::new()));

        let gossip = CreditGossip {
            node_id: identity.node_id().clone(),
            balance_bucket: 500,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        // Should have been rejected — no balance added
        let balances = peer_balances.load_full();
        assert_eq!(balances.len(), 0);
    }

    #[tokio::test]
    async fn process_implausible_balance_ignored() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();
        let signature = sign_balance_report(&node_id, 200_000_000, timestamp, &identity);

        let peer_balances = Arc::new(ArcSwap::from_pointee(Vec::new()));

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 200_000_000,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        let balances = peer_balances.load_full();
        assert_eq!(balances.len(), 0);
    }
}
