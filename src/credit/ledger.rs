use std::sync::Arc;

use dashmap::DashMap;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use tokio::sync::{mpsc, watch, RwLock};

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    CreditBalance, CreditGossip, CreditTransaction, NetworkCommand, NodeId, PriorityTier, ShardId,
    SwarmMessage,
};

/// Earning/spending rates (credits per unit).
/// IMPORTANT: Both earn and spend use `rate * tokens` (no layer multiplier) to prevent
/// credit inflation. A 22-layer model serving 100 tokens earns 10*100=1000 credits,
/// and the consumer spends 10*100=1000 credits — balanced.
/// All rates are configurable via `[pool.credit_rates]` in config.toml.
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
    peer_balances: Arc<RwLock<Vec<i64>>>,
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
        peer_balances: Arc<RwLock<Vec<i64>>>,
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
                    tracing::error!("CRITICAL: Failed to restore credit balance — lock unavailable at startup. Balance may be zero.");
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
            // Check for a per-pool override first
            if let Ok(pool_state) = ss.credits.pool_state.try_read() {
                if let Some(ref ps) = *pool_state {
                    if let Some(rates) = ss.credits.pool_credit_rates.get(&ps.pool_id) {
                        return rates.value().clone();
                    }
                }
            }
            // Fall back to global config rates
            ss.config.pool.credit_rates.clone()
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
                    tracing::debug!(error = %e, "Pool credit forwarding failed — crediting locally");
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

    /// Apply a penalty (e.g., for serve failure/timeout).
    pub async fn apply_penalty(&self, reason: &str) -> Result<(), SwarmError> {
        let rates = self.credit_rates();
        self.apply_credit(-rates.penalty_serve_failure, false)
            .await?;
        self.persist_balance().await?;

        tracing::warn!(
            penalty = rates.penalty_serve_failure,
            reason,
            "Applied credit penalty"
        );

        Ok(())
    }

    /// Record a completed transaction to the database.
    pub fn record_transaction(&self, tx: &CreditTransaction) -> Result<(), SwarmError> {
        tracing::info!(
            tx_id = %tx.id,
            credit_delta = tx.amount,
            "DIAG: record_transaction"
        );
        self.db
            .put_json(TREE_TRANSACTIONS, &tx.id.to_string(), tx)?;
        Ok(())
    }

    /// Calculate the current priority tier based on balance and network percentile.
    pub async fn calculate_tier(&self) -> PriorityTier {
        let bal = self.balance.read().await;
        let percentile = self.estimate_percentile(bal.balance).await;
        super::priority::calculate_tier(bal.balance, percentile)
    }

    /// Bucket the current balance for gossip (round to nearest 100 for privacy).
    pub async fn balance_bucket(&self) -> i64 {
        let bal = self.balance.read().await;
        bucket_balance(bal.balance)
    }

    pub fn peer_balances(&self) -> &Arc<RwLock<Vec<i64>>> {
        &self.peer_balances
    }

    /// Estimate this node's percentile in the network.
    async fn estimate_percentile(&self, balance: i64) -> f32 {
        let balances = self.peer_balances.read().await;
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
    async fn persist_balance(&self) -> Result<(), SwarmError> {
        let bal = self.balance.read().await;
        self.db.put_json(TREE_CREDITS, KEY_BALANCE, &*bal)?;
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

        // Gossip balance every 5 minutes
        let mut gossip_interval = tokio::time::interval(std::time::Duration::from_secs(300));
        gossip_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Persist balance every 60 seconds
        let mut persist_interval = tokio::time::interval(std::time::Duration::from_secs(60));
        persist_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Earn shard hosting credits every hour
        let mut hosting_interval = tokio::time::interval(std::time::Duration::from_secs(3600));
        hosting_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Clean up expired escrows every 5 minutes
        let mut escrow_cleanup_interval =
            tokio::time::interval(std::time::Duration::from_secs(300));
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
                    let bucket = self.balance_bucket().await;
                    let tier = self.calculate_tier().await;

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
                                pending > 0,
                            ).await {
                                tracing::warn!(error = %e, pending, "Failed to flush pending credit earn");
                                // Put it back so it's not lost
                                ss.credits.pending_credit_earn.fetch_add(pending, std::sync::atomic::Ordering::Release);
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
/// This replicates what `CreditLedger::apply_credit` + `persist_balance` do, so that
/// callers like `InferenceRouter` and `PipelineExecutor` don't bypass persistence and
/// proper accounting.
pub async fn apply_credit_direct(
    balance: &Arc<RwLock<CreditBalance>>,
    db: &crate::storage::db::Database,
    delta: i64,
    is_earning: bool,
) -> Result<(), SwarmError> {
    {
        let mut bal = balance.write().await;
        // SEC-I1: saturating arithmetic to prevent overflow
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
            "Credit balance updated (direct)"
        );

        // Persist while we still hold the balance data (clone for serialization)
        db.put_json(TREE_CREDITS, KEY_BALANCE, &*bal)?;
    }

    Ok(())
}

/// Maximum staleness for a signed balance report (5 minutes).
const BALANCE_REPORT_MAX_AGE_SECS: i64 = 300;

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

    // Timestamp freshness check
    let now = chrono::Utc::now();
    let age_secs = (now - gossip.timestamp).num_seconds().abs();
    if age_secs > BALANCE_REPORT_MAX_AGE_SECS {
        return Err(SwarmError::CreditError(format!(
            "Stale balance report from {}: {}s old (max {}s)",
            gossip.node_id, age_secs, BALANCE_REPORT_MAX_AGE_SECS,
        )));
    }

    // Verify Ed25519 signature
    if gossip.signature.len() != 64 {
        return Err(SwarmError::InvalidSignature);
    }

    let verifying_key =
        VerifyingKey::from_bytes(&gossip.node_id.0).map_err(|_| SwarmError::InvalidSignature)?;

    let sig = Signature::from_bytes(
        gossip
            .signature
            .as_slice()
            .try_into()
            .map_err(|_| SwarmError::InvalidSignature)?,
    );

    let payload =
        build_balance_report_payload(&gossip.node_id, gossip.balance_bucket, gossip.timestamp);

    verifying_key
        .verify(&payload, &sig)
        .map_err(|_| SwarmError::InvalidSignature)?;

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
    peer_balances: &Arc<RwLock<Vec<i64>>>,
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
            // The Vec is rebuilt from the map's values for percentile calculation.
            if let Some(map) = peer_balance_map {
                // Cap the map to prevent unbounded growth from departed peers
                const MAX_PEERS: usize = 10_000;
                if map.len() < MAX_PEERS || map.contains_key(&gossip.node_id) {
                    map.insert(gossip.node_id.clone(), gossip.balance_bucket);
                }

                // Rebuild the Vec from the deduped map
                let mut balances = peer_balances.write().await;
                balances.clear();
                balances.extend(map.iter().map(|e| *e.value()));
            } else {
                // Fallback: raw push (used in tests without SharedState)
                const MAX_BALANCE_VEC_PEERS: usize = 1000;
                let mut balances = peer_balances.write().await;
                balances.push(gossip.balance_bucket);
                if balances.len() > MAX_BALANCE_VEC_PEERS {
                    let excess = balances.len() - MAX_BALANCE_VEC_PEERS;
                    balances.drain(..excess);
                }
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
            Arc::new(RwLock::new(Vec::new())),
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
            Arc::new(RwLock::new(Vec::new())),
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
    async fn apply_penalty_reduces_balance() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let node_id = NodeId([3u8; 32]);
        let balance = Arc::new(RwLock::new(CreditBalance {
            node_id: node_id.clone(),
            balance: 100,
            lifetime_earned: 100,
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
            Arc::new(RwLock::new(Vec::new())),
        );

        ledger.apply_penalty("test timeout").await.unwrap();

        let bal = balance.read().await;
        assert_eq!(bal.balance, 50); // 100 - 50 penalty
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
            Arc::new(RwLock::new(Vec::new())),
        );

        // No peers: default percentile
        let tier = ledger.calculate_tier().await;
        assert_eq!(tier, PriorityTier::Silver); // balance > 0, percentile 0.5

        // Add peer balances: our 500 should be above most
        {
            let mut balances = ledger.peer_balances().write().await;
            for b in [100, 200, 300, 150, 250, 50, 400, 350, 180, 220] {
                balances.push(b);
            }
        }

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
            Arc::new(RwLock::new(Vec::new())),
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

    #[tokio::test]
    async fn process_signed_gossip_adds_balance() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();
        let signature = sign_balance_report(&node_id, 500, timestamp, &identity);

        let peer_balances = Arc::new(RwLock::new(Vec::new()));

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 500,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        let balances = peer_balances.read().await;
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

        let peer_balances = Arc::new(RwLock::new(Vec::new()));

        let gossip = CreditGossip {
            node_id: identity.node_id().clone(),
            balance_bucket: 500,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        // Should have been rejected — no balance added
        let balances = peer_balances.read().await;
        assert_eq!(balances.len(), 0);
    }

    #[tokio::test]
    async fn process_implausible_balance_ignored() {
        let identity = crate::identity::Identity::generate();
        let node_id = identity.node_id().clone();
        let timestamp = chrono::Utc::now();
        let signature = sign_balance_report(&node_id, 200_000_000, timestamp, &identity);

        let peer_balances = Arc::new(RwLock::new(Vec::new()));

        let gossip = CreditGossip {
            node_id,
            balance_bucket: 200_000_000,
            timestamp,
            signature,
        };

        process_balance_gossip(&peer_balances, &gossip, None).await;

        let balances = peer_balances.read().await;
        assert_eq!(balances.len(), 0);
    }
}
