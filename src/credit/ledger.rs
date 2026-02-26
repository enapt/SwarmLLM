use std::sync::Arc;

use tokio::sync::{mpsc, watch, RwLock};

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{
    CreditBalance, CreditTransaction, NetworkCommand, NodeId, PriorityTier, ShardId, SwarmMessage,
};

/// Earning/spending rates (credits per unit).
/// These are initial values — tunable via config in the future.
pub const RATE_INFERENCE_SERVE: i64 = 10; // per layer per token
pub const RATE_SHARD_HOSTING: i64 = 1; // per GB per hour
pub const RATE_SHARD_SEEDING: i64 = 5; // per GB transferred
pub const RATE_RELAY_SERVICE: i64 = 2; // per connection hour
pub const RATE_INFERENCE_CONSUME: i64 = 8; // per layer per token (cost)
pub const PENALTY_SERVE_FAILURE: i64 = 50; // per incident

/// Database tree name for credit data.
const TREE_CREDITS: &str = "credits";
const KEY_BALANCE: &str = "balance";
/// Tree name for credit transaction history (spec: `credit_txns`).
pub const TREE_TRANSACTIONS: &str = "credit_txns";

/// CreditLedger tracks the local node's credit balance and transaction history.
///
/// It persists balance and transactions to sled, and gossips balance buckets
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
        // sled reads are fast (in-memory B-tree), so this is safe in constructor.
        let restored = db
            .get_json::<CreditBalance>(TREE_CREDITS, KEY_BALANCE)
            .ok()
            .flatten();

        if let Some(restored_balance) = restored {
            if restored_balance.node_id == node_id {
                // Use try_write to avoid blocking if somehow locked;
                // fall back to best-effort if lock not available immediately.
                if let Ok(mut bal) = balance.try_write() {
                    tracing::info!(
                        balance = restored_balance.balance,
                        lifetime_earned = restored_balance.lifetime_earned,
                        "Restored credit balance from database"
                    );
                    *bal = restored_balance;
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
        }
    }

    /// Set a shared state reference for pool credit forwarding.
    pub fn set_shared_state(&mut self, shared_state: std::sync::Arc<crate::daemon::SharedState>) {
        self.shared_state = Some(shared_state);
    }

    /// Get the current credit balance.
    pub async fn get_balance(&self) -> CreditBalance {
        self.balance.read().await.clone()
    }

    /// Earn credits for serving inference.
    ///
    /// If this node is a pool member (not owner), forwards credits to the pool owner.
    pub async fn earn_inference(
        &self,
        request_id: uuid::Uuid,
        tokens: u32,
        layers: u32,
    ) -> Result<i64, SwarmError> {
        let amount = RATE_INFERENCE_SERVE * (layers as i64) * (tokens as i64);
        self.apply_credit(amount, true).await?;
        self.persist_balance().await?;

        tracing::info!(
            amount,
            tokens,
            layers,
            request_id = %request_id,
            "Earned credits for inference serving"
        );

        // Forward credits to pool owner if we're a member (not owner)
        if let Some(ref ss) = self.shared_state {
            if let Err(e) = crate::pool::forward::forward_credits_to_owner(ss, amount).await {
                tracing::debug!(error = %e, "Pool credit forwarding skipped");
            }
        }

        Ok(amount)
    }

    /// Spend credits for consuming inference.
    pub async fn spend_inference(
        &self,
        request_id: uuid::Uuid,
        tokens: u32,
        layers: u32,
    ) -> Result<i64, SwarmError> {
        let amount = RATE_INFERENCE_CONSUME * (layers as i64) * (tokens as i64);
        self.apply_credit(-amount, false).await?;
        self.persist_balance().await?;

        tracing::info!(
            amount,
            tokens,
            layers,
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
        let amount = (RATE_SHARD_HOSTING as f64 * size_gb * hours as f64) as i64;
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
        let amount = (RATE_SHARD_SEEDING as f64 * gb) as i64;
        if amount > 0 {
            self.apply_credit(amount, true).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Earn credits for relay service.
    pub async fn earn_relay_service(&self, duration_seconds: u64) -> Result<i64, SwarmError> {
        let hours = duration_seconds as f64 / 3600.0;
        let amount = (RATE_RELAY_SERVICE as f64 * hours) as i64;
        if amount > 0 {
            self.apply_credit(amount, true).await?;
            self.persist_balance().await?;
        }
        Ok(amount)
    }

    /// Apply a penalty (e.g., for serve failure/timeout).
    pub async fn apply_penalty(&self, reason: &str) -> Result<(), SwarmError> {
        self.apply_credit(-PENALTY_SERVE_FAILURE, false).await?;
        self.persist_balance().await?;

        tracing::warn!(
            penalty = PENALTY_SERVE_FAILURE,
            reason,
            "Applied credit penalty"
        );

        Ok(())
    }

    /// Record a completed transaction to the database.
    pub fn record_transaction(&self, tx: &CreditTransaction) -> Result<(), SwarmError> {
        self.db
            .put_json(TREE_TRANSACTIONS, &tx.id.to_string(), tx)?;
        Ok(())
    }

    /// Update the peer balance list from a gossip message.
    /// Balances are bucketed (rounded to nearest 100) for privacy.
    pub async fn update_peer_balance(&self, balance_bucket: i64) {
        let mut balances = self.peer_balances.write().await;
        balances.push(balance_bucket);
        // Keep a rolling window of the most recent 1000 observations
        if balances.len() > 1000 {
            let excess = balances.len() - 1000;
            balances.drain(..excess);
        }
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

    /// Get a reference to the peer balances for external use.
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
    async fn apply_credit(&self, delta: i64, is_earning: bool) -> Result<(), SwarmError> {
        let mut bal = self.balance.write().await;
        bal.balance += delta;
        bal.last_updated = chrono::Utc::now();

        if is_earning {
            bal.lifetime_earned += delta.unsigned_abs();
        } else {
            bal.lifetime_spent += delta.unsigned_abs();
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
        tracing::info!("CreditLedger running");

        // Gossip balance every 5 minutes
        let mut gossip_interval = tokio::time::interval(std::time::Duration::from_secs(300));
        gossip_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // Persist balance every 60 seconds
        let mut persist_interval = tokio::time::interval(std::time::Duration::from_secs(60));
        persist_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut shutdown_rx = self.shutdown_rx.clone();

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        // Final persist on shutdown
                        let _ = self.persist_balance().await;
                        tracing::info!("CreditLedger shutting down");
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

                    // Send credit gossip via network
                    let msg = SwarmMessage::CreditGossip(crate::types::CreditGossip {
                        node_id: self.node_id.clone(),
                        balance_bucket: bucket,
                        timestamp: chrono::Utc::now(),
                    });

                    if let Err(e) = self.network_tx.send(NetworkCommand::Broadcast(msg)).await {
                        tracing::warn!(error = %e, "Failed to send credit gossip");
                    }
                }
                _ = persist_interval.tick() => {
                    if let Err(e) = self.persist_balance().await {
                        tracing::warn!(error = %e, "Failed to persist credit balance");
                    }
                }
            }
        }

        Ok(())
    }
}

/// Bucket a balance value for privacy-preserving gossip.
/// Rounds to the nearest 100.
fn bucket_balance(balance: i64) -> i64 {
    (balance / 100) * 100
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
        assert_eq!(bucket_balance(-150), -100);
        assert_eq!(bucket_balance(-50), 0);
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
            .earn_inference(uuid::Uuid::new_v4(), 10, 2)
            .await
            .unwrap();

        // 10 credits/layer/token * 2 layers * 10 tokens = 200
        assert_eq!(earned, 200);

        let bal = balance.read().await;
        assert_eq!(bal.balance, 200);
        assert_eq!(bal.lifetime_earned, 200);
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
            .spend_inference(uuid::Uuid::new_v4(), 5, 3)
            .await
            .unwrap();

        // 8 credits/layer/token * 3 layers * 5 tokens = 120
        assert_eq!(spent, 120);

        let bal = balance.read().await;
        assert_eq!(bal.balance, 880); // 1000 - 120
        assert_eq!(bal.lifetime_spent, 120);
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
        for b in [100, 200, 300, 150, 250, 50, 400, 350, 180, 220] {
            ledger.update_peer_balance(b).await;
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
            .earn_inference(uuid::Uuid::new_v4(), 10, 1)
            .await
            .unwrap();

        // Check it was persisted
        let stored: CreditBalance = db.get_json(TREE_CREDITS, KEY_BALANCE).unwrap().unwrap();
        assert_eq!(stored.balance, 100); // 10 * 1 * 10
        assert_eq!(stored.node_id, node_id);
    }
}
