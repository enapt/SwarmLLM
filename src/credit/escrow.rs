use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{CreditBalance, NodeId};

/// Sled tree name for persisted escrow entries.
const TREE_ESCROW: &str = "escrow";

/// Default escrow threshold — requests costing more than this get escrowed.
pub const DEFAULT_ESCROW_THRESHOLD: i64 = 10;

/// Escrow entry TTL in seconds (10 minutes).
pub const ESCROW_TTL_SECS: u64 = 600;

/// Status of an escrow entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EscrowStatus {
    Pending,
    Released,
    Refunded,
    Expired,
}

/// An escrow entry holding credits during an inference request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EscrowEntry {
    pub id: uuid::Uuid,
    pub request_id: uuid::Uuid,
    pub amount: i64,
    pub from_node: NodeId,
    pub to_node: Option<NodeId>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub status: EscrowStatus,
}

/// Manages credit escrow for large inference requests.
///
/// When a request's estimated cost exceeds `escrow_threshold`, credits are
/// held in escrow before pipeline execution begins. On success, they are
/// released to the serving node. On failure/timeout, they are refunded.
pub struct EscrowManager {
    entries: DashMap<uuid::Uuid, EscrowEntry>,
    db: Database,
    threshold: i64,
}

impl EscrowManager {
    pub fn new(db: Database, threshold: i64) -> Self {
        let entries = DashMap::new();
        // Load persisted escrows from DB
        if let Ok(persisted) = db.iter_json::<EscrowEntry>(TREE_ESCROW) {
            for entry in persisted {
                if entry.status == EscrowStatus::Pending {
                    entries.insert(entry.id, entry);
                }
            }
        }
        Self {
            entries,
            db,
            threshold,
        }
    }

    /// Check if a request amount exceeds the escrow threshold.
    pub fn needs_escrow(&self, estimated_cost: i64) -> bool {
        estimated_cost > self.threshold
    }

    /// Create an escrow entry, deducting credits from the requester's balance.
    ///
    /// Returns the escrow ID on success.
    pub async fn create_escrow(
        &self,
        request_id: uuid::Uuid,
        amount: i64,
        from_node: &NodeId,
        balance: &Arc<RwLock<CreditBalance>>,
    ) -> Result<uuid::Uuid, SwarmError> {
        // Deduct from requester balance
        {
            let mut bal = balance.write().await;
            bal.balance = bal.balance.saturating_sub(amount);
            bal.lifetime_spent = bal.lifetime_spent.saturating_add(amount as u64);
            bal.last_updated = chrono::Utc::now();
        }

        let entry = EscrowEntry {
            id: uuid::Uuid::new_v4(),
            request_id,
            amount,
            from_node: from_node.clone(),
            to_node: None,
            created_at: chrono::Utc::now(),
            status: EscrowStatus::Pending,
        };

        // Persist to sled
        if let Err(e) = self
            .db
            .put_json(TREE_ESCROW, &entry.id.to_string(), &entry)
        {
            tracing::warn!(error = %e, "Failed to persist escrow entry");
        }

        let escrow_id = entry.id;
        self.entries.insert(entry.id, entry);

        tracing::info!(
            escrow_id = %escrow_id,
            request_id = %request_id,
            amount,
            "Created escrow"
        );

        Ok(escrow_id)
    }

    /// Release escrowed credits to the serving node on successful completion.
    /// The requester's credits are not returned — they are considered paid.
    pub async fn release_escrow(
        &self,
        escrow_id: uuid::Uuid,
        to_node: &NodeId,
    ) -> Result<i64, SwarmError> {
        let mut entry = self
            .entries
            .get_mut(&escrow_id)
            .ok_or_else(|| SwarmError::CreditError("Escrow not found".into()))?;

        if entry.status != EscrowStatus::Pending {
            return Err(SwarmError::CreditError(format!(
                "Escrow {} is {:?}, not Pending",
                escrow_id, entry.status
            )));
        }

        entry.status = EscrowStatus::Released;
        entry.to_node = Some(to_node.clone());
        let amount = entry.amount;

        // Persist updated status
        if let Err(e) = self
            .db
            .put_json(TREE_ESCROW, &escrow_id.to_string(), &*entry)
        {
            tracing::warn!(error = %e, "Failed to persist escrow release");
        }

        drop(entry);

        // Credit the serving node's perspective (in a real network this would
        // be sent to the serving node; locally we just log it).
        // The requester already paid — no balance change for requester.
        tracing::info!(
            escrow_id = %escrow_id,
            amount,
            to_node = %to_node,
            "Released escrow"
        );

        Ok(amount)
    }

    /// Refund escrowed credits to the requester on failure or timeout.
    pub async fn refund_escrow(
        &self,
        escrow_id: uuid::Uuid,
        balance: &Arc<RwLock<CreditBalance>>,
    ) -> Result<i64, SwarmError> {
        let mut entry = self
            .entries
            .get_mut(&escrow_id)
            .ok_or_else(|| SwarmError::CreditError("Escrow not found".into()))?;

        if entry.status != EscrowStatus::Pending {
            return Err(SwarmError::CreditError(format!(
                "Escrow {} is {:?}, not Pending",
                escrow_id, entry.status
            )));
        }

        entry.status = EscrowStatus::Refunded;
        let amount = entry.amount;

        // Persist updated status
        if let Err(e) = self
            .db
            .put_json(TREE_ESCROW, &escrow_id.to_string(), &*entry)
        {
            tracing::warn!(error = %e, "Failed to persist escrow refund");
        }

        drop(entry);

        // Return credits to requester
        {
            let mut bal = balance.write().await;
            bal.balance = bal.balance.saturating_add(amount);
            bal.lifetime_spent = bal.lifetime_spent.saturating_sub(amount as u64);
            bal.last_updated = chrono::Utc::now();
        }

        tracing::info!(
            escrow_id = %escrow_id,
            amount,
            "Refunded escrow"
        );

        Ok(amount)
    }

    /// Get an escrow entry by ID.
    pub fn get_escrow(&self, escrow_id: &uuid::Uuid) -> Option<EscrowEntry> {
        self.entries.get(escrow_id).map(|e| e.clone())
    }

    /// Get an escrow entry by request ID.
    pub fn get_by_request_id(&self, request_id: &uuid::Uuid) -> Option<EscrowEntry> {
        self.entries
            .iter()
            .find(|e| e.request_id == *request_id)
            .map(|e| e.clone())
    }

    /// Expire stale pending escrows (older than ESCROW_TTL_SECS).
    /// Returns the number of expired entries.
    pub async fn cleanup_expired(
        &self,
        balance: &Arc<RwLock<CreditBalance>>,
    ) -> usize {
        let now = chrono::Utc::now();
        let ttl = chrono::Duration::seconds(ESCROW_TTL_SECS as i64);
        let mut expired_ids = Vec::new();

        for entry in self.entries.iter() {
            if entry.status == EscrowStatus::Pending && (now - entry.created_at) > ttl {
                expired_ids.push(entry.id);
            }
        }

        let count = expired_ids.len();
        for id in expired_ids {
            if let Some(mut entry) = self.entries.get_mut(&id) {
                entry.status = EscrowStatus::Expired;
                let amount = entry.amount;

                // Persist
                let _ = self.db.put_json(TREE_ESCROW, &id.to_string(), &*entry);
                drop(entry);

                // Refund the expired amount
                {
                    let mut bal = balance.write().await;
                    bal.balance = bal.balance.saturating_add(amount);
                    bal.lifetime_spent = bal.lifetime_spent.saturating_sub(amount as u64);
                    bal.last_updated = chrono::Utc::now();
                }

                tracing::info!(
                    escrow_id = %id,
                    amount,
                    "Expired escrow — refunded"
                );
            }
        }

        count
    }

    /// Number of active (pending) escrows.
    pub fn pending_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.status == EscrowStatus::Pending)
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_balance(initial: i64) -> Arc<RwLock<CreditBalance>> {
        Arc::new(RwLock::new(CreditBalance {
            node_id: NodeId([1u8; 32]),
            balance: initial,
            lifetime_earned: initial.max(0) as u64,
            lifetime_spent: 0,
            last_updated: chrono::Utc::now(),
        }))
    }

    #[tokio::test]
    async fn create_and_release_escrow() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        let balance = make_balance(1000);
        let from = NodeId([1u8; 32]);
        let to = NodeId([2u8; 32]);
        let request_id = uuid::Uuid::new_v4();

        let escrow_id = em
            .create_escrow(request_id, 100, &from, &balance)
            .await
            .unwrap();

        // Balance should be reduced
        assert_eq!(balance.read().await.balance, 900);

        // Release to serving node
        let released = em.release_escrow(escrow_id, &to).await.unwrap();
        assert_eq!(released, 100);

        // Balance stays the same (credits went to serving node)
        assert_eq!(balance.read().await.balance, 900);

        // Entry should be Released
        let entry = em.get_escrow(&escrow_id).unwrap();
        assert_eq!(entry.status, EscrowStatus::Released);
        assert_eq!(entry.to_node, Some(to));
    }

    #[tokio::test]
    async fn create_and_refund_escrow() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        let balance = make_balance(500);
        let from = NodeId([1u8; 32]);
        let request_id = uuid::Uuid::new_v4();

        let escrow_id = em
            .create_escrow(request_id, 200, &from, &balance)
            .await
            .unwrap();
        assert_eq!(balance.read().await.balance, 300);

        let refunded = em.refund_escrow(escrow_id, &balance).await.unwrap();
        assert_eq!(refunded, 200);
        assert_eq!(balance.read().await.balance, 500);

        let entry = em.get_escrow(&escrow_id).unwrap();
        assert_eq!(entry.status, EscrowStatus::Refunded);
    }

    #[tokio::test]
    async fn double_release_fails() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        let balance = make_balance(1000);
        let from = NodeId([1u8; 32]);
        let to = NodeId([2u8; 32]);

        let escrow_id = em
            .create_escrow(uuid::Uuid::new_v4(), 100, &from, &balance)
            .await
            .unwrap();

        em.release_escrow(escrow_id, &to).await.unwrap();
        let result = em.release_escrow(escrow_id, &to).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn refund_after_release_fails() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        let balance = make_balance(1000);
        let from = NodeId([1u8; 32]);
        let to = NodeId([2u8; 32]);

        let escrow_id = em
            .create_escrow(uuid::Uuid::new_v4(), 100, &from, &balance)
            .await
            .unwrap();

        em.release_escrow(escrow_id, &to).await.unwrap();
        let result = em.refund_escrow(escrow_id, &balance).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn expire_stale_escrows() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        let balance = make_balance(1000);
        let from = NodeId([1u8; 32]);

        let escrow_id = em
            .create_escrow(uuid::Uuid::new_v4(), 50, &from, &balance)
            .await
            .unwrap();
        assert_eq!(balance.read().await.balance, 950);

        // Manually set creation time to the past
        if let Some(mut entry) = em.entries.get_mut(&escrow_id) {
            entry.created_at =
                chrono::Utc::now() - chrono::Duration::seconds(ESCROW_TTL_SECS as i64 + 10);
        }

        let expired = em.cleanup_expired(&balance).await;
        assert_eq!(expired, 1);
        assert_eq!(balance.read().await.balance, 1000);

        let entry = em.get_escrow(&escrow_id).unwrap();
        assert_eq!(entry.status, EscrowStatus::Expired);
    }

    #[test]
    fn needs_escrow_threshold() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, 10);

        assert!(!em.needs_escrow(5));
        assert!(!em.needs_escrow(10));
        assert!(em.needs_escrow(11));
        assert!(em.needs_escrow(100));
    }

    #[tokio::test]
    async fn get_by_request_id() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        let balance = make_balance(1000);
        let from = NodeId([1u8; 32]);
        let request_id = uuid::Uuid::new_v4();

        em.create_escrow(request_id, 100, &from, &balance)
            .await
            .unwrap();

        let entry = em.get_by_request_id(&request_id).unwrap();
        assert_eq!(entry.request_id, request_id);
        assert_eq!(entry.amount, 100);
    }

    #[tokio::test]
    async fn pending_count() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        let balance = make_balance(1000);
        let from = NodeId([1u8; 32]);
        let to = NodeId([2u8; 32]);

        assert_eq!(em.pending_count(), 0);

        let id1 = em
            .create_escrow(uuid::Uuid::new_v4(), 100, &from, &balance)
            .await
            .unwrap();
        let _id2 = em
            .create_escrow(uuid::Uuid::new_v4(), 200, &from, &balance)
            .await
            .unwrap();
        assert_eq!(em.pending_count(), 2);

        em.release_escrow(id1, &to).await.unwrap();
        assert_eq!(em.pending_count(), 1);
    }

    #[tokio::test]
    async fn escrow_persists_across_reload() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(dir.path()).unwrap();
        let balance = make_balance(1000);
        let from = NodeId([1u8; 32]);
        let request_id = uuid::Uuid::new_v4();

        {
            let em = EscrowManager::new(db.clone(), DEFAULT_ESCROW_THRESHOLD);
            em.create_escrow(request_id, 150, &from, &balance)
                .await
                .unwrap();
        }

        // Reload
        let em2 = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
        assert_eq!(em2.pending_count(), 1);
        let entry = em2.get_by_request_id(&request_id).unwrap();
        assert_eq!(entry.amount, 150);
    }
}
