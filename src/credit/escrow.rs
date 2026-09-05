use std::sync::Arc;

use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::credit::ledger::{apply_credit_direct_noted, CreditDelta};
use crate::error::SwarmError;
use crate::storage::db::Database;
use crate::types::{CreditBalance, NodeId};

/// redb tree key for persisted escrow entries.
const TREE_ESCROW: &str = "escrow";

/// Default escrow threshold — requests costing more than this get escrowed.
pub const DEFAULT_ESCROW_THRESHOLD: i64 = 10;

/// What `refund_escrow` says when the entry is gone — the expiry sweep
/// removes it after refunding, so this is "already settled", not "lost".
const ERR_NOT_FOUND: &str = "Escrow not found";

/// Is this refund error just "somebody already settled it"?
///
/// A distributed request may legitimately outlive `ESCROW_TTL_SECS`, and when
/// it does the expiry sweep has already refunded and removed the entry before
/// the request finishes. The caller must be able to tell that apart from a
/// real failure, and it lives HERE, beside the strings it matches, so the two
/// cannot drift across modules — the same reason `reclassify_flattened_error`
/// keeps its markers next to the variants that produce them.
pub fn already_settled(e: &SwarmError) -> bool {
    match e {
        SwarmError::CreditError(m) => {
            m == ERR_NOT_FOUND || (m.starts_with("Escrow ") && m.contains(", not Pending"))
        }
        _ => false,
    }
}

/// Escrow entry TTL in seconds (10 minutes).
const ESCROW_TTL_SECS: u64 = 600;

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
        if amount <= 0 {
            return Err(SwarmError::CreditError(format!(
                "Escrow amount must be positive, got {amount}"
            )));
        }
        let escrow_id = uuid::Uuid::new_v4();
        tracing::info!(
            escrow_id = %escrow_id,
            request_id = %request_id,
            amount,
            from = %from_node,
            "DIAG: escrow created"
        );
        let entry = EscrowEntry {
            id: escrow_id,
            request_id,
            amount,
            from_node: from_node.clone(),
            to_node: None,
            created_at: chrono::Utc::now(),
            status: EscrowStatus::Pending,
        };

        // SEC: Deduct balance FIRST, then persist escrow. If we crash between the
        // balance write and the escrow write, the balance is deducted but no escrow
        // exists — the user loses credits. This is better than the reverse: if we
        // crash after persisting escrow but before deducting balance, cleanup_expired
        // would refund into a balance that was never deducted, creating free credits.
        if let Err(e) = apply_credit_direct_noted(
            balance,
            &self.db,
            -amount,
            CreditDelta::Spending,
            "escrow_reserve",
        )
        .await
        {
            tracing::warn!(error = %e, "Failed to persist balance for escrow deduction");
            return Err(SwarmError::CreditError(format!(
                "Failed to persist balance: {e}"
            )));
        }

        // Now persist the escrow entry. If we crash here, the balance was already
        // deducted but the escrow won't be refunded — user loses credits (safe).
        if let Err(e) = self.db.put_json(TREE_ESCROW, &entry.id.to_string(), &entry) {
            tracing::warn!(error = %e, "Failed to persist escrow entry");
            return Err(SwarmError::CreditError(format!(
                "Failed to persist escrow: {e}"
            )));
        }

        let escrow_id = entry.id;
        self.entries.insert(entry.id, entry);

        tracing::info!(
            escrow_id = %escrow_id,
            request_id = %request_id,
            amount,
            state = "pending",
            "DIAG: escrow hold"
        );

        Ok(escrow_id)
    }

    /// Settle a completed request against its escrow.
    ///
    /// `actual_cost` is what the request really consumed. The escrow held an
    /// *estimate* built from `max_tokens`, and the difference is reconciled
    /// here — refunded when the request used less than reserved, charged when
    /// it used more (possible because the estimate covers completion tokens
    /// while the real cost also counts the prompt).
    ///
    /// Reconciling matters more than it sounds. The estimate is
    /// `RATE_INFERENCE_CONSUME * max_tokens`, which at the shipped defaults of
    /// 10 and 2048 is 20,480 credits — charged in full whether the model
    /// returned two thousand tokens or one. An operator debugging a failing
    /// setup reported a balance of -41,400 after a handful of requests, which
    /// is that estimate applied twice over. The non-escrow path next to this
    /// one always charged actual usage, so the two disagreed by orders of
    /// magnitude depending only on whether the estimate crossed the escrow
    /// threshold.
    pub async fn release_escrow(
        &self,
        escrow_id: uuid::Uuid,
        to_node: &NodeId,
        actual_cost: i64,
        balance: &Arc<RwLock<CreditBalance>>,
    ) -> Result<i64, SwarmError> {
        // Snapshot the entry while holding the DashMap shard lock briefly,
        // then drop the lock before the synchronous redb write. Holding the
        // RefMut across put_json otherwise blocked every other access on the
        // same shard for the disk-write duration.
        let snapshot = {
            let mut entry = self
                .entries
                .get_mut(&escrow_id)
                .ok_or_else(|| SwarmError::CreditError(ERR_NOT_FOUND.into()))?;

            if entry.status != EscrowStatus::Pending {
                return Err(SwarmError::CreditError(format!(
                    "Escrow {} is {:?}, not Pending",
                    escrow_id, entry.status
                )));
            }

            entry.status = EscrowStatus::Released;
            entry.to_node = Some(to_node.clone());
            entry.clone()
        };
        let amount = snapshot.amount;

        if let Err(e) = self
            .db
            .put_json(TREE_ESCROW, &escrow_id.to_string(), &snapshot)
        {
            tracing::warn!(error = %e, "Failed to persist escrow release");
        }

        // Remove from in-memory map — entry is persisted to DB
        self.entries.remove(&escrow_id);

        // Reconcile the estimate against what was actually used. Clamp at zero
        // so a negative actual can never mint credits.
        let actual = actual_cost.max(0);
        let delta = amount - actual;
        if delta > 0 {
            // Reserved more than needed — hand the remainder back.
            if let Err(e) = apply_credit_direct_noted(
                balance,
                &self.db,
                delta,
                CreditDelta::Refund,
                "escrow_settle",
            )
            .await
            {
                tracing::warn!(error = %e, "Failed to persist escrow over-reservation refund");
            }
        } else if delta < 0 {
            // Used more than reserved (long prompt, small max_tokens).
            // `apply_credit_direct` takes a SIGNED delta and adds it — `kind`
            // only selects which lifetime counter moves — so a charge passes
            // the negative through rather than negating it.
            if let Err(e) = apply_credit_direct_noted(
                balance,
                &self.db,
                delta,
                CreditDelta::Spending,
                "escrow_settle_adjust",
            )
            .await
            {
                tracing::warn!(error = %e, "Failed to persist escrow shortfall charge");
            }
        }

        tracing::info!(
            escrow_id = %escrow_id,
            reserved = amount,
            actual,
            reconciled = delta,
            to_node = %to_node,
            state = "released",
            "DIAG: escrow release"
        );

        Ok(actual)
    }

    /// Refund escrowed credits to the requester on failure or timeout.
    pub async fn refund_escrow(
        &self,
        escrow_id: uuid::Uuid,
        balance: &Arc<RwLock<CreditBalance>>,
    ) -> Result<i64, SwarmError> {
        // Snapshot under the DashMap shard lock, drop the lock, then write
        // to redb. Holding the RefMut across the synchronous put_json blocks
        // every other DashMap access on the same shard for the disk-write
        // duration — same pattern as release_escrow above.
        let snapshot = {
            let mut entry = self
                .entries
                .get_mut(&escrow_id)
                .ok_or_else(|| SwarmError::CreditError(ERR_NOT_FOUND.into()))?;

            if entry.status != EscrowStatus::Pending {
                return Err(SwarmError::CreditError(format!(
                    "Escrow {} is {:?}, not Pending",
                    escrow_id, entry.status
                )));
            }

            entry.status = EscrowStatus::Refunded;
            entry.clone()
        };
        let amount = snapshot.amount;

        // Persist updated status BEFORE modifying balance to prevent
        // double-refund on crash. On DB failure we leave the in-memory
        // entry as Refunded — the next restart will see it as Refunded too
        // (we already updated under the lock above) so the balance won't
        // be double-refunded; the cost is one stuck "Refunded with no
        // balance update" state, surfaced via the warn! below if the
        // subsequent apply_credit_direct also fails.
        if let Err(e) = self
            .db
            .put_json(TREE_ESCROW, &escrow_id.to_string(), &snapshot)
        {
            // Revert in-memory status since DB failed.
            if let Some(mut entry) = self.entries.get_mut(&escrow_id) {
                entry.status = EscrowStatus::Pending;
            }
            return Err(SwarmError::Database(format!(
                "Failed to persist escrow refund: {e}"
            )));
        }

        // Remove from in-memory map — entry is persisted to DB
        self.entries.remove(&escrow_id);

        // Return credits to requester. CreditDelta::Refund leaves
        // `lifetime_spent` alone (monotonic) and only adjusts `balance`.
        if let Err(e) = apply_credit_direct_noted(
            balance,
            &self.db,
            amount,
            CreditDelta::Refund,
            "escrow_refund",
        )
        .await
        {
            tracing::warn!(error = %e, "Failed to persist refunded balance");
        }

        tracing::info!(
            escrow_id = %escrow_id,
            amount,
            "Refunded escrow"
        );

        Ok(amount)
    }

    /// Get an escrow entry by ID (used by integration tests).
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
    pub async fn cleanup_expired(&self, balance: &Arc<RwLock<CreditBalance>>) -> usize {
        let now = chrono::Utc::now();
        let ttl = chrono::Duration::seconds(ESCROW_TTL_SECS as i64);
        let mut expired_ids = Vec::new();

        for entry in self.entries.iter() {
            if entry.status == EscrowStatus::Pending && (now - entry.created_at) > ttl {
                expired_ids.push(entry.id);
            }
        }

        let mut count = 0;
        for id in expired_ids {
            if let Some(mut entry) = self.entries.get_mut(&id) {
                if entry.status != EscrowStatus::Pending {
                    continue; // Already released/refunded by another path
                }
                // Don't increment count yet — only count successfully completed
                // expiries. The persist+refund flow below has multiple revert
                // paths; counting up-front would log "Cleaned up N expired
                // escrows" when N includes rollbacks.
                entry.status = EscrowStatus::Expired;
                let amount = entry.amount;

                // Persist — if DB write fails, revert status to prevent double-refund on restart
                if let Err(e) = self.db.put_json(TREE_ESCROW, &id.to_string(), &*entry) {
                    tracing::error!(escrow_id = %id, error = %e, "Failed to persist escrow expiry — reverting to Pending");
                    entry.status = EscrowStatus::Pending;
                    drop(entry);
                    continue;
                }
                drop(entry);
                // Do NOT remove from in-memory map yet — the balance persist below
                // may fail, in which case we need the entry present to revert it
                // back to Pending for retry. Remove only after the refund succeeds.

                // Refund the expired amount. We deliberately DO NOT use
                // `apply_credit_direct_noted(..., CreditDelta::Refund, "escrow_expire_refund")` here even
                // though the accounting semantics match — `cleanup_expired`
                // requires strict crash-safety to support its retry loop:
                // on persist failure, the in-memory balance MUST be reverted
                // and the escrow status MUST go back to Pending so the next
                // tick retries. `apply_credit_direct` deliberately doesn't
                // revert in-memory on persist failure (small crash window
                // is acceptable for hot-path callers), so reusing it here
                // would let the retry tick double-credit when it succeeds.
                let balance_persisted = {
                    let mut bal = balance.write().await;
                    let old_balance = bal.balance;
                    bal.balance = bal.balance.saturating_add(amount);
                    bal.last_updated = chrono::Utc::now();
                    if let Err(e) = self.db.put_json(
                        crate::credit::ledger::TREE_CREDITS,
                        crate::credit::ledger::KEY_BALANCE,
                        &*bal,
                    ) {
                        // Revert in-memory balance to match DB state
                        bal.balance = old_balance;
                        // Also revert escrow back to Pending so next cleanup retries the refund
                        if let Some(mut esc) = self.entries.get_mut(&id) {
                            esc.status = EscrowStatus::Pending;
                            if let Err(e2) = self.db.put_json(TREE_ESCROW, &id.to_string(), &*esc) {
                                tracing::error!(
                                    escrow_id = %id,
                                    error = %e2,
                                    "Failed to persist escrow revert to Pending — escrow state diverged from DB"
                                );
                            }
                        }
                        tracing::error!(
                            escrow_id = %id,
                            error = %e,
                            "Failed to persist credit balance after escrow expiry — reverted escrow to Pending for retry"
                        );
                        false
                    } else {
                        true
                    }
                };

                if balance_persisted {
                    // Refund completed successfully — safe to remove from in-memory map
                    self.entries.remove(&id);
                    count += 1;
                    // Log ONLY on successful refund. Logging unconditionally
                    // makes the failure path log "refunded" while leaving the
                    // escrow Pending — operator audits via grep see false
                    // positives.
                    tracing::info!(
                        escrow_id = %id,
                        amount,
                        "Expired escrow — refunded"
                    );
                }
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

    /// Total amount currently held in pending escrows. Useful for
    /// distinguishing "credits actually spent" from "credits temporarily
    /// locked awaiting release/refund" in admin surfaces.
    pub fn pending_total(&self) -> i64 {
        self.entries
            .iter()
            .filter(|e| e.status == EscrowStatus::Pending)
            .map(|e| e.amount)
            .sum()
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
            lifetime_refunded: 0,
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
        let released = em
            .release_escrow(escrow_id, &to, 100, &balance)
            .await
            .unwrap();
        assert_eq!(released, 100);

        // Balance stays the same (credits went to serving node)
        assert_eq!(balance.read().await.balance, 900);

        // Entry should be removed from in-memory map after release (persisted to DB)
        assert!(em.get_escrow(&escrow_id).is_none());
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

        // Entry should be removed from in-memory map after refund (persisted to DB)
        assert!(em.get_escrow(&escrow_id).is_none());
    }

    /// The bug an operator hit as a -41,400 balance: the escrow reserves
    /// `RATE_INFERENCE_CONSUME * max_tokens` (20,480 at shipped defaults) and
    /// release used to keep all of it regardless of what the request used.
    #[tokio::test]
    async fn release_refunds_the_unused_reservation() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, 10);
        let balance = make_balance(100_000);
        let (from, to) = (NodeId([1u8; 32]), NodeId([2u8; 32]));

        // Reserve for max_tokens=2048 at 10/token.
        let escrow_id = em
            .create_escrow(uuid::Uuid::new_v4(), 20_480, &from, &balance)
            .await
            .unwrap();
        assert_eq!(balance.read().await.balance, 79_520);

        // The model answered in one token after a 9-token prompt.
        let actual = 10 * (9 + 1);
        let settled = em
            .release_escrow(escrow_id, &to, actual, &balance)
            .await
            .unwrap();

        assert_eq!(settled, actual, "release should report real usage");
        assert_eq!(
            balance.read().await.balance,
            100_000 - actual,
            "only actual usage should leave the balance"
        );
    }

    /// A long prompt with a small max_tokens can cost more than reserved —
    /// the estimate only covers completion tokens.
    #[tokio::test]
    async fn release_charges_the_shortfall() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, 10);
        let balance = make_balance(100_000);
        let (from, to) = (NodeId([1u8; 32]), NodeId([2u8; 32]));

        let escrow_id = em
            .create_escrow(uuid::Uuid::new_v4(), 100, &from, &balance)
            .await
            .unwrap();
        let settled = em
            .release_escrow(escrow_id, &to, 250, &balance)
            .await
            .unwrap();

        assert_eq!(settled, 250);
        assert_eq!(balance.read().await.balance, 100_000 - 250);
    }

    /// A negative actual must never mint credits.
    #[tokio::test]
    async fn release_clamps_negative_actual_to_zero() {
        let db = Database::open_temp().unwrap();
        let em = EscrowManager::new(db, 10);
        let balance = make_balance(1000);
        let (from, to) = (NodeId([1u8; 32]), NodeId([2u8; 32]));

        let escrow_id = em
            .create_escrow(uuid::Uuid::new_v4(), 100, &from, &balance)
            .await
            .unwrap();
        em.release_escrow(escrow_id, &to, -500, &balance)
            .await
            .unwrap();
        assert_eq!(
            balance.read().await.balance,
            1000,
            "full reservation returned, nothing minted"
        );
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

        em.release_escrow(escrow_id, &to, 100, &balance)
            .await
            .unwrap();
        // Entry removed from map after release — second release fails (not found)
        let result = em.release_escrow(escrow_id, &to, 100, &balance).await;
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

        em.release_escrow(escrow_id, &to, 100, &balance)
            .await
            .unwrap();
        // Entry removed from map after release — refund fails (not found)
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

        // Entry should be removed from in-memory map after expiry (persisted to DB)
        assert!(em.get_escrow(&escrow_id).is_none());
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

        em.release_escrow(id1, &to, 100, &balance).await.unwrap();
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
