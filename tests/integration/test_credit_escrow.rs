//! Integration tests for the credit escrow lifecycle (Phase 11).
//!
//! Tests the full create → fund → release/refund/expire flow through
//! the EscrowManager, including persistence across reloads and concurrent
//! escrow management.

use std::sync::Arc;
use tokio::sync::RwLock;

use swarmllm::credit::escrow::{EscrowManager, EscrowStatus, DEFAULT_ESCROW_THRESHOLD};
use swarmllm::storage::db::Database;
use swarmllm::types::{CreditBalance, NodeId};

fn make_balance(initial: i64) -> Arc<RwLock<CreditBalance>> {
    Arc::new(RwLock::new(CreditBalance {
        node_id: NodeId([1u8; 32]),
        balance: initial,
        lifetime_earned: initial.max(0) as u64,
        lifetime_spent: 0,
        last_updated: chrono::Utc::now(),
    }))
}

/// Test the full escrow lifecycle: create → release to provider.
/// Verifies that credits are deducted on creation and stay deducted
/// after release (they go to the serving node).
#[tokio::test]
async fn test_escrow_create_and_release() {
    let db = Database::open_temp().unwrap();
    let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
    let balance = make_balance(1000);
    let from = NodeId([1u8; 32]);
    let to = NodeId([2u8; 32]);
    let request_id = uuid::Uuid::new_v4();

    // Create escrow — should deduct from balance
    let escrow_id = em
        .create_escrow(request_id, 100, &from, &balance)
        .await
        .unwrap();
    assert_eq!(balance.read().await.balance, 900);
    assert_eq!(em.pending_count(), 1);

    // Verify we can look it up by request_id
    let entry = em.get_by_request_id(&request_id).unwrap();
    assert_eq!(entry.amount, 100);
    assert_eq!(entry.status, EscrowStatus::Pending);
    assert!(entry.to_node.is_none());

    // Release to serving node — credits stay with provider
    let released = em.release_escrow(escrow_id, &to).await.unwrap();
    assert_eq!(released, 100);
    assert_eq!(balance.read().await.balance, 900); // Not refunded
    assert_eq!(em.pending_count(), 0);

    // Entry should be Released with to_node set
    let entry = em.get_escrow(&escrow_id).unwrap();
    assert_eq!(entry.status, EscrowStatus::Released);
    assert_eq!(entry.to_node, Some(to));
}

/// Test the escrow dispute path: create → refund.
/// Verifies that credits are returned to the requester on failure.
#[tokio::test]
async fn test_escrow_dispute_refund() {
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

    // Refund on failure — credits return to requester
    let refunded = em.refund_escrow(escrow_id, &balance).await.unwrap();
    assert_eq!(refunded, 200);
    assert_eq!(balance.read().await.balance, 500);

    let entry = em.get_escrow(&escrow_id).unwrap();
    assert_eq!(entry.status, EscrowStatus::Refunded);
}

/// Test that cleanup_expired correctly identifies non-expired escrows.
/// Since we cannot backdate entries via the public API, we verify that
/// freshly created escrows are NOT expired, and that the cleanup function
/// returns 0 when nothing has timed out.
#[tokio::test]
async fn test_escrow_cleanup_no_false_positives() {
    let db = Database::open_temp().unwrap();
    let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
    let balance = make_balance(1000);
    let from = NodeId([1u8; 32]);

    // Create two escrows
    em.create_escrow(uuid::Uuid::new_v4(), 100, &from, &balance)
        .await
        .unwrap();
    em.create_escrow(uuid::Uuid::new_v4(), 200, &from, &balance)
        .await
        .unwrap();
    assert_eq!(balance.read().await.balance, 700);
    assert_eq!(em.pending_count(), 2);

    // Cleanup should find nothing expired (just created)
    let expired = em.cleanup_expired(&balance).await;
    assert_eq!(expired, 0);
    assert_eq!(em.pending_count(), 2);
    // Balance unchanged since nothing was refunded
    assert_eq!(balance.read().await.balance, 700);
}

/// Test the escrow threshold: requests below the threshold don't need escrow.
#[tokio::test]
async fn test_escrow_threshold_check() {
    let db = Database::open_temp().unwrap();
    let em = EscrowManager::new(db, 10);

    assert!(!em.needs_escrow(5));  // Below threshold
    assert!(!em.needs_escrow(10)); // At threshold (not above)
    assert!(em.needs_escrow(11));  // Above threshold
    assert!(em.needs_escrow(100)); // Well above threshold
}

/// Test that double-release of the same escrow is rejected.
/// Prevents double-spending where a provider claims credits twice.
#[tokio::test]
async fn test_escrow_double_release_rejected() {
    let db = Database::open_temp().unwrap();
    let em = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
    let balance = make_balance(1000);
    let from = NodeId([1u8; 32]);
    let to = NodeId([2u8; 32]);

    let escrow_id = em
        .create_escrow(uuid::Uuid::new_v4(), 100, &from, &balance)
        .await
        .unwrap();

    // First release succeeds
    em.release_escrow(escrow_id, &to).await.unwrap();

    // Second release fails
    let result = em.release_escrow(escrow_id, &to).await;
    assert!(result.is_err());
}

/// Test that refunding after release is rejected.
/// Prevents a requester from reclaiming credits already paid to provider.
#[tokio::test]
async fn test_escrow_refund_after_release_rejected() {
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

    // Cannot refund a released escrow
    let result = em.refund_escrow(escrow_id, &balance).await;
    assert!(result.is_err());

    // Balance should still reflect the original deduction
    assert_eq!(balance.read().await.balance, 900);
}

/// Test escrow persistence across manager reloads.
/// Verifies that pending escrows survive a restart.
#[tokio::test]
async fn test_escrow_persists_across_reload() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let balance = make_balance(1000);
    let from = NodeId([1u8; 32]);
    let request_id = uuid::Uuid::new_v4();

    // Create escrow with first manager instance
    {
        let em = EscrowManager::new(db.clone(), DEFAULT_ESCROW_THRESHOLD);
        em.create_escrow(request_id, 150, &from, &balance)
            .await
            .unwrap();
        assert_eq!(em.pending_count(), 1);
    }

    // Create a new manager — should reload from DB
    let em2 = EscrowManager::new(db, DEFAULT_ESCROW_THRESHOLD);
    assert_eq!(em2.pending_count(), 1);

    let entry = em2.get_by_request_id(&request_id).unwrap();
    assert_eq!(entry.amount, 150);
    assert_eq!(entry.status, EscrowStatus::Pending);
}
