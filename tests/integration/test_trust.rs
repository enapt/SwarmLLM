//! Integration tests for the trust/reputation scoring system (Phase 11).
//!
//! Tests trust score updates, clamping, decay, persistence, and the
//! sybil resistance properties (low trust for unverified nodes).

use dashmap::DashMap;

use swarmllm::credit::trust::{
    TrustEvent, TrustManager, DEFAULT_TRUST, TRUST_INFERENCE_SUCCESS, TRUST_SPOT_CHECK_FAIL,
};
use swarmllm::storage::db::Database;
use swarmllm::types::{NodeId, PeerInfo};

fn make_peer(node_id: NodeId) -> PeerInfo {
    PeerInfo {
        node_id,
        addresses: vec![],
        capability: None,
        last_seen: chrono::Utc::now(),
        latency_ms: Some(50),
        trust_score: DEFAULT_TRUST,
        peer_id_bytes: None,
        active_request_count: 0,
        first_seen: 0,
        verified_transaction_count: 0,
    }
}

/// Test that successful inference increases a peer's trust score.
#[test]
fn test_trust_score_update_on_success() {
    let db = Database::open_temp().unwrap();
    let tm = TrustManager::new(db);
    let registry = DashMap::new();
    let node = NodeId([1u8; 32]);
    registry.insert(node.clone(), make_peer(node.clone()));

    let score = tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);

    // DEFAULT_TRUST (0.5) + TRUST_INFERENCE_SUCCESS (0.01) = 0.51
    assert!((score - (DEFAULT_TRUST + TRUST_INFERENCE_SUCCESS)).abs() < 0.001);

    // Verify it's also reflected in the registry
    let peer_score = registry.get(&node).unwrap().trust_score;
    assert!((peer_score - score).abs() < 0.001);
}

/// Test that failed/dishonest behavior decreases trust score.
#[test]
fn test_trust_score_penalty_on_failure() {
    let db = Database::open_temp().unwrap();
    let tm = TrustManager::new(db);
    let registry = DashMap::new();
    let node = NodeId([2u8; 32]);
    registry.insert(node.clone(), make_peer(node.clone()));

    // Spot check failure should decrease trust
    let score = tm.update_trust(&registry, &node, TrustEvent::SpotCheckFail);
    assert!((score - (DEFAULT_TRUST + TRUST_SPOT_CHECK_FAIL)).abs() < 0.001);
    assert!(score < DEFAULT_TRUST);

    // Signature violation is even more severe
    let score2 = tm.update_trust(&registry, &node, TrustEvent::SignatureViolation);
    assert!(score2 < score);
}

/// Test sybil resistance: a node not in the registry gets trust computed
/// from DB/default, and repeated bad behavior drives it to zero.
#[test]
fn test_sybil_resistance_low_trust() {
    let db = Database::open_temp().unwrap();
    let tm = TrustManager::new(db);
    let registry: DashMap<NodeId, PeerInfo> = DashMap::new();
    let suspicious_node = NodeId([99u8; 32]);

    // Node is NOT in the peer registry — like a sybil that just appeared
    // Multiple bad events should drive trust to near-zero
    let mut score = DEFAULT_TRUST;
    for _ in 0..5 {
        score = tm.update_trust(&registry, &suspicious_node, TrustEvent::InvalidGossip);
    }
    // 0.5 + 5 * (-0.05) = 0.25
    assert!((score - 0.25).abs() < 0.01);

    // Signature violations drive it down further
    score = tm.update_trust(&registry, &suspicious_node, TrustEvent::SignatureViolation);
    // 0.25 + (-0.2) = 0.05
    assert!((score - 0.05).abs() < 0.01);

    // One more drives it to zero (clamped)
    score = tm.update_trust(&registry, &suspicious_node, TrustEvent::SignatureViolation);
    assert!((score - 0.0).abs() < 0.01);
}

/// Test that trust scores persist to the database and survive reload.
#[test]
fn test_trust_score_persistence() {
    let dir = tempfile::tempdir().unwrap();
    let db = Database::open(dir.path()).unwrap();
    let tm = TrustManager::new(db.clone());
    let registry = DashMap::new();
    let node = NodeId([5u8; 32]);
    registry.insert(node.clone(), make_peer(node.clone()));

    // Apply several trust events
    tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);
    tm.update_trust(&registry, &node, TrustEvent::ValidTransaction);
    // Score: 0.5 + 0.01 + 0.02 = 0.53

    // Reload from fresh TrustManager
    let tm2 = TrustManager::new(db);
    let loaded = tm2.get_trust(&node);
    assert!((loaded - 0.53).abs() < 0.001);
}

/// Test decay moves all scores toward the default (0.5) over time.
#[test]
fn test_trust_decay_toward_default() {
    let db = Database::open_temp().unwrap();
    let tm = TrustManager::new(db);
    let registry = DashMap::new();

    // High-trust node
    let high_node = NodeId([10u8; 32]);
    let mut high_peer = make_peer(high_node.clone());
    high_peer.trust_score = 1.0;
    registry.insert(high_node.clone(), high_peer);

    // Low-trust node
    let low_node = NodeId([11u8; 32]);
    let mut low_peer = make_peer(low_node.clone());
    low_peer.trust_score = 0.0;
    registry.insert(low_node.clone(), low_peer);

    // Apply decay
    tm.decay_all(&registry);

    let high_score = registry.get(&high_node).unwrap().trust_score;
    let low_score = registry.get(&low_node).unwrap().trust_score;

    // High should decrease toward 0.5: 1.0 + 0.01*(0.5-1.0) = 0.995
    assert!(high_score < 1.0);
    assert!(high_score > 0.99);

    // Low should increase toward 0.5: 0.0 + 0.01*(0.5-0.0) = 0.005
    assert!(low_score > 0.0);
    assert!(low_score < 0.01);
}

/// Test hydrate_from_db restores trust scores into a fresh registry.
#[test]
fn test_trust_hydrate_from_db() {
    let db = Database::open_temp().unwrap();
    let tm = TrustManager::new(db);
    let registry = DashMap::new();
    let node = NodeId([20u8; 32]);
    registry.insert(node.clone(), make_peer(node.clone()));

    // Update trust twice
    tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);
    tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);
    // Score: 0.5 + 0.01 + 0.01 = 0.52

    // Create fresh registry with default scores
    let registry2 = DashMap::new();
    registry2.insert(node.clone(), make_peer(node.clone()));
    assert!((registry2.get(&node).unwrap().trust_score - 0.5).abs() < 0.001);

    // Hydrate should restore the 0.52 score
    tm.hydrate_from_db(&registry2);
    let restored = registry2.get(&node).unwrap().trust_score;
    assert!((restored - 0.52).abs() < 0.001);
}
