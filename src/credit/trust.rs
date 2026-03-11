use dashmap::DashMap;

use crate::storage::db::Database;
use crate::types::NodeId;

/// Sled tree name for persisted trust scores.
const TREE_TRUST_SCORES: &str = "trust_scores";

/// Default trust score for newly discovered peers.
pub const DEFAULT_TRUST: f32 = 0.5;

/// Trust score adjustments for various events.
pub const TRUST_INFERENCE_SUCCESS: f32 = 0.01;
pub const TRUST_SPOT_CHECK_FAIL: f32 = -0.1;
pub const TRUST_INVALID_GOSSIP: f32 = -0.05;
pub const TRUST_VALID_TRANSACTION: f32 = 0.02;
pub const TRUST_SIGNATURE_VIOLATION: f32 = -0.2;
/// Trust penalty for nodes sharing a /24 subnet with many other nodes (Sybil indicator).
pub const TRUST_SUBNET_CLUSTERING: f32 = -0.03;

/// Decay rate toward the default trust per health ping cycle.
/// Each ping, trust moves 1% toward DEFAULT_TRUST.
pub const TRUST_DECAY_RATE: f32 = 0.01;

/// Reason for a trust score update, for logging and auditing.
#[derive(Debug, Clone, Copy)]
pub enum TrustEvent {
    InferenceSuccess,
    SpotCheckFail,
    InvalidGossip,
    ValidTransaction,
    SignatureViolation,
    /// Multiple nodes sharing the same /24 subnet — potential Sybil attack.
    SubnetClustering,
}

impl TrustEvent {
    fn delta(self) -> f32 {
        match self {
            Self::InferenceSuccess => TRUST_INFERENCE_SUCCESS,
            Self::SpotCheckFail => TRUST_SPOT_CHECK_FAIL,
            Self::InvalidGossip => TRUST_INVALID_GOSSIP,
            Self::ValidTransaction => TRUST_VALID_TRANSACTION,
            Self::SignatureViolation => TRUST_SIGNATURE_VIOLATION,
            Self::SubnetClustering => TRUST_SUBNET_CLUSTERING,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::InferenceSuccess => "inference_success",
            Self::SpotCheckFail => "spot_check_fail",
            Self::InvalidGossip => "invalid_gossip",
            Self::ValidTransaction => "valid_transaction",
            Self::SignatureViolation => "signature_violation",
            Self::SubnetClustering => "subnet_clustering",
        }
    }
}

/// TrustManager tracks per-peer trust scores, persists them to redb,
/// and provides update/query methods used by the scheduler and ledger.
pub struct TrustManager {
    db: Database,
}

impl TrustManager {
    pub fn new(db: Database) -> Self {
        Self { db }
    }

    /// Update the trust score for a peer after a trust-affecting event.
    /// The updated score is clamped to [0.0, 1.0] and persisted to redb.
    /// Also updates the live PeerInfo in the peer_registry.
    pub fn update_trust(
        &self,
        peer_registry: &DashMap<NodeId, crate::types::PeerInfo>,
        node_id: &NodeId,
        event: TrustEvent,
    ) -> f32 {
        let delta = event.delta();
        let new_score = if let Some(mut peer) = peer_registry.get_mut(node_id) {
            peer.trust_score = (peer.trust_score + delta).clamp(0.0, 1.0);
            peer.trust_score
        } else {
            // Peer not in registry — compute from DB or default
            let current = self.get_trust(node_id);
            let score = (current + delta).clamp(0.0, 1.0);
            // Apply to registry if peer reconnects before next restart
            if let Some(mut peer) = peer_registry.get_mut(node_id) {
                peer.trust_score = score;
            }
            score
        };

        // Persist to DB
        let key = hex::encode(node_id.0);
        if let Err(e) = self.db.put_json(TREE_TRUST_SCORES, &key, &new_score) {
            tracing::warn!(error = %e, node = %node_id, "Failed to persist trust score");
        }

        tracing::debug!(
            node = %node_id,
            event = event.name(),
            score_delta = delta,
            new_score,
            "DIAG: trust score update"
        );

        new_score
    }

    /// Get the current trust score for a peer.
    /// Falls back to DEFAULT_TRUST if unknown.
    pub fn get_trust(&self, node_id: &NodeId) -> f32 {
        let key = hex::encode(node_id.0);
        self.db
            .get_json::<f32>(TREE_TRUST_SCORES, &key)
            .ok()
            .flatten()
            .unwrap_or(DEFAULT_TRUST)
    }

    /// Apply time-based decay toward DEFAULT_TRUST for all known peers.
    /// Called once per health ping cycle. Moves each score 1% toward 0.5.
    pub fn decay_all(&self, peer_registry: &DashMap<NodeId, crate::types::PeerInfo>) {
        for mut entry in peer_registry.iter_mut() {
            let old = entry.trust_score;
            // Linear interpolation toward DEFAULT_TRUST
            entry.trust_score = old + TRUST_DECAY_RATE * (DEFAULT_TRUST - old);
            // Persist the decayed score
            let key = hex::encode(entry.node_id.0);
            let _ = self
                .db
                .put_json(TREE_TRUST_SCORES, &key, &entry.trust_score);
        }
    }

    /// Load persisted trust scores into the peer registry at startup.
    /// For peers already in the registry, overrides the default trust_score
    /// with the persisted value.
    pub fn hydrate_from_db(&self, peer_registry: &DashMap<NodeId, crate::types::PeerInfo>) {
        if let Ok(entries) = self.db.iter_raw(TREE_TRUST_SCORES) {
            for (key_bytes, val_bytes) in entries {
                let key_str = match std::str::from_utf8(&key_bytes) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                let node_id_bytes: [u8; 32] = match hex::decode(key_str) {
                    Ok(b) if b.len() == 32 => {
                        let mut arr = [0u8; 32];
                        arr.copy_from_slice(&b);
                        arr
                    }
                    _ => continue,
                };
                let trust: f32 = match serde_json::from_slice(&val_bytes) {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                let node_id = NodeId(node_id_bytes);
                if let Some(mut peer) = peer_registry.get_mut(&node_id) {
                    // Clamp and reject non-finite values from DB to prevent
                    // NaN/Infinity injection via crafted database entries
                    if trust.is_finite() {
                        peer.trust_score = trust.clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::PeerInfo;

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
            is_lan_peer: false,
        }
    }

    #[test]
    fn update_trust_inference_success() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db);
        let registry = DashMap::new();
        let node = NodeId([1u8; 32]);
        registry.insert(node.clone(), make_peer(node.clone()));

        let score = tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);
        assert!((score - 0.51).abs() < 0.001);
    }

    #[test]
    fn update_trust_spot_check_fail() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db);
        let registry = DashMap::new();
        let node = NodeId([2u8; 32]);
        registry.insert(node.clone(), make_peer(node.clone()));

        let score = tm.update_trust(&registry, &node, TrustEvent::SpotCheckFail);
        assert!((score - 0.4).abs() < 0.001);
    }

    #[test]
    fn update_trust_clamps_to_zero() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db);
        let registry = DashMap::new();
        let node = NodeId([3u8; 32]);
        let mut peer = make_peer(node.clone());
        peer.trust_score = 0.05;
        registry.insert(node.clone(), peer);

        let score = tm.update_trust(&registry, &node, TrustEvent::SignatureViolation);
        assert!((score - 0.0).abs() < 0.001);
    }

    #[test]
    fn update_trust_clamps_to_one() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db);
        let registry = DashMap::new();
        let node = NodeId([4u8; 32]);
        let mut peer = make_peer(node.clone());
        peer.trust_score = 0.99;
        registry.insert(node.clone(), peer);

        let score = tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);
        assert!((score - 1.0).abs() < 0.001);
    }

    #[test]
    fn trust_persists_to_db() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db.clone());
        let registry = DashMap::new();
        let node = NodeId([5u8; 32]);
        registry.insert(node.clone(), make_peer(node.clone()));

        tm.update_trust(&registry, &node, TrustEvent::ValidTransaction);

        // Verify it was persisted
        let loaded = tm.get_trust(&node);
        assert!((loaded - 0.52).abs() < 0.001);
    }

    #[test]
    fn get_trust_unknown_peer_returns_default() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db);
        let node = NodeId([6u8; 32]);

        assert!((tm.get_trust(&node) - DEFAULT_TRUST).abs() < 0.001);
    }

    #[test]
    fn decay_moves_toward_default() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db);
        let registry = DashMap::new();

        let high_node = NodeId([7u8; 32]);
        let mut high_peer = make_peer(high_node.clone());
        high_peer.trust_score = 1.0;
        registry.insert(high_node.clone(), high_peer);

        let low_node = NodeId([8u8; 32]);
        let mut low_peer = make_peer(low_node.clone());
        low_peer.trust_score = 0.0;
        registry.insert(low_node.clone(), low_peer);

        tm.decay_all(&registry);

        let high_score = registry.get(&high_node).unwrap().trust_score;
        let low_score = registry.get(&low_node).unwrap().trust_score;

        // High score should decrease toward 0.5
        assert!(high_score < 1.0);
        assert!(high_score > 0.99); // 1.0 + 0.01*(0.5-1.0) = 0.995

        // Low score should increase toward 0.5
        assert!(low_score > 0.0);
        assert!(low_score < 0.01); // 0.0 + 0.01*(0.5-0.0) = 0.005
    }

    #[test]
    fn hydrate_from_db_restores_scores() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db.clone());
        let registry = DashMap::new();
        let node = NodeId([9u8; 32]);
        registry.insert(node.clone(), make_peer(node.clone()));

        // Set trust via update
        tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);
        tm.update_trust(&registry, &node, TrustEvent::InferenceSuccess);

        // Create a fresh registry and hydrate
        let registry2 = DashMap::new();
        registry2.insert(node.clone(), make_peer(node.clone()));
        assert!((registry2.get(&node).unwrap().trust_score - 0.5).abs() < 0.001);

        tm.hydrate_from_db(&registry2);
        let restored = registry2.get(&node).unwrap().trust_score;
        assert!((restored - 0.52).abs() < 0.001);
    }

    #[test]
    fn update_trust_without_peer_in_registry() {
        let db = Database::open_temp().unwrap();
        let tm = TrustManager::new(db);
        let registry = DashMap::new();
        let node = NodeId([10u8; 32]);

        // No peer in registry — should still persist
        let score = tm.update_trust(&registry, &node, TrustEvent::InvalidGossip);
        assert!((score - 0.45).abs() < 0.001);

        // Check DB persistence
        let loaded = tm.get_trust(&node);
        assert!((loaded - 0.45).abs() < 0.001);
    }
}
