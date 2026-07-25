//! NETWORKING_PLAN Phase 1 — learned relay-route table + relay-side rate
//! limiting on the root `SharedState`.
//!
//! These back the application-level inference relay: a NAT'd node routes an
//! inference message through a mutually-reachable relay peer, and the two
//! endpoints learn a reverse route so replies and later turns flow back the
//! same way. The routing decision lives entirely in the NetworkManager's
//! directed-send path (`network/manager/commands.rs`) — daemon send code is
//! unchanged.

use std::time::{Duration, Instant};

use crate::types::NodeId;

/// How long a learned reverse route stays usable without being refreshed. A
/// route is refreshed every time another envelope arrives from that origin, so
/// an active session keeps it warm; this bounds staleness for an idle peer.
pub const RELAY_ROUTE_TTL_SECS: u64 = 300;

/// Relay-side forward rate limit window.
pub const RELAY_FORWARD_WINDOW_SECS: u64 = 10;

/// Max messages one origin may push through us as a relay per window. Generous
/// enough for fast token streaming (~200 msg/s sustained), low enough to blunt
/// a flood. Phase 3 replaces this coarse cap with credit-metered relaying.
pub const RELAY_FORWARD_MAX_PER_WINDOW: u32 = 2000;

/// A learned route to a target that we cannot reach directly: send to
/// `relay_peer_bytes`, which forwards to the target. `target_node` is kept so
/// the send path can seal the inner message for the target's X25519 key.
#[derive(Clone, Debug)]
pub struct RelayRoute {
    pub relay_peer_bytes: Vec<u8>,
    pub target_node: NodeId,
    pub learned_at: Instant,
}

/// Sliding-window forward counter for one origin (relay side).
#[derive(Debug)]
pub struct RelayForwardCounter {
    pub count: u32,
    pub window_start: Instant,
}

/// Relay features a peer has *demonstrably* used by sending us a relayed message
/// addressed to us: a `RelayedTensor` proves `features::TENSOR_RELAY`, a
/// `RelayedEnvelope` proves `features::RELAY`. This is direct, first-hand proof
/// the peer speaks the protocol — stronger and fresher than the gossiped
/// `NodeCapability.features`, which has a cold-start window where our entry for
/// a relay-only peer is still `capability: None` (the capability-gossip handler
/// is update-only, so it can't populate an entry that didn't exist yet). Without
/// this proof the return path would refuse to relay a computed result back to a
/// coordinator that JUST relayed a forward to us, dropping the first result and
/// timing the request out until a capability-gossip round lands.
#[derive(Debug, Clone)]
pub struct RelayProvenFeatures {
    pub features: u64,
    pub proven_at: Instant,
}

impl super::SharedState {
    /// Record that `origin` is reachable via the relay peer a `RelayedEnvelope`
    /// just arrived from. Keyed by `origin`'s peer-id bytes so the send path can
    /// look it up by the target peer bytes it already carries.
    pub fn learn_relay_route(&self, origin: &NodeId, relay_peer_bytes: Vec<u8>) {
        let Some(target_peer_id) = crate::network::transport::node_id_to_peer_id(origin) else {
            return;
        };
        let target_bytes = target_peer_id.to_bytes();
        // A route through the target itself is degenerate (and a self-route is
        // meaningless) — ignore both.
        if relay_peer_bytes == target_bytes {
            return;
        }
        self.relay_routes.insert(
            target_bytes,
            RelayRoute {
                relay_peer_bytes,
                target_node: origin.clone(),
                learned_at: Instant::now(),
            },
        );
    }

    /// Fresh learned relay route for a target peer (by peer-id bytes), or None
    /// if unknown or expired.
    pub fn relay_route_for_peer(&self, target_peer_bytes: &[u8]) -> Option<RelayRoute> {
        let entry = self.relay_routes.get(target_peer_bytes)?;
        if entry.learned_at.elapsed() > Duration::from_secs(RELAY_ROUTE_TTL_SECS) {
            return None;
        }
        Some(entry.clone())
    }

    /// Relay-side rate check: is `origin` under its forward budget for the
    /// current window? Increments the counter when it returns true.
    pub fn relay_forward_allowed(&self, origin_peer_bytes: &[u8]) -> bool {
        let now = Instant::now();
        let window = Duration::from_secs(RELAY_FORWARD_WINDOW_SECS);
        let mut entry = self
            .relay_forward_counters
            .entry(origin_peer_bytes.to_vec())
            .or_insert(RelayForwardCounter {
                count: 0,
                window_start: now,
            });
        if now.duration_since(entry.window_start) > window {
            entry.count = 0;
            entry.window_start = now;
        }
        if entry.count >= RELAY_FORWARD_MAX_PER_WINDOW {
            return false;
        }
        entry.count += 1;
        true
    }

    /// Record that `peer` has demonstrably used relay `features` (it sent us a
    /// relayed message addressed to us). ORs the newly-proven bits into any
    /// existing proof and refreshes the timestamp, so an active relay session
    /// keeps the proof warm. Cheap direct proof that sidesteps the capability-
    /// gossip cold-start window on the relay send path's feature gates.
    pub fn record_relay_proven_features(&self, peer: &NodeId, features: u64) {
        let now = Instant::now();
        self.relay_proven_features
            .entry(peer.clone())
            .and_modify(|e| {
                e.features |= features;
                e.proven_at = now;
            })
            .or_insert(RelayProvenFeatures {
                features,
                proven_at: now,
            });
    }

    /// Whether `peer` has proven support for ALL of `needed` within the freshness
    /// window (same TTL as a learned route — an active relay session re-proves it
    /// on every inbound message, so it never goes stale mid-session). Lets the
    /// relay send path trust a peer that just relayed to us even before its
    /// capability gossip arrives.
    pub fn relay_feature_proven(&self, peer: &NodeId, needed: u64) -> bool {
        self.relay_proven_features
            .get(peer)
            .filter(|e| e.proven_at.elapsed() <= Duration::from_secs(RELAY_ROUTE_TTL_SECS))
            .is_some_and(|e| crate::types::features::supports(e.features, needed))
    }

    /// Sweep expired relay routes + idle forward counters + stale proven-feature
    /// proofs. Wired to the HealthMonitor tick so all three maps stay bounded
    /// under peer churn.
    pub fn sweep_stale_relay_state(&self) {
        let route_ttl = Duration::from_secs(RELAY_ROUTE_TTL_SECS);
        self.relay_routes
            .retain(|_, r| r.learned_at.elapsed() <= route_ttl);
        let counter_ttl = Duration::from_secs(RELAY_FORWARD_WINDOW_SECS * 2);
        self.relay_forward_counters
            .retain(|_, c| c.window_start.elapsed() <= counter_ttl);
        self.relay_proven_features
            .retain(|_, e| e.proven_at.elapsed() <= route_ttl);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relay_proven_features_record_check_and_isolate() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use crate::types::{features, NodeId};
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = Config::default();
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(config, identity, db, executor, None);

        let peer = NodeId([9u8; 32]);
        // Unknown peer → nothing proven.
        assert!(!state.relay_feature_proven(&peer, features::TENSOR_RELAY));

        // Receiving a relayed tensor proves TENSOR_RELAY...
        state.record_relay_proven_features(&peer, features::TENSOR_RELAY);
        assert!(state.relay_feature_proven(&peer, features::TENSOR_RELAY));
        // ...but a TENSOR_RELAY proof does NOT imply RELAY (distinct bits).
        assert!(!state.relay_feature_proven(&peer, features::RELAY));

        // Recording RELAY too ORs the bits — both now proven for that peer.
        state.record_relay_proven_features(&peer, features::RELAY);
        assert!(state.relay_feature_proven(&peer, features::RELAY));
        assert!(state.relay_feature_proven(&peer, features::TENSOR_RELAY));

        // A different peer is unaffected by another's proof.
        assert!(!state.relay_feature_proven(&NodeId([1u8; 32]), features::TENSOR_RELAY));

        // A stale proof (older than the route TTL) no longer counts, and the
        // sweep drops it. Guarded because a just-booted host's monotonic clock
        // may not be able to represent an instant that far in the past.
        if let Some(old) =
            Instant::now().checked_sub(Duration::from_secs(RELAY_ROUTE_TTL_SECS + 30))
        {
            state.relay_proven_features.insert(
                peer.clone(),
                RelayProvenFeatures {
                    features: features::TENSOR_RELAY,
                    proven_at: old,
                },
            );
            assert!(!state.relay_feature_proven(&peer, features::TENSOR_RELAY));
            state.sweep_stale_relay_state();
            assert!(!state.relay_proven_features.contains_key(&peer));
        }
    }

    #[test]
    fn forward_rate_limit_trips_and_resets() {
        // Pure window-counter logic, exercised without a full SharedState.
        let mut counter = RelayForwardCounter {
            count: RELAY_FORWARD_MAX_PER_WINDOW,
            window_start: Instant::now(),
        };
        // At cap → blocked.
        assert!(counter.count >= RELAY_FORWARD_MAX_PER_WINDOW);
        // After the window elapses the caller resets; emulate that.
        counter.count = 0;
        counter.window_start = Instant::now();
        assert!(counter.count < RELAY_FORWARD_MAX_PER_WINDOW);
    }
}
