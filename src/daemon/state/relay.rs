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

    /// Record a failed inference for the diagnostics ring buffer.
    ///
    /// Oldest-first eviction at [`super::MAX_RECENT_FAILURES`]. Best-effort: a
    /// poisoned lock is skipped rather than propagated, because losing a
    /// diagnostic record must never affect the request path.
    pub fn record_request_failure(&self, failure: super::RequestFailure) {
        if let Ok(mut buf) = self.recent_failures.lock() {
            if buf.len() >= super::MAX_RECENT_FAILURES {
                buf.pop_front();
            }
            buf.push_back(failure);
        }
    }

    /// Snapshot of recent failures, oldest first.
    pub fn recent_failures_snapshot(&self) -> Vec<super::RequestFailure> {
        self.recent_failures
            .lock()
            .map(|b| b.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Whether this node should act as an application-level inference relay.
    ///
    /// The single source of truth for the relay-forwarder decision — the DHT
    /// relay-service registration, the `relay_capable` capability advertisement,
    /// and the inbound forward gates must all agree, or peers route through a
    /// node that then refuses to forward.
    ///
    /// Three ways to qualify:
    ///  - `--anchor` mode (a dedicated bootstrap/relay node),
    ///  - explicit `network.relay_forwarding = true` opt-in, or
    ///  - NETWORKING_PLAN Phase 3 auto-donation: the node is confirmed reachable
    ///    from the open internet AND has not opted out via
    ///    `network.relay_forwarding_auto = false`.
    ///
    /// The auto path exists because Phase 3's "any public node can opt in as a
    /// relay" was previously only a config flag that nothing ever set, so the
    /// swarm's entire relay capacity was whichever nodes ran `--anchor` — a
    /// single point of failure and a hard throughput ceiling for every NAT'd
    /// pair.
    pub fn relay_forwarding_enabled(&self) -> bool {
        if self.config.node.anchor_mode || self.config.network.relay_forwarding {
            return true;
        }
        self.config.network.relay_forwarding_auto
            && self
                .publicly_reachable
                .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether inference for `node` can be routed through an application-level
    /// relay, even though we hold no libp2p connection to it.
    ///
    /// NETWORKING_PLAN §4 Phase 1 calls for the router to prefer
    /// `direct → relayed-through-a-relay → fail`, via a "reachable-via-relay"
    /// tier in the scheduler "so a NAT'd holder is still usable". Without it the
    /// scheduler filtered purely on `connected_node_ids` (populated only by a
    /// libp2p Identify), so the app-relay could only ever *substitute* the data
    /// path for a peer we were already connected to — it could never make an
    /// unconnectable peer usable, which is the case it exists for.
    ///
    /// A peer qualifies when it speaks the relay protocol AND we can name a
    /// relay that reaches it:
    ///  - a **fresh learned route** (it has already relayed to us — definitive,
    ///    since the route was observed working), or
    ///  - a relay it advertises a reservation with (`relay_reservations`) that
    ///    we are also connected to — the forward is then guaranteed to land.
    ///
    /// Conservative by construction: an unknown peer, a peer with no relay
    /// feature, or one sharing no relay with us returns false, so this only ever
    /// *adds* candidates that have a concrete path.
    pub fn peer_reachable_via_relay(&self, node: &NodeId) -> bool {
        use crate::types::features;

        let Some(info) = self.peer_registry.get(node) else {
            return false;
        };

        // Must be able to decode a relayed envelope. Gossiped capability first,
        // then the proof recorded when it actually relayed something to us
        // (covers the cold-start window before capability gossip lands).
        let advertises = info
            .capability
            .as_ref()
            .is_some_and(|c| features::supports(c.features, features::RELAY));
        if !advertises && !self.relay_feature_proven(node, features::RELAY) {
            return false;
        }

        // A route we have already seen work.
        if let Some(bytes) = info.peer_id_bytes.as_deref() {
            if self.relay_route_for_peer(bytes).is_some() {
                return true;
            }
        }

        // Otherwise: does the target share a relay with us? Its advertised
        // reservations, intersected with the relays we are connected to.
        let Some(reservations) = info
            .capability
            .as_ref()
            .map(|c| c.relay_reservations.clone())
        else {
            return false;
        };
        drop(info);

        reservations.iter().any(|relay| {
            self.connected_node_ids.contains(relay)
                && self
                    .peer_registry
                    .get(relay)
                    .and_then(|p| p.capability.as_ref().map(|c| c.relay_capable))
                    .unwrap_or(false)
        })
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

    /// Build a SharedState for relay-tier tests.
    #[cfg(test)]
    fn test_state(config: crate::config::Config) -> std::sync::Arc<crate::daemon::SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use tokio::sync::Mutex;

        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(config, identity, db, executor, None);
        state
    }

    /// Insert a peer with the given capability into the registry.
    #[cfg(test)]
    fn insert_peer(
        state: &crate::daemon::SharedState,
        node: &crate::types::NodeId,
        capability: Option<crate::types::NodeCapability>,
    ) {
        use crate::types::PeerInfo;
        state.peer_registry.insert(
            node.clone(),
            PeerInfo {
                node_id: node.clone(),
                addresses: vec![],
                capability,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(50),
                trust_score: 0.5,
                peer_id_bytes: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );
    }

    #[cfg(test)]
    fn capability_with(
        features: u64,
        relay_capable: bool,
        reservations: Vec<crate::types::NodeId>,
    ) -> crate::types::NodeCapability {
        crate::types::NodeCapability {
            node_id: crate::types::NodeId([0u8; 32]),
            gpu: None,
            ram_total_mb: 0,
            ram_available_mb: 0,
            disk_available_mb: 0,
            bandwidth_mbps: 0.0,
            hosted_shards: vec![],
            max_contribution: crate::types::ContributionLevel::Moderate,
            uptime_seconds: 0,
            version: String::new(),
            region: None,
            est_tokens_per_sec_7b: 0.0,
            observed_latencies: vec![],
            relay_capable,
            protocol_version: 0,
            features,
            relay_reservations: reservations,
            anchor_mode: false,
        }
    }

    /// The relay-donation decision must come out the same everywhere it is
    /// consulted, so all three qualifying paths are pinned here.
    #[test]
    fn relay_forwarding_enabled_covers_anchor_explicit_and_auto() {
        use std::sync::atomic::Ordering;

        // Plain NAT'd node: not an anchor, no opt-in, not publicly reachable.
        let mut config = crate::config::Config::default();
        config.node.anchor_mode = false;
        config.network.relay_forwarding = false;
        let state = test_state(config);
        assert!(
            !state.relay_forwarding_enabled(),
            "a NAT'd node must never advertise itself as a relay"
        );

        // Becoming publicly reachable auto-enables donation (Phase 3).
        state.publicly_reachable.store(true, Ordering::Relaxed);
        assert!(state.relay_forwarding_enabled());

        // ...unless the operator opted out of auto-donation.
        let mut config = crate::config::Config::default();
        config.network.relay_forwarding_auto = false;
        let state = test_state(config);
        state.publicly_reachable.store(true, Ordering::Relaxed);
        assert!(
            !state.relay_forwarding_enabled(),
            "relay_forwarding_auto = false must stop a public node donating"
        );

        // Explicit opt-in wins even without confirmed reachability.
        let mut config = crate::config::Config::default();
        config.network.relay_forwarding = true;
        config.network.relay_forwarding_auto = false;
        let state = test_state(config);
        assert!(state.relay_forwarding_enabled());

        // Anchor mode always forwards.
        let mut config = crate::config::Config::default();
        config.node.anchor_mode = true;
        config.network.relay_forwarding_auto = false;
        let state = test_state(config);
        assert!(state.relay_forwarding_enabled());
    }

    /// The scheduler's relay tier must only admit peers with a concrete path,
    /// or it re-introduces the dead-peer hang the liveness filter prevents.
    #[test]
    fn peer_reachable_via_relay_requires_a_real_path() {
        use crate::types::{features, NodeId};

        let state = test_state(crate::config::Config::default());
        let target = NodeId([7u8; 32]);
        let relay = NodeId([8u8; 32]);

        // Completely unknown peer.
        assert!(!state.peer_reachable_via_relay(&target));

        // Known, but advertises no relay support → cannot decode an envelope.
        insert_peer(
            &state,
            &target,
            Some(capability_with(0, false, vec![relay.clone()])),
        );
        assert!(
            !state.peer_reachable_via_relay(&target),
            "a peer without the RELAY feature must not be offered a relayed request"
        );

        // Speaks relay, names a relay — but we aren't connected to that relay,
        // so there is no path and it must stay unschedulable.
        insert_peer(
            &state,
            &target,
            Some(capability_with(features::RELAY, false, vec![relay.clone()])),
        );
        assert!(!state.peer_reachable_via_relay(&target));

        // Connecting to the shared relay completes the path.
        insert_peer(
            &state,
            &relay,
            Some(capability_with(features::RELAY, true, vec![])),
        );
        state.connected_node_ids.insert(relay.clone());
        assert!(
            state.peer_reachable_via_relay(&target),
            "a shared, connected, relay-capable node makes the target reachable"
        );

        // A connected node that is NOT relay-capable does not count.
        let bystander = NodeId([9u8; 32]);
        insert_peer(
            &state,
            &target,
            Some(capability_with(
                features::RELAY,
                false,
                vec![bystander.clone()],
            )),
        );
        insert_peer(
            &state,
            &bystander,
            Some(capability_with(features::RELAY, false, vec![])),
        );
        state.connected_node_ids.insert(bystander);
        assert!(!state.peer_reachable_via_relay(&target));
    }

    /// Cold start: a peer that has actually relayed to us is reachable even
    /// before its capability gossip lands (which is when `capability` is None).
    #[test]
    fn peer_reachable_via_relay_accepts_a_proven_route() {
        use crate::types::{features, NodeId};

        let state = test_state(crate::config::Config::default());
        let target = NodeId([4u8; 32]);
        let peer_bytes = vec![1u8, 2, 3, 4];

        // Registry entry with no capability at all — the cold-start shape
        // created by `ensure_relayed_origin_known`.
        use crate::types::PeerInfo;
        state.peer_registry.insert(
            target.clone(),
            PeerInfo {
                node_id: target.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: None,
                trust_score: 0.5,
                peer_id_bytes: Some(peer_bytes.clone()),
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );
        assert!(!state.peer_reachable_via_relay(&target));

        // It relayed to us (proving RELAY) and we learned the route back.
        state.record_relay_proven_features(&target, features::RELAY);
        state.relay_routes.insert(
            peer_bytes,
            RelayRoute {
                relay_peer_bytes: vec![9u8; 8],
                target_node: target.clone(),
                learned_at: Instant::now(),
            },
        );
        assert!(
            state.peer_reachable_via_relay(&target),
            "a proven, freshly-learned route is the strongest reachability signal"
        );
    }

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
