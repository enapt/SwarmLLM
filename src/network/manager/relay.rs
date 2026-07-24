//! NETWORKING_PLAN Phase 1 — application-level inference relay.
//!
//! Two responsibilities live here:
//!  - **Outbound** (`try_relay_send`): when a directed inference message can't
//!    reach its target directly (both NAT'd, no hole-punch), wrap it in a
//!    `RelayedEnvelope` sealed end-to-end for the target and send it to a
//!    mutually-reachable relay peer instead.
//!  - **Inbound** (`handle_relayed_envelope`): either we are the relay (forward
//!    to the target) or the final recipient (open + inject the inner message
//!    into the normal dispatch feed as though the origin sent it directly).
//!
//! The relay never sees plaintext — the inner message is ephemeral-sealed for
//! `relay_to`'s X25519 key (see `crypto::relay_seal`). Forwarding is bounded to
//! a single hop: a relay only forwards an envelope whose transport-authenticated
//! sender equals its `origin`, which blocks loops and traffic amplification.

use libp2p::PeerId;

use crate::network::protocol::SwarmRequest;
use crate::types::{NodeId, PeerInfo, RelayedEnvelope, SwarmMessage};

use super::NetworkManager;

impl NetworkManager {
    /// Whether a `SwarmMessage` is eligible to be routed through a relay. Only
    /// the remote-generate inference fast path — the prompt request, streamed
    /// tokens, and its cancel — qualifies. TP AllReduce and bulk tensor/shard
    /// transfers are never relayed (LAN-local or their own transport).
    fn is_relay_eligible(msg: &SwarmMessage) -> bool {
        matches!(
            msg,
            SwarmMessage::RemoteGenerateRequest(_)
                | SwarmMessage::StreamingToken(_)
                | SwarmMessage::CancelInference(_)
        )
    }

    /// Correlation id bound into the envelope AAD + used for logging.
    fn relay_correlation_id(msg: &SwarmMessage) -> uuid::Uuid {
        match msg {
            SwarmMessage::RemoteGenerateRequest(r) => r.request_id,
            SwarmMessage::StreamingToken(t) => t.request_id,
            SwarmMessage::CancelInference(c) => c.request_id,
            _ => uuid::Uuid::new_v4(),
        }
    }

    /// Reverse-resolve a target's NodeId from its peer-id bytes via the
    /// persistent `peer_registry` (which survives disconnects, unlike
    /// `peer_to_node`). O(peers) — only hit on the first message to a target,
    /// before a relay route is learned.
    fn node_id_for_peer_bytes(&self, target_peer_bytes: &[u8]) -> Option<NodeId> {
        self.shared_state
            .peer_registry
            .iter()
            .find(|e| {
                e.value()
                    .peer_id_bytes
                    .as_deref()
                    .is_some_and(|b| b == target_peer_bytes)
            })
            .map(|e| e.key().clone())
    }

    /// Pick a connected, relay-capable peer to route toward `target` through.
    /// Skips the target itself and the local node. Returns the relay's peer-id
    /// bytes.
    fn pick_connected_relay(&self, target: &NodeId) -> Option<Vec<u8>> {
        let local = self.shared_state.identity.node_id();
        for peer_id in self.swarm.connected_peers() {
            let Some(node) = self.peer_to_node.get(peer_id).map(|r| r.clone()) else {
                continue;
            };
            if &node == target || node == *local {
                continue;
            }
            let is_relay = self
                .shared_state
                .peer_registry
                .get(&node)
                .and_then(|p| p.capability.as_ref().map(|c| c.relay_capable))
                .unwrap_or(false);
            if is_relay {
                return Some(peer_id.to_bytes());
            }
        }
        None
    }

    /// Whether `target` advertises the RELAY protocol feature (can receive a
    /// relayed envelope). Guards backward-compat: never wrap traffic for a peer
    /// that hasn't negotiated the feature — an older node advertises no features
    /// (0) and is correctly skipped.
    fn target_supports_relay(&self, target: &NodeId) -> bool {
        self.shared_state
            .peer_registry
            .get(target)
            .and_then(|p| {
                p.capability.as_ref().map(|c| {
                    crate::types::features::supports(c.features, crate::types::features::RELAY)
                })
            })
            .unwrap_or(false)
    }

    /// Whether we hold at least one DIRECT (non-relay-circuit) connection to a
    /// peer. A peer reachable only via a relay circuit can't reliably round-trip
    /// request_response, so the relay send path prefers the app-level relay for
    /// it (NETWORKING_PLAN Phase 1).
    pub(super) fn has_direct_connection(&self, peer: &PeerId) -> bool {
        self.peer_direct_conns
            .get(peer)
            .is_some_and(|s| !s.is_empty())
    }

    /// Try to deliver `msg` to a target we can't reliably reach directly by
    /// routing it through a relay peer. Takes `&msg` so the caller keeps
    /// ownership and can fall through to a best-effort direct send if no relay
    /// path exists. Returns true if an envelope was dispatched to a relay.
    pub(super) fn try_relay_send(&mut self, target_peer_bytes: &[u8], msg: &SwarmMessage) -> bool {
        if !Self::is_relay_eligible(msg) {
            return false;
        }
        let local_node = self.shared_state.identity.node_id().clone();

        // Resolve the relay peer + the target NodeId (needed to seal e2e).
        let (relay_peer_bytes, target_node) =
            match self.shared_state.relay_route_for_peer(target_peer_bytes) {
                Some(route) => (route.relay_peer_bytes, route.target_node),
                None => {
                    // No learned route (first message to this target). Proactively
                    // pick any connected relay-capable peer.
                    let Some(target_node) = self.node_id_for_peer_bytes(target_peer_bytes) else {
                        return false;
                    };
                    if !self.target_supports_relay(&target_node) {
                        return false;
                    }
                    let Some(relay_peer_bytes) = self.pick_connected_relay(&target_node) else {
                        return false;
                    };
                    (relay_peer_bytes, target_node)
                }
            };

        let Some(relay_peer_id) = Self::resolve_peer_id(&relay_peer_bytes, "relay") else {
            return false;
        };
        if !self.swarm.is_connected(&relay_peer_id) {
            // Learned route went stale — forget it so we stop trying.
            self.shared_state.relay_routes.remove(target_peer_bytes);
            return false;
        }

        let request_id = Self::relay_correlation_id(msg);
        let env = match crate::crypto::relay_seal::seal_relayed_message(
            local_node,
            target_node.clone(),
            request_id,
            msg,
        ) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(error = %e, "relay seal failed — dropping message");
                return false;
            }
        };
        let req = SwarmRequest::Message(Box::new(SwarmMessage::RelayedEnvelope(env)));
        let req_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&relay_peer_id, req);
        self.pending_rr_observability.insert(
            req_id,
            ("relayed".to_string(), std::time::Instant::now(), None),
        );
        tracing::debug!(
            %relay_peer_id,
            target = %target_node,
            %request_id,
            "NETWORKING_PLAN: routed inference message via relay"
        );
        true
    }

    /// Handle an inbound `RelayedEnvelope`. Either we are the final recipient
    /// (open + inject the inner message) or the relay (forward to the target).
    /// Called inline from `handle_request`, which sends the ACK.
    pub(super) fn handle_relayed_envelope(&mut self, immediate_peer: PeerId, env: RelayedEnvelope) {
        let local = self.shared_state.identity.node_id().clone();

        if env.relay_to == local {
            // We are the target. Open the sealed inner message and dispatch it
            // as though `origin` sent it directly.
            let local_secret = crate::crypto::session::ed25519_to_x25519_secret(
                &self.shared_state.identity.signing_key_bytes(),
            );
            let inner = match crate::crypto::relay_seal::open_relayed_message(&local_secret, &env) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(error = %e, origin = %env.origin, "failed to open relayed envelope — dropping");
                    return;
                }
            };
            // Single-hop invariant: a relayed envelope must never itself carry
            // another envelope.
            if matches!(inner, SwarmMessage::RelayedEnvelope(_)) {
                tracing::warn!(origin = %env.origin, "nested relayed envelope — dropping");
                return;
            }
            // Register the origin as a known peer (reachable via relay) so the
            // dispatch security gates pass and the return path can stamp its
            // peer bytes — then learn the reverse route so replies flow back the
            // same way.
            self.ensure_relayed_origin_known(&env.origin);
            self.shared_state
                .learn_relay_route(&env.origin, immediate_peer.to_bytes());
            if let Err(e) = self.dispatch_authenticated_as(Some(env.origin.clone()), inner) {
                tracing::warn!(error = %e, "relayed inner message dropped — dispatch backpressured");
            }
            return;
        }

        // We are the relay. Forward to the target if we can and are willing.
        let relay_enabled = self.shared_state.config.node.anchor_mode
            || self.shared_state.config.network.relay_forwarding;
        if !relay_enabled {
            return;
        }
        // Single-hop invariant: only forward an envelope whose immediate sender
        // is its own origin. Bounds relaying to exactly one hop; blocks loops
        // and traffic amplification.
        let Some(sender_node) = self.peer_to_node.get(&immediate_peer).map(|r| r.clone()) else {
            return;
        };
        if sender_node != env.origin {
            tracing::debug!(origin = %env.origin, sender = %sender_node, "refusing to re-relay a relayed envelope (not first hop)");
            return;
        }
        if !self
            .shared_state
            .relay_forward_allowed(&immediate_peer.to_bytes())
        {
            tracing::debug!(origin = %env.origin, "relay forward rate limit — dropping");
            return;
        }
        // Only forward to a DIRECTLY connected target (star-topology through
        // this relay). No multi-hop.
        let Some(target_peer_id) = crate::network::transport::node_id_to_peer_id(&env.relay_to)
        else {
            return;
        };
        if !self.swarm.is_connected(&target_peer_id) {
            tracing::debug!(target = %env.relay_to, "relay: target not connected — cannot forward");
            return;
        }
        let req = SwarmRequest::Message(Box::new(SwarmMessage::RelayedEnvelope(env)));
        let req_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&target_peer_id, req);
        self.pending_rr_observability.insert(
            req_id,
            ("relay_forward".to_string(), std::time::Instant::now(), None),
        );
    }

    /// Register a relayed message's `origin` as a known peer reachable via
    /// relay. Populates `peer_id_map` (so dispatch can stamp `sender_peer_bytes`
    /// on a relayed `RemoteGenerateRequest`) and a minimal `peer_registry` entry
    /// (so the "known peer" security gates pass). We ARE relaying inference with
    /// this peer, so it is legitimately known — via the relay, not a direct
    /// link. A richer existing entry is never clobbered.
    fn ensure_relayed_origin_known(&self, origin: &NodeId) {
        let Some(peer_id) = crate::network::transport::node_id_to_peer_id(origin) else {
            return;
        };
        let peer_bytes = peer_id.to_bytes();
        self.shared_state
            .peer_id_map
            .insert(origin.clone(), peer_bytes.clone());
        if !self.shared_state.peer_registry.contains_key(origin) {
            let now_ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            self.shared_state.peer_registry.insert(
                origin.clone(),
                PeerInfo {
                    node_id: origin.clone(),
                    addresses: vec![],
                    capability: None,
                    last_seen: chrono::Utc::now(),
                    latency_ms: None,
                    trust_score: 0.5,
                    peer_id_bytes: Some(peer_bytes),
                    active_request_count: 0,
                    first_seen: now_ts,
                    verified_transaction_count: 0,
                    is_lan_peer: false,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::types::features;

    #[test]
    fn relay_feature_negotiation() {
        // A node advertising the full feature set (this build) supports relay.
        assert!(features::supports(features::ALL, features::RELAY));
        // An older node advertising no features (serde default 0) does not —
        // so we never wrap traffic it can't decode.
        assert!(!features::supports(0, features::RELAY));
    }
}
