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
//!
//! The TARGET cannot see who handed the envelope to the relay, and the seal is
//! anonymous, so the origin is proven by its signature inside the seal
//! (`features::RELAY_SIGNED`, FUTURE_WORK #298): an unsigned body, a replayed
//! one, or one carrying a message no node relays is dropped before anything
//! is learned from it.

use libp2p::PeerId;

use crate::network::protocol::SwarmRequest;
use crate::types::{NodeId, PeerInfo, RelayedEnvelope, SwarmMessage};

use super::NetworkManager;

/// Why an unsigned relayed body is refused, for the log.
const UNSIGNED: &str =
    "it is unsigned, so its origin cannot be proven (a sender older than RELAY_SIGNED, or a forgery)";

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
                | SwarmMessage::ResendTokens(_)
                | SwarmMessage::RequesterGuesses(_)
        )
    }

    /// Correlation id bound into the envelope AAD + used for logging.
    fn relay_correlation_id(msg: &SwarmMessage) -> uuid::Uuid {
        match msg {
            SwarmMessage::RemoteGenerateRequest(r) => r.request_id,
            SwarmMessage::StreamingToken(t) => t.request_id,
            SwarmMessage::CancelInference(c) => c.request_id,
            SwarmMessage::ResendTokens(r) => r.request_id,
            SwarmMessage::RequesterGuesses(g) => g.request_id,
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

    /// Ranked list of connected relay-capable peers to route toward `target`
    /// through (NETWORKING_PLAN Phase 3). A relay the target ALSO advertises
    /// being connected to (`relay_reservations`) comes first — the forward is
    /// then guaranteed to land at the target — followed by any other connected
    /// relay as a fallback. Skips the target itself and the local node.
    fn pick_connected_relays(&self, target: &NodeId) -> Vec<Vec<u8>> {
        let local = self.shared_state.identity.node_id();
        let target_reservations: Vec<NodeId> = self
            .shared_state
            .peer_registry
            .get(target)
            .and_then(|p| p.capability.as_ref().map(|c| c.relay_reservations.clone()))
            .unwrap_or_default();
        let mut preferred: Vec<Vec<u8>> = Vec::new();
        let mut fallback: Vec<Vec<u8>> = Vec::new();
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
            if !is_relay {
                continue;
            }
            if target_reservations.contains(&node) {
                preferred.push(peer_id.to_bytes());
            } else {
                fallback.push(peer_id.to_bytes());
            }
        }
        preferred.extend(fallback);
        preferred
    }

    /// Whether `target` speaks a relay `feature`: `RELAY` (can receive a
    /// relayed envelope), `TENSOR_RELAY` (a relayed pipeline tensor — distinct,
    /// and a learned route does NOT prove it) or `RELAY_SIGNED` (reads a body
    /// signed by its origin). Guards backward-compat: never send a peer what it
    /// has not negotiated — an older node advertises no features (0) and is
    /// correctly skipped.
    fn target_speaks(&self, target: &NodeId, feature: u64) -> bool {
        // Direct proof first: a peer that just relayed a PROVEN body to us
        // demonstrably speaks what it used, even if its capability gossip
        // hasn't populated our registry entry yet. This is the cold-start
        // return-path fix — otherwise a reply or computed result can't be
        // relayed back to an origin known only via `ensure_relayed_origin_known`
        // (capability None) until a capability-gossip round lands.
        if self.shared_state.relay_feature_proven(target, feature) {
            return true;
        }
        self.shared_state
            .peer_registry
            .get(target)
            .and_then(|p| {
                p.capability
                    .as_ref()
                    .map(|c| crate::types::features::supports(c.features, feature))
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

    /// Whether this node forwards relay traffic (so it should register itself as
    /// a DHT relay-service provider). NETWORKING_PLAN Phase 3.
    pub(super) fn is_relay_forwarder(&self) -> bool {
        self.shared_state.relay_forwarding_enabled()
    }

    /// Count connected relay-capable peers — the redundancy that decides whether
    /// to seek more relays from the DHT (NETWORKING_PLAN Phase 3).
    pub(super) fn count_connected_relays(&self) -> usize {
        let local = self.shared_state.identity.node_id();
        self.swarm
            .connected_peers()
            .filter(|pid| {
                self.peer_to_node
                    .get(pid)
                    .map(|n| n.clone())
                    .is_some_and(|node| {
                        &node != local
                            && self
                                .shared_state
                                .peer_registry
                                .get(&node)
                                .and_then(|p| p.capability.as_ref().map(|c| c.relay_capable))
                                .unwrap_or(false)
                    })
            })
            .count()
    }

    /// Handle relay-provider DHT results (NETWORKING_PLAN Phase 3): dial any
    /// discovered relay peer we aren't already connected to, so we keep multiple
    /// relay paths available and survive the loss of the bootstrap anchor.
    /// Best-effort — Kademlia supplies the addresses it learned during the query.
    pub(super) fn handle_relay_providers_found(
        &mut self,
        providers: &std::collections::HashSet<PeerId>,
    ) {
        let local = *self.swarm.local_peer_id();
        let mut dialed = 0;
        for peer_id in providers {
            if peer_id == &local || self.swarm.is_connected(peer_id) {
                continue;
            }
            if self.dial_checked(*peer_id, "relay_providers").is_ok() {
                dialed += 1;
            }
        }
        if dialed > 0 {
            tracing::info!(
                dialed,
                "NETWORKING_PLAN: dialing DHT-discovered relay(s) for redundancy"
            );
        }
    }

    /// Try to deliver `msg` to a target we can't reliably reach directly by
    /// routing it through a relay peer. Takes `&msg` so the caller keeps
    /// ownership and can fall through to a best-effort direct send if no relay
    /// path exists. Returns true if an envelope was dispatched to a relay.
    pub(super) fn try_relay_send(&mut self, target_peer_bytes: &[u8], msg: &SwarmMessage) -> bool {
        if !Self::is_relay_eligible(msg) {
            return false;
        }

        // Resolve the target NodeId (needed to seal e2e): from a learned route
        // if we have one (proven to speak relay), else reverse-lookup +
        // feature-gate so we never wrap traffic an older peer can't decode.
        let learned = self.shared_state.relay_route_for_peer(target_peer_bytes);
        let target_node = match &learned {
            Some(route) => route.target_node.clone(),
            None => {
                let Some(tn) = self.node_id_for_peer_bytes(target_peer_bytes) else {
                    return false;
                };
                if !self.target_speaks(&tn, crate::types::features::RELAY) {
                    return false;
                }
                tn
            }
        };
        // Only to a target that reads signed bodies (#298): an older one cannot
        // parse one, and its reply would come back unsigned and be dropped
        // here after it had done the work (security review). Refused here,
        // the send falls through to the direct path at once.
        if !self.target_speaks(&target_node, crate::types::features::RELAY_SIGNED) {
            return false;
        }

        // Candidate relays: the learned route first (known to reach the target),
        // then fresh connected relays (mutually-reachable first) as failover if
        // the learned relay has since dropped.
        let mut candidates: Vec<Vec<u8>> = Vec::new();
        if let Some(route) = &learned {
            candidates.push(route.relay_peer_bytes.clone());
        }
        for r in self.pick_connected_relays(&target_node) {
            if !candidates.contains(&r) {
                candidates.push(r);
            }
        }

        let request_id = Self::relay_correlation_id(msg);
        for relay_peer_bytes in candidates {
            let Some(relay_peer_id) = Self::resolve_peer_id(&relay_peer_bytes, "relay") else {
                continue;
            };
            if !self.swarm.is_connected(&relay_peer_id) {
                continue;
            }
            let env = match crate::crypto::relay_seal::seal_relayed_message(
                &self.shared_state.identity,
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
            self.pending_rr_observability
                .insert(req_id, super::PendingRrSend::new("relayed"));
            tracing::debug!(
                %relay_peer_id,
                target = %target_node,
                %request_id,
                "NETWORKING_PLAN: routed inference message via relay"
            );
            return true;
        }

        // No candidate relay is connected. A stale learned route is now known
        // dead — forget it so we stop trying that relay.
        if learned.is_some() {
            self.shared_state.relay_routes.remove(target_peer_bytes);
        }
        false
    }

    /// Try to deliver an already-encoded tensor frame (`encoded` = the plaintext
    /// wire bytes of a `LayerForward` when `is_result=false`, or a `LayerResult`
    /// when `is_result=true`) to `target_peer_bytes` through a relay. The tensor
    /// is ephemeral-sealed for the target's static key, so the relay stays blind
    /// and the target opens it without a session. Returns true if dispatched to
    /// a relay; the caller then skips its direct send. NETWORKING_PLAN.
    pub(super) fn try_relay_tensor(
        &mut self,
        target_peer_bytes: &[u8],
        encoded: &[u8],
        is_result: bool,
        request_id: uuid::Uuid,
    ) -> bool {
        let local_node = self.shared_state.identity.node_id().clone();
        let learned = self.shared_state.relay_route_for_peer(target_peer_bytes);
        let target_node = match &learned {
            Some(route) => route.target_node.clone(),
            None => match self.node_id_for_peer_bytes(target_peer_bytes) {
                Some(tn) => tn,
                None => return false,
            },
        };
        // TENSOR_RELAY is a distinct capability from RELAY, and a learned route
        // (which may have come from a message relay) does not prove it — always
        // gate so we never send a tensor an older peer can't decode — nor one
        // whose result would come back unsigned (`RELAY_SIGNED`, #298).
        if !self.target_speaks(
            &target_node,
            crate::types::features::TENSOR_RELAY | crate::types::features::RELAY_SIGNED,
        ) {
            return false;
        }

        let mut candidates: Vec<Vec<u8>> = Vec::new();
        if let Some(route) = &learned {
            candidates.push(route.relay_peer_bytes.clone());
        }
        for r in self.pick_connected_relays(&target_node) {
            if !candidates.contains(&r) {
                candidates.push(r);
            }
        }

        for relay_peer_bytes in candidates {
            let Some(relay_peer_id) = Self::resolve_peer_id(&relay_peer_bytes, "relay") else {
                continue;
            };
            if !self.swarm.is_connected(&relay_peer_id) {
                continue;
            }
            let (ephemeral_pub, sealed) = match crate::crypto::relay_seal::seal_relayed_tensor(
                &self.shared_state.identity,
                &target_node,
                &request_id,
                crate::crypto::relay_seal::RelayedKind::of_tensor(is_result),
                encoded,
            ) {
                Ok(x) => x,
                Err(e) => {
                    tracing::warn!(error = %e, "relay tensor seal failed — dropping");
                    return false;
                }
            };
            let rt = crate::network::protocol::RelayedTensor {
                relay_to: target_node.clone(),
                origin: local_node.clone(),
                request_id,
                is_result,
                ephemeral_pub,
                sealed,
            };
            let req = SwarmRequest::RelayedTensor(rt);
            let req_id = self
                .swarm
                .behaviour_mut()
                .request_response
                .send_request(&relay_peer_id, req);
            self.pending_rr_observability
                .insert(req_id, super::PendingRrSend::new("relayed_tensor"));
            tracing::debug!(
                %relay_peer_id,
                target = %target_node,
                %request_id,
                is_result,
                "NETWORKING_PLAN: routed tensor via relay"
            );
            return true;
        }

        if learned.is_some() {
            self.shared_state.relay_routes.remove(target_peer_bytes);
        }
        false
    }

    /// Handle an inbound `RelayedEnvelope`. Either we are the final recipient
    /// (open + inject the inner message) or the relay (forward to the target).
    /// Called inline from `handle_request`, which sends the ACK.
    pub(super) fn handle_relayed_envelope(&mut self, immediate_peer: PeerId, env: RelayedEnvelope) {
        let local = self.shared_state.identity.node_id().clone();

        if env.relay_to == local {
            // We are the target. Open the sealed inner message and dispatch it
            // as though `origin` sent it directly — only once `origin` is
            // PROVEN to have sealed it (its signature, #298), the body is not a
            // replay, and it is one of the messages a node ever relays.
            let local_secret = self.shared_state.identity.x25519_secret();
            let (inner, sent_at_ms, digest) =
                match crate::crypto::relay_seal::open_relayed_message(&local_secret, &env) {
                    Ok(crate::crypto::relay_seal::Opened::Proven {
                        inner,
                        sent_at_ms,
                        digest,
                    }) => (inner, sent_at_ms, digest),
                    Ok(crate::crypto::relay_seal::Opened::Unsigned) => {
                        self.refuse_relayed(immediate_peer, &env.origin, "envelope", &UNSIGNED);
                        return;
                    }
                    Err(e) => {
                        let why = format!("it did not open: {e}");
                        self.refuse_relayed(immediate_peer, &env.origin, "envelope", &why);
                        return;
                    }
                };
            // Only what `try_relay_send` relays — never, say, a holder update
            // or a nested envelope (the single-hop invariant) — even from a
            // proven origin.
            if !Self::is_relay_eligible(&inner) {
                let why = format!("it carries a {}, which no node relays", inner.kind_name());
                self.refuse_relayed(immediate_peer, &env.origin, "envelope", &why);
                return;
            }
            if let Err(why) = self.admit_relayed(&env.origin, &immediate_peer, digest, sent_at_ms) {
                self.refuse_relayed(immediate_peer, &env.origin, "envelope", &why);
                return;
            }
            self.take_proven_relayed_origin(
                &env.origin,
                immediate_peer,
                crate::types::features::RELAY,
            );
            if let Err(e) = self.dispatch_authenticated_as(Some(env.origin.clone()), inner) {
                tracing::warn!(error = %e, "relayed inner message dropped — dispatch backpressured");
            }
            return;
        }

        // We are the relay. Forward to the target if we can and are willing.
        let relay_enabled = self.shared_state.relay_forwarding_enabled();
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
        // NETWORKING_PLAN Phase 3 — meter forwarded bytes so relaying earns
        // credit at the seeding rate (informational/priority; see
        // `earn_relay_forwarding`).
        self.shared_state.relay_inference_bytes.fetch_add(
            env.sealed.len() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let req = SwarmRequest::Message(Box::new(SwarmMessage::RelayedEnvelope(env)));
        let req_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&target_peer_id, req);
        self.pending_rr_observability
            .insert(req_id, super::PendingRrSend::new("relay_forward"));
    }

    /// Handle an inbound `RelayedTensor` (NETWORKING_PLAN tensor relay). Either
    /// we are the final recipient — open the ephemeral-sealed tensor with our
    /// static key (no session needed), decode it, and inject it into dispatch as
    /// though `origin` sent it directly — or we are the relay and forward it to
    /// the target. Called inline from `handle_request`, which sends the ACK.
    pub(super) fn handle_relayed_tensor(
        &mut self,
        immediate_peer: PeerId,
        rt: crate::network::protocol::RelayedTensor,
    ) {
        let local = self.shared_state.identity.node_id().clone();

        if rt.relay_to == local {
            // We are the target. Open with our static key; dispatch only a
            // body its origin is PROVEN to have sealed (#298), never twice.
            let secret = self.shared_state.identity.x25519_secret();
            let (plaintext, sent_at_ms, digest) =
                match crate::crypto::relay_seal::open_relayed_tensor(
                    &secret,
                    &rt.origin,
                    &rt.relay_to,
                    &rt.request_id,
                    crate::crypto::relay_seal::RelayedKind::of_tensor(rt.is_result),
                    &rt.ephemeral_pub,
                    &rt.sealed,
                ) {
                    Ok(crate::crypto::relay_seal::Opened::Proven {
                        inner,
                        sent_at_ms,
                        digest,
                    }) => (inner, sent_at_ms, digest),
                    Ok(crate::crypto::relay_seal::Opened::Unsigned) => {
                        self.refuse_relayed(immediate_peer, &rt.origin, "tensor", &UNSIGNED);
                        return;
                    }
                    Err(e) => {
                        let why = format!("it did not open: {e}");
                        self.refuse_relayed(immediate_peer, &rt.origin, "tensor", &why);
                        return;
                    }
                };
            if let Err(why) = self.admit_relayed(&rt.origin, &immediate_peer, digest, sent_at_ms) {
                self.refuse_relayed(immediate_peer, &rt.origin, "tensor", &why);
                return;
            }
            self.take_proven_relayed_origin(
                &rt.origin,
                immediate_peer,
                crate::types::features::TENSOR_RELAY,
            );
            // A result goes through the one intake for results, so a chain
            // hop passes a relayed next hop's failure on too (`hop_forwards`).
            if rt.is_result {
                let r = match crate::network::protocol::decode_layer_result(&plaintext) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(error = %e, "relayed LayerResult decode failed");
                        return;
                    }
                };
                if let Err(e) = self.dispatch_layer_result(Some(rt.origin.clone()), r) {
                    tracing::warn!(error = %e, "relayed tensor dropped — dispatch backpressured");
                }
                return;
            }
            let msg = {
                match crate::network::protocol::decode_layer_forward(&plaintext) {
                    Ok(mut f) => {
                        // Stamp the origin's peer bytes so `handle_layer_forward`
                        // routes the computed result back to the origin — where
                        // the return relay picks it up. (The origin builds the
                        // forward with `None`; the direct receive path stamps
                        // the RR sender, so the relay path must stamp the origin.)
                        if let Some(pid) = crate::network::transport::node_id_to_peer_id(&rt.origin)
                        {
                            f.sender_peer_bytes = Some(pid.to_bytes());
                        }
                        SwarmMessage::LayerForward(f)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "relayed LayerForward decode failed");
                        return;
                    }
                }
            };
            if let Err(e) = self.dispatch_authenticated_as(Some(rt.origin.clone()), msg) {
                tracing::warn!(error = %e, "relayed tensor dropped — dispatch backpressured");
            }
            return;
        }

        // We are the relay. Forward to the target if willing + able (single hop).
        let relay_enabled = self.shared_state.relay_forwarding_enabled();
        if !relay_enabled {
            return;
        }
        let Some(sender_node) = self.peer_to_node.get(&immediate_peer).map(|r| r.clone()) else {
            return;
        };
        if sender_node != rt.origin {
            tracing::debug!(origin = %rt.origin, sender = %sender_node, "refusing to re-relay a relayed tensor (not first hop)");
            return;
        }
        if !self
            .shared_state
            .relay_forward_allowed(&immediate_peer.to_bytes())
        {
            tracing::debug!(origin = %rt.origin, "relay tensor forward rate limit — dropping");
            return;
        }
        let Some(target_peer_id) = crate::network::transport::node_id_to_peer_id(&rt.relay_to)
        else {
            return;
        };
        if !self.swarm.is_connected(&target_peer_id) {
            tracing::debug!(target = %rt.relay_to, "relay: tensor target not connected — cannot forward");
            return;
        }
        self.shared_state
            .relay_inference_bytes
            .fetch_add(rt.sealed.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let req = SwarmRequest::RelayedTensor(rt);
        let req_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&target_peer_id, req);
        self.pending_rr_observability
            .insert(req_id, super::PendingRrSend::new("relay_tensor_forward"));
    }

    /// Say why a relayed body addressed here was refused — once per window
    /// per delivering peer and kind. `claimed_origin` is only what the body
    /// CLAIMS; the peer that handed it over is the one fact a forger cannot
    /// choose, and a forger can send refusals as fast as it likes (security
    /// review of #298).
    fn refuse_relayed(
        &mut self,
        immediate_peer: PeerId,
        claimed_origin: &NodeId,
        what: &'static str,
        why: &dyn std::fmt::Display,
    ) {
        if let super::PeerFailureLog::Emit { suppressed } =
            self.observe_repeated_peer_failure(immediate_peer, &format!("refused relayed {what}"))
        {
            tracing::warn!(
                relayed_by = %immediate_peer,
                %claimed_origin,
                suppressed,
                "refusing a relayed {what}: {why}"
            );
        }
    }

    /// Take a proven relayed body into the replay memory, counted against the
    /// network the peer that delivered it reaches this node from — the one
    /// admission both relayed kinds go through (#299: counted per peer id,
    /// eight minted key pairs filled the memory). A peer whose connection is
    /// not on record counts with every unaddressed one, never as a network of
    /// its own.
    fn admit_relayed(
        &mut self,
        origin: &NodeId,
        delivered_by: &PeerId,
        digest: [u8; 32],
        sent_at_ms: u64,
    ) -> Result<(), crate::crypto::relay_seal::Refused> {
        let network = self
            .peer_remote_addrs
            .get(delivered_by)
            .map(|remote| remote.group)
            .unwrap_or(crate::network::helpers::NetGroup::UNADDRESSED);
        self.relay_replay
            .admit(origin, &network, digest, sent_at_ms)
    }

    /// What a relayed body PROVEN to come from `origin` teaches, and only such
    /// a body (#298 — an unproven one taught a forger's choice of route,
    /// registry entry and features for any NodeId): the origin is a known peer
    /// reachable via relay, so the dispatch security gates pass and the return
    /// path can stamp its peer bytes; replies flow back the way it came
    /// (`immediate_peer`); and it demonstrably speaks `relay_feature` and
    /// `RELAY_SIGNED`, so a reply is never refused by the feature gate — nor
    /// sent unsigned — before its capability gossip arrives.
    fn take_proven_relayed_origin(
        &self,
        origin: &NodeId,
        immediate_peer: PeerId,
        relay_feature: u64,
    ) {
        self.ensure_relayed_origin_known(origin);
        self.shared_state
            .learn_relay_route(origin, immediate_peer.to_bytes());
        self.shared_state.record_relay_proven_features(
            origin,
            relay_feature | crate::types::features::RELAY_SIGNED,
        );
    }

    /// Register a relayed message's `origin` as a known peer reachable via
    /// relay. Populates `peer_id_map` (so dispatch can stamp `sender_peer_bytes`
    /// on a relayed `RemoteGenerateRequest`) and a minimal `peer_registry` entry
    /// (so the "known peer" security gates pass). We ARE relaying inference with
    /// this peer, so it is legitimately known — via the relay, not a direct
    /// link. A richer existing entry is never clobbered.
    ///
    /// Both maps keep the identify path's caps (#299 — an origin costs nothing
    /// to mint, so one connection could grow them until the stale-peer sweep):
    /// `peer_id_map` through `remember_peer_id`, and the registry holds at most
    /// `MAX_UNCONNECTED_REGISTERED` nodes this node is not connected to, the
    /// one unseen longest making room. A connected peer is never displaced for
    /// one. Every proven body marks its origin seen, so an origin still
    /// sending is never the one unseen longest: minted origins, each used and
    /// dropped, make room among themselves (security review of #299 — an
    /// entry evicted between its body and dispatch is "an unknown peer").
    fn ensure_relayed_origin_known(&self, origin: &NodeId) {
        let Some(peer_id) = crate::network::transport::node_id_to_peer_id(origin) else {
            return;
        };
        let peer_bytes = peer_id.to_bytes();
        self.remember_peer_id(origin, peer_bytes.clone());
        let seen_again = match self.shared_state.peer_registry.get_mut(origin) {
            Some(mut entry) => {
                entry.last_seen = chrono::Utc::now();
                true
            }
            None => false,
        };
        if !seen_again {
            let (unconnected, oldest) = self.unconnected_registry_entries(origin);
            if unconnected >= super::identify::MAX_UNCONNECTED_REGISTERED {
                if let Some(oldest) = oldest {
                    self.shared_state.peer_registry.remove(&oldest);
                }
            }
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
                    ack_srtt_ms: None,
                    active_request_count: 0,
                    first_seen: now_ts,
                    verified_transaction_count: 0,
                    is_lan_peer: false,
                    goodput_bytes_per_sec: None,
                    goodput_samples: 0,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::{mpsc, watch};

    use super::*;
    use crate::crypto::relay_seal::{
        seal_relayed_message, seal_relayed_tensor, seal_unsigned_message, seal_unsigned_tensor,
        RelayedKind,
    };
    use crate::identity::Identity;
    use crate::types::{features, AuthenticatedMessage, CancelInference};

    #[test]
    fn relay_feature_negotiation() {
        // A node advertising the full feature set (this build) supports relay.
        assert!(features::supports(features::ALL, features::RELAY));
        // An older node advertising no features (serde default 0) does not —
        // so we never wrap traffic it can't decode.
        assert!(!features::supports(0, features::RELAY));
    }

    /// A manager that is the TARGET of relayed bodies, and the dispatch feed
    /// it hands them to. Never run: the handlers are called directly.
    struct Target {
        manager: NetworkManager,
        dispatched: mpsc::Receiver<AuthenticatedMessage>,
        _keep: (
            tempfile::TempDir,
            mpsc::Sender<crate::types::NetworkCommand>,
            watch::Sender<bool>,
            mpsc::Sender<crate::types::ModelId>,
        ),
    }

    impl Target {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let db = crate::storage::db::Database::open(dir.path()).unwrap();
            let config = crate::config::Config::default();
            let (state, _, _) = crate::daemon::SharedState::new(
                config.clone(),
                Identity::generate(),
                db,
                Arc::new(tokio::sync::Mutex::new(
                    crate::inference::executor::ModelExecutor::new(),
                )),
                None,
            );
            let (command_tx, command_rx) = mpsc::channel(8);
            let (out_tx, dispatched) = mpsc::channel(8);
            let (shutdown_tx, shutdown_rx) = watch::channel(false);
            let (dht_tx, dht_rx) = mpsc::channel(8);
            let manager = NetworkManager::new(
                state.clone(),
                &state.identity,
                &config,
                command_rx,
                command_tx.clone(),
                out_tx,
                shutdown_rx,
                None,
                dht_rx,
            )
            .unwrap();
            Self {
                manager,
                dispatched,
                _keep: (dir, command_tx, shutdown_tx, dht_tx),
            }
        }

        fn id(&self) -> NodeId {
            self.manager.shared_state.identity.node_id().clone()
        }

        /// What `origin` taught this node: a route, a registry entry, a proof.
        fn learned_about(&self, origin: &NodeId) -> bool {
            !self.manager.shared_state.relay_routes.is_empty()
                || self.manager.shared_state.peer_registry.contains_key(origin)
                || self
                    .manager
                    .shared_state
                    .relay_feature_proven(origin, features::RELAY_SIGNED)
        }
    }

    fn cancel() -> SwarmMessage {
        SwarmMessage::CancelInference(CancelInference {
            request_id: uuid::Uuid::new_v4(),
        })
    }

    /// FUTURE_WORK #298 at the call site: an envelope is dispatched as its
    /// origin's only when the origin signed it, it carries a message a node
    /// relays, and it was not taken before — and nothing is learned about the
    /// origin from one that fails.
    #[tokio::test]
    async fn a_relayed_envelope_is_dispatched_only_when_its_origin_is_proven() {
        let mut target = Target::new();
        let origin = Identity::generate();
        let relay = PeerId::random();
        let target_id = target.id();
        let seal = |msg: &SwarmMessage| {
            seal_relayed_message(&origin, target_id.clone(), uuid::Uuid::new_v4(), msg).unwrap()
        };

        let unsigned = seal_unsigned_message(
            origin.node_id(),
            &target_id,
            uuid::Uuid::new_v4(),
            &cancel(),
        );
        let not_relayed = seal(&SwarmMessage::PeerExchangeResponse(
            crate::types::PeerExchangeResponse { peers: vec![] },
        ));
        let proven = seal(&cancel());

        target.manager.handle_relayed_envelope(relay, unsigned);
        assert!(
            target.dispatched.try_recv().is_err(),
            "unsigned: its origin is a claim"
        );
        target.manager.handle_relayed_envelope(relay, not_relayed);
        assert!(
            target.dispatched.try_recv().is_err(),
            "no node relays this message"
        );
        assert!(!target.learned_about(origin.node_id()));

        target
            .manager
            .handle_relayed_envelope(relay, proven.clone());
        let taken = target
            .dispatched
            .try_recv()
            .expect("a proven envelope is dispatched");
        assert_eq!(taken.sender.as_ref(), Some(origin.node_id()));
        assert!(matches!(taken.message, SwarmMessage::CancelInference(_)));
        assert!(target.learned_about(origin.node_id()));
        assert!(target
            .manager
            .shared_state
            .relay_feature_proven(origin.node_id(), features::RELAY | features::RELAY_SIGNED));

        target.manager.handle_relayed_envelope(relay, proven);
        assert!(target.dispatched.try_recv().is_err(), "a replay is refused");
    }

    #[tokio::test]
    async fn a_relayed_tensor_is_dispatched_only_when_its_origin_is_proven() {
        let mut target = Target::new();
        let origin = Identity::generate();
        let relay = PeerId::random();
        let request_id = uuid::Uuid::new_v4();
        let forward = crate::types::LayerForward {
            request_id,
            sequence_num: 1,
            index_pos: 0,
            activations: vec![0; 8],
            format: crate::types::TensorFormat::FP16,
            model_id: crate::types::ModelId("m".into()),
            layer_range: (0, 4),
            tp_meta: None,
            vision_embeddings: None,
            chain: Vec::new(),
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: Vec::new(),
            spec_logits_requested: false,
            spec_walk_at_tail: false,
            coupling_seed: None,
            stream_seq: None,
            truncate_kv_to: None,
            prompt_cache: None,
            chunk_meta: None,
            sampling: None,
            prompt_span: None,
            image_spans: Vec::new(),
            draft_parents: Vec::new(),
            draft_tree_from: None,
        };
        let encoded = crate::network::protocol::encode_layer_forward(&forward).unwrap();
        let target_id = target.id();
        let tensor = |signed, is_result| {
            let (ephemeral_pub, sealed) = if signed {
                seal_relayed_tensor(
                    &origin,
                    &target_id,
                    &request_id,
                    RelayedKind::of_tensor(is_result),
                    &encoded,
                )
                .unwrap()
            } else {
                seal_unsigned_tensor(origin.node_id(), &target_id, &request_id, &encoded)
            };
            crate::network::protocol::RelayedTensor {
                relay_to: target_id.clone(),
                origin: origin.node_id().clone(),
                request_id,
                is_result,
                ephemeral_pub,
                sealed,
            }
        };

        target
            .manager
            .handle_relayed_tensor(relay, tensor(false, false));
        assert!(
            target.dispatched.try_recv().is_err(),
            "unsigned: its origin is a claim"
        );
        let mut flipped = tensor(true, false);
        flipped.is_result = true;
        target.manager.handle_relayed_tensor(relay, flipped);
        assert!(
            target.dispatched.try_recv().is_err(),
            "a relay flipped the frame's kind"
        );
        assert!(!target.learned_about(origin.node_id()));

        let proven = tensor(true, false);
        target.manager.handle_relayed_tensor(relay, proven.clone());
        let taken = target
            .dispatched
            .try_recv()
            .expect("a proven tensor is dispatched");
        assert_eq!(taken.sender.as_ref(), Some(origin.node_id()));
        assert!(matches!(taken.message, SwarmMessage::LayerForward(_)));
        assert!(target.learned_about(origin.node_id()));

        target.manager.handle_relayed_tensor(relay, proven);
        assert!(target.dispatched.try_recv().is_err(), "a replay is refused");
    }

    /// A peer that cannot read signed bodies is never relayed to: it could not
    /// parse one, and its reply would come back unsigned and be dropped here
    /// after it had done the work (security review of #298). Seen through the
    /// one thing a send past the gate does with a route whose relay is not
    /// connected: it forgets the route.
    #[tokio::test]
    async fn a_peer_that_cannot_sign_back_is_never_relayed_to() {
        let mut sender = Target::new();
        let peer = Identity::generate();
        let peer_bytes = crate::network::transport::node_id_to_peer_id(peer.node_id())
            .unwrap()
            .to_bytes();
        let state = sender.manager.shared_state.clone();
        state.learn_relay_route(peer.node_id(), PeerId::random().to_bytes());
        state.record_relay_proven_features(peer.node_id(), features::RELAY);

        assert!(!sender.manager.try_relay_send(&peer_bytes, &cancel()));
        assert!(
            state.relay_route_for_peer(&peer_bytes).is_some(),
            "refused before any relay was tried"
        );

        state.record_relay_proven_features(peer.node_id(), features::RELAY_SIGNED);
        assert!(!sender.manager.try_relay_send(&peer_bytes, &cancel()));
        assert!(
            state.relay_route_for_peer(&peer_bytes).is_none(),
            "past the gate: the unconnected relay's route is forgotten"
        );
    }

    /// Record `peer`'s connection as coming from `addr`, as a real one would be.
    fn connected_from(target: &mut Target, peer: PeerId, addr: &str) {
        connected_by(
            target,
            peer,
            libp2p::core::ConnectedPoint::Listener {
                local_addr: "/ip4/0.0.0.0/tcp/8810".parse().unwrap(),
                send_back_addr: addr.parse().unwrap(),
            },
        );
    }

    /// Record `peer`'s connection as the swarm would, with the default
    /// bootstrap names configured.
    fn connected_by(target: &mut Target, peer: PeerId, endpoint: libp2p::core::ConnectedPoint) {
        let names = crate::network::helpers::ConfiguredNames::of_bootstrap(
            &crate::config::network::default_bootstrap_peers(),
        );
        crate::network::manager::record_peer_remote(
            &mut target.manager.peer_remote_addrs,
            peer,
            &endpoint,
            &names,
        );
    }

    /// FUTURE_WORK #299: the replay memory's share is the delivering peer's
    /// NETWORK, not its peer id — a second key pair in the same /16 shares
    /// the first one's share, and a peer in another network still has its own.
    #[tokio::test]
    async fn relays_in_one_network_share_one_share_of_the_replay_memory() {
        let mut target = Target::new();
        target.manager.relay_replay =
            crate::crypto::relay_seal::RelayReplayGuard::with_network_share(2);
        let (first, second, elsewhere) = (PeerId::random(), PeerId::random(), PeerId::random());
        connected_from(&mut target, first, "/ip4/203.0.113.5/tcp/40000");
        connected_from(&mut target, second, "/ip4/203.0.7.9/tcp/40001");
        connected_from(&mut target, elsewhere, "/ip4/198.51.100.7/tcp/40002");
        let target_id = target.id();
        let mut deliver = |by: PeerId| {
            let origin = Identity::generate();
            let body =
                seal_relayed_message(&origin, target_id.clone(), uuid::Uuid::new_v4(), &cancel())
                    .unwrap();
            target.manager.handle_relayed_envelope(by, body);
            target.dispatched.try_recv().is_ok()
        };
        assert!(deliver(first));
        assert!(deliver(first));
        assert!(
            !deliver(second),
            "another key pair in the same network has no share of its own"
        );
        assert!(deliver(elsewhere), "another network is still served");
    }

    fn peer_entry(node: &NodeId, latency_ms: Option<u32>) -> PeerInfo {
        PeerInfo {
            node_id: node.clone(),
            addresses: vec!["/ip4/203.0.113.5/tcp/8810".into()],
            capability: None,
            last_seen: chrono::Utc::now(),
            latency_ms,
            trust_score: 0.5,
            peer_id_bytes: None,
            ack_srtt_ms: None,
            active_request_count: 0,
            first_seen: 0,
            verified_transaction_count: 0,
            is_lan_peer: false,
            goodput_bytes_per_sec: None,
            goodput_samples: 0,
        }
    }

    /// FUTURE_WORK #299: proven relayed origins — anyone can mint one — are
    /// registered within their own bound and never displace a peer this node
    /// is connected to, nor make the identify path's cap disconnect one.
    #[tokio::test]
    async fn proven_relayed_origins_never_displace_a_connected_peer() {
        let target = Target::new();
        let state = target.manager.shared_state.clone();
        let connected: Vec<NodeId> = (0..200u32)
            .map(|i| {
                let mut id = [0u8; 32];
                id[..4].copy_from_slice(&i.to_le_bytes());
                NodeId(id)
            })
            .collect();
        for (i, node) in connected.iter().enumerate() {
            state
                .peer_registry
                .insert(node.clone(), peer_entry(node, Some(10 + i as u32)));
            state.connected_node_ids.insert(node.clone());
        }

        for _ in 0..100 {
            target
                .manager
                .ensure_relayed_origin_known(Identity::generate().node_id());
        }
        assert_eq!(
            state.peer_registry.len(),
            200 + super::super::identify::MAX_UNCONNECTED_REGISTERED,
            "the origins are bounded by their own cap"
        );
        assert!(connected
            .iter()
            .all(|node| state.peer_registry.contains_key(node)));

        let mut target = target;
        target.manager.evict_one_registered_peer(&connected[0]);
        assert!(
            connected
                .iter()
                .all(|node| state.peer_registry.contains_key(node)),
            "the registry cap drops a node this node is not connected to first"
        );
        assert_eq!(
            state.peer_registry.len(),
            200 + super::super::identify::MAX_UNCONNECTED_REGISTERED - 1
        );
    }

    /// FUTURE_WORK #299: the persistent NodeId → PeerId map keeps its cap on
    /// the relay path too, and a full one makes room with a mapping for a node
    /// no longer registered rather than refusing the newest.
    #[tokio::test]
    async fn a_proven_relayed_origin_is_mapped_within_the_cap() {
        let target = Target::new();
        let state = target.manager.shared_state.clone();
        let cap = super::super::identify::MAX_PEER_TO_NODE;
        for i in 0..cap as u32 {
            let mut id = [0u8; 32];
            id[..4].copy_from_slice(&i.to_le_bytes());
            state.peer_id_map.insert(NodeId(id), vec![1]);
        }
        let origin = Identity::generate();
        target.manager.ensure_relayed_origin_known(origin.node_id());
        assert_eq!(state.peer_id_map.len(), cap);
        assert!(state.peer_id_map.contains_key(origin.node_id()));
    }

    /// Security review of #299: the default anchor is dialed by name and
    /// relays for every NAT'd node. Neither a peer reaching this node through
    /// the anchor's circuit nor one dialed by a name it chose may fill the
    /// share the anchor's own deliveries are counted in.
    #[tokio::test]
    async fn nothing_reaching_this_node_through_the_anchor_fills_the_anchors_share() {
        let mut target = Target::new();
        target.manager.relay_replay =
            crate::crypto::relay_seal::RelayReplayGuard::with_network_share(2);
        let anchor_host = "/dns4/swarmllm.duckdns.org/tcp/8810";
        let (anchor, via_circuit, named) = (PeerId::random(), PeerId::random(), PeerId::random());
        connected_by(
            &mut target,
            anchor,
            libp2p::core::ConnectedPoint::Dialer {
                address: format!("{anchor_host}/p2p/{anchor}").parse().unwrap(),
                role_override: libp2p::core::Endpoint::Dialer,
                port_use: libp2p::core::transport::PortUse::Reuse,
            },
        );
        connected_by(
            &mut target,
            via_circuit,
            libp2p::core::ConnectedPoint::Listener {
                local_addr: format!("{anchor_host}/p2p/{anchor}/p2p-circuit")
                    .parse()
                    .unwrap(),
                send_back_addr: format!("/p2p/{via_circuit}").parse().unwrap(),
            },
        );
        connected_by(
            &mut target,
            named,
            libp2p::core::ConnectedPoint::Dialer {
                address: format!("/dns4/chosen.example/tcp/8810/p2p/{named}")
                    .parse()
                    .unwrap(),
                role_override: libp2p::core::Endpoint::Dialer,
                port_use: libp2p::core::transport::PortUse::Reuse,
            },
        );
        let target_id = target.id();
        let mut deliver = |by: PeerId| {
            let origin = Identity::generate();
            let body =
                seal_relayed_message(&origin, target_id.clone(), uuid::Uuid::new_v4(), &cancel())
                    .unwrap();
            target.manager.handle_relayed_envelope(by, body);
            target.dispatched.try_recv().is_ok()
        };
        for flooder in [via_circuit, named] {
            assert!(deliver(flooder) && deliver(flooder));
            assert!(!deliver(flooder), "the flooder's own share is full");
        }
        assert!(
            deliver(anchor) && deliver(anchor),
            "the anchor's own deliveries are still taken"
        );
    }

    /// Security review of #299: a proven body marks its origin seen, so an
    /// origin still sending is never the one unseen longest, and origins
    /// minted to churn the registry make room among themselves.
    #[tokio::test]
    async fn an_origin_still_sending_keeps_its_registry_entry() {
        let target = Target::new();
        let state = target.manager.shared_state.clone();
        let cap = super::super::identify::MAX_UNCONNECTED_REGISTERED;
        let honest = Identity::generate();
        target.manager.ensure_relayed_origin_known(honest.node_id());
        state
            .peer_registry
            .get_mut(honest.node_id())
            .unwrap()
            .last_seen = chrono::Utc::now() - chrono::Duration::seconds(100);
        for _ in 1..cap {
            target
                .manager
                .ensure_relayed_origin_known(Identity::generate().node_id());
        }
        target.manager.ensure_relayed_origin_known(honest.node_id());
        for _ in 0..cap / 2 {
            target
                .manager
                .ensure_relayed_origin_known(Identity::generate().node_id());
        }
        assert!(state.peer_registry.contains_key(honest.node_id()));
        assert_eq!(state.peer_registry.len(), cap);
    }

    /// Second security-review pass on #299: libp2p-relay opens a circuit over
    /// the connection it already has to the relay and never dials the address
    /// it was handed, so an attacker's relay could advertise circuits "via"
    /// any prefix and mint a network per prefix. Every circuit through one
    /// relay counts in the network of this node's connection to that relay.
    #[tokio::test]
    async fn a_circuit_address_names_no_network_of_its_own() {
        let mut target = Target::new();
        target.manager.relay_replay =
            crate::crypto::relay_seal::RelayReplayGuard::with_network_share(2);
        let relay = PeerId::random();
        connected_from(&mut target, relay, "/ip4/203.0.113.5/tcp/40000");
        let through = |target: &mut Target, written: &str| {
            let peer = PeerId::random();
            connected_by(
                target,
                peer,
                libp2p::core::ConnectedPoint::Dialer {
                    address: format!("{written}/tcp/1/p2p/{relay}/p2p-circuit/p2p/{peer}")
                        .parse()
                        .unwrap(),
                    role_override: libp2p::core::Endpoint::Dialer,
                    port_use: libp2p::core::transport::PortUse::Reuse,
                },
            );
            peer
        };
        let first = through(&mut target, "/ip4/1.0.0.1");
        let second = through(&mut target, "/ip4/2.0.0.1");
        let named = through(&mut target, "/dns4/swarmllm.duckdns.org");
        let target_id = target.id();
        let mut deliver = |by: PeerId| {
            let origin = Identity::generate();
            let body =
                seal_relayed_message(&origin, target_id.clone(), uuid::Uuid::new_v4(), &cancel())
                    .unwrap();
            target.manager.handle_relayed_envelope(by, body);
            target.dispatched.try_recv().is_ok()
        };
        assert!(deliver(first) && deliver(first));
        assert!(
            !deliver(second),
            "another prefix written in the address is no network"
        );
        assert!(!deliver(named), "nor is the anchor's name");
        assert!(
            deliver(relay),
            "the relay's own deliveries are counted apart"
        );
    }

    /// Third security-review pass on #299: `peer_remote_addrs` keeps a peer's
    /// LATEST connection, so a relay that also reached this node by a circuit
    /// left every circuit through it in one group shared by all such relays.
    /// A circuit through it still counts in its direct connection's network.
    #[tokio::test]
    async fn a_relay_that_also_came_by_circuit_keeps_its_network() {
        let mut target = Target::new();
        target.manager.relay_replay =
            crate::crypto::relay_seal::RelayReplayGuard::with_network_share(2);
        let (relay, neighbour, other) = (PeerId::random(), PeerId::random(), PeerId::random());
        connected_from(&mut target, relay, "/ip4/203.0.113.5/tcp/40000");
        connected_from(&mut target, neighbour, "/ip4/203.0.200.9/tcp/40000");
        connected_from(&mut target, other, "/ip4/198.51.100.7/tcp/40000");
        // The relay's newest connection: an inbound circuit through `other`.
        connected_by(
            &mut target,
            relay,
            libp2p::core::ConnectedPoint::Listener {
                local_addr: format!("/ip4/198.51.100.7/tcp/40000/p2p/{other}/p2p-circuit")
                    .parse()
                    .unwrap(),
                send_back_addr: format!("/p2p/{relay}").parse().unwrap(),
            },
        );
        let through = |target: &mut Target, via: PeerId| {
            let peer = PeerId::random();
            connected_by(
                target,
                peer,
                libp2p::core::ConnectedPoint::Dialer {
                    address: format!("/p2p/{via}/p2p-circuit/p2p/{peer}")
                        .parse()
                        .unwrap(),
                    role_override: libp2p::core::Endpoint::Dialer,
                    port_use: libp2p::core::transport::PortUse::Reuse,
                },
            );
            peer
        };
        let (first, second) = (through(&mut target, relay), through(&mut target, neighbour));
        let target_id = target.id();
        let mut deliver = |by: PeerId| {
            let origin = Identity::generate();
            let body =
                seal_relayed_message(&origin, target_id.clone(), uuid::Uuid::new_v4(), &cancel())
                    .unwrap();
            target.manager.handle_relayed_envelope(by, body);
            target.dispatched.try_recv().is_ok()
        };
        assert!(deliver(first) && deliver(first));
        assert!(
            !deliver(second),
            "circuits through two relays in one /16 share one share, whichever came by circuit"
        );
    }
}
