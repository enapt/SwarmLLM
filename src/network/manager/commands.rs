//! Outbound command + GossipSub broadcast handlers.
//!
//! `handle_outbound_command` dispatches `NetworkCommand` from daemon tasks to
//! the per-message handlers (most live in `tensors.rs` / `shard_transfer.rs`).
//! `handle_broadcast` signs + seals + publishes a `SwarmMessage` to the
//! appropriate GossipSub topic, buffering on early-startup publish failures.

use libp2p::gossipsub::IdentTopic;

use crate::network::protocol::{self, PrefixKvDataResp, SwarmRequest, SwarmResponse, TOPIC_MODELS};
use crate::types::{NetworkCommand, SwarmMessage};

/// Next-hop lookups in flight at once (`NetworkCommand::DialPeer`). Each costs
/// a few Kademlia requests and is asked for only by a chained run this node
/// could not hand on; the cap bounds what admitted forwards can ask of it.
pub(super) const MAX_NEXT_HOP_LOOKUPS: usize = 16;

/// How long a next hop a lookup found no address for is not looked up again.
/// The hop is named by whoever sent the forward, so a name nobody in the DHT
/// knows must not buy a walk every time it is named (security review, 10-20).
pub(super) const NEXT_HOP_MISS_MEMORY: std::time::Duration = std::time::Duration::from_secs(60);
/// Misses remembered at once; the oldest goes first.
const MAX_NEXT_HOP_MISSES: usize = 256;

/// Next hops a lookup found nothing for, and when.
#[derive(Default)]
pub(super) struct NextHopMisses {
    at: std::collections::HashMap<libp2p::PeerId, std::time::Instant>,
}

impl NextHopMisses {
    /// A lookup for `peer` found no address worth dialling.
    pub(super) fn note(&mut self, peer: libp2p::PeerId, now: std::time::Instant) {
        self.at
            .retain(|_, at| now.saturating_duration_since(*at) < NEXT_HOP_MISS_MEMORY);
        if self.at.len() >= MAX_NEXT_HOP_MISSES {
            if let Some(oldest) = self.at.iter().min_by_key(|(_, at)| **at).map(|(p, _)| *p) {
                self.at.remove(&oldest);
            }
        }
        self.at.insert(peer, now);
    }

    /// Was `peer` looked up for nothing within [`NEXT_HOP_MISS_MEMORY`]?
    pub(super) fn recently(&self, peer: &libp2p::PeerId, now: std::time::Instant) -> bool {
        self.at
            .get(peer)
            .is_some_and(|at| now.saturating_duration_since(*at) < NEXT_HOP_MISS_MEMORY)
    }
}

use super::{NetworkManager, MAX_BUFFERED_GOSSIP};

impl NetworkManager {
    /// Of `addrs`, those worth dialling from here — `peer_cache::filter_dialable`,
    /// the rule every dial path follows. A chained run's next hop is dialled
    /// with what this node's registry holds and with what a DHT lookup
    /// answered, and a lookup's answer is any peer's word: unfiltered, it could
    /// point this node at a private address on a network it is not on.
    pub(super) fn dialable_from_here(&self, addrs: &[String]) -> Vec<libp2p::Multiaddr> {
        let local_addrs = self.shared_state.listen_multiaddrs.load();
        crate::network::peer_cache::filter_dialable(addrs, self.swarm.local_peer_id(), &local_addrs)
            .iter()
            .filter_map(|a| a.parse().ok())
            .collect()
    }
}

/// `SWARMLLM_TEST_TENSOR_DELAY_MS=N` holds every outbound tensor message — a
/// forward or a result — N ms before it is sent, so a split
/// between two nodes on ONE machine sees a round trip 2N longer. It is how the
/// split's speed at a same-city or same-country distance is measured and tuned
/// without a second site (`tc netem` needs root). Unset or 0: never delays.
/// Read once.
pub(super) fn test_tensor_delay() -> Option<std::time::Duration> {
    static DELAY: std::sync::OnceLock<Option<std::time::Duration>> = std::sync::OnceLock::new();
    *DELAY.get_or_init(|| {
        std::env::var("SWARMLLM_TEST_TENSOR_DELAY_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|&ms| ms > 0)
            .map(std::time::Duration::from_millis)
    })
}

impl NetworkManager {
    /// Handle outbound commands from daemon tasks.
    pub(super) async fn handle_outbound_command(&mut self, cmd: NetworkCommand) {
        if let Some(delay) = test_tensor_delay() {
            // The command that STARTS a send, once: `SendTensor` comes back as
            // `SendEncodedTensor` after its off-loop encryption, and delaying
            // both counted the delay twice.
            let is_tensor = matches!(
                cmd,
                NetworkCommand::SendTensor { .. } | NetworkCommand::SendTensorResult { .. }
            );
            if is_tensor && !self.releasing_delayed {
                let tx = self.delayed_cmd_tx.clone();
                tokio::spawn(async move {
                    tokio::time::sleep(delay).await;
                    let _ = tx.send(cmd).await;
                });
                return;
            }
        }
        let cmd_name = match &cmd {
            NetworkCommand::Broadcast(_) => "Broadcast",
            NetworkCommand::SendTensor { .. } => "SendTensor",
            NetworkCommand::SendEncodedTensor { .. } => "SendEncodedTensor",
            NetworkCommand::FailEncodedTensor { .. } => "FailEncodedTensor",
            NetworkCommand::SendTensorResult { .. } => "SendTensorResult",
            NetworkCommand::SendStreamingToken { .. } => "SendStreamingToken",
            NetworkCommand::SendShardRequest { .. } => "SendShardRequest",
            NetworkCommand::SendAllReduceRequest { .. } => "SendAllReduceRequest",
            NetworkCommand::SendAllReduceResponse { .. } => "SendAllReduceResponse",
            NetworkCommand::SendRingChunk { .. } => "SendRingChunk",
            NetworkCommand::SendDirectMessage { .. } => "SendDirectMessage",
            NetworkCommand::DialAddress(_) => "DialAddress",
            NetworkCommand::DialPeer { .. } => "DialPeer",
            NetworkCommand::StartProviding(_) => "StartProviding",
            NetworkCommand::StopProviding(_) => "StopProviding",
            NetworkCommand::SendPrefixKvFetch { .. } => "SendPrefixKvFetch",
            NetworkCommand::DeliverPrefixKvResponse { .. } => "DeliverPrefixKvResponse",
            NetworkCommand::DeliverShardResponse { .. } => "DeliverShardResponse",
        };
        tracing::debug!(cmd = cmd_name, "DIAG: handling outbound command");
        match cmd {
            NetworkCommand::Broadcast(msg) => {
                self.handle_broadcast(msg).await;
            }
            NetworkCommand::SendTensor {
                target_peer_bytes,
                forward,
            } => {
                self.handle_send_tensor(target_peer_bytes, forward);
            }
            NetworkCommand::SendEncodedTensor {
                target_peer_bytes,
                payload,
                request_id,
                step,
                activation_bytes,
            } => {
                self.handle_send_encoded_tensor(
                    target_peer_bytes,
                    payload,
                    request_id,
                    step,
                    activation_bytes,
                );
            }
            NetworkCommand::FailEncodedTensor {
                target_peer_bytes,
                request_id,
                step,
                reason,
            } => {
                self.handle_fail_encoded_tensor(target_peer_bytes, request_id, step, reason);
            }
            NetworkCommand::SendTensorResult {
                target_peer_bytes,
                result,
            } => {
                self.handle_send_tensor_result(target_peer_bytes, result);
            }
            NetworkCommand::SendStreamingToken {
                target_peer_bytes,
                token,
            } => {
                self.handle_send_streaming_token(target_peer_bytes, token);
            }
            NetworkCommand::SendShardRequest {
                target_peer_bytes,
                request,
            } => {
                self.handle_send_shard_request(target_peer_bytes, request);
            }
            NetworkCommand::SendAllReduceRequest {
                target_peer_bytes,
                request,
            } => {
                self.handle_send_rr_message(
                    target_peer_bytes,
                    SwarmMessage::TpAllReduceRequest(request),
                    "AllReduceRequest",
                    None,
                );
            }
            NetworkCommand::SendAllReduceResponse {
                target_peer_bytes,
                response,
            } => {
                self.handle_send_rr_message(
                    target_peer_bytes,
                    SwarmMessage::TpAllReduceResponse(response),
                    "AllReduceResponse",
                    None,
                );
            }
            NetworkCommand::SendRingChunk {
                target_peer_bytes,
                chunk,
            } => {
                self.handle_send_rr_message(
                    target_peer_bytes,
                    SwarmMessage::TpRingChunk(chunk),
                    "RingChunk",
                    None,
                );
            }
            NetworkCommand::SendDirectMessage {
                target_peer_bytes,
                message,
                delivery_request_id,
            } => {
                self.handle_send_rr_message(
                    target_peer_bytes,
                    message,
                    "DirectMessage",
                    delivery_request_id,
                );
            }
            NetworkCommand::DialAddress(addr_str) => {
                match addr_str.parse::<libp2p::Multiaddr>() {
                    Ok(addr) => {
                        tracing::info!(%addr, "Dialing peer from invite code");
                        // Dial the original address
                        if let Err(e) = self.dial_checked(addr.clone(), "invite_code") {
                            tracing::warn!(%addr, error = %e, "Failed to dial invite peer");
                        }
                        // Also try localhost variant for same-machine peers
                        let addr_s = addr.to_string();
                        if !addr_s.contains("/ip4/127.0.0.1/") {
                            let localhost_addr = addr_s
                                .split("/ip4/")
                                .enumerate()
                                .map(|(i, part)| {
                                    if i == 1 {
                                        // Replace the IP portion
                                        let rest = part.split_once('/').map(|x| x.1).unwrap_or("");
                                        format!("127.0.0.1/{rest}")
                                    } else {
                                        part.to_string()
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("/ip4/");
                            if let Ok(lo_addr) = localhost_addr.parse::<libp2p::Multiaddr>() {
                                tracing::debug!(%lo_addr, "Also trying localhost variant");
                                let _ = self.dial_checked(lo_addr, "invite_code_loopback");
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(addr = %addr_str, error = %e, "Invalid multiaddr in DialAddress");
                    }
                }
            }
            NetworkCommand::DialPeer {
                peer_id_bytes,
                hints,
            } => {
                let Ok(peer_id) = libp2p::PeerId::from_bytes(&peer_id_bytes) else {
                    tracing::debug!("DialPeer: not a peer id");
                    return;
                };
                if self.swarm.is_connected(&peer_id) {
                    return;
                }
                let addrs = self.dialable_from_here(&hints);
                let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(peer_id)
                    .addresses(addrs.clone())
                    .condition(libp2p::swarm::dial_opts::PeerCondition::DisconnectedAndNotDialing)
                    .extend_addresses_through_behaviour()
                    .build();
                match self.dial_checked(opts, "next_hop") {
                    Ok(()) => tracing::info!(
                        %peer_id,
                        hints = addrs.len(),
                        "Dialing a chained run's next hop"
                    ),
                    // Nothing knows where it is: a node the swarm has not
                    // introduced to this one is in no routing table here. Ask
                    // the DHT, and dial what it answers (`events.rs`).
                    Err(libp2p::swarm::DialError::NoAddresses) => {
                        if self.next_hop_lookups.len() < MAX_NEXT_HOP_LOOKUPS
                            && !self.next_hop_lookups.values().any(|p| p == &peer_id)
                            && !self
                                .next_hop_misses
                                .recently(&peer_id, std::time::Instant::now())
                        {
                            let query = self
                                .swarm
                                .behaviour_mut()
                                .kademlia
                                .get_closest_peers(peer_id);
                            self.next_hop_lookups.insert(query, peer_id);
                            tracing::info!(%peer_id, "Looking up a chained run's next hop");
                        }
                    }
                    Err(e) => tracing::debug!(%peer_id, error = %e, "Next-hop dial skipped"),
                }
            }
            NetworkCommand::StartProviding(shards) => {
                let _ = crate::network::discovery::start_providing_shards(&mut self.swarm, &shards);
            }
            NetworkCommand::StopProviding(shards) => {
                crate::network::discovery::stop_providing_shards(&mut self.swarm, &shards);
            }
            NetworkCommand::DeliverPrefixKvResponse {
                ticket,
                request_id,
                payload,
            } => {
                let Some((stored_request_id, stored_at, channel)) =
                    self.pending_prefix_kv_inbound.remove(&ticket)
                else {
                    tracing::debug!(%ticket, "DeliverPrefixKvResponse: no pending inbound fetch");
                    return;
                };
                if stored_request_id != request_id {
                    tracing::warn!(
                        %ticket,
                        stored = %stored_request_id,
                        got = %request_id,
                        "DeliverPrefixKvResponse: request_id mismatch — sending miss"
                    );
                }
                let age_ms = stored_at.elapsed().as_millis();
                let resp = SwarmResponse::PrefixKvData(PrefixKvDataResp {
                    request_id: stored_request_id,
                    payload: payload.clone(),
                });
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, resp)
                    .is_err()
                {
                    tracing::debug!(%ticket, "DeliverPrefixKvResponse: channel closed");
                }
                tracing::debug!(
                    %ticket,
                    request_id = %stored_request_id,
                    age_ms,
                    hit = payload.is_some(),
                    "DIAG: served PrefixKvFetch"
                );
            }
            NetworkCommand::DeliverShardResponse {
                ticket,
                data,
                total_size,
            } => {
                let Some((stored_at, channel)) = self.pending_shard_responses.remove(&ticket)
                else {
                    tracing::debug!(
                        %ticket,
                        "DeliverShardResponse: no pending inbound shard fetch"
                    );
                    return;
                };
                let bytes_served = data.len() as u64;
                let age_ms = stored_at.elapsed().as_millis();
                let resp =
                    SwarmResponse::ShardData(crate::types::ShardResponse { data, total_size });
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, resp)
                    .is_ok()
                {
                    if bytes_served > 0 {
                        self.shared_state
                            .shard_bytes_served
                            .fetch_add(bytes_served, std::sync::atomic::Ordering::Relaxed);
                    }
                } else {
                    tracing::debug!(
                        %ticket,
                        "DeliverShardResponse: channel closed before reply"
                    );
                }
                tracing::debug!(
                    %ticket,
                    bytes_served,
                    age_ms,
                    "DIAG: served ShardTransfer"
                );
            }
            NetworkCommand::SendPrefixKvFetch {
                target_peer_bytes,
                request_id,
                model_id,
                block_hash,
            } => {
                let peer_id = match libp2p::PeerId::from_bytes(&target_peer_bytes) {
                    Ok(p) => p,
                    Err(e) => {
                        tracing::warn!(error = %e, "SendPrefixKvFetch: invalid peer bytes");
                        // Resolve the caller's oneshot with None so they don't hang.
                        if let Some((_, tx)) = self
                            .shared_state
                            .pending_prefix_kv_fetches
                            .remove(&request_id)
                        {
                            let _ = tx.send(None);
                        }
                        return;
                    }
                };
                if !self.swarm.is_connected(&peer_id) {
                    tracing::debug!(%peer_id, "SendPrefixKvFetch: peer not connected, aborting");
                    if let Some((_, tx)) = self
                        .shared_state
                        .pending_prefix_kv_fetches
                        .remove(&request_id)
                    {
                        let _ = tx.send(None);
                    }
                    return;
                }
                let req = SwarmRequest::PrefixKvFetch(protocol::PrefixKvFetchReq {
                    request_id,
                    model_id,
                    block_hash,
                });
                let outbound = self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_request(&peer_id, req);
                self.pending_prefix_kv_outbound.insert(outbound, request_id);
                tracing::debug!(
                    %peer_id,
                    ?outbound,
                    fetch_id = %request_id,
                    "DIAG: dispatched PrefixKvFetch"
                );
            }
        }
    }

    /// Broadcast a message via GossipSub.
    pub(super) async fn handle_broadcast(&mut self, msg: SwarmMessage) {
        let topic = match &msg {
            SwarmMessage::ShardAnnounce(_)
            | SwarmMessage::NodeCapabilityUpdate(_)
            | SwarmMessage::ModelManifest(_)
            | SwarmMessage::ShardDownloadProgress(_)
            | SwarmMessage::HfSourceGossip(_)
            | SwarmMessage::PrefixCacheAnnounce(_) => TOPIC_MODELS,
            SwarmMessage::CreditGossip(_) => crate::network::protocol::TOPIC_CREDITS,
            SwarmMessage::HealthPing { .. } | SwarmMessage::HealthPong { .. } => {
                crate::network::protocol::TOPIC_HEALTH
            }
            SwarmMessage::NicknameGossip(_) => crate::network::protocol::TOPIC_IDENTITY,
            SwarmMessage::PoolMessage(_) => crate::network::protocol::TOPIC_POOLS,
            SwarmMessage::RegionShardSummary(_)
            | SwarmMessage::ModelDemandGossip(_)
            | SwarmMessage::WishlistAnnouncement(_)
            | SwarmMessage::PoolModelAvailability(_) => crate::network::protocol::TOPIC_REGIONS,
            // AllReduce responses broadcast to TP group via gossip (small group, LAN-local)
            SwarmMessage::TpAllReduceResponse(_) => crate::network::protocol::TOPIC_HEALTH,
            // Inference and credit transaction messages go via request_response, not gossipsub
            _ => return,
        };

        match protocol::encode_message(&msg) {
            Ok(data) => {
                // SEC: Sign and seal gossip with Ed25519 identity + epoch PSK.
                // Unsigned gossip allows any peer to forge messages (shard announcements,
                // model manifests, health pings, credit gossip) under arbitrary NodeIds.
                let publish_data = match self
                    .shared_state
                    .gossip_sealer
                    .seal_signed(&data, &self.shared_state.identity)
                {
                    Ok(sealed) => sealed,
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to seal gossip message, dropping");
                        return;
                    }
                };

                // Same scoping as the subscription, or a private network would
                // publish where it does not listen. Computed once: the buffer
                // below replays by name, so buffering the unscoped one would
                // send a private network's backlog onto the public topic.
                let topic_str = crate::network::protocol::topic_for_network(
                    topic,
                    self.shared_state
                        .config
                        .network
                        .gossip_network_id
                        .as_deref(),
                );
                let gossip_topic = IdentTopic::new(topic_str.clone());
                match self
                    .swarm
                    .behaviour_mut()
                    .gossipsub
                    .publish(gossip_topic, publish_data.clone())
                {
                    Ok(_) => tracing::debug!(topic, "Published message to GossipSub"),
                    Err(e) => {
                        // NET-I4: Buffer messages that fail at startup (no peers), capped to prevent memory leak
                        if self.buffered_gossip.len() < MAX_BUFFERED_GOSSIP {
                            tracing::debug!(topic, error = %e, "Failed to publish to GossipSub, buffering");
                            self.buffered_gossip.push((topic_str.clone(), publish_data));
                        } else if matches!(
                            e,
                            libp2p::gossipsub::PublishError::NoPeersSubscribedToTopic
                        ) {
                            // Nobody is listening yet. That is the normal state
                            // of every node between starting and finding the
                            // swarm, and the permanent state of an isolated or
                            // private one — not a fault, and not something to
                            // warn about once per attempt. Measured 2026-08-09:
                            // 263 warnings in 12 minutes on a node whose only
                            // distinction was having no gossip peers.
                            //
                            // A full buffer with peers PRESENT is different —
                            // that is real backpressure, and still warns below.
                            tracing::debug!(
                                topic,
                                cap = MAX_BUFFERED_GOSSIP,
                                "No peers subscribed yet; dropping gossip (buffer full)"
                            );
                        } else {
                            tracing::warn!(
                                topic,
                                error = %e,
                                cap = MAX_BUFFERED_GOSSIP,
                                "Gossip buffer full, dropping message"
                            );
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to encode outbound message");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::sync::{mpsc, watch};

    use super::NetworkManager;

    /// A next hop a lookup found nothing for is not looked up again for
    /// `NEXT_HOP_MISS_MEMORY` — the name came from whoever sent a forward —
    /// and the memory holds at most `MAX_NEXT_HOP_MISSES`.
    #[test]
    fn a_next_hop_nobody_knows_is_not_looked_up_again_for_a_while() {
        use super::{NextHopMisses, MAX_NEXT_HOP_MISSES, NEXT_HOP_MISS_MEMORY};
        let mut misses = NextHopMisses::default();
        let peer = libp2p::PeerId::random();
        let t0 = std::time::Instant::now();
        assert!(!misses.recently(&peer, t0));
        misses.note(peer, t0);
        assert!(misses.recently(&peer, t0 + NEXT_HOP_MISS_MEMORY / 2));
        assert!(!misses.recently(&peer, t0 + NEXT_HOP_MISS_MEMORY));
        for _ in 0..(MAX_NEXT_HOP_MISSES + 10) {
            misses.note(libp2p::PeerId::random(), t0);
        }
        assert!(misses.at.len() <= MAX_NEXT_HOP_MISSES);
    }

    /// A next hop is dialled only where `filter_dialable` would dial: a node
    /// on a public address is not sent to a private one a lookup answered.
    #[tokio::test]
    async fn a_next_hop_is_dialled_only_at_addresses_worth_dialling_from_here() {
        let dir = tempfile::tempdir().unwrap();
        let db = crate::storage::db::Database::open(dir.path()).unwrap();
        let config = crate::config::Config::default();
        let (state, _, _) = crate::daemon::SharedState::new(
            config.clone(),
            crate::identity::Identity::generate(),
            db,
            Arc::new(tokio::sync::Mutex::new(
                crate::inference::executor::ModelExecutor::new(),
            )),
            None,
        );
        let (command_tx, command_rx) = mpsc::channel(8);
        let (out_tx, _dispatched) = mpsc::channel(8);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);
        let (_dht_tx, dht_rx) = mpsc::channel(8);
        let manager = NetworkManager::new(
            state.clone(),
            &state.identity,
            &config,
            command_rx,
            command_tx,
            out_tx,
            shutdown_rx,
            None,
            dht_rx,
        )
        .unwrap();
        state
            .listen_multiaddrs
            .store(Arc::new(vec!["/ip4/203.0.113.5/tcp/8810".to_string()]));
        let answered = [
            "/ip4/192.168.1.1/tcp/22".to_string(),
            "/ip4/198.51.100.7/tcp/8810".to_string(),
        ];
        let dialled: Vec<String> = manager
            .dialable_from_here(&answered)
            .iter()
            .map(|a| a.to_string())
            .collect();
        assert_eq!(dialled, vec!["/ip4/198.51.100.7/tcp/8810".to_string()]);
    }
}
