//! Outbound command + GossipSub broadcast handlers.
//!
//! `handle_outbound_command` dispatches `NetworkCommand` from daemon tasks to
//! the per-message handlers (most live in `tensors.rs` / `shard_transfer.rs`).
//! `handle_broadcast` signs + seals + publishes a `SwarmMessage` to the
//! appropriate GossipSub topic, buffering on early-startup publish failures.

use libp2p::gossipsub::IdentTopic;

use crate::network::protocol::{self, PrefixKvDataResp, SwarmRequest, SwarmResponse, TOPIC_MODELS};
use crate::types::{NetworkCommand, SwarmMessage};

use super::{NetworkManager, MAX_BUFFERED_GOSSIP};

impl NetworkManager {
    /// Handle outbound commands from daemon tasks.
    pub(super) async fn handle_outbound_command(&mut self, cmd: NetworkCommand) {
        let cmd_name = match &cmd {
            NetworkCommand::Broadcast(_) => "Broadcast",
            NetworkCommand::SendTensor { .. } => "SendTensor",
            NetworkCommand::SendEncodedTensor { .. } => "SendEncodedTensor",
            NetworkCommand::SendTensorResult { .. } => "SendTensorResult",
            NetworkCommand::SendStreamingToken { .. } => "SendStreamingToken",
            NetworkCommand::SendShardRequest { .. } => "SendShardRequest",
            NetworkCommand::SendAllReduceRequest { .. } => "SendAllReduceRequest",
            NetworkCommand::SendAllReduceResponse { .. } => "SendAllReduceResponse",
            NetworkCommand::SendRingChunk { .. } => "SendRingChunk",
            NetworkCommand::SendDirectMessage { .. } => "SendDirectMessage",
            NetworkCommand::DialAddress(_) => "DialAddress",
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
                num_layers,
                activation_bytes,
            } => {
                self.handle_send_encoded_tensor(
                    target_peer_bytes,
                    payload,
                    request_id,
                    num_layers,
                    activation_bytes,
                );
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
                        if let Err(e) = self.swarm.dial(addr.clone()) {
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
                                let _ = self.swarm.dial(lo_addr);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(addr = %addr_str, error = %e, "Invalid multiaddr in DialAddress");
                    }
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
