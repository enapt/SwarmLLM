//! Top-level swarm event dispatch.
//!
//! `handle_swarm_event` is the single match-on-`SwarmEvent` function that all
//! incoming libp2p activity flows through. It branches into the per-protocol
//! handlers in sibling modules — `requests.rs` for request_response,
//! `identify.rs` for Identify, `connections.rs` for ConnectionEstablished and
//! ConnectionClosed, `dht.rs` for DHT provider results — and handles the rest
//! inline: gossipsub Message (signed-seal verify + dispatcher push), gossipsub
//! Subscribed (replay buffered topic publishes), AutoNAT StatusChanged
//! (auto-relay listen activation on private NAT), mDNS Discovered/Expired
//! (LAN peer count + peer_registry sync), Kademlia DHT record verify,
//! ExternalAddrConfirmed (Kademlia → Server mode), plus the various
//! connection-error edges.

use libp2p::gossipsub::{self, IdentTopic};
use libp2p::request_response;
use libp2p::swarm::SwarmEvent;
use libp2p::Multiaddr;

use crate::network::behaviour::SwarmBehaviourEvent;
use crate::network::helpers::swarm_event_name;
use crate::network::protocol::{self, SwarmRequest, SwarmResponse};
use crate::types::SwarmMessage;

use super::NetworkManager;

impl NetworkManager {
    pub(super) async fn handle_swarm_event(&mut self, event: SwarmEvent<SwarmBehaviourEvent>) {
        tracing::debug!(event_type = %swarm_event_name(&event), "DIAG: processing swarm event");
        match event {
            // ── GossipSub messages ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(gossipsub::Event::Message {
                propagation_source,
                message,
                ..
            })) => {
                // SEC: All gossip MUST be signed + sealed. No unsigned fallback.
                let decoded = self
                    .shared_state
                    .gossip_sealer
                    .open_signed(&message.data)
                    .map_err(|e| {
                        tracing::warn!(
                            source = ?message.source,
                            error = %e,
                            "Rejecting unsigned/invalid gossip message"
                        );
                        e
                    })
                    .and_then(|(sender_pub, plaintext)| {
                        let msg = protocol::decode_message(&plaintext)?;
                        Ok((crate::types::NodeId(sender_pub), msg))
                    });

                match decoded {
                    Ok((sender_node_id, msg)) => {
                        // NET-M10: Reject gossip messages with timestamps older than 5 minutes
                        // or more than 30s in the future. One-sided check per gotcha #44 —
                        // symmetric .abs()-style windows double the effective replay window.
                        let now_epoch = chrono::Utc::now().timestamp() as u64;
                        const SKEW_TOLERANCE_SECS: u64 = 30;
                        const MAX_AGE_SECS: u64 = 300;
                        let stale_or_future = |ts: u64| -> bool {
                            ts > now_epoch + SKEW_TOLERANCE_SECS
                                || now_epoch.saturating_sub(ts) > MAX_AGE_SECS
                        };
                        let too_old = match &msg {
                            SwarmMessage::HealthPing { timestamp, .. }
                            | SwarmMessage::HealthPong { timestamp, .. } => {
                                stale_or_future(*timestamp)
                            }
                            SwarmMessage::ShardAnnounce(ann) => {
                                stale_or_future(ann.timestamp.timestamp() as u64)
                            }
                            SwarmMessage::CreditGossip(gossip) => {
                                stale_or_future(gossip.timestamp.timestamp() as u64)
                            }
                            _ => false,
                        };
                        if too_old {
                            tracing::debug!(
                                source = %propagation_source,
                                "Dropping stale gossip message (>5 min old)"
                            );
                        } else {
                            tracing::debug!(
                                source = %propagation_source,
                                sender = %sender_node_id,
                                "Received signed GossipSub message"
                            );
                            let authed = crate::types::AuthenticatedMessage {
                                sender: Some(sender_node_id),
                                message: msg,
                            };
                            if let Err(e) = self.outbound_tx.try_send(authed) {
                                self.shared_state
                                    .metrics
                                    .channel_metrics
                                    .network_out
                                    .record_dropped();
                                tracing::warn!(error = %e, "Dispatcher backpressured, dropping gossipsub message");
                            } else {
                                self.shared_state
                                    .metrics
                                    .channel_metrics
                                    .network_out
                                    .record_sent();
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(error = %e, "Failed to decode/verify gossipsub message");
                    }
                }
            }

            // ── JSON request/response (control messages, shard transfers) ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::Message { peer, message, .. },
            )) => match message {
                request_response::Message::Request {
                    request, channel, ..
                } => {
                    let kind = match &request {
                        SwarmRequest::Message(_) => "message",
                        SwarmRequest::ShardTransfer(_) => "shard",
                        SwarmRequest::TensorPayload(_) => "tensor",
                        SwarmRequest::PrefixKvFetch(_) => "prefix_kv_fetch",
                    };
                    tracing::info!(%peer, kind, "Received request");
                    self.handle_request(peer, request, channel).await;
                }
                request_response::Message::Response {
                    request_id,
                    response,
                } => {
                    let kind = match &response {
                        SwarmResponse::Message(_) => "message",
                        SwarmResponse::ShardData(_) => "shard",
                        SwarmResponse::Ack => "ack",
                        SwarmResponse::TensorPayload(_) => "tensor",
                        SwarmResponse::PrefixKvData(_) => "prefix_kv_data",
                    };
                    let was_tensor = self.pending_tensor_outbound.contains_key(&request_id);
                    tracing::info!(
                        %peer,
                        kind,
                        ?request_id,
                        was_tensor_forward = was_tensor,
                        pending_tensor_out = self.pending_tensor_outbound.len(),
                        "DIAG: received response"
                    );
                    // Clean up tensor outbound tracking (response received = not a failure)
                    self.pending_tensor_outbound.remove(&request_id);
                    self.pending_tensor_result_outbound.remove(&request_id);
                    self.handle_response(peer, request_id, response).await;
                }
            },

            // ── Request/response failures ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::OutboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                tracing::warn!(
                    %peer,
                    ?request_id,
                    %error,
                    is_connected = self.swarm.is_connected(&peer),
                    pending_tensor_out = self.pending_tensor_outbound.len(),
                    pending_channels = self.pending_tensor_channels.len(),
                    "DIAG: OutboundFailure"
                );
                // Check if this was a pending tensor forward — notify the pipeline
                if let Some((inference_uuid, sent_at, _target, _, _)) =
                    self.pending_tensor_outbound.remove(&request_id)
                {
                    let age_ms = sent_at.elapsed().as_millis();
                    tracing::error!(
                        %peer,
                        inference_request_id = %inference_uuid,
                        %error,
                        age_ms,
                        "Tensor forward OutboundFailure — notifying pipeline"
                    );
                    // Send an error LayerResult so the pipeline can failover immediately
                    self.fail_tensor_forward(
                        inference_uuid,
                        &peer,
                        format!("OutboundFailure: {error}"),
                    );
                }
                // Log result-send fallback failures with UUID context.
                // We can't notify the upstream requester from here — their pipeline
                // has its own timeout via their pending_tensor_outbound watchdog.
                if let Some((result_uuid, _)) =
                    self.pending_tensor_result_outbound.remove(&request_id)
                {
                    tracing::error!(
                        %peer,
                        inference_request_id = %result_uuid,
                        %error,
                        "Tensor result fallback OutboundFailure — upstream will timeout"
                    );
                }
                if let Some((label, _)) = self.pending_rr_observability.remove(&request_id) {
                    tracing::warn!(
                        %peer,
                        label,
                        %error,
                        "rr-message OutboundFailure — upstream will handle via its own timeout"
                    );
                }
                // Item 8 Phase 2: unblock a pending prefix-KV fetch on failure.
                if let Some(uuid) = self.pending_prefix_kv_outbound.remove(&request_id) {
                    if let Some((_, tx)) = self.shared_state.pending_prefix_kv_fetches.remove(&uuid)
                    {
                        let _ = tx.send(None);
                    }
                }
                // Check if this was a pending shard download request
                if let Some((_peer_id, shard_id)) = self.pending_shard_requests.remove(&request_id)
                {
                    let progress = self
                        .shard_download_progress
                        .get(&shard_id)
                        .copied()
                        .unwrap_or(0);
                    tracing::error!(
                        %peer,
                        model = %shard_id.model_id,
                        shard_index = shard_id.index,
                        %error,
                        bytes_downloaded = progress,
                        "DIAG: shard download OutboundFailure — attempting peer failover"
                    );
                    // Try another peer; fall back to HF only after retries exhausted.
                    self.retry_shard_or_fallback(shard_id, peer, &format!("{error}"));
                }
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::InboundFailure {
                    peer,
                    request_id,
                    error,
                    ..
                },
            )) => {
                // Note: pending_tensor_channels is keyed by Uuid (from the parsed
                // message), not InboundRequestId — we can't directly remove the entry
                // here. The stale timeout cleanup (every 30s) handles orphaned channels.
                tracing::warn!(
                    %peer,
                    ?request_id,
                    %error,
                    pending_channels = self.pending_tensor_channels.len(),
                    "DIAG: InboundFailure — response send may have failed, stale cleanup will reclaim"
                );
            }
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RequestResponse(
                request_response::Event::ResponseSent {
                    peer, request_id, ..
                },
            )) => {
                tracing::info!(%peer, ?request_id, "DIAG: ResponseSent event — response written to wire");
            }

            // ── GossipSub peer subscribed — flush matching buffered messages (NET-I4) ──
            //
            // Only the just-subscribed topic is eligible for replay — a
            // Subscribed{peer_id, topic=X} event tells us the mesh now has
            // at least one peer on topic X, but says nothing about topic Y.
            // Before this filter, ANY Subscribed event iterated the whole
            // buffer and called publish() on every entry; publish() would
            // still return Err for topics with no subscribers (gossipsub
            // routes correctly, so this wasn't an info leak), but the
            // entry got re-buffered, wasting a full O(buffer) pass per
            // Subscribed event on a multi-topic mesh.
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Gossipsub(
                gossipsub::Event::Subscribed { peer_id, topic },
            )) => {
                tracing::debug!(%peer_id, %topic, "Peer subscribed to topic");
                let subscribed_topic_str = topic.to_string();
                let has_match = self
                    .buffered_gossip
                    .iter()
                    .any(|(t, _)| t == &subscribed_topic_str);
                if has_match {
                    let mut remaining = Vec::with_capacity(self.buffered_gossip.len());
                    let mut replayed = 0;
                    for (topic_str, data) in std::mem::take(&mut self.buffered_gossip) {
                        if topic_str != subscribed_topic_str {
                            remaining.push((topic_str, data));
                            continue;
                        }
                        let gossip_topic = IdentTopic::new(&topic_str);
                        match self
                            .swarm
                            .behaviour_mut()
                            .gossipsub
                            .publish(gossip_topic, data.clone())
                        {
                            Ok(_) => replayed += 1,
                            Err(_) => remaining.push((topic_str, data)),
                        }
                    }
                    self.buffered_gossip = remaining;
                    if replayed > 0 {
                        tracing::info!(
                            topic = %subscribed_topic_str,
                            count = replayed,
                            "Replayed buffered GossipSub messages for newly-subscribed topic"
                        );
                    }
                }
            }

            // ── Relay server events ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::RelayServer(event)) => {
                crate::network::relay::handle_relay_server_event(event, &self.shared_state);
            }

            // ── AutoNAT status changes ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Autonat(
                libp2p::autonat::Event::StatusChanged { old, new },
            )) => {
                tracing::info!(?old, ?new, "AutoNAT status changed");
                {
                    if let Ok(mut stats) = self.shared_state.metrics.node_stats.try_write() {
                        stats.nat_status = Some(format!("{new:?}"));
                    }
                }
                // NET-M3: Auto-listen on relay when NAT is detected as Private
                if matches!(new, libp2p::autonat::NatStatus::Private)
                    && !self.relay_activated
                    && self.shared_state.config.network.auto_relay
                {
                    self.relay_activated = true;
                    tracing::info!(target: "swarmllm::network::manager", "NAT detected, activating relay listener");

                    // Try bootstrap peers as relay candidates — they are most likely
                    // to be publicly reachable and have relay enabled.
                    let bootstrap_addrs = &self.shared_state.config.network.bootstrap_peers;
                    let mut relayed = false;
                    for addr_str in bootstrap_addrs {
                        if let Ok(maddr) = addr_str.parse::<Multiaddr>() {
                            // Extract the peer ID from the multiaddr (/p2p/<peer_id>)
                            let maybe_pid = maddr.iter().find_map(|proto| {
                                if let libp2p::multiaddr::Protocol::P2p(pid) = proto {
                                    Some(pid)
                                } else {
                                    None
                                }
                            });
                            if let Some(relay_pid) = maybe_pid {
                                // Build a relay-listen address without the trailing /p2p
                                let base: Multiaddr = maddr
                                    .iter()
                                    .take_while(|p| {
                                        !matches!(p, libp2p::multiaddr::Protocol::P2p(_))
                                    })
                                    .collect();
                                let relay_addr =
                                    crate::network::relay::relay_listen_addr(&relay_pid, &base);
                                match self.swarm.listen_on(relay_addr.clone()) {
                                    Ok(_) => {
                                        tracing::info!(
                                            relay_peer = %relay_pid,
                                            %relay_addr,
                                            "Relay listen activated"
                                        );
                                        relayed = true;
                                        break; // One relay is sufficient
                                    }
                                    Err(e) => {
                                        tracing::debug!(
                                            relay_peer = %relay_pid,
                                            error = %e,
                                            "Failed to listen via relay peer"
                                        );
                                    }
                                }
                            }
                        }
                    }

                    if !relayed && !bootstrap_addrs.is_empty() {
                        tracing::warn!(
                            "NAT detected but no relay peers accepted — node may be unreachable"
                        );
                    }
                }
            }

            // ── Identify ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Identify(
                libp2p::identify::Event::Received {
                    peer_id,
                    info,
                    connection_id,
                },
            )) => {
                self.handle_identify_received(peer_id, info, connection_id);
            }

            // ── Kademlia ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Kademlia(
                libp2p::kad::Event::OutboundQueryProgressed { id, result, .. },
            )) => {
                use libp2p::kad::QueryResult;
                match result {
                    QueryResult::GetRecord(Ok(libp2p::kad::GetRecordOk::FoundRecord(
                        peer_record,
                    ))) => {
                        // Verify Ed25519 signature on DHT records before trusting
                        match crate::network::discovery::verify_dht_value(&peer_record.record.value)
                        {
                            Ok((pubkey, payload)) => {
                                tracing::debug!(
                                    key = ?peer_record.record.key,
                                    signer = %hex::encode(&pubkey[..8]),
                                    payload_len = payload.len(),
                                    "DHT record verified"
                                );
                                // Process verified payload: deserialize NodeCapability
                                // and update peer registry with the advertised capabilities.
                                let key_bytes = peer_record.record.key.as_ref();
                                let key_str = String::from_utf8_lossy(key_bytes);
                                if key_str.starts_with("/swarm/node/") {
                                    if let Ok(cap) = serde_json::from_slice::<
                                        crate::types::NodeCapability,
                                    >(payload)
                                    {
                                        let node_id = crate::types::NodeId(pubkey);
                                        if let Some(mut entry) =
                                            self.shared_state.peer_registry.get_mut(&node_id)
                                        {
                                            entry.capability = Some(cap);
                                        }
                                    }
                                }
                            }
                            Err(_) => {
                                tracing::warn!(
                                    key = ?peer_record.record.key,
                                    "DHT record failed signature verification — ignoring"
                                );
                            }
                        }
                    }
                    // S5: DHT provider query results — merge discovered holders
                    // into the bounded shard_holders cache.
                    QueryResult::GetProviders(Ok(
                        libp2p::kad::GetProvidersOk::FoundProviders { providers, .. },
                    )) => {
                        self.handle_dht_providers_found(id, &providers);
                    }
                    QueryResult::GetProviders(Ok(
                        libp2p::kad::GetProvidersOk::FinishedWithNoAdditionalRecord { .. },
                    )) => {
                        // Query finished — clean up tracking
                        self.pending_provider_queries.remove(&id);
                    }
                    QueryResult::GetProviders(Err(ref e)) => {
                        tracing::debug!(error = ?e, "DHT provider query failed — cleaning up");
                        self.pending_provider_queries.remove(&id);
                    }
                    _ => {
                        tracing::debug!(?result, "Kademlia query progressed");
                    }
                }
            }

            // ── mDNS ──
            SwarmEvent::Behaviour(SwarmBehaviourEvent::Mdns(libp2p::mdns::Event::Discovered(
                peers,
            ))) => {
                for (peer_id, addr) in peers {
                    // Do NOT add mDNS addresses to Kademlia. Kademlia's periodic
                    // routing table refresh dials all known addresses, creating
                    // duplicate connections every 30s that corrupt request_response
                    // routing. The identify protocol handles address exchange after
                    // connection is established.
                    if !self.swarm.is_connected(&peer_id) {
                        tracing::info!(
                            %peer_id, %addr,
                            "LAN peer discovered automatically — no configuration needed"
                        );
                        // Use Disconnected (not DisconnectedAndNotDialing) so mDNS
                        // can override a failing bootstrap dial attempt. Without this,
                        // a peer that restarts with a new identity can't reconnect
                        // because the stale bootstrap dial blocks mDNS.
                        let opts = libp2p::swarm::dial_opts::DialOpts::peer_id(peer_id)
                            .condition(libp2p::swarm::dial_opts::PeerCondition::Disconnected)
                            .addresses(vec![addr])
                            .build();
                        if let Err(e) = self.swarm.dial(opts) {
                            tracing::debug!(%peer_id, error = %e, "mDNS: dial skipped");
                        }
                    } else {
                        tracing::debug!(%peer_id, "mDNS: already connected, skipping");
                    }
                    // Mark as LAN peer if we can derive their NodeId
                    if let Some(node_id) = self.peer_to_node.get(&peer_id) {
                        if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&*node_id) {
                            if !peer.is_lan_peer {
                                peer.is_lan_peer = true;
                                drop(peer);
                                // Increment LAN peer count and notify via unified activity event
                                let count = self
                                    .shared_state
                                    .lan_peer_count
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                                    + 1;
                                let msg = format!(
                                    "Found {} peer{} on your local network",
                                    count,
                                    if count == 1 { "" } else { "s" }
                                );
                                tracing::info!(lan_peers = count, message = %msg, "LAN peer discovery update");
                                self.shared_state.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "network",
                                        "lan_peer_discovered",
                                        msg,
                                    )
                                    .with_detail_num(count as i64)
                                    .with_toast("success", 8000),
                                );
                            }
                        }
                    }
                }
            }

            SwarmEvent::Behaviour(SwarmBehaviourEvent::Mdns(libp2p::mdns::Event::Expired(
                peers,
            ))) => {
                for (peer_id, _addr) in peers {
                    tracing::debug!(%peer_id, "mDNS: peer expired");
                    // Decrement LAN peer count if this was a tracked LAN peer
                    if let Some(node_id) = self.peer_to_node.get(&peer_id) {
                        if let Some(mut peer) = self.shared_state.peer_registry.get_mut(&*node_id) {
                            if peer.is_lan_peer {
                                peer.is_lan_peer = false;
                                drop(peer);
                                let _ = self.shared_state.lan_peer_count.fetch_update(
                                    std::sync::atomic::Ordering::Relaxed,
                                    std::sync::atomic::Ordering::Relaxed,
                                    |v| v.checked_sub(1),
                                );
                            }
                        }
                    }
                }
            }

            SwarmEvent::ConnectionEstablished {
                peer_id,
                connection_id,
                num_established,
                endpoint,
                ..
            } => {
                self.handle_connection_established(
                    peer_id,
                    connection_id,
                    num_established,
                    &endpoint,
                );
            }

            SwarmEvent::ConnectionClosed {
                peer_id,
                connection_id,
                cause,
                num_established,
                ..
            } => {
                self.handle_connection_closed(peer_id, connection_id, cause, num_established);
            }

            SwarmEvent::NewListenAddr { address, .. } => {
                tracing::info!(%address, "New listen address");
            }

            // NET-I7: Switch Kademlia to Server mode when external address is confirmed
            SwarmEvent::ExternalAddrConfirmed { address } => {
                tracing::info!(%address, "External address confirmed — switching Kademlia to Server mode");
                self.swarm
                    .behaviour_mut()
                    .kademlia
                    .set_mode(Some(libp2p::kad::Mode::Server));
            }

            SwarmEvent::OutgoingConnectionError { peer_id, error, .. } => {
                tracing::debug!(
                    ?peer_id, %error,
                    "Outgoing connection failed"
                );
            }

            SwarmEvent::IncomingConnectionError { error, .. } => {
                tracing::debug!(
                    %error,
                    "Incoming connection failed"
                );
            }

            other => {
                tracing::trace!(?other, "Unhandled swarm event");
            }
        }
    }
}
