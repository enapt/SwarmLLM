//! Request/response handlers for the libp2p request_response substream.
//!
//! `handle_request` services inbound `SwarmRequest` (Message/PEX, ShardTransfer,
//! PrefixKvFetch, TensorPayload). `handle_response` processes inbound replies
//! to our outbound requests (PEX response, ShardData chunks, TensorPayload
//! results, PrefixKvData snapshots, plain Ack).
//!
//! Both fire from `events.rs::handle_swarm_event` when the swarm surfaces a
//! `request_response::Event::Message`. They mutate a wide swath of
//! NetworkManager state: `pex_inbound_timestamps`, `pending_shard_requests`,
//! `pending_prefix_kv_*`, `pending_tensor_channels`, `shard_*` progress maps.

use libp2p::request_response::{self, OutboundRequestId};

use crate::model::acquisition::AcquisitionCommand;
use crate::network::helpers::is_non_public_addr;
use crate::network::protocol::{self, PrefixKvDataResp, SwarmRequest, SwarmResponse};
use crate::types::{NetworkCommand, SwarmMessage};

use super::{
    NetworkManager, MAX_INBOUND_PREFIX_FETCHES, MAX_INBOUND_SHARD_FETCHES,
    MAX_PENDING_SHARD_REQUESTS, MAX_PENDING_TENSOR_CHANNELS, PEX_MAX_PER_WINDOW, PEX_WINDOW,
};

impl NetworkManager {
    pub(super) async fn handle_request(
        &mut self,
        peer: libp2p::PeerId,
        request: SwarmRequest,
        channel: request_response::ResponseChannel<SwarmResponse>,
    ) {
        self.refresh_peer_last_seen(&peer);
        match request {
            SwarmRequest::Message(mut msg) => {
                // Handle PEX messages inline instead of forwarding to dispatcher
                match *msg {
                    SwarmMessage::PeerExchangeRequest => {
                        let now_pex = std::time::Instant::now();
                        self.pex_inbound_timestamps
                            .retain(|t| now_pex.duration_since(*t) < PEX_WINDOW);
                        if self.pex_inbound_timestamps.len() >= PEX_MAX_PER_WINDOW {
                            tracing::debug!(%peer, limit = PEX_MAX_PER_WINDOW, window_secs = PEX_WINDOW.as_secs(), "PEX rate limit exceeded, dropping request");
                            let _ = self
                                .swarm
                                .behaviour_mut()
                                .request_response
                                .send_response(channel, SwarmResponse::Ack);
                            return;
                        }
                        self.pex_inbound_timestamps.push(now_pex);
                        tracing::debug!(%peer, "Handling PEX request");
                        // Respond with up to 20 known peer addresses (filter out self)
                        let local_node_id = self.shared_state.identity.node_id();
                        let peers: Vec<String> = self
                            .shared_state
                            .peer_registry
                            .iter()
                            .filter(|entry| entry.key() != local_node_id)
                            .flat_map(|entry| entry.addresses.clone())
                            .filter(|addr| !is_non_public_addr(addr))
                            .take(20)
                            .collect();
                        let pex_resp = SwarmMessage::PeerExchangeResponse(
                            crate::types::PeerExchangeResponse { peers },
                        );
                        let resp = SwarmResponse::Message(Box::new(pex_resp));
                        if self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, resp)
                            .is_err()
                        {
                            tracing::debug!(%peer, "Failed to send PEX response (channel closed)");
                        }
                        return;
                    }
                    SwarmMessage::PeerExchangeResponse(ref pex_resp) => {
                        // SEC: Only process PEX from registered peers to prevent topology manipulation
                        let is_registered = self
                            .peer_to_node
                            .get(&peer)
                            .is_some_and(|n| self.shared_state.peer_registry.contains_key(&*n));
                        if !is_registered {
                            tracing::debug!(%peer, "Ignoring PEX response from unregistered peer");
                        } else {
                            tracing::debug!(%peer, count = pex_resp.peers.len(), "Received PEX response (via request)");
                            self.handle_pex_response(&pex_resp.peers);
                        }
                        // ACK and return
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, SwarmResponse::Ack);
                        return;
                    }
                    SwarmMessage::RelayedEnvelope(env) => {
                        // NETWORKING_PLAN Phase 1 — handled inline like PEX (in
                        // the NetworkManager, not the dispatch loop): either
                        // forward to the target or open + inject the inner
                        // message. ACK below confirms delivery to this hop.
                        self.handle_relayed_envelope(peer, env);
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, SwarmResponse::Ack);
                        return;
                    }
                    _ => {
                        // Attach sender peer identity for messages that need it
                        match &mut *msg {
                            SwarmMessage::VisionEncodeRequest(ref mut req) => {
                                req.sender_peer_bytes = Some(peer.to_bytes());
                            }
                            SwarmMessage::TpAllReduceRequest(ref mut req) => {
                                req.sender_peer_bytes = Some(peer.to_bytes());
                            }
                            _ => {}
                        }
                        // Forward all other messages to dispatcher with authenticated sender
                        tracing::debug!(%peer, "Handling protocol message request");
                        if let Err(e) = self.dispatch_authenticated(Some(&peer), *msg) {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_dropped();
                            tracing::warn!(error = %e, "Dispatcher backpressured, dropping request message");
                        } else {
                            self.shared_state
                                .metrics
                                .channel_metrics
                                .network_out
                                .record_sent();
                        }
                    }
                }
                // NET-M7: Log send_response errors
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, SwarmResponse::Ack)
                    .is_err()
                {
                    tracing::debug!(%peer, "Failed to send ACK (channel closed)");
                }
            }
            SwarmRequest::ShardTransfer(shard_req) => {
                // SEC: Only serve shards to peers in our peer_registry (authenticated + known).
                // Without this, any node that completes a Noise handshake can exfiltrate shards.
                let peer_node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                if let Some(ref nid) = peer_node_id {
                    if !self.shared_state.peer_registry.contains_key(nid) {
                        tracing::warn!(%peer, "Shard transfer from unknown peer — rejecting");
                        let _ = self.swarm.behaviour_mut().request_response.send_response(
                            channel,
                            SwarmResponse::ShardData(crate::types::ShardResponse::empty()),
                        );
                        return;
                    }
                } else {
                    tracing::warn!(%peer, "Shard transfer from unmapped peer — rejecting");
                    let _ = self.swarm.behaviour_mut().request_response.send_response(
                        channel,
                        SwarmResponse::ShardData(crate::types::ShardResponse::empty()),
                    );
                    return;
                }

                tracing::info!(
                    %peer,
                    model = %shard_req.shard_id.model_id,
                    index = shard_req.shard_id.index,
                    offset = shard_req.chunk_offset,
                    chunk_size = shard_req.chunk_size,
                    "Shard transfer request"
                );

                // SEC: do the disk read + bandwidth-throttle sleep OFF the
                // swarm event loop (gotcha #11). Awaiting them inline
                // suspends ALL network activity for the duration — at a
                // 1 Mbps cap, a 4 MB chunk would freeze the loop ~32s.
                // Stash the channel and spawn a task that posts the
                // response back via internal_cmd_tx; same pattern as the
                // PrefixKvFetch path immediately below.
                if self.pending_shard_responses.len() >= MAX_INBOUND_SHARD_FETCHES {
                    tracing::warn!(
                        %peer,
                        "ShardTransfer: inbound queue full, replying empty"
                    );
                    let resp = SwarmResponse::ShardData(crate::types::ShardResponse::empty());
                    let _ = self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, resp);
                    return;
                }
                let prepared = self.prepare_shard_read(&shard_req);
                let bw_limit = self
                    .shared_state
                    .cfg()
                    .resources
                    .shard_upload_mbps(self.shared_state.contribution());
                let ticket = uuid::Uuid::new_v4();
                self.pending_shard_responses
                    .insert(ticket, (std::time::Instant::now(), channel));
                let net_tx = self.internal_cmd_tx.clone();
                tokio::spawn(async move {
                    let (data, total_size) = match prepared {
                        Some((path, offset, chunk_size, model_id, shard_index)) => {
                            let resp = super::shard_transfer::read_shard_chunk_async(
                                path,
                                offset,
                                chunk_size,
                                model_id,
                                shard_index,
                            )
                            .await;
                            // Enforce upload bandwidth cap. 0 = unlimited
                            // (default). Only throttles shard serving, not
                            // tensor forwards.
                            if bw_limit > 0 {
                                if let SwarmResponse::ShardData(ref sr) = resp {
                                    if !sr.data.is_empty() {
                                        let bytes = sr.data.len() as u64;
                                        let limit_bytes_per_sec = bw_limit * 125_000; // Mbps → bytes/s
                                        let delay_ms = (bytes * 1000) / limit_bytes_per_sec;
                                        if delay_ms > 0 {
                                            tokio::time::sleep(std::time::Duration::from_millis(
                                                delay_ms,
                                            ))
                                            .await;
                                        }
                                    }
                                }
                            }
                            match resp {
                                SwarmResponse::ShardData(sr) => (sr.data, sr.total_size),
                                _ => (Vec::new(), 0),
                            }
                        }
                        None => (Vec::new(), 0),
                    };
                    let bytes_served = data.len() as u64;
                    let cmd = NetworkCommand::DeliverShardResponse {
                        ticket,
                        data,
                        total_size,
                    };
                    if net_tx.send(cmd).await.is_err() {
                        tracing::debug!(
                            %ticket,
                            bytes_served,
                            "ShardTransfer: internal_cmd_tx closed before delivery"
                        );
                    }
                });
            }
            SwarmRequest::PrefixKvFetch(req) => {
                // Item 8 Phase 2b: inbound cross-node prefix KV fetch.
                // Authenticated-peer gate mirrors TensorPayload. Spawn a
                // task to pull the snapshot from the local worker via IPC,
                // and stash the ResponseChannel so the eventual
                // `DeliverPrefixKvResponse` command can emit the reply.
                let peer_node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                let is_authenticated = match &peer_node_id {
                    Some(nid) => self.shared_state.peer_registry.contains_key(nid),
                    None => false,
                };
                if !is_authenticated {
                    tracing::warn!(%peer, "PrefixKvFetch from unauthenticated peer — rejecting");
                    let resp = SwarmResponse::PrefixKvData(PrefixKvDataResp {
                        request_id: req.request_id,
                        payload: None,
                    });
                    let _ = self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, resp);
                    return;
                }
                if self.pending_prefix_kv_inbound.len() >= MAX_INBOUND_PREFIX_FETCHES {
                    tracing::warn!(%peer, "PrefixKvFetch: inbound queue full, replying miss");
                    let resp = SwarmResponse::PrefixKvData(PrefixKvDataResp {
                        request_id: req.request_id,
                        payload: None,
                    });
                    let _ = self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, resp);
                    return;
                }
                let ticket = uuid::Uuid::new_v4();
                let inner_request_id = req.request_id;
                self.pending_prefix_kv_inbound.insert(
                    ticket,
                    (inner_request_id, std::time::Instant::now(), channel),
                );
                let state = self.shared_state.clone();
                let net_tx = self.internal_cmd_tx.clone();
                let model_id = req.model_id.clone();
                let model_id_for_task = model_id.clone();
                let block_hash = req.block_hash;
                tokio::spawn(async move {
                    let payload = state
                        .model_process_pool
                        .fetch_local_snapshot(&model_id_for_task, block_hash)
                        .await;
                    let cmd = NetworkCommand::DeliverPrefixKvResponse {
                        ticket,
                        request_id: inner_request_id,
                        payload,
                    };
                    if let Err(e) = net_tx.send(cmd).await {
                        tracing::debug!(error = %e, "PrefixKvFetch serve: command send failed");
                    }
                });
                tracing::debug!(
                    %peer,
                    request_id = %inner_request_id,
                    model = %model_id,
                    %ticket,
                    "DIAG: PrefixKvFetch: serving inbound fetch via worker IPC"
                );
            }
            SwarmRequest::TensorPayload(payload) => {
                // SEC: Only accept tensor payloads from authenticated peers in peer_registry.
                // Without this, any node completing a Noise handshake can inject activations
                // into in-flight inference pipelines, corrupting output silently.
                let peer_node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                let is_authenticated = match &peer_node_id {
                    Some(nid) => self.shared_state.peer_registry.contains_key(nid),
                    None => false,
                };
                if !is_authenticated {
                    tracing::warn!(%peer, "TensorPayload from unauthenticated peer — rejecting");
                    let _ = self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, SwarmResponse::Ack);
                    return;
                }

                tracing::info!(
                    %peer,
                    payload_len = payload.len(),
                    "DIAG: inbound TensorPayload request"
                );
                if let Some(request_id) = self.handle_tensor_payload(peer, &payload) {
                    // Store the ResponseChannel so we can send the computed result as
                    // the actual response (single substream per token, no separate request).
                    if self.pending_tensor_channels.len() >= MAX_PENDING_TENSOR_CHANNELS {
                        tracing::warn!(%peer, %request_id, "pending_tensor_channels full — responding with error LayerResult");
                        // Respond with an error LayerResult so the requester's oneshot resolves
                        // immediately instead of waiting for the ~600s request_timeout.
                        let err = crate::types::LayerResult::error(
                            request_id,
                            "server tensor-channel capacity exceeded",
                        );
                        let resp = match protocol::encode_layer_result(&err) {
                            Ok(bytes) => SwarmResponse::TensorPayload(bytes),
                            Err(_) => SwarmResponse::Ack,
                        };
                        let _ = self
                            .swarm
                            .behaviour_mut()
                            .request_response
                            .send_response(channel, resp);
                    } else {
                        self.pending_tensor_channels
                            .insert(request_id, (std::time::Instant::now(), peer, channel));
                        tracing::info!(
                            %peer,
                            %request_id,
                            pending_channels = self.pending_tensor_channels.len(),
                            "DIAG: stored ResponseChannel for tensor forward"
                        );
                    }
                } else {
                    // LayerResult or decode failure — just ACK since there's no round-trip
                    if self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, SwarmResponse::Ack)
                        .is_err()
                    {
                        tracing::debug!(%peer, "Failed to send tensor ACK (channel closed)");
                    }
                }
            }
            SwarmRequest::RelayedTensor(rt) => {
                // NETWORKING_PLAN tensor relay — handled inline (forward to the
                // target, or open + dispatch if we are the target). The ACK
                // confirms delivery to this hop; the computed result returns as
                // a SEPARATE relayed tensor, never on this substream (avoids the
                // bidirectional-over-relay problem).
                self.handle_relayed_tensor(peer, rt);
                if self
                    .swarm
                    .behaviour_mut()
                    .request_response
                    .send_response(channel, SwarmResponse::Ack)
                    .is_err()
                {
                    tracing::debug!(%peer, "Failed to ACK relayed tensor (channel closed)");
                }
            }
        }
    }

    pub(super) async fn handle_response(
        &mut self,
        peer: libp2p::PeerId,
        request_id: OutboundRequestId,
        response: SwarmResponse,
    ) {
        self.refresh_peer_last_seen(&peer);
        match response {
            SwarmResponse::Message(msg) => {
                // Handle PEX response inline
                if let SwarmMessage::PeerExchangeResponse(ref pex_resp) = *msg {
                    // Measure RTT from rr_ping send time
                    if let Some((_sent_peer, sent_at)) = self.ping_sent_times.remove(&request_id) {
                        let rtt_ms = sent_at.elapsed().as_millis() as u32;
                        // Clone NodeId out of peer_to_node Ref before touching peer_registry —
                        // holding a Ref across another DashMap mutating call is the gotcha #10
                        // pattern that has bitten us before.
                        let node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                        if let Some(node_id) = node_id {
                            if let Some(mut peer_info) =
                                self.shared_state.peer_registry.get_mut(&node_id)
                            {
                                peer_info.latency_ms = Some(rtt_ms);
                                // Auto-detect LAN peer from low latency (< 5ms)
                                if rtt_ms < 5 && !peer_info.is_lan_peer {
                                    peer_info.is_lan_peer = true;
                                    drop(peer_info);
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
                                    tracing::info!(%peer, rtt_ms, lan_peers = count, message = %msg, "LAN peer discovery update");
                                    self.shared_state.emit_activity(
                                        crate::daemon::state::ActivityEvent::new(
                                            "network",
                                            "lan_peer_discovered",
                                            msg,
                                        )
                                        .with_detail_num(count as i64)
                                        .with_toast("success", 8000),
                                    );
                                } else {
                                    tracing::debug!(%peer, rtt_ms, "Peer RTT measured");
                                }
                            }
                        }
                    }
                    // SEC: Only process PEX from registered peers
                    let is_registered = self
                        .peer_to_node
                        .get(&peer)
                        .is_some_and(|n| self.shared_state.peer_registry.contains_key(&*n));
                    if is_registered {
                        tracing::debug!(%peer, count = pex_resp.peers.len(), "Received PEX response");
                        self.handle_pex_response(&pex_resp.peers);
                    } else {
                        tracing::debug!(%peer, "Ignoring PEX response from unregistered peer");
                    }
                    return;
                }
                if let Err(e) = self.dispatch_authenticated(Some(&peer), *msg) {
                    self.shared_state
                        .metrics
                        .channel_metrics
                        .network_out
                        .record_dropped();
                    tracing::warn!(%peer, error = %e, "Dispatcher backpressured, dropping response message");
                } else {
                    self.shared_state
                        .metrics
                        .channel_metrics
                        .network_out
                        .record_sent();
                }
            }
            SwarmResponse::ShardData(data) => {
                tracing::debug!(
                    %peer,
                    bytes = data.data.len(),
                    total_size = data.total_size,
                    "Received shard data chunk"
                );
                // Route to AcquisitionManager — always clean up tracking state
                // NET-C1: Look up by OutboundRequestId for correct correlation
                if let Some((_, shard_id)) = self.pending_shard_requests.remove(&request_id) {
                    // Cap total_size to prevent unbounded download loops from malicious peers
                    let max_shard_bytes =
                        self.shared_state.cfg().model.shard_size_mb * 1024 * 1024 * 2;
                    if data.total_size > max_shard_bytes {
                        tracing::warn!(
                            %peer,
                            total_size = data.total_size,
                            max = max_shard_bytes,
                            "Rejecting shard download — total_size exceeds limit"
                        );
                        self.shard_download_progress.remove(&shard_id);
                        self.shard_last_progress_at.remove(&shard_id);
                        self.shard_p2p_retries.remove(&shard_id);
                        return;
                    }
                    let offset = self
                        .shard_download_progress
                        .get(&shard_id)
                        .copied()
                        .unwrap_or(0);
                    let chunk_len = data.data.len() as u64;

                    // Empty response = peer doesn't have the shard — retry with next peer or HF
                    if data.total_size == 0 || data.data.is_empty() {
                        tracing::warn!(
                            %peer,
                            model = %shard_id.model_id,
                            shard = shard_id.index,
                            "Peer returned empty shard data — trying next peer or HF fallback"
                        );
                        self.retry_shard_or_fallback(shard_id, peer, "empty response");
                        return;
                    }

                    // Forward to AcquisitionManager if it has a job for this model
                    // (user-initiated downloads via the "Download" button).
                    // Auto-manage P2P downloads don't have jobs, so we also write directly.
                    if let Some(ref acq_tx) = self.acquisition_tx {
                        let _ = acq_tx.try_send(AcquisitionCommand::ShardDataReceived {
                            shard_id: shard_id.clone(),
                            offset,
                            data: data.data.clone(),
                            total_size: data.total_size,
                        });
                    }

                    // NetworkManager is the sole writer for shard chunks (both
                    // user-initiated acquisitions and auto-manage P2P downloads).
                    // AcquisitionManager only tracks progress + verifies the final
                    // file. See commit: p2p dual-writer race fix.
                    tracing::debug!(
                        model = %shard_id.model_id,
                        shard = shard_id.index,
                        write_offset = offset,
                        chunk_len,
                        total_size = data.total_size,
                        "DIAG: writing shard chunk to disk"
                    );
                    if let Err(e) = self.shard_store.write_chunk(
                        &shard_id.model_id,
                        shard_id.index,
                        offset,
                        &data.data,
                    ) {
                        crate::log_failure!(
                            &e,
                            model = %shard_id.model_id,
                            shard = shard_id.index,
                            error = %e,
                            "Failed to write P2P shard chunk to disk"
                        );
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "download",
                                "shard_write_failed",
                                format!(
                                    "Failed to write shard {} of {} to disk: {}",
                                    crate::types::ShardId::display_index_short(shard_id.index),
                                    shard_id.model_id,
                                    e
                                ),
                            )
                            .with_model(shard_id.model_id.0.clone())
                            .with_detail_num(shard_id.index as i64)
                            .with_toast("error", 6000),
                        );
                    }

                    // Update acquisition_progress so the frontend download bar moves
                    if let Some(mut entry) = self
                        .shared_state
                        .models
                        .acquisition_progress
                        .get_mut(&shard_id.model_id)
                    {
                        entry.downloaded_bytes = entry.downloaded_bytes.saturating_add(chunk_len);
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_id.index) {
                            sp.downloaded_bytes = sp.downloaded_bytes.saturating_add(chunk_len);
                        }
                        // Estimate speed from chunk timing
                        if chunk_len > 0 {
                            entry.speed_bytes_per_sec = chunk_len; // rough per-chunk estimate
                        }
                    }

                    // Update progress tracking
                    let new_offset = offset + chunk_len;
                    if new_offset < data.total_size {
                        // More chunks needed — re-register and request next chunk
                        // SEC: enforce cap even for continuation requests to prevent unbounded growth
                        if self.pending_shard_requests.len() >= MAX_PENDING_SHARD_REQUESTS {
                            // Don't just drop the local maps: that leaves
                            // `acquisition_progress` / `p2p_download_permits`
                            // orphaned (UI shows perpetual download, semaphore
                            // permit blocks future fetches). Route through the
                            // standard retry/fallback path so HF takeover and
                            // permit release happen.
                            tracing::warn!(
                                %peer,
                                model = %shard_id.model_id,
                                index = shard_id.index,
                                "Shard download continuation dropped — pending_shard_requests at cap; routing to retry/fallback"
                            );
                            self.retry_shard_or_fallback(
                                shard_id,
                                peer,
                                "pending_shard_requests at cap on continuation",
                            );
                        } else {
                            self.shard_download_progress
                                .insert(shard_id.clone(), new_offset);
                            self.shard_last_progress_at
                                .insert(shard_id.clone(), std::time::Instant::now());

                            let next_req = crate::types::ShardRequest {
                                shard_id: shard_id.clone(),
                                chunk_offset: new_offset,
                                chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE, // 32MB chunks
                            };
                            let req = SwarmRequest::ShardTransfer(next_req);
                            let new_req_id = self
                                .swarm
                                .behaviour_mut()
                                .request_response
                                .send_request(&peer, req);
                            self.pending_shard_requests
                                .insert(new_req_id, (peer, shard_id));
                        }
                    } else {
                        // Download complete for this shard
                        self.shard_download_progress.remove(&shard_id);
                        self.shard_p2p_retries.remove(&shard_id);
                        self.shard_last_progress_at.remove(&shard_id);
                        // Allow future P2P attempts for this shard if it gets re-downloaded later.
                        self.shared_state.models.shard_p2p_failed.remove(&shard_id);
                        // Success clears any accumulated download backoff.
                        self.shared_state
                            .models
                            .clear_shard_download_backoff(&shard_id);

                        // Finalize: rename .tmp → .bin atomically
                        if let Err(e) = self
                            .shard_store
                            .finalize_shard(&shard_id.model_id, shard_id.index)
                        {
                            crate::log_failure!(
                                &e,
                                model = %shard_id.model_id,
                                shard = shard_id.index,
                                error = %e,
                                "Failed to finalize P2P shard (.tmp → .bin rename)"
                            );
                            self.shared_state.emit_activity(
                                crate::daemon::state::ActivityEvent::new(
                                    "download",
                                    "shard_finalize_failed",
                                    format!(
                                        "Failed to finalize shard {} of {}: {}",
                                        crate::types::ShardId::display_index_short(shard_id.index),
                                        shard_id.model_id,
                                        e
                                    ),
                                )
                                .with_model(shard_id.model_id.0.clone())
                                .with_detail_num(shard_id.index as i64)
                                .with_toast("error", 6000),
                            );
                        }

                        // Verify the content hash BEFORE this shard becomes visible
                        // to anyone else.
                        //
                        // This is the untrusted path — the bytes came from a peer —
                        // and it was the one path that did not check. The
                        // HuggingFace download verifies (auto_manage/download.rs),
                        // and a load-time check exists, but between finalize and
                        // those the shard was already recorded as held and
                        // announced to the swarm, so a corrupt or forged shard was
                        // advertised and re-served to other nodes until a periodic
                        // scan happened to re-hash it (~5 min). Reported by an
                        // external security review, 2026-07-28.
                        //
                        // Policy on failure is the one documented in CLAUDE.md for
                        // shard integrity errors: quarantine, penalise the sender's
                        // trust, and let the normal retry path fetch it again —
                        // crucially without announcing ourselves as a holder.
                        let shard_info = self
                            .shared_state
                            .model_registry
                            .get_manifest(&shard_id.model_id)
                            .and_then(|m| {
                                m.shards.iter().find(|s| s.index == shard_id.index).cloned()
                            });
                        // Only enforce when the manifest actually carries a
                        // hash to check against.
                        //
                        // `verify_shard` treats an all-zero hash as a FAILURE
                        // ("placeholder required"), which is right for a
                        // deliberate integrity audit but wrong as a gate on
                        // accepting a download: a manifest without hashes means
                        // we have nothing to compare to, not that the bytes are
                        // bad. Enforcing it unconditionally — as this did when
                        // first written — rejected and quarantined every P2P
                        // shard of any model whose manifest lacks hashes,
                        // making that model impossible to acquire over the
                        // network at all. Caught by a soak run against
                        // meta-llama-3.1-8b within hours of shipping.
                        let manifest_has_hash =
                            shard_info.as_ref().is_some_and(|i| i.hash != [0u8; 32]);
                        if !manifest_has_hash {
                            tracing::debug!(
                                model = %shard_id.model_id,
                                shard = shard_id.index,
                                "Manifest carries no hash for this shard — accepting unverified"
                            );
                        }
                        if let Some(info) = shard_info.filter(|_| manifest_has_hash) {
                            if let Err(e) = self.shard_store.verify_shard(&shard_id.model_id, &info)
                            {
                                // An INCOMPLETE transfer is not evidence about
                                // the sender: the bytes that did arrive may be
                                // perfectly good and the connection simply
                                // dropped. Only bytes that arrived in FULL and
                                // still hash wrong implicate the peer.
                                //
                                // Observed 2026-07-29: one peer produced four
                                // failures whose computed hash differed every
                                // time for the same shard — the signature of a
                                // truncated transfer, not corrupt storage —
                                // while also timing out constantly. Penalising
                                // trust there lowers the reputation of an
                                // honest node on a bad link.
                                let incomplete =
                                    matches!(e, crate::error::SwarmError::ShardIncomplete { .. });
                                // `incomplete` decides ATTRIBUTION (whether the
                                // sender's trust is docked, below) and must stay.
                                // The SEVERITY is a separate question and is
                                // derived, so a third `verify_shard` outcome —
                                // `ShardNotFound` is a 404, i.e. nobody's fault —
                                // does not inherit this branch's ERROR.
                                if incomplete {
                                    crate::log_failure!(
                                        &e,
                                        model = %shard_id.model_id,
                                        shard = shard_id.index,
                                        peer = %peer,
                                        error = %e,
                                        "P2P shard transfer incomplete — discarding and retrying, \
                                         NOT penalising the sender"
                                    );
                                } else {
                                    crate::log_failure!(
                                        &e,
                                        model = %shard_id.model_id,
                                        shard = shard_id.index,
                                        peer = %peer,
                                        error = %e,
                                        "P2P shard failed hash verification — quarantining, not announcing"
                                    );
                                }
                                let _ = self
                                    .shard_store
                                    .delete_shard(&shard_id.model_id, shard_id.index);
                                self.shared_state
                                    .models
                                    .shard_p2p_failed
                                    .insert(shard_id.clone());
                                if !incomplete {
                                    if let Some(node) =
                                        self.peer_to_node.get(&peer).map(|n| n.clone())
                                    {
                                        self.shared_state.credits.trust_manager.update_trust(
                                            &self.shared_state.peer_registry,
                                            &node,
                                            crate::credit::trust::TrustEvent::ShardVerificationFail,
                                        );
                                    }
                                }
                                self.shared_state.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "download",
                                        "shard_verification_failed",
                                        format!(
                                            "Shard {} of {} did not arrive intact and will be fetched again",
                                            crate::types::ShardId::display_index_short(
                                                shard_id.index
                                            ),
                                            shard_id.model_id
                                        ),
                                    )
                                    .with_model(shard_id.model_id.0.clone())
                                    .with_detail_num(shard_id.index as i64)
                                    .with_toast("warn", 6000),
                                );
                                return;
                            }
                        }

                        // Mark acquisition as complete so frontend clears the download bar
                        if let Some(mut entry) = self
                            .shared_state
                            .models
                            .acquisition_progress
                            .get_mut(&shard_id.model_id)
                        {
                            let was_complete = entry
                                .shard_progress
                                .get(&shard_id.index)
                                .map(|sp| {
                                    matches!(
                                        sp.state,
                                        crate::model::acquisition::ShardState::Complete
                                    )
                                })
                                .unwrap_or(false);
                            if let Some(sp) = entry.shard_progress.get_mut(&shard_id.index) {
                                sp.state = crate::model::acquisition::ShardState::Complete;
                                sp.downloaded_bytes = sp.total_bytes;
                            }
                            if !was_complete {
                                entry.downloaded_shards = entry.downloaded_shards.saturating_add(1);
                            }
                            if entry.total_shards > 0
                                && entry.downloaded_shards >= entry.total_shards
                            {
                                entry.state = crate::model::acquisition::AcquisitionState::Complete;
                                entry.downloaded_bytes = entry.total_bytes;
                            }
                        }
                        // Remove the acquisition entry after a delay only when the
                        // entire model is done — not after each individual shard.
                        let model_done = self
                            .shared_state
                            .models
                            .acquisition_progress
                            .get(&shard_id.model_id)
                            .map(|e| {
                                matches!(
                                    e.state,
                                    crate::model::acquisition::AcquisitionState::Complete
                                )
                            })
                            .unwrap_or(false);
                        if model_done {
                            self.shared_state
                                .schedule_acquisition_cleanup(shard_id.model_id.clone());
                        }

                        // Register ourselves as a holder of this shard
                        let local_node_id = self.shared_state.identity.node_id().clone();
                        self.shared_state
                            .model_registry
                            .record_shard_holder(shard_id.clone(), local_node_id.clone());

                        // Queue shard announce — can't call handle_broadcast inline
                        // because we're inside a swarm event handler (causes re-entrant panic).
                        // The announce will be sent on the next event loop iteration.
                        self.deferred_broadcasts
                            .push(crate::types::SwarmMessage::ShardAnnounce(
                                crate::types::ShardAnnounce {
                                    node_id: local_node_id,
                                    shards: vec![shard_id.clone()],
                                    timestamp: chrono::Utc::now(),
                                    // One shard we just fetched — incremental.
                                    complete_for_models: Vec::new(),
                                },
                            ));

                        // Load the model with the new shard (spawned async — can't block event loop)
                        crate::model::auto_manage::spawn_check_and_load(
                            self.shared_state.clone(),
                            shard_id.model_id.clone(),
                        );
                        self.shared_state.models.auto_manage_notify.notify_one();

                        // Release the P2P download semaphore permit parked by
                        // AutoShardManager::trigger_download. The shard is
                        // verified on disk; the slot is free for the next one.
                        self.shared_state
                            .models
                            .p2p_download_permits
                            .remove(&shard_id);

                        tracing::info!(
                            model = %shard_id.model_id,
                            index = shard_id.index,
                            "P2P shard download complete — registered and announced"
                        );
                        let mname = self
                            .shared_state
                            .model_registry
                            .get_manifest(&shard_id.model_id)
                            .map(|m| m.name.clone());
                        let peer_node_id = self.peer_to_node.get(&peer).map(|r| r.clone());
                        // Nickname-or-short-id via the canonical helper when we know the
                        // NodeId; fall back to the libp2p peer id only when we don't.
                        let peer_label = peer_node_id
                            .as_ref()
                            .map(|nid| {
                                crate::identity::nickname::short_display_name(
                                    nid,
                                    &self.shared_state.nickname_registry,
                                )
                            })
                            .unwrap_or_else(|| format!("{}", peer).chars().take(12).collect());
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "download",
                                "shard_p2p_complete",
                                format!(
                                    "Shard {} of {} downloaded from peer {}",
                                    crate::types::ShardId::display_index_short(shard_id.index),
                                    mname.as_deref().unwrap_or(&shard_id.model_id.0),
                                    peer_label
                                ),
                            )
                            .with_model(shard_id.model_id.0.clone())
                            .with_node(format!("{}", peer))
                            .with_detail_num(shard_id.index as i64),
                        );
                    }
                } else {
                    tracing::warn!(
                        %peer,
                        "Received shard data but no pending request found"
                    );
                }
            }
            SwarmResponse::TensorPayload(payload) => {
                // Tensor data in a response (e.g. LayerResult sent as response)
                tracing::info!(
                    %peer,
                    payload_len = payload.len(),
                    "DIAG: received TensorPayload response"
                );
                let _ = self.handle_tensor_payload(peer, &payload);
            }
            SwarmResponse::Ack => {
                tracing::debug!(%peer, "Received ACK");
            }
            SwarmResponse::PrefixKvData(resp) => {
                // Item 8 Phase 2: route to the caller's oneshot via the
                // Uuid→libp2p OutboundRequestId mapping. SharedState owns
                // the oneshot so the daemon caller's RAII guard can clean
                // up on cancellation without us noticing.
                let bytes_len = resp.payload.as_ref().map(|b| b.len()).unwrap_or(0);
                let hit = resp.payload.is_some();
                tracing::debug!(
                    %peer,
                    ?request_id,
                    inner_request_id = %resp.request_id,
                    hit,
                    bytes_len,
                    "DIAG: received PrefixKvData response"
                );
                if let Some(uuid) = self.pending_prefix_kv_outbound.remove(&request_id) {
                    if let Some((_, tx)) = self.shared_state.pending_prefix_kv_fetches.remove(&uuid)
                    {
                        let _ = tx.send(resp.payload);
                    } else {
                        tracing::debug!(
                            %peer,
                            fetch_uuid = %uuid,
                            "PrefixKvData response: no matching oneshot (caller cancelled?)"
                        );
                    }
                } else {
                    tracing::debug!(
                        %peer,
                        ?request_id,
                        "PrefixKvData response for unknown fetch (already timed out?)"
                    );
                }
            }
        }
    }
}
