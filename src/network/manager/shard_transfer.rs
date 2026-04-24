//! Shard transfer — P2P shard download request dispatch, retry/HF fallback,
//! and free-function helpers for serving shard chunks from disk.
//!
//! Extracted from manager/mod.rs to isolate the P2P shard sub-protocol from
//! the swarm event loop + gossip/pex/tensor logic. State fields live on
//! NetworkManager in mod.rs; this file is a pub(super) impl block.

use crate::network::protocol::{SwarmRequest, SwarmResponse};

use super::{NetworkManager, MAX_PENDING_SHARD_REQUESTS};

impl NetworkManager {
    /// Retry a failed P2P shard download: try another peer, or if retries are
    /// exhausted, mark the shard as P2P-failed and wake auto-manage to try HF.
    ///
    /// Returns `true` if a retry was dispatched to another peer; `false` if we
    /// gave up or fell back to HF (caller should treat the download as ended).
    pub(super) fn retry_shard_or_fallback(
        &mut self,
        shard_id: crate::types::ShardId,
        failed_peer: libp2p::PeerId,
        reason: &str,
    ) -> bool {
        const MAX_P2P_RETRIES: u32 = 5;

        tracing::debug!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            %failed_peer,
            reason,
            "DIAG: retry_shard_or_fallback entered"
        );

        // Clear any partial progress — restarting from offset 0 with a fresh peer.
        self.shard_download_progress.remove(&shard_id);
        self.shard_last_progress_at.remove(&shard_id);

        let retries = self.shard_p2p_retries.entry(shard_id.clone()).or_insert(0);
        *retries += 1;
        let retry_num = *retries;

        if retry_num <= MAX_P2P_RETRIES {
            // Pick next peer holding this shard, excluding the failed one.
            let local_nid = self.shared_state.identity.node_id().clone();
            let failed_nid = self.peer_to_node.get(&failed_peer).map(|r| r.clone());
            let other_holders: Vec<_> = self
                .shared_state
                .model_registry
                .shard_holders(&shard_id)
                .into_iter()
                .filter(|n| {
                    if *n == local_nid {
                        return false;
                    }
                    match &failed_nid {
                        Some(fp) => n != fp,
                        None => true,
                    }
                })
                .collect();

            if !other_holders.is_empty() {
                let next_target = self.shared_state.select_best_peer(&other_holders);
                if let Some(next_bytes) = self
                    .shared_state
                    .peer_registry
                    .get(&next_target)
                    .and_then(|p| p.peer_id_bytes.clone())
                {
                    let mname = self
                        .shared_state
                        .model_registry
                        .get_manifest(&shard_id.model_id)
                        .map(|m| m.name.clone());
                    self.shared_state.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "download",
                            "shard_download_p2p",
                            format!(
                                "Retrying shard {} of {} from another peer ({}, attempt {}/{})",
                                crate::types::ShardId::display_index_short(shard_id.index),
                                mname.as_deref().unwrap_or(&shard_id.model_id.0),
                                reason,
                                retry_num,
                                MAX_P2P_RETRIES,
                            ),
                        )
                        .with_model(shard_id.model_id.0.clone())
                        .with_node(format!("{}", next_target))
                        .with_detail_num(shard_id.index as i64)
                        .with_detail_str("retry".to_string()),
                    );
                    let retry_req = crate::types::ShardRequest {
                        shard_id: shard_id.clone(),
                        chunk_offset: 0,
                        chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE,
                    };
                    self.handle_send_shard_request(next_bytes, retry_req);
                    return true;
                }
            }
        }

        // Retries exhausted or no more peers — fall back to HuggingFace.
        // Release the P2P semaphore permit so the HF path (which acquires
        // its own permit in a separate trigger_download call) doesn't
        // starve behind the dead P2P request.
        self.shard_p2p_retries.remove(&shard_id);
        self.shared_state
            .models
            .p2p_download_permits
            .remove(&shard_id);
        self.shared_state
            .models
            .shard_p2p_failed
            .insert(shard_id.clone());
        // Clear the per-shard progress entry so auto_manage/scoring.rs's
        // is_shard_in_progress gate lets the HF download through.
        if let Some(mut entry) = self
            .shared_state
            .models
            .acquisition_progress
            .get_mut(&shard_id.model_id)
        {
            entry.shard_progress.remove(&shard_id.index);
        }
        tracing::info!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            retry_num,
            has_hf_source = self.shared_state.models.hf_sources.contains_key(&shard_id.model_id),
            "P2P exhausted — entering HF fallback branch"
        );
        let mname = self
            .shared_state
            .model_registry
            .get_manifest(&shard_id.model_id)
            .map(|m| m.name.clone());
        if self
            .shared_state
            .models
            .hf_sources
            .contains_key(&shard_id.model_id)
        {
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "shard_download_started",
                    format!(
                        "P2P failed after {} attempts ({}) — falling back to HuggingFace for shard {} of {}",
                        retry_num,
                        reason,
                        crate::types::ShardId::display_index_short(shard_id.index),
                        mname.as_deref().unwrap_or(&shard_id.model_id.0),
                    ),
                )
                .with_model(shard_id.model_id.0.clone())
                .with_detail_num(shard_id.index as i64)
                .with_detail_str("hf_fallback".to_string()),
            );
            // Wake auto-manage; shard_p2p_failed forces the HF path even if peers are registered.
            self.shared_state.models.auto_manage_notify.notify_one();
        } else {
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "shard_transfer_failed",
                    format!(
                        "No peers or HF source for shard {} of {} ({})",
                        crate::types::ShardId::display_index_short(shard_id.index),
                        mname.as_deref().unwrap_or(&shard_id.model_id.0),
                        reason,
                    ),
                )
                .with_model(shard_id.model_id.0.clone())
                .with_detail_num(shard_id.index as i64)
                .with_detail_str("no_source".to_string())
                .with_toast("warning", 6000),
            );
            // Mark acquisition as failed so the UI doesn't spin forever.
            if let Some(mut entry) = self
                .shared_state
                .models
                .acquisition_progress
                .get_mut(&shard_id.model_id)
            {
                entry.state = crate::model::acquisition::AcquisitionState::Failed {
                    reason: format!("P2P transfer failed: {reason}"),
                };
            }
            self.shared_state
                .schedule_acquisition_cleanup(shard_id.model_id.clone());
        }
        false
    }

    /// Send a shard transfer request to a specific peer.
    pub(super) fn handle_send_shard_request(
        &mut self,
        target_peer_bytes: Vec<u8>,
        request: crate::types::ShardRequest,
    ) {
        let Some(peer_id) = Self::resolve_peer_id(&target_peer_bytes, "shard request") else {
            return;
        };

        tracing::info!(
            %peer_id,
            model = %request.shard_id.model_id,
            index = request.shard_id.index,
            offset = request.chunk_offset,
            "Sending shard transfer request to peer"
        );

        // SEC: Cap pending shard requests to prevent memory exhaustion from
        // malicious peers that send partial chunks in an infinite loop.
        if self.pending_shard_requests.len() >= MAX_PENDING_SHARD_REQUESTS {
            tracing::warn!(
                count = self.pending_shard_requests.len(),
                "Pending shard requests at capacity — dropping new request"
            );
            return;
        }

        // Pre-check: if libp2p has no live connection, send_request would
        // queue into a dead ConnectionId and silently hang for 600s (gotcha
        // #14). Fail fast via the retry path so the next peer (or HF
        // fallback) kicks in immediately.
        if !self.swarm.is_connected(&peer_id) {
            tracing::warn!(
                %peer_id,
                shard = ?request.shard_id,
                "Dropping shard request — peer not connected; routing to failover"
            );
            self.retry_shard_or_fallback(request.shard_id, peer_id, "peer not connected");
            return;
        }
        let shard_id = request.shard_id.clone();
        let chunk_offset = request.chunk_offset;
        let req = SwarmRequest::ShardTransfer(request);
        // NET-C1: Track by OutboundRequestId for correct request-response correlation
        let outbound_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        self.pending_shard_requests
            .insert(outbound_id, (peer_id, shard_id.clone()));
        // Resume support: seed the per-shard write offset from the request's
        // chunk_offset so the first received chunk lands at the resume position
        // (and write_chunk's truncate-on-zero-offset path doesn't wipe an
        // existing partial .tmp file).
        self.shard_download_progress
            .insert(shard_id.clone(), chunk_offset);
        self.shard_last_progress_at
            .insert(shard_id, std::time::Instant::now());
    }
}

/// Read a shard chunk from disk using spawn_blocking to avoid stalling the
/// async event loop. This is a free function (not on NetworkManager) so
/// `&self` is not captured across the await point.
pub(super) async fn read_shard_chunk_async(
    path: std::path::PathBuf,
    offset: u64,
    chunk_size: u64,
    model_id: crate::types::ModelId,
    shard_index: u32,
) -> SwarmResponse {
    let result = tokio::task::spawn_blocking(move || {
        use std::io::{Read, Seek, SeekFrom};
        match std::fs::File::open(&path) {
            Ok(mut file) => {
                let total_size = file.metadata().map(|m| m.len()).unwrap_or(0);
                let chunk_size = chunk_size.min(crate::network::protocol::SHARD_CHUNK_SIZE);
                if let Err(e) = file.seek(SeekFrom::Start(offset)) {
                    tracing::warn!(error = %e, "Failed to seek in shard file");
                    return SwarmResponse::ShardData(crate::types::ShardResponse {
                        data: vec![],
                        total_size: 0,
                    });
                }
                let read_len = chunk_size.min(total_size.saturating_sub(offset)) as usize;
                let mut buf = vec![0u8; read_len];
                match file.read_exact(&mut buf) {
                    Ok(()) => {
                        tracing::debug!(
                            model = %model_id,
                            shard = shard_index,
                            offset,
                            bytes = buf.len(),
                            total_size,
                            "Serving shard chunk from file"
                        );
                        SwarmResponse::ShardData(crate::types::ShardResponse {
                            data: buf,
                            total_size,
                        })
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Failed to read shard file");
                        SwarmResponse::ShardData(crate::types::ShardResponse {
                            data: vec![],
                            total_size: 0,
                        })
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to open shard file");
                SwarmResponse::ShardData(crate::types::ShardResponse {
                    data: vec![],
                    total_size: 0,
                })
            }
        }
    })
    .await;

    match result {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(error = %e, "spawn_blocking panicked during shard read");
            SwarmResponse::ShardData(crate::types::ShardResponse {
                data: vec![],
                total_size: 0,
            })
        }
    }
}
