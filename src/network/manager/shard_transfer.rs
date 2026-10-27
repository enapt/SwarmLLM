//! Shard transfer — P2P shard download request dispatch, retry/HF fallback,
//! and free-function helpers for serving shard chunks from disk.
//!
//! Extracted from manager/mod.rs to isolate the P2P shard sub-protocol from
//! the swarm event loop + gossip/pex/tensor logic. State fields live on
//! NetworkManager in mod.rs; this file is a pub(super) impl block.

use crate::network::protocol::{SwarmRequest, SwarmResponse};

use super::{NetworkManager, MAX_PENDING_SHARD_REQUESTS};

impl NetworkManager {
    /// Should this P2P transfer stop because somebody else now owns the
    /// shard's `.tmp`? Returns the reason, for the log.
    ///
    /// Confirmed on this project's own node, 2026-09-12: a P2P transfer of
    /// llama-3.2-3b shard 1 stalled, the 180 s permit sweep released its slot
    /// and declared "HF fallback will fire next cycle", the HuggingFace
    /// download ran and **completed the shard at 17:09:35** — and the P2P chain
    /// was still asking peers for it at 17:10:26, from offset 0. Two watchdogs
    /// with no agreement between them: the 30 s one keeps retrying the P2P
    /// chain, the 180 s one hands the shard to HuggingFace.
    ///
    /// Nothing stopped the loser writing. `ShardStore::write_chunk` truncates
    /// at `offset == 0`, so a peer answering one of those late retries during
    /// the HuggingFace download would have truncated its partial file to
    /// nothing mid-transfer — and after the rename it would recreate a `.tmp`
    /// for a shard already complete on disk. It did not happen only because
    /// those peers stayed silent, which is why the transfer stalled in the
    /// first place.
    ///
    /// The rule is the one the HuggingFace path already follows: one writer per
    /// shard file. A transfer owns the file while its slot is parked in
    /// `p2p_download_permits` (the slot holds the claim); if the claim is held
    /// and the slot is gone, the claim belongs to somebody else and this
    /// transfer is over.
    fn shard_transfer_lost_its_file(
        &self,
        shard_id: &crate::types::ShardId,
    ) -> Option<&'static str> {
        let claimed = self.shared_state.models.shard_download_claimed(shard_id);
        if !claimed {
            return None;
        }
        if self
            .shared_state
            .models
            .p2p_download_permits
            .contains_key(shard_id)
        {
            return None;
        }
        Some("another download owns this shard's file")
    }

    /// End a P2P transfer whose file now belongs to somebody else.
    ///
    /// Deliberately does NOT remove the `.tmp`: the point is that the file is
    /// no longer ours. Deleting it is the collision, not the cure.
    pub(super) fn abandon_shard_transfer_if_not_ours(
        &mut self,
        shard_id: &crate::types::ShardId,
    ) -> bool {
        let Some(reason) = self.shard_transfer_lost_its_file(shard_id) else {
            return false;
        };
        tracing::info!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            reason,
            "Stopping a peer transfer — this shard is being fetched another way"
        );
        self.shard_download_progress.remove(shard_id);
        self.shard_last_progress_at.remove(shard_id);
        self.shard_p2p_retries.remove(shard_id);
        self.shard_busy_holders.remove(shard_id);
        self.pending_shard_requests
            .retain(|_, (_, sid)| sid != shard_id);
        true
    }

    /// Has this model's download been cancelled? If so, stop the transfer and
    /// tidy up; returns whether it did.
    ///
    /// Asked on every chunk, because a P2P transfer is a chain of
    /// request/response hops across the network event loop rather than a loop
    /// with a flag to read — so without this, Cancel reached the HuggingFace
    /// path and silently did nothing on the P2P one, and the same button meant
    /// two different things depending on where the shard happened to come from.
    ///
    /// Deliberately NOT `retry_shard_or_fallback`: that exists for a transfer
    /// that FAILED and its job is to find the bytes somewhere else, which for a
    /// cancel is the opposite of what was asked. This ends the transfer.
    pub(super) fn abort_shard_transfer_if_cancelled(
        &mut self,
        shard_id: &crate::types::ShardId,
    ) -> bool {
        // Asked of the flag the TRANSFER holds, not of the model's current
        // entry in `download_cancel_flags` — that entry is replaced when a
        // download starts after a cancel, and a transfer reading the map would
        // then be shown a newer, unset flag and sail through the cancel meant
        // for it. A transfer with no parked slot has already been released by
        // something else and has nothing to abort.
        let cancelled = self
            .shared_state
            .models
            .p2p_download_permits
            .get(shard_id)
            .map(|slot| slot.is_cancelled())
            .unwrap_or(false);
        if !cancelled {
            return false;
        }

        tracing::info!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            "P2P shard transfer cancelled by user — stopping"
        );

        self.shard_download_progress.remove(shard_id);
        self.shard_last_progress_at.remove(shard_id);
        self.shard_p2p_retries.remove(shard_id);
        self.shard_busy_holders.remove(shard_id);
        self.pending_shard_requests
            .retain(|_, (_, sid)| sid != shard_id);

        // Remove the partial file, then release the permit and the writer
        // claim — in that order, so nothing can start writing this shard
        // between the two and have its file deleted underneath it.
        if let Err(e) = std::fs::remove_file(
            self.shard_store
                .shard_tmp_path(&shard_id.model_id, shard_id.index),
        ) {
            if e.kind() != std::io::ErrorKind::NotFound {
                tracing::warn!(
                    model = %shard_id.model_id,
                    shard = shard_id.index,
                    error = %e,
                    "Could not remove the partial file of a cancelled transfer"
                );
            }
        }
        self.shared_state
            .models
            .p2p_download_permits
            .remove(shard_id);

        // Clear the per-shard progress so the dashboard stops showing a bar for
        // something nobody is fetching.
        if let Some(mut entry) = self
            .shared_state
            .models
            .acquisition_progress
            .get_mut(&shard_id.model_id)
        {
            entry.shard_progress.remove(&shard_id.index);
        }
        true
    }

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

        // A cancelled download must not be resurrected by its own failure. The
        // chunk path already stops on the flag, but a transfer that fails at
        // the same moment arrives here instead, and this function's whole job
        // is to go and find the bytes from somebody else.
        if self.abort_shard_transfer_if_cancelled(&shard_id) {
            // `false` = the download has ended. `true` would tell the caller a
            // retry is on its way to another peer, and there is none.
            return false;
        }
        // Nor may a failure restart a transfer whose shard is now being
        // fetched another way — that is how a P2P chain outlived the
        // HuggingFace download that had already completed the shard.
        if self.abandon_shard_transfer_if_not_ours(&shard_id) {
            return false;
        }

        // Clear any partial progress — restarting from offset 0 with a fresh peer.
        self.shard_download_progress.remove(&shard_id);
        self.shard_last_progress_at.remove(&shard_id);

        // Cap shard_p2p_retries map: every retry adds at most one entry per
        // unique ShardId. On a seeder serving many distinct shards under
        // hostile churn the map can grow without bound. Soft-evict the
        // oldest-by-key entry when at cap to keep memory predictable
        // (R97). Real eviction happens when retries exhaust or completion
        // succeeds (lines 108, requests.rs:504/642).
        const MAX_P2P_RETRY_ENTRIES: usize = 4096;
        if !self.shard_p2p_retries.contains_key(&shard_id)
            && self.shard_p2p_retries.len() >= MAX_P2P_RETRY_ENTRIES
        {
            if let Some(victim) = self.shard_p2p_retries.keys().next().cloned() {
                self.shard_p2p_retries.remove(&victim);
                self.shard_busy_holders.remove(&victim);
                tracing::warn!(
                    cap = MAX_P2P_RETRY_ENTRIES,
                    "shard_p2p_retries at cap; evicted one entry"
                );
            }
        }
        let retries = self.shard_p2p_retries.entry(shard_id.clone()).or_insert(0);
        *retries += 1;
        let retry_num = *retries;

        if retry_num <= MAX_P2P_RETRIES {
            // Pick next peer holding this shard, excluding the failed one.
            let failed_nid = self.peer_to_node.get(&failed_peer).map(|r| r.clone());
            let other_holders: Vec<_> = self
                .shared_state
                .model_registry
                .shard_holders(&shard_id)
                .into_iter()
                .filter(|n| match &failed_nid {
                    Some(fp) => n != fp,
                    None => true,
                })
                .collect();

            // The retry is scoped exactly like the first attempt. It used to
            // pick straight out of `shard_holders`, so private mode held until
            // a transfer failed and not afterwards.
            if let Some(next_target) = self.shared_state.select_best_allowed_peer(&other_holders) {
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
                                "Retrying part {} of {} from another computer ({}, attempt {}/{})",
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
        self.shard_busy_holders.remove(&shard_id);
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
                        "P2P failed after {} attempts ({}) — falling back to HuggingFace for part {} of {}",
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
            // No HF fallback to hand off to — this is a terminal failure, so
            // back the shard off. (The HF-fallback branch above deliberately
            // does NOT back off: it wants an immediate HF retry via
            // `shard_p2p_failed`.)
            let (fails, delay) = self
                .shared_state
                .models
                .record_shard_download_failure(&shard_id);
            tracing::info!(
                model = %shard_id.model_id,
                shard = shard_id.index,
                fails,
                backoff_secs = delay,
                "P2P exhausted with no HF source — backing off shard"
            );
            self.shared_state.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "shard_transfer_failed",
                    format!(
                        "No computers or HF source for part {} of {} ({})",
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

    /// A chunk of one of this node's parts landed from `peer`. Every other
    /// part waiting on that holder starts its count of busy answers again
    /// (#340): a holder shares its upload among networks, so a node fetching
    /// more parts from it than its share carries has some waiting while the
    /// others land, and each such part raced its siblings for the next free
    /// share. Counted as a holder that stayed busy, one part lost 16 times in a
    /// minute and left a holder that was sending this node its other parts
    /// (rig, 2026-10-27).
    pub(super) fn note_chunk_landed_from(&mut self, peer: &libp2p::PeerId) {
        if self.shard_busy_holders.is_empty() {
            return;
        }
        let Some(holder) = self.peer_to_node.get(peer).map(|r| r.clone()) else {
            return;
        };
        for busy in self.shard_busy_holders.values_mut() {
            busy.holder_is_serving_us(&holder);
        }
    }

    /// A holder answered a part request BUSY (#335): its upload limit is
    /// spoken for. Keep the bytes already written and ask again at the same
    /// offset — a holder that is not busy at once, else the first to come free
    /// when its wait ends. Never a failure, never a restart from 0, which is
    /// what the empty refusal costs an older build. Bounded: past
    /// [`MOST_BUSY_ANSWERS`] answers the transfer takes the ordinary retry
    /// path, as a failure would.
    pub(super) fn ask_again_after_busy(
        &mut self,
        shard_id: crate::types::ShardId,
        busy_peer: libp2p::PeerId,
        retry_after_ms: u32,
    ) {
        if self.abort_shard_transfer_if_cancelled(&shard_id)
            || self.abandon_shard_transfer_if_not_ours(&shard_id)
        {
            return;
        }
        let Some(busy_holder) = self.peer_to_node.get(&busy_peer).map(|r| r.clone()) else {
            self.retry_shard_or_fallback(shard_id, busy_peer, "busy holder no longer known");
            return;
        };
        let now = std::time::Instant::now();
        let wait = std::time::Duration::from_millis(u64::from(retry_after_ms))
            .clamp(BUSY_WAIT_LEAST, super::upload_pacer::RELEASED_WITHIN);
        if !self.shard_busy_holders.contains_key(&shard_id)
            && self.shard_busy_holders.len() >= MAX_BUSY_ENTRIES
        {
            // Entries leave with their transfer; this only bounds a map no
            // transfer cleans up (the retry map's own cap does the same).
            if let Some(victim) = self.shard_busy_holders.keys().next().cloned() {
                self.shard_busy_holders.remove(&victim);
            }
        }
        let busy = self.shard_busy_holders.entry(shard_id.clone()).or_default();
        busy.note(busy_holder, now + wait);
        if busy.answers > MOST_BUSY_ANSWERS {
            self.shard_busy_holders.remove(&shard_id);
            self.retry_shard_or_fallback(shard_id, busy_peer, "every holder stayed busy");
            return;
        }
        let ask = next_ask(
            busy,
            &self.shared_state.model_registry.shard_holders(&shard_id),
            now,
        );
        // The holder answered, so the transfer is alive: the permit sweep
        // must not read the wait as a stall.
        self.shared_state
            .models
            .note_p2p_transfer_progress(&shard_id);
        let offset = self
            .shard_download_progress
            .get(&shard_id)
            .copied()
            .unwrap_or(0);
        let (candidates, at) = match ask {
            NextAsk::Now(free) => (free, None),
            NextAsk::At(when, free) => (free, Some(when)),
        };
        let target = self
            .shared_state
            .select_best_allowed_peer(&candidates)
            .and_then(|n| {
                self.shared_state
                    .peer_registry
                    .get(&n)
                    .and_then(|p| p.peer_id_bytes.clone())
            });
        let Some(target_peer_bytes) = target else {
            self.retry_shard_or_fallback(shard_id, busy_peer, "no holder to ask after busy");
            return;
        };
        tracing::debug!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            %busy_peer,
            offset,
            wait_ms = at.map(|t| t.saturating_duration_since(now).as_millis() as u64),
            "Part holder busy — asking again at the same offset"
        );
        let request = crate::types::ShardRequest {
            shard_id,
            chunk_offset: offset,
            chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE,
        };
        match at {
            None => self.handle_send_shard_request(target_peer_bytes, request),
            Some(when) => {
                let tx = self.internal_cmd_tx.clone();
                tokio::spawn(async move {
                    tokio::time::sleep_until(tokio::time::Instant::from_std(when)).await;
                    let _ = tx
                        .send(crate::types::NetworkCommand::ResumeShardAfterBusy {
                            target_peer_bytes,
                            request,
                        })
                        .await;
                });
            }
        }
    }

    /// The wait after a busy answer has passed (#335): ask again, unless the
    /// transfer has moved on meanwhile — ended (its busy record goes with it),
    /// cancelled, fetched another way, or started again with a request of its
    /// own in flight. Asking then would reset that transfer's offset to ours.
    pub(super) fn resume_shard_after_busy(
        &mut self,
        target_peer_bytes: Vec<u8>,
        request: crate::types::ShardRequest,
    ) {
        let shard_id = &request.shard_id;
        let still_waiting = self.shard_busy_holders.contains_key(shard_id)
            && self.shard_download_progress.get(shard_id) == Some(&request.chunk_offset)
            && !self
                .pending_shard_requests
                .values()
                .any(|(_, sid)| sid == shard_id);
        if !still_waiting
            || self.abort_shard_transfer_if_cancelled(shard_id)
            || self.abandon_shard_transfer_if_not_ours(shard_id)
        {
            tracing::debug!(
                model = %shard_id.model_id,
                shard = shard_id.index,
                "Not asking again after busy — the transfer has moved on"
            );
            return;
        }
        self.handle_send_shard_request(target_peer_bytes, request);
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
                    return SwarmResponse::ShardData(crate::types::ShardResponse::empty());
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
                        SwarmResponse::ShardData(crate::types::ShardResponse::empty())
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to open shard file");
                SwarmResponse::ShardData(crate::types::ShardResponse::empty())
            }
        }
    })
    .await;

    match result {
        Ok(response) => response,
        Err(e) => {
            tracing::warn!(error = %e, "spawn_blocking panicked during shard read");
            SwarmResponse::ShardData(crate::types::ShardResponse::empty())
        }
    }
}

/// The shortest wait before asking a busy holder again, whatever it named.
const BUSY_WAIT_LEAST: std::time::Duration = std::time::Duration::from_secs(1);

/// Busy answers in a row — no chunk landing between, of this part or of
/// another from the same holder (#340) — one transfer takes
/// before it is handled as a failure. Each wait is at most
/// `upload_pacer::RELEASED_WITHIN`, so a part gives up on holders that stay
/// busy within ~5 min.
const MOST_BUSY_ANSWERS: u32 = 16;

/// Transfers with busy-holder memory at once; only a bound.
const MAX_BUSY_ENTRIES: usize = 4096;

/// Holders of one part that answered BUSY, and until when (#335).
#[derive(Debug, Default)]
pub(super) struct BusyHolders {
    /// Busy answers since a chunk last landed, from every holder.
    answers: u32,
    /// Each busy holder and when its wait ends. A few holders at most.
    until: Vec<(crate::types::NodeId, std::time::Instant)>,
}

impl BusyHolders {
    fn note(&mut self, holder: crate::types::NodeId, until: std::time::Instant) {
        self.answers += 1;
        match self.until.iter_mut().find(|(n, _)| *n == holder) {
            Some(entry) => entry.1 = until,
            None => self.until.push((holder, until)),
        }
    }

    fn busy_at(&self, holder: &crate::types::NodeId, at: std::time::Instant) -> bool {
        self.until.iter().any(|(n, u)| n == holder && *u > at)
    }

    /// `holder` landed a chunk of another of this node's parts: a holder that
    /// is sending this node bytes has not "stayed busy", so the count starts
    /// again (#340).
    fn holder_is_serving_us(&mut self, holder: &crate::types::NodeId) {
        if self.until.iter().any(|(n, _)| n == holder) {
            self.answers = 0;
        }
    }
}

/// Where a downloader turned away BUSY asks next.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum NextAsk {
    /// Now, one of these holders (none of them busy).
    Now(Vec<crate::types::NodeId>),
    /// At this instant — when the first busy holder comes free — one of these.
    At(std::time::Instant, Vec<crate::types::NodeId>),
}

/// The holders worth asking after a busy answer: those not busy now; if every
/// holder is, those free when the first wait ends. The caller picks among
/// them with the same scoped choice as any request (`select_best_allowed_peer`).
pub(super) fn next_ask(
    busy: &BusyHolders,
    holders: &[crate::types::NodeId],
    now: std::time::Instant,
) -> NextAsk {
    let free: Vec<_> = holders
        .iter()
        .filter(|h| !busy.busy_at(h, now))
        .cloned()
        .collect();
    if !free.is_empty() {
        return NextAsk::Now(free);
    }
    let soonest = busy
        .until
        .iter()
        .filter(|(n, _)| holders.contains(n))
        .map(|(_, u)| *u)
        .min()
        .unwrap_or(now);
    let free_then = holders
        .iter()
        .filter(|h| !busy.busy_at(h, soonest))
        .cloned()
        .collect();
    NextAsk::At(soonest, free_then)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// After a busy answer the downloader asks a holder that is not busy at
    /// once; when every holder is busy it waits for the FIRST to come free —
    /// never for the one that just answered, if another is free sooner.
    #[test]
    fn a_busy_holder_is_passed_over_and_the_first_free_one_is_waited_for() {
        use crate::types::NodeId;
        use std::time::{Duration, Instant};
        let (a, b, c) = (NodeId([1; 32]), NodeId([2; 32]), NodeId([3; 32]));
        let now = Instant::now();
        let mut busy = BusyHolders::default();
        busy.note(a.clone(), now + Duration::from_secs(5));
        assert_eq!(
            next_ask(&busy, &[a.clone(), b.clone(), c.clone()], now),
            NextAsk::Now(vec![b.clone(), c.clone()])
        );
        busy.note(b.clone(), now + Duration::from_secs(3));
        busy.note(c.clone(), now + Duration::from_secs(9));
        assert_eq!(
            next_ask(&busy, &[a.clone(), b.clone(), c.clone()], now),
            NextAsk::At(now + Duration::from_secs(3), vec![b.clone()])
        );
        // A holder's newer wait replaces its older one, and every answer counts.
        busy.note(b.clone(), now + Duration::from_secs(7));
        assert_eq!(busy.answers, 4);
        assert_eq!(
            next_ask(&busy, &[a.clone(), b.clone(), c], now),
            NextAsk::At(now + Duration::from_secs(5), vec![a])
        );
        // A holder that has since gone from the registry is not waited for.
        assert_eq!(
            next_ask(&busy, std::slice::from_ref(&b), now),
            NextAsk::At(now + Duration::from_secs(7), vec![b])
        );
    }

    /// #340: a part waiting on a holder that lands this node's other parts
    /// has not met a holder that stayed busy — its count starts again — while
    /// a holder it is not waiting on changes nothing.
    #[test]
    fn a_part_waiting_on_a_holder_that_serves_its_siblings_keeps_waiting() {
        use crate::types::NodeId;
        use std::time::{Duration, Instant};
        let (a, b) = (NodeId([1; 32]), NodeId([2; 32]));
        let now = Instant::now();
        let mut busy = BusyHolders::default();
        for i in 0..MOST_BUSY_ANSWERS {
            busy.note(a.clone(), now + Duration::from_secs(u64::from(i)));
        }
        busy.holder_is_serving_us(&b);
        assert_eq!(busy.answers, MOST_BUSY_ANSWERS, "not a holder it waits on");
        busy.holder_is_serving_us(&a);
        assert_eq!(busy.answers, 0);
        assert!(busy.busy_at(&a, now), "the wait itself stands");
    }

    /// The serve side clamps to `SHARD_CHUNK_SIZE`, and a requester walks the
    /// file by the bytes it ACTUALLY received.
    ///
    /// Together these are what make the chunk size safe to change in a
    /// mixed-version swarm: a peer on an older build asks for 32 MiB and is
    /// served the current maximum, and its `new_offset = offset + chunk_len`
    /// loop simply takes more turns. Lowering the constant without both halves
    /// holding would strand every older peer mid-transfer.
    #[tokio::test]
    async fn a_chunk_request_is_clamped_and_the_walk_still_covers_the_file() {
        let dir = std::env::temp_dir().join(format!("swarmllm-chunk-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("shard_000.bin");

        // Deterministic contents so a reassembled copy can be compared byte for
        // byte, and not a multiple of the chunk size — the final short read is
        // exactly where an off-by-one would hide.
        let total: usize = (crate::network::protocol::SHARD_CHUNK_SIZE as usize * 2) + 12_345;
        let original: Vec<u8> = (0..total).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &original).unwrap();

        let model_id = crate::types::ModelId("chunk-test".to_string());
        let mut got: Vec<u8> = Vec::with_capacity(total);
        let mut offset: u64 = 0;
        let mut rounds = 0;

        loop {
            // Ask for far more than the limit, as an older peer would.
            let resp =
                read_shard_chunk_async(path.clone(), offset, u64::MAX / 2, model_id.clone(), 0)
                    .await;
            let SwarmResponse::ShardData(sr) = resp else {
                panic!("expected ShardData");
            };
            assert!(
                sr.data.len() as u64 <= crate::network::protocol::SHARD_CHUNK_SIZE,
                "server served {} bytes, above the {} limit — the clamp is gone, and a peer \
                 that asked for more would be handed a message the transport may refuse",
                sr.data.len(),
                crate::network::protocol::SHARD_CHUNK_SIZE
            );
            assert_eq!(
                sr.total_size, total as u64,
                "total_size must describe the file"
            );
            assert!(
                !sr.data.is_empty(),
                "a chunk inside the file must not be empty"
            );

            got.extend_from_slice(&sr.data);
            offset += sr.data.len() as u64;
            rounds += 1;
            assert!(rounds < 1000, "the walk is not terminating");
            if offset >= sr.total_size {
                break;
            }
        }

        assert_eq!(got.len(), total, "the walk did not cover the whole file");
        assert_eq!(
            got, original,
            "the reassembled shard differs from the original"
        );
        assert!(
            rounds >= 3,
            "expected the clamp to force several rounds for a file of {total} bytes, got {rounds}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
