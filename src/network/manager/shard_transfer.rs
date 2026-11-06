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
    /// `p2p_download_permits` (the slot holds the claim, from the first ask —
    /// #418); once the slot is gone the transfer is over, whether the claim
    /// went to somebody else or to nobody (`p2p_transfer_still_owns_the_file`).
    fn shard_transfer_lost_its_file(
        &self,
        shard_id: &crate::types::ShardId,
    ) -> Option<&'static str> {
        let models = &self.shared_state.models;
        if models.p2p_transfer_still_owns_the_file(shard_id) {
            None
        } else if models.shard_download_claimed(shard_id) {
            Some("another download owns this shard's file")
        } else {
            Some("this transfer was released — it stalled or was stopped")
        }
    }

    /// Forget this node's record of a part's peer transfer: where it stands,
    /// its clocks, retries and busy holders, and its asks still out — an answer
    /// to one of those finds no record and is dropped.
    fn forget_part_transfer(&mut self, shard_id: &crate::types::ShardId) {
        self.shard_download_progress.remove(shard_id);
        self.shard_last_progress_at.remove(shard_id);
        self.shard_p2p_retries.remove(shard_id);
        self.shard_busy_holders.remove(shard_id);
        self.pending_shard_requests
            .retain(|_, (_, sid)| sid != shard_id);
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
        self.forget_part_transfer(shard_id);
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

        self.forget_part_transfer(shard_id);

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
                    // The next computer carries on where the last one stopped
                    // (#191): the bytes on disk are kept, never fetched again.
                    let resume_at = self.resume_offset(&shard_id);
                    tracing::info!(
                        model = %shard_id.model_id,
                        shard = shard_id.index,
                        %failed_peer,
                        resume_at,
                        "Asking another computer for the rest of a part"
                    );
                    let retry_req = crate::types::ShardRequest {
                        shard_id: shard_id.clone(),
                        chunk_offset: resume_at,
                        chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE,
                    };
                    self.handle_send_shard_request(next_bytes, retry_req, PartAsk::Continues);
                    return true;
                }
            }
        }

        // Retries exhausted or no more peers — fall back to HuggingFace.
        // Release the P2P semaphore permit so the HF path (which acquires
        // its own permit in a separate trigger_download call) doesn't
        // starve behind the dead P2P request.
        self.shard_download_progress.remove(&shard_id);
        self.shard_last_progress_at.remove(&shard_id);
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
            .map(|t| t.next_offset)
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
            None => self.handle_send_shard_request(target_peer_bytes, request, PartAsk::Continues),
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
            && self
                .shard_download_progress
                .get(shard_id)
                .map(|t| t.next_offset)
                == Some(request.chunk_offset)
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
        self.handle_send_shard_request(target_peer_bytes, request, PartAsk::Continues);
    }

    /// Where a transfer that lost its computer carries on (#191): the next
    /// byte its record asks for, or the end of the partial file where that is
    /// shorter (a file removed or cut under it) — never past the bytes the disk
    /// holds. `write_chunk` writes with `std::fs` and no buffer of its own, so
    /// the file's length is what reached the system (gotcha #424 was a buffered
    /// `tokio::fs::File` on the origin path).
    fn resume_offset(&self, shard_id: &crate::types::ShardId) -> u64 {
        resume_from(
            self.shard_download_progress.get(shard_id),
            self.shard_store
                .tmp_size(&shard_id.model_id, shard_id.index),
        )
    }

    /// This node could not write a chunk it was sent (#191): its disk is full
    /// or refusing. No other computer can fix that, so the transfer ends —
    /// libtorrent pauses a torrent on a disk error — instead of fetching the
    /// rest of the part into a file with a hole in it, whose failed check
    /// would then have blamed the sender for our disk. The bytes written stay
    /// for the next attempt (which resumes from the file), the slot and the
    /// writer claim go, and the part backs off like any failed download.
    pub(super) fn end_transfer_after_a_write_failure(&mut self, shard_id: &crate::types::ShardId) {
        self.forget_part_transfer(shard_id);
        self.shared_state
            .models
            .p2p_download_permits
            .remove(shard_id);
        let (fails, backoff_secs) = self
            .shared_state
            .models
            .record_shard_download_failure(shard_id);
        if let Some(mut entry) = self
            .shared_state
            .models
            .acquisition_progress
            .get_mut(&shard_id.model_id)
        {
            entry.shard_progress.remove(&shard_id.index);
        }
        tracing::warn!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            fails,
            backoff_secs,
            "Stopped a part download this computer could not write — keeping what it wrote"
        );
    }

    /// Send a shard transfer request to a specific peer. `ask` says whether
    /// it starts a transfer (a new record) or carries one on — a retry, or an
    /// ask after a busy answer — keeping who sent the bytes before it.
    pub(super) fn handle_send_shard_request(
        &mut self,
        target_peer_bytes: Vec<u8>,
        request: crate::types::ShardRequest,
        ask: PartAsk,
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

        let shard_id = request.shard_id.clone();
        let chunk_offset = request.chunk_offset;
        // Resume support: seed the per-shard write offset from the request's
        // chunk_offset so the first received chunk lands at the resume position
        // (and write_chunk's truncate-on-zero-offset path doesn't wipe an
        // existing partial .tmp file). Recorded BEFORE the connection check
        // below: its failover resumes from this record, and with none it
        // would ask from 0 and truncate the partial file (#191).
        match ask {
            PartAsk::Starts => {
                // One transfer per part: a second start (the button pressed
                // twice) would write the same file from two chains. A record
                // whose slot is gone is a transfer ended out of this map's
                // reach (the stall sweep): forgotten, it blocks nothing.
                if self.shard_download_progress.contains_key(&shard_id) {
                    if self
                        .shared_state
                        .models
                        .p2p_transfer_still_owns_the_file(&shard_id)
                    {
                        tracing::info!(
                            model = %shard_id.model_id,
                            shard = shard_id.index,
                            "A peer download of this part is already under way — not starting another"
                        );
                        return;
                    }
                    self.forget_part_transfer(&shard_id);
                }
                // Every transfer owns its file from its first ask (#418).
                if !self.shared_state.models.hold_for_a_peer_transfer(&shard_id) {
                    tracing::info!(
                        model = %shard_id.model_id,
                        shard = shard_id.index,
                        "Another download is writing this part — not starting a peer transfer"
                    );
                    return;
                }
                self.shard_download_progress
                    .insert(shard_id.clone(), PartTransfer::starting_at(chunk_offset));
            }
            PartAsk::Continues => {
                self.shard_download_progress
                    .entry(shard_id.clone())
                    .and_modify(|t| t.next_offset = chunk_offset)
                    .or_insert_with(|| PartTransfer::starting_at(chunk_offset));
            }
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
        let req = SwarmRequest::ShardTransfer(request);
        // NET-C1: Track by OutboundRequestId for correct request-response correlation
        let outbound_id = self
            .swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer_id, req);
        self.pending_shard_requests
            .insert(outbound_id, (peer_id, shard_id.clone()));
        self.shard_last_progress_at
            .insert(shard_id, std::time::Instant::now());
    }
}

/// The offset a transfer carries on from: its record's next byte, never past
/// the `on_disk` bytes its partial file holds; 0 with no record (#191).
fn resume_from(record: Option<&PartTransfer>, on_disk: u64) -> u64 {
    record.map_or(0, |t| t.next_offset).min(on_disk)
}

/// Whether a part request starts a transfer or carries one on (#191) — a
/// REQUIRED argument of `handle_send_shard_request`, because only a transfer
/// that carries on may keep the record of who sent the bytes before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PartAsk {
    /// A download starting (the auto-manage pass, a "download" button): a
    /// new record — and bytes before its offset came from nobody on record.
    Starts,
    /// The same transfer asking again: after a busy answer, or of another
    /// computer once one stopped answering.
    Continues,
}

/// One part's peer transfer as it stands (#191): the byte the next chunk is
/// asked for, and who sent the bytes before it.
///
/// Prior art: BitTorrent keeps the blocks of a piece whichever peers sent
/// them, and a piece that fails its hash implicates every peer that took
/// part — libtorrent's smart ban hashes each block to find the one that
/// differed ("smart ban", blog.libtorrent.org, 2011). A part here has one
/// hash, of the whole file, so a part with several senders blames none: a
/// failed check implicates a computer only when every byte came from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct PartTransfer {
    /// The offset the next chunk is asked for; the bytes before it are on disk.
    pub(super) next_offset: u64,
    senders: PartSenders,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartSenders {
    /// Nothing written yet, and nothing before the transfer's start.
    Nobody,
    /// Every byte from this one computer.
    Only(libp2p::PeerId),
    /// Several computers — or bytes from before the transfer started, which
    /// nobody recorded (a partial file an earlier attempt or run left).
    Several,
}

impl PartTransfer {
    /// A transfer starting at `offset`.
    pub(super) fn starting_at(offset: u64) -> Self {
        Self {
            next_offset: offset,
            senders: if offset == 0 {
                PartSenders::Nobody
            } else {
                PartSenders::Several
            },
        }
    }

    /// `len` bytes from `peer` were written at `offset`. A write at 0
    /// truncates the file first, so whoever sent it is then the only sender.
    pub(super) fn landed(&mut self, peer: libp2p::PeerId, offset: u64, len: u64) {
        self.senders = match self.senders {
            _ if offset == 0 => PartSenders::Only(peer),
            PartSenders::Only(only) if only == peer => PartSenders::Only(only),
            _ => PartSenders::Several,
        };
        self.next_offset = offset.saturating_add(len);
    }

    /// The one computer every byte came from — the only one a failed check
    /// may blame.
    pub(super) fn sole_sender(&self) -> Option<libp2p::PeerId> {
        match self.senders {
            PartSenders::Only(peer) => Some(peer),
            PartSenders::Nobody | PartSenders::Several => None,
        }
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

    /// #191: a part names its sender only while every byte came from one
    /// computer; a second sender, or bytes from before the transfer began,
    /// name none — and a write at 0, which truncates, starts over.
    #[test]
    fn a_part_names_its_sender_only_while_every_byte_came_from_it() {
        let (a, b) = (libp2p::PeerId::random(), libp2p::PeerId::random());
        let mut t = PartTransfer::starting_at(0);
        assert_eq!(t.sole_sender(), None, "nothing written yet");
        t.landed(a, 0, 8);
        t.landed(a, 8, 8);
        assert_eq!((t.sole_sender(), t.next_offset), (Some(a), 16));
        t.landed(b, 16, 8);
        assert_eq!(
            (t.sole_sender(), t.next_offset),
            (None, 24),
            "a second sender"
        );
        t.landed(a, 24, 8);
        assert_eq!(t.sole_sender(), None, "several stays several");
        t.landed(b, 0, 8);
        assert_eq!(
            (t.sole_sender(), t.next_offset),
            (Some(b), 8),
            "a write at 0 truncates the file first"
        );

        // Resumed from a partial file an earlier attempt left: its first
        // bytes came from nobody on record.
        let mut resumed = PartTransfer::starting_at(16);
        resumed.landed(a, 16, 8);
        assert_eq!(resumed.sole_sender(), None);
    }

    /// #191: a transfer that lost its computer carries on from its record,
    /// never past what its file holds — and resuming there writes the part
    /// whole, byte for byte, keeping what the first computer sent.
    #[test]
    fn a_part_resumed_on_another_computer_keeps_its_bytes_and_ends_whole() {
        let dir = std::env::temp_dir().join(format!(
            "swarmllm-resume-test-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let store = crate::model::shard::ShardStore::new(&dir);
        let model = crate::types::ModelId("resume-test".to_string());
        let part: Vec<u8> = (0..40_000u32).map(|i| (i % 251) as u8).collect();
        let (a, b) = (libp2p::PeerId::random(), libp2p::PeerId::random());

        // The first computer sends two chunks, then stops answering.
        let mut t = PartTransfer::starting_at(0);
        for (at, len) in [(0usize, 16_000usize), (16_000, 16_000)] {
            store
                .write_chunk(&model, 3, at as u64, &part[at..at + len])
                .unwrap();
            t.landed(a, at as u64, len as u64);
        }
        let on_disk = store.tmp_size(&model, 3);
        assert_eq!(resume_from(Some(&t), on_disk), 32_000);
        assert_eq!(resume_from(None, on_disk), 0, "no record: from the start");
        assert_eq!(
            resume_from(Some(&t), 10_000),
            10_000,
            "a file cut short is resumed from its end"
        );

        // The next computer sends the rest from there.
        let at = resume_from(Some(&t), on_disk);
        store
            .write_chunk(&model, 3, at, &part[at as usize..])
            .unwrap();
        t.landed(b, at, part.len() as u64 - at);
        let written = std::fs::read(store.shard_tmp_path(&model, 3)).unwrap();
        assert!(written == part, "the resumed part is not the part");
        assert_eq!(t.sole_sender(), None, "two senders");
        let _ = std::fs::remove_dir_all(&dir);
    }

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
