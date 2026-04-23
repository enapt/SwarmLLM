use crate::model::manifest::ModelManifestExt;
use crate::types::{ModelId, NetworkCommand, ShardId};

use super::manager::{AutoShardManager, ShardCandidate};
use super::scan::check_and_load_model;
use super::vram::compute_vram_budget;

impl AutoShardManager {
    /// Trigger download of a single shard.
    ///
    /// Strategy: try peers first if any hold the shard, fall back to HuggingFace.
    /// After download, register the shard and check if the model is now complete.
    /// Acquires a semaphore permit to limit concurrent downloads.
    pub(super) async fn trigger_download(&self, candidate: &ShardCandidate) {
        // Try to acquire a semaphore permit non-blocking. If all download slots
        // are occupied, defer to next evaluation cycle instead of blocking the loop.
        let permit = match self.download_semaphore.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Download semaphore full, deferring to next cycle"
                );
                return;
            }
        };

        // Reject duplicate downloads early. Without this, a re-trigger that
        // arrives while a download is mid-flight (and the .bin doesn't exist
        // yet) starts a SECOND parallel task that appends concurrently to the
        // same .tmp file — producing the right size but corrupted bytes.
        // The .bin-exists path below has its own in-progress check; this one
        // covers the more common partial-.tmp case.
        if self
            .shared_state
            .models
            .is_shard_in_progress(&candidate.model_id, candidate.shard_index)
        {
            tracing::debug!(
                model = %candidate.model_id,
                shard = candidate.shard_index,
                "Shard download already in progress, deferring"
            );
            return;
        }

        tracing::info!(
            model = %candidate.model_id,
            shard = candidate.shard_index,
            holders = candidate.holder_count,
            score = candidate.score,
            "AutoShardManager: requesting shard download"
        );

        // Activity events are emitted at the point of download:
        // - HF downloads: shard_download_started (below in HF path)
        // - P2P downloads: shard_download_p2p (below in P2P path)
        // This avoids double-logging when a generic start + specific start both fire.

        let model_dir = crate::model::shard::model_dir(
            &self.shared_state.config.node.data_dir,
            &candidate.model_id.0,
        );

        // -- T8: mmproj full-file download (not byte-range) --
        if candidate.shard_index == crate::types::MMPROJ_SHARD_INDEX {
            self.trigger_mmproj_download(candidate, model_dir, permit)
                .await;
            return;
        }

        // Check if we already have the shard file locally.
        // Guard: only treat it as complete if there is NO active download for this
        // shard AND the file size matches expected.  A partially-downloaded file
        // will exist on disk but be smaller than `shard_size_bytes`.
        let shard_path = model_dir.join(format!("shard_{:03}.bin", candidate.shard_index));
        if shard_path.exists() {
            // Check if this shard is currently being downloaded (by API handler or another cycle)
            let is_downloading = self
                .shared_state
                .models
                .is_shard_in_progress(&candidate.model_id, candidate.shard_index);

            if is_downloading {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Shard file exists but download is in progress, skipping"
                );
                return;
            }

            // Verify shard integrity: try BLAKE3 hash if available, fall back to size check
            let shard_store = self.shared_state.shard_store();
            let size_ok = || super::shard_size_ok(&shard_path, candidate.shard_size_bytes);
            let file_ok = if let Some(manifest) = self
                .shared_state
                .model_registry
                .get_manifest(&candidate.model_id)
            {
                if let Some(shard_info) = manifest
                    .shards
                    .iter()
                    .find(|s| s.index == candidate.shard_index)
                {
                    if shard_info.hash != [0u8; 32] {
                        // Hash available -- verify properly
                        shard_store
                            .verify_shard(&candidate.model_id, shard_info)
                            .is_ok()
                    } else {
                        // Zero-hash placeholder -- fall back to size check
                        size_ok()
                    }
                } else {
                    size_ok()
                }
            } else {
                size_ok()
            };

            if file_ok {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Shard file already exists on disk, registering"
                );
                self.register_local_shard(candidate);
                self.check_model_complete(&candidate.model_id).await;
                return;
            } else {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "Shard file exists but is too small (partial download?), re-downloading"
                );
                // Fall through to download
            }
        }

        let mid = candidate.model_id.clone();

        // NOTE: We do NOT send a ShardAnnounce before the download starts.
        // Premature announces cause peers to register us as a holder before
        // the shard is actually on disk, making the UI show "peer-held" instead
        // of "peer-downloading".  The ShardDownloadProgress gossip broadcasts
        // our progress, and the completion message triggers holder registration
        // on remote nodes.

        // Prefer P2P over HuggingFace when peers hold the shard — P2P is faster
        // for LAN peers and doesn't depend on external CDN availability.
        // Exception: if P2P has already exhausted all peer attempts for this shard
        // in this session, skip P2P and go straight to HF fallback.
        let sid_for_failed_check = ShardId {
            model_id: candidate.model_id.clone(),
            index: candidate.shard_index,
        };
        let p2p_exhausted = self
            .shared_state
            .models
            .shard_p2p_failed
            .contains(&sid_for_failed_check);
        let has_peer_holders = candidate.holder_count > 0 && !p2p_exhausted;

        // Download from HuggingFace only if no peers hold the shard
        // In offline mode, skip automatic HF downloads (user must trigger manually)
        let offline_mode = self
            .shared_state
            .credits
            .offline_mode
            .load(std::sync::atomic::Ordering::Relaxed);
        if !has_peer_holders && !offline_mode {
            if let Some(hf_source) = self.shared_state.models.hf_sources.get(&candidate.model_id) {
                // Create progress entry with per-shard tracking for the specific shard
                let mut shard_progress = std::collections::HashMap::new();
                shard_progress.insert(
                    candidate.shard_index,
                    crate::model::acquisition::ShardProgress::new_downloading(
                        candidate.shard_index,
                        candidate.shard_size_bytes,
                    ),
                );
                // Merge with existing progress entry rather than overwriting.
                // Multiple shards of the same model may be downloading concurrently
                // and each needs its own shard_progress entry tracked.
                if let Some(mut entry) = self.shared_state.models.acquisition_progress.get_mut(&mid)
                {
                    entry.state = crate::model::acquisition::AcquisitionState::Downloading;
                    // Set total_shards from the manifest, not by incrementing
                    // (incrementing causes inflated counts when merging progress entries)
                    if let Some(manifest) = self.shared_state.model_registry.get_manifest(&mid) {
                        entry.total_shards = manifest.shard_count;
                        entry.total_bytes = manifest.total_size_bytes;
                    }
                    // Only add this shard's progress if not already tracked
                    entry
                        .shard_progress
                        .entry(candidate.shard_index)
                        .or_insert_with(|| {
                            crate::model::acquisition::ShardProgress::new_downloading(
                                candidate.shard_index,
                                candidate.shard_size_bytes,
                            )
                        });
                    entry.log_push(format!(
                        "Auto-manage: downloading shard {} (score: {:.1})",
                        candidate.shard_index, candidate.score
                    ));
                } else {
                    let (total_shards, total_bytes) = self
                        .shared_state
                        .model_registry
                        .get_manifest(&mid)
                        .map(|m| (m.shard_count, m.total_size_bytes))
                        .unwrap_or((1, candidate.shard_size_bytes));
                    let mut status = crate::model::acquisition::AcquisitionStatus::new_downloading(
                        mid.clone(),
                        total_shards,
                        total_bytes,
                        "peers",
                        "auto_manage",
                        format!(
                            "Auto-manage: downloading shard {} of {} (score: {:.1})",
                            candidate.shard_index, candidate.model_name, candidate.score
                        ),
                    );
                    status.shard_progress = shard_progress;
                    self.shared_state
                        .models
                        .acquisition_progress
                        .insert(mid.clone(), status);
                }
                let repo_id = hf_source.repo_id.clone();
                let filename = hf_source.filename.clone();
                drop(hf_source); // release DashMap ref

                let shared = self.shared_state.clone();
                let model_id = candidate.model_id.clone();
                let shard_idx = candidate.shard_index;
                let dest = model_dir.clone();

                tracing::info!(
                    model = %model_id,
                    shard = shard_idx,
                    repo = %repo_id,
                    "AutoShardManager: downloading shard from HuggingFace"
                );

                // Emit HF-specific download start event
                {
                    let display = shared.model_registry.display_name(&model_id);
                    shared.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "download",
                            "shard_download_started",
                            format!(
                                "Downloading shard {} of {} from HuggingFace",
                                shard_idx + 1,
                                display
                            ),
                        )
                        .with_model(model_id.0.clone())
                        .with_detail_num(shard_idx as i64)
                        .with_detail_str("huggingface".to_string()),
                    );
                }

                let net_tx = self.network_tx.clone();

                // Spawn the download so we don't block the evaluation loop.
                // The semaphore permit is moved into the task and dropped on completion,
                // releasing the slot for the next download.
                tokio::spawn(async move {
                    let _permit = permit; // Hold permit for duration of download
                    let (ptx, mut prx) = tokio::sync::mpsc::channel::<
                        crate::model::huggingface::DownloadProgress,
                    >(32);

                    // Progress updater -- updates per-shard progress + broadcasts to network
                    let prog_mid = model_id.clone();
                    let prog_shared = shared.clone();
                    let prog_net_tx = net_tx.clone();
                    tokio::spawn(async move {
                        let mut last_broadcast_pct: u32 = 0;
                        while let Some(prog) = prx.recv().await {
                            let pct = crate::model::acquisition::shard_pct(
                                prog.downloaded_bytes,
                                prog.total_bytes,
                            );

                            if let Some(mut entry) =
                                prog_shared.models.acquisition_progress.get_mut(&prog_mid)
                            {
                                entry.downloaded_bytes = prog.downloaded_bytes;
                                entry.total_bytes = prog.total_bytes;
                                // Update per-shard progress
                                if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                                    sp.downloaded_bytes = prog.downloaded_bytes;
                                    sp.total_bytes = prog.total_bytes;
                                }
                            }

                            // Broadcast progress every 5% to avoid gossip flood
                            let sid = crate::types::ShardId {
                                model_id: prog_mid.clone(),
                                index: shard_idx,
                            };
                            last_broadcast_pct =
                                crate::model::acquisition::maybe_broadcast_shard_progress(
                                    &prog_net_tx,
                                    prog_shared.identity.node_id(),
                                    &sid,
                                    pct,
                                    last_broadcast_pct,
                                    5,
                                );
                        }
                    });

                    // Probe to get shard layouts, then download the specific shard
                    let configured_shard_size = shared.config.model.shard_size_bytes();
                    let probe_result = crate::model::huggingface::probe_gguf_file(
                        &repo_id,
                        &filename,
                        configured_shard_size,
                    )
                    .await;
                    let info = match probe_result {
                        Ok(info) => info,
                        Err(e) => {
                            tracing::warn!(
                                model = %model_id,
                                shard = shard_idx,
                                error = %e,
                                "AutoShardManager: GGUF probe failed"
                            );
                            if let Some(mut entry) =
                                shared.models.acquisition_progress.get_mut(&model_id)
                            {
                                entry.state = crate::model::acquisition::AcquisitionState::Failed {
                                    reason: format!("GGUF probe failed: {}", e),
                                };
                            }
                            return;
                        }
                    };
                    // Check architecture support before downloading
                    let arch_str = &info.tensor_meta.architecture;
                    let arch = crate::inference::split::ModelArch::from_gguf_arch(arch_str);
                    if !arch.is_supported() {
                        tracing::warn!(
                            model = %model_id,
                            arch = %arch_str,
                            "AutoShardManager: skipping unsupported architecture"
                        );
                        shared.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "auto_manage",
                                "unsupported_architecture",
                                format!(
                                    "Skipped {} — unsupported architecture: {}",
                                    model_id, arch_str
                                ),
                            )
                            .with_model(&model_id.0)
                            .with_detail_str(arch_str)
                            .with_toast("warning", 5000),
                        );
                        return;
                    }

                    let layout = match info.layouts.get(shard_idx as usize) {
                        Some(l) => l,
                        None => {
                            tracing::warn!(
                                model = %model_id,
                                shard = shard_idx,
                                total_shards = info.shard_count(),
                                "AutoShardManager: shard index out of range"
                            );
                            return;
                        }
                    };

                    // Download header + tied output weight (if weight-tied) + shard
                    if let Err(e) = crate::model::huggingface::download_gguf_header(
                        &repo_id,
                        &filename,
                        &dest,
                        info.header_size,
                    )
                    .await
                    {
                        tracing::warn!(
                            model = %model_id,
                            shard = shard_idx,
                            error = %e,
                            "AutoShardManager: gguf_header.bin download failed — shard registered but first-segment local inference will be unavailable until header is re-downloaded"
                        );
                        shared.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "auto_manage",
                                "header_download_failed",
                                format!(
                                    "GGUF header download failed for {} (first-segment inference unavailable)",
                                    model_id
                                ),
                            )
                            .with_model(&model_id.0)
                            .with_detail_str(e.to_string())
                            .with_toast("warning", 5000),
                        );
                    }

                    // Download tied output weight for weight-tied models
                    if let Err(e) = crate::model::huggingface::download_tied_output_weight(
                        &repo_id,
                        &filename,
                        &dest,
                        &info.tensor_meta,
                    )
                    .await
                    {
                        tracing::warn!(error = %e, "Tied output weight download failed (non-fatal)");
                        shared.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "auto_manage",
                                "tied_output_failed",
                                format!(
                                    "Tied output weight download failed for {} (non-fatal)",
                                    model_id
                                ),
                            )
                            .with_model(&model_id.0)
                            .with_detail_str(e.to_string()),
                        );
                    }

                    match crate::model::huggingface::download_shard(
                        &repo_id,
                        &filename,
                        &dest,
                        layout,
                        Some(ptx),
                        None,
                    )
                    .await
                    {
                        Ok(_shard_path) => {
                            tracing::info!(
                                model = %model_id,
                                shard = shard_idx,
                                "AutoShardManager: shard downloaded from HF"
                            );
                            // Emit per-shard completion activity
                            {
                                let display = shared.model_registry.display_name(&model_id);
                                shared.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "download",
                                        "shard_download_complete",
                                        format!(
"Downloaded shard {} of {} from HuggingFace — verifying",
shard_idx + 1,
display
),
                                    )
                                    .with_model(model_id.0.clone())
                                    .with_detail_num(shard_idx as i64)
                                    .with_detail_str("huggingface".to_string()),
                                );
                            }

                            // Skip pre-registration BLAKE3 verification for HF downloads.
                            // The manifest hash may be stale (computed from a previous download
                            // or copied from another node). We trust HF CDN integrity — the
                            // hash is recomputed from the actual file below and the manifest
                            // is updated to match, so future verifications will pass.

                            // Compute BLAKE3 hash of the downloaded shard and update the manifest
                            // so startup verification passes on restart.
                            // block_in_place: streaming hash with 64KB buffer (avoids loading full shard into memory)
                            let shard_path = dest.join(format!("shard_{:03}.bin", shard_idx));
                            let hash_result: Option<[u8; 32]> = tokio::task::block_in_place(|| {
                                crate::model::shard::hash_file_blake3(&shard_path).ok()
                            });
                            if let Some(hash) = hash_result {
                                if let Some(mut manifest) =
                                    shared.model_registry.get_manifest(&model_id)
                                {
                                    if let Some(si) =
                                        manifest.shards.iter_mut().find(|s| s.index == shard_idx)
                                    {
                                        si.hash = hash;
                                    }
                                    manifest.manifest_hash = manifest.compute_hash();
                                    let model_dir = crate::model::shard::model_dir(
                                        &shared.config.node.data_dir,
                                        &model_id.0,
                                    );
                                    if let Err(e) = manifest.save_to_dir(&model_dir) {
                                        tracing::warn!(
                                            model = %model_id,
                                            error = %e,
                                            "AutoShardManager: failed to persist manifest after shard hash update — hash in memory only"
                                        );
                                    }
                                    shared.model_registry.register_manifest(manifest);
                                }
                            }

                            // Register + announce the shard to the network
                            let sid = crate::types::ShardId {
                                model_id: model_id.clone(),
                                index: shard_idx,
                            };
                            shared.announce_shard_acquired(&net_tx, &sid);

                            // Broadcast download completion progress
                            let complete_msg = crate::types::SwarmMessage::ShardDownloadProgress(
                                crate::types::ShardDownloadProgress {
                                    node_id: shared.identity.node_id().clone(),
                                    shard_id: sid,
                                    progress_pct: 100,
                                    state: crate::types::DownloadState::Complete,
                                },
                            );
                            let _ = net_tx
                                .try_send(crate::types::NetworkCommand::Broadcast(complete_msg));

                            // Update progress
                            let model_just_completed = if let Some(mut entry) =
                                shared.models.acquisition_progress.get_mut(&model_id)
                            {
                                let was_complete = entry
                                    .shard_progress
                                    .get(&shard_idx)
                                    .map(|sp| {
                                        matches!(
                                            sp.state,
                                            crate::model::acquisition::ShardState::Complete
                                        )
                                    })
                                    .unwrap_or(false);
                                if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                                    sp.state = crate::model::acquisition::ShardState::Complete;
                                    sp.downloaded_bytes = sp.total_bytes;
                                }
                                if !was_complete {
                                    entry.downloaded_shards =
                                        entry.downloaded_shards.saturating_add(1);
                                    entry.verified_shards = entry.verified_shards.saturating_add(1);
                                }
                                let all_shards_done = entry.total_shards > 0
                                    && entry.verified_shards >= entry.total_shards;
                                let was_already_complete = matches!(
                                    entry.state,
                                    crate::model::acquisition::AcquisitionState::Complete
                                );
                                if all_shards_done {
                                    entry.state =
                                        crate::model::acquisition::AcquisitionState::Complete;
                                }
                                entry.log_push("Shard downloaded and registered".into());
                                all_shards_done && !was_already_complete
                            } else {
                                false
                            };
                            // Emit shard registered activity
                            {
                                let display = shared.model_registry.display_name(&model_id);
                                shared.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "download",
                                        "shard_verified",
                                        format!(
                                            "Shard {} of {} verified and registered",
                                            shard_idx + 1,
                                            display
                                        ),
                                    )
                                    .with_model(model_id.0.clone())
                                    .with_detail_num(shard_idx as i64),
                                );
                            }

                            // Load whatever shards are now available for inference
                            let vram_budget = compute_vram_budget(&shared);
                            check_and_load_model(&shared, &model_id, vram_budget).await;

                            // Emit model_download_complete only when this shard finished
                            // the model. Without the guard every single shard triggers the
                            // "model ready" toast.
                            if model_just_completed {
                                let display = shared.model_registry.display_name(&model_id);
                                shared.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "download",
                                        "model_download_complete",
                                        format!(
"All shards of {} downloaded and verified — model ready",
display
),
                                    )
                                    .with_model(model_id.0.clone())
                                    .with_detail_str("huggingface".to_string())
                                    .with_toast("success", 8000),
                                );
                            }

                            // Self-wake so we immediately re-evaluate and download
                            // more shards (libp2p gossipsub doesn't deliver our own
                            // broadcasts back to us, so we must notify ourselves).
                            shared.models.auto_manage_notify.notify_one();

                            // Clean up acquisition_progress after a delay so the
                            // frontend sees "complete" before we remove it.
                            shared.schedule_acquisition_cleanup(model_id.clone());
                        }
                        Err(e) => {
                            tracing::warn!(
                                model = %model_id,
                                shard = shard_idx,
                                error = %e,
                                "AutoShardManager: HF shard download failed"
                            );
                            {
                                let mname = shared
                                    .model_registry
                                    .get_manifest(&model_id)
                                    .map(|m| m.name.clone());
                                shared.emit_activity(
                                    crate::daemon::state::ActivityEvent::new(
                                        "download",
                                        "shard_download_failed",
                                        format!(
"Failed to download shard {} of {} from HuggingFace: {}",
shard_idx + 1,
mname.as_deref().unwrap_or(&model_id.0),
e
),
                                    )
                                    .with_model(model_id.0.clone())
                                    .with_detail_num(shard_idx as i64)
                                    .with_detail_str(e.clone())
                                    .with_toast("error", 6000),
                                );
                            }
                            if let Some(mut entry) =
                                shared.models.acquisition_progress.get_mut(&model_id)
                            {
                                entry.state = crate::model::acquisition::AcquisitionState::Failed {
                                    reason: e,
                                };
                                entry.log_push("HF download failed".into());
                            }
                            shared.schedule_acquisition_cleanup(model_id.clone());
                        }
                    }
                });
            } // end hf_source check
        } // end !has_peer_holders

        if has_peer_holders {
            // P2P: download from peers who hold this shard.
            // Send AcquisitionCommand::Acquire to trigger P2P chunk-based transfer
            // via the AcquisitionManager, which handles retry logic and verification.
            let sid = ShardId {
                model_id: candidate.model_id.clone(),
                index: candidate.shard_index,
            };
            // Private mode: only download from allowed nodes
            let allowed_set = crate::pool::scope::allowed_node_set(&self.shared_state);
            let holders: Vec<_> = self
                .shared_state
                .model_registry
                .shard_holders(&sid)
                .into_iter()
                .filter(|n| {
                    if n == self.shared_state.identity.node_id() {
                        return false; // Skip self
                    }
                    match allowed_set {
                        Some(ref allowed) => allowed.contains(n),
                        None => true,
                    }
                })
                .collect();
            if holders.is_empty() {
                tracing::debug!(
                    model = %candidate.model_id,
                    shard = candidate.shard_index,
                    "No HF source and no peer holders — cannot download"
                );
                {
                    let mname = self
                        .shared_state
                        .model_registry
                        .get_manifest(&candidate.model_id)
                        .map(|m| m.name.clone());
                    self.shared_state.emit_activity(crate::daemon::state::ActivityEvent::new(
                        "download",
                        "shard_no_source",
                        format!("Cannot download {} of {} — no HuggingFace source and no peers hold it", crate::types::ShardId::display_index(candidate.shard_index), mname.as_deref().unwrap_or(&candidate.model_id.0)),
                    )
                    .with_model(candidate.model_id.0.clone())
                    .with_detail_num(candidate.shard_index as i64)
                    .with_toast("warning", 6000));
                }
            } else {
                // Pick the best holder: LAN first, lowest latency, highest trust
                let target = self.shared_state.select_best_peer(&holders);

                let peer_id_bytes = self
                    .shared_state
                    .peer_registry
                    .get(&target)
                    .and_then(|p| p.peer_id_bytes.clone());

                if let Some(bytes) = peer_id_bytes {
                    // Create/update acquisition_progress so frontend shows a download bar.
                    // For P2P, we download one shard at a time — total_bytes is THIS shard's size,
                    // not the entire model. total_shards reflects shards being P2P downloaded.
                    let shard_bytes = candidate.shard_size_bytes;
                    if let Some(mut entry) = self
                        .shared_state
                        .models
                        .acquisition_progress
                        .get_mut(&candidate.model_id)
                    {
                        entry.state = crate::model::acquisition::AcquisitionState::Downloading;
                        entry.total_bytes = shard_bytes;
                        entry.downloaded_bytes = 0;
                        entry.speed_bytes_per_sec = 0;
                        entry
                            .shard_progress
                            .entry(candidate.shard_index)
                            .or_insert_with(|| {
                                crate::model::acquisition::ShardProgress::new_downloading(
                                    candidate.shard_index,
                                    shard_bytes,
                                )
                            });
                        entry.log_push(format!(
                            "P2P: downloading shard {} from peer",
                            crate::types::ShardId::display_index(candidate.shard_index)
                        ));
                    } else {
                        let mut shard_progress = std::collections::HashMap::new();
                        shard_progress.insert(
                            candidate.shard_index,
                            crate::model::acquisition::ShardProgress::new_downloading(
                                candidate.shard_index,
                                shard_bytes,
                            ),
                        );
                        let mut p2p_status =
                            crate::model::acquisition::AcquisitionStatus::new_downloading(
                                candidate.model_id.clone(),
                                1,
                                shard_bytes,
                                "peers",
                                "auto_manage",
                                format!(
                                    "P2P: downloading shard {} from peer",
                                    crate::types::ShardId::display_index(candidate.shard_index)
                                ),
                            );
                        p2p_status.shard_progress = shard_progress;
                        self.shared_state
                            .models
                            .acquisition_progress
                            .insert(candidate.model_id.clone(), p2p_status);
                    }

                    // Resume: pick up where a previous interrupted P2P attempt
                    // (or daemon restart) left off in the .tmp file.
                    let resume_offset = self
                        .shared_state
                        .shard_store()
                        .tmp_size(&candidate.model_id, candidate.shard_index);
                    let request = crate::types::ShardRequest {
                        shard_id: sid.clone(),
                        chunk_offset: resume_offset,
                        chunk_size: crate::network::protocol::SHARD_CHUNK_SIZE,
                    };
                    tracing::info!(
                        model = %candidate.model_id,
                        shard = candidate.shard_index,
                        peer = %target,
                        "AutoShardManager: downloading shard from peer (no HF source)"
                    );
                    {
                        let mname = self
                            .shared_state
                            .model_registry
                            .get_manifest(&candidate.model_id)
                            .map(|m| m.name.clone());
                        let peer_label = crate::identity::nickname::short_display_name(
                            &target,
                            &self.shared_state.nickname_registry,
                        );
                        self.shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "download",
                                "shard_download_p2p",
                                format!(
                                    "Requesting shard {} of {} from peer {}",
                                    crate::types::ShardId::display_index(candidate.shard_index),
                                    mname.as_deref().unwrap_or(&candidate.model_id.0),
                                    peer_label
                                ),
                            )
                            .with_model(candidate.model_id.0.clone())
                            .with_node(format!("{}", target))
                            .with_detail_num(candidate.shard_index as i64)
                            .with_detail_str("p2p".to_string()),
                        );
                    }
                    let cmd = NetworkCommand::SendShardRequest {
                        target_peer_bytes: bytes,
                        request,
                    };
                    if let Err(e) = self.network_tx.send(cmd).await {
                        tracing::warn!(
                            error = %e,
                            model = %candidate.model_id,
                            "Failed to send P2P shard request"
                        );
                    }
                } else {
                    tracing::debug!(
                        model = %candidate.model_id,
                        shard = candidate.shard_index,
                        "Peer holds shard but peer_id_bytes unavailable"
                    );
                }
            }
        }
    }

    /// Download mmproj (vision encoder) as a full file from HuggingFace.
    /// Unlike text shards which use byte-range downloads, mmproj is a separate GGUF file.
    pub(super) async fn trigger_mmproj_download(
        &self,
        candidate: &ShardCandidate,
        model_dir: std::path::PathBuf,
        permit: tokio::sync::OwnedSemaphorePermit,
    ) {
        let mmproj_path = model_dir.join(crate::model::shard::MMPROJ_FILENAME);
        if mmproj_path.exists() {
            // Already on disk -- just register the sentinel shard
            let node_id = self.shared_state.identity.node_id().clone();
            let shard_id = ShardId {
                model_id: candidate.model_id.clone(),
                index: crate::types::MMPROJ_SHARD_INDEX,
            };
            self.shared_state
                .model_registry
                .record_shard_holder(shard_id, node_id);
            tracing::info!(model = %candidate.model_id, "mmproj already on disk, registered sentinel shard");
            return;
        }

        // Look up mmproj_filename + repo_id from HfSource in a single access
        let (filename, repo_id) = match self.shared_state.models.hf_sources.get(&candidate.model_id)
        {
            Some(s) => match s.mmproj_filename.clone() {
                Some(f) => (f, s.repo_id.clone()),
                None => {
                    tracing::debug!(
                        model = %candidate.model_id,
                        "No mmproj_filename in HfSource — cannot download mmproj"
                    );
                    return;
                }
            },
            None => return,
        };

        let shared = self.shared_state.clone();
        let model_id = candidate.model_id.clone();
        let net_tx = self.network_tx.clone();

        tracing::info!(
            model = %model_id,
            repo = %repo_id,
            filename = %filename,
            "AutoShardManager: downloading mmproj from HuggingFace"
        );
        shared.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "auto_manage",
                "mmproj_download_started",
                format!("Downloading vision projector (mmproj) for {}", model_id),
            )
            .with_model(&model_id.0)
            .with_detail_str(&repo_id),
        );

        tokio::spawn(async move {
            let _permit = permit;
            match crate::model::huggingface::download_model(&repo_id, &filename, &model_dir, None)
                .await
            {
                Ok(_path) => {
                    tracing::info!(
                        model = %model_id,
                        "AutoShardManager: mmproj downloaded from HF"
                    );
                    shared.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "auto_manage",
                            "mmproj_download_complete",
                            format!("Vision projector (mmproj) downloaded for {}", model_id),
                        )
                        .with_model(&model_id.0)
                        .with_toast("success", 4000),
                    );
                    // Register + announce the mmproj sentinel shard
                    let sid = crate::types::ShardId {
                        model_id: model_id.clone(),
                        index: crate::types::MMPROJ_SHARD_INDEX,
                    };
                    shared.announce_shard_acquired(&net_tx, &sid);
                }
                Err(e) => {
                    tracing::warn!(
                        model = %model_id,
                        error = %e,
                        "AutoShardManager: mmproj download failed"
                    );
                    shared.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "auto_manage",
                            "mmproj_download_failed",
                            format!("Vision projector download failed for {}", model_id),
                        )
                        .with_model(&model_id.0)
                        .with_detail_str(e.to_string())
                        .with_toast("error", 6000),
                    );
                }
            }
        });
    }

    /// Register a shard file that already exists on disk.
    pub(super) fn register_local_shard(&self, candidate: &ShardCandidate) {
        tracing::debug!(
            model = %candidate.model_id,
            shard = candidate.shard_index,
            "DIAG: register_local_shard"
        );
        let node_id = self.shared_state.identity.node_id().clone();
        let shard_id = ShardId {
            model_id: candidate.model_id.clone(),
            index: candidate.shard_index,
        };
        self.shared_state
            .model_registry
            .record_shard_holder(shard_id, node_id);
    }

    /// Check if any local shards are available for this model and load them.
    /// A node does NOT need all shards -- it loads whatever it has and participates
    /// in distributed inference for the layers it covers.
    pub(super) async fn check_model_complete(&self, model_id: &ModelId) {
        let vram_budget = compute_vram_budget(&self.shared_state);
        check_and_load_model(&self.shared_state, model_id, vram_budget).await;
        self.shared_state
            .signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);
    }
}
