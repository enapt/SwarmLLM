use axum::extract::State;
use axum::Json;
use serde::Deserialize;

use crate::api::server::AppState;
use crate::error::ApiError;
use crate::model::manifest::ModelManifestExt;

use super::{gguf_filename_to_model_id, progress::spawn_progress_updater, validate_hf_inputs};

#[derive(Debug, Deserialize)]
pub struct HfShardDownloadRequest {
    pub repo_id: String,
    pub filename: String,
    /// Which shard indices to download (e.g. [0,1,2] for the first 3 shards).
    /// If empty, the server will probe the file and return shard info without downloading.
    #[serde(default)]
    pub shards: Vec<u32>,
    /// Optional: target an existing model_id so downloaded shards merge into its directory.
    /// If omitted, a new model_id is derived from the filename.
    #[serde(default)]
    pub model_id: Option<String>,
    /// When true AND `shards` is empty: compute a deterministic fair share of shards
    /// based on the node's identity and peer count. Each node claims `ceil(shard_count / (peers + 1))`
    /// shards, with assignment determined by BLAKE3(node_id || model_id) for consistency.
    /// Peers with auto-manage enabled will auto-acquire the remaining shards.
    #[serde(default)]
    pub peer_fair_share: bool,
}

pub async fn hf_download_shards(
    State(state): State<AppState>,
    Json(body): Json<HfShardDownloadRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let repo_id = body.repo_id;
    let filename = body.filename;
    let shard_indices = body.shards;
    let peer_fair_share = body.peer_fair_share;

    validate_hf_inputs(&repo_id, &filename)?;

    if shard_indices.is_empty() && !peer_fair_share {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "shards array is required (e.g. [0, 1, 2])".into(),
        )));
    }

    // peer_fair_share is mutually exclusive with an explicit shards list —
    // the fair-share assignment computation only runs when shards is empty
    // (see `fair_share_peer_count` below). Mixing the two would silently
    // ignore peer_fair_share and the dashboard would show a misleading
    // "fair share mode" label on the resulting download.
    if peer_fair_share && !shard_indices.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "peer_fair_share is only meaningful when shards is empty — \
             pass either an explicit shards array OR peer_fair_share=true, \
             not both"
                .into(),
        )));
    }

    if shard_indices.len() > 256 {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Too many shards requested (max 256)".into(),
        )));
    }

    tracing::info!(
        repo_id = %repo_id,
        filename = %filename,
        shard_count = shard_indices.len(),
        peer_fair_share,
        "DIAG: hf_download_shards handler"
    );

    // peer_fair_share: compute shard assignment deterministically.
    // Deferred until after probe (we need shard_count), so store the peer count now.
    // Count only peers that are actually CONNECTED. `peer_registry` is
    // deliberately preserved across disconnects (see the scheduler-liveness
    // rule), so using it divides the model among peers that may be long gone —
    // this node then takes a small slice of a model nobody else is holding and
    // every request fails with "No node available for layer 0". Reported
    // 2026-07-29: a fair-share fetch of 4 shards left an unusable model while
    // none of the 4 connected peers advertised it at all.
    //
    // `connected_node_ids` is the same oracle the scheduler uses to decide who
    // can serve a segment, which is exactly the question being asked here.
    let fair_share_peer_count = if peer_fair_share && shard_indices.is_empty() {
        Some(state.shared_state.connected_node_ids.len())
    } else {
        None
    };
    let fair_share_node_id = state.shared_state.identity.node_id().clone();

    // Use provided model_id if it matches an existing model, otherwise derive from filename.
    // Always sanitize to prevent path traversal.
    let safe_name = if let Some(ref mid) = body.model_id {
        if mid.len() > 256 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "model_id must be 256 characters or fewer".into(),
            )));
        }
        let sanitized = crate::model::shard::sanitize_path_component(mid);
        if state.model_dir(&sanitized).exists() {
            sanitized
        } else {
            gguf_filename_to_model_id(&filename)
        }
    } else {
        gguf_filename_to_model_id(&filename)
    };

    let dest_dir = state.model_dir(&safe_name);

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        shards = ?shard_indices,
        dest = %dest_dir.display(),
        "Starting HuggingFace shard download"
    );

    let model_id_str = safe_name.clone();
    let mid = crate::types::ModelId(model_id_str.clone());

    // ── Trust: pin this model as user-approved ──────────────────────────
    // User explicitly chose to download → set Pinned trust level so auto-manage
    // will propagate shards for this model across the network.
    {
        let mut trust = state
            .shared_state
            .models
            .model_trust
            .entry(mid.clone())
            .or_insert_with(crate::types::ModelTrustInfo::new_pinned);
        if !trust.pinned_by_user {
            trust.pinned_by_user = true;
            if trust.trust_level < crate::types::ModelTrustLevel::Pinned {
                trust.trust_level = crate::types::ModelTrustLevel::Pinned;
            }
        }
        if let Err(e) = state
            .shared_state
            .db
            .put_json("model_trust", &mid.0, trust.value())
        {
            tracing::warn!(error = %e, model = %mid.0, "Failed to persist model trust pin — may be lost on restart");
        }
    }

    // ── Synchronous probe + architecture check ──────────────────────────
    // Probe before spawning the download task so we can return an immediate
    // HTTP error for unsupported architectures.
    //
    // **This is NOT fast**, whatever an earlier comment here claimed ("reads
    // ~few KB header"). `GGUF_HEADER_PROBE_SIZE` is **16 MB**, fetched as a
    // range request, on top of a HEAD — and both retry with 5/30/120s backoff.
    // On an ordinary home connection that is the ~25 seconds a tester reported
    // sitting on a zero-byte `.tmp` before anything appeared to happen
    // (2026-07-26). It is the FIRST thing a new user does after picking a
    // model, and a stalled-looking download is exactly when someone concludes
    // it is broken and kills it.
    //
    // The size is deliberate — large-vocabulary headers approach 10 MB, so the
    // margin avoids a second round trip — so the honest fix is to say what is
    // happening rather than to pretend it is instant.
    state.shared_state.emit_activity(
        crate::daemon::state::ActivityEvent::new(
            "download",
            "hf_probe_started",
            format!("Checking {filename} on HuggingFace before downloading"),
        )
        .with_detail_str(&repo_id)
        .with_toast("info", 4000),
    );
    let configured_shard_size = state.shared_state.config.model.shard_size_bytes();
    let info =
        crate::model::huggingface::probe_gguf_file(&repo_id, &filename, configured_shard_size)
            .await
            .map_err(|e| {
                tracing::error!(error = %e, "HuggingFace probe failed");
                // SEC / contract: HuggingFace is the upstream service, so a
                // probe failure is an upstream error (502 Bad Gateway), not
                // local "this server can't serve" (503). Matches the variant
                // used in probe.rs and search.rs and prevents leaking local-
                // vs-upstream topology via 503/502 differentiation.
                ApiError(crate::error::SwarmError::ProviderError {
                    status: 502,
                    body: format!(
                        "HuggingFace probe failed: {}",
                        crate::api::scrub_truncate_error(&e)
                    ),
                })
            })?;

    let arch_str = &info.tensor_meta.architecture;
    let model_arch = crate::inference::split::ModelArch::from_gguf_arch(arch_str);
    if !model_arch.is_supported() {
        let msg = format!(
            "Unsupported architecture '{}'. Supported: {}",
            arch_str,
            crate::inference::split::ModelArch::supported_list().join(", ")
        );
        tracing::warn!(%arch_str, "Refusing download: unsupported architecture");
        return Err(ApiError(crate::error::SwarmError::Validation(msg)));
    }

    // Create initial acquisition progress entry with per-shard progress so that
    // auto-manage can detect these downloads are already in flight and skip them.
    let log_msg = if peer_fair_share && shard_indices.is_empty() {
        format!("Computing fair share of {} from HuggingFace...", filename)
    } else {
        format!(
            "Downloading shards {:?} of {} from HuggingFace...",
            shard_indices, filename
        )
    };
    let mut initial_shard_progress = std::collections::HashMap::new();
    for &idx in &shard_indices {
        initial_shard_progress.insert(
            idx,
            crate::model::acquisition::ShardProgress::new_downloading(idx, 0),
        );
    }
    let mut status = crate::model::acquisition::AcquisitionStatus::new_downloading(
        mid.clone(),
        shard_indices.len() as u32,
        0,
        "huggingface",
        "user",
        log_msg,
    );
    status.shard_progress = initial_shard_progress;
    let shared = state.shared_state.clone();

    // Clone values needed both in the spawn and the response
    let response_model_id = model_id_str.clone();
    let response_shards = shard_indices.clone();

    // Register download: AcquisitionStatus + cancel flag atomically.
    let cancel_flag = shared.models.begin_download(mid.clone(), status);

    // Capture network_tx for broadcasting HfSourceGossip + ModelManifest after download
    let network_tx = state.network_tx.clone();

    tokio::spawn(async move {
        let shutdown_rx = shared.shutdown_rx();
        let download_mid = mid.clone();
        let download_shared = shared.clone();

        // peer_fair_share: download just ONE seed shard. Auto-manage handles the rest.
        // Each node picks a deterministic shard (based on node_id hash) so that
        // different nodes seed different shards when they add the same model.
        let shard_indices = if let Some(peer_count) = fair_share_peer_count {
            let total_shards = info.shard_count() as u32;

            // Deterministic shard selection: hash(node_id || model_id) → shard index
            let seed_shard = crate::types::hash_parts_to_u32(&[
                fair_share_node_id.0.as_ref(),
                model_id_str.as_bytes(),
            ]) % total_shards;

            let assigned = vec![seed_shard];

            tracing::info!(
                total_shards,
                peers = peer_count,
                seed_shard,
                "peer_fair_share: seeding 1 shard (auto-manage will acquire more as needed)"
            );

            // Update acquisition progress with the single seed shard
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.total_shards = 1;
                entry.log_push(format!(
                    "Seeding shard {seed_shard}/{total_shards} — auto-manage will acquire more as peers join"
                ));
                entry.shard_progress.insert(
                    seed_shard,
                    crate::model::acquisition::ShardProgress::new_downloading(seed_shard, 0),
                );
            }
            assigned
        } else {
            shard_indices
        };

        if let Some(mut entry) = download_shared
            .models
            .acquisition_progress
            .get_mut(&download_mid)
        {
            // Set total_bytes to the sum of requested shards only (not full model size)
            let requested_bytes: u64 = shard_indices
                .iter()
                .filter_map(|&idx| info.layouts.get(idx as usize))
                .map(|l| l.size_bytes)
                .sum();
            entry.total_bytes = requested_bytes;
            // Don't overwrite total_shards — keep as the requested count, not the full model count
            entry.log_push(format!(
                "Probed: {} shards, {:.1} MB total",
                info.shard_count(),
                info.total_size as f64 / (1024.0 * 1024.0)
            ));
            // Set per-shard total_bytes now that we know sizes from the probe
            for &idx in &shard_indices {
                if let Some(layout) = info.layouts.get(idx as usize) {
                    if let Some(sp) = entry.shard_progress.get_mut(&idx) {
                        sp.total_bytes = layout.size_bytes;
                    }
                }
            }
        }

        // Download GGUF header (~6MB) — needed for manifest generation
        if let Err(e) = crate::model::huggingface::download_gguf_header(
            &repo_id,
            &filename,
            &dest_dir,
            info.header_size,
        )
        .await
        {
            tracing::error!(error = %e, "GGUF header download failed");
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.state =
                    crate::model::acquisition::AcquisitionState::Failed { reason: e.clone() };
                entry.log_push(format!("Header download failed: {}", e));
            }
            download_shared.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "hf_download_failed",
                    format!("GGUF header download failed: {}", e),
                )
                .with_model(download_mid.0.clone())
                .with_detail_str(e)
                .with_toast("error", 6000),
            );
            return;
        }

        // Download tied output weight if model is weight-tied (no output.weight tensor).
        // This is needed by the last node in distributed inference for logit projection.
        if let Err(e) = crate::model::huggingface::download_tied_output_weight(
            &repo_id,
            &filename,
            &dest_dir,
            &info.tensor_meta,
        )
        .await
        {
            tracing::warn!(error = %e, "Tied output weight download failed (non-fatal)");
        }

        // Generate manifest from header BEFORE downloading shard data.
        // Pass empty shard_indices — no shards to register yet (they don't exist on disk).
        let header_path = dest_dir.join(crate::model::shard::HEADER_FILENAME);
        let manifest_result = generate_manifest_from_header(&ManifestGenParams {
            header_path: &header_path,
            model_id_str: &model_id_str,
            filename: &filename,
            total_size: info.total_size,
            shard_count: info.shard_count(),
            shard_indices: &[],
            shared: &download_shared,
            precomputed_layouts: Some(&info.layouts),
        });

        if let Err(e) = &manifest_result {
            tracing::error!(error = %e, "Manifest generation failed (early broadcast skipped)");
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.log_push(format!("Manifest generation failed: {e}"));
            }
            let display = download_shared.model_registry.display_name(&download_mid);
            download_shared.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "manifest_gen_failed",
                    format!(
                        "Early manifest generation failed for {} — will retry after shards download: {}",
                        display, e
                    ),
                )
                .with_model(download_mid.0.clone())
                .with_toast("warning", 6000),
            );
            // Continue with downloads anyway — manifest can be regenerated later
        }

        // Record HF source so auto-manager (and peers) know where to fetch shards
        let hf_source = crate::daemon::HfSource {
            repo_id: repo_id.clone(),
            filename: filename.clone(),
            mmproj_filename: None,
        };
        download_shared.models.hf_sources.insert(
            crate::types::ModelId(model_id_str.clone()),
            hf_source.clone(),
        );
        let _ = download_shared
            .db
            .put_json("hf_sources", &model_id_str, &hf_source);
        let hf_source_path = dest_dir.join(crate::model::shard::HF_SOURCE_FILENAME);
        let hf_source_json = serde_json::to_string_pretty(&hf_source).unwrap_or_default();
        let _ =
            tokio::task::spawn_blocking(move || std::fs::write(&hf_source_path, hf_source_json))
                .await;

        // Broadcast HfSourceGossip + ModelManifest EARLY so peers can start
        // auto-acquiring shards immediately (before our shard data downloads finish).
        if let Some(ref ntx) = network_tx {
            let gossip_msg =
                crate::types::SwarmMessage::HfSourceGossip(crate::types::HfSourceGossip {
                    model_id: crate::types::ModelId(model_id_str.clone()),
                    repo_id: repo_id.clone(),
                    filename: filename.clone(),
                    publisher: download_shared.identity.node_id().clone(),
                    mmproj_filename: None,
                });
            let _ = ntx
                .send(crate::types::NetworkCommand::Broadcast(gossip_msg))
                .await;

            if let Some(manifest) = download_shared
                .model_registry
                .get_manifest(&crate::types::ModelId(model_id_str.clone()))
            {
                let _ = ntx
                    .send(crate::types::NetworkCommand::Broadcast(
                        crate::types::SwarmMessage::ModelManifest(manifest),
                    ))
                    .await;
            }

            tracing::info!(model = %model_id_str, "Broadcast manifest + HF source EARLY (before shard downloads)");

            // Broadcast download intent for each shard so peers know we're
            // working on them and auto-manage won't duplicate the download.
            let our_node_id = download_shared.identity.node_id().clone();
            for &idx in &shard_indices {
                let intent_msg = crate::types::SwarmMessage::ShardDownloadProgress(
                    crate::types::ShardDownloadProgress {
                        node_id: our_node_id.clone(),
                        shard_id: crate::types::ShardId {
                            model_id: crate::types::ModelId(model_id_str.clone()),
                            index: idx,
                        },
                        progress_pct: 0,
                        state: crate::types::DownloadState::Downloading,
                    },
                );
                let _ = ntx
                    .send(crate::types::NetworkCommand::Broadcast(intent_msg))
                    .await;
            }
        }

        // NOTE: Do NOT wake auto-manage here. Shards aren't downloaded yet,
        // so holder_count == 0 and auto-manage would race to download them
        // from HF. The notify happens AFTER downloads complete (line ~1666).

        // ── Phase 2: Download shard data ────────────────────────────────

        let (ptx, prx) =
            tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);

        spawn_progress_updater(shared.clone(), mid.clone(), prx);

        // Download individual layer-aligned shards
        let total_shard_bytes: u64 = shard_indices
            .iter()
            .filter_map(|&idx| info.layouts.get(idx as usize))
            .map(|layout| layout.size_bytes)
            .sum();

        let mut cumulative_downloaded: u64 = 0;
        let mut failed = false;

        for &shard_idx in &shard_indices {
            // Check cancellation flag and shutdown before each shard download
            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) || *shutdown_rx.borrow() {
                let reason = if *shutdown_rx.borrow() {
                    "Cancelled by daemon shutdown"
                } else {
                    "Cancelled by user"
                };
                tracing::info!(model = %model_id_str, reason, "Download cancelled");
                if let Some(mut entry) = download_shared
                    .models
                    .acquisition_progress
                    .get_mut(&download_mid)
                {
                    entry.state = crate::model::acquisition::AcquisitionState::Failed {
                        reason: reason.to_string(),
                    };
                    entry.log_push(reason.to_string());
                }
                // Clean up cancel flag
                download_shared
                    .models
                    .download_cancel_flags
                    .remove(&download_mid);
                return;
            }

            let layout = match info.layouts.get(shard_idx as usize) {
                Some(l) => l,
                None => {
                    tracing::error!(
                        shard_idx,
                        max = info.layouts.len().saturating_sub(1),
                        "Shard index out of range"
                    );
                    failed = true;
                    break;
                }
            };

            let (shard_tx, mut shard_rx) =
                tokio::sync::mpsc::channel::<crate::model::huggingface::DownloadProgress>(64);
            let progress_tx = ptx.clone();
            let base_downloaded = cumulative_downloaded;
            let total = total_shard_bytes;
            let shard_progress_shared = shared.clone();
            let shard_progress_mid = mid.clone();
            let gossip_ntx = network_tx.clone();
            let gossip_node_id = shared.identity.node_id().clone();
            let gossip_model_id = model_id_str.clone();
            let progress_task = tokio::spawn(async move {
                let mut last_broadcast_pct: u32 = 0;
                while let Some(prog) = shard_rx.recv().await {
                    // Forward cumulative bytes to the overall progress updater
                    let _ = progress_tx.try_send(crate::model::huggingface::DownloadProgress {
                        downloaded_bytes: base_downloaded + prog.downloaded_bytes,
                        total_bytes: total,
                    });
                    // Update per-shard progress directly
                    if let Some(mut entry) = shard_progress_shared
                        .models
                        .acquisition_progress
                        .get_mut(&shard_progress_mid)
                    {
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                            sp.downloaded_bytes = prog.downloaded_bytes;
                            if sp.total_bytes == 0 {
                                sp.total_bytes = prog.total_bytes;
                            }
                        }
                    }
                    // Broadcast progress to peers every ~2% so they see near real-time updates
                    let pct = crate::model::acquisition::shard_pct(
                        prog.downloaded_bytes,
                        prog.total_bytes,
                    );
                    if let Some(ref ntx) = gossip_ntx {
                        let sid = crate::types::ShardId {
                            model_id: crate::types::ModelId(gossip_model_id.clone()),
                            index: shard_idx,
                        };
                        last_broadcast_pct =
                            crate::model::acquisition::maybe_broadcast_shard_progress(
                                ntx,
                                &gossip_node_id,
                                &sid,
                                pct,
                                last_broadcast_pct,
                                2,
                            );
                    }
                }
            });

            match crate::model::huggingface::download_shard(
                &repo_id,
                &filename,
                &dest_dir,
                layout,
                Some(shard_tx),
                Some(cancel_flag.as_ref()),
            )
            .await
            {
                Ok(_shard_path) => {
                    progress_task.abort();
                    cumulative_downloaded += layout.size_bytes;

                    if let Some(mut entry) = download_shared
                        .models
                        .acquisition_progress
                        .get_mut(&download_mid)
                    {
                        entry.downloaded_shards += 1;
                        entry.log_push(format!("Shard {} downloaded", shard_idx));
                        // Mark this shard's progress as complete so check_and_load_model
                        // won't skip it as "still downloading"
                        if let Some(sp) = entry.shard_progress.get_mut(&shard_idx) {
                            sp.state = crate::model::acquisition::ShardState::Complete;
                            sp.downloaded_bytes = sp.total_bytes;
                        }
                    }

                    // Register + announce the shard to the network
                    let shard_id = crate::types::ShardId {
                        model_id: crate::types::ModelId(model_id_str.clone()),
                        index: shard_idx,
                    };
                    if let Some(ref ntx) = network_tx {
                        download_shared.announce_shard_acquired(ntx, &shard_id);
                    } else {
                        // No network channel — just register locally
                        download_shared.model_registry.record_shard_holder(
                            shard_id,
                            download_shared.identity.node_id().clone(),
                        );
                    }
                }
                Err(e) => {
                    progress_task.abort();
                    tracing::error!(error = %e, shard_idx, "Shard download failed");
                    if let Some(mut entry) = download_shared
                        .models
                        .acquisition_progress
                        .get_mut(&download_mid)
                    {
                        entry.failed_shards += 1;
                        entry.log_push(format!("Shard {} failed: {}", shard_idx, e));
                    }
                    download_shared.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "download",
                            "shard_download_failed",
                            format!("Shard {} download failed: {}", shard_idx + 1, e),
                        )
                        .with_model(download_mid.0.clone())
                        .with_detail_num(shard_idx as i64)
                        .with_detail_str(e.to_string())
                        .with_toast("error", 6000),
                    );
                    failed = true;
                    break;
                }
            }
        }

        // Drop the progress sender so the updater task exits
        drop(ptx);

        // Clean up cancel flag
        download_shared
            .models
            .download_cancel_flags
            .remove(&download_mid);

        if failed {
            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.state = crate::model::acquisition::AcquisitionState::Failed {
                    reason: "One or more shard downloads failed".to_string(),
                };
            }
        } else {
            tracing::info!(
                model = %model_id_str,
                shards = ?shard_indices,
                "All shard downloads complete"
            );

            // Regenerate manifest with correct BLAKE3 hashes now that shard files
            // exist on disk. The early manifest had [0u8; 32] placeholders.
            if let Err(e) = generate_manifest_from_header(&ManifestGenParams {
                header_path: &header_path,
                model_id_str: &model_id_str,
                filename: &filename,
                total_size: info.total_size,
                shard_count: info.shard_count(),
                shard_indices: &shard_indices,
                shared: &download_shared,
                precomputed_layouts: Some(&info.layouts),
            }) {
                tracing::error!(error = %e, model = %model_id_str, "Final manifest regeneration failed after shard download");
                let display = download_shared.model_registry.display_name(&download_mid);
                download_shared.emit_activity(
                    crate::daemon::state::ActivityEvent::new(
                        "download",
                        "manifest_gen_failed",
                        format!(
                            "Could not finalize manifest for {} — model may not register correctly: {}",
                            display, e
                        ),
                    )
                    .with_model(download_mid.0.clone())
                    .with_toast("error", 8000),
                );
            }

            if let Some(mut entry) = download_shared
                .models
                .acquisition_progress
                .get_mut(&download_mid)
            {
                entry.state = crate::model::acquisition::AcquisitionState::Complete;
                entry.verified_shards = shard_indices.len() as u32;
                entry.log_push("All shards downloaded and registered".to_string());
            }

            // Load available shards for inference (partial is fine)
            let vram_budget = crate::model::auto_manage::compute_vram_budget(&download_shared);
            crate::model::auto_manage::check_and_load_model(
                &download_shared,
                &crate::types::ModelId(model_id_str.clone()),
                vram_budget,
            )
            .await;

            // Notify dashboard that models have changed
            download_shared.signal_dashboard(crate::daemon::state::DashboardSignal::ModelsChanged);

            // Emit activity event for HF download completion
            download_shared.emit_activity(
                crate::daemon::state::ActivityEvent::new(
                    "download",
                    "hf_download_complete",
                    format!("HuggingFace download complete: {}", model_id_str),
                )
                .with_model(model_id_str.clone())
                .with_detail_str("huggingface".to_string())
                .with_toast("success", 8000),
            );

            // Wake auto-manage again to re-evaluate (maybe download more shards)
            download_shared.models.auto_manage_notify.notify_one();

            // Clean up acquisition_progress after a delay so the frontend sees
            // the "complete" state and triggers a re-render before we remove it.
            download_shared.schedule_acquisition_cleanup(download_mid.clone());
        }
    });

    Ok(Json(serde_json::json!({
        "status": "started",
        "model_id": response_model_id,
        "shards": response_shards,
        "peer_fair_share": peer_fair_share,
    })))
}

struct ManifestGenParams<'a> {
    header_path: &'a std::path::Path,
    model_id_str: &'a str,
    filename: &'a str,
    total_size: u64,
    shard_count: u32,
    shard_indices: &'a [u32],
    shared: &'a std::sync::Arc<crate::daemon::SharedState>,
    /// Pre-computed layouts from probe. When provided, these are used directly
    /// instead of recomputing (avoids shard_count mismatch between probe and manifest).
    precomputed_layouts: Option<&'a [crate::inference::split::LayerShardLayout]>,
}

/// Generate a manifest from a downloaded GGUF header and register shards.
fn generate_manifest_from_header(params: &ManifestGenParams<'_>) -> Result<(), String> {
    use crate::inference::split::GgufTensorMeta;

    let header_path = params.header_path;
    let total_size = params.total_size;
    let shard_count = params.shard_count;

    // Parse model metadata from the GGUF header
    let meta = GgufTensorMeta::from_gguf_file(header_path)
        .map_err(|e| format!("Failed to parse GGUF header: {e}"))?;

    let model_id = crate::types::ModelId(params.model_id_str.to_string());
    let num_layers = meta.block_count as u32;

    // Build a friendly model name from the GGUF metadata or filename
    let model_name = meta
        .model_name
        .clone()
        .unwrap_or_else(|| params.filename.trim_end_matches(".gguf").to_string());

    // Architecture already extracted by GgufTensorMeta above — no need to re-read the file
    let architecture = crate::model::manifest::gguf_arch_to_model_architecture(&meta.architecture);

    let model_dir = header_path
        .parent()
        .ok_or_else(|| "GGUF header path has no parent directory".to_string())?;

    let computed_layouts;
    let layouts: &[crate::inference::split::LayerShardLayout] = if let Some(precomputed) =
        params.precomputed_layouts
    {
        precomputed
    } else {
        computed_layouts = crate::inference::split::compute_layer_shard_layouts(&meta, shard_count);
        &computed_layouts
    };
    let shards = crate::model::manifest::build_shard_infos_from_layouts(model_dir, layouts);

    let node_id = params.shared.identity.node_id().clone();

    let manifest = crate::model::manifest::build_manifest_from_gguf(
        crate::model::manifest::ManifestFromGguf {
            id: model_id.clone(),
            name: model_name,
            architecture,
            num_layers,
            total_size_bytes: total_size,
            shard_count,
            shards,
            publisher: node_id.clone(),
        },
    );

    // Save manifest to disk
    manifest.save_to_dir(model_dir).map_err(|e| e.to_string())?;

    // Register manifest in the model registry
    params
        .shared
        .model_registry
        .register_manifest(manifest.clone());

    // Store GGUF metadata
    params.shared.gguf_meta.insert(model_id.clone(), meta);

    // Register this node as holder of the downloaded shards
    for &shard_idx in params.shard_indices {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_idx,
        };
        params
            .shared
            .model_registry
            .record_shard_holder(shard_id, node_id.clone());
    }

    tracing::info!(
        model = %model_id,
        shards_registered = params.shard_indices.len(),
        num_layers,
        "Generated manifest and registered shards from HF download"
    );

    Ok(())
}
