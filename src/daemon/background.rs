//! Post-spawn background tasks kicked off by the daemon's run loop.
//!
//! Each `spawn_*` function fires off a `tokio::spawn` that runs until the
//! shutdown channel fires (or the task completes naturally). These are
//! best-effort background chores: shard verification, IP geolocation,
//! startup broadcasts, key rotation, opening a browser, auto-loading
//! models, and handling SIGHUP config reloads.

use std::path::PathBuf;
use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use crate::config::Config;
use crate::types::{NetworkCommand, SwarmMessage};

use super::helpers::{detect_region_from_ip, open_browser};
use super::state::SharedState;

/// BLAKE3 hash check runs after API is up so the dashboard is responsive
/// immediately. Bad shards are quarantined.
pub(super) fn spawn_shard_verification(
    shared_state: Arc<SharedState>,
    data_dir: PathBuf,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        // Small delay to let the API server bind and first WS clients connect
        tokio::select! {
            _ = shutdown_rx.changed() => { return; }
            _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {}
        }
        let shard_store = crate::model::shard::ShardStore::new(&data_dir);
        let mut verified = 0u32;
        let mut quarantined = 0u32;
        for manifest in shared_state.model_registry.models() {
            for shard_info in &manifest.shards {
                // Only verify shards we registered (i.e., we are a holder)
                let sid = crate::types::ShardId {
                    model_id: manifest.id.clone(),
                    index: shard_info.index,
                };
                if !shared_state
                    .model_registry
                    .shard_holders(&sid)
                    .contains(shared_state.identity.node_id())
                {
                    continue;
                }
                let shard_path = shard_store.shard_path(&manifest.id, shard_info.index);
                if !shard_path.exists() {
                    continue;
                }
                // Skip zero-hash shards (hash not yet computed)
                if shard_info.hash == [0u8; 32] {
                    verified += 1;
                    continue;
                }
                // Run BLAKE3 verification in a blocking thread
                let mid = manifest.id.clone();
                let si = shard_info.clone();
                let store = crate::model::shard::ShardStore::new(&data_dir);
                let result =
                    tokio::task::spawn_blocking(move || store.verify_shard(&mid, &si)).await;
                match result {
                    Ok(Ok(())) => {
                        verified += 1;
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(
                            model = %manifest.id,
                            shard = shard_info.index,
                            error = %e,
                            "Background shard verification failed — quarantined"
                        );
                        shared_state
                            .model_registry
                            .remove_shard_holder(&sid, shared_state.identity.node_id());
                        shared_state.emit_activity(
                            crate::daemon::state::ActivityEvent::new(
                                "model",
                                "shard_verify_failed",
                                format!(
                                    "Shard {} of {} failed verification — quarantined",
                                    crate::types::ShardId::display_index_short(shard_info.index),
                                    manifest.name
                                ),
                            )
                            .with_model(manifest.id.0.clone())
                            .with_model_name(manifest.name.clone())
                            .with_detail_num(shard_info.index as i64)
                            .with_detail_str(format!("{e}"))
                            .with_toast("warning", 5000),
                        );
                        quarantined += 1;
                    }
                    Err(_) => {
                        // JoinError — spawn_blocking panicked
                        quarantined += 1;
                    }
                }
            }
        }
        if quarantined > 0 {
            tracing::warn!(
                verified,
                quarantined,
                "Background shard verification complete — some shards quarantined"
            );
        } else {
            tracing::info!(
                verified,
                "Background shard verification complete — all shards OK"
            );
        }
        shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "system",
                "shard_verified",
                format!(
                    "Verified {} shards{}",
                    verified,
                    if quarantined > 0 {
                        format!(" ({quarantined} quarantined)")
                    } else {
                        String::new()
                    }
                ),
            )
            .with_detail_num(verified as i64),
        );
    });
}

/// Auto-detect region via IP geolocation (non-blocking, best-effort). If the
/// user configured a region explicitly, apply that instead.
pub(super) fn spawn_region_detection(
    shared_state: Arc<SharedState>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    if shared_state.config.identity.region.is_none() {
        let geo_state = shared_state.clone();
        tokio::spawn(async move {
            tokio::select! {
                result = detect_region_from_ip() => {
                    match result {
                        Some(code) => {
                            tracing::info!(region = %code, "Auto-detected region via IP geolocation");
                            *geo_state.detected_region.write().await = Some(code);
                        }
                        None => {
                            tracing::debug!(
                                "IP geolocation unavailable — network map will show unknown region"
                            );
                        }
                    }
                }
                _ = shutdown_rx.changed() => {}
            }
        });
    } else {
        let state = shared_state.clone();
        tokio::spawn(async move {
            *state.detected_region.write().await = state.config.identity.region.clone();
        });
    }
}

/// Broadcast shard announcements and manifests shortly after startup so peers
/// discover our shards quickly (don't wait for the 30s health tick).
pub(super) fn spawn_initial_announcements(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        // Wait for peer connections to establish, abort on shutdown
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {}
            _ = shutdown_rx.changed() => { return; }
        }

        let node_id = shared_state.identity.node_id().clone();

        // Broadcast shard announcements
        let hosted_shards = shared_state.model_registry.shards_for_node(&node_id);

        if !hosted_shards.is_empty() {
            // S5: Register as DHT provider for local shards
            let _ = network_tx
                .send(NetworkCommand::StartProviding(hosted_shards.clone()))
                .await;

            let announce = crate::types::ShardAnnounce {
                node_id: node_id.clone(),
                shards: hosted_shards,
                timestamp: chrono::Utc::now(),
            };
            tracing::info!(
                shards = announce.shards.len(),
                "Broadcasting initial shard announcement"
            );
            let _ = network_tx
                .send(NetworkCommand::Broadcast(SwarmMessage::ShardAnnounce(
                    announce,
                )))
                .await;
        }

        // Broadcast manifests for models where we hold at least one shard
        // (not just models we originally published). This allows shard-holding
        // nodes to propagate manifests to pure-consumer peers.
        let hosted_models: std::collections::HashSet<String> = shared_state
            .model_registry
            .all_shard_entries()
            .into_iter()
            .filter_map(|(shard_id, holders)| {
                if holders.contains(&node_id) {
                    Some(shard_id.model_id.0.clone())
                } else {
                    None
                }
            })
            .collect();
        for manifest in shared_state.model_registry.models() {
            if manifest.publisher == node_id || hosted_models.contains(&manifest.id.0) {
                let _ = network_tx
                    .send(NetworkCommand::Broadcast(SwarmMessage::ModelManifest(
                        manifest,
                    )))
                    .await;
            }
        }
    });
}

/// Spawn key rotation task (evicts stale sessions + ephemeral re-keying).
pub(super) fn spawn_key_rotation(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
) {
    let sm = shared_state.session_manager.clone();
    let node_id = shared_state.identity.node_id().clone();
    tokio::spawn(async move {
        crate::crypto::key_rotation::run_key_rotation(
            sm,
            network_tx,
            node_id,
            shared_state,
            shutdown_rx,
        )
        .await;
    });
}

/// Open browser on first start if configured.
pub(super) fn spawn_browser_open(config: &Config, mut shutdown_rx: watch::Receiver<bool>) {
    if !config.ui.open_browser_on_start {
        return;
    }
    let url = format!("http://localhost:{}", config.node.listen_port);
    // Check if config file exists — if not, open setup wizard
    let config_path = config.node.data_dir.join("config.toml");
    let target = if config_path.exists() {
        format!("{url}/admin")
    } else {
        format!("{url}/setup")
    };
    tokio::spawn(async move {
        // Small delay to let the server bind, abort on shutdown
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_millis(500)) => {
                if let Err(e) = open_browser(&target) {
                    tracing::debug!(error = %e, "Could not open browser automatically");
                }
            }
            _ = shutdown_rx.changed() => {}
        }
    });
}

/// Item 8 Phase 1: drain the model-process pool's prefix-cache manifest
/// channel and (a) gossip each update as `SwarmMessage::PrefixCacheAnnounce`
/// so peers can index our cached blocks, (b) fold our own blocks into the
/// loopback view by recording them under our `node_id` in the cross-node
/// index. This lets a single-node setup verify the wire path end-to-end —
/// after sending a prompt we should see our own NodeId returned by
/// `cross_node_prefix_holders` on the matching block hashes.
pub(super) fn spawn_prefix_announce_forwarder(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut rx: mpsc::Receiver<crate::inference::process_pool::PrefixManifestEvent>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        let our_id = shared_state.identity.node_id().clone();
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => break,
                msg = rx.recv() => {
                    let Some(event) = msg else { break };
                    let block_count = event.blocks.len();
                    // Loopback verification: record our own blocks under our
                    // NodeId so a single-node test can `cross_node_prefix_holders`
                    // and observe end-to-end indexing without needing a peer.
                    let block_hashes: Vec<[u8; 32]> = event
                        .blocks
                        .iter()
                        .map(|b| b.block_hash)
                        .collect();
                    let (added, removed) = shared_state.models.replace_peer_prefix_blocks(
                        our_id.clone(),
                        event.model_id.clone(),
                        block_hashes,
                    );
                    tracing::debug!(
                        model = %event.model_id,
                        blocks = block_count,
                        added,
                        removed,
                        "DIAG: prefix-cache loopback indexed (self)"
                    );
                    let announce = crate::types::PrefixCacheAnnounce {
                        node_id: our_id.clone(),
                        model_id: event.model_id,
                        blocks: event.blocks,
                        timestamp: chrono::Utc::now(),
                    };
                    let cmd = NetworkCommand::Broadcast(SwarmMessage::PrefixCacheAnnounce(announce));
                    if let Err(e) = network_tx.send(cmd).await {
                        tracing::debug!(error = %e, "prefix-cache announce: network_tx send failed");
                    }
                }
            }
        }
    });
}

/// Auto-load models that have local shards available. Popular models
/// (by historical request count) are loaded first so they get VRAM priority
/// on restart.
pub(super) fn spawn_model_autoload(
    shared_state: Arc<SharedState>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        // Brief delay to let shard announcements propagate, abort on shutdown
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(2)) => {}
            _ = shutdown_rx.changed() => { return; }
        }
        let mut manifests = shared_state.model_registry.models();
        manifests.sort_by(|a, b| {
            let count_a = shared_state
                .models
                .model_request_counts
                .get(&a.id)
                .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0);
            let count_b = shared_state
                .models
                .model_request_counts
                .get(&b.id)
                .map(|c| c.value().load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0);
            count_b.cmp(&count_a)
        });
        for m in &manifests {
            if shared_state.has_split_model(&m.id) {
                continue;
            }
            // Recompute VRAM budget each iteration since loading a model consumes VRAM
            let vram_budget = crate::model::auto_manage::compute_vram_budget(&shared_state);
            crate::model::auto_manage::check_and_load_model(&shared_state, &m.id, vram_budget)
                .await;
        }
    });
}

/// SIGHUP config reload handler. No-op on non-Unix platforms.
#[cfg(unix)]
pub(super) fn spawn_sighup_handler(
    shared_state: Arc<SharedState>,
    config: Config,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        let mut sighup = match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::hangup())
        {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!(error = %e, "Failed to register SIGHUP handler — config reload via signal disabled");
                return;
            }
        };
        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = sighup.recv() => {
                    let config_path = config.node.data_dir.join("config.toml");
                    tracing::info!(
                        "SIGHUP received — reloading config from {}",
                        config_path.display()
                    );
                    match crate::config::reload_operational_params(&config_path) {
                        Ok(params) => {
                            let old = crate::config::OperationalParams::from_config(&config);
                            if params != old {
                                tracing::info!(
                                    ?params,
                                    "Config reloaded with changes"
                                );
                            } else {
                                tracing::info!(
                                    "Config reloaded — no changes detected"
                                );
                            }
                            shared_state.apply_config_reload(params);
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                "Failed to reload config on SIGHUP"
                            );
                        }
                    }
                }
            }
        }
    });
}

#[cfg(not(unix))]
pub(super) fn spawn_sighup_handler(
    _shared_state: Arc<SharedState>,
    _config: Config,
    _shutdown_rx: watch::Receiver<bool>,
) {
}
