use std::sync::Arc;

use tokio::sync::{mpsc, watch, RwLock};

use crate::identity::nickname::NicknameRecordExt;
use crate::inference::router::RouterCommand;
use crate::model::manifest::ModelManifestExt;
use crate::types::{AuthenticatedMessage, EphemeralKeyExchange, NetworkCommand, SwarmMessage};

use super::state::{SharedState, TpAllReduceCollector};

/// Maximum number of concurrent LayerForward tasks.
const MAX_CONCURRENT_FORWARDS: usize = 64;
/// Maximum concurrent forwards per individual peer to prevent single-peer semaphore exhaustion.
const MAX_FORWARDS_PER_PEER: usize = 8;
/// Zstd compression level for tensor wire payloads.
const ZSTD_COMPRESS_LEVEL: i32 = 3;
/// Maximum age (ms) for regional gossip messages before they're considered stale.
const GOSSIP_STALENESS_MS: u64 = 15 * 60 * 1000;

/// Pipeline sealing: encrypt the token IDs in a LayerResult for the requester's X25519 key.
/// If `requester_node_id` is present, seals `token_ids` into `sealed_token_ids` and clears
/// the plaintext `token_ids`. Falls back silently on crypto errors (result sent unsealed).
fn seal_layer_result(result: &mut crate::types::LayerResult, requester_node_id: Option<&[u8; 32]>) {
    let requester_bytes = match requester_node_id {
        Some(b) => b,
        None => return,
    };
    if result.token_ids.is_empty() {
        return; // Only seal final-segment results that have token IDs
    }
    let requester_x25519 = match crate::crypto::session::ed25519_pubkey_to_x25519(requester_bytes) {
        Some(pk) => pk,
        None => {
            tracing::warn!(request_id = %result.request_id, "Pipeline seal: invalid requester pubkey");
            return;
        }
    };
    // Serialize token IDs to JSON bytes, then seal
    let token_json = serde_json::to_vec(&result.token_ids).unwrap_or_default();
    match crate::crypto::pipeline_seal::seal_prompt(
        result.request_id,
        &token_json,
        &requester_x25519,
    ) {
        Ok(sealed) => {
            match serde_json::to_vec(&sealed) {
                Ok(sealed_bytes) => {
                    tracing::debug!(
                        request_id = %result.request_id,
                        num_tokens = result.token_ids.len(),
                        "Pipeline seal: sealed token IDs for requester"
                    );
                    result.sealed_token_ids = Some(sealed_bytes);
                    result.token_ids.clear(); // Don't send plaintext
                }
                Err(e) => {
                    tracing::warn!(
                        request_id = %result.request_id,
                        error = %e,
                        "Pipeline seal: failed to serialize SealedPrompt — clearing plaintext"
                    );
                    // SEC: Never send plaintext tokens when sealing was intended
                    result.token_ids.clear();
                }
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %result.request_id,
                error = %e,
                "Pipeline seal: encryption failed — clearing plaintext"
            );
            // SEC: Never send plaintext tokens when sealing was intended
            result.token_ids.clear();
        }
    }
}

/// Track inference participation: increment forwards_served and earn credits (non-blocking).
/// `max_layers` caps the credited range to the model's actual layer count,
/// preventing credit inflation from forged layer_range values.
/// Estimate VRAM usage from shard files on disk (no model loading).
pub fn estimate_vram_from_shard_dir(
    model_dir: &std::path::Path,
    layer_start: usize,
    layer_end: usize,
    total_layers: usize,
) -> u64 {
    let total_bytes: u64 = std::fs::read_dir(model_dir)
        .into_iter()
        .flatten()
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name();
            let name = name.to_str()?;
            if name.starts_with("shard_") && name.ends_with(".bin") {
                entry.metadata().ok().map(|m| m.len())
            } else {
                None
            }
        })
        .sum();
    if total_layers == 0 {
        return total_bytes / (1024 * 1024);
    }
    let layer_fraction = (layer_end - layer_start) as f64 / total_layers as f64;
    ((total_bytes as f64 * layer_fraction) / (1024.0 * 1024.0)) as u64
}

fn track_forward_participation(
    shared_state: &SharedState,
    _layer_start: usize,
    _layer_end: usize,
    _max_layers: usize,
) {
    if let Ok(mut stats) = shared_state.node_stats.try_write() {
        stats.forwards_served += 1;
    }
    // Credits are earned per-token (not per-layer) to stay balanced with the
    // consume side. For the forward path, we earn a fixed per-forward amount
    // since we don't know the total token count yet (single decode step).
    // The pipeline orchestrator earns the bulk credit at completion.
    let earned = crate::credit::ledger::RATE_INFERENCE_SERVE; // 1 token per forward step
                                                              // Use atomic accumulator to prevent credit loss on lock contention.
                                                              // The CreditLedger periodic persist (every 60s) will flush this to DB.
    shared_state
        .pending_credit_earn
        .fetch_add(earned, std::sync::atomic::Ordering::Relaxed);
}

/// Dispatch inbound network messages to the appropriate subsystem.
///
/// Inference-related messages (InferenceRequest, LayerForward, LayerResult,
/// InferenceError, PipelineAssignment) are routed to the InferenceRouter.
/// CreditGossip messages are used to update the peer balance distribution.
/// Other messages (health, discovery) are handled by their respective
/// subsystems directly via SharedState or are already handled by NetworkManager.
pub(crate) async fn dispatch_network_messages(
    network_out_rx: &mut mpsc::Receiver<AuthenticatedMessage>,
    router_tx: &mpsc::Sender<RouterCommand>,
    credit_peer_balances: Arc<RwLock<Vec<i64>>>,
    shared_state: &Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let forward_semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_CONCURRENT_FORWARDS));
    // SEC: Per-peer concurrent forward counter to prevent single-peer semaphore exhaustion
    let peer_forward_counts: Arc<
        dashmap::DashMap<crate::types::NodeId, std::sync::atomic::AtomicUsize>,
    > = Arc::new(dashmap::DashMap::new());
    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    break;
                }
            }
            authed_msg = network_out_rx.recv() => {
                match authed_msg {
                    Some(AuthenticatedMessage { sender: authenticated_sender, message: msg }) => {
                        match msg {
                            // LayerResult: route to pending pipeline executor via oneshot channel
                            SwarmMessage::LayerResult(ref result) => {
                                let sender = match authenticated_sender {
                                    Some(ref s) => s,
                                    None => {
                                        tracing::warn!("LayerResult from unauthenticated peer — dropping");
                                        continue;
                                    }
                                };
                                if !shared_state.peer_registry.contains_key(sender) {
                                    tracing::warn!(sender = %sender, "LayerResult from unknown peer — dropping");
                                    continue;
                                }
                                tracing::info!(
                                    request_id = %result.request_id,
                                    tokens = result.token_ids.len(),
                                    activations_bytes = result.activations.len(),
                                    finish = ?result.finish_reason,
                                    pending_count = shared_state.pending_layer_results.len(),
                                    "DIAG: dispatcher received LayerResult"
                                );
                                if let Some((_, tx)) = shared_state
                                    .pending_layer_results
                                    .remove(&result.request_id)
                                {
                                    if tx.send(result.clone()).is_err() {
                                        tracing::warn!(
                                            request_id = %result.request_id,
                                            tokens = result.token_ids.len(),
                                            finish = ?result.finish_reason,
                                            "DIAG: LayerResult delivered but pipeline receiver DROPPED"
                                        );
                                    } else {
                                        tracing::info!(
                                            request_id = %result.request_id,
                                            tokens = result.token_ids.len(),
                                            activations_bytes = result.activations.len(),
                                            finish = ?result.finish_reason,
                                            pending_remaining = shared_state.pending_layer_results.len(),
                                            "DIAG: LayerResult delivered to pipeline"
                                        );
                                    }
                                } else {
                                    tracing::warn!(
                                        request_id = %result.request_id,
                                        tokens = result.token_ids.len(),
                                        finish = ?result.finish_reason,
                                        pending_count = shared_state.pending_layer_results.len(),
                                        "DIAG: No pending channel for LayerResult — already timed out or duplicate"
                                    );
                                }
                            }
                            // LayerForward: process locally using split inference engine,
                            // then send back a LayerResult to the requesting node.
                            SwarmMessage::LayerForward(forward) => {
                                if let Some(ref sender) = authenticated_sender {
                                    if !shared_state.peer_registry.contains_key(sender) {
                                        tracing::warn!(sender = %sender, "LayerForward from unknown peer — dropping");
                                        continue;
                                    }
                                } else {
                                    tracing::warn!("LayerForward without authenticated sender — dropping");
                                    continue;
                                }
                                tracing::info!(
                                    request_id = %forward.request_id,
                                    seq = forward.sequence_num,
                                    layer_range = ?forward.layer_range,
                                    activation_bytes = forward.activations.len(),
                                    has_sender = forward.sender_peer_bytes.is_some(),
                                    "DIAG: dispatcher received LayerForward, spawning handler"
                                );
                                // SEC: Per-peer concurrent forward limit to prevent single-peer exhaustion
                                let peer_sender = authenticated_sender.clone().expect("guarded by Some check above");
                                let peer_count = peer_forward_counts
                                    .entry(peer_sender.clone())
                                    .or_insert_with(|| std::sync::atomic::AtomicUsize::new(0));
                                let current = peer_count.load(std::sync::atomic::Ordering::Relaxed);
                                if current >= MAX_FORWARDS_PER_PEER {
                                    tracing::warn!(
                                        sender = %peer_sender,
                                        current,
                                        max = MAX_FORWARDS_PER_PEER,
                                        "LayerForward rejected — per-peer limit reached"
                                    );
                                    continue;
                                }
                                peer_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let permit = match forward_semaphore.clone().try_acquire_owned() {
                                    Ok(p) => p,
                                    Err(_) => {
                                        // Decrement unconditionally — use the entry ref we already hold
                                        // to avoid racing with concurrent DashMap removal
                                        peer_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                                        tracing::warn!("LayerForward rejected — forward semaphore full");
                                        continue;
                                    }
                                };
                                let ss = shared_state.clone();
                                let ntx = network_tx.clone();
                                let pfc = peer_forward_counts.clone();
                                let ps = peer_sender;
                                tokio::spawn(async move {
                                    let _permit = permit;
                                    handle_layer_forward(ss, ntx, forward).await;
                                    // Decrement per-peer count when done
                                    pfc.get(&ps).map(|c| c.fetch_sub(1, std::sync::atomic::Ordering::Relaxed));
                                });
                            }
                            // StreamingToken: route to registered streaming channel
                            SwarmMessage::StreamingToken(ref token) => {
                                let sender = match authenticated_sender {
                                    Some(ref s) => s,
                                    None => {
                                        tracing::warn!("StreamingToken from unauthenticated peer — dropping");
                                        continue;
                                    }
                                };
                                if !shared_state.peer_registry.contains_key(sender) {
                                    tracing::warn!(sender = %sender, "StreamingToken from unknown peer — dropping");
                                    continue;
                                }
                                // Clone the sender to drop the DashMap Ref (read lock) before
                                // awaiting send() or calling remove() — avoids deadlock.
                                let maybe_tx = shared_state
                                    .streaming_token_txs
                                    .get(&token.request_id)
                                    .map(|r| r.clone());
                                if let Some(tx) = maybe_tx {
                                    // Use try_send to avoid blocking the dispatch loop
                                    // if the client isn't consuming fast enough.
                                    if tx.try_send(token.clone()).is_err() {
                                        tracing::debug!(
                                            request_id = %token.request_id,
                                            "Streaming token channel closed or full"
                                        );
                                        shared_state.streaming_token_txs.remove(&token.request_id);
                                    }
                                }
                            }
                            // T13: VisionEncodeRequest — encode image using local mmproj
                            SwarmMessage::VisionEncodeRequest(req) => {
                                // SEC: Only accept from known, authenticated peers
                                if let Some(ref sender) = authenticated_sender {
                                    if !shared_state.peer_registry.contains_key(sender) {
                                        tracing::warn!(
                                            sender = %sender,
                                            "VisionEncodeRequest from unknown peer — dropping"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::warn!("VisionEncodeRequest without authenticated sender — dropping");
                                    continue;
                                }
                                let permit = match forward_semaphore.clone().try_acquire_owned() {
                                    Ok(p) => p,
                                    Err(_) => {
                                        tracing::warn!("VisionEncodeRequest rejected — forward semaphore full");
                                        continue;
                                    }
                                };
                                let ss = shared_state.clone();
                                let ntx = network_tx.clone();
                                tokio::spawn(async move {
                                    let _permit = permit;
                                    handle_vision_encode_request(ss, ntx, req).await;
                                });
                            }
                            // T13: VisionEncodeResponse — fire pending oneshot
                            SwarmMessage::VisionEncodeResponse(resp) => {
                                // Verify the authenticated sender matches the expected responder
                                // stored when the VisionEncodeRequest was sent.
                                // SEC: Peek at entry first — only remove after sender is validated
                                // to avoid dropping the oneshot sender on mismatch (which would hang the pipeline).
                                let sender_ok = if let Some(entry) = shared_state
                                    .pending_vision_results
                                    .get(&resp.request_id)
                                {
                                    let expected_node = &entry.0;
                                    if let Some(ref sender) = authenticated_sender {
                                        if sender != expected_node {
                                            tracing::warn!(
                                                request_id = %resp.request_id,
                                                expected = %expected_node,
                                                actual = %sender,
                                                "VisionEncodeResponse sender mismatch — dropping"
                                            );
                                            false
                                        } else {
                                            true
                                        }
                                    } else {
                                        tracing::warn!(
                                            request_id = %resp.request_id,
                                            "VisionEncodeResponse without authenticated sender — dropping"
                                        );
                                        false
                                    }
                                } else {
                                    false
                                };
                                if sender_ok {
                                    if let Some((_, (_expected_node, tx))) = shared_state
                                        .pending_vision_results
                                        .remove(&resp.request_id)
                                    {
                                        let _ = tx.send(resp);
                                    }
                                }
                            }
                            msg @ SwarmMessage::InferenceRequest(_)
                            | msg @ SwarmMessage::PipelineAssignment(_)
                            | msg @ SwarmMessage::InferenceError(_) => {
                                // SEC: Require authenticated sender for all inference control messages
                                if let Some(ref sender) = authenticated_sender {
                                    if !shared_state.peer_registry.contains_key(sender) {
                                        tracing::warn!(sender = %sender, "Inference message from unknown peer — dropping");
                                        continue;
                                    }
                                } else {
                                    tracing::warn!("Inference message without authenticated sender — dropping");
                                    continue;
                                }
                                if let Err(e) = router_tx
                                    .send(RouterCommand::NetworkMessage(msg))
                                    .await
                                {
                                    tracing::warn!(
                                        error = %e,
                                        "Failed to route inference message to router"
                                    );
                                }
                            }
                            SwarmMessage::CreditGossip(gossip) => {
                                // SEC: Verify sender matches the gossip's node_id
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &gossip.node_id {
                                        tracing::warn!(
                                            sender = %sender,
                                            claimed = %gossip.node_id,
                                            "Credit gossip rejected: sender mismatch"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated CreditGossip");
                                    continue;
                                }
                                // Use peer_credit_balances DashMap for deduplication:
                                // each peer gets exactly one entry, preventing Sybil stuffing.
                                crate::credit::ledger::process_balance_gossip(
                                    &credit_peer_balances,
                                    &gossip,
                                    Some(&shared_state.peer_credit_balances),
                                ).await;
                            }
                            SwarmMessage::ModelVote(_) => {
                                // Model governance voting is not enforced — users add models directly.
                            }
                            SwarmMessage::CreditTransaction(tx) => {
                                tracing::debug!(
                                    tx_id = %tx.id,
                                    from = %tx.from,
                                    to = %tx.to,
                                    amount = tx.amount,
                                    "Received credit transaction"
                                );
                                // SEC: Verify the transport-authenticated sender is a party to this tx.
                                // Prevents relaying forged transactions under someone else's identity.
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &tx.from && sender != &tx.to {
                                        tracing::warn!(
                                            tx_id = %tx.id,
                                            sender = %sender,
                                            from = %tx.from,
                                            to = %tx.to,
                                            "Credit tx rejected: sender is not a party to this transaction"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated CreditTransaction");
                                    continue;
                                }
                                // SEC-C3: Reject duplicate transactions (UUID replay check)
                                if let Ok(Some(_)) = shared_state.db.get_json::<crate::types::CreditTransaction>(
                                    crate::credit::ledger::TREE_TRANSACTIONS,
                                    &tx.id.to_string(),
                                ) {
                                    tracing::warn!(tx_id = %tx.id, "Rejecting replayed credit transaction");
                                    continue;
                                }
                                // SEC: Verify dual Ed25519 signatures before accepting.
                                // Without this check, any peer can forge arbitrary credit transactions.
                                {
                                    use ed25519_dalek::VerifyingKey;
                                    let from_key = match VerifyingKey::from_bytes(&tx.from.0) {
                                        Ok(k) => k,
                                        Err(_) => {
                                            tracing::warn!(tx_id = %tx.id, "Credit tx rejected: invalid from key");
                                            continue;
                                        }
                                    };
                                    let to_key = match VerifyingKey::from_bytes(&tx.to.0) {
                                        Ok(k) => k,
                                        Err(_) => {
                                            tracing::warn!(tx_id = %tx.id, "Credit tx rejected: invalid to key");
                                            continue;
                                        }
                                    };
                                    // verify_transaction also checks replay, but we already checked above
                                    if let Err(e) = crate::credit::transaction::verify_single_signatures(&tx, &from_key, &to_key) {
                                        tracing::warn!(
                                            tx_id = %tx.id,
                                            error = %e,
                                            "Credit tx rejected: signature verification failed"
                                        );
                                        continue;
                                    }
                                }
                                // Anti-gaming validation for network transactions
                                {
                                    let mut ag = shared_state.anti_gaming.lock().await;
                                    match ag.check_and_record_transaction(&tx.from, &tx.to, tx.amount) {
                                        Ok(_decision) => {}
                                        Err(violation) => {
                                            tracing::warn!(
                                                tx_id = %tx.id,
                                                violation = %violation,
                                                "Anti-gaming rejected credit transaction"
                                            );
                                            continue;
                                        }
                                    }
                                }
                                // Record the transaction and apply balance change
                                // if we are the recipient
                                let local_id = shared_state.identity.node_id().clone();
                                if tx.to == local_id {
                                    if let Err(e) = crate::credit::ledger::apply_credit_direct(
                                        &shared_state.credit_balance,
                                        &shared_state.db,
                                        tx.amount,
                                        true,
                                    ).await {
                                        tracing::warn!(error = %e, "Failed to apply credit transaction");
                                    }
                                    let bal = shared_state.credit_balance.read().await;
                                    tracing::info!(
                                        amount = tx.amount,
                                        balance = bal.balance,
                                        "Applied incoming credit transaction"
                                    );
                                }
                                let key = tx.id.to_string();
                                if let Err(e) = shared_state.db.put_json(crate::credit::ledger::TREE_TRANSACTIONS, &key, &tx) {
                                    tracing::warn!(error = %e, "Failed to store credit transaction");
                                }
                            }
                            // Process shard announcements from peers
                            SwarmMessage::ShardAnnounce(announce) => {
                                // SEC: Verify the authenticated sender matches the announce's node_id.
                                // Prevents peers from announcing shards under another node's identity.
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &announce.node_id {
                                        tracing::warn!(
                                            sender = %sender,
                                            claimed = %announce.node_id,
                                            shards = announce.shards.len(),
                                            "Shard announce rejected: sender mismatch"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated ShardAnnounce");
                                    continue;
                                }
                                // SEC: Cap shards per announce to prevent shard_holders memory exhaustion
                                const MAX_SHARDS_PER_ANNOUNCE: usize = 512;
                                if announce.shards.len() > MAX_SHARDS_PER_ANNOUNCE {
                                    tracing::warn!(
                                        node_id = %announce.node_id,
                                        shards = announce.shards.len(),
                                        max = MAX_SHARDS_PER_ANNOUNCE,
                                        "ShardAnnounce exceeds shard limit — dropping"
                                    );
                                    continue;
                                }
                                tracing::info!(
                                    node_id = %announce.node_id,
                                    shards = announce.shards.len(),
                                    "Received shard announce from peer"
                                );
                                // Refresh last_seen so health monitor doesn't remove active peers
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&announce.node_id) {
                                    peer.last_seen = chrono::Utc::now();
                                }
                                // Group shards by model for activity logging
                                let mut models_announced: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                                for shard_id in &announce.shards {
                                    shared_state.model_registry
                                        .record_shard_holder(shard_id.clone(), announce.node_id.clone());
                                    *models_announced.entry(shard_id.model_id.0.clone()).or_insert(0) += 1;
                                }
                                // Emit activity for each model announced
                                let peer_label = shared_state.nickname_registry.get(&announce.node_id)
                                    .map(|r| r.nickname.clone())
                                    .unwrap_or_else(|| format!("{}", announce.node_id).chars().take(8).collect());
                                for (mid, count) in &models_announced {
                                    let mname = shared_state.model_registry
                                        .get_manifest(&crate::types::ModelId(mid.clone()))
                                        .map(|m| m.name.clone());
                                    shared_state.emit_activity(crate::daemon::state::ActivityEvent {
                                        category: "model",
                                        kind: "shard_announced",
                                        message: format!("{} announced {} shard{} of {}", peer_label, count, if *count != 1 { "s" } else { "" }, mname.as_deref().unwrap_or(mid)),
                                        model_id: Some(mid.clone()),
                                        model_name: mname,
                                        node_id: Some(format!("{}", announce.node_id)),
                                        detail_num: Some(*count as i64),
                                        detail_str: None,
                                    });
                                }
                                // Wake auto-manage so it re-evaluates rarity scores —
                                // new shard holders change which shards are most needed.
                                shared_state.auto_manage_notify.notify_one();
                            }
                            // Process model manifests from peers — register in model_registry
                            SwarmMessage::ModelManifest(manifest) => {
                                // SEC: Require transport-authenticated sender (prevents anonymous injection).
                                // We do NOT require sender == publisher because any node that holds shards
                                // should be able to re-gossip a manifest they received from the publisher.
                                // Content integrity is guaranteed by verify_hash_strict() below.
                                if authenticated_sender.is_none() {
                                    tracing::debug!("Dropping unauthenticated ModelManifest");
                                    continue;
                                }
                                tracing::info!(
                                    model = %manifest.id,
                                    name = %manifest.name,
                                    shards = manifest.shard_count,
                                    publisher = %manifest.publisher,
                                    "Received model manifest from network"
                                );
                                // Strict verification for network-received manifests:
                                // reject zero-hash to prevent gossip poisoning.
                                match manifest.verify_hash_strict() {
                                    Ok(()) => {
                                        let is_new = shared_state
                                            .model_registry
                                            .get_manifest(&manifest.id)
                                            .is_none();
                                        shared_state.model_registry.register_manifest(manifest.clone());
                                        // Wake auto-manage when a genuinely new model appears
                                        if is_new {
                                            shared_state.auto_manage_notify.notify_one();
                                            shared_state.emit_activity(crate::daemon::state::ActivityEvent {
                                                category: "model",
                                                kind: "model_discovered",
                                                message: format!("Discovered model: {} ({} shards)", manifest.name, manifest.shard_count),
                                                model_id: Some(manifest.id.0.clone()),
                                                model_name: Some(manifest.name.clone()),
                                                node_id: None,
                                                detail_num: Some(manifest.shard_count as i64),
                                                detail_str: Some(format!("{:?}", manifest.architecture)),
                                            });
                                        }
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            error = %e,
                                            "Manifest hash verification failed — rejecting"
                                        );
                                    }
                                }
                            }
                            // Process capability updates from peers
                            SwarmMessage::NodeCapabilityUpdate(cap) => {
                                // SEC: Verify sender matches claimed node_id
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &cap.node_id {
                                        tracing::warn!(
                                            claimed = %cap.node_id,
                                            actual = %sender,
                                            "NodeCapabilityUpdate sender mismatch — dropping"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated NodeCapabilityUpdate");
                                    continue;
                                }
                                tracing::debug!(
                                    node_id = %cap.node_id,
                                    hosted_shards = cap.hosted_shards.len(),
                                    "Received capability update from peer"
                                );
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&cap.node_id) {
                                    peer.capability = Some(cap.clone());
                                    peer.last_seen = chrono::Utc::now();
                                }
                            }
                            // Nickname gossip from peers
                            SwarmMessage::NicknameGossip(gossip) => {
                                let record = &gossip.record;
                                // SEC: Verify gossip sender matches the record's node_id.
                                // Prevents peers from injecting nicknames for other nodes.
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &record.node_id {
                                        tracing::warn!(
                                            sender = %sender,
                                            claimed = %record.node_id,
                                            "Nickname gossip rejected: sender mismatch"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated NicknameGossip");
                                    continue;
                                }
                                // Age check: reject messages older than 24 hours
                                let age = chrono::Utc::now() - record.timestamp;
                                if age > chrono::Duration::hours(24) {
                                    tracing::debug!(
                                        node_id = %record.node_id,
                                        "Rejecting stale nickname gossip (>24h old)"
                                    );
                                } else if record.verify().is_err() {
                                    tracing::warn!(
                                        node_id = %record.node_id,
                                        "Rejecting nickname gossip with invalid signature"
                                    );
                                } else {
                                    // SEC: Only accept nicknames from peers we've seen to prevent
                                    // Sybil memory exhaustion via pre-generated Ed25519 keypairs.
                                    // Hard cap as secondary defense.
                                    if !shared_state.peer_registry.contains_key(&record.node_id)
                                        && !shared_state.nickname_registry.contains_key(&record.node_id)
                                        && shared_state.nickname_registry.len() >= 10_000
                                    {
                                        tracing::debug!(
                                            node_id = %record.node_id,
                                            "Rejecting nickname from unknown peer (registry cap)"
                                        );
                                        continue;
                                    }
                                    // Timestamp-wins: only update if newer
                                    let should_insert = match shared_state
                                        .nickname_registry
                                        .get(&record.node_id)
                                    {
                                        Some(existing) => record.timestamp > existing.timestamp,
                                        None => true,
                                    };
                                    if should_insert {
                                        tracing::info!(
                                            node_id = %record.node_id,
                                            nickname = %record.nickname,
                                            "Accepted nickname from peer"
                                        );
                                        shared_state
                                            .nickname_registry
                                            .insert(record.node_id.clone(), record.clone());
                                        // Persist
                                        let store = crate::identity::nickname::NicknameStore::new(
                                            shared_state.db.clone(),
                                        );
                                        if let Err(e) = store.put_record(record) {
                                            tracing::warn!(error = %e, "Failed to persist nickname");
                                        }
                                    }
                                }
                            }
                            // Route pool messages to the PoolManager
                            SwarmMessage::PoolMessage(pool_msg) => {
                                let sender = match authenticated_sender {
                                    Some(ref s) => s,
                                    None => {
                                        tracing::warn!("PoolMessage without authenticated sender — dropping");
                                        continue;
                                    }
                                };
                                // SEC: Verify inner identity matches authenticated sender
                                // to prevent spoofing pool messages from other nodes
                                let inner_ok = match &pool_msg {
                                    crate::types::PoolMessage::CreditForward(fwd) => fwd.from_node_id == *sender,
                                    crate::types::PoolMessage::MemberLeft { node_id, .. } => node_id == sender,
                                    // Invitation/Acceptance/Removal are verified by crypto sigs in pool manager
                                    _ => true,
                                };
                                if !inner_ok {
                                    tracing::warn!(
                                        sender = %sender,
                                        "PoolMessage inner identity mismatch — dropping"
                                    );
                                    continue;
                                }
                                if let Some(ref tx) = *shared_state.pool_tx.read().await {
                                    let cmd = match pool_msg {
                                        crate::types::PoolMessage::Invitation(inv) => {
                                            Some(crate::pool::types::PoolCommand::InboundInvitation {
                                                invitation: inv,
                                            })
                                        }
                                        crate::types::PoolMessage::BlindedInvitation(blinded) => {
                                            Some(crate::pool::types::PoolCommand::InboundBlindedInvitation {
                                                blinded,
                                            })
                                        }
                                        crate::types::PoolMessage::Acceptance(acc) => {
                                            Some(crate::pool::types::PoolCommand::InboundAcceptance {
                                                acceptance: acc,
                                            })
                                        }
                                        crate::types::PoolMessage::StateGossip(state) => {
                                            Some(crate::pool::types::PoolCommand::PoolStateGossip {
                                                state,
                                            })
                                        }
                                        crate::types::PoolMessage::CreditForward(fwd) => {
                                            Some(crate::pool::types::PoolCommand::ProcessCreditForward {
                                                forward: fwd,
                                            })
                                        }
                                        crate::types::PoolMessage::Removal(rem) => {
                                            Some(crate::pool::types::PoolCommand::InboundRemoval {
                                                removal: rem,
                                            })
                                        }
                                        crate::types::PoolMessage::MemberLeft { pool_id, node_id, signature } => {
                                            Some(crate::pool::types::PoolCommand::InboundMemberLeft {
                                                pool_id,
                                                node_id,
                                                signature,
                                            })
                                        }
                                        crate::types::PoolMessage::JoinRequest { code_hash, requester, signature: _ } => {
                                            // SEC: Transport layer already verified the sender's identity.
                                            // The signature field provides an additional binding but is
                                            // validated by the pool manager if needed.
                                            Some(crate::pool::types::PoolCommand::InboundJoinRequest {
                                                code_hash,
                                                requester,
                                            })
                                        }
                                    };
                                    if let Some(cmd) = cmd {
                                        if let Err(e) = tx.send(cmd).await {
                                            tracing::warn!(error = %e, "Failed to route pool message");
                                        }
                                    }
                                }
                            }
                            // HuggingFace source gossip — store so auto-manage can download shards
                            SwarmMessage::HfSourceGossip(gossip) => {
                                // SEC: Verify sender matches claimed publisher
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &gossip.publisher {
                                        tracing::warn!(
                                            claimed = %gossip.publisher,
                                            actual = %sender,
                                            "HfSourceGossip sender mismatch — dropping"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated HfSourceGossip");
                                    continue;
                                }
                                // SEC: Length limits on untrusted strings
                                if gossip.repo_id.len() > 256 || gossip.filename.len() > 256 {
                                    tracing::warn!(
                                        repo_id_len = gossip.repo_id.len(),
                                        filename_len = gossip.filename.len(),
                                        "HfSourceGossip strings too long — dropping"
                                    );
                                    continue;
                                }
                                // SEC: Validate repo_id format (owner/repo) and filename to prevent
                                // URL injection when constructing HuggingFace download URLs
                                let repo_valid = {
                                    let parts: Vec<&str> = gossip.repo_id.splitn(2, '/').collect();
                                    parts.len() == 2
                                        && parts.iter().all(|p| {
                                            !p.is_empty()
                                                && p.chars().all(|c| {
                                                    c.is_alphanumeric()
                                                        || c == '-'
                                                        || c == '_'
                                                        || c == '.'
                                                })
                                        })
                                };
                                let filename_valid = !gossip.filename.is_empty()
                                    && !gossip.filename.contains('/')
                                    && !gossip.filename.contains('\\')
                                    && !gossip.filename.contains('\0')
                                    && gossip.filename.chars().all(|c| {
                                        c.is_alphanumeric()
                                            || c == '-'
                                            || c == '_'
                                            || c == '.'
                                    });
                                if !repo_valid || !filename_valid {
                                    tracing::warn!(
                                        repo = %gossip.repo_id,
                                        filename = %gossip.filename,
                                        "HfSourceGossip invalid repo_id/filename format — dropping"
                                    );
                                    continue;
                                }
                                let mid = gossip.model_id.clone();
                                if !shared_state.hf_sources.contains_key(&mid) {
                                    tracing::info!(
                                        model = %mid,
                                        repo = %gossip.repo_id,
                                        filename = %gossip.filename,
                                        publisher = %gossip.publisher,
                                        "Received HfSourceGossip — storing HF source"
                                    );
                                    let source = crate::daemon::HfSource {
                                        repo_id: gossip.repo_id.clone(),
                                        filename: gossip.filename.clone(),
                                        mmproj_filename: gossip.mmproj_filename.clone(),
                                    };
                                    shared_state.hf_sources.insert(mid.clone(), source.clone());
                                    // Persist to DB
                                    let _ = shared_state.db.put_json("hf_sources", &mid.0, &source);
                                    // Wake the AutoShardManager so it evaluates promptly
                                    shared_state.auto_manage_notify.notify_one();
                                }
                            }
                            SwarmMessage::ShardDownloadProgress(progress) => {
                                // SEC: Verify sender matches claimed node_id
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &progress.node_id {
                                        tracing::warn!(
                                            claimed = %progress.node_id,
                                            actual = %sender,
                                            "ShardDownloadProgress sender mismatch — dropping"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated ShardDownloadProgress");
                                    continue;
                                }
                                // Update peer download state in shared state
                                let local_nid = shared_state.identity.node_id();
                                if progress.node_id != *local_nid {
                                    if progress.state == crate::types::DownloadState::Complete || progress.progress_pct >= 100 {
                                        // Download finished — remove from download tracking
                                        if let Some(mut entry) = shared_state.peer_shard_downloads.get_mut(&progress.shard_id) {
                                            entry.retain(|(nid, _)| *nid != progress.node_id);
                                        }
                                        // Register the peer as a shard holder now
                                        // (the ShardAnnounce gossip will also arrive,
                                        //  but this gives immediate consistency)
                                        shared_state.model_registry
                                            .record_shard_holder(progress.shard_id.clone(), progress.node_id.clone());
                                        // Wake auto-manage — peer completed a download, rarity changed
                                        shared_state.auto_manage_notify.notify_one();
                                    } else {
                                        // Update or insert download progress
                                        let mut entry = shared_state.peer_shard_downloads.entry(progress.shard_id.clone()).or_default();
                                        if let Some(pos) = entry.iter().position(|(nid, _)| *nid == progress.node_id) {
                                            entry[pos].1 = progress.progress_pct;
                                        } else {
                                            entry.push((progress.node_id.clone(), progress.progress_pct));
                                        }
                                    }
                                    tracing::debug!(
                                        node = %progress.node_id,
                                        model = %progress.shard_id.model_id,
                                        shard = progress.shard_id.index,
                                        pct = progress.progress_pct,
                                        state = %progress.state,
                                        "Peer shard download progress"
                                    );
                                }
                            }
                            // Health pings: update sender's load and respond with pong
                            SwarmMessage::HealthPing { nonce, node_id: Some(sender_id), active_request_count, .. } => {
                                // SEC: Verify sender matches the health ping's node_id
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &sender_id {
                                        tracing::warn!(
                                            sender = %sender,
                                            claimed = %sender_id,
                                            "Health ping rejected: sender mismatch"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated HealthPing");
                                    continue;
                                }
                                // Update the sender's active request count in peer_registry
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&sender_id) {
                                    peer.active_request_count = active_request_count;
                                    peer.last_seen = chrono::Utc::now();
                                }

                                // Respond with a pong containing our own load
                                let ts = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_secs();
                                let our_load = shared_state.active_pipelines.len() as u32;
                                let our_id = Some(shared_state.identity.node_id().clone());
                                let pong = SwarmMessage::HealthPong {
                                    nonce,
                                    timestamp: ts,
                                    node_id: our_id,
                                    active_request_count: our_load,
                                };
                                // Unicast pong to the pinger instead of broadcasting O(N²)
                                if let Some(peer_bytes) = shared_state
                                    .peer_id_map
                                    .get(&sender_id)
                                    .map(|r| r.clone())
                                {
                                    let _ = network_tx
                                        .send(NetworkCommand::SendDirectMessage {
                                            target_peer_bytes: peer_bytes,
                                            message: pong,
                                        })
                                        .await;
                                } else {
                                    // Fallback to broadcast if peer_id unknown
                                    let _ =
                                        network_tx.send(NetworkCommand::Broadcast(pong)).await;
                                }
                            }
                            // Health pongs: update the sender's load in peer_registry
                            SwarmMessage::HealthPong { node_id: Some(sender_id), active_request_count, .. } => {
                                // SEC: Verify sender matches the health pong's node_id
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &sender_id {
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated HealthPong");
                                    continue;
                                }
                                if let Some(mut peer) = shared_state.peer_registry.get_mut(&sender_id) {
                                    peer.active_request_count = active_request_count;
                                    peer.last_seen = chrono::Utc::now();
                                }
                            }
                            // Ephemeral key exchange for forward secrecy
                            SwarmMessage::EphemeralKeyExchange(exchange) => {
                                // SEC: Verify transport-authenticated sender matches exchange.node_id.
                                // The Noise protocol authenticates the transport, so we trust the PeerId→NodeId
                                // mapping. This prevents a peer from injecting ephemeral keys for another node.
                                if let Some(ref sender) = authenticated_sender {
                                    if sender != &exchange.node_id {
                                        tracing::warn!(
                                            sender = %sender,
                                            claimed = %exchange.node_id,
                                            "Ephemeral key exchange rejected: sender mismatch"
                                        );
                                        continue;
                                    }
                                } else {
                                    tracing::debug!("Dropping unauthenticated EphemeralKeyExchange");
                                    continue;
                                }
                                let sm = shared_state.session_manager.clone();
                                let our_id = shared_state.identity.node_id().clone();
                                if exchange.node_id == our_id {
                                    // Ignore our own broadcast
                                } else if exchange.is_initiator {
                                    // Peer wants to re-key: accept and reply
                                    let response_pub = sm.accept_ephemeral_exchange(
                                        &exchange.node_id,
                                        &exchange.ephemeral_pubkey,
                                    );
                                    let reply = SwarmMessage::EphemeralKeyExchange(EphemeralKeyExchange {
                                        session_id: exchange.session_id,
                                        node_id: our_id,
                                        ephemeral_pubkey: response_pub,
                                        is_initiator: false,
                                    });
                                    // Send reply directly to the initiator (not broadcast)
                                    // to prevent other peers from intercepting the ephemeral key.
                                    let target = shared_state
                                        .peer_id_map
                                        .get(&exchange.node_id)
                                        .map(|r| r.value().clone());
                                    if let Some(target_bytes) = target {
                                        let _ = network_tx.send(NetworkCommand::SendDirectMessage {
                                            target_peer_bytes: target_bytes,
                                            message: reply,
                                        }).await;
                                    } else {
                                        tracing::warn!(
                                            node_id = %exchange.node_id,
                                            "Cannot reply to ephemeral key exchange — no PeerId mapping"
                                        );
                                    }
                                } else {
                                    // Response to our initiation: complete the exchange
                                    sm.complete_ephemeral_session(
                                        &exchange.node_id,
                                        &exchange.ephemeral_pubkey,
                                    );
                                }
                            }
                            // Tensor-parallel AllReduce: collect partial from a TP rank
                            SwarmMessage::TpAllReduceRequest(req) => {
                                if let Some(ref sender) = authenticated_sender {
                                    if !shared_state.peer_registry.contains_key(sender) {
                                        tracing::warn!(sender = %sender, "TpAllReduceRequest from unknown peer — dropping");
                                        continue;
                                    }
                                } else {
                                    continue;
                                }
                                if req.tp_size < 2 || req.tp_size as usize > 32 {
                                    tracing::warn!(tp_size = req.tp_size, "TpAllReduceRequest tp_size out of range [2,32] — dropping");
                                    continue;
                                }
                                let key = (req.request_id, req.layer_idx);
                                let tp_size = req.tp_size;
                                let ss = shared_state.clone();
                                let ntx = network_tx.clone();

                                // Extract sender peer bytes from the request context
                                // (embedded by NetworkManager when receiving the rr request)
                                let sender_peer = req.sender_peer_bytes.clone();

                                // SEC: Cap pending_tp_partials to prevent OOM from AllReduce flooding
                                const MAX_PENDING_TP_PARTIALS: usize = 512;
                                if !ss.pending_tp_partials.contains_key(&key)
                                    && ss.pending_tp_partials.len() >= MAX_PENDING_TP_PARTIALS
                                {
                                    tracing::warn!("pending_tp_partials full ({MAX_PENDING_TP_PARTIALS}) — dropping TpAllReduceRequest");
                                    continue;
                                }

                                let all_arrived = {
                                    let mut entry = ss.pending_tp_partials
                                        .entry(key)
                                        .or_insert_with(|| TpAllReduceCollector::new(tp_size));
                                    entry.insert(req, sender_peer)
                                };

                                if all_arrived {
                                    // All partials collected — reduce and respond
                                    tokio::spawn(async move {
                                        let collector = ss.pending_tp_partials.remove(&key);
                                        if let Some((_, collector)) = collector {
                                            match collector.reduce_sum() {
                                                Ok((reduced_data, shape)) => {
                                                    let resp = crate::types::TpAllReduceResponse {
                                                        request_id: key.0,
                                                        layer_idx: key.1,
                                                        reduced_data,
                                                        shape,
                                                    };
                                                    // Deliver to local registry (coordinator is also a TP rank)
                                                    ss.allreduce_registry.deliver(resp.clone());
                                                    // Unicast response to each remote TP participant (not broadcast)
                                                    for peer_bytes in collector.sender_peers.iter().flatten() {
                                                        let _ = ntx.send(NetworkCommand::SendAllReduceResponse {
                                                            target_peer_bytes: peer_bytes.clone(),
                                                            response: resp.clone(),
                                                        }).await;
                                                    }
                                                }
                                                Err(e) => {
                                                    tracing::warn!(
                                                        error = %e,
                                                        request_id = %key.0,
                                                        layer_idx = key.1,
                                                        "AllReduce sum failed"
                                                    );
                                                }
                                            }
                                        }
                                    });
                                }
                            }
                            // Tensor-parallel AllReduce response: deliver to waiting pipeline
                            SwarmMessage::TpAllReduceResponse(resp) => {
                                match authenticated_sender {
                                    Some(ref sender) => {
                                        if !shared_state.peer_registry.contains_key(sender) {
                                            tracing::warn!(sender = %sender, "TpAllReduceResponse from unknown peer — dropping");
                                            continue;
                                        }
                                    }
                                    None => {
                                        tracing::warn!("TpAllReduceResponse from unauthenticated peer — dropping");
                                        continue;
                                    }
                                }
                                let delivered = shared_state.allreduce_registry.deliver(resp.clone());
                                tracing::debug!(
                                    request_id = %resp.request_id,
                                    layer_idx = resp.layer_idx,
                                    reduced_bytes = resp.reduced_data.len(),
                                    delivered,
                                    "AllReduce response received"
                                );
                            }
                            // Regional shard summary gossip (Phase 18)
                            SwarmMessage::RegionShardSummary(summary) => {
                                // Authenticate sender
                                match &authenticated_sender {
                                    Some(sender) if *sender != summary.publisher => {
                                        tracing::warn!(sender = %sender, claimed = %summary.publisher, "RegionShardSummary sender mismatch — dropping");
                                        continue;
                                    }
                                    None => {
                                        tracing::debug!("Dropping unauthenticated RegionShardSummary");
                                        continue;
                                    }
                                    Some(_) => {} // Sender matches publisher — proceed
                                }
                                if summary.region.len() > 8 || summary.shard_counts.len() > 512 {
                                    continue;
                                }
                                // Reject stale summaries
                                let now_ms = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64;
                                if now_ms.saturating_sub(summary.timestamp_ms) > GOSSIP_STALENESS_MS {
                                    tracing::debug!(
                                        region = %summary.region,
                                        model = %summary.model_id,
                                        age_ms = now_ms.saturating_sub(summary.timestamp_ms),
                                        "Dropping stale RegionShardSummary"
                                    );
                                    continue;
                                }
                                let key = (summary.region.clone(), summary.model_id.clone());
                                // Keep the most recent summary per (region, model)
                                let should_update = shared_state
                                    .region_shard_summaries
                                    .get(&key)
                                    .map(|existing| summary.timestamp_ms > existing.timestamp_ms)
                                    .unwrap_or(true);
                                if should_update {
                                    // Cap to prevent unbounded growth from malicious/diverse gossip
                                    const MAX_REGION_SUMMARIES: usize = 10_000;
                                    if shared_state.region_shard_summaries.len() >= MAX_REGION_SUMMARIES
                                        && !shared_state.region_shard_summaries.contains_key(&key)
                                    {
                                        tracing::debug!("region_shard_summaries at cap, dropping new entry");
                                        continue;
                                    }
                                    tracing::debug!(
                                        region = %summary.region,
                                        model = %summary.model_id,
                                        node_count = summary.region_node_count,
                                        shard_entries = summary.shard_counts.len(),
                                        "RegionShardSummary updated"
                                    );
                                    shared_state.region_shard_summaries.insert(key, summary);
                                }
                            }

                            // Model demand gossip (Phase 18)
                            SwarmMessage::ModelDemandGossip(demand) => {
                                // Authenticate sender
                                match &authenticated_sender {
                                    Some(sender) if *sender != demand.publisher => {
                                        tracing::warn!(sender = %sender, claimed = %demand.publisher, "ModelDemandGossip sender mismatch — dropping");
                                        continue;
                                    }
                                    None => {
                                        tracing::debug!("Dropping unauthenticated ModelDemandGossip");
                                        continue;
                                    }
                                    Some(_) => {} // Sender matches publisher — proceed
                                }
                                if demand.region.len() > 8 {
                                    continue;
                                }
                                // Reject stale demand
                                let now_ms = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH)
                                    .unwrap_or_default()
                                    .as_millis() as u64;
                                if now_ms.saturating_sub(demand.timestamp_ms) > GOSSIP_STALENESS_MS {
                                    tracing::debug!(
                                        model = %demand.model_id,
                                        region = %demand.region,
                                        "Dropping stale ModelDemandGossip"
                                    );
                                    continue;
                                }
                                let key = (demand.model_id.clone(), demand.region.clone());
                                // Cap to prevent unbounded growth
                                const MAX_DEMAND_ENTRIES: usize = 10_000;
                                if shared_state.region_demand.len() >= MAX_DEMAND_ENTRIES
                                    && !shared_state.region_demand.contains_key(&key)
                                {
                                    continue;
                                }
                                // EMA blend: 0.8 * old + 0.2 * incoming
                                let new_rate = if let Some(existing) = shared_state.region_demand.get(&key) {
                                    *existing * 0.8 + demand.decayed_rate * 0.2
                                } else {
                                    demand.decayed_rate
                                };
                                shared_state.region_demand.insert(key, new_rate);
                                tracing::debug!(
                                    model = %demand.model_id,
                                    region = %demand.region,
                                    decayed_rate = demand.decayed_rate,
                                    blended_rate = new_rate,
                                    "ModelDemandGossip processed"
                                );
                            }

                            // Other messages handled by NetworkManager
                            _ => {}
                        }
                    }
                    None => break,
                }
            }
        }
    }
}

async fn handle_layer_forward(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    forward: crate::types::LayerForward,
) {
    let request_id = forward.request_id;
    let sender_peer_bytes = match forward.sender_peer_bytes {
        Some(ref bytes) => bytes.clone(),
        None => {
            tracing::warn!(request_id = %request_id, "LayerForward missing sender_peer_bytes");
            return;
        }
    };

    let forward_start = std::time::Instant::now();
    tracing::info!(
        request_id = %request_id,
        seq = forward.sequence_num,
        activation_bytes = forward.activations.len(),
        model_id = %forward.model_id,
        layer_range = ?forward.layer_range,
        "DIAG: processing LayerForward locally"
    );

    let model_id = forward.model_id.clone();

    // Determine our layer range from the manifest and local shards
    let manifest = match shared_state.model_registry.get_manifest(&model_id) {
        Some(m) => m,
        None => {
            send_error_result(
                &network_tx,
                &sender_peer_bytes,
                request_id,
                "No manifest for model",
            )
            .await;
            return;
        }
    };

    // Figure out which shard indices we hold locally
    let local_node_id = shared_state.identity.node_id().clone();
    let mut local_shard_indices: Vec<u32> = Vec::new();
    for shard_info in &manifest.shards {
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: shard_info.index,
        };
        let holders = shared_state.model_registry.shard_holders(&shard_id);
        if holders.contains(&local_node_id) {
            local_shard_indices.push(shard_info.index);
        }
    }

    if local_shard_indices.is_empty() {
        send_error_result(
            &network_tx,
            &sender_peer_bytes,
            request_id,
            "No local shards for model",
        )
        .await;
        return;
    }

    // Layer range is required in the forward message — no guessing
    let (layer_start, layer_end, total_layers) = {
        let (ls, le) = forward.layer_range;
        let total = manifest.num_layers as usize;
        (ls as usize, le as usize, total)
    };

    if layer_start >= layer_end || layer_end > total_layers {
        send_error_result(
            &network_tx,
            &sender_peer_bytes,
            request_id,
            &format!(
                "Invalid layer range [{layer_start}..{layer_end}) for model with {total_layers} layers"
            ),
        )
        .await;
        return;
    }

    // is_first requires shard 0 (token_embd.weight is at tensor offset 0)
    // is_last requires the final shard (output.weight spans to the end of the file)
    let has_shard_0 = local_shard_indices.contains(&0);
    let last_shard_idx = manifest.shard_count.saturating_sub(1);
    let has_last_shard = local_shard_indices.contains(&last_shard_idx);
    let is_first = layer_start == 0 && has_shard_0;
    let is_last = layer_end >= total_layers && has_last_shard;

    // Ensure the split model metadata entry exists (lightweight — no GPU loading).
    // Re-check after computation to avoid overwriting a concurrent insert.
    let split_key = (model_id.clone(), layer_start, layer_end);
    if !shared_state.split_models.contains_key(&split_key) {
        let shard_store = crate::model::shard::ShardStore::new(&shared_state.config.node.data_dir);
        let model_dir = shard_store
            .models_dir()
            .join(crate::model::shard::sanitize_path_component(&model_id.0));

        // Estimate VRAM from shard file sizes on disk (no model loading)
        let vram_estimate_mb =
            estimate_vram_from_shard_dir(&model_dir, layer_start, layer_end, total_layers);

        // Read metadata from GGUF header file
        let header_path = model_dir.join("gguf_header.bin");
        let entry = crate::inference::split::SplitModelEntry::from_header(
            &header_path,
            layer_start,
            layer_end,
            is_first,
            is_last,
            vram_estimate_mb,
        );

        // VRAM-aware eviction before inserting
        let vram_budget = crate::model::auto_manage::compute_vram_budget(&shared_state)
            .or(shared_state.config.inference.max_split_model_memory_mb);
        if let Some(budget_mb) = vram_budget {
            let evicted = crate::inference::split::evict_split_models_lru(
                &shared_state.split_models,
                &shared_state.active_pipelines,
                budget_mb,
                entry.estimated_vram_mb,
            );
            if evicted > 0 {
                tracing::info!(
                    evicted,
                    budget_mb,
                    "Evicted LRU split models for VRAM budget"
                );
            }
        }
        // Re-check: a concurrent task may have inserted while we computed above.
        // or_insert avoids overwriting the concurrent entry (and its VRAM eviction).
        shared_state
            .split_models
            .entry(split_key.clone())
            .or_insert(entry);
    }

    // Touch the metadata entry
    if let Some(entry) = shared_state.split_models.get(&split_key) {
        entry.value().touch();
    }

    // Capture requester_node_id before moving forward into the process pool
    let requester_node_id = forward.requester_node_id;

    // Route forward pass to subprocess via process pool
    let result = shared_state.model_process_pool.forward(forward).await;

    let result = match result {
        Ok(r) => r,
        Err(e) => {
            send_error_result(
                &network_tx,
                &sender_peer_bytes,
                request_id,
                &format!("Worker: {e}"),
            )
            .await;
            return;
        }
    };

    let forward_elapsed = forward_start.elapsed();
    tracing::info!(
        request_id = %request_id,
        tokens = result.token_ids.len(),
        activations_bytes = result.activations.len(),
        is_last,
        elapsed_ms = forward_elapsed.as_millis() as u64,
        model_id = %model_id,
        layers = format!("[{layer_start}..{layer_end})"),
        "DIAG: LayerForward processed via worker subprocess"
    );

    track_forward_participation(&shared_state, layer_start, layer_end, total_layers);

    // Pipeline sealing: encrypt token IDs for requester if this is the final segment
    let mut result = result;
    if is_last {
        seal_layer_result(&mut result, requester_node_id.as_ref());
    }

    // Send back as a separate request to the originating peer
    if let Err(e) = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: sender_peer_bytes,
            result,
        })
        .await
    {
        tracing::warn!(error = %e, "Failed to send LayerResult back to peer");
    }
}

/// Handle a VisionEncodeRequest: encode the image using local mmproj and respond.
async fn handle_vision_encode_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    req: crate::types::VisionEncodeRequest,
) {
    let model_id = &req.model_id;

    // SEC: Reject oversized image payloads BEFORE loading vision module to prevent
    // a malicious peer from triggering expensive module loading with large payloads.
    const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024; // 20 MB
    if req.image_data.len() > MAX_IMAGE_BYTES {
        tracing::warn!(
            request_id = %req.request_id,
            size = req.image_data.len(),
            max = MAX_IMAGE_BYTES,
            "VisionEncodeRequest image_data too large — rejecting"
        );
        return;
    }

    tracing::info!(
        request_id = %req.request_id,
        model = %model_id,
        image_bytes = req.image_data.len(),
        "Handling VisionEncodeRequest"
    );

    // Load or get the vision module
    let vision_module = if let Some(entry) = shared_state.vision_modules.get(model_id) {
        entry.value().clone()
    } else {
        // Try to load mmproj on-demand
        let model_dir = shared_state
            .config
            .node
            .data_dir
            .join("models")
            .join(crate::model::shard::sanitize_path_component(&model_id.0));
        let mmproj_path = model_dir.join("mmproj.gguf");
        if !mmproj_path.exists() {
            tracing::warn!(
                request_id = %req.request_id,
                model = %model_id,
                "VisionEncodeRequest received but no mmproj.gguf found"
            );
            return;
        }
        match crate::inference::vision::load_from_mmproj_gguf(
            &mmproj_path,
            &candle_core::Device::Cpu,
        ) {
            Ok(module) => {
                let module = Arc::new(module);
                shared_state
                    .vision_modules
                    .insert(model_id.clone(), module.clone());
                module
            }
            Err(e) => {
                tracing::warn!(
                    request_id = %req.request_id,
                    error = %e,
                    "Failed to load mmproj for VisionEncodeRequest"
                );
                return;
            }
        }
    };

    // Size check already done above (before module loading).
    // This is a defense-in-depth second check.
    if req.image_data.len() > MAX_IMAGE_BYTES {
        return;
    }

    // Decode JPEG image into ImageData
    let img = match image::load_from_memory(&req.image_data) {
        Ok(dyn_img) => {
            let rgb = dyn_img.to_rgb8();
            let (w, h) = rgb.dimensions();
            crate::types::ImageData {
                rgb_bytes: rgb.into_raw(),
                width: w,
                height: h,
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %req.request_id,
                error = %e,
                "Failed to decode image in VisionEncodeRequest"
            );
            return;
        }
    };

    // Encode image to vision embeddings (CPU-bound)
    let encode_result = tokio::task::block_in_place(|| vision_module.encode_images(&[img]));
    match encode_result {
        Ok(embeddings) => {
            // Compress embeddings with zstd for wire transfer
            let (num_tokens, hidden_dim) = embeddings.dims2().unwrap_or((0, 0));
            let raw_bytes: Vec<u8> = embeddings
                .to_dtype(candle_core::DType::F16)
                .and_then(|t| t.to_vec2::<half::f16>())
                .map(|v: Vec<Vec<half::f16>>| {
                    let mut bytes = Vec::with_capacity(num_tokens * hidden_dim * 2);
                    for row in v {
                        for f in row {
                            bytes.extend_from_slice(&f.to_le_bytes());
                        }
                    }
                    bytes
                })
                .unwrap_or_default();
            let compressed =
                zstd::encode_all(std::io::Cursor::new(&raw_bytes), ZSTD_COMPRESS_LEVEL)
                    .unwrap_or(raw_bytes);

            let response = crate::types::VisionEncodeResponse {
                request_id: req.request_id,
                embeddings: compressed,
                num_tokens: num_tokens as u32,
                hidden_dim: hidden_dim as u32,
            };

            tracing::info!(
                request_id = %req.request_id,
                num_tokens,
                hidden_dim,
                compressed_bytes = response.embeddings.len(),
                "VisionEncodeRequest completed, sending response"
            );

            let msg = if let Some(target_bytes) = &req.sender_peer_bytes {
                NetworkCommand::SendDirectMessage {
                    target_peer_bytes: target_bytes.clone(),
                    message: SwarmMessage::VisionEncodeResponse(response),
                }
            } else {
                tracing::warn!(request_id = %req.request_id, "VisionEncodeResponse has no sender — dropping");
                return;
            };
            if let Err(e) = network_tx.send(msg).await {
                tracing::warn!(error = %e, "Failed to send VisionEncodeResponse");
            }
        }
        Err(e) => {
            tracing::warn!(
                request_id = %req.request_id,
                error = %e,
                "Vision encoding failed"
            );
        }
    }
}

/// Send an error LayerResult back to the requesting peer.
async fn send_error_result(
    network_tx: &mpsc::Sender<NetworkCommand>,
    target_peer_bytes: &[u8],
    request_id: uuid::Uuid,
    error: &str,
) {
    tracing::warn!(request_id = %request_id, error, "LayerForward processing failed");
    // Sanitize error for network — don't leak internal paths, layer counts, or model topology
    let sanitized = if error.contains("layer range") || error.contains("layer_start") {
        "Layer configuration error".to_string()
    } else if error.contains("No local shards") || error.contains("shard") {
        "Required shards not available".to_string()
    } else {
        // Truncate and strip paths
        let msg = error.chars().take(100).collect::<String>();
        msg.replace(['/', '\\'], "")
    };
    let result = crate::types::LayerResult {
        request_id,
        token_ids: vec![],
        finish_reason: Some(crate::types::NetworkFinishReason::Error(sanitized)),
        activations: vec![],
        sealed_token_ids: None,
    };
    let _ = network_tx
        .send(NetworkCommand::SendTensorResult {
            target_peer_bytes: target_peer_bytes.to_vec(),
            result,
        })
        .await;
}
