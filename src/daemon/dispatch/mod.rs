use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use crate::identity::nickname::NicknameRecordExt;
use crate::inference::router::RouterCommand;
use crate::model::manifest::ModelManifestExt;
use crate::types::{AuthenticatedMessage, EphemeralKeyExchange, NetworkCommand, SwarmMessage};

use super::state::{SharedState, TpAllReduceCollector};

/// Maximum concurrent LayerForward tasks, at the highest contribution level.
///
/// This is work done FOR OTHER PEOPLE, so how much of it a node accepts has to
/// follow what its owner agreed to give — see [`max_concurrent_forwards`].
const MAX_CONCURRENT_FORWARDS_MAX: usize = 64;

/// Concurrent peer forwards this node will run at once, by contribution level.
///
/// This was a flat 64 on every node whatever its owner had chosen, and
/// `Minimal` is the DEFAULT — so a stock home machine would accept 64
/// simultaneous inference forwards from the swarm. On the hardware this is
/// actually aimed at (gaming PCs, home desktops) that is precisely how a node
/// gets swamped: each forward is a real model step, and 64 of them at once
/// makes the machine unusable for the person sitting in front of it.
///
/// A contribution setting has to mean something everywhere resources are spent,
/// not only where it was convenient to plumb it. This is the concurrency half
/// of that; `vram_fraction_for` in `config/node.rs` is the memory half.
///
/// The floor is deliberately not 1: a single-permit node cannot participate in
/// tensor-parallel work at all, and refusing everything is its own kind of
/// broken. `Minimal` still accepts real work — just not an unbounded amount of
/// it.
///
/// **This is also the only bound on inference upload bandwidth, deliberately.**
/// Shard serving is rate-limited by `ResourceConfig::shard_upload_mbps`, and the
/// obvious symmetry would be to throttle tensor forwards the same way. That
/// would make things worse, not better. A forward is latency-critical: the
/// coordinator is holding a segment timeout open waiting for it, so slowing the
/// bytes down does not produce a slower answer, it produces a *timeout* — and
/// the peer has then burned the compute for nothing and taken a serve-failure
/// penalty for it. Throttling converts a degraded success into a failure.
///
/// Bounding concurrency instead limits how many forwards can be in flight, and
/// therefore the peak, without stretching any single one past its deadline. The
/// complementary half is admission: `state.metrics.peer_speed` measures each
/// peer's real prefill/decode rate and both sizes the segment timeouts and ranks
/// candidates, so a node on a thin uplink is chosen less often rather than
/// being handed work it will fail to deliver. Refusing work you cannot serve is
/// the correct control here; serving it slowly is not.
fn max_concurrent_forwards(contribution: &swarmllm_types::ContributionMode) -> usize {
    match contribution {
        swarmllm_types::ContributionMode::Minimal => 8,
        swarmllm_types::ContributionMode::Moderate => 24,
        swarmllm_types::ContributionMode::Maximum => MAX_CONCURRENT_FORWARDS_MAX,
    }
}

/// Concurrent forwards from ONE peer, so a single peer cannot take the whole
/// semaphore. Half the total, floored at 4 — below that a tensor-parallel group
/// (24 per-layer forwards per token step at full size) cannot make progress at
/// all, and the cap stops being a limit and starts being a failure.
fn max_forwards_per_peer(contribution: &swarmllm_types::ContributionMode) -> usize {
    (max_concurrent_forwards(contribution) / 2).max(4)
}
const MAX_NICKNAME_REGISTRY: usize = 10_000;
/// Interval for sweeping stale zero-count entries from peer_forward_counts.
const FORWARD_COUNTS_CLEANUP_SECS: u64 = 60;
/// Zstd compression level for tensor wire payloads.
pub(super) const ZSTD_COMPRESS_LEVEL: i32 = 3;
/// Maximum age (ms) for regional gossip messages before they're considered stale.
const GOSSIP_STALENESS_MS: u64 = 15 * 60 * 1000;
/// Clock-skew tolerance for gossip timestamp checks. Per gotcha #44, the
/// future-side check MUST be one-sided — `now_ms.saturating_sub(ts) > MAX`
/// silently accepts future-dated messages because saturating_sub returns 0
/// when ts > now. Future-dated messages then ride the entire 15-minute
/// staleness window AND beat any honest contemporaneous update on the
/// "most recent wins" comparison.
const GOSSIP_SKEW_MS: u64 = 30_000;
/// Maximum AllReduce partials in flight before dropping new TpAllReduceRequests (DoS guard).
const MAX_PENDING_TP_PARTIALS: usize = 512;
/// Maximum cached regional shard summaries (per region+model pair).
const MAX_REGION_SUMMARIES: usize = 10_000;
/// Maximum demand rate entries across all (model, region) pairs.
const MAX_DEMAND_ENTRIES: usize = 10_000;
/// R130: Cap entries per WishlistAnnouncement. Caps the wire size AND the
/// per-publisher slice of `state.models.foreign_wishlist`. 64 keeps the
/// headline (a swarm normally has dozens of models, not hundreds).
const MAX_WISHLIST_ANNOUNCE_ENTRIES: usize = 64;
/// R134: Cap entries per PoolModelAvailability gossip. Senders rank by
/// recent local-host activity; receivers reject announcements over this
/// cap. 128 is comfortable headroom over typical pool catalog sizes.
pub(crate) const MAX_POOL_MODEL_ANNOUNCE_ENTRIES: usize = 128;
/// R134: maximum cached `foreign_pool_catalog` entries across all pools.
/// On insertion, evicts the oldest entry by `received_at_ms` to stay
/// under the cap. Keeps memory bounded even under a hostile-publisher
/// flood.
pub const MAX_FOREIGN_POOL_CATALOG_ENTRIES: usize = 5_000;
/// R134: drop foreign_pool_catalog entries older than this. Two hours
/// matches `FOREIGN_WISHLIST_MAX_AGE_MS` — both signals expire on the
/// same cadence so a pool that goes dark stops appearing in discovery.
pub const FOREIGN_POOL_CATALOG_MAX_AGE_MS: u64 = 2 * 60 * 60 * 1000;
/// SEC: Cap shards per ShardAnnounce to prevent shard_holders memory exhaustion.
const MAX_SHARDS_PER_ANNOUNCE: usize = 512;
/// SEC: Cap blocks per PrefixCacheAnnounce. A 7B model at 64-token blocks
/// tops out at ~120 blocks per 8K prompt; 1024 leaves headroom for larger
/// contexts without unbounded growth.
const MAX_BLOCKS_PER_ANNOUNCE: usize = 1024;
/// Maximum age (s) for nickname gossip before rejection. Paired with
/// `GOSSIP_SKEW_MS` for the future-side check — kept in seconds because
/// nickname records carry `chrono::DateTime<Utc>` (not the `u64`-ms epoch
/// the other gossip helpers use). One-sided per gotcha #44.
const NICK_GOSSIP_MAX_AGE_SECS: i64 = 24 * 60 * 60;
/// Future-side skew tolerance (s) for nickname gossip. Same value as
/// `GOSSIP_SKEW_MS` but in seconds to match the i64-seconds age math.
const NICK_GOSSIP_SKEW_SECS: i64 = 30;

/// Generic one-sided staleness check (gotcha #44). Returns `true` when
/// the timestamp is within `[now - max_age, now + skew]`; `false` (with a
/// log tagged by `kind`) when future-dated past `skew` or older than
/// `max_age`. Time units must be consistent across `ts`/`now`/`max_age`
/// /`skew` — typically all-ms or all-secs. Pre-existing call sites with
/// `.abs()` symmetric windows silently double the effective replay
/// window (see gotcha #44 / #32) — never re-introduce that pattern.
pub(crate) fn timestamp_fresh_one_sided(
    ts: u64,
    now: u64,
    max_age: u64,
    skew: u64,
    kind: &'static str,
) -> bool {
    if ts > now.saturating_add(skew) {
        tracing::warn!(kind, ts, now, "Dropping future-dated gossip");
        return false;
    }
    let age = now.saturating_sub(ts);
    if age > max_age {
        tracing::debug!(kind, age, "Dropping stale gossip");
        return false;
    }
    true
}

/// One-sided staleness check for regional gossip messages (gotcha #44),
/// in milliseconds with the dispatch defaults (15 min staleness, 30s
/// skew). Centralised here so every gossip handler enforces the
/// invariant identically.
fn gossip_timestamp_fresh(ts_ms: u64, now_ms: u64, kind: &'static str) -> bool {
    timestamp_fresh_one_sided(ts_ms, now_ms, GOSSIP_STALENESS_MS, GOSSIP_SKEW_MS, kind)
}

/// Pipeline sealing: encrypt the token IDs in a LayerResult for the requester's X25519 key.
/// If `requester_node_id` is present, seals `token_ids` into `sealed_token_ids` and clears
/// the plaintext `token_ids`. Falls back silently on crypto errors (result sent unsealed).
pub(super) fn seal_layer_result(
    result: &mut crate::types::LayerResult,
    requester_node_id: Option<&[u8; 32]>,
) {
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

/// Estimate VRAM usage (in MiB) from shard files on disk (no model
/// loading). Sums every `shard_*.bin` byte length in `model_dir` and
/// scales by the requested layer-range fraction; if `total_layers == 0`
/// the full sum is returned unscaled.
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
    credit_peer_balances: Arc<arc_swap::ArcSwap<Vec<i64>>>,
    shared_state: &Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    let contribution = shared_state.config.node.contribution.clone();
    let forward_limit = max_concurrent_forwards(&contribution);
    let per_peer_limit = max_forwards_per_peer(&contribution);
    tracing::info!(
        ?contribution,
        forward_limit,
        per_peer_limit,
        "Peer-work concurrency set from the contribution level"
    );
    let forward_semaphore = Arc::new(tokio::sync::Semaphore::new(forward_limit));
    // SEC: Per-peer concurrent forward counter to prevent single-peer semaphore exhaustion
    let peer_forward_counts: Arc<
        dashmap::DashMap<crate::types::NodeId, std::sync::atomic::AtomicUsize>,
    > = Arc::new(dashmap::DashMap::new());
    let mut cleanup_interval =
        tokio::time::interval(std::time::Duration::from_secs(FORWARD_COUNTS_CLEANUP_SECS));
    cleanup_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
                    _ = shutdown_rx.changed() => {
                        if *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    _ = cleanup_interval.tick() => {
                        // Sweep stale zero-count entries from transient peers
                        peer_forward_counts.retain(|_, v| v.load(std::sync::atomic::Ordering::Relaxed) > 0);
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
                                                tracing::warn!(msg_type = "LayerResult", "message from unauthenticated peer — dropping");
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
                                        // Resolve via the choke point so a result
                                        // is only accepted from the node this
                                        // request is actually waiting on — a
                                        // failed-over request still has the
                                        // abandoned forward outstanding.
                                        if shared_state.resolve_pending_layer_result(
                                            Some(sender),
                                            result.clone(),
                                        ) {
                                            tracing::info!(
                                                request_id = %result.request_id,
                                                tokens = result.token_ids.len(),
                                                activations_bytes = result.activations.len(),
                                                finish = ?result.finish_reason,
                                                pending_remaining = shared_state.pending_layer_results.len(),
                                                "DIAG: LayerResult delivered to pipeline"
                                            );
                                        } else {
                                            // Hedge losers (R136 L2) and genuine timeouts both
                                            // arrive here. Hedge-loser is normal operation under
                                            // hedge_enabled, so debug-level — the rare genuine
                                            // timeout case loses some signal but operators can
                                            // still see it via -v.
                                            tracing::debug!(
                                                request_id = %result.request_id,
                                                tokens = result.token_ids.len(),
                                                finish = ?result.finish_reason,
                                                pending_count = shared_state.pending_layer_results.len(),
                                                "DIAG: No pending channel for LayerResult — timed out, duplicate, or hedge loser"
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
                                            tracing::warn!(msg_type = "LayerForward", "message without authenticated sender — dropping");
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
                                        // SEC: Per-peer concurrent forward limit to prevent single-peer exhaustion.
                                        // Use optimistic fetch_add, then revert on overshoot — load-then-check-then-add
                                        // would let two concurrent dispatcher iterations both pass the check at
                                        // MAX-1 and admit MAX+1 forwards from one peer.
                                        let peer_sender = authenticated_sender.clone().expect("guarded by Some check above");
                                        let peer_count = peer_forward_counts
                                            .entry(peer_sender.clone())
                                            .or_insert_with(|| std::sync::atomic::AtomicUsize::new(0));
                                        let prev = peer_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                        if prev >= per_peer_limit {
                                            peer_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                                            tracing::warn!(
                                                sender = %peer_sender,
                                                current = prev,
                                                max = per_peer_limit,
                                                "LayerForward rejected — per-peer limit reached"
                                            );
                                            continue;
                                        }
                                        let permit = match forward_semaphore.clone().try_acquire_owned() {
                                            Ok(p) => p,
                                            Err(_) => {
                                                // Decrement unconditionally — use the entry ref we already hold
                                                // to avoid racing with concurrent DashMap removal
                                                peer_count.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                                                tracing::warn!(sender = %peer_sender, "LayerForward rejected — forward semaphore full");
                                                continue;
                                            }
                                        };
                                        let ss = shared_state.clone();
                                        let ntx = network_tx.clone();
                                        let pfc = peer_forward_counts.clone();
                                        let ps = peer_sender;
                                        let forward_request_id = forward.request_id;
                                        let abort_registry = shared_state.clone();
                                        // The forward carries the authenticated sender's peer bytes
                                        // (set on the decrypt path), which is what the
                                        // disconnect sweep matches on.
                                        let coordinator_bytes =
                                            forward.sender_peer_bytes.clone().unwrap_or_default();
                                        // Shared with the task so a forward that finishes before the
                                        // abort handle is registered does not strand its entry.
                                        let forward_finished =
                                            std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                                        let finished_in_task = forward_finished.clone();
                                        let handle = tokio::spawn(async move {
                                            let _permit = permit;
                                            layer_forward::handle_layer_forward(ss.clone(), ntx, forward).await;
                                            // Whether it finished or was abandoned, this request is no
                                            // longer in flight — drop the abort handle so the map does
                                            // not accumulate one entry per forward ever received.
                                            ss.clear_inbound_forward_abort(
                                                &forward_request_id,
                                                &finished_in_task,
                                            );
                                            // Decrement per-peer count; remove entry if zero to prevent unbounded growth
                                            if let Some(c) = pfc.get(&ps) {
                                                let prev = c.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                                                drop(c); // release DashMap ref before remove
                                                if prev <= 1 {
                                                    pfc.remove(&ps);
                                                }
                                            }
                                        });
                                        // Registered after the spawn because the abort handle does not
                                        // exist until then; the helper withdraws the entry again if the
                                        // task has already finished. A `CancelInference` naming this
                                        // request can now actually stop it — aborting drops the future,
                                        // which drops the worker's ResponseGuard and cancels the
                                        // compute (R147).
                                        abort_registry.register_inbound_forward_abort(
                                            forward_request_id,
                                            handle.abort_handle(),
                                            coordinator_bytes,
                                            &forward_finished,
                                        );
                                    }
                                    // StreamingToken: route to registered streaming channel
                                    SwarmMessage::StreamingToken(ref token) => {
                                        let sender = match authenticated_sender {
                                            Some(ref s) => s,
                                            None => {
                                                tracing::warn!(msg_type = "StreamingToken", "message from unauthenticated peer — dropping");
                                                continue;
                                            }
                                        };
                                        if !shared_state.peer_registry.contains_key(sender) {
                                            tracing::warn!(sender = %sender, "StreamingToken from unknown peer — dropping");
                                            continue;
                                        }
                                        // Clone the sender to drop the DashMap Ref (read lock) before
                                        // awaiting send() or calling remove() — avoids deadlock.
                                        // Take the channel ONLY if this token
                                        // belongs to the attempt currently in
                                        // flight. A failed request keeps its id
                                        // when it is retried, so a late token
                                        // from the abandoned attempt looks
                                        // identical to a live one if routing
                                        // goes by id alone — and if that token
                                        // is the abandoned attempt's terminal
                                        // failure, it kills a healthy retry and
                                        // blames whichever peer had just taken
                                        // it over. Same lesson as
                                        // `PendingLayerResult::awaiting`.
                                        let maybe_tx = shared_state
                                            .streaming_token_txs
                                            .get(&token.request_id)
                                            .and_then(|r| {
                                                if authenticated_sender
                                                    .as_ref()
                                                    .is_some_and(|s| s == &r.expected_peer)
                                                {
                                                    Some(r.tx.clone())
                                                } else {
                                                    tracing::debug!(
                                                        request_id = %token.request_id,
                                                        "streaming token from a peer this request is no longer being served by — dropping"
                                                    );
                                                    None
                                                }
                                            });
                                        if let Some(tx) = maybe_tx {
                                            // `try_send`, so a client that is not reading cannot
                                            // block the dispatch loop for every other request.
                                            //
                                            // **Full and closed are opposite situations and used to
                                            // be handled identically.** A closed channel means the
                                            // request is over and the entry should go. A full one
                                            // means the consumer is momentarily behind — and
                                            // removing the channel for that discarded not one
                                            // token but EVERY REMAINING TOKEN of the reply, since
                                            // each later one then finds no sender and is dropped
                                            // silently. A moment of backpressure truncated the
                                            // whole answer.
                                            //
                                            // Order does not matter here: `StreamReassembler`
                                            // sequences by `token_id`, so handing a delayed token
                                            // to a task to deliver cannot reorder the reply. That
                                            // is what makes recovering it safe rather than merely
                                            // less bad.
                                            match tx.try_send(token.clone()) {
                                                Ok(()) => {}
                                                Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                                                    shared_state.streaming_token_txs.remove(&token.request_id);
                                                }
                                                Err(tokio::sync::mpsc::error::TrySendError::Full(tok)) => {
                                                    // At `warn`: nodes run at `info`, so the
                                                    // `debug!` this replaces was invisible in
                                                    // every real log — and it was the one line
                                                    // that could have said whether a truncated
                                                    // reply was the network losing packets or
                                                    // this node dropping them.
                                                    tracing::warn!(
                                                        request_id = %token.request_id,
                                                        token_id = tok.token_id,
                                                        "streaming token channel full — consumer is behind, delivering out of band"
                                                    );
                                                    tokio::spawn(async move {
                                                        // Bounded, so a vanished consumer cannot
                                                        // strand this task forever.
                                                        let _ = tx
                                                            .send_timeout(
                                                                tok,
                                                                std::time::Duration::from_secs(10),
                                                            )
                                                            .await;
                                                    });
                                                }
                                            }
                                        }
                                    }
                                    // Remote-generate fast path: single-segment coordinator
                                    // asks us to run the full decode locally and stream tokens.
                                    SwarmMessage::RemoteGenerateRequest(mut req) => {
                                        if let Some(ref sender) = authenticated_sender {
                                            if !shared_state.peer_registry.contains_key(sender) {
                                                tracing::warn!(sender = %sender, "RemoteGenerateRequest from unknown peer — dropping");
                                                continue;
                                            }
                                        } else {
                                            tracing::warn!(msg_type = "RemoteGenerateRequest", "message without authenticated sender — dropping");
                                            continue;
                                        }
                                        // Stamp the authenticated sender's peer bytes so the
                                        // handler knows where to send StreamingTokens back.
                                        if req.sender_peer_bytes.is_none() {
                                            if let Some(ref sender) = authenticated_sender {
                                                req.sender_peer_bytes = shared_state
                                                    .peer_id_map
                                                    .get(sender)
                                                    .map(|r| r.value().clone());
                                            }
                                        }
                                        let permit = match forward_semaphore.clone().try_acquire_owned() {
                                            Ok(p) => p,
                                            Err(_) => {
                                                tracing::warn!(sender = %authenticated_sender.as_ref().map(|s| s.to_string()).unwrap_or_default(), "RemoteGenerateRequest rejected — forward semaphore full");
                                                continue;
                                            }
                                        };
                                        let ss = shared_state.clone();
                                        let ntx = network_tx.clone();
                                        tokio::spawn(async move {
                                            let _permit = permit;
                                            remote_generate::handle_remote_generate_request(ss, ntx, req).await;
                                        });
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
                                            tracing::warn!(msg_type = "VisionEncodeRequest", "message without authenticated sender — dropping");
                                            continue;
                                        }
                                        let permit = match forward_semaphore.clone().try_acquire_owned() {
                                            Ok(p) => p,
                                            Err(_) => {
                                                tracing::warn!(sender = %authenticated_sender.as_ref().map(|s| s.to_string()).unwrap_or_default(), "VisionEncodeRequest rejected — forward semaphore full");
                                                continue;
                                            }
                                        };
                                        let ss = shared_state.clone();
                                        let ntx = network_tx.clone();
                                        tokio::spawn(async move {
                                            let _permit = permit;
                                            vision::handle_vision_encode_request(ss, ntx, req).await;
                                        });
                                    }
                                    // T13: VisionEncodeResponse — fire pending oneshot
                                    SwarmMessage::VisionEncodeResponse(resp) => {
                                        // Atomically remove the entry, then validate the sender against
                                        // the stored expected_node. A peek+remove dance had a TOCTOU with
                                        // the health-monitor stale-entry sweep (health/monitor.rs:622)
                                        // that could remove the entry between the peek and the remove,
                                        // silently dropping a valid response.
                                        // On sender mismatch we re-insert so a later valid response can
                                        // still land — entries are bounded by request timeouts upstream.
                                        if let Some((_, (expected_node, tx))) = shared_state
                                            .pending_vision_results
                                            .remove(&resp.request_id)
                                        {
                                            match &authenticated_sender {
                                                Some(sender) if sender == &expected_node => {
                                                    let _ = tx.send(resp);
                                                }
                                                Some(sender) => {
                                                    tracing::warn!(
                                                        request_id = %resp.request_id,
                                                        expected = %expected_node,
                                                        actual = %sender,
                                                        "VisionEncodeResponse sender mismatch — dropping"
                                                    );
                                                    shared_state.pending_vision_results.insert(
                                                        resp.request_id,
                                                        (expected_node, tx),
                                                    );
                                                }
                                                None => {
                                                    tracing::warn!(
                                                        request_id = %resp.request_id,
                                                        "VisionEncodeResponse without authenticated sender — dropping"
                                                    );
                                                    shared_state.pending_vision_results.insert(
                                                        resp.request_id,
                                                        (expected_node, tx),
                                                    );
                                                }
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
                                            tracing::warn!(msg_type = "Inference", "message without authenticated sender — dropping");
                                            continue;
                                        }
                                        // try_send to avoid blocking the dispatch loop on a backlogged
                                        // router channel. A full router queue is the expected backpressure
                                        // signal — drop the inbound inference message and log; the peer
                                        // will retry. Same policy as the StreamingToken path above.
                                        if let Err(e) = router_tx
                                            .try_send(RouterCommand::NetworkMessage(msg))
                                        {
                                            tracing::warn!(
                                                error = %e,
                                                "Failed to route inference message to router (channel full or closed)"
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
                                            Some(&shared_state.credits.peer_credit_balances),
                                        ).await;
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
                                        // SEC: Freshness window — gotcha #32 / #44 one-sided staleness.
                                        // Shared invariant with the balance report (same skew + max age).
                                        if let Err(e) = crate::credit::ledger::check_signed_freshness(
                                            tx.timestamp,
                                            crate::credit::ledger::CLOCK_SKEW_TOLERANCE_SECS,
                                            crate::credit::ledger::BALANCE_REPORT_MAX_AGE_SECS,
                                            "credit tx",
                                        ) {
                                            tracing::warn!(tx_id = %tx.id, error = %e, "Rejecting credit tx");
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
                                            // verify_single_signatures checks both signatures; replay already checked above
                                            if let Err(e) = crate::credit::transaction::verify_single_signatures(&tx, &from_key, &to_key) {
                                                tracing::warn!(
                                                    tx_id = %tx.id,
                                                    error = %e,
                                                    "Credit tx rejected: signature verification failed"
                                                );
                                                continue;
                                            }
                                        }
                                        // Anti-gaming validation for network transactions.
                                        // Use try_lock to avoid blocking the dispatch loop on contention —
                                        // the periodic AG sweep in health/monitor.rs holds the same mutex for
                                        // cleanup. Skipping a check on contention is acceptable: the dispatcher
                                        // has already verified signatures + replay; AG just adds rate-window
                                        // and subnet heuristics. Same pattern as health/monitor.rs:128.
                                        match shared_state.credits.anti_gaming.try_lock() {
                                            Ok(mut ag) => {
                                                match ag.check_and_record_transaction(&tx.from, &tx.to, tx.amount) {
                                                    Ok(decision) => {
                                                        if decision == crate::credit::anti_gaming::SpotCheckDecision::RequiresVerification {
                                                            tracing::info!(
                                                                tx_id = %tx.id,
                                                                from = %tx.from,
                                                                to = %tx.to,
                                                                amount = tx.amount,
                                                                "Anti-gaming: spot check recommended for transaction"
                                                            );
                                                        }
                                                    }
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
                                            Err(_) => {
                                                tracing::debug!(
                                                    tx_id = %tx.id,
                                                    "anti_gaming contended, skipping rate-window check"
                                                );
                                            }
                                        }
                                        // Record the transaction and apply balance change
                                        // if we are the recipient
                                        let local_id = shared_state.identity.node_id().clone();
                                        if tx.to == local_id {
                                            if let Err(e) = crate::credit::ledger::apply_credit_direct_noted(
                                                &shared_state.credits.credit_balance,
                                                &shared_state.db,
                                                tx.amount,
                                                crate::credit::ledger::CreditDelta::Earning,
                                            "peer_credit_tx_in").await {
                                                tracing::warn!(error = %e, "Failed to apply credit transaction");
                                            }
                                            let bal = shared_state.credits.credit_balance.read().await;
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
                                        if announce.shards.len() > MAX_SHARDS_PER_ANNOUNCE {
                                            tracing::warn!(
                                                node_id = %announce.node_id,
                                                shards = announce.shards.len(),
                                                max = MAX_SHARDS_PER_ANNOUNCE,
                                                "ShardAnnounce exceeds shard limit — dropping"
                                            );
                                            continue;
                                        }
                                        // Reject oversized model_id strings (memory DoS prevention)
                                        if announce.shards.iter().any(|s| s.model_id.0.len() > 256) {
                                            tracing::warn!(
                                                node_id = %announce.node_id,
                                                "ShardAnnounce contains oversized model_id — dropping"
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
                                            // Don't record holders for a backup-copy model name —
                                            // it would inflate replica counts for a model that
                                            // isn't real. Mirrors the manifest-ingress guard.
                                            if crate::model::manifest::is_backup_artifact_id(&shard_id.model_id.0) {
                                                continue;
                                            }
                                            shared_state.model_registry
                                                .record_shard_holder(shard_id.clone(), announce.node_id.clone());
                                            *models_announced.entry(shard_id.model_id.0.clone()).or_insert(0) += 1;
                                        }
                                        // Retract whatever this node no longer holds, for the
                                        // models it declared complete. Additive-only handling
                                        // left a peer claiming shards it had deleted, and the
                                        // scheduler kept assigning it layers it could not serve.
                                        // Runs AFTER the inserts above so an announce that both
                                        // adds and drops shards of one model lands atomically
                                        // from the scheduler's point of view.
                                        for model_id in &announce.complete_for_models {
                                            let keep: std::collections::HashSet<u32> = announce
                                                .shards
                                                .iter()
                                                .filter(|s| s.model_id == *model_id)
                                                .map(|s| s.index)
                                                .collect();
                                            let dropped = shared_state.model_registry
                                                .retain_node_shards_for_model(model_id, &announce.node_id, &keep);
                                            if dropped > 0 {
                                                tracing::info!(
                                                    node_id = %announce.node_id,
                                                    model = %model_id,
                                                    dropped,
                                                    retained = keep.len(),
                                                    "Peer retracted shards it no longer hosts"
                                                );
                                            }
                                        }
                                        // Structured record of what was actually ingested.
                                        // The activity events below are a bounded ring buffer, so
                                        // a busy peer's entries displace a quiet one's — which made
                                        // "this peer never announced model X" indistinguishable
                                        // from "its entry scrolled off" while investigating a peer
                                        // that held the shards but never became a routing
                                        // candidate. This line is per-announce and does not scroll.
                                        tracing::debug!(
                                            node_id = %announce.node_id,
                                            shards_in_announce = announce.shards.len(),
                                            models = ?models_announced,
                                            complete_for = ?announce.complete_for_models.len(),
                                            "DIAG: shard announce ingested"
                                        );
                                        // Emit activity for each model announced
                                        let peer_label = crate::identity::nickname::short_display_name(
                                            &announce.node_id,
                                            &shared_state.nickname_registry,
                                        );
                                        for (mid, count) in &models_announced {
                                            let mname = shared_state.model_registry
                                                .get_manifest(&crate::types::ModelId(mid.clone()))
                                                .map(|m| m.name.clone());
                                            shared_state.emit_activity(crate::daemon::state::ActivityEvent::new(
                                                "model",
                                                "shard_announced",
                                                format!("{} announced {} shard{} of {}", peer_label, count, if *count != 1 { "s" } else { "" }, mname.as_deref().unwrap_or(mid)),
                                            )
                                            .with_model(mid.clone())
                                            .with_node(format!("{}", announce.node_id))
                                            .with_detail_num(*count as i64));
                                        }
                                        // Wake auto-manage so it re-evaluates rarity scores —
                                        // new shard holders change which shards are most needed.
                                        shared_state.models.auto_manage_notify.notify_one();
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
                                        // Refuse copied-folder model names (`<model>.FULLBACKUP`,
                                        // `<model>.old`) at the network boundary. A peer on an
                                        // older build still re-gossips these; accepting one lets
                                        // a stale local-copy name spread swarm-wide under an id no
                                        // node can ever resolve to a real download. register_manifest
                                        // also nets these, but rejecting here skips the auto-manage
                                        // wake and surfaces *why* to the operator.
                                        if crate::model::manifest::is_backup_artifact_id(&manifest.id.0) {
                                            tracing::warn!(
                                                model = %manifest.id,
                                                publisher = %manifest.publisher,
                                                "Rejecting peer manifest: backup-copy name"
                                            );
                                            if let Some(ref sender) = authenticated_sender {
                                                shared_state.emit_activity(
                                                    crate::daemon::state::ActivityEvent::new(
                                                        "security",
                                                        "manifest_rejected_backup",
                                                        format!(
                                                            "Rejected \"{}\" from {}: looks like a local backup copy, not a real model",
                                                            manifest.id.0, sender
                                                        ),
                                                    )
                                                    .with_model(manifest.id.0.clone())
                                                    .with_node(format!("{}", sender))
                                                    .with_toast("warning", 6000),
                                                );
                                            }
                                            continue;
                                        }
                                        // DEBUG, not INFO: `register_manifest` below is the
                                        // choke point every adoption path funnels through, and
                                        // it announces at INFO when the manifest is genuinely
                                        // new or changed. Announcing here too meant a settled
                                        // swarm logged both lines on every re-gossip of every
                                        // unchanged manifest — together roughly half the volume
                                        // of a real node's log. Keep this one for tracing the
                                        // network path specifically.
                                        tracing::debug!(
                                            model = %manifest.id,
                                            name = %manifest.name,
                                            shards = manifest.shard_count,
                                            publisher = %manifest.publisher,
                                            "Received model manifest from network"
                                        );
                                        // SEC (R105): cap peer-gossiped manifest size BEFORE
                                        // accepting. The manifest_hash check is self-
                                        // referential — a peer can compute a valid hash over
                                        // ANY payload they construct, including one with
                                        // 100k tensor entries × 100 shards inflating registry
                                        // memory by hundreds of MB per gossiped manifest.
                                        // Real-world manifests have ≤ 256 shards × ≤ 16k
                                        // tensor entries each (the 70B-class Llama upper
                                        // bound). Anything beyond is hostile/malformed.
                                        const MAX_SHARDS_PER_MANIFEST: usize = 256;
                                        const MAX_TENSORS_PER_SHARD: usize = 16_384;
                                        if manifest.shards.len() > MAX_SHARDS_PER_MANIFEST {
                                            tracing::warn!(
                                                model = %manifest.id,
                                                shards = manifest.shards.len(),
                                                cap = MAX_SHARDS_PER_MANIFEST,
                                                "Rejecting peer manifest: too many shards"
                                            );
                                            if let Some(ref sender) = authenticated_sender {
                                                shared_state.emit_activity(
                                                    crate::daemon::state::ActivityEvent::new(
                                                        "security",
                                                        "manifest_rejected",
                                                        format!(
                                                            "Rejected manifest from {}: {} shards exceeds cap of {}",
                                                            sender, manifest.shards.len(), MAX_SHARDS_PER_MANIFEST
                                                        ),
                                                    )
                                                    .with_model(manifest.id.0.clone())
                                                    .with_node(format!("{}", sender))
                                                    .with_detail_num(manifest.shards.len() as i64)
                                                    .with_toast("warning", 6000),
                                                );
                                            }
                                            continue;
                                        }
                                        let oversize_tensor_count = manifest
                                            .shards
                                            .iter()
                                            .map(|s| s.tensors.len())
                                            .filter(|n| *n > MAX_TENSORS_PER_SHARD)
                                            .max();
                                        if let Some(n) = oversize_tensor_count {
                                            tracing::warn!(
                                                model = %manifest.id,
                                                cap = MAX_TENSORS_PER_SHARD,
                                                "Rejecting peer manifest: a shard has too many tensor entries"
                                            );
                                            if let Some(ref sender) = authenticated_sender {
                                                shared_state.emit_activity(
                                                    crate::daemon::state::ActivityEvent::new(
                                                        "security",
                                                        "manifest_rejected",
                                                        format!(
                                                            "Rejected manifest from {}: {} tensors in a shard exceeds cap of {}",
                                                            sender, n, MAX_TENSORS_PER_SHARD
                                                        ),
                                                    )
                                                    .with_model(manifest.id.0.clone())
                                                    .with_node(format!("{}", sender))
                                                    .with_detail_num(n as i64)
                                                    .with_toast("warning", 6000),
                                                );
                                            }
                                            continue;
                                        }
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
                                                    shared_state.models.auto_manage_notify.notify_one();
                                                    shared_state.emit_activity(crate::daemon::state::ActivityEvent::new(
                                                        "model",
                                                        "model_discovered",
                                                        format!(
        "Discovered new model on network: {} — {:?} arch, {} layers, {} shards",
        manifest.name, manifest.architecture, manifest.num_layers, manifest.shard_count
        ),
                                                    )
                                                    .with_model(manifest.id.0.clone())
                                                    .with_model_name(manifest.name.clone())
                                                    .with_detail_num(manifest.shard_count as i64)
                                                    .with_detail_str(format!("{:?}", manifest.architecture)));
                                                }
                                            }
                                            Err(e) => {
                                                // Name the model AND the sender.
                                                // Without them this line is
                                                // unattributable: it fired 3189
                                                // times in 90 minutes on this
                                                // node carrying nothing but two
                                                // hex strings, so neither an
                                                // operator nor a maintainer
                                                // could tell which model was
                                                // being rejected, whether it was
                                                // one peer or thirteen, or
                                                // whether it mattered. The
                                                // sibling arm above already
                                                // reports both.
                                                tracing::warn!(
                                                    error = %e,
                                                    model = %manifest.id,
                                                    model_name = %manifest.name,
                                                    from_peer = ?authenticated_sender,
                                                    shard_count = manifest.shard_count,
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
                                        // SEC: cap inbound vec sizes before iterating. Each
                                        // observed_latencies entry inserts into a shared
                                        // DashMap via merge_peer_segment_latency — without
                                        // a length cap a malicious peer can drive memory
                                        // growth via a single capability update. Mirrors the
                                        // ShardAnnounce/PrefixCacheAnnounce/RegionShardSummary
                                        // pattern (already capped at 512/1024/512). Same
                                        // concern for hosted_shards (bounded already by the
                                        // single peer.capability slot, but a 100k-entry vec
                                        // would still OOM the deserialiser).
                                        const MAX_OBSERVED_LATENCIES: usize = 256;
                                        const MAX_HOSTED_SHARDS_IN_CAP: usize = 1024;
                                        if cap.observed_latencies.len() > MAX_OBSERVED_LATENCIES
                                            || cap.hosted_shards.len() > MAX_HOSTED_SHARDS_IN_CAP
                                        {
                                            tracing::warn!(
                                                node_id = %cap.node_id,
                                                obs = cap.observed_latencies.len(),
                                                hosted = cap.hosted_shards.len(),
                                                "NodeCapabilityUpdate: payload exceeds caps — dropping"
                                            );
                                            continue;
                                        }
                                        tracing::debug!(
                                            node_id = %cap.node_id,
                                            hosted_shards = cap.hosted_shards.len(),
                                            observed_latencies = cap.observed_latencies.len(),
                                            "Received capability update from peer"
                                        );
                                        // Merge the sender's observed-latency snapshot into
                                        // our local EMA before installing the capability (so
                                        // the snapshot doesn't have to survive the clone).
                                        // Trust-weighted: weight = sender's trust score; a
                                        // zero-trust sender is a no-op. Own-node observations
                                        // are skipped — we already have direct samples.
                                        let sender_trust = shared_state
                                            .peer_registry
                                            .get(&cap.node_id)
                                            .map(|p| p.trust_score)
                                            .unwrap_or(0.5);
                                        let own_id = shared_state.identity.node_id();
                                        for obs in &cap.observed_latencies {
                                            if &obs.peer == own_id || obs.peer == cap.node_id {
                                                continue;
                                            }
                                            shared_state.merge_peer_segment_latency(
                                                &obs.peer,
                                                obs.ms_per_layer,
                                                sender_trust,
                                            );
                                        }
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
                                        // Age check: one-sided per gotcha #44. Without the future-dated
                                        // rejection, an attacker can pre-sign with `timestamp = now+23.9h`
                                        // and the record will win the timestamp-tiebreaker for the next day,
                                        // squatting any peer's nickname.
                                        let age_secs = (chrono::Utc::now() - record.timestamp).num_seconds();
                                        if age_secs < -NICK_GOSSIP_SKEW_SECS {
                                            tracing::debug!(
                                                node_id = %record.node_id,
                                                age_secs,
                                                "Rejecting future-dated nickname gossip"
                                            );
                                        } else if age_secs > NICK_GOSSIP_MAX_AGE_SECS {
                                            tracing::debug!(
                                                node_id = %record.node_id,
                                                age_secs,
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
                                                && shared_state.nickname_registry.len() >= MAX_NICKNAME_REGISTRY
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
                                                tracing::warn!(msg_type = "PoolMessage", "message without authenticated sender — dropping");
                                                continue;
                                            }
                                        };
                                        // SEC: Verify inner identity matches authenticated sender
                                        // to prevent spoofing pool messages from other nodes
                                        let inner_ok = match &pool_msg {
                                            crate::types::PoolMessage::CreditForward(fwd) => fwd.from_node_id == *sender,
                                            crate::types::PoolMessage::MemberLeft { node_id, .. } => node_id == sender,
                                            crate::types::PoolMessage::JoinRequest { requester, .. } => requester == sender,
                                            crate::types::PoolMessage::DeviceStatsReport { node_id, .. } => node_id == sender,
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
                                        // Clone the sender out of the RwLock to drop the read guard
                                        // BEFORE awaiting send(). Holding the guard across send().await
                                        // would block the dispatch loop AND block any writer trying to
                                        // install/replace pool_tx (gotcha: tokio RwLock starves writers
                                        // while readers are parked).
                                        let pool_tx_clone = shared_state
                                            .credits
                                            .pool_tx
                                            .read()
                                            .await
                                            .as_ref()
                                            .cloned();
                                        if let Some(tx) = pool_tx_clone {
                                            let cmd = match pool_msg {
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
                                                crate::types::PoolMessage::StateDiff(diff) => {
                                                    Some(crate::pool::types::PoolCommand::PoolStateDiffGossip {
                                                        diff,
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
                                                crate::types::PoolMessage::MemberLeft { pool_id, node_id, left_at, nonce, signature } => {
                                                    Some(crate::pool::types::PoolCommand::InboundMemberLeft {
                                                        pool_id,
                                                        node_id,
                                                        left_at,
                                                        nonce,
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
                                                crate::types::PoolMessage::DeviceStatsReport { pool_id, node_id, device_name, stats } => {
                                                    Some(crate::pool::types::PoolCommand::InboundDeviceStatsReport {
                                                        pool_id,
                                                        node_id,
                                                        device_name,
                                                        stats,
                                                    })
                                                }
                                            };
                                            if let Some(cmd) = cmd {
                                                // Use try_send so a slow PoolManager (large
                                                // StateGossip merge, slow DB write) doesn't
                                                // block the dispatch loop and starve every
                                                // other inbound message (LayerResult,
                                                // CreditTransaction, …). Consistent with the
                                                // InferenceRequest / StreamingToken paths.
                                                if let Err(e) = tx.try_send(cmd) {
                                                    shared_state
                                                        .metrics
                                                        .channel_metrics
                                                        .network_out
                                                        .record_dropped();
                                                    tracing::warn!(error = %e, "Failed to route pool message (channel full or closed)");
                                                } else {
                                                    shared_state
                                                        .metrics
                                                        .channel_metrics
                                                        .network_out
                                                        .record_sent();
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
                                        // SEC: cap distinct hf_sources entries. Without this, a
                                        // peer can gossip thousands of unique model_ids to inflate
                                        // the in-memory map and the persisted "hf_sources" tree.
                                        const MAX_HF_SOURCES: usize = 1024;
                                        if !shared_state.models.hf_sources.contains_key(&mid)
                                            && shared_state.models.hf_sources.len() >= MAX_HF_SOURCES
                                        {
                                            tracing::warn!(
                                                model = %mid,
                                                cap = MAX_HF_SOURCES,
                                                "HfSourceGossip dropped — hf_sources at capacity"
                                            );
                                            // R141: throttled activity event so the dashboard
                                            // surfaces "we're missing models" instead of silently
                                            // dropping. fetch_add is per-process; first drop +
                                            // every 50th after fires an event — caps log/UI noise
                                            // when a malicious peer floods unique model_ids.
                                            use std::sync::atomic::{AtomicU64, Ordering};
                                            static DROP_COUNTER: AtomicU64 = AtomicU64::new(0);
                                            let prev = DROP_COUNTER.fetch_add(1, Ordering::Relaxed);
                                            if prev == 0 || prev.is_multiple_of(50) {
                                                shared_state.emit_activity(
                                                    crate::daemon::state::ActivityEvent::new(
                                                        "capacity",
                                                        "hf_sources_cap_reached",
                                                        format!(
                                                            "Discovered model catalogue is full ({MAX_HF_SOURCES} entries). New models from peers are being dropped — remove unused models in Settings to free slots."
                                                        ),
                                                    )
                                                    .with_detail_num(MAX_HF_SOURCES as i64)
                                                    .with_toast("warning", 6000),
                                                );
                                            }
                                            continue;
                                        }
                                        if !shared_state.models.hf_sources.contains_key(&mid) {
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
                                            shared_state.models.hf_sources.insert(mid.clone(), source.clone());
                                            // Persist to DB
                                            let _ = shared_state.db.put_json("hf_sources", &mid.0, &source);
                                            // Also write hf_source.json to disk so discover_hf_sources finds it on restart
                                            let model_dir = shared_state.model_dir(&mid.0);
                                            {
                                                let json_str = serde_json::to_string_pretty(&source).unwrap_or_default();
                                                tokio::task::spawn_blocking(move || {
                                                    if model_dir.is_dir() {
                                                        let hf_path = model_dir.join(crate::model::shard::HF_SOURCE_FILENAME);
                                                        if !hf_path.exists() {
                                                            let _ = std::fs::write(&hf_path, json_str);
                                                        }
                                                    }
                                                });
                                            }
                                            // Wake the AutoShardManager so it evaluates promptly
                                            shared_state.models.auto_manage_notify.notify_one();
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
                                                if let Some(mut entry) = shared_state.models.peer_shard_downloads.get_mut(&progress.shard_id) {
                                                    entry.retain(|(nid, _)| *nid != progress.node_id);
                                                }
                                                // Register the peer as a shard holder now
                                                // (the ShardAnnounce gossip will also arrive,
                                                //  but this gives immediate consistency)
                                                shared_state.model_registry
                                                    .record_shard_holder(progress.shard_id.clone(), progress.node_id.clone());
                                                // Wake auto-manage — peer completed a download, rarity changed
                                                shared_state.models.auto_manage_notify.notify_one();
                                            } else {
                                                // Update or insert download progress.
                                                // Cap the per-shard list to avoid unbounded growth when
                                                // many peers race to download a popular shard — each access
                                                // of this Vec is linear, so uncapped growth creates an
                                                // O(n) scan on every gossip message. When full, evict the
                                                // highest-progress entry: near-complete peers will self-remove
                                                // via the completion path (is_complete branch above) within
                                                // seconds, so preemptively evicting them costs little. The
                                                // in-progress peers with lower pct carry the more useful
                                                // operational signal and stay visible.
                                                const MAX_PEER_DOWNLOADS_PER_SHARD: usize = 64;
                                                let mut entry = shared_state.models.peer_shard_downloads.entry(progress.shard_id.clone()).or_default();
                                                if let Some(pos) = entry.iter().position(|(nid, _)| *nid == progress.node_id) {
                                                    entry[pos].1 = progress.progress_pct;
                                                } else {
                                                    if entry.len() >= MAX_PEER_DOWNLOADS_PER_SHARD {
                                                        if let Some((max_pos, _)) = entry
                                                            .iter()
                                                            .enumerate()
                                                            .max_by_key(|(_, (_, pct))| *pct)
                                                        {
                                                            entry.swap_remove(max_pos);
                                                        }
                                                    }
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
                                        let ts = crate::types::unix_now_secs();
                                        let our_load = shared_state.active_inference_load();
                                        let our_id = Some(shared_state.identity.node_id().clone());
                                        let pong = SwarmMessage::HealthPong {
                                            nonce,
                                            timestamp: ts,
                                            node_id: our_id,
                                            active_request_count: our_load,
                                        };
                                        // Unicast pong to the pinger instead of broadcasting O(N²)
                                        // try_send so a saturated network_tx
                                        // (large tensor transfer in flight) can't
                                        // block the dispatch loop. A missed pong
                                        // recovers next health-monitor tick.
                                        //
                                        // The ping reached us over the gossipsub mesh, which
                                        // relays it from peers we hold no direct connection to.
                                        // The pong goes back over request_response, which does
                                        // not relay HealthPong (`is_relay_eligible` refuses it),
                                        // so a departed-but-still-gossiping peer would otherwise
                                        // get an undeliverable send every 30s forever. Resolve
                                        // through the liveness oracle, not `peer_id_map`, which
                                        // survives disconnects by design.
                                        match shared_state.resolve_connected_peer_id_bytes(&sender_id) {
                                            Some(peer_bytes) => {
                                                if let Err(e) = network_tx.try_send(NetworkCommand::SendDirectMessage {
                                                    target_peer_bytes: peer_bytes,
                                                    message: pong,
                                                    delivery_request_id: None,
                                                }) {
                                                    tracing::debug!(error = %e, "Dropping HealthPong: network_tx busy");
                                                }
                                            }
                                            // Connected, but no PeerId mapping yet — broadcast so
                                            // the pinger still learns our load. Requires an active
                                            // connection, so this cannot become the 30s loop above.
                                            None if shared_state.connected_node_ids.contains(&sender_id) => {
                                                if let Err(e) = network_tx.try_send(NetworkCommand::Broadcast(pong)) {
                                                    tracing::debug!(error = %e, "Dropping HealthPong broadcast: network_tx busy");
                                                }
                                            }
                                            None => {
                                                tracing::trace!(
                                                    peer = %sender_id,
                                                    "Skipping HealthPong — ping arrived via gossip mesh but peer is not connected"
                                                );
                                            }
                                        }
                                    }
                                    // Health pongs: update the sender's load in peer_registry
                                    SwarmMessage::HealthPong { node_id: Some(sender_id), active_request_count, nonce, .. } => {
                                        // SEC: Verify sender matches the health pong's node_id
                                        if let Some(ref sender) = authenticated_sender {
                                            if sender != &sender_id {
                                                continue;
                                            }
                                        } else {
                                            tracing::debug!("Dropping unauthenticated HealthPong");
                                            continue;
                                        }
                                        // The round trip we just completed IS a
                                        // latency measurement, and it was being
                                        // thrown away. `latency_ms` was written from
                                        // exactly one other place — an occasional
                                        // rr_ping/PEX exchange — so what the dashboard
                                        // and `/api/admin/peers` present as a peer's
                                        // latency was an artefact of whenever that last
                                        // happened, `None` for peers it never happened
                                        // to, and never refreshed. Observed 2026-08-05:
                                        // two of three connected peers read `lat=None`
                                        // after 47 minutes of healthy two-way traffic.
                                        //
                                        // Not merely cosmetic: `tp_max_latency_ms`
                                        // admits peers to a tensor-parallel group on
                                        // this number.
                                        //
                                        // Only the CURRENT nonce counts. A pong echoing
                                        // an older one arrived after the next ping went
                                        // out, so measuring it against the newer send
                                        // time would report a far-too-small RTT.
                                        let rtt_ms = {
                                            let guard = shared_state.last_health_ping.lock();
                                            match *guard {
                                                Some((sent_nonce, sent_at)) if sent_nonce == nonce => {
                                                    Some(sent_at.elapsed().as_millis().min(u32::MAX as u128) as u32)
                                                }
                                                _ => None,
                                            }
                                        };
                                        if let Some(mut peer) = shared_state.peer_registry.get_mut(&sender_id) {
                                            peer.active_request_count = active_request_count;
                                            peer.last_seen = chrono::Utc::now();
                                            if let Some(rtt) = rtt_ms {
                                                peer.latency_ms = Some(rtt);
                                            }
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
                                            // Direct-only: EphemeralKeyExchange is not relay-eligible,
                                            // so a target we no longer hold a connection to can only
                                            // be dropped by the send path.
                                            let target = shared_state
                                                .resolve_connected_peer_id_bytes(&exchange.node_id);
                                            if let Some(target_bytes) = target {
                                                // try_send to keep the dispatch
                                                // loop from blocking on a saturated
                                                // network_tx. A dropped key reply
                                                // re-runs on the next exchange.
                                                if let Err(e) = network_tx.try_send(NetworkCommand::SendDirectMessage {
                                                    target_peer_bytes: target_bytes,
                                                    message: reply,
                                                    delivery_request_id: None,
                                                }) {
                                                    tracing::warn!(error = %e, "Dropping ephemeral key reply: network_tx busy");
                                                }
                                            } else {
                                                tracing::debug!(
                                                    node_id = %exchange.node_id,
                                                    "Cannot reply to ephemeral key exchange — peer not connected or no PeerId mapping"
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

                                        // Cap check: reject when key is new AND we're at the
                                        // limit. The contains_key + len check + entry().or_insert
                                        // sequence below is NOT atomic across multiple
                                        // dispatch_network_messages tasks, but only this single
                                        // task touches pending_tp_partials inserts on the
                                        // dispatch side, so the visible window is one iteration.
                                        // If this loop is ever parallelised, restructure with
                                        // entry-first to keep the cap exact.
                                        if !ss.pending_tp_partials.contains_key(&key)
                                            && ss.pending_tp_partials.len() >= MAX_PENDING_TP_PARTIALS
                                        {
                                            tracing::warn!(capacity = MAX_PENDING_TP_PARTIALS, "pending_tp_partials full — dropping TpAllReduceRequest");
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
                                                tracing::warn!(msg_type = "TpAllReduceResponse", "message from unauthenticated peer — dropping");
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
                                    SwarmMessage::TpRingChunk(chunk) => {
                                        // Ring AllReduce chunk: route to the allreduce registry
                                        match authenticated_sender {
                                            Some(ref sender) => {
                                                if !shared_state.peer_registry.contains_key(sender) {
                                                    tracing::warn!(sender = %sender, "TpRingChunk from unknown peer — dropping");
                                                    continue;
                                                }
                                            }
                                            None => {
                                                tracing::warn!(msg_type = "TpRingChunk", "message from unauthenticated peer — dropping");
                                                continue;
                                            }
                                        }
                                        let delivered = shared_state.ring_chunk_registry.deliver(
                                            chunk.request_id,
                                            chunk.layer_idx,
                                            chunk.step,
                                            chunk.chunk_data.clone(),
                                        );
                                        tracing::debug!(
                                            request_id = %chunk.request_id,
                                            layer_idx = chunk.layer_idx,
                                            step = chunk.step,
                                            chunk_idx = chunk.chunk_idx,
                                            is_allgather = chunk.is_allgather,
                                            delivered,
                                            "Ring AllReduce chunk received"
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
                                        if summary.region.len() > 8 || summary.shard_counts.len() > 512
                                            || summary.model_id.0.len() > 256
                                        {
                                            continue;
                                        }
                                        // Don't track region availability for a backup-copy model
                                        // name (`<model>.FULLBACKUP`) — see manifest ingress guard.
                                        if crate::model::manifest::is_backup_artifact_id(&summary.model_id.0) {
                                            continue;
                                        }
                                        let now_ms = crate::types::unix_now_ms();
                                        if !gossip_timestamp_fresh(summary.timestamp_ms, now_ms, "RegionShardSummary") {
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
                                        if demand.region.len() > 8 || demand.model_id.0.len() > 256 {
                                            continue;
                                        }
                                        let now_ms = crate::types::unix_now_ms();
                                        if !gossip_timestamp_fresh(demand.timestamp_ms, now_ms, "ModelDemandGossip") {
                                            continue;
                                        }
                                        // SEC: reject NaN/Inf from peer-supplied f64. Without this,
                                        // a single gossiped `decayed_rate: NaN` poisons the EMA
                                        // blend permanently for that (model, region) pair —
                                        // every subsequent `*existing * 0.8 + NaN * 0.2` stays NaN
                                        // until restart, corrupting auto-manage replication scoring.
                                        if !demand.decayed_rate.is_finite() {
                                            tracing::warn!(
                                                model = %demand.model_id,
                                                region = %demand.region,
                                                rate = demand.decayed_rate,
                                                "ModelDemandGossip non-finite decayed_rate — dropping"
                                            );
                                            continue;
                                        }
                                        let key = (demand.model_id.clone(), demand.region.clone());
                                        if shared_state.region_demand.len() >= MAX_DEMAND_ENTRIES
                                            && !shared_state.region_demand.contains_key(&key)
                                        {
                                            continue;
                                        }
                                        // EMA blend: 0.8 * old + 0.2 * incoming.
                                        // SEC: also guard the cached `*existing` against
                                        // NaN — gotcha #98 closed the receive-side guard
                                        // on `decayed_rate`, but a stale NaN from a
                                        // pre-R102 DB rehydrate or local-decay race
                                        // (manager.rs::decay_request_counts) still
                                        // permanently poisons the EMA: `NaN * 0.8 + x * 0.2 = NaN`.
                                        let new_rate = if let Some(existing) = shared_state.region_demand.get(&key) {
                                            let prev = if existing.is_finite() { *existing } else { 0.0 };
                                            prev * 0.8 + demand.decayed_rate * 0.2
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

                                    // R130: cross-pool wishlist gossip. Publisher's top-K
                                    // wishlist entries; receivers blend the foreign interest
                                    // into their own wishlist score as a soft boost.
                                    SwarmMessage::WishlistAnnouncement(announce) => {
                                        match &authenticated_sender {
                                            Some(sender) if *sender != announce.publisher => {
                                                tracing::warn!(
                                                    sender = %sender,
                                                    claimed = %announce.publisher,
                                                    "WishlistAnnouncement sender mismatch — dropping"
                                                );
                                                continue;
                                            }
                                            None => {
                                                tracing::debug!("Dropping unauthenticated WishlistAnnouncement");
                                                continue;
                                            }
                                            Some(_) => {}
                                        }
                                        // Skip self-announces — own wishlist already in `state.models.wishlist`.
                                        if announce.publisher == *shared_state.identity.node_id() {
                                            continue;
                                        }
                                        if announce.entries.len() > MAX_WISHLIST_ANNOUNCE_ENTRIES {
                                            tracing::warn!(
                                                publisher = %announce.publisher,
                                                entries = announce.entries.len(),
                                                max = MAX_WISHLIST_ANNOUNCE_ENTRIES,
                                                "WishlistAnnouncement exceeds entry cap — dropping"
                                            );
                                            continue;
                                        }
                                        let now_ms = crate::types::unix_now_ms();
                                        if !gossip_timestamp_fresh(announce.timestamp_ms, now_ms, "WishlistAnnouncement") {
                                            continue;
                                        }
                                        let foreign = &shared_state.models.foreign_wishlist;
                                        let pre_existing = announce.entries.iter().any(|e| {
                                            foreign.contains_key(&(announce.publisher.clone(), e.model_id.clone()))
                                        });
                                        if !pre_existing && foreign.len() >= crate::daemon::state::MAX_FOREIGN_WISHLIST_ENTRIES {
                                            tracing::debug!("foreign_wishlist at cap, dropping new WishlistAnnouncement");
                                            continue;
                                        }
                                        let entries_pairs: Vec<(crate::types::ModelId, u32)> = announce
                                            .entries
                                            .iter()
                                            .map(|e| (e.model_id.clone(), e.score))
                                            .collect();
                                        let (added, removed) = shared_state.models.apply_wishlist_announcement(
                                            announce.publisher.clone(),
                                            &entries_pairs,
                                            announce.timestamp_ms,
                                        );
                                        tracing::debug!(
                                            publisher = %announce.publisher,
                                            entries = announce.entries.len(),
                                            added,
                                            removed,
                                            "WishlistAnnouncement processed"
                                        );
                                    }

                                    // R134: inter-pool model availability. Pool owners opt in to
                                    // advertise which model_ids their pool can serve. Receivers
                                    // cache as a discovery signal — surfaces in the admin REST
                                    // surface as "Pool X also serves Y". Does NOT change routing.
                                    SwarmMessage::PoolModelAvailability(announce) => {
                                        use ed25519_dalek::Verifier;
                                        match &authenticated_sender {
                                            Some(sender) if *sender != announce.pool_id => {
                                                tracing::warn!(
                                                    sender = %sender,
                                                    claimed = %announce.pool_id,
                                                    "PoolModelAvailability sender mismatch — dropping"
                                                );
                                                continue;
                                            }
                                            None => {
                                                tracing::debug!("Dropping unauthenticated PoolModelAvailability");
                                                continue;
                                            }
                                            Some(_) => {}
                                        }
                                        if announce.pool_id == *shared_state.identity.node_id() {
                                            // Self-announce; nothing to learn.
                                            continue;
                                        }
                                        if announce.model_ids.len() > MAX_POOL_MODEL_ANNOUNCE_ENTRIES {
                                            tracing::warn!(
                                                pool = %announce.pool_id,
                                                count = announce.model_ids.len(),
                                                "PoolModelAvailability exceeds cap — dropping"
                                            );
                                            continue;
                                        }
                                        let now_ms = crate::types::unix_now_ms();
                                        if !gossip_timestamp_fresh(announce.timestamp_ms, now_ms, "PoolModelAvailability") {
                                            continue;
                                        }
                                        let owner_key = match ed25519_dalek::VerifyingKey::from_bytes(&announce.pool_id.0) {
                                            Ok(k) => k,
                                            Err(_) => {
                                                tracing::warn!(pool = %announce.pool_id, "Invalid pool owner key in PoolModelAvailability");
                                                continue;
                                            }
                                        };
                                        let payload = crate::pool::crypto::pool_model_availability_payload(
                                            &announce.pool_id,
                                            &announce.model_ids,
                                            announce.timestamp_ms,
                                        );
                                        let sig_bytes: &[u8; 64] = match announce.owner_signature.as_slice().try_into() {
                                            Ok(b) => b,
                                            Err(_) => {
                                                tracing::warn!(pool = %announce.pool_id, "Invalid signature length in PoolModelAvailability");
                                                continue;
                                            }
                                        };
                                        let sig = ed25519_dalek::Signature::from_bytes(sig_bytes);
                                        if owner_key.verify(&payload, &sig).is_err() {
                                            tracing::warn!(pool = %announce.pool_id, "Invalid owner signature in PoolModelAvailability");
                                            continue;
                                        }

                                        // R135 sec follow-up: enforce the k-anonymity floor on RECEIVE,
                                        // not just on PUBLISH. A malicious operator can disable the
                                        // floor on their own publisher; the receive-side check makes
                                        // ingestion conservative regardless. We can only check when
                                        // we've already received the foreign pool's PoolState via the
                                        // separate StateGossip channel — if we haven't, accept the
                                        // announcement (the next state gossip will confirm or refute
                                        // membership before any routing decisions actually fire).
                                        let local_min_members =
                                            shared_state.cfg().pool.share_model_catalog_min_members;
                                        if local_min_members > 1 {
                                            if let Some(ps_entry) =
                                                shared_state.credits.pool_registry.get(&announce.pool_id)
                                            {
                                                let cached_count = ps_entry.value().members.len();
                                                if (cached_count as u32) < local_min_members {
                                                    tracing::warn!(
                                                        pool = %announce.pool_id,
                                                        members = cached_count,
                                                        floor = local_min_members,
                                                        "PoolModelAvailability rejected: cached pool size below local k-anonymity floor"
                                                    );
                                                    continue;
                                                }
                                            }
                                        }

                                        shared_state.credits.apply_pool_model_availability(
                                            &announce.pool_id,
                                            &announce.model_ids,
                                            announce.timestamp_ms,
                                            now_ms,
                                            FOREIGN_POOL_CATALOG_MAX_AGE_MS,
                                            MAX_FOREIGN_POOL_CATALOG_ENTRIES,
                                        );
                                        tracing::debug!(
                                            pool = %announce.pool_id,
                                            entries = announce.model_ids.len(),
                                            "PoolModelAvailability processed"
                                        );
                                    }

                                    // Item 8 Phase 1: cross-node prefix-cache announcement.
                                    // Each peer broadcasts the BLAKE3 chained-hash list of
                                    // its locally-cached prompt-prefix blocks for `model_id`.
                                    // We update the local index so Phase 2's KV-fetch path
                                    // can ask "who has this block hash?" and pick a peer.
                                    SwarmMessage::PrefixCacheAnnounce(announce) => {
                                        match &authenticated_sender {
                                            Some(sender) if *sender != announce.node_id => {
                                                tracing::warn!(
                                                    sender = %sender,
                                                    claimed = %announce.node_id,
                                                    "PrefixCacheAnnounce sender mismatch — dropping"
                                                );
                                                continue;
                                            }
                                            None => {
                                                tracing::debug!("Dropping unauthenticated PrefixCacheAnnounce");
                                                continue;
                                            }
                                            Some(_) => {}
                                        }
                                        // Drop self-announces — local cache hits are served
                                        // from the in-process PrefixCache directly.
                                        if announce.node_id == *shared_state.identity.node_id() {
                                            continue;
                                        }
                                        if announce.blocks.len() > MAX_BLOCKS_PER_ANNOUNCE {
                                            tracing::warn!(
                                                node_id = %announce.node_id,
                                                blocks = announce.blocks.len(),
                                                max = MAX_BLOCKS_PER_ANNOUNCE,
                                                "PrefixCacheAnnounce exceeds block limit — dropping"
                                            );
                                            continue;
                                        }
                                        if announce.model_id.0.len() > 256 {
                                            tracing::warn!(
                                                node_id = %announce.node_id,
                                                "PrefixCacheAnnounce oversized model_id — dropping"
                                            );
                                            continue;
                                        }
                                        let block_hashes: Vec<[u8; 32]> = announce
                                            .blocks
                                            .iter()
                                            .map(|b| b.block_hash)
                                            .collect();
                                        let blocks_count = block_hashes.len();
                                        let (added, removed) = shared_state.models.replace_peer_prefix_blocks(
                                            announce.node_id.clone(),
                                            announce.model_id.clone(),
                                            block_hashes,
                                        );
                                        tracing::debug!(
                                            node_id = %announce.node_id,
                                            model = %announce.model_id,
                                            blocks = blocks_count,
                                            added,
                                            removed,
                                            "DIAG: PrefixCacheAnnounce indexed"
                                        );
                                    }

                                    // SEC (R105): explicitly reject HealthPing/Pong with
                                    // missing node_id rather than silently swallowing
                                    // them via the generic catch-all. The Some(node_id)
                                    // arms above carry the authenticated-sender check;
                                    // a peer sending node_id: None bypasses both the
                                    // pong path and any log signal. Surface so it's
                                    // observable.
                                    SwarmMessage::HealthPing { node_id: None, .. }
                                    | SwarmMessage::HealthPong { node_id: None, .. } => {
                                        tracing::debug!(
                                            "Dropping HealthPing/Pong with missing node_id"
                                        );
                                    }
                                    // Cross-node inference cancellation. Originator broadcasts
                                    // this when the local request flips its cancel flag (or the
                                    // SSE client hangs up); receiver aborts the matching
                                    // remote-generate decode so the worker stops streaming
                                    // wasted tokens back.
                                    SwarmMessage::CancelInference(cancel) => {
                                        if authenticated_sender.is_none() {
                                            tracing::debug!("Dropping unauthenticated CancelInference");
                                            continue;
                                        }
                                        let mut aborted_something = false;
                                        if let Some((_, (abort, _))) = shared_state
                                            .inbound_generate_aborts
                                            .remove(&cancel.request_id)
                                        {
                                            tracing::info!(
                                                request_id = %cancel.request_id,
                                                sender = ?authenticated_sender,
                                                "CancelInference: aborting inbound remote-generate"
                                            );
                                            abort.abort();
                                            aborted_something = true;
                                        }
                                        // The segment-forward sibling. Until this existed, a
                                        // coordinator that had given up on a segment told us so and
                                        // we ignored it: the handler found no remote-generate,
                                        // logged "no in-flight decode", and we computed the whole
                                        // abandoned prefill anyway while every other request queued
                                        // behind it.
                                        if let Some((_, (abort, _))) = shared_state
                                            .inbound_forward_aborts
                                            .remove(&cancel.request_id)
                                        {
                                            tracing::info!(
                                                request_id = %cancel.request_id,
                                                sender = ?authenticated_sender,
                                                "CancelInference: abandoning inbound segment forward"
                                            );
                                            abort.abort();
                                            aborted_something = true;
                                        }
                                        if !aborted_something {
                                            tracing::debug!(
                                                request_id = %cancel.request_id,
                                                "CancelInference: nothing in flight for request"
                                            );
                                        }
                                        // Also reach the worker directly. Two
                                        // cases need this beyond the abort
                                        // handle above:
                                        //   - a LayerForward already dispatched
                                        //     to the worker (e.g. we are the
                                        //     losing holder of a hedge race)
                                        //     has no abort handle — the
                                        //     coordinator simply stopped
                                        //     waiting, and without this the
                                        //     worker computes a reply nobody
                                        //     will read;
                                        //   - the abort fires between the
                                        //     worker receiving the request and
                                        //     the daemon-side future being
                                        //     dropped.
                                        // Idempotent on the worker side.
                                        shared_state
                                            .model_process_pool
                                            .cancel_request(cancel.request_id)
                                            .await;
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

pub(crate) mod layer_forward;
mod remote_generate;
mod vision;

#[cfg(test)]
mod contribution_limits_tests {
    use super::{max_concurrent_forwards, max_forwards_per_peer, MAX_CONCURRENT_FORWARDS_MAX};
    use swarmllm_types::ContributionMode;

    /// **A node must not be swamped with other people's work beyond what its
    /// owner agreed to give.** This was a flat 64 concurrent forwards on every
    /// node — and `Minimal` is the DEFAULT — so a stock home machine accepted 64
    /// simultaneous model steps from the swarm.
    #[test]
    fn peer_work_is_bounded_by_the_contribution_level() {
        let min = max_concurrent_forwards(&ContributionMode::Minimal);
        let mod_ = max_concurrent_forwards(&ContributionMode::Moderate);
        let max = max_concurrent_forwards(&ContributionMode::Maximum);

        assert!(
            min < mod_ && mod_ < max,
            "more contribution must mean more accepted work, got {min}/{mod_}/{max}"
        );
        assert_eq!(
            max, MAX_CONCURRENT_FORWARDS_MAX,
            "an explicit offer keeps the old ceiling"
        );
        assert!(
            min <= 8,
            "the DEFAULT must not let a home machine be swamped, got {min}"
        );
    }

    /// The point of the per-peer cap is that ONE peer cannot take everything.
    /// It has to stay strictly below the total at every level, or it stops
    /// being a cap at all.
    #[test]
    fn one_peer_can_never_take_the_whole_budget() {
        for c in [
            ContributionMode::Minimal,
            ContributionMode::Moderate,
            ContributionMode::Maximum,
        ] {
            let total = max_concurrent_forwards(&c);
            let per_peer = max_forwards_per_peer(&c);
            assert!(
                per_peer < total,
                "{c:?}: one peer ({per_peer}) must not be able to take the whole \
                 budget ({total})"
            );
            assert!(
                per_peer >= 4,
                "{c:?}: below 4 a tensor-parallel group cannot progress, got {per_peer}"
            );
        }
    }
}
