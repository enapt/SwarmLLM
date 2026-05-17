//! Inference router: owns the request queue, priority/tier gating,
//! multi-turn KV cache bookkeeping, and batch-vs-single dispatch.
//!
//! The router event loop lives in this file. Per-path execution (local
//! batched, distributed batched, per-request) lives in sibling modules.

mod batch;
mod distributed_exec;
mod local_exec;
mod spot_check;
#[cfg(test)]
mod tests;
mod types;

use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use crate::credit::priority;
use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::kv_cache::KvCacheManager;
use crate::types::{InferenceRequest, NetworkCommand, SwarmMessage};

use batch::execute_batch;
use distributed_exec::{execute_request, finalize_request};
use types::QueuedRequest;

pub use types::{
    InferenceOutput, InferenceResultTx, RouterCommand, StreamingTokenEvent, StreamingTokenTx,
    TokenLogProbEntry,
};

/// Classify a router error as a transient remote failure that warrants a
/// single retry with a fresh pipeline assembly.
///
/// Targets:
/// - "peer never acknowledged" — RR_ACK_TIMEOUT_SECS sweep fired
///   (libp2p rr silent-drop or peer died without TCP RST).
/// - "remote-generate timed out" — first-token wait exceeded.
/// - "OutboundFailure" — explicit libp2p delivery failure.
///
/// On retry the scheduler re-runs and picks a different holder via the
/// `connected_node_ids` filter, so a dead peer is not selected again.
fn is_transient_remote_failure(err: &SwarmError) -> bool {
    let msg = err.to_string();
    msg.contains("never acknowledged")
        || msg.contains("silent drop")
        || msg.contains("remote-generate timed out")
        || msg.contains("OutboundFailure")
}

const KV_CACHE_CLEANUP_INTERVAL_SECS: u64 = 30;
/// Maximum depth of the inference request queue. Requests are rejected with 503 when full.
const MAX_QUEUE_DEPTH: usize = 512;
/// TTL for the cached network credit percentile used in priority tiering.
/// The raw scan is O(n) over peer_credit_balances; at high request rates on
/// a swarm with thousands of peers this becomes an expensive per-submit cost
/// on the single-threaded router task. Priority tier is quantized
/// (Bronze/Silver/Gold/Platinum) so sub-second staleness has no observable
/// effect on scheduling.
const PERCENTILE_CACHE_TTL_MS: u128 = 500;

/// The InferenceRouter is the brain of distributed inference.
///
/// It receives inference requests, places them in a priority queue,
/// assembles pipelines using the scheduler, and kicks off execution.
///
/// When `max_batch_size > 1`, compatible requests (same model) are
/// grouped into batches and executed together — sharing the model lock
/// for local inference to reduce contention.
pub struct InferenceRouter {
    shared_state: Arc<SharedState>,
    command_rx: mpsc::Receiver<RouterCommand>,
    network_tx: mpsc::Sender<NetworkCommand>,
    shutdown_rx: watch::Receiver<bool>,
    queue: BinaryHeap<QueuedRequest>,
    scheduler: crate::inference::scheduler::PipelineScheduler,
    kv_cache: KvCacheManager,
    max_concurrent: usize,
    active_count: Arc<AtomicUsize>,
    /// Notify used to wake the drain loop when a new request is queued,
    /// replacing the fixed 50ms polling interval.
    queue_notify: Arc<tokio::sync::Notify>,
    max_batch_size: usize,
    batch_timeout: std::time::Duration,
    /// Sender for spawned tasks to send commands back to the router (e.g., KV-cache updates).
    self_tx: mpsc::Sender<RouterCommand>,
    config_watch_rx: watch::Receiver<crate::config::OperationalParams>,
}

impl InferenceRouter {
    pub fn new(
        shared_state: Arc<SharedState>,
        command_rx: mpsc::Receiver<RouterCommand>,
        command_tx: mpsc::Sender<RouterCommand>,
        network_tx: mpsc::Sender<NetworkCommand>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Self {
        let kv_cache_ttl = std::time::Duration::from_secs(
            shared_state
                .config
                .inference
                .kv_cache_ttl_secs
                .unwrap_or(crate::inference::process_pool::DEFAULT_KV_CACHE_TTL_SECS),
        );
        let max_concurrent = shared_state.config.inference.max_concurrent_requests as usize;
        let max_batch_size = (shared_state.config.inference.max_batch_size as usize).max(1);
        let batch_timeout =
            std::time::Duration::from_millis(shared_state.config.inference.batch_timeout_ms);
        let mut kv_cache = KvCacheManager::new(kv_cache_ttl);

        // Restore persisted multi-turn sessions from previous run
        match kv_cache.restore_from_db(&shared_state.db) {
            Ok(count) if count > 0 => {
                tracing::info!(count, "Restored KV-cache sessions from previous run");
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to restore KV-cache sessions");
            }
            _ => {}
        }

        let config_watch_rx = shared_state.config_watch_rx();
        Self {
            shared_state: shared_state.clone(),
            command_rx,
            network_tx,
            shutdown_rx,
            queue: BinaryHeap::new(),
            scheduler: crate::inference::scheduler::PipelineScheduler::new(shared_state),
            kv_cache,
            max_concurrent,
            active_count: Arc::new(AtomicUsize::new(0)),
            queue_notify: Arc::new(tokio::sync::Notify::new()),
            max_batch_size,
            batch_timeout,
            self_tx: command_tx,
            config_watch_rx,
        }
    }

    /// Run the router event loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        tracing::info!(
            max_batch_size = self.max_batch_size,
            batch_timeout_ms = self.batch_timeout.as_millis() as u64,
            "InferenceRouter running"
        );

        // KV-cache cleanup interval
        let mut cache_cleanup = tokio::time::interval(std::time::Duration::from_secs(
            KV_CACHE_CLEANUP_INTERVAL_SECS,
        ));
        cache_cleanup.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("InferenceRouter shutting down");
                        // Persist multi-turn sessions for next startup
                        let privacy_mode = self.shared_state.config.inference.privacy_mode;
                        match self.kv_cache.save_to_db(&self.shared_state.db, privacy_mode) {
                            Ok(count) => {
                                tracing::info!(count, "Saved KV-cache sessions for next startup");
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Failed to persist KV-cache sessions");
                            }
                        }
                        break;
                    }
                }
                cmd = self.command_rx.recv() => {
                    match cmd {
                        Some(RouterCommand::Submit { request, result_tx }) => {
                            self.handle_submit(request, result_tx, None);
                            self.drain_queue().await;
                        }
                        Some(RouterCommand::StreamSubmit { request, result_tx, token_tx }) => {
                            self.handle_submit(request, result_tx, Some(token_tx));
                            self.drain_queue().await;
                        }
                        Some(RouterCommand::NetworkMessage(msg)) => {
                            self.handle_network_message(msg).await;
                        }
                        Some(RouterCommand::UpdateCacheTokens { session_id, total_tokens, prompt }) => {
                            if let Some(internal_id) = self.kv_cache.get_internal_id(&session_id) {
                                self.kv_cache.update_cached_tokens(&internal_id, total_tokens);
                                self.kv_cache.update_cached_prompt(&internal_id, prompt);
                                tracing::debug!(
                                    session_id,
                                    total_tokens,
                                    "Updated multi-turn KV-cache token count"
                                );
                            }
                        }
                        None => {
                            tracing::info!("Command channel closed, shutting down");
                            break;
                        }
                    }
                }
                _ = self.queue_notify.notified() => {
                    self.drain_queue().await;
                }
                _ = self.config_watch_rx.changed() => {
                    let params = self.config_watch_rx.borrow().clone();
                    let new_max = params.max_concurrent_requests as usize;
                    let new_batch = (params.max_batch_size as usize).max(1);
                    if new_max != self.max_concurrent || new_batch != self.max_batch_size {
                        tracing::info!(
                            old_max_concurrent = self.max_concurrent,
                            new_max_concurrent = new_max,
                            old_max_batch = self.max_batch_size,
                            new_max_batch = new_batch,
                            "Hot-reloaded inference router config"
                        );
                        self.max_concurrent = new_max;
                        self.max_batch_size = new_batch;
                    }
                }
                _ = cache_cleanup.tick() => {
                    let expired = self.kv_cache.cleanup_expired();
                    if expired > 0 {
                        tracing::debug!(expired, "Cleaned up expired KV-cache sessions");
                    }
                    // Also clean up per-request tensor KV-caches
                    let tensor_expired = self.shared_state.kv_cache_store.cleanup_expired();
                    if tensor_expired > 0 {
                        tracing::debug!(tensor_expired, "Cleaned up expired per-request KV-caches");
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a new inference submission.
    ///
    /// Checks credit balance / priority tier before queueing.
    fn handle_submit(
        &mut self,
        request: InferenceRequest,
        result_tx: InferenceResultTx,
        token_tx: Option<StreamingTokenTx>,
    ) {
        // Calculate priority tier from credit balance and network percentile.
        // Per spec: "Credit errors: degrade priority tier, never block"
        let balance = {
            if let Ok(bal) = self.shared_state.credits.credit_balance.try_read() {
                bal.balance
            } else {
                0
            }
        };

        // Compute network percentile from peer credit balances, cached at the
        // module-level PERCENTILE_CACHE_TTL_MS interval.
        let network_percentile = {
            let now = std::time::Instant::now();
            let mut cache = self.shared_state.credits.credit_percentile_cache.lock();
            if now.duration_since(cache.0).as_millis() < PERCENTILE_CACHE_TTL_MS {
                cache.1
            } else {
                let mut count = 0u32;
                let mut below = 0u32;
                for entry in self.shared_state.credits.peer_credit_balances.iter() {
                    count += 1;
                    if *entry.value() < balance {
                        below += 1;
                    }
                }
                let pct = if count == 0 {
                    0.5
                } else {
                    below as f32 / count as f32
                };
                *cache = (now, pct);
                pct
            }
        };

        let priority = priority::calculate_tier(balance, network_percentile);

        // Enforce minimum balance for inference requests.
        // Nodes below the floor must contribute (host shards, serve inference) before consuming.
        // Local API requests use NodeId([0;32]) as sentinel — always allow those from localhost.
        let is_local = request.requester == crate::types::NodeId([0u8; 32]);
        if !is_local
            && crate::credit::ledger::MIN_BALANCE_FOR_INFERENCE != 0
            && balance < crate::credit::ledger::MIN_BALANCE_FOR_INFERENCE
        {
            tracing::warn!(
                balance,
                min = crate::credit::ledger::MIN_BALANCE_FOR_INFERENCE,
                requester = %request.requester,
                "Inference request rejected — balance below minimum"
            );
            let send_res = result_tx.send(Err(SwarmError::InsufficientCredits {
                balance,
                required: crate::credit::ledger::MIN_BALANCE_FOR_INFERENCE,
            }));
            if send_res.is_err() {
                tracing::debug!(
                    requester = %request.requester,
                    "Credit-rejection error not delivered (oneshot receiver dropped)"
                );
            }
            return;
        }

        let mut adjusted_request = request;
        adjusted_request.priority = priority;

        // Track per-model request count for popularity-based prune scoring.
        // Only track models in the registry to prevent unbounded map growth from
        // cloud-proxied model names, typos, or other arbitrary strings.
        if self
            .shared_state
            .model_registry
            .get_manifest(&adjusted_request.model_id)
            .is_some()
        {
            self.shared_state
                .models
                .model_request_counts
                .entry(adjusted_request.model_id.clone())
                .or_insert_with(|| std::sync::atomic::AtomicU64::new(0))
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // Validate model exists (check registry, split_models, and loaded model)
        {
            let has_manifest = self
                .shared_state
                .model_registry
                .get_manifest(&adjusted_request.model_id)
                .is_some();
            let has_split = self
                .shared_state
                .has_split_model(&adjusted_request.model_id);
            let has_loaded = {
                let model_loaded = self
                    .shared_state
                    .model_loaded
                    .load(std::sync::atomic::Ordering::Relaxed);
                if model_loaded {
                    // Check if the loaded model matches the requested model
                    if let Ok(info) = self.shared_state.loaded_model_info.try_read() {
                        info.as_ref()
                            .map(|i| i.name == adjusted_request.model_id.0)
                            .unwrap_or(false)
                    } else {
                        true // Can't read, assume ok
                    }
                } else {
                    false
                }
            };
            if !has_manifest && !has_split && !has_loaded {
                let _ = result_tx.send(Err(self
                    .shared_state
                    .model_registry
                    .model_not_found_error(&adjusted_request.model_id)));
                return;
            }
        }

        // Reject when queue is full to prevent memory exhaustion from request flooding.
        // max_concurrent gates execution slots; this caps the waiting queue depth.
        if self.queue.len() >= MAX_QUEUE_DEPTH {
            tracing::warn!(
                queue_len = self.queue.len(),
                "Inference queue full — rejecting request"
            );
            let send_res = result_tx.send(Err(crate::error::SwarmError::ServiceUnavailable(
                "Inference queue is full. Please try again later.".to_string(),
            )));
            if send_res.is_err() {
                tracing::debug!(
                    request_id = %adjusted_request.id,
                    "Queue-full error not delivered (oneshot receiver dropped)"
                );
            }
            return;
        }

        tracing::info!(
            request_id = %adjusted_request.id,
            model = %adjusted_request.model_id,
            priority = ?adjusted_request.priority,
            "Queued inference request"
        );

        self.queue.push(QueuedRequest {
            request: adjusted_request,
            result_tx,
            token_tx,
        });

        // Wake the drain loop immediately instead of waiting for 50ms poll.
        self.queue_notify.notify_one();
    }

    /// Log network messages routed to the inference router.
    /// Actual dispatch (LayerForward, pipeline execution) is handled in `dispatch.rs`.
    async fn handle_network_message(&mut self, msg: SwarmMessage) {
        match msg {
            SwarmMessage::LayerResult(result) => {
                tracing::debug!(
                    request_id = %result.request_id,
                    tokens = result.token_ids.len(),
                    "Received layer result from network"
                );
            }
            SwarmMessage::InferenceError(err) => {
                tracing::warn!(
                    request_id = %err.request_id,
                    error = %err.error,
                    recoverable = err.recoverable,
                    "Received inference error from network"
                );
            }
            SwarmMessage::InferenceRequest(req) => {
                tracing::debug!(
                    request_id = %req.id,
                    requester = %req.requester,
                    model = %req.model_id,
                    "Received remote inference request via router"
                );
            }
            other => {
                tracing::trace!(
                    kind = ?std::mem::discriminant(&other),
                    "Router ignoring unhandled network message"
                );
            }
        }
    }

    /// Collect a batch of compatible requests (same model + same priority tier)
    /// from the priority queue.
    ///
    /// Returns up to `max_batch_size` requests that all target the same
    /// `model_id` AND share the same `PriorityTier` as the first request.
    /// Incompatible requests (different model or lower tier) are pushed back
    /// into the queue.
    ///
    /// CORRECTNESS: the per-tier concurrency cap in `drain_queue` is checked
    /// against the head-of-queue's tier (via `peek()`). If we co-batched
    /// lower-tier requests with a higher-tier head, those lower-tier
    /// requests would bypass their stricter cap (e.g. a Platinum head would
    /// pull Bronze followers through under the 2× Platinum cap, defeating
    /// Bronze's 1/4× isolation). Same-tier batching keeps the cap invariant
    /// regardless of which request in the batch is examined.
    fn collect_batch(&mut self, max_size: usize) -> Vec<QueuedRequest> {
        let first = match self.queue.pop() {
            Some(q) => q,
            None => return vec![],
        };

        if max_size <= 1 {
            return vec![first];
        }

        let target_model = first.request.model_id.clone();
        let target_priority = first.request.priority;
        let mut batch = vec![first];
        let mut deferred = Vec::new();

        while batch.len() < max_size {
            match self.queue.pop() {
                Some(q) => {
                    if q.request.model_id == target_model && q.request.priority == target_priority {
                        batch.push(q);
                    } else {
                        deferred.push(q);
                    }
                }
                None => break,
            }
        }

        // Push back incompatible requests
        for q in deferred {
            self.queue.push(q);
        }

        batch
    }

    /// Drain the priority queue and execute requests, batching when possible.
    ///
    /// When `max_batch_size > 1`, multiple compatible requests (same model) are
    /// collected and dispatched together. For local inference, the batch shares
    /// a single model lock acquisition — requests are processed sequentially
    /// within the batch but without re-acquiring the lock between them.
    async fn drain_queue(&mut self) {
        while !self.queue.is_empty() {
            // Enforce per-tier concurrent limits: the next request's tier
            // determines how many slots it can use (Bronze=1/4, Silver=1/2,
            // Gold=base, Platinum=2x).
            let next_tier = self
                .queue
                .peek()
                .map(|q| q.request.priority)
                .unwrap_or(crate::types::PriorityTier::Bronze);
            let tier_max = priority::max_concurrent_for_tier(next_tier, self.max_concurrent);
            if self.active_count.load(Ordering::Relaxed) >= tier_max {
                break;
            }

            // If batching is enabled and we have a partial batch (>1 request),
            // wait briefly for more requests to arrive before dispatching.
            // Skip the wait for single requests to avoid unnecessary latency.
            if self.max_batch_size > 1
                && self.queue.len() > 1
                && self.queue.len() < self.max_batch_size
                && !self.batch_timeout.is_zero()
            {
                let notify = self.queue_notify.clone();
                let timeout = self.batch_timeout;
                // Wait for either more requests or timeout
                let _ = tokio::time::timeout(timeout, notify.notified()).await;
                // After waiting, drain any new commands that arrived
                self.drain_pending_commands().await;
            }

            // Re-read active_count after the await so a task that completed
            // during the batch_timeout window contributes its freed slot.
            // Cap the collection size against remaining tier headroom — the
            // multi-batch dispatch below does `fetch_add(batch_size)` in one
            // shot, so collecting more than `tier_max - active` items would
            // bypass the per-tier concurrency cap (Bronze=¼, Silver=½),
            // letting low-credit users co-batch up to `max_batch_size` and
            // erase the credit-tier isolation guarantee. R107 fix.
            let active = self.active_count.load(Ordering::Relaxed);
            if active >= tier_max {
                break;
            }
            let headroom = tier_max - active;
            let collect_cap = self.max_batch_size.min(headroom);
            let batch = self.collect_batch(collect_cap);
            if batch.is_empty() {
                break;
            }

            let batch_size = batch.len();

            // For single-request batches, use the original unbatched path
            if batch_size == 1 {
                let mut batch = batch;
                let queued = batch.pop().expect("batch_size==1 guarantees non-empty");
                self.dispatch_single(queued);
                continue;
            }

            // Multi-request batch: dispatch as a group
            tracing::info!(batch_size, "Dispatching inference batch");

            // Each request in the batch counts toward active_count
            self.active_count.fetch_add(batch_size, Ordering::Relaxed);

            let active_count = self.active_count.clone();
            let queue_notify = self.queue_notify.clone();
            let shared_state = self.shared_state.clone();
            let network_tx = self.network_tx.clone();
            let scheduler = self.scheduler.clone();

            tokio::spawn(async move {
                execute_batch(
                    shared_state,
                    network_tx,
                    scheduler,
                    batch,
                    active_count,
                    queue_notify,
                )
                .await;
            });
        }
    }

    /// Drain any pending commands from the command channel without blocking.
    /// Used during batch assembly to pick up requests that arrived while waiting.
    async fn drain_pending_commands(&mut self) {
        loop {
            match self.command_rx.try_recv() {
                Ok(RouterCommand::Submit { request, result_tx }) => {
                    self.handle_submit(request, result_tx, None);
                }
                Ok(RouterCommand::StreamSubmit {
                    request,
                    result_tx,
                    token_tx,
                }) => {
                    self.handle_submit(request, result_tx, Some(token_tx));
                }
                Ok(RouterCommand::NetworkMessage(msg)) => {
                    self.handle_network_message(msg).await;
                }
                Ok(RouterCommand::UpdateCacheTokens {
                    session_id,
                    total_tokens,
                    prompt,
                }) => {
                    if let Some(internal_id) = self.kv_cache.get_internal_id(&session_id) {
                        self.kv_cache
                            .update_cached_tokens(&internal_id, total_tokens);
                        self.kv_cache.update_cached_prompt(&internal_id, prompt);
                    }
                }
                Err(_) => break,
            }
        }
    }

    /// Dispatch a single request (non-batched path).
    ///
    /// If the request has a `session_id`, checks whether the KV-cache from
    /// a previous turn can be reused (multi-turn prefix matching). On a cache
    /// hit, the tensor-level KV-cache is preserved and `start_pos` is set
    /// to skip redundant prefill.
    fn dispatch_single(&mut self, queued: QueuedRequest) {
        // Pipeline affinity: get previous pipeline assignment for KV cache locality
        let preferred_pipeline = if let Some(ref session_id) = queued.request.session_id {
            if let Some(internal_id) = self.kv_cache.get_internal_id(session_id) {
                self.kv_cache.get_previous_pipeline(&internal_id)
            } else {
                None
            }
        } else {
            None
        };

        // Check for multi-turn KV-cache reuse. Hoist the chatml_fallback
        // allocation outside the if-let so the same prompt String is reused
        // by both check_multi_turn_reuse (above) and register_multi_turn
        // (below) on the cache-miss branch — was being allocated twice on
        // every cold-start session-keyed request.
        let session_prompt =
            queued.request.session_id.as_ref().map(|_| {
                crate::inference::chat_template::chatml_fallback(&queued.request.messages)
            });

        let cache_start_pos = if let (Some(session_id), Some(prompt)) =
            (queued.request.session_id.as_ref(), session_prompt.as_ref())
        {
            // Collect active peer IDs into a HashSet for O(1) holder lookup
            // inside check_multi_turn_reuse. Use connected_node_ids (the
            // connectivity oracle, gotcha #86) — peer_registry is preserved
            // across mid-pipeline disconnects, so a peer that hung up is
            // still in peer_registry; using it here would let a stale
            // session reuse KV against a node that no longer holds it.
            let active_peers: std::collections::HashSet<crate::types::NodeId> = self
                .shared_state
                .connected_node_ids
                .iter()
                .map(|e| e.key().clone())
                .collect();

            match self
                .kv_cache
                .check_multi_turn_reuse(session_id, prompt, &active_peers)
            {
                crate::inference::kv_cache::CacheReuse::Hit { start_pos } => {
                    tracing::info!(
                        session_id,
                        start_pos,
                        request_id = %queued.request.id,
                        "Multi-turn KV-cache hit"
                    );
                    Some(start_pos)
                }
                crate::inference::kv_cache::CacheReuse::Miss => {
                    tracing::debug!(
                        session_id,
                        request_id = %queued.request.id,
                        "Multi-turn KV-cache miss"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Register multi-turn KV-cache session so subsequent turns can
        // find this session via check_multi_turn_reuse. Use chatml_fallback
        // consistently (same template used for the prefix check above).
        // Skip if session already exists (cache_start_pos.is_some()) — don't overwrite
        // existing session's pipeline/cache_holders with empty data.
        if cache_start_pos.is_none() {
            if let (Some(session_id), Some(prompt)) =
                (queued.request.session_id.as_ref(), session_prompt)
            {
                self.kv_cache.register_multi_turn(
                    session_id,
                    queued.request.id,
                    crate::types::PipelineAssignment {
                        request_id: queued.request.id,
                        segments: vec![],
                        standbys: vec![],
                        tp_groups: vec![],
                        supports_speculative: false,
                    },
                    0,
                    prompt,
                );
            }
        }

        self.active_count.fetch_add(1, Ordering::Relaxed);
        let active_count = self.active_count.clone();
        let queue_notify = self.queue_notify.clone();
        let shared_state = self.shared_state.clone();
        let network_tx = self.network_tx.clone();
        let scheduler = self.scheduler.clone();
        let self_tx = self.self_tx.clone();
        let request = queued.request;
        let result_tx = queued.result_tx;
        let token_tx = queued.token_tx;

        tokio::spawn(async move {
            // RAII guard so a panic anywhere inside the spawned closure
            // doesn't leak the active_pipelines entry or the active_count
            // tier-cap counter — those would otherwise stay until process
            // exit and silently throttle the daemon to ServiceUnavailable
            // after enough panicking requests.
            struct ActivePipelineGuard {
                shared_state: Arc<crate::daemon::SharedState>,
                count: Arc<std::sync::atomic::AtomicUsize>,
                queue_notify: Arc<tokio::sync::Notify>,
                request_id: uuid::Uuid,
                armed: bool,
            }
            impl ActivePipelineGuard {
                fn disarm(&mut self) {
                    self.armed = false;
                }
            }
            impl Drop for ActivePipelineGuard {
                fn drop(&mut self) {
                    if self.armed {
                        self.shared_state.active_pipelines.remove(&self.request_id);
                        self.count
                            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        // Wake drain_queue so the next queued request can dispatch.
                        self.queue_notify.notify_one();
                    }
                }
            }
            let mut active_guard = ActivePipelineGuard {
                shared_state: shared_state.clone(),
                count: active_count.clone(),
                queue_notify: queue_notify.clone(),
                request_id: request.id,
                armed: true,
            };

            let request_start = std::time::Instant::now();
            tracing::info!(
                request_id = %request.id,
                model = %request.model_id,
                priority = ?request.priority,
                "DIAG: dispatch_single starting inference"
            );

            // Create escrow for large requests (estimated cost > threshold)
            let estimated_cost = crate::credit::ledger::RATE_INFERENCE_CONSUME
                * request.sampling_params.max_tokens as i64;
            let escrow_id = if shared_state
                .credits
                .escrow_manager
                .needs_escrow(estimated_cost)
            {
                match shared_state
                    .credits
                    .escrow_manager
                    .create_escrow(
                        request.id,
                        estimated_cost,
                        &request.requester,
                        &shared_state.credits.credit_balance,
                    )
                    .await
                {
                    Ok(id) => {
                        tracing::debug!(
                            request_id = %request.id,
                            escrow_id = %id,
                            amount = estimated_cost,
                            "Credit escrow created"
                        );
                        Some(id)
                    }
                    Err(e) => {
                        tracing::warn!(
                            request_id = %request.id,
                            error = %e,
                            "Failed to create escrow — proceeding without"
                        );
                        None
                    }
                }
            } else {
                None
            };

            // Retry once on transient remote failures (silent rr drops or
            // mid-flight peer disconnects). We reset `preferred_pipeline`
            // to None so the second attempt re-runs the scheduler — which
            // filters out the dead/dropped peer via `connected_node_ids`
            // and picks a different holder.
            let mut output = execute_request(
                shared_state.clone(),
                network_tx.clone(),
                scheduler.clone(),
                request.clone(),
                token_tx.clone(),
                preferred_pipeline,
            )
            .await;
            if matches!(&output, Err(e) if is_transient_remote_failure(e)) {
                tracing::warn!(
                    request_id = %request.id,
                    error = %output.as_ref().err().unwrap(),
                    "DIAG: inference transient failure — retrying with fresh pipeline"
                );
                output = execute_request(
                    shared_state.clone(),
                    network_tx,
                    scheduler,
                    request.clone(),
                    token_tx,
                    None,
                )
                .await;
            }

            let elapsed = request_start.elapsed();
            // Record latency for Prometheus histogram
            match &output {
                Ok(ref result) => {
                    let latency_secs = elapsed.as_secs_f64();
                    tracing::info!(
                        request_id = %request.id,
                        model = %request.model_id,
                        elapsed_ms = elapsed.as_millis() as u64,
                        prompt_tokens = result.prompt_tokens,
                        completion_tokens = result.completion_tokens,
                        "DIAG: inference completed"
                    );
                    if let Ok(mut samples) = shared_state.metrics.inference_latency_samples.write()
                    {
                        if samples.len() >= 1000 {
                            samples.pop_front();
                        }
                        samples.push_back(latency_secs);
                    }
                    // CORRECTNESS (R105): keep a monotonic total alongside
                    // the bounded ring. Prometheus histogram `_count` /
                    // `_sum` MUST be monotonically non-decreasing for
                    // `rate()` / `increase()` to work; the ring's count
                    // capped at 1000 and could fall when it wrapped.
                    shared_state
                        .metrics
                        .inference_latency_total_count
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let micros = (latency_secs * 1_000_000.0).round() as u64;
                    shared_state
                        .metrics
                        .inference_latency_total_micros
                        .fetch_add(micros, std::sync::atomic::Ordering::Relaxed);
                }
                Err(ref e) => {
                    tracing::error!(
                        request_id = %request.id,
                        model = %request.model_id,
                        elapsed_ms = elapsed.as_millis() as u64,
                        error = %e,
                        "DIAG: inference FAILED"
                    );
                }
            }

            finalize_request(&shared_state, &request, &output, escrow_id).await;

            // Release escrow on success. Refund on failure is handled by
            // finalize_request — calling refund_escrow again here races and
            // logs a spurious "Escrow not found" warning.
            if let (Some(eid), Ok(_)) = (escrow_id, &output) {
                if let Err(e) = shared_state
                    .credits
                    .escrow_manager
                    .release_escrow(eid, shared_state.identity.node_id())
                    .await
                {
                    tracing::warn!(escrow_id = %eid, error = %e, "Failed to release escrow");
                }
            }

            // Update multi-turn KV-cache with actual token count so subsequent
            // turns can skip prefill via start_pos
            if let (Some(ref session_id), Ok(ref result)) = (&request.session_id, &output) {
                let total_tokens = result.prompt_tokens + result.completion_tokens;
                let prompt = crate::inference::chat_template::chatml_fallback(&request.messages);
                let _ = self_tx
                    .send(RouterCommand::UpdateCacheTokens {
                        session_id: session_id.clone(),
                        total_tokens,
                        prompt,
                    })
                    .await;
                // SWARM-SPEC Layer 3: feed the conversation history
                // learner. record_response_completion stamps idle-time
                // anchor; observe_user_turn captures the first token
                // of the user's latest message so the histogram of
                // next-turn first-tokens converges over the session.
                // Uses the standalone tokenizer (R136 follow-on)
                // loaded from gguf_header.bin — same cache the
                // n-gram-only spec path uses.
                let now_ms = crate::types::unix_now_ms();
                shared_state
                    .metrics
                    .prefetch_orchestrator
                    .record_response_completion(session_id, now_ms);
                // Find the most-recent user-role message
                let latest_user = request
                    .messages
                    .iter()
                    .rev()
                    .find(|m| matches!(m.role, crate::types::Role::User))
                    .map(|m| m.content.clone())
                    .unwrap_or_default();
                if !latest_user.is_empty() {
                    if let Some(tok) = shared_state.standalone_tokenizer(&request.model_id) {
                        let ids = tok.encode(&latest_user);
                        if let Some(first) = ids.first() {
                            let first_u32 = *first as u32;
                            shared_state
                                .metrics
                                .prefetch_orchestrator
                                .observe_user_turn(session_id, first_u32);
                        }
                    }
                }
                // SWARM-SPEC Layer 3 dispatch: query the orchestrator
                // for top-K candidates. When prefetch_enabled is on AND
                // history is sufficient AND idle window has elapsed,
                // returns Vec<u32> of predicted first-tokens. We
                // record_dispatch immediately (counter for throttling)
                // and emit an ActivityEvent so the dashboard surfaces
                // the decision. Actual K-layer activation prefetch is
                // workload-dependent and deferred to a model-size /
                // hardware specific follow-up — for small models on
                // fast hardware the prefill saving is in the noise.
                let prefetch_cfg = crate::inference::prefetch::PrefetchConfig {
                    enabled: shared_state.config.inference.prefetch_enabled,
                    min_idle_ms: shared_state.config.inference.prefetch_min_idle_ms,
                    min_turns_for_prediction: shared_state
                        .config
                        .inference
                        .prefetch_min_turns_for_prediction,
                    max_candidates: shared_state.config.inference.prefetch_max_candidates as usize,
                    ..Default::default()
                };
                let candidates = shared_state.metrics.prefetch_orchestrator.should_prefetch(
                    session_id,
                    now_ms,
                    prefetch_cfg,
                );
                if !candidates.is_empty() {
                    shared_state.metrics.prefetch_orchestrator.record_dispatch();
                    tracing::info!(
                        session_id = %session_id,
                        model_id = %request.model_id,
                        candidate_count = candidates.len(),
                        first_candidates = ?&candidates[..candidates.len().min(3)],
                        "SWARM-SPEC L3: prefetch would fire — observability-only (K-layer compute deferred)"
                    );
                    shared_state.emit_activity(
                        crate::daemon::state::ActivityEvent::new(
                            "system",
                            "prefetch_decision",
                            format!(
                                "Predictive prefetch: {} candidate(s) for session {}",
                                candidates.len(),
                                &session_id[..session_id.len().min(8)]
                            ),
                        )
                        .with_model(request.model_id.0.clone()),
                    );
                }
            }

            // Disarm the guard — normal completion does the same work below.
            active_guard.disarm();
            // Remove from active pipelines
            shared_state.active_pipelines.remove(&request.id);

            // Decrement active count so new requests can be dispatched, then
            // wake drain_queue so the next queued request actually starts.
            // Without the notify, queued requests sat indefinitely until the
            // next Submit arrived (the only other drain trigger).
            active_count.fetch_sub(1, Ordering::Relaxed);
            queue_notify.notify_one();

            if result_tx.send(output).is_err() {
                tracing::warn!(
                    request_id = %request.id,
                    "DIAG: result_tx receiver dropped — client disconnected before result"
                );
            }
        });
    }
}
