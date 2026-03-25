use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, watch};

use crate::credit::priority;
use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::chat_template;
use crate::inference::kv_cache::KvCacheManager;
use crate::inference::pipeline::PipelineExecutor;
use crate::inference::scheduler::PipelineScheduler;
use crate::types::{InferenceRequest, NetworkCommand, NodeId, PipelineAssignment, SwarmMessage};

/// Result channel for returning inference output to API callers.
pub type InferenceResultTx = oneshot::Sender<Result<InferenceOutput, SwarmError>>;

/// A queued inference request with its result channel and priority ordering.
struct QueuedRequest {
    request: InferenceRequest,
    result_tx: InferenceResultTx,
    /// If set, tokens are sent incrementally for SSE streaming.
    token_tx: Option<StreamingTokenTx>,
}

impl Eq for QueuedRequest {}
impl PartialEq for QueuedRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request.priority == other.request.priority
            && self.request.created_at == other.request.created_at
    }
}

impl PartialOrd for QueuedRequest {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueuedRequest {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Higher priority first, then earlier created_at (FIFO within same tier)
        self.request
            .priority
            .cmp(&other.request.priority)
            .then_with(|| other.request.created_at.cmp(&self.request.created_at))
    }
}

/// Output from a completed inference request.
#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub request_id: uuid::Uuid,
    pub content: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: String,
    /// The session ID for multi-turn KV-cache reuse. Echoed back from the
    /// request or auto-generated if the router created one.
    pub session_id: Option<String>,
    /// Per-token log probabilities (populated when logprobs=true in request).
    pub token_logprobs: Vec<TokenLogProbEntry>,
}

/// A single token's log probability info for the logprobs response field.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenLogProbEntry {
    /// The token text.
    pub token: String,
    /// Log probability of this token.
    pub logprob: f32,
    /// Top-N alternative tokens with their logprobs.
    pub top_logprobs: Vec<(String, f32)>,
}

/// Sender for incremental streaming tokens from distributed inference.
pub type StreamingTokenTx = mpsc::Sender<StreamingTokenEvent>;

/// A single token event sent during streaming distributed inference.
#[derive(Debug, Clone)]
pub struct StreamingTokenEvent {
    pub text: String,
    pub finish_reason: Option<String>,
}

/// Command sent to the InferenceRouter from the API layer or network.
pub enum RouterCommand {
    /// Submit a new inference request with a channel for the result.
    Submit {
        request: InferenceRequest,
        result_tx: InferenceResultTx,
    },
    /// Submit a streaming inference request. Tokens are sent incrementally
    /// on `token_tx`. The final `InferenceOutput` is still sent on `result_tx`
    /// for stats/credit accounting.
    StreamSubmit {
        request: InferenceRequest,
        result_tx: InferenceResultTx,
        token_tx: StreamingTokenTx,
    },
    /// A network message relevant to inference (LayerForward, LayerResult, etc.)
    NetworkMessage(SwarmMessage),
    /// Update multi-turn KV-cache token count after inference completes.
    UpdateCacheTokens {
        session_id: String,
        total_tokens: u32,
        prompt: String,
    },
}

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
    scheduler: PipelineScheduler,
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
                .unwrap_or(600),
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

        Self {
            shared_state: shared_state.clone(),
            command_rx,
            network_tx,
            shutdown_rx,
            queue: BinaryHeap::new(),
            scheduler: PipelineScheduler::new(shared_state),
            kv_cache,
            max_concurrent,
            active_count: Arc::new(AtomicUsize::new(0)),
            queue_notify: Arc::new(tokio::sync::Notify::new()),
            max_batch_size,
            batch_timeout,
            self_tx: command_tx,
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
        let mut cache_cleanup = tokio::time::interval(std::time::Duration::from_secs(30));
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
            if let Ok(bal) = self.shared_state.credit_balance.try_read() {
                bal.balance
            } else {
                0
            }
        };

        // Compute network percentile from peer credit balances.
        // O(n) scan instead of O(n log n) sort — we only need the rank.
        let network_percentile = {
            let mut count = 0u32;
            let mut below = 0u32;
            for entry in self.shared_state.peer_credit_balances.iter() {
                count += 1;
                if *entry.value() < balance {
                    below += 1;
                }
            }
            if count == 0 {
                0.5 // No peers known, default to median
            } else {
                below as f32 / count as f32
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
            let _ = result_tx.send(Err(SwarmError::CreditError(format!(
                "Insufficient credits: balance {} is below minimum {} required for inference. \
                 Contribute by hosting model shards or serving inference to earn credits.",
                balance,
                crate::credit::ledger::MIN_BALANCE_FOR_INFERENCE
            ))));
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
                .split_models
                .iter()
                .any(|e| e.key().0 == adjusted_request.model_id);
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
                let available: Vec<String> = self
                    .shared_state
                    .model_registry
                    .models()
                    .iter()
                    .map(|m| m.id.0.clone())
                    .collect();
                let msg = if available.is_empty() {
                    format!(
                        "Model '{}' not found. No models are available — download shards first.",
                        adjusted_request.model_id.0
                    )
                } else {
                    format!(
                        "Model '{}' not found. Available models: {}",
                        adjusted_request.model_id.0,
                        available.join(", ")
                    )
                };
                let _ = result_tx.send(Err(crate::error::SwarmError::ModelNotAvailable(
                    crate::types::ModelId(msg),
                )));
                return;
            }
        }

        // Reject when queue is full to prevent memory exhaustion from request flooding.
        // max_concurrent gates execution slots; this caps the waiting queue depth.
        const MAX_QUEUE_DEPTH: usize = 512;
        if self.queue.len() >= MAX_QUEUE_DEPTH {
            tracing::warn!(
                queue_len = self.queue.len(),
                "Inference queue full — rejecting request"
            );
            let _ = result_tx.send(Err(crate::error::SwarmError::ServiceUnavailable(
                "Inference queue is full. Please try again later.".to_string(),
            )));
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
                    "Router ignoring unhandled network message: {:?}",
                    std::mem::discriminant(&other)
                );
            }
        }
    }

    /// Collect a batch of compatible requests (same model) from the priority queue.
    ///
    /// Returns up to `max_batch_size` requests that all target the same model_id.
    /// Incompatible requests (different model) are pushed back into the queue.
    fn collect_batch(&mut self, max_size: usize) -> Vec<QueuedRequest> {
        let first = match self.queue.pop() {
            Some(q) => q,
            None => return vec![],
        };

        if max_size <= 1 {
            return vec![first];
        }

        let target_model = first.request.model_id.clone();
        let mut batch = vec![first];
        let mut deferred = Vec::new();

        while batch.len() < max_size {
            match self.queue.pop() {
                Some(q) => {
                    if q.request.model_id == target_model {
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
            let active = self.active_count.load(Ordering::Relaxed);
            let next_tier = self
                .queue
                .peek()
                .map(|q| q.request.priority)
                .unwrap_or(crate::types::PriorityTier::Bronze);
            let tier_max = priority::max_concurrent_for_tier(next_tier, self.max_concurrent);
            if active >= tier_max {
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

            let batch = self.collect_batch(self.max_batch_size);
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
            let shared_state = self.shared_state.clone();
            let network_tx = self.network_tx.clone();
            let scheduler = self.scheduler.clone();

            tokio::spawn(async move {
                execute_batch(shared_state, network_tx, scheduler, batch, active_count).await;
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

        // Check for multi-turn KV-cache reuse
        let cache_start_pos = if let Some(ref session_id) = queued.request.session_id {
            // Collect active peer IDs for pipeline validation
            let active_peers: Vec<crate::types::NodeId> = self
                .shared_state
                .peer_registry
                .iter()
                .map(|e| e.key().clone())
                .collect();

            // Build the prompt to check prefix matching
            let prompt = {
                // Use a quick ChatML fallback for prefix comparison — the
                // actual template doesn't matter as long as we're consistent.
                crate::inference::chat_template::chatml_fallback(&queued.request.messages)
            };

            match self
                .kv_cache
                .check_multi_turn_reuse(session_id, &prompt, &active_peers)
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
            if let Some(ref session_id) = queued.request.session_id {
                let prompt =
                    crate::inference::chat_template::chatml_fallback(&queued.request.messages);
                self.kv_cache.register_multi_turn(
                    session_id,
                    queued.request.id,
                    crate::types::PipelineAssignment {
                        request_id: queued.request.id,
                        segments: vec![],
                        standbys: vec![],
                        tp_groups: vec![],
                    },
                    0,
                    prompt,
                );
            }
        }

        self.active_count.fetch_add(1, Ordering::Relaxed);
        let active_count = self.active_count.clone();
        let shared_state = self.shared_state.clone();
        let network_tx = self.network_tx.clone();
        let scheduler = self.scheduler.clone();
        let self_tx = self.self_tx.clone();
        let request = queued.request;
        let result_tx = queued.result_tx;
        let token_tx = queued.token_tx;

        tokio::spawn(async move {
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
            let escrow_id = if shared_state.escrow_manager.needs_escrow(estimated_cost) {
                match shared_state
                    .escrow_manager
                    .create_escrow(
                        request.id,
                        estimated_cost,
                        &request.requester,
                        &shared_state.credit_balance,
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

            let output = execute_request(
                shared_state.clone(),
                network_tx,
                scheduler,
                request.clone(),
                token_tx,
                preferred_pipeline,
            )
            .await;

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
                    if let Ok(mut samples) = shared_state.inference_latency_samples.write() {
                        if samples.len() >= 1000 {
                            samples.pop_front();
                        }
                        samples.push_back(latency_secs);
                    }
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

            // Release or refund escrow
            if let Some(eid) = escrow_id {
                match &output {
                    Ok(_) => {
                        // Release escrow — credits stay deducted (already charged)
                        if let Err(e) = shared_state
                            .escrow_manager
                            .release_escrow(eid, shared_state.identity.node_id())
                            .await
                        {
                            tracing::warn!(escrow_id = %eid, error = %e, "Failed to release escrow");
                        }
                    }
                    Err(_) => {
                        // Refund escrow — return credits on failure
                        if let Err(e) = shared_state
                            .escrow_manager
                            .refund_escrow(eid, &shared_state.credit_balance)
                            .await
                        {
                            tracing::warn!(escrow_id = %eid, error = %e, "Failed to refund escrow");
                        }
                    }
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
            }

            // Remove from active pipelines
            shared_state.active_pipelines.remove(&request.id);

            // Decrement active count so new requests can be dispatched
            active_count.fetch_sub(1, Ordering::Relaxed);

            if result_tx.send(output).is_err() {
                tracing::warn!(
                    request_id = %request.id,
                    "DIAG: result_tx receiver dropped — client disconnected before result"
                );
            }
        });
    }
}

/// Finalize a completed request: update stats and apply credit charges.
/// When `escrow_id` is `Some`, the escrow already deducted credits — skip the
/// direct charge to avoid double-billing the local API consumer.
async fn finalize_request(
    shared_state: &SharedState,
    request: &InferenceRequest,
    output: &Result<InferenceOutput, SwarmError>,
    escrow_id: Option<uuid::Uuid>,
) {
    if let Err(ref e) = output {
        tracing::error!(
            request_id = %request.id,
            model = %request.model_id,
            error = %e,
            "Inference request failed"
        );
        // Emit failure activity
        let mname = shared_state
            .model_registry
            .get_manifest(&request.model_id)
            .map(|m| m.name.clone());
        shared_state.emit_activity(crate::daemon::state::ActivityEvent {
            category: "inference",
            kind: "inference_failed",
            message: format!("Inference failed: {}", e),
            model_id: Some(request.model_id.0.clone()),
            model_name: mname,
            node_id: None,
            detail_num: None,
            detail_str: Some(format!("{}", e)),
            toast_level: Some("warning"),
            toast_duration_ms: Some(5000),
            shard_index: None,
            freed_bytes: None,
            holder_count_before: None,
            holder_count_after: None,
            remaining_local_shards: None,
            timestamp: None,
        });
    }

    // Local API requests use NodeId([0; 32]) as requester sentinel
    let is_local_api_request = request.requester == crate::types::NodeId([0u8; 32]);

    if let Ok(ref result) = output {
        if let Ok(mut stats) = shared_state.node_stats.try_write() {
            stats.requests_served += 1;
        }
        // Update Prometheus metrics
        shared_state
            .inference_requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Emit inference completion activity
        {
            let mname = shared_state
                .model_registry
                .get_manifest(&request.model_id)
                .map(|m| m.name.clone());
            let display = mname.as_deref().unwrap_or(&request.model_id.0);
            let total_tokens = result.prompt_tokens + result.completion_tokens;
            shared_state.emit_activity(crate::daemon::state::ActivityEvent {
                category: "inference",
                kind: "inference_completed",
                message: format!(
                    "Completed on {} — {} prompt + {} generated = {} total tokens ({})",
                    display,
                    result.prompt_tokens,
                    result.completion_tokens,
                    total_tokens,
                    result.finish_reason,
                ),
                model_id: Some(request.model_id.0.clone()),
                model_name: mname,
                node_id: None,
                detail_num: Some(total_tokens as i64),
                detail_str: Some(result.finish_reason.clone()),
                toast_level: None,
                toast_duration_ms: None,
                shard_index: None,
                freed_bytes: None,
                holder_count_before: None,
                holder_count_after: None,
                remaining_local_shards: None,
                timestamp: None,
            });
        }

        // Credit operations:
        // - Per-layer earn credits are handled in process_local_segment
        //   and handle_layer_forward (each node earns for layers it processed)
        // - Here we debit the local API consumer for requesting inference
        // - Skip if escrow was used — escrow already deducted the estimated cost
        // - Pool members (slaves): charge goes to the MASTER's balance via credit forward.
        //   The slave's dashboard is fully usable; usage is billed to the pool owner.
        let is_pool_member = {
            let ps = shared_state.pool_state.read().await;
            ps.as_ref()
                .map(|s| s.pool_id != *shared_state.identity.node_id())
                .unwrap_or(false)
        };
        if is_local_api_request && escrow_id.is_none() {
            let total_tokens = result.prompt_tokens + result.completion_tokens;
            let spent = crate::credit::ledger::RATE_INFERENCE_CONSUME * total_tokens as i64;

            if is_pool_member {
                // Slave device: forward the spend to the master's balance.
                // Use the same credit forward mechanism as earning, but negative.
                if let Some(ref tx) = *shared_state.pool_tx.read().await {
                    let my_id = shared_state.identity.node_id();
                    let pool_id = {
                        let ps = shared_state.pool_state.read().await;
                        ps.as_ref().map(|s| s.pool_id.clone())
                    };
                    if let Some(pid) = pool_id {
                        let forward = crate::pool::crypto::create_credit_forward(
                            &shared_state.identity,
                            &pid,
                            my_id,
                            &pid,
                            -spent, // negative = spend deduction
                        );
                        let _ = tx
                            .send(crate::pool::types::PoolCommand::ProcessCreditForward { forward })
                            .await;
                        tracing::debug!(
                            spent,
                            request_id = %request.id,
                            "Forwarded inference spend to pool owner"
                        );
                    }
                }
            } else {
                // Normal node or pool owner: charge locally
                if let Err(e) = crate::credit::ledger::apply_credit_direct(
                    &shared_state.credit_balance,
                    &shared_state.db,
                    -spent,
                    false,
                )
                .await
                {
                    tracing::warn!(error = %e, "Failed to persist credit spend");
                }
                tracing::debug!(
                    spent,
                    total_tokens,
                    request_id = %request.id,
                    "Spent credits for consuming inference"
                );
            }
        }
    }

    // Clean up per-request KV-cache entries now that the request is done.
    // EXCEPT when the request has a session_id — those entries persist for
    // multi-turn reuse. They'll be cleaned up by the TTL-based expiry instead.
    if request.session_id.is_none() {
        let req_id_str = request.id.to_string();
        shared_state.kv_cache_store.cleanup_request_id(&req_id_str);
    } else {
        tracing::debug!(
            request_id = %request.id,
            session_id = ?request.session_id,
            "Preserving KV-cache for multi-turn session"
        );
    }
}

/// Execute a batch of requests that target the same model.
///
/// For local inference (full model loaded), the batch shares a single executor
/// lock acquisition — requests are processed sequentially within the lock,
/// avoiding repeated lock acquire/release overhead.
///
/// For distributed inference, each request gets its own pipeline and they
/// execute concurrently.
async fn execute_batch(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    batch: Vec<QueuedRequest>,
    active_count: Arc<AtomicUsize>,
) {
    let batch_size = batch.len();
    let is_split_mode = shared_state.config.inference.shard_range.is_some();
    let model_loaded = shared_state
        .model_loaded
        .load(std::sync::atomic::Ordering::Acquire);

    if model_loaded && !is_split_mode {
        // Local inference batch: hold the executor lock once, process all requests
        execute_local_batch(shared_state, batch, active_count).await;
    } else {
        // Distributed inference batch: spawn each request independently
        // They'll assemble their own pipelines and run concurrently.
        execute_distributed_batch(shared_state, network_tx, scheduler, batch, active_count).await;
    }

    tracing::debug!(batch_size, "Batch execution complete");
}

/// RAII guard that decrements active_count for any unprocessed batch items on drop.
/// Ensures active_count is always decremented even if batch processing panics mid-loop.
struct BatchCleanup {
    active_count: Arc<AtomicUsize>,
    remaining: usize,
}

impl BatchCleanup {
    fn complete_one(&mut self) {
        if self.remaining > 0 {
            self.active_count.fetch_sub(1, Ordering::Relaxed);
            self.remaining -= 1;
        }
    }
}

impl Drop for BatchCleanup {
    fn drop(&mut self) {
        if self.remaining > 0 {
            self.active_count
                .fetch_sub(self.remaining, Ordering::Relaxed);
        }
    }
}

/// Execute a batch of requests locally, sharing the model lock.
///
/// Acquires the executor mutex once and processes all requests sequentially.
/// Each request gets its own generation call and independent output.
async fn execute_local_batch(
    shared_state: Arc<SharedState>,
    batch: Vec<QueuedRequest>,
    active_count: Arc<AtomicUsize>,
) {
    let mut executor = shared_state.executor.lock().await;
    let batch_size = batch.len();
    let mut cleanup = BatchCleanup {
        active_count: active_count.clone(),
        remaining: batch_size,
    };

    tracing::info!(batch_size, "Executing local inference batch");

    for queued in batch {
        let request = queued.request;
        let result_tx = queued.result_tx;
        let token_tx = queued.token_tx;

        let output = if executor.is_loaded() {
            let prompt = {
                let info = shared_state.loaded_model_info.read().await;
                match info.as_ref() {
                    Some(i) => chat_template::build_prompt(
                        &request.messages,
                        i.chat_template.as_deref(),
                        &i.bos_token,
                        &i.eos_token,
                    ),
                    None => chat_template::chatml_fallback(&request.messages),
                }
            };

            tracing::info!(
                request_id = %request.id,
                model = %request.model_id,
                "Executing inference locally (batched)"
            );

            // Extract chat-template stop strings (e.g. "<|user|>", "<|im_end|>")
            let local_stop_strings = {
                let info = shared_state.loaded_model_info.read().await;
                let tmpl = info.as_ref().and_then(|i| i.chat_template.as_deref());
                chat_template::extract_stop_strings(tmpl)
            };

            // Use streaming generation if the request has a token channel
            if let Some(ref tx) = token_tx {
                let tx = tx.clone();
                let session_id = request.session_id.clone();
                let mut accumulated = String::new();
                let stop_strings = local_stop_strings.clone();
                let mut hit_stop = false;
                match executor.generate_stream(
                    &prompt,
                    &request.sampling_params,
                    |token: &str| -> bool {
                        accumulated.push_str(token);
                        // Check for chat template stop strings
                        if let Some(stop) = stop_strings
                            .iter()
                            .find(|s| accumulated.contains(s.as_str()))
                        {
                            // Truncate accumulated text at the stop string
                            if let Some(pos) = accumulated.find(stop.as_str()) {
                                accumulated.truncate(pos);
                            }
                            hit_stop = true;
                            return false; // Signal to stop generation
                        }
                        let event = StreamingTokenEvent {
                            text: token.to_string(),
                            finish_reason: None,
                        };
                        tx.try_send(event).is_ok()
                    },
                ) {
                    Ok(gen_result) => {
                        let finish = if hit_stop {
                            "stop".to_string()
                        } else {
                            gen_result.finish_reason.as_str().to_string()
                        };
                        // Send final done event
                        let done_event = StreamingTokenEvent {
                            text: String::new(),
                            finish_reason: Some(finish.clone()),
                        };
                        let _ = tx.try_send(done_event);
                        Ok(InferenceOutput {
                            request_id: request.id,
                            content: accumulated,
                            prompt_tokens: gen_result.prompt_tokens,
                            completion_tokens: gen_result.completion_tokens,
                            finish_reason: finish,
                            session_id,
                            token_logprobs: vec![],
                        })
                    }
                    Err(e) => Err(e),
                }
            } else {
                match executor.generate(&prompt, &request.sampling_params) {
                    Ok((mut content, gen_result)) => {
                        // Check for chat template stop strings in generated content
                        let mut finish = gen_result.finish_reason.as_str().to_string();
                        for stop in &local_stop_strings {
                            if let Some(pos) = content.find(stop.as_str()) {
                                content.truncate(pos);
                                finish = "stop".to_string();
                                break;
                            }
                        }
                        // Strip trailing partial stop strings
                        for stop in &local_stop_strings {
                            for end_len in (1..stop.len()).rev() {
                                let prefix = &stop[..end_len];
                                if content.ends_with(prefix) {
                                    content.truncate(content.len() - end_len);
                                    break;
                                }
                            }
                        }
                        Ok(InferenceOutput {
                            request_id: request.id,
                            content,
                            prompt_tokens: gen_result.prompt_tokens,
                            completion_tokens: gen_result.completion_tokens,
                            finish_reason: finish,
                            session_id: request.session_id.clone(),
                            token_logprobs: vec![],
                        })
                    }
                    Err(e) => Err(e),
                }
            }
        } else {
            Err(SwarmError::NoModelLoaded)
        };

        finalize_request(&shared_state, &request, &output, None).await;
        shared_state.active_pipelines.remove(&request.id);
        cleanup.complete_one();
        if result_tx.send(output).is_err() {
            tracing::warn!(
                request_id = %request.id,
                "DIAG: batch result_tx receiver dropped"
            );
        }
    }

    tracing::debug!(batch_size, "Local batch complete");
}

/// Execute a batch of distributed inference requests concurrently.
///
/// Each request gets its own pipeline. They share the active_count
/// and are finalized independently.
async fn execute_distributed_batch(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    batch: Vec<QueuedRequest>,
    active_count: Arc<AtomicUsize>,
) {
    let mut handles = Vec::with_capacity(batch.len());

    for queued in batch {
        let shared_state = shared_state.clone();
        let network_tx = network_tx.clone();
        let scheduler = scheduler.clone();
        let active_count = active_count.clone();

        handles.push(tokio::spawn(async move {
            let request = queued.request;
            let result_tx = queued.result_tx;
            let token_tx = queued.token_tx;

            let output = execute_request(
                shared_state.clone(),
                network_tx,
                scheduler,
                request.clone(),
                token_tx,
                None, // No pipeline affinity for batched requests
            )
            .await;

            finalize_request(&shared_state, &request, &output, None).await;
            shared_state.active_pipelines.remove(&request.id);
            // Return true to signal the join loop that we already decremented
            active_count.fetch_sub(1, Ordering::Relaxed);
            if result_tx.send(output).is_err() {
                tracing::warn!(
                    request_id = %request.id,
                    "DIAG: distributed batch result_tx receiver dropped"
                );
            }
            true // task completed normally, already decremented
        }));
    }

    // Wait for all requests in the batch to complete.
    // Only decrement if the task panicked BEFORE it could decrement itself.
    for handle in handles {
        match handle.await {
            Ok(_) => {} // task already decremented
            Err(_) => {
                active_count.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }
}

/// Execute a single inference request — either locally or via distributed pipeline.
async fn execute_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    request: InferenceRequest,
    token_tx: Option<StreamingTokenTx>,
    preferred_pipeline: Option<PipelineAssignment>,
) -> Result<InferenceOutput, SwarmError> {
    let model_id = &request.model_id;

    // Update model trust on inference request — promotes to DemandVerified
    // after threshold, enabling auto-manage to propagate this model.
    {
        let mut trust = shared_state
            .model_trust
            .entry(model_id.clone())
            .or_insert_with(crate::types::ModelTrustInfo::new_discovered);
        trust.record_request();
        // Persist on promotion only (not every request)
        if trust.total_requests == 3 {
            let _ = shared_state
                .db
                .put_json("model_trust", &model_id.0, trust.value());
        }
    }

    // Check if we can handle this entirely locally.
    // Use the atomic flag to avoid locking the executor mutex just to check readiness.
    // Skip the llama.cpp path when a LoRA adapter is requested — LoRA is only
    // supported on the split model (candle) path via forward_with_lora().
    let local_node_id = shared_state.identity.node_id().clone();
    let is_split_mode = shared_state.config.inference.shard_range.is_some();
    let has_lora = request.lora_adapter.is_some();
    if shared_state
        .model_loaded
        .load(std::sync::atomic::Ordering::Acquire)
        && !is_split_mode
        && !has_lora
    {
        // Local-only inference path (single node has the model loaded)
        let mut executor = shared_state.executor.lock().await;
        tracing::info!(
            request_id = %request.id,
            model = %model_id,
            "Executing inference locally"
        );

        let prompt = {
            let info = shared_state.loaded_model_info.read().await;
            match info.as_ref() {
                Some(i) => chat_template::build_prompt(
                    &request.messages,
                    i.chat_template.as_deref(),
                    &i.bos_token,
                    &i.eos_token,
                ),
                None => chat_template::chatml_fallback(&request.messages),
            }
        };

        // Use streaming generation if token_tx is present
        if let Some(ref tx) = token_tx {
            let tx = tx.clone();
            let mut accumulated = String::new();
            let gen_result = executor.generate_stream(
                &prompt,
                &request.sampling_params,
                |token: &str| -> bool {
                    accumulated.push_str(token);
                    let event = StreamingTokenEvent {
                        text: token.to_string(),
                        finish_reason: None,
                    };
                    tx.try_send(event).is_ok()
                },
            )?;
            // Send final done event
            let done_event = StreamingTokenEvent {
                text: String::new(),
                finish_reason: Some(gen_result.finish_reason.as_str().to_string()),
            };
            if tx.try_send(done_event).is_err() {
                tracing::warn!(
                    request_id = %request.id,
                    "DIAG: streaming done_event send failed — receiver dropped"
                );
            }
            return Ok(InferenceOutput {
                request_id: request.id,
                content: accumulated,
                prompt_tokens: gen_result.prompt_tokens,
                completion_tokens: gen_result.completion_tokens,
                finish_reason: gen_result.finish_reason.as_str().to_string(),
                session_id: request.session_id.clone(),
                token_logprobs: vec![],
            });
        }

        let (content, gen_result) = executor.generate(&prompt, &request.sampling_params)?;

        return Ok(InferenceOutput {
            request_id: request.id,
            content,
            prompt_tokens: gen_result.prompt_tokens,
            completion_tokens: gen_result.completion_tokens,
            finish_reason: gen_result.finish_reason.as_str().to_string(),
            session_id: request.session_id.clone(),
            token_logprobs: vec![],
        });
    }

    // ── On-demand shard loading ────────────────────────────────────────
    // If this model has shards on disk but they aren't loaded in split_models,
    // load them now (with LRU eviction if needed) instead of failing.
    {
        let already_loaded = shared_state
            .split_models
            .iter()
            .any(|e| e.key().0 == *model_id);
        if !already_loaded {
            let model_dir = shared_state
                .config
                .node
                .data_dir
                .join("models")
                .join(&model_id.0);
            let has_shards_on_disk = model_dir.exists()
                && (model_dir.join("shard_000.bin").exists()
                    || model_dir.join("model.gguf").exists());

            if has_shards_on_disk {
                tracing::info!(
                    request_id = %request.id,
                    model = %model_id,
                    "On-demand loading: model has shards on disk but not loaded"
                );

                // check_and_load_model has internal TOCTOU guard via loading_models.
                // If another task is already loading, it returns immediately.
                // We then wait on the notify for up to 60s.
                let maybe_notify = shared_state
                    .loading_models
                    .get(model_id)
                    .map(|r| r.value().clone());

                if let Some(notify) = maybe_notify {
                    // Another task is loading — wait for it
                    let _ =
                        tokio::time::timeout(std::time::Duration::from_secs(60), notify.notified())
                            .await;
                } else {
                    // No one loading — trigger load (guard inside check_and_load_model)
                    let vram_budget = crate::model::auto_manage::compute_vram_budget(&shared_state);
                    crate::model::auto_manage::check_and_load_model(
                        &shared_state,
                        model_id,
                        vram_budget,
                    )
                    .await;
                }
            }
        }
    }

    // Distributed inference path: assemble pipeline across nodes
    // Check if the requested model exists in the registry before attempting pipeline assembly.
    // This gives a clearer error than "No model loaded" when the model name is wrong.
    {
        let has_manifest = shared_state.model_registry.get_manifest(model_id).is_some();
        let has_split = shared_state
            .split_models
            .iter()
            .any(|e| e.key().0 == *model_id);
        if !has_manifest && !has_split {
            return Err(SwarmError::PipelineError(format!(
                "Model '{}' not found. Available models: {}",
                model_id.0,
                shared_state
                    .model_registry
                    .models()
                    .iter()
                    .map(|m| m.id.0.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }

    // S5: Fire-and-forget DHT provider query to pre-warm the shard holder cache.
    // Results arrive asynchronously and are merged into model_registry by NetworkManager.
    // First request for a model may miss the cache, but subsequent ones benefit.
    let _ = shared_state.dht_query_tx.try_send(model_id.clone());

    let schedule_start = std::time::Instant::now();
    tracing::info!(
        request_id = %request.id,
        model = %model_id,
        "Assembling distributed pipeline"
    );

    // Pipeline affinity: reuse previous pipeline if all nodes are still connected
    let assignment = if let Some(prev) = preferred_pipeline {
        let all_connected = prev.segments.iter().all(|seg| {
            seg.node_id == local_node_id || shared_state.peer_registry.contains_key(&seg.node_id)
        });
        if all_connected && !prev.segments.is_empty() {
            tracing::info!(
                request_id = %request.id,
                segments = prev.segments.len(),
                "Reusing previous pipeline (KV cache affinity)"
            );
            PipelineAssignment {
                request_id: request.id,
                ..prev
            }
        } else {
            scheduler.assemble_pipeline_for(model_id, &local_node_id, request.id)?
        }
    } else {
        scheduler.assemble_pipeline_for(model_id, &local_node_id, request.id)?
    };
    let schedule_ms = schedule_start.elapsed().as_millis() as u64;

    tracing::info!(
        request_id = %request.id,
        segments = assignment.segments.len(),
        standbys = assignment.standbys.len(),
        schedule_ms,
        "DIAG: pipeline assembled"
    );
    for (i, seg) in assignment.segments.iter().enumerate() {
        tracing::info!(
            request_id = %request.id,
            segment = i,
            node = %seg.node_id,
            layer_start = seg.layer_range.0,
            layer_end = seg.layer_range.1,
            "Pipeline segment"
        );
    }

    // Store assignment in shared state for monitoring
    let assignment_ref = assignment.clone();
    shared_state
        .active_pipelines
        .insert(request.id, assignment.clone());

    // Execute the distributed pipeline
    let execute_start = std::time::Instant::now();
    let network_tx_for_error = network_tx.clone();
    let mut pipeline = PipelineExecutor::new(
        shared_state.clone(),
        network_tx,
        request.clone(),
        assignment,
    );

    let result = pipeline.execute(token_tx).await;
    let execute_ms = execute_start.elapsed().as_millis() as u64;
    match &result {
        Ok(output) => {
            tracing::info!(
                request_id = %request.id,
                schedule_ms,
                execute_ms,
                total_ms = schedule_ms + execute_ms,
                prompt_tokens = output.prompt_tokens,
                completion_tokens = output.completion_tokens,
                finish_reason = %output.finish_reason,
                "DIAG: execute_request completed successfully"
            );

            // Update trust for all remote peers that participated in the pipeline
            for seg in &assignment_ref.segments {
                if seg.node_id != local_node_id {
                    shared_state.trust_manager.update_trust(
                        &shared_state.peer_registry,
                        &seg.node_id,
                        crate::credit::trust::TrustEvent::InferenceSuccess,
                    );
                }
            }

            // Spot-check: probabilistically verify remote peer output
            spot_check_distributed_result(
                &shared_state,
                &request,
                &assignment_ref,
                &local_node_id,
                output,
            )
            .await;
        }
        Err(ref e) => {
            tracing::error!(
                request_id = %request.id,
                schedule_ms,
                execute_ms,
                "DIAG: execute_request failed: {e}"
            );

            // Apply credit penalty for distributed inference failure
            let penalty = shared_state.config.pool.credit_rates.penalty_serve_failure;
            if let Err(pe) = crate::credit::ledger::apply_credit_direct(
                &shared_state.credit_balance,
                &shared_state.db,
                -penalty,
                false,
            )
            .await
            {
                tracing::warn!(error = %pe, "Failed to apply failure penalty");
            } else {
                tracing::info!(
                    penalty,
                    request_id = %request.id,
                    "Applied credit penalty for distributed inference failure"
                );
            }

            // Broadcast pipeline error so peers can update shard availability
            crate::inference::pipeline::broadcast_pipeline_error(
                &network_tx_for_error,
                request.id,
                &e.to_string(),
            )
            .await;
        }
    }
    result
}

/// Probabilistic spot-check of distributed inference results.
///
/// After a successful distributed inference, randomly selects remote peers
/// (based on AntiGaming spot-check rate) and validates the output is plausible.
/// On failure, reduces trust for the offending peer.
async fn spot_check_distributed_result(
    shared_state: &Arc<SharedState>,
    request: &InferenceRequest,
    assignment: &PipelineAssignment,
    local_node_id: &NodeId,
    output: &InferenceOutput,
) {
    // Only spot-check if there are remote peers in the pipeline
    let remote_peers: Vec<NodeId> = assignment
        .segments
        .iter()
        .filter(|s| s.node_id != *local_node_id)
        .map(|s| s.node_id.clone())
        .collect();
    if remote_peers.is_empty() {
        return;
    }

    // Ask anti-gaming whether this request should be spot-checked
    let should_check = {
        let ag = shared_state.anti_gaming.lock().await;
        let rate = ag.effective_spot_check_rate(&remote_peers[0]);
        rand::random::<f64>() < rate
    };

    if !should_check {
        return;
    }

    tracing::info!(
        request_id = %request.id,
        remote_peers = remote_peers.len(),
        "Spot-check: verifying distributed inference result"
    );

    // Validation 1: Check output is non-empty and reasonable
    let text = &output.content;
    if text.is_empty() && output.completion_tokens > 0 {
        tracing::warn!(
            request_id = %request.id,
            "Spot-check FAIL: empty text with non-zero completion_tokens"
        );
        for peer in &remote_peers {
            report_spot_check_failure(shared_state, peer).await;
        }
        return;
    }

    // Validation 2: Check for garbage output (all same char, all whitespace for long outputs)
    if output.completion_tokens > 10 {
        let chars: Vec<char> = text.chars().collect();
        if !chars.is_empty() {
            let first = chars[0];
            if chars.iter().all(|&c| c == first) {
                tracing::warn!(
                    request_id = %request.id,
                    repeated_char = ?first,
                    "Spot-check FAIL: output is all repeated characters"
                );
                for peer in &remote_peers {
                    report_spot_check_failure(shared_state, peer).await;
                }
                return;
            }
        }
    }

    // Validation 3: Token count consistency
    if output.completion_tokens == 0 && !text.is_empty() {
        tracing::warn!(
            request_id = %request.id,
            text_len = text.len(),
            "Spot-check FAIL: non-empty text but zero completion_tokens"
        );
        for peer in &remote_peers {
            report_spot_check_failure(shared_state, peer).await;
        }
        return;
    }

    tracing::debug!(
        request_id = %request.id,
        "Spot-check PASS: output appears valid"
    );
}

/// Report a spot-check failure for a peer: reduce trust and log the penalty.
async fn report_spot_check_failure(shared_state: &Arc<SharedState>, peer: &NodeId) {
    // Update trust score
    shared_state.trust_manager.update_trust(
        &shared_state.peer_registry,
        peer,
        crate::credit::trust::TrustEvent::SpotCheckFail,
    );

    // Report to anti-gaming
    let penalty = {
        let mut ag = shared_state.anti_gaming.lock().await;
        ag.report_spot_check_failure(peer)
    };

    tracing::warn!(
        peer = %peer,
        penalty = ?penalty,
        "Spot-check failure reported — trust reduced"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ChatMessage, ModelId, PriorityTier, Role, SamplingParams};

    fn make_request(priority: PriorityTier) -> InferenceRequest {
        InferenceRequest {
            id: uuid::Uuid::new_v4(),
            model_id: ModelId("test".into()),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".into(),
                images: vec![],
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: crate::types::NodeId([0u8; 32]),
            priority,
            created_at: chrono::Utc::now(),
            session_id: None,
            lora_adapter: None,
        }
    }

    fn make_request_with_model(priority: PriorityTier, model: &str) -> InferenceRequest {
        InferenceRequest {
            id: uuid::Uuid::new_v4(),
            model_id: ModelId(model.into()),
            messages: vec![ChatMessage {
                role: Role::User,
                content: "hello".into(),
                images: vec![],
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: crate::types::NodeId([0u8; 32]),
            priority,
            created_at: chrono::Utc::now(),
            session_id: None,
            lora_adapter: None,
        }
    }

    #[test]
    fn priority_ordering() {
        let (tx_a, _) = oneshot::channel();
        let (tx_b, _) = oneshot::channel();
        let (tx_c, _) = oneshot::channel();

        let mut queue = BinaryHeap::new();
        queue.push(QueuedRequest {
            request: make_request(PriorityTier::Bronze),
            result_tx: tx_a,
            token_tx: None,
        });
        queue.push(QueuedRequest {
            request: make_request(PriorityTier::Platinum),
            result_tx: tx_b,
            token_tx: None,
        });
        queue.push(QueuedRequest {
            request: make_request(PriorityTier::Silver),
            result_tx: tx_c,
            token_tx: None,
        });

        // Highest priority should come out first
        let first = queue.pop().unwrap();
        assert_eq!(first.request.priority, PriorityTier::Platinum);
        let second = queue.pop().unwrap();
        assert_eq!(second.request.priority, PriorityTier::Silver);
        let third = queue.pop().unwrap();
        assert_eq!(third.request.priority, PriorityTier::Bronze);
    }

    #[test]
    fn collect_batch_groups_same_model() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let mut config = Config::default();
        config.inference.max_batch_size = 4;
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (shared_state, _, _) = SharedState::new(config, identity, db, executor, None);

        let (cmd_tx, cmd_rx) = mpsc::channel(64);
        let (net_tx, _net_rx) = mpsc::channel(64);
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let _ = shutdown_tx;

        let mut router = InferenceRouter::new(shared_state, cmd_rx, cmd_tx, net_tx, shutdown_rx);

        // Add 3 requests for model "alpha", 2 for model "beta"
        for _ in 0..3 {
            let (tx, _) = oneshot::channel();
            router.queue.push(QueuedRequest {
                request: make_request_with_model(PriorityTier::Silver, "alpha"),
                result_tx: tx,
                token_tx: None,
            });
        }
        for _ in 0..2 {
            let (tx, _) = oneshot::channel();
            router.queue.push(QueuedRequest {
                request: make_request_with_model(PriorityTier::Silver, "beta"),
                result_tx: tx,
                token_tx: None,
            });
        }

        // Collect batch of max 4 — should get all from one model
        let batch = router.collect_batch(4);
        // All items in the batch should have the same model
        let model = &batch[0].request.model_id;
        assert!(batch.iter().all(|q| &q.request.model_id == model));
        // The remaining queue should have the other model's requests
        assert!(!router.queue.is_empty());
    }

    #[test]
    fn collect_batch_single_returns_one() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = Config::default(); // max_batch_size = 1
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (shared_state, _, _) = SharedState::new(config, identity, db, executor, None);

        let (_cmd_tx, cmd_rx) = mpsc::channel(64);
        let (net_tx, _net_rx) = mpsc::channel(64);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let mut router =
            InferenceRouter::new(shared_state, cmd_rx, _cmd_tx.clone(), net_tx, shutdown_rx);

        // Add 3 requests
        for _ in 0..3 {
            let (tx, _) = oneshot::channel();
            router.queue.push(QueuedRequest {
                request: make_request(PriorityTier::Silver),
                result_tx: tx,
                token_tx: None,
            });
        }

        // With max_batch_size=1, should only get 1
        let batch = router.collect_batch(1);
        assert_eq!(batch.len(), 1);
        assert_eq!(router.queue.len(), 2);
    }

    #[test]
    fn collect_batch_respects_max_size() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let mut config = Config::default();
        config.inference.max_batch_size = 2;
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (shared_state, _, _) = SharedState::new(config, identity, db, executor, None);

        let (_cmd_tx, cmd_rx) = mpsc::channel(64);
        let (net_tx, _net_rx) = mpsc::channel(64);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let mut router =
            InferenceRouter::new(shared_state, cmd_rx, _cmd_tx.clone(), net_tx, shutdown_rx);

        // Add 5 requests all same model
        for _ in 0..5 {
            let (tx, _) = oneshot::channel();
            router.queue.push(QueuedRequest {
                request: make_request(PriorityTier::Silver),
                result_tx: tx,
                token_tx: None,
            });
        }

        // With max_batch_size=2, should only get 2
        let batch = router.collect_batch(2);
        assert_eq!(batch.len(), 2);
        assert_eq!(router.queue.len(), 3);
    }

    #[test]
    fn collect_batch_empty_queue() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = Config::default();
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (shared_state, _, _) = SharedState::new(config, identity, db, executor, None);

        let (_cmd_tx, cmd_rx) = mpsc::channel(64);
        let (net_tx, _net_rx) = mpsc::channel(64);
        let (_shutdown_tx, shutdown_rx) = watch::channel(false);

        let mut router =
            InferenceRouter::new(shared_state, cmd_rx, _cmd_tx.clone(), net_tx, shutdown_rx);

        let batch = router.collect_batch(4);
        assert!(batch.is_empty());
    }

    #[test]
    fn default_batch_config() {
        let config = crate::config::Config::default();
        assert_eq!(config.inference.max_batch_size, 1);
        assert_eq!(config.inference.batch_timeout_ms, 50);
    }
}
