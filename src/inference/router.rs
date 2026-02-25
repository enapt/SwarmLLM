use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, oneshot, watch};

use crate::daemon::SharedState;
use crate::error::SwarmError;
use crate::inference::executor::build_chat_prompt;
use crate::inference::kv_cache::KvCacheManager;
use crate::inference::pipeline::PipelineExecutor;
use crate::inference::scheduler::PipelineScheduler;
use crate::types::{InferenceRequest, NetworkCommand, SwarmMessage};

/// Result channel for returning inference output to API callers.
pub type InferenceResultTx = oneshot::Sender<Result<InferenceOutput, SwarmError>>;

/// A queued inference request with its result channel and priority ordering.
struct QueuedRequest {
    request: InferenceRequest,
    result_tx: InferenceResultTx,
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
}

/// Command sent to the InferenceRouter from the API layer or network.
pub enum RouterCommand {
    /// Submit a new inference request with a channel for the result.
    Submit {
        request: InferenceRequest,
        result_tx: InferenceResultTx,
    },
    /// A network message relevant to inference (LayerForward, LayerResult, etc.)
    NetworkMessage(SwarmMessage),
}

/// The InferenceRouter is the brain of distributed inference.
///
/// It receives inference requests, places them in a priority queue,
/// assembles pipelines using the scheduler, and kicks off execution.
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
}

impl InferenceRouter {
    pub fn new(
        shared_state: Arc<SharedState>,
        command_rx: mpsc::Receiver<RouterCommand>,
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
        Self {
            shared_state: shared_state.clone(),
            command_rx,
            network_tx,
            shutdown_rx,
            queue: BinaryHeap::new(),
            scheduler: PipelineScheduler::new(shared_state),
            kv_cache: KvCacheManager::new(kv_cache_ttl),
            max_concurrent,
            active_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Run the router event loop.
    pub async fn run(mut self) -> Result<(), SwarmError> {
        tracing::info!("InferenceRouter running");

        // Drain interval — process queued requests periodically
        let mut drain_interval = tokio::time::interval(std::time::Duration::from_millis(50));
        drain_interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        // KV-cache cleanup interval
        let mut cache_cleanup = tokio::time::interval(std::time::Duration::from_secs(30));
        cache_cleanup.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = self.shutdown_rx.changed() => {
                    if *self.shutdown_rx.borrow() {
                        tracing::info!("InferenceRouter shutting down");
                        break;
                    }
                }
                cmd = self.command_rx.recv() => {
                    match cmd {
                        Some(RouterCommand::Submit { request, result_tx }) => {
                            self.handle_submit(request, result_tx);
                        }
                        Some(RouterCommand::NetworkMessage(msg)) => {
                            self.handle_network_message(msg).await;
                        }
                        None => {
                            tracing::info!("Command channel closed, shutting down");
                            break;
                        }
                    }
                }
                _ = drain_interval.tick() => {
                    self.drain_queue().await;
                }
                _ = cache_cleanup.tick() => {
                    let expired = self.kv_cache.cleanup_expired();
                    if expired > 0 {
                        tracing::debug!(expired, "Cleaned up expired KV-cache sessions");
                    }
                }
            }
        }

        Ok(())
    }

    /// Handle a new inference submission.
    ///
    /// Checks credit balance / priority tier before queueing.
    fn handle_submit(&mut self, request: InferenceRequest, result_tx: InferenceResultTx) {
        // Check credit balance — Bronze tier nodes are deprioritized but not blocked
        // per spec: "Credit errors: degrade priority tier, never block"
        let balance = {
            if let Ok(bal) = self.shared_state.credit_balance.try_read() {
                bal.balance
            } else {
                0
            }
        };

        let priority = if balance < 0 {
            // Negative balance → force Bronze tier
            tracing::debug!(
                request_id = %request.id,
                balance,
                "Negative credit balance, degrading to Bronze tier"
            );
            crate::types::PriorityTier::Bronze
        } else {
            request.priority
        };

        let mut adjusted_request = request;
        adjusted_request.priority = priority;

        tracing::info!(
            request_id = %adjusted_request.id,
            model = %adjusted_request.model_id,
            priority = ?adjusted_request.priority,
            "Queued inference request"
        );

        self.queue.push(QueuedRequest {
            request: adjusted_request,
            result_tx,
        });
    }

    /// Handle network messages (LayerResult, InferenceError, etc.)
    async fn handle_network_message(&mut self, msg: SwarmMessage) {
        match msg {
            SwarmMessage::LayerResult(result) => {
                tracing::debug!(
                    request_id = %result.request_id,
                    tokens = result.token_ids.len(),
                    "Received layer result from network"
                );
                // Pipeline executor handles this via its own channels
                // Store in active_pipelines for monitoring
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
                tracing::info!(
                    request_id = %req.id,
                    requester = %req.requester,
                    model = %req.model_id,
                    "Received remote inference request"
                );
                // Remote requests get queued with a result that routes back over network
                // For now, log and skip (full cross-node routing in wiring step)
            }
            _ => {}
        }
    }

    /// Drain the priority queue and execute requests.
    async fn drain_queue(&mut self) {
        while self.active_count.load(Ordering::Relaxed) < self.max_concurrent {
            let queued = match self.queue.pop() {
                Some(q) => q,
                None => break,
            };

            self.active_count.fetch_add(1, Ordering::Relaxed);
            let active_count = self.active_count.clone();
            let shared_state = self.shared_state.clone();
            let network_tx = self.network_tx.clone();
            let scheduler = self.scheduler.clone();
            let request = queued.request;
            let result_tx = queued.result_tx;

            tokio::spawn(async move {
                let output =
                    execute_request(shared_state.clone(), network_tx, scheduler, request.clone())
                        .await;

                // Update stats and apply credit events
                let local_node_id = shared_state.identity.node_id().clone();
                let is_remote_request = request.requester != local_node_id;

                if let Ok(ref result) = output {
                    if let Ok(mut stats) = shared_state.node_stats.try_write() {
                        stats.requests_served += 1;
                    }

                    // Credit operations: earn if we served a remote request,
                    // spend if we consumed inference as the requester
                    let total_tokens = result.prompt_tokens + result.completion_tokens;
                    let layers = 1u32; // Local path = 1 logical layer pass

                    if is_remote_request {
                        // We served someone else — earn credits
                        let mut bal = shared_state.credit_balance.write().await;
                        let earned = crate::credit::ledger::RATE_INFERENCE_SERVE
                            * layers as i64
                            * total_tokens as i64;
                        bal.balance += earned;
                        bal.lifetime_earned += earned as u64;
                        bal.last_updated = chrono::Utc::now();
                        tracing::debug!(
                            earned,
                            request_id = %request.id,
                            "Earned credits for serving inference"
                        );
                    } else {
                        // We requested inference — spend credits
                        let mut bal = shared_state.credit_balance.write().await;
                        let spent = crate::credit::ledger::RATE_INFERENCE_CONSUME
                            * layers as i64
                            * total_tokens as i64;
                        bal.balance -= spent;
                        bal.lifetime_spent += spent as u64;
                        bal.last_updated = chrono::Utc::now();
                        tracing::debug!(
                            spent,
                            request_id = %request.id,
                            "Spent credits for consuming inference"
                        );
                    }
                }

                // Remove from active pipelines
                shared_state.active_pipelines.remove(&request.id);

                // Decrement active count so new requests can be dispatched
                active_count.fetch_sub(1, Ordering::Relaxed);

                let _ = result_tx.send(output);
            });
        }
    }
}

/// Execute a single inference request — either locally or via distributed pipeline.
async fn execute_request(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    request: InferenceRequest,
) -> Result<InferenceOutput, SwarmError> {
    let model_id = &request.model_id;

    // Check if we can handle this entirely locally
    let local_node_id = shared_state.identity.node_id().clone();
    let mut executor = shared_state.executor.lock().await;

    // Only use local executor when the full model is loaded (not in split/shard mode)
    let is_split_mode = shared_state.config.inference.shard_range.is_some();
    if executor.is_loaded() && !is_split_mode {
        // Local-only inference path (single node has the model loaded)
        tracing::info!(
            request_id = %request.id,
            model = %model_id,
            "Executing inference locally"
        );

        let prompt = build_chat_prompt(&request.messages);
        let (content, gen_result) = executor.generate(&prompt, &request.sampling_params)?;

        return Ok(InferenceOutput {
            request_id: request.id,
            content,
            prompt_tokens: gen_result.prompt_tokens,
            completion_tokens: gen_result.completion_tokens,
            finish_reason: gen_result.finish_reason.as_str().to_string(),
        });
    }
    drop(executor);

    // Distributed inference path: assemble pipeline across nodes
    tracing::info!(
        request_id = %request.id,
        model = %model_id,
        "Assembling distributed pipeline"
    );

    let assignment = scheduler.assemble_pipeline(model_id, &local_node_id)?;

    // Store assignment in shared state for monitoring
    shared_state
        .active_pipelines
        .insert(request.id, assignment.clone());

    // Execute the distributed pipeline
    let mut pipeline = PipelineExecutor::new(
        shared_state.clone(),
        network_tx,
        request.clone(),
        assignment,
    );

    pipeline.execute().await
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
            }],
            sampling_params: SamplingParams::default(),
            stream: false,
            requester: crate::types::NodeId([0u8; 32]),
            priority,
            created_at: chrono::Utc::now(),
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
        });
        queue.push(QueuedRequest {
            request: make_request(PriorityTier::Platinum),
            result_tx: tx_b,
        });
        queue.push(QueuedRequest {
            request: make_request(PriorityTier::Silver),
            result_tx: tx_c,
        });

        // Highest priority should come out first
        let first = queue.pop().unwrap();
        assert_eq!(first.request.priority, PriorityTier::Platinum);
        let second = queue.pop().unwrap();
        assert_eq!(second.request.priority, PriorityTier::Silver);
        let third = queue.pop().unwrap();
        assert_eq!(third.request.priority, PriorityTier::Bronze);
    }
}
