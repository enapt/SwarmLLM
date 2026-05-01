//! Batch-dispatch entrypoint: routes a collected batch to either the
//! local-model path or the distributed pipeline path.

use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::SharedState;
use crate::inference::scheduler::PipelineScheduler;
use crate::types::NetworkCommand;

use super::distributed_exec::execute_distributed_batch;
use super::local_exec::execute_local_batch;
use super::types::QueuedRequest;

/// Execute a batch of requests that target the same model.
///
/// For local inference (full model loaded), the batch shares a single executor
/// lock acquisition — requests are processed sequentially within the lock,
/// avoiding repeated lock acquire/release overhead.
///
/// For distributed inference, each request gets its own pipeline and they
/// execute concurrently.
pub(super) async fn execute_batch(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    scheduler: PipelineScheduler,
    batch: Vec<QueuedRequest>,
    active_count: Arc<AtomicUsize>,
    queue_notify: Arc<tokio::sync::Notify>,
) {
    let batch_size = batch.len();
    let is_split_mode = shared_state.config.inference.shard_range.is_some();
    let model_loaded = shared_state
        .model_loaded
        .load(std::sync::atomic::Ordering::Acquire);

    if model_loaded && !is_split_mode {
        // Local inference batch: hold the executor lock once, process all requests
        execute_local_batch(shared_state, batch, active_count, queue_notify).await;
    } else {
        // Distributed inference batch: spawn each request independently
        // They'll assemble their own pipelines and run concurrently.
        execute_distributed_batch(
            shared_state,
            network_tx,
            scheduler,
            batch,
            active_count,
            queue_notify,
        )
        .await;
    }

    tracing::debug!(batch_size, "Batch execution complete");
}
