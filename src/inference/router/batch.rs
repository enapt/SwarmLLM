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
    // Increment the tier-cap counter INSIDE this spawned task — pairs
    // with the fetch_sub side that lives inside BatchCleanup (local) and
    // the per-request completion arms (distributed). The same R103/R138
    // hygiene as `dispatch_single`: if `tokio::spawn` of this task
    // failed or the task was dropped before running, the counter would
    // leak `batch_size` slots permanently. Performing the add here
    // guarantees the spawn succeeded; the sub paths (BatchCleanup Drop +
    // explicit fetch_sub arms in distributed) handle every exit.
    active_count.fetch_add(batch_size, std::sync::atomic::Ordering::Relaxed);
    let is_split_mode = shared_state.config.inference.shard_range.is_some();
    // Every request in a batch targets the same model (see `collect_batch`),
    // so the first one settles it for all of them.
    //
    // This asks whether the local executor holds THE REQUESTED model, not the
    // bare `model_loaded` flag. The flag is global — "a model is loaded" — and
    // `execute_local_batch` never looks at `request.model_id`, so dispatching
    // on it alone answered requests for other models with whichever model
    // happened to be resident. See `SharedState::local_executor_serves`.
    let serves_locally = match batch.first() {
        Some(q) => {
            shared_state
                .local_executor_serves(&q.request.model_id)
                .await
        }
        None => false,
    };

    if serves_locally && !is_split_mode {
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
