use std::collections::BinaryHeap;

use tokio::sync::{mpsc, oneshot, watch};

use super::types::{QueuedRequest, RouterCommand};
use super::InferenceRouter;
use crate::types::{ChatMessage, InferenceRequest, ModelId, PriorityTier, Role, SamplingParams};

fn make_test_shared_state(
    config: crate::config::Config,
) -> (
    std::sync::Arc<crate::daemon::SharedState>,
    tempfile::TempDir,
) {
    use crate::daemon::SharedState;
    use crate::identity::Identity;
    use crate::inference::executor::ModelExecutor;
    use crate::storage::db::Database;
    use std::sync::Arc;
    use tokio::sync::Mutex;

    let identity = Identity::generate();
    let temp = tempfile::tempdir().unwrap();
    let db = Database::open(temp.path()).unwrap();
    let executor = Arc::new(Mutex::new(ModelExecutor::new()));
    let (shared_state, _, _) = SharedState::new(config, identity, db, executor, None);
    (shared_state, temp)
}

fn make_test_router(
    config: crate::config::Config,
) -> (
    InferenceRouter,
    mpsc::Sender<RouterCommand>,
    tempfile::TempDir,
) {
    let (shared_state, temp) = make_test_shared_state(config);
    let (cmd_tx, cmd_rx) = mpsc::channel(64);
    let (net_tx, _net_rx) = mpsc::channel(64);
    let (_shutdown_tx, shutdown_rx) = watch::channel(false);
    let router = InferenceRouter::new(shared_state, cmd_rx, cmd_tx.clone(), net_tx, shutdown_rx);
    (router, cmd_tx, temp)
}

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
        cancel: None,
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
        cancel: None,
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
        trace: std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
            uuid::Uuid::new_v4(),
            "test-model",
            "chat",
        )),
    });
    queue.push(QueuedRequest {
        request: make_request(PriorityTier::Platinum),
        result_tx: tx_b,
        token_tx: None,
        trace: std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
            uuid::Uuid::new_v4(),
            "test-model",
            "chat",
        )),
    });
    queue.push(QueuedRequest {
        request: make_request(PriorityTier::Silver),
        result_tx: tx_c,
        token_tx: None,
        trace: std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
            uuid::Uuid::new_v4(),
            "test-model",
            "chat",
        )),
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
    let mut config = crate::config::Config::default();
    config.inference.max_batch_size = 4;
    let (mut router, _cmd_tx, _temp) = make_test_router(config);

    // Add 3 requests for model "alpha", 2 for model "beta"
    for _ in 0..3 {
        let (tx, _) = oneshot::channel();
        router.queue.push(QueuedRequest {
            request: make_request_with_model(PriorityTier::Silver, "alpha"),
            result_tx: tx,
            token_tx: None,
            trace: std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
                uuid::Uuid::new_v4(),
                "test-model",
                "chat",
            )),
        });
    }
    for _ in 0..2 {
        let (tx, _) = oneshot::channel();
        router.queue.push(QueuedRequest {
            request: make_request_with_model(PriorityTier::Silver, "beta"),
            result_tx: tx,
            token_tx: None,
            trace: std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
                uuid::Uuid::new_v4(),
                "test-model",
                "chat",
            )),
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
    let config = crate::config::Config::default(); // max_batch_size = 1
    let (mut router, _cmd_tx, _temp) = make_test_router(config);

    // Add 3 requests
    for _ in 0..3 {
        let (tx, _) = oneshot::channel();
        router.queue.push(QueuedRequest {
            request: make_request(PriorityTier::Silver),
            result_tx: tx,
            token_tx: None,
            trace: std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
                uuid::Uuid::new_v4(),
                "test-model",
                "chat",
            )),
        });
    }

    // With max_batch_size=1, should only get 1
    let batch = router.collect_batch(1);
    assert_eq!(batch.len(), 1);
    assert_eq!(router.queue.len(), 2);
}

#[test]
fn collect_batch_respects_max_size() {
    let mut config = crate::config::Config::default();
    config.inference.max_batch_size = 2;
    let (mut router, _cmd_tx, _temp) = make_test_router(config);

    // Add 5 requests all same model
    for _ in 0..5 {
        let (tx, _) = oneshot::channel();
        router.queue.push(QueuedRequest {
            request: make_request(PriorityTier::Silver),
            result_tx: tx,
            token_tx: None,
            trace: std::sync::Arc::new(crate::inference::trace::RequestTrace::new(
                uuid::Uuid::new_v4(),
                "test-model",
                "chat",
            )),
        });
    }

    // With max_batch_size=2, should only get 2
    let batch = router.collect_batch(2);
    assert_eq!(batch.len(), 2);
    assert_eq!(router.queue.len(), 3);
}

#[test]
fn collect_batch_empty_queue() {
    let (mut router, _cmd_tx, _temp) = make_test_router(crate::config::Config::default());

    let batch = router.collect_batch(4);
    assert!(batch.is_empty());
}

#[test]
fn default_batch_config() {
    let config = crate::config::Config::default();
    assert_eq!(config.inference.max_batch_size, 1);
    assert_eq!(config.inference.batch_timeout_ms, 50);
}

// --- retry classification -------------------------------------------------
//
// A peer whose worker is broken reports `ServiceUnavailable`. That must be
// retryable against a different holder, but ONLY when a remote segment was
// actually involved — the identical wording from our own worker is terminal.

#[test]
fn peer_service_unavailable_is_retryable() {
    use crate::error::SwarmError;
    // Observed live from a third-party node: a worker binary that could not be
    // spawned. `assemblies=1` — the request failed with no second attempt.
    let err = SwarmError::ServiceUnavailable("spawn worker: No such file or directory".into());
    assert!(super::remote_peer_could_not_serve(&err));

    // Also matches when the peer's message arrives already stringified through
    // the network layer rather than as a typed variant.
    let relayed =
        SwarmError::Inference("Service unavailable: worker closed connection mid-generate".into());
    assert!(super::remote_peer_could_not_serve(&relayed));
}

#[test]
fn ordinary_failures_are_not_treated_as_peer_unavailable() {
    use crate::error::SwarmError;
    // Our own bug: retrying cannot help, and charging a peer would be wrong.
    assert!(!super::remote_peer_could_not_serve(&SwarmError::Internal(
        "shape mismatch in rms-norm".into()
    )));
    // Bad input: identical on every retry.
    assert!(!super::remote_peer_could_not_serve(
        &SwarmError::Validation("max_tokens must be positive".into())
    ));
    assert!(!super::remote_peer_could_not_serve(
        &SwarmError::ModelNotAvailable(ModelId("llama-3.2-1b".into()))
    ));
}

#[test]
fn peer_unavailable_is_kept_out_of_the_transient_classifier() {
    use crate::error::SwarmError;
    // `is_transient_remote_failure` is consulted without knowing whether the
    // attempt used a remote segment, so it must NOT match on wording our own
    // worker also produces. The remote-only case is gated separately by the
    // caller on `trace.remote_segments() > 0`.
    let err = SwarmError::ServiceUnavailable("spawn worker: No such file or directory".into());
    assert!(!super::is_transient_remote_failure(&err));

    // The genuinely remote-only signals stay matched.
    assert!(super::is_transient_remote_failure(&SwarmError::Inference(
        "peer never acknowledged the request".into()
    )));

    // The typed variant the remote-generate fast path raises for a silent
    // peer must keep its single re-routed retry — reclassifying it away from
    // `PipelineError` (for the 503) must not cost the retry.
    assert!(super::is_transient_remote_failure(
        &SwarmError::PeerUnresponsive(
            "remote-generate: peer never acknowledged request_id=x (silent drop or disconnect)"
                .into()
        )
    ));
    assert!(super::is_transient_remote_failure(
        &SwarmError::PeerUnresponsive(
            "remote-generate timed out waiting for token (first=true)".into()
        )
    ));
}

/// The coordinator matches a peer's failure as TEXT off the wire, while the
/// retry decision matches a typed error. Both go through one predicate, so a
/// message that triggers the retry must also bar the peer from it — otherwise
/// the retry re-picks the node that just failed, which is what happened live
/// (`assemblies=2`, same node id both times).
#[test]
fn the_retry_and_the_blacklist_agree_on_what_counts() {
    use crate::error::SwarmError;
    let wire_message = "Service unavailable: worker closed connection mid-generate";
    assert!(super::message_means_peer_cannot_serve(wire_message));
    assert!(super::remote_peer_could_not_serve(&SwarmError::Inference(
        wire_message.to_string()
    )));

    // A peer failing for a reason of its own is not barred: retrying elsewhere
    // would not have helped and the node is still good for later segments.
    assert!(!super::message_means_peer_cannot_serve(
        "Validation error: max_tokens must be positive"
    ));
}

/// A wallet that could not be READ is not a wallet that is EMPTY.
///
/// `credit_balance` is a writer-fair `RwLock`, so `try_read` fails whenever a
/// writer is merely queued — an inbound credit transaction, a penalty, an escrow
/// expiry. The router used to substitute `0` for that, which is the single most
/// damaging value the wallet could hold: the request was refused and the caller
/// was told their balance was too low, from evidence that only said a lock was
/// busy for an instant. The spec line above the read has always said credit
/// errors degrade the tier and never block.
#[test]
fn an_unreadable_balance_never_refuses_the_request() {
    use super::refuse_for_insufficient_credit;

    // The case that regressed: floor active, balance unknown.
    assert!(
        !refuse_for_insufficient_credit(false, None, 100),
        "an unread balance must fall through, not be counted as below the floor"
    );

    // A balance we actually read, and which is actually below the floor, still
    // refuses — the fix must not disable the check it is protecting.
    assert!(refuse_for_insufficient_credit(false, Some(99), 100));
    assert!(!refuse_for_insufficient_credit(false, Some(100), 100));

    // Local requests are never refused, read or not.
    assert!(!refuse_for_insufficient_credit(true, Some(-5000), 100));
    assert!(!refuse_for_insufficient_credit(true, None, 100));

    // Credits are dormant: a zero floor gates nothing, whatever the balance.
    assert!(!refuse_for_insufficient_credit(false, Some(-5000), 0));
    assert!(!refuse_for_insufficient_credit(false, None, 0));
}
