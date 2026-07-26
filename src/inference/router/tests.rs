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
