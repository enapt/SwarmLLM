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
        tools: None,
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
        tools: None,
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
    // Batching is ON by default, measured at about 40% more aggregate
    // throughput under concurrency with no cost to a single request. The
    // subject here is that the value reaches the router at all, not what it
    // happens to be — see `InferenceConfig::max_batch_size`.
    let config = crate::config::Config::default();
    assert_eq!(config.inference.max_batch_size, 8);
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

/// The error single-peer delegation actually produces when its one peer goes
/// away. Both delegation shapes are assembled with no standby by design, on the
/// stated reasoning that the retry re-routes and "the request can always come
/// home" — and the error a lost peer raises there, `SegmentFailoverExhausted`,
/// was on neither of the retry gate's two lists (gotcha #456).
#[test]
fn a_segment_that_ran_out_of_machines_is_retryable() {
    use crate::error::SwarmError;
    let err = SwarmError::SegmentFailoverExhausted(
        "Segment 0 failed with no standby available: Peer departed: its connection closed and \
         it could not be reached again"
            .into(),
    );
    assert!(super::segment_ran_out_of_machines(&err));

    // Deliberately kept out of the two existing predicates. It is gated by the
    // caller on a remote segment having been involved, exactly like
    // `remote_peer_could_not_serve` — a purely local pipeline that exhausted
    // its standbys has nothing new to route to.
    assert!(!super::is_transient_remote_failure(&err));
    assert!(!super::remote_peer_could_not_serve(&err));
}

/// The negative control: everything else keeps its old verdict, so the new arm
/// cannot be turning terminal failures into doubled work.
#[test]
fn ordinary_failures_do_not_look_like_an_exhausted_segment() {
    use crate::error::SwarmError;
    for err in [
        SwarmError::Internal("shape mismatch in rms-norm".into()),
        SwarmError::Validation("prompt is 41000 tokens, the limit is 8192".into()),
        SwarmError::ModelNotAvailable(ModelId("llama-3.2-1b".into())),
        SwarmError::ServiceUnavailable("spawn worker: No such file or directory".into()),
    ] {
        assert!(
            !super::segment_ran_out_of_machines(&err),
            "{err} is not an exhausted segment"
        );
    }
}

/// The whole retry decision, as four terms rather than four reads of one long
/// condition.
#[test]
fn the_retry_gate_weighs_every_term() {
    use crate::error::SwarmError;
    let departed = SwarmError::SegmentFailoverExhausted(
        "Segment 0 failed with no standby available: Peer departed".into(),
    );

    // The case these fixes are about: a delegated peer vanished, a remote
    // segment was involved, nothing has been streamed, the client is waiting.
    assert!(super::should_retry_after(&departed, true, false, false));

    // A purely local pipeline that exhausted its standbys has nowhere new to
    // go — the same gating `remote_peer_could_not_serve` already carries.
    assert!(!super::should_retry_after(&departed, false, false, false));

    // The client has gone (gotcha #445).
    assert!(!super::should_retry_after(&departed, true, true, false));

    // Text has already reached the client. A retry restarts generation, so the
    // reader would watch the reply begin a second time.
    assert!(!super::should_retry_after(&departed, true, false, true));

    // The streamed guard binds every class, not just the new one — a silent
    // drop mid-reply is the same hazard.
    let silent = SwarmError::PeerUnresponsive(
        "remote-generate: peer never acknowledged request_id=x (silent drop or disconnect)".into(),
    );
    assert!(super::should_retry_after(&silent, true, false, false));
    assert!(!super::should_retry_after(&silent, true, false, true));

    // And a terminal failure is still terminal on every combination.
    let ours = SwarmError::Internal("shape mismatch in rms-norm".into());
    for remote in [true, false] {
        assert!(!super::should_retry_after(&ours, remote, false, false));
    }
}

/// A peer that sat on a segment for its whole deadline is retried on a fresh
/// plan — by TYPE, since the deadline's wording matches none of the prose the
/// transient classifier knows — and only when a remote segment was involved,
/// because the variant can come back reclassified off the wire.
#[test]
fn a_segment_deadline_is_retried_only_with_a_remote_segment_involved() {
    use crate::error::SwarmError;
    let silent = SwarmError::PeerUnresponsive(
        "Timed out waiting for segment result (296s, 28 layers)".into(),
    );
    // The wording alone never qualified it — that is the gap this closes.
    assert!(!super::is_transient_remote_failure(&silent));
    assert!(super::should_retry_after(&silent, true, false, false));
    assert!(
        !super::should_retry_after(&silent, false, false, false),
        "with no remote segment there is nothing different to route to"
    );
    // The streamed and cancelled guards bind it like every other class.
    assert!(!super::should_retry_after(&silent, true, false, true));
    assert!(!super::should_retry_after(&silent, true, true, false));
}

/// This node's own memory refusal is re-planned with no remote segment
/// involved — the one local failure that is.
///
/// Its `ServiceUnavailable` sibling deliberately is not: a dead worker or a
/// failed spawn re-plans to the identical route, so retrying it fails twice.
/// This one is different only because the re-plan is handed a fact it did not
/// have (`note_local_memory_refusal`), which is what stops it repeating itself.
#[test]
fn a_local_memory_refusal_is_replanned_and_its_siblings_are_not() {
    use crate::error::SwarmError;
    let out_of_memory = SwarmError::LocalMemoryUnavailable(
        "qwen2.5-14b needs about 10374 MB of memory but this node's budget allows 8890 MB".into(),
    );
    // No remote segment, nothing streamed, client still there: re-plan.
    assert!(super::should_retry_after(
        &out_of_memory,
        false,
        false,
        false
    ));

    // The control that matters. The same 503 shape from a dead worker is NOT
    // re-planned without a remote segment — retrying our own worker failure
    // just fails twice, which is why the memory case needed its own variant
    // rather than a widening of this one.
    let dead_worker = SwarmError::ServiceUnavailable("worker is dead".into());
    assert!(!super::should_retry_after(
        &dead_worker,
        false,
        false,
        false
    ));

    // And every other term still binds: a client that has gone, and a reply
    // that has already started, are not re-planned whatever the failure.
    assert!(!super::should_retry_after(
        &out_of_memory,
        false,
        true,
        false
    ));
    assert!(!super::should_retry_after(
        &out_of_memory,
        false,
        false,
        true
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

/// A request that generated real tokens and then lost its peer hands those
/// tokens to the caller rather than nothing.
///
/// Report #028: a 4m43s reply on a 14B, already decoding, was discarded
/// outright when the tail peer's connection dropped. The retry is still tried
/// first — a complete answer beats a truncated one — but when nothing else can
/// serve the request, the work that was done is strictly better than a 503.
#[test]
fn a_reply_generated_before_the_failure_reaches_the_caller() {
    use crate::error::SwarmError;

    let (state, _tmp) = make_test_shared_state(crate::config::Config::default());
    let id = uuid::Uuid::new_v4();

    state.note_salvaged_reply(id, salvage(id, "the first half of an answer"));

    let out = super::salvaged_reply_if_lost(
        &state,
        id,
        Err(SwarmError::SegmentFailoverExhausted(
            "tail peer gone".into(),
        )),
    );

    let out = out.expect("a failure with something salvaged returns the salvage");
    assert_eq!(out.content, "the first half of an answer");
    assert_eq!(
        out.finish_reason,
        crate::inference::FINISH_REASON_INTERRUPTED,
        "the caller must be able to tell this reply is unfinished"
    );

    // Taken, not copied — a second delivery would double-report the request.
    assert!(state.take_salvaged_reply(id).is_none());
}

/// The control: a failure with nothing generated keeps its error.
///
/// This is the half that makes the change safe. The error carries the class,
/// the hint and the peer attribution; replacing it with an empty `200` would be
/// gotcha #433's lie pointing the other way — a failure dressed as a reply.
#[test]
fn a_failure_with_nothing_generated_keeps_its_error() {
    use crate::error::SwarmError;

    let (state, _tmp) = make_test_shared_state(crate::config::Config::default());
    let id = uuid::Uuid::new_v4();

    // Nothing recorded at all.
    let out = super::salvaged_reply_if_lost(
        &state,
        id,
        Err(SwarmError::SegmentFailoverExhausted("no standby".into())),
    );
    assert!(matches!(out, Err(SwarmError::SegmentFailoverExhausted(_))));

    // And an empty reply is refused at the recording end, so it can never
    // become a salvage later.
    state.note_salvaged_reply(id, salvage(id, ""));
    assert!(
        state.take_salvaged_reply(id).is_none(),
        "an empty salvage is not a salvage"
    );
}

/// A successful reply is never replaced, even when a salvage happens to exist.
///
/// The first attempt can fail and record a partial while the retry succeeds in
/// full; the complete answer must win.
#[test]
fn a_successful_reply_is_never_replaced_by_a_salvage() {
    let (state, _tmp) = make_test_shared_state(crate::config::Config::default());
    let id = uuid::Uuid::new_v4();

    state.note_salvaged_reply(id, salvage(id, "half an answer"));

    let mut complete = salvage(id, "the whole answer");
    complete.finish_reason = "stop".to_string();

    let out = super::salvaged_reply_if_lost(&state, id, Ok(complete))
        .expect("an Ok is returned untouched");
    assert_eq!(out.content, "the whole answer");
    assert_eq!(out.finish_reason, "stop");
}

/// Both attempts can salvage; the one that got further is the useful one.
#[test]
fn the_longer_of_two_salvaged_attempts_is_the_one_kept() {
    let (state, _tmp) = make_test_shared_state(crate::config::Config::default());
    let id = uuid::Uuid::new_v4();

    state.note_salvaged_reply(id, salvage(id, "twelve tokens in"));
    state.note_salvaged_reply(id, salvage(id, "four"));
    assert_eq!(
        state.take_salvaged_reply(id).unwrap().content,
        "twelve tokens in",
        "a shorter second attempt must not overwrite a longer first one"
    );

    // ...and in the other order, so this is about length rather than arrival.
    state.note_salvaged_reply(id, salvage(id, "four"));
    state.note_salvaged_reply(id, salvage(id, "twelve tokens in"));
    assert_eq!(
        state.take_salvaged_reply(id).unwrap().content,
        "twelve tokens in"
    );
}

/// A salvage is per-request state and is released with the rest of it.
///
/// Left behind, it would be handed to nobody and held for the daemon's life —
/// the leak `release_request_state` exists to prevent.
#[test]
fn a_salvage_is_released_with_the_rest_of_the_requests_state() {
    let (state, _tmp) = make_test_shared_state(crate::config::Config::default());
    let id = uuid::Uuid::new_v4();

    state.note_salvaged_reply(id, salvage(id, "something"));
    state.release_request_state(&id);
    assert!(state.take_salvaged_reply(id).is_none());
}

/// Which failures may keep their partial reply at all.
#[test]
fn only_an_unstreamed_reply_with_tokens_may_be_salvaged() {
    use crate::inference::pipeline::distributed::may_salvage;

    assert!(
        may_salvage(false, &[1, 2, 3]),
        "the reported case: not streamed, tokens generated, all of it discarded"
    );
    assert!(
        !may_salvage(false, &[]),
        "nothing generated — the error is the better answer"
    );
    assert!(
        !may_salvage(true, &[1, 2, 3]),
        "a streamed reply already reached the client, and turning this into an \
         Ok would make the OpenAI encoder re-emit the whole reply (gotcha #414)"
    );
    assert!(!may_salvage(true, &[]));
}

fn salvage(request_id: uuid::Uuid, content: &str) -> super::InferenceOutput {
    super::InferenceOutput {
        request_id,
        content: content.to_string(),
        prompt_tokens: 7,
        completion_tokens: content.split_whitespace().count() as u32,
        finish_reason: crate::inference::FINISH_REASON_INTERRUPTED.to_string(),
        session_id: None,
        token_logprobs: Vec::new(),
        matched_stop_sequence: None,
        trace: None,
    }
}
