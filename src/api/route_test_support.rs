//! Test support for the route a streamed reply names in its body (#251 (3)):
//! an `AppState` with no daemon behind it, and a router that answers one
//! streamed request the way the real one does — it attaches a trace with a
//! route to the token sender it was handed, streams a reply, and returns the
//! final output carrying that trace's snapshot.

use std::sync::Arc;

use crate::api::server::AppState;
use crate::inference::router::{InferenceOutput, RouterCommand, StreamingTokenEvent};
use crate::inference::trace::{local_segment, RequestTrace, Route};

/// Everything an `AppState` needs, kept alive for the test's length.
pub(crate) struct TestApp {
    pub(crate) state: AppState,
    _dir: tempfile::TempDir,
    _rx: Box<dyn std::any::Any + Send>,
}

pub(crate) fn test_app() -> TestApp {
    let dir = tempfile::tempdir().unwrap();
    let db = crate::storage::db::Database::open(dir.path()).unwrap();
    let executor = Arc::new(tokio::sync::Mutex::new(
        crate::inference::executor::ModelExecutor::new(),
    ));
    let (shared, a, b) = crate::daemon::SharedState::new(
        crate::config::Config::default(),
        crate::identity::Identity::generate(),
        db.clone(),
        executor.clone(),
        None,
    );
    let state = AppState {
        config: crate::config::Config::default(),
        db,
        executor,
        router_tx: None,
        acquisition_tx: None,
        network_tx: None,
        shared_state: shared,
        rate_limiter: crate::api::middleware::RateLimiter::new(60, 200),
        bootstrap_nonces: Arc::new(dashmap::DashMap::new()),
    };
    TestApp {
        state,
        _dir: dir,
        _rx: Box::new((a, b)),
    }
}

/// The route the fake router serves on: this computer's first half and a
/// peer's second — two segments, one peer.
pub(crate) fn two_computer_trace(request_id: uuid::Uuid) -> Arc<RequestTrace> {
    let trace = Arc::new(RequestTrace::new(request_id, "m", "chat"));
    let mut segments = local_segment(&crate::types::NodeId([1u8; 32]), (0, 11));
    let mut far = local_segment(&crate::types::NodeId([9u8; 32]), (11, 22)).remove(0);
    far.index = 1;
    far.is_local = false;
    segments.push(far);
    trace.mark_assembled(Route::Distributed, segments, 0);
    trace
}

/// A router that answers ONE streamed request: `reply` as tokens, then a
/// finish, on [`two_computer_trace`]'s route.
pub(crate) fn fake_router(reply: &'static str) -> tokio::sync::mpsc::Sender<RouterCommand> {
    let (router_tx, mut router_rx) = tokio::sync::mpsc::channel::<RouterCommand>(4);
    tokio::spawn(async move {
        let Some(RouterCommand::StreamSubmit {
            request,
            result_tx,
            token_tx,
        }) = router_rx.recv().await
        else {
            return;
        };
        let trace = two_computer_trace(request.id);
        // What the router does at admission (`router::mod`): the trace rides
        // the sender, and nothing else tells the API code holding the receiver.
        let token_tx = token_tx.with_trace(trace.clone());
        let _ = token_tx
            .send(StreamingTokenEvent {
                text: reply.to_string(),
                finish_reason: None,
                matched_stop_sequence: None,
            })
            .await;
        let _ = token_tx
            .send(StreamingTokenEvent {
                text: String::new(),
                finish_reason: Some("stop".into()),
                matched_stop_sequence: None,
            })
            .await;
        let _ = result_tx.send(Ok(InferenceOutput {
            request_id: request.id,
            content: reply.to_string(),
            reasoning: None,
            prompt_tokens: 3,
            completion_tokens: 1,
            finish_reason: "stop".into(),
            session_id: None,
            token_logprobs: Vec::new(),
            matched_stop_sequence: None,
            trace: Some(trace.snapshot()),
        }));
    });
    router_tx
}

/// The `data:` payloads of an SSE body, read to its end.
pub(crate) async fn sse_payloads(response: axum::response::Response) -> Vec<String> {
    let bytes = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        axum::body::to_bytes(response.into_body(), usize::MAX),
    )
    .await
    .expect("the stream must end")
    .expect("readable body");
    String::from_utf8_lossy(&bytes)
        .lines()
        .filter_map(|l| l.strip_prefix("data: ").or_else(|| l.strip_prefix("data:")))
        .map(str::to_string)
        .collect()
}

/// A router that answers ONE request that is not streamed, on
/// [`two_computer_trace`]'s route — the final output carries its snapshot, as
/// the router's completion arm attaches it.
pub(crate) fn fake_unstreamed_router(
    reply: &'static str,
) -> tokio::sync::mpsc::Sender<RouterCommand> {
    let (router_tx, mut router_rx) = tokio::sync::mpsc::channel::<RouterCommand>(4);
    tokio::spawn(async move {
        let Some(RouterCommand::Submit { request, result_tx }) = router_rx.recv().await else {
            return;
        };
        let trace = two_computer_trace(request.id);
        let _ = result_tx.send(Ok(InferenceOutput {
            request_id: request.id,
            content: reply.to_string(),
            reasoning: None,
            prompt_tokens: 3,
            completion_tokens: 1,
            finish_reason: "stop".into(),
            session_id: None,
            token_logprobs: Vec::new(),
            matched_stop_sequence: None,
            trace: Some(trace.snapshot()),
        }));
    });
    router_tx
}
