//! V5+V8 (responses_api_v2): resumable SSE + background streaming.
//!
//! The OpenAI Responses API lets a caller run an inference with
//! `background=true` and `stream=true`, then disconnect and resume the
//! stream later by passing `?starting_after={seq}` on a GET. v1 rejected
//! that combination; V8 ships it.
//!
//! Design:
//!
//! - Each in-flight background response has a `BackgroundState` entry
//!   keyed by response id. State holds a cancel flag, a bounded event
//!   buffer, a completion flag, and a `tokio::sync::Notify` that fires
//!   whenever a new event lands in the buffer.
//! - A POST with `background=true&stream=true` returns `202 Accepted`
//!   plus a `Location` header pointing at the GET-with-stream resume
//!   path. The server immediately starts running the inference
//!   internally via a spawned task that writes every SSE event into
//!   the buffer (instead of a live client socket).
//! - A GET with `stream=true&starting_after={seq}` replays every
//!   cached event whose sequence_number is strictly greater than
//!   `seq`, then subscribes to the notifier and live-tails as new
//!   events land. When the state is marked completed the stream closes.
//! - Buffer cap is 2000 events; when exceeded we drop the oldest. A
//!   client resuming with a cursor that points before the dropped
//!   window sees only the still-buffered tail (sequence numbers remain
//!   monotonic, so no duplicates — just a gap if they fell too far
//!   behind).

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use dashmap::DashMap;
use serde::Deserialize;
use tokio::sync::{Mutex, Notify};

use super::stream::{self as responses_stream, SSE_KEEPALIVE_INTERVAL_SECS};
use super::types::*;
use super::{store, BACKGROUND_CANCEL};
use crate::api::server::AppState;
use crate::error::{ApiError, SwarmError};

/// Hard cap on the in-memory event buffer for a single background
/// response. 2000 events comfortably covers a multi-thousand-token
/// generation (one event per token plus lifecycle events).
pub(crate) const EVENT_BUFFER_CAP: usize = 2000;

/// A single buffered SSE event, stored in replayable form. We can't
/// store the axum `Event` directly (it's not `Clone`), so we hold the
/// event name + JSON body and reconstruct the Event on replay.
#[derive(Clone)]
pub struct BufferedEvent {
    pub sequence_number: u64,
    pub event_name: String,
    pub data: serde_json::Value,
}

impl BufferedEvent {
    pub(super) fn to_event(&self) -> Event {
        Event::default()
            .event(&self.event_name)
            .data(serde_json::to_string(&self.data).unwrap_or_default())
    }
}

/// Per-response background state. Shared between the running inference
/// task (which pushes events) and any number of resuming GET handlers
/// (which replay + live-tail). Completion is signaled both via the
/// `completed` flag and by dropping the notifier's final wake.
pub(crate) struct BackgroundState {
    pub cancel: Arc<AtomicBool>,
    pub events: Mutex<std::collections::VecDeque<BufferedEvent>>,
    pub completed: AtomicBool,
    pub notify: Notify,
}

impl BackgroundState {
    fn new(cancel: Arc<AtomicBool>) -> Self {
        Self {
            cancel,
            events: Mutex::new(std::collections::VecDeque::new()),
            completed: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    /// Push a new event into the buffer and wake any waiting resumers.
    /// The buffer is capped at EVENT_BUFFER_CAP; when full, the oldest
    /// events are dropped first.
    async fn push(&self, event: BufferedEvent) {
        {
            let mut events = self.events.lock().await;
            if events.len() >= EVENT_BUFFER_CAP {
                // Drop the oldest event. VecDeque's pop_front is O(1) so
                // this stays cheap even at the cap (2000).
                events.pop_front();
            }
            events.push_back(event);
            // Drop the guard before notifying so woken resumers don't
            // immediately contend on the same lock.
        }
        self.notify.notify_waiters();
    }

    async fn mark_completed(&self) {
        self.completed.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    async fn events_after(&self, after: i64) -> Vec<BufferedEvent> {
        let events = self.events.lock().await;
        events
            .iter()
            .filter(|e| (e.sequence_number as i64) > after)
            .cloned()
            .collect()
    }
}

/// Registry of in-flight background streaming responses. Entries are
/// removed when the background task finishes.
pub(crate) static BACKGROUND_STATE: std::sync::LazyLock<DashMap<String, Arc<BackgroundState>>> =
    std::sync::LazyLock::new(DashMap::new);

/// Register a new background streaming response. Returns the shared
/// state. The cancel flag is also inserted into the legacy
/// `BACKGROUND_CANCEL` map so POST /cancel still works uniformly.
pub(crate) fn register_background_stream(
    response_id: &str,
    cancel: Arc<AtomicBool>,
) -> Arc<BackgroundState> {
    let state = Arc::new(BackgroundState::new(cancel.clone()));
    BACKGROUND_STATE.insert(response_id.to_string(), state.clone());
    // Mirror the cancel flag into the legacy map so the existing
    // cancel_response handler doesn't need a second lookup path.
    BACKGROUND_CANCEL.insert(response_id.to_string(), cancel);
    state
}

/// Deregister background state for a response id. Called when the
/// inference task is finishing — marks completed first so any
/// currently-waiting GETs see the final state before the entry is
/// removed.
pub(crate) async fn deregister_background_stream(response_id: &str) {
    if let Some((_, state)) = BACKGROUND_STATE.remove(response_id) {
        state.mark_completed().await;
    }
    BACKGROUND_CANCEL.remove(response_id);
}

/// Lookup helper used by the resume GET.
pub(crate) fn lookup_background_state(response_id: &str) -> Option<Arc<BackgroundState>> {
    BACKGROUND_STATE.get(response_id).map(|e| e.value().clone())
}

// ============================================================================
// Entry point from create_response: background=true && stream=true
// ============================================================================

/// Run the background-streaming path. The POST handler returns 202
/// Accepted with a Location header immediately; inference runs in a
/// spawned task that writes each SSE event into the shared buffer.
pub async fn start_background_stream(
    state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
    prior: Option<store::ResponsesRecord>,
) -> Result<Response, ApiError> {
    // Pre-flight: translate request synchronously so a 400 is a 400 and
    // not a 202 + empty buffer. Matches V1's discipline.
    let mut chat_req = super::translate::request_to_chat(&req, prior.as_ref())?;
    chat_req.stream = true;

    let response_id = format!("resp_{}", uuid::Uuid::new_v4().simple());
    let created_at = chrono::Utc::now().timestamp();

    // Seed redb with a queued placeholder so a GET without stream=true
    // (the polling form) also returns meaningful state.
    let queued = build_queued_response(&req, &response_id, created_at);
    let record =
        store::ResponsesRecord::new(req.clone(), queued, created_at, store::DEFAULT_TTL_SECS);
    store::store(&state.db, &record).map_err(ApiError)?;

    let cancel = Arc::new(AtomicBool::new(false));
    let bg_state = register_background_stream(&response_id, cancel);

    // Spawn the inference driver. It consumes the SSE generator from
    // stream::build_response_event_stream's output, pushing each event
    // into the buffer instead of yielding to a socket.
    let state_for_task = state.clone();
    let headers_for_task = headers.clone();
    let req_for_task = req.clone();
    let id_for_task = response_id.clone();
    let id_for_cleanup = response_id.clone();
    let bg_state_for_task = bg_state.clone();
    let bg_state_for_cleanup = bg_state.clone();
    let chat_req_for_task = chat_req;
    tokio::spawn(async move {
        // Wrap in catch_unwind so a panic inside drive_background_stream
        // (e.g., during run_streaming_buffered, or any inference panic)
        // doesn't leak the BACKGROUND_STATE / BACKGROUND_CANCEL entries
        // forever. Without this guard, a panicked task would leave the
        // entry with completed=false, and any waiting GET resume client
        // would receive 15s SSE keepalives indefinitely until disconnect.
        use futures::FutureExt;
        let outcome = std::panic::AssertUnwindSafe(drive_background_stream(
            state_for_task,
            headers_for_task,
            req_for_task,
            id_for_task,
            created_at,
            chat_req_for_task,
            bg_state_for_task,
        ))
        .catch_unwind()
        .await;
        if outcome.is_err() {
            tracing::error!(
                response_id = %id_for_cleanup,
                "background streaming task panicked — marking completed and cleaning up"
            );
            // Push a cancelled-shaped terminal event so resumers see a
            // close, then mark + deregister.
            let seq_marker = u64::MAX; // will sort last; resumers see it after any cached events
            bg_state_for_cleanup
                .push(BufferedEvent {
                    sequence_number: seq_marker,
                    event_name: "response.failed".into(),
                    data: serde_json::json!({
                        "type": "response.failed",
                        "sequence_number": seq_marker,
                        "response": {
                            "id": id_for_cleanup,
                            "status": "failed",
                            "error": {
                                "code": "internal_error",
                                "message": "background task panicked"
                            }
                        }
                    }),
                })
                .await;
            deregister_background_stream(&id_for_cleanup).await;
        }
    });

    // Build a 202 Accepted response with a Location header so the
    // caller knows where to GET and resume.
    let location = format!(
        "/v1/responses/{}?stream=true&starting_after=-1",
        response_id
    );
    let mut resp = (
        StatusCode::ACCEPTED,
        Json(serde_json::json!({
            "id": response_id,
            "object": "response",
            "status": "queued",
            "model": req.model,
            "created_at": created_at,
            "background": true,
            "stream": true,
        })),
    )
        .into_response();
    if let Ok(val) = HeaderValue::from_str(&location) {
        resp.headers_mut().insert(header::LOCATION, val);
    }
    Ok(resp)
}

/// Drive the inference stream internally, pushing each SSE event into
/// the buffer. Finalization writes the completed record to redb via the
/// stream generator itself (store_db is passed in).
async fn drive_background_stream(
    app_state: AppState,
    headers: axum::http::HeaderMap,
    req: ResponsesRequest,
    response_id: String,
    created_at: i64,
    chat_req: crate::api::openai::types::ChatCompletionRequest,
    bg_state: Arc<BackgroundState>,
) {
    use futures::StreamExt;

    let state_for_chat = app_state.clone();
    let headers_for_chat = headers.clone();
    let chat_future = async move {
        crate::api::openai::chat_completions(
            State(state_for_chat),
            headers_for_chat,
            crate::api::server::JsonBody(chat_req),
        )
        .await
    };

    // Reuse the local streaming event generator. We'll iterate it and
    // bucket each event into the buffer instead of yielding to a
    // client.
    let store_db = req.store.unwrap_or(true).then(|| app_state.db.clone());
    let item_id = format!("msg_{}", uuid::Uuid::new_v4().simple());

    // Drive via a helper that exposes a (name, data, seq) triple per
    // event. We don't have that on the generator directly — it only
    // exposes axum Events. As a compromise, we emit Events and shim
    // them into buffered form by parsing the event back out.
    // Simpler: call the raw generator through an adapter that yields
    // (name, value, seq). To avoid duplicating the generator, we
    // instead fork the existing generator's output into a wrapper.
    let stream = responses_stream::run_streaming_buffered(
        chat_future,
        req.clone(),
        response_id.clone(),
        item_id,
        created_at,
        store_db,
    );

    let mut stream = Box::pin(stream);
    while let Some(buffered) = stream.next().await {
        // Also check the cancel flag here; if tripped, break.
        if bg_state.cancel.load(Ordering::SeqCst) {
            // Record a cancelled final event so resumers see a terminal.
            let seq = buffered.sequence_number;
            bg_state
                .push(BufferedEvent {
                    sequence_number: seq,
                    event_name: "response.cancelled".into(),
                    data: serde_json::json!({
                        "type": "response.cancelled",
                        "sequence_number": seq,
                        "response": {
                            "id": response_id,
                            "status": "cancelled",
                        }
                    }),
                })
                .await;
            break;
        }
        bg_state.push(buffered).await;
    }

    deregister_background_stream(&response_id).await;
}

// ============================================================================
// Resume GET: /v1/responses/:id?stream=true&starting_after={seq}
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct GetResponseParams {
    #[serde(default)]
    pub stream: Option<bool>,
    /// Cursor: replay buffered events with sequence_number strictly
    /// greater than this value. `-1` (or unset) means "from the start".
    #[serde(default)]
    pub starting_after: Option<i64>,
}

/// Unified GET handler. Non-streaming GET (no `stream=true` param)
/// falls back to the normal retrieve behavior; a streaming resume GET
/// replays the buffered events + live-tails.
pub async fn get_response_maybe_stream(
    State(app_state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Query(params): Query<GetResponseParams>,
) -> Result<Response, ApiError> {
    crate::api::openai::responses::validate_response_id(&id)?;
    let wants_stream = params.stream.unwrap_or(false);
    if !wants_stream {
        // Original get_response behavior.
        return crate::api::openai::responses::get_response(
            State(app_state),
            axum::extract::Path(id),
        )
        .await;
    }

    let after = params.starting_after.unwrap_or(-1);
    let bg = lookup_background_state(&id);
    match bg {
        Some(state) => Ok(serve_resume_stream(state, after)),
        None => {
            // No live background task — the response may have completed
            // already, or it never existed. If the record exists,
            // synthesize a replay-from-completed stream from the
            // stored record so clients that reconnect after completion
            // still see a clean SSE close.
            match store::load(&app_state.db, &id).map_err(ApiError)? {
                Some(record) => Ok(serve_completed_replay(record, after)),
                None => Err(ApiError(SwarmError::Validation(format!(
                    "Response `{id}` not found or expired."
                )))),
            }
        }
    }
}

/// Maximum time a resumer will wait between events before giving up.
/// Caps the worst-case "stuck on a stale state entry" hang in case the
/// background task panics before its catch_unwind cleanup runs (or any
/// future bug skips cleanup). 1 hour is generous for the longest plausible
/// inference but still guarantees the connection won't be held forever.
const RESUME_STREAM_MAX_IDLE_SECS: u64 = 3600;

/// Live-tail resume: replay cached events after `after`, then wait for
/// new events until completion.
fn serve_resume_stream(state: Arc<BackgroundState>, after: i64) -> Response {
    let stream = async_stream::stream! {
        let mut last_seen = after;

        loop {
            // Register the notification future *before* checking the
            // event buffer. Tokio's Notify only wakes already-registered
            // waiters, so a producer call to notify_waiters() between
            // our events_after() check and our notified().await would
            // otherwise be a lost-wakeup, stalling this resumer for up
            // to RESUME_STREAM_MAX_IDLE_SECS. The standard idiom is:
            //   register → check → drain → await
            // — by the time we await, any push that happened after we
            // registered will fire the future immediately.
            let notified = state.notify.notified();
            tokio::pin!(notified);

            let pending = state.events_after(last_seen).await;
            for ev in pending {
                last_seen = ev.sequence_number as i64;
                yield Ok::<_, Infallible>(ev.to_event());
            }

            if state.completed.load(Ordering::SeqCst) {
                // Final drain in case events landed between our last
                // check and the completed flag being set.
                let final_pending = state.events_after(last_seen).await;
                for ev in final_pending {
                    yield Ok(ev.to_event());
                }
                break;
            }

            // Wait for new events with a hard idle cap so a stale state
            // entry can't keep this connection open forever.
            let wait = tokio::time::timeout(
                std::time::Duration::from_secs(RESUME_STREAM_MAX_IDLE_SECS),
                notified.as_mut(),
            )
            .await;
            if wait.is_err() {
                tracing::warn!(
                    idle_secs = RESUME_STREAM_MAX_IDLE_SECS,
                    "resume stream idle timeout — closing connection"
                );
                break;
            }
        }
    };

    Sse::new(stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response()
}

/// Completed-response replay: the response already finished, so we
/// reconstruct a minimal lifecycle sequence from the stored record so
/// the caller's SSE iterator still closes cleanly.
fn serve_completed_replay(record: store::ResponsesRecord, after: i64) -> Response {
    let replayed = build_completed_replay_events(&record.response, after);
    let stream = futures::stream::iter(
        replayed
            .into_iter()
            .map(|e| Ok::<_, Infallible>(e.to_event())),
    );

    Sse::new(stream)
        .keep_alive(
            KeepAlive::new().interval(std::time::Duration::from_secs(SSE_KEEPALIVE_INTERVAL_SECS)),
        )
        .into_response()
}

/// Pure helper: build the synthetic replay event sequence for a stored
/// response, filtered by the `after` cursor. Extracted for unit testing.
/// Minimal lifecycle: response.created, response.in_progress, and the
/// terminal event derived from status. Partial-state events (output
/// items, content parts) are omitted — the caller can read the full
/// completed response via the terminal event's `response` field rather
/// than have us fabricate mid-run state.
fn build_completed_replay_events(resp: &ResponsesResponse, after: i64) -> Vec<BufferedEvent> {
    let mut events: Vec<BufferedEvent> = Vec::new();
    let mut seq: u64 = 0;

    events.push(BufferedEvent {
        sequence_number: seq,
        event_name: "response.created".into(),
        data: serde_json::json!({
            "type": "response.created",
            "sequence_number": seq,
            "response": resp,
        }),
    });
    seq += 1;
    events.push(BufferedEvent {
        sequence_number: seq,
        event_name: "response.in_progress".into(),
        data: serde_json::json!({
            "type": "response.in_progress",
            "sequence_number": seq,
            "response": resp,
        }),
    });
    seq += 1;

    let terminal_name = match resp.status {
        ResponseStatus::Failed => "response.failed",
        ResponseStatus::Incomplete => "response.incomplete",
        ResponseStatus::Cancelled => "response.cancelled",
        _ => "response.completed",
    };
    events.push(BufferedEvent {
        sequence_number: seq,
        event_name: terminal_name.into(),
        data: serde_json::json!({
            "type": terminal_name,
            "sequence_number": seq,
            "response": resp,
        }),
    });

    events
        .into_iter()
        .filter(|e| (e.sequence_number as i64) > after)
        .collect()
}

// ============================================================================
// Queued placeholder builder
// ============================================================================

fn build_queued_response(
    req: &ResponsesRequest,
    response_id: &str,
    created_at: i64,
) -> ResponsesResponse {
    let mut q =
        super::build_response_skeleton(req, response_id, created_at, ResponseStatus::Queued);
    q.background = Some(true);
    q
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[tokio::test]
    async fn buffer_caps_at_limit_and_drops_oldest() {
        let state = BackgroundState::new(Arc::new(AtomicBool::new(false)));
        for i in 0..(EVENT_BUFFER_CAP + 50) as u64 {
            state
                .push(BufferedEvent {
                    sequence_number: i,
                    event_name: "x".into(),
                    data: serde_json::json!({}),
                })
                .await;
        }
        let events = state.events.lock().await;
        assert_eq!(events.len(), EVENT_BUFFER_CAP);
        // Oldest should be 50 (we dropped 0..50).
        assert_eq!(events.front().unwrap().sequence_number, 50);
        assert_eq!(
            events.back().unwrap().sequence_number,
            (EVENT_BUFFER_CAP + 49) as u64
        );
    }

    #[tokio::test]
    async fn events_after_filters_by_cursor() {
        let state = BackgroundState::new(Arc::new(AtomicBool::new(false)));
        for i in 0..5u64 {
            state
                .push(BufferedEvent {
                    sequence_number: i,
                    event_name: "x".into(),
                    data: serde_json::json!({}),
                })
                .await;
        }
        let after_2 = state.events_after(2).await;
        let seqs: Vec<u64> = after_2.iter().map(|e| e.sequence_number).collect();
        assert_eq!(seqs, vec![3, 4]);

        let after_neg = state.events_after(-1).await;
        assert_eq!(after_neg.len(), 5);
    }

    fn test_response(status: ResponseStatus) -> ResponsesResponse {
        ResponsesResponse {
            id: "resp_test".into(),
            object: "response".into(),
            created_at: 0,
            status,
            model: "m".into(),
            output: Vec::new(),
            output_text: Some("hi".into()),
            usage: ResponsesUsage::default(),
            error: None,
            incomplete_details: None,
            previous_response_id: None,
            instructions: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            temperature: None,
            top_p: None,
            max_output_tokens: None,
            truncation: None,
            metadata: None,
            user: None,
            reasoning: None,
            text: None,
            modalities: None,
            service_tier: None,
            background: None,
            extras: HashMap::new(),
        }
    }

    #[test]
    fn completed_replay_emits_three_events_in_order() {
        let resp = test_response(ResponseStatus::Completed);
        let events = build_completed_replay_events(&resp, -1);
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_name, "response.created");
        assert_eq!(events[0].sequence_number, 0);
        assert_eq!(events[1].event_name, "response.in_progress");
        assert_eq!(events[1].sequence_number, 1);
        assert_eq!(events[2].event_name, "response.completed");
        assert_eq!(events[2].sequence_number, 2);
    }

    #[test]
    fn completed_replay_cursor_skips_earlier_events() {
        let resp = test_response(ResponseStatus::Completed);
        // After seq=1 → only the terminal event survives.
        let events = build_completed_replay_events(&resp, 1);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_name, "response.completed");
        assert_eq!(events[0].sequence_number, 2);
    }

    #[test]
    fn completed_replay_picks_terminal_event_from_status() {
        for (status, name) in [
            (ResponseStatus::Completed, "response.completed"),
            (ResponseStatus::Failed, "response.failed"),
            (ResponseStatus::Incomplete, "response.incomplete"),
            (ResponseStatus::Cancelled, "response.cancelled"),
        ] {
            let resp = test_response(status);
            let events = build_completed_replay_events(&resp, -1);
            assert_eq!(events.last().unwrap().event_name, name);
        }
    }

    #[tokio::test]
    async fn register_and_lookup_background_state() {
        let id = format!("resp_test_{}", uuid::Uuid::new_v4().simple());
        let cancel = Arc::new(AtomicBool::new(false));
        let state = register_background_stream(&id, cancel.clone());
        let fetched = lookup_background_state(&id).expect("registered");
        assert!(Arc::ptr_eq(&state, &fetched));
        assert!(!state.cancel.load(Ordering::SeqCst));
        cancel.store(true, Ordering::SeqCst);
        // Cancel is a separate flag, but both maps share the same
        // Arc<AtomicBool>, so flipping one flips the other.
        assert!(fetched.cancel.load(Ordering::SeqCst));

        deregister_background_stream(&id).await;
        assert!(lookup_background_state(&id).is_none());
    }

    #[tokio::test]
    async fn mark_completed_sets_completed_flag() {
        // Notify wake-up has subtle timing semantics (only wakes already-
        // registered waiters), so this test focuses on the flag — the
        // Notify behavior is exercised end-to-end in
        // serve_resume_stream's integration path, not here.
        let state = Arc::new(BackgroundState::new(Arc::new(AtomicBool::new(false))));
        assert!(!state.completed.load(Ordering::SeqCst));
        state.mark_completed().await;
        assert!(state.completed.load(Ordering::SeqCst));
    }
}
