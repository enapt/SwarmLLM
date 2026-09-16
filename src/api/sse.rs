//! Provider-neutral streaming primitives shared by SSE endpoints.
//!
//! OpenAI and Anthropic formats are different enough that the final SSE
//! serialization lives in each provider module, but the intermediate event
//! type used between the inference loop and the SSE encoder is the same
//! shape for OpenAI-style streams and is kept here for reuse.

use tokio::sync::mpsc;

// ---- Raw SSE framing for non-axum-Sse streams ----
//
// Some handlers (Claude subscription, Claude Code sessions) produce a pre-
// formatted byte stream wrapped in `axum::body::Body::from_stream` rather
// than routing through `axum::response::Sse`. These helpers keep the `data:`
// and `event:` framing uniform across those handlers.

/// Format a value as an SSE `data:` frame (for byte streams).
/// Accepts any `serde::Serialize` — typed response structs or raw JSON values.
/// On serialization failure (e.g. NaN in a logprob) emits a structured error
/// event rather than an empty `data:` line — empty data is a valid SSE
/// no-op event and would silently hang clients waiting for a specific
/// payload (e.g. `[DONE]` or `message_stop`).
pub fn data_frame<S: serde::Serialize>(value: &S) -> bytes::Bytes {
    match serde_json::to_string(value) {
        Ok(json) => bytes::Bytes::from(format!("data: {json}\n\n")),
        Err(e) => {
            tracing::error!(error = %e, "SSE data_frame: serialization failed");
            bytes::Bytes::from_static(b"data: {\"error\":\"serialization_failed\"}\n\n")
        }
    }
}

/// Format a named SSE event frame (`event: ...\ndata: ...`) for byte streams.
pub fn event_frame<S: serde::Serialize>(event_type: &str, value: &S) -> bytes::Bytes {
    match serde_json::to_string(value) {
        Ok(json) => bytes::Bytes::from(format!("event: {event_type}\ndata: {json}\n\n")),
        Err(e) => {
            tracing::error!(error = %e, event_type, "SSE event_frame: serialization failed");
            bytes::Bytes::from(format!(
                "event: {event_type}\ndata: {{\"error\":\"serialization_failed\"}}\n\n"
            ))
        }
    }
}

/// Terminal `data: [DONE]` frame used by OpenAI-compatible streams.
pub fn done_frame() -> bytes::Bytes {
    bytes::Bytes::from_static(b"data: [DONE]\n\n")
}

/// Intermediate stream event emitted by the inference loop and consumed by
/// the OpenAI-format SSE encoder.
pub enum StreamEvent {
    Delta {
        content: Option<String>,
        role: Option<String>,
        finish_reason: Option<String>,
    },
    /// A failure, in the same terms the non-streaming sibling would report it.
    ///
    /// `error_type` is a required field rather than an encoder-side default
    /// precisely so a new call site has to say what kind of failure this is.
    /// The encoder used to stamp every one of them `server_error`, which told
    /// the caller that this server had broken when in fact their prompt was too
    /// long. Fill it from `crate::error::classify_error`, never by hand.
    ///
    /// `hint` / `hint_key` are the ACTIONABLE half, and they were absent here
    /// while the non-streaming envelope (`error.rs`'s `IntoResponse`) and the
    /// Anthropic surface both carried them — so the same failure told a
    /// streaming caller strictly less than a non-streaming one about what to
    /// do next. Prefer [`StreamEvent::from_error`], which fills all four from
    /// the single sources and cannot forget one.
    Error {
        message: String,
        error_type: &'static str,
        hint: Option<&'static str>,
        hint_key: Option<&'static str>,
    },
    /// A complete set of tool calls recovered from a local model's output.
    ///
    /// Additive variant rather than a field on `Delta`, which has a dozen
    /// construction sites that have nothing to do with tools.
    ///
    /// Emitted as ONE event carrying whole calls rather than the fragment
    /// sequence a cloud provider streams. A local model's tool call can only be
    /// recognised once its text is complete — mid-stream we cannot tell a tool
    /// call from prose that happens to start with a brace — so fragments would
    /// mean emitting text we might have to retract. Clients that concatenate
    /// streamed `tool_calls` deltas handle a single complete delta correctly,
    /// since the index/id/name/arguments fields are all present at once.
    ToolCalls {
        calls: Vec<crate::api::openai::StreamToolCall>,
    },
    /// OpenAI 2024+ spec: when the request includes
    /// `stream_options: {"include_usage": true}`, an extra terminal chunk
    /// is emitted right before `[DONE]` with `choices: []` and the usage
    /// object populated. This event carries the token counts to the
    /// encoder; emit it ONLY when the request opted in.
    Usage {
        prompt_tokens: u32,
        completion_tokens: u32,
    },
    Done,
}

impl StreamEvent {
    /// Turn a `SwarmError` into a streamed failure frame — the ONE way to build
    /// one from a typed error.
    ///
    /// It reads the two single sources together: `classify_error` for the kind,
    /// `error_hint_with_key` for the advice. Three call sites each rebuilt the
    /// first by hand and none of them carried the second, so a streaming caller
    /// got the message and never the hint, while the non-streaming sibling for
    /// the identical failure got both. A constructor rather than a doc note
    /// because "remember to add the hint too" is exactly the obligation this
    /// codebase has repeatedly shown does not survive a new call site.
    pub fn from_error(err: &crate::error::SwarmError) -> Self {
        let (hint_key, hint) = match crate::error::error_hint_with_key(err) {
            Some((key, text)) => (Some(key), Some(text)),
            None => (None, None),
        };
        StreamEvent::Error {
            message: err.to_string(),
            error_type: crate::error::classify_error(err).2,
            hint,
            hint_key,
        }
    }
}

/// Send the initial `role: "assistant"` delta that opens the streaming response.
/// Returns `false` if the client has already disconnected.
pub async fn send_role_preamble(tx: &mpsc::Sender<StreamEvent>) -> bool {
    tx.send(StreamEvent::Delta {
        content: None,
        role: Some("assistant".into()),
        finish_reason: None,
    })
    .await
    .is_ok()
}

/// One-line human-readable progress note for an SSE comment frame.
///
/// Deliberately plain text, not JSON: it is a comment, so nothing parses it —
/// its only reader is a person watching a stream that would otherwise look
/// dead. Says "still reading" rather than naming an internal phase, and omits
/// the ETA entirely rather than inventing one before the rate is known.
pub fn format_progress_comment(s: &crate::inference::trace::ProgressSnapshot) -> String {
    let what = match s.phase {
        "loading_model" => "loading model".to_string(),
        "prefill" => match s.percent {
            Some(pct) => format!("reading prompt {pct}% ({}/{} tokens)", s.done, s.total),
            None => "reading prompt".to_string(),
        },
        other => other.to_string(),
    };
    match s.eta_ms {
        Some(ms) if ms >= 1000 => format!("{what}, about {}s left", ms / 1000),
        _ => what,
    }
}

/// Marks the machine-readable half of a progress keep-alive.
///
/// SSE comments are the only thing that can ride a `/v1/chat/completions`
/// stream with no risk to the clients already on it: every conforming reader
/// drops a line beginning with `:`, so an OpenAI SDK sees exactly what it saw
/// before. A `data:` frame carrying a non-chat-completion object would not be —
/// some clients deserialise every frame strictly — and an `event:` name is
/// invisible to anything not using `EventSource`. So the status goes out as a
/// comment, with this prefix so a reader that DOES want it can pick it out from
/// the prose.
///
/// The prefix is part of the wire contract with the dashboard. It is not a
/// private handshake: anything may read it, and nothing has to.
pub const STATUS_COMMENT_PREFIX: &str = "swarmllm-status ";

/// The machine-readable progress line.
///
/// `to_string` never emits a raw newline — the SSE encoder ASSERTS on one, and
/// a panic inside the ticker would take the whole response down.
pub fn format_status_comment(status: &crate::inference::trace::LiveStatus) -> Option<String> {
    serde_json::to_string(status)
        .ok()
        .map(|json| format!("{STATUS_COMMENT_PREFIX}{json}"))
}

/// The comment lines one keep-alive tick carries.
///
/// Two of them, for different readers:
///
/// 1. **Prose**, for a person watching the stream in a terminal. Always
///    present — an empty comment is still a valid keep-alive, which is what a
///    request with nothing to report emits.
/// 2. **Structured**, for a client that can show what is happening. Emitted
///    whenever there is a trace at all, which crucially includes the phases
///    BEFORE any worker has reported: queued, planning, contacting other
///    computers. That window is exactly the one a fixed "Thinking…" used to
///    cover, and the daemon has always known what was in it.
///
/// Separated from the ticker so the lines can be asserted without driving a
/// stream, and folded back into ONE event by the ticker so there is still only
/// one keep-alive on the wire.
fn keep_alive_lines(trace: Option<&crate::inference::trace::RequestTrace>) -> Vec<String> {
    let prose = trace
        .and_then(|t| t.progress())
        .map(|s| format_progress_comment(&s))
        .unwrap_or_default();

    let mut lines = vec![prose];
    if let Some(line) = trace
        .map(|t| t.live_status())
        .and_then(|s| format_status_comment(&s))
    {
        lines.push(line);
    }
    lines
}

// ---- Keep-alive / progress ticker ----

/// Interleaved keep-alive comments carrying the request's progress, for merging
/// with a token stream.
///
/// **Why this is shared rather than written per surface.** Both SSE encoders
/// had a byte-identical copy of this, and both carried the same defect: the
/// ticker slept the whole interval and only *then* checked whether the response
/// had finished. `merge` ends when both halves end, so every streamed response
/// stayed open until that in-flight sleep expired — measured 2026-08-25 on this
/// machine, an 8-token reply delivered in 0.5 s held its connection to 15.0 s,
/// and the same request answered non-streaming in 0.56 s. Clients that stop at
/// `[DONE]` never saw it; anything reading to end-of-stream waited, and the
/// server held a task and a connection per stream for the remainder of the
/// interval either way.
///
/// The comment above the old copy said *"the ticker MUST terminate"* and was
/// right about the hazard it had in mind (an unbounded ticker holds the response
/// open for ever). Terminating late is the same bug with a bound on it.
///
/// So the wait is cancellable: whichever of the interval and the finish signal
/// comes first ends it. A dropped sender ends it too — the token stream is gone,
/// so there is nothing left to keep alive.
pub(crate) fn progress_ticker(
    progress: Option<(std::sync::Arc<crate::daemon::SharedState>, uuid::Uuid)>,
    finished: tokio::sync::watch::Receiver<bool>,
    // A `Duration` rather than a count of seconds so a test can drive this at
    // millisecond scale and assert exactly, instead of either waiting out a
    // real interval or pulling in tokio's `test-util` clock.
    interval: std::time::Duration,
) -> impl futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>>
       + Send
       + 'static {
    futures::stream::unfold((progress, finished), move |(p, mut finished)| async move {
        tokio::select! {
            _ = tokio::time::sleep(interval) => {}
            changed = finished.changed() => {
                // Err means the sender is gone with the token stream.
                if changed.is_err() {
                    return None;
                }
            }
        }
        if *finished.borrow() {
            return None;
        }
        let trace = p
            .as_ref()
            .and_then(|(state, rid)| state.active_traces.get(rid).map(|t| t.clone()));

        let event = keep_alive_lines(trace.as_deref())
            .into_iter()
            .fold(axum::response::sse::Event::default(), |e, line| {
                e.comment(line)
            });

        Some((Ok(event), (p, finished)))
    })
}

#[cfg(test)]
mod error_frame_tests {
    use super::*;
    use crate::error::SwarmError;

    /// **A streamed failure carries the same advice as its non-streaming
    /// sibling.** The message alone says what went wrong; the hint says what to
    /// do about it, and the streaming surface used to drop it — so the same
    /// failure told a streaming caller strictly less. Asserted against
    /// `error_hint_with_key`, which is the one source both surfaces read.
    #[test]
    fn a_streamed_failure_carries_the_same_hint_as_the_non_streaming_one() {
        let err = SwarmError::ModelNotAvailable(crate::types::ModelId("llama-3.2-3b".to_string()));
        let expected =
            crate::error::error_hint_with_key(&err).expect("this variant has a hint to carry");

        match StreamEvent::from_error(&err) {
            StreamEvent::Error {
                message,
                error_type,
                hint,
                hint_key,
            } => {
                assert!(
                    message.contains("llama-3.2-3b"),
                    "the message must name the failure: {message}"
                );
                assert_eq!(error_type, crate::error::classify_error(&err).2);
                assert_eq!(
                    hint_key,
                    Some(expected.0),
                    "the streamed frame must carry the same hint KEY the \
                     dashboard translates"
                );
                assert_eq!(
                    hint,
                    Some(expected.1),
                    "the streamed frame must carry the same English hint the \
                     non-streaming envelope sends"
                );
            }
            _ => panic!("from_error must build an Error frame"),
        }
    }

    /// A variant with no advice sends no advice — not an empty string, and not
    /// a guess. The encoder omits the fields entirely in that case.
    #[test]
    fn a_failure_with_no_advice_carries_none() {
        let err = SwarmError::Internal("something we did wrong".into());
        if crate::error::error_hint_with_key(&err).is_none() {
            match StreamEvent::from_error(&err) {
                StreamEvent::Error { hint, hint_key, .. } => {
                    assert_eq!(hint, None);
                    assert_eq!(hint_key, None);
                }
                _ => panic!("from_error must build an Error frame"),
            }
        }
    }
}

#[cfg(test)]
mod ticker_tests {
    use super::*;
    use futures::StreamExt;
    use std::time::Duration;

    /// The defect this exists for: a response whose tokens are done must not be
    /// Both halves of a tick, and what each is for.
    #[test]
    fn a_keep_alive_carries_prose_for_a_person_and_json_for_a_client() {
        use crate::inference::trace::RequestTrace;
        let trace = RequestTrace::new(uuid::Uuid::new_v4(), "m", "chat");
        trace.mark_dequeued();
        trace.set_progress("prefill", 256, 1024);

        let lines = super::keep_alive_lines(Some(&trace));
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].contains("reading prompt 25%"),
            "the prose half is for someone watching a terminal, got: {}",
            lines[0]
        );
        let status = lines[1]
            .strip_prefix(super::STATUS_COMMENT_PREFIX)
            .expect("the structured half carries the marker a client looks for");
        let parsed: serde_json::Value = serde_json::from_str(status).expect("valid JSON");
        assert_eq!(parsed["phase"], "reading_prompt");
        assert_eq!(parsed["percent"], 25);
    }

    /// The window the whole thing exists for: a request that has been admitted
    /// and has no worker report yet still says what it is doing. Before this,
    /// the only thing on the wire for that window was an empty comment.
    #[test]
    fn a_request_with_nothing_to_report_yet_still_says_what_it_is_doing() {
        use crate::inference::trace::RequestTrace;
        let trace = RequestTrace::new(uuid::Uuid::new_v4(), "m", "chat");

        let lines = super::keep_alive_lines(Some(&trace));
        assert_eq!(lines[0], "", "there is no prose to write yet");
        assert!(
            lines[1].contains("\"phase\":\"queued\""),
            "the structured half covers the phases prose never had, got: {}",
            lines[1]
        );
    }

    /// No trace at all — a surface that did not pass one — must still keep the
    /// connection alive, and must not invent a status for a request it cannot
    /// see.
    #[test]
    fn no_trace_means_a_bare_keep_alive_and_no_invented_status() {
        let lines = super::keep_alive_lines(None);
        assert_eq!(lines, vec![String::new()]);
    }

    /// The wiring, not the helper. A correct `keep_alive_lines` the ticker does
    /// not call is the shape of gotcha #601 — a fix written, tested and never
    /// reached — so this drives the real ticker over a real trace held in a
    /// real `active_traces` and reads what came out on the wire.
    #[tokio::test]
    async fn the_ticker_really_puts_the_status_on_the_wire() {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::inference::trace::RequestTrace;
        use crate::storage::db::Database;

        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(tokio::sync::Mutex::new(ModelExecutor::new()));
        let (state, _a, _b) = crate::daemon::SharedState::new(
            crate::config::Config::default(),
            Identity::generate(),
            db,
            executor,
            None,
        );

        let rid = uuid::Uuid::new_v4();
        let trace = std::sync::Arc::new(RequestTrace::new(rid, "m", "chat"));
        trace.mark_dequeued();
        state.active_traces.insert(rid, trace);

        let (_tx, rx) = tokio::sync::watch::channel(false);
        let ticker = progress_ticker(Some((state, rid)), rx, Duration::from_millis(5));
        futures::pin_mut!(ticker);
        let event = tokio::time::timeout(Duration::from_secs(5), ticker.next())
            .await
            .expect("a live request must get keep-alives")
            .expect("stream must yield")
            .expect("infallible");

        // `Event` keeps its wire bytes in a `BytesMut` that it prints in full.
        let on_the_wire = format!("{event:?}");
        assert!(
            on_the_wire.contains("swarmllm-status"),
            "the ticker must EMIT the status line, not merely be able to: {on_the_wire}"
        );
        assert!(
            on_the_wire.contains("planning"),
            "and it must carry the phase the trace is actually in: {on_the_wire}"
        );
    }

    /// held open for the rest of the keep-alive interval.
    ///
    /// Measured on the live node before the fix — an 8-token reply delivered in
    /// 0.5 s held its connection to 15.0 s, the configured interval, while the
    /// same request answered non-streaming in 0.56 s.
    ///
    /// The hour-long interval is what makes this decisive rather than flaky: a
    /// ticker that waits it out cannot possibly answer inside the timeout, and
    /// one that reacts to the signal answers immediately.
    #[tokio::test]
    async fn a_finished_response_ends_the_ticker_without_waiting_out_the_interval() {
        let (tx, rx) = tokio::sync::watch::channel(false);
        let ticker = progress_ticker(None, rx, Duration::from_secs(3600));
        futures::pin_mut!(ticker);

        tx.send(true).expect("receiver is alive");
        let ended = tokio::time::timeout(Duration::from_millis(500), ticker.next())
            .await
            .expect("the ticker waited out its interval instead of reacting to the finish signal");
        assert!(
            ended.is_none(),
            "the ticker must END once the response is finished, not emit again"
        );
    }

    /// The hazard the original comment was written about, still covered: an
    /// unfinished response keeps getting keep-alives, so a slow request does not
    /// look dead to the client or to an intermediary.
    #[tokio::test]
    async fn an_unfinished_response_still_gets_keep_alives() {
        let (_tx, rx) = tokio::sync::watch::channel(false);
        let ticker = progress_ticker(None, rx, Duration::from_millis(20));
        futures::pin_mut!(ticker);
        for i in 0..3 {
            let item = tokio::time::timeout(Duration::from_secs(5), ticker.next())
                .await
                .unwrap_or_else(|_| panic!("keep-alive {i} never arrived"));
            assert!(
                item.is_some(),
                "a live response must keep receiving keep-alives"
            );
        }
    }

    /// A token stream that goes away without a terminal frame — a client
    /// disconnecting mid-reply — must not leave the ticker running. Dropping the
    /// sender is that signal.
    #[tokio::test]
    async fn a_dropped_stream_ends_the_ticker() {
        let (tx, rx) = tokio::sync::watch::channel(false);
        let ticker = progress_ticker(None, rx, Duration::from_secs(3600));
        futures::pin_mut!(ticker);
        drop(tx);
        let ended = tokio::time::timeout(Duration::from_millis(500), ticker.next())
            .await
            .expect("a ticker whose stream is gone must not keep waiting");
        assert!(ended.is_none());
    }
}
