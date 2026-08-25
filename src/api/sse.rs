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
    Error {
        message: String,
        error_type: &'static str,
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
        let text = p
            .as_ref()
            .and_then(|(state, rid)| state.active_traces.get(rid).and_then(|t| t.progress()))
            .map(|s| format_progress_comment(&s))
            // No snapshot yet, or already streaming: an empty comment is a
            // valid keep-alive, which is what this subsumes.
            .unwrap_or_default();
        Some((
            Ok(axum::response::sse::Event::default().comment(text)),
            (p, finished),
        ))
    })
}

#[cfg(test)]
mod ticker_tests {
    use super::*;
    use futures::StreamExt;
    use std::time::Duration;

    /// The defect this exists for: a response whose tokens are done must not be
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
