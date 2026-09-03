//! One rule for every long wait in a request's life: watch the request's
//! cancel flag while waiting, and stop the moment it flips.
//!
//! `InferenceRequest::cancel` is the single cancellation signal. The
//! non-streaming API flips it when the client's connection drops
//! (`CancelOnDisconnect`), both streaming surfaces flip it the instant their
//! SSE body's receiver is gone, and `/v1/responses/{id}/cancel` flips it by
//! hand. Until 2026-09-03 the pipeline read it in exactly one place — the top
//! of the per-token loop — so a cancel that landed during the PROMPT pass was
//! seen only when the prompt pass ended. On a card that is seconds. On a
//! processor-only node running an agent's 14,000-token prompt through a 14B it
//! is longer than any client waits, and a tester found a worker at 400% CPU for
//! 81 minutes, with a thermal warning, after three abandoned attempts — each of
//! which had queued its own full prompt pass behind the last (the second half
//! of gotcha #441; #445).
//!
//! The waits that matter are the ones that can run for minutes with nothing to
//! send: the worker's answer to a local segment (`ModelProcessPool::
//! forward_for_request`) and a remote segment's result (`PipelineExecutor::
//! wait_for_result`). Both go through [`unless_cancelled`]. Dropping the wait is
//! what stops the work: the pool's `ResponseGuard` sends `CancelRequest` to the
//! worker on drop, and the worker skips a forward it has not started and stops
//! between layers if it has; the remote caller sends `CancelInference` to the
//! peer. The router then declines to retry a request whose flag is set.
//!
//! The flag is polled rather than awaited because it is an `AtomicBool` shared
//! with code that has no runtime handle; [`CANCEL_POLL`] bounds the latency of
//! noticing, and a quarter-second is nothing against the minutes it saves.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crate::error::SwarmError;

/// The message a wait ends with when the request was cancelled underneath it.
///
/// `ServiceUnavailable`, deliberately: it is exempt from the peer penalty
/// (`failure_is_penalty_worthy`) — nobody's hardware failed, the client left —
/// and the router refuses to retry a cancelled request, so the "peer could not
/// serve" retry that variant otherwise invites never fires. The status it maps
/// to reaches nobody: the client is the reason it exists.
pub(crate) const REQUEST_ABANDONED: &str =
    "Request abandoned by the client before the reply was ready";

/// How long a cancel can go unnoticed inside a watched wait.
pub(crate) const CANCEL_POLL: Duration = Duration::from_millis(250);

/// The error a watched wait returns once the request is cancelled.
pub(crate) fn request_abandoned() -> SwarmError {
    SwarmError::ServiceUnavailable(REQUEST_ABANDONED.into())
}

/// Did this error come from [`unless_cancelled`] noticing a cancelled request?
///
/// Matches the exact marker this module produces and nothing else — the same
/// pattern as `REMOTE_GENERATE_NOT_HOSTED`: a constant owned here, not prose
/// matched by substring.
pub(crate) fn is_request_abandoned(err: &SwarmError) -> bool {
    matches!(err, SwarmError::ServiceUnavailable(m) if m == REQUEST_ABANDONED)
}

/// Await `fut`, but stop — dropping it — the moment `cancel` reads true.
///
/// With no flag this is `fut.await`, exactly as before. With one, the flag is
/// checked before the wait starts (a cancel that arrived earlier costs nothing)
/// and every [`CANCEL_POLL`] during it. Dropping `fut` is the mechanism, so the
/// future handed in must be one whose drop tells the other side to stop: the
/// pool's response wait (its guard sends `CancelRequest`), a oneshot waiter the
/// caller follows with `CancelInference`.
pub(crate) async fn unless_cancelled<T, F>(
    fut: F,
    cancel: Option<&Arc<AtomicBool>>,
) -> Result<T, SwarmError>
where
    F: Future<Output = Result<T, SwarmError>>,
{
    let Some(flag) = cancel else {
        return fut.await;
    };
    if flag.load(Ordering::Acquire) {
        return Err(request_abandoned());
    }
    tokio::pin!(fut);
    let mut poll = tokio::time::interval(CANCEL_POLL);
    poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            r = &mut fut => return r,
            _ = poll.tick() => {
                if flag.load(Ordering::Acquire) {
                    return Err(request_abandoned());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flag(set: bool) -> Arc<AtomicBool> {
        Arc::new(AtomicBool::new(set))
    }

    /// The case the module exists for: a wait with nothing to say for minutes
    /// ends within a poll or two of the client leaving.
    #[tokio::test]
    async fn a_pending_wait_ends_when_the_flag_flips() {
        let f = flag(false);
        let flipper = f.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            flipper.store(true, Ordering::Release);
        });
        let started = std::time::Instant::now();
        let r: Result<(), SwarmError> =
            unless_cancelled(std::future::pending::<Result<(), SwarmError>>(), Some(&f)).await;
        assert!(is_request_abandoned(&r.unwrap_err()));
        assert!(
            started.elapsed() < Duration::from_secs(3),
            "must notice within a few polls, took {:?}",
            started.elapsed()
        );
    }

    /// A cancel that landed BEFORE the wait costs nothing: the future is never
    /// polled.
    #[tokio::test]
    async fn an_already_cancelled_request_does_not_start_the_wait() {
        let f = flag(true);
        let r: Result<(), SwarmError> =
            unless_cancelled(std::future::pending::<Result<(), SwarmError>>(), Some(&f)).await;
        assert!(is_request_abandoned(&r.unwrap_err()));
    }

    /// A wait that completes returns its own result, whatever the flag — the
    /// flag ends waiting, it never rewrites an answer that arrived.
    #[tokio::test]
    async fn a_completed_wait_returns_its_result() {
        let f = flag(false);
        let ok: Result<u8, SwarmError> =
            unless_cancelled(std::future::ready(Ok(7u8)), Some(&f)).await;
        assert_eq!(ok.unwrap(), 7);
        let err: Result<u8, SwarmError> = unless_cancelled(
            std::future::ready(Err(SwarmError::Internal("own".into()))),
            Some(&f),
        )
        .await;
        assert!(matches!(err, Err(SwarmError::Internal(m)) if m == "own"));
    }

    /// No flag, no watching: the wait is exactly `fut.await`.
    #[tokio::test]
    async fn no_flag_means_the_plain_await() {
        let r: Result<u8, SwarmError> = unless_cancelled(std::future::ready(Ok(3u8)), None).await;
        assert_eq!(r.unwrap(), 3);
    }

    /// Only this module's marker is recognised — not the variant, not the text
    /// under another variant.
    #[test]
    fn only_the_marker_is_recognised() {
        assert!(is_request_abandoned(&request_abandoned()));
        assert!(!is_request_abandoned(&SwarmError::ServiceUnavailable(
            "worker is dead".into()
        )));
        assert!(!is_request_abandoned(&SwarmError::PipelineError(
            REQUEST_ABANDONED.into()
        )));
    }
}
