//! Answering a peer's `PromptAdmissionAsk` (FUTURE_WORK #252 (b)): would this
//! node's conversation memory hold a split's prompt over the ranges the plan
//! gives it? Asked by the split's coordinator before the prompt pass, so a "no"
//! costs one round trip instead of every earlier machine's read of the prompt.
//!
//! The question is WEIGHED, never granted (vLLM's `can_allocate` beside its
//! `allocate`): `ModelProcessPool::weigh_peer_prompt` runs the same arithmetic
//! this node's own pipeline is admitted by (#263) and records nothing — no
//! promise, no reserved positions, no release of an ended conversation, no
//! entry under the asker's request id. The request's forwards are admitted as
//! they always were when they arrive. A security review (2026-10-18) is why:
//! an ask that held room let any peer park this node's whole conversation
//! memory, refreshable forever, and an ask under a live request's id took that
//! request's answer slot.

use std::sync::Arc;

use tokio::sync::mpsc;

use crate::daemon::state::SharedState;
use crate::inference::process_pool::PeerPromptWeight;
use crate::types::{
    NetworkCommand, NodeId, PromptAdmissionAnswer, PromptAdmissionAsk, PromptAdmissionVerdict,
    SwarmMessage,
};

/// The most ranges one ask may name. A plan gives one machine a handful of
/// segments at most; an ask naming more is not one an honest coordinator sends.
const MAX_ASKED_RANGES: usize = 16;

/// Is this an ask an honest coordinator could have sent? Every range non-empty,
/// a few of them, and a prompt to weigh.
fn ask_is_well_formed(ask: &PromptAdmissionAsk) -> bool {
    !ask.layer_ranges.is_empty()
        && ask.layer_ranges.len() <= MAX_ASKED_RANGES
        && ask.layer_ranges.iter().all(|&(start, end)| start < end)
        && ask.prompt_tokens > 0
}

/// Take `asker`'s ask: answer at once what needs no worker, and weigh the rest
/// in a task of its own (the worker may take up to its bound to answer).
///
/// The cheap verdicts come first so that only a question that will really be
/// weighed costs a task. Past this node's [`AskBudget`] the asker is told "go
/// ahead" straight away — the answer every split had before the question
/// existed — rather than made to wait out its own bound.
///
/// [`AskBudget`]: crate::daemon::state::prompt_admission::AskBudget
pub(super) fn on_ask(
    shared_state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    asker: NodeId,
    ask: PromptAdmissionAsk,
) {
    if !ask_is_well_formed(&ask) {
        tracing::debug!(sender = %asker, request_id = %ask.request_id, "Dropping a malformed PromptAdmissionAsk");
        return;
    }
    // Answered over the connection the ask came in on, never a dial to a
    // stale address the registry still carries for it.
    let Some(asker_peer_bytes) = shared_state.resolve_connected_peer_id_bytes(&asker) else {
        tracing::debug!(sender = %asker, "PromptAdmissionAsk from a peer no longer connected");
        return;
    };
    // A node on a metered connection takes no new work for other machines; its
    // forwards would be refused, so the re-plan should plan none here.
    if shared_state.metered_hold_now() {
        let verdict = PromptAdmissionVerdict::Refused {
            message: super::metered_refusal(),
            layers_that_fit: 0,
        };
        answer_unweighed(&network_tx, asker, asker_peer_bytes, &ask, verdict);
        return;
    }
    if !shared_state
        .prompt_ask_budget
        .try_spend(&asker, std::time::Instant::now())
    {
        let verdict = PromptAdmissionVerdict::Proceed;
        answer_unweighed(&network_tx, asker, asker_peer_bytes, &ask, verdict);
        return;
    }
    tokio::spawn(async move {
        let weight = shared_state
            .model_process_pool
            .weigh_peer_prompt(&ask.model_id, ask.layer_ranges.clone(), ask.prompt_tokens)
            .await;
        let verdict = match weight {
            PeerPromptWeight::Fits => PromptAdmissionVerdict::Proceed,
            PeerPromptWeight::DoesNotFit {
                message,
                layers_that_fit,
            } => PromptAdmissionVerdict::Refused {
                // The words a forward's own refusal would have carried.
                message: super::layer_forward::sanitize_peer_facing_error(&message),
                layers_that_fit,
            },
        };
        answer(network_tx, asker, asker_peer_bytes, ask, verdict).await;
    });
}

/// A verdict that needed no worker, sent from the dispatcher itself: no task,
/// and no wait on a full network queue — a question sent at wire speed costs
/// one `try_send` and a debug line, never a task and an info line each
/// (security review, 2026-10-18). Dropped if the queue is full: the asker's
/// wait ends and its segments admit themselves, as for any unanswered ask.
fn answer_unweighed(
    network_tx: &mpsc::Sender<NetworkCommand>,
    asker: NodeId,
    asker_peer_bytes: Vec<u8>,
    ask: &PromptAdmissionAsk,
    verdict: PromptAdmissionVerdict,
) {
    tracing::debug!(
        request_id = %ask.request_id,
        sender = %asker,
        verdict = ?verdict,
        "answered a peer's up-front prompt admission without weighing it"
    );
    if network_tx
        .try_send(answer_to(asker_peer_bytes, ask, verdict))
        .is_err()
    {
        tracing::debug!(sender = %asker, "network queue full — PromptAdmissionAnswer dropped");
    }
}

fn answer_to(
    asker_peer_bytes: Vec<u8>,
    ask: &PromptAdmissionAsk,
    verdict: PromptAdmissionVerdict,
) -> NetworkCommand {
    NetworkCommand::SendDirectMessage {
        target_peer_bytes: asker_peer_bytes,
        message: SwarmMessage::PromptAdmissionAnswer(PromptAdmissionAnswer {
            request_id: ask.request_id,
            ask_id: ask.ask_id,
            verdict,
        }),
        delivery_request_id: None,
    }
}

/// A verdict the worker weighed — at most [`AskBudget`]'s figures a minute,
/// so each is worth an info line.
///
/// [`AskBudget`]: crate::daemon::state::prompt_admission::AskBudget
async fn answer(
    network_tx: mpsc::Sender<NetworkCommand>,
    asker: NodeId,
    asker_peer_bytes: Vec<u8>,
    ask: PromptAdmissionAsk,
    verdict: PromptAdmissionVerdict,
) {
    tracing::info!(
        request_id = %ask.request_id,
        sender = %asker,
        model_id = %ask.model_id,
        ranges = ?ask.layer_ranges,
        prompt_tokens = ask.prompt_tokens,
        verdict = ?verdict,
        "DIAG: answered a peer's up-front prompt admission"
    );
    if let Err(e) = network_tx
        .send(answer_to(asker_peer_bytes, &ask, verdict))
        .await
    {
        tracing::warn!(error = %e, "Failed to send PromptAdmissionAnswer");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ask(ranges: Vec<(u32, u32)>, prompt_tokens: u32) -> PromptAdmissionAsk {
        PromptAdmissionAsk {
            request_id: uuid::Uuid::new_v4(),
            ask_id: uuid::Uuid::new_v4(),
            model_id: crate::types::ModelId("m".into()),
            layer_ranges: ranges,
            prompt_tokens,
        }
    }

    #[test]
    fn only_an_ask_an_honest_coordinator_could_send_is_answered() {
        assert!(ask_is_well_formed(&ask(vec![(0, 12), (20, 22)], 500)));
        assert!(!ask_is_well_formed(&ask(vec![], 500)), "no range");
        assert!(
            !ask_is_well_formed(&ask(vec![(12, 12)], 500)),
            "an empty range"
        );
        assert!(
            !ask_is_well_formed(&ask(vec![(12, 4)], 500)),
            "a reversed range"
        );
        assert!(!ask_is_well_formed(&ask(vec![(0, 1)], 0)), "no prompt");
        assert!(!ask_is_well_formed(&ask(
            (0..17).map(|i| (i, i + 1)).collect(),
            500
        )));
    }
}
