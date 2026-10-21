//! Asking a split's PEERS whether they hold its prompt before the pass (FUTURE_WORK #252 (b)).
//!
//! #263 admits the prompt on this node's own segments before any segment runs.
//! A peer's segment was still admitted only when its forward arrived — after
//! every machine before it had read the prompt — and a refusal then barred the
//! peer from the re-plan altogether, though most of its layers might fit (the
//! peerkv rig's tight case: a 503 after up to 712 s). Each peer whose segment
//! runs after other work is now asked first (`types::PromptAdmissionAsk`), at
//! the same moment this node's worker is, and the request waits for the
//! answers — one round trip, bounded.
//!
//! Precedent: vLLM's scheduler asks its block manager `can_allocate` for a whole
//! sequence group before the prefill, never per layer during it — and the
//! question allocates nothing; `allocate` follows only once the group is
//! scheduled. So here: a peer WEIGHS the prompt and holds no room for it
//! (`dispatch::prompt_admission`), and its forward is admitted when it arrives,
//! as before. Petals asks no server up front — its client keeps the outputs it
//! already has and re-routes only the span that failed — which a split here
//! cannot yet do mid-pass, so the question moves to the front.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::daemon::state::SharedState;
use crate::types::{
    ModelId, NetworkCommand, NodeId, PipelineSegment, PromptAdmissionAsk, PromptAdmissionVerdict,
    SwarmMessage,
};

/// A round trip counted for a peer this node has not timed.
const UNTIMED_ROUND_TRIP: Duration = Duration::from_millis(500);
/// The most one round trip is counted at. Past it the wait has ended, the pass
/// has gone ahead and each segment admits itself — exactly the behaviour before
/// the ask — so this bounds what a peer that never answers costs a request,
/// never what a slow link may do.
const ROUND_TRIP_COUNTED_AT_MOST: Duration = Duration::from_secs(2);

/// The peers to ask, each with every range the plan gives it: every peer with
/// a segment that runs after other work. A peer whose ONLY segment is the
/// pass's first is not asked — nothing runs before it, so its own forward's
/// admission answers just as early and wastes no one's read of the prompt.
pub(super) fn peers_to_ask(
    segments: &[PipelineSegment],
    local: &NodeId,
) -> Vec<(NodeId, Vec<(u32, u32)>)> {
    let mut asks: Vec<(NodeId, Vec<(u32, u32)>)> = Vec::new();
    for segment in segments.iter().filter(|s| &s.node_id != local) {
        match asks.iter_mut().find(|(node, _)| node == &segment.node_id) {
            Some((_, ranges)) => ranges.push(segment.layer_range),
            None => asks.push((segment.node_id.clone(), vec![segment.layer_range])),
        }
    }
    if let Some(first) = segments.first() {
        asks.retain(|(node, ranges)| !(node == &first.node_id && ranges.len() == 1));
    }
    asks
}

/// How long to wait for the answers: the asked peer's own bound on its worker
/// (`process_pool::UP_FRONT_ADMISSION_WAIT` — a worker answers one message at a
/// time, so one busy with another request's forward answers late) plus two
/// round trips to the slowest peer asked, as its forwards have measured it.
pub(super) fn answer_wait(round_trips_ms: impl IntoIterator<Item = Option<u32>>) -> Duration {
    let slowest = round_trips_ms
        .into_iter()
        .map(|ms| {
            ms.map_or(UNTIMED_ROUND_TRIP, |ms| {
                Duration::from_millis(u64::from(ms))
            })
            .min(ROUND_TRIP_COUNTED_AT_MOST)
        })
        .max()
        .unwrap_or(UNTIMED_ROUND_TRIP);
    crate::inference::process_pool::UP_FRONT_ADMISSION_WAIT + slowest * 2
}

/// A peer that said its conversation memory does not hold the prompt.
pub(super) struct PeerRefusal {
    pub node_id: NodeId,
    /// The refusal in the peer's words — what its forward would have said —
    /// bounded and with its control characters blanked ([`peer_words`]).
    pub message: String,
    pub layers_that_fit: u32,
}

/// The most of a peer's refusal kept: an honest one is the itemised memory
/// refusal, under `sanitize_peer_facing_error`'s 512 characters.
const PEER_WORDS_BYTES: usize = 1024;

/// A peer's refusal as this node repeats it — to its log and to the caller.
/// The text is the peer's, so it is cut to [`PEER_WORDS_BYTES`] and a control
/// character (a newline forging a log line, an escape sequence) becomes a
/// space; the "Service unavailable: " its error type printed is dropped, as
/// this node's own error adds it back.
pub(super) fn peer_words(message: &str) -> String {
    let message = message
        .strip_prefix("Service unavailable: ")
        .unwrap_or(message);
    crate::api::prefix_within_bytes(message, PEER_WORDS_BYTES)
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect()
}

/// The waits registered for one request's asks, removed on every exit.
struct Waiters<'a> {
    state: &'a SharedState,
    keys: Vec<(uuid::Uuid, NodeId)>,
}

impl Drop for Waiters<'_> {
    fn drop(&mut self) {
        for key in &self.keys {
            self.state.pending_prompt_admissions.remove(key);
        }
    }
}

/// Ask every peer in `asks` that understands the question, and wait (bounded)
/// for their answers. Returns the refusals; a peer that said yes, that was not
/// asked (an older build), or that did not answer in time leaves its segments
/// to admit themselves as their forwards arrive, as before. A request
/// cancelled while it waits stops waiting and hears no refusal — the pass it
/// would have run is abandoned anyway.
pub(super) async fn ask_peers(
    shared_state: &SharedState,
    network_tx: &mpsc::Sender<NetworkCommand>,
    request_id: uuid::Uuid,
    model_id: &ModelId,
    asks: Vec<(NodeId, Vec<(u32, u32)>)>,
    prompt_tokens: u32,
    cancel: Option<&Arc<AtomicBool>>,
) -> Vec<PeerRefusal> {
    let mut waiters = Waiters {
        state: shared_state,
        keys: Vec::new(),
    };
    let mut pending = Vec::new();
    let mut round_trips = Vec::new();
    for (node_id, layer_ranges) in asks {
        if !shared_state
            .peer_advertises_feature(&node_id, swarmllm_types::node::features::PROMPT_ADMISSION)
        {
            continue;
        }
        // Asked only over a live connection: the forwards that follow need one
        // too, and a dial to a stale address would spend the whole wait.
        let Some(target_peer_bytes) = shared_state.resolve_connected_peer_id_bytes(&node_id) else {
            continue;
        };
        let (tx, rx) = tokio::sync::oneshot::channel();
        // Keyed by THIS question, not the request: a re-plan keeps the request
        // id, and a late answer to the last attempt's question must not settle
        // this one's.
        let ask_id = uuid::Uuid::new_v4();
        let key = (ask_id, node_id.clone());
        match shared_state.pending_prompt_admissions.entry(key.clone()) {
            dashmap::mapref::entry::Entry::Vacant(slot) => {
                slot.insert(tx);
            }
            dashmap::mapref::entry::Entry::Occupied(_) => continue,
        }
        waiters.keys.push(key);
        let message = SwarmMessage::PromptAdmissionAsk(PromptAdmissionAsk {
            request_id,
            ask_id,
            model_id: model_id.clone(),
            layer_ranges,
            prompt_tokens,
        });
        let sent = network_tx
            .send(NetworkCommand::SendDirectMessage {
                target_peer_bytes,
                message,
                delivery_request_id: None,
            })
            .await;
        if sent.is_err() {
            continue;
        }
        round_trips.push(
            shared_state
                .peer_registry
                .get(&node_id)
                .and_then(|p| p.round_trip_ms()),
        );
        pending.push((node_id, rx));
    }
    if pending.is_empty() {
        return Vec::new();
    }
    let asked = pending.len();
    let wait = answer_wait(round_trips);
    let asked_at = tokio::time::Instant::now();
    let deadline = asked_at + wait;
    let all_answers =
        futures::future::join_all(pending.into_iter().map(|(node_id, rx)| async move {
            (node_id, tokio::time::timeout_at(deadline, rx).await)
        }));
    let answers =
        match crate::inference::cancel::unless_cancelled(async { Ok(all_answers.await) }, cancel)
            .await
        {
            Ok(answers) => answers,
            Err(_) => return Vec::new(),
        };
    let answered = answers
        .iter()
        .filter(|(_, a)| matches!(a, Ok(Ok(_))))
        .count();
    let refusals: Vec<PeerRefusal> = answers
        .into_iter()
        .filter_map(|(node_id, answer)| match answer {
            Ok(Ok(PromptAdmissionVerdict::Refused {
                message,
                layers_that_fit,
            })) => Some(PeerRefusal {
                node_id,
                message: peer_words(&message),
                layers_that_fit,
            }),
            _ => None,
        })
        .collect();
    tracing::info!(
        %request_id,
        model_id = %model_id,
        prompt_tokens,
        asked,
        answered,
        refused = refusals.len(),
        took_ms = asked_at.elapsed().as_millis() as u64,
        wait_ms = wait.as_millis() as u64,
        "DIAG: KV admission up front — asked this split's peers"
    );
    refusals
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(node: u8, range: (u32, u32)) -> PipelineSegment {
        PipelineSegment {
            node_id: NodeId([node; 32]),
            shard_id: crate::types::ShardId {
                model_id: ModelId("m".into()),
                index: 0,
            },
            layer_range: range,
        }
    }

    /// Every peer whose segment runs after other work is asked, for every
    /// range it runs; a peer whose only segment opens the pass is not (its
    /// forward answers as early), and this node never is (#263 asks it).
    #[test]
    fn a_peer_is_asked_for_every_range_it_runs_after_other_work() {
        let local = NodeId([9; 32]);
        let names = |asks: Vec<(NodeId, Vec<(u32, u32)>)>| -> Vec<(u8, Vec<(u32, u32)>)> {
            asks.into_iter().map(|(n, r)| (n.0[0], r)).collect()
        };

        // This node first: every peer after it is asked.
        let ours_first = [
            segment(9, (0, 10)),
            segment(1, (10, 20)),
            segment(2, (20, 32)),
        ];
        assert_eq!(
            names(peers_to_ask(&ours_first, &local)),
            vec![(1, vec![(10, 20)]), (2, vec![(20, 32)])]
        );

        // A peer opens the pass: it is not asked, the rest are.
        let peer_first = [
            segment(1, (0, 10)),
            segment(9, (10, 20)),
            segment(2, (20, 32)),
        ];
        assert_eq!(
            names(peers_to_ask(&peer_first, &local)),
            vec![(2, vec![(20, 32)])]
        );

        // A peer that opens the pass AND runs a later range is asked for both.
        let twice = [
            segment(1, (0, 10)),
            segment(2, (10, 20)),
            segment(1, (20, 32)),
        ];
        assert_eq!(
            names(peers_to_ask(&twice, &local)),
            vec![(1, vec![(0, 10), (20, 32)]), (2, vec![(10, 20)])]
        );

        // Ours alone: nobody.
        assert!(peers_to_ask(&[segment(9, (0, 32))], &local).is_empty());
    }

    /// A peer's refusal is repeated without the prefix its error type printed,
    /// cut short, and with no control character left to forge a log line.
    #[test]
    fn a_peers_refusal_is_repeated_bounded_and_on_one_line() {
        assert_eq!(
            peer_words("Service unavailable: not enough memory for 40 layers"),
            "not enough memory for 40 layers"
        );
        assert_eq!(
            peer_words("no room\n2026-10-18T00:00:00Z  INFO forged\u{1b}[31m"),
            "no room 2026-10-18T00:00:00Z  INFO forged [31m"
        );
        let long = "é".repeat(PEER_WORDS_BYTES);
        let kept = peer_words(&long);
        assert!(kept.len() <= PEER_WORDS_BYTES && kept.chars().all(|c| c == 'é'));
    }

    /// The wait is the peer's own bound on its worker plus two round trips to
    /// the slowest peer asked; an untimed peer counts half a second, and no
    /// round trip counts more than two.
    #[test]
    fn the_wait_for_answers_follows_the_slowest_link_asked() {
        let ms = Duration::from_millis;
        assert_eq!(answer_wait([Some(20), Some(150)]), ms(1000 + 300));
        assert_eq!(answer_wait([Some(20), None]), ms(1000 + 1000));
        assert_eq!(answer_wait([Some(9_000)]), ms(1000 + 4000));
        assert_eq!(answer_wait(Vec::new()), ms(1000 + 1000));
    }
}
