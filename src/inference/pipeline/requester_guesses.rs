//! Guessing for a split this node handed to its head (#318) — the requester's
//! half; the head's is `daemon::state::requester_guesses`.
//!
//! A user holding none of a split model hands the request to the machine
//! holding its first layers (`remote_generate::try_delegated_split`), which
//! leads it among its peers and guesses with what IT holds — usually nothing
//! better than prompt lookup. This node often holds a small model of the
//! family whole. It follows the reply as its ids stream in
//! (`StreamingToken::reply_ids`), drafts ahead of it, and sends each window to
//! the head (`SwarmMessage::RequesterGuesses`). The head checks those that go
//! past its reply each pass, and waits for the next window only where it is
//! due soon enough to pay (`ngram_only_spec::worth_waiting_ms`, #318 (a)); a
//! guess is kept only where the target's own sample equals it, so a late or
//! wrong window costs rows, never a token.
//!
//! No published system does this (prior-art search, 2026-10-20); its parts
//! are AMUSD's rollback, PEARL's post-verify drafting and PipeInfer's handling
//! of a stale run. The alternative it replaces is not a co-located drafter —
//! "Speculation at a Distance" (arXiv 2606.25091) argues for one — but a head
//! with none at all.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, watch};

use crate::daemon::state::SharedState;
use crate::types::{
    GuessesAsked, ModelId, NetworkCommand, NodeId, ReplyIds, RequesterGuesses, SamplingParams,
    MAX_REQUESTER_GUESSES,
};

use super::engine_drafter::{self, DrafterSpec, EngineDrafter, TargetHere};

/// How many guesses past the reply this node keeps drafted, in guesses per
/// check: the head checks a check's worth each pass, and a window spends a
/// round trip and a drafting call on its way, while the reply moves on.
const PASSES_AHEAD: u32 = 4;

/// `SWARMLLM_REQUESTER_GUESSES=0`: never offer a head this node's guesses —
/// the other arm of an A/B inside one binary.
fn switched_on() -> bool {
    std::env::var("SWARMLLM_REQUESTER_GUESSES").map_or(true, |v| v.trim() != "0")
}

/// Whether this node offers guesses for a split it hands over. ⚠ Never to a
/// head that does not advertise `features::REQUESTER_GUESSES`: an older head
/// cannot read the message, and would not send the ids the drafter follows.
fn offers(head_takes_them: bool, switched_on: bool, drafter_here: bool) -> bool {
    head_takes_them && switched_on && drafter_here
}

/// The reply's ids as the head sent them, put back together in order. The
/// head's record is append-only, so ids overlapping what is held add nothing;
/// ids past a gap (a token that never arrived) are dropped — the drafter
/// follows the reply whole or not at all. Never more than the reply may hold
/// (`max_tokens`): the ids come off the wire.
struct ReplyIdFeed {
    ids: Vec<u32>,
    limit: usize,
}

impl ReplyIdFeed {
    fn new(limit: usize) -> Self {
        Self {
            ids: Vec::new(),
            limit,
        }
    }

    /// The ids `more` adds to the reply, if any.
    fn apply<'a>(&mut self, more: &'a ReplyIds) -> &'a [u32] {
        let from = more.from as usize;
        let Some(overlap) = self.ids.len().checked_sub(from) else {
            return &[];
        };
        let new = more.ids.get(overlap..).unwrap_or(&[]);
        let new = &new[..new.len().min(self.limit.saturating_sub(self.ids.len()))];
        self.ids.extend_from_slice(new);
        new
    }
}

/// What this node's drafter holds of the reply: the ids the head has sent,
/// then the guesses drafted past them.
#[derive(Default)]
struct Line {
    tokens: Vec<u32>,
    /// How many of `tokens` are the reply's own.
    known: usize,
}

#[derive(Debug, PartialEq, Eq)]
enum Heard {
    /// The reply has not moved.
    Nothing,
    /// The reply moved along the guesses: they still stand.
    Along,
    /// The reply left the guesses (or ran past them): the drafter follows it.
    Away,
}

impl Line {
    fn hear(&mut self, reply: &[u32]) -> Heard {
        if reply.len() <= self.known {
            return Heard::Nothing;
        }
        self.known = reply.len();
        if self.tokens.get(..reply.len()) == Some(reply) {
            return Heard::Along;
        }
        self.tokens = reply.to_vec();
        Heard::Away
    }

    fn ahead(&self) -> usize {
        self.tokens.len() - self.known
    }

    /// The window to send: the guesses past the reply, at most a message's worth.
    fn window(&self, request_id: uuid::Uuid) -> Option<RequesterGuesses> {
        let past = &self.tokens[self.known..];
        (!past.is_empty()).then(|| RequesterGuesses {
            request_id,
            after: self.known as u32,
            tokens: past[..past.len().min(MAX_REQUESTER_GUESSES)].to_vec(),
        })
    }
}

/// The guessing for one hand-off, while it runs: dropping it stops the
/// drafting task (which then releases its drafter's cache), on every way the
/// hand-off ends.
pub(super) struct GuessingAhead {
    stop: Arc<AtomicBool>,
    reply: watch::Sender<Vec<u32>>,
    feed: ReplyIdFeed,
    heard: bool,
}

impl GuessingAhead {
    /// What came with the next reply token, in the reply's order. `false` when
    /// the head is not taking guesses — the reply's FIRST token came without
    /// ids — and the guessing should be dropped: a head that takes them records
    /// each token before it emits it, so its first token always carries one
    /// (security review of #318: a head whose owner turned speculation off
    /// still advertises the bit, and this node drafted for nothing).
    pub(super) fn hear(&mut self, ids: Option<&ReplyIds>) -> bool {
        let Some(ids) = ids else {
            return self.heard;
        };
        self.heard = true;
        let new = self.feed.apply(ids);
        if !new.is_empty() {
            self.reply.send_modify(|reply| reply.extend_from_slice(new));
        }
        true
    }
}

impl Drop for GuessingAhead {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Start guessing for a split handed to `head` under `wire_id`, where this
/// node can: the head takes guesses, the switch is on, this node read the
/// prompt with the target's own tokenizer (`prompt_ids`) and holds a drafter
/// for the target that its memory would load. `None` otherwise, and the
/// hand-off goes as before.
#[allow(clippy::too_many_arguments)]
pub(super) fn start(
    state: &Arc<SharedState>,
    network_tx: &mpsc::Sender<NetworkCommand>,
    head: &NodeId,
    head_bytes: &[u8],
    wire_id: uuid::Uuid,
    target: &ModelId,
    prompt_ids: Option<&[u32]>,
    sampling: &SamplingParams,
    requester: crate::inference::process_pool::Requester,
) -> Option<(GuessesAsked, GuessingAhead)> {
    let ids = prompt_ids?;
    let head_takes_them =
        state.peer_advertises_feature(head, crate::types::features::REQUESTER_GUESSES);
    let on = switched_on();
    // Looked for only where it would be used: the choice weighs memory.
    let spec = (head_takes_them && on)
        .then(|| {
            engine_drafter::engine_drafter_for(
                state,
                target,
                crate::inference::model_worker::drafter_cache_positions(
                    ids.len(),
                    Some(sampling.max_tokens),
                ),
                &TargetHere {
                    ranges: &[],
                    requester,
                },
            )
        })
        .flatten();
    if !offers(head_takes_them, on, spec.is_some()) {
        return None;
    }
    let spec = spec?;
    // Shared noise, as DSD draws it (`dsd.rs`): at a temperature above zero a
    // close drafter then reproduces the target's SAMPLE. The head draws with
    // it only where it read the prompt as this many tokens.
    let coupling_seed = (sampling.temperature > 0.0
        && std::env::var("SWARMLLM_SPEC_COUPLING").as_deref() != Ok("0"))
    .then(rand::random::<u64>);
    let gamma = state.cfg().inference.guesses_per_check();
    let stop = Arc::new(AtomicBool::new(false));
    let (reply_tx, reply_rx) = watch::channel(Vec::new());
    tracing::info!(
        %wire_id,
        head = %head,
        drafter = %spec.model_id,
        coupled = coupling_seed.is_some(),
        "requester guesses: guessing for the split handed to its head"
    );
    tokio::spawn(guess_ahead(
        Drafting {
            state: state.clone(),
            network_tx: network_tx.clone(),
            head_bytes: head_bytes.to_vec(),
            wire_id,
            sampling: sampling.clone(),
            coupling_seed,
            chunk: gamma,
            ahead: (gamma * PASSES_AHEAD).min(MAX_REQUESTER_GUESSES as u32) as usize,
            stop: stop.clone(),
        },
        spec,
        ids.to_vec(),
        reply_rx,
    ));
    Some((
        GuessesAsked {
            coupling_seed,
            prompt_tokens: ids.len() as u32,
        },
        GuessingAhead {
            stop,
            reply: reply_tx,
            feed: ReplyIdFeed::new(sampling.max_tokens as usize),
            heard: false,
        },
    ))
}

struct Drafting {
    state: Arc<SharedState>,
    network_tx: mpsc::Sender<NetworkCommand>,
    head_bytes: Vec<u8>,
    wire_id: uuid::Uuid,
    sampling: SamplingParams,
    coupling_seed: Option<u64>,
    /// Guesses asked of the drafter a call.
    chunk: u32,
    /// Guesses kept drafted past the reply.
    ahead: usize,
    stop: Arc<AtomicBool>,
}

/// Follow the reply and keep `ahead` guesses drafted past it, sending a window
/// after every call. Ends when the hand-off does (its [`GuessingAhead`]
/// dropped) or the drafter fails — the head then goes on without guesses.
async fn guess_ahead(
    d: Drafting,
    spec: DrafterSpec,
    prompt_ids: Vec<u32>,
    mut reply_rx: watch::Receiver<Vec<u32>>,
) {
    let prompt_len = prompt_ids.len();
    let key = engine_drafter::draft_key();
    let mut drafter = EngineDrafter::new(
        spec.clone(),
        key,
        prompt_ids.clone(),
        engine_drafter::release_of(d.state.clone(), key),
    );
    // The drafter reads the prompt while the head does.
    let read = engine_drafter::read_ahead(
        d.state.clone(),
        &spec,
        key,
        prompt_ids,
        &d.sampling,
        Some(d.stop.clone()),
    );
    if let Some(Ok(Ok(_))) = read.finish().await {
        drafter.prompt_read();
    }

    let mut line = Line::default();
    // The drafter stopped short of what it was asked (not confident of the
    // next token): drafting on from the same place would stop there again, so
    // it waits for the reply to leave its guesses.
    let mut stalled = false;
    let (mut calls, mut drafted, mut windows) = (0u32, 0usize, 0u32);
    while !d.stop.load(Ordering::Relaxed) {
        let reply = reply_rx.borrow_and_update().clone();
        if line.hear(&reply) == Heard::Away {
            drafter.follow(prompt_len, &reply);
            stalled = false;
        }
        if line.known == 0 || stalled || line.ahead() >= d.ahead {
            if reply_rx.changed().await.is_err() {
                break;
            }
            continue;
        }
        let want = d.chunk.min((d.ahead - line.ahead()) as u32);
        calls += 1;
        let guesses = match drafter
            .draft(
                &d.state,
                want,
                &d.sampling,
                &line.tokens,
                d.coupling_seed,
                Some(d.stop.clone()),
            )
            .await
        {
            Ok(g) => g,
            Err(e) => {
                if !d.stop.load(Ordering::Relaxed) {
                    tracing::info!(
                        wire_id = %d.wire_id,
                        error = %e,
                        "requester guesses: the drafter failed — the head goes on without guesses"
                    );
                }
                break;
            }
        };
        drafter.settle(guesses.len());
        drafter.push(&guesses);
        line.tokens.extend_from_slice(&guesses);
        drafted += guesses.len();
        stalled = guesses.len() < want as usize;
        let Some(window) = line.window(d.wire_id) else {
            continue;
        };
        windows += 1;
        let sent = d
            .network_tx
            .send(NetworkCommand::SendDirectMessage {
                target_peer_bytes: d.head_bytes.clone(),
                message: crate::types::SwarmMessage::RequesterGuesses(window),
                delivery_request_id: None,
            })
            .await;
        if sent.is_err() {
            break;
        }
    }
    tracing::info!(
        wire_id = %d.wire_id,
        reply_ids = line.known,
        calls,
        drafted,
        windows,
        "requester guesses: done"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(from: u32, ids: &[u32]) -> ReplyIds {
        ReplyIds {
            from,
            ids: ids.to_vec(),
        }
    }

    /// The reply is put together in order: a resend overlapping what is held
    /// adds only what is new, and ids past a gap are dropped rather than
    /// joined on as if they followed.
    #[test]
    fn the_reply_ids_are_put_together_in_order() {
        let mut feed = ReplyIdFeed::new(100);
        assert_eq!(feed.apply(&ids(0, &[1, 2])), &[1, 2]);
        assert_eq!(feed.apply(&ids(1, &[2, 3])), &[3]);
        assert!(feed.apply(&ids(0, &[1])).is_empty());
        assert!(feed.apply(&ids(5, &[9])).is_empty());
        assert_eq!(feed.ids, vec![1, 2, 3]);
    }

    /// A head that takes guesses records each token before it emits it, so its
    /// first token carries ids; one that does not take them never sends any,
    /// and the guessing stops at its first token. A LATER token without ids is
    /// only one whose ids went with an earlier token.
    #[test]
    fn a_head_whose_first_token_carries_no_ids_is_not_guessed_for() {
        let guessing = || GuessingAhead {
            stop: Arc::new(AtomicBool::new(false)),
            reply: watch::channel(Vec::new()).0,
            feed: ReplyIdFeed::new(16),
            heard: false,
        };
        assert!(!guessing().hear(None));
        let mut g = guessing();
        assert!(g.hear(Some(&ids(0, &[1, 2, 3]))));
        assert!(g.hear(None));
        assert_eq!(*g.reply.borrow(), vec![1, 2, 3]);
    }

    /// A head cannot grow the reply past what it may hold: ids past
    /// `max_tokens` are dropped.
    #[test]
    fn the_reply_ids_stop_at_the_replys_length() {
        let mut feed = ReplyIdFeed::new(3);
        assert_eq!(feed.apply(&ids(0, &[1, 2])), &[1, 2]);
        assert_eq!(feed.apply(&ids(2, &[3, 4, 5])), &[3]);
        assert!(feed.apply(&ids(3, &[6])).is_empty());
        assert_eq!(feed.ids, vec![1, 2, 3]);
    }

    /// A reply that moves along the guesses keeps them, and the window then
    /// starts where the reply has reached; one that leaves them drops them.
    #[test]
    fn guesses_the_reply_agrees_with_are_kept_and_the_rest_dropped() {
        let id = uuid::Uuid::new_v4();
        let mut line = Line::default();
        assert_eq!(line.hear(&[1]), Heard::Away);
        line.tokens.extend_from_slice(&[2, 3, 4]);
        assert_eq!(
            line.window(id).map(|w| (w.after, w.tokens)),
            Some((1, vec![2, 3, 4]))
        );

        assert_eq!(line.hear(&[1, 2]), Heard::Along);
        assert_eq!(line.ahead(), 2);
        assert_eq!(
            line.window(id).map(|w| (w.after, w.tokens)),
            Some((2, vec![3, 4]))
        );
        assert_eq!(line.hear(&[1, 2]), Heard::Nothing);

        assert_eq!(line.hear(&[1, 2, 7]), Heard::Away);
        assert_eq!((line.tokens.clone(), line.ahead()), (vec![1, 2, 7], 0));
        assert!(line.window(id).is_none());
    }

    /// A window never carries more than a head takes: past the cap it is cut,
    /// not sent whole to be dropped unread.
    #[test]
    fn a_window_is_cut_to_what_a_head_takes() {
        let mut line = Line::default();
        line.hear(&[1]);
        line.tokens
            .extend(std::iter::repeat_n(5, MAX_REQUESTER_GUESSES + 3));
        let w = line.window(uuid::Uuid::new_v4()).unwrap();
        assert_eq!(w.tokens.len(), MAX_REQUESTER_GUESSES);
    }

    /// Guesses go only to a head that takes them, and only with the switch on
    /// and a drafter here.
    #[test]
    fn a_head_that_does_not_take_guesses_is_never_offered_them() {
        assert!(offers(true, true, true));
        assert!(!offers(false, true, true));
        assert!(!offers(true, false, true));
        assert!(!offers(true, true, false));
    }
}
