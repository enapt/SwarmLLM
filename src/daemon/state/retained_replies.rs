//! Serving-side memory of a fast-path reply's tokens, so the requester can ask
//! for the ones that never arrived (gotcha #438).
//!
//! Every content token of a remote-generate reply is its own fire-and-forget
//! `request_response` send, and the transport orders nothing between them —
//! so the requester reassembles by `token_id` and releases only the
//! consecutive run. That is correct, and it means ONE lost send strands every
//! token after it: measured on 2026-08-20 as ten replies from one WAN peer
//! that arrived as `emitted=3 missing=57 buffered=17` — a hole early, the rest
//! of the reply sitting in a buffer behind it until a 15 s deadline gave up.
//!
//! The serving node now keeps what it sent for a short while and answers
//! `SwarmMessage::ResendTokens` with the range asked for, plus the terminal
//! token once the reply has finished, since that is the frame the requester
//! is waiting on. Bounded three ways — replies retained, tokens per reply,
//! resends per reply — because a peer can ask for anything, and swept on the
//! health tick.
//!
//! **A resend goes only to the peer the reply was for.** The requester's peer
//! id is recorded when retention starts and every ask is checked against it;
//! a third node asking for someone else's reply gets nothing. The tokens are
//! model output, which the requester is entitled to and nobody else is.

use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::types::StreamingToken;

/// Replies kept at once. Oldest-by-activity is evicted when a new one starts
/// past this. Sized for a node serving many peers, bounded so a burst of
/// requests cannot turn retention into a memory hog: at the per-reply cap
/// below and ~80 bytes a token this is tens of MB in the worst case and a
/// few KB in the ordinary one.
pub(crate) const MAX_RETAINED_REPLIES: usize = 64;
/// Content tokens kept per reply. Matches the largest reply the API admits;
/// anything past it is simply not resendable.
pub(crate) const MAX_RETAINED_TOKENS_PER_REPLY: usize = 8192;
/// Longest run of tokens one ask may name. A hole is a few tokens wide; a
/// requester asking for hundreds is asking for the reply again.
pub(crate) const MAX_RESEND_SPAN: u32 = 256;
/// Asks answered per reply before further ones are refused.
pub(crate) const MAX_RESENDS_PER_REPLY: u32 = 32;
/// How long a reply stays after its last activity (a token sent, an ask
/// answered). Longer than the requester's straggler wait plus its resend
/// budget, so an ask arriving late in a slow request still finds its tokens.
pub(crate) const RETAINED_REPLY_TTL: Duration = Duration::from_secs(120);

/// One reply's tokens, in send order. `tokens[i].token_id == i` by
/// construction on the serving side, which is what makes a range ask a slice.
#[derive(Debug)]
pub(crate) struct RetainedReply {
    /// The requester, as libp2p peer-id bytes — the only destination a resend
    /// is ever sent to.
    pub(crate) target_peer_bytes: Vec<u8>,
    tokens: Vec<StreamingToken>,
    terminal: Option<StreamingToken>,
    last_activity: Instant,
    resends_served: u32,
}

/// What an ask was answered with, for the caller to log and act on.
#[derive(Debug)]
pub(crate) enum ResendOutcome {
    /// Tokens to send, in order, followed by the terminal token when the
    /// reply has finished. May be empty if the range named nothing we hold.
    Send {
        tokens: Vec<StreamingToken>,
        terminal: Option<StreamingToken>,
    },
    /// No reply retained under that id: it expired, was evicted, or was never
    /// ours.
    UnknownRequest,
    /// The asker is not the peer this reply is for.
    WrongPeer,
    /// This reply has been asked about too often.
    TooManyAsks,
}

/// Every reply this node is currently prepared to resend from.
#[derive(Debug, Default)]
pub(crate) struct RetainedReplies {
    map: DashMap<uuid::Uuid, RetainedReply>,
}

impl RetainedReplies {
    /// Begin retaining a reply for `request_id`, going to `target_peer_bytes`.
    /// Evicts the least recently active replies if the cap is reached.
    pub(crate) fn start(&self, request_id: uuid::Uuid, target_peer_bytes: Vec<u8>) {
        while self.map.len() >= MAX_RETAINED_REPLIES {
            let oldest = self
                .map
                .iter()
                .min_by_key(|e| e.value().last_activity)
                .map(|e| *e.key());
            match oldest {
                Some(id) => {
                    self.map.remove(&id);
                }
                None => break,
            }
        }
        self.map.insert(
            request_id,
            RetainedReply {
                target_peer_bytes,
                tokens: Vec::new(),
                terminal: None,
                last_activity: Instant::now(),
                resends_served: 0,
            },
        );
    }

    /// Remember a content token as sent. A no-op for a reply not being
    /// retained, and past the per-reply cap.
    pub(crate) fn push(&self, request_id: uuid::Uuid, token: &StreamingToken) {
        if let Some(mut r) = self.map.get_mut(&request_id) {
            if r.tokens.len() < MAX_RETAINED_TOKENS_PER_REPLY {
                debug_assert_eq!(
                    r.tokens.len() as u32,
                    token.token_id,
                    "retained tokens must be pushed in sequence order"
                );
                r.tokens.push(token.clone());
            }
            r.last_activity = Instant::now();
        }
    }

    /// Remember the terminal token, so a late ask also learns the reply ended.
    pub(crate) fn finish(&self, request_id: uuid::Uuid, terminal: &StreamingToken) {
        if let Some(mut r) = self.map.get_mut(&request_id) {
            r.terminal = Some(terminal.clone());
            r.last_activity = Instant::now();
        }
    }

    /// Answer an ask for `[from, to)` of `request_id` from `asker_peer_bytes`.
    pub(crate) fn resend(
        &self,
        request_id: uuid::Uuid,
        from: u32,
        to: u32,
        asker_peer_bytes: &[u8],
    ) -> ResendOutcome {
        let Some(mut r) = self.map.get_mut(&request_id) else {
            return ResendOutcome::UnknownRequest;
        };
        if r.target_peer_bytes != asker_peer_bytes {
            return ResendOutcome::WrongPeer;
        }
        if r.resends_served >= MAX_RESENDS_PER_REPLY {
            return ResendOutcome::TooManyAsks;
        }
        r.resends_served += 1;
        r.last_activity = Instant::now();
        let held = r.tokens.len() as u32;
        let start = from.min(held);
        let end = to.min(held).min(start.saturating_add(MAX_RESEND_SPAN));
        let tokens = r.tokens[start as usize..end as usize].to_vec();
        ResendOutcome::Send {
            tokens,
            terminal: r.terminal.clone(),
        }
    }

    /// Drop replies idle past `ttl`. Returns how many went.
    pub(crate) fn sweep(&self, ttl: Duration) -> usize {
        let before = self.map.len();
        let now = Instant::now();
        self.map
            .retain(|_, r| now.duration_since(r.last_activity) < ttl);
        before - self.map.len()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(id: uuid::Uuid, n: u32) -> StreamingToken {
        StreamingToken {
            request_id: id,
            token_id: n,
            finish_reason: None,
            text: format!("t{n}"),
            usage: None,
            matched_stop_sequence: None,
            logprob: None,
        }
    }

    fn done(id: uuid::Uuid, n: u32) -> StreamingToken {
        StreamingToken {
            finish_reason: Some(crate::types::NetworkFinishReason::Stop),
            text: String::new(),
            ..tok(id, n)
        }
    }

    #[test]
    fn a_hole_is_answered_with_exactly_the_range_asked_for() {
        let r = RetainedReplies::default();
        let id = uuid::Uuid::new_v4();
        r.start(id, vec![1, 2, 3]);
        for n in 0..10 {
            r.push(id, &tok(id, n));
        }
        match r.resend(id, 3, 6, &[1, 2, 3]) {
            ResendOutcome::Send { tokens, terminal } => {
                let ids: Vec<u32> = tokens.iter().map(|t| t.token_id).collect();
                assert_eq!(ids, vec![3, 4, 5]);
                assert!(terminal.is_none(), "the reply has not finished");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_finished_reply_resends_its_terminal_token_too() {
        let r = RetainedReplies::default();
        let id = uuid::Uuid::new_v4();
        r.start(id, vec![9]);
        for n in 0..4 {
            r.push(id, &tok(id, n));
        }
        r.finish(id, &done(id, 4));
        match r.resend(id, 2, 100, &[9]) {
            ResendOutcome::Send { tokens, terminal } => {
                assert_eq!(tokens.len(), 2, "clamped to what is held");
                assert_eq!(terminal.map(|t| t.token_id), Some(4));
            }
            other => panic!("{other:?}"),
        }
    }

    /// Model output goes to the peer that asked for the reply and nobody else.
    #[test]
    fn only_the_requester_is_answered() {
        let r = RetainedReplies::default();
        let id = uuid::Uuid::new_v4();
        r.start(id, vec![1]);
        r.push(id, &tok(id, 0));
        assert!(matches!(r.resend(id, 0, 1, &[2]), ResendOutcome::WrongPeer));
        assert!(matches!(
            r.resend(uuid::Uuid::new_v4(), 0, 1, &[1]),
            ResendOutcome::UnknownRequest
        ));
    }

    #[test]
    fn asks_are_bounded_in_span_and_in_number() {
        let r = RetainedReplies::default();
        let id = uuid::Uuid::new_v4();
        r.start(id, vec![1]);
        for n in 0..(MAX_RESEND_SPAN + 50) {
            r.push(id, &tok(id, n));
        }
        match r.resend(id, 0, u32::MAX, &[1]) {
            ResendOutcome::Send { tokens, .. } => {
                assert_eq!(tokens.len() as u32, MAX_RESEND_SPAN)
            }
            other => panic!("{other:?}"),
        }
        for _ in 1..MAX_RESENDS_PER_REPLY {
            assert!(matches!(
                r.resend(id, 0, 1, &[1]),
                ResendOutcome::Send { .. }
            ));
        }
        assert!(matches!(
            r.resend(id, 0, 1, &[1]),
            ResendOutcome::TooManyAsks
        ));
    }

    #[test]
    fn retention_is_capped_and_swept() {
        let r = RetainedReplies::default();
        let first = uuid::Uuid::new_v4();
        r.start(first, vec![1]);
        for _ in 0..MAX_RETAINED_REPLIES {
            r.start(uuid::Uuid::new_v4(), vec![1]);
        }
        assert_eq!(r.len(), MAX_RETAINED_REPLIES);
        assert!(
            matches!(r.resend(first, 0, 1, &[1]), ResendOutcome::UnknownRequest),
            "the oldest reply is the one evicted"
        );
        assert_eq!(r.sweep(Duration::from_secs(3600)), 0);
        assert_eq!(r.sweep(Duration::ZERO), MAX_RETAINED_REPLIES);
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn tokens_past_the_per_reply_cap_are_not_kept() {
        let r = RetainedReplies::default();
        let id = uuid::Uuid::new_v4();
        r.start(id, vec![1]);
        for n in 0..(MAX_RETAINED_TOKENS_PER_REPLY as u32 + 5) {
            r.push(id, &tok(id, n));
        }
        let held = match r.resend(
            id,
            MAX_RETAINED_TOKENS_PER_REPLY as u32 - 1,
            MAX_RETAINED_TOKENS_PER_REPLY as u32 + 5,
            &[1],
        ) {
            ResendOutcome::Send { tokens, .. } => tokens.len(),
            other => panic!("{other:?}"),
        };
        assert_eq!(held, 1);
    }
}
