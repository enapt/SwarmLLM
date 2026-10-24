//! The owner's earlier replies, for a copied run to find (FUTURE_WORK #337).
//!
//! A copied run (`ngram_lookup::copied_run`) looks in the prompt and the reply
//! so far. An agent's opening calls repeat its EARLIER sessions, which neither
//! holds: measured offline on 400 SWE-agent sessions, a store of earlier
//! replies took the share of reply tokens a copy predicts on a session's first
//! three replies from 28% to 60%, and over whole sessions from 64.6% to 73.3%
//! (`docs/plans/faster_than_local.md` § 7.6). SuffixDecoding (2411.04975) keeps
//! the same kind of store; llama.cpp's lookup decoding keeps one across runs.
//!
//! What it may hold and who may read it is the point of this module:
//!
//! - **Only the owner's replies, read only for the owner.** [`OwnerReplies::for_the_owner`]
//!   is the one way in, and asks the request: the owner asked
//!   (`InferenceRequest::requested_by`, read from the headers at the API) and
//!   this node is not leading the split for another node (`worker_requester`).
//!   A store's matches show in a reply's SPEED — REST's datastore leaked
//!   through it at over 25 tokens a second (2411.01076) — so a reply for
//!   anyone else neither reads it nor is kept in it. "The owner" is whoever
//!   presents this node's key directly: in a pool every device shares the
//!   key, so a pool member asking THIS node directly is its owner here — and
//!   could already read its stored Responses replies with that key.
//! - **A peer is never shown a copy of text it has not already seen.** A
//!   copied guess is SENT: the first layers run it, the tail checks or draws
//!   it. Each reply is kept with the peers that served it — every one of its
//!   tokens crossed them — and a run is copied from it only into a split whose
//!   peers are ALL among those ([`Exposure`]). A reply's last token is not
//!   kept: the tail drew it, the peers before the tail may never have been
//!   sent it. **And in the form they saw it**: a peer is sent a guess's TOKEN
//!   IDS where it runs the model's first layers or walks the guesses (a
//!   chained check carries them through every hop), hidden states otherwise. A
//!   run goes as ids to a peer only from a reply that peer certainly knew as
//!   ids — its first peer was sent them, its tail drew them — never one it was
//!   sent as hidden states (prompt privacy treats ids leaving as worse than
//!   states, `local_embedding_privacy`). Each path says who sees what, for a
//!   guess it would send (naming more is safe) and for the reply it keeps
//!   (naming fewer is safe): `dsd_flow::flow_exposure`,
//!   `PipelineExecutor::exposure_of_a_check`.
//! - **Never saved, never shared, never announced**: memory only, bounded per
//!   model ([`KEPT_PER_MODEL`] tokens, [`MOST_REPLIES`] replies) and in models
//!   ([`MODELS_KEPT`]).
//!
//! The lookup is llama.cpp's for its cross-run cache (`common/ngram-cache.cpp`),
//! at its LAX thresholds: the last [`HITS_WEIGHED`] occurrences of the context's
//! last 4, 3, then 2 tokens vote on the next token, and a run is copied only
//! where the leader holds half of them — and for 2 tokens, only where the
//! bigram was seen twice. A longer n-gram that fails the vote ends the lookup;
//! it does not fall back to a shorter one. Measured (`combo.py`, 2026-10-24):
//! the vote cut the store's rows 24% for 0.2 points of coverage; ending a run
//! where the hits stop agreeing (#338's cut, which pays on the conversation's
//! own text) saved no rows here and cost 0.5 points, so the run is today's:
//! the most recent leading hit's continuation, as long as its match goes back.

use std::collections::VecDeque;
use std::sync::{Arc, OnceLock};

use dashmap::DashMap;

use crate::inference::ngram_lookup::{CopiedRun, NgramHitSource, NgramLookupConfig, MATCHED_MOST};
use crate::types::{InferenceRequest, NodeId, RequestedBy};

/// Tokens kept per model. 256K kept 96% of an unbounded store's gain on the
/// 1.18M reply tokens of the 400 sessions (11.2 of 11.7 points), 128K less;
/// 4 MB a model with its index, 1.5 MB of tables more.
pub const KEPT_PER_MODEL: usize = 256 * 1024;

/// Replies kept per model, however short: the replies' own records stay
/// bounded too (400-token replies fill the store at ~650).
pub const MOST_REPLIES: usize = 4096;

/// Models kept; the one written longest ago goes first.
pub const MODELS_KEPT: usize = 4;

/// Occurrences of an n-gram that vote: the most recent. 32 and 64 measured
/// the same (73.3%).
const HITS_WEIGHED: usize = 32;

/// Links of an n-gram's chain walked in one lookup, matching or not — the
/// lookup's cost bound (zlib's `max_chain`).
const WALK_MOST: usize = 256;

/// The n-gram sizes indexed. The lookup's own config chooses among them.
const SMALLEST_N: usize = 2;
const LARGEST_N: usize = 4;

/// Slots in each n-gram size's table of most recent starts.
const SLOT_BITS: u32 = 17;

/// Between two replies: no token id is this, so no match runs across it and
/// no copy runs past a reply's end.
const BETWEEN: u32 = u32::MAX;

/// No earlier start.
const NONE: u32 = u32::MAX;

/// `SWARMLLM_FLOW_COPY_STORE=0|off|false`: neither read nor keep the owner's
/// earlier replies, for an A/B inside one binary. Read once.
fn store_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_FLOW_COPY_STORE").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// Every model's store (`state.models.owner_replies`).
#[derive(Default)]
pub struct OwnerReplies {
    by_model: DashMap<String, ModelReplies>,
    writes: std::sync::atomic::AtomicU64,
}

impl OwnerReplies {
    /// The store for `request`, or `None` where the request may neither read
    /// nor be kept in it: not the owner's, or this node leading the split for
    /// another (module doc). The ONLY way to a store's contents.
    pub fn for_the_owner(self: &Arc<Self>, request: &InferenceRequest) -> Option<EarlierReplies> {
        let owners = request.requested_by == RequestedBy::Owner
            && crate::inference::pipeline::worker_requester(request)
                == crate::inference::process_pool::Requester::Owner;
        if !owners || !store_on() {
            return None;
        }
        Some(EarlierReplies {
            store: Arc::clone(self),
            model: request.model_id.0.clone(),
        })
    }

    fn keep(&self, model: &str, reply: &[u32], seen: &Exposure) {
        // The last token only the tail is sure to have seen (module doc).
        let Some((_, kept)) = reply.split_last() else {
            return;
        };
        if kept.len() < SMALLEST_N {
            return;
        }
        let written = self
            .writes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        {
            let mut store = self
                .by_model
                .entry(model.to_string())
                .or_insert_with(ModelReplies::new);
            store.written = written;
            store.append(kept, seen);
        }
        // After the write, with no entry held (an entry guard and a read of
        // the same map deadlock, gotcha #10): writers racing to two new models
        // still leave at most MODELS_KEPT.
        while self.by_model.len() > MODELS_KEPT {
            let oldest = self
                .by_model
                .iter()
                .filter(|e| e.key() != model)
                .min_by_key(|e| e.value().written)
                .map(|e| e.key().clone());
            match oldest {
                Some(oldest) => {
                    self.by_model.remove(&oldest);
                }
                None => break,
            }
        }
    }

    fn copied_run(
        &self,
        model: &str,
        context: &[u32],
        shown: &Exposure,
        cfg: NgramLookupConfig,
        most: usize,
    ) -> Option<CopiedRun> {
        self.by_model
            .get(model)?
            .copied_run(context, shown, cfg, most)
    }
}

/// Which peers a reply — or a guess — reaches, and which of them as token ids.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Exposure {
    /// Sorted, deduplicated; `ids` among them.
    seen_by: Arc<[NodeId]>,
    /// Sorted, deduplicated.
    ids: Arc<[NodeId]>,
}

impl Exposure {
    /// `peers` are reached, `ids` of them as token ids (each of `ids` is
    /// reached too, named among `peers` or not).
    pub fn new(
        peers: impl IntoIterator<Item = NodeId>,
        ids: impl IntoIterator<Item = NodeId>,
    ) -> Self {
        let sorted = |mut v: Vec<NodeId>| -> Arc<[NodeId]> {
            v.sort_unstable_by_key(|p| p.0);
            v.dedup();
            v.into()
        };
        let ids: Vec<NodeId> = ids.into_iter().collect();
        let peers: Vec<NodeId> = peers.into_iter().chain(ids.iter().cloned()).collect();
        Self {
            seen_by: sorted(peers),
            ids: sorted(ids),
        }
    }

    /// The peers reached.
    pub fn peers(&self) -> &[NodeId] {
        &self.seen_by
    }

    /// Whether a run copied from a reply seen as `self` may be sent as
    /// `shown`: every peer it would reach saw the reply, and every peer it
    /// would reach as ids knew the reply as ids.
    fn covers(&self, shown: &Exposure) -> bool {
        let within = |few: &[NodeId], all: &[NodeId]| {
            few.iter()
                .all(|p| all.binary_search_by_key(&p.0, |s| s.0).is_ok())
        };
        within(&shown.seen_by, &self.seen_by) && within(&shown.ids, &self.ids)
    }
}

/// One owner's request's view of the store. Built only by
/// [`OwnerReplies::for_the_owner`].
pub struct EarlierReplies {
    store: Arc<OwnerReplies>,
    model: String,
}

impl EarlierReplies {
    /// What followed the context's last tokens in an earlier reply that may
    /// be sent as `shown`, at most `most` tokens; `None` where nothing matched
    /// or the earlier replies did not agree (module doc). `shown` names every
    /// peer the run may reach and every one it may reach as ids: naming MORE
    /// than will be is safe, fewer is the leak.
    pub fn copied_run(
        &self,
        context: &[u32],
        shown: &Exposure,
        cfg: NgramLookupConfig,
        most: usize,
    ) -> Option<CopiedRun> {
        self.store
            .copied_run(&self.model, context, shown, cfg, most)
    }

    /// Keep this reply's tokens as `seen`: the peers it reached, and those
    /// CERTAIN to know it as ids — naming FEWER than did is safe, more is the
    /// leak.
    pub fn keep(&self, reply: &[u32], seen: &Exposure) {
        self.store.keep(&self.model, reply, seen);
    }
}

/// One kept reply: where it lies and who was shown it.
struct Reply {
    start: u32,
    /// Its separator's position: one past its last token.
    end: u32,
    shown: Exposure,
}

/// One model's kept replies, indexed by n-gram as LZ77 indexes its window: a
/// table of each hash's most recent start, and for every start the one before
/// it with the same hash. Positions are absolute and only grow; one before
/// `base` has been dropped, and a walk stops there.
struct ModelReplies {
    base: u32,
    tokens: VecDeque<u32>,
    /// Per size (`n - SMALLEST_N`): each position's earlier start with the
    /// same hash, beside `tokens`.
    earlier: Vec<VecDeque<u32>>,
    /// Per size: each hash's most recent start.
    latest: Vec<Vec<u32>>,
    replies: VecDeque<Reply>,
    written: u64,
}

fn slot(gram: impl Iterator<Item = u32>) -> usize {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for t in gram {
        h = (h ^ u64::from(t)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    // The high bits of a multiply mix best.
    (h.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> (64 - SLOT_BITS)) as usize
}

impl ModelReplies {
    fn new() -> Self {
        let sizes = LARGEST_N - SMALLEST_N + 1;
        Self {
            base: 0,
            tokens: VecDeque::new(),
            earlier: (0..sizes).map(|_| VecDeque::new()).collect(),
            latest: (0..sizes).map(|_| vec![NONE; 1 << SLOT_BITS]).collect(),
            replies: VecDeque::new(),
            written: 0,
        }
    }

    fn end(&self) -> u64 {
        u64::from(self.base) + self.tokens.len() as u64
    }

    fn append(&mut self, reply: &[u32], shown: &Exposure) {
        // A reply longer than the store keeps its end.
        let reply = &reply[reply.len().saturating_sub(KEPT_PER_MODEL - 1)..];
        while !self.replies.is_empty()
            && (self.tokens.len() + reply.len() + 1 > KEPT_PER_MODEL
                || self.replies.len() >= MOST_REPLIES)
        {
            self.drop_oldest();
        }
        // Positions are u32: past ~4 billion kept tokens, start again.
        if self.end() + reply.len() as u64 + 1 >= u64::from(NONE) {
            let written = self.written;
            *self = Self::new();
            self.written = written;
        }
        let start = self.end() as u32;
        // Replies of one split share one record of its peers.
        let shown = match self.replies.back() {
            Some(last) if last.shown == *shown => last.shown.clone(),
            _ => shown.clone(),
        };
        for &t in reply.iter().chain(std::iter::once(&BETWEEN)) {
            self.tokens.push_back(t);
            for chain in &mut self.earlier {
                chain.push_back(NONE);
            }
        }
        for n in SMALLEST_N..=LARGEST_N.min(reply.len()) {
            let k = n - SMALLEST_N;
            for s in 0..=reply.len() - n {
                let at = start + s as u32;
                let h = slot(reply[s..s + n].iter().copied());
                self.earlier[k][(at - self.base) as usize] = self.latest[k][h];
                self.latest[k][h] = at;
            }
        }
        self.replies.push_back(Reply {
            start,
            end: start + reply.len() as u32,
            shown,
        });
    }

    fn drop_oldest(&mut self) {
        let Some(oldest) = self.replies.pop_front() else {
            return;
        };
        let gone = (oldest.end + 1 - self.base) as usize;
        self.tokens.drain(..gone);
        for chain in &mut self.earlier {
            chain.drain(..gone);
        }
        self.base = oldest.end + 1;
    }

    fn reply_at(&self, at: u32) -> Option<&Reply> {
        let i = self.replies.partition_point(|r| r.start <= at);
        self.replies.get(i.checked_sub(1)?)
    }

    fn copied_run(
        &self,
        context: &[u32],
        split: &Exposure,
        cfg: NgramLookupConfig,
        most: usize,
    ) -> Option<CopiedRun> {
        let largest = cfg.max_ngram_size.min(LARGEST_N).min(context.len());
        let smallest = cfg.min_ngram_size.max(SMALLEST_N);
        if most == 0 || largest < smallest {
            return None;
        }
        for n in (smallest..=largest).rev() {
            let needle = &context[context.len() - n..];
            let hits = self.hits(needle, split);
            if hits.is_empty() {
                continue;
            }
            let (leader, votes) = leading_next(&self.tokens, &hits, n);
            let (least, share_pct) = if n == SMALLEST_N { (2, 50) } else { (1, 50) };
            if hits.len() < least || votes * 100 < share_pct * hits.len() {
                return None;
            }
            let i = *hits.iter().find(|&&i| self.tokens[i + n] == leader)?;
            let mut matched = n;
            while matched < MATCHED_MOST
                && matched < context.len()
                && i > matched - n
                && self.tokens[i - (matched - n) - 1] == context[context.len() - matched - 1]
            {
                matched += 1;
            }
            let tokens: Vec<u32> = self
                .tokens
                .range(i + n..)
                .take_while(|&&t| t != BETWEEN)
                .take(most)
                .copied()
                .collect();
            return Some(CopiedRun {
                tokens,
                matched,
                source: NgramHitSource::EarlierReply,
            });
        }
        None
    }

    /// The most recent occurrences of `needle` with a token after them, in
    /// replies `split` may be shown (`Exposure::covers`): their indices into
    /// `tokens`, newest first.
    fn hits(&self, needle: &[u32], split: &Exposure) -> Vec<usize> {
        let n = needle.len();
        let k = n - SMALLEST_N;
        let mut at = self.latest[k][slot(needle.iter().copied())];
        let mut hits = Vec::new();
        let mut walked = 0;
        while at != NONE && at >= self.base && walked < WALK_MOST && hits.len() < HITS_WEIGHED {
            walked += 1;
            let i = (at - self.base) as usize;
            let next = self.earlier[k][i];
            let same = (0..n).all(|j| self.tokens[i + j] == needle[j]);
            // Every reply ends in `BETWEEN`, so `i + n` is in range.
            if same
                && self.tokens[i + n] != BETWEEN
                && self.reply_at(at).is_some_and(|r| r.shown.covers(split))
            {
                hits.push(i);
            }
            at = next;
        }
        hits
    }
}

/// The token most of `hits` continue with and how many do; a tie goes to the
/// one the most recent hit continues with.
fn leading_next(tokens: &VecDeque<u32>, hits: &[usize], n: usize) -> (u32, usize) {
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for &i in hits {
        let next = tokens[i + n];
        match counts.iter_mut().find(|(t, _)| *t == next) {
            Some((_, c)) => *c += 1,
            None => counts.push((next, 1)),
        }
    }
    // `max_by_key` keeps the LAST of equal keys; the first seen is the newest.
    counts
        .iter()
        .rev()
        .max_by_key(|(_, c)| *c)
        .copied()
        .unwrap_or((BETWEEN, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{KeepAsked, ModelId, SamplingParams};

    fn peer(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    fn request(by: RequestedBy) -> InferenceRequest {
        InferenceRequest::local(
            ModelId("m".into()),
            Vec::new(),
            SamplingParams::default(),
            true,
            None,
            None,
            None,
            KeepAsked::Unasked,
            by,
        )
    }

    fn cfg() -> NgramLookupConfig {
        NgramLookupConfig::default()
    }

    fn owner(store: &Arc<OwnerReplies>) -> EarlierReplies {
        store.for_the_owner(&request(RequestedBy::Owner)).unwrap()
    }

    /// Peers `peers` reached, `ids` of them as token ids.
    fn via(peers: &[u8], ids: &[u8]) -> Exposure {
        Exposure::new(peers.iter().map(|&b| peer(b)), ids.iter().map(|&b| peer(b)))
    }

    /// The case it exists for: a new conversation's text matches an earlier
    /// reply's, and the earlier reply's continuation is copied — through the
    /// same peers that served it.
    #[test]
    fn an_earlier_reply_is_copied_into_a_split_through_the_peers_that_saw_it() {
        let store = Arc::new(OwnerReplies::default());
        let earlier = owner(&store);
        earlier.keep(&[10, 11, 12, 13, 14, 15, 16, 17, 99], &via(&[1, 2], &[]));
        let run = earlier
            .copied_run(&[5, 6, 11, 12, 13], &via(&[2, 1], &[]), cfg(), 32)
            .unwrap();
        assert_eq!(
            run.tokens,
            vec![14, 15, 16, 17],
            "the reply's last token is not kept"
        );
        assert_eq!(run.matched, 3);
        assert_eq!(run.source, NgramHitSource::EarlierReply);
        // A split through ONE of them: it has seen the reply.
        assert!(earlier
            .copied_run(&[11, 12, 13], &via(&[2], &[]), cfg(), 32)
            .is_some());
    }

    /// The privacy rule: a split with a peer that did not serve the earlier
    /// reply is copied nothing from it.
    #[test]
    fn a_peer_is_never_shown_a_copy_of_a_reply_it_did_not_serve() {
        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        s.keep(&[10, 11, 12, 13, 14, 15, 99], &via(&[1], &[]));
        for peers in [vec![3], vec![1, 3]] {
            assert_eq!(
                s.copied_run(&[11, 12, 13], &via(&peers, &[]), cfg(), 32),
                None,
                "{peers:?}"
            );
        }
        // The hit it was refused is still there for the split that saw it.
        assert!(s
            .copied_run(&[11, 12, 13], &via(&[1], &[]), cfg(), 32)
            .is_some());
    }

    /// Only the owner's request reads or is kept: a pool member's forward, a
    /// peer's hand-off, and a split this node leads for another node all get
    /// nothing to hold.
    #[test]
    fn only_the_owners_request_reaches_the_store() {
        let store = Arc::new(OwnerReplies::default());
        assert!(store
            .for_the_owner(&request(RequestedBy::NotTheOwner))
            .is_none());
        let mut led = request(RequestedBy::Owner);
        led.route_override = Some(crate::inference::route_override::RoutePlanOverride {
            lead_here: true,
            ..Default::default()
        });
        assert!(store.for_the_owner(&led).is_none());
        assert!(store.for_the_owner(&request(RequestedBy::Owner)).is_some());
    }

    /// llama.cpp's lax vote: a bigram seen once copies nothing, twice with the
    /// same next token copies; earlier replies split evenly between two next
    /// tokens still copy (half is enough); a longer n-gram whose vote fails
    /// ends the lookup rather than falling back to a shorter one.
    #[test]
    fn earlier_replies_are_copied_from_only_where_they_agree() {
        let one = via(&[1], &[]);
        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        s.keep(&[1, 2, 7, 8, 99], &one);
        assert_eq!(
            s.copied_run(&[9, 1, 2], &one, cfg(), 32),
            None,
            "a bigram seen once"
        );
        s.keep(&[1, 2, 7, 8, 99], &one);
        assert_eq!(
            s.copied_run(&[9, 1, 2], &one, cfg(), 32).unwrap().tokens,
            vec![7, 8]
        );

        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        s.keep(&[3, 4, 5, 6, 99], &one);
        s.keep(&[3, 4, 5, 7, 99], &one);
        // Two of two hits, split 1:1 — the newest leads.
        assert_eq!(
            s.copied_run(&[3, 4, 5], &one, cfg(), 32).unwrap().tokens,
            vec![7]
        );
        s.keep(&[3, 4, 5, 8, 99], &one);
        // One in three for each: under half, and no fallback to [4, 5].
        assert_eq!(s.copied_run(&[3, 4, 5], &one, cfg(), 32), None);
    }

    /// A run never goes past its reply's end, and a match never runs back
    /// across the reply before it.
    #[test]
    fn a_copy_stays_inside_one_earlier_reply() {
        let one = via(&[1], &[]);
        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        s.keep(&[20, 21, 22, 99], &one);
        s.keep(&[23, 24, 25, 26, 99], &one);
        // [21, 22] ends the first reply: nothing kept follows it there.
        assert_eq!(s.copied_run(&[20, 21, 22], &one, cfg(), 32), None);
        // [22, 23] spans the boundary and was never an n-gram of either.
        assert_eq!(s.copied_run(&[22, 23], &one, cfg(), 32), None);
        let run = s.copied_run(&[22, 23, 24, 25], &one, cfg(), 32).unwrap();
        assert_eq!(run.tokens, vec![26]);
        assert_eq!(
            run.matched, 3,
            "matched back to the reply's start, not past it"
        );
    }

    /// A peer that knew a reply only as hidden states is never sent that
    /// reply's token ids: a guess that reaches a peer as ids is copied only
    /// from replies that peer knew as ids; one reaching it as states, from
    /// either.
    #[test]
    fn a_peer_shown_hidden_states_is_never_sent_the_replys_ids() {
        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        s.keep(&[10, 11, 12, 13, 14, 99], &via(&[1], &[]));
        assert_eq!(
            s.copied_run(&[11, 12, 13], &via(&[], &[1]), cfg(), 32),
            None
        );
        assert!(s
            .copied_run(&[11, 12, 13], &via(&[1], &[]), cfg(), 32)
            .is_some());

        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        // Peer 1 ran the first layers (ids), peer 2 the rest (states).
        s.keep(&[20, 21, 22, 23, 24, 99], &via(&[2], &[1]));
        for shown in [via(&[2], &[1]), via(&[1], &[]), via(&[1, 2], &[])] {
            assert!(
                s.copied_run(&[21, 22, 23], &shown, cfg(), 32).is_some(),
                "{shown:?}"
            );
        }
        // Ids to the peer that was sent states — a chained check's hop, or a
        // tail that walks — refused.
        for shown in [via(&[1], &[2]), via(&[], &[1, 2])] {
            assert_eq!(
                s.copied_run(&[21, 22, 23], &shown, cfg(), 32),
                None,
                "{shown:?}"
            );
        }
    }

    /// Short replies are bounded by count as well as by tokens.
    #[test]
    fn the_store_keeps_at_most_its_replies() {
        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        let one = via(&[1], &[]);
        for i in 0..(MOST_REPLIES as u32 + 10) {
            s.keep(&[i, i + 1, i + 2], &one);
        }
        let m = store.by_model.get("m").unwrap();
        assert_eq!(m.replies.len(), MOST_REPLIES);
        assert_eq!(m.earlier[0].len(), m.tokens.len());
        // Replies of one split share one record of its peers.
        let first = &m.replies.front().unwrap().shown.seen_by;
        let last = &m.replies.back().unwrap().shown.seen_by;
        assert!(Arc::ptr_eq(first, last));
    }

    /// The store is bounded: past `KEPT_PER_MODEL` the oldest reply goes, and
    /// nothing is found in it; past `MODELS_KEPT` the model written longest
    /// ago goes.
    #[test]
    fn the_store_keeps_its_bounds() {
        let one = via(&[1], &[]);
        let store = Arc::new(OwnerReplies::default());
        let s = owner(&store);
        let first: Vec<u32> = vec![40, 41, 42, 43, 99];
        s.keep(&first, &one);
        assert!(s.copied_run(&[40, 41, 42], &one, cfg(), 32).is_some());
        let filler: Vec<u32> = (0..KEPT_PER_MODEL as u32 / 2).map(|i| 1000 + i).collect();
        s.keep(&filler, &one);
        s.keep(&filler, &one);
        assert_eq!(
            s.copied_run(&[40, 41, 42], &one, cfg(), 32),
            None,
            "dropped"
        );
        {
            let m = store.by_model.get("m").unwrap();
            assert!(m.tokens.len() <= KEPT_PER_MODEL);
            assert_eq!(m.earlier[0].len(), m.tokens.len());
        }
        for i in 0..MODELS_KEPT {
            let mut r = request(RequestedBy::Owner);
            r.model_id = ModelId(format!("other{i}"));
            store.for_the_owner(&r).unwrap().keep(&[1, 2, 3, 4], &one);
        }
        assert_eq!(store.by_model.len(), MODELS_KEPT);
        assert!(!store.by_model.contains_key("m"), "the oldest model went");
    }
}
