//! A split's prompt weighed on its PEERS before the pass (FUTURE_WORK #252 (b)).
//!
//! #263 admits a prompt on this node's own segments before any segment runs;
//! a peer's segment was still admitted when its forward arrived — after every
//! machine before it had read the prompt. The coordinator now asks each peer
//! first (`types::PromptAdmissionAsk`), and both ends keep a little state:
//!
//! - **the coordinator** remembers what each peer said for a request, so the
//!   re-plan holds a peer to the layers it said fit rather than barring it
//!   ([`SharedState::note_peer_prompt_refusal`]), and waits for answers
//!   (`pending_prompt_admissions`);
//! - **the asked node** limits how many questions it weighs ([`AskBudget`]).
//!   A question is weighed and nothing is held for it
//!   (`ModelProcessPool::weigh_peer_prompt`), but weighing still takes a turn
//!   of the worker's one message loop — a stream synchronize and a device
//!   reading — so it is rationed per asking peer and for the node as a whole.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use dashmap::DashMap;

use super::SharedState;
use crate::types::NodeId;
use swarmllm_types::LayersCounted;

/// The questions one peer may have weighed here in [`ASK_WINDOW`]. An honest
/// coordinator asks once per split request it sends through this node (and
/// again on a re-plan), so this is a dozen split requests a minute from ONE
/// coordinator.
pub(crate) const ASKS_PER_PEER: usize = 12;
/// The questions this node weighs in [`ASK_WINDOW`] from every peer together —
/// what bounds the worker time many identities could spend, one a second.
pub(crate) const ASKS_PER_NODE: usize = 60;
pub(crate) const ASK_WINDOW: Duration = Duration::from_secs(60);

/// How many questions this node has weighed lately, per asking peer and in
/// all. Past either limit a question is answered "go ahead" at once without
/// the worker — what every split did before the question existed: its
/// segments admit themselves as their forwards arrive. So a flood costs the
/// honest coordinators only their early answer, never a refusal.
#[derive(Default)]
pub(crate) struct AskBudget {
    windows: std::sync::Mutex<AskWindows>,
}

#[derive(Default)]
struct AskWindows {
    by_peer: HashMap<NodeId, VecDeque<Instant>>,
    all: VecDeque<Instant>,
}

impl AskBudget {
    /// May `peer`'s question be weighed now? Spends one of its questions and
    /// one of the node's if so. A peer whose questions have all aged out is
    /// forgotten, so the map holds only peers that asked within the window.
    pub(crate) fn try_spend(&self, peer: &NodeId, now: Instant) -> bool {
        self.try_spend_within(peer, now, ASKS_PER_PEER, ASKS_PER_NODE)
    }

    /// [`Self::try_spend`] with limits of the caller's own, for another kind
    /// of peer-triggered work rationed the same way (`layer_forward`'s next-hop
    /// dials). One budget, one pair of limits: never spend one instance under
    /// two.
    pub(crate) fn try_spend_within(
        &self,
        peer: &NodeId,
        now: Instant,
        per_peer: usize,
        in_all: usize,
    ) -> bool {
        let Ok(mut w) = self.windows.lock() else {
            return false;
        };
        let fresh = |at: &Instant| now.saturating_duration_since(*at) < ASK_WINDOW;
        while w.all.front().is_some_and(|at| !fresh(at)) {
            w.all.pop_front();
        }
        w.by_peer.retain(|_, asks| {
            while asks.front().is_some_and(|at| !fresh(at)) {
                asks.pop_front();
            }
            !asks.is_empty()
        });
        if w.all.len() >= in_all
            || w.by_peer
                .get(peer)
                .is_some_and(|asks| asks.len() >= per_peer)
        {
            return false;
        }
        w.all.push_back(now);
        w.by_peer.entry(peer.clone()).or_default().push_back(now);
        true
    }
}

/// What a peer said fits when it refused a request before the pass:
/// `layers` of the model, and what kind of count that is (`counted`).
///
/// - [`LayersCounted::Loading`]: the widest run from `from`, the first layer
///   it was asked for, whose weights would load
///   (`ModelProcessPool::peer_load_refusal`) — REAL layers there.
/// - [`LayersCounted::Caches`]: the asked layers whose caches fit
///   (`model_worker::layers_that_fit`) — any layers, each costing the same.
/// - [`LayersCounted::Unsaid`]: a peer that predates the field; either.
///
/// **Why it is kept whole.** A peer's bounds count AVERAGE layers, and a
/// range is weighed as what it loads (#353). Read bare, a count of layers
/// that load was a bound in the wrong unit: an Australian peer refused
/// [16..66) of Llama 3.3 70B saying 45 would load — 45 of the model's lighter
/// middle layers, 21,585 MB — and the re-plan gave it [28..75), 22,740 MB,
/// whose weighed charge (44.9) was under 45. Refused again at the same spot,
/// and the request failed (2026-10-26, v0.3.255). The scheduler weighs each
/// kind in its own unit (`NodeCandidate::prompt_fit_counts_caches`,
/// `RangeWeights::ceiling_of_a_fit`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeerPromptFit {
    pub from: u32,
    pub layers: u32,
    pub counted: LayersCounted,
}

/// The verdicts kept per peer and request. A request is re-planned at most
/// a few times (`router::MAX_REPLANS`), each asking a peer once; past this the
/// loosest count goes.
const MOST_FITS_KEPT: usize = 8;

impl SharedState {
    /// A peer refused this request's prompt when asked before the pass, and
    /// said what fits ([`PeerPromptFit`]). The re-plan holds that peer's TOTAL
    /// layers to it, as #263 holds this node
    /// (`local_layers_fitting_for_request`). One verdict per start and kind,
    /// the fewest layers kept, so a second refusal can only tighten it;
    /// verdicts counted otherwise are all kept, since which is tighter depends
    /// on the weights where each was counted. Released with the request
    /// ([`SharedState::release_request_state`]).
    pub fn note_peer_prompt_refusal(
        &self,
        request_id: uuid::Uuid,
        peer: &NodeId,
        said: PeerPromptFit,
    ) {
        let mut by_peer = self.peer_prompt_refusals.entry(request_id).or_default();
        let fits = by_peer.entry(peer.clone()).or_default();
        match fits
            .iter_mut()
            .find(|fit| fit.from == said.from && fit.counted == said.counted)
        {
            Some(fit) => fit.layers = fit.layers.min(said.layers),
            None => fits.push(said),
        }
        if fits.len() > MOST_FITS_KEPT {
            if let Some(loosest) = (0..fits.len()).max_by_key(|&i| fits[i].layers) {
                fits.swap_remove(loosest);
            }
        }
    }

    /// What `peer` said fits of this request, if it refused the prompt when
    /// asked — every verdict kept.
    pub fn peer_prompt_fits_for_request(
        &self,
        request_id: uuid::Uuid,
        peer: &NodeId,
    ) -> Vec<PeerPromptFit> {
        self.peer_prompt_refusals
            .get(&request_id)
            .and_then(|by_peer| by_peer.get(peer).cloned())
            .unwrap_or_default()
    }
}

/// What each peer said, per request (the coordinator's side).
pub(crate) type PeerPromptRefusals = DashMap<uuid::Uuid, HashMap<NodeId, Vec<PeerPromptFit>>>;

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    /// One peer may have a dozen questions weighed a minute, every peer
    /// together sixty; a question past either is answered without the worker,
    /// and a minute later the peer may ask again. A peer whose questions aged
    /// out is forgotten.
    #[test]
    fn questions_are_weighed_a_few_a_minute_per_peer_and_in_all() {
        let budget = AskBudget::default();
        let t = Instant::now();
        let (a, b) = (node(1), node(2));
        for _ in 0..ASKS_PER_PEER {
            assert!(budget.try_spend(&a, t));
        }
        assert!(!budget.try_spend(&a, t), "past the peer's share");
        assert!(budget.try_spend(&b, t), "another peer has its own");

        // Many identities still meet the node's limit.
        let mut spent = ASKS_PER_PEER + 1;
        let mut n = 3u8;
        while spent < ASKS_PER_NODE {
            assert!(budget.try_spend(&node(n), t));
            spent += 1;
            n += 1;
        }
        assert!(!budget.try_spend(&node(250), t), "past the node's limit");

        let later = t + ASK_WINDOW;
        assert!(
            budget.try_spend(&a, later),
            "a minute later it may ask again"
        );
        assert_eq!(
            budget.windows.lock().unwrap().by_peer.len(),
            1,
            "the peers whose questions aged out are forgotten"
        );
    }
}
