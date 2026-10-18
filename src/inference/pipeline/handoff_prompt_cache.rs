//! What this node believes a peer's prompt cache holds after a whole-model
//! hand-off (FUTURE_WORK #3 (c)).
//!
//! A peer that runs a whole model for this node (`remote_generate`, the
//! hand-off) runs it through its worker's `Generate`, which stores the prompt
//! in the worker's `PrefixCache` and looks the next one up there. So the next
//! turn of a conversation handed to the same peer reads only what is new — on
//! the live swarm a repeated 505-token prompt that took ~0 to read was priced
//! 20.2 s (0.25x of its price, twice). The planner could not know: a peer
//! announces its cache only when its owner opts in
//! (`inference.share_prefix_cache_with_peers`, off by default — a block hash is
//! a commitment to the prompt it came from).
//!
//! So this node remembers what IT sent where, as SGLang's cache-aware router
//! does (an approximate tree per worker, built from the requests it routed,
//! never from the worker) and llm-d's approximate prefix scorer does (the
//! pod a prompt went to, recorded after scheduling). Their known weakness is
//! drift: the router never sees an eviction or a restart. Here a belief
//! - lasts [`BELIEF_TTL`], the idle time after which a worker is unloaded, so
//!   it cannot describe a worker unloaded and loaded again since;
//! - is priced only while the peer still reports the whole model loaded
//!   (`scheduler::gather_candidates`);
//! - is forgotten when a hand-off to that peer fails;
//! - is a PRICE, never a pin: a faster or less loaded route still wins.
//!
//! What remains unseen — the peer's cache evicting under memory pressure, a
//! card reclaim and reload inside the window, a peer whose block size or
//! cache settings differ from the defaults — costs one prompt read at the
//! price of a cache hit, on a route the planner would otherwise have priced
//! by the same read.
//!
//! The keys are [`split_prompt_cache::chain_keys`]: keyed with this process's
//! secret, at the default `prefix_cache_block_tokens` (64), and never sent
//! anywhere.
//!
//! [`split_prompt_cache::chain_keys`]: super::split_prompt_cache::chain_keys

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use dashmap::DashMap;

use crate::types::{ModelId, NodeId};

/// How long a peer is believed to keep a prompt handed to it: the default idle
/// unload (`prune`'s 300 s threshold). A conversation's next turn usually comes
/// sooner; one that does not is priced cold, as before.
pub(crate) const BELIEF_TTL: Duration = Duration::from_secs(300);

/// Tokens per block — the default `prefix_cache_block_tokens`, the boundaries
/// a peer's worker stores a prompt at.
const BLOCK_TOKENS: u32 = super::split_prompt_cache::BLOCK_TOKENS;

/// The fewest blocks worth remembering: below this a re-read is cheap.
const MIN_BLOCKS: usize = 2;

/// Prompts remembered per (peer, model): one per conversation a peer serves
/// for this node, the latest of each.
const CHAINS_PER_PEER_MODEL: usize = 8;

/// (peer, model) pairs remembered at most; past it the table is cleared —
/// beliefs are only an optimisation.
const MAX_PAIRS: usize = 4096;

struct Sent {
    keys: Vec<[u8; 32]>,
    at: Instant,
}

fn beliefs() -> &'static DashMap<(NodeId, ModelId), Vec<Sent>> {
    static BELIEFS: OnceLock<DashMap<(NodeId, ModelId), Vec<Sent>>> = OnceLock::new();
    BELIEFS.get_or_init(DashMap::new)
}

/// `peer` ran `model` whole for this node on the prompt `ids`, and its worker
/// now holds that prompt's blocks. A prompt that extends one remembered
/// replaces it (the worker keeps only the longer, covered-prefix pruning); one
/// a remembered prompt already covers refreshes it.
pub(crate) fn note_handed_off(peer: &NodeId, model: &ModelId, ids: &[u32], now: Instant) {
    let keys = super::split_prompt_cache::chain_keys(ids, BLOCK_TOKENS);
    if keys.len() < MIN_BLOCKS {
        return;
    }
    let table = beliefs();
    if table.len() >= MAX_PAIRS && !table.contains_key(&(peer.clone(), model.clone())) {
        table.clear();
    }
    let mut chains = table.entry((peer.clone(), model.clone())).or_default();
    chains.retain(|sent| now.saturating_duration_since(sent.at) < BELIEF_TTL);
    if let Some(covering) = chains.iter_mut().find(|sent| sent.keys.starts_with(&keys)) {
        covering.at = now;
        return;
    }
    chains.retain(|sent| !keys.starts_with(&sent.keys));
    chains.insert(0, Sent { keys, at: now });
    chains.truncate(CHAINS_PER_PEER_MODEL);
}

/// A hand-off to `peer` failed: what it holds is no longer known.
pub(crate) fn forget(peer: &NodeId, model: &ModelId) {
    beliefs().remove(&(peer.clone(), model.clone()));
}

/// Does this node believe any peer holds a prompt of `model`? The cheap
/// question `cached_prefix::plan_prompt` asks before tokenizing.
pub(crate) fn believes_any_of(model: &ModelId) -> bool {
    beliefs().iter().any(|e| &e.key().1 == model)
}

/// Each peer believed to hold the opening of the prompt `ids` for a
/// whole-model run of `model`, with how many tokens of it: whole blocks, and
/// never the last token — the worker reads at least one position for the
/// reply's first logits.
pub(crate) fn handed_off_openings(
    model: &ModelId,
    ids: &[u32],
    now: Instant,
) -> Vec<(NodeId, u32)> {
    let table = beliefs();
    if !table.iter().any(|e| &e.key().1 == model) {
        return Vec::new();
    }
    let keys = super::split_prompt_cache::chain_keys(ids, BLOCK_TOKENS);
    let most = u32::try_from(ids.len().saturating_sub(1)).unwrap_or(u32::MAX);
    table
        .iter()
        .filter(|e| &e.key().1 == model)
        .filter_map(|e| {
            let blocks = e
                .value()
                .iter()
                .filter(|sent| now.saturating_duration_since(sent.at) < BELIEF_TTL)
                .map(|sent| {
                    sent.keys
                        .iter()
                        .zip(&keys)
                        .take_while(|(held, asked)| held == asked)
                        .count()
                })
                .max()?;
            let tokens = u32::try_from(blocks)
                .unwrap_or(u32::MAX)
                .saturating_mul(BLOCK_TOKENS)
                .min(most);
            (tokens > 0).then(|| (e.key().0.clone(), tokens))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(b: u8) -> NodeId {
        NodeId([b; 32])
    }

    /// A conversation's turn: the opening every turn repeats, then this turn.
    fn turn(opening: u32, this_turn: u32, salt: u32) -> Vec<u32> {
        (0..opening)
            .chain((0..this_turn).map(|i| 100_000 + salt * 1_000 + i))
            .collect()
    }

    /// **The next turn handed to the same peer is credited what that peer
    /// read last turn** — whole blocks of the shared opening, never the last
    /// token — and no other peer, and no other model, is credited anything.
    #[test]
    fn a_peer_is_credited_the_opening_it_was_handed() {
        let model = ModelId("handoff-credit".into());
        let other_model = ModelId("handoff-credit-other".into());
        let (be, it) = (node(1), node(2));
        let now = Instant::now();
        note_handed_off(&be, &model, &turn(505, 0, 0), now);

        // Turn 2: the 505-token opening plus 40 new tokens — 7 whole blocks.
        let next = turn(505, 40, 1);
        assert_eq!(
            handed_off_openings(&model, &next, now),
            vec![(be.clone(), 448)]
        );
        assert!(handed_off_openings(&other_model, &next, now).is_empty());

        // A different conversation shares nothing.
        assert!(handed_off_openings(&model, &(7_000..7_600).collect::<Vec<u32>>(), now).is_empty());

        // The prompt itself again: every block but never the last token.
        let same: Vec<u32> = turn(512, 0, 0);
        note_handed_off(&it, &model, &same, now);
        let credited = handed_off_openings(&model, &same, now);
        assert!(credited.contains(&(it, 511)), "{credited:?}");
    }

    /// A belief lasts the idle-unload window and no longer; a failed hand-off
    /// forgets it at once; a prompt too short to matter is not remembered.
    #[test]
    fn a_belief_ages_out_and_a_failure_forgets_it() {
        let model = ModelId("handoff-ttl".into());
        let peer = node(3);
        let then = Instant::now();
        note_handed_off(&peer, &model, &turn(640, 0, 0), then);
        let next = turn(640, 10, 1);
        assert_eq!(handed_off_openings(&model, &next, then).len(), 1);
        assert!(handed_off_openings(&model, &next, then + BELIEF_TTL).is_empty());

        note_handed_off(&peer, &model, &turn(640, 0, 0), then);
        forget(&peer, &model);
        assert!(handed_off_openings(&model, &next, then).is_empty());

        let short = ModelId("handoff-short".into());
        note_handed_off(&peer, &short, &turn(100, 0, 0), then);
        assert!(!believes_any_of(&short), "one block is not worth a belief");
    }

    /// A growing conversation keeps ONE belief per peer, as the worker keeps
    /// one entry: the longer prompt replaces the one it extends.
    #[test]
    fn a_growing_conversation_is_one_belief() {
        let model = ModelId("handoff-grow".into());
        let peer = node(4);
        let now = Instant::now();
        let first = turn(256, 0, 0);
        let mut second = first.clone();
        second.extend(500..700);
        note_handed_off(&peer, &model, &first, now);
        note_handed_off(&peer, &model, &second, now);
        let chains = beliefs().get(&(peer.clone(), model.clone())).unwrap().len();
        assert_eq!(chains, 1);
        // The shorter prompt again is covered, and refreshes rather than adds.
        note_handed_off(&peer, &model, &first, now);
        assert_eq!(beliefs().get(&(peer, model)).unwrap().len(), 1);
    }
}
