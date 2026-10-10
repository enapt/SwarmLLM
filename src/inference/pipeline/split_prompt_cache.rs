//! A split keeps its conversation's prompt between turns (FUTURE_WORK #10,
//! `docs/plans/split_prompt_cache.md`) — the coordinator's half.
//!
//! An agent re-sends the whole conversation every turn, so every turn read the
//! whole prompt through every segment of a split again: each segment's cache
//! for the request is released when the request ends, and the next turn is a
//! new request. Now a prompt pass carries a [`PromptCacheHint`]: the prompt's
//! blocks named by a keyed chain (vLLM's prefix-caching rule, each key covering
//! its parent's), and how far every segment of this plan is believed to hold
//! them already (`resume_at`). Each segment restores that opening from its own
//! store — the worker's `PrefixCache`, charged and evicted with the rest of its
//! cache (gotcha #440) — or refuses, and stores the prompt's blocks after the
//! pass; its answer says how many, which is what this node believes next turn.
//! Each segment keeps its own entries, as each pipeline stage does in
//! vLLM-Ascend's KVPP; the belief may be stale, which costs a miss and never a
//! wrong reply (arXiv 2606.17059's point about stale cache metadata): a miss is
//! answered by sending the prompt again from position 0, once.
//!
//! The keys are hashed with a secret of this process, so a segment holds names
//! it cannot turn back into text — the middle of a boomerang learns nothing it
//! did not already see — and two coordinators' entries never meet.

use std::sync::OnceLock;

use dashmap::DashMap;

use crate::types::{ModelId, NodeId, PipelineSegment, PromptCacheHint};

/// Tokens per block. Small enough that a turn's new text rarely wastes much of
/// a block, large enough that a long prompt's keys stay a few KB on the wire
/// (a 32K-token prompt is 512 keys, 16 KB, beside megabytes of hidden states).
pub(super) const BLOCK_TOKENS: u32 = 64;

/// The fewest blocks worth keeping: below this a turn's re-read is cheap.
const MIN_BLOCKS: usize = 2;

/// Chains remembered per (segment, model): a coordinator serving several
/// conversations on one split keeps each one's latest.
const CHAINS_PER_SEGMENT: usize = 8;

/// (segment node, model, layer range) beliefs this node keeps at most; past it
/// the table is cleared — beliefs are only an optimisation.
const MAX_BELIEFS: usize = 4096;

/// `SWARMLLM_SPLIT_PROMPT_CACHE=0` sends every prompt pass whole, as before —
/// the A/B arm. Read once.
pub(super) fn switched_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_SPLIT_PROMPT_CACHE").as_deref(),
            Ok("0") | Ok("off") | Ok("false")
        )
    })
}

/// This process's key for naming prompt blocks. Not persisted: after a restart
/// the old entries are never named again and age out of the segments' caches.
fn secret() -> &'static [u8; 32] {
    static SECRET: OnceLock<[u8; 32]> = OnceLock::new();
    SECRET.get_or_init(rand::random)
}

/// The key of every FULL block of `ids`: `key[i] = H(key[i-1] ‖ block i)`,
/// keyed with this process's secret.
pub(super) fn chain_keys(ids: &[u32], block_tokens: u32) -> Vec<[u8; 32]> {
    chain_keys_with(secret(), ids, block_tokens)
}

fn chain_keys_with(secret: &[u8; 32], ids: &[u32], block_tokens: u32) -> Vec<[u8; 32]> {
    let bt = block_tokens as usize;
    if bt == 0 {
        return Vec::new();
    }
    let mut parent = [0u8; 32];
    ids.chunks_exact(bt)
        .map(|block| {
            let mut h = blake3::Hasher::new_keyed(secret);
            h.update(&parent);
            for id in block {
                h.update(&id.to_le_bytes());
            }
            parent = *h.finalize().as_bytes();
            parent
        })
        .collect()
}

type BeliefKey = (NodeId, ModelId, (u32, u32));

fn beliefs() -> &'static DashMap<BeliefKey, Vec<Vec<[u8; 32]>>> {
    static B: OnceLock<DashMap<BeliefKey, Vec<Vec<[u8; 32]>>>> = OnceLock::new();
    B.get_or_init(DashMap::new)
}

/// How many leading blocks of `keys` a segment is believed to hold.
fn believed_blocks(segment: &PipelineSegment, keys: &[[u8; 32]]) -> usize {
    let key = (
        segment.node_id.clone(),
        segment.shard_id.model_id.clone(),
        segment.layer_range,
    );
    beliefs().get(&key).map_or(0, |chains| {
        chains
            .iter()
            .map(|c| c.iter().zip(keys).take_while(|(a, b)| a == b).count())
            .max()
            .unwrap_or(0)
    })
}

/// Where a prompt pass may resume: the longest opening EVERY segment of the
/// plan is believed to hold, in whole blocks, short of the prompt — the last
/// position must be computed for its logits.
pub(super) fn resume_at(
    segments: &[PipelineSegment],
    keys: &[[u8; 32]],
    prompt_len: usize,
    block_tokens: u32,
) -> u32 {
    let bt = block_tokens as usize;
    if bt == 0 || prompt_len == 0 {
        return 0;
    }
    let held = segments
        .iter()
        .map(|s| believed_blocks(s, keys))
        .min()
        .unwrap_or(0)
        .min((prompt_len - 1) / bt);
    if held < MIN_BLOCKS {
        0
    } else {
        (held * bt) as u32
    }
}

/// What a segment said it stored: the first `blocks` keys of `hint`.
pub(super) fn note_stored(segment: &PipelineSegment, hint: &PromptCacheHint, blocks: u32) {
    let key = (
        segment.node_id.clone(),
        segment.shard_id.model_id.clone(),
        segment.layer_range,
    );
    let blocks = (blocks as usize).min(hint.keys.len());
    let map = beliefs();
    if blocks == 0 {
        map.remove(&key);
        return;
    }
    if map.len() >= MAX_BELIEFS && !map.contains_key(&key) {
        map.clear();
    }
    let chain = hint.keys[..blocks].to_vec();
    let mut entry = map.entry(key).or_default();
    // A chain this one extends is covered by it — a growing conversation keeps
    // one — and the newest goes to the back.
    entry.retain(|c| !chain.starts_with(c));
    entry.push(chain);
    if entry.len() > CHAINS_PER_SEGMENT {
        let over = entry.len() - CHAINS_PER_SEGMENT;
        entry.drain(..over);
    }
}

/// A segment refused to restore what it was believed to hold: believe nothing
/// of it for this model and range until it says otherwise.
pub(super) fn forget(segment: &PipelineSegment) {
    beliefs().remove(&(
        segment.node_id.clone(),
        segment.shard_id.model_id.clone(),
        segment.layer_range,
    ));
}

/// The ids, packed as a first segment reads a multi-position input (i64 LE).
pub(super) fn pack_ids(ids: &[u32]) -> Vec<u8> {
    super::pack_verify_tokens_to_le_bytes(ids)
}

/// Does an error say a segment no longer holds what it was asked to restore?
/// Read through `reclassify_flattened_error`, which is where a class that
/// crossed the worker's or the network's string hop is recovered.
pub(super) fn is_cache_miss(err: &crate::error::SwarmError) -> bool {
    matches!(err, crate::error::SwarmError::PromptCacheMiss(_))
        || matches!(
            crate::error::reclassify_flattened_error(&err.to_string()),
            Some(crate::error::SwarmError::PromptCacheMiss(_))
        )
}

impl super::PipelineExecutor {
    /// The prompt's ids, when this pass should keep its prompt: a prompt pass
    /// of a plain split — no image, nothing pre-embedded, no tensor-parallel
    /// group — of two or more segments, every one of them this node's or a
    /// peer advertising `features::SPLIT_PROMPT_CACHE`, a tokenizer here, and
    /// a prompt of at least [`MIN_BLOCKS`] blocks. `None` runs the pass as
    /// before.
    pub(super) fn split_prompt_to_keep(
        &self,
        sequence_num: u32,
        prompt_bytes: &[u8],
        vision: bool,
        pre_embedded: bool,
    ) -> Option<Vec<u32>> {
        let segments = &self.assignment.segments;
        if sequence_num != 0
            || vision
            || pre_embedded
            || !switched_on()
            || segments.len() < 2
            || !self.assignment.tp_groups.is_empty()
        {
            return None;
        }
        let me = self.shared_state.identity.node_id();
        let every_segment_keeps = segments.iter().all(|s| {
            s.node_id == *me
                || self.shared_state.peer_advertises_feature(
                    &s.node_id,
                    swarmllm_types::node::features::SPLIT_PROMPT_CACHE,
                )
        });
        if !every_segment_keeps {
            return None;
        }
        let ids = self.tokenize_prompt(prompt_bytes)?;
        (ids.len() >= MIN_BLOCKS * BLOCK_TOKENS as usize).then_some(ids)
    }

    /// Run a prompt pass that resumes from what every segment is believed to
    /// hold and stores the prompt for the next turn. A segment that no longer
    /// holds it — or fails while the pass resumes — answers a miss, and the
    /// prompt is sent again from position 0, once: that pass restores nothing,
    /// so it cannot miss, and a failover runs in it as it always has.
    pub(super) async fn prompt_pass_keeping_the_prompt(
        &mut self,
        request_id: uuid::Uuid,
        ids: Vec<u32>,
        generated_ids: &[u32],
    ) -> Result<crate::types::LayerResult, crate::error::SwarmError> {
        let keys = chain_keys(&ids, BLOCK_TOKENS);
        let resume = resume_at(&self.assignment.segments, &keys, ids.len(), BLOCK_TOKENS);
        let first = self
            .kept_prompt_pass(request_id, &ids, keys.clone(), resume, generated_ids)
            .await;
        match first {
            Err(e) if resume > 0 && is_cache_miss(&e) => {
                tracing::info!(
                    %request_id,
                    resumed_from = resume,
                    error = %e,
                    // A miss, or any failure of a resumed pass (a stand-in
                    // could not hold the opening) — the error says which.
                    "a resumed prompt pass did not finish — reading the prompt again from the start"
                );
                for s in &self.assignment.segments {
                    forget(s);
                }
                // The resumed pass left every segment unrestorable for this
                // request, and it stays so: a request's retained history is
                // released in one place (`per_request_state_is_released_in_one_place`).
                // A failure mid-reply is continued by the router (#236) instead.
                self.kept_prompt_pass(request_id, &ids, keys, 0, generated_ids)
                    .await
            }
            other => other,
        }
    }

    async fn kept_prompt_pass(
        &mut self,
        request_id: uuid::Uuid,
        ids: &[u32],
        keys: Vec<[u8; 32]>,
        resume: u32,
        generated_ids: &[u32],
    ) -> Result<crate::types::LayerResult, crate::error::SwarmError> {
        if resume > 0 {
            // Positions `0..resume` are never sent, so a stand-in replayed from
            // the retained history mid-reply would start from a hole: say so
            // rather than leave it to the contiguity check.
            for s in &self.assignment.segments {
                self.shared_state
                    .retained_activations
                    .mark_unrestorable(request_id, s.layer_range);
            }
        }
        tracing::info!(
            %request_id,
            prompt_tokens = ids.len(),
            resumed_from = resume,
            blocks = keys.len(),
            "DIAG: a split prompt pass keeping its prompt"
        );
        self.prompt_cache_hint = Some(PromptCacheHint {
            block_tokens: BLOCK_TOKENS,
            keys,
            resume_at: resume,
        });
        // In pieces where the plan reads them (FUTURE_WORK #171): each piece
        // carries the hint, the first restores, the final stores. A miss is the
        // caller's to answer (a retry from 0); any other failure reads this
        // pass whole, below.
        // A kept pass is never chained (each segment's answer says what it
        // stored), so pieces may cross a boundary between two peers here.
        if self.reads_in_pieces((ids.len() as u32).saturating_sub(resume), true) {
            match self
                .prompt_pass_in_pieces(request_id, ids, resume, generated_ids)
                .await
            {
                Ok(result) => {
                    self.prompt_cache_hint = None;
                    return Ok(result);
                }
                Err(failed)
                    if self.request.is_cancelled()
                        || !failed.every_piece_answered
                        || is_cache_miss(&failed.error) =>
                {
                    self.prompt_cache_hint = None;
                    return Err(failed.error);
                }
                Err(failed) => tracing::warn!(
                    %request_id,
                    error = %failed.error,
                    "a prompt pass in pieces did not finish — reading it whole instead"
                ),
            }
        }
        let result = self
            .forward_through_segments_checked(
                request_id,
                0,
                resume as usize,
                pack_ids(&ids[resume as usize..]),
                None,
                false,
                generated_ids,
            )
            .await;
        self.prompt_cache_hint = None;
        result
    }

    /// What segment `idx` said it stored of the prompt being kept — what this
    /// node believes of it next turn. An answer without the count (a segment
    /// that stored nothing) believes nothing.
    pub(super) fn note_prompt_blocks_stored(
        &self,
        idx: usize,
        sequence_num: u32,
        result: &crate::types::LayerResult,
    ) {
        if sequence_num != 0 {
            return;
        }
        if let (Some(hint), Some(segment)) =
            (&self.prompt_cache_hint, self.assignment.segments.get(idx))
        {
            note_stored(segment, hint, result.prompt_blocks_stored.unwrap_or(0));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ShardId;

    fn seg(node: u8, range: (u32, u32)) -> PipelineSegment {
        PipelineSegment {
            node_id: NodeId([node; 32]),
            shard_id: ShardId {
                model_id: ModelId("split-prompt-cache-test".into()),
                index: 0,
            },
            layer_range: range,
        }
    }

    /// A key names the whole prompt up to its block: two prompts that differ
    /// in their first block differ in every key after it, and the same prompt
    /// under another coordinator's secret shares no key at all.
    #[test]
    fn a_key_names_everything_before_it() {
        let a: Vec<u32> = (0..256).collect();
        let mut b = a.clone();
        b[3] = 9999;
        let (ka, kb) = (
            chain_keys_with(&[1; 32], &a, 64),
            chain_keys_with(&[1; 32], &b, 64),
        );
        assert_eq!(ka.len(), 4);
        assert!(ka.iter().zip(&kb).all(|(x, y)| x != y));
        let longer: Vec<u32> = (0..300).collect();
        assert_eq!(
            chain_keys_with(&[1; 32], &longer, 64)[..4],
            ka[..],
            "a longer prompt extends the chain"
        );
        assert!(chain_keys_with(&[2; 32], &a, 64)
            .iter()
            .zip(&ka)
            .all(|(x, y)| x != y));
    }

    /// The resume point is what EVERY segment holds, in whole blocks, and
    /// leaves the last position to compute; a refusal forgets that segment.
    #[test]
    fn a_prompt_resumes_from_what_every_segment_holds() {
        let ids: Vec<u32> = (0..1000).map(|i| i * 7).collect();
        let keys = chain_keys(&ids, 64);
        let (a, b) = (seg(11, (0, 10)), seg(12, (10, 28)));
        let plan = vec![a.clone(), b.clone()];
        assert_eq!(
            resume_at(&plan, &keys, ids.len(), 64),
            0,
            "nothing believed yet"
        );
        let hint = PromptCacheHint {
            block_tokens: 64,
            keys: keys.clone(),
            resume_at: 0,
        };
        note_stored(&a, &hint, 15);
        note_stored(&b, &hint, 9);
        assert_eq!(
            resume_at(&plan, &keys, ids.len(), 64),
            9 * 64,
            "the least any segment holds"
        );
        // A prompt that ends inside what they hold still computes its last position.
        assert_eq!(resume_at(&plan, &keys, 9 * 64, 64), 8 * 64);
        // Fewer than two blocks is not worth a resume.
        assert_eq!(resume_at(&plan, &keys, 100, 64), 0);
        forget(&b);
        assert_eq!(resume_at(&plan, &keys, ids.len(), 64), 0);
    }

    /// The next turn's prompt shares the opening: it resumes from it, and a
    /// different conversation on the same split does not.
    #[test]
    fn the_next_turn_resumes_and_another_conversation_does_not() {
        let turn1: Vec<u32> = (0..640).map(|i| i + 5).collect();
        let mut turn2 = turn1.clone();
        turn2.extend(1000..1200);
        let other: Vec<u32> = (0..900).map(|i| i + 50_000).collect();
        let s = seg(21, (0, 28));
        let plan = vec![s.clone()];
        let k1 = chain_keys(&turn1, 64);
        note_stored(
            &s,
            &PromptCacheHint {
                block_tokens: 64,
                keys: k1,
                resume_at: 0,
            },
            10,
        );
        assert_eq!(
            resume_at(&plan, &chain_keys(&turn2, 64), turn2.len(), 64),
            640
        );
        assert_eq!(
            resume_at(&plan, &chain_keys(&other, 64), other.len(), 64),
            0
        );
    }

    #[test]
    fn a_miss_is_recognised_after_the_string_hops() {
        let direct = crate::error::SwarmError::PromptCacheMiss("x".into());
        assert!(is_cache_miss(&direct));
        let flattened = crate::error::SwarmError::Inference(format!("segment 1: {direct}"));
        assert!(is_cache_miss(&flattened));
        let wrapped = crate::error::SwarmError::ServiceUnavailable(format!("worker: {direct}"));
        assert!(is_cache_miss(&wrapped), "before the generic markers");
        assert!(!is_cache_miss(&crate::error::SwarmError::Inference(
            "boom".into()
        )));
    }
}
