//! How much of a request's prompt this node's own worker already holds.
//!
//! **Why routing needs it.** A node without a graphics card stands its API
//! fast path aside whenever a peer is connected, so the router decides where
//! every one of its requests runs — and it priced running the model here on
//! the FULL prompt, knowing nothing of the prefix cache. Field report
//! 2026-09-26: an agent harness sends the same ~9,200-token system prompt plus
//! a growing history on every turn. Turn 1 ran here (678 s) and cached it;
//! turn 2 was priced as another 9,280-token read, a chain through a peer's card
//! priced cheaper, and it took 847 s — against a cache hit that re-reads ~80
//! tokens. Every later turn of every agent conversation on such a node could go
//! the same way.
//!
//! **What it does.** Render the prompt exactly as the executor will
//! ([`crate::inference::pipeline::render_prompt_from_header`]), tokenize it with
//! the same tokenizer the worker builds from the same header, chain-hash it at
//! the worker's block size ([`crate::inference::split::compute_block_hashes`])
//! and walk those hashes against the blocks the worker last reported holding
//! (`models.peer_prefix_blocks[our id]`, written by the loopback forwarder in
//! `daemon::background` on every snapshot). The count of leading blocks it
//! holds is the part of the prompt a whole-model run here will not read again.
//!
//! This is SGLang's cache-aware routing (v0.4: route on a radix tree of each
//! worker's cache, keyed by the prompt prefix itself) fitted to our cost model
//! rather than to a threshold, and it keeps both lessons from its history:
//! the key is the FULL rendered prefill input, tools included — keying on
//! `messages[0]` made unrelated conversations look identical
//! (sgl-project/sglang#26263) — and a hit is a PRICE, not a pin, so load and a
//! genuinely faster route still win (their `cache_threshold` +
//! `balance_*_threshold`). See `docs/FUTURE_WORK.md` #10.
//!
//! **Only this node's own cache is credited.** A whole-model run here goes
//! through `Generate`, the one path that consults the cache
//! (`PipelineExecutor::try_local_generate_fastpath`; the three speculative paths
//! before it all decline a plan that is entirely local). A PEER's announced
//! cache would be priced and not delivered whenever the n-gram path takes the
//! request — it runs before `remote_generate` for a remote single segment, and
//! sends forwards, which never look the cache up.
//!
//! **Errs toward pricing cold.** Blocks are whole-block matches, so the credit
//! never exceeds what `PrefixCache::lookup` will hydrate. A self-entry the worker
//! has since evicted — or lost with the worker — costs one uncached local
//! prefill, which is exactly what this request would have cost without the
//! credit; that run re-inserts and re-announces, correcting the index.

use crate::daemon::SharedState;
use crate::types::{InferenceRequest, ModelId, NodeId};

/// What the planner knows about a request's prompt.
///
/// Converts from the bare `Option<u32>` estimate, so a caller with no rendered
/// prompt in hand — the admin plan preview, the tests — passes exactly what it
/// passed before and gets no cache credit.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PromptPlan {
    /// Prompt length in tokens: exact when the prompt was tokenized, else the
    /// router's estimate. `None` when the caller does not know.
    pub tokens: Option<u32>,
    /// Leading tokens of this prompt that THIS node's worker already holds in
    /// its prefix cache. Zero unless tokenized and matched.
    pub cached_locally: u32,
}

impl From<Option<u32>> for PromptPlan {
    fn from(tokens: Option<u32>) -> Self {
        Self {
            tokens,
            cached_locally: 0,
        }
    }
}

/// [`plan_prompt`], with the tokenizing kept off the async runtime.
///
/// Rendering, tokenizing and hashing a prompt is CPU work that scales with it —
/// 46 ms for 2,744 tokens measured on the rig, plus ~90 ms the first time a
/// model's tokenizer is built — and a runtime thread must not be held for
/// that. The cheap question, "has our worker reported anything for this
/// model", is answered inline, so the common request never leaves its thread.
pub(crate) async fn plan_prompt_off_thread(
    state: &std::sync::Arc<SharedState>,
    request: &InferenceRequest,
) -> Option<PromptPlan> {
    if !worth_planning(state, request) {
        return None;
    }
    let (state, request) = (state.clone(), request.clone());
    tokio::task::spawn_blocking(move || plan_prompt(&state, &request))
        .await
        .ok()
        .flatten()
}

/// Could this request be credited at all? The cheap half of [`plan_prompt`]:
/// no images, and our worker has reported a cache for this model.
fn worth_planning(state: &SharedState, request: &InferenceRequest) -> bool {
    // An image becomes embeddings the text tokenizer never sees, and the
    // worker's vision path is not the one this hash chain describes.
    !request.messages.iter().any(|m| !m.images.is_empty())
        && state.model_process_pool.prefix_cache_block_tokens() > 0
        && worker_reported_a_cache(state, state.identity.node_id(), &request.model_id)
}

/// The prompt plan for `request`: its exact length and how much of it this
/// node's worker holds, when that can be known; `None` otherwise, and the
/// caller falls back to its estimate.
///
/// Tokenizes ONLY when the worker has reported a cache for this model — the
/// common request pays one map lookup and nothing else. Blocking: call it
/// through [`plan_prompt_off_thread`] from async code.
pub(crate) fn plan_prompt(state: &SharedState, request: &InferenceRequest) -> Option<PromptPlan> {
    if !worth_planning(state, request) {
        return None;
    }
    let our_id = state.identity.node_id();
    let block_tokens = state.model_process_pool.prefix_cache_block_tokens();
    let tokenizer = state.standalone_tokenizer(&request.model_id)?;
    let header_path = state
        .model_dir(&request.model_id.0)
        .join(crate::model::shard::HEADER_FILENAME);
    let header = crate::inference::pipeline::template_from_header(&header_path)?;
    let prompt = crate::inference::pipeline::render_prompt_from_header(request, &header);
    let ids: Vec<u32> = tokenizer
        .encode(&prompt)
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let blocks = crate::inference::split::compute_block_hashes(&ids, block_tokens);
    let held_blocks = leading_blocks_held(state, our_id, &request.model_id, &blocks);
    Some(PromptPlan {
        tokens: Some(u32::try_from(ids.len()).unwrap_or(u32::MAX)),
        cached_locally: cached_tokens(held_blocks, block_tokens, ids.len()),
    })
}

/// Has this node's worker reported holding any block of this model?
fn worker_reported_a_cache(state: &SharedState, node: &NodeId, model: &ModelId) -> bool {
    state
        .models
        .peer_prefix_blocks
        .get(node)
        .is_some_and(|models| models.get(model).is_some_and(|set| !set.is_empty()))
}

/// How many of `blocks`, from the first, the node last reported holding.
///
/// Leading blocks only: the hash chain makes block `k` a commitment to
/// everything before it, so a hole means nothing after it is usable.
fn leading_blocks_held(
    state: &SharedState,
    node: &NodeId,
    model: &ModelId,
    blocks: &[crate::types::PrefixBlockEntry],
) -> usize {
    let Some(models) = state.models.peer_prefix_blocks.get(node) else {
        return 0;
    };
    let Some(held) = models.get(model) else {
        return 0;
    };
    blocks
        .iter()
        .take_while(|b| held.contains(&b.block_hash))
        .count()
}

/// Tokens a whole-model run will not read again: the held blocks, never the
/// whole prompt — the worker always forwards at least one token, because the
/// reply's first logits come from it (`PrefixCache::lookup`'s `usable_max`).
fn cached_tokens(held_blocks: usize, block_tokens: usize, prompt_len: usize) -> u32 {
    let held = held_blocks
        .saturating_mul(block_tokens)
        .min(prompt_len.saturating_sub(1));
    u32::try_from(held).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_estimate_carries_no_credit() {
        let plan: PromptPlan = Some(9_280).into();
        assert_eq!(plan.tokens, Some(9_280));
        assert_eq!(plan.cached_locally, 0);
        assert_eq!(PromptPlan::from(None), PromptPlan::default());
    }

    /// The credit is whole blocks, and never the last token: the worker
    /// forwards at least one position to get the reply's first logits.
    #[test]
    fn the_credit_is_whole_blocks_and_leaves_one_token_to_read() {
        assert_eq!(cached_tokens(143, 64, 9_280), 9_152);
        // Every block held and the prompt an exact multiple: one token stays.
        assert_eq!(cached_tokens(4, 64, 256), 255);
        assert_eq!(cached_tokens(0, 64, 9_280), 0);
        assert_eq!(cached_tokens(3, 64, 0), 0);
    }
}
