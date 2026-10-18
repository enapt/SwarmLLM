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
//! **A whole model is credited only for this node's own cache.** A whole-model
//! run here goes through `Generate`, the one path that consults the cache
//! (`PipelineExecutor::try_local_generate_fastpath`; the three speculative paths
//! before it all decline a plan that is entirely local). A PEER's announced
//! cache would be priced and not delivered whenever the n-gram path takes the
//! request — it runs before `remote_generate` for a remote single segment, and
//! sends forwards, which never look the cache up.
//!
//! **A segment of a split is credited for what this node believes it holds**
//! (`PromptPlan::split_openings`, FUTURE_WORK #10): the segments' own answers
//! after earlier prompt passes this node led, the beliefs a split's prompt pass
//! resumes from (`pipeline::split_prompt_cache`). That credit IS delivered: a
//! split's prompt pass goes through `forward_through_segments` on every path,
//! and a whole model — where the n-gram trap above lies — is never credited by
//! it. Without it, two warm peers were told apart by speed alone, and a
//! conversation's next turn could go to the one holding ANOTHER conversation's
//! opening and read its whole prompt again.
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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PromptPlan {
    /// Prompt length in tokens: exact when the prompt was tokenized, else the
    /// router's estimate. `None` when the caller does not know.
    pub tokens: Option<u32>,
    /// Leading tokens of this prompt that THIS node's worker already holds in
    /// its prefix cache. Zero unless tokenized and matched.
    pub cached_locally: u32,
    /// The nodes — peers or this one — believed to hold the opening of this
    /// prompt as a SEGMENT of a split, with the layer range and how many
    /// tokens (`pipeline::held_openings`). Empty unless tokenized and believed.
    pub(crate) split_openings: Vec<crate::inference::pipeline::HeldOpening>,
    /// The most tokens this request's reply may use — its `max_tokens` — so
    /// the most decode passes it can take. The cost model charges its
    /// per-token terms for no more passes than this
    /// (`NodeCandidate::forward_passes`, FUTURE_WORK #3). `None` where no
    /// request is behind the plan (the admin preview, the tests): priced at
    /// `ASSUMED_FORWARD_PASSES` exactly as before.
    pub reply_tokens_at_most: Option<u32>,
}

impl From<Option<u32>> for PromptPlan {
    fn from(tokens: Option<u32>) -> Self {
        Self {
            tokens,
            ..Self::default()
        }
    }
}

impl PromptPlan {
    /// This plan, for a request whose reply may use at most `max_tokens`.
    /// Set by the router where it plans a real request
    /// (`distributed_exec`), the one place both of its prompt plans meet.
    pub(crate) fn with_reply_of_at_most(mut self, max_tokens: u32) -> Self {
        self.reply_tokens_at_most = Some(max_tokens);
        self
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
    if !worth_planning(state, request)
        && !worth_counting(&request.messages, request.tools.as_deref())
    {
        return None;
    }
    let (state, request) = (state.clone(), request.clone());
    tokio::task::spawn_blocking(move || plan_prompt(&state, &request))
        .await
        .ok()
        .flatten()
}

/// Could this request be credited at all? The cheap half of [`plan_prompt`]:
/// no images, and our worker has reported a cache for this model or this node
/// believes a split's segments hold something of it.
fn worth_planning(state: &SharedState, request: &InferenceRequest) -> bool {
    // An image becomes embeddings the text tokenizer never sees, and neither
    // the worker's vision path nor a split's (which keeps no prompt) is the
    // one these hash chains describe.
    !request.messages.iter().any(|m| !m.images.is_empty())
        && (local_cache_worth_asking(state, request)
            || crate::inference::pipeline::believes_any_of(&request.model_id))
}

/// The estimate past which a prompt is counted exactly, in tokens.
///
/// **The estimate under-counts dense text by half.** Four bytes a token is
/// prose; a tester's 10 KB prompt was estimated at 2,495 tokens and is 3,729
/// (field report 2026-10-16), and the plan priced its prefill and every
/// holder's conversation memory for two-thirds of it. Counting costs ~46 ms for
/// 2,744 tokens, which a short chat should not pay for an error of a few
/// tokens; a prompt this long takes seconds to read anywhere, and a cache a
/// third short is a refusal waiting at the holder (#263).
const COUNT_EXACTLY_FROM_TOKENS: u32 = 1_024;

/// Is the router's estimate of this prompt long enough to be worth counting
/// exactly? Not for an image, which the text tokenizer never sees.
fn worth_counting(
    messages: &[crate::types::ChatMessage],
    tools: Option<&[serde_json::Value]>,
) -> bool {
    !messages.iter().any(|m| !m.images.is_empty())
        && crate::inference::estimate_prompt_tokens(messages, tools) >= COUNT_EXACTLY_FROM_TOKENS
}

/// Has our own worker reported a prefix cache for this model?
fn local_cache_worth_asking(state: &SharedState, request: &InferenceRequest) -> bool {
    state.model_process_pool.prefix_cache_block_tokens() > 0
        && worker_reported_a_cache(state, state.identity.node_id(), &request.model_id)
}

/// The prompt plan for `request`: its exact length, how much of it this
/// node's worker holds, and which split segments are believed to hold its
/// opening, when that can be known; `None` otherwise, and the caller falls
/// back to its estimate.
///
/// Tokenizes ONLY when there is something to match, or a prompt long enough
/// that the estimate's error matters ([`worth_counting`]) — the common request
/// pays a map lookup and a pass over the belief table, and nothing else.
/// Blocking: call it through [`plan_prompt_off_thread`] from async code.
pub(crate) fn plan_prompt(state: &SharedState, request: &InferenceRequest) -> Option<PromptPlan> {
    if !worth_planning(state, request)
        && !worth_counting(&request.messages, request.tools.as_deref())
    {
        return None;
    }
    let our_id = state.identity.node_id();
    let tokenizer = state.standalone_tokenizer(&request.model_id)?;
    let header_path = state
        .model_dir(&request.model_id.0)
        .join(crate::model::shard::HEADER_FILENAME);
    let header = crate::inference::pipeline::template_from_header(&header_path)?;
    // As every path's prompt is built (`PipelineExecutor::build_prompt_with_header`):
    // a reply continued after a failure carries what its reader already has.
    let mut prompt = crate::inference::pipeline::render_prompt_from_header(request, &header);
    if let Some(c) = &request.continuation {
        prompt.push_str(&c.text);
    }
    let ids: Vec<u32> = tokenizer
        .encode(&prompt)
        .into_iter()
        .map(|t| t as u32)
        .collect();
    let cached_locally = if local_cache_worth_asking(state, request) {
        let block_tokens = state.model_process_pool.prefix_cache_block_tokens();
        let blocks = crate::inference::split::compute_block_hashes(&ids, block_tokens);
        let held_blocks = leading_blocks_held(state, our_id, &request.model_id, &blocks);
        cached_tokens(held_blocks, block_tokens, ids.len())
    } else {
        0
    };
    Some(PromptPlan {
        tokens: Some(u32::try_from(ids.len()).unwrap_or(u32::MAX)),
        cached_locally,
        split_openings: crate::inference::pipeline::held_openings(&request.model_id, &ids),
        // The caller's: `with_reply_of_at_most`, beside the fallback plan.
        reply_tokens_at_most: None,
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
        assert!(plan.split_openings.is_empty());
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

    /// **A long prompt is counted, not estimated** (field report 2026-10-16:
    /// 2,495 estimated, 3,729 read). A short chat keeps the free estimate, and
    /// an image — embeddings the tokenizer never sees — is never counted.
    #[test]
    fn a_long_prompt_is_counted_exactly_and_a_short_one_is_estimated() {
        let asked = |content: String, images: usize| {
            vec![crate::types::ChatMessage {
                role: crate::types::Role::User,
                content,
                images: (0..images)
                    .map(|_| crate::types::ImageData {
                        rgb_bytes: vec![0; 3],
                        width: 1,
                        height: 1,
                    })
                    .collect(),
                tool: Default::default(),
            }]
        };
        assert!(!worth_counting(&asked("Hi".into(), 0), None));
        assert!(worth_counting(&asked("fn x() {} ".repeat(1_000), 0), None));
        assert!(!worth_counting(&asked("fn x() {} ".repeat(1_000), 1), None));
    }
}
