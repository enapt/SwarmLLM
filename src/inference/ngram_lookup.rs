//! N-gram prompt lookup speculation (SWARM-SPEC Layer 1.1 / 1.2).
//!
//! Looks for matches of the last few generated tokens earlier in the prompt
//! and/or in the recent generation tail. On a hit, returns the next K tokens
//! after the match as a speculative draft — no draft model required.
//!
//! Reference: <https://github.com/apoorvumang/prompt-lookup-decoding> and
//! HuggingFace's `prompt_lookup_num_tokens` / `max_matching_ngram_size`
//! parameters.
//!
//! # Why this matters for SwarmLLM
//!
//! Most of SwarmLLM's actual workload (Claude Code subscriptions, MCP tool
//! use, RAG, code completion) is "input-grounded": output copies entities,
//! identifiers, code fragments, format strings, etc. from the prompt. For
//! those workloads, prompt-lookup achieves 2.4-4.2× speedup (PROMTEC, ACL
//! 2025) with zero new model dependencies — just a hash table.
//!
//! # Algorithm
//!
//! For each candidate n in `max_ngram_size .. min_ngram_size`:
//!   - Take the last `n` tokens of the current context as the search needle.
//!   - Scan the searchable prefix (prompt OR prompt + recent generation) for
//!     all matches of that needle.
//!   - On the FIRST match found (scanning right-to-left, so most-recent
//!     match wins), return up to `num_pred_tokens` tokens after the match
//!     as the speculative draft.
//!   - If no match at this n, decrease n and retry.
//!
//! Falling back from larger n to smaller n preserves the quality bias —
//! longer matches are more likely to be predictive than shorter ones, but
//! we'd rather emit a short candidate than nothing.
//!
//! # Edge cases
//!
//! - Match position must be strictly less than `context.len() - n` (so the
//!   match isn't the search needle itself).
//! - If the match is at the very tail (its trailing tokens are partial
//!   inside the searchable region), the candidate is truncated.
//! - Empty context returns empty draft.

/// Default maximum n-gram size to try. Larger n means higher-quality matches
/// but lower hit rate. 4 matches the HuggingFace default behavior on the
/// canonical implementation.
pub const DEFAULT_MAX_NGRAM_SIZE: usize = 4;

/// Default minimum n-gram size — drop below this and matches become noise.
/// `n=2` matches single bigram phrases ("the", "of") which are too common.
pub const DEFAULT_MIN_NGRAM_SIZE: usize = 2;

/// Default number of candidate tokens to emit per match. Matches the
/// HuggingFace `prompt_lookup_num_tokens` default. Larger draft = larger
/// per-round speedup IF accepted, but more wasted bandwidth on rejection.
pub const DEFAULT_NUM_PRED_TOKENS: usize = 10;

/// Configuration for the n-gram lookup speculator.
#[derive(Clone, Copy, Debug)]
pub struct NgramLookupConfig {
    pub max_ngram_size: usize,
    pub min_ngram_size: usize,
    pub num_pred_tokens: usize,
}

impl Default for NgramLookupConfig {
    fn default() -> Self {
        Self {
            max_ngram_size: DEFAULT_MAX_NGRAM_SIZE,
            min_ngram_size: DEFAULT_MIN_NGRAM_SIZE,
            num_pred_tokens: DEFAULT_NUM_PRED_TOKENS,
        }
    }
}

/// Find a speculative continuation for `context` by looking up its tail in
/// `searchable`. Returns at most `cfg.num_pred_tokens` tokens; empty Vec
/// when no match is found.
///
/// `context` is the full token sequence so far (prompt + generation tail).
/// `searchable` is the slice to search in — typically `&context` itself
/// (search the prompt for matches of the recent tail), but it can also be
/// a separate buffer (e.g. last 500 generated tokens, or prompt + last 500).
///
/// Time complexity: O(searchable.len() × max_ngram_size) worst case.
/// Practical cost on typical 2K-prompt + 100-token tail: <1ms.
pub fn find_candidate(context: &[u32], searchable: &[u32], cfg: NgramLookupConfig) -> Vec<u32> {
    if context.is_empty() || searchable.is_empty() {
        return Vec::new();
    }
    let max_n = cfg.max_ngram_size.min(context.len()).min(searchable.len());
    let min_n = cfg.min_ngram_size.max(1);
    if max_n < min_n {
        return Vec::new();
    }

    for n in (min_n..=max_n).rev() {
        let needle = &context[context.len() - n..];
        // Iterate match positions right-to-left so the MOST RECENT match
        // wins. Most recent match is most likely to be in-context for the
        // current generation step (RAG/code: the chunk the user just
        // pasted, not a sibling chunk pasted earlier).
        //
        // The upper bound `searchable.len() - n` means the needle window
        // must fit inside `searchable`. The lower bound is 0 inclusive.
        let max_start = searchable.len().saturating_sub(n);
        // Skip positions that would self-match the needle inside context.
        // The self-match position is `context.len() - n` ONLY when
        // searchable === context (same slice). We can't cheaply detect
        // slice identity, so we use a value comparison: any position
        // whose window equals the needle AND whose successor would
        // produce the needle on re-search is filtered.
        for start in (0..=max_start).rev() {
            let window = &searchable[start..start + n];
            if window != needle {
                continue;
            }
            // Self-match guard: if the match position is the literal end of
            // searchable (i.e. start + n == searchable.len()), skip it
            // because the predicted tokens would just be the needle
            // itself.
            let candidate_start = start + n;
            if candidate_start >= searchable.len() {
                continue;
            }
            let candidate_end = (candidate_start + cfg.num_pred_tokens).min(searchable.len());
            return searchable[candidate_start..candidate_end].to_vec();
        }
    }
    Vec::new()
}

/// Cascade entry point used by the speculative loop: try prompt lookup
/// first, then generated-output lookup over the last `recent_gen_window`
/// tokens. Returns the first non-empty draft.
///
/// `prompt_tokens` is the full prompt + generation so far; the function
/// itself decides which sub-slices to search. This keeps callers from
/// having to maintain separate buffers.
///
/// The "generated-output lookup" pass searches the recent generation tail
/// against itself — captures "the model is in a repeating-pattern groove"
/// (lists, refactor edits, format adherence) that the prompt-only search
/// misses.
pub fn cascade_find_candidate(
    prompt_tokens: &[u32],
    prompt_len: usize,
    recent_gen_window: usize,
    cfg: NgramLookupConfig,
) -> (Vec<u32>, NgramHitSource) {
    // Pass 1: search the prompt for matches of the recent context tail.
    // Searches the prompt slice only — the most common "input-grounded"
    // case (copying from prompt to output).
    if prompt_len > 0 && prompt_len <= prompt_tokens.len() {
        let cand = find_candidate(prompt_tokens, &prompt_tokens[..prompt_len], cfg);
        if !cand.is_empty() {
            return (cand, NgramHitSource::Prompt);
        }
    }

    // Pass 2: search the recent generation tail for self-matches. Useful
    // when the model is enumerating a list or applying a repeating
    // transformation. Searching too large a window slows the lookup
    // without adding hit rate, so we cap.
    if prompt_tokens.len() > prompt_len {
        let gen_start = prompt_tokens.len().saturating_sub(recent_gen_window);
        let gen_start = gen_start.max(prompt_len);
        if gen_start < prompt_tokens.len() {
            let cand = find_candidate(prompt_tokens, &prompt_tokens[gen_start..], cfg);
            if !cand.is_empty() {
                return (cand, NgramHitSource::Generation);
            }
        }
    }

    (Vec::new(), NgramHitSource::None)
}

/// Where the lookup hit fired — for telemetry / cascade-decision logging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgramHitSource {
    /// Match found in the original prompt — input-grounded copying.
    Prompt,
    /// Match found in the recent generation tail — model is repeating
    /// a pattern.
    Generation,
    /// No match at any n.
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_match_returns_empty() {
        let ctx = vec![1, 2, 3, 4, 5];
        let search = vec![100, 200, 300];
        let cfg = NgramLookupConfig::default();
        assert!(find_candidate(&ctx, &search, cfg).is_empty());
    }

    #[test]
    fn exact_match_returns_continuation() {
        // Context ends in [3, 4]; searchable has [3, 4, 10, 11, 12]
        // followed by [3, 4] again at the end.
        //
        // Right-to-left scan finds the trailing [3, 4] first at start=5;
        // start+n=7=searchable.len() so candidate_start >= len → skip.
        // Next match is at start=0, candidate_start=2, candidate window
        // [10, 11, 12, 3, 4] (capped at num_pred_tokens=5 = full).
        // We DO emit the trailing [3, 4] — if the model is going to
        // produce them anyway, predicting them is fine.
        let search = vec![3, 4, 10, 11, 12, 3, 4];
        let ctx = vec![99, 3, 4];
        let cfg = NgramLookupConfig {
            max_ngram_size: 2,
            min_ngram_size: 2,
            num_pred_tokens: 5,
        };
        assert_eq!(find_candidate(&ctx, &search, cfg), vec![10, 11, 12, 3, 4]);
    }

    #[test]
    fn falls_back_from_long_to_short_ngram() {
        // No 4-gram match but a 2-gram match exists.
        let search = vec![1, 2, 99, 50, 60, 1, 2, 70, 80, 90];
        let ctx = vec![100, 100, 1, 2];
        let cfg = NgramLookupConfig {
            max_ngram_size: 4,
            min_ngram_size: 2,
            num_pred_tokens: 3,
        };
        // Right-to-left: trailing [1, 2] is the self-match → skip. Next
        // is at index 5 → [70, 80, 90].
        assert_eq!(find_candidate(&ctx, &search, cfg), vec![70, 80, 90]);
    }

    #[test]
    fn most_recent_match_wins_for_rag_use_case() {
        // RAG-style: the chunk the user just pasted is at the end of the
        // prompt. We want the LATEST match, not the earliest.
        let search = vec![
            5, 6, 7, 100, // earlier chunk, "5,6,7" → "100"
            8, 9, 10, // separator
            5, 6, 7, 200, // recent chunk, "5,6,7" → "200"
        ];
        let ctx = vec![999, 5, 6, 7];
        let cfg = NgramLookupConfig {
            max_ngram_size: 3,
            min_ngram_size: 3,
            num_pred_tokens: 1,
        };
        assert_eq!(find_candidate(&ctx, &search, cfg), vec![200]);
    }

    #[test]
    fn caps_at_num_pred_tokens() {
        let search = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let ctx = vec![99, 1, 2];
        let cfg = NgramLookupConfig {
            max_ngram_size: 2,
            min_ngram_size: 2,
            num_pred_tokens: 3,
        };
        assert_eq!(find_candidate(&ctx, &search, cfg), vec![3, 4, 5]);
    }

    #[test]
    fn cascade_prefers_prompt_match() {
        let prompt_len = 5;
        let prompt = vec![1, 2, 99, 50, 60];
        // Generated tokens: 7,8,9,1,2 (self-match in generation tail too)
        let mut full = prompt.clone();
        full.extend([7u32, 8, 9, 1, 2]);
        let cfg = NgramLookupConfig {
            max_ngram_size: 2,
            min_ngram_size: 2,
            num_pred_tokens: 1,
        };
        let (cand, source) = cascade_find_candidate(&full, prompt_len, 5, cfg);
        // Prompt has [1,2] at index 0 → continuation is [99]. The cascade
        // should prefer the prompt match over the generation-tail match.
        assert_eq!(cand, vec![99]);
        assert_eq!(source, NgramHitSource::Prompt);
    }

    #[test]
    fn cascade_falls_through_to_generation() {
        // Prompt has NO match for the context tail; generation tail does.
        let prompt_len = 3;
        let prompt = vec![100, 200, 300];
        let mut full = prompt.clone();
        // Generation: 5,6,99,5,6 — context tail is [5, 6], which matches
        // earlier in the generation but NOT in the prompt.
        full.extend([5u32, 6, 99, 5, 6]);
        let cfg = NgramLookupConfig {
            max_ngram_size: 2,
            min_ngram_size: 2,
            num_pred_tokens: 1,
        };
        let (cand, source) = cascade_find_candidate(&full, prompt_len, 10, cfg);
        assert_eq!(cand, vec![99]);
        assert_eq!(source, NgramHitSource::Generation);
    }

    #[test]
    fn cascade_no_match_returns_none() {
        let prompt = vec![1, 2, 3];
        let mut full = prompt.clone();
        full.extend([10u32, 20, 30]);
        let cfg = NgramLookupConfig::default();
        let (cand, source) = cascade_find_candidate(&full, prompt.len(), 100, cfg);
        assert!(cand.is_empty());
        assert_eq!(source, NgramHitSource::None);
    }

    #[test]
    fn empty_context_safe() {
        let cfg = NgramLookupConfig::default();
        assert!(find_candidate(&[], &[1, 2, 3], cfg).is_empty());
        assert!(find_candidate(&[1, 2, 3], &[], cfg).is_empty());
    }

    #[test]
    fn ngram_larger_than_context_clamps_down() {
        // max_ngram_size=4 but context only has 2 tokens — should clamp.
        let search = vec![1, 2, 99];
        let ctx = vec![1, 2];
        let cfg = NgramLookupConfig {
            max_ngram_size: 4,
            min_ngram_size: 1,
            num_pred_tokens: 1,
        };
        assert_eq!(find_candidate(&ctx, &search, cfg), vec![99]);
    }
}
