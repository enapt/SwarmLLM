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
//! The speculative loops' cascade ([`copied_run`]) weighs the matched
//! n-gram's last 32 occurrences instead of the first one found, and ends the
//! run where they stop agreeing (#338).
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
    find_occurrences(context, searchable, cfg, 1)
        .map(|(starts, _)| {
            let from = starts[0];
            searchable[from..(from + cfg.num_pred_tokens).min(searchable.len())].to_vec()
        })
        .unwrap_or_default()
}

/// Where [`find_candidate`] finds `context`'s tail in `searchable`: the
/// indices in `searchable` where its continuations start — the most recent
/// occurrence first, at most `most` of them, never empty — and the n-gram
/// that matched (the longest with any occurrence).
fn find_occurrences(
    context: &[u32],
    searchable: &[u32],
    cfg: NgramLookupConfig,
    most: usize,
) -> Option<(Vec<usize>, usize)> {
    if context.is_empty() || searchable.is_empty() || most == 0 {
        return None;
    }
    let max_n = cfg.max_ngram_size.min(context.len()).min(searchable.len());
    let min_n = cfg.min_ngram_size.max(1);
    if max_n < min_n {
        return None;
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
        let mut starts = Vec::new();
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
            starts.push(candidate_start);
            if starts.len() == most {
                break;
            }
        }
        if !starts.is_empty() {
            return Some((starts, n));
        }
    }
    None
}

/// How many of the matched n-gram's most recent occurrences a copied run
/// weighs ([`copied_run`], #338).
pub const OCCURRENCES_WEIGHED: usize = 32;

/// A copied run goes on while the token most of the occurrences still
/// following it continue with holds at least this share of them, in percent
/// ([`agreed_run`], #338).
const AGREEING_PCT: usize = 67;

/// `SWARMLLM_COPY_AGREEMENT=0|off|false`: a copied run is the most recent
/// occurrence's continuation, whatever the others did (before #338), for an
/// A/B inside one binary. Read once.
fn copy_agreement_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_COPY_AGREEMENT").ok().as_deref(),
            Some("0") | Some("off") | Some("false")
        )
    })
}

/// What the occurrences whose continuations start at `starts` (most recent
/// first) go on with, walked together: at each token the one most of those
/// still following continue with (a tie goes to the most recent), ended once
/// it holds under [`AGREEING_PCT`] of them — the first token is always taken —
/// and, once ONE occurrence is left, that one's continuation. At most `most`
/// tokens.
///
/// SuffixDecoding (arXiv 2411.04975) stops a draft the same way, where the
/// frequency its tree records for the next token falls (`min_token_prob`).
/// Measured offline on 400 SWE-agent sessions (`docs/plans/faster_than_local.md`
/// § 7.6): with each run capped as the flow caps it, the share of reply tokens
/// a copy covered stayed at 64.6% while the rows copied fell from 3.29 to 2.36
/// a reply token (−28%). Not used on the owner's earlier replies
/// (`owner_replies`), where it saved no rows and cost 0.5 points.
fn agreed_run(searchable: &[u32], starts: &[usize], most: usize) -> Vec<u32> {
    let mut run = Vec::new();
    let mut following = starts.to_vec();
    while run.len() < most {
        if let [only] = following[..] {
            let rest = most - run.len();
            run.extend(searchable.iter().skip(only).take(rest));
            break;
        }
        let mut leader: Option<(u32, usize)> = None;
        for (i, &at) in following.iter().enumerate() {
            let Some(&token) = searchable.get(at) else {
                continue;
            };
            let counted_already = following[..i]
                .iter()
                .any(|&before| searchable.get(before) == Some(&token));
            if counted_already {
                continue;
            }
            let votes = following
                .iter()
                .filter(|&&other| searchable.get(other) == Some(&token))
                .count();
            if leader.is_none_or(|(_, most_votes)| votes > most_votes) {
                leader = Some((token, votes));
            }
        }
        let Some((token, votes)) = leader else {
            break;
        };
        // An occurrence that has ended still counts against the leader.
        if !run.is_empty() && votes * 100 < AGREEING_PCT * following.len() {
            break;
        }
        run.push(token);
        following = following
            .into_iter()
            .filter(|&at| searchable.get(at) == Some(&token))
            .map(|at| at + 1)
            .collect();
    }
    run
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
    match copied_run(
        prompt_tokens,
        prompt_len,
        recent_gen_window,
        cfg,
        cfg.num_pred_tokens,
    ) {
        Some(run) => (run.tokens, run.source),
        None => (Vec::new(), NgramHitSource::None),
    }
}

/// How far back a match is measured ([`CopiedRun::matched`]).
pub const MATCHED_MOST: usize = 64;

/// A run of tokens that followed an earlier occurrence of the context's tail
/// ([`copied_run`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopiedRun {
    /// The tokens after the occurrence, at most the count asked for.
    pub tokens: Vec<u32>,
    /// How many of the context's last tokens equal the ones before the run:
    /// the n-gram that found it, measured back while they go on matching, up
    /// to [`MATCHED_MOST`]. The longer the match, the likelier the copy goes
    /// on — SuffixDecoding (arXiv 2411.04975) sizes its guess by it (α·p).
    pub matched: usize,
    pub source: NgramHitSource,
}

/// The cascade [`cascade_find_candidate`] runs — the prompt first, then the
/// last `recent_gen_window` tokens of the reply — with up to `most` tokens of
/// the run and how far its match goes back. The run is what the matched
/// n-gram's last [`OCCURRENCES_WEIGHED`] occurrences AGREE on
/// ([`agreed_run`], #338); the match is measured back from the most recent.
pub fn copied_run(
    context: &[u32],
    prompt_len: usize,
    recent_gen_window: usize,
    cfg: NgramLookupConfig,
    most: usize,
) -> Option<CopiedRun> {
    // Pass 1: search the prompt for matches of the recent context tail.
    // Searches the prompt slice only — the most common "input-grounded"
    // case (copying from prompt to output).
    let prompt = (prompt_len > 0 && prompt_len <= context.len())
        .then_some((0..prompt_len, NgramHitSource::Prompt));
    // Pass 2: search the recent generation tail for self-matches. Useful
    // when the model is enumerating a list or applying a repeating
    // transformation. Searching too large a window slows the lookup
    // without adding hit rate, so we cap.
    let gen_start = context
        .len()
        .saturating_sub(recent_gen_window)
        .max(prompt_len);
    let reply = (gen_start < context.len())
        .then_some((gen_start..context.len(), NgramHitSource::Generation));
    let weighed = if copy_agreement_on() {
        OCCURRENCES_WEIGHED
    } else {
        1
    };
    [prompt, reply]
        .into_iter()
        .flatten()
        .find_map(|(range, source)| {
            let searchable = &context[range];
            let (starts, n) = find_occurrences(context, searchable, cfg, weighed)?;
            let from = starts[0];
            let tokens = agreed_run(searchable, &starts, most);
            if tokens.is_empty() {
                return None;
            }
            let mut matched = n;
            while matched < MATCHED_MOST
                && matched < from
                && matched < context.len()
                && searchable[from - matched - 1] == context[context.len() - matched - 1]
            {
                matched += 1;
            }
            Some(CopiedRun {
                tokens,
                matched,
                source,
            })
        })
}

/// Where the lookup hit fired — for telemetry / cascade-decision logging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgramHitSource {
    /// Match found in the original prompt — input-grounded copying.
    Prompt,
    /// Match found in the recent generation tail — model is repeating
    /// a pattern.
    Generation,
    /// Match found in one of the owner's earlier replies
    /// (`owner_replies`, #337).
    EarlierReply,
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
    fn a_copied_run_measures_how_far_back_its_match_goes() {
        // The reply's last four tokens occur in the prompt; the token before
        // them (12) matches too, the one before that (77 against 11) does not.
        let prompt: Vec<u32> = (10..20).collect();
        let mut context = prompt.clone();
        context.extend([77u32, 12, 13, 14, 15, 16]);
        let run = copied_run(&context, prompt.len(), 500, NgramLookupConfig::default(), 3).unwrap();
        assert_eq!(run.tokens, vec![17, 18, 19]);
        assert_eq!(run.matched, 5);
        assert_eq!(run.source, NgramHitSource::Prompt);
        // The cascade it serves still answers as before.
        let (cand, source) =
            cascade_find_candidate(&context, prompt.len(), 500, NgramLookupConfig::default());
        assert_eq!(cand, vec![17, 18, 19]);
        assert_eq!(source, NgramHitSource::Prompt);

        // A reply copying the prompt from its start matches as far back as it
        // has copied, measured no further than MATCHED_MOST.
        let prompt: Vec<u32> = (0..100).collect();
        let mut context = prompt.clone();
        context.extend(0u32..80);
        let run = copied_run(
            &context,
            prompt.len(),
            500,
            NgramLookupConfig::default(),
            32,
        )
        .unwrap();
        assert_eq!(run.tokens, (80u32..100).collect::<Vec<_>>());
        assert_eq!(run.matched, MATCHED_MOST);
    }

    #[test]
    fn a_copied_run_ends_where_the_earlier_occurrences_stop_agreeing() {
        // Four earlier occurrences of the reply's last tokens [1, 2, 3, 4],
        // oldest first, each followed by a separator of its own.
        let prompt: Vec<u32> = [
            &[1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 100][..],
            &[1, 2, 3, 4, 5, 6, 7, 50, 60, 61, 101],
            &[1, 2, 3, 4, 5, 6, 7, 9, 70, 71, 102],
            &[1, 2, 3, 4, 5, 6, 30, 31, 80, 81, 103],
        ]
        .concat();
        let mut context = prompt.clone();
        context.extend([90u32, 1, 2, 3, 4]);
        let run = copied_run(
            &context,
            prompt.len(),
            500,
            NgramLookupConfig::default(),
            32,
        )
        .unwrap();
        // All four go on with 5 and 6, three of four with 7 (the newest goes
        // to 30); after 7 the three name 9, 50 and 8, one each — no
        // agreement, so the run ends there rather than copying the newest
        // occurrence's 30, 31, 80, ….
        assert_eq!(run.tokens, vec![5, 6, 7]);
        // The match is measured back from the most recent occurrence: the
        // reply's 90 does not match the separator 102 before it.
        assert_eq!(run.matched, 4);

        // Two of three is not agreement either (under 67%).
        let prompt: Vec<u32> = [
            &[1, 2, 3, 4, 5, 6, 100][..],
            &[1, 2, 3, 4, 5, 6, 101],
            &[1, 2, 3, 4, 5, 7, 102],
        ]
        .concat();
        let mut context = prompt.clone();
        context.extend([90u32, 1, 2, 3, 4]);
        let run = copied_run(
            &context,
            prompt.len(),
            500,
            NgramLookupConfig::default(),
            32,
        )
        .unwrap();
        // 5 by all three; then 6 by two of them against the newest's 7 —
        // under 67%, so the run ends.
        assert_eq!(run.tokens, vec![5]);
    }

    #[test]
    fn a_copied_run_with_one_occurrence_copies_it_and_a_tie_goes_to_the_newest() {
        // One occurrence: its whole continuation, as far as asked.
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut context = prompt.clone();
        context.extend([90u32, 1, 2, 3, 4]);
        let run = copied_run(&context, prompt.len(), 500, NgramLookupConfig::default(), 4).unwrap();
        assert_eq!(run.tokens, vec![5, 6, 7, 8]);

        // Two occurrences that disagree from the first token: the first is
        // always taken, a tie going to the newest — which alone follows it
        // then, so its continuation is copied as before.
        let prompt: Vec<u32> = vec![1, 2, 3, 4, 10, 11, 12, 1, 2, 3, 4, 20, 21, 22];
        let mut context = prompt.clone();
        context.extend([90u32, 1, 2, 3, 4]);
        let run = copied_run(
            &context,
            prompt.len(),
            500,
            NgramLookupConfig::default(),
            32,
        )
        .unwrap();
        assert_eq!(run.tokens, vec![20, 21, 22]);
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
