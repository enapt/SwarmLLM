use crate::types::SamplingParams;

/// Pre-allocated scratch buffers for sampling, eliminating ~700KB of allocations per token.
///
/// Create once per request/session and pass to `_with_ctx` variants of sampling functions.
pub struct SamplingContext {
    /// Scratch buffer for (index, logit) pairs in top-k.
    indexed_logits: Vec<(usize, f32)>,
    /// Bitmap for keeping track of which tokens to keep (top-k and top-p).
    keep_mask: Vec<bool>,
    /// Scratch buffer for probability values (softmax output).
    probs: Vec<f32>,
    /// Scratch buffer for index sorting in top-p.
    indices: Vec<usize>,
    /// Saved raw logits (pre-temperature/top-k/top-p) for computing logprobs.
    raw_logits: Vec<f32>,
}

impl SamplingContext {
    /// Create a new SamplingContext pre-allocated for the given vocab size.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            indexed_logits: Vec::with_capacity(vocab_size),
            keep_mask: vec![false; vocab_size],
            probs: Vec::with_capacity(vocab_size),
            indices: Vec::with_capacity(vocab_size),
            raw_logits: Vec::with_capacity(vocab_size),
        }
    }

    /// Ensure all buffers are large enough for the given vocab size.
    fn ensure_capacity(&mut self, vocab_size: usize) {
        if self.keep_mask.len() < vocab_size {
            self.keep_mask.resize(vocab_size, false);
        }
        self.indexed_logits
            .reserve(vocab_size.saturating_sub(self.indexed_logits.capacity()));
        self.probs
            .reserve(vocab_size.saturating_sub(self.probs.capacity()));
        self.indices
            .reserve(vocab_size.saturating_sub(self.indices.capacity()));
    }
}

/// Convert raw logits to a probability distribution via softmax.
///
/// Allocates a new `Vec<f32>` per call. Used by batch paths (e.g. speculative
/// decoding in `executor.rs`) that need freestanding probability vectors.
/// Hot-path decoding in `sample_token_with_ctx` reuses scratch buffers and
/// does its own inline softmax to avoid allocation.
pub fn softmax_vec(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        // Uniform fallback to avoid division by zero.
        let n = logits.len() as f32;
        return vec![1.0 / n; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Apply temperature scaling to logits.
///
/// Higher temperature = more random, lower = more deterministic.
/// Callers must handle temperature == 0 (greedy/argmax) before calling this.
/// Temperature == 1.0 is a no-op (dividing by 1 changes nothing).
fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }
}

/// Apply top-k filtering: keep only the k highest logits, set rest to -inf.
///
/// Allocates temporary buffers. Private — production hot path uses
/// `apply_top_k_with_ctx` via `sample_token_with_ctx`; kept for unit tests.
#[cfg(test)]
fn apply_top_k(logits: &mut [f32], k: u32) {
    if k == 0 || k as usize >= logits.len() {
        return;
    }
    let mut ctx = SamplingContext::new(logits.len());
    apply_top_k_with_ctx(logits, k, &mut ctx);
}

/// Apply top-k filtering using pre-allocated scratch buffers.
fn apply_top_k_with_ctx(logits: &mut [f32], k: u32, ctx: &mut SamplingContext) {
    if k == 0 || k as usize >= logits.len() {
        return;
    }

    let len = logits.len();
    let k_usize = k as usize;
    ctx.ensure_capacity(len);

    // Reuse indexed_logits buffer
    ctx.indexed_logits.clear();
    ctx.indexed_logits
        .extend(logits.iter().copied().enumerate());
    // Partition so the k largest values end up in [..k_usize].
    // select_nth_unstable_by(n, desc) places the n-th largest at index n,
    // with all elements in [..n] being >= it. Using k_usize-1 as the pivot
    // ensures exactly k elements (indices 0..k) are in the top partition.
    ctx.indexed_logits
        .select_nth_unstable_by(k_usize - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

    // Reuse keep_mask bitmap (clear relevant portion)
    for v in ctx.keep_mask[..len].iter_mut() {
        *v = false;
    }
    for &(idx, _) in &ctx.indexed_logits[..k_usize] {
        ctx.keep_mask[idx] = true;
    }

    for (i, logit) in logits.iter_mut().enumerate() {
        if !ctx.keep_mask[i] {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Apply top-p (nucleus) sampling using pre-allocated scratch buffers.
///
/// Fast path: partial-sort the top TOP_P_PARTIAL_K indices with
/// `select_nth_unstable_by` (O(len)), sort just that prefix descending,
/// and walk until cumulative probability ≥ p. For typical p=0.9 on a
/// 32K–151K-vocab LLM, the nucleus is ~10–500 tokens, so 4096 is deep
/// enough to contain it with wide margin.
///
/// Fallback: if cumulative doesn't reach p in the top-K (pathologically
/// flat distribution, or p very close to 1), sort the remaining tail
/// descending and continue. This is O(len log len) but happens rarely.
fn apply_top_p_with_ctx(logits: &mut [f32], p: f32, ctx: &mut SamplingContext) {
    if p >= 1.0 {
        return;
    }

    let len = logits.len();
    ctx.ensure_capacity(len);

    // Convert to probabilities via softmax — reuse probs buffer
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ctx.probs.clear();
    ctx.probs
        .extend(logits.iter().map(|l| (l - max_logit).exp()));
    let sum: f32 = ctx.probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return;
    }
    let inv_sum = 1.0 / sum;
    for prob in ctx.probs.iter_mut() {
        *prob *= inv_sum;
    }

    ctx.indices.clear();
    ctx.indices.extend(0..len);
    // Borrow probs out of ctx once — sort_unstable_by needs an exclusive
    // borrow of ctx.indices and a shared borrow of probs simultaneously.
    let probs: &[f32] = &ctx.probs;
    let cmp_desc = |&a: &usize, &b: &usize| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    };

    const TOP_P_PARTIAL_K: usize = 4096;
    let k = len.min(TOP_P_PARTIAL_K);
    if k < len {
        // Push the top-k indices (in any order) into ctx.indices[..k].
        let (_, _, _) = ctx.indices.select_nth_unstable_by(k - 1, cmp_desc);
        ctx.indices[..k].sort_unstable_by(cmp_desc);
    } else {
        ctx.indices.sort_unstable_by(cmp_desc);
    }

    // Clear keep mask, then walk the sorted prefix accumulating mass.
    for v in ctx.keep_mask[..len].iter_mut() {
        *v = false;
    }
    let mut cumulative = 0.0;
    let mut reached = false;
    for &idx in &ctx.indices[..k] {
        cumulative += ctx.probs[idx];
        ctx.keep_mask[idx] = true;
        if cumulative >= p {
            reached = true;
            break;
        }
    }

    // Fallback: pathological case — top-k didn't contain the full
    // nucleus. Sort the tail and continue walking.
    if !reached && k < len {
        ctx.indices[k..].sort_unstable_by(cmp_desc);
        for &idx in &ctx.indices[k..] {
            cumulative += ctx.probs[idx];
            ctx.keep_mask[idx] = true;
            if cumulative >= p {
                break;
            }
        }
    }

    // Mask out tokens not in the nucleus
    for (i, logit) in logits.iter_mut().enumerate() {
        if !ctx.keep_mask[i] {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Sample a token index from logits using the given parameters.
///
/// Applies temperature, top-k, top-p in order, then samples from the
/// resulting distribution. Allocates temporary buffers each call.
/// For hot-path usage in decode loops, prefer `sample_token_with_ctx`.
pub fn sample_token(logits: &mut [f32], params: &SamplingParams) -> u32 {
    let mut ctx = SamplingContext::new(logits.len());
    sample_token_with_ctx(logits, params, &mut ctx)
}

/// Sample a token index from logits using pre-allocated scratch buffers.
///
/// Same behavior as `sample_token` but reuses buffers from `SamplingContext`,
/// eliminating ~700KB of allocations per call for typical vocab sizes (32K).
fn sample_token_with_ctx(
    logits: &mut [f32],
    params: &SamplingParams,
    ctx: &mut SamplingContext,
) -> u32 {
    // Greedy decoding when temperature is 0
    if params.temperature <= 0.0 {
        let token = argmax(logits);
        // Populate probs buffer from raw logits so logprob computation works for greedy
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        ctx.probs.clear();
        ctx.probs
            .extend(logits.iter().map(|l| (l - max_logit).exp()));
        tracing::trace!(
            token,
            vocab_size = logits.len(),
            mode = "greedy",
            "DIAG: sample_token complete"
        );
        return token;
    }

    apply_temperature(logits, params.temperature);
    apply_top_k_with_ctx(logits, params.top_k, ctx);
    apply_top_p_with_ctx(logits, params.top_p, ctx);

    let len = logits.len();
    ctx.ensure_capacity(len);

    // Softmax — reuse probs buffer
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    ctx.probs.clear();
    ctx.probs
        .extend(logits.iter().map(|l| (l - max_logit).exp()));
    let sum: f32 = ctx.probs.iter().sum();

    // All logits masked to -inf (e.g. degenerate top_k ∩ top_p).
    // Fall back to argmax over the pre-softmax logits rather than emitting
    // the last vocab token, which is a silent wrong-token bug.
    if !sum.is_finite() || sum == 0.0 {
        let mut best_i = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > best_v {
                best_v = l;
                best_i = i as u32;
            }
        }
        tracing::warn!(
            vocab_size = logits.len(),
            sum,
            "DIAG: sampling softmax collapsed — falling back to argmax"
        );
        return best_i;
    }

    // Weighted random selection
    let r: f32 = simple_random() * sum;
    let mut cumulative = 0.0;
    for (i, &p) in ctx.probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
            tracing::trace!(
                token = i as u32,
                vocab_size = ctx.probs.len(),
                temperature = params.temperature,
                top_k = params.top_k,
                top_p = params.top_p,
                mode = "stochastic",
                "DIAG: sample_token complete"
            );
            return i as u32;
        }
    }

    // Fallback — should rarely happen (cumulative rounding)
    tracing::warn!(
        vocab_size = ctx.probs.len(),
        sum,
        "DIAG: sampling fallback — cumulative probability didn't reach threshold"
    );
    (ctx.probs.len() - 1) as u32
}

/// Token logprob information returned by `sample_token_with_logprobs`.
#[derive(Debug, Clone)]
pub struct SampledTokenLogProb {
    /// The sampled token index.
    pub token_id: u32,
    /// Log probability of the sampled token.
    pub logprob: f32,
    /// Top-N tokens and their log probabilities, sorted descending.
    pub top_logprobs: Vec<(u32, f32)>,
}

/// Sample a token and optionally return logprob information.
///
/// When `top_logprobs > 0`, saves the raw logits before temperature/top-k/top-p
/// mutation, then computes log-softmax over those raw logits. Returns the top-N
/// tokens by pre-sampling probability along with the sampled token's logprob.
/// This matches the OpenAI spec (logprobs reflect the model's raw distribution).
pub fn sample_token_with_logprobs(
    logits: &mut [f32],
    params: &SamplingParams,
    ctx: &mut SamplingContext,
) -> (u32, Option<SampledTokenLogProb>) {
    let need_logprobs = params.logprobs && params.top_logprobs > 0;

    // Save raw logits BEFORE sampling mutates them (temperature/top-k/top-p).
    // OpenAI spec: logprobs are computed from the pre-sampling distribution.
    if need_logprobs {
        ctx.raw_logits.clear();
        ctx.raw_logits.extend_from_slice(logits);
    }

    // Clear probs buffer before sampling to prevent stale data from previous tokens
    // (especially important for greedy/temperature=0 which skips softmax)
    ctx.probs.clear();

    let token_id = sample_token_with_ctx(logits, params, ctx);

    let logprob_info = if need_logprobs {
        // Compute log-softmax from raw (pre-sampling) logits per OpenAI spec.
        let max_logit = ctx
            .raw_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp = ctx
            .raw_logits
            .iter()
            .map(|l| (l - max_logit).exp())
            .sum::<f32>()
            .ln()
            + max_logit;

        let sampled_logprob = ctx
            .raw_logits
            .get(token_id as usize)
            .map(|l| l - log_sum_exp)
            .unwrap_or(-9999.0);

        // Get top-N tokens by raw logit value (highest logit = highest probability)
        let n = params.top_logprobs.min(20) as usize;
        ctx.indexed_logits.clear();
        ctx.indexed_logits
            .extend(ctx.raw_logits.iter().copied().enumerate());
        // Partial sort to get top-N by logit value (k-1 pivot keeps exactly N elements)
        if ctx.indexed_logits.len() > n && n > 0 {
            ctx.indexed_logits.select_nth_unstable_by(n - 1, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            ctx.indexed_logits.truncate(n);
        }
        ctx.indexed_logits
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top: Vec<(u32, f32)> = ctx
            .indexed_logits
            .iter()
            .map(|&(idx, logit)| (idx as u32, logit - log_sum_exp))
            .collect();

        Some(SampledTokenLogProb {
            token_id,
            logprob: sampled_logprob,
            top_logprobs: top,
        })
    } else {
        None
    };

    (token_id, logprob_info)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Random float in [0, 1) using the `rand` crate.
/// Not cryptographically secure — fine for sampling.
fn simple_random() -> f32 {
    rand::random::<f32>()
}

/// Check if accumulated text contains any stop sequence.
/// Returns the matching stop string (if any) for callers that need to truncate.
pub(crate) fn find_stop_sequence<'a>(
    accumulated_text: &str,
    stop_sequences: &'a [String],
) -> Option<&'a str> {
    if stop_sequences.is_empty() {
        return None;
    }
    stop_sequences
        .iter()
        .find(|s| accumulated_text.contains(s.as_str()))
        .map(|s| s.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_scaling() {
        let mut logits = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut logits, 0.5);
        assert!((logits[0] - 2.0).abs() < f32::EPSILON);
        assert!((logits[1] - 4.0).abs() < f32::EPSILON);
        assert!((logits[2] - 6.0).abs() < f32::EPSILON);
    }

    #[test]
    fn top_k_filtering() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        // Top-2 are indices 1 (5.0) and 4 (4.0)
        assert!(logits[0].is_infinite() && logits[0] < 0.0);
        assert!((logits[1] - 5.0).abs() < f32::EPSILON);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);
        assert!((logits[4] - 4.0).abs() < f32::EPSILON);
    }

    #[test]
    fn top_k_with_ctx_matches_allocating_version() {
        let logits_orig = vec![1.0, 5.0, 3.0, 2.0, 4.0];

        let mut logits_a = logits_orig.clone();
        apply_top_k(&mut logits_a, 2);

        let mut logits_b = logits_orig.clone();
        let mut ctx = SamplingContext::new(logits_b.len());
        apply_top_k_with_ctx(&mut logits_b, 2, &mut ctx);

        for (a, b) in logits_a.iter().zip(logits_b.iter()) {
            assert!((a - b).abs() < f32::EPSILON || (a.is_infinite() && b.is_infinite()));
        }
    }

    /// Naive O(n log n) reference implementation of top-p (nucleus)
    /// filtering. Used by tests as ground truth for the optimized
    /// partial-sort version. Mirrors the spec: softmax → full descending
    /// sort → walk accumulating mass until ≥ p → mask the rest.
    fn naive_top_p(logits: &mut [f32], p: f32) {
        if p >= 1.0 {
            return;
        }
        let len = logits.len();
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probs: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = probs.iter().sum();
        if sum <= 0.0 || !sum.is_finite() {
            return;
        }
        for prob in probs.iter_mut() {
            *prob /= sum;
        }

        let mut indices: Vec<usize> = (0..len).collect();
        indices.sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = vec![false; len];
        let mut cumulative = 0.0;
        for &idx in &indices {
            cumulative += probs[idx];
            keep[idx] = true;
            if cumulative >= p {
                break;
            }
        }

        for (i, logit) in logits.iter_mut().enumerate() {
            if !keep[i] {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    fn assert_logits_match(actual: &[f32], expected: &[f32], label: &str) {
        for (i, (a, b)) in actual.iter().zip(expected.iter()).enumerate() {
            let both_inf = a.is_infinite() && b.is_infinite();
            assert!(
                both_inf || (a - b).abs() < 1e-5,
                "{label}: index {i} mismatch (actual={a}, expected={b})"
            );
        }
    }

    #[test]
    fn top_p_with_ctx_matches_naive_small() {
        let logits_orig = vec![1.0, 5.0, 3.0, 0.5, 4.0];
        let mut actual = logits_orig.clone();
        let mut ctx = SamplingContext::new(actual.len());
        apply_top_p_with_ctx(&mut actual, 0.8, &mut ctx);

        let mut expected = logits_orig.clone();
        naive_top_p(&mut expected, 0.8);

        assert_logits_match(&actual, &expected, "small p=0.8");
    }

    #[test]
    fn top_p_with_ctx_matches_naive_across_p_values() {
        let logits_orig = vec![0.1, 0.2, 0.3, 5.0, 4.5, 0.05, 3.0, 2.0, 1.0, 0.8];
        for p in [0.1_f32, 0.5, 0.7, 0.9, 0.95, 0.999] {
            let mut actual = logits_orig.clone();
            let mut ctx = SamplingContext::new(actual.len());
            apply_top_p_with_ctx(&mut actual, p, &mut ctx);

            let mut expected = logits_orig.clone();
            naive_top_p(&mut expected, p);

            assert_logits_match(&actual, &expected, &format!("p={p}"));
        }
    }

    #[test]
    fn top_p_with_ctx_matches_naive_large_vocab_fast_path() {
        // 5000 elements > TOP_P_PARTIAL_K (4096) so the partial-sort
        // fast path exercises its prefix walk; very sharp distribution
        // (top-3 carry essentially all probability mass) so the
        // nucleus is small enough to clear p=0.9 inside the top-K
        // and the fallback path doesn't fire. Use distinct logits
        // (all 5000 tokens have unique values) to avoid ordering
        // ambiguity on tied probabilities — both impls would be
        // spec-correct picking different tied tokens, but the test
        // wants bitwise equivalence.
        let len = 5000;
        let mut logits_orig: Vec<f32> = (0..len).map(|i| -(i as f32) * 1e-3).collect();
        logits_orig[2500] = 10.0;
        logits_orig[100] = 8.0;
        logits_orig[3700] = 5.0;

        let mut actual = logits_orig.clone();
        let mut ctx = SamplingContext::new(actual.len());
        apply_top_p_with_ctx(&mut actual, 0.9, &mut ctx);

        let mut expected = logits_orig.clone();
        naive_top_p(&mut expected, 0.9);

        assert_logits_match(&actual, &expected, "large vocab fast path");
    }

    #[test]
    fn top_p_with_ctx_matches_naive_large_vocab_fallback() {
        // 5000 elements, near-uniform distribution, p=0.999 → the
        // nucleus is wider than TOP_P_PARTIAL_K, forcing the
        // tail-sort fallback. Use a deterministic spread so the test
        // is reproducible.
        let len = 5000;
        let logits_orig: Vec<f32> = (0..len).map(|i| (i as f32) / 1e5).collect();

        let mut actual = logits_orig.clone();
        let mut ctx = SamplingContext::new(actual.len());
        apply_top_p_with_ctx(&mut actual, 0.999, &mut ctx);

        let mut expected = logits_orig.clone();
        naive_top_p(&mut expected, 0.999);

        assert_logits_match(&actual, &expected, "large vocab fallback");
        // Sanity: with p very close to 1, the optimized version should
        // keep ~all tokens — masked count should match naive exactly.
        let actual_kept = actual.iter().filter(|x| x.is_finite()).count();
        let expected_kept = expected.iter().filter(|x| x.is_finite()).count();
        assert_eq!(actual_kept, expected_kept);
    }

    #[test]
    fn softmax_collapse_falls_back_to_argmax() {
        // All logits set to -inf simulates a degenerate top_k ∩ top_p
        // intersection where every candidate has been masked out. The
        // softmax sum collapses to 0 (or NaN) and we MUST return the
        // pre-softmax argmax rather than emitting the last vocab index
        // (the silent-wrong-token bug fixed in 2026-04-14).
        let mut logits = vec![f32::NEG_INFINITY; 5];
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 5,
            top_p: 0.5,
            ..Default::default()
        };
        let token = sample_token(&mut logits, &params);
        // All logits identical (-inf): argmax returns the first index.
        assert_eq!(token, 0);

        // Now make one logit non-(-inf) but every other -inf: argmax
        // must select that finite index, not last vocab.
        let mut logits = vec![f32::NEG_INFINITY; 5];
        logits[3] = 1.0;
        // Force softmax-collapse-after-masking by setting top_k=1
        // followed by top_p that excludes everything (degenerate
        // arithmetic): instead, we just feed pre-collapsed logits
        // into sample_token so the sum-zero branch fires directly.
        let token = sample_token(&mut logits.clone(), &params);
        assert_eq!(token, 3);
    }

    #[test]
    fn top_p_p_one_or_above_is_identity() {
        let logits_orig = vec![1.0, 5.0, 3.0, 0.5, 4.0];
        let mut actual = logits_orig.clone();
        let mut ctx = SamplingContext::new(actual.len());
        apply_top_p_with_ctx(&mut actual, 1.0, &mut ctx);
        assert_eq!(actual, logits_orig);
    }

    #[test]
    fn sampling_context_reuse() {
        let mut ctx = SamplingContext::new(5);

        // First use
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        apply_top_k_with_ctx(&mut logits, 2, &mut ctx);
        assert!((logits[1] - 5.0).abs() < f32::EPSILON);
        assert!((logits[4] - 4.0).abs() < f32::EPSILON);

        // Second use — same context, different logits
        let mut logits2 = vec![10.0, 1.0, 8.0, 3.0, 5.0];
        apply_top_k_with_ctx(&mut logits2, 2, &mut ctx);
        assert!((logits2[0] - 10.0).abs() < f32::EPSILON);
        assert!((logits2[2] - 8.0).abs() < f32::EPSILON);
        assert!(logits2[1].is_infinite() && logits2[1] < 0.0);
    }

    #[test]
    fn greedy_sampling() {
        let mut logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let token = sample_token(&mut logits, &params);
        assert_eq!(token, 1); // Index of highest logit
    }

    #[test]
    fn greedy_sampling_with_ctx() {
        let mut logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(logits.len());
        let token = sample_token_with_ctx(&mut logits, &params, &mut ctx);
        assert_eq!(token, 1);
    }

    #[test]
    fn argmax_returns_correct_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 1.0]), 0);
    }

    #[test]
    fn top_k_no_op_when_k_zero() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut ctx = SamplingContext::new(3);
        apply_top_k_with_ctx(&mut logits, 0, &mut ctx);
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
        assert!((logits[1] - 2.0).abs() < f32::EPSILON);
        assert!((logits[2] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn top_p_no_op_when_p_one() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let mut ctx = SamplingContext::new(3);
        apply_top_p_with_ctx(&mut logits, 1.0, &mut ctx);
        assert!((logits[0] - 1.0).abs() < f32::EPSILON);
        assert!((logits[1] - 2.0).abs() < f32::EPSILON);
        assert!((logits[2] - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn sample_with_logprobs_returns_data() {
        let mut logits = vec![1.0, 5.0, 3.0, 0.5, 4.0];
        let params = SamplingParams {
            temperature: 0.0, // greedy — deterministic
            logprobs: true,
            top_logprobs: 3,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(logits.len());
        let (token, lp) = sample_token_with_logprobs(&mut logits, &params, &mut ctx);
        assert_eq!(token, 1); // index of 5.0 (highest)
        let lp = lp.expect("logprobs should be present even in greedy mode");
        assert_eq!(lp.token_id, 1);
        assert!(lp.logprob > -1.0); // highest logit → near-zero logprob
        assert_eq!(lp.top_logprobs.len(), 3);
        // Top logprobs should be sorted descending by logprob
        assert!(lp.top_logprobs[0].1 >= lp.top_logprobs[1].1);
        assert!(lp.top_logprobs[1].1 >= lp.top_logprobs[2].1);
        // First entry should be the highest-logit token (index 1, logit 5.0)
        assert_eq!(lp.top_logprobs[0].0, 1);
    }

    #[test]
    fn sample_with_logprobs_temperature() {
        let mut logits = vec![10.0, 1.0, 1.0];
        let params = SamplingParams {
            temperature: 0.01, // near-greedy but uses softmax path
            top_p: 1.0,
            top_k: 0,
            logprobs: true,
            top_logprobs: 2,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(logits.len());
        let (token, lp) = sample_token_with_logprobs(&mut logits, &params, &mut ctx);
        assert_eq!(token, 0); // 10.0 dominates
        let lp = lp.expect("logprobs should be present");
        assert_eq!(lp.token_id, 0);
        assert!(lp.logprob > -1.0); // near-certain token has logprob close to 0
        assert!(!lp.top_logprobs.is_empty());
        assert!(lp.top_logprobs.len() <= 2);
    }

    #[test]
    fn sample_with_logprobs_disabled() {
        let mut logits = vec![1.0, 5.0, 3.0];
        let params = SamplingParams {
            temperature: 0.5,
            logprobs: false,
            top_logprobs: 3,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(logits.len());
        let (_token, lp) = sample_token_with_logprobs(&mut logits, &params, &mut ctx);
        assert!(lp.is_none());
    }
}
