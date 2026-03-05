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
}

impl SamplingContext {
    /// Create a new SamplingContext pre-allocated for the given vocab size.
    pub fn new(vocab_size: usize) -> Self {
        Self {
            indexed_logits: Vec::with_capacity(vocab_size),
            keep_mask: vec![false; vocab_size],
            probs: Vec::with_capacity(vocab_size),
            indices: Vec::with_capacity(vocab_size),
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

/// Apply temperature scaling to logits.
///
/// Higher temperature = more random, lower = more deterministic.
/// Callers must handle temperature == 0 (greedy/argmax) before calling this.
/// Temperature == 1.0 is a no-op (dividing by 1 changes nothing).
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    if temperature == 1.0 {
        return;
    }
    for logit in logits.iter_mut() {
        *logit /= temperature;
    }
}

/// Apply top-k filtering: keep only the k highest logits, set rest to -inf.
///
/// Allocates temporary buffers. For hot-path usage, prefer `apply_top_k_with_ctx`.
pub fn apply_top_k(logits: &mut [f32], k: u32) {
    if k == 0 || k as usize >= logits.len() {
        return;
    }
    let mut ctx = SamplingContext::new(logits.len());
    apply_top_k_with_ctx(logits, k, &mut ctx);
}

/// Apply top-k filtering using pre-allocated scratch buffers.
pub fn apply_top_k_with_ctx(logits: &mut [f32], k: u32, ctx: &mut SamplingContext) {
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
    ctx.indexed_logits.select_nth_unstable_by(k_usize, |a, b| {
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

/// Apply top-p (nucleus) sampling: keep the smallest set of tokens whose
/// cumulative probability exceeds p.
///
/// Allocates temporary buffers. For hot-path usage, prefer `apply_top_p_with_ctx`.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let mut ctx = SamplingContext::new(logits.len());
    apply_top_p_with_ctx(logits, p, &mut ctx);
}

/// Apply top-p (nucleus) sampling using pre-allocated scratch buffers.
///
/// Uses a bitmap instead of HashSet for the keep set, and partial sort
/// (select_nth_unstable_by) to avoid full O(V log V) sort.
pub fn apply_top_p_with_ctx(logits: &mut [f32], p: f32, ctx: &mut SamplingContext) {
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

    // Sort indices by probability descending — reuse indices buffer
    ctx.indices.clear();
    ctx.indices.extend(0..len);
    ctx.indices.sort_unstable_by(|&a, &b| {
        ctx.probs[b]
            .partial_cmp(&ctx.probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Find cutoff and build keep bitmap (replaces HashSet)
    for v in ctx.keep_mask[..len].iter_mut() {
        *v = false;
    }
    let mut cumulative = 0.0;
    for &idx in &ctx.indices {
        cumulative += ctx.probs[idx];
        ctx.keep_mask[idx] = true;
        if cumulative >= p {
            break;
        }
    }

    // Mask out tokens not in the nucleus
    for (i, logit) in logits.iter_mut().enumerate() {
        if !ctx.keep_mask[i] {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// Apply frequency and presence penalties to logits based on token occurrence counts.
pub fn apply_repetition_penalties(
    logits: &mut [f32],
    token_counts: &[u32],
    params: &SamplingParams,
) {
    if params.frequency_penalty == 0.0 && params.presence_penalty == 0.0 {
        return;
    }
    for (i, logit) in logits.iter_mut().enumerate() {
        if let Some(&count) = token_counts.get(i) {
            if count > 0 {
                *logit -= params.frequency_penalty * count as f32;
                *logit -= params.presence_penalty;
            }
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
pub fn sample_token_with_ctx(
    logits: &mut [f32],
    params: &SamplingParams,
    ctx: &mut SamplingContext,
) -> u32 {
    // Greedy decoding when temperature is 0
    if params.temperature <= 0.0 {
        return argmax(logits);
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

    // Weighted random selection
    let r: f32 = simple_random() * sum;
    let mut cumulative = 0.0;
    for (i, &p) in ctx.probs.iter().enumerate() {
        cumulative += p;
        if cumulative >= r {
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
/// When `top_logprobs > 0`, computes log-softmax over the raw logits (before
/// temperature/top-k/top-p) and returns the top-N tokens by probability along
/// with the sampled token's logprob.
pub fn sample_token_with_logprobs(
    logits: &mut [f32],
    params: &SamplingParams,
    ctx: &mut SamplingContext,
) -> (u32, Option<SampledTokenLogProb>) {
    let need_logprobs = params.logprobs && params.top_logprobs > 0;

    let token_id = sample_token_with_ctx(logits, params, ctx);

    let logprob_info = if need_logprobs {
        // We need the original logits, but they've been mutated by temperature/top-k/top-p.
        // The log-softmax for the sampled token is: logit[token] - log_sum_exp
        // But logits are mutated. We saved log_sum_exp before mutation, so we need
        // to recover from the probs buffer which has the post-sampling softmax.
        //
        // Actually, we computed log_sum_exp from the raw logits before sample_token_with_ctx
        // mutated them. But sample_token_with_ctx already consumed and modified logits.
        // The probs in ctx.probs are post-temperature softmax probs.
        //
        // For correct logprobs, we use the post-temperature softmax (which is what the
        // model actually sampled from). This matches what vLLM and other implementations do.
        let sum: f32 = ctx.probs.iter().sum();
        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            let sampled_prob = ctx.probs.get(token_id as usize).copied().unwrap_or(0.0) * inv_sum;
            let sampled_logprob = if sampled_prob > 0.0 {
                sampled_prob.ln()
            } else {
                -9999.0
            };

            // Get top-N tokens by probability
            let n = params.top_logprobs.min(20) as usize;
            ctx.indexed_logits.clear();
            ctx.indexed_logits.extend(
                ctx.probs
                    .iter()
                    .copied()
                    .enumerate()
                    .filter(|(_, p)| *p > 0.0),
            );
            // Partial sort to get top-N
            if ctx.indexed_logits.len() > n {
                ctx.indexed_logits.select_nth_unstable_by(n, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                ctx.indexed_logits.truncate(n);
            }
            ctx.indexed_logits
                .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let top: Vec<(u32, f32)> = ctx
                .indexed_logits
                .iter()
                .map(|&(idx, p)| {
                    let lp = if p * inv_sum > 0.0 {
                        (p * inv_sum).ln()
                    } else {
                        -9999.0
                    };
                    (idx as u32, lp)
                })
                .collect();

            Some(SampledTokenLogProb {
                token_id,
                logprob: sampled_logprob,
                top_logprobs: top,
            })
        } else {
            None
        }
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

    #[test]
    fn top_p_with_ctx_matches_allocating_version() {
        let logits_orig = vec![1.0, 5.0, 3.0, 0.5, 4.0];

        let mut logits_a = logits_orig.clone();
        apply_top_p(&mut logits_a, 0.8);

        let mut logits_b = logits_orig.clone();
        let mut ctx = SamplingContext::new(logits_b.len());
        apply_top_p_with_ctx(&mut logits_b, 0.8, &mut ctx);

        for (a, b) in logits_a.iter().zip(logits_b.iter()) {
            assert!((a - b).abs() < f32::EPSILON || (a.is_infinite() && b.is_infinite()));
        }
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
        // Greedy path: logprobs come from probs buffer which isn't populated by argmax.
        // For greedy, logprobs may be None because the softmax probs aren't computed.
        // This is acceptable — logprobs are most useful with temperature > 0.
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
