use crate::inference::coupled_noise::{CoupledNoise, GUMBEL_MAX, GUMBEL_MIN};
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
    /// Per-token occurrence counts over the generated history. Reused
    /// across decode iterations: zeroed at start of each penalty
    /// application. Size = vocab_size, indexed by token id.
    penalty_counts: Vec<u32>,
}

impl SamplingContext {
    /// Create an empty SamplingContext; every buffer grows on first use.
    ///
    /// Deliberately NOT sized to the vocabulary up front. Most callers build
    /// one per sampled token (`tensor_util::sample_token_with_*`), and the
    /// default top-k path touches only O(k) of it — so eager vocab-sized
    /// buffers were ~4 MB of fresh allocation and page faults per token at a
    /// 152k vocabulary, for nothing. `vocab_size` is kept for the call sites.
    pub fn new(_vocab_size: usize) -> Self {
        Self {
            indexed_logits: Vec::new(),
            keep_mask: Vec::new(),
            probs: Vec::new(),
            indices: Vec::new(),
            raw_logits: Vec::new(),
            penalty_counts: Vec::new(),
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

fn penalties_active(frequency_penalty: f32, presence_penalty: f32) -> bool {
    frequency_penalty != 0.0 || presence_penalty != 0.0
}

/// Does sampling for these parameters read the reply's token history at all?
///
/// The history exists for ONE consumer, the repetition penalties, and they
/// return before reading it when both are zero — the default. So this is the
/// single answer to "must the tokens generated so far travel with a forward",
/// asked by every path that ships them: the local worker path, the
/// distributed send and its failover (`pipeline::distributed`). Shipping them
/// anyway cost 4 bytes a token on the wire plus ~7 as a JSON array into the
/// peer's worker, on EVERY step — 8 KB per forward by token 2000, more than
/// the hidden state it rides with.
pub(crate) fn sampler_reads_history(params: &SamplingParams) -> bool {
    penalties_active(params.frequency_penalty, params.presence_penalty)
}

/// Apply OpenAI-style frequency + presence penalties in-place.
///
/// Formula (per OpenAI API spec): for each vocab token j,
///   `logits[j] -= count_j * frequency_penalty + I[count_j > 0] * presence_penalty`
/// where count_j is the number of occurrences of token j in `generated_ids`
/// (the completion-so-far, NOT including prompt tokens). Both penalties
/// expected in the documented [-2.0, 2.0] range; we don't clamp here so
/// callers that pass out-of-range values get the documented behavior.
///
/// Penalty application MUST happen BEFORE temperature scaling so the
/// post-temperature distribution is meaningful (penalties scaled by
/// temperature otherwise become inverse-proportional to creativity, which
/// inverts the intent of frequency_penalty=1.0 + temperature=2.0).
///
/// `ctx.penalty_counts` is reused as a vocab-sized histogram; this avoids
/// allocating per-token in the decode hot path. Cost: O(vocab + history)
/// per call.
pub(crate) fn apply_repetition_penalties(
    logits: &mut [f32],
    generated_ids: &[u32],
    frequency_penalty: f32,
    presence_penalty: f32,
    ctx: &mut SamplingContext,
) {
    if generated_ids.is_empty()
        || !penalties_active(frequency_penalty, presence_penalty)
        || logits.is_empty()
    {
        return;
    }
    let vocab = logits.len();
    if ctx.penalty_counts.len() < vocab {
        ctx.penalty_counts.resize(vocab, 0);
    } else {
        for c in &mut ctx.penalty_counts[..vocab] {
            *c = 0;
        }
    }
    for &tok in generated_ids {
        let i = tok as usize;
        if i < vocab {
            ctx.penalty_counts[i] = ctx.penalty_counts[i].saturating_add(1);
        }
    }
    for (i, &c) in ctx.penalty_counts[..vocab].iter().enumerate() {
        if c == 0 {
            continue;
        }
        logits[i] -= (c as f32) * frequency_penalty + presence_penalty;
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
///
/// Repetition penalties (frequency_penalty / presence_penalty) are NOT
/// applied here — pass `generated_ids` via `sample_token_with_history`
/// in decode loops that need them. This signature stays history-free
/// for tests and one-shot validation paths.
pub fn sample_token(logits: &mut [f32], params: &SamplingParams) -> u32 {
    let mut ctx = SamplingContext::new(logits.len());
    sample_token_with_ctx(logits, params, &[], &mut ctx, None)
}

/// Sample a token index applying frequency/presence penalties from the
/// generated-token history. Use this in decode loops; pass the
/// completion-so-far as `generated_ids` (NOT including prompt tokens,
/// matching the OpenAI spec). Empty `generated_ids` is equivalent to
/// `sample_token`.
pub fn sample_token_with_history(
    logits: &mut [f32],
    params: &SamplingParams,
    generated_ids: &[u32],
    ctx: &mut SamplingContext,
) -> u32 {
    sample_token_with_ctx(logits, params, generated_ids, ctx, None)
}

/// Sample the token that will occupy absolute sequence `position`, drawing from
/// the request's SHARED noise stream instead of this process's random numbers
/// (`inference::coupled_noise`). The same filters as every other sampler —
/// penalties, temperature, top-k, top-p — then Gumbel-max over what survives:
/// an exact sample of the same distribution, but one a drafter holding the same
/// seed can reproduce. Temperature 0 is the argmax, as everywhere.
pub fn sample_token_coupled(
    logits: &mut [f32],
    params: &SamplingParams,
    generated_ids: &[u32],
    ctx: &mut SamplingContext,
    noise: &CoupledNoise,
    position: u64,
) -> u32 {
    sample_token_with_ctx(logits, params, generated_ids, ctx, Some((noise, position)))
}

/// How many of a row's likeliest tokens a sampled draw chooses among — the
/// request's `top_k`, where it is above 0 and below the vocabulary — or `None`
/// for a draw over the whole row. The ONE answer to which path
/// [`sample_token_with_ctx`] takes, asked by [`coupled_draw_reads_only_top_k`].
fn draws_among_top_k(params: &SamplingParams, vocab: usize) -> Option<usize> {
    let k = params.top_k as usize;
    (k > 0 && k < vocab).then_some(k)
}

/// Where a coupled draw ([`sample_token_coupled`]) reads only the row's `k`
/// likeliest tokens, `Some(k)`: the request samples (temperature above 0), names
/// a `top_k` below the vocabulary, and has no penalty that would read its
/// history first. A drafter on a card then has those `k` ranked there
/// (`tree_rank::top_k_on_the_card`) instead of reading every row back whole, and
/// draws with [`sample_coupled_from_top_k`] — the same token.
pub(crate) fn coupled_draw_reads_only_top_k(
    params: &SamplingParams,
    vocab: usize,
) -> Option<usize> {
    (params.temperature > 0.0 && params.frequency_penalty == 0.0 && params.presence_penalty == 0.0)
        .then(|| draws_among_top_k(params, vocab))
        .flatten()
}

/// One of a row's likeliest tokens, as a card ranked it: its log-probability
/// at the request's temperature and its RAW score, which the draw divides by
/// the temperature itself, as it does a whole row's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct RankedToken {
    pub token: u32,
    pub logp: f32,
    pub logit: f32,
}

/// The token [`sample_token_coupled`] draws from the whole row, drawn from its
/// `k` likeliest alone — where [`coupled_draw_reads_only_top_k`] answered
/// `Some(k)` and `likeliest` holds that row's `k` highest scores (fewer only
/// where fewer are finite), in any order. The same draw
/// ([`draw_among_candidates`]) over the same candidates, in the same token
/// order; only a tie AT the k-th place may be broken the other way, as the
/// host's own selection breaks it arbitrarily.
pub(crate) fn sample_coupled_from_top_k(
    likeliest: &[RankedToken],
    params: &SamplingParams,
    ctx: &mut SamplingContext,
    noise: &CoupledNoise,
    position: u64,
    vocab: usize,
) -> u32 {
    ctx.indexed_logits.clear();
    ctx.indexed_logits
        .extend(likeliest.iter().map(|c| (c.token as usize, c.logit)));
    ctx.indexed_logits.sort_unstable_by_key(|&(i, _)| i);
    draw_among_candidates(params, ctx, vocab, simple_random, Some((noise, position)))
}

/// `argmax_i (score_i + G(position, i))` over `(token, score)` pairs, where
/// `score` is the temperature-scaled logit of a token that survived the filters —
/// Gumbel-max, an exact sample of the softmax over those scores.
///
/// **Pruned exactly, never approximately.** Noise is computed only for tokens
/// that could still win: the best candidate scores at least
/// `max + GUMBEL_MIN`, and no noise exceeds `GUMBEL_MAX`, so a token scoring below
/// `max + GUMBEL_MIN - GUMBEL_MAX` cannot be the argmax. Evaluating noise only for
/// SOME other subset (a drafter's candidates, say) would bias the sample; this
/// bound cannot. Ties go to the lowest token id. `None` when nothing is finite.
fn gumbel_argmax(
    candidates: impl Iterator<Item = (usize, f32)> + Clone,
    noise: &CoupledNoise,
    position: u64,
) -> Option<u32> {
    let max = candidates
        .clone()
        .filter(|&(_, l)| l.is_finite())
        .map(|(_, l)| l)
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let floor = f64::from(max) + GUMBEL_MIN - GUMBEL_MAX;
    let mut best: Option<(u32, f64)> = None;
    for (i, l) in candidates {
        if !l.is_finite() || f64::from(l) < floor {
            continue;
        }
        let score = f64::from(l) + noise.gumbel(position, i as u32);
        if best.is_none_or(|(_, b)| score > b) {
            best = Some((i as u32, score));
        }
    }
    best.map(|(i, _)| i)
}

/// Are the rows a verify of `n_drafts` drafts reads (γ+1 of them) all finite?
/// A peer supplies them; see `sampled_accept_reject`'s non-finite guard.
pub(crate) fn verify_rows_are_finite(n_drafts: usize, rows: &[Vec<f32>]) -> bool {
    rows.iter()
        .take(n_drafts + 1)
        .all(|row| row.iter().all(|v| v.is_finite()))
}

/// The walk a verify's last segment runs when asked to
/// (`LayerForward::spec_walk_at_tail`): the drafts it keeps, then the token it
/// sampled where they parted (or the bonus after all of them) — what the
/// coordinator would have derived from these rows with `sampled_accept_reject`,
/// sent as token ids instead of a vocabulary per position.
///
/// Errs where the coordinator's `VerifyReply::accept` would: rows that do not
/// number drafts + 1, or that are not finite.
pub(crate) fn walk_verified_positions(
    drafts: &[u32],
    rows: &[Vec<f32>],
    params: &SamplingParams,
    generated: &[u32],
    coupling: Option<(&CoupledNoise, u64)>,
) -> Result<Vec<u32>, String> {
    if rows.len() != drafts.len() + 1 {
        return Err(format!(
            "spec walk: {} positions verified for {} drafts",
            rows.len(),
            drafts.len()
        ));
    }
    if !verify_rows_are_finite(drafts.len(), rows) {
        return Err("spec walk: non-finite logits".into());
    }
    let (mut kept, next, _) = sampled_accept_reject(drafts, rows, params, generated, coupling);
    kept.push(next);
    Ok(kept)
}

/// [`walk_verified_positions`] for a tree of guesses (#134,
/// `inference::draft_tree`): sample at the row before the first guess, keep
/// the guess following it that carries the sample, sample there, and so on
/// until no guess carries it. Answers the kept tokens, then that last sample.
///
/// Exact at any temperature, for the reason the chain's rule is: a row's
/// guesses carry distinct tokens, so "keep the guess equal to the model's own
/// draw" is [`sampled_accept_reject`]'s rule along whichever path the draws
/// take (SpecExec, arXiv 2406.02532 §3). A draw is keyed by the absolute
/// position it fills, `first + depth`, so a drafter holding the same seed
/// reproduces it on every branch.
///
/// `tree` holds every row of the forward; the rows from `guessed_from` are the
/// guesses, `drafts` their tokens; the rows before are tokens already kept.
/// Each row the walk reads is checked finite, as the chain's are.
pub(crate) fn walk_verified_tree(
    tree: &crate::inference::draft_tree::DraftTree,
    drafts: &[u32],
    rows: &[Vec<f32>],
    params: &SamplingParams,
    generated: &[u32],
    coupling: Option<(&CoupledNoise, u64)>,
) -> Result<Vec<u32>, String> {
    let guessed_from = tree_guesses_from(tree, drafts.len(), rows.len())?;
    let vocab = rows.first().map_or(0, Vec::len);
    let mut ctx = SamplingContext::new(vocab);
    let mut history = generated.to_vec();
    let mut walked = Vec::new();
    let mut at = guessed_from - 1;
    loop {
        if !rows[at].iter().all(|v| v.is_finite()) {
            return Err("spec walk: non-finite logits".into());
        }
        let mut row = rows[at].clone();
        let pick = match coupling {
            Some((noise, first)) => sample_token_coupled(
                &mut row,
                params,
                &history,
                &mut ctx,
                noise,
                first + tree.depth(at) as u64,
            ),
            None => sample_token_with_history(&mut row, params, &history, &mut ctx),
        };
        walked.push(pick);
        match tree
            .children(Some(at))
            .find(|&c| c >= guessed_from && drafts[c - guessed_from] == pick)
        {
            Some(next) => {
                history.push(pick);
                at = next;
            }
            None => return Ok(walked),
        }
    }
}

/// What the last segment answers for the rows of a tree that FLOWS (#134 (b),
/// `LayerForward::draft_tree_from`): every row's own draw — the token the
/// model samples after that row's path — for the requester to walk. `rows` are
/// this forward's, the tree's last; the `read_before` rows ahead of them were
/// answered by earlier forwards.
///
/// Exact at any temperature, for [`walk_verified_tree`]'s reason: the walk
/// keeps the child carrying the draw at the row it stands on, and emits that
/// draw whether or not a child carries it, so every emitted token is the
/// model's own sample along the path emitted. A row's draw is made once and
/// read only by a walk standing on that row. With shared noise it is keyed by
/// the absolute position it fills, `first + depth` (`first`: one past the
/// tree's start), so the drafter holding the seed reproduces it on every branch; without,
/// each row draws independently, which is exact too — a path reads one draw
/// per position. Penalties read each path's own tokens, which no row here
/// knows: the caller refuses such a request before it reaches this.
pub(crate) fn draw_each_row(
    tree: &crate::inference::draft_tree::DraftTree,
    read_before: usize,
    rows: &[Vec<f32>],
    params: &SamplingParams,
    coupling: Option<(&CoupledNoise, u64)>,
) -> Result<Vec<u32>, String> {
    let vocab = rows.first().map_or(0, Vec::len);
    let mut ctx = SamplingContext::new(vocab);
    draws_of_each_row(tree, read_before, rows, |logits, depth| {
        if !logits.iter().all(|v| v.is_finite()) {
            return Err("spec draws: non-finite logits".to_string());
        }
        let mut row = logits.clone();
        Ok(match coupling {
            Some((noise, first)) => {
                sample_token_coupled(&mut row, params, &[], &mut ctx, noise, first + depth)
            }
            None => sample_token_with_history(&mut row, params, &[], &mut ctx),
        })
    })
}

/// A tree that flows, drawn row by row: row `q` of this forward is the tree's
/// row `read_before + q`, drawn by `draw` at its depth in the tree — the
/// noise's key past the tree's start. The ONE statement of which row and key a
/// flowing row's draw gets, whether the row arrived whole ([`draw_each_row`])
/// or as its top k from a card (`tree_rank::draw_each_row_on_the_card`).
pub(crate) fn draws_of_each_row<R>(
    tree: &crate::inference::draft_tree::DraftTree,
    read_before: usize,
    rows: &[R],
    mut draw: impl FnMut(&R, u64) -> Result<u32, String>,
) -> Result<Vec<u32>, String> {
    if read_before + rows.len() != tree.len() {
        return Err(format!(
            "spec draws: {} rows read after {read_before} of a tree of {} rows",
            rows.len(),
            tree.len()
        ));
    }
    rows.iter()
        .enumerate()
        .map(|(q, row)| draw(row, tree.depth(read_before + q) as u64))
        .collect()
}

/// [`walk_verified_tree`] for a greedy request with no penalties, from each
/// row's argmax alone — what a card answers instead of every row's vocabulary.
pub(crate) fn argmax_walk_tree(
    tree: &crate::inference::draft_tree::DraftTree,
    drafts: &[u32],
    picks: &[u32],
) -> Result<Vec<u32>, String> {
    let guessed_from = tree_guesses_from(tree, drafts.len(), picks.len())?;
    let mut walked = Vec::new();
    let mut at = guessed_from - 1;
    loop {
        let pick = picks[at];
        walked.push(pick);
        match tree
            .children(Some(at))
            .find(|&c| c >= guessed_from && drafts[c - guessed_from] == pick)
        {
            Some(next) => at = next,
            None => return Ok(walked),
        }
    }
}

/// The first guessed row of a tree check of `rows` rows carrying `drafts`
/// guesses, refused where no honest coordinator sends it: rows that are not
/// the tree's, no row before the guesses, kept rows that are not a chain from
/// the cache, or a guess hanging off a kept row other than the last.
fn tree_guesses_from(
    tree: &crate::inference::draft_tree::DraftTree,
    drafts: usize,
    rows: usize,
) -> Result<usize, String> {
    if rows != tree.len() || drafts >= rows {
        return Err(format!(
            "spec walk: {rows} positions verified for a tree of {} rows and {drafts} guesses",
            tree.len()
        ));
    }
    let guessed_from = rows - drafts;
    for r in 0..guessed_from {
        if tree.parent(r) != r.checked_sub(1) {
            return Err("spec walk: the kept rows of a tree are not a chain".into());
        }
    }
    for r in guessed_from..rows {
        if tree.parent(r).is_none_or(|p| p + 1 < guessed_from) {
            return Err("spec walk: a guess follows a row that is already kept".into());
        }
    }
    Ok(guessed_from)
}

/// Accept-reject for a DRAFT THAT CARRIES NO DISTRIBUTION, honouring the
/// caller's sampling parameters.
///
/// An n-gram draft proposes a token with no probability behind it — `q = δ_x` —
/// and speculative sampling then reduces to: accept with probability
/// `min(1, p(x)/q(x)) = p(x)`, otherwise draw from the residual
/// `norm((p − q)₊)`, i.e. `p` with `x` removed and renormalised. "Draw `t ~ p`;
/// keep the draft iff `t == x`" has exactly those two branches, so sampling each
/// position through the real sampler and keeping a match IS that rule — at any
/// temperature, with no separate machinery.
/// `accepting_only_on_a_match_preserves_the_sampled_distribution` below pins
/// it, with a control that fails if the metric could
/// not detect a bias.
///
/// **Why this exists beside `greedy_accept_reject`.** That one takes the raw
/// argmax, which ignores temperature, top-k, top-p AND the repetition
/// penalties. It is correct for the paths that are greedy-only by construction
/// (a draft MODEL needs the draft's own probabilities to do this properly, which
/// is a different algorithm), but it meant the same request got a different
/// answer depending on whether it was served locally or across peers — the local
/// worker has always sampled properly. It also meant n-gram speculation could
/// only ever run at temperature 0, which is not what any client asks for: the
/// OpenAI surface defaults to 0.7 and the Anthropic one to 1.0.
///
/// `generated` is the history the penalties apply against, and it grows as
/// drafts are accepted, so each position sees what it would have seen had the
/// tokens been produced one at a time.
///
/// The non-finite guard is NOT optional and is the reason this cannot simply
/// call the sampler in a loop: a peer supplies these logits, and NaN comparisons
/// are non-deterministic, so a malicious segment could otherwise steer which
/// tokens get accepted. Same treatment as the greedy sibling — reject the whole
/// round.
pub(crate) fn sampled_accept_reject(
    drafts: &[u32],
    spec_logits: &[Vec<f32>],
    params: &crate::types::SamplingParams,
    generated: &[u32],
    coupling: Option<(&CoupledNoise, u64)>,
) -> (Vec<u32>, u32, bool) {
    // Too few rows is refused here, not only by the callers that check first:
    // indexing row `drafts.len()` would panic on a short reply, and a rule every
    // caller must remember to guard is one a new caller will not (gotcha #29 —
    // the contract is γ+1 rows).
    if spec_logits.len() < drafts.len() + 1 || !verify_rows_are_finite(drafts.len(), spec_logits) {
        return (Vec::new(), 0, false);
    }
    let vocab = spec_logits.first().map(|r| r.len()).unwrap_or(0);
    let mut ctx = SamplingContext::new(vocab);
    let mut history: Vec<u32> = generated.to_vec();
    let mut accepted: Vec<u32> = Vec::with_capacity(drafts.len());
    let mut bonus: u32 = 0;
    // Row i predicts the token at absolute position `first + i`; with shared
    // noise that position keys the draw, so a drafter holding the same seed
    // reproduces it (`coupled_noise`).
    let draw =
        |row: &mut [f32], history: &[u32], i: usize, ctx: &mut SamplingContext| match coupling {
            Some((noise, first)) => {
                sample_token_coupled(row, params, history, ctx, noise, first + i as u64)
            }
            None => sample_token_with_history(row, params, history, ctx),
        };
    for (i, &q) in drafts.iter().enumerate() {
        let mut row = spec_logits[i].clone();
        let pick = draw(&mut row, &history, i, &mut ctx);
        if pick == q {
            accepted.push(q);
            history.push(q);
        } else {
            bonus = pick;
            break;
        }
    }
    let all_accepted = accepted.len() == drafts.len();
    if all_accepted {
        let mut row = spec_logits[drafts.len()].clone();
        bonus = draw(&mut row, &history, drafts.len(), &mut ctx);
    }
    (accepted, bonus, all_accepted)
}

/// Sample a token index from logits using pre-allocated scratch buffers.
///
/// Same behavior as `sample_token` but reuses buffers from `SamplingContext`,
/// eliminating ~700KB of allocations per call for typical vocab sizes (32K).
fn sample_token_with_ctx(
    logits: &mut [f32],
    params: &SamplingParams,
    generated_ids: &[u32],
    ctx: &mut SamplingContext,
    coupled: Option<(&CoupledNoise, u64)>,
) -> u32 {
    // Repetition penalties (frequency + presence) MUST run before
    // temperature scaling per OpenAI spec — applying after temperature
    // would scale the penalty inversely with creativity and silently
    // invert the user's intent. No-op when penalties are 0 or history
    // is empty.
    apply_repetition_penalties(
        logits,
        generated_ids,
        params.frequency_penalty,
        params.presence_penalty,
        ctx,
    );

    // Greedy decoding when temperature is 0. Nothing else to compute: logprobs
    // are taken from `ctx.raw_logits` by the caller that wants them, so the
    // vocab-wide `exp` pass that used to fill `ctx.probs` here was read by
    // nothing — about two-thirds of a greedy token's sampling time at 152k.
    if params.temperature <= 0.0 {
        let token = argmax(logits);
        tracing::trace!(
            token,
            vocab_size = logits.len(),
            mode = "greedy",
            "DIAG: sample_token complete"
        );
        return token;
    }

    if let Some(k) = draws_among_top_k(params, logits.len()) {
        sample_among_top_k(logits, params, k, ctx, simple_random, coupled)
    } else {
        sample_full_vocab(logits, params, ctx, simple_random, coupled)
    }
}

/// The `k` highest logits as `(index, logit)`, returned sorted by INDEX.
///
/// One pass with a rising floor: a logit is kept only if it beats the k-th
/// best seen so far, and the buffer is cut back to `k` whenever it reaches
/// twice that. On a real distribution the floor settles within the first few
/// thousand entries and the rest of the vocabulary costs one comparison each —
/// where the version this replaced copied the whole vocabulary into
/// `(usize, f32)` pairs (2.4 MB at 152k) to partition it.
///
/// `-inf` and NaN are never candidates (`>` is false for both): a `-inf` logit
/// has probability exactly zero, and a NaN one has no probability at all.
/// Ties at the k-th place are resolved arbitrarily, as the partition did.
fn select_top_k_candidates(logits: &[f32], k: usize, out: &mut Vec<(usize, f32)>) {
    let desc = |a: &(usize, f32), b: &(usize, f32)| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    };
    out.clear();
    let cap = k.saturating_mul(2).max(k + 16);
    out.reserve(cap);
    let mut floor = f32::NEG_INFINITY;
    for (i, &l) in logits.iter().enumerate() {
        if l > floor {
            out.push((i, l));
            if out.len() >= cap {
                out.select_nth_unstable_by(k - 1, desc);
                out.truncate(k);
                floor = out[k - 1].1;
            }
        }
    }
    if out.len() > k {
        out.select_nth_unstable_by(k - 1, desc);
        out.truncate(k);
    }
    out.sort_unstable_by_key(|&(i, _)| i);
}

/// Sample with top-k active, touching only the `k` candidates after selection.
///
/// llama.cpp's chain does the same (`llama_sampler_top_k` shrinks
/// `cur_p->size`, and top-p and the draw then run on what is left). The
/// vocabulary-wide path below computed softmax twice more over all 152k
/// entries — once for top-p, once for the draw — when every entry outside the
/// top `k` is `-inf` and contributes exactly nothing.
///
/// **It picks the same token as [`sample_full_vocab`] for the same uniform
/// draw**, not merely from the same distribution, because it is the same
/// arithmetic on the same values:
/// - a masked entry's `exp(-inf - max)` is exactly `+0.0`, and adding `+0.0`
///   leaves an f32 sum unchanged, so summing only the candidates, IN INDEX
///   ORDER, gives the full-vocabulary sum bit for bit — which is why the
///   candidates are kept sorted by index rather than by value;
/// - the maximum over candidates IS the maximum over the masked vocabulary;
/// - the final walk visits candidates in index order, as the full walk did,
///   and only ever stops on a non-zero entry.
///
/// Two degenerate cases differ, both previously wrong: a uniform draw of
/// exactly `0.0` (probability 2^-24 per token) made the full walk return
/// index 0 even when index 0 was masked, and its rounding fallback returned
/// the LAST VOCABULARY ENTRY; here both land on a candidate. And a NaN logit,
/// which sent the full path to an argmax, is simply never a candidate.
/// `a_top_k_sample_is_the_same_token_the_full_vocabulary_path_picks` pins the
/// equivalence against `sample_full_vocab` itself.
fn sample_among_top_k(
    logits: &[f32],
    params: &SamplingParams,
    k: usize,
    ctx: &mut SamplingContext,
    uniform: impl FnOnce() -> f32,
    coupled: Option<(&CoupledNoise, u64)>,
) -> u32 {
    select_top_k_candidates(logits, k, &mut ctx.indexed_logits);
    draw_among_candidates(params, ctx, logits.len(), uniform, coupled)
}

/// [`sample_among_top_k`] after the selection: temperature, top-p and the draw
/// over `ctx.indexed_logits` — the row's `top_k` likeliest as (token, raw
/// logit), in token order. The ONE draw over a row's top k, whether the host
/// chose them from the whole row or the card did
/// ([`sample_coupled_from_top_k`]). `vocab` is only reported.
fn draw_among_candidates(
    params: &SamplingParams,
    ctx: &mut SamplingContext,
    vocab: usize,
    uniform: impl FnOnce() -> f32,
    coupled: Option<(&CoupledNoise, u64)>,
) -> u32 {
    let cands = &mut ctx.indexed_logits;

    // Temperature on the candidates only — the same division
    // `apply_temperature` performs, and it cannot reorder them.
    if params.temperature != 1.0 {
        for c in cands.iter_mut() {
            c.1 /= params.temperature;
        }
    }

    // Top-p over the candidates, exactly as `apply_top_p_with_ctx` computes it
    // over the masked vocabulary.
    if params.top_p < 1.0 && !cands.is_empty() {
        let max_logit = cands.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        ctx.probs.clear();
        ctx.probs
            .extend(cands.iter().map(|c| (c.1 - max_logit).exp()));
        let sum: f32 = ctx.probs.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            let inv_sum = 1.0 / sum;
            for prob in ctx.probs.iter_mut() {
                *prob *= inv_sum;
            }
            let probs: &[f32] = &ctx.probs;
            ctx.indices.clear();
            ctx.indices.extend(0..cands.len());
            ctx.indices.sort_unstable_by(|&a, &b| {
                probs[b]
                    .partial_cmp(&probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if ctx.keep_mask.len() < cands.len() {
                ctx.keep_mask.resize(cands.len(), false);
            }
            let keep = &mut ctx.keep_mask[..cands.len()];
            keep.fill(false);
            let mut cumulative = 0.0;
            let mut reached = false;
            for &pos in &ctx.indices {
                cumulative += probs[pos];
                keep[pos] = true;
                if cumulative >= params.top_p {
                    reached = true;
                    break;
                }
            }
            if reached {
                let mut pos = 0;
                cands.retain(|_| {
                    let kept = keep[pos];
                    pos += 1;
                    kept
                });
            }
        }
    }

    // The shared-noise draw: Gumbel-max over the same survivors.
    if let Some((noise, position)) = coupled {
        if let Some(token) = gumbel_argmax(cands.iter().copied(), noise, position) {
            return token;
        }
    }

    // The draw: unnormalised softmax over the survivors, walked in index order.
    let max_logit = cands.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
    ctx.probs.clear();
    ctx.probs
        .extend(cands.iter().map(|c| (c.1 - max_logit).exp()));
    let sum: f32 = ctx.probs.iter().sum();
    if !sum.is_finite() || sum == 0.0 {
        // Same fallback as the full path: the first highest logit, or index 0
        // when nothing is finite at all.
        let mut best_i = 0u32;
        let mut best_v = f32::NEG_INFINITY;
        for &(i, l) in cands.iter() {
            if l > best_v {
                best_v = l;
                best_i = i as u32;
            }
        }
        tracing::warn!(
            vocab_size = vocab,
            sum,
            "DIAG: sampling softmax collapsed — falling back to argmax"
        );
        return best_i;
    }
    let r: f32 = uniform() * sum;
    let mut cumulative = 0.0;
    for (&(i, _), &p) in cands.iter().zip(ctx.probs.iter()) {
        cumulative += p;
        if cumulative >= r {
            tracing::trace!(
                token = i as u32,
                vocab_size = vocab,
                temperature = params.temperature,
                top_k = params.top_k,
                top_p = params.top_p,
                mode = "stochastic",
                "DIAG: sample_token complete"
            );
            return i as u32;
        }
    }
    tracing::warn!(
        vocab_size = vocab,
        sum,
        "DIAG: sampling fallback — cumulative probability didn't reach threshold"
    );
    cands.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

/// Sample over the whole vocabulary: temperature, top-k mask, top-p mask, then
/// a softmax draw. The path for requests with top-k off, and the reference
/// [`sample_among_top_k`] is tested against.
fn sample_full_vocab(
    logits: &mut [f32],
    params: &SamplingParams,
    ctx: &mut SamplingContext,
    uniform: impl FnOnce() -> f32,
    coupled: Option<(&CoupledNoise, u64)>,
) -> u32 {
    apply_temperature(logits, params.temperature);
    apply_top_k_with_ctx(logits, params.top_k, ctx);
    apply_top_p_with_ctx(logits, params.top_p, ctx);

    // The shared-noise draw over what the masks left finite.
    if let Some((noise, position)) = coupled {
        if let Some(token) = gumbel_argmax(logits.iter().copied().enumerate(), noise, position) {
            return token;
        }
    }

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
        tracing::warn!(
            vocab_size = logits.len(),
            sum,
            "DIAG: sampling softmax collapsed — falling back to argmax"
        );
        return argmax(logits);
    }

    // Weighted random selection
    let r: f32 = uniform() * sum;
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
    sample_token_with_logprobs_history(logits, params, &[], ctx)
}

/// Same as `sample_token_with_logprobs` but applies repetition penalties
/// from `generated_ids`. Use this in decode loops with non-zero
/// frequency/presence penalty AND logprobs enabled.
///
/// Note: `raw_logits` is saved BEFORE penalty application so the returned
/// logprobs reflect the model's raw distribution (matches OpenAI spec —
/// logprobs are not penalty-adjusted, only the sampling distribution is).
pub fn sample_token_with_logprobs_history(
    logits: &mut [f32],
    params: &SamplingParams,
    generated_ids: &[u32],
    ctx: &mut SamplingContext,
) -> (u32, Option<SampledTokenLogProb>) {
    let need_logprobs = params.logprobs && params.top_logprobs > 0;

    // Save raw logits BEFORE sampling mutates them (temperature/top-k/top-p
    // or penalties). OpenAI spec: logprobs are computed from the
    // pre-sampling distribution — NOT the penalty-adjusted one.
    if need_logprobs {
        ctx.raw_logits.clear();
        ctx.raw_logits.extend_from_slice(logits);
    }

    // Clear probs buffer before sampling to prevent stale data from previous tokens
    // (especially important for greedy/temperature=0 which skips softmax)
    ctx.probs.clear();

    let token_id = sample_token_with_ctx(logits, params, generated_ids, ctx, None);

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

/// The one greedy pick over a row of logits: the FIRST of equal maxima (the
/// lowest id), as llama.cpp's greedy sampler (`llama_sampler_greedy_apply`,
/// strict `>`), `gumbel_argmax` and the card's ranking (`tree_rank`) break a
/// tie. A NaN never wins; a row with nothing comparable answers 0. The
/// coordinator's greedy check (`pipeline::speculative`) and the worker's SWIFT
/// check (`model_worker::sample_argmax`) call it too: with copies that broke
/// ties apart, an exact tie could read one token without speculation and
/// another with it (sweep 2026-10-24). A check whose rows stay on a card picks
/// through `tree_rank::greedy_picks`, the same rule; candle's card `argmax`
/// breaks a tie by thread order and never decides a reply (guard
/// `no_reply_is_picked_by_candles_argmax`).
pub(crate) fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
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

    /// The top-k path must pick the SAME token as the full-vocabulary path for
    /// the same uniform draw — same arithmetic, not just the same distribution
    /// (see `sample_among_top_k`). Varies vocabulary size, spread, k, p,
    /// temperature and the draw, and reuses both contexts across cases the way
    /// a decode loop does.
    #[test]
    fn a_top_k_sample_is_the_same_token_the_full_vocabulary_path_picks() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut ctx_full = SamplingContext::new(0);
        let mut ctx_fast = SamplingContext::new(0);
        let mut differing_tokens = std::collections::HashSet::new();
        for case in 0..600usize {
            let vocab = 300 + (next() % 6000) as usize;
            let spread = [1.0f32, 6.0, 20.0][case % 3];
            let logits: Vec<f32> = (0..vocab)
                .map(|_| ((next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32 * spread)
                .collect();
            let params = SamplingParams {
                temperature: [0.7f32, 1.0, 1.3, 0.2][(case / 3) % 4],
                top_k: [1u32, 2, 40, 64, 250][(case / 12) % 5],
                top_p: [0.9f32, 1.0, 0.5, 0.99, 0.05][(case / 60) % 5],
                ..Default::default()
            };
            for u in [1e-6f32, 0.1, 0.37, 0.5, 0.83, 0.999_999] {
                let mut full = logits.clone();
                let want = sample_full_vocab(&mut full, &params, &mut ctx_full, || u, None);
                let got = sample_among_top_k(
                    &logits,
                    &params,
                    params.top_k as usize,
                    &mut ctx_fast,
                    || u,
                    None,
                );
                assert_eq!(
                    got, want,
                    "case {case}: vocab {vocab}, k {}, p {}, t {}, u {u}",
                    params.top_k, params.top_p, params.temperature
                );
                differing_tokens.insert(got);
            }
        }
        // The draws must actually have exercised the walk, not all landed on
        // the argmax.
        assert!(differing_tokens.len() > 500, "{}", differing_tokens.len());
    }

    /// The selector returns exactly the k largest, in index order, whatever
    /// order the vocabulary presents them in — including more candidates than
    /// its compaction buffer and a vocabulary with fewer finite entries than k.
    #[test]
    fn the_top_k_selector_keeps_the_k_largest_in_index_order() {
        let mut out = Vec::new();
        let rising: Vec<f32> = (0..10_000).map(|i| i as f32).collect();
        select_top_k_candidates(&rising, 40, &mut out);
        assert_eq!(
            out.iter().map(|c| c.0).collect::<Vec<_>>(),
            (9_960..10_000).collect::<Vec<_>>()
        );
        let falling: Vec<f32> = rising.iter().rev().copied().collect();
        select_top_k_candidates(&falling, 3, &mut out);
        assert_eq!(out, vec![(0, 9999.0), (1, 9998.0), (2, 9997.0)]);

        let mut sparse = vec![f32::NEG_INFINITY; 1000];
        sparse[7] = 1.0;
        sparse[900] = 2.0;
        sparse[500] = f32::NAN;
        select_top_k_candidates(&sparse, 40, &mut out);
        assert_eq!(out, vec![(7, 1.0), (900, 2.0)]);
    }

    /// Every logit masked: the same fallback as the full path — index 0 —
    /// rather than a panic on an empty candidate list.
    #[test]
    fn a_top_k_sample_over_nothing_finite_falls_back_like_the_full_path() {
        let logits = vec![f32::NEG_INFINITY; 100];
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(0);
        assert_eq!(
            sample_among_top_k(&logits, &params, 40, &mut ctx, || 0.5, None),
            0
        );
        let mut full = logits.clone();
        assert_eq!(
            sample_full_vocab(&mut full, &params, &mut ctx, || 0.5, None),
            0
        );
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
        let token = sample_token_with_ctx(&mut logits, &params, &[], &mut ctx, None);
        assert_eq!(token, 1);
    }

    #[test]
    fn argmax_returns_correct_index() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[5.0, 1.0, 1.0]), 0);
        // An exact tie goes to the lowest id, as llama.cpp's greedy sampler
        // and the card's ranking break it.
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 2.0, 2.0]), 1);
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

    #[test]
    fn repetition_penalties_no_op_when_zero() {
        let mut logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let original = logits.clone();
        let mut ctx = SamplingContext::new(logits.len());
        apply_repetition_penalties(&mut logits, &[1, 1, 2], 0.0, 0.0, &mut ctx);
        assert_eq!(logits, original);
    }

    #[test]
    fn repetition_penalties_no_op_when_history_empty() {
        let mut logits = vec![1.0, 5.0, 3.0];
        let original = logits.clone();
        let mut ctx = SamplingContext::new(logits.len());
        apply_repetition_penalties(&mut logits, &[], 1.5, 0.5, &mut ctx);
        assert_eq!(logits, original);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        // OpenAI formula: logits[j] -= count[j] * frequency_penalty
        let mut logits = vec![10.0, 10.0, 10.0];
        let mut ctx = SamplingContext::new(logits.len());
        // Token 1 appears 3 times; token 2 appears once; token 0 unused.
        apply_repetition_penalties(&mut logits, &[1, 1, 1, 2], 1.0, 0.0, &mut ctx);
        assert!((logits[0] - 10.0).abs() < 1e-6);
        assert!((logits[1] - 7.0).abs() < 1e-6); // 10 - 3*1
        assert!((logits[2] - 9.0).abs() < 1e-6); // 10 - 1*1
    }

    #[test]
    fn presence_penalty_is_boolean_not_count() {
        // OpenAI formula: logits[j] -= 1[count[j] > 0] * presence_penalty.
        // Token 1 appears 5 times — same penalty as appearing once.
        let mut logits = vec![10.0, 10.0, 10.0];
        let mut ctx = SamplingContext::new(logits.len());
        apply_repetition_penalties(&mut logits, &[1, 1, 1, 1, 1, 2], 0.0, 0.7, &mut ctx);
        assert!((logits[0] - 10.0).abs() < 1e-6);
        assert!((logits[1] - 9.3).abs() < 1e-6);
        assert!((logits[2] - 9.3).abs() < 1e-6);
    }

    #[test]
    fn frequency_and_presence_combine() {
        let mut logits = vec![10.0, 10.0];
        let mut ctx = SamplingContext::new(logits.len());
        apply_repetition_penalties(&mut logits, &[1, 1, 1], 0.5, 0.3, &mut ctx);
        // Token 1: count=3, present → 10 - (3 * 0.5 + 0.3) = 8.2
        assert!((logits[0] - 10.0).abs() < 1e-6);
        assert!((logits[1] - 8.2).abs() < 1e-6);
    }

    #[test]
    fn negative_penalty_encourages_repetition() {
        // Per OpenAI docs, negative values "encourage repetition".
        let mut logits = vec![10.0, 10.0];
        let mut ctx = SamplingContext::new(logits.len());
        apply_repetition_penalties(&mut logits, &[1, 1], -1.0, -0.5, &mut ctx);
        assert!((logits[0] - 10.0).abs() < 1e-6);
        // Token 1: 10 - (2 * -1.0 + -0.5) = 12.5
        assert!((logits[1] - 12.5).abs() < 1e-6);
    }

    #[test]
    fn out_of_range_token_id_ignored() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        let mut ctx = SamplingContext::new(logits.len());
        // Token id 99 is past vocab — must NOT panic, must NOT corrupt logits.
        apply_repetition_penalties(&mut logits, &[99, 100], 1.0, 1.0, &mut ctx);
        assert_eq!(logits, original);
    }

    #[test]
    fn sample_token_with_history_applies_freq_penalty() {
        // With strong freq penalty + greedy temperature, the previously
        // sampled token should be suppressed below the next-best.
        let mut logits = vec![10.0, 9.0, 8.0]; // greedy would pick 0
        let params = SamplingParams {
            temperature: 0.0,
            frequency_penalty: 2.0,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(logits.len());
        // Token 0 appears once; penalty drops it to 10 - 1*2 = 8.0
        // but logits[1] = 9.0 > 8.0, so token 1 wins.
        let token = sample_token_with_history(&mut logits, &params, &[0], &mut ctx);
        assert_eq!(token, 1);
    }
}

#[cfg(test)]
mod speculative_acceptance_tests {
    use super::*;
    use crate::types::SamplingParams;

    /// Draw `n` tokens the plain way, and `n` more through the accept/reject
    /// procedure the local speculator uses, then compare the two distributions.
    ///
    /// **This is the whole correctness argument for speculating at a non-zero
    /// temperature, so it is worth stating.** Speculative sampling with a
    /// deterministic draft `x` (an n-gram guess carries no distribution, so
    /// `q = δ_x`) says: accept `x` with probability `min(1, p(x)/q(x)) = p(x)`,
    /// and on rejection draw from the residual `norm((p − q)₊)`, which is `p`
    /// with `x` removed and renormalised.
    ///
    /// "Draw `t ~ p`; accept iff `t == x`" has exactly those two branches —
    /// acceptance happens with probability `p(x)`, and conditioned on rejection
    /// `t` is distributed as `p` restricted to `≠ x`. So the speculator does not
    /// need a bespoke rejection rule, and it is NOT limited to greedy decoding:
    /// sampling every position with the real sampler and keeping the draft only
    /// when it matches already IS the rule.
    ///
    /// The test exists because that equivalence is easy to assert and easy to
    /// get wrong, and a mistake would not fail loudly — it would quietly bias
    /// every sampled reply toward whatever the n-gram happened to guess.
    #[test]
    fn accepting_only_on_a_match_preserves_the_sampled_distribution() {
        let vocab = 12usize;
        // A distribution with real spread, so a bias would show. Deliberately
        // not uniform: under a uniform target every token is equally likely to
        // match and a bias toward the draft would be invisible.
        let base: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        // The token the speculator would propose. Picked as a HIGH-probability
        // one: that is the case where acceptance is frequent, so any error in
        // the accept branch has the most room to distort the result.
        let draft = 3u32;

        let draws = 60_000;
        let mut plain = vec![0u32; vocab];
        let mut spec = vec![0u32; vocab];
        let mut ctx = SamplingContext::new(vocab);

        for _ in 0..draws {
            let mut l = base.clone();
            plain[sample_token_with_history(&mut l, &params, &[], &mut ctx) as usize] += 1;

            // The speculator's step: sample this position, then keep the draft
            // only if the sampler independently produced it.
            let mut l2 = base.clone();
            let t = sample_token_with_history(&mut l2, &params, &[], &mut ctx);
            let emitted = if t == draft { draft } else { t };
            spec[emitted as usize] += 1;
        }

        // Total variation distance between the two empirical distributions.
        // With 60k draws per arm, sampling noise on a 12-token vocabulary sits
        // around 0.01; 0.03 leaves room for that while still failing on a real
        // bias (accepting unconditionally puts ~0.3 of the mass on `draft`).
        let tv: f32 = (0..vocab)
            .map(|i| {
                let a = plain[i] as f32 / draws as f32;
                let b = spec[i] as f32 / draws as f32;
                (a - b).abs()
            })
            .sum::<f32>()
            / 2.0;
        assert!(
            tv < 0.03,
            "speculative acceptance shifted the distribution (total variation {tv:.4})\n\
             plain: {plain:?}\n spec:  {spec:?}"
        );
    }

    /// The control for the test above: if the draft were accepted without being
    /// checked against the sampler, the distribution WOULD move, and by much
    /// more than the tolerance. Without this, a test that passes proves only
    /// that the metric is insensitive.
    #[test]
    fn accepting_a_draft_unconditionally_would_fail_that_test() {
        let vocab = 12usize;
        let base: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.37).sin() * 2.0).collect();
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let draft = 3u32;
        let draws = 20_000;
        let mut plain = vec![0u32; vocab];
        let mut bad = vec![0u32; vocab];
        let mut ctx = SamplingContext::new(vocab);

        for _ in 0..draws {
            let mut l = base.clone();
            plain[sample_token_with_history(&mut l, &params, &[], &mut ctx) as usize] += 1;
            // The bug being ruled out: trusting the draft.
            bad[draft as usize] += 1;
        }
        let tv: f32 = (0..vocab)
            .map(|i| {
                let a = plain[i] as f32 / draws as f32;
                let b = bad[i] as f32 / draws as f32;
                (a - b).abs()
            })
            .sum::<f32>()
            / 2.0;
        assert!(
            tv > 0.3,
            "the metric cannot detect an obvious bias (total variation {tv:.4})"
        );
    }

    /// The walk a verify's last segment answers with is exactly what the
    /// coordinator would have derived from the same rows, and it refuses what
    /// the coordinator refuses — otherwise which side walked would change the
    /// reply, or a tail would answer malformed rows with a token.
    #[test]
    fn the_tails_walk_is_the_coordinators_accept_sent_as_ids() {
        let greedy = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let rows = vec![
            vec![0.1, 5.0, 0.2],
            vec![7.0, 0.5, 0.3],
            vec![0.0, 0.1, 9.0],
        ];
        for drafts in [vec![1u32, 0], vec![1, 2], vec![0, 0]] {
            let (mut want, next, _) = sampled_accept_reject(&drafts, &rows, &greedy, &[], None);
            want.push(next);
            assert_eq!(
                walk_verified_positions(&drafts, &rows, &greedy, &[], None).unwrap(),
                want,
                "drafts {drafts:?}"
            );
        }
        assert_eq!(
            walk_verified_positions(&[1, 0], &rows, &greedy, &[], None).unwrap(),
            vec![1, 0, 2],
            "every draft kept, then the bonus"
        );
        assert!(walk_verified_positions(&[1], &rows, &greedy, &[], None).is_err());
        let mut poisoned = rows;
        poisoned[2][1] = f32::INFINITY;
        assert!(walk_verified_positions(&[1, 0], &poisoned, &greedy, &[], None).is_err());
    }
}

#[cfg(test)]
mod coupled_sampling_tests {
    use super::*;
    use crate::inference::coupled_noise::CoupledNoise;

    fn tv(a: &[u32], b: &[u32], n: u32) -> f64 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).abs() / f64::from(n))
            .sum::<f64>()
            / 2.0
    }

    fn logits(vocab: usize) -> Vec<f32> {
        (0..vocab).map(|i| (i as f32 * 0.37).sin() * 2.0).collect()
    }

    /// **The exactness the whole scheme rests on.** A shared-noise sample, taken
    /// at fresh positions, must follow the SAME distribution the ordinary sampler
    /// draws from — on both the top-k path and the full-vocabulary path, with
    /// top-p in play. A bias here would quietly change every sampled reply of a
    /// split request. The control below proves the metric can see one.
    #[test]
    fn a_coupled_sample_follows_the_ordinary_samplers_distribution() {
        let vocab = 24usize;
        let base = logits(vocab);
        let noise = CoupledNoise::new(0xC0FFEE);
        for params in [
            SamplingParams {
                temperature: 0.8,
                top_k: 10,
                top_p: 0.9,
                ..Default::default()
            },
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
                ..Default::default()
            },
        ] {
            let n = 60_000u32;
            let (mut plain, mut coupled) = (vec![0u32; vocab], vec![0u32; vocab]);
            let mut ctx = SamplingContext::new(vocab);
            for pos in 0..n {
                let mut a = base.clone();
                plain[sample_token_with_history(&mut a, &params, &[], &mut ctx) as usize] += 1;
                let mut b = base.clone();
                let t =
                    sample_token_coupled(&mut b, &params, &[], &mut ctx, &noise, u64::from(pos));
                coupled[t as usize] += 1;
            }
            let d = tv(&plain, &coupled, n);
            assert!(d < 0.02, "coupled sampling moved the distribution: tv {d:.4} for {params:?}\n plain {plain:?}\n coupled {coupled:?}");
        }
    }

    /// The control: dropping the noise (always taking the argmax) is the kind of
    /// mistake the test above must catch, and it moves the distribution by far
    /// more than the tolerance.
    #[test]
    fn a_coupled_draw_that_ignored_its_noise_would_fail_that_test() {
        let vocab = 24usize;
        let base = logits(vocab);
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let n = 20_000u32;
        let (mut plain, mut greedy) = (vec![0u32; vocab], vec![0u32; vocab]);
        let mut ctx = SamplingContext::new(vocab);
        for _ in 0..n {
            let mut a = base.clone();
            plain[sample_token_with_history(&mut a, &params, &[], &mut ctx) as usize] += 1;
            greedy[argmax(&base) as usize] += 1;
        }
        assert!(tv(&plain, &greedy, n) > 0.3);
    }

    /// Why it exists: a drafter whose distribution is CLOSE to the target's draws
    /// the target's token far more often with the same noise than a fixed guess is
    /// accepted by an independent sample (p(guess)).
    #[test]
    fn shared_noise_makes_a_close_distribution_draw_the_same_token() {
        let vocab = 64usize;
        let target: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.11).cos() * 1.5).collect();
        let draft: Vec<f32> = target
            .iter()
            .enumerate()
            .map(|(i, l)| l + ((i as f32 * 1.7).sin() * 0.15))
            .collect();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let noise = CoupledNoise::new(7);
        let mut ctx = SamplingContext::new(vocab);
        let n = 20_000u32;
        let (mut same_coupled, mut same_independent) = (0u32, 0u32);
        let guess = argmax(&draft);
        for pos in 0..n {
            let (mut t, mut d) = (target.clone(), draft.clone());
            let tt = sample_token_coupled(&mut t, &params, &[], &mut ctx, &noise, u64::from(pos));
            let dd = sample_token_coupled(&mut d, &params, &[], &mut ctx, &noise, u64::from(pos));
            same_coupled += u32::from(tt == dd);
            let mut t2 = target.clone();
            same_independent +=
                u32::from(sample_token_with_history(&mut t2, &params, &[], &mut ctx) == guess);
        }
        let (c, i) = (
            f64::from(same_coupled) / f64::from(n),
            f64::from(same_independent) / f64::from(n),
        );
        assert!(
            c > 0.85 && c > i + 0.5,
            "coupled agreement {c:.3} vs a fixed guess accepted {i:.3}"
        );
    }

    /// Pruning is exact: the pick equals an unpruned brute-force Gumbel argmax,
    /// including on logits spread far wider than the pruning margin.
    #[test]
    fn pruning_never_changes_the_coupled_pick() {
        let noise = CoupledNoise::new(99);
        for spread in [1.0f32, 10.0, 80.0] {
            let l: Vec<f32> = (0..500)
                .map(|i| (i as f32 * 0.713).sin() * spread)
                .collect();
            for pos in 0..200u64 {
                let brute = l
                    .iter()
                    .enumerate()
                    .map(|(i, &x)| (i as u32, f64::from(x) + noise.gumbel(pos, i as u32)))
                    .fold(
                        (0u32, f64::NEG_INFINITY),
                        |b, c| if c.1 > b.1 { c } else { b },
                    )
                    .0;
                assert_eq!(
                    gumbel_argmax(l.iter().copied().enumerate(), &noise, pos),
                    Some(brute),
                    "spread {spread} pos {pos}"
                );
            }
        }
    }

    /// A coupled draw from a row's top k alone (`sample_coupled_from_top_k`:
    /// what a drafter on a card draws from the k the card ranked) is the draw
    /// `sample_token_coupled` makes from the whole row, at the temperatures,
    /// top-k and top-p a request carries — whatever order the k arrive in.
    #[test]
    fn a_coupled_draw_from_the_top_k_is_the_draw_from_the_whole_row() {
        let vocab = 3000usize;
        let noise = CoupledNoise::new(0x5EED);
        let mut ctx = SamplingContext::new(vocab);
        let mut not_the_argmax = 0u32;
        for spread in [2.0f32, 8.0, 30.0] {
            let row: Vec<f32> = (0..vocab)
                .map(|i| ((i * 7919) % 3001) as f32 / 3001.0 * spread + (i as f32 * 0.0917).cos())
                .collect();
            let mut order: Vec<usize> = (0..vocab).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
            for (temperature, top_k, top_p) in [
                (0.7f32, 40u32, 1.0f32),
                (0.7, 40, 0.9),
                (1.0, 8, 1.0),
                (1.3, 64, 0.5),
                (0.3, 1, 1.0),
            ] {
                let params = SamplingParams {
                    temperature,
                    top_k,
                    top_p,
                    ..Default::default()
                };
                let k = coupled_draw_reads_only_top_k(&params, vocab)
                    .expect("a sampled request with a top_k draws among its top k");
                assert_eq!(k, top_k as usize);
                // Likeliest FIRST, as a card hands them over — not the token
                // order the draw walks.
                let likeliest: Vec<RankedToken> = order[..k]
                    .iter()
                    .map(|&i| RankedToken {
                        token: i as u32,
                        logp: 0.0,
                        logit: row[i],
                    })
                    .collect();
                for pos in 0..300u64 {
                    let whole =
                        sample_token_coupled(&mut row.clone(), &params, &[], &mut ctx, &noise, pos);
                    let top = sample_coupled_from_top_k(
                        &likeliest, &params, &mut ctx, &noise, pos, vocab,
                    );
                    assert_eq!(
                        top, whole,
                        "spread {spread}, T {temperature}, top_k {top_k}, top_p {top_p}, pos {pos}"
                    );
                    not_the_argmax += u32::from(whole as usize != order[0]);
                }
            }
        }
        // The comparison saw real sampling, not the argmax every time.
        assert!(
            not_the_argmax > 1000,
            "{not_the_argmax} draws off the argmax"
        );
    }

    /// Where the draw reads more than a row's top k — the whole row, or a
    /// history for its penalties — or is no draw at all, the card's ranking is
    /// not asked for it.
    #[test]
    fn a_draw_that_reads_more_than_the_top_k_is_not_handed_only_the_top_k() {
        let sampled = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            ..Default::default()
        };
        assert_eq!(coupled_draw_reads_only_top_k(&sampled, 151_936), Some(40));
        for (what, params, vocab) in [
            (
                "greedy",
                SamplingParams {
                    temperature: 0.0,
                    ..sampled.clone()
                },
                151_936,
            ),
            (
                "no top_k",
                SamplingParams {
                    top_k: 0,
                    ..sampled.clone()
                },
                151_936,
            ),
            ("top_k past the vocabulary", sampled.clone(), 40),
            (
                "a frequency penalty",
                SamplingParams {
                    frequency_penalty: 0.5,
                    ..sampled.clone()
                },
                151_936,
            ),
            (
                "a presence penalty",
                SamplingParams {
                    presence_penalty: 0.1,
                    ..sampled.clone()
                },
                151_936,
            ),
        ] {
            assert_eq!(
                coupled_draw_reads_only_top_k(&params, vocab),
                None,
                "{what}"
            );
        }
    }

    /// Temperature 0 is the argmax whatever the noise.
    #[test]
    fn a_greedy_request_ignores_the_noise() {
        let base = logits(24);
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut ctx = SamplingContext::new(24);
        for pos in 0..50u64 {
            let mut l = base.clone();
            assert_eq!(
                sample_token_coupled(&mut l, &params, &[], &mut ctx, &CoupledNoise::new(pos), pos),
                argmax(&base)
            );
        }
    }

    /// **Positions must line up**, and this is the trap a relative key falls
    /// into: a drafter with the SAME distribution, drawing with the same noise at
    /// the positions the sampler keys by, has every draft accepted at
    /// temperature 1.0 — and off by one position it is mostly refused.
    #[test]
    fn a_drafter_keyed_at_the_samplers_positions_is_accepted_and_one_off_is_not() {
        let vocab = 48usize;
        let rows: Vec<Vec<f32>> = (0..9)
            .map(|r| {
                (0..vocab)
                    .map(|i| ((i * 7 + r * 3) as f32 * 0.29).sin() * 1.2)
                    .collect()
            })
            .collect();
        let params = SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            ..Default::default()
        };
        let noise = CoupledNoise::new(0xABCD);
        let first = 1000u64;
        let draft_at = |offset: u64| -> Vec<u32> {
            let mut ctx = SamplingContext::new(vocab);
            let mut history = Vec::new();
            (0..8)
                .map(|j| {
                    let mut row = rows[j].clone();
                    let t = sample_token_coupled(
                        &mut row,
                        &params,
                        &history,
                        &mut ctx,
                        &noise,
                        first + offset + j as u64,
                    );
                    history.push(t);
                    t
                })
                .collect()
        };
        let aligned = draft_at(0);
        let (kept, _, all) =
            sampled_accept_reject(&aligned, &rows, &params, &[], Some((&noise, first)));
        assert!(
            all && kept == aligned,
            "aligned drafts must all be kept: kept {} of 8",
            kept.len()
        );

        let mut refused = 0;
        for trial in 0..20u64 {
            let n = CoupledNoise::new(trial);
            let mut ctx = SamplingContext::new(vocab);
            let drafts: Vec<u32> = (0..8)
                .map(|j| {
                    sample_token_coupled(
                        &mut rows[j].clone(),
                        &params,
                        &[],
                        &mut ctx,
                        &n,
                        first + 1 + j as u64,
                    )
                })
                .collect();
            let (kept, _, _) =
                sampled_accept_reject(&drafts, &rows, &params, &[], Some((&n, first)));
            refused += usize::from(kept.len() < 8);
        }
        assert!(
            refused >= 18,
            "one position off should break agreement: {refused}/20 refused"
        );
    }

    /// A reply with fewer rows than drafts + 1 is refused by the rule itself —
    /// never an index panic, whichever caller forgot to check.
    #[test]
    fn too_few_rows_are_refused_not_indexed() {
        let rows = vec![vec![0.1f32, 2.0], vec![1.0, 0.0]];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(
            sampled_accept_reject(&[1, 0, 1], &rows, &params, &[], None),
            (Vec::new(), 0, false)
        );
    }
}

#[cfg(test)]
mod tree_walk_tests {
    use super::*;
    use crate::inference::draft_tree::DraftTree;

    fn rows(n: usize, vocab: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                (0..vocab)
                    .map(|_| {
                        s = s
                            .wrapping_mul(6364136223846793005)
                            .wrapping_add(1442695040888963407);
                        ((s >> 33) as f32 / (1u64 << 31) as f32) * 6.0 - 3.0
                    })
                    .collect()
            })
            .collect()
    }

    fn argmax(row: &[f32]) -> u32 {
        row.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                if x > b.1 {
                    (i, x)
                } else {
                    b
                }
            })
            .0 as u32
    }

    /// A tree that is a chain walks exactly as the chain's walk does, greedy
    /// and drawn with shared noise — the tree's rule IS the chain's rule.
    #[test]
    fn a_chain_shaped_tree_walks_as_the_chain_does() {
        let noise = CoupledNoise::new(7);
        for (seed, temperature) in [(1u64, 0.0f32), (2, 0.8), (3, 1.0)] {
            let r = rows(5, 40, seed);
            let params = SamplingParams {
                temperature,
                ..Default::default()
            };
            // Drafts: what the rows pick at the first two positions, then a miss.
            let mut drafts = Vec::new();
            let mut history = Vec::new();
            let mut ctx = SamplingContext::new(40);
            for (i, row) in r.iter().take(2).enumerate() {
                let mut row = row.clone();
                let t = sample_token_coupled(
                    &mut row,
                    &params,
                    &history,
                    &mut ctx,
                    &noise,
                    11 + i as u64,
                );
                drafts.push(t);
                history.push(t);
            }
            drafts.extend([39, 38]);
            let chain =
                walk_verified_positions(&drafts, &r, &params, &[], Some((&noise, 11))).unwrap();
            let tree = DraftTree::new(10, vec![None, Some(0), Some(1), Some(2), Some(3)]).unwrap();
            let walked =
                walk_verified_tree(&tree, &drafts, &r, &params, &[], Some((&noise, 11))).unwrap();
            assert_eq!(walked, chain, "temperature {temperature}");
            assert!(walked.len() >= 3, "kept the two drafts that were the draws");
        }
    }

    /// Branches: the walk follows the guess carrying the model's own pick at
    /// each row, whichever sibling it is, and stops with the pick no guess carries.
    #[test]
    fn the_walk_follows_the_branch_that_carries_each_pick() {
        // Row 0 kept; rows 1, 2 follow it; row 3 follows 2; row 4 follows 3.
        let tree = DraftTree::new(20, vec![None, Some(0), Some(0), Some(2), Some(3)]).unwrap();
        let r = rows(5, 50, 9);
        let picks: Vec<u32> = r.iter().map(|row| argmax(row)).collect();
        // Row 1 carries something else; row 2 carries row 0's pick; row 3 row
        // 2's; row 4 does NOT carry row 3's.
        let other = |t: u32| (t + 1) % 50;
        let drafts = vec![other(picks[0]), picks[0], picks[2], other(picks[3])];
        let greedy = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let walked = walk_verified_tree(&tree, &drafts, &r, &greedy, &[], None).unwrap();
        assert_eq!(walked, vec![picks[0], picks[2], picks[3]]);
        assert_eq!(argmax_walk_tree(&tree, &drafts, &picks).unwrap(), walked);
        // The coordinator reads the same path back as rows.
        let mut tokens = vec![0u32];
        tokens.extend(&drafts);
        assert_eq!(
            crate::inference::draft_tree::kept_rows(&tree, 1, &tokens, &walked[..2]),
            Some(vec![2, 3])
        );
    }

    /// A tree that flows (#134 (b)): the tail answers each row's own draw and
    /// the requester walks them (`draft_tree::walk_draws`). That must emit
    /// EXACTLY what the tail's own walk emits with the same noise — greedy and
    /// drawn — and a row's draw must not depend on which forward carried it.
    #[test]
    fn walking_each_rows_draw_is_the_tails_own_walk() {
        use crate::inference::draft_tree::{walk_draws, WalkEnd};
        let noise = CoupledNoise::new(11);
        // Row 0 is the check's own token; 1 and 2 follow it; 3 follows 2; 4
        // follows 3; 5 follows 1.
        let parents = vec![None, Some(0), Some(0), Some(2), Some(3), Some(1)];
        let tree = DraftTree::new(30, parents).unwrap();
        for (seed, temperature) in [(5u64, 0.0f32), (6, 0.7), (7, 1.0)] {
            let r = rows(6, 40, seed);
            let params = SamplingParams {
                temperature,
                ..Default::default()
            };
            let draws = draw_each_row(&tree, 0, &r, &params, Some((&noise, 31))).unwrap();
            // Guesses that carry the draws down rows 0 → 2 → 3, then miss.
            let other = |t: u32| (t + 1) % 40;
            let drafts = vec![other(draws[0]), draws[0], draws[2], other(draws[3]), 7];
            let mut tokens = vec![0u32];
            tokens.extend(&drafts);
            let at_tail =
                walk_verified_tree(&tree, &drafts, &r, &params, &[], Some((&noise, 31))).unwrap();
            let known: Vec<Option<u32>> = draws.iter().copied().map(Some).collect();
            let (emitted, end) = walk_draws(&tree, &tokens, &known, 0);
            assert_eq!(emitted, at_tail, "temperature {temperature}");
            assert_eq!(emitted, vec![draws[0], draws[2], draws[3]]);
            assert_eq!(
                end,
                WalkEnd::Missed {
                    row: 3,
                    token: draws[3]
                }
            );
            // The same rows read in two forwards draw the same.
            let first = draw_each_row(
                &DraftTree::new(30, vec![None, Some(0), Some(0)]).unwrap(),
                0,
                &r[..3],
                &params,
                Some((&noise, 31)),
            )
            .unwrap();
            let later = draw_each_row(&tree, 3, &r[3..], &params, Some((&noise, 31))).unwrap();
            assert_eq!([first, later].concat(), draws);
            // A walk reaching a row whose draw is not back waits there.
            let mut partial = known.clone();
            partial[3] = None;
            assert_eq!(
                walk_draws(&tree, &tokens, &partial, 0),
                (vec![draws[0], draws[2]], WalkEnd::Waiting { row: 3 })
            );
        }
        // Greedy draws are each row's argmax.
        let r = rows(6, 40, 8);
        let greedy = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let draws = draw_each_row(&tree, 0, &r, &greedy, None).unwrap();
        assert_eq!(draws, r.iter().map(|row| argmax(row)).collect::<Vec<_>>());
        // Rows the tree does not end with, or a non-finite one, are refused.
        assert!(draw_each_row(&tree, 1, &r, &greedy, None).is_err());
        let mut bad = r.clone();
        bad[4][2] = f32::INFINITY;
        assert!(draw_each_row(&tree, 0, &bad, &greedy, None).is_err());
    }

    #[test]
    fn a_tree_no_honest_coordinator_sends_is_refused() {
        let r = rows(3, 10, 4);
        let greedy = SamplingParams::default();
        // A guess hanging off a kept row that is not the last.
        let t = DraftTree::new(5, vec![None, Some(0), Some(0)]).unwrap();
        assert!(walk_verified_tree(&t, &[1], &r, &greedy, &[], None).is_err());
        // Kept rows that are not a chain.
        let t = DraftTree::new(5, vec![None, None, Some(1)]).unwrap();
        assert!(walk_verified_tree(&t, &[1], &r, &greedy, &[], None).is_err());
        // Rows that are not the tree's.
        let t = DraftTree::new(5, vec![None, Some(0)]).unwrap();
        assert!(walk_verified_tree(&t, &[1], &r, &greedy, &[], None).is_err());
        // A non-finite row the walk reads.
        let t = DraftTree::new(5, vec![None, Some(0), Some(1)]).unwrap();
        let mut bad = r.clone();
        bad[0][3] = f32::NAN;
        assert!(walk_verified_tree(&t, &[1, 2], &bad, &greedy, &[], None).is_err());
    }
}
