//! Guessing ahead with this model for a bigger one — the in-engine drafter of
//! speculation across computers (`pipeline::engine_drafter`, the worker's
//! `DaemonMsg::Draft`).

use crate::error::SwarmError;

use super::kv_cache::KvCacheStore;
use super::model::SplitModel;

/// The longest read a draft call makes as single positions on a card (see
/// `draft_after`): past a few positions, one pass reading the weights once
/// wins even against captured steps.
const SINGLE_STEP_READ_MAX: usize = 4;

/// How many followers of each guess a tree considers ([`SplitModel::draft_tree_after`]):
/// SpecExec's branching factor is 16; past ~8 the extra followers of a node
/// are rarely likelier than a deeper guess elsewhere.
pub(super) const TREE_CHILDREN: usize = 8;

use crate::inference::draft_tree::GrownTree;

/// What one tree-growing call adds to ([`SplitModel::draft_tree_after`]).
#[derive(Debug, Clone, Copy)]
pub struct TreeGrowth<'a> {
    /// The tree grown so far after the same token, rated from it: read again
    /// in one forward, then added to. An empty one grows from nothing.
    pub seed: &'a GrownTree,
    /// Guesses to add.
    pub rows: usize,
    /// Guesses read in one forward while it grows.
    pub per_step: usize,
    /// Stop adding guesses once the call has run this long — a tree grown
    /// while a check is on its way is wanted when the check's answer is
    /// (`pipeline::dsd_tree`). `None` adds all `rows`.
    pub grow_for: Option<std::time::Duration>,
}

/// A drafter's tree that FLOWS, kept between calls (#134 (b),
/// [`SplitModel::grow_flow`]): the requester's [`FlowTree`] as this model
/// holds it — the same rows in the same order: a proposal is a row of both
/// the moment it is made, and every other change is one of the requester's
/// [`FlowOp`]s — with the rows it has read in its cache (`base + read`
/// positions), each row's path rating, and the followers of read rows it has
/// not proposed yet. A call reads what it has not read, one forward a level,
/// and proposes; nothing is read twice and nothing is cut, but at a rebase.
///
/// [`FlowTree`]: crate::inference::draft_tree::FlowTree
/// [`FlowOp`]: crate::inference::draft_tree::FlowOp
#[derive(Debug)]
pub struct FlowGrowth {
    flow: crate::inference::draft_tree::FlowTree,
    read: usize,
    rating: Vec<f32>,
    queue: std::collections::BinaryHeap<Candidate>,
    /// Each read row's likeliest followers and their log-probabilities —
    /// what a copied row under it is rated by (`FlowOp::Copy`).
    followers: Vec<Vec<(u32, f32)>>,
    /// A copied row not yet rated: its rating waits for its parent's read.
    unrated: Vec<bool>,
}

impl FlowGrowth {
    /// The tree as this session holds it.
    #[cfg(test)]
    pub(crate) fn flow(&self) -> &crate::inference::draft_tree::FlowTree {
        &self.flow
    }
}

/// What one call of a tree that flows asks for ([`SplitModel::grow_flow`]).
#[derive(Debug, Clone, Copy)]
pub struct FlowStep<'a> {
    /// The requester's changes since the last call, in its order.
    pub ops: &'a [crate::inference::draft_tree::FlowOp],
    /// Rows to propose a level, at most.
    pub rows: usize,
    /// Levels: each reads the last one's proposals in one forward and
    /// proposes after them, so a branch grows up to this many rows deep in
    /// one call (0 reads as 1). At one a call, a walk advanced 0.63-0.74
    /// tokens a call whatever the rows (8, 16 or 32, 2026-10-22) — the depth
    /// the drafter added, never the rows it added, was what bound it.
    pub levels: usize,
    /// Propose none past this many live rows (the walk's row and those
    /// following it), the proposals counted: past ~128 a row's chance of
    /// lying on the reply's path is too small to pay for its place
    /// (`docs/plans/large_model_splits.md` § "A tree that flows").
    pub live_most: usize,
}

/// A guess waiting for a place in the tree, rated by its path's log-probability.
#[derive(Debug)]
struct Candidate {
    score: f32,
    own: f32,
    parent: Option<usize>,
    token: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.score.total_cmp(&other.score).is_eq()
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.score.total_cmp(&other.score)
    }
}

/// The `k` likeliest tokens of `logits` at `temperature` (1 for a greedy
/// request) with their log-probabilities, likeliest first; `drawn`, when given,
/// is rated first whatever its probability — the coupled draw is the
/// drafter's best guess at what the target's sampler picks.
fn likeliest(logits: &[f32], temperature: f32, k: usize, drawn: Option<u32>) -> Vec<(u32, f32)> {
    let Some((t, log_sum)) = row_log_sum(logits, temperature) else {
        return Vec::new();
    };
    let logp = |i: usize| logits[i] / t - log_sum;
    let best = likeliest_indices(logits, k);
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    if let Some(d) = drawn.filter(|&d| (d as usize) < logits.len()) {
        // Rated as the likeliest sibling: ahead of every other follower.
        let top = best.first().map_or(0.0, |&i| logp(i));
        out.push((d, top.max(logp(d as usize))));
    }
    out.extend(
        best.into_iter()
            .filter(|&i| Some(i as u32) != drawn)
            .map(|i| (i as u32, logp(i)))
            .take(k.saturating_sub(usize::from(drawn.is_some()))),
    );
    out
}

/// A row's temperature (1 for a greedy request) and log-sum-exp at it; `None`
/// where nothing is finite.
fn row_log_sum(logits: &[f32], temperature: f32) -> Option<(f32, f32)> {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let finite = logits.iter().copied().filter(|x| x.is_finite());
    let max = finite.clone().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    Some((
        t,
        max / t + finite.map(|x| ((x - max) / t).exp()).sum::<f32>().ln(),
    ))
}

/// [`likeliest`] with each token's raw score beside its log-probability, no
/// draw: what the card's `tree_rank::top_k_on_the_card` answers, ranked here
/// for a row read back whole.
fn likeliest_scored(
    logits: &[f32],
    temperature: f32,
    k: usize,
) -> Vec<crate::inference::sampling::RankedToken> {
    let Some((t, log_sum)) = row_log_sum(logits, temperature) else {
        return Vec::new();
    };
    likeliest_indices(logits, k)
        .into_iter()
        .map(|i| crate::inference::sampling::RankedToken {
            token: i as u32,
            logp: logits[i] / t - log_sum,
            logit: logits[i],
        })
        .collect()
}

/// The `k` highest finite scores' indices, highest first, a tie to the
/// smaller index — one pass, each index offered to a list of `k`.
fn likeliest_indices(logits: &[f32], k: usize) -> Vec<usize> {
    let mut best: Vec<usize> = Vec::with_capacity(k + 1);
    if k == 0 {
        return best;
    }
    for (i, &x) in logits.iter().enumerate() {
        if !x.is_finite() {
            continue;
        }
        if best.len() < k {
            best.push(i);
            if best.len() == k {
                best.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
            }
        } else if x > logits[best[k - 1]] {
            best[k - 1] = i;
            let mut j = k - 1;
            while j > 0 && logits[best[j]] > logits[best[j - 1]] {
                best.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    if best.len() < k {
        best.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    }
    best
}

/// Each row's followers, likeliest first: (token, log-probability).
type RankedRows = Vec<Vec<(u32, f32)>>;

/// Each row's likeliest tokens for a coupled draw, best first, with raw scores.
type RankedTopK = Vec<Vec<crate::inference::sampling::RankedToken>>;

/// [`likeliest`] for every row of a `[rows, vocab]` logits tensor, ranked on
/// its own device: only `k` (token, log-probability) pairs a row come back,
/// never the row. `None` where a row's sum is not finite — the host ranks
/// those as before. Asked on a card only ([`StepScores::read`]); on the
/// processor one pass over the row is cheaper than `k`.
///
/// A tree step read its rows back whole — 8 × 152K logits, 4.9 MB, for a
/// Qwen drafter — and scanned each on the host three times: ~12 ms of a ~28 ms
/// step on the TH↔AU link (2026-10-21), and ~100 ms for the 65-row seed a call
/// beside a check starts from. The chain drafter learned the same lesson
/// (`argmax_guesses_on_card`); EAGLE-2 and SpecExec rank on the device too
/// (`torch.topk`). On a card the two launches of `kernels/tree_rank.cu` rank
/// the level (`inference::tree_rank`, card builds only: one read of each row,
/// one copy back); elsewhere, or with `SWARMLLM_TREE_TOPK=composed`, the
/// composition.
fn likeliest_rows_where_they_are(
    logits: &candle_core::Tensor,
    temperature: f32,
    k: usize,
) -> Result<Option<RankedRows>, SwarmError> {
    #[cfg(feature = "candle-cuda")]
    {
        use crate::inference::tree_rank::{ranked_on_the_card, CardRanking};
        match ranked_on_the_card(logits, temperature, k) {
            Ok(CardRanking::Ranked(ranked)) => return Ok(Some(ranked)),
            Ok(CardRanking::NotFinite) => return Ok(None),
            Ok(CardRanking::NotTaken) => {}
            // A kernel that will not load or launch costs the level its speed,
            // never the drafter: the composition ranks it as before.
            Err(e) => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        error = %e,
                        "the card's tree ranking failed; ranking with candle's ops instead"
                    );
                });
            }
        }
    }
    likeliest_rows_composed(logits, temperature, k)
}

/// [`likeliest_rows_where_they_are`] composed from candle ops: `k` picks of
/// the row's maximum, each written over with −∞, reproduce [`likeliest`]'s
/// order; ties may break differently. About thirty launches and three reads
/// back a level.
fn likeliest_rows_composed(
    logits: &candle_core::Tensor,
    temperature: f32,
    k: usize,
) -> Result<Option<RankedRows>, SwarmError> {
    use candle_core::{DType, Tensor};
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let scaled = logits.to_dtype(DType::F32).map_err(SwarmError::internal)?;
    let scaled = if t == 1.0 {
        scaled
    } else {
        (scaled / f64::from(t)).map_err(SwarmError::internal)?
    };
    // A buffer of its own: each pick is written over in place below, and
    // `to_dtype` of an f32 tensor shares the caller's storage.
    let scores = scaled
        .contiguous()
        .and_then(|s| s.copy())
        .map_err(SwarmError::internal)?;
    let (rows, vocab) = scores.dims2().map_err(SwarmError::internal)?;
    let k = k.min(vocab);
    if rows == 0 || k == 0 {
        return Ok(Some(vec![Vec::new(); rows]));
    }
    let log_sum: Vec<f32> = scores
        .log_sum_exp(1)
        .and_then(|s| s.to_vec1())
        .map_err(SwarmError::internal)?;
    if log_sum.iter().any(|s| !s.is_finite()) {
        return Ok(None);
    }
    let taken = Tensor::full(f32::NEG_INFINITY, (rows, 1), scores.device())
        .map_err(SwarmError::internal)?;
    let mut ids = Vec::with_capacity(k);
    let mut values = Vec::with_capacity(k);
    for _ in 0..k {
        let best = scores.argmax_keepdim(1).map_err(SwarmError::internal)?;
        values.push(scores.gather(&best, 1).map_err(SwarmError::internal)?);
        scores
            .scatter_set(&best, &taken, 1)
            .map_err(SwarmError::internal)?;
        ids.push(best);
    }
    let ids: Vec<Vec<u32>> = Tensor::cat(&ids, 1)
        .and_then(|t| t.to_vec2())
        .map_err(SwarmError::internal)?;
    let values: Vec<Vec<f32>> = Tensor::cat(&values, 1)
        .and_then(|t| t.to_vec2())
        .map_err(SwarmError::internal)?;
    if values.iter().flatten().any(|v| v.is_nan()) {
        return Ok(None);
    }
    Ok(Some(
        ids.into_iter()
            .zip(values)
            .zip(log_sum)
            .map(|((ids, values), log_sum)| {
                ids.into_iter()
                    .zip(values)
                    // Past a row's finite scores: the host never offers those.
                    .take_while(|&(_, v)| v.is_finite())
                    .map(|(id, v)| (id, v - log_sum))
                    .collect()
            })
            .collect(),
    ))
}

/// [`likeliest`] for a row the card ranked for a coupled draw
/// ([`RowScores::Candidates`], best first): its `k` likeliest, `drawn` first
/// and rated as the likeliest sibling — the draw is always among them.
fn likeliest_of_ranked(
    ranked: &[crate::inference::sampling::RankedToken],
    k: usize,
    drawn: Option<u32>,
) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    if let Some(d) = drawn {
        let top = ranked.first().map_or(0.0, |c| c.logp);
        let own = ranked.iter().find(|c| c.token == d).map_or(top, |c| c.logp);
        out.push((d, top.max(own)));
    }
    out.extend(
        ranked
            .iter()
            .take(k)
            .filter(|c| Some(c.token) != drawn)
            .map(|c| (c.token, c.logp))
            .take(k.saturating_sub(usize::from(drawn.is_some()))),
    );
    out
}

/// A coupled level's rows ranked on the card for the draw: each row's
/// likeliest `max(top_k, TREE_CHILDREN)` with their raw scores
/// (`tree_rank::top_k_on_the_card`), where the draw reads only the request's
/// top k (`sampling::coupled_draw_reads_only_top_k`). `None`: read the rows
/// whole, as before.
///
/// A sampled reply read every row back whole — 608 KB at 152K — and ranked it
/// on the host: 34-43 ms a flow call against 25-29 for a greedy one on the
/// TH↔AU link (2026-10-23, coder-1.5B Q8_0, temperature 0.7), the cost
/// [`likeliest_rows_where_they_are`] had already taken out of greedy replies.
fn likeliest_for_a_coupled_draw(
    logits: &candle_core::Tensor,
    sampling: &crate::types::SamplingParams,
) -> Result<Option<(RankedTopK, usize)>, SwarmError> {
    let vocab = logits
        .dim(candle_core::D::Minus1)
        .map_err(SwarmError::internal)?;
    let Some(k) = crate::inference::sampling::coupled_draw_reads_only_top_k(sampling, vocab) else {
        return Ok(None);
    };
    #[cfg(feature = "candle-cuda")]
    {
        use crate::inference::tree_rank::{top_k_on_the_card, CardTopK};
        match top_k_on_the_card(logits, sampling.temperature, k.max(TREE_CHILDREN)) {
            Ok(CardTopK::Rows(rows)) => return Ok(Some((rows, vocab))),
            Ok(CardTopK::NotTaken | CardTopK::NotFinite) => {}
            // A kernel that will not load or launch costs the level its
            // speed, never the drafter: the rows are read whole.
            Err(e) => {
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        error = %e,
                        "the card's top-k for a sampled reply failed; reading its rows whole"
                    );
                });
            }
        }
    }
    #[cfg(not(feature = "candle-cuda"))]
    let _ = k;
    Ok(None)
}

/// A chain's sampled guess from the card: its one row's likeliest ranked there
/// ([`likeliest_for_a_coupled_draw`]), the draw among the request's top k
/// (`sampling::sample_coupled_from_top_k`, the token `sample_token_coupled`
/// draws from the whole row) and its probability — never the row read back.
/// `None`: read it whole.
fn coupled_guess_on_the_card(
    logits: &candle_core::Tensor,
    sampling: &crate::types::SamplingParams,
    ctx: &mut crate::inference::sampling::SamplingContext,
    noise: &crate::inference::coupled_noise::CoupledNoise,
    position: u64,
) -> Result<Option<(u32, f32)>, SwarmError> {
    let row = logits
        .flatten_all()
        .and_then(|t| t.unsqueeze(0))
        .map_err(SwarmError::internal)?;
    let Some((rows, vocab)) = likeliest_for_a_coupled_draw(&row, sampling)? else {
        return Ok(None);
    };
    let Some(ranked) = rows.first().filter(|r| !r.is_empty()) else {
        return Ok(None);
    };
    let k = (sampling.top_k as usize).min(ranked.len());
    let t = crate::inference::sampling::sample_coupled_from_top_k(
        &ranked[..k],
        sampling,
        ctx,
        noise,
        position,
        vocab,
    );
    let logp = ranked
        .iter()
        .find(|c| c.token == t)
        .map_or(f32::NEG_INFINITY, |c| c.logp);
    Ok(Some((t, logp.exp())))
}

/// A row's scores as a tree step hands them to its queue: the logits, ranked
/// on the host (with the coupled draw where the request samples), the
/// followers the card already ranked ([`likeliest_rows_where_they_are`]), or a
/// sampled row's likeliest the card ranked for its draw
/// ([`likeliest_for_a_coupled_draw`]).
enum RowScores<'a> {
    Logits(&'a [f32]),
    Ranked(&'a [(u32, f32)]),
    Candidates {
        likeliest: &'a [crate::inference::sampling::RankedToken],
        vocab: usize,
    },
}

impl RowScores<'_> {
    /// This row's followers for the queue, likeliest first
    /// ([`TREE_CHILDREN`]); with `draw` — the request's noise and the position
    /// the follower will occupy — the coupled draw first. `history` is what
    /// the sampler reads before drawing from a whole row (its penalties); a row
    /// is ranked on the card only where nothing reads it. The ONE way a tree's
    /// growth turns a row into followers, rounds and flow alike.
    fn followers(
        &self,
        sampling: &crate::types::SamplingParams,
        draw: Option<(&crate::inference::coupled_noise::CoupledNoise, u64)>,
        history: impl FnOnce() -> Vec<u32>,
        ctx: &mut crate::inference::sampling::SamplingContext,
    ) -> Vec<(u32, f32)> {
        match *self {
            // Ranked only where nothing is drawn (`StepScores::read`).
            RowScores::Ranked(ranked) => ranked.to_vec(),
            RowScores::Logits(logits)
                if draw.is_some()
                    && crate::inference::sampling::coupled_draw_reads_only_top_k(
                        sampling,
                        logits.len(),
                    )
                    .is_some() =>
            {
                let ranked = likeliest_scored(
                    logits,
                    sampling.temperature,
                    (sampling.top_k as usize).max(TREE_CHILDREN),
                );
                followers_drawn_from_top_k(&ranked, sampling, draw, ctx, logits.len())
            }
            RowScores::Logits(logits) => {
                let drawn = draw.map(|(noise, position)| {
                    let mut row = logits.to_vec();
                    crate::inference::sampling::sample_token_coupled(
                        &mut row,
                        sampling,
                        &history(),
                        ctx,
                        noise,
                        position,
                    )
                });
                likeliest(logits, sampling.temperature, TREE_CHILDREN, drawn)
            }
            RowScores::Candidates { likeliest, vocab } => {
                followers_drawn_from_top_k(likeliest, sampling, draw, ctx, vocab)
            }
        }
    }
}

/// A row's followers from its likeliest tokens, best first — the request's
/// `top_k` of them at least, where the coupled draw reads only those
/// (`sampling::coupled_draw_reads_only_top_k`): the draw among the first
/// `top_k`, rated first, then the others rated by the chance the far end
/// draws each ([`followers_rated_by_the_shared_noise`]). The ONE way a sampled
/// row is drawn and ranked, whether the card ranked it or this host did.
fn followers_drawn_from_top_k(
    ranked: &[crate::inference::sampling::RankedToken],
    sampling: &crate::types::SamplingParams,
    draw: Option<(&crate::inference::coupled_noise::CoupledNoise, u64)>,
    ctx: &mut crate::inference::sampling::SamplingContext,
    vocab: usize,
) -> Vec<(u32, f32)> {
    // Nothing finite: no followers, and nothing to draw (`likeliest`).
    if ranked.is_empty() {
        return Vec::new();
    }
    let Some((noise, position)) = draw else {
        return likeliest_of_ranked(ranked, TREE_CHILDREN, None);
    };
    // Only the first `top_k` can ever be drawn, so only they may be followers:
    // `ranked` holds at least `TREE_CHILDREN`, and below that a token past
    // `top_k` was a row the far end could never accept (review, 2026-10-23).
    let drawable = &ranked[..(sampling.top_k as usize).min(ranked.len())];
    let drawn = crate::inference::sampling::sample_coupled_from_top_k(
        drawable, sampling, ctx, noise, position, vocab,
    );
    if sampled_rows_rated_by_probability() {
        return likeliest_of_ranked(drawable, TREE_CHILDREN, Some(drawn));
    }
    followers_rated_by_the_shared_noise(drawable, drawn, sampling.temperature, noise, position)
}

/// `SWARMLLM_TREE_SAMPLED_RATING=probability`: a sampled row's followers rated
/// by the drafter's probability at the request's temperature, as before
/// 2026-10-23 — the A/B against [`followers_rated_by_the_shared_noise`].
fn sampled_rows_rated_by_probability() -> bool {
    static BY_PROBABILITY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BY_PROBABILITY.get_or_init(|| {
        std::env::var("SWARMLLM_TREE_SAMPLED_RATING").is_ok_and(|v| v == "probability")
    })
}

/// A sampled row's followers rated by the chance the far end DRAWS each: the
/// log-softmax of `logit + T·g` over the ranked tokens, `g` the request's noise
/// at that token and position — the very noise the far end's draw adds
/// (`sampling::draw_each_row`). A token the drafter likes whose noise is low is
/// unlikely to be the draw, since the far end sees that low noise too; the
/// drafter's probability at `T` alone rated it as likely as any (Gumbel-top-k,
/// Kool et al. 1903.06059, here with the target's own noise). The drawn token
/// comes first, rated as the likeliest sibling; the rest by that score.
///
/// Why `T·g` and no other scale (2026-10-23, 1,200 positions of Qwen2.5-Coder-7B
/// replies sampled at 0.7, the coder-1.5B Q8_0 drafter teacher-forced): the
/// fitted scale of the softmax was 1/T, so the drafter's logits err by about a
/// unit-scale Gumbel, and at T → 0 this is the greedy rule's own (calibrated)
/// probability. Claimed per rank [0.716, 0.115, 0.051, 0.030, 0.018] against hit
/// [0.733, 0.122, 0.046, 0.026, 0.017]; the probability at T claimed [0.795,
/// 0.159, 0.043, 0.019, 0.010] — the second child overrated, the tail halved, so
/// a tree grew too few of the alternatives that hold the draw. Trees of 128
/// grown by each rating: 6.90 tokens a round against 6.15 (greedy 7.19).
fn followers_rated_by_the_shared_noise(
    ranked: &[crate::inference::sampling::RankedToken],
    drawn: u32,
    temperature: f32,
    noise: &crate::inference::coupled_noise::CoupledNoise,
    position: u64,
) -> Vec<(u32, f32)> {
    let t = f64::from(temperature);
    let score: Vec<f64> = ranked
        .iter()
        .map(|c| f64::from(c.logit) + t * noise.gumbel(position, c.token))
        .collect();
    let max = score.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_sum = max + score.iter().map(|s| (s - max).exp()).sum::<f64>().ln();
    let mut order: Vec<usize> = (0..ranked.len()).collect();
    order.sort_by(|&a, &b| score[b].total_cmp(&score[a]));
    let top = score[order[0]] - log_sum;
    let own = ranked
        .iter()
        .position(|c| c.token == drawn)
        .map_or(top, |i| score[i] - log_sum);
    let mut out = Vec::with_capacity(TREE_CHILDREN);
    out.push((drawn, top.max(own) as f32));
    out.extend(
        order
            .into_iter()
            .filter(|&i| ranked[i].token != drawn)
            .take(TREE_CHILDREN - 1)
            .map(|i| (ranked[i].token, (score[i] - log_sum) as f32)),
    );
    out
}

/// Each row of a tree step's `[rows, vocab]` logits for its queue: ranked on
/// the card where nothing needs the whole row — a sampled row's draw needs
/// only its top k — else read back whole.
enum StepScores {
    Logits(Vec<Vec<f32>>),
    Ranked(RankedRows),
    Candidates { rows: RankedTopK, vocab: usize },
}

impl StepScores {
    fn read(
        logits: &candle_core::Tensor,
        sampling: &crate::types::SamplingParams,
        coupled: bool,
    ) -> Result<Self, SwarmError> {
        // `SWARMLLM_TREE_RANK=host` ranks every row on the host, as before.
        if logits.device().is_cuda() && std::env::var("SWARMLLM_TREE_RANK").as_deref() != Ok("host")
        {
            if !coupled {
                if let Some(ranked) =
                    likeliest_rows_where_they_are(logits, sampling.temperature, TREE_CHILDREN)?
                {
                    return Ok(Self::Ranked(ranked));
                }
            } else if let Some((rows, vocab)) = likeliest_for_a_coupled_draw(logits, sampling)? {
                return Ok(Self::Candidates { rows, vocab });
            }
        }
        logits
            .to_dtype(candle_core::DType::F32)
            .and_then(|t| t.to_vec2())
            .map(Self::Logits)
            .map_err(SwarmError::internal)
    }

    fn len(&self) -> usize {
        match self {
            Self::Logits(rows) => rows.len(),
            Self::Ranked(rows) => rows.len(),
            Self::Candidates { rows, .. } => rows.len(),
        }
    }

    fn row(&self, i: usize) -> RowScores<'_> {
        match self {
            Self::Logits(rows) => RowScores::Logits(&rows[i]),
            Self::Ranked(rows) => RowScores::Ranked(&rows[i]),
            Self::Candidates { rows, vocab } => RowScores::Candidates {
                likeliest: &rows[i],
                vocab: *vocab,
            },
        }
    }
}

impl SplitModel {
    /// Read `append` into this request's cache from position `keep`, then guess
    /// `gamma` tokens, reading each back in so the next can follow it — all but
    /// the last, which nothing has followed yet. The cache is cut back to `keep`
    /// first; afterwards it holds `keep + append.len() + gamma - 1` positions.
    ///
    /// A cache holding FEWER than `keep` positions is refused, not read on top
    /// of: the call continues a context this worker no longer has — the entry
    /// expired while other rounds drafted, or the worker was replaced — and
    /// reading `append` at position `keep` over it would guess from a context
    /// that was never written, with nothing failing. Refused, the reply goes
    /// on without guessing (`pipeline::dsd`, `drafting_off`).
    ///
    /// A guess is the token that will occupy position `keep + append.len() + j`.
    /// With `noise` it is drawn the way the target's sampler will draw that
    /// position (`sampling::sample_token_coupled`: the request's parameters,
    /// the reply so far in `history` plus the guesses before it, the shared
    /// noise keyed by the position); without, it is the plain argmax, as the
    /// llama.cpp drafter guesses (`pipeline::speculative::draft_next_gamma`).
    ///
    /// With `stop_below`, guessing stops early — after the guess that falls
    /// below it — once this model's own probability for a guess is under that
    /// figure: an unsure guess is the one the check is most likely to refuse,
    /// and every guess after it is guessed on top of it. Hugging Face's
    /// assisted generation does the same (`ConfidenceCriteria`: the softmax of
    /// the latest scores at the token just produced, compared with
    /// `assistant_confidence_threshold`, the token kept). The probability is
    /// read at the request's temperature (1 for a greedy request), before
    /// top-k and top-p.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_after(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        keep: usize,
        append: &[u32],
        gamma: usize,
        sampling: &crate::types::SamplingParams,
        history: &[u32],
        noise: Option<&crate::inference::coupled_noise::CoupledNoise>,
        prefill_chunk_tokens: usize,
        stop_below: Option<f32>,
    ) -> Result<Vec<u32>, SwarmError> {
        let mut logits = self.read_after(kv, request_id, keep, append, prefill_chunk_tokens)?;
        let first = keep + append.len();
        if noise.is_none() && self.device.is_cuda() {
            return self.argmax_guesses_on_card(
                kv,
                request_id,
                first,
                logits,
                gamma,
                sampling.temperature,
                stop_below,
                None,
            );
        }
        let mut history = history.to_vec();
        let mut ctx = crate::inference::sampling::SamplingContext::new(0);
        let mut guesses = Vec::with_capacity(gamma);
        for j in 0..gamma {
            // A sampled guess whose draw reads only the row's top k is drawn
            // from the card's ranking of them; else the row is read whole.
            let on_the_card = match noise {
                Some(noise) if self.device.is_cuda() => coupled_guess_on_the_card(
                    &logits,
                    sampling,
                    &mut ctx,
                    noise,
                    (first + j) as u64,
                )?,
                _ => None,
            };
            let (t, unsure) = match on_the_card {
                Some((t, probability)) => (t, stop_below.is_some_and(|floor| probability < floor)),
                None => {
                    let mut row: Vec<f32> = logits
                        .flatten_all()
                        .and_then(|t| t.to_dtype(candle_core::DType::F32))
                        .and_then(|t| t.to_vec1())
                        .map_err(SwarmError::internal)?;
                    // Read before sampling, which may rewrite the row in place.
                    let raw = stop_below.map(|floor| (row.clone(), floor));
                    let t = match noise {
                        Some(noise) => crate::inference::sampling::sample_token_coupled(
                            &mut row,
                            sampling,
                            &history,
                            &mut ctx,
                            noise,
                            (first + j) as u64,
                        ),
                        // The first of equal maxima, as every argmax here picks.
                        None => {
                            row.iter()
                                .enumerate()
                                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                                    if x > best.1 {
                                        (i, x)
                                    } else {
                                        best
                                    }
                                })
                                .0 as u32
                        }
                    };
                    let unsure = raw.is_some_and(|(raw, floor)| {
                        probability_at(&raw, t as usize, sampling.temperature) < floor
                    });
                    (t, unsure)
                }
            };
            guesses.push(t);
            history.push(t);
            if unsure {
                break;
            }
            if j + 1 < gamma {
                let next = self.token_tensor(t)?;
                logits = self.forward(&next, first + j, kv, request_id)?;
            }
        }
        Ok(guesses)
    }

    /// Cut this request's cache back to `keep` and read `append` into it,
    /// answering the last position's logits — the start of every drafting call
    /// ([`Self::draft_after`], [`Self::draft_tree_after`]). A cache holding
    /// fewer than `keep` positions is refused (see `draft_after`).
    fn read_after(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        keep: usize,
        append: &[u32],
        prefill_chunk_tokens: usize,
    ) -> Result<candle_core::Tensor, SwarmError> {
        let held = kv.request_positions(self.kv_model_key(), request_id);
        if held < keep {
            return Err(SwarmError::ServiceUnavailable(format!(
                "the guessing model holds {held} positions of this reply, not the {keep} \
                 this call continues from — its cache expired or its worker was replaced"
            )));
        }
        // Also empties a first read's cache (`keep` 0) of whatever an earlier
        // call that failed part-way left in it.
        kv.truncate_request_to(self.kv_model_key(), request_id, keep)?;
        // A short read that continues the reply goes through as single
        // positions on a card: each is a decode step, sent as a CUDA graph at
        // ~3 ms on a 0.5B, where ONE multi-position forward of the same 2-4
        // tokens takes the uncaptured prompt path at 12-17 ms (measured on the
        // split rig, 2026-09-30). Most calls read two — the guess the last call
        // never fed back, and the check's own token — so this was most of a
        // call's cost. On the processor one pass reads the weights once and
        // stays cheaper; the prompt's first read is a prompt pass either way.
        let logits =
            if keep > 0 && append.len() <= SINGLE_STEP_READ_MAX && self.runs_entirely_on_card() {
                let mut last = None;
                for (i, &id) in append.iter().enumerate() {
                    let step = self.token_tensor(id)?;
                    last = Some(self.forward(&step, keep + i, kv, request_id)?);
                }
                last.expect("append is not empty — the worker refuses an empty draft call")
            } else {
                let input = self.tensor_from_ids(append)?;
                self.forward_prompt_in_chunks(
                    &input,
                    keep,
                    kv,
                    request_id,
                    None,
                    false,
                    prefill_chunk_tokens,
                )?
            };
        Ok(logits)
    }

    /// Grow a tree of guesses after reading `append` from position `keep`
    /// (#134, `docs/plans/large_model_splits.md` § "Draft trees"): the call
    /// [`Self::draft_after`] makes for a chain, for a check that keeps the
    /// longest run along whichever branch the target takes.
    ///
    /// Best-first over cumulative log-probability, as SpecExec (arXiv
    /// 2406.02532) and EAGLE-2's rerank grow theirs: each step adds the
    /// `per_step` likeliest guesses not yet in the tree, reads them in ONE tree
    /// forward of this model (`with_draft_tree`), and queues each one's
    /// [`TREE_CHILDREN`] likeliest followers. The last step's guesses are not
    /// read — nothing follows them yet. A guess's rating is its probability at
    /// the request's temperature (1 for a greedy request); with `noise` the
    /// draw the target's sampler will make there is rated first among its
    /// siblings, as `draft_after` guesses it.
    ///
    /// `growth.seed` is a tree already grown after the same token, rated from
    /// it — what a check's outcome left of the last one (`GrownTree::after`).
    /// Its rows are read in one forward and their followers queued without the
    /// children it holds, so the call goes on where that growth stopped: the
    /// queue is the one a single call would have had. `growth.rows` more are
    /// added, fewer once `growth.grow_for` has passed. A seed whose rows do
    /// not come after their parents, or name a token past the vocabulary, is
    /// refused.
    ///
    /// The tree is laid out with its likeliest path first
    /// (`draft_tree::trunk_first`), so a check that keeps that path keeps a
    /// prefix of the rows. Returns each guess's token, parent among the
    /// guesses (`None`: it follows `append`'s last token) and path rating
    /// ([`GrownTree`]), the seed's among them. The cache is cut back to
    /// `keep + append.len()` afterwards: it holds what the check will
    /// confirm, never a guess.
    #[allow(clippy::too_many_arguments)]
    pub fn draft_tree_after(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        keep: usize,
        append: &[u32],
        growth: TreeGrowth<'_>,
        sampling: &crate::types::SamplingParams,
        history: &[u32],
        noise: Option<&crate::inference::coupled_noise::CoupledNoise>,
        prefill_chunk_tokens: usize,
    ) -> Result<GrownTree, SwarmError> {
        use crate::inference::draft_tree::{reorder, trunk_first, DraftTree, MAX_TREE_ROWS};
        let started = std::time::Instant::now();
        let seed = growth.seed;
        let seeded = seed.tokens.len();
        if seed.parents.len() != seeded || seed.path_logp.len() != seeded {
            return Err(SwarmError::Validation(
                "a tree to grow on has tokens, parents and ratings of different lengths".into(),
            ));
        }
        if seeded > MAX_TREE_ROWS {
            return Err(SwarmError::Validation(format!(
                "a tree to grow on of {seeded} rows is past the {MAX_TREE_ROWS} one may hold"
            )));
        }
        if let Some(vocab) = self.embedding_rows() {
            if seed.tokens.iter().any(|&t| t as usize >= vocab) {
                return Err(SwarmError::Validation(
                    "a tree to grow on names a token past the vocabulary".into(),
                ));
            }
        }
        let rows = seeded.saturating_add(growth.rows).min(MAX_TREE_ROWS);
        let per_step = growth.per_step.max(1);
        let out_of_time = || {
            growth
                .grow_for
                .is_some_and(|limit| started.elapsed() >= limit)
        };
        // A seed row's children, by its index (0: the rows that follow the
        // last token read): already in the tree, so never queued again.
        let mut held: Vec<Vec<u32>> = vec![Vec::new(); seeded + 1];
        for (i, parent) in seed.parents.iter().enumerate() {
            match parent {
                None => held[0].push(seed.tokens[i]),
                Some(p) if usize::from(*p) < i => held[usize::from(*p) + 1].push(seed.tokens[i]),
                Some(_) => {
                    return Err(SwarmError::Validation(format!(
                        "row {i} of a tree to grow on comes before its parent"
                    )));
                }
            }
        }
        let logits = self.read_after(kv, request_id, keep, append, prefill_chunk_tokens)?;
        let base = keep + append.len();
        let coupled = noise.is_some();
        let root = logits
            .flatten_all()
            .and_then(|t| t.unsqueeze(0))
            .map_err(SwarmError::internal)
            .and_then(|t| StepScores::read(&t, sampling, coupled))?;
        let mut tree = DraftTree::new(base, Vec::new()).map_err(SwarmError::Internal)?;
        let mut tokens: Vec<u32> = Vec::with_capacity(rows);
        // Each guess's log-probability among its siblings, and along its path.
        let mut own: Vec<f32> = Vec::with_capacity(rows);
        let mut path: Vec<f32> = Vec::with_capacity(rows);
        let mut queue = std::collections::BinaryHeap::new();
        let mut ctx = crate::inference::sampling::SamplingContext::new(0);
        let path_tokens = |tree: &DraftTree, tokens: &[u32], row: Option<usize>| -> Vec<u32> {
            let mut out: Vec<u32> = std::iter::successors(row, |&r| tree.parent(r))
                .map(|r| tokens[r])
                .collect();
            out.reverse();
            out
        };
        let mut queue_followers = |queue: &mut std::collections::BinaryHeap<Candidate>,
                                   tree: &DraftTree,
                                   tokens: &[u32],
                                   parent: Option<usize>,
                                   so_far: f32,
                                   scores: RowScores<'_>,
                                   held: &[u32]| {
            let position = base + parent.map_or(0, |p| tree.depth(p) + 1);
            let followers = scores.followers(
                sampling,
                noise.map(|noise| (noise, position as u64)),
                || {
                    let mut seen = history.to_vec();
                    seen.extend(path_tokens(tree, tokens, parent));
                    seen
                },
                &mut ctx,
            );
            for (token, logp) in followers {
                if held.contains(&token) {
                    continue;
                }
                queue.push(Candidate {
                    score: so_far + logp,
                    own: logp,
                    parent,
                    token,
                });
            }
        };
        queue_followers(&mut queue, &tree, &tokens, None, 0.0, root.row(0), &held[0]);
        let outcome = (|| -> Result<(), SwarmError> {
            // The seed's rows go in first, parents first as it lists them, and
            // are read in one forward so their followers can be queued.
            for i in 0..seeded {
                tree.push(seed.parents[i].map(usize::from))
                    .map_err(SwarmError::Internal)?;
                let parent_rating = seed.parents[i].map_or(0.0, |p| seed.path_logp[usize::from(p)]);
                tokens.push(seed.tokens[i]);
                own.push(seed.path_logp[i] - parent_rating);
                path.push(seed.path_logp[i]);
            }
            if seeded > 0 && seeded < rows {
                let input = self.tensor_from_ids(&tokens)?;
                let out = self.with_draft_tree(tree.clone(), |m| {
                    m.forward_verify_all_positions(&input, base, kv, request_id)
                })??;
                let read = out
                    .squeeze(0)
                    .map_err(SwarmError::internal)
                    .and_then(|t| StepScores::read(&t, sampling, coupled))?;
                for row in 0..read.len() {
                    queue_followers(
                        &mut queue,
                        &tree,
                        &tokens,
                        Some(row),
                        path[row],
                        read.row(row),
                        &held[row + 1],
                    );
                }
            }
            // A step's guesses are rated before it starts, so each is added
            // whatever the time; only reading them — for their followers —
            // waits on it.
            while tokens.len() < rows {
                let mut added = Vec::with_capacity(per_step);
                while added.len() < per_step && tokens.len() < rows {
                    let Some(c) = queue.pop() else { break };
                    let row = tree.push(c.parent).map_err(SwarmError::Internal)?;
                    tokens.push(c.token);
                    own.push(c.own);
                    path.push(c.score);
                    added.push(row);
                }
                // Past the time given, the step's guesses are kept unread.
                if added.is_empty() || tokens.len() >= rows || out_of_time() {
                    break;
                }
                let input = self.tensor_from_ids(&tokens[added[0]..])?;
                let out = self.with_draft_tree(tree.clone(), |m| {
                    m.forward_verify_all_positions(&input, base + added[0], kv, request_id)
                })??;
                let read = out
                    .squeeze(0)
                    .map_err(SwarmError::internal)
                    .and_then(|t| StepScores::read(&t, sampling, coupled))?;
                for (i, &row) in added.iter().enumerate() {
                    queue_followers(
                        &mut queue,
                        &tree,
                        &tokens,
                        Some(row),
                        path[row],
                        read.row(i),
                        &[],
                    );
                }
            }
            Ok(())
        })();
        // Whatever happened, the cache holds the confirmed context only.
        kv.truncate_request_to(self.kv_model_key(), request_id, base)?;
        outcome?;
        let order = trunk_first(&tree, &own);
        let rated: Vec<(u32, f32)> = tokens.iter().copied().zip(path.iter().copied()).collect();
        let (laid, rated) = reorder(&tree, &rated, &order).map_err(SwarmError::Internal)?;
        Ok(GrownTree {
            parents: (0..laid.len())
                .map(|r| laid.parent(r).map(|p| p as u16))
                .collect(),
            tokens: rated.iter().map(|&(t, _)| t).collect(),
            path_logp: rated.iter().map(|&(_, s)| s).collect(),
        })
    }

    /// Start a tree that flows (#134 (b)): cut this request's cache back to
    /// `keep`, read `append` (the context since, NOT the token the tree starts
    /// from), and hold a tree of one row, `root`, at `keep + append.len()` —
    /// read by the first [`Self::grow_flow`] call. A cache holding fewer than
    /// `keep` positions is refused, as [`Self::draft_after`] refuses it.
    pub fn start_flow(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        keep: usize,
        append: &[u32],
        root: u32,
        prefill_chunk_tokens: usize,
    ) -> Result<FlowGrowth, SwarmError> {
        if let Some(vocab) = self.embedding_rows() {
            if root as usize >= vocab || append.iter().any(|&t| t as usize >= vocab) {
                return Err(SwarmError::Validation(
                    "a tree that flows names a token past the vocabulary".into(),
                ));
            }
        }
        if append.is_empty() {
            let held = kv.request_positions(self.kv_model_key(), request_id);
            if held < keep {
                return Err(SwarmError::ServiceUnavailable(format!(
                    "the guessing model holds {held} positions of this reply, not the {keep} \
                     this call continues from — its cache expired or its worker was replaced"
                )));
            }
            kv.truncate_request_to(self.kv_model_key(), request_id, keep)?;
        } else {
            self.read_after(kv, request_id, keep, append, prefill_chunk_tokens)?;
        }
        Ok(FlowGrowth {
            flow: crate::inference::draft_tree::FlowTree::new(keep + append.len(), root)
                .map_err(SwarmError::Validation)?,
            read: 0,
            rating: vec![0.0],
            queue: std::collections::BinaryHeap::new(),
            followers: Vec::new(),
            unrated: vec![false],
        })
    }

    /// One call of a tree that flows (#134 (b)): apply the requester's
    /// changes, then for each of `step.levels` levels read the rows not read
    /// yet in ONE tree forward (their followers join the queue) and propose up
    /// to `step.rows` rows, best-first by path rating, each following a live
    /// row and carrying a token no row following its parent carries. Each
    /// proposal is a row of this tree the moment it is made — the next
    /// level reads it — so the requester must push every one, in this order,
    /// and sends no op for it. Answers the proposals as (parent row, token).
    ///
    /// The last level's proposals are read by the next call. A level costs a
    /// forward; EAGLE-2 grows a draft the same way (one forward a depth, top-8
    /// by path probability, 5 deep), and FlowSpec, the published design
    /// nearest this one, grows depth 5 every step. Followers are rated as
    /// [`Self::draft_tree_after`] rates them (`TREE_CHILDREN` a row, the
    /// coupled draw first where `noise` is given, keyed by the position it
    /// fills). A rebase cuts the cache to the tree's old start and reads the
    /// walk's path there as context; every row kept is read again by this
    /// call. On an error the session must not be used again.
    #[allow(clippy::too_many_arguments)]
    pub fn grow_flow(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        session: &mut FlowGrowth,
        step: FlowStep<'_>,
        sampling: &crate::types::SamplingParams,
        noise: Option<&crate::inference::coupled_noise::CoupledNoise>,
        prefill_chunk_tokens: usize,
    ) -> Result<Vec<(u16, u32)>, SwarmError> {
        use crate::inference::draft_tree::{FlowOp, MAX_TREE_ROWS};
        let vocab = self.embedding_rows();
        for &op in step.ops {
            match op {
                FlowOp::Push { parent, token } | FlowOp::Copy { parent, token } => {
                    if vocab.is_some_and(|v| token as usize >= v) {
                        return Err(SwarmError::Validation(
                            "a tree that flows names a token past the vocabulary".into(),
                        ));
                    }
                    let parent = usize::from(parent);
                    // A row the model drew that no row carried is as sure as
                    // its parent. A copied one is rated by this model's own
                    // guess once its parent is read (`read_flow_rows`).
                    let rating = session.rating.get(parent).copied().unwrap_or(0.0);
                    session
                        .flow
                        .push(parent, token)
                        .map_err(SwarmError::Validation)?;
                    session.rating.push(rating);
                    session.unrated.push(matches!(op, FlowOp::Copy { .. }));
                }
                FlowOp::Advance { row } => session
                    .flow
                    .advance(usize::from(row))
                    .map_err(SwarmError::Validation)?,
                FlowOp::Rebase => {
                    let rebased = session.flow.rebase().map_err(SwarmError::Validation)?;
                    let mut new_index = vec![None; session.rating.len()];
                    for (new, &old) in rebased.kept.iter().enumerate() {
                        new_index[old] = Some(new);
                    }
                    session.rating = rebased
                        .kept
                        .iter()
                        .map(|&old| session.rating[old])
                        .collect();
                    session.unrated = rebased
                        .kept
                        .iter()
                        .map(|&old| session.unrated[old])
                        .collect();
                    // Every row kept is read again, from the start.
                    session.followers.clear();
                    session.queue = std::mem::take(&mut session.queue)
                        .into_iter()
                        .filter_map(|c| {
                            let parent = new_index[c.parent?]?;
                            Some(Candidate {
                                parent: Some(parent),
                                ..c
                            })
                        })
                        .collect();
                    kv.truncate_request_to(self.kv_model_key(), request_id, rebased.old_base)?;
                    if !rebased.path.is_empty() {
                        let input = self.tensor_from_ids(&rebased.path)?;
                        self.forward_prompt_in_chunks(
                            &input,
                            rebased.old_base,
                            kv,
                            request_id,
                            None,
                            false,
                            prefill_chunk_tokens,
                        )?;
                    }
                    session.read = 0;
                }
            }
        }
        let base = session.flow.base();
        let held = kv.request_positions(self.kv_model_key(), request_id);
        if held != base + session.read {
            return Err(SwarmError::ServiceUnavailable(format!(
                "the guessing model holds {held} positions of this reply, not the {} its tree \
                 continues from — its cache expired or its worker was replaced",
                base + session.read
            )));
        }
        let mut out = Vec::new();
        for _ in 0..step.levels.max(1) {
            self.read_flow_rows(kv, request_id, session, sampling, noise)?;
            let before = out.len();
            let mut live = session.flow.live_rows();
            while out.len() - before < step.rows
                && live < step.live_most
                && session.flow.len() < MAX_TREE_ROWS
            {
                let Some(c) = session.queue.pop() else { break };
                let Some(parent) = c.parent else { continue };
                if !session.flow.is_live(parent) || session.flow.has_child(parent, c.token) {
                    continue;
                }
                session
                    .flow
                    .push(parent, c.token)
                    .map_err(SwarmError::Internal)?;
                session.rating.push(c.score);
                session.unrated.push(false);
                out.push((parent as u16, c.token));
                live += 1;
            }
            if out.len() == before {
                break;
            }
        }
        Ok(out)
    }

    /// The rows of a tree that flows not read yet, read in ONE tree forward;
    /// the followers of the live ones join the queue.
    fn read_flow_rows(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        session: &mut FlowGrowth,
        sampling: &crate::types::SamplingParams,
        noise: Option<&crate::inference::coupled_noise::CoupledNoise>,
    ) -> Result<(), SwarmError> {
        let flow = &session.flow;
        if session.read >= flow.len() {
            return Ok(());
        }
        let base = flow.base();
        let from = session.read;
        let input = self.tensor_from_ids(&flow.tokens()[from..])?;
        let out = self.with_draft_tree(flow.tree().clone(), |m| {
            m.forward_verify_all_positions(&input, base + from, kv, request_id)
        })??;
        let coupled = noise.is_some();
        let read = out
            .squeeze(0)
            .map_err(SwarmError::internal)
            .and_then(|t| StepScores::read(&t, sampling, coupled))?;
        let mut ctx = crate::inference::sampling::SamplingContext::new(0);
        session.followers.resize(from, Vec::new());
        for (i, row) in (from..flow.len()).enumerate() {
            // A lost branch's followers could never be proposed.
            if !flow.is_live(row) {
                session.followers.push(Vec::new());
                continue;
            }
            // A flow's sampler reads no history (`dsd_flow::flow_possible`).
            let followers = read.row(i).followers(
                sampling,
                noise.map(|noise| (noise, (base + flow.tree().depth(row) + 1) as u64)),
                Vec::new,
                &mut ctx,
            );
            if session.unrated[row] {
                // A copied row is as likely as this model guesses its token
                // after its parent — no likelier than the least of the
                // parent's own followers where they do not list it — so a
                // wrong copy's followers do not crowd out its own guesses.
                if let Some(parent) = flow.tree().parent(row) {
                    if let Some(listed) = session.followers.get(parent) {
                        let token = flow.tokens()[row];
                        let logp = listed
                            .iter()
                            .find(|&&(t, _)| t == token)
                            .or_else(|| listed.iter().min_by(|a, b| a.1.total_cmp(&b.1)))
                            .map_or(0.0, |&(_, logp)| logp);
                        session.rating[row] = session.rating[parent] + logp;
                    }
                }
                session.unrated[row] = false;
            }
            for &(token, logp) in &followers {
                if flow.has_child(row, token) {
                    continue;
                }
                session.queue.push(Candidate {
                    score: session.rating[row] + logp,
                    own: logp,
                    parent: Some(row),
                    token,
                });
            }
            session.followers.push(followers);
        }
        session.read = flow.len();
        Ok(())
    }

    /// Argmax guesses that never leave the card between steps: each step's
    /// token is chosen on the card and is the next step's input there, so the
    /// host records step j+1 while the card still runs step j, and every guess
    /// comes back in ONE transfer at the end.
    ///
    /// The host loop read every row back to choose its token — ~600 KB and a
    /// wait for the card per guess — so recording (~3 ms on a 0.5B) and the
    /// card's own work (~2.7 ms) ran one after the other: 5.7 ms a guess on the
    /// split rig (2026-09-30). With `stop_below`, each guess's probability is
    /// computed on the card too and the guesses are cut after the first unsure
    /// one here — the set the host loop keeps; the steps after it are drafted
    /// and dropped, and the next call cuts the drafter's cache back as always.
    /// Ties between equal maxima may break differently from the host's
    /// first-index rule; a guess is only a guess, and the check decides.
    /// With `skips`, each step runs without those layers
    /// ([`Self::card_guesses`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn argmax_guesses_on_card(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        first: usize,
        mut logits: candle_core::Tensor,
        gamma: usize,
        temperature: f32,
        stop_below: Option<f32>,
        skips: Option<&[bool]>,
    ) -> Result<Vec<u32>, SwarmError> {
        use candle_core::Tensor;
        // A read-ahead (`engine_drafter::read_ahead`) asks for no guesses: the
        // prompt is in the cache and that is the whole call. Without this,
        // `Tensor::stack` of nothing failed every read-ahead on a card, and the
        // first round then read the prompt a second time from scratch.
        if gamma == 0 {
            return Ok(Vec::new());
        }
        let t = if temperature > 0.0 {
            f64::from(temperature)
        } else {
            1.0
        };
        let mut ids: Vec<Tensor> = Vec::with_capacity(gamma);
        let mut mass: Vec<Tensor> = Vec::with_capacity(gamma);
        for j in 0..gamma {
            let row = logits
                .flatten_all()
                .and_then(|r| r.to_dtype(candle_core::DType::F32))
                .map_err(SwarmError::internal)?;
            let best = row.argmax(0).map_err(SwarmError::internal)?;
            if stop_below.is_some() {
                // Σ exp((x - max) / T): the guess's probability is its inverse.
                let z = row
                    .max_keepdim(0)
                    .and_then(|m| row.broadcast_sub(&m))
                    .and_then(|d| d / t)
                    .and_then(|d| d.exp())
                    .and_then(|e| e.sum(0))
                    .map_err(SwarmError::internal)?;
                mass.push(z);
            }
            ids.push(best.clone());
            if j + 1 < gamma {
                let next = best.reshape((1, 1)).map_err(SwarmError::internal)?;
                logits = match skips {
                    Some(skips) => {
                        self.forward_with_skip_mask(&next, first + j, kv, request_id, skips)?
                    }
                    None => self.forward(&next, first + j, kv, request_id)?,
                };
            }
        }
        let mut guesses: Vec<u32> = Tensor::stack(&ids, 0)
            .and_then(|v| v.to_vec1())
            .map_err(SwarmError::internal)?;
        if let Some(floor) = stop_below {
            let z: Vec<f32> = Tensor::stack(&mass, 0)
                .and_then(|v| v.to_vec1())
                .map_err(SwarmError::internal)?;
            // A row that was not finite (its mass NaN) cuts too, as the host's
            // `probability_at` reads such a row as probability 0.
            if let Some(k) = z.iter().position(|&z| {
                let p = 1.0 / z;
                p.is_nan() || p < floor
            }) {
                guesses.truncate(k + 1);
            }
        }
        Ok(guesses)
    }

    /// `gamma` guesses at what follows `token` at `index_pos`, made on the
    /// card alone by skipping the layers `skips` names — the run the
    /// processor holds (`SplitModel::card_guess_skips`) — and checked by the
    /// whole model next (`model_worker::LocalSpec`). Each guess is the card's
    /// greedy pick and the next step's input there, all read back once
    /// ([`Self::argmax_guesses_on_card`]); the guesses' cache entries are cut
    /// back to `index_pos` before it returns, so the check starts from the
    /// cache a step would.
    pub(crate) fn card_guesses(
        &mut self,
        kv: &KvCacheStore,
        request_id: &str,
        index_pos: usize,
        token: u32,
        gamma: usize,
        skips: &[bool],
    ) -> Result<Vec<u32>, SwarmError> {
        if gamma == 0 {
            return Ok(Vec::new());
        }
        let input = self.token_tensor(token)?;
        let guessed = self
            .forward_with_skip_mask(&input, index_pos, kv, request_id, skips)
            .and_then(|logits| {
                self.argmax_guesses_on_card(
                    kv,
                    request_id,
                    index_pos + 1,
                    logits,
                    gamma,
                    0.0,
                    None,
                    Some(skips),
                )
            });
        // Cut back whether or not the guesses came: a check never starts on a
        // guess's cache.
        let key = self.kv_model_key().to_string();
        kv.truncate_request_to(&key, request_id, index_pos)?;
        guessed
    }
}

/// `softmax(logits / T)[token]`, with `T` the request's temperature or 1 for a
/// greedy one — the probability a drafter gives its own guess.
fn probability_at(logits: &[f32], token: usize, temperature: f32) -> f32 {
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return 0.0;
    }
    let sum: f64 = logits.iter().map(|&x| f64::from((x - max) / t).exp()).sum();
    let own = logits
        .get(token)
        .map_or(f64::NEG_INFINITY, |&x| f64::from((x - max) / t));
    (own.exp() / sum) as f32
}

#[cfg(test)]
mod tests {
    use super::{likeliest, likeliest_rows_where_they_are, probability_at};

    /// A sampled row whose draw reads only its top k is drawn and ranked
    /// through `followers_drawn_from_top_k` — whether this host ranked it or a
    /// card did — and gives the same followers either way: the whole row's
    /// coupled draw first, then the others as the shared noise rates them. A
    /// row whose draw reads the whole row (no top_k) keeps the whole-row path.
    #[test]
    fn a_sampled_row_follows_from_its_top_k_as_from_the_whole_row() {
        use super::{
            followers_rated_by_the_shared_noise, likeliest_scored, RowScores, TREE_CHILDREN,
        };
        use crate::inference::coupled_noise::CoupledNoise;
        use crate::inference::sampling::{sample_token_coupled, SamplingContext};
        use crate::types::SamplingParams;
        let vocab = 5000usize;
        let row: Vec<f32> = (0..vocab)
            .map(|i| ((i * 7919) % 5003) as f32 / 5003.0 * 14.0 + (i as f32 * 0.31).sin())
            .collect();
        let noise = CoupledNoise::new(77);
        let mut ctx = SamplingContext::new(vocab);
        let mut drawn_off_the_top = 0u32;
        for (temperature, top_k, top_p) in [
            (0.7f32, 40u32, 1.0f32),
            (0.7, 40, 0.9),
            (1.0, 3, 1.0),
            (1.3, 64, 0.95),
            (0.8, 0, 1.0),
        ] {
            let sampling = SamplingParams {
                temperature,
                top_k,
                top_p,
                ..Default::default()
            };
            let ranked = likeliest_scored(&row, temperature, (top_k as usize).max(TREE_CHILDREN));
            for pos in 0..200u64 {
                let drawn =
                    sample_token_coupled(&mut row.clone(), &sampling, &[], &mut ctx, &noise, pos);
                let drawable = &ranked[..(top_k as usize).min(ranked.len())];
                let want = if top_k > 0 {
                    followers_rated_by_the_shared_noise(drawable, drawn, temperature, &noise, pos)
                } else {
                    likeliest(&row, temperature, TREE_CHILDREN, Some(drawn))
                };
                assert_eq!(want[0].0, drawn);
                let what = format!("T {temperature}, top_k {top_k}, top_p {top_p}, pos {pos}");
                let followers = RowScores::Logits(&row).followers(
                    &sampling,
                    Some((&noise, pos)),
                    Vec::new,
                    &mut ctx,
                );
                // The far end draws among its top_k alone: a follower past it
                // is a row that can never be accepted.
                if top_k > 0 {
                    assert!(
                        followers
                            .iter()
                            .all(|(t, _)| drawable.iter().any(|c| c.token == *t)),
                        "a follower outside the top {top_k}: {what}"
                    );
                }
                assert_eq!(followers, want, "read whole: {what}");
                if top_k > 0 {
                    assert_eq!(
                        RowScores::Candidates {
                            likeliest: &ranked,
                            vocab
                        }
                        .followers(
                            &sampling,
                            Some((&noise, pos)),
                            Vec::new,
                            &mut ctx
                        ),
                        want,
                        "ranked as a card hands it over: {what}"
                    );
                }
                drawn_off_the_top += u32::from(drawn != ranked[0].token);
            }
        }
        assert!(
            drawn_off_the_top > 100,
            "{drawn_off_the_top} draws not the likeliest"
        );
        // Nothing finite: no followers, and nothing drawn.
        let none = vec![f32::NEG_INFINITY; 64];
        let sampling = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            ..Default::default()
        };
        assert!(RowScores::Logits(&none)
            .followers(&sampling, Some((&noise, 3)), Vec::new, &mut ctx)
            .is_empty());
    }

    /// A sampled row's children are rated by the chance the far end draws
    /// each. Where the drafter's logits err from the target's by a unit-scale
    /// Gumbel (about what Qwen2.5-Coder-1.5B does against the 7B, 2026-10-23),
    /// the far end's coupled draw `argmax(z/T + g)` is `argmax(y + T·g + ε)`, so
    /// by the Gumbel-max theorem `softmax(y + T·g)` IS that chance: what each
    /// rank claims must be how often it is the draw. The drafter's probability
    /// at T claimed the second child half again too likely and the tail half
    /// as likely as it is — a tree grew too few of the children that hold it.
    #[test]
    fn a_sampled_rows_children_are_rated_by_the_noise_the_far_end_draws_with() {
        use super::{followers_rated_by_the_shared_noise, TREE_CHILDREN};
        use crate::inference::coupled_noise::CoupledNoise;
        use crate::inference::sampling::RankedToken;
        let (noise, error, spread) = (
            CoupledNoise::new(11),
            CoupledNoise::new(12),
            CoupledNoise::new(13),
        );
        let vocab = 24u32;
        let trials = 10_000u64;
        for temperature in [0.7f32, 1.0, 1.3] {
            let t = f64::from(temperature);
            let (mut claimed, mut hit) = ([0f64; TREE_CHILDREN], [0f64; TREE_CHILDREN]);
            let mut the_noise_reordered = 0u32;
            for trial in 0..trials {
                let y: Vec<f64> = (0..vocab).map(|x| 1.5 * spread.gumbel(trial, x)).collect();
                let log_sum_t = {
                    let m = y.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    m / t + y.iter().map(|v| ((v - m) / t).exp()).sum::<f64>().ln()
                };
                let mut ranked: Vec<RankedToken> = (0..vocab)
                    .map(|x| RankedToken {
                        token: x,
                        logp: (y[x as usize] / t - log_sum_t) as f32,
                        logit: y[x as usize] as f32,
                    })
                    .collect();
                ranked.sort_by(|a, b| b.logit.total_cmp(&a.logit));
                let argmax = |score: &dyn Fn(u32) -> f64| {
                    (0..vocab)
                        .max_by(|&a, &b| score(a).total_cmp(&score(b)))
                        .unwrap()
                };
                let drawn = argmax(&|x| y[x as usize] + t * noise.gumbel(trial, x));
                let target = argmax(&|x| {
                    y[x as usize] + t * noise.gumbel(trial, x) + error.gumbel(trial, x)
                });
                let children =
                    followers_rated_by_the_shared_noise(&ranked, drawn, temperature, &noise, trial);
                assert_eq!(children.len(), TREE_CHILDREN);
                assert_eq!(children[0].0, drawn, "the far end's likeliest draw first");
                for (rank, &(token, logp)) in children.iter().enumerate() {
                    claimed[rank] += f64::from(logp).exp();
                    hit[rank] += f64::from(u8::from(token == target));
                }
                the_noise_reordered += u32::from(
                    children[1..]
                        .windows(2)
                        .any(|w| y[w[0].0 as usize] < y[w[1].0 as usize]),
                );
            }
            for rank in 0..TREE_CHILDREN {
                let (c, h) = (claimed[rank] / trials as f64, hit[rank] / trials as f64);
                assert!(
                    (c - h).abs() < 0.012 + 0.1 * h,
                    "T {temperature}, child {rank}: claims {c:.4}, is the draw {h:.4}"
                );
            }
            assert!(
                the_noise_reordered > 1000,
                "T {temperature}: the noise reordered {the_noise_reordered} rows' children"
            );
        }
    }

    /// The rows ranked as tensors (on a card in a node; here on the processor,
    /// by the same operations) give each row's followers in the order and at
    /// the log-probabilities the host's one-pass ranking does.
    #[test]
    fn rows_ranked_as_tensors_agree_with_the_hosts_ranking() {
        let vocab = 500usize;
        // Distinct values, so no tie can order two followers differently.
        let rows: Vec<Vec<f32>> = (0..5u32)
            .map(|r| {
                (0..vocab as u32)
                    .map(|i| ((i * 7919 + r * 104_729) % 1009) as f32 / 37.0 - 9.0)
                    .collect()
            })
            .collect();
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let logits =
            candle_core::Tensor::from_vec(flat, (rows.len(), vocab), &candle_core::Device::Cpu)
                .unwrap();
        for temperature in [0.0f32, 0.7, 1.3] {
            let ranked = likeliest_rows_where_they_are(&logits, temperature, 8)
                .unwrap()
                .expect("finite rows are ranked");
            assert_eq!(ranked.len(), rows.len());
            for (row, got) in rows.iter().zip(&ranked) {
                let want = likeliest(row, temperature, 8, None);
                assert_eq!(
                    got.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
                    want.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
                    "temperature {temperature}"
                );
                for (&(_, g), &(_, w)) in got.iter().zip(&want) {
                    assert!((g - w).abs() < 1e-4, "{g} against {w} at {temperature}");
                }
            }
        }
        // Fewer finite scores than followers asked: only the finite come back,
        // as the host offers only those.
        let mut short = vec![f32::NEG_INFINITY; vocab];
        short[3] = 1.0;
        short[9] = 2.0;
        let one =
            candle_core::Tensor::from_vec(short.clone(), (1, vocab), &candle_core::Device::Cpu)
                .unwrap();
        let ranked = likeliest_rows_where_they_are(&one, 0.0, 8)
            .unwrap()
            .unwrap();
        assert_eq!(
            ranked[0].iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            vec![9, 3]
        );
        assert_eq!(likeliest(&short, 0.0, 8, None).len(), 2);
        // A row with nothing finite is left to the host.
        let none = candle_core::Tensor::from_vec(
            vec![f32::NEG_INFINITY; vocab],
            (1, vocab),
            &candle_core::Device::Cpu,
        )
        .unwrap();
        assert!(likeliest_rows_where_they_are(&none, 0.0, 8)
            .unwrap()
            .is_none());
    }

    /// The card's two launches (`kernels/tree_rank.cu`) rank a level as the
    /// host's one-pass ranking does — the same followers in the same order,
    /// ties to the smaller id as the host breaks them, log-probabilities to
    /// 1e-4 — at a drafter's shapes: 8 rows of a Qwen vocabulary, the 65-row
    /// seed, Gemma's 262K, a tiny vocabulary, suppressed (−∞) tokens, fewer
    /// finite scores than followers. A NaN or a row with nothing finite is left
    /// to the host.
    ///
    /// ⚠ Needs a card: run `cargo test --features candle-cuda --lib
    /// the_card_ranks -- --nocapture` and look for no SKIPPED line.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_card_ranks_a_level_as_the_host_does() {
        use crate::inference::tree_rank::{ranked_on_the_card, CardRanking};
        use candle_core::{Device, Tensor};
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED tree-rank against the host: no CUDA device ({e})");
                return;
            }
        };
        let ranked = |rows: &[Vec<f32>], temperature: f32, k: usize| {
            let vocab = rows[0].len();
            let flat: Vec<f32> = rows.iter().flatten().copied().collect();
            let t = Tensor::from_vec(flat, (rows.len(), vocab), &dev).unwrap();
            ranked_on_the_card(&t, temperature, k).unwrap()
        };
        let check = |rows: &[Vec<f32>], what: &str| {
            for (temperature, k) in [(0.0f32, 8usize), (0.7, 8), (1.3, 3)] {
                let CardRanking::Ranked(got) = ranked(rows, temperature, k) else {
                    panic!("{what}: finite rows are ranked on the card");
                };
                assert_eq!(got.len(), rows.len(), "{what}");
                for (r, (row, got)) in rows.iter().zip(&got).enumerate() {
                    let want = likeliest(row, temperature, k, None);
                    assert_eq!(
                        got.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
                        want.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
                        "{what}: row {r} at temperature {temperature}"
                    );
                    // Against an f64 sum: the host's one f32 sum over a
                    // vocabulary drops every term below half its last bit.
                    let t = f64::from(if temperature > 0.0 { temperature } else { 1.0 });
                    let finite = || row.iter().filter(|x| x.is_finite()).map(|&x| f64::from(x));
                    let max = finite().fold(f64::NEG_INFINITY, f64::max);
                    let lse = max / t + finite().map(|x| ((x - max) / t).exp()).sum::<f64>().ln();
                    for &(id, g) in got {
                        let w = f64::from(row[id as usize]) / t - lse;
                        assert!((f64::from(g) - w).abs() < 1e-4, "{what}: {g} against {w}");
                    }
                }
            }
        };
        let random = |rows: usize, vocab: usize| -> Vec<Vec<f32>> {
            Tensor::randn(0f32, 3.0, (rows, vocab), &dev)
                .unwrap()
                .to_vec2()
                .unwrap()
        };
        check(&random(8, 151_936), "8 rows of a Qwen vocabulary");
        check(&random(65, 151_936), "the 65-row seed");
        check(&random(3, 262_144), "Gemma's vocabulary");
        check(&random(2, 500), "a tiny vocabulary");
        // Few distinct values: every follower ties with others.
        let tied: Vec<Vec<f32>> = (0..4u32)
            .map(|r| {
                (0..151_936u32)
                    .map(|i| ((i * 7919 + r * 104_729) % 1009) as f32 / 37.0 - 9.0)
                    .collect()
            })
            .collect();
        check(&tied, "tied scores");
        let mut suppressed = random(8, 151_936);
        for row in &mut suppressed {
            for x in row.iter_mut().step_by(3) {
                *x = f32::NEG_INFINITY;
            }
        }
        check(&suppressed, "suppressed tokens");
        let mut short = vec![vec![f32::NEG_INFINITY; 151_936]; 2];
        short[0][3] = 1.0;
        short[0][140_000] = 2.0;
        short[1][77] = -4.0;
        let CardRanking::Ranked(got) = ranked(&short, 0.0, 8) else {
            panic!("rows with a few finite scores are ranked");
        };
        assert_eq!(
            got[0].iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            [140_000, 3]
        );
        assert_eq!(got[1].iter().map(|&(t, _)| t).collect::<Vec<_>>(), [77]);
        check(&short, "a few finite scores");

        let mut nan = random(3, 151_936);
        nan[1][9_000] = f32::NAN;
        assert!(matches!(ranked(&nan, 0.0, 8), CardRanking::NotFinite));
        let mut inf = random(3, 151_936);
        inf[2][5] = f32::INFINITY;
        assert!(matches!(ranked(&inf, 0.0, 8), CardRanking::NotFinite));
        let empty = vec![vec![f32::NEG_INFINITY; 151_936]];
        assert!(matches!(ranked(&empty, 0.0, 8), CardRanking::NotFinite));
        // More followers than the kernel keeps: composed instead.
        assert!(matches!(
            ranked(&random(1, 500), 0.0, 9),
            CardRanking::NotTaken
        ));
    }

    /// The card's top k for a sampled row (`tree_rank::top_k_on_the_card`,
    /// `kernels/tree_rank.cu`'s radix select) is the host's
    /// (`likeliest_scored`): the same scores in the same order, each the raw
    /// score of its token, log-probabilities to 1e-4 of an f64 sum, and the
    /// same tokens wherever no tie sits at the k-th place (a tie there is
    /// broken either way, as the host's own selection breaks it) — at a
    /// drafter's shapes, with suppressed tokens and too few finite scores. A
    /// NaN or nothing finite is left to the host, and so is a `k` the kernel
    /// does not keep. Then the end of it: a level's followers, drawn on the
    /// card's ranking, are the followers drawn from the rows read whole.
    ///
    /// ⚠ Needs a card: run `cargo test --features candle-cuda --lib
    /// the_card_ranks_a_sampled -- --nocapture` and look for no SKIPPED line.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_card_ranks_a_sampled_rows_top_k_as_the_host_does() {
        use super::{likeliest_scored, RowScores, StepScores};
        use crate::inference::coupled_noise::CoupledNoise;
        use crate::inference::sampling::SamplingContext;
        use crate::inference::tree_rank::{top_k_on_the_card, CardTopK, MOST_CANDIDATES};
        use crate::types::SamplingParams;
        use candle_core::{Device, Tensor};
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED the card's top k against the host: no CUDA device ({e})");
                return;
            }
        };
        let tensor = |rows: &[Vec<f32>]| {
            let vocab = rows[0].len();
            let flat: Vec<f32> = rows.iter().flatten().copied().collect();
            Tensor::from_vec(flat, (rows.len(), vocab), &dev).unwrap()
        };
        let top = |rows: &[Vec<f32>], temperature: f32, k: usize| {
            top_k_on_the_card(&tensor(rows), temperature, k).unwrap()
        };
        let check = |rows: &[Vec<f32>], what: &str| {
            for (temperature, k) in [
                (0.7f32, 40usize),
                (1.0, MOST_CANDIDATES),
                (1.3, 8),
                (0.7, 1),
            ] {
                let CardTopK::Rows(got) = top(rows, temperature, k) else {
                    panic!("{what}: finite rows are ranked on the card");
                };
                assert_eq!(got.len(), rows.len(), "{what}");
                for (r, (row, got)) in rows.iter().zip(&got).enumerate() {
                    let want = likeliest_scored(row, temperature, k);
                    let what = format!("{what}, row {r}, T {temperature}, k {k}");
                    assert_eq!(got.len(), want.len(), "{what}");
                    assert_eq!(
                        got.iter().map(|c| c.logit).collect::<Vec<_>>(),
                        want.iter().map(|c| c.logit).collect::<Vec<_>>(),
                        "{what}: the same scores, likeliest first"
                    );
                    let mut seen = std::collections::HashSet::new();
                    for (j, c) in got.iter().enumerate() {
                        assert_eq!(
                            c.logit, row[c.token as usize],
                            "{what}: a token's own score"
                        );
                        assert!(seen.insert(c.token), "{what}: {} twice", c.token);
                        if j > 0 && got[j - 1].logit == c.logit {
                            assert!(
                                got[j - 1].token < c.token,
                                "{what}: a tie to the smaller id"
                            );
                        }
                    }
                    let kth = want.last().map(|c| c.logit);
                    let tie_at_the_edge = want.len() == k
                        && row.iter().filter(|&&x| Some(x) == kth).count()
                            > want.iter().filter(|c| Some(c.logit) == kth).count();
                    if !tie_at_the_edge {
                        assert_eq!(
                            got.iter().map(|c| c.token).collect::<Vec<_>>(),
                            want.iter().map(|c| c.token).collect::<Vec<_>>(),
                            "{what}: the same tokens"
                        );
                    }
                    let t = f64::from(temperature);
                    let finite = || row.iter().filter(|x| x.is_finite()).map(|&x| f64::from(x));
                    let max = finite().fold(f64::NEG_INFINITY, f64::max);
                    let lse = max / t + finite().map(|x| ((x - max) / t).exp()).sum::<f64>().ln();
                    for c in got {
                        let w = f64::from(c.logit) / t - lse;
                        assert!(
                            (f64::from(c.logp) - w).abs() < 1e-4,
                            "{what}: {} against {w}",
                            c.logp
                        );
                    }
                }
            }
        };
        let random = |rows: usize, vocab: usize| -> Vec<Vec<f32>> {
            Tensor::randn(0f32, 3.0, (rows, vocab), &dev)
                .unwrap()
                .to_vec2()
                .unwrap()
        };
        check(&random(8, 151_936), "8 rows of a Qwen vocabulary");
        check(&random(65, 151_936), "the 65-row seed");
        check(&random(2, 262_208), "Gemma 3's vocabulary");
        check(&random(3, 32_000), "a Llama 2 vocabulary");
        check(&random(2, 500), "a tiny vocabulary");
        let tied: Vec<Vec<f32>> = (0..4u32)
            .map(|r| {
                (0..151_936u32)
                    .map(|i| ((i * 7919 + r * 104_729) % 1009) as f32 / 37.0 - 9.0)
                    .collect()
            })
            .collect();
        check(&tied, "tied scores");
        let mut suppressed = random(8, 151_936);
        for row in &mut suppressed {
            for x in row.iter_mut().step_by(3) {
                *x = f32::NEG_INFINITY;
            }
        }
        check(&suppressed, "suppressed tokens");
        let mut short = vec![vec![f32::NEG_INFINITY; 151_936]; 2];
        short[0][3] = 1.0;
        short[0][140_000] = 2.0;
        short[1][77] = -4.0;
        check(&short, "a few finite scores");
        let CardTopK::Rows(got) = top(&short, 0.7, 40) else {
            panic!("rows with a few finite scores are ranked");
        };
        assert_eq!(
            got[0].iter().map(|c| c.token).collect::<Vec<_>>(),
            [140_000, 3]
        );
        assert_eq!(got[1].iter().map(|c| c.token).collect::<Vec<_>>(), [77]);

        let mut nan = random(3, 151_936);
        nan[1][9_000] = f32::NAN;
        assert!(matches!(top(&nan, 0.7, 40), CardTopK::NotFinite));
        let mut inf = random(3, 151_936);
        inf[2][5] = f32::INFINITY;
        assert!(matches!(top(&inf, 0.7, 40), CardTopK::NotFinite));
        let empty = vec![vec![f32::NEG_INFINITY; 151_936]];
        assert!(matches!(top(&empty, 0.7, 40), CardTopK::NotFinite));
        assert!(matches!(
            top(&random(1, 500), 0.7, MOST_CANDIDATES + 1),
            CardTopK::NotTaken
        ));
        assert!(matches!(top(&random(1, 40), 0.7, 40), CardTopK::NotTaken));

        // A level read for a coupled draw: the card's rows give the followers
        // the rows read whole give.
        let rows = random(8, 151_936);
        let logits = tensor(&rows);
        let noise = CoupledNoise::new(0xD1CE);
        let mut ctx = SamplingContext::new(151_936);
        for (top_k, top_p) in [(40u32, 1.0f32), (40, 0.9), (5, 1.0)] {
            let sampling = SamplingParams {
                temperature: 0.7,
                top_k,
                top_p,
                ..Default::default()
            };
            let read = StepScores::read(&logits, &sampling, true).unwrap();
            assert!(
                matches!(read, StepScores::Candidates { .. }),
                "a sampled level is ranked on the card (top_k {top_k})"
            );
            for (i, row) in rows.iter().enumerate() {
                for pos in 0..20u64 {
                    let card =
                        read.row(i)
                            .followers(&sampling, Some((&noise, pos)), Vec::new, &mut ctx);
                    let host = RowScores::Logits(row).followers(
                        &sampling,
                        Some((&noise, pos)),
                        Vec::new,
                        &mut ctx,
                    );
                    let what = format!("row {i}, top_k {top_k}, top_p {top_p}, pos {pos}");
                    // The same draw and followers; the log-probabilities as
                    // close as the host's one f32 sum over the vocabulary
                    // allows (it drops every term below half its last bit).
                    assert_eq!(
                        card.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
                        host.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
                        "{what}"
                    );
                    for (&(_, a), &(_, b)) in card.iter().zip(&host) {
                        assert!((a - b).abs() < 1e-3, "{what}: {a} against {b}");
                    }
                }
            }
        }
        let whole = SamplingParams {
            temperature: 0.7,
            top_k: 0,
            ..Default::default()
        };
        assert!(
            matches!(
                StepScores::read(&logits, &whole, true).unwrap(),
                StepScores::Logits(_)
            ),
            "a draw over the whole row reads the rows whole"
        );

        // A chain's guess, one row at a time: the whole row's draw and its
        // probability.
        let sampling = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            ..Default::default()
        };
        for (i, row) in rows.iter().enumerate() {
            let one = logits.narrow(0, i, 1).unwrap();
            for pos in 0..20u64 {
                let (t, p) =
                    super::coupled_guess_on_the_card(&one, &sampling, &mut ctx, &noise, pos)
                        .unwrap()
                        .expect("a sampled guess is drawn on the card");
                let want = crate::inference::sampling::sample_token_coupled(
                    &mut row.clone(),
                    &sampling,
                    &[],
                    &mut ctx,
                    &noise,
                    pos,
                );
                assert_eq!(t, want, "row {i}, pos {pos}");
                let q = super::probability_at(row, t as usize, 0.7);
                assert!(
                    (p - q).abs() <= 1e-3 * q + 1e-7,
                    "row {i}, pos {pos}: {p} against {q}"
                );
            }
        }
        let first_row = logits.narrow(0, 0, 1).unwrap();
        assert_eq!(
            super::coupled_guess_on_the_card(&first_row, &whole, &mut ctx, &noise, 0).unwrap(),
            None,
            "a draw over the whole row reads it whole"
        );
    }

    #[test]
    fn a_guess_probability_is_the_softmax_at_the_requests_temperature() {
        let logits = [2.0f32, 1.0, 0.0];
        let p = probability_at(&logits, 0, 0.0);
        let e = [2.0f64.exp(), 1.0f64.exp(), 1.0];
        assert!((f64::from(p) - e[0] / (e[0] + e[1] + e[2])).abs() < 1e-6);
        // A higher temperature flattens it.
        assert!(probability_at(&logits, 0, 2.0) < p);
        assert_eq!(probability_at(&[f32::NEG_INFINITY; 3], 0, 1.0), 0.0);
    }
}
