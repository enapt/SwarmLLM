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
const TREE_CHILDREN: usize = 8;

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

/// A guess waiting for a place in the tree, rated by its path's log-probability.
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
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let finite = logits.iter().copied().filter(|x| x.is_finite());
    let max = finite.clone().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Vec::new();
    }
    let log_sum = max / t + finite.map(|x| ((x - max) / t).exp()).sum::<f32>().ln();
    let logp = |i: usize| logits[i] / t - log_sum;
    let mut best: Vec<usize> = Vec::with_capacity(k + 1);
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

/// Each row's followers, likeliest first: (token, log-probability).
type RankedRows = Vec<Vec<(u32, f32)>>;

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
/// (`torch.topk`). `k` picks of the row's maximum, each written over with −∞,
/// reproduce [`likeliest`]'s order; ties may break differently.
fn likeliest_rows_where_they_are(
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

/// A row's scores as a tree step hands them to its queue: the logits, ranked
/// on the host (with the coupled draw where the request samples), or the
/// followers the card already ranked ([`likeliest_rows_where_they_are`]).
enum RowScores<'a> {
    Logits(&'a [f32]),
    Ranked(&'a [(u32, f32)]),
}

/// Each row of a tree step's `[rows, vocab]` logits for its queue: ranked on
/// the card where nothing needs the whole row, else read back whole.
enum StepScores {
    Logits(Vec<Vec<f32>>),
    Ranked(RankedRows),
}

impl StepScores {
    fn read(
        logits: &candle_core::Tensor,
        temperature: f32,
        coupled: bool,
    ) -> Result<Self, SwarmError> {
        // A coupled draw reads the whole row (`sample_token_coupled`).
        // `SWARMLLM_TREE_RANK=host` ranks every row on the host, as before.
        if !coupled
            && logits.device().is_cuda()
            && std::env::var("SWARMLLM_TREE_RANK").as_deref() != Ok("host")
        {
            if let Some(ranked) = likeliest_rows_where_they_are(logits, temperature, TREE_CHILDREN)?
            {
                return Ok(Self::Ranked(ranked));
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
        }
    }

    fn row(&self, i: usize) -> RowScores<'_> {
        match self {
            Self::Logits(rows) => RowScores::Logits(&rows[i]),
            Self::Ranked(rows) => RowScores::Ranked(&rows[i]),
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
            );
        }
        let mut history = history.to_vec();
        let mut ctx = crate::inference::sampling::SamplingContext::new(0);
        let mut guesses = Vec::with_capacity(gamma);
        for j in 0..gamma {
            let mut row: Vec<f32> = logits
                .flatten_all()
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec1())
                .map_err(SwarmError::internal)?;
            // Read before sampling, which may rewrite the row in place.
            let unsure = stop_below.map(|floor| (row.clone(), floor));
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
            guesses.push(t);
            history.push(t);
            if let Some((raw, floor)) = unsure {
                if probability_at(&raw, t as usize, sampling.temperature) < floor {
                    break;
                }
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
            .and_then(|t| StepScores::read(&t, sampling.temperature, coupled))?;
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
            let followers = match scores {
                RowScores::Ranked(ranked) => ranked.to_vec(),
                RowScores::Logits(logits) => {
                    let drawn = noise.map(|noise| {
                        let mut seen = history.to_vec();
                        seen.extend(path_tokens(tree, tokens, parent));
                        let position = base + parent.map_or(0, |p| tree.depth(p) + 1);
                        let mut row = logits.to_vec();
                        crate::inference::sampling::sample_token_coupled(
                            &mut row,
                            sampling,
                            &seen,
                            &mut ctx,
                            noise,
                            position as u64,
                        )
                    });
                    likeliest(logits, sampling.temperature, TREE_CHILDREN, drawn)
                }
            };
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
                    .and_then(|t| StepScores::read(&t, sampling.temperature, coupled))?;
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
                    .and_then(|t| StepScores::read(&t, sampling.temperature, coupled))?;
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
                logits = self.forward(&next, first + j, kv, request_id)?;
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
