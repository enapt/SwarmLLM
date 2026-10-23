//! A drafter's tree level ranked on a card in two launches
//! (`kernels/tree_rank.cu`): each row's log-sum-exp and its eight likeliest
//! tokens, read back in ONE copy.
//!
//! `split::draft::likeliest_rows_where_they_are` asks [`ranked_on_the_card`]
//! first and composes only where it answers [`CardRanking::NotTaken`]. The
//! composition is about thirty launches and three waits on the card a level,
//! and its eight argmax launches gave a row one block each — 134 µs apiece and
//! 21% of the card's time in a 0.5B drafter's three-level call (nsys,
//! 2026-10-22). The kernel reads each row once with up to 64 blocks.
//!
//! NOT bit-identical to the composition (the log-sum-exp sums in another
//! order; a tie goes to the smaller id, as the host's ranking breaks it). The
//! ranking decides only what is guessed, never the reply. A/B
//! `SWARMLLM_TREE_TOPK=composed`.
//!
//! A SAMPLED reply's level is ranked by the second pair
//! ([`top_k_on_the_card`]): each row's request `top_k` likeliest (≤
//! [`MOST_CANDIDATES`]) with their raw scores, which is all its coupled draw
//! reads (`sampling::coupled_draw_reads_only_top_k`) — the rows were read back
//! whole before (2026-10-23). A/B `SWARMLLM_TREE_COUPLED_DRAW=host`.

use candle_core::{DType, Device, Result, Tensor};

/// The most followers the kernel ranks a row (`TOPK` in the kernel).
pub(crate) const MOST_FOLLOWERS: usize = 8;
/// What the kernel writes a row: the log-sum-exp, then `MOST_FOLLOWERS`
/// log-probabilities, then their ids as exact floats.
const OUT_WIDTH: usize = 1 + 2 * MOST_FOLLOWERS;
/// What pass 1 writes a chunk: max, sum, flag, values, ids.
const PARTIAL_WIDTH: usize = 3 + 2 * MOST_FOLLOWERS;
const PASS1_THREADS: u32 = 256;
/// Pass 2 gives each chunk a thread, so no row is cut into more chunks.
const MOST_CHUNKS: usize = 64;
/// A row is cut into chunks of at least this many scores: fewer and a block's
/// 256 threads would each read only a handful.
const LEAST_CHUNK: usize = 2048;
/// Ids go back as exact floats.
const WIDEST_VOCAB: usize = 1 << 24;

/// PTX for the ranking kernels, compiled from `kernels/tree_rank.cu` by
/// `build.rs` and loaded through candle's `get_or_load_custom_func`.
const TREE_RANK_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/tree_rank.ptx"));

/// `SWARMLLM_TREE_TOPK=composed` ranks with the composed candle ops, so the
/// kernel can be A/B'd inside ONE binary. Read once: it sits on every level.
fn fused() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_TREE_TOPK").as_deref() != Ok("composed"))
}

/// What the card made of a level's rows.
pub(crate) enum CardRanking {
    /// The kernel does not take these logits — not on a card, more followers
    /// asked than it keeps, a vocabulary too wide for exact float ids, or
    /// switched off. The caller composes.
    NotTaken,
    /// A row held a NaN or +inf, or nothing finite: the host ranks the level,
    /// as it did when the composition found a sum that was not finite.
    NotFinite,
    /// Each row's followers, likeliest first, as (token, log-probability):
    /// only finite scores, as the host offers only those.
    Ranked(Vec<Vec<(u32, f32)>>),
}

/// Each row of `logits` (`[rows, vocab]`) ranked at `temperature` (1 for a
/// greedy request): its `k` likeliest followers, `k <= MOST_FOLLOWERS`.
pub(crate) fn ranked_on_the_card(
    logits: &Tensor,
    temperature: f32,
    k: usize,
) -> Result<CardRanking> {
    if !fused() || !matches!(logits.device(), Device::Cuda(_)) || k > MOST_FOLLOWERS {
        return Ok(CardRanking::NotTaken);
    }
    let Ok((rows, vocab)) = logits.dims2() else {
        return Ok(CardRanking::NotTaken);
    };
    if rows == 0 || vocab == 0 || vocab >= WIDEST_VOCAB || rows > usize::from(u16::MAX) {
        return Ok(CardRanking::NotTaken);
    }
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    // The composition's `scaled / t` is candle's `affine(1 / t, 0)`.
    let inv_t = (1.0 / f64::from(t)) as f32;
    let scores = logits.to_dtype(DType::F32)?.contiguous()?;
    let out: Vec<Vec<f32>> = scores.apply_op1_no_bwd(&TreeRank { inv_t })?.to_vec2()?;
    let mut ranked = Vec::with_capacity(rows);
    for row in out {
        if !row[0].is_finite() {
            return Ok(CardRanking::NotFinite);
        }
        ranked.push(
            (0..k)
                .map(|j| (row[1 + MOST_FOLLOWERS + j] as u32, row[1 + j]))
                .take_while(|&(_, logp)| logp.is_finite())
                .collect(),
        );
    }
    Ok(CardRanking::Ranked(ranked))
}

struct TreeRank {
    inv_t: f32,
}

impl candle_core::CustomOp1 for TreeRank {
    fn name(&self) -> &'static str {
        "tree-rank"
    }

    /// Never reached: [`ranked_on_the_card`] hands only card tensors to this
    /// op, and every other device composes.
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!(
            "tree-rank is card-only; likeliest_rows_where_they_are composes elsewhere"
        )
    }

    /// Two launches: every chunk of every row (`tree_rank_partials`), then a
    /// row's chunks merged (`tree_rank_merge`). Both outputs come from
    /// `alloc_fully_overwritten`: pass 1 writes every chunk's whole partial
    /// (an empty chunk writes an empty one), pass 2 every row's whole answer.
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device.clone();
        let Some((start, end)) = l.contiguous_offsets() else {
            candle_core::bail!("tree-rank: logits must be contiguous");
        };
        let &[rows, vocab] = l.shape().dims() else {
            candle_core::bail!("tree-rank: logits must be [rows, vocab]");
        };
        if end - start != rows * vocab {
            candle_core::bail!("tree-rank: {} scores for {rows} x {vocab}", end - start);
        }
        let logits = s.as_cuda_slice::<f32>()?.slice(start..end);
        let chunks = vocab.div_ceil(LEAST_CHUNK).clamp(1, MOST_CHUNKS);
        let mut partials = dev.alloc_fully_overwritten::<f32>(rows * chunks * PARTIAL_WIDTH)?;
        let mut out = dev.alloc_fully_overwritten::<f32>(rows * OUT_WIDTH)?;
        let (vocab_i, chunks_i) = (vocab as i32, chunks as i32);

        let partial =
            dev.get_or_load_custom_func("tree_rank_partials", "swarmllm_tree_rank", TREE_RANK_PTX)?;
        let mut builder = partial.builder();
        builder.arg(&logits);
        builder.arg(&mut partials);
        builder.arg(&vocab_i);
        builder.arg(&chunks_i);
        builder.arg(&self.inv_t);
        let cfg = LaunchConfig {
            grid_dim: (chunks as u32, rows as u32, 1),
            block_dim: (PASS1_THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: ffi. The logits are `rows * vocab` contiguous f32 (checked
        // above); the kernel reads only those and writes `rows * chunks *
        // PARTIAL_WIDTH` floats of `partials`, allocated at that length.
        // `ranked_on_the_card` bounds `rows` by the grid's y limit and `vocab`
        // below 2^24, so both fit i32.
        unsafe { builder.launch(cfg) }.w()?;

        let merge =
            dev.get_or_load_custom_func("tree_rank_merge", "swarmllm_tree_rank", TREE_RANK_PTX)?;
        let mut builder = merge.builder();
        builder.arg(&partials);
        builder.arg(&mut out);
        builder.arg(&chunks_i);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (MOST_CHUNKS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: ffi. Reads the partials pass 1 wrote (`chunks <=
        // MOST_CHUNKS`, the block's width), writes `rows * OUT_WIDTH` floats of
        // `out`, allocated at that length. One stream: pass 1 has finished.
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, dev),
            candle_core::Shape::from((rows, OUT_WIDTH)),
        ))
    }
}

/// The most of a row's likeliest tokens [`top_k_on_the_card`] hands back
/// (`KMAX` in the kernel) — above the API's default `top_k` of 40.
pub(crate) const MOST_CANDIDATES: usize = 64;
/// What `tree_topk_merge` writes a row: the log-sum-exp, then
/// `MOST_CANDIDATES` log-probabilities, ids as exact floats and raw scores.
const TOPK_OUT_WIDTH: usize = 1 + 3 * MOST_CANDIDATES;
/// What `tree_topk_partials` writes a chunk: max, sum, flag, values, ids.
const TOPK_PARTIAL_WIDTH: usize = 3 + 2 * MOST_CANDIDATES;
const TOPK_THREADS: usize = 256;
/// A pass-1 chunk: each of the block's threads keeps 16 scores in registers.
const TOPK_CHUNK: usize = TOPK_THREADS * 16;
/// Pass 2's candidates, every chunk's k: 32 a thread.
const TOPK_MERGE_MOST: usize = TOPK_THREADS * 32;

/// `SWARMLLM_TREE_COUPLED_DRAW=host` reads a sampled reply's rows back whole
/// and draws on the host, as before 2026-10-23 — the A/B of
/// [`top_k_on_the_card`] inside one binary. Read once: it sits on every level.
fn top_k_on_card_wanted() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_TREE_COUPLED_DRAW").as_deref() != Ok("host"))
}

/// What the card made of a level's rows for a coupled draw.
pub(crate) enum CardTopK {
    /// The kernels do not take these logits — not on a card, `k` past
    /// [`MOST_CANDIDATES`] or not below the vocabulary, a vocabulary too wide,
    /// or switched off. The caller reads the rows whole.
    NotTaken,
    /// A row held a NaN or +inf, or nothing finite: the host reads the level.
    NotFinite,
    /// Each row's `k` likeliest at the temperature, best first (a tie to the
    /// smaller id), finite scores only.
    Rows(Vec<Vec<crate::inference::sampling::RankedToken>>),
}

/// Each row of `logits` (`[rows, vocab]`): its `k` likeliest tokens at
/// `temperature`, with their log-probabilities and RAW scores — what a coupled
/// draw among the row's top k reads (`sampling::sample_coupled_from_top_k`),
/// in two launches and one copy back of `rows × 772` bytes instead of the rows.
pub(crate) fn top_k_on_the_card(logits: &Tensor, temperature: f32, k: usize) -> Result<CardTopK> {
    if !top_k_on_card_wanted()
        || !matches!(logits.device(), Device::Cuda(_))
        || k == 0
        || k > MOST_CANDIDATES
    {
        return Ok(CardTopK::NotTaken);
    }
    let Ok((rows, vocab)) = logits.dims2() else {
        return Ok(CardTopK::NotTaken);
    };
    let chunks = vocab.div_ceil(TOPK_CHUNK).max(1);
    if rows == 0
        || vocab <= k
        || vocab >= WIDEST_VOCAB
        || rows > usize::from(u16::MAX)
        || chunks > TOPK_THREADS
        || chunks * k > TOPK_MERGE_MOST
    {
        return Ok(CardTopK::NotTaken);
    }
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    let inv_t = (1.0 / f64::from(t)) as f32;
    let scores = logits.to_dtype(DType::F32)?.contiguous()?;
    let out: Vec<Vec<f32>> = scores
        .apply_op1_no_bwd(&TreeTopK {
            inv_t,
            k: k as i32,
            chunks,
        })?
        .to_vec2()?;
    let mut ranked = Vec::with_capacity(rows);
    for row in out {
        if !row[0].is_finite() {
            return Ok(CardTopK::NotFinite);
        }
        ranked.push(
            (0..k)
                .map(|j| crate::inference::sampling::RankedToken {
                    token: row[1 + MOST_CANDIDATES + j] as u32,
                    logp: row[1 + j],
                    logit: row[1 + 2 * MOST_CANDIDATES + j],
                })
                .take_while(|c| c.logp.is_finite())
                .collect(),
        );
    }
    Ok(CardTopK::Rows(ranked))
}

struct TreeTopK {
    inv_t: f32,
    k: i32,
    chunks: usize,
}

impl candle_core::CustomOp1 for TreeTopK {
    fn name(&self) -> &'static str {
        "tree-top-k"
    }

    /// Never reached: [`top_k_on_the_card`] hands only card tensors to this op.
    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &candle_core::Layout,
    ) -> Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("tree-top-k is card-only; the host reads the rows whole elsewhere")
    }

    /// Two launches: every chunk of every row (`tree_topk_partials`), then a
    /// row's chunks merged (`tree_topk_merge`), which also reads each kept
    /// token's raw score. Both outputs are written whole: pass 1 every chunk's
    /// partial (empty places as -inf), pass 2 every row's answer.
    fn cuda_fwd(
        &self,
        s: &candle_core::CudaStorage,
        l: &candle_core::Layout,
    ) -> Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s.device.clone();
        let Some((start, end)) = l.contiguous_offsets() else {
            candle_core::bail!("tree-top-k: logits must be contiguous");
        };
        let &[rows, vocab] = l.shape().dims() else {
            candle_core::bail!("tree-top-k: logits must be [rows, vocab]");
        };
        if end - start != rows * vocab {
            candle_core::bail!("tree-top-k: {} scores for {rows} x {vocab}", end - start);
        }
        let chunks = self.chunks;
        if chunks == 0
            || chunks > TOPK_THREADS
            || vocab.div_ceil(chunks) > TOPK_CHUNK
            || chunks * self.k as usize > TOPK_MERGE_MOST
        {
            candle_core::bail!("tree-top-k: {chunks} chunks do not fit {vocab} scores");
        }
        let logits = s.as_cuda_slice::<f32>()?.slice(start..end);
        let mut partials =
            dev.alloc_fully_overwritten::<f32>(rows * chunks * TOPK_PARTIAL_WIDTH)?;
        let mut out = dev.alloc_fully_overwritten::<f32>(rows * TOPK_OUT_WIDTH)?;
        let (vocab_i, chunks_i) = (vocab as i32, chunks as i32);

        let partial =
            dev.get_or_load_custom_func("tree_topk_partials", "swarmllm_tree_rank", TREE_RANK_PTX)?;
        let mut builder = partial.builder();
        builder.arg(&logits);
        builder.arg(&mut partials);
        builder.arg(&vocab_i);
        builder.arg(&chunks_i);
        builder.arg(&self.inv_t);
        builder.arg(&self.k);
        let cfg = LaunchConfig {
            grid_dim: (chunks as u32, rows as u32, 1),
            block_dim: (TOPK_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: ffi. The logits are `rows * vocab` contiguous f32 (checked
        // above); a block reads one chunk of `ceil(vocab / chunks) <=
        // TOPK_CHUNK` scores, the most its threads hold, and writes its
        // `TOPK_PARTIAL_WIDTH` floats of `partials`, allocated at `rows *
        // chunks` of them. `k <= MOST_CANDIDATES` (`top_k_on_the_card`).
        unsafe { builder.launch(cfg) }.w()?;

        let merge =
            dev.get_or_load_custom_func("tree_topk_merge", "swarmllm_tree_rank", TREE_RANK_PTX)?;
        let mut builder = merge.builder();
        builder.arg(&partials);
        builder.arg(&logits);
        builder.arg(&mut out);
        builder.arg(&vocab_i);
        builder.arg(&chunks_i);
        builder.arg(&self.k);
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (TOPK_THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: ffi. Reads the partials pass 1 wrote (`chunks <=
        // TOPK_THREADS`, one thread a chunk's sums; `chunks * k <=
        // TOPK_MERGE_MOST`, the candidates its threads hold) and one raw score
        // a kept id (< vocab, from pass 1), writes `rows * TOPK_OUT_WIDTH`
        // floats of `out`, allocated at that length. One stream: pass 1 has
        // finished.
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, dev),
            candle_core::Shape::from((rows, TOPK_OUT_WIDTH)),
        ))
    }
}

/// `sampling::draw_each_row` for a tree that flows with the request's noise,
/// its rows still on the card (`[rows, vocab]`): each row's draw reads only its
/// `top_k` (`sampling::coupled_draw_reads_only_top_k`), so the card ranks those
/// ([`top_k_on_the_card`]) and the host draws from them — the same draws, for
/// 772 bytes a row instead of the row (608 KB at 152K, copied and scanned three
/// times). `None` where the kernels do not take the rows: read them whole.
///
/// The host path refuses a row with any non-finite score; so does this, by one
/// summed scalar that any NaN or ±inf poisons, as the greedy walk on the card
/// checks.
pub(crate) fn draw_each_row_on_the_card(
    logits: &Tensor,
    tree: &crate::inference::draft_tree::DraftTree,
    read_before: usize,
    params: &crate::types::SamplingParams,
    coupling: (&crate::inference::coupled_noise::CoupledNoise, u64),
) -> std::result::Result<Option<Vec<u32>>, String> {
    let fail = |e: candle_core::Error| format!("spec draws on the card: {e}");
    let (_, vocab) = logits.dims2().map_err(fail)?;
    let Some(k) = crate::inference::sampling::coupled_draw_reads_only_top_k(params, vocab) else {
        return Ok(None);
    };
    let total: f32 = logits
        .sum_all()
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.to_scalar())
        .map_err(fail)?;
    if !total.is_finite() {
        return Err("spec draws: non-finite logits".into());
    }
    let rows = match top_k_on_the_card(logits, params.temperature, k) {
        Ok(CardTopK::Rows(rows)) => rows,
        Ok(CardTopK::NotFinite) => return Err("spec draws: non-finite logits".into()),
        Ok(CardTopK::NotTaken) => return Ok(None),
        // A kernel that will not load or launch costs the draws their speed,
        // never the reply: the rows are read whole.
        Err(e) => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    error = %e,
                    "the card's top-k for a sampled tree failed; reading its rows whole"
                );
            });
            return Ok(None);
        }
    };
    let (noise, first) = coupling;
    let mut ctx = crate::inference::sampling::SamplingContext::new(0);
    crate::inference::sampling::draws_of_each_row(tree, read_before, &rows, |likeliest, depth| {
        if likeliest.is_empty() {
            return Err("spec draws: non-finite logits".to_string());
        }
        Ok(crate::inference::sampling::sample_coupled_from_top_k(
            likeliest,
            params,
            &mut ctx,
            noise,
            first + depth,
            vocab,
        ))
    })
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::draw_each_row_on_the_card;
    use crate::inference::coupled_noise::CoupledNoise;
    use crate::inference::draft_tree::DraftTree;
    use crate::types::SamplingParams;
    use candle_core::{Device, Tensor};

    /// A tree that flows, drawn on the card (`draw_each_row_on_the_card`), gets
    /// the draws `sampling::draw_each_row` makes from its rows read whole —
    /// each row at its own depth's noise, rows read in an earlier forward
    /// skipped — and refuses what that refuses; a draw that reads the whole row
    /// is left to it.
    ///
    /// ⚠ Needs a card: run `cargo test --features candle-cuda --lib
    /// a_flowing_trees_draws -- --nocapture` and look for no SKIPPED line.
    #[test]
    fn a_flowing_trees_draws_on_the_card_are_its_draws_from_whole_rows() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED the card's flowing draws: no CUDA device ({e})");
                return;
            }
        };
        let vocab = 151_936;
        // Two rows read in an earlier forward, then four in this one.
        let tree =
            DraftTree::new(40, vec![None, Some(0), Some(0), Some(1), Some(3), Some(2)]).unwrap();
        let (read_before, rows) = (2, 4);
        let logits = Tensor::randn(0f32, 3.0, (rows, vocab), &dev).unwrap();
        let whole: Vec<Vec<f32>> = logits.to_vec2().unwrap();
        let noise = CoupledNoise::new(0xF10);
        let mut drawn_off_the_top = 0;
        for (top_k, top_p) in [(40u32, 1.0f32), (40, 0.9), (8, 1.0), (64, 0.95)] {
            let params = SamplingParams {
                temperature: 0.7,
                top_k,
                top_p,
                ..Default::default()
            };
            for first in [41u64, 1_000, 77_777] {
                let want = crate::inference::sampling::draw_each_row(
                    &tree,
                    read_before,
                    &whole,
                    &params,
                    Some((&noise, first)),
                )
                .unwrap();
                let got = draw_each_row_on_the_card(
                    &logits,
                    &tree,
                    read_before,
                    &params,
                    (&noise, first),
                )
                .unwrap()
                .expect("a sampled flowing tree is drawn on the card");
                assert_eq!(got, want, "top_k {top_k}, top_p {top_p}, from {first}");
                for (row, &t) in whole.iter().zip(&got) {
                    let best = (0..vocab)
                        .max_by(|&a, &b| row[a].total_cmp(&row[b]))
                        .unwrap();
                    drawn_off_the_top += usize::from(t as usize != best);
                }
            }
        }
        assert!(
            drawn_off_the_top > 10,
            "{drawn_off_the_top} draws not the likeliest"
        );

        let whole_row = SamplingParams {
            temperature: 0.7,
            top_k: 0,
            ..Default::default()
        };
        assert_eq!(
            draw_each_row_on_the_card(&logits, &tree, read_before, &whole_row, (&noise, 41)),
            Ok(None),
            "a draw over the whole row reads the rows whole"
        );
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 40,
            ..Default::default()
        };
        let mut poisoned = whole.clone();
        poisoned[3][123] = f32::NEG_INFINITY;
        let flat: Vec<f32> = poisoned.into_iter().flatten().collect();
        let poisoned = Tensor::from_vec(flat, (rows, vocab), &dev).unwrap();
        assert!(
            draw_each_row_on_the_card(&poisoned, &tree, read_before, &params, (&noise, 41))
                .is_err(),
            "a non-finite score is refused, as the host refuses it"
        );
        assert!(
            draw_each_row_on_the_card(&logits, &tree, 1, &params, (&noise, 41)).is_err(),
            "rows that are not the tree's last are refused"
        );
    }
}
