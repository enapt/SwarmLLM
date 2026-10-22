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
