//! Single-position (decode) attention on the CPU, straight over the KV cache.
//!
//! For one query position the two attention matmuls are tiny and awkward:
//! `[b, kvh, n_rep, d] × [b, kvh, d, S]` and `[b, kvh, n_rep, S] × [b, kvh, S, d]`
//! with `n_rep` of 1–4 rows. Measured on llama-3.2-3b at ~920 KV they cost
//! **1.3 ms per layer — 26% of a decode step — for ~11 MFLOP**: the generic
//! GEMM's packing, dispatch and the transposed K view dominate, not the
//! arithmetic. This kernel does the same computation as a handful of dot
//! products and axpys over the cache in the layout it is already stored in
//! (`[b, kvh, S, d]`, rows contiguous).
//!
//! **Each K row and each V row is read ONCE for every query head of its
//! group**, and the positions are split into fixed [`attn_rows::CHUNK`]s computed
//! side by side and folded in chunk order (`attn_rows::merge_into`). The first version said it read a group's K and V
//! once and read them once PER QUERY HEAD — 3x the cache traffic on
//! llama-3.2-3b, 7x on Qwen2.5-7B — which is why its decode slowed with context
//! ~2.5x faster than llama.cpp's (#119). Fixed 2026-09-26, A/B inside one
//! binary at 4 decode threads (the width calibration picks on the Ryzen),
//! min of 2 interleaved: llama-3.2-3b at ~2,080 cached 81.6 → 70.7 ms/token and
//! level at ~544 (62.2 / 61.5), so growth from 544 to 2,080 went +19.1 →
//! +8.2 ms against llama.cpp's +7.3; Qwen2.5-7B at ~2,080 141.6 → 134.1.
//! llama.cpp's own tiled single-token CPU kernel fixes the same thing for the
//! same reason (PrismML-Eng/llama.cpp PR #254: "every KV row is loaded … once
//! per q head sharing it").
//!
//! Scope: `q_len == 1`, CPU, f32, K/V with contiguous `[S, d]` planes (what
//! `KvCache` hands out), optional additive mask row and Gemma-2 soft-cap. Anything
//! else returns `Ok(None)` and the caller keeps the matmul path — the kernel is
//! an accelerator, never a requirement.
//!
//! `SWARMLLM_DECODE_ATTN=standard` disables it for A/B inside one binary, the
//! same discipline as `SWARMLLM_FORCE_STANDARD_ATTN` and `SWARMLLM_DECODE_THREADS`.
//!
//! Numerics: a row's arithmetic is [`crate::inference::attn_rows`]', the SAME
//! the prompt kernel (`prefill_attn`) gives that row, so a position computed
//! alone (a prompt-cache hit's last token) reads exactly what it would have
//! read inside the prompt (#179). Against the matmul path it agrees to ~1e-6
//! (`decode_kernel_matches_the_matmul_path`): the dot products sum in a
//! different order from the GEMM's blocking.

use candle_core::{CpuStorage, Device, Layout, Result, Storage, Tensor};
use rayon::prelude::*;

use crate::inference::attn_rows::{self, CHUNK};

/// `SWARMLLM_TREE_ATTN=standard` → a tree of guesses on a card is read by the
/// matmul path over its `[rows, kv_len]` mask, as before
/// [`gqa_tree_attention_cuda`] (A/B switch). `SWARMLLM_DECODE_ATTN=standard`
/// turns this off too.
pub(crate) fn tree_kernel_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        decode_kernel_enabled()
            && !matches!(
                std::env::var("SWARMLLM_TREE_ATTN").as_deref(),
                Ok("standard") | Ok("off") | Ok("0")
            )
    })
}

/// `SWARMLLM_DECODE_ATTN=standard` → never use the kernel (A/B switch).
pub fn decode_kernel_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        !matches!(
            std::env::var("SWARMLLM_DECODE_ATTN").as_deref(),
            Ok("standard") | Ok("off") | Ok("0")
        )
    })
}

/// A borrowed f32 plane view: `base[offset + i*stride_i + j*stride_j + ...]`.
pub(crate) struct Planes<'a> {
    pub(crate) data: &'a [f32],
    pub(crate) offset: usize,
    /// strides for (b, h, s, d)
    pub(crate) strides: [usize; 4],
}

pub(crate) fn f32_planes<'a>(storage: &'a Storage, layout: &Layout) -> Option<Planes<'a>> {
    let data = match storage {
        Storage::Cpu(CpuStorage::F32(v)) => v.as_slice(),
        _ => return None,
    };
    let dims = layout.dims();
    let strides = layout.stride();
    if dims.len() != 4 || strides.len() != 4 {
        return None;
    }
    // The last two axes (S, d) must be a dense plane: row stride == d, unit
    // inner stride. Head/batch strides may be anything (a `narrow` along S
    // leaves a gap between heads).
    if strides[3] != 1 || strides[2] != dims[3] {
        return None;
    }
    Some(Planes {
        data,
        offset: layout.start_offset(),
        strides: [strides[0], strides[1], strides[2], strides[3]],
    })
}

/// Attention for a single query position. Returns `Ok(None)` when the inputs
/// are outside the kernel's scope (caller falls back to the matmul path).
///
/// `q`: `[b, n_head, 1, d]`; `k`, `v`: `[b, n_kv_head, S, d]`; `mask`: an
/// additive row broadcastable to `[S]` (`0.0` visible, `-inf` masked) or `None`;
/// `scale` multiplies the raw score; `softcap` is Gemma-2's tanh cap;
/// `key_origin` is the absolute position of `k`'s first key — 0, except for a
/// windowed layer, whose keys start at its window ([`attn_rows`]).
pub fn gqa_decode_attention_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    softcap: Option<f32>,
    key_origin: usize,
) -> Result<Option<Tensor>> {
    if !decode_kernel_enabled() || !matches!(q.device(), Device::Cpu) {
        return Ok(None);
    }
    if q.dtype() != candle_core::DType::F32
        || k.dtype() != candle_core::DType::F32
        || v.dtype() != candle_core::DType::F32
    {
        return Ok(None);
    }
    let (b, n_head, q_len, d) = q.dims4()?;
    if q_len != 1 {
        return Ok(None);
    }
    let (kb, n_kv_head, s_len, kd) = k.dims4()?;
    if kb != b || kd != d || n_kv_head == 0 || n_head % n_kv_head != 0 || s_len == 0 {
        return Ok(None);
    }
    if v.dims4()? != (b, n_kv_head, s_len, d) {
        return Ok(None);
    }
    let n_rep = n_head / n_kv_head;

    // The mask, if any, as one additive row over S. Anything not reducible to
    // that (a genuinely per-row mask for q_len == 1 cannot exist) → fallback.
    let mask_row: Option<Vec<f32>> = match mask {
        None => None,
        Some(m) => {
            let flat = m.flatten_all()?;
            let n = flat.dim(0)?;
            if n == s_len {
                Some(flat.to_vec1::<f32>()?)
            } else if n % s_len == 0 {
                // e.g. [n_rep, S] broadcast copies — every row identical for one
                // query position; take the first.
                let all = flat.to_vec1::<f32>()?;
                let first = &all[..s_len];
                if all.chunks_exact(s_len).all(|r| r == first) {
                    Some(first.to_vec())
                } else {
                    return Ok(None);
                }
            } else {
                return Ok(None);
            }
        }
    };

    let q = q.contiguous()?;
    let qv = q.flatten_all()?.to_vec1::<f32>()?; // b * n_head * d
    let (k_storage, k_layout) = k.storage_and_layout();
    let (v_storage, v_layout) = v.storage_and_layout();
    let Some(kp) = f32_planes(&k_storage, k_layout) else {
        return Ok(None);
    };
    let Some(vp) = f32_planes(&v_storage, v_layout) else {
        return Ok(None);
    };

    // The positions this step reads: all of them, unless the mask hides some
    // at either end — then the ones it shows, as a prompt row would (#179).
    let seen = vec![attn_rows::visible_range(mask_row.as_deref(), s_len); n_rep];
    let chunks = attn_rows::chunks_spanned(key_origin, s_len);

    // Pass 1, one task per (batch, kv head, chunk of positions): each K row and
    // each V row of the chunk is read ONCE, for every query head of the group.
    let n_chunks = chunks.len();
    let row = attn_rows::partial_len(d);
    let mut partial = vec![0f32; b * n_kv_head * n_chunks * n_rep * row];
    partial
        .par_chunks_mut(n_rep * row)
        .enumerate()
        .for_each(|(t, part)| {
            let g = t / n_chunks;
            let (bi, h) = (g / n_kv_head, g % n_kv_head);
            let k_base = kp.offset + bi * kp.strides[0] + h * kp.strides[1];
            let v_base = vp.offset + bi * vp.strides[0] + h * vp.strides[1];
            let q0 = (bi * n_head + h * n_rep) * d;
            let mut scores = vec![0f32; n_rep * CHUNK];
            attn_rows::chunk_partials(
                &qv[q0..q0 + n_rep * d],
                &seen,
                &kp.data[k_base..k_base + s_len * d],
                &vp.data[v_base..v_base + s_len * d],
                d,
                key_origin,
                chunks.start + t % n_chunks,
                &|_| mask_row.as_deref(),
                scale,
                softcap,
                &mut scores,
                part,
            );
        });

    // Pass 2, one task per (batch, kv head): each row folds its chunks in
    // order, as a prompt task folds them (`attn_rows::merge_into`).
    let mut out = vec![0f32; b * n_head * d];
    out.par_chunks_mut(n_rep * d)
        .enumerate()
        .for_each(|(g, out_g)| {
            let mut acc = vec![0f32; row];
            for (r, o) in out_g.chunks_exact_mut(d).enumerate() {
                attn_rows::empty_partial(&mut acc);
                for c in 0..n_chunks {
                    let at = ((g * n_chunks + c) * n_rep + r) * row;
                    attn_rows::merge_into(&mut acc, &partial[at..at + row]);
                }
                attn_rows::finish_row(&acc, o);
            }
        });

    Ok(Some(Tensor::from_vec(
        out,
        (b, n_head, 1, d),
        &Device::Cpu,
    )?))
}

/// The most cached positions per block on the CARD — `DA_CHUNK` in
/// `kernels/decode_attn.cu`, what its shared memory is sized for, and the
/// chunk a long cache gets ([`chunk_for`]). Smaller than the processor's
/// [`CHUNK`] because a card needs blocks to fill it: at 200 positions on a
/// 4-KV-head model, 256 would be four blocks on a 40-SM card.
#[cfg(any(feature = "candle-cuda", test))]
pub(crate) const CUDA_CHUNK: usize = 64;

/// The chunks the card kernel takes, smallest first — its `chunk` argument,
/// at most [`CUDA_CHUNK`] (`DA_CHUNK`, what its shared memory is sized for).
#[cfg(any(feature = "candle-cuda", test))]
const CARD_CHUNKS: [usize; 3] = [16, 32, CUDA_CHUNK];

/// The most blocks a smaller chunk may make ([`chunk_for`]).
#[cfg(any(feature = "candle-cuda", test))]
const CHUNK_MAX_BLOCKS: usize = 256;

/// The most chunks a smaller chunk may make: the merge walks a head's chunks
/// one after another, ~0.45 µs each ([`chunk_for`]).
#[cfg(any(feature = "candle-cuda", test))]
const CHUNK_MAX_CHUNKS: usize = 64;

/// Positions per block for `groups` (batch × KV heads) over `s_len` cached
/// positions: the smallest of [`CARD_CHUNKS`] that keeps the blocks within
/// [`CHUNK_MAX_BLOCKS`] and the chunks within [`CHUNK_MAX_CHUNKS`], else
/// [`CUDA_CHUNK`].
///
/// A block's time is its chunk's positions one after another, so a short
/// cache in few blocks idles the card: TinyLlama at ~200 positions is 16
/// blocks of 64 on a 40-SM RTX 3070 Laptop, and its attention took 39.8 µs a
/// layer against llama.cpp's ~8 — 12.0 µs in chunks of 16 (nsys, FUTURE_WORK
/// #309). A long cache has blocks to spare, and each extra chunk costs the
/// merge: Llama 3.2 3B at ~4,000 positions took 224 + 29 µs in chunks of 64
/// and 237 + 57 in chunks of 32. Measured end to end (2026-10-19, 160-token
/// replies, each chunk alternated twice in one binary), the rule picks the
/// fastest arm at every point: TinyLlama ~200 positions → 16 (300.6 tok/s
/// against 252 in 64), 1,700 → 32 (209 against 201); Llama 3.2 3B ~200 → 16
/// (122 against 117), ~600 → 32 (level), ~4,000 → 64 (68.5 against 63.7 in
/// 32). The rule against fixed 64, alternated twice in one binary: TinyLlama
/// 297-304 tok/s against 252-258 (llama.cpp 299), Gemma 3 1B 196-209 against
/// 175, Gemma 2 2B 126-127 against 118-120, Llama 3.2 3B 121-122 against
/// 116-117; behind 1,848 and 672-token prompts +4 and +3 %; at 4,320 it is 64
/// and the reply is the same byte for byte. It reads the head layout and the
/// length only — never the card — so a reply is the same on every card, and a
/// captured step only updates its arguments as the chunk grows (a reply
/// crossing 16 → 32 rebuilt no graph).
#[cfg(any(feature = "candle-cuda", test))]
pub(crate) fn chunk_for(groups: usize, s_len: usize) -> usize {
    CARD_CHUNKS
        .into_iter()
        .find(|&c| {
            let chunks = s_len.div_ceil(c);
            chunks <= CHUNK_MAX_CHUNKS && groups * chunks <= CHUNK_MAX_BLOCKS
        })
        .unwrap_or(CUDA_CHUNK)
}

/// `SWARMLLM_DECODE_ATTN_CHUNK=16|32|64` holds the card kernel to one chunk
/// for an A/B inside one binary (`64` = the kernel before [`chunk_for`]);
/// unset, [`chunk_for`] chooses per call.
#[cfg(feature = "candle-cuda")]
fn fixed_chunk() -> Option<usize> {
    static CHUNK: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *CHUNK.get_or_init(|| {
        std::env::var("SWARMLLM_DECODE_ATTN_CHUNK")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|c| CARD_CHUNKS.contains(c))
    })
}

/// Whether a cache of ONE chunk is merged by the combine kernel too, so a
/// decode step launches the same two kernels at every length.
///
/// A decode step on a card is a captured CUDA graph, updated in place each
/// step while its kernels stay the same (`inference::cuda_graph`). Writing a
/// single chunk's result directly and adding the combine only past
/// [`CUDA_CHUNK`] positions changed every layer's kernels at position 64 (and
/// back when the next conversation started below it): the driver refuses the
/// update and every group is instantiated afresh — 790 ms in ONE step of a
/// Gemma 3 4B reply on the RTX 3070 Laptop (2026-10-19), against ~12 ms for
/// the steps around it. A reply to a short question (Gemma's template adds no
/// system prompt) stalled once mid-answer, and a 160-token decode averaged
/// 45-56 tokens a second instead of ~80. A one-chunk merge is bit-identical to
/// the direct write (its one weight is `expf(0) = 1`; pinned by
/// `one_chunk_merged_is_bit_identical_to_the_direct_write`) and costs one
/// small launch a layer while the cache is one chunk.
/// `SWARMLLM_DECODE_ATTN_COMBINE=when-needed` restores the direct write for an
/// A/B inside one binary.
#[cfg(feature = "candle-cuda")]
fn combine_every_length() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("SWARMLLM_DECODE_ATTN_COMBINE").as_deref() != Ok("when-needed")
    })
}
/// Query heads per KV head one block holds (`DA_MAX_REP`); head dims it
/// handles (`DA_MAX_D`, and a multiple of 32).
const CUDA_MAX_REP: usize = 16;
const CUDA_MAX_D: usize = 256;

/// Does the card kernel answer every one-position attention of a model with
/// these head counts? The geometry half of [`gqa_decode_attention_cuda`]'s
/// scope — the rest (a mask, a cache view whose rows are not dense) is about
/// one call, never about the model.
///
/// **The half cache is only kept where this is true** (`layers::kv_storage`):
/// a decode step the kernel declines falls to the matmul path, which reads an
/// f32 cache, so a half cache there would be widened from scratch on every
/// token — O(history) per step, the cost the f16 mirror was built to remove.
/// One predicate, so the storage decision and the kernel cannot disagree
/// about which models the kernel takes.
pub(crate) fn card_kernel_covers(n_head: usize, n_kv_head: usize, head_dim: usize) -> bool {
    decode_kernel_enabled() && card_kernel_geometry(n_head, n_kv_head, head_dim)
}

/// The geometry alone, whatever this process's A/B switch says — what a
/// coordinator asks of a PEER's model (`layers::kv_storage_as_advertised`),
/// whose own switch it cannot see.
pub(crate) fn card_kernel_geometry(n_head: usize, n_kv_head: usize, head_dim: usize) -> bool {
    n_kv_head > 0
        && n_head.is_multiple_of(n_kv_head)
        && n_head / n_kv_head <= CUDA_MAX_REP
        && head_dim > 0
        && head_dim.is_multiple_of(32)
        && head_dim <= CUDA_MAX_D
}

/// PTX for the card kernel, compiled from `kernels/decode_attn.cu` by
/// `build.rs` and loaded through candle's `get_or_load_custom_func`.
#[cfg(feature = "candle-cuda")]
pub(crate) const DECODE_ATTN_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/decode_attn.ptx"));

/// [`gqa_decode_attention_cpu`]'s twin on a CARD: one query position, straight
/// over the f32 cache in its stored layout, each K and V row read once for its
/// whole group, split into fixed chunks merged by flash-decoding's reduction.
/// It replaces two cuBLAS matmuls around a softmax (plus the scale and the
/// copies around them) — ~0.1 ms per layer at 512 positions, 0.32 at 8K on an
/// RTX 3070 — with two launches, the chunks and their merge, at every cache
/// length (`combine_every_length`).
///
/// Scope: CUDA, an f32 query over an f32 OR f16 cache (K and V alike — the
/// half cache a card keeps, `layers::KvStorage::F16`), `q_len == 1`, no mask
/// (a decode step on a card never has one), a geometry [`card_kernel_covers`],
/// K/V with dense `[S, d]` rows. Anything else returns `Ok(None)` and the
/// caller keeps the matmul path. `SWARMLLM_DECODE_ATTN=standard` turns both
/// kernels off for an A/B inside one binary.
///
/// Not bit-identical to the matmul path (another summation order); pinned
/// against it within float tolerance by
/// `cuda_decode_kernel_matches_the_matmul_path`, and replies are judged against
/// llama.cpp.
pub fn gqa_decode_attention_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    softcap: Option<f32>,
) -> Result<Option<Tensor>> {
    if !decode_kernel_enabled() || !q.device().is_cuda() || mask.is_some() {
        return Ok(None);
    }
    let cache_dtype = k.dtype();
    if q.dtype() != candle_core::DType::F32
        || !matches!(
            cache_dtype,
            candle_core::DType::F32 | candle_core::DType::F16
        )
        || v.dtype() != cache_dtype
    {
        return Ok(None);
    }
    let (b, n_head, q_len, d) = q.dims4()?;
    let (kb, n_kv_head, s_len, kd) = k.dims4()?;
    if q_len != 1
        || kb != b
        || kd != d
        || !card_kernel_covers(n_head, n_kv_head, d)
        || s_len == 0
        || v.dims4()? != (b, n_kv_head, s_len, d)
    {
        return Ok(None);
    }
    for t in [k, v] {
        let st = t.stride();
        if st[3] != 1 || st[2] != d {
            return Ok(None);
        }
    }
    #[cfg(feature = "candle-cuda")]
    {
        let q = q.contiguous()?;
        let op = cuda::DecodeAttn {
            scale,
            softcap: softcap.unwrap_or(0.0),
            combine_one_chunk: combine_every_length(),
            chunk: fixed_chunk(),
            tree: None,
        };
        Ok(Some(q.apply_op3_no_bwd(k, v, &op)?))
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        let _ = (scale, softcap);
        Ok(None)
    }
}

/// The most rows of a tree [`gqa_tree_attention_cuda`] reads. Each row
/// re-reads the cache — one block per (row, KV head, chunk) — so its time
/// grows with rows × keys, where the matmul path pays a near-fixed cost
/// (widening a half cache, two cuBLAS products): on the RTX 3070 Laptop
/// (`tree_attention_timing`, min of 30 calls each synchronised, half cache,
/// 2026-10-22) the kernel against the matmul path, µs a call — the 0.5B
/// drafter's heads (14/2, d 64) at 900 keys: 8 rows 188 / 477, 25 rows 380 /
/// 473, 32 rows 456 / 431; a 7B's (28/4, d 128) at 900 keys: 8 rows 235 / 305,
/// 12 rows 328 / 318, 25 rows 597 / 394. On the rig a 7B's 25-row tree
/// forwards on the kernel took this node's half 30% longer (near 14.4-15.3 s
/// a 400-token reply against 10.1-12.6) while the drafter's 8-row levels
/// gained 9%. At 8 rows it wins at every shape measured — and 8 is what a
/// captured forward may carry ([`crate::inference::cuda_graph::MAX_POSITIONS`]),
/// the place a mask cannot be. Over the HALF cache a card keeps, every tree
/// goes to the tiled kernel instead ([`tree_tiles`]), whose blocks share each
/// chunk's K/V reads among a tile of rows; this cap is the per-row kernel's,
/// which an f32 cache (and `SWARMLLM_TREE_ATTN_TILED=0`) still reads.
pub(crate) const TREE_KERNEL_MOST_ROWS: usize = 8;

/// Positions a block of the TILED tree kernel stages for head width `d`
/// (`kernels/decode_attn.cu`'s `tree_attn_tiled_f16kv_d128` / `_d256`): fixed
/// per width, so a captured call only updates its chunk count as the cache
/// grows.
#[cfg(feature = "candle-cuda")]
fn tiled_chunk(d: usize) -> usize {
    if d <= 128 {
        64
    } else {
        32
    }
}

/// Query vectors (a tree row × a query head of its group) one block of the
/// tiled kernel serves, for head width `d`.
#[cfg(feature = "candle-cuda")]
fn tiled_queries(d: usize) -> usize {
    if d <= 128 {
        32
    } else {
        16
    }
}

/// The fewest rows a tree's forward must have for the TILED kernel to read it
/// ([`tree_tiles`]). `SWARMLLM_TREE_ATTN_TILED=0` (or `off`) never tiles — a
/// tree past [`TREE_KERNEL_MOST_ROWS`] goes back to the matmul path; a number
/// tiles from that many rows, for an A/B inside one binary.
fn tree_tiled_from_rows() -> Option<usize> {
    static FROM: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FROM.get_or_init(
        || match std::env::var("SWARMLLM_TREE_ATTN_TILED").as_deref() {
            Ok("0") | Ok("off") | Ok("standard") => None,
            Ok(v) => Some(
                v.parse::<usize>()
                    .ok()
                    .filter(|&n| n > 0)
                    .unwrap_or(TREE_TILED_FROM_ROWS),
            ),
            Err(_) => Some(TREE_TILED_FROM_ROWS),
        },
    )
}

/// Rows from which a tree's forward over the half cache is read by the tiled
/// kernel (the default of [`tree_tiled_from_rows`]): every tree. It beat one
/// block a row at every row count measured (`tree_attention_timing`, 1-64
/// rows, 300 and 900 keys, the 0.5B's and a 7B's heads) and in whole forwards:
/// a 7B's near half at 8 rows 10.6-10.9 ms against 11.8-12.1, the 0.5B
/// drafter's 8-row level 600 deep 5.1-5.2 against 6.6-6.8 (2026-10-22).
const TREE_TILED_FROM_ROWS: usize = 1;

/// Calls the TILED tree kernel has read in this process — what a test reads to
/// know it ran (`a_wide_tree_on_the_card_reads_what_each_path_reads`).
#[cfg(feature = "candle-cuda")]
static TILED_TREE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `TILED_TREE_CALLS`.
#[cfg(all(test, feature = "candle-cuda"))]
pub(crate) fn tiled_tree_calls() -> u64 {
    TILED_TREE_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Does the TILED tree kernel read this call? A half cache whose rows start on
/// 16 bytes (the kernel stages eight halves a load), at least
/// [`tree_tiled_from_rows`] rows. The geometry and the visibility's reach are
/// [`gqa_tree_attention_cuda`]'s, checked before.
fn tree_tiles(rows: usize, k: &Tensor, v: &Tensor) -> bool {
    let Some(from) = tree_tiled_from_rows() else {
        return false;
    };
    rows >= from
        && k.dtype() == candle_core::DType::F16
        && v.dtype() == candle_core::DType::F16
        && [k, v].into_iter().all(|t| {
            t.layout().start_offset().is_multiple_of(8)
                && t.stride()[..2].iter().all(|s| s.is_multiple_of(8))
        })
}

/// Who reads a tiled tree's context — every position before its base, which
/// each of its rows sees whole. `SWARMLLM_TREE_ATTN_PAST=tiled` (or `0`,
/// `off`): the tiled kernel, as before; a number: flash in that many runs
/// from a run of one position, for an A/B inside one binary; unset: flash in
/// [`auto_past_runs`] runs of at least [`PAST_RUN_LEAST`], once the call is
/// [`PAST_FLASH_FROM`] work or more.
#[cfg(feature = "candle-cuda")]
#[derive(Clone, Copy, Debug)]
enum PastReader {
    Tiled,
    Runs(usize),
    Auto,
}

#[cfg(feature = "candle-cuda")]
fn past_reader() -> PastReader {
    static READER: std::sync::OnceLock<PastReader> = std::sync::OnceLock::new();
    *READER.get_or_init(
        || match std::env::var("SWARMLLM_TREE_ATTN_PAST").as_deref() {
            // A build without flash attention has nothing else to read it.
            _ if !cfg!(feature = "flash-attn") => PastReader::Tiled,
            Ok("tiled") | Ok("0") | Ok("off") => PastReader::Tiled,
            Ok(v) => v
                .parse::<usize>()
                .ok()
                .filter(|&n| (1..=PAST_MOST_RUNS).contains(&n))
                .map_or(PastReader::Auto, PastReader::Runs),
            Err(_) => PastReader::Auto,
        },
    )
}

/// The fewest context positions a run of flash reads by default: below
/// `runs` × this the context stays on the tiled kernel. It keeps a run at a
/// key tile or more (32 positions at head width 128 on Ampere,
/// `flash_fwd_launch_template.h`).
#[cfg(feature = "candle-cuda")]
const PAST_RUN_LEAST: usize = 32;

/// The most runs flash splits a context into.
#[cfg(feature = "candle-cuda")]
const PAST_MOST_RUNS: usize = 16;

/// Rows × context positions × head width from which flash reads a tiled
/// tree's context by default. Below it the tiled kernel alone is as fast or
/// faster: it is one launch to flash's three (the queries' conversion, flash,
/// the tiled kernel over the rest), and its CUDA-core arithmetic is small.
/// Measured on the op (`tree_context_on_flash_timing`, RTX 3070 Laptop,
/// 2026-10-22): the 0.5B drafter's heads (d 64) 8 rows 600 deep 74 µs tiled
/// against 89 at best (307k), 16 rows 80 against 87 (614k), 24 rows 97
/// against 93 (922k), 8 rows 1,400 deep 98 against 90 (717k); a 7B's (d 128)
/// 8 rows 600 deep 81 against 82 (614k), 16 rows 101 against 90 (1.2M). The
/// drafter's whole flow call 600 deep was 14.2-14.5 ms tiled against
/// 15.0-15.1 with flash at every size.
#[cfg(feature = "candle-cuda")]
const PAST_FLASH_FROM: usize = 700_000;

/// Runs a context is split into: about one of flash's blocks a
/// multiprocessor, a block being a KV head × 128 query vectors × one run —
/// flash-decoding's split, so a tree of a few rows does not leave the card
/// idle. Measured (`tree_context_on_flash_timing`, 40 multiprocessors): the
/// best run count put 32-40 blocks on the card at every shape — a 7B's 24
/// rows (8 blocks a run) 4-8 runs, its 8 rows 8, its 64 rows 2; twice that
/// was 5-15% slower. A function of the shape and the card only, never of the
/// context's length: a captured check keeps its allocations as the context
/// grows (a graph whose allocation changes is rebuilt, not updated).
#[cfg(feature = "candle-cuda")]
fn auto_past_runs(query_vectors: usize, n_kv_head: usize, sms: usize) -> usize {
    let blocks = n_kv_head * query_vectors.div_ceil(128);
    (sms / blocks.max(1)).clamp(1, PAST_MOST_RUNS)
}

/// Multiprocessors of the card `dev` is on.
#[cfg(feature = "candle-cuda")]
fn card_sms(dev: &Device) -> Option<usize> {
    use candle_core::cuda_backend::cudarc::driver::sys::CUdevice_attribute;
    // Asked once per card: the count never changes, and the asker is often a
    // check being recorded as a CUDA graph, where nothing should reach the
    // driver that need not (gotcha #841's class; review, 2026-10-23).
    static KNOWN: std::sync::Mutex<Vec<(usize, Option<usize>)>> = std::sync::Mutex::new(Vec::new());
    let (Device::Cuda(c), candle_core::DeviceLocation::Cuda { gpu_id }) = (dev, dev.location())
    else {
        return None;
    };
    let mut known = KNOWN
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(&(_, sms)) = known.iter().find(|(card, _)| *card == gpu_id) {
        return sms;
    }
    let sms = c
        .cuda_stream()
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok()
        .and_then(|n| usize::try_from(n).ok());
    known.push((gpu_id, sms));
    sms
}

/// How flash reads a tiled tree's context of `base` positions — `(runs,
/// positions a run)`, covering `runs * run_len <= base`; the tiled kernel
/// reads the rest of it and the tree (`cuda::PastOnFlash`). `None`: the tiled
/// kernel reads it all ([`past_reader`]; a call under [`PAST_FLASH_FROM`]; a
/// context too short for the runs; a build without flash).
#[cfg(feature = "candle-cuda")]
fn past_runs(
    rows: usize,
    n_kv_head: usize,
    n_rep: usize,
    d: usize,
    base: usize,
    dev: &Device,
) -> Option<(usize, usize)> {
    let (runs, least) = match past_reader() {
        PastReader::Tiled => return None,
        PastReader::Runs(n) => (n, 1),
        PastReader::Auto if rows * base * d < PAST_FLASH_FROM => return None,
        PastReader::Auto => (
            auto_past_runs(rows * n_rep, n_kv_head, card_sms(dev)?),
            PAST_RUN_LEAST,
        ),
    };
    let run_len = base / runs;
    (run_len >= least).then_some((runs, run_len))
}

/// A tree's queries as flash reads its context: `q_rows` `[rows, n_head, 1,
/// d]` f32 → `[1, rows * n_rep, n_kv_head, d]` half, query vector `row * n_rep
/// + r` of KV head `h` being query head `h * n_rep + r` of `row`. A group's
/// heads become rows of one head (flash's own `seqlenq_ngroups_swapped` for
/// decode), so a block reads its KV head's keys once for all of them rather
/// than once a query head. Flash takes half queries — what a chain's check on
/// flash takes too (`layers::run_attention`).
#[cfg(feature = "candle-cuda")]
fn flash_queries(
    q_rows: &Tensor,
    rows: usize,
    n_kv_head: usize,
    n_rep: usize,
    d: usize,
) -> Result<Tensor> {
    q_rows
        .reshape((1, rows, n_kv_head, n_rep, d))?
        .transpose(2, 3)?
        .to_dtype(candle_core::DType::F16)?
        .reshape((1, rows * n_rep, n_kv_head, d))
}

// A captured tree's forward builds no mask, so the kernel must take every row
// count a capture may carry (`SplitModel::tree_steps_may_be_captured`).
const _: () = assert!(crate::inference::cuda_graph::MAX_POSITIONS <= TREE_KERNEL_MOST_ROWS);

/// A TREE of guesses' attention on a card (#134): every row of the forward
/// reads the confirmed context and its own ancestors, through `visibility` —
/// `[rows, VISIBILITY_WORDS]` u32 on the card, a bit per tree row
/// (`draft_tree::RowVisibility`, made once a forward by `layers::TreeRows`) —
/// over the cache as it is stored, f32 or half.
///
/// The decode kernel with each row a batch element of its own over the same
/// cache (`kernels/decode_attn.cu`): its launches are the same two whatever
/// the tree or the cache length, where the matmul path it replaces sized two
/// cuBLAS products and a `[rows, kv_len]` mask by both — and, over the half
/// cache a card keeps, widened the whole history to f32 every call. That is
/// FlashInfer's packed custom mask, the shape SGLang's and vLLM's tree drafts
/// run under CUDA graphs. Each row re-reads the cache, so the work grows with
/// rows × history — hence [`TREE_KERNEL_MOST_ROWS`]: a drafter's level of 8.
///
/// Scope: the decode kernel's ([`gqa_decode_attention_cuda`]), at most
/// [`TREE_KERNEL_MOST_ROWS`] query rows unless the tiled kernel reads the call
/// ([`tree_tiles`]: a half cache, any rows), a cache of `tree_base` context
/// positions plus at most `32 * VISIBILITY_WORDS` tree rows. Anything else `Ok(None)` — the caller
/// keeps the matmul path over the mask. `SWARMLLM_TREE_ATTN=standard` for an
/// A/B. Not bit-identical to that path; pinned against it by
/// `cuda_tree_kernel_matches_the_masked_matmul_path`, replies judged against
/// llama.cpp.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gqa_tree_attention_cuda(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    visibility: &Tensor,
    tree_base: usize,
    scale: f32,
    softcap: Option<f32>,
) -> Result<Option<Tensor>> {
    use crate::inference::draft_tree::VISIBILITY_WORDS;
    if !tree_kernel_enabled() || !q.device().is_cuda() {
        return Ok(None);
    }
    let cache_dtype = k.dtype();
    if q.dtype() != candle_core::DType::F32
        || !matches!(
            cache_dtype,
            candle_core::DType::F32 | candle_core::DType::F16
        )
        || v.dtype() != cache_dtype
        || visibility.dtype() != candle_core::DType::U32
        || !visibility.device().same_device(q.device())
    {
        return Ok(None);
    }
    let (b, n_head, rows, d) = q.dims4()?;
    let (kb, n_kv_head, s_len, kd) = k.dims4()?;
    let tiled = tree_tiles(rows, k, v);
    if rows == 0
        || (rows > TREE_KERNEL_MOST_ROWS && !tiled)
        || kb != b
        || kd != d
        || !card_kernel_covers(n_head, n_kv_head, d)
        || s_len <= tree_base
        || s_len - tree_base > 32 * VISIBILITY_WORDS
        || v.dims4()? != (b, n_kv_head, s_len, d)
        || visibility.dims() != [rows, VISIBILITY_WORDS]
    {
        return Ok(None);
    }
    for t in [k, v] {
        let st = t.stride();
        if st[3] != 1 || st[2] != d {
            return Ok(None);
        }
    }
    #[cfg(feature = "candle-cuda")]
    {
        // Each row a batch element: `[b * rows, n_head, 1, d]`, rows of a
        // batch one after another, as the kernel numbers them.
        let q_rows = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b * rows, n_head, 1, d))?;
        // The context on flash's tensor cores, the tree's own rows on the
        // tiled kernel (one K/V batch: a tree is one request's).
        let flash_runs = past_runs(
            rows,
            n_kv_head,
            n_head / n_kv_head,
            d,
            tree_base,
            q.device(),
        );
        let past = match (tiled && b == 1, flash_runs) {
            (true, Some((runs, run_len))) => Some(cuda::PastOnFlash {
                q: flash_queries(&q_rows, rows, n_kv_head, n_head / n_kv_head, d)?,
                runs,
                run_len,
            }),
            _ => None,
        };
        let runs = past.as_ref().map_or(0, |p| p.runs);
        let op = cuda::DecodeAttn {
            scale,
            softcap: softcap.unwrap_or(0.0),
            combine_one_chunk: combine_every_length(),
            chunk: fixed_chunk(),
            tree: Some(cuda::TreeOperands {
                visibility: visibility.clone(),
                rows,
                base: tree_base,
                tiled,
                past,
            }),
        };
        let out = q_rows.apply_op3_no_bwd(k, v, &op)?;
        if tiled {
            TILED_TREE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // What a rig greps to know the kernel ran (an A/B whose arms measure
        // alike proves nothing without it).
        // Once for each kernel: per row, tiled, tiled after flash.
        static RAN: [std::sync::Once; 3] = [
            std::sync::Once::new(),
            std::sync::Once::new(),
            std::sync::Once::new(),
        ];
        RAN[usize::from(tiled) + usize::from(runs > 0)].call_once(|| {
            tracing::debug!(
                target: "swarmllm::inference::decode_attn",
                rows,
                s_len,
                tiled,
                flash_runs = runs,
                "a tree's attention runs on the card's tree kernel"
            )
        });
        // Back to `[b, n_head, rows, d]` — a view of the kernel's
        // `[b, rows, n_head, d]`, which the caller's `transpose(1, 2)` reads
        // in place.
        Ok(Some(out.reshape((b, rows, n_head, d))?.transpose(1, 2)?))
    }
    #[cfg(not(feature = "candle-cuda"))]
    {
        let _ = (scale, softcap);
        Ok(None)
    }
}

#[cfg(feature = "candle-cuda")]
mod cuda {
    use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape, Tensor};

    /// A tree's rows ([`super::gqa_tree_attention_cuda`]).
    pub(super) struct TreeOperands {
        /// `[rows, VISIBILITY_WORDS]` u32 on the card.
        pub(super) visibility: Tensor,
        /// Query rows a K/V batch serves.
        pub(super) rows: usize,
        /// The cache position of the tree's first row.
        pub(super) base: usize,
        /// Read by the TILED kernel (`tree_attn_tiled_f16kv_*`, a half cache
        /// only — [`super::tree_tiles`]) instead of one block per row.
        pub(super) tiled: bool,
        /// The context flash attention reads instead ([`super::past_runs`]):
        /// the tiled kernel then reads only the positions after it.
        pub(super) past: Option<PastOnFlash>,
    }

    /// A tree's context on flash attention's tensor cores
    /// ([`super::past_runs`]): positions `0 .. runs * run_len`, every one
    /// of them before the tree's base, so every row sees all of them.
    pub(super) struct PastOnFlash {
        /// The rows' queries as flash reads them: `[1, rows * n_rep,
        /// n_kv_head, d]` half — query vector `row * n_rep + r` of each KV
        /// head, a group's heads as rows (flash's `seqlenq_ngroups_swapped`),
        /// so one block reads its K/V for all of them.
        pub(super) q: Tensor,
        /// Equal runs of the context, one flash batch element (and block) each.
        pub(super) runs: usize,
        /// Positions a run.
        pub(super) run_len: usize,
    }

    pub(super) struct DecodeAttn {
        pub(super) scale: f32,
        /// 0 = none.
        pub(super) softcap: f32,
        /// A one-chunk cache goes through the combine too
        /// ([`super::combine_every_length`]).
        pub(super) combine_one_chunk: bool,
        /// Positions per block, one of [`super::CARD_CHUNKS`]; `None` =
        /// [`super::chunk_for`] the call's shape ([`super::fixed_chunk`]).
        pub(super) chunk: Option<usize>,
        /// `None`: one query row a batch, every position read (a decode step).
        pub(super) tree: Option<TreeOperands>,
    }

    /// The cache's K and V as the kernel reads them — f32, or the half cache
    /// a card keeps. One kernel body serves both (`kernels/decode_attn.cu`).
    enum CacheViews<'a> {
        F32(
            candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>,
            candle_core::cuda_backend::cudarc::driver::CudaView<'a, f32>,
        ),
        F16(
            candle_core::cuda_backend::cudarc::driver::CudaView<'a, half::f16>,
            candle_core::cuda_backend::cudarc::driver::CudaView<'a, half::f16>,
        ),
    }

    impl CustomOp3 for DecodeAttn {
        fn name(&self) -> &'static str {
            "decode-attention"
        }

        /// Never reached: [`super::gqa_decode_attention_cuda`] returns `None`
        /// off a card, and the processor has its own kernel.
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!("decode-attention is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            s1: &candle_core::CudaStorage,
            l1: &Layout,
            s2: &candle_core::CudaStorage,
            l2: &Layout,
            s3: &candle_core::CudaStorage,
            l3: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use candle_core::backend::BackendStorage;
            use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use candle_core::cuda_backend::WrapErr;

            let dev = s1.device.clone();
            let Some((qo, qe)) = l1.contiguous_offsets() else {
                candle_core::bail!("decode-attention: q must be contiguous");
            };
            // `b` counts query rows: a tree's rows are batch elements of
            // their own over their batch's cache (`kv_b`).
            let (b, n_head, _, d) = l1.shape().dims4()?;
            let (kv_b, n_kv_head, s_len, _) = l2.shape().dims4()?;
            let (ks, vs) = (l2.stride(), l3.stride());
            if ks[3] != 1 || ks[2] != d || vs[3] != 1 || vs[2] != d || qe - qo != b * n_head * d {
                candle_core::bail!("decode-attention: unexpected layout");
            }
            let rows = self.tree.as_ref().map_or(1, |t| t.rows);
            if b != kv_b * rows {
                candle_core::bail!(
                    "decode-attention: {b} query rows over {kv_b} caches of {rows} rows"
                );
            }
            // The visibility's view, held for the launch.
            let vis_storage = self
                .tree
                .as_ref()
                .map(|t| t.visibility.storage_and_layout());
            let vis = match &vis_storage {
                None => None,
                Some((storage, layout)) => {
                    let candle_core::Storage::Cuda(c) = &**storage else {
                        candle_core::bail!("decode-attention: a tree's visibility off the card");
                    };
                    let Some((o, e)) = layout.contiguous_offsets() else {
                        candle_core::bail!(
                            "decode-attention: a tree's visibility must be contiguous"
                        );
                    };
                    let words = crate::inference::draft_tree::VISIBILITY_WORDS;
                    if e - o != rows * words
                        || s_len > self.tree.as_ref().map_or(0, |t| t.base) + 32 * words
                    {
                        candle_core::bail!(
                            "decode-attention: a tree's visibility does not cover the cache"
                        );
                    }
                    Some(c.as_cuda_slice::<u32>()?.slice(o..e))
                }
            };
            let n_rep = n_head / n_kv_head;
            let tiled = self.tree.as_ref().is_some_and(|t| t.tiled);
            let chunk = if tiled {
                super::tiled_chunk(d)
            } else {
                self.chunk
                    .unwrap_or_else(|| super::chunk_for(b * n_kv_head, s_len))
            };
            if !tiled && !super::CARD_CHUNKS.contains(&chunk) {
                candle_core::bail!("decode-attention: no kernel for {chunk}-position chunks");
            }
            // A tree's context on flash: the positions it reads, which the
            // kernels below skip (their views start after them, the tree's
            // base counted from there).
            let past = self.tree.as_ref().and_then(|t| t.past.as_ref());
            let base = self.tree.as_ref().map_or(0, |t| t.base);
            let skip = past.map_or(0, |p| p.runs * p.run_len);
            if let Some(p) = past {
                if !tiled
                    || kv_b != 1
                    || skip == 0
                    || skip > base
                    || p.q.dims() != [1, rows * n_rep, n_kv_head, d]
                {
                    candle_core::bail!(
                        "decode-attention: flash reads the context of one tiled tree only \
                         ({} runs of {} before base {base}, queries {:?})",
                        p.runs,
                        p.run_len,
                        p.q.dims()
                    );
                }
            }
            let kernel_len = s_len - skip;
            let n_chunks = kernel_len.div_ceil(chunk);
            // The tiled kernel writes partials only.
            let merged = tiled || n_chunks > 1 || self.combine_one_chunk;

            // Each run's output and log-sum-exp, merged by the combine below.
            #[cfg(feature = "flash-attn")]
            let past_out = match past {
                None => None,
                Some(p) => {
                    let (qs, ql) = p.q.storage_and_layout();
                    let candle_core::Storage::Cuda(qs) = &*qs else {
                        candle_core::bail!("decode-attention: a tree's flash queries off the card");
                    };
                    Some(candle_flash_attn::flash_attn_key_runs(
                        qs,
                        ql,
                        s2,
                        &l2.narrow(2, 0, skip)?.transpose(1, 2)?,
                        s3,
                        &l3.narrow(2, 0, skip)?.transpose(1, 2)?,
                        p.runs,
                        self.scale,
                        (self.softcap > 0.0).then_some(self.softcap),
                    )?)
                }
            };
            #[cfg(not(feature = "flash-attn"))]
            let past_out: Option<(
                candle_core::cuda_backend::cudarc::driver::CudaSlice<half::f16>,
                candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>,
            )> = match past {
                None => None,
                Some(_) => candle_core::bail!("decode-attention: built without flash attention"),
            };

            let q = s1.as_cuda_slice::<f32>()?.slice(qo..qe);
            // The cache views start where their layouts say (past what flash
            // read); every read the kernel makes is within `b` batches ×
            // `n_kv_head` heads × `kernel_len` rows of `d` from there, by the
            // strides passed. The element type picks the entry point: the half
            // cache a card keeps (`layers::KvStorage::F16`) or the f32 one.
            let (k_at, v_at) = (l2.start_offset() + skip * d, l3.start_offset() + skip * d);
            let cache = match (s2.dtype(), s3.dtype()) {
                (candle_core::DType::F16, candle_core::DType::F16) => CacheViews::F16(
                    s2.as_cuda_slice::<half::f16>()?.slice(k_at..),
                    s3.as_cuda_slice::<half::f16>()?.slice(v_at..),
                ),
                (candle_core::DType::F32, candle_core::DType::F32) => CacheViews::F32(
                    s2.as_cuda_slice::<f32>()?.slice(k_at..),
                    s3.as_cuda_slice::<f32>()?.slice(v_at..),
                ),
                (kd, vd) => candle_core::bail!("decode-attention: a {kd:?} K beside a {vd:?} V"),
            };

            // Every element of both buffers is assigned: `out` by the one
            // kernel or the combine, `partial` by every (block, head, dim).
            let mut out = dev.alloc_fully_overwritten::<f32>(b * n_head * d)?;
            let mut partial = if merged {
                Some(
                    dev.alloc_fully_overwritten::<f32>(b * n_kv_head * n_chunks * n_rep * (d + 2))?,
                )
            } else {
                None
            };

            let (k_sb, k_sh, v_sb, v_sh) = (ks[0] as i64, ks[1] as i64, vs[0] as i64, vs[1] as i64);
            let (n_kv_i, n_rep_i, d_i, s_len_i) =
                (n_kv_head as i32, n_rep as i32, d as i32, kernel_len as i32);
            if tiled {
                let CacheViews::F16(k, v) = &cache else {
                    candle_core::bail!(
                        "decode-attention: the tiled tree kernel reads a half cache"
                    );
                };
                // Eight halves a load: every row the kernel reads starts on 16 bytes.
                if [k_at, v_at]
                    .into_iter()
                    .chain([ks[0], ks[1], vs[0], vs[1]])
                    .any(|x| !x.is_multiple_of(8))
                    || d > 256
                    || !d.is_multiple_of(32)
                {
                    candle_core::bail!(
                        "decode-attention: a cache the tiled tree kernel cannot stage"
                    );
                }
                let Some(p) = partial.as_mut() else {
                    candle_core::bail!("decode-attention: the tiled tree kernel writes partials");
                };
                let tree = self.tree.as_ref().expect("tiled is a tree's");
                let func = dev.get_or_load_custom_func(
                    if d <= 128 {
                        "tree_attn_tiled_f16kv_d128"
                    } else {
                        "tree_attn_tiled_f16kv_d256"
                    },
                    "swarmllm_decode_attn",
                    super::DECODE_ATTN_PTX,
                )?;
                let cfg = LaunchConfig {
                    grid_dim: (
                        (kv_b * n_kv_head) as u32,
                        n_chunks as u32,
                        (rows * n_rep).div_ceil(super::tiled_queries(d)) as u32,
                    ),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let rows_i = rows as i32;
                let (vis_words_i, base_i) = (
                    crate::inference::draft_tree::VISIBILITY_WORDS as i32,
                    (tree.base - skip) as i32,
                );
                let null: u64 = 0;
                let mut builder = func.builder();
                builder.arg(&q);
                builder.arg(k);
                builder.arg(v);
                builder.arg(p);
                builder.arg(&n_kv_i);
                builder.arg(&n_rep_i);
                builder.arg(&d_i);
                builder.arg(&s_len_i);
                builder.arg(&k_sb);
                builder.arg(&k_sh);
                builder.arg(&v_sb);
                builder.arg(&v_sh);
                builder.arg(&self.scale);
                builder.arg(&self.softcap);
                builder.arg(&rows_i);
                match &vis {
                    Some(v) => builder.arg(v),
                    None => builder.arg(&null),
                };
                builder.arg(&vis_words_i);
                builder.arg(&base_i);
                // SAFETY: ffi. Shapes, strides, the word alignment and scope
                // (d % 32 == 0, d <= 256, n_rep <= 16) are checked by the
                // caller and above; the kernel reads `kv_b` batches ×
                // `n_kv_head` heads × `s_len` rows of `d` halves from the
                // views by the strides passed, `rows * n_head * d` query floats
                // and `rows` rows of the visibility's words, and writes every
                // (row, head, chunk, query head) partial of `d + 2`.
                unsafe { builder.launch(cfg) }.w()?;
            } else {
                let func = dev.get_or_load_custom_func(
                    match cache {
                        CacheViews::F32(..) => "decode_attn_f32",
                        CacheViews::F16(..) => "decode_attn_f16kv",
                    },
                    "swarmllm_decode_attn",
                    super::DECODE_ATTN_PTX,
                )?;
                let cfg = LaunchConfig {
                    grid_dim: ((b * n_kv_head) as u32, n_chunks as u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let chunk_i = chunk as i32;
                let (rows_i, vis_words_i, base_i) = (
                    rows as i32,
                    crate::inference::draft_tree::VISIBILITY_WORDS as i32,
                    self.tree.as_ref().map_or(0, |t| t.base) as i32,
                );
                // A null pointer for whichever output this launch does not write,
                // and for the visibility of a call with no tree.
                let null: u64 = 0;
                {
                    let mut builder = func.builder();
                    builder.arg(&q);
                    match &cache {
                        CacheViews::F32(k, v) => {
                            builder.arg(k);
                            builder.arg(v);
                        }
                        CacheViews::F16(k, v) => {
                            builder.arg(k);
                            builder.arg(v);
                        }
                    }
                    match partial.as_mut() {
                        None => {
                            builder.arg(&mut out);
                            builder.arg(&null);
                        }
                        Some(p) => {
                            builder.arg(&null);
                            builder.arg(p);
                        }
                    }
                    builder.arg(&n_kv_i);
                    builder.arg(&n_rep_i);
                    builder.arg(&d_i);
                    builder.arg(&s_len_i);
                    builder.arg(&k_sb);
                    builder.arg(&k_sh);
                    builder.arg(&v_sb);
                    builder.arg(&v_sh);
                    builder.arg(&self.scale);
                    builder.arg(&self.softcap);
                    builder.arg(&chunk_i);
                    builder.arg(&rows_i);
                    match &vis {
                        Some(v) => builder.arg(v),
                        None => builder.arg(&null),
                    };
                    builder.arg(&vis_words_i);
                    builder.arg(&base_i);
                    // SAFETY: ffi. Shapes, strides and scope (d % 32 == 0, d <= 256,
                    // n_rep <= 16, chunk <= 64) are checked by the caller and above; the kernel
                    // reads within the views (a tree's visibility: `rows` rows of
                    // its words, the cache past its base within them, checked
                    // above) and writes `out` or `partial`, each sized for exactly
                    // what it assigns.
                    unsafe { builder.launch(cfg) }.w()?;
                }
            }
            if let Some(p) = partial.as_ref() {
                let combine = dev.get_or_load_custom_func(
                    "decode_attn_combine_f32",
                    "swarmllm_decode_attn",
                    super::DECODE_ATTN_PTX,
                )?;
                let cfg = LaunchConfig {
                    grid_dim: ((b * n_head) as u32, 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let n_chunks_i = n_chunks as i32;
                let past_runs_i = past.map_or(0, |p| p.runs) as i32;
                let null: u64 = 0;
                let mut builder = combine.builder();
                builder.arg(p);
                builder.arg(&mut out);
                builder.arg(&n_kv_i);
                builder.arg(&n_rep_i);
                builder.arg(&d_i);
                builder.arg(&n_chunks_i);
                match &past_out {
                    Some((o, lse)) => {
                        builder.arg(o);
                        builder.arg(lse);
                    }
                    None => {
                        builder.arg(&null);
                        builder.arg(&null);
                    }
                };
                builder.arg(&past_runs_i);
                // SAFETY: ffi. `partial` holds `n_chunks` rows of `d + 2` per
                // (group, head), all written above; `out` is `b * n_head * d`;
                // flash's `past_runs` runs of `rows * n_rep` query vectors × KV
                // heads (output × `d`), written above with the call checked to
                // be one K/V batch's.
                unsafe { builder.launch(cfg) }.w()?;
            }
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from_dims(&[b, n_head, 1, d]),
            ))
        }
    }
}

/// 8-lane dot product (independent accumulators so LLVM vectorises it).
#[inline(always)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (x, y) in ca.iter().zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0f32;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    acc.iter().sum::<f32>() + tail
}

/// `y += a * x` over equal-length slices.
#[inline(always)]
pub(crate) fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The largest gap, NaN if any gap is: `f32::max` drops a NaN, so a kernel
    /// answering NaN would pass a tolerance folded with it (gotcha #882).
    fn nan_max(worst: f32, gap: f32) -> f32 {
        if worst.is_nan() || gap.is_nan() {
            f32::NAN
        } else {
            worst.max(gap)
        }
    }

    /// The kernel against the exact composition it replaces: grouped matmul →
    /// scaled softmax → matmul, on the shapes llama-3.2-3b decodes with.
    /// How the mask of a case is built.
    #[derive(Clone, Copy, Debug)]
    enum Masked {
        No,
        /// The newest position hidden.
        Last,
        /// The first `CHUNK + 44` positions hidden: a whole chunk contributes
        /// nothing and the next one only partly — the -inf merge path.
        LeadingChunk,
    }

    #[test]
    fn decode_kernel_matches_the_matmul_path() {
        let dev = Device::Cpu;
        for (b, n_head, n_kv_head, s_len, softcap, masked) in [
            (1usize, 24usize, 8usize, 37usize, None, Masked::No),
            (1, 24, 8, 920, None, Masked::Last),
            (1, 32, 32, 300, None, Masked::No),
            (1, 8, 2, 64, Some(50.0f32), Masked::Last),
            // Several chunks, the last one partial, n_rep = 7 (Qwen2.5-7B).
            (1, 28, 4, 2055, None, Masked::No),
            (1, 24, 8, 1000, None, Masked::LeadingChunk),
            (1, 8, 2, 700, Some(30.0), Masked::LeadingChunk),
            // Exactly one chunk, and one chunk plus one position.
            (1, 24, 8, CHUNK, None, Masked::No),
            (1, 24, 8, CHUNK + 1, None, Masked::Last),
            (2, 24, 8, 600, None, Masked::Last),
        ] {
            let d = 128;
            let q = Tensor::randn(0f32, 1.0, (b, n_head, 1, d), &dev).unwrap();
            // K/V as a `narrow` of a larger buffer, the way KvCache hands them out.
            let kbuf = Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 17, d), &dev).unwrap();
            let vbuf = Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 17, d), &dev).unwrap();
            let k = kbuf.narrow(2, 0, s_len).unwrap();
            let v = vbuf.narrow(2, 0, s_len).unwrap();
            let mask = match masked {
                Masked::No => None,
                Masked::Last => {
                    let mut m = vec![0f32; s_len];
                    m[s_len - 1] = f32::NEG_INFINITY;
                    Some(Tensor::from_vec(m, (1, s_len), &dev).unwrap())
                }
                Masked::LeadingChunk => {
                    let mut m = vec![0f32; s_len];
                    m[..CHUNK + 44].fill(f32::NEG_INFINITY);
                    Some(Tensor::from_vec(m, (1, s_len), &dev).unwrap())
                }
            };
            let scale = 1.0 / (d as f32).sqrt();

            // Reference: the grouped-matmul composition.
            let n_rep = n_head / n_kv_head;
            let qg = q.reshape((b, n_kv_head, n_rep, d)).unwrap();
            let att = qg.matmul(&k.t().unwrap()).unwrap();
            let att = (att * scale as f64).unwrap();
            let att = match softcap {
                Some(c) => ((att / c as f64).unwrap().tanh().unwrap() * c as f64).unwrap(),
                None => att,
            };
            let att = match &mask {
                Some(m) => att.broadcast_add(m).unwrap(),
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
            let want = att
                .matmul(&v.contiguous().unwrap())
                .unwrap()
                .reshape((b, n_head, 1, d))
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            let got = gqa_decode_attention_cpu(&q, &k, &v, mask.as_ref(), scale, softcap, 0)
                .unwrap()
                .expect("kernel applies to these inputs")
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got.len(), want.len());
            // Two f32 summation orders agree to ~1e-7 absolute on outputs of
            // order 0.1-1; a wrong head or a dropped position is O(0.3). Bound
            // the absolute error and the relative error away from zero, so a
            // near-cancelling output does not fail on noise.
            let worst_abs = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, nan_max);
            let worst_rel = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(0.05))
                .fold(0f32, nan_max);
            assert!(
                worst_abs < 1e-5 && worst_rel < 1e-4,
                "b={b} heads {n_head}/{n_kv_head} S={s_len} softcap={softcap:?} masked={masked:?}: worst abs {worst_abs} rel {worst_rel}"
            );
        }
    }

    /// The card kernel against the matmul composition it replaces, computed on
    /// the processor from the same numbers. ⚠ Gated, so no default build runs
    /// it: `cargo test --features candle-cuda --lib decode_attn -- --nocapture`,
    /// and look for the SKIPPED line before believing a pass.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn cuda_decode_kernel_matches_the_matmul_path() {
        let card = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED cuda decode attention: no CUDA device ({e})");
                return;
            }
        };
        let cpu = Device::Cpu;
        for case in [
            (1usize, 24usize, 8usize, 37usize, 128usize, None),
            (1, 28, 4, 200, 128, None),
            // Several chunks, the last one partial, n_rep = 7 (Qwen2.5-7B).
            (1, 28, 4, 2055, 128, None),
            (1, 32, 32, 300, 128, None),
            (1, 32, 32, 300, 64, None),
            (1, 8, 4, 700, 256, Some(50.0f32)),
            (1, 32, 8, 8192, 128, None),
            // Exactly one chunk, and one chunk plus one position.
            (1, 24, 8, CUDA_CHUNK, 128, None),
            (1, 24, 8, CUDA_CHUNK + 1, 128, None),
            (2, 24, 8, 600, 128, None),
        ]
        .into_iter()
        // Every shape twice: over the f32 cache, and over the half cache a card
        // keeps (FUTURE_WORK #194) — the reference then computed from the same
        // rounded values, so the two arms differ in nothing but the read.
        .flat_map(|case| {
            [
                (case, candle_core::DType::F32),
                (case, candle_core::DType::F16),
            ]
        }) {
            let ((b, n_head, n_kv_head, s_len, d, softcap), cache_dtype) = case;
            let q = Tensor::randn(0f32, 1.0, (b, n_head, 1, d), &cpu).unwrap();
            let as_stored = |t: Tensor| {
                t.to_dtype(cache_dtype)
                    .unwrap()
                    .to_dtype(candle_core::DType::F32)
                    .unwrap()
            };
            let kbuf =
                as_stored(Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 17, d), &cpu).unwrap());
            let vbuf =
                as_stored(Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 17, d), &cpu).unwrap());
            let scale = 1.0 / (d as f32).sqrt();

            let (k, v) = (
                kbuf.narrow(2, 0, s_len).unwrap(),
                vbuf.narrow(2, 0, s_len).unwrap(),
            );
            let n_rep = n_head / n_kv_head;
            let qg = q.reshape((b, n_kv_head, n_rep, d)).unwrap();
            let att = (qg.matmul(&k.t().unwrap()).unwrap() * scale as f64).unwrap();
            let att = match softcap {
                Some(c) => ((att / c as f64).unwrap().tanh().unwrap() * c as f64).unwrap(),
                None => att,
            };
            let att = candle_nn::ops::softmax_last_dim(&att).unwrap();
            let want = att
                .matmul(&v.contiguous().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            // On the card: the cache as a `narrow` of a larger buffer, the way
            // `KvCache` hands it out.
            let on_card = |t: &Tensor| {
                t.to_dtype(cache_dtype)
                    .unwrap()
                    .to_device(&card)
                    .unwrap()
                    .narrow(2, 0, s_len)
                    .unwrap()
            };
            let (qc, kc, vc) = (q.to_device(&card).unwrap(), on_card(&kbuf), on_card(&vbuf));
            let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            // The entry point as callers reach it, then every chunk the kernel
            // is built for (`SWARMLLM_DECODE_ATTN_CHUNK`).
            let mut arms = vec![(
                "entry point".to_string(),
                flat(
                    gqa_decode_attention_cuda(&qc, &kc, &vc, None, scale, softcap)
                        .unwrap()
                        .expect("the card kernel applies to these inputs"),
                ),
            )];
            for chunk in CARD_CHUNKS {
                let op = cuda::DecodeAttn {
                    scale,
                    softcap: softcap.unwrap_or(0.0),
                    combine_one_chunk: true,
                    chunk: Some(chunk),
                    tree: None,
                };
                arms.push((
                    format!("{chunk}-position chunks"),
                    flat(qc.apply_op3_no_bwd(&kc, &vc, &op).unwrap()),
                ));
            }
            for (arm, got) in arms {
                assert_eq!(got.len(), want.len());
                let worst_abs = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, nan_max);
                let worst_rel = got
                    .iter()
                    .zip(&want)
                    .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(0.05))
                    .fold(0f32, nan_max);
                assert!(
                    worst_abs < 1e-5 && worst_rel < 1e-4,
                    "{arm}: b={b} heads {n_head}/{n_kv_head} S={s_len} d={d} softcap={softcap:?} cache {cache_dtype:?}: worst abs {worst_abs} rel {worst_rel}"
                );
            }
        }
    }

    /// A tree of guesses' attention on the card (#134) against the path it
    /// replaces: the matmul composition over the tree's additive mask
    /// (`RowVisibility::mask`, the processor's `matmul_attention`), from the
    /// same numbers. Drafter and 7B head layouts, a later level of a tree whose
    /// first rows are cached, the last level of a tree of every row one may
    /// hold (bits in every word), a one-row tree, a soft cap, a long context;
    /// the f32 and the half cache; the query as the transposed view a layer
    /// hands over. A forward past [`TREE_KERNEL_MOST_ROWS`] is declined. The entry
    /// point `standard_attention` reaches under a tree's scope must be the
    /// kernel's own answer BIT FOR BIT — the matmul path on the card would
    /// pass the tolerance too, so only that proves the kernel ran.
    ///
    /// ⚠ Gated: run it on a card with `--nocapture` and look for the SKIPPED
    /// line before believing a pass.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn cuda_tree_kernel_matches_the_masked_matmul_path() {
        use crate::inference::draft_tree::{DraftTree, MAX_TREE_ROWS};
        let card = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED cuda tree attention: no CUDA device ({e})");
                return;
            }
        };
        let cpu = Device::Cpu;
        // (n_head, n_kv_head, d, base, tree rows, rows already cached, softcap)
        for case in [
            (14usize, 2usize, 64usize, 37usize, 8usize, 0usize, None),
            (14, 2, 64, 300, 40, 32, None),
            (28, 4, 128, 600, 8, 0, None),
            (8, 4, 256, 100, 7, 0, Some(50.0f32)),
            (14, 2, 64, 5, MAX_TREE_ROWS, MAX_TREE_ROWS - 8, None),
            (14, 2, 64, 20, 1, 0, None),
            (32, 8, 128, 2000, 8, 0, None),
            // Past the per-row kernel's rows: the tiled kernel's over a half
            // cache — a 7B's near-half check of a flow's 24 rows, a 64-row
            // tree, every row a tree may hold, a later level after cached
            // rows, a head width of each tiled entry with a part-filled tile
            // and a part-filled word row (96: 48 words; 160: 80).
            (28, 4, 128, 600, 24, 0, None),
            (14, 2, 64, 300, 64, 0, None),
            (14, 2, 64, 5, MAX_TREE_ROWS, 0, None),
            (32, 8, 128, 2000, 40, 8, None),
            (8, 4, 256, 100, 20, 0, Some(50.0f32)),
            (12, 4, 96, 50, 16, 0, None),
            (10, 2, 160, 40, 12, 0, None),
        ]
        .into_iter()
        .flat_map(|case| {
            [
                (case, candle_core::DType::F32),
                (case, candle_core::DType::F16),
            ]
        }) {
            let ((n_head, n_kv_head, d, base, tree_rows, before, softcap), cache_dtype) = case;
            // A branching tree, the same every run: every fifth row starts
            // afresh, the rest follow an earlier row.
            let mut seed = 0x9e37_79b9u64 ^ (tree_rows as u64);
            let parents = (0..tree_rows)
                .map(|i| {
                    seed = seed
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add(1_442_695_040_888_963_407);
                    (i % 5 != 0).then(|| ((seed >> 33) as usize % i) as u16)
                })
                .collect();
            let tree = DraftTree::new(base, parents).unwrap();
            let seq = tree_rows - before;
            let (positions, seen) = tree.rows_for(base + before, seq).unwrap();
            let kv_len = base + tree_rows;
            let mask = Tensor::from_slice(&seen.mask(), (seq, kv_len), &cpu).unwrap();

            let q = Tensor::randn(0f32, 1.0, (1, seq, n_head, d), &cpu)
                .unwrap()
                .transpose(1, 2)
                .unwrap();
            let as_stored = |t: Tensor| {
                t.to_dtype(cache_dtype)
                    .unwrap()
                    .to_dtype(candle_core::DType::F32)
                    .unwrap()
            };
            let kbuf =
                as_stored(Tensor::randn(0f32, 1.0, (1, n_kv_head, kv_len + 17, d), &cpu).unwrap());
            let vbuf =
                as_stored(Tensor::randn(0f32, 1.0, (1, n_kv_head, kv_len + 17, d), &cpu).unwrap());
            let (k, v) = (
                kbuf.narrow(2, 0, kv_len).unwrap(),
                vbuf.narrow(2, 0, kv_len).unwrap(),
            );
            let want = crate::inference::layers::matmul_attention(
                &q,
                &k,
                &v,
                Some(&mask),
                d,
                n_head,
                n_kv_head,
                softcap,
            )
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

            let on_card = |t: &Tensor| {
                t.to_dtype(cache_dtype)
                    .unwrap()
                    .to_device(&card)
                    .unwrap()
                    .narrow(2, 0, kv_len)
                    .unwrap()
            };
            let (qc, kc, vc) = (q.to_device(&card).unwrap(), on_card(&kbuf), on_card(&vbuf));
            let vis = Tensor::from_slice(
                &seen.words,
                (seq, crate::inference::draft_tree::VISIBILITY_WORDS),
                &card,
            )
            .unwrap();
            // The layer's scale (f64, then f32): at d 96 the f32 quotient is one
            // step away, and the entry point below must match bit for bit.
            let scale = crate::inference::attn_softmax::scale_from_head_dim(d) as f32;
            let flat = |t: Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let Some(entry) =
                gqa_tree_attention_cuda(&qc, &kc, &vc, &vis, base, scale, softcap).unwrap()
            else {
                // Only a forward past the per-row kernel's rows that the tiled
                // kernel does not take (an f32 cache) is declined.
                assert!(
                    seq > TREE_KERNEL_MOST_ROWS && !tree_tiles(seq, &kc, &vc),
                    "the tree kernel declined {seq} rows (heads {n_head}/{n_kv_head} d={d} cache {cache_dtype:?})"
                );
                continue;
            };
            let entry = flat(entry);
            if seq == TREE_KERNEL_MOST_ROWS && !tree_tiles(seq + 1, &kc, &vc) {
                // One row more is the matmul path's.
                let wider = Tensor::randn(0f32, 1.0, (1, n_head, seq + 1, d), &card).unwrap();
                let wider_vis = Tensor::zeros(
                    (seq + 1, crate::inference::draft_tree::VISIBILITY_WORDS),
                    candle_core::DType::U32,
                    &card,
                )
                .unwrap();
                assert!(gqa_tree_attention_cuda(
                    &wider,
                    &kc,
                    &vc,
                    &wider_vis,
                    base.saturating_sub(1),
                    scale,
                    softcap
                )
                .unwrap()
                .is_none());
            }
            let through_attention = {
                let _tree = crate::inference::layers::RowPositions::for_tree(positions, seen);
                flat(
                    crate::inference::layers::standard_attention(
                        &qc,
                        &kc,
                        &vc,
                        Some(&mask.to_device(&card).unwrap()),
                        d,
                        n_head,
                        n_kv_head,
                        softcap,
                        0,
                    )
                    .unwrap(),
                )
            };
            assert!(
                entry
                    .iter()
                    .zip(&through_attention)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "standard_attention under a tree's scope is not the tree kernel's answer \
                 (heads {n_head}/{n_kv_head} d={d} base {base} rows {tree_rows} cache {cache_dtype:?})"
            );
            // Did flash read the context at the entry point (its default)?
            let n_rep = n_head / n_kv_head;
            let entry_on_flash = tree_tiles(seq, &kc, &vc)
                && past_runs(seq, n_kv_head, n_rep, d, base, &card).is_some();
            let mut arms = vec![("entry point".to_string(), entry, entry_on_flash)];
            // Each per-row chunk, and the tiled kernel over a half cache —
            // alone, and after flash read the context in a run, in runs that
            // leave a part of it to the tiled kernel, and in the most runs.
            let half = cache_dtype == candle_core::DType::F16;
            let tiled_arm = half.then_some((None, true, None));
            let flash_arms = [1usize, 3, 7, PAST_MOST_RUNS]
                .into_iter()
                .filter(|&runs| half && cfg!(feature = "flash-attn") && base >= runs)
                .map(|runs| (None, true, Some(runs)));
            let q_rows = qc
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape((seq, n_head, 1, d))
                .unwrap();
            for (chunk, tiled, runs) in CARD_CHUNKS
                .into_iter()
                .map(|c| (Some(c), false, None))
                .chain(tiled_arm)
                .chain(flash_arms)
            {
                let op = cuda::DecodeAttn {
                    scale,
                    softcap: softcap.unwrap_or(0.0),
                    combine_one_chunk: true,
                    chunk,
                    tree: Some(cuda::TreeOperands {
                        visibility: vis.clone(),
                        rows: seq,
                        base,
                        tiled,
                        past: runs.map(|runs| cuda::PastOnFlash {
                            q: flash_queries(&q_rows, seq, n_kv_head, n_rep, d).unwrap(),
                            runs,
                            run_len: base / runs,
                        }),
                    }),
                };
                let got = q_rows
                    .apply_op3_no_bwd(&kc, &vc, &op)
                    .unwrap()
                    .reshape((1, seq, n_head, d))
                    .unwrap()
                    .transpose(1, 2)
                    .unwrap();
                arms.push((
                    match (chunk, runs) {
                        (Some(c), _) => format!("{c}-position chunks"),
                        (None, None) => "tiled".to_string(),
                        (None, Some(r)) => format!("tiled after flash in {r} runs"),
                    },
                    flat(got),
                    runs.is_some(),
                ));
            }
            // Flash reads half queries (as a chain's check on flash does) and
            // sums its weights in half: its arms are held to the matmul path
            // over the same half queries, at a half's precision.
            let want_half_q = half.then(|| {
                crate::inference::layers::matmul_attention(
                    &q.to_dtype(candle_core::DType::F16)
                        .unwrap()
                        .to_dtype(candle_core::DType::F32)
                        .unwrap(),
                    &k,
                    &v,
                    Some(&mask),
                    d,
                    n_head,
                    n_kv_head,
                    softcap,
                )
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
            });
            for (arm, got, on_flash) in arms {
                let (want, most_abs, most_rel) = match (&want_half_q, on_flash) {
                    (Some(w), true) => (w, 2e-3, 2e-2),
                    _ => (&want, 1e-5, 1e-4),
                };
                assert_eq!(got.len(), want.len());
                let worst_abs = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, nan_max);
                let worst_rel = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(0.05))
                    .fold(0f32, nan_max);
                if on_flash {
                    eprintln!(
                        "FLASH {arm}: heads {n_head}/{n_kv_head} d={d} base {base} rows {tree_rows}: worst abs {worst_abs} rel {worst_rel}"
                    );
                }
                assert!(
                    worst_abs < most_abs && worst_rel < most_rel,
                    "{arm}: heads {n_head}/{n_kv_head} d={d} base {base} rows {tree_rows} after {before} softcap={softcap:?} cache {cache_dtype:?}: worst abs {worst_abs} rel {worst_rel}"
                );
            }
        }
    }

    /// A tree's call with NO mask that the tree kernel cannot take fails —
    /// never runs as an ordinary decode or matmul call over every cached row.
    /// A captured tree level builds no mask (`RecordedStep::tree_staged`), so
    /// such a call would read its siblings: a one-row call the scope does not
    /// describe passed the first version of the check (it asked for more than
    /// one row and a scope that matched the call). With the mask given, both
    /// calls run.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn a_tree_call_the_kernel_cannot_take_without_a_mask_fails() {
        use crate::inference::draft_tree::DraftTree;
        let card = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED a tree's unmasked call: no CUDA device ({e})");
                return;
            }
        };
        let (n_head, n_kv_head, d, base) = (14usize, 2usize, 64usize, 20usize);
        let call = |rows: usize,
                    keys: usize,
                    scope_rows: usize,
                    with_mask: bool,
                    cache_dtype: candle_core::DType| {
            let parents = (0..scope_rows)
                .map(|i| i.checked_sub(1).map(|p| p as u16))
                .collect();
            let tree = DraftTree::new(base, parents).unwrap();
            let (positions, seen) = tree.rows_for(base, scope_rows).unwrap();
            let mask = with_mask.then(|| {
                let line: Vec<f32> = (0..rows * keys).map(|_| 0.0).collect();
                Tensor::from_slice(&line, (rows, keys), &card).unwrap()
            });
            let q = Tensor::randn(0f32, 1.0, (1, n_head, rows, d), &card).unwrap();
            let cache = || {
                Tensor::randn(0f32, 1.0, (1, n_kv_head, keys, d), &card)
                    .unwrap()
                    .to_dtype(cache_dtype)
                    .unwrap()
            };
            let (k, v) = (cache(), cache());
            let _tree = crate::inference::layers::RowPositions::for_tree(positions, seen);
            crate::inference::layers::standard_attention(
                &q,
                &k,
                &v,
                mask.as_ref(),
                d,
                n_head,
                n_kv_head,
                None,
                0,
            )
        };
        let (f16, f32) = (candle_core::DType::F16, candle_core::DType::F32);
        let wide = TREE_KERNEL_MOST_ROWS + 1;
        // Past the per-row kernel's rows over an f32 cache (no tiled kernel
        // reads one), its scope matching the call.
        assert!(call(wide, base + wide, wide, false, f32).is_err());
        assert!(call(wide, base + wide, wide, true, f32).is_ok());
        // The same rows over the half cache are the tiled kernel's.
        assert_eq!(
            call(wide, base + wide, wide, false, f16).is_ok(),
            tree_tiled_from_rows().is_some_and(|from| wide >= from)
        );
        // One row over more keys than the scope's tree ends at.
        assert!(call(1, base + 3, 1, false, f16).is_err());
        assert!(call(1, base + 3, 1, true, f16).is_ok());
        // The kernel's own call runs without one.
        assert!(call(1, base + 1, 1, false, f16).is_ok());
    }

    /// What a tree's attention costs on the card, the tree kernel against the
    /// matmul path over its mask (`standard_attention` under
    /// `SWARMLLM_TREE_ATTN=standard`'s branch: widen a half cache, two cuBLAS
    /// products, the mask, a softmax), by rows, for the 0.5B drafter's heads
    /// (14/2, d 64) and a 7B's (28/4, d 128), over a half cache. Min of 30
    /// calls, each synchronised. Where the kernel's per-row re-read of the
    /// cache overtakes the matmul path's fixed cost sets the kernel's row cap.
    /// `cargo test --release --features cuda --lib tree_attention_timing --
    /// --ignored --nocapture`.
    #[cfg(feature = "candle-cuda")]
    #[test]
    #[ignore]
    fn tree_attention_timing() {
        let card = Device::new_cuda(0).expect("a card");
        for (n_head, n_kv_head, d) in [(14usize, 2usize, 64usize), (28, 4, 128)] {
            for s_len in [300usize, 900] {
                for rows in [1usize, 2, 4, 8, 12, 16, 24, 32, 64] {
                    let base = s_len - rows;
                    let q = Tensor::randn(0f32, 1.0, (1, rows, n_head, d), &card)
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap();
                    let cache = || {
                        Tensor::randn(0f32, 1.0, (1, n_kv_head, s_len + 64, d), &card)
                            .unwrap()
                            .to_dtype(candle_core::DType::F16)
                            .unwrap()
                            .narrow(2, 0, s_len)
                            .unwrap()
                    };
                    let (k, v) = (cache(), cache());
                    let parents = (0..rows)
                        .map(|i| i.checked_sub(1).map(|p| p as u16))
                        .collect();
                    let tree = crate::inference::draft_tree::DraftTree::new(base, parents).unwrap();
                    let (_, seen) = tree.rows_for(base, rows).unwrap();
                    let vis = Tensor::from_slice(
                        &seen.words,
                        (rows, crate::inference::draft_tree::VISIBILITY_WORDS),
                        &card,
                    )
                    .unwrap();
                    let mask = Tensor::from_slice(&seen.mask(), (rows, s_len), &card).unwrap();
                    let scale = 1.0 / (d as f32).sqrt();
                    let time = |f: &dyn Fn() -> Tensor| {
                        let mut best = f64::MAX;
                        for i in 0..33 {
                            card.synchronize().unwrap();
                            let t = std::time::Instant::now();
                            let out = f();
                            card.synchronize().unwrap();
                            drop(out);
                            if i >= 3 {
                                best = best.min(t.elapsed().as_secs_f64() * 1e6);
                            }
                        }
                        best
                    };
                    // The op itself, past the entry point's row cap: one
                    // block a row, and the tiled kernel.
                    let op_timed = |tiled: bool| {
                        time(&|| {
                            let op = cuda::DecodeAttn {
                                scale,
                                softcap: 0.0,
                                combine_one_chunk: true,
                                chunk: None,
                                tree: Some(cuda::TreeOperands {
                                    visibility: vis.clone(),
                                    rows,
                                    base,
                                    tiled,
                                    past: None,
                                }),
                            };
                            q.transpose(1, 2)
                                .unwrap()
                                .contiguous()
                                .unwrap()
                                .reshape((rows, n_head, 1, d))
                                .unwrap()
                                .apply_op3_no_bwd(&k, &v, &op)
                                .unwrap()
                        })
                    };
                    let (kernel, tiled) = (op_timed(false), op_timed(true));
                    let matmul = time(&|| {
                        let (kw, vw) = (
                            k.to_dtype(candle_core::DType::F32).unwrap(),
                            v.to_dtype(candle_core::DType::F32).unwrap(),
                        );
                        crate::inference::layers::matmul_attention(
                            &q,
                            &kw,
                            &vw,
                            Some(&mask),
                            d,
                            n_head,
                            n_kv_head,
                            None,
                        )
                        .unwrap()
                    });
                    eprintln!(
                        "TIMING heads {n_head}/{n_kv_head} d {d} keys {s_len} rows {rows}: \
                         kernel {kernel:.1} us, tiled {tiled:.1} us, matmul path {matmul:.1} us"
                    );
                }
            }
        }
    }

    /// What a tiled tree's attention costs with its context on flash, by the
    /// runs flash splits it into, against the tiled kernel reading it all —
    /// the 0.5B drafter's heads and a 7B's, a flow's row counts, a short, a
    /// middling and a long context. Min of 30 calls, each synchronised, the
    /// query conversion included. `cargo test --release --features cuda --lib
    /// tree_context_on_flash_timing -- --ignored --nocapture`.
    #[cfg(all(feature = "candle-cuda", feature = "flash-attn"))]
    #[test]
    #[ignore]
    fn tree_context_on_flash_timing() {
        let card = Device::new_cuda(0).expect("a card");
        let sms = card_sms(&card).expect("the card's multiprocessors");
        for (n_head, n_kv_head, d) in [(14usize, 2usize, 64usize), (28, 4, 128)] {
            let n_rep = n_head / n_kv_head;
            for base in [600usize, 1400, 4000] {
                for rows in [8usize, 16, 24, 32, 64] {
                    let s_len = base + rows;
                    let q = Tensor::randn(0f32, 1.0, (1, rows, n_head, d), &card).unwrap();
                    let cache = || {
                        Tensor::randn(0f32, 1.0, (1, n_kv_head, s_len + 64, d), &card)
                            .unwrap()
                            .to_dtype(candle_core::DType::F16)
                            .unwrap()
                            .narrow(2, 0, s_len)
                            .unwrap()
                    };
                    let (k, v) = (cache(), cache());
                    let parents = (0..rows)
                        .map(|i| i.checked_sub(1).map(|p| p as u16))
                        .collect();
                    let tree = crate::inference::draft_tree::DraftTree::new(base, parents).unwrap();
                    let (_, seen) = tree.rows_for(base, rows).unwrap();
                    let vis = Tensor::from_slice(
                        &seen.words,
                        (rows, crate::inference::draft_tree::VISIBILITY_WORDS),
                        &card,
                    )
                    .unwrap();
                    let scale = 1.0 / (d as f32).sqrt();
                    let q_rows = q.reshape((rows, n_head, 1, d)).unwrap();
                    let time = |runs: Option<usize>| {
                        let mut best = f64::MAX;
                        for i in 0..33 {
                            card.synchronize().unwrap();
                            let t = std::time::Instant::now();
                            let op = cuda::DecodeAttn {
                                scale,
                                softcap: 0.0,
                                combine_one_chunk: true,
                                chunk: None,
                                tree: Some(cuda::TreeOperands {
                                    visibility: vis.clone(),
                                    rows,
                                    base,
                                    tiled: true,
                                    past: runs.map(|runs| cuda::PastOnFlash {
                                        q: flash_queries(&q_rows, rows, n_kv_head, n_rep, d)
                                            .unwrap(),
                                        runs,
                                        run_len: base / runs,
                                    }),
                                }),
                            };
                            let out = q_rows.apply_op3_no_bwd(&k, &v, &op).unwrap();
                            card.synchronize().unwrap();
                            drop(out);
                            if i >= 3 {
                                best = best.min(t.elapsed().as_secs_f64() * 1e6);
                            }
                        }
                        best
                    };
                    let auto = auto_past_runs(rows * n_rep, n_kv_head, sms);
                    let by_runs = [1usize, 2, 4, 8, 12, 16]
                        .into_iter()
                        .map(|runs| format!("{runs}: {:.1}", time(Some(runs))))
                        .collect::<Vec<_>>()
                        .join(", ");
                    eprintln!(
                        "FLASH heads {n_head}/{n_kv_head} d {d} context {base} rows {rows}: \
                         tiled {:.1} us; flash runs (auto {auto}) {by_runs} us",
                        time(None)
                    );
                }
            }
        }
    }

    /// A cache of ONE chunk merged by the combine kernel must equal the direct
    /// write BIT FOR BIT: the combine runs at every length so a captured decode
    /// step keeps its kernels as the cache passes one chunk
    /// ([`super::combine_every_length`]), and that must move the launch count
    /// and nothing a reply can see. Every length up to one chunk, both caches,
    /// a soft cap, several query heads a group.
    ///
    /// ⚠ Gated, like the test above: run it on a card with `--nocapture` and
    /// look for the SKIPPED line before believing a pass.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn one_chunk_merged_is_bit_identical_to_the_direct_write() {
        use candle_core::DType;
        let card = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED decode attention one-chunk merge: no CUDA device ({e})");
                return;
            }
        };
        let mut compared = 0;
        for (b, n_head, n_kv_head, d, softcap) in [
            (1usize, 24usize, 8usize, 128usize, None),
            (1, 8, 4, 256, Some(50.0f32)),
            (2, 32, 32, 64, None),
        ] {
            for s_len in [1, 2, 17, CUDA_CHUNK - 1, CUDA_CHUNK] {
                for cache_dtype in [DType::F32, DType::F16] {
                    let q = Tensor::randn(0f32, 1.0, (b, n_head, 1, d), &card).unwrap();
                    let cache = || {
                        Tensor::randn(0f32, 1.0, (b, n_kv_head, s_len + 9, d), &card)
                            .unwrap()
                            .to_dtype(cache_dtype)
                            .unwrap()
                            .narrow(2, 0, s_len)
                            .unwrap()
                    };
                    let (k, v) = (cache(), cache());
                    let run = |combine_one_chunk: bool| {
                        let op = cuda::DecodeAttn {
                            scale: 1.0 / (d as f32).sqrt(),
                            softcap: softcap.unwrap_or(0.0),
                            combine_one_chunk,
                            chunk: Some(CUDA_CHUNK),
                            tree: None,
                        };
                        q.apply_op3_no_bwd(&k, &v, &op)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap()
                    };
                    let (direct, merged) = (run(false), run(true));
                    let differ = direct
                        .iter()
                        .zip(&merged)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        differ,
                        0,
                        "b={b} heads {n_head}/{n_kv_head} S={s_len} d={d} softcap={softcap:?} \
                         cache {cache_dtype:?}: {differ} of {} differ",
                        direct.len()
                    );
                    compared += 1;
                }
            }
        }
        assert_eq!(compared, 3 * 5 * 2);
    }

    /// [`chunk_for`] at every point it was measured (FUTURE_WORK #309, 2026-10-19), and at its edges.
    #[test]
    fn the_card_chunk_grows_with_the_cache_and_the_heads() {
        // TinyLlama 1.1B: 4 KV heads.
        assert_eq!(chunk_for(4, 200), 16);
        assert_eq!(chunk_for(4, 1_700), 32);
        // Llama 3.2 3B: 8 KV heads.
        assert_eq!(chunk_for(8, 200), 16);
        assert_eq!(chunk_for(8, 600), 32);
        assert_eq!(chunk_for(8, 4_000), 64);
        // One KV head (Gemma 3 1B): the merge's chunk count binds, not the blocks.
        assert_eq!(chunk_for(1, 1_024), 16);
        assert_eq!(chunk_for(1, 1_025), 32);
        assert_eq!(chunk_for(1, 4_000), 64);
        // The edges: 64 chunks of 16 and 256 blocks are allowed, one more is not.
        assert_eq!(chunk_for(4, 1_024), 16);
        assert_eq!(chunk_for(4, 1_025), 32);
        assert_eq!(chunk_for(16, 256), 16);
        assert_eq!(chunk_for(16, 257), 32);
        // Past what any smaller chunk allows, the widest; and a one-position cache.
        assert_eq!(chunk_for(32, 100_000), CUDA_CHUNK);
        assert_eq!(chunk_for(1, 1), 16);
        // It only ever answers a chunk the kernel takes.
        for groups in [1, 2, 4, 8, 32, 64, 300] {
            for s_len in [1, 15, 16, 17, 255, 1_000, 5_000, 70_000] {
                assert!(CARD_CHUNKS.contains(&chunk_for(groups, s_len)));
            }
        }
    }

    #[test]
    fn the_kernel_declines_what_it_cannot_index() {
        let dev = Device::Cpu;
        let q = Tensor::randn(0f32, 1.0, (1, 8, 2, 64), &dev).unwrap(); // q_len 2
        let k = Tensor::randn(0f32, 1.0, (1, 8, 10, 64), &dev).unwrap();
        assert!(gqa_decode_attention_cpu(&q, &k, &k, None, 0.1, None, 0)
            .unwrap()
            .is_none());
        let q1 = Tensor::randn(0f32, 1.0, (1, 8, 1, 64), &dev).unwrap();
        // K whose (S, d) plane is not dense: a transposed view.
        let kt = k.transpose(2, 3).unwrap();
        assert!(gqa_decode_attention_cpu(&q1, &kt, &kt, None, 0.1, None, 0)
            .unwrap()
            .is_none());
    }
}
