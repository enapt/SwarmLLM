//! One query row's attention on the processor: the ONE arithmetic both CPU
//! kernels compute it with, whichever forward the row is in (#179).
//!
//! A row's attention used to depend on the forward that carried it. The decode
//! kernel (`decode_attn`, one position) and the prompt kernel (`prefill_attn`,
//! several) summed in different orders, and the prompt kernel sized its key
//! tiles by the FORWARD, not by what the row can see, so the vectorised `exp`
//! split each row's scores differently with the prompt's length. Each result
//! was within ~1e-6 of the exact answer. None was reproducible across forwards.
//! A prompt-cache hit is exactly such a change: the cached prefix is skipped, so
//! the last prompt token runs alone, as a decode step. Llama-3.2-3B at
//! `temperature: 0` then gave one reply on a fresh node and another on every
//! repeat (FUTURE_WORK #179), and the same 55-token prompt gave different logits
//! at position 54 read in a 55- or a 56-token forward.
//!
//! The rule is Thinking Machines' "batch invariance" ("Defeating Nondeterminism
//! in LLM Inference", 2025): a token's reduction order must be identical whether
//! 0 or many tokens of its sequence are in the KV cache, so split the keys by a
//! fixed SIZE, never a fixed count. SGLang's deterministic mode does the same
//! with fixed split-KV sizes. Here that means:
//!
//! * **A row reads the keys from the first to the last its mask leaves
//!   visible** ([`visible_range`]). A causal prompt row at position `p` and a
//!   decode step at `p` both read `0..=p`; a windowed row reads its window
//!   whether the keys before it were cut off (a decode step) or masked (a
//!   prompt). Keys the mask hides at either end are not read at all, rather
//!   than read as `-inf`.
//! * **Keys go in fixed [`CHUNK`]s of ABSOLUTE position** — the caller says
//!   where its first key sits (`key_origin`), because a windowed layer hands
//!   attention its keys from the window's start, which moves with the forward
//!   (`LayerWeights::keys_in_window`). Per chunk:
//!   scores `dot(q, k) * scale` (then soft-cap, then mask), the chunk's max,
//!   `exp` over exactly the row's slice of the chunk, the sum, then `Σ p·v`
//!   ([`axpy`] in key order). A row's chunks are folded in chunk order
//!   ([`merge_into`], flash-attention's online merge), whether they were
//!   computed one after another or side by side.
//! * **One dot product and one [`axpy`] ARE the arithmetic; every loop here
//!   computes them, whatever its blocking.** A product is a fused multiply-add
//!   wherever the processor has one; a dot product runs eight lanes over the
//!   dimension and adds them as `((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 +
//!   l7))`, the order the AVX2 kernel's horizontal adds take. The blocked loops
//!   change only how many rows and keys share a loaded vector
//!   (`the_vector_kernels_are_the_portable_arithmetic_exactly`), and
//!   `a_row_s_attention_does_not_depend_on_the_forward_it_is_in` pins the whole
//!   row bit for bit.
//!
//! The vector kernels are AVX2 + FMA, chosen at compile time as `fast_math`'s
//! are (gotcha #253: the x86-64-v3 and native builds have both); every other
//! build runs the portable twin. Each build is invariant in itself; two builds
//! need not agree with each other.
//!
//! What it does not cover: the matmul paths (`layers::matmul_attention`, where
//! a kernel declines, and every card) and the card's kernels. A card keeps
//! #179's behaviour; FUTURE_WORK #313.

use std::ops::Range;

/// Keys per chunk. Fixed, never derived from the thread count or the forward,
/// so every row splits its keys at the same places. 256 rows of K is 128 KB at
/// `d = 128`, inside L2.
pub(crate) const CHUNK: usize = 256;

/// One row's partial for one chunk: `[max, Σ exp, o[0..d]]`.
pub(crate) const fn partial_len(d: usize) -> usize {
    d + 2
}

/// The keys a row reads: from the first its mask leaves visible to one past
/// the last, or every key when there is no mask; empty when it sees none.
/// Masks here are additive with exact `-inf` for a hidden key
/// (`split::executor::causal_mask_with_spans`, `window_mask`), so a causal row
/// at position `p` reads `0..p + 1` whether the forward holds one position or
/// a whole prompt.
pub(crate) fn visible_range(mask_row: Option<&[f32]>, s_len: usize) -> Range<usize> {
    let Some(m) = mask_row else {
        return 0..s_len;
    };
    let m = &m[..s_len];
    match m.iter().position(|&x| x != f32::NEG_INFINITY) {
        None => 0..0,
        Some(first) => {
            first
                ..1 + m
                    .iter()
                    .rposition(|&x| x != f32::NEG_INFINITY)
                    .unwrap_or(first)
        }
    }
}

/// The absolute chunks keys `key_origin..key_origin + s_len` fall in.
pub(crate) fn chunks_spanned(key_origin: usize, s_len: usize) -> Range<usize> {
    if s_len == 0 {
        return 0..0;
    }
    key_origin / CHUNK..(key_origin + s_len - 1) / CHUNK + 1
}

/// The partials of `nr` query rows of one KV group against absolute chunk `ac`
/// (one of [`chunks_spanned`]).
///
/// `q`: the rows, `nr × d`, unscaled. `seen[r]`: row `r`'s [`visible_range`].
/// `k`, `v`: the group's planes, `s_len × d`, whose first key is at absolute
/// position `key_origin`. `mask(r)`: row `r`'s additive mask over all `s_len`
/// keys, if any. `scores`: scratch of at least `nr * CHUNK`. Writes `part`
/// (`nr × partial_len(d)`) whole; a row that sees no key of the chunk gets
/// max `-inf`, which [`merge_into`] skips.
#[allow(clippy::too_many_arguments)]
pub(crate) fn chunk_partials<'m>(
    q: &[f32],
    seen: &[Range<usize>],
    k: &[f32],
    v: &[f32],
    d: usize,
    key_origin: usize,
    ac: usize,
    mask: &dyn Fn(usize) -> Option<&'m [f32]>,
    scale: f32,
    softcap: Option<f32>,
    scores: &mut [f32],
    part: &mut [f32],
) {
    let nr = seen.len();
    let w = partial_len(d);
    let s_len = k.len() / d;
    assert!(q.len() >= nr * d && v.len() >= s_len * d);
    assert!(scores.len() >= nr * CHUNK && part.len() >= nr * w);
    // This chunk's keys, as indices into `k`: `c0..c1`.
    let c0 = (ac * CHUNK).max(key_origin) - key_origin;
    let c1 = ((ac + 1) * CHUNK).min(key_origin + s_len) - key_origin;
    // The keys of this chunk row `r` reads, as indices into `k`; empty for a
    // row that sees none of them or (below) whose every one is masked.
    let mut keys: Vec<Range<usize>> = seen
        .iter()
        .map(|s| {
            let (a, b) = (s.start.max(c0), s.end.min(c1));
            if a < b {
                a..b
            } else {
                c0..c0
            }
        })
        .collect();
    for p in part[..nr * w].chunks_exact_mut(w) {
        empty_partial(p);
    }
    // Row `r`'s score for key `j` sits at `scores[r * CHUNK + j - c0]`.

    // 1. Raw scores: a tile of rows against a tile of keys, both held in L1,
    //    so a key row is read from L2 once per row tile — not once per four
    //    rows (#119's defect: the cache re-read per query head) — rows four at a
    //    time inside it.
    let mut r = 0;
    while r < nr {
        let rt = (nr - r).min(ROW_TILE);
        let tile = keys_read(&keys[r..r + rt], c0);
        let mut kt = tile.start;
        while kt < tile.end {
            let kn = (tile.end - kt).min(KEY_TILE);
            let mut rr = r;
            while rr < r + rt {
                let rb = (r + rt - rr).min(4);
                let want = keys_read(&keys[rr..rr + rb], c0);
                let (from, to) = (want.start.max(kt), want.end.min(kt + kn));
                if from < to {
                    qk_rows(
                        &q[rr * d..(rr + rb) * d],
                        rb,
                        &k[from * d..to * d],
                        to - from,
                        d,
                        &mut scores[rr * CHUNK + from - c0..],
                    );
                }
                rr += rb;
            }
            kt += kn;
        }
        r += rt;
    }

    // 2. Per row: scale → soft-cap → mask, the max, exp, the sum — over
    //    exactly the row's keys, so the vectorised `exp` splits them the same
    //    way whichever forward the row is in.
    for (r, rk) in keys.iter_mut().enumerate() {
        if Range::is_empty(rk) {
            continue;
        }
        let sc = &mut scores[r * CHUNK + rk.start - c0..r * CHUNK + rk.end - c0];
        // One loop per case, so each vectorises; the operations on an
        // element are the same in every case: scale, soft-cap, mask.
        match (softcap, mask(r)) {
            (None, None) => sc.iter_mut().for_each(|x| *x *= scale),
            (None, Some(m)) => {
                for (x, &mj) in sc.iter_mut().zip(&m[rk.clone()]) {
                    *x = *x * scale + mj;
                }
            }
            (Some(cap), m) => {
                for (j, x) in rk.clone().zip(sc.iter_mut()) {
                    let mut y = cap * ((*x * scale) / cap).tanh();
                    if let Some(m) = m {
                        y += m[j];
                    }
                    *x = y;
                }
            }
        }
        let max = lane_max(sc);
        if max == f32::NEG_INFINITY {
            // Every key of the chunk masked for this row: it contributes nothing.
            *rk = c0..c0;
            continue;
        }
        sc.iter_mut().for_each(|x| *x -= max);
        crate::inference::fast_math::exp_inplace(sc);
        part[r * w] = max;
        part[r * w + 1] = lane_sum(sc);
    }

    // 3. o_r = Σ_j p_rj · v_j in key order.
    let span = keys_read(&keys, c0);
    if nr <= KEYS_OUTER_MAX_ROWS {
        // A decode step's few rows (one per query head of the group): each
        // value row is read once, in order, straight from memory, for every
        // row that sees it — the cache streams through once.
        for j in span {
            let vj = &v[j * d..(j + 1) * d];
            for (i, rk) in keys.iter().enumerate() {
                if rk.contains(&j) {
                    axpy(
                        scores[i * CHUNK + j - c0],
                        vj,
                        &mut part[i * w + 2..(i + 1) * w],
                    );
                }
            }
        }
        return;
    }
    // A prompt's rows: keys in sub-blocks small enough to stay in L1 while
    // every row of the task is accumulated over them, four rows at a time with
    // their outputs held in registers. A sub-block every row of the four reads
    // in full goes through the blocked loop; one at a row's edge, row by row.
    let mut a = span.start;
    while a < span.end {
        let b = (a + KEY_SUB).min(span.end);
        let mut r = 0;
        while r < nr {
            let rb = (nr - r).min(4);
            let block = &keys[r..r + rb];
            if block.iter().all(|k| k.start <= a && b <= k.end) {
                pv_rows(
                    &scores[r * CHUNK + a - c0..],
                    rb,
                    &v[a * d..b * d],
                    b - a,
                    d,
                    &mut part[r * w + 2..],
                    w,
                );
            } else {
                for (i, rk) in (r..r + rb).zip(block) {
                    let o = &mut part[i * w + 2..(i + 1) * w];
                    for j in a.max(rk.start)..b.min(rk.end) {
                        axpy(scores[i * CHUNK + j - c0], &v[j * d..(j + 1) * d], o);
                    }
                }
            }
            r += rb;
        }
        a = b;
    }
}

/// Rows per score tile and keys per tile: 32 query rows (16 KB at `d = 128`)
/// beside 16 key rows (8 KB) fit a 32 KB L1.
const ROW_TILE: usize = 32;
const KEY_TILE: usize = 16;
/// Keys per sub-block of a prompt's `Σ p·v` loop: 32 value rows, 16 KB at
/// `d = 128`.
const KEY_SUB: usize = 32;
/// Up to this many rows a chunk's `Σ p·v` walks the keys in order instead.
const KEYS_OUTER_MAX_ROWS: usize = 8;

/// The largest of `xs` (`-inf` for none) — exact whatever the order, so it
/// is taken eight lanes at a time.
fn lane_max(xs: &[f32]) -> f32 {
    let mut acc = [f32::NEG_INFINITY; 8];
    let (blocks, tail) = xs.as_chunks::<8>();
    for b in blocks {
        for (a, &x) in acc.iter_mut().zip(b) {
            if x > *a {
                *a = x;
            }
        }
    }
    let mut max = f32::NEG_INFINITY;
    for &x in acc.iter().chain(tail) {
        if x > max {
            max = x;
        }
    }
    max
}

/// `Σ xs` in ONE order: eight lanes over the slice in order, added as the
/// dot product's tree, then the elements past the last eight, in order. A
/// row's chunk sum is this in both kernels.
fn lane_sum(xs: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (blocks, tail) = xs.as_chunks::<8>();
    for b in blocks {
        for (a, &x) in acc.iter_mut().zip(b) {
            *a += x;
        }
    }
    let mut rest = 0f32;
    for &x in tail {
        rest += x;
    }
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + rest
}

/// The keys any of these rows reads: from the first start to the last end, or
/// empty (at `c0`) when none reads any.
fn keys_read(block: &[Range<usize>], c0: usize) -> Range<usize> {
    let from = block
        .iter()
        .filter(|k| !k.is_empty())
        .map(|k| k.start)
        .min();
    let to = block.iter().map(|k| k.end).max().unwrap_or(c0);
    from.map_or(c0..c0, |from| from..to)
}

/// A partial that has seen no key: max `-inf`, nothing summed. A row's merge
/// starts here.
pub(crate) fn empty_partial(p: &mut [f32]) {
    p[0] = f32::NEG_INFINITY;
    p[1..].fill(0.0);
}

/// Fold one chunk's partial `p` into a row's merge `acc` (both
/// `[max, Σ exp, o]`) — flash-attention's online form of flash-decoding's
/// reduction: `m = max(m_acc, m_p)`, then `acc · e^(m_acc − m) + p · e^(m_p − m)`.
/// Both kernels fold a row's chunks in chunk order through this, so a row
/// merges the same way whether its chunks were computed one after another
/// (a prompt task) or side by side (a decode step).
pub(crate) fn merge_into(acc: &mut [f32], p: &[f32]) {
    if p[0] == f32::NEG_INFINITY {
        return;
    }
    let m = acc[0].max(p[0]);
    // An empty `acc` has max -inf: its weight is e^-inf = 0, and the fold
    // takes `p` exactly (0·0 + y·1).
    let (a, b) = ((acc[0] - m).exp(), (p[0] - m).exp());
    acc[0] = m;
    for (x, &y) in acc[1..].iter_mut().zip(&p[1..]) {
        *x = *x * a + y * b;
    }
}

/// The row's attention from its merge: `o / Σ`, or zeros for a row that saw
/// no key.
pub(crate) fn finish_row(acc: &[f32], o: &mut [f32]) {
    if acc[0] == f32::NEG_INFINITY || acc[1] <= 0.0 {
        o.fill(0.0);
        return;
    }
    let inv = 1.0 / acc[1];
    for (x, &y) in o.iter_mut().zip(&acc[2..]) {
        *x = y * inv;
    }
}

/// `y += a · x`, element by element, each a fused multiply-add where the
/// processor has one. THE value-sum step: every `Σ p·v` here is this, in key
/// order.
pub(crate) fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    let n = x.len().min(y.len());
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    // SAFETY: AVX2 + FMA are compiled in; both slices hold `n` elements.
    unsafe {
        avx::axpy(a, x.as_ptr(), y.as_mut_ptr(), n)
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    portable::axpy(a, &x[..n], &mut y[..n])
}

/// `out[r * CHUNK + j] = dot(q_r, k_j)` for `rb` rows (≤ 4) and keys `0..keys`.
fn qk_rows(q: &[f32], rb: usize, k: &[f32], keys: usize, d: usize, out: &mut [f32]) {
    assert!(q.len() >= rb * d && k.len() >= keys * d);
    assert!(rb >= 1 && out.len() >= (rb - 1) * CHUNK + keys);
    // As many keys a call as keep the products in eight vector registers.
    macro_rules! rows {
        ($r:literal; $full:literal; $($c:literal),*) => {{
            let mut j = 0;
            while j + $full <= keys {
                qk::<$r, $full>(&q[..$r * d], &k[j * d..], d, &mut out[j..]);
                j += $full;
            }
            match keys - j {
                $($c => qk::<$r, $c>(&q[..$r * d], &k[j * d..], d, &mut out[j..]),)*
                _ => {}
            }
        }};
    }
    match rb {
        4 => rows!(4; 2; 1),
        3 => rows!(3; 2; 1),
        2 => rows!(2; 4; 3, 2, 1),
        1 => rows!(1; 8; 7, 6, 5, 4, 3, 2, 1),
        _ => unreachable!("qk_rows takes 1-4 rows"),
    }
}

/// `R × C` scores (`R · C ≤ 8`): `out[r * CHUNK + c] = dot(q_r, k_c)`.
#[inline(always)]
fn qk<const R: usize, const C: usize>(q: &[f32], k: &[f32], d: usize, out: &mut [f32]) {
    assert!(R * C <= 8 && q.len() >= R * d && k.len() >= C * d);
    assert!(out.len() >= (R - 1) * CHUNK + C);
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    // SAFETY: AVX2 + FMA are compiled in; the asserts above bound every read
    // and write the kernel makes.
    unsafe {
        avx::qk::<R, C>(q.as_ptr(), k.as_ptr(), d, out.as_mut_ptr())
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    )))]
    portable::qk::<R, C>(q, k, d, out)
}

/// `o_r += Σ_{j < keys} p_rj · v_j` for `rb` rows (≤ 4), in key order — [`axpy`]
/// per key, rows sharing each loaded value vector. Row `r`'s probabilities are
/// `p[r * CHUNK..]`; its output is `o[r * o_stride..][..d]`.
fn pv_rows(p: &[f32], rb: usize, v: &[f32], keys: usize, d: usize, o: &mut [f32], o_stride: usize) {
    if keys == 0 {
        return;
    }
    assert!((1..=4).contains(&rb) && v.len() >= keys * d);
    assert!(p.len() >= (rb - 1) * CHUNK + keys && o.len() >= (rb - 1) * o_stride + d);
    macro_rules! go {
        ($r:literal) => {{
            #[cfg(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "fma"
            ))]
            // SAFETY: AVX2 + FMA are compiled in; the asserts above bound
            // every read and write.
            unsafe {
                avx::pv::<$r>(p.as_ptr(), v.as_ptr(), keys, d, o.as_mut_ptr(), o_stride)
            }
            #[cfg(not(all(
                target_arch = "x86_64",
                target_feature = "avx2",
                target_feature = "fma"
            )))]
            portable::pv::<$r>(p, v, keys, d, o, o_stride)
        }};
    }
    match rb {
        4 => go!(4),
        3 => go!(3),
        2 => go!(2),
        _ => go!(1),
    }
}

/// The arithmetic in plain Rust: what every build without AVX2 + FMA runs,
/// and what the vector kernels are checked against bit for bit.
#[cfg(any(
    test,
    not(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))
))]
mod portable {
    use super::CHUNK;

    /// `a · b + c`, fused where the processor fuses (the x86-64 FMA targets
    /// and every aarch64), else a product then a sum — never a software
    /// `fma` call.
    #[inline(always)]
    fn madd(a: f32, b: f32, c: f32) -> f32 {
        #[cfg(any(target_feature = "fma", target_arch = "aarch64"))]
        {
            a.mul_add(b, c)
        }
        #[cfg(not(any(target_feature = "fma", target_arch = "aarch64")))]
        {
            a * b + c
        }
    }

    /// The module doc's dot product: lane `i` accumulates dimensions
    /// `i, i + 8, …` in order, the lanes are added as a tree, then the
    /// dimensions past the last eight, in order.
    pub(super) fn dot(q: &[f32], k: &[f32]) -> f32 {
        let d = q.len().min(k.len());
        let lanes = d / 8;
        let mut acc = [0f32; 8];
        for b in 0..lanes {
            for (i, a) in acc.iter_mut().enumerate() {
                *a = madd(q[b * 8 + i], k[b * 8 + i], *a);
            }
        }
        let mut tail = 0f32;
        for x in lanes * 8..d {
            tail = madd(q[x], k[x], tail);
        }
        ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail
    }

    pub(super) fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
        for (yi, &xi) in y.iter_mut().zip(x) {
            *yi = madd(a, xi, *yi);
        }
    }

    pub(super) fn qk<const R: usize, const C: usize>(
        q: &[f32],
        k: &[f32],
        d: usize,
        out: &mut [f32],
    ) {
        for r in 0..R {
            for c in 0..C {
                out[r * CHUNK + c] = dot(&q[r * d..(r + 1) * d], &k[c * d..(c + 1) * d]);
            }
        }
    }

    pub(super) fn pv<const R: usize>(
        p: &[f32],
        v: &[f32],
        keys: usize,
        d: usize,
        o: &mut [f32],
        o_stride: usize,
    ) {
        for j in 0..keys {
            for r in 0..R {
                let row = &mut o[r * o_stride..r * o_stride + d];
                axpy(p[r * CHUNK + j], &v[j * d..(j + 1) * d], row);
            }
        }
    }
}

/// The vector kernels — [`portable`]'s arithmetic, eight lanes at a time.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
mod avx {
    use super::CHUNK;
    use std::arch::x86_64::*;

    /// Eight dot products' lane accumulators → their eight sums, each
    /// `((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7))` (`portable::dot`'s
    /// tree; a sum of two is the same either way round).
    #[inline(always)]
    unsafe fn lane_trees(v: [__m256; 8]) -> __m256 {
        let t0 = _mm256_hadd_ps(v[0], v[1]);
        let t1 = _mm256_hadd_ps(v[2], v[3]);
        let t2 = _mm256_hadd_ps(v[4], v[5]);
        let t3 = _mm256_hadd_ps(v[6], v[7]);
        let u0 = _mm256_hadd_ps(t0, t1);
        let u1 = _mm256_hadd_ps(t2, t3);
        _mm256_add_ps(
            _mm256_permute2f128_ps(u0, u1, 0x20),
            _mm256_permute2f128_ps(u0, u1, 0x31),
        )
    }

    /// `R × C` scores, `R · C ≤ 8`. `q`: `R` rows of `d`; `k`: `C` rows of `d`;
    /// `out[r * CHUNK + c]`.
    #[inline(always)]
    pub(super) unsafe fn qk<const R: usize, const C: usize>(
        q: *const f32,
        k: *const f32,
        d: usize,
        out: *mut f32,
    ) {
        let lanes = d / 8;
        let mut acc = [[_mm256_setzero_ps(); C]; R];
        for b in 0..lanes {
            let mut kv = [_mm256_setzero_ps(); C];
            for (c, kc) in kv.iter_mut().enumerate() {
                *kc = _mm256_loadu_ps(k.add(c * d + b * 8));
            }
            for (r, acc_r) in acc.iter_mut().enumerate() {
                let qv = _mm256_loadu_ps(q.add(r * d + b * 8));
                for (a, kc) in acc_r.iter_mut().zip(&kv) {
                    *a = _mm256_fmadd_ps(qv, *kc, *a);
                }
            }
        }
        let mut v = [_mm256_setzero_ps(); 8];
        for r in 0..R {
            for c in 0..C {
                v[r * C + c] = acc[r][c];
            }
        }
        let mut sums = [0f32; 8];
        _mm256_storeu_ps(sums.as_mut_ptr(), lane_trees(v));
        for r in 0..R {
            for c in 0..C {
                let mut tail = 0f32;
                for x in lanes * 8..d {
                    tail = (*q.add(r * d + x)).mul_add(*k.add(c * d + x), tail);
                }
                *out.add(r * CHUNK + c) = sums[r * C + c] + tail;
            }
        }
    }

    /// `R` rows of `o_r += p_rj · v_j` for `j` in key order, sixteen output
    /// lanes at a time held in registers across the keys.
    #[inline(always)]
    pub(super) unsafe fn pv<const R: usize>(
        p: *const f32,
        v: *const f32,
        keys: usize,
        d: usize,
        o: *mut f32,
        o_stride: usize,
    ) {
        let mut x = 0;
        while x + 16 <= d {
            let mut acc = [[_mm256_setzero_ps(); 2]; R];
            for (r, a) in acc.iter_mut().enumerate() {
                a[0] = _mm256_loadu_ps(o.add(r * o_stride + x));
                a[1] = _mm256_loadu_ps(o.add(r * o_stride + x + 8));
            }
            for j in 0..keys {
                let v0 = _mm256_loadu_ps(v.add(j * d + x));
                let v1 = _mm256_loadu_ps(v.add(j * d + x + 8));
                for (r, a) in acc.iter_mut().enumerate() {
                    let pr = _mm256_broadcast_ss(&*p.add(r * CHUNK + j));
                    a[0] = _mm256_fmadd_ps(pr, v0, a[0]);
                    a[1] = _mm256_fmadd_ps(pr, v1, a[1]);
                }
            }
            for (r, a) in acc.iter().enumerate() {
                _mm256_storeu_ps(o.add(r * o_stride + x), a[0]);
                _mm256_storeu_ps(o.add(r * o_stride + x + 8), a[1]);
            }
            x += 16;
        }
        if x + 8 <= d {
            let mut acc = [_mm256_setzero_ps(); R];
            for (r, a) in acc.iter_mut().enumerate() {
                *a = _mm256_loadu_ps(o.add(r * o_stride + x));
            }
            for j in 0..keys {
                let v0 = _mm256_loadu_ps(v.add(j * d + x));
                for (r, a) in acc.iter_mut().enumerate() {
                    let pr = _mm256_broadcast_ss(&*p.add(r * CHUNK + j));
                    *a = _mm256_fmadd_ps(pr, v0, *a);
                }
            }
            for (r, a) in acc.iter().enumerate() {
                _mm256_storeu_ps(o.add(r * o_stride + x), *a);
            }
            x += 8;
        }
        for xx in x..d {
            for j in 0..keys {
                for r in 0..R {
                    let y = o.add(r * o_stride + xx);
                    *y = (*p.add(r * CHUNK + j)).mul_add(*v.add(j * d + xx), *y);
                }
            }
        }
    }

    /// `y += a · x` over `n` elements.
    #[inline(always)]
    pub(super) unsafe fn axpy(a: f32, x: *const f32, y: *mut f32, n: usize) {
        let av = _mm256_set1_ps(a);
        let mut i = 0;
        while i + 8 <= n {
            let r = _mm256_fmadd_ps(av, _mm256_loadu_ps(x.add(i)), _mm256_loadu_ps(y.add(i)));
            _mm256_storeu_ps(y.add(i), r);
            i += 8;
        }
        while i < n {
            *y.add(i) = a.mul_add(*x.add(i), *y.add(i));
            i += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u32) -> Vec<f32> {
        let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn bits(x: &[f32]) -> Vec<u32> {
        x.iter().map(|f| f.to_bits()).collect()
    }

    /// Every blocked score and value-sum is the portable arithmetic bit for
    /// bit — at every block shape the dispatch uses, and at head
    /// widths with and without a tail. On an AVX2 + FMA build this checks the
    /// vector kernels against the portable arithmetic.
    #[test]
    fn the_vector_kernels_are_the_portable_arithmetic_exactly() {
        for d in [64usize, 80, 100, 128, 256] {
            let q = noise(4 * d, d as u32);
            let k = noise(8 * d, 7 + d as u32);
            let want = |r: usize, c: usize| {
                let mut one = [0f32; 1];
                portable::qk::<1, 1>(&q[r * d..], &k[c * d..], d, &mut one);
                one[0]
            };
            for rb in 1..=4usize {
                for keys in 1..=8usize {
                    let mut out = vec![f32::NAN; 4 * CHUNK];
                    qk_rows(&q[..rb * d], rb, &k[..keys * d], keys, d, &mut out);
                    for r in 0..rb {
                        for c in 0..keys {
                            assert_eq!(
                                out[r * CHUNK + c].to_bits(),
                                want(r, c).to_bits(),
                                "d={d} rows={rb} keys={keys} at {r},{c}"
                            );
                        }
                    }
                }
            }

            let keys = 37;
            let p = noise(4 * CHUNK, 3 + d as u32);
            let v = noise(keys * d, 11 + d as u32);
            let w = partial_len(d);
            let start = noise(4 * w, 5);
            for rb in 1..=4usize {
                let mut got = start.clone();
                pv_rows(&p, rb, &v, keys, d, &mut got, w);
                let mut want = start.clone();
                match rb {
                    1 => portable::pv::<1>(&p, &v, keys, d, &mut want, w),
                    2 => portable::pv::<2>(&p, &v, keys, d, &mut want, w),
                    3 => portable::pv::<3>(&p, &v, keys, d, &mut want, w),
                    _ => portable::pv::<4>(&p, &v, keys, d, &mut want, w),
                }
                assert_eq!(bits(&got), bits(&want), "pv d={d} rows={rb}");
            }

            let mut got = start[..d].to_vec();
            axpy(0.37, &v[..d], &mut got);
            let mut want = start[..d].to_vec();
            portable::axpy(0.37, &v[..d], &mut want);
            assert_eq!(bits(&got), bits(&want), "axpy d={d}");
        }
    }

    #[test]
    fn a_row_reads_from_its_first_to_its_last_visible_key() {
        let inf = f32::NEG_INFINITY;
        assert_eq!(visible_range(None, 9), 0..9);
        assert_eq!(visible_range(Some(&[0.0, 0.0, inf, inf]), 4), 0..2);
        // Keys hidden at the start (a window) are not read either.
        assert_eq!(visible_range(Some(&[inf, 0.0, 0.0, inf]), 4), 1..3);
        // A hidden key between visible ones stays inside the row.
        assert_eq!(visible_range(Some(&[0.0, inf, 0.0]), 3), 0..3);
        assert_eq!(visible_range(Some(&[inf, inf]), 2), 0..0);
        assert_eq!(chunks_spanned(0, 1), 0..1);
        assert_eq!(chunks_spanned(0, CHUNK + 1), 0..2);
        assert_eq!(chunks_spanned(CHUNK - 1, 2), 0..2);
        assert_eq!(chunks_spanned(3 * CHUNK + 5, 10), 3..4);
    }
}
