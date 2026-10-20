use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5K, BlockQ5_0, BlockQ5_1, BlockQ6K,
    BlockQ8K, BlockQ8_0, BlockQ8_1, QK8_0, QK_K,
};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[inline(always)]
pub(crate) unsafe fn hsum_float_8(x: __m256) -> f32 {
    let res = _mm256_extractf128_ps(x, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

#[inline(always)]
pub(crate) unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4));
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = scale_product(x.d, y.d);
            let bx = q4_0_bytes(x);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = scale_product(x.d, y.d);
            let bx = q8_0_bytes(x);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }
}

/// SwarmLLM patch: Q5_0 against Q8_0 on AVX2 — upstream candle had no SIMD
/// kernel for Q5_0, Q4_1 or Q5_1 and ran the scalar `vec_dot_unopt`, though
/// llama.cpp writes Q5_0 wherever a row is too narrow for Q4_K (a Q4_K_M of a
/// model 896 wide is 44% Q5_0 by bytes). llama.cpp's `ggml_vec_dot_q5_0_q8_0`
/// (`ggml-cpu/arch/x86/quants.c`), operation for operation, so the result is
/// llama.cpp's arithmetic rather than the scalar loop's.
#[inline(always)]
pub(crate) fn vec_dot_q5_0_q8_0(n: usize, xs: &[BlockQ5_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q5_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = scale_product(x.d, y.d);
            let bx = q5_0_bytes(x);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }
}

/// SwarmLLM patch: Q4_1 against Q8_1 on AVX2 — llama.cpp's
/// `ggml_vec_dot_q4_1_q8_1`: the products in eight lanes, the minimum's term
/// (`m * s`) summed apart and added at the end.
#[inline(always)]
pub(crate) fn vec_dot_q4_1_q8_1(n: usize, xs: &[BlockQ4_1], ys: &[BlockQ8_1]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q4_1_q8_1: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut summs = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            summs += _mm256_cvtss_f32(scale_product(x.m, y.s));
            let d = scale_product(x.d, y.d);
            let bx = q4_1_bytes(x);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_us8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc) + summs
    }
}

/// SwarmLLM patch: Q5_1 against Q8_1 on AVX2 — llama.cpp's
/// `ggml_vec_dot_q5_1_q8_1`, as `vec_dot_q4_1_q8_1` with the fifth bit (its
/// `fmadd(q, dx * dy, acc)` is this one's: the two factors of an FMA commute
/// exactly).
#[inline(always)]
pub(crate) fn vec_dot_q5_1_q8_1(n: usize, xs: &[BlockQ5_1], ys: &[BlockQ8_1]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q5_1_q8_1: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        let mut summs = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()) {
            summs += _mm256_cvtss_f32(scale_product(x.m, y.s));
            let d = scale_product(x.d, y.d);
            let bx = q5_1_bytes(x);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_us8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc) + summs
    }
}

/// SwarmLLM patch: `x × y` of two f16 scales, as f32, broadcast — the
/// single-row kernels' `f16::to_f32(x) * f16::to_f32(y)`, bit for bit, built
/// so it cannot wait on the accumulator. A scalar `f16::to_f32` compiles to
/// `vpinsrw $0, mem, %xmmN, %xmmM` (LLVM treats the other lanes as unused), and
/// that MERGES into whichever register the allocator picks for `N`: when it
/// picked the accumulator (2026-10-20, measured), every block's scale waited
/// for the previous block's FMA and the chain ran ~15 cycles a block instead of
/// ~4 — a Q8_0 row 3.5 against 1.5 ns a block, and which kernel was hit moved
/// with register allocation. Both halves are packed in a general register and
/// moved with `vmovd`, which writes the whole vector register.
#[inline(always)]
unsafe fn scale_product(x: f16, y: f16) -> __m256 {
    let halves = x.to_bits() as u32 | (y.to_bits() as u32) << 16;
    let both = _mm_cvtph_ps(_mm_cvtsi32_si128(halves as i32));
    _mm256_broadcastss_ps(_mm_mul_ss(both, _mm_movehdup_ps(both)))
}

/// SwarmLLM patch: a several-row pass's `scale_product`s for one block, in one
/// conversion — lane `1 + r` is `x × y_of(r)` for row `r < R` (lane 0 is `x × x`,
/// unused). Same bits as `scale_product` per row: each lane is one f16 → f32
/// conversion and one IEEE product, and multiplication commutes exactly.
#[inline(always)]
unsafe fn block_scale_products<const R: usize>(x: f16, y_of: impl Fn(usize) -> f16) -> __m256 {
    const { assert!(R <= 7, "eight lanes: the column's scale and seven rows") };
    let mut lanes = [x.to_bits() as u64, 0u64];
    for r in 0..R {
        let slot = r + 1;
        lanes[slot / 4] |= (y_of(r).to_bits() as u64) << (16 * (slot % 4));
    }
    let s = _mm256_cvtph_ps(_mm_set_epi64x(lanes[1] as i64, lanes[0] as i64));
    _mm256_mul_ps(_mm256_permutevar8x32_ps(s, _mm256_setzero_si256()), s)
}

/// Lane `1 + r` of a `block_scale_products` result, broadcast.
#[inline(always)]
unsafe fn row_lane(v: __m256, r: usize) -> __m256 {
    _mm256_permutevar8x32_ps(v, _mm256_set1_epi32(r as i32 + 1))
}

/// Spreads 32 bits to 32 bytes, 0xFF where a bit is set — llama.cpp's
/// `bytes_from_bits_32`: the fifth bit of Q5_0 / Q5_1, bit `j` to byte `j`.
#[inline(always)]
unsafe fn bytes_from_bits_32(bits: &[u8; 4]) -> __m256i {
    let x32 = u32::from_le_bytes(*bits) as i32;
    let shuf_mask = _mm256_set_epi64x(
        0x0303030303030303,
        0x0202020202020202,
        0x0101010101010101,
        0x0000000000000000,
    );
    let bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    let bytes = _mm256_or_si256(bytes, _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe));
    _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1))
}

// A weight block's 32 values as bytes — the column side of the kernels below.
// The single-row and several-row kernels unpack through the same function, so
// they cannot drift apart.

/// Q8_0: the bytes as stored.
#[inline(always)]
unsafe fn q8_0_bytes(x: &BlockQ8_0) -> __m256i {
    _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i)
}

/// Q4_0: the nibbles less 8, in -8..=7.
#[inline(always)]
unsafe fn q4_0_bytes(x: &BlockQ4_0) -> __m256i {
    _mm256_sub_epi8(bytes_from_nibbles_32(x.qs.as_ptr()), _mm256_set1_epi8(8))
}

/// Q5_0: nibble plus fifth bit, less 16, in -16..=15 (a clear fifth bit ORs in
/// 0xF0, which is the nibble less 16 in two's complement).
#[inline(always)]
unsafe fn q5_0_bytes(x: &BlockQ5_0) -> __m256i {
    let high = _mm256_andnot_si256(bytes_from_bits_32(&x.qh), _mm256_set1_epi8(0xF0u8 as i8));
    _mm256_or_si256(bytes_from_nibbles_32(x.qs.as_ptr()), high)
}

/// Q5_1: nibble plus fifth bit, in 0..=31 (the minimum is added apart).
#[inline(always)]
unsafe fn q5_1_bytes(x: &BlockQ5_1) -> __m256i {
    let high = _mm256_and_si256(bytes_from_bits_32(&x.qh), _mm256_set1_epi8(0x10));
    _mm256_or_si256(bytes_from_nibbles_32(x.qs.as_ptr()), high)
}

/// SwarmLLM patch: the several-row kernels of the 32-value formats. A Q8_0 /
/// Q4_0 / Q5_0 column has little to unpack, so what one call per row wastes is
/// its single accumulator: every block's product waits on the one before it.
/// `R` rows a pass keep `R` independent chains in flight — llamafile's tinyBLAS
/// (`ggml-cpu/llamafile/sgemm.cpp`, `tinyBLAS_Q0_AVX`) tiles the same per-block
/// sequence 4×2 on AVX2 for this reason. Bit-identical to the per-row loop:
/// each row keeps its own `acc` (and `summs`), sees the single-row kernel's
/// operations in its order, and the column is unpacked by the same function.
macro_rules! rows_of_a_32_value_format {
    ($rows_fn:ident, $pass:ident, $single:ident, $block:ty, $vec_dot_ty:ty, $unpack:ident, signed) => {
        rows_of_a_32_value_format!(@wrapper $rows_fn, $pass, $single, $block, $vec_dot_ty);

        /// `R` rows of the single-row kernel sharing one pass over the column.
        #[inline(always)]
        unsafe fn $pass<const R: usize>(xs: &[$block], ys: &[$vec_dot_ty], kb: usize, out: &mut [f32]) {
            let mut acc = [_mm256_setzero_ps(); R];
            for (bi, x) in xs.iter().enumerate() {
                let d_all = block_scale_products::<R>(x.d, |r| ys[r * kb + bi].d);
                let bx = $unpack(x);
                let ax = _mm256_sign_epi8(bx, bx);
                for r in 0..R {
                    let y = &ys[r * kb + bi];
                    let d = row_lane(d_all, r);
                    let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
                    let q = mul_sum_us8_pairs_float(ax, _mm256_sign_epi8(by, bx));
                    acc[r] = _mm256_fmadd_ps(d, q, acc[r]);
                }
            }
            for r in 0..R {
                out[r] = hsum_float_8(acc[r]);
            }
        }
    };
    ($rows_fn:ident, $pass:ident, $single:ident, $block:ty, $vec_dot_ty:ty, $unpack:ident, with_minimum) => {
        rows_of_a_32_value_format!(@wrapper $rows_fn, $pass, $single, $block, $vec_dot_ty);

        /// `R` rows of the single-row kernel sharing one pass over the column.
        #[inline(always)]
        unsafe fn $pass<const R: usize>(xs: &[$block], ys: &[$vec_dot_ty], kb: usize, out: &mut [f32]) {
            let mut acc = [_mm256_setzero_ps(); R];
            // Lane `1 + r` is row r's `summs`: the same f32 additions, in the
            // same order, as the single-row kernel's scalar.
            let mut summs = _mm256_setzero_ps();
            for (bi, x) in xs.iter().enumerate() {
                let d_all = block_scale_products::<R>(x.d, |r| ys[r * kb + bi].d);
                summs = _mm256_add_ps(summs, block_scale_products::<R>(x.m, |r| ys[r * kb + bi].s));
                let bx = $unpack(x);
                for r in 0..R {
                    let y = &ys[r * kb + bi];
                    let d = row_lane(d_all, r);
                    let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
                    let q = mul_sum_us8_pairs_float(bx, by);
                    acc[r] = _mm256_fmadd_ps(d, q, acc[r]);
                }
            }
            for r in 0..R {
                out[r] = hsum_float_8(acc[r]) + _mm256_cvtss_f32(row_lane(summs, r));
            }
        }
    };
    (@wrapper $rows_fn:ident, $pass:ident, $single:ident, $block:ty, $vec_dot_ty:ty) => {
        /// One column against `rows` activation rows — passes of four rows, a
        /// 2-3 row tail in one pass ([`tail_in_one_pass`]), a lone row through
        /// the single-row kernel.
        #[inline(always)]
        pub(crate) fn $rows_fn(n: usize, xs: &[$block], ys: &[$vec_dot_ty], rows: usize, out: &mut [f32]) {
            let kb = xs.len();
            debug_assert!(n.is_multiple_of(QK8_0), "{}: {n} is not divisible by {QK8_0}", stringify!($rows_fn));
            debug_assert_eq!(ys.len(), rows * kb, "{}: rows × blocks", stringify!($rows_fn));
            debug_assert!(out.len() >= rows);
            const R: usize = 4;
            let mut r0 = 0;
            while r0 + R <= rows {
                unsafe { $pass::<R>(xs, &ys[r0 * kb..(r0 + R) * kb], kb, &mut out[r0..r0 + R]) };
                r0 += R;
            }
            let left = rows - r0;
            if tail_in_one_pass(left) {
                let (ys, out) = (&ys[r0 * kb..rows * kb], &mut out[r0..rows]);
                unsafe {
                    if left == 3 {
                        $pass::<3>(xs, ys, kb, out)
                    } else {
                        $pass::<2>(xs, ys, kb, out)
                    }
                }
                return;
            }
            for r in r0..rows {
                out[r] = $single(n, xs, &ys[r * kb..(r + 1) * kb]);
            }
        }
    };
}

rows_of_a_32_value_format!(vec_dot_rows_q8_0_q8_0, dot_q8_0_q8_0_rows, vec_dot_q8_0_q8_0, BlockQ8_0, BlockQ8_0, q8_0_bytes, signed);
rows_of_a_32_value_format!(vec_dot_rows_q4_0_q8_0, dot_q4_0_q8_0_rows, vec_dot_q4_0_q8_0, BlockQ4_0, BlockQ8_0, q4_0_bytes, signed);
rows_of_a_32_value_format!(vec_dot_rows_q5_0_q8_0, dot_q5_0_q8_0_rows, vec_dot_q5_0_q8_0, BlockQ5_0, BlockQ8_0, q5_0_bytes, signed);
rows_of_a_32_value_format!(vec_dot_rows_q4_1_q8_1, dot_q4_1_q8_1_rows, vec_dot_q4_1_q8_1, BlockQ4_1, BlockQ8_1, q4_1_bytes, with_minimum);
rows_of_a_32_value_format!(vec_dot_rows_q5_1_q8_1, dot_q5_1_q8_1_rows, vec_dot_q5_1_q8_1, BlockQ5_1, BlockQ8_1, q5_1_bytes, with_minimum);

/// Q4_1: the nibbles, in 0..=15 (the minimum is added apart).
#[inline(always)]
unsafe fn q4_1_bytes(x: &BlockQ4_1) -> __m256i {
    bytes_from_nibbles_32(x.qs.as_ptr())
}

#[inline(always)]
unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    const K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}

#[inline(always)]
unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
    const K_SHUFFLE: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i))
}

#[inline(always)]
unsafe fn get_scale_shuffle_q3k(i: usize) -> __m256i {
    const K_SHUFFLE: [u8; 128] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i))
}

#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q6k_8k: {n} is not divisible by {QK_K}"
    );

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let m2 = _mm256_set1_epi8(3);
        let m32s = _mm256_set1_epi8(32);
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let mut q4 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 128 {
                let is = j * 4;
                let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));

                let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
                qh = qh.add(32);

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
                let q4h_1 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2), 4);
                let q4h_2 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2), 4);
                let q4h_3 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2), 4);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                let p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                let p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                let p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                let p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}

/// SwarmLLM patch (2026-10-14): do the 2 or 3 rows left after the last full
/// pass of a several-row kernel go through ONE pass sized to them?
///
/// The kernels below take rows four at a time and used to hand the rest to the
/// single-row kernel, one row each — harmless at a prompt's 128 rows, and the
/// WHOLE call for a speculative check: 1-2 guesses are a 2-3 row matmul, so
/// every row redid the column-side unpack the several-row kernel exists to
/// share. On a processor tail (Proxmox, xlam-2-3b, 20 layers) each extra
/// position cost ~45% of a one-position forward (59 / 86 / 115 ms for 1 / 2 / 3)
/// and the guess-length controller rightly chose no guesses at all. The `R`-row
/// kernels are generic in `R` and every row keeps its own accumulators, so the
/// pass at R = 2 or 3 is bit-identical to R = 4 and to the per-row loop
/// (`examples/qmatmul_bench` asserts it). `SWARMLLM_QMATMUL_TAIL=per-row` = the
/// old tail, the A/B inside one binary. Read once.
#[inline(always)]
fn tail_in_one_pass(left: usize) -> bool {
    static PER_ROW: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    (left == 2 || left == 3)
        && !*PER_ROW.get_or_init(|| {
            std::env::var("SWARMLLM_QMATMUL_TAIL").is_ok_and(|v| v == "per-row")
        })
}

/// SwarmLLM patch: one Q6_K column against `rows` Q8_K rows at once — the
/// Q6_K sibling of `vec_dot_rows_q4k_q8k`, same contract (bit-identical to the
/// per-row `vec_dot_q6k_q8k`; only the column-side nibble/high-bit assembly and
/// the widened scales are shared). Q6_K's row side is a larger fraction of the
/// work than Q4_K's (the `q8s` correction term is per row), so the gain is
/// smaller; four rows per pass keeps the tile in registers.
#[inline(always)]
pub(crate) fn vec_dot_rows_q6k_q8k(
    n: usize,
    xs: &[BlockQ6K],
    ys: &[BlockQ8K],
    rows: usize,
    out: &mut [f32],
) {
    let kb = xs.len();
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_rows_q6k_q8k: {n} is not divisible by {QK_K}"
    );
    debug_assert_eq!(ys.len(), rows * kb, "vec_dot_rows_q6k_q8k: rows × blocks");
    debug_assert!(out.len() >= rows);
    const R: usize = 4;
    let mut r0 = 0;
    while r0 + R <= rows {
        unsafe {
            dot_q6k_q8k_rows::<R>(xs, &ys[r0 * kb..(r0 + R) * kb], kb, &mut out[r0..r0 + R]);
        }
        r0 += R;
    }
    let left = rows - r0;
    if tail_in_one_pass(left) {
        let (ys, out) = (&ys[r0 * kb..rows * kb], &mut out[r0..rows]);
        unsafe {
            if left == 3 {
                dot_q6k_q8k_rows::<3>(xs, ys, kb, out)
            } else {
                dot_q6k_q8k_rows::<2>(xs, ys, kb, out)
            }
        }
        return;
    }
    for r in r0..rows {
        out[r] = vec_dot_q6k_q8k(n, xs, &ys[r * kb..(r + 1) * kb]);
    }
}

/// `R` rows of `vec_dot_q6k_q8k` sharing one pass over the column. Kept in
/// lock-step with the single-row kernel: change one, change both, and re-run
/// `examples/qmatmul_bench` (exact-equality assert).
#[inline(always)]
unsafe fn dot_q6k_q8k_rows<const R: usize>(xs: &[BlockQ6K], ys: &[BlockQ8K], kb: usize, out: &mut [f32]) {
    let m4 = _mm256_set1_epi8(0xF);
    let m2 = _mm256_set1_epi8(3);
    let m32s = _mm256_set1_epi8(32);
    let mut acc = [_mm256_setzero_ps(); R];

    for (bi, x) in xs.iter().enumerate() {
        let xd = x.d.to_f32();
        let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
        let mut q4 = x.ql.as_ptr();
        let mut qh = x.qh.as_ptr();
        let mut q8p: [*const i8; R] = [std::ptr::null(); R];
        let mut ds = [0f32; R];
        for r in 0..R {
            let y = &ys[r * kb + bi];
            q8p[r] = y.qs.as_ptr();
            ds[r] = y.d * xd;
        }
        let mut sumi = [_mm256_setzero_si256(); R];

        for j in 0..QK_K / 128 {
            let is = j * 4;
            // Column side, shared across rows: the widened scales and the four
            // assembled 6-bit operand vectors.
            let sc_0 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is)));
            let sc_1 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 1)));
            let sc_2 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 2)));
            let sc_3 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales, get_scale_shuffle(is + 3)));

            let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
            qh = qh.add(32);

            let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
            let q4h_1 =
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2), 4);
            let q4h_2 =
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2), 4);
            let q4h_3 =
                _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2), 4);

            let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
            let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
            let q4_2 =
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
            let q4_3 =
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

            // Row side, exactly the single-row kernel's sequence per row.
            for r in 0..R {
                let q8 = q8p[r];
                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                let q8_1 = _mm256_loadu_si256(q8.add(32) as *const __m256i);
                let q8_2 = _mm256_loadu_si256(q8.add(64) as *const __m256i);
                let q8_3 = _mm256_loadu_si256(q8.add(96) as *const __m256i);
                q8p[r] = q8.add(128);

                let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                let p16_0 = _mm256_madd_epi16(sc_0, p16_0);
                let p16_1 = _mm256_madd_epi16(sc_1, p16_1);
                let p16_2 = _mm256_madd_epi16(sc_2, p16_2);
                let p16_3 = _mm256_madd_epi16(sc_3, p16_3);

                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_0, p16_1));
                sumi[r] = _mm256_add_epi32(sumi[r], _mm256_add_epi32(p16_2, p16_3));
            }
        }
        for r in 0..R {
            acc[r] = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&ds[r]),
                _mm256_cvtepi32_ps(sumi[r]),
                acc[r],
            );
        }
    }
    for r in 0..R {
        out[r] = hsum_float_8(acc[r]);
    }
}

#[inline(always)]
unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
}

#[inline(always)]
pub(crate) fn vec_dot_q2k_q8k(n: usize, xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q2k_q8k: {n} is not divisible by {QK_K}"
    );

    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let m4 = _mm_set1_epi8(0xF);

        let mut acc = _mm256_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            let mut q2 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let scales8 = _mm_and_si128(mins_and_scales, m4);
            let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
            let mins = _mm256_cvtepi8_epi16(mins8);
            let prod =
                _mm256_madd_epi16(mins, _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i));

            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

            let all_scales = _mm256_cvtepi8_epi16(scales8);
            let l_scales = _mm256_extracti128_si256(all_scales, 0);
            let h_scales = _mm256_extracti128_si256(all_scales, 1);
            let scales = [
                mm256_set_m128i(l_scales, l_scales),
                mm256_set_m128i(h_scales, h_scales),
            ];

            let mut sumi = _mm256_setzero_si256();

            for scale in scales {
                let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
                q2 = q2.add(32);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let q2_0 = _mm256_and_si256(q2bits, m3);
                let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
                let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
                let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

                let p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                let p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                let p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                let p3 = _mm256_maddubs_epi16(q2_3, q8_3);

                let p0 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(0)), p0);
                let p1 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(1)), p1);
                let p2 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(2)), p2);
                let p3 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(3)), p3);

                let p0 = _mm256_add_epi32(p0, p1);
                let p2 = _mm256_add_epi32(p2, p3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }

        hsum_float_8(acc)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q3k_q8k(n: usize, xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q3k_q8k: {n} is not divisible by {QK_K}"
    );

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut aux = [0u32; 3];

    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let mone = _mm256_set1_epi8(1);
        let m32 = _mm_set1_epi8(32);

        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();

            let mut q3 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            LittleEndian::read_u32_into(&x.scales, &mut aux);
            let scales128 = _mm_set_epi32(
                (((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4)) as i32,
                (((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4)) as i32,
                ((aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4)) as i32,
                ((aux[0] & KMASK2) | (((aux[2]) & KMASK1) << 4)) as i32,
            );
            let scales128 = _mm_sub_epi8(scales128, m32);
            let all_scales = _mm256_cvtepi8_epi16(scales128);
            let l_scales = _mm256_extracti128_si256(all_scales, 0);
            let h_scales = _mm256_extracti128_si256(all_scales, 1);
            let scales = [
                mm256_set_m128i(l_scales, l_scales),
                mm256_set_m128i(h_scales, h_scales),
            ];

            // high bit
            let hbits = _mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i);

            let mut sumi = _mm256_setzero_si256();

            for (j, scale) in scales.iter().enumerate() {
                // load low 2 bits
                let q3bits = _mm256_loadu_si256(q3 as *const __m256i);
                q3 = q3.add(32);

                // Prepare low and high bits
                // We hardcode the shifts here to avoid loading them into a separate register
                let q3l_0 = _mm256_and_si256(q3bits, m3);
                let q3h_0 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 0)), 0)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 4)), 4)
                };
                let q3h_0 = _mm256_slli_epi16(q3h_0, 2);

                let q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
                let q3h_1 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 1)), 1)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 5)), 5)
                };
                let q3h_1 = _mm256_slli_epi16(q3h_1, 2);

                let q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
                let q3h_2 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 2)), 2)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 6)), 6)
                };
                let q3h_2 = _mm256_slli_epi16(q3h_2, 2);

                let q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
                let q3h_3 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 3)), 3)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 7)), 7)
                };
                let q3h_3 = _mm256_slli_epi16(q3h_3, 2);

                // load Q8 quants
                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we
                // can use _mm256_maddubs_epi16, and then subtract. The high bit part has the 2
                // already subtracted (and so, it is zero if the high bit was not set, and 2 if the
                // high bit was set)
                let q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
                let q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
                let q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
                let q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

                let p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
                let p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
                let p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

                let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                // multiply with scales
                let p16_0 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(0)), p16_0);
                let p16_1 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(1)), p16_1);
                let p16_2 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(2)), p16_2);
                let p16_3 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(3)), p16_3);

                // accumulate
                let p16_0 = _mm256_add_epi32(p16_0, p16_1);
                let p16_2 = _mm256_add_epi32(p16_2, p16_3);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
            }

            // multiply with block scale and accumulate
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}

/// SwarmLLM patch: one Q2_K column against `rows` Q8_K rows at once, as
/// [`vec_dot_rows_q4k_q8k`] does for Q4_K (FUTURE_WORK #138: a mixture-of-
/// experts prompt on the processor, whose Q2_K files are this type). Every row
/// keeps its own accumulators and sees `vec_dot_q2k_q8k`'s FLOAT operations in
/// the same order; the column-side work — scale and min unpack, the 2-bit
/// split, the shuffled scales — is done once for all rows. The integer partial
/// sums are added in another grouping, which is exact (wrapping `i32` addition
/// is associative), so the result is bit-identical. Rows four at a time; the
/// tail takes the single-row kernel.
#[inline(always)]
pub(crate) fn vec_dot_rows_q2k_q8k(
    n: usize,
    xs: &[BlockQ2K],
    ys: &[BlockQ8K],
    rows: usize,
    out: &mut [f32],
) {
    let kb = xs.len();
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_rows_q2k_q8k: {n} is not divisible by {QK_K}"
    );
    debug_assert_eq!(ys.len(), rows * kb, "vec_dot_rows_q2k_q8k: rows × blocks");
    debug_assert!(out.len() >= rows);
    const R: usize = 4;
    let mut r0 = 0;
    while r0 + R <= rows {
        unsafe {
            dot_q2k_q8k_rows::<R>(xs, &ys[r0 * kb..(r0 + R) * kb], kb, &mut out[r0..r0 + R]);
        }
        r0 += R;
    }
    let left = rows - r0;
    if tail_in_one_pass(left) {
        let (ys, out) = (&ys[r0 * kb..rows * kb], &mut out[r0..rows]);
        unsafe {
            if left == 3 {
                dot_q2k_q8k_rows::<3>(xs, ys, kb, out)
            } else {
                dot_q2k_q8k_rows::<2>(xs, ys, kb, out)
            }
        }
        return;
    }
    for r in r0..rows {
        out[r] = vec_dot_q2k_q8k(n, xs, &ys[r * kb..(r + 1) * kb]);
    }
}

/// `R` rows of `vec_dot_q2k_q8k` sharing one pass over the column. Kept in
/// lock-step with the single-row kernel: change one, change both
/// (`multi_row_kernels_are_bit_identical_to_the_per_row_ordering`).
#[inline(always)]
unsafe fn dot_q2k_q8k_rows<const R: usize>(xs: &[BlockQ2K], ys: &[BlockQ8K], kb: usize, out: &mut [f32]) {
    let m3 = _mm256_set1_epi8(3);
    let m4 = _mm_set1_epi8(0xF);
    let mut acc = [_mm256_setzero_ps(); R];

    for (bi, x) in xs.iter().enumerate() {
        let xd = x.d.to_f32();
        let xdmin = x.dmin.to_f32();

        let mins_and_scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
        let scales8 = _mm_and_si128(mins_and_scales, m4);
        let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
        let mins = _mm256_cvtepi8_epi16(mins8);
        let all_scales = _mm256_cvtepi8_epi16(scales8);
        let l_scales = _mm256_extracti128_si256(all_scales, 0);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(l_scales, l_scales),
            mm256_set_m128i(h_scales, h_scales),
        ];

        // Row-side: the mins term first, as the single-row kernel adds it.
        let mut ds = [0f32; R];
        let mut q8p: [*const i8; R] = [std::ptr::null(); R];
        for r in 0..R {
            let y = &ys[r * kb + bi];
            ds[r] = y.d * xd;
            let dmin = -y.d * xdmin;
            let prod =
                _mm256_madd_epi16(mins, _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i));
            acc[r] = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc[r]);
            q8p[r] = y.qs.as_ptr();
        }

        let mut sumi = [_mm256_setzero_si256(); R];
        let mut q2 = x.qs.as_ptr();
        for scale in scales {
            let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
            q2 = q2.add(32);
            macro_rules! part {
                ($i:literal, $shift:literal) => {{
                    let q2_i = _mm256_and_si256(_mm256_srli_epi16(q2bits, $shift), m3);
                    let sc_i = _mm256_shuffle_epi8(scale, get_scale_shuffle_q3k($i));
                    for r in 0..R {
                        let q8 = _mm256_loadu_si256(q8p[r].add(32 * $i) as *const __m256i);
                        let p = _mm256_madd_epi16(sc_i, _mm256_maddubs_epi16(q2_i, q8));
                        sumi[r] = _mm256_add_epi32(sumi[r], p);
                    }
                }};
            }
            part!(0, 0);
            part!(1, 2);
            part!(2, 4);
            part!(3, 6);
            for p in q8p.iter_mut() {
                *p = p.add(128);
            }
        }
        for r in 0..R {
            acc[r] = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&ds[r]),
                _mm256_cvtepi32_ps(sumi[r]),
                acc[r],
            );
        }
    }
    for r in 0..R {
        out[r] = hsum_float_8(acc[r]);
    }
}

/// SwarmLLM patch: one Q3_K column against `rows` Q8_K rows at once — the same
/// contract as [`vec_dot_rows_q2k_q8k`]: the column's 2-bit and high-bit split
/// and its shuffled scales done once, each row's FLOAT operations in
/// `vec_dot_q3k_q8k`'s order, the exact integer sums regrouped.
#[inline(always)]
pub(crate) fn vec_dot_rows_q3k_q8k(
    n: usize,
    xs: &[BlockQ3K],
    ys: &[BlockQ8K],
    rows: usize,
    out: &mut [f32],
) {
    let kb = xs.len();
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_rows_q3k_q8k: {n} is not divisible by {QK_K}"
    );
    debug_assert_eq!(ys.len(), rows * kb, "vec_dot_rows_q3k_q8k: rows × blocks");
    debug_assert!(out.len() >= rows);
    const R: usize = 4;
    let mut r0 = 0;
    while r0 + R <= rows {
        unsafe {
            dot_q3k_q8k_rows::<R>(xs, &ys[r0 * kb..(r0 + R) * kb], kb, &mut out[r0..r0 + R]);
        }
        r0 += R;
    }
    let left = rows - r0;
    if tail_in_one_pass(left) {
        let (ys, out) = (&ys[r0 * kb..rows * kb], &mut out[r0..rows]);
        unsafe {
            if left == 3 {
                dot_q3k_q8k_rows::<3>(xs, ys, kb, out)
            } else {
                dot_q3k_q8k_rows::<2>(xs, ys, kb, out)
            }
        }
        return;
    }
    for r in r0..rows {
        out[r] = vec_dot_q3k_q8k(n, xs, &ys[r * kb..(r + 1) * kb]);
    }
}

/// `R` rows of `vec_dot_q3k_q8k` sharing one pass over the column. Kept in
/// lock-step with the single-row kernel: change one, change both.
#[inline(always)]
unsafe fn dot_q3k_q8k_rows<const R: usize>(xs: &[BlockQ3K], ys: &[BlockQ8K], kb: usize, out: &mut [f32]) {
    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;
    let mut aux = [0u32; 3];
    let m3 = _mm256_set1_epi8(3);
    let mone = _mm256_set1_epi8(1);
    let m32 = _mm_set1_epi8(32);
    let mut acc = [_mm256_setzero_ps(); R];

    for (bi, x) in xs.iter().enumerate() {
        let xd = x.d.to_f32();

        LittleEndian::read_u32_into(&x.scales, &mut aux);
        let scales128 = _mm_set_epi32(
            (((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4)) as i32,
            (((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4)) as i32,
            ((aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4)) as i32,
            ((aux[0] & KMASK2) | (((aux[2]) & KMASK1) << 4)) as i32,
        );
        let scales128 = _mm_sub_epi8(scales128, m32);
        let all_scales = _mm256_cvtepi8_epi16(scales128);
        let l_scales = _mm256_extracti128_si256(all_scales, 0);
        let h_scales = _mm256_extracti128_si256(all_scales, 1);
        let scales = [
            mm256_set_m128i(l_scales, l_scales),
            mm256_set_m128i(h_scales, h_scales),
        ];
        let hbits = _mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i);

        let mut ds = [0f32; R];
        let mut q8p: [*const i8; R] = [std::ptr::null(); R];
        for r in 0..R {
            let y = &ys[r * kb + bi];
            ds[r] = y.d * xd;
            q8p[r] = y.qs.as_ptr();
        }

        let mut sumi = [_mm256_setzero_si256(); R];
        let mut q3 = x.qs.as_ptr();
        for (j, scale) in scales.iter().enumerate() {
            let q3bits = _mm256_loadu_si256(q3 as *const __m256i);
            q3 = q3.add(32);
            // Part `i` of this half: low 2 bits shifted by `$lo`, the high bit
            // `$h` of the first half's mask or `$h + 4` of the second's.
            macro_rules! part {
                ($i:literal, $lo:literal, $h:literal, $h2:literal) => {{
                    let q3l = _mm256_and_si256(_mm256_srli_epi16(q3bits, $lo), m3);
                    let q3h = if j == 0 {
                        _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, $h)), $h)
                    } else {
                        _mm256_srli_epi16(
                            _mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, $h2)),
                            $h2,
                        )
                    };
                    let q3h = _mm256_slli_epi16(q3h, 2);
                    let sc = _mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k($i));
                    for r in 0..R {
                        let q8 = _mm256_loadu_si256(q8p[r].add(32 * $i) as *const __m256i);
                        let q8s = _mm256_maddubs_epi16(q3h, q8);
                        let p16 = _mm256_maddubs_epi16(q3l, q8);
                        let p16 = _mm256_sub_epi16(p16, q8s);
                        let p16 = _mm256_madd_epi16(sc, p16);
                        sumi[r] = _mm256_add_epi32(sumi[r], p16);
                    }
                }};
            }
            part!(0, 0, 0, 4);
            part!(1, 2, 1, 5);
            part!(2, 4, 2, 6);
            part!(3, 6, 3, 7);
            for p in q8p.iter_mut() {
                *p = p.add(128);
            }
        }
        for r in 0..R {
            acc[r] = _mm256_fmadd_ps(
                _mm256_broadcast_ss(&ds[r]),
                _mm256_cvtepi32_ps(sumi[r]),
                acc[r],
            );
        }
    }
    for r in 0..R {
        out[r] = hsum_float_8(acc[r]);
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);

        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 64 {
                let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16l = _mm256_maddubs_epi16(q4l, q8l);
                let p16l = _mm256_madd_epi16(scale_l, p16l);
                sumi = _mm256_add_epi32(sumi, p16l);

                let q8h = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16h = _mm256_maddubs_epi16(q4h, q8h);
                let p16h = _mm256_madd_epi16(scale_h, p16h);
                sumi = _mm256_add_epi32(sumi, p16h);
            }

            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }

        let acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        let acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

        hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
    }
}

/// SwarmLLM patch: one Q4_K column against `rows` Q8_K rows at once.
///
/// `ys` holds `rows` rows of `xs.len()` blocks each, contiguous. Writes
/// `out[r] = vec_dot_q4k_q8k(n, xs, row r)`, bit-identical to calling that per
/// row: every row keeps its own integer and float accumulators and sees exactly
/// the operations `vec_dot_q4k_q8k` performs, in the same order; only the
/// column-side work — scale unpack, nibble split, the shuffled scale vectors —
/// is done once and shared. That column-side work is the larger half of the
/// single-row kernel at prefill sizes (measured: the per-row cost of the tiled
/// matmul plateaued from m=8 to m=128, so the weight READ was amortised and the
/// per-row UNPACK was not), which is what makes this worth having. Rows are
/// taken four at a time (below); a tail of 2-3 rows shares one pass too
/// ([`tail_in_one_pass`]) and a lone row reuses the single-row kernel.
#[inline(always)]
pub(crate) fn vec_dot_rows_q4k_q8k(
    n: usize,
    xs: &[BlockQ4K],
    ys: &[BlockQ8K],
    rows: usize,
    out: &mut [f32],
) {
    let kb = xs.len();
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_rows_q4k_q8k: {n} is not divisible by {QK_K}"
    );
    debug_assert_eq!(ys.len(), rows * kb, "vec_dot_rows_q4k_q8k: rows × blocks");
    debug_assert!(out.len() >= rows);
    // Four rows per pass, not eight: eight accumulator pairs plus the column
    // operands exceed the 16 ymm registers and spill; measured 0.080 → 0.067
    // ms/row (attn proj) and 0.198 → 0.161 (ffn up) at m=128 going 8 → 4.
    const R: usize = 4;
    let mut r0 = 0;
    while r0 + R <= rows {
        unsafe {
            dot_q4k_q8k_rows::<R>(xs, &ys[r0 * kb..(r0 + R) * kb], kb, &mut out[r0..r0 + R]);
        }
        r0 += R;
    }
    let left = rows - r0;
    if tail_in_one_pass(left) {
        let (ys, out) = (&ys[r0 * kb..rows * kb], &mut out[r0..rows]);
        unsafe {
            if left == 3 {
                dot_q4k_q8k_rows::<3>(xs, ys, kb, out)
            } else {
                dot_q4k_q8k_rows::<2>(xs, ys, kb, out)
            }
        }
        return;
    }
    for r in r0..rows {
        out[r] = vec_dot_q4k_q8k(n, xs, &ys[r * kb..(r + 1) * kb]);
    }
}

/// `R` rows of `vec_dot_q4k_q8k` sharing one pass over the column. Kept in
/// lock-step with the single-row kernel above: change one, change both, and
/// re-run `examples/qmatmul_bench` (exact-equality assert).
#[inline(always)]
unsafe fn dot_q4k_q8k_rows<const R: usize>(xs: &[BlockQ4K], ys: &[BlockQ8K], kb: usize, out: &mut [f32]) {
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;
    let m4 = _mm256_set1_epi8(0xF);
    let mut acc = [_mm256_setzero_ps(); R];
    let mut acc_m = [_mm_setzero_ps(); R];

    for (bi, x) in xs.iter().enumerate() {
        let xd = x.d.to_f32();
        let xdmin = x.dmin.to_f32();

        // Column-side unpack, once for all rows (identical to the single-row
        // kernel's prologue).
        LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);
        utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
        let uaux = utmp[1] & KMASK1;
        utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
        utmp[2] = uaux;
        utmp[0] &= KMASK1;
        let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
            utmp[3] as i32,
            utmp[2] as i32,
            utmp[1] as i32,
            utmp[0] as i32,
        ));
        let mins = _mm256_extracti128_si256(mins_and_scales, 1);
        let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
        let scales = mm256_set_m128i(sc128, sc128);

        // Row-side: the mins term and the per-row scale, exactly as the
        // single-row kernel computes them (`d = y.d * x.d`, `dmin = -y.d * x.dmin`).
        let mut ds = [0f32; R];
        let mut q8p: [*const i8; R] = [std::ptr::null(); R];
        for r in 0..R {
            let y = &ys[r * kb + bi];
            ds[r] = y.d * xd;
            let dmin = -y.d * xdmin;
            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(mins, q8s);
            acc_m[r] = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m[r]);
            q8p[r] = y.qs.as_ptr();
        }

        let mut sumi = [_mm256_setzero_si256(); R];
        let mut q4 = x.qs.as_ptr();
        for j in 0..QK_K / 64 {
            let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
            let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

            let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
            q4 = q4.add(32);
            let q4l = _mm256_and_si256(q4bits, m4);
            let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            for r in 0..R {
                let q8l = _mm256_loadu_si256(q8p[r] as *const __m256i);
                let p16l = _mm256_maddubs_epi16(q4l, q8l);
                let p16l = _mm256_madd_epi16(scale_l, p16l);
                sumi[r] = _mm256_add_epi32(sumi[r], p16l);

                let q8h = _mm256_loadu_si256(q8p[r].add(32) as *const __m256i);
                let p16h = _mm256_maddubs_epi16(q4h, q8h);
                let p16h = _mm256_madd_epi16(scale_h, p16h);
                sumi[r] = _mm256_add_epi32(sumi[r], p16h);
                q8p[r] = q8p[r].add(64);
            }
        }

        for r in 0..R {
            let vd = _mm256_set1_ps(ds[r]);
            acc[r] = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi[r]), acc[r]);
        }
    }

    for r in 0..R {
        let am = _mm_add_ps(acc_m[r], _mm_movehl_ps(acc_m[r], acc_m[r]));
        let am = _mm_add_ss(am, _mm_movehdup_ps(am));
        out[r] = hsum_float_8(acc[r]) + _mm_cvtss_f32(am);
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q5k_q8k(n: usize, xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q5k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let mzero = _mm_setzero_si128();
        let mone = _mm256_set1_epi8(1);

        let mut acc = _mm256_setzero_ps();
        let mut summs = 0.0;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mut q5 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs += dmin * _mm_extract_epi32(hsum, 0) as f32;

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = mm256_set_m128i(sc128, sc128);

            let hbits = _mm256_loadu_si256(x.qh.as_ptr() as *const __m256i);
            let mut hmask = mone;

            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 64 {
                let scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                let scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                let q5bits = _mm256_loadu_si256(q5 as *const __m256i);
                q5 = q5.add(32);

                //Similar to q3k we hardcode the shifts here to avoid loading them into a separate register
                let q5l_0 = _mm256_and_si256(q5bits, m4);
                let q5l_0_shift_input = _mm256_and_si256(hbits, hmask);
                let q5l_0_right_shift = match j {
                    0 => _mm256_srli_epi16(q5l_0_shift_input, 0),
                    1 => _mm256_srli_epi16(q5l_0_shift_input, 2),
                    2 => _mm256_srli_epi16(q5l_0_shift_input, 4),
                    3 => _mm256_srli_epi16(q5l_0_shift_input, 6),
                    _ => unreachable!(),
                };
                let q5h_0 = _mm256_slli_epi16(q5l_0_right_shift, 4);
                let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                hmask = _mm256_slli_epi16(hmask, 1);

                let q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                let q5l_1_shift_input = _mm256_and_si256(hbits, hmask);
                let q5l_1_right_shift = match j {
                    0 => _mm256_srli_epi16(q5l_1_shift_input, 1),
                    1 => _mm256_srli_epi16(q5l_1_shift_input, 3),
                    2 => _mm256_srli_epi16(q5l_1_shift_input, 5),
                    3 => _mm256_srli_epi16(q5l_1_shift_input, 7),
                    _ => unreachable!(),
                };

                let q5h_1 = _mm256_slli_epi16(q5l_1_right_shift, 4);
                let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                hmask = _mm256_slli_epi16(hmask, 1);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

                let p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                let p16_1 = _mm256_madd_epi16(scale_1, p16_1);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            }
            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc) + summs
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k(n: usize, xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q8k_8k: {n} is not divisible by {QK_K}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sumi = _mm256_setzero_si256();
            let x_qs = xs.qs.as_ptr();
            let y_qs = ys.qs.as_ptr();
            for j in (0..QK_K).step_by(32) {
                let xs = _mm256_loadu_si256(x_qs.add(j) as *const __m256i);
                let ys = _mm256_loadu_si256(y_qs.add(j) as *const __m256i);

                let xs0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xs, 0));
                let ys0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ys, 0));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(xs0, ys0));

                let xs1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xs, 1));
                let ys1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ys, 1));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(xs1, ys1));
            }
            let d = _mm256_set1_ps(xs.d * ys.d);
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}
