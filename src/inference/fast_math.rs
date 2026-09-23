//! Vectorised `expf` for the CPU inference path, and the fused SiLU×gate op
//! built on it.
//!
//! Every `exp` on the CPU forward pass was a scalar libm call: the fused
//! attention softmax (one per score — ~540 M for an 896-token llama-3.2-3b
//! prompt), SiLU in the FFN (one per gate element — ~205 M for the same
//! prompt), and the decode attention kernel's softmax. At ~20 ns each that is
//! several seconds of a 35 s prompt, and the profiler filed it under
//! "attention" and "activation * gate" where it looked like tensor work.
//!
//! `exp_inplace` evaluates eight lanes at a time with the Cephes polynomial
//! (the `exp_ps` of sse_mathfun / llama.cpp's `ggml_v_expf` lineage): range
//! reduction by `ln 2` split in two, a degree-5 polynomial on the remainder,
//! and the power of two assembled into the exponent bits. Maximum relative
//! error against libm is ~2 ulp (pinned by `vectorised_exp_tracks_libm`);
//! inputs below about −87.3 underflow to 0 where libm would return a
//! denormal, and inputs above 88.37 saturate — both fine for softmax (shifted
//! inputs are ≤ 0) and SiLU (where `exp(-x)` → 0 or ∞ gives the right limit).
//! Without AVX2 the helpers fall back to `f32::exp`.
//!
//! Not bit-identical to libm; the consumers are tolerance-tested against the
//! composed reference (`fused_matches_composed_reference` at 1e-6,
//! `fused_silu_gate_matches_candle`), which is the same bar the fused softmax
//! itself was held to when it replaced four candle ops.

use candle_core::{CpuStorage, CustomOp2, DType, Device, Layout, Result, Shape, Tensor};
use rayon::prelude::*;

/// `xs[i] = exp(xs[i])` for every element.
#[inline]
pub fn exp_inplace(xs: &mut [f32]) {
    #[cfg(target_feature = "avx2")]
    {
        // SAFETY: compiled only when the target has AVX2 (+FMA comes with the
        // x86-64-v3 / native targets this project builds for).
        unsafe { exp_inplace_avx2(xs) }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        for x in xs.iter_mut() {
            *x = x.exp();
        }
    }
}

/// `out[i] = gate[i] / (1 + exp(-gate[i])) * up[i]` — SiLU(gate) × up in one pass.
#[inline]
pub fn silu_mul_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    #[cfg(target_feature = "avx2")]
    {
        unsafe { silu_mul_avx2(gate, up, out) }
    }
    #[cfg(not(target_feature = "avx2"))]
    {
        for ((g, u), o) in gate.iter().zip(up).zip(out.iter_mut()) {
            *o = g / (1.0 + (-g).exp()) * u;
        }
    }
}

#[cfg(target_feature = "avx2")]
mod avx2 {
    use std::arch::x86_64::*;

    /// Eight `expf` at once. Cephes coefficients; see the module doc.
    #[inline(always)]
    pub(super) unsafe fn exp256_ps(x: __m256) -> __m256 {
        let x = _mm256_min_ps(x, _mm256_set1_ps(88.376_26));
        let x = _mm256_max_ps(x, _mm256_set1_ps(-88.376_26));
        // n = floor(x * log2(e) + 0.5)
        let fx = _mm256_fmadd_ps(
            x,
            _mm256_set1_ps(std::f32::consts::LOG2_E),
            _mm256_set1_ps(0.5),
        );
        let fx = _mm256_floor_ps(fx);
        // r = x - n*ln2, with ln2 split so the product is exact
        let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693_359_4), x);
        let r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.121_944_4e-4), r);
        let mut y = _mm256_set1_ps(1.987_569_2e-4);
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.398_2e-3));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(8.333_452e-3));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(4.166_579_6e-2));
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(1.666_666_5e-1));
        // Cephes p5 = 5.0000001201e-1, which is 0.5 exactly in f32.
        y = _mm256_fmadd_ps(y, r, _mm256_set1_ps(0.5));
        let r2 = _mm256_mul_ps(r, r);
        y = _mm256_fmadd_ps(y, r2, r);
        y = _mm256_add_ps(y, _mm256_set1_ps(1.0));
        // 2^n via the exponent field
        let n = _mm256_cvttps_epi32(fx);
        let n = _mm256_add_epi32(n, _mm256_set1_epi32(0x7f));
        let n = _mm256_slli_epi32(n, 23);
        _mm256_mul_ps(y, _mm256_castsi256_ps(n))
    }

    #[inline(always)]
    pub(super) unsafe fn exp_inplace_avx2(xs: &mut [f32]) {
        let (chunks, rest) = xs.as_chunks_mut::<8>();
        for c in chunks.iter_mut() {
            let v = _mm256_loadu_ps(c.as_ptr());
            _mm256_storeu_ps(c.as_mut_ptr(), exp256_ps(v));
        }
        for x in rest {
            *x = x.exp();
        }
    }

    #[inline(always)]
    pub(super) unsafe fn silu_mul_avx2(gate: &[f32], up: &[f32], out: &mut [f32]) {
        let n = gate.len();
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let mut i = 0;
        while i + 8 <= n {
            let g = _mm256_loadu_ps(gate.as_ptr().add(i));
            let u = _mm256_loadu_ps(up.as_ptr().add(i));
            let e = exp256_ps(_mm256_sub_ps(zero, g));
            let s = _mm256_div_ps(g, _mm256_add_ps(one, e));
            _mm256_storeu_ps(out.as_mut_ptr().add(i), _mm256_mul_ps(s, u));
            i += 8;
        }
        while i < n {
            out[i] = gate[i] / (1.0 + (-gate[i]).exp()) * up[i];
            i += 1;
        }
    }
}
#[cfg(target_feature = "avx2")]
use avx2::{exp_inplace_avx2, silu_mul_avx2};

/// PTX for the fused decode kernels, compiled from `kernels/fused_decode.cu`
/// by `build.rs` and loaded through candle's `get_or_load_custom_func`.
#[cfg(feature = "candle-cuda")]
pub(crate) const FUSED_DECODE_PTX: &str =
    include_str!(concat!(env!("OUT_DIR"), "/fused_decode.ptx"));

/// `SWARMLLM_FUSE_SILU_MUL=0` puts the two-kernel candle composition back on
/// the CUDA path, so the fusion can be A/B'd inside ONE binary — which is how
/// every other submission-count change here was attributed
/// (`.claude/rules/diagnosis.md` § 4). Read once and cached: this sits on the
/// per-layer decode path.
#[cfg(feature = "candle-cuda")]
fn fuse_silu_mul_on_cuda() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_FUSE_SILU_MUL").as_deref() != Ok("0"))
}

/// `silu(gate) * up` as one fused pass where possible; the candle composition
/// (two ops, two temporaries) everywhere else.
///
/// Fused on the CPU since the AVX2 work, and on CUDA since the decode budget
/// showed `usilu_f32` and `bmul_f32` running once each per layer — two
/// launches, two allocations and two frees for an elementwise pass over a few
/// KB. → `docs/invariants/inference.md` § "A decode token is bound by GPU
/// submission COUNT, not bandwidth".
pub fn silu_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let shapes_fusable = gate.dtype() == DType::F32
        && up.dtype() == DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
        && gate.dims() == up.dims();

    if shapes_fusable {
        match gate.device() {
            Device::Cpu => return gate.apply_op2_no_bwd(up, &SiluMul),
            #[cfg(feature = "candle-cuda")]
            Device::Cuda(_) if up.device().same_device(gate.device()) => {
                if fuse_silu_mul_on_cuda() {
                    return gate.apply_op2_no_bwd(up, &SiluMul);
                }
            }
            _ => {}
        }
    }
    candle_nn::ops::silu(gate)? * up
}

#[derive(Debug, Clone)]
struct SiluMul;

impl CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "silu-mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(g), CpuStorage::F32(u)) = (s1, s2) else {
            candle_core::bail!("silu-mul: f32 only");
        };
        let (Some((go, ge)), Some((uo, ue))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("silu-mul: inputs must be contiguous");
        };
        let g = &g[go..ge];
        let u = &u[uo..ue];
        if g.len() != u.len() {
            candle_core::bail!("silu-mul: shape mismatch {} vs {}", g.len(), u.len());
        }
        let mut out = vec![0f32; g.len()];
        // Row-sized chunks keep the parallel split coarse enough to matter
        // and aligned to rows for the usual [tokens, ffn] shape.
        let chunk = l1.dims().last().copied().unwrap_or(g.len()).max(1024);
        out.par_chunks_mut(chunk)
            .zip(g.par_chunks(chunk))
            .zip(u.par_chunks(chunk))
            .for_each(|((o, g), u)| silu_mul_into(g, u, o));
        Ok((CpuStorage::F32(out), Shape::from_dims(l1.shape().dims())))
    }

    /// One `silu_mul_f32` launch in place of candle's `usilu_f32` + `bmul_f32`.
    ///
    /// Removes a launch, an allocation and a free per layer. The output comes
    /// from `alloc_fully_overwritten` because the kernel's grid-stride loop
    /// assigns every element it owns — the precondition that accessor
    /// documents.
    ///
    /// ⚠ **Bit-identical to the composed path**, not merely close: the kernel
    /// evaluates candle's own `silu_fwd` expression and then candle's `bmul`,
    /// in that order. `cuda_silu_mul_is_bit_identical_to_the_composed_path`
    /// pins it, so an A/B of this switch moves the submission count and
    /// nothing else.
    #[cfg(feature = "candle-cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::WrapErr;

        let dev = s1.device.clone();
        let g = s1.as_cuda_slice::<f32>()?;
        let u = s2.as_cuda_slice::<f32>()?;
        let (Some((go, ge)), Some((uo, ue))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("silu-mul: inputs must be contiguous");
        };
        let numel = ge - go;
        if numel != ue - uo {
            candle_core::bail!("silu-mul: shape mismatch {numel} vs {}", ue - uo);
        }
        let g = g.slice(go..ge);
        let u = u.slice(uo..ue);

        let out = dev.alloc_fully_overwritten::<f32>(numel)?;
        let func =
            dev.get_or_load_custom_func("silu_mul_f32", "swarmllm_fused_decode", FUSED_DECODE_PTX)?;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let mut builder = func.builder();
        builder.arg(&numel);
        builder.arg(&g);
        builder.arg(&u);
        builder.arg(&out);
        // SAFETY: ffi. Shapes and contiguity are checked above; the kernel
        // reads `numel` elements of each input and writes `numel` of the
        // output, all three allocated at that length.
        unsafe { builder.launch(cfg) }.w()?;

        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(out, dev),
            Shape::from_dims(l1.shape().dims()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectorised_exp_tracks_libm() {
        // Dense sweep of the range the consumers use: softmax inputs are
        // shifted to ≤ 0, SiLU sees ±|gate| (a few tens at most), and both
        // ends are checked up to where libm itself leaves normal numbers.
        let mut xs: Vec<f32> = (-80_000..=80_000).map(|i| i as f32 / 1000.0).collect();
        let want: Vec<f32> = xs.iter().map(|x| x.exp()).collect();
        exp_inplace(&mut xs);
        let mut worst = 0f32;
        for (got, want) in xs.iter().zip(&want) {
            let rel = (got - want).abs() / want;
            if rel > worst {
                worst = rel;
            }
        }
        assert!(worst < 3e-7, "max relative error vs libm: {worst}");
        // Underflow and the masked value behave as softmax needs them to.
        let mut tail = vec![-200.0f32, f32::NEG_INFINITY, 0.0, -0.0];
        exp_inplace(&mut tail);
        assert_eq!(tail[0], 0.0);
        assert_eq!(tail[1], 0.0);
        assert_eq!(tail[2], 1.0);
        assert_eq!(tail[3], 1.0);
    }

    #[test]
    fn fused_silu_gate_matches_candle() {
        let dev = Device::Cpu;
        for (rows, cols) in [(1usize, 8usize), (3, 1000), (7, 8192), (1, 13)] {
            let gate = Tensor::randn(0f32, 3.0, (rows, cols), &dev).unwrap();
            let up = Tensor::randn(0f32, 1.0, (rows, cols), &dev).unwrap();
            let want = (candle_nn::ops::silu(&gate).unwrap() * &up)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = silu_mul(&gate, &up)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1e-6))
                .fold(0f32, f32::max);
            assert!(worst < 2e-6, "{rows}x{cols}: worst rel diff {worst}");
        }
    }

    /// The CUDA fusion must agree with the two kernels it replaces EXACTLY,
    /// not to a tolerance.
    ///
    /// That is the whole point of the bar: the fused kernel exists to cut a
    /// launch, an allocation and a free per layer, so an A/B between it and
    /// the composed path has to isolate the submission count. Any arithmetic
    /// difference at all would mean replies change with the switch, and then
    /// "did this help" and "is this still correct" stop being separable
    /// questions. `kernels/fused_decode.cu` evaluates candle's own `silu_fwd`
    /// and `bmul` expressions in that order to make it reachable.
    ///
    /// ⚠ Gated, so no default build compiles it — run
    /// `cargo test --features candle-cuda --all-targets` after touching
    /// either side (`arch-inference.md` § "Cross-feature compile checks").
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn cuda_silu_mul_is_bit_identical_to_the_composed_path() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                // No card on this machine: the compile is the check here.
                //
                // ⚠ **Say so out loud.** A test that skips silently passes
                // identically whether it ran or not — the vacuous-test trap of
                // `.claude/rules/diagnosis.md` § 5, and the same shape gotcha
                // #259 fixed by asserting the condition it depended on really
                // held. This skip fires on a missing `LD_LIBRARY_PATH` just as
                // readily as on a missing GPU, so it is reachable on the one
                // box that CAN check this. Run with `--nocapture` and look for
                // this line before believing a pass.
                eprintln!("SKIPPED cuda_silu_mul bit-identity: no CUDA device ({e})");
                return;
            }
        };
        // Shapes a real FFN produces: one decode row against several
        // intermediate widths, plus a prefill-sized block and a width that is
        // not a multiple of the block size.
        for (rows, cols) in [(1usize, 5632usize), (1, 8960), (128, 5632), (3, 1001)] {
            let gate = Tensor::randn(0f32, 3.0, (rows, cols), &dev).unwrap();
            let up = Tensor::randn(0f32, 1.0, (rows, cols), &dev).unwrap();

            let want = (candle_nn::ops::silu(&gate).unwrap() * &up)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = gate
                .apply_op2_no_bwd(&up, &SiluMul)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();

            assert_eq!(got.len(), want.len(), "{rows}x{cols}: length");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "{rows}x{cols}: element {i} differs: fused {g} vs composed {w}"
                );
            }
        }
    }
}
