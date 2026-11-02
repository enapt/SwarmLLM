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

/// `out[i] = gelu(gate[i]) * up[i]`, GELU in its tanh form — candle's f32
/// `gelu` (`candle_core::op::Gelu::f32`) operation for operation, so that a
/// gated product built here holds the bits `(gate.gelu()? * up)?` gives on
/// the processor. Gemma 4's experts (llama.cpp `LLM_FFN_GELU`, which is
/// `ggml_gelu`, the same tanh form).
#[inline]
pub fn gelu_mul_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    const SQRT_TWO_OVER_PI_F32: f32 = 0.797_884_6; // candle's `0.79788456080286535587989211986876373`
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for ((&v, &u), o) in gate.iter().zip(up).zip(out.iter_mut()) {
        let g = 0.5 * v * (1.0 + f32::tanh(SQRT_TWO_OVER_PI_F32 * v * (1.0 + 0.044715 * v * v)));
        *o = g * u;
    }
}

/// gpt-oss's `alpha` in [`swiglu_oai_mul_into`] (llama.cpp
/// `LLM_FFN_SWIGLU_OAI_MOE`, `llama-graph.cpp`; HF's `GptOssExperts.alpha`).
pub const SWIGLU_OAI_ALPHA: f32 = 1.702;
/// gpt-oss's clamp in [`swiglu_oai_mul_into`] (HF's `swiglu_limit`).
pub const SWIGLU_OAI_LIMIT: f32 = 7.0;

/// `out[i] = swiglu_oai(gate[i], up[i])` — gpt-oss's activation, llama.cpp's
/// `ggml_compute_forward_swiglu_oai_f32` expression for expression: the gate
/// clamped from ABOVE only (`std::min`), `up` both ways (`std::clamp`), a
/// sigmoid sharpened by [`SWIGLU_OAI_ALPHA`] through libm's `expf`, and `up`
/// lifted by one. Scalar on purpose, as llama.cpp's is: `f32::exp` is the
/// same libm call, so a product here holds llama.cpp's bits. A NaN gate stays
/// NaN, as `std::min(NaN, limit)` does.
#[inline]
pub fn swiglu_oai_mul_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    const LIMIT: f32 = SWIGLU_OAI_LIMIT;
    for ((&g, &u), o) in gate.iter().zip(up).zip(out.iter_mut()) {
        let x = if LIMIT < g { LIMIT } else { g };
        // `std::clamp` and `f32::clamp` agree on every input, NaN and -0 too.
        let y = u.clamp(-LIMIT, LIMIT);
        let out_glu = x / (1.0 + (SWIGLU_OAI_ALPHA * (-x)).exp());
        *o = out_glu * (y + 1.0);
    }
}

/// [`swiglu_oai_mul_into`] over two tensors of one shape: one pass on the
/// processor, ONE launch on a card (`swiglu_oai_mul_f32`, llama.cpp's CUDA
/// expression; a card's `exp` is not libm's, so a card's product is judged
/// against llama.cpp's, never by its bits), candle's composition elsewhere.
/// The composition's scalar `minimum` / `clamp` copy their scalars from the
/// host, which a captured decode step cannot hold (#390 step 7).
pub fn swiglu_oai_mul(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let fusable = gate.dtype() == DType::F32
        && up.dtype() == DType::F32
        && gate.is_contiguous()
        && up.is_contiguous()
        && gate.dims() == up.dims()
        && up.device().same_device(gate.device());
    if fusable {
        match gate.device() {
            Device::Cpu => return gate.apply_op2_no_bwd(up, &SwigluOaiMul),
            #[cfg(feature = "candle-cuda")]
            Device::Cuda(_) => return gate.apply_op2_no_bwd(up, &SwigluOaiMul),
            _ => {}
        }
    }
    swiglu_oai_composed(gate, up)
}

/// [`swiglu_oai_mul`] as candle ops. On the processor each op is llama.cpp's
/// (candle's f32 `exp` is libm's too), so it holds [`swiglu_oai_mul_into`]'s
/// bits there — pinned by `swiglu_oai_is_llama_cpps_expression`.
fn swiglu_oai_composed(gate: &Tensor, up: &Tensor) -> Result<Tensor> {
    let x = gate.minimum(SWIGLU_OAI_LIMIT)?;
    let y = up.clamp(-SWIGLU_OAI_LIMIT, SWIGLU_OAI_LIMIT)?;
    let sigmoid_den = x
        .affine(-f64::from(SWIGLU_OAI_ALPHA), 0.0)?
        .exp()?
        .affine(1.0, 1.0)?;
    (x / sigmoid_den)? * y.affine(1.0, 1.0)?
}

#[derive(Debug, Clone)]
struct SwigluOaiMul;

impl CustomOp2 for SwigluOaiMul {
    fn name(&self) -> &'static str {
        "swiglu-oai-mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let (CpuStorage::F32(g), CpuStorage::F32(u)) = (s1, s2) else {
            candle_core::bail!("swiglu-oai-mul: f32 only");
        };
        let (Some((go, ge)), Some((uo, ue))) = (l1.contiguous_offsets(), l2.contiguous_offsets())
        else {
            candle_core::bail!("swiglu-oai-mul: inputs must be contiguous");
        };
        let g = &g[go..ge];
        let u = &u[uo..ue];
        if g.len() != u.len() {
            candle_core::bail!("swiglu-oai-mul: shape mismatch {} vs {}", g.len(), u.len());
        }
        let mut out = vec![0f32; g.len()];
        let chunk = l1.dims().last().copied().unwrap_or(g.len()).max(1024);
        out.par_chunks_mut(chunk)
            .zip(g.par_chunks(chunk))
            .zip(u.par_chunks(chunk))
            .for_each(|((o, g), u)| swiglu_oai_mul_into(g, u, o));
        Ok((CpuStorage::F32(out), Shape::from_dims(l1.shape().dims())))
    }

    /// One `swiglu_oai_mul_f32` launch (`kernels/fused_decode.cu`), its output
    /// from `alloc_fully_overwritten` as [`SiluMul`]'s is.
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
            candle_core::bail!("swiglu-oai-mul: inputs must be contiguous");
        };
        let numel = ge - go;
        if numel != ue - uo {
            candle_core::bail!("swiglu-oai-mul: shape mismatch {numel} vs {}", ue - uo);
        }
        let g = g.slice(go..ge);
        let u = u.slice(uo..ue);
        let out = dev.alloc_fully_overwritten::<f32>(numel)?;
        let func = dev.get_or_load_custom_func(
            "swiglu_oai_mul_f32",
            "swarmllm_fused_decode",
            FUSED_DECODE_PTX,
        )?;
        let cfg = LaunchConfig::for_num_elems(numel as u32);
        let (alpha, limit) = (SWIGLU_OAI_ALPHA, SWIGLU_OAI_LIMIT);
        let mut builder = func.builder();
        builder.arg(&numel);
        builder.arg(&alpha);
        builder.arg(&limit);
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

    /// gpt-oss's activation is llama.cpp's `ggml_compute_forward_swiglu_oai_f32`
    /// bit for bit — over inputs either side of both clamps, the clamps
    /// themselves, the sigmoid's saturation and a NaN gate — and so is the
    /// candle composition a card runs. The figure is llama.cpp's own op's
    /// (`examples/swiglu_oai_reference.c`); the C expression transcribed below
    /// says which element parts, should one.
    #[test]
    fn swiglu_oai_is_llama_cpps_expression() {
        // ops.cpp, verbatim: x = std::min(g, limit); y = std::clamp(u, -limit,
        // limit); out_glu = x / (1.f + expf(alpha * (-x))); out_glu * (y + 1.f).
        fn llama_cpp(g: f32, u: f32) -> f32 {
            let (alpha, limit) = (1.702f32, 7.0f32);
            let x = if limit < g { limit } else { g };
            let y = u.clamp(-limit, limit); // std::clamp
            let out_glu = x / (1.0 + (alpha * (-x)).exp());
            out_glu * (y + 1.0)
        }
        let mut gate: Vec<f32> = vec![
            -90.0, -7.5, -7.0, -1.0, -0.0, 0.0, 0.3, 6.99, 7.0, 7.01, 50.0,
        ];
        let mut up: Vec<f32> = vec![
            -50.0, -7.01, -7.0, -1.0, 0.0, -0.0, 0.7, 6.99, 7.0, 7.01, 90.0,
        ];
        // And a spread wide enough that most values sit past a clamp.
        let mut s = 0x9e37_79b9u32;
        for _ in 0..4093 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            gate.push((s % 40_001) as f32 / 2_000.0 - 10.0);
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            up.push((s % 40_001) as f32 / 2_000.0 - 10.0);
        }
        let want: Vec<u32> = gate
            .iter()
            .zip(&up)
            .map(|(&g, &u)| llama_cpp(g, u).to_bits())
            .collect();
        assert!(gate.iter().any(|&g| g > SWIGLU_OAI_LIMIT));
        assert!(up.iter().any(|&u| u < -SWIGLU_OAI_LIMIT));

        // llama.cpp's own op on these inputs (`examples/swiglu_oai_reference.c`,
        // 2026-11-02, master 4b1a27f): the FNV-1a hash of its output's bytes.
        // The figure is glibc's `expf`: Apple's libm rounds a few of these
        // exponentials differently (CI's macos-15 job, 2026-11-02), as
        // llama.cpp built there would. Elsewhere the transcription stands, on
        // that platform's own `expf`, and the kernels below are held to it.
        let fnv = want
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
        if cfg!(all(target_os = "linux", target_env = "gnu")) {
            assert_eq!(fnv, 0xa594_237e_1c52_159c, "llama.cpp's output");
        }

        let mut out = vec![0f32; gate.len()];
        swiglu_oai_mul_into(&gate, &up, &mut out);
        let got: Vec<u32> = out.iter().map(|o| o.to_bits()).collect();
        assert_eq!(got, want, "the slice kernel");

        let n = gate.len();
        let (g, u) = (
            Tensor::from_vec(gate.clone(), (n / 8, 8), &Device::Cpu).unwrap(),
            Tensor::from_vec(up.clone(), (n / 8, 8), &Device::Cpu).unwrap(),
        );
        for (name, t) in [
            ("the processor op", swiglu_oai_mul(&g, &u).unwrap()),
            ("the composition", swiglu_oai_composed(&g, &u).unwrap()),
        ] {
            let got: Vec<u32> = t
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|o| o.to_bits())
                .collect();
            assert_eq!(got, want, "{name}");
        }

        // A NaN gate stays NaN (`std::min(NaN, limit)` is the NaN).
        let mut nan = [0f32];
        swiglu_oai_mul_into(&[f32::NAN], &[1.0], &mut nan);
        assert!(nan[0].is_nan());
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
    /// The card's one-launch `swiglu_oai_mul_f32` against the processor's
    /// llama.cpp expression: a card's `expf` is not libm's, so within a few
    /// units in the last place, never by bits — and the gate past the limit,
    /// `up` past both ends, are exercised (σ 6 reaches ±7 often).
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn cuda_swiglu_oai_matches_the_processors_expression() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED cuda_swiglu_oai: no CUDA device ({e})");
                return;
            }
        };
        for (rows, cols) in [(1usize, 2880usize), (8, 2880), (3, 1001)] {
            let gate = Tensor::randn(0f32, 6.0, (rows, cols), &Device::Cpu).unwrap();
            let up = Tensor::randn(0f32, 6.0, (rows, cols), &Device::Cpu).unwrap();
            let want = swiglu_oai_mul(&gate, &up)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let got = swiglu_oai_mul(&gate.to_device(&dev).unwrap(), &up.to_device(&dev).unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "{rows}x{cols}: element {i}: card {g} vs processor {w}"
                );
            }
        }
    }

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
