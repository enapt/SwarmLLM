//! The residual stream between two norm points, and the ONE place "add it up,
//! then RMS-norm it" is computed.
//!
//! Every dense transformer layer passes its residual stream through the same
//! shape twice:
//!
//! ```text
//! h = h + attn(norm_1(h))        ← add, then norm_2 of the sum
//! h = h + ffn(norm_2(h))         ← add, then the NEXT layer's norm_1 (or the final norm)
//! ```
//!
//! On CUDA candle composes each "add, then norm" as `badd_f32` + `rmsnorm_f32`:
//! two launches plus an allocation and a free for the sum. Decode there is
//! bound by how many times the CPU thread talks to the driver, not by
//! arithmetic (`docs/invariants/inference.md` § "A decode token is bound by
//! GPU submission COUNT, not bandwidth"), so each pair is worth fusing:
//! **−1 launch, −1 allocation, −1 free per site, two sites per layer.**
//!
//! # Why a type and not a helper at each site
//!
//! The second site straddles the layer boundary — the add ends layer `i` and
//! the norm begins layer `i + 1` (or is the final norm). Fusing it means NOT
//! doing the add where the layer ends, and there are eight hand-written copies
//! of that pattern across `SplitModel`'s single-request and batched loops (four
//! layer variants each). [`Residual`] carries the not-yet-taken sum across the
//! boundary, and [`Residual::add_norm`] is the single place it is resolved — so
//! a new layer variant gets the fusion by construction rather than by
//! remembering (`.claude/rules/architecture.md` § "One invariant, N paths").
//!
//! # What must hold
//!
//! * **Bit-identity.** The fused kernel reproduces candle's `rmsnorm` statement
//!   for statement (`kernels/fused_decode.cu`), so a reply must not move by a
//!   single bit when the switch is flipped. That is what lets the A/B measure
//!   the submission count and nothing else; `examples/kernel_count_ab.sh` checks
//!   both halves in one run.
//! * **Wherever a real tensor is required, the sum is materialised** — before a
//!   device transition, for a captured layer, and as the output of a segment
//!   that is not the last. [`Residual::into_tensor`] is that, and it is exactly
//!   the add the loop used to do.
//! * **Off CUDA nothing changes.** The composed path is the old code, in the old
//!   order.
//!
//! `SWARMLLM_FUSE_ADD_RMSNORM=0` restores the composed path on CUDA for a
//! one-binary A/B, the same discipline as `SWARMLLM_FUSE_SILU_MUL`.

use candle_core::quantized::QTensor;
use candle_core::{Device, Module, Result, Tensor};

/// An RMS norm whose weight and epsilon the fused kernel can read.
///
/// Same construction and the same `forward` as
/// `candle_transformers::quantized_nn::RmsNorm`, which keeps both fields
/// private — the only reason this type exists. `forward` calls
/// `candle_nn::ops::rms_norm`, exactly as that one does, so every norm that is
/// NOT fused computes what it always did.
#[derive(Debug, Clone)]
pub struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    /// Dequantize a GGUF norm weight onto its own device, as
    /// `quantized_nn::RmsNorm::from_qtensor` does.
    pub fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

/// The hidden state between two norm points: a tensor, or a sum not yet taken.
///
/// `Pending { delta, base }` means `delta + base`, in that operand order —
/// which is the order every site in `SplitModel` wrote it (`attn + residual`,
/// `x + residual`), so the composed path adds exactly as before.
pub(crate) enum Residual {
    Ready(Tensor),
    Pending { delta: Tensor, base: Tensor },
}

impl Residual {
    /// The sum `delta + base`, left untaken until something needs it.
    pub(crate) fn pending(delta: Tensor, base: Tensor) -> Self {
        Self::Pending { delta, base }
    }

    /// The device the (eventual) tensor lives on.
    pub(crate) fn device(&self) -> &Device {
        match self {
            Self::Ready(t) => t.device(),
            Self::Pending { base, .. } => base.device(),
        }
    }

    /// Resolve this residual and normalise it: `(residual, norm(residual))`.
    ///
    /// The residual is what the NEXT residual add consumes; the normed value is
    /// what the projections consume. A pending sum goes through
    /// [`add_rms_norm`], which fuses the add into the norm on CUDA.
    pub(crate) fn add_norm(self, norm: &RmsNorm) -> Result<(Tensor, Tensor)> {
        match self {
            Self::Ready(x) => {
                let normed = norm.forward(&x)?;
                Ok((x, normed))
            }
            Self::Pending { delta, base } => add_rms_norm(&delta, &base, norm),
        }
    }

    /// The residual as a plain tensor — the add the loop used to do at the end
    /// of every layer, taken only where something needs the value itself.
    pub(crate) fn into_tensor(self) -> Result<Tensor> {
        match self {
            Self::Ready(x) => Ok(x),
            Self::Pending { delta, base } => delta + base,
        }
    }
}

/// `SWARMLLM_FUSE_ADD_RMSNORM=0` puts candle's two-kernel composition back on
/// the CUDA path, so the fusion can be A/B'd inside ONE binary. Read once and
/// cached: this sits on the per-layer decode path twice.
#[cfg(feature = "candle-cuda")]
fn fuse_add_rms_norm_on_cuda() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("SWARMLLM_FUSE_ADD_RMSNORM").as_deref() != Ok("0"))
}

/// `(a + b, rms_norm(a + b))` — one kernel on CUDA, candle's two ops elsewhere.
///
/// Returns `(sum, normed)`. On the fused path both are zero-copy views of one
/// 2N allocation: the normed half at offset 0 (it feeds every projection, so it
/// gets the plain layout) and the sum at offset N (it only ever feeds the next
/// residual add, which slices by layout). `Tensor::reshape` of a contiguous
/// `narrow` shares storage — checked in candle's source, since one hidden
/// `copy_strided_src` would hand the whole saving back.
pub(crate) fn add_rms_norm(a: &Tensor, b: &Tensor, norm: &RmsNorm) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "candle-cuda")]
    if fusable_on_cuda(a, b, norm) {
        let both = a.apply_op3_no_bwd(
            b,
            &norm.weight,
            &cuda::AddRmsNorm {
                eps: norm.eps as f32,
            },
        )?;
        let normed = both.narrow(0, 0, 1)?.reshape(a.shape())?;
        let sum = both.narrow(0, 1, 1)?.reshape(a.shape())?;
        return Ok((sum, normed));
    }
    let sum = (a + b)?;
    let normed = norm.forward(&sum)?;
    Ok((sum, normed))
}

/// Every precondition the fused kernel relies on, checked rather than assumed:
/// the kernel indexes rows of `ncols` contiguous f32s and reads `alpha` by
/// column. Anything else takes the composed path, which handles it.
#[cfg(feature = "candle-cuda")]
fn fusable_on_cuda(a: &Tensor, b: &Tensor, norm: &RmsNorm) -> bool {
    use candle_core::DType;
    if !matches!(a.device(), Device::Cuda(_)) || !fuse_add_rms_norm_on_cuda() {
        return false;
    }
    let Some(&ncols) = a.dims().last() else {
        return false;
    };
    a.dtype() == DType::F32
        && b.dtype() == DType::F32
        && norm.weight.dtype() == DType::F32
        && a.dims() == b.dims()
        && ncols > 0
        && a.elem_count() > 0
        && a.is_contiguous()
        && b.is_contiguous()
        && norm.weight.is_contiguous()
        && norm.weight.dims() == [ncols]
        && b.device().same_device(a.device())
        && norm.weight.device().same_device(a.device())
}

#[cfg(feature = "candle-cuda")]
mod cuda {
    use candle_core::{CpuStorage, CustomOp3, Layout, Result, Shape};

    pub(super) struct AddRmsNorm {
        pub(super) eps: f32,
    }

    impl CustomOp3 for AddRmsNorm {
        fn name(&self) -> &'static str {
            "add-rms-norm"
        }

        /// Never reached: [`super::add_rms_norm`] sends every non-CUDA tensor
        /// to candle's composed ops, which ARE the CPU implementation. A second
        /// CPU implementation here could only ever disagree with them.
        fn cpu_fwd(
            &self,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            candle_core::bail!(
                "add-rms-norm is CUDA-only; add_rms_norm routes other devices to the composed ops"
            )
        }

        /// One `add_rmsnorm_f32` launch in place of `badd_f32` + `rmsnorm_f32`.
        ///
        /// Output is ONE buffer of `2 * numel`: `[normed | sum]`, shaped
        /// `(2, ..dims)` so the caller splits it with `narrow` on dim 0. From
        /// `alloc_fully_overwritten`, because both of the kernel's loops assign
        /// every column of their row. Launch geometry is candle's `rmsnorm`
        /// rule exactly — one block per row, 32 threads below 1024 columns and
        /// 1024 at or above — which the kernel's reduction depends on for
        /// bit-identity.
        fn cuda_fwd(
            &self,
            s1: &candle_core::CudaStorage,
            l1: &Layout,
            s2: &candle_core::CudaStorage,
            l2: &Layout,
            s3: &candle_core::CudaStorage,
            l3: &Layout,
        ) -> Result<(candle_core::CudaStorage, Shape)> {
            use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use candle_core::cuda_backend::WrapErr;

            let dev = s1.device.clone();
            let (Some((ao, ae)), Some((bo, be)), Some((wo, we))) = (
                l1.contiguous_offsets(),
                l2.contiguous_offsets(),
                l3.contiguous_offsets(),
            ) else {
                candle_core::bail!("add-rms-norm: inputs must be contiguous");
            };
            let dims = l1.shape().dims();
            let numel = ae - ao;
            let ncols = *dims.last().unwrap_or(&0);
            if numel == 0 || ncols == 0 || numel % ncols != 0 {
                candle_core::bail!("add-rms-norm: bad shape {dims:?}");
            }
            if be - bo != numel || we - wo != ncols {
                candle_core::bail!(
                    "add-rms-norm: shape mismatch a={numel} b={} alpha={} ncols={ncols}",
                    be - bo,
                    we - wo
                );
            }
            let nrows = numel / ncols;

            let a = s1.as_cuda_slice::<f32>()?.slice(ao..ae);
            let b = s2.as_cuda_slice::<f32>()?.slice(bo..be);
            let alpha = s3.as_cuda_slice::<f32>()?.slice(wo..we);

            let mut out = dev.alloc_fully_overwritten::<f32>(2 * numel)?;
            let block_size: u32 = if ncols < 1024 { 32 } else { 1024 };
            let cfg = LaunchConfig {
                grid_dim: (nrows as u32, 1, 1),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            };
            let func = dev.get_or_load_custom_func(
                "add_rmsnorm_f32",
                "swarmllm_fused_decode",
                crate::inference::fast_math::FUSED_DECODE_PTX,
            )?;
            let ncols_i = ncols as i32;
            let block_i = block_size as i32;
            {
                // `&mut`: cudarc implements `PushKernelArg` for a
                // `CudaViewMut` only by mutable reference (0.19.9, launch.rs).
                let (mut normed, mut sum) = out.split_at_mut(numel);
                let mut builder = func.builder();
                builder.arg(&a);
                builder.arg(&b);
                builder.arg(&alpha);
                builder.arg(&mut normed);
                builder.arg(&mut sum);
                builder.arg(&ncols_i);
                builder.arg(&block_i);
                builder.arg(&self.eps);
                // SAFETY: ffi. Contiguity and every length are checked above;
                // the kernel reads `nrows * ncols` of `a` and `b` and `ncols` of
                // `alpha`, and writes `nrows * ncols` into each half of `out`.
                unsafe { builder.launch(cfg) }.w()?;
            }

            let mut out_dims = Vec::with_capacity(dims.len() + 1);
            out_dims.push(2);
            out_dims.extend_from_slice(dims);
            Ok((
                candle_core::CudaStorage::wrap_cuda_slice(out, dev),
                Shape::from_dims(&out_dims),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    fn norm(dim: usize, dev: &Device) -> RmsNorm {
        let w = (Tensor::rand(0.5f32, 1.5, dim, &Device::Cpu).unwrap())
            .to_device(dev)
            .unwrap();
        RmsNorm::from_qtensor(QTensor::quantize(&w, GgmlDType::F32).unwrap(), 1e-5).unwrap()
    }

    fn bits(t: &Tensor) -> Vec<u32> {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// Off CUDA the pending residual resolves through exactly the ops the loop
    /// used to write — `delta + base`, then `rms_norm` — so a CPU node's replies
    /// cannot move with this change.
    #[test]
    fn a_pending_residual_resolves_to_the_composed_ops_on_the_cpu() {
        let dev = Device::Cpu;
        let n = norm(96, &dev);
        let delta = Tensor::randn(0f32, 1.0, (2, 3, 96), &dev).unwrap();
        let base = Tensor::randn(0f32, 1.0, (2, 3, 96), &dev).unwrap();

        let want_sum = (&delta + &base).unwrap();
        let want_normed = n.forward(&want_sum).unwrap();
        let (sum, normed) = Residual::pending(delta.clone(), base.clone())
            .add_norm(&n)
            .unwrap();
        assert_eq!(bits(&sum), bits(&want_sum));
        assert_eq!(bits(&normed), bits(&want_normed));

        let materialised = Residual::pending(delta, base).into_tensor().unwrap();
        assert_eq!(bits(&materialised), bits(&want_sum));

        // A Ready residual is normalised as-is.
        let (same, normed2) = Residual::Ready(want_sum.clone()).add_norm(&n).unwrap();
        assert_eq!(bits(&same), bits(&want_sum));
        assert_eq!(bits(&normed2), bits(&want_normed));
    }

    /// Our `RmsNorm` must compute what `quantized_nn::RmsNorm` did — every norm
    /// in the model moved to it, fused or not.
    #[test]
    fn our_rms_norm_matches_the_one_it_replaced() {
        let dev = Device::Cpu;
        let w = Tensor::rand(0.5f32, 1.5, 64, &dev).unwrap();
        let theirs = candle_transformers::quantized_nn::RmsNorm::from_qtensor(
            QTensor::quantize(&w, GgmlDType::F32).unwrap(),
            1e-6,
        )
        .unwrap();
        let ours =
            RmsNorm::from_qtensor(QTensor::quantize(&w, GgmlDType::F32).unwrap(), 1e-6).unwrap();
        let x = Tensor::randn(0f32, 2.0, (3, 5, 64), &dev).unwrap();
        assert_eq!(
            bits(&ours.forward(&x).unwrap()),
            bits(&theirs.forward(&x).unwrap())
        );
    }

    /// The CUDA fusion must agree with `badd_f32` + `rmsnorm_f32` EXACTLY, and
    /// its two outputs must be views of one allocation, not copies.
    ///
    /// Bit-identity is the bar for the reason `fast_math`'s silu×up test gives:
    /// the change is worth a few submissions per layer, far below what a clock
    /// resolves here, so it ships on the count — and a count A/B means nothing
    /// unless both arms compute the same thing.
    ///
    /// The widths cover both of candle's launch geometries (32 threads below
    /// 1024 columns, 1024 at or above), a width that is not a multiple of
    /// either, a prefill-sized block and a batch.
    ///
    /// ⚠ Gated, so no default build compiles it — run
    /// `cargo test --features candle-cuda --lib residual_norm -- --nocapture`
    /// and look for the SKIPPED line before believing a pass.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn cuda_add_rms_norm_is_bit_identical_to_the_composed_path() {
        let dev = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED add_rms_norm bit-identity: no CUDA device ({e})");
                return;
            }
        };
        assert!(
            fuse_add_rms_norm_on_cuda(),
            "SWARMLLM_FUSE_ADD_RMSNORM=0 makes this test compare the composed path with itself"
        );
        for shape in [
            vec![1usize, 1, 2048],
            vec![1, 1, 896],
            vec![1, 1, 1000],
            vec![1, 1, 3072],
            vec![1, 128, 3072],
            vec![4, 1, 4096],
        ] {
            let ncols = *shape.last().unwrap();
            let n = norm(ncols, &dev);
            let a = Tensor::randn(0f32, 1.0, shape.as_slice(), &dev).unwrap();
            let b = Tensor::randn(0f32, 3.0, shape.as_slice(), &dev).unwrap();
            assert!(
                fusable_on_cuda(&a, &b, &n),
                "{shape:?} should take the fused path"
            );

            let want_sum = (&a + &b).unwrap();
            let want_normed = n.forward(&want_sum).unwrap();
            let (sum, normed) = add_rms_norm(&a, &b, &n).unwrap();

            assert_eq!(sum.dims(), shape.as_slice());
            assert_eq!(normed.dims(), shape.as_slice());
            // Zero-copy: one storage, normed at offset 0, sum right after it.
            assert_eq!(normed.layout().start_offset(), 0, "{shape:?}: normed moved");
            assert_eq!(
                sum.layout().start_offset(),
                a.elem_count(),
                "{shape:?}: sum was copied"
            );
            assert!(normed.is_contiguous() && sum.is_contiguous());

            for (label, got, want) in [("sum", &sum, &want_sum), ("normed", &normed, &want_normed)]
            {
                let (g, w) = (bits(got), bits(want));
                assert_eq!(g.len(), w.len());
                if let Some(i) = g.iter().zip(&w).position(|(x, y)| x != y) {
                    panic!(
                        "{shape:?} {label}: element {i} differs: fused {} vs composed {}",
                        f32::from_bits(g[i]),
                        f32::from_bits(w[i])
                    );
                }
            }
        }
    }
}
