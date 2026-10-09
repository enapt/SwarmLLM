//! Writing a step's K or V into a cache buffer of ANOTHER dtype — the half
//! cache a card keeps (`layers::KvStorage::F16`, FUTURE_WORK #194), fed by
//! projections that answer in f32.
//!
//! [`write_into`] is `slice_set` that converts. On a card, f32 into f16 is ONE
//! launch (`kernels/kv_append.cu`): composed from candle ops it was a cast and
//! then a copy — a launch, an allocation and a free more per tensor per layer
//! per decoded token than the f32 cache cost, on a decode step bound by how
//! many launches it makes (`docs/invariants/inference.md` § "A decode token is
//! bound by GPU submission COUNT"). Anything else — the processor, or the
//! direction the card never takes — converts with candle and copies, which is
//! correct everywhere and only ever runs off the per-token path (a hydrated
//! snapshot, a test).

use candle_core::{Result, Tensor};

/// Write `src` into `buffer` at `offset` along `dim`, converting to `buffer`'s
/// dtype — `Tensor::slice_set`'s contract (both contiguous, equal on every
/// other dim), with the dtypes allowed to differ.
pub(crate) fn write_into(buffer: &Tensor, src: &Tensor, dim: usize, offset: usize) -> Result<()> {
    if src.dtype() == buffer.dtype() {
        return buffer.slice_set(src, dim, offset);
    }
    #[cfg(feature = "candle-cuda")]
    if buffer.device().is_cuda()
        && buffer.dtype() == candle_core::DType::F16
        && src.dtype() == candle_core::DType::F32
        && converting_write_enabled()
    {
        let src = src.contiguous()?;
        return buffer.inplace_op2(&src, &cuda::WriteHalf { dim, offset });
    }
    buffer.slice_set(&src.to_dtype(buffer.dtype())?, dim, offset)
}

/// `SWARMLLM_KV_WRITE=compose` → cast with candle and `slice_set`, as two
/// launches — the A/B for the one-launch write inside one binary.
#[cfg(feature = "candle-cuda")]
fn converting_write_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("SWARMLLM_KV_WRITE").as_deref() != Ok("compose"))
}

/// PTX for `kernels/kv_append.cu`, compiled by `build.rs`.
#[cfg(feature = "candle-cuda")]
pub(crate) const KV_APPEND_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kv_append.ptx"));

#[cfg(feature = "candle-cuda")]
mod cuda {
    use candle_core::{CpuStorage, InplaceOp2, Layout, Result};

    /// f32 `src` into the f16 buffer at `offset` along `dim`.
    pub(super) struct WriteHalf {
        pub(super) dim: usize,
        pub(super) offset: usize,
    }

    impl InplaceOp2 for WriteHalf {
        fn name(&self) -> &'static str {
            "kv-write-f16"
        }

        /// Never reached: [`super::write_into`] takes this op only for a
        /// buffer on a card.
        fn cpu_fwd(
            &self,
            _: &mut CpuStorage,
            _: &Layout,
            _: &CpuStorage,
            _: &Layout,
        ) -> Result<()> {
            candle_core::bail!("kv-write-f16 is CUDA-only")
        }

        fn cuda_fwd(
            &self,
            dst: &mut candle_core::CudaStorage,
            dl: &Layout,
            src: &candle_core::CudaStorage,
            sl: &Layout,
        ) -> Result<()> {
            use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
            use candle_core::cuda_backend::WrapErr;

            let (Some((so, se)), Some((dst_start, _))) =
                (sl.contiguous_offsets(), dl.contiguous_offsets())
            else {
                candle_core::bail!("kv-write-f16: the buffer and the source must be contiguous");
            };
            let (dims, sdims) = (dl.dims(), sl.dims());
            let dim = self.dim;
            let same_elsewhere = dims.len() == sdims.len()
                && dim < dims.len()
                && dims
                    .iter()
                    .zip(sdims)
                    .enumerate()
                    .all(|(i, (a, b))| i == dim || a == b);
            if !same_elsewhere || self.offset + sdims[dim] > dims[dim] {
                candle_core::bail!(
                    "kv-write-f16: {sdims:?} does not fit {dims:?} at {} along dim {dim}",
                    self.offset
                );
            }
            // `slice_set`'s 2D view: every dim before `dim` is a row; `dim`
            // and everything after it, one row's run of elements.
            let rows: usize = dims[..dim].iter().product();
            let inner: usize = dims[dim + 1..].iter().product();
            let cols = sdims[dim] * inner;
            let dst_row_stride = dims[dim] * inner;
            let dst_offset = self.offset * inner;
            let total = rows * cols;
            if total == 0 {
                return Ok(());
            }

            let dev = src.device.clone();
            let src = src.as_cuda_slice::<f32>()?.slice(so..se);
            let dst = dst.as_cuda_slice_mut::<half::f16>()?;
            let mut dst = dst.slice_mut(dst_start..);
            let func = dev.get_or_load_custom_func(
                "kv_write_f16",
                "swarmllm_kv_append",
                super::KV_APPEND_PTX,
            )?;
            let cfg = LaunchConfig::for_num_elems(total as u32);
            let mut builder = func.builder();
            builder.arg(&src);
            builder.arg(&mut dst);
            builder.arg(&rows);
            builder.arg(&cols);
            builder.arg(&dst_row_stride);
            builder.arg(&dst_offset);
            // SAFETY: ffi. Both are contiguous (checked above); the kernel reads
            // `rows * cols` elements of `src` and writes the same number into
            // `dst`, every one inside `rows * dst_row_stride` elements from its
            // start because `offset + n <= dims[dim]`.
            unsafe { builder.launch(cfg) }.w()?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, IndexOp};

    /// The processor path: a converting write lands the rounded values where
    /// `slice_set` would have put the originals, and touches nothing else.
    #[test]
    fn a_converting_write_lands_where_slice_set_would() {
        let dev = Device::Cpu;
        let buffer = Tensor::zeros((1usize, 2, 8, 4), DType::F16, &dev).unwrap();
        let src = Tensor::randn(0f32, 1.0, (1usize, 2, 3, 4), &dev).unwrap();
        write_into(&buffer, &src, 2, 2).unwrap();
        let written = buffer.narrow(2, 2, 3).unwrap();
        assert_eq!(
            written
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            src.to_dtype(DType::F16)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );
        let untouched = buffer
            .i((.., .., 0..2, ..))
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert!(untouched.iter().all(|x| *x == 0.0));
    }

    /// The card's one-launch write against candle's cast + `slice_set`, bit for
    /// bit: both round to nearest even, so a half cache written either way holds
    /// the same values the f16 flash mirror did. ⚠ Gated, so no default build
    /// runs it: `cargo test --release --features cuda --lib kv_write`, and look
    /// for the SKIPPED line before believing a pass.
    #[cfg(feature = "candle-cuda")]
    #[test]
    fn the_cards_converting_write_matches_cast_then_slice_set() {
        let card = match Device::new_cuda(0) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED kv_write on the card: no CUDA device ({e})");
                return;
            }
        };
        for (src_len, at, cap) in [
            (1usize, 0usize, 64usize),
            (1, 37, 64),
            (130, 5, 512),
            (64, 0, 64),
        ] {
            let src = Tensor::randn(0f32, 3.0, (1usize, 4, src_len, 128), &card).unwrap();
            let fused = Tensor::zeros((1usize, 4, cap, 128), DType::F16, &card).unwrap();
            let composed = Tensor::zeros((1usize, 4, cap, 128), DType::F16, &card).unwrap();
            fused
                .inplace_op2(&src, &cuda::WriteHalf { dim: 2, offset: at })
                .unwrap();
            composed
                .slice_set(&src.to_dtype(DType::F16).unwrap(), 2, at)
                .unwrap();
            let a = fused.flatten_all().unwrap().to_vec1::<half::f16>().unwrap();
            let b = composed
                .flatten_all()
                .unwrap()
                .to_vec1::<half::f16>()
                .unwrap();
            assert!(
                a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "src_len={src_len} at={at} cap={cap}: the one-launch write differs from cast + slice_set"
            );
        }
    }
}
