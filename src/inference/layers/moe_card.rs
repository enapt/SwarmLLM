//! The card's grouped mixture-of-experts products: every chosen (token, expert)
//! pair of one projection in ONE launch of the vendored candle-kernels
//! `moe_gemm_gguf` (one row per pair, a decode step) or `moe_gemm_gguf_prefill`
//! (tensor cores, a prompt) — `MoeFfn::stacked_products` is the caller.
//!
//! This is candle-nn 0.10.1's `moe::moe_gemm_gguf`, read 2026-10-16, with the
//! quant types of the vendored kernels: theirs stop at Q8_0 and Q2_K-Q6_K, and
//! ours add the 32-block types (Q4_0, Q4_1, Q5_0, Q5_1). A row whose width is not
//! a multiple of 256 cannot hold a K-quant, so llama.cpp's quantizer stores it in
//! one of these (`tensor_type_fallback` in `llama-quant.cpp`): every Gemma 4
//! 26B-A4B upload's down experts (704 wide) — before, each such layer ran one
//! expert at a time on a card, and its RAM-held experts read a prompt on the
//! processor (FUTURE_WORK #292). [`kernel_type`] is the one list of what the
//! kernels read; `StackedExperts::fit` asks it.

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Result, Tensor};

/// The vendored kernels' id for `dtype` (`quant_type` / `gguf_type` in
/// `candle-kernels/src/moe/moe_gguf.cu` and `moe_wmma_gguf.cu`), or `None` for a
/// type they do not read.
pub(crate) fn kernel_type(dtype: GgmlDType) -> Option<i32> {
    match dtype {
        GgmlDType::Q8_0 => Some(0),
        GgmlDType::Q4K => Some(1),
        GgmlDType::Q2K => Some(2),
        GgmlDType::Q3K => Some(3),
        GgmlDType::Q5K => Some(4),
        GgmlDType::Q6K => Some(5),
        GgmlDType::Q4_0 => Some(15),
        GgmlDType::Q4_1 => Some(16),
        GgmlDType::Q5_0 => Some(17),
        GgmlDType::Q5_1 => Some(18),
        _ => None,
    }
}

/// `input` [m, k] f32 on a card through `weights` [n_experts, n, k], each row of
/// `sorted_token_ids` (the (token, pick) pairs sorted by expert, `expert_ids`
/// beside them) through its expert: [m · topk, n] f32, each row scaled by its
/// pick's weight in `topk_weights` when given. `prompt` sends it to the
/// tensor-core kernel, its input in f16.
#[cfg(feature = "candle-cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_gemm_gguf(
    input: &Tensor,
    weights: &QTensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    prompt: bool,
) -> Result<Tensor> {
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::cuda_backend::kernels::ffi;
    use candle_core::op::BackpropOp;
    use candle_core::{DType, Storage};
    use core::ffi::c_void;

    if input.dtype() != DType::F32 {
        candle_core::bail!("moe_gemm_gguf only accepts f32 inputs");
    }
    let (mut size_m, size_k) = input.dims2()?;
    if topk_weights.is_none() {
        size_m *= topk;
    }
    let (num_experts, size_n, size_k1) = weights.shape().dims3()?;
    if size_k != size_k1 {
        candle_core::bail!("moe_gemm_gguf: input width {size_k} against weight {size_k1}");
    }
    if size_k % 8 != 0 {
        candle_core::bail!("moe_gemm_gguf: input width {size_k} is not a multiple of 8");
    }
    let Some(gguf_dtype) = kernel_type(weights.dtype()) else {
        candle_core::bail!("moe_gemm_gguf does not read {:?} weights", weights.dtype());
    };
    let dev = input.device().as_cuda_device()?;
    let weight_ptr = weights.device_ptr()?;

    let topk_weights = topk_weights.as_ref().map(|w| w.storage_and_layout());
    let topk_weights_ptr = match &topk_weights {
        Some((storage, _)) => match &**storage {
            Storage::Cuda(c) => {
                let w = c.as_cuda_slice::<f32>()?;
                w.device_ptr(w.stream()).0 as *const f32
            }
            _ => candle_core::bail!("topk_weights must be a cuda tensor"),
        },
        None => std::ptr::null(),
    };
    let (sorted_token_ids, _) = sorted_token_ids.storage_and_layout();
    let sorted_token_ids = match &*sorted_token_ids {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("sorted_token_ids must be a cuda tensor"),
    };
    let (experts_ids, _) = experts_ids.storage_and_layout();
    let experts_ids = match &*experts_ids {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("experts_ids must be a cuda tensor"),
    };

    // Every (pair, output column) is assigned by either kernel — no zero-fill,
    // as candle-nn allocates it.
    let output = unsafe { dev.alloc::<f32>(size_m * size_n) }?;
    let stream = dev.cuda_stream().cu_stream() as i64;
    let token_ids_ptr = sorted_token_ids.device_ptr(sorted_token_ids.stream()).0 as *const i32;
    let expert_ids_ptr = experts_ids.device_ptr(experts_ids.stream()).0 as *const i32;
    let output_ptr = output.device_ptr(output.stream()).0 as *mut c_void;
    unsafe {
        if prompt {
            let input = input.to_dtype(DType::F16)?;
            let (input, _) = input.storage_and_layout();
            let input_ptr = match &*input {
                Storage::Cuda(c) => {
                    let c = c.as_cuda_slice::<half::f16>()?;
                    c.device_ptr(c.stream()).0 as *const c_void
                }
                _ => candle_core::bail!("input must be a cuda tensor"),
            };
            ffi::moe_gemm_gguf_prefill(
                input_ptr,
                weight_ptr,
                token_ids_ptr,
                expert_ids_ptr,
                topk_weights_ptr,
                output_ptr,
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                0, // f16 input
                gguf_dtype,
                stream,
            );
        } else {
            let (input, _) = input.storage_and_layout();
            let input = match &*input {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle_core::bail!("input must be a cuda tensor"),
            };
            ffi::moe_gemm_gguf(
                input.device_ptr(input.stream()).0 as *const f32,
                weight_ptr as *const c_void,
                token_ids_ptr,
                expert_ids_ptr,
                topk_weights_ptr,
                output_ptr,
                num_experts as i32,
                topk as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                gguf_dtype,
                stream,
            );
        }
    }
    let output = candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone());
    Ok(Tensor::from_storage(
        Storage::Cuda(output),
        (size_m, size_n),
        BackpropOp::none(),
        false,
    ))
}

#[cfg(not(feature = "candle-cuda"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn moe_gemm_gguf(
    _input: &Tensor,
    _weights: &QTensor,
    _topk_weights: &Option<Tensor>,
    _sorted_token_ids: &Tensor,
    _experts_ids: &Tensor,
    _topk: usize,
    _prompt: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_gguf runs on a card only")
}
