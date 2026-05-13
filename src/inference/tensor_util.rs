//! Tensor serialization/deserialization and token sampling utilities.

use candle_core::{DType, Device, Tensor};

use crate::error::SwarmError;
use crate::inference::quant;

/// dtype_tag values used by `tensor_to_bytes` / `bytes_to_tensor`.
/// 0 = raw little-endian f32 (legacy default)
/// 1 = Q8_0 group-32 symmetric quantized (see `inference::quant`)
const DTYPE_TAG_F32: u32 = 0;
const DTYPE_TAG_Q8_0: u32 = 1;

pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>, SwarmError> {
    let tensor = tensor.to_dtype(DType::F32).map_err(SwarmError::internal)?;
    let shape = tensor.shape().dims();
    let data = tensor
        .flatten_all()
        .map_err(SwarmError::internal)?
        .to_vec1::<f32>()
        .map_err(SwarmError::internal)?;

    let mut bytes = Vec::new();
    // ndim
    bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    // shape
    for &dim in shape {
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    // dtype tag (0 = f32)
    bytes.extend_from_slice(&DTYPE_TAG_F32.to_le_bytes());
    // raw f32 data
    for val in &data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    Ok(bytes)
}

/// Q8_0-encoded variant of `tensor_to_bytes` for hidden-state activations.
///
/// Wire layout: same header as f32 (`ndim + shape + dtype_tag=Q8_0`), followed
/// by Q8_0 blocks (34 bytes per group of 32 f32 values, see `inference::quant`).
/// Compresses ~3.76× vs the f32 form. Receivers must use `bytes_to_tensor`,
/// which dispatches on the dtype tag.
pub fn tensor_to_bytes_q8_0(tensor: &Tensor) -> Result<Vec<u8>, SwarmError> {
    let tensor = tensor.to_dtype(DType::F32).map_err(SwarmError::internal)?;
    let shape = tensor.shape().dims();
    let data = tensor
        .flatten_all()
        .map_err(SwarmError::internal)?
        .to_vec1::<f32>()
        .map_err(SwarmError::internal)?;

    let qbytes = quant::quantize_q8_0(&data);

    let mut bytes = Vec::with_capacity(4 + shape.len() * 4 + 4 + qbytes.len());
    bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    for &dim in shape {
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    bytes.extend_from_slice(&DTYPE_TAG_Q8_0.to_le_bytes());
    bytes.extend_from_slice(&qbytes);
    Ok(bytes)
}

/// Element-wise add of two tensors in tensor_to_bytes format.
/// Both must have the same shape. Returns the sum in tensor_to_bytes format.
pub fn tensor_bytes_add(a: &[u8], b: &[u8]) -> Result<Vec<u8>, SwarmError> {
    let ta = bytes_to_tensor(a)?;
    let tb = bytes_to_tensor(b)?;
    let sum = ta
        .add(&tb)
        .map_err(|e| SwarmError::Internal(format!("Tensor add: {e}")))?;
    tensor_to_bytes(&sum)
}

/// Extract raw f32 bytes from a tensor (no header, just flat f32 LE data).
/// Used by AllReduce to ensure consistent data format across TP ranks.
pub fn tensor_to_raw_f32(tensor: &Tensor) -> Result<Vec<u8>, SwarmError> {
    let tensor = tensor.to_dtype(DType::F32).map_err(SwarmError::internal)?;
    let data = tensor
        .flatten_all()
        .map_err(SwarmError::internal)?
        .to_vec1::<f32>()
        .map_err(SwarmError::internal)?;
    Ok(data.iter().flat_map(|f| f.to_le_bytes()).collect())
}

/// Reconstruct tensor bytes (with header) from raw f32 data and shape.
/// Inverse of `tensor_to_raw_f32` — produces the format that `bytes_to_tensor` expects.
pub fn raw_f32_to_tensor_bytes(raw: &[u8], shape: &[u32]) -> Vec<u8> {
    let ndim = shape.len() as u32;
    let mut bytes = Vec::with_capacity(4 + shape.len() * 4 + 4 + raw.len());
    bytes.extend_from_slice(&ndim.to_le_bytes());
    for &dim in shape {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.extend_from_slice(&0u32.to_le_bytes()); // dtype tag: f32
    bytes.extend_from_slice(raw);
    bytes
}

/// Deserialize bytes back to a candle Tensor.
pub fn bytes_to_tensor(bytes: &[u8]) -> Result<Tensor, SwarmError> {
    if bytes.len() < 4 {
        return Err(SwarmError::Internal("Tensor bytes too short".into()));
    }

    let mut pos = 0;

    // Validate minimum header size: ndim(4) + dtype(4) = 8 bytes minimum
    let ndim = u32::from_le_bytes(
        bytes[pos..pos + 4]
            .try_into()
            .map_err(|_| SwarmError::Internal("Tensor bytes too short for ndim".into()))?,
    ) as usize;
    pos += 4;

    // Sanity-check ndim to avoid OOM on malicious input
    if ndim > 8 {
        return Err(SwarmError::Internal(format!(
            "Tensor ndim {} exceeds maximum 8",
            ndim
        )));
    }

    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        if pos + 4 > bytes.len() {
            return Err(SwarmError::Internal(
                "Tensor bytes truncated in shape".into(),
            ));
        }
        let dim = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .map_err(|_| SwarmError::Internal("Tensor shape parse error".into()))?,
        ) as usize;
        shape.push(dim);
        pos += 4;
    }

    if pos + 4 > bytes.len() {
        return Err(SwarmError::Internal(
            "Tensor bytes truncated at dtype".into(),
        ));
    }
    let dtype_tag = u32::from_le_bytes(
        bytes[pos..pos + 4]
            .try_into()
            .map_err(|_| SwarmError::Internal("Tensor dtype parse error".into()))?,
    );
    pos += 4;

    let num_elements: usize = shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| SwarmError::Internal("Tensor shape overflow".into()))?;

    const MAX_TENSOR_ELEMENTS: usize = 32 * 1024 * 1024; // 32M elements = 128MB of f32
    if num_elements > MAX_TENSOR_ELEMENTS {
        return Err(SwarmError::Internal(format!(
            "Tensor too large: {} elements (max {})",
            num_elements, MAX_TENSOR_ELEMENTS
        )));
    }
    if num_elements == 0 {
        return Err(SwarmError::Internal("Tensor has zero elements".into()));
    }

    let data = match dtype_tag {
        DTYPE_TAG_F32 => {
            let mut data = Vec::with_capacity(num_elements);
            for _ in 0..num_elements {
                if pos + 4 > bytes.len() {
                    // Truncated wire payload from a peer is a network/remote
                    // fault, not a local code bug — `Inference` (rather than
                    // `Internal`) so the upstream caller doesn't surface it
                    // as a 500.
                    return Err(SwarmError::Inference("Tensor data truncated".into()));
                }
                let val = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
                if !val.is_finite() {
                    // NaN/Inf in an inference activation isn't a code bug
                    // (could be an fp16-overflow on a CUDA layer that
                    // promoted to Inf before serialization). Inference is
                    // the right error class — Internal would map this to
                    // HTTP 500 even though it's a model/runtime fault.
                    return Err(SwarmError::Inference(
                        "Tensor contains non-finite values (NaN/Inf)".into(),
                    ));
                }
                data.push(val);
                pos += 4;
            }
            data
        }
        DTYPE_TAG_Q8_0 => {
            let payload_len = quant::q8_0_byte_len(num_elements);
            if pos + payload_len > bytes.len() {
                return Err(SwarmError::Inference(
                    "Tensor Q8_0 payload truncated".into(),
                ));
            }
            let data = quant::dequantize_q8_0(&bytes[pos..pos + payload_len], num_elements)
                .map_err(SwarmError::Inference)?;
            // Mirror the F32 path's non-finite guard — a malicious or
            // broken peer could ship a Q8_0 block whose dequantized values
            // include NaN/Inf and corrupt subsequent attention.
            if data.iter().any(|v: &f32| !v.is_finite()) {
                return Err(SwarmError::Inference(
                    "Tensor Q8_0 dequantized to non-finite values (NaN/Inf)".into(),
                ));
            }
            data
        }
        unknown => {
            return Err(SwarmError::Internal(format!(
                "Unknown tensor dtype tag: {unknown}"
            )));
        }
    };

    let tensor =
        Tensor::from_vec(data, shape.as_slice(), &Device::Cpu).map_err(SwarmError::internal)?;
    Ok(tensor)
}

/// Sample the next token from logits using full sampling parameters.
pub fn sample_token(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32, SwarmError> {
    sample_token_with_params(
        logits,
        &crate::types::SamplingParams {
            temperature,
            top_p,
            ..Default::default()
        },
    )
}

/// Sample the next token from logits using full SamplingParams (top_k,
/// temperature, top_p). Does NOT apply frequency/presence penalties —
/// pass an empty history. For decode loops with non-zero penalties,
/// use `sample_token_with_params_history`.
///
/// Converts the tensor to a flat `Vec<f32>` and delegates to
/// `sampling::sample_token`.
pub fn sample_token_with_params(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
) -> Result<u32, SwarmError> {
    sample_token_with_params_history(logits, params, &[])
}

/// Same as `sample_token_with_params` but applies frequency/presence
/// penalties from `generated_ids` (the completion-so-far). Empty history
/// is equivalent to `sample_token_with_params` (no penalty).
pub fn sample_token_with_params_history(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
    generated_ids: &[u32],
) -> Result<u32, SwarmError> {
    let logits = logits.squeeze(0).map_err(SwarmError::internal)?;
    let logits = logits.to_dtype(DType::F32).map_err(SwarmError::internal)?;
    let mut logits_vec = logits.to_vec1::<f32>().map_err(SwarmError::internal)?;

    if logits_vec.is_empty() {
        return Err(SwarmError::Internal("Empty logits".into()));
    }

    let mut ctx = crate::inference::sampling::SamplingContext::new(logits_vec.len());
    Ok(crate::inference::sampling::sample_token_with_history(
        &mut logits_vec,
        params,
        generated_ids,
        &mut ctx,
    ))
}

/// Sample a token from logits with optional logprob collection.
/// When `params.logprobs` is true, returns `(token_id, Some(logprob))`.
///
/// History-free: pass `generated_ids = &[]`. For decode loops with
/// frequency_penalty / presence_penalty, use
/// `sample_token_with_logprob_history` instead.
pub fn sample_token_with_logprob(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
) -> Result<(u32, Option<f32>), SwarmError> {
    sample_token_with_logprob_history(logits, params, &[])
}

/// Same as `sample_token_with_logprob` but applies frequency/presence
/// penalties from `generated_ids` (the completion-so-far token list).
/// Use this in decode loops; pass an empty slice to skip penalties.
pub fn sample_token_with_logprob_history(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
    generated_ids: &[u32],
) -> Result<(u32, Option<f32>), SwarmError> {
    let logits_squeezed = logits.squeeze(0).map_err(SwarmError::internal)?;
    let logits_f32 = logits_squeezed
        .to_dtype(DType::F32)
        .map_err(SwarmError::internal)?;
    let mut logits_vec = logits_f32.to_vec1::<f32>().map_err(SwarmError::internal)?;
    if logits_vec.is_empty() {
        return Err(SwarmError::Internal("Empty logits".into()));
    }
    let mut ctx = crate::inference::sampling::SamplingContext::new(logits_vec.len());
    if !params.logprobs {
        let token_id = crate::inference::sampling::sample_token_with_history(
            &mut logits_vec,
            params,
            generated_ids,
            &mut ctx,
        );
        return Ok((token_id, None));
    }
    let (token_id, info) = crate::inference::sampling::sample_token_with_logprobs_history(
        &mut logits_vec,
        params,
        generated_ids,
        &mut ctx,
    );
    let logprob = info.map(|i| i.logprob);
    Ok((token_id, logprob))
}
