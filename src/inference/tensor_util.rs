//! Tensor serialization/deserialization and token sampling utilities.

use candle_core::{DType, Device, Tensor};

use crate::error::SwarmError;

pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>, SwarmError> {
    let tensor = tensor
        .to_dtype(DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let shape = tensor.shape().dims();
    let data = tensor
        .flatten_all()
        .map_err(|e| SwarmError::Internal(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;

    let mut bytes = Vec::new();
    // ndim
    bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    // shape
    for &dim in shape {
        bytes.extend_from_slice(&(dim as u32).to_le_bytes());
    }
    // dtype tag (0 = f32)
    bytes.extend_from_slice(&0u32.to_le_bytes());
    // raw f32 data
    for val in &data {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
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
    let tensor = tensor
        .to_dtype(DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let data = tensor
        .flatten_all()
        .map_err(|e| SwarmError::Internal(e.to_string()))?
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
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
    let _dtype_tag = u32::from_le_bytes(
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

    let mut data = Vec::with_capacity(num_elements);
    for _ in 0..num_elements {
        if pos + 4 > bytes.len() {
            return Err(SwarmError::Internal("Tensor data truncated".into()));
        }
        let val = f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap());
        if !val.is_finite() {
            return Err(SwarmError::Internal(
                "Tensor contains non-finite values (NaN/Inf)".into(),
            ));
        }
        data.push(val);
        pos += 4;
    }

    let tensor = Tensor::from_vec(data, shape.as_slice(), &Device::Cpu)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
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

/// Sample the next token from logits using full SamplingParams (top_k, frequency/presence penalty).
///
/// Converts the tensor to a flat `Vec<f32>` and delegates to `sampling::sample_token`
/// which handles the full temperature → top-k → top-p → softmax → sample pipeline.
pub fn sample_token_with_params(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
) -> Result<u32, SwarmError> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let mut logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;

    if logits_vec.is_empty() {
        return Err(SwarmError::Internal("Empty logits".into()));
    }

    Ok(crate::inference::sampling::sample_token(
        &mut logits_vec,
        params,
    ))
}

/// Sample a token from logits with optional logprob collection.
/// When `params.logprobs` is true, returns `(token_id, Some(logprob))`.
pub fn sample_token_with_logprob(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
) -> Result<(u32, Option<f32>), SwarmError> {
    if !params.logprobs {
        return sample_token_with_params(logits, params).map(|t| (t, None));
    }
    let logits_squeezed = logits
        .squeeze(0)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let logits_f32 = logits_squeezed
        .to_dtype(DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let mut logits_vec = logits_f32
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    if logits_vec.is_empty() {
        return Err(SwarmError::Internal("Empty logits".into()));
    }
    let mut ctx = crate::inference::sampling::SamplingContext::new(logits_vec.len());
    let (token_id, info) =
        crate::inference::sampling::sample_token_with_logprobs(&mut logits_vec, params, &mut ctx);
    let logprob = info.map(|i| i.logprob);
    Ok((token_id, logprob))
}
