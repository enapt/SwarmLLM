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

    const MAX_TENSOR_ELEMENTS: usize = 64 * 1024 * 1024; // 64M elements = 256MB of f32
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

/// Sample the next token with optional logprobs. Returns (token_id, Option<logprob_info>).
/// When `logprobs=true` in params, collects the top-N log probabilities.
/// Token ID with its logprob, used for logprobs response.
pub type TokenLogProbs = Vec<(u32, f32)>;

pub fn sample_token_with_params_and_logprobs(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
) -> Result<(u32, Option<TokenLogProbs>), SwarmError> {
    let logits_squeezed = logits
        .squeeze(0)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let logits_f32 = logits_squeezed
        .to_dtype(candle_core::DType::F32)
        .map_err(|e| SwarmError::Internal(e.to_string()))?;
    let mut logits_vec = logits_f32
        .to_vec1::<f32>()
        .map_err(|e| SwarmError::Internal(e.to_string()))?;

    if logits_vec.is_empty() {
        return Err(SwarmError::Internal("Empty logits".into()));
    }

    // Use the full sampling with logprobs
    let mut ctx = crate::inference::sampling::SamplingContext::new(logits_vec.len());
    let (token_id, logprob_info) =
        crate::inference::sampling::sample_token_with_logprobs(&mut logits_vec, params, &mut ctx);

    let top = logprob_info.map(|info| {
        let mut result = vec![(info.token_id, info.logprob)];
        for (tid, lp) in info.top_logprobs {
            if tid != info.token_id {
                result.push((tid, lp));
            }
        }
        result
    });

    Ok((token_id, top))
}

/// Sample the next token from logits using full SamplingParams (top_k, frequency/presence penalty).
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

    let temperature = params.temperature;
    let top_p = params.top_p;

    if temperature <= 0.0 {
        // Greedy: argmax — O(V)
        let (idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        return Ok(idx as u32);
    }

    // Match sampling.rs order: temperature → top-k → top-p → sample
    crate::inference::sampling::apply_temperature(&mut logits_vec, temperature);
    crate::inference::sampling::apply_top_k(&mut logits_vec, params.top_k);

    // Softmax — O(V)
    let max_val = logits_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_vec.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = probs.iter().sum();
    if sum <= 0.0 || !sum.is_finite() {
        return Ok(0);
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }

    // Top-p >= 1.0: sample directly from full distribution — O(V), no sort needed
    if top_p >= 1.0 {
        let r: f32 = rand::random();
        let mut cumulative = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumulative += p;
            if r < cumulative {
                return Ok(i as u32);
            }
        }
        return Ok((probs.len() - 1) as u32);
    }

    // Top-p < 1.0: use partial sort — O(V + K log K) where K << V
    // First pass: partition top-K candidates via select_nth_unstable_by (O(V))
    // then sort only those K elements (O(K log K))
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    let k = 256.min(probs.len() - 1);
    indices.select_nth_unstable_by(k, |&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Sort top-K+1 elements descending by probability
    indices[..=k].sort_unstable_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Scan top-K for cumulative >= top_p
    let mut cumulative = 0.0;
    let mut cutoff = k + 1;
    for (i, &idx) in indices[..=k].iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= top_p {
            cutoff = i + 1;
            break;
        }
    }

    // If top-K wasn't enough (very flat distribution), fall back to full sort
    if cumulative < top_p {
        indices[k + 1..].sort_unstable_by(|&a, &b| {
            probs[b]
                .partial_cmp(&probs[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for (i, &idx) in indices[k + 1..].iter().enumerate() {
            cumulative += probs[idx];
            if cumulative >= top_p {
                cutoff = k + 1 + i + 1;
                break;
            }
        }
    }

    // Renormalize and sample from the top-p subset
    let subset = &indices[..cutoff];
    let subset_sum: f32 = subset.iter().map(|&i| probs[i]).sum();
    let r: f32 = rand::random();
    let mut cumulative = 0.0;
    let inv_subset = 1.0 / subset_sum;
    for &idx in subset {
        cumulative += probs[idx] * inv_subset;
        if r < cumulative {
            return Ok(idx as u32);
        }
    }

    Ok(subset[subset.len() - 1] as u32)
}
