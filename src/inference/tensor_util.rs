//! Tensor serialization/deserialization and token sampling utilities.

use candle_core::{DType, Device, Tensor};

use crate::error::SwarmError;
use crate::inference::quant;

/// dtype_tag values used by `tensor_to_bytes` / `bytes_to_tensor`.
/// 0 = raw little-endian f32 (legacy default)
/// 1 = Q8_0 group-32 symmetric quantized (see `inference::quant`)
const DTYPE_TAG_F32: u32 = 0;
const DTYPE_TAG_Q8_0: u32 = 1;

/// How many payload bytes a count of values takes in the encoding `dtype_tag`
/// names — `None` for a tag this build does not decode; the length function
/// answers `None` for a count that overflows. The ONE rule for both readers
/// (`bytes_to_tensor` and `activation_shape_matches`): a new encoding taught
/// to only one would refuse a healthy peer's activation (FUTURE_WORK #98).
fn payload_len(dtype_tag: u32) -> Option<fn(usize) -> Option<usize>> {
    match dtype_tag {
        DTYPE_TAG_F32 => Some(|n| n.checked_mul(4)),
        DTYPE_TAG_Q8_0 => Some(quant::q8_0_byte_len_checked),
        _ => None,
    }
}

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
///
/// **The payload bounds the allocation — there is deliberately no fixed cap on
/// the element count.** The shape comes off the wire, and the risk it carries
/// is `Vec::with_capacity(num_elements)`: a twelve-byte message declaring a
/// billion elements would reserve 4 GB before the first bounds check. Checking
/// the declared count against the bytes that are ACTUALLY here answers that
/// exactly, and caps the allocation at roughly the size of a message the
/// transport already accepted (128 MB per activation on the wire, 512 MB over
/// worker IPC).
///
/// This replaced a flat `MAX_TENSOR_ELEMENTS = 32 * 1024 * 1024`, added in a
/// March 2026 hardening pass for the same reason and correct about the hazard.
/// A fixed element count is not a memory bound, though — it is a bound on the
/// WORK, and it silently became the shortest prompt any distributed pipeline
/// could carry: a hidden state is `positions × hidden_dim` elements, so 32 M is
/// exactly 8192 positions at hidden 4096 and only 4096 at the 8192-wide hidden
/// of a 70 B. Reported from the field on v0.3.153 (gotcha #451) — an 11.2 k
/// token agent prompt on an 8 B model produced 43,876,352 elements and every
/// attempt failed identically, on a pipeline that was otherwise fine.
///
/// A payload LONGER than the shape needs is still accepted: callers hand this
/// whole buffers whose tail belongs to something else.
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

    if num_elements == 0 {
        return Err(SwarmError::Internal("Tensor has zero elements".into()));
    }

    // How many bytes this payload must contain for the shape it declares.
    // Unknown dtype is answered here, before anything is allocated.
    let Some(len_of) = payload_len(dtype_tag) else {
        return Err(SwarmError::Internal(format!(
            "Unknown tensor dtype tag: {dtype_tag}"
        )));
    };
    let Some(required) = len_of(num_elements) else {
        return Err(SwarmError::Internal(format!(
            "Tensor shape overflow: {num_elements} elements of dtype {dtype_tag}"
        )));
    };
    let available = bytes.len() - pos;
    if required > available {
        // Truncated wire payload from a peer is a network/remote fault, not a
        // local code bug — `Inference` (rather than `Internal`) so the
        // upstream caller doesn't surface it as a 500.
        return Err(SwarmError::Inference(format!(
            "Tensor data truncated: shape {shape:?} needs {required} bytes, {available} present"
        )));
    }
    let payload = &bytes[pos..pos + required];

    let data = match dtype_tag {
        DTYPE_TAG_F32 => {
            let mut data = Vec::with_capacity(num_elements);
            for chunk in payload.as_chunks::<4>().0 {
                let val = f32::from_le_bytes(*chunk);
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
            }
            data
        }
        DTYPE_TAG_Q8_0 => {
            let data =
                quant::dequantize_q8_0(payload, num_elements).map_err(SwarmError::Inference)?;
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
/// Converts the tensor to a flat `Vec<f32>` and delegates, through
/// `sample_token_with_params_history`, to `sampling::sample_token_with_history`.
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
    held: Option<&mut crate::inference::reply_grammar::HeldReply>,
) -> Result<(u32, Option<f32>), SwarmError> {
    sample_token_with_logprob_history(logits, params, &[], held)
}

/// Same as `sample_token_with_logprob` but applies frequency/presence
/// penalties from `generated_ids` (the completion-so-far token list).
/// Use this in decode loops; pass an empty slice to skip penalties.
///
/// `held` is the reply's grammar (#336), REQUIRED: the tokens it rules out are
/// set to −∞ before anything else reads the scores (so penalties, top-k/top-p
/// and the logprob all see only what the reply may take — the logprob is the
/// one the token was drawn with), and it is told the token drawn. A draw for
/// a request whose sampling carries a grammar but is handed none is REFUSED:
/// that is a draw site that forgot it, and a reply quietly let off its JSON
/// is the bug this argument exists to stop.
pub fn sample_token_with_logprob_history(
    logits: &Tensor,
    params: &crate::types::SamplingParams,
    generated_ids: &[u32],
    mut held: Option<&mut crate::inference::reply_grammar::HeldReply>,
) -> Result<(u32, Option<f32>), SwarmError> {
    if params.reply_grammar.is_some() && held.is_none() {
        return Err(SwarmError::Internal(
            "a reply held to JSON was drawn without its grammar".into(),
        ));
    }
    let logits_squeezed = logits.squeeze(0).map_err(SwarmError::internal)?;
    let logits_f32 = logits_squeezed
        .to_dtype(DType::F32)
        .map_err(SwarmError::internal)?;
    let mut logits_vec = logits_f32.to_vec1::<f32>().map_err(SwarmError::internal)?;
    if logits_vec.is_empty() {
        return Err(SwarmError::Internal("Empty logits".into()));
    }
    if let Some(held) = held.as_mut() {
        held.mask(&mut logits_vec)?;
    }
    let mut ctx = crate::inference::sampling::SamplingContext::new(logits_vec.len());
    let (token_id, logprob) = if !params.logprobs {
        let token_id = crate::inference::sampling::sample_token_with_history(
            &mut logits_vec,
            params,
            generated_ids,
            &mut ctx,
        );
        (token_id, None)
    } else {
        let (token_id, info) = crate::inference::sampling::sample_token_with_logprobs_history(
            &mut logits_vec,
            params,
            generated_ids,
            &mut ctx,
        );
        (token_id, info.map(|i| i.logprob))
    };
    if let Some(held) = held {
        held.took(token_id)?;
    }
    Ok((token_id, logprob))
}

/// How many sequence positions a wire-format activation buffer carries, read
/// from its shape header without decoding the payload.
///
/// Activations cross a segment boundary as `[batch, seq, hidden]` (a prompt
/// pass) or `[batch, 1, hidden]` (a decode step), so the position count is the
/// second-to-last dimension. A 2-D `[seq, hidden]` buffer is read the same way.
///
/// Exists for retention accounting (`daemon::state::retained_activations`),
/// which has to know a step's span to prove the history it holds is
/// CONTIGUOUS — a replay assembled from a history with a hole in it would
/// rebuild a plausible cache that is not the one the failed machine had, and
/// nothing downstream could tell. Reading the header is the cheap half of
/// `bytes_to_tensor`; it deliberately does not validate the payload, because
/// the payload is validated when the buffer is actually decoded.
pub fn activation_positions(bytes: &[u8]) -> Option<u32> {
    if bytes.len() < 4 {
        return None;
    }
    let ndim = u32::from_le_bytes(bytes[0..4].try_into().ok()?) as usize;
    if ndim < 2 || bytes.len() < 4 + ndim * 4 {
        return None;
    }
    let dim = |i: usize| -> Option<u32> {
        let off = 4 + i * 4;
        u32::from_le_bytes(bytes[off..off + 4].try_into().ok()?).into()
    };
    dim(ndim - 2)
}

/// How many token positions a forward carries — the ONE answer, read by the
/// serving node's "tokens served" and serve credit (`dispatch::layer_forward`)
/// and by the worker's own decode timing, which times a step of exactly one
/// (`process_pool::forward_is_one_decode_step`, FUTURE_WORK #3 (f)).
pub(crate) fn positions_carried(forward: &crate::types::LayerForward) -> u32 {
    forward_positions(
        &forward.activations,
        forward.pre_embedded || forward.layer_range.0 > 0,
        crate::inference::model_worker::first_segment_input_is_text(
            forward.index_pos,
            forward.prompt_cache.is_some(),
            forward.prompt_span.is_some(),
        ),
    )
}

/// How many token positions a forward's payload carries — the unit "tokens
/// served" and the serve credit are counted in. Asked through
/// [`positions_carried`].
///
/// Three payloads cross a segment boundary, and the byte length means something
/// different in each:
/// - **hidden states** (`hidden_states`: any segment past the first, or a
///   pre-embedded first one) — `[batch, seq, hidden]` in f32 or Q8_0, counted
///   from the shape header ([`activation_positions`]), because the length is
///   seq × hidden × bytes;
/// - **prompt text** (a first segment's whole prompt pass from position 0 —
///   `model_worker::first_segment_input_is_text`, the worker's own rule) — ~4
///   characters a token, an estimate kept cheap on the serving path;
/// - **token ids** (a first segment's later steps, a resumed pass, a piece of
///   a pass) — 8 bytes each: one for a decode step, the guesses for a check, a
///   piece's positions.
///
/// It divided every prompt pass's bytes by 4, which for hidden states is
/// tokens × hidden size: a 20-token prompt pass of a 2048-wide model counted as
/// 40,965 tokens, and the dashboard's "tokens served" and tok/s read ~2000×
/// high (field report #004, 2026-10-01). A later step counted 1 whatever it
/// carried, so a check's guesses were not counted at all.
pub(crate) fn forward_positions(activations: &[u8], hidden_states: bool, prompt_text: bool) -> u32 {
    let positions = if hidden_states {
        activation_positions(activations).map(|p| p as usize)
    } else if prompt_text {
        Some(String::from_utf8_lossy(activations).chars().count() / 4)
    } else {
        Some(activations.len() / 8)
    };
    positions.unwrap_or(1).clamp(1, u32::MAX as usize) as u32
}

/// Does a segment's RETURNED activation carry the same shape as the one it was
/// handed, in a well-formed payload for whatever encoding the peer chose?
///
/// **The shape is a property of the TENSOR, not of its bytes.** A hidden state
/// crosses a segment boundary as f32 or as Q8_0 (`tensor_to_bytes_q8_0`,
/// ~3.76x smaller), and which one is decided by the node that WROTE it — its own
/// `inference.activation_compression`. Two honest nodes that differ on that
/// setting carry the same `[1, 28, 4096]` as 458,772 bytes (f32) against the
/// 121,876 bytes (Q8_0) this node sent — measured from a public v0.3.200 peer. Comparing byte lengths called that "the wrong activation shape",
/// failed the healthy peer over, and ended the request when it had no standby
/// (FUTURE_WORK #98, seen at the .201 gate and again 2026-09-24).
///
/// What the old check existed for is kept: a malformed or mis-shaped tensor
/// from a broken or malicious peer is still refused HERE, before it is handed
/// to the next worker (gotcha #20). The returned payload must be long enough
/// for the shape and dtype it declares, by the same arithmetic `bytes_to_tensor`
/// uses, and its dtype must be one this build decodes.
pub fn activation_shape_matches(sent: &[u8], returned: &[u8]) -> bool {
    let header = |bytes: &[u8]| -> Option<(Vec<u32>, u32, usize)> {
        let word = |at: usize| -> Option<u32> {
            Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
        };
        let ndim = word(0)? as usize;
        if ndim == 0 || ndim > 8 {
            return None;
        }
        let shape: Vec<u32> = (0..ndim).map(|i| word(4 + i * 4)).collect::<Option<_>>()?;
        let dtype_at = 4 + ndim * 4;
        Some((shape, word(dtype_at)?, dtype_at + 4))
    };
    let (Some((want, _, _)), Some((got, dtype, payload_at))) = (header(sent), header(returned))
    else {
        return false;
    };
    if want != got {
        return false;
    }
    let Some(elements) = got
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
        .filter(|&n| n > 0)
    else {
        return false;
    };
    payload_len(dtype)
        .and_then(|len_of| len_of(elements))
        .is_some_and(|need| returned.len().saturating_sub(payload_at) >= need)
}

/// Several hidden states laid end to end along their positions, as ONE
/// payload in the same encoding — what `tensor_to_bytes` (or `_q8_0`) of the
/// concatenated tensor would have written, byte for byte, with no decode.
///
/// For the flow's batched tail (`dsd_flow`, FUTURE_WORK #369): answers the far
/// computer sent back close together are read here in one forward instead of
/// one each. `None` whenever the parts cannot be joined EXACTLY, and the caller
/// then reads them one at a time:
/// - a header that does not parse, or a payload shorter than its shape needs;
/// - parts that differ in encoding or in any dimension but the positions
///   (`ndim - 2`), or a leading dimension other than 1 (a batch of two would
///   interleave);
/// - a Q8_0 part whose values do not fill whole 32-value blocks — its last
///   block's scale would cover the next part's first values.
///
/// Only the bytes each shape needs are copied: a payload LONGER than its shape
/// is accepted by `bytes_to_tensor`, and its tail is not this tensor's.
pub(crate) fn join_positions(parts: &[&[u8]]) -> Option<Vec<u8>> {
    let parts: Vec<PositionsPart> = parts
        .iter()
        .map(|b| PositionsPart::parse(b))
        .collect::<Option<_>>()?;
    let first = parts.first()?;
    let at = first.shape.len() - 2;
    let mut positions: u32 = 0;
    for part in &parts {
        let same_but_positions = part.dtype == first.dtype
            && part.shape.len() == first.shape.len()
            && part
                .shape
                .iter()
                .zip(&first.shape)
                .enumerate()
                .all(|(i, (a, b))| i == at || a == b);
        if !same_but_positions {
            return None;
        }
        positions = positions.checked_add(part.shape[at])?;
    }
    let payload_bytes = parts
        .iter()
        .try_fold(0usize, |acc, p| acc.checked_add(p.payload.len()))?;
    let mut joined = Vec::with_capacity(4 + first.shape.len() * 4 + 4 + payload_bytes);
    joined.extend_from_slice(&(first.shape.len() as u32).to_le_bytes());
    for (i, &dim) in first.shape.iter().enumerate() {
        let dim = if i == at { positions } else { dim };
        joined.extend_from_slice(&dim.to_le_bytes());
    }
    joined.extend_from_slice(&first.dtype.to_le_bytes());
    for part in &parts {
        joined.extend_from_slice(part.payload);
    }
    Some(joined)
}

/// One hidden-state buffer as [`join_positions`] and [`take_positions`] read
/// it: its shape, its encoding and exactly the payload bytes the shape needs.
/// `None` for a header that does not parse, a leading dimension other than 1,
/// no values, a payload shorter than the shape needs, or a Q8_0 buffer whose
/// values do not fill whole 32-value blocks.
struct PositionsPart<'a> {
    shape: Vec<u32>,
    dtype: u32,
    payload: &'a [u8],
}

impl<'a> PositionsPart<'a> {
    fn parse(bytes: &'a [u8]) -> Option<Self> {
        let word = |at: usize| -> Option<u32> {
            Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
        };
        let ndim = word(0)? as usize;
        if !(2..=8).contains(&ndim) {
            return None;
        }
        let shape: Vec<u32> = (0..ndim).map(|i| word(4 + i * 4)).collect::<Option<_>>()?;
        if shape[..ndim - 2].iter().any(|&d| d != 1) {
            return None;
        }
        let dtype = word(4 + ndim * 4)?;
        let values = shape
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))
            .filter(|&n| n > 0)?;
        if dtype == DTYPE_TAG_Q8_0 && values % quant::GROUP_SIZE != 0 {
            return None;
        }
        let need = payload_len(dtype)?(values)?;
        let at = 4 + ndim * 4 + 4;
        let payload = bytes.get(at..at.checked_add(need)?)?;
        Some(Self {
            shape,
            dtype,
            payload,
        })
    }
}

/// Positions `from..to` of one hidden-state buffer, in its own encoding — what
/// `tensor_to_bytes` (or `_q8_0`) of the narrowed tensor would have written,
/// byte for byte, with no decode. The inverse of [`join_positions`].
///
/// For the continuous stream's spares (`dsd_stream`, FUTURE_WORK #381): one
/// forward through this node's layers carries the chunk's rows and a spare
/// restart's, and each goes to the far end as its own buffer. `None` wherever
/// the cut cannot be EXACT, and the caller decodes instead:
/// - anything [`PositionsPart::parse`] refuses;
/// - an empty range, or one past the buffer's positions;
/// - a Q8_0 buffer whose positions do not each fill whole 32-value blocks —
///   a block's one scale would cover values on both sides of the cut.
pub(crate) fn take_positions(bytes: &[u8], from: u32, to: u32) -> Option<Vec<u8>> {
    let part = PositionsPart::parse(bytes)?;
    let at = part.shape.len() - 2;
    if from >= to || to > part.shape[at] {
        return None;
    }
    let per_position = part.shape[at + 1..]
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d as usize))?;
    if part.dtype == DTYPE_TAG_Q8_0 && per_position % quant::GROUP_SIZE != 0 {
        return None;
    }
    let len_of = payload_len(part.dtype)?;
    let start = len_of((from as usize).checked_mul(per_position)?)?;
    let end = len_of((to as usize).checked_mul(per_position)?)?;
    let payload = part.payload.get(start..end)?;
    let mut taken = Vec::with_capacity(4 + part.shape.len() * 4 + 4 + payload.len());
    taken.extend_from_slice(&(part.shape.len() as u32).to_le_bytes());
    for (i, &dim) in part.shape.iter().enumerate() {
        let dim = if i == at { to - from } else { dim };
        taken.extend_from_slice(&dim.to_le_bytes());
    }
    taken.extend_from_slice(&part.dtype.to_le_bytes());
    taken.extend_from_slice(payload);
    Some(taken)
}

#[cfg(test)]
mod join_positions_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// Distinct values per position, so a wrong order or offset shows.
    fn hidden(positions: usize, width: usize, start: f32) -> Tensor {
        let values: Vec<f32> = (0..positions * width)
            .map(|i| start + i as f32 * 0.01 - (i % 7) as f32)
            .collect();
        Tensor::from_vec(values, (1, positions, width), &Device::Cpu).unwrap()
    }

    #[test]
    fn joined_parts_are_the_bytes_of_the_joined_tensor_in_either_encoding() {
        let parts = [hidden(3, 64, 0.0), hidden(1, 64, 50.0), hidden(5, 64, -9.0)];
        let whole = Tensor::cat(&parts, 1).unwrap();
        for encode in [tensor_to_bytes, tensor_to_bytes_q8_0] {
            let bytes: Vec<Vec<u8>> = parts.iter().map(|t| encode(t).unwrap()).collect();
            let refs: Vec<&[u8]> = bytes.iter().map(Vec::as_slice).collect();
            assert_eq!(join_positions(&refs).unwrap(), encode(&whole).unwrap());
        }
    }

    #[test]
    fn taken_positions_are_the_bytes_of_the_narrowed_tensor_in_either_encoding() {
        let whole = hidden(5, 64, 3.0);
        for encode in [tensor_to_bytes, tensor_to_bytes_q8_0] {
            let bytes = encode(&whole).unwrap();
            for (from, to) in [(0, 5), (0, 3), (3, 5), (2, 3)] {
                let narrowed = whole
                    .narrow(1, from as usize, (to - from) as usize)
                    .unwrap();
                assert_eq!(
                    take_positions(&bytes, from, to).unwrap(),
                    encode(&narrowed).unwrap(),
                    "{from}..{to}"
                );
            }
            assert_eq!(take_positions(&bytes, 2, 2), None, "an empty range");
            assert_eq!(take_positions(&bytes, 4, 6), None, "past the positions");
        }
        // A Q8_0 position of 48 values ends inside a block: no exact cut.
        let ragged = tensor_to_bytes_q8_0(&hidden(2, 48, 0.0)).unwrap();
        assert_eq!(take_positions(&ragged, 0, 1), None);
        // Cut and joined again, a buffer is itself.
        let bytes = tensor_to_bytes_q8_0(&whole).unwrap();
        let (a, b) = (
            take_positions(&bytes, 0, 2).unwrap(),
            take_positions(&bytes, 2, 5).unwrap(),
        );
        assert_eq!(join_positions(&[&a, &b]).unwrap(), bytes);
    }

    #[test]
    fn a_tail_past_the_shape_is_not_copied() {
        let a = tensor_to_bytes(&hidden(2, 32, 1.0)).unwrap();
        let mut a_long = a.clone();
        a_long.extend_from_slice(&[0xAB; 12]);
        let b = tensor_to_bytes(&hidden(1, 32, 2.0)).unwrap();
        assert_eq!(join_positions(&[&a_long, &b]), join_positions(&[&a, &b]));
    }

    #[test]
    fn parts_that_cannot_be_joined_exactly_are_refused() {
        let f32_part = tensor_to_bytes(&hidden(2, 64, 0.0)).unwrap();
        let q8_part = tensor_to_bytes_q8_0(&hidden(2, 64, 0.0)).unwrap();
        assert_eq!(
            join_positions(&[&f32_part, &q8_part]),
            None,
            "mixed encodings"
        );
        let wider = tensor_to_bytes(&hidden(2, 96, 0.0)).unwrap();
        assert_eq!(
            join_positions(&[&f32_part, &wider]),
            None,
            "different widths"
        );
        // 3 positions x 48 = 144 values: four whole blocks and half of a fifth.
        let partial = tensor_to_bytes_q8_0(&hidden(3, 48, 0.0)).unwrap();
        assert_eq!(
            join_positions(&[&partial, &partial]),
            None,
            "a partial block"
        );
        let batch_of_two =
            tensor_to_bytes(&Tensor::ones((2, 2, 64), DType::F32, &Device::Cpu).unwrap()).unwrap();
        assert_eq!(
            join_positions(&[&batch_of_two, &batch_of_two]),
            None,
            "a batch of two"
        );
        let mut short = f32_part.clone();
        short.truncate(short.len() - 1);
        assert_eq!(
            join_positions(&[&f32_part, &short]),
            None,
            "a short payload"
        );
        assert_eq!(join_positions(&[]), None, "nothing to join");
    }
}

#[cfg(test)]
mod activation_shape_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn hidden(positions: usize) -> Tensor {
        Tensor::ones((1, positions, 64), candle_core::DType::F32, &Device::Cpu).unwrap()
    }

    /// The field case: this node sends Q8_0, an honest peer on the other
    /// setting answers f32. Same tensor, 3.76x the bytes — not a wrong shape.
    #[test]
    fn a_differently_encoded_reply_of_the_same_shape_is_accepted() {
        let sent = tensor_to_bytes_q8_0(&hidden(28)).unwrap();
        let returned = tensor_to_bytes(&hidden(28)).unwrap();
        assert_ne!(
            sent.len(),
            returned.len(),
            "the premise: the byte lengths differ"
        );
        assert!(activation_shape_matches(&sent, &returned));
        assert!(
            activation_shape_matches(&returned, &sent),
            "and the other way round"
        );
    }

    /// What the check is FOR still holds: a different shape, a truncated
    /// payload or an unknown encoding is refused before the next worker sees it.
    #[test]
    fn a_wrong_shape_or_a_malformed_payload_is_still_refused() {
        let sent = tensor_to_bytes(&hidden(28)).unwrap();
        assert!(!activation_shape_matches(
            &sent,
            &tensor_to_bytes(&hidden(27)).unwrap()
        ));
        let mut truncated = tensor_to_bytes(&hidden(28)).unwrap();
        truncated.truncate(truncated.len() - 1);
        assert!(!activation_shape_matches(&sent, &truncated));
        let mut unknown = tensor_to_bytes(&hidden(28)).unwrap();
        let dtype_at = 4 + 3 * 4;
        unknown[dtype_at..dtype_at + 4].copy_from_slice(&7u32.to_le_bytes());
        assert!(!activation_shape_matches(&sent, &unknown));
        assert!(!activation_shape_matches(&sent, &[]));
    }
}

#[cfg(test)]
mod retention_header_tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn a_forwards_position_count_is_read_from_its_header() {
        // A prompt pass: 7 positions of a 4-wide hidden state.
        let prompt = Tensor::zeros((1, 7, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        let bytes = tensor_to_bytes(&prompt).unwrap();
        assert_eq!(activation_positions(&bytes), Some(7));

        // A decode step.
        let step = Tensor::zeros((1, 1, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            activation_positions(&tensor_to_bytes(&step).unwrap()),
            Some(1)
        );

        // The Q8_0 encoding carries the same header, so the count does not
        // depend on which encoder the sender chose.
        let q8 = tensor_to_bytes_q8_0(&prompt).unwrap();
        assert_eq!(activation_positions(&q8), Some(7));

        // Two dimensions, `[seq, hidden]`.
        let flat = Tensor::zeros((5, 4), candle_core::DType::F32, &Device::Cpu).unwrap();
        assert_eq!(
            activation_positions(&tensor_to_bytes(&flat).unwrap()),
            Some(5)
        );

        // Nothing readable rather than a guess.
        assert_eq!(activation_positions(&[]), None);
        assert_eq!(activation_positions(&[1, 0, 0, 0, 9, 9, 9, 9]), None);
    }
}
