use crate::error::SwarmError;
use crate::types::{LayerResult, NetworkFinishReason};

use super::{MAX_ACTIVATION_SIZE, MAX_RESULT_TOKENS, TENSOR_TAG_RESULT};

pub fn encode_layer_result(result: &LayerResult) -> Result<Vec<u8>, SwarmError> {
    let num_tokens = result.token_ids.len();
    if num_tokens > MAX_RESULT_TOKENS {
        return Err(SwarmError::Network(format!(
            "LayerResult token_ids too large: {num_tokens} > {MAX_RESULT_TOKENS}"
        )));
    }
    if result.activations.len() > MAX_ACTIVATION_SIZE {
        return Err(SwarmError::Network(format!(
            "LayerResult activations too large: {} > {MAX_ACTIVATION_SIZE}",
            result.activations.len()
        )));
    }
    let mut buf = Vec::with_capacity(1 + 25 + num_tokens * 4 + result.activations.len());

    // Message type tag
    buf.push(TENSOR_TAG_RESULT);
    buf.extend_from_slice(result.request_id.as_bytes());
    buf.extend_from_slice(&(num_tokens as u32).to_le_bytes());
    for &token in &result.token_ids {
        buf.extend_from_slice(&token.to_le_bytes());
    }

    match &result.finish_reason {
        None => buf.push(0),
        Some(NetworkFinishReason::Stop) => buf.push(1),
        Some(NetworkFinishReason::MaxTokens) => buf.push(2),
        Some(NetworkFinishReason::Error(msg)) => {
            buf.push(3);
            // Error message length + message — capped to match the 4KB decode-side limit.
            let bytes = msg.as_bytes();
            let cap = bytes.len().min(4096);
            // Slice on a UTF-8 boundary ≤ cap.
            let mut end = cap;
            while end > 0 && !msg.is_char_boundary(end) {
                end -= 1;
            }
            let slice = &bytes[..end];
            buf.extend_from_slice(&(slice.len() as u32).to_le_bytes());
            buf.extend_from_slice(slice);
        }
    }

    // Append activations (for intermediate pipeline segments)
    buf.extend_from_slice(&(result.activations.len() as u32).to_le_bytes());
    buf.extend_from_slice(&result.activations);

    Ok(buf)
}

/// Decode binary into a LayerResult.
/// Expects the 1-byte tag prefix to already be stripped (or handles both cases).
pub fn decode_layer_result(data: &[u8]) -> Result<LayerResult, SwarmError> {
    // Skip the tag byte if present
    let data = if !data.is_empty() && data[0] == TENSOR_TAG_RESULT {
        &data[1..]
    } else {
        data
    };

    if data.len() < 21 {
        return Err(SwarmError::Network(
            "LayerResult envelope too short".to_string(),
        ));
    }

    let request_id = uuid::Uuid::from_bytes(
        data[0..16]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid UUID".into()))?,
    );
    let num_tokens = u32::from_le_bytes(
        data[16..20]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid num_tokens".into()))?,
    ) as usize;

    // SECURITY: Cap num_tokens to prevent OOM from crafted messages
    if num_tokens > MAX_RESULT_TOKENS {
        return Err(SwarmError::Network(format!(
            "num_tokens exceeds maximum ({MAX_RESULT_TOKENS})"
        )));
    }

    let tokens_end = 20 + num_tokens * 4;
    if data.len() < tokens_end + 1 {
        return Err(SwarmError::Network("LayerResult truncated".to_string()));
    }

    let mut token_ids = Vec::with_capacity(num_tokens);
    for i in 0..num_tokens {
        let start = 20 + i * 4;
        let token = u32::from_le_bytes(
            data[start..start + 4]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid token id".into()))?,
        );
        token_ids.push(token);
    }

    let mut pos = tokens_end;
    let finish_reason = match data[pos] {
        0 => {
            pos += 1;
            None
        }
        1 => {
            pos += 1;
            Some(NetworkFinishReason::Stop)
        }
        2 => {
            pos += 1;
            Some(NetworkFinishReason::MaxTokens)
        }
        3 => {
            pos += 1;
            // Error: read message length + message
            if pos + 4 <= data.len() {
                let msg_len_bytes: [u8; 4] = data[pos..pos + 4].try_into().unwrap_or([0; 4]); // safe: bounds already checked above
                let msg_len = u32::from_le_bytes(msg_len_bytes) as usize;
                pos += 4;
                // SEC: Cap error message to 4KB to prevent 256MB allocation from oversized msg_len
                let capped_len = msg_len.min(4096).min(data.len() - pos);
                let msg = String::from_utf8_lossy(&data[pos..pos + capped_len]).to_string();
                pos += msg_len.min(data.len() - pos); // advance past full message even if truncated
                Some(NetworkFinishReason::Error(msg))
            } else {
                Some(NetworkFinishReason::Error(String::new()))
            }
        }
        t => {
            return Err(SwarmError::Network(format!(
                "Unknown finish reason tag: {t}"
            )))
        }
    };

    // Read activations if present (capped at 128MB to prevent abuse)
    let activations = if pos + 4 <= data.len() {
        let act_len = u32::from_le_bytes(
            data[pos..pos + 4]
                .try_into()
                .expect("slice is exactly 4 bytes after bounds check"),
        ) as usize;
        pos += 4;
        if act_len > MAX_ACTIVATION_SIZE {
            return Err(SwarmError::Network(format!(
                "Activation data too large: {act_len} bytes"
            )));
        }
        if act_len > 0 && pos + act_len <= data.len() {
            data[pos..pos + act_len].to_vec()
        } else {
            vec![]
        }
    } else {
        vec![]
    };

    Ok(LayerResult {
        request_id,
        token_ids,
        finish_reason,
        activations,
        sealed_token_ids: None,
    })
}

// ---- Encrypted Tensor Encoding ----
//
// Wire format for TENSOR_TAG_ENCRYPTED (0x10):
//   [0]        tag = 0x10
//   [1..17]    request_id (UUID, 16 bytes) — cleartext AAD
//   [17..21]   sequence_num (u32 LE) — cleartext AAD
//   [21..25]   index_pos (u32 LE) — cleartext AAD
//   [25]       format tag (0=FP16, 1=FP32, 2=INT8) — cleartext AAD
//   [26..30]   layer_start (u32 LE) — cleartext AAD
//   [30..34]   layer_end (u32 LE) — cleartext AAD
//   [34..36]   model_id_len (u16 LE) — cleartext AAD
//   [36..36+M] model_id (UTF-8) — cleartext AAD
//   [36+M..40+M] sealed_len (u32 LE)
//   [40+M..]   sealed activations (nonce + ciphertext + AEAD tag)
//
// The AAD for the AEAD is the header bytes [1..36+M].
