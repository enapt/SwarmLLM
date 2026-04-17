use crate::error::SwarmError;
use crate::types::{LayerForward, ModelId, TensorFormat};

use super::TENSOR_TAG_ENCRYPTED;

pub fn encode_layer_forward_encrypted(
    forward: &LayerForward,
    sealed_activations: Vec<u8>,
) -> Result<Vec<u8>, SwarmError> {
    let sealed_len = sealed_activations.len();
    let model_id_bytes = forward.model_id.0.as_bytes();
    if model_id_bytes.len() > u16::MAX as usize {
        return Err(SwarmError::Network(format!(
            "Model ID too long: {} bytes (max {})",
            model_id_bytes.len(),
            u16::MAX
        )));
    }
    let total = 1 + 25 + 8 + 2 + model_id_bytes.len() + 4 + sealed_len;
    let mut buf = Vec::with_capacity(total);

    buf.push(TENSOR_TAG_ENCRYPTED);
    buf.extend_from_slice(forward.request_id.as_bytes());
    buf.extend_from_slice(&forward.sequence_num.to_le_bytes());
    buf.extend_from_slice(&forward.index_pos.to_le_bytes());
    let fmt_tag: u8 = match forward.format {
        TensorFormat::FP16 => 0,
        TensorFormat::FP32 => 1,
        TensorFormat::INT8 => 2,
    };
    buf.push(fmt_tag);
    // layer_range (required)
    let (layer_start, layer_end) = forward.layer_range;
    buf.extend_from_slice(&layer_start.to_le_bytes());
    buf.extend_from_slice(&layer_end.to_le_bytes());
    // model_id
    buf.extend_from_slice(&(model_id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(model_id_bytes);
    // sealed payload
    buf.extend_from_slice(&(sealed_len as u32).to_le_bytes());
    buf.extend_from_slice(&sealed_activations);

    // Optional: tp_meta trailer (marker 0x02 + tp_rank(1) + tp_size(1) + single_layer(4) + phase(1))
    if let Some(ref tp) = forward.tp_meta {
        buf.push(0x02);
        buf.push(tp.tp_rank);
        buf.push(tp.tp_size);
        buf.extend_from_slice(&tp.single_layer.to_le_bytes());
        let phase_byte: u8 = match tp.phase {
            crate::types::TpPhase::Full => 0,
            crate::types::TpPhase::AttnOnly => 1,
            crate::types::TpPhase::FfnOnly => 2,
            crate::types::TpPhase::EmbedOnly => 3,
        };
        buf.push(phase_byte);
        buf.push(if forward.pre_embedded { 1 } else { 0 });
    }

    // Optional: speculative trailer (marker 0x03). In encrypted mode, the draft
    // tokens travel in the cleartext trailer — they are not sensitive (they
    // are just candidate IDs). The sealed activations already carry the
    // coordinator's position; drafts ride alongside as plaintext metadata.
    if !forward.draft_tokens.is_empty() {
        if forward.draft_tokens.len() > u16::MAX as usize {
            return Err(SwarmError::Network(format!(
                "draft_tokens too long: {} > {}",
                forward.draft_tokens.len(),
                u16::MAX
            )));
        }
        buf.push(0x03);
        let flags: u8 = if forward.spec_logits_requested { 1 } else { 0 };
        buf.push(flags);
        buf.extend_from_slice(&(forward.draft_tokens.len() as u16).to_le_bytes());
        for t in &forward.draft_tokens {
            buf.extend_from_slice(&t.to_le_bytes());
        }
    }

    Ok(buf)
}

/// Decode an encrypted tensor envelope.
/// Returns the cleartext header fields (as a LayerForward with empty activations)
/// plus the sealed activation bytes, along with the AAD bytes.
/// The caller must decrypt the sealed bytes using the SessionManager.
pub fn decode_layer_forward_encrypted(
    data: &[u8],
) -> Result<(LayerForward, Vec<u8>, Vec<u8>), SwarmError> {
    // Skip tag byte if present
    let data = if !data.is_empty() && data[0] == TENSOR_TAG_ENCRYPTED {
        &data[1..]
    } else {
        data
    };

    // Header: uuid(16) + seq(4) + idx_pos(4) + fmt(1) + layer_start(4) + layer_end(4) + model_id_len(2) = 35
    if data.len() < 35 {
        return Err(SwarmError::Network(
            "Encrypted tensor envelope too short".to_string(),
        ));
    }

    let request_id = uuid::Uuid::from_bytes(
        data[0..16]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid UUID".into()))?,
    );
    let sequence_num = u32::from_le_bytes(
        data[16..20]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid sequence_num".into()))?,
    );
    let index_pos = u32::from_le_bytes(
        data[20..24]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid index_pos".into()))?,
    );
    let format = match data[24] {
        0 => TensorFormat::FP16,
        1 => TensorFormat::FP32,
        2 => TensorFormat::INT8,
        t => {
            return Err(SwarmError::Network(format!(
                "Unknown tensor format tag: {t}"
            )))
        }
    };
    let layer_start = u32::from_le_bytes(
        data[25..29]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid layer_start".into()))?,
    );
    let layer_end = u32::from_le_bytes(
        data[29..33]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid layer_end".into()))?,
    );
    let mid_len = u16::from_le_bytes(
        data[33..35]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid model_id_len".into()))?,
    ) as usize;

    let mid_start = 35;
    if data.len() < mid_start + mid_len + 4 {
        return Err(SwarmError::Network(
            "Encrypted tensor model_id/sealed truncated".to_string(),
        ));
    }
    let model_id_str = std::str::from_utf8(&data[mid_start..mid_start + mid_len])
        .map_err(|_| SwarmError::Network("Invalid model_id UTF-8".into()))?;
    let model_id = ModelId(model_id_str.to_string());

    // AAD is everything from uuid through model_id (before sealed_len)
    let aad_end = mid_start + mid_len;
    let aad = data[..aad_end].to_vec();

    let sealed_len_start = aad_end;
    let sealed_len = u32::from_le_bytes(
        data[sealed_len_start..sealed_len_start + 4]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid sealed_len".into()))?,
    ) as usize;

    let sealed_start = sealed_len_start + 4;
    if data.len() < sealed_start + sealed_len {
        return Err(SwarmError::Network(
            "Encrypted tensor data truncated".to_string(),
        ));
    }

    let sealed = data[sealed_start..sealed_start + sealed_len].to_vec();

    // Optional: tp_meta trailer after sealed data (marker 0x02 + 7 bytes)
    let tp_meta_start = sealed_start + sealed_len;
    let (tp_meta, tp_pre_embedded, mut cursor) =
        if data.len() >= tp_meta_start + 9 && data[tp_meta_start] == 0x02 {
            let tp_rank = data[tp_meta_start + 1];
            let tp_size = data[tp_meta_start + 2];
            let single_layer = u32::from_le_bytes(
                data[tp_meta_start + 3..tp_meta_start + 7]
                    .try_into()
                    .map_err(|_| SwarmError::Network("Invalid tp single_layer".into()))?,
            );
            let phase = match data[tp_meta_start + 7] {
                1 => crate::types::TpPhase::AttnOnly,
                2 => crate::types::TpPhase::FfnOnly,
                3 => crate::types::TpPhase::EmbedOnly,
                _ => crate::types::TpPhase::Full,
            };
            let pre_embedded = data[tp_meta_start + 8] != 0;
            (
                Some(crate::types::TensorParallelMeta {
                    tp_rank,
                    tp_size,
                    single_layer,
                    phase,
                }),
                pre_embedded,
                tp_meta_start + 9,
            )
        } else {
            (None, false, tp_meta_start)
        };

    // Optional: speculative trailer (marker 0x03)
    let (draft_tokens, spec_logits_requested) = if data.len() >= cursor + 4 && data[cursor] == 0x03
    {
        let flags = data[cursor + 1];
        let num_drafts = u16::from_le_bytes(
            data[cursor + 2..cursor + 4]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid num_drafts".into()))?,
        ) as usize;
        cursor += 4;
        if data.len() < cursor + num_drafts * 4 {
            return Err(SwarmError::Network("draft_tokens truncated".into()));
        }
        let mut drafts = Vec::with_capacity(num_drafts);
        for i in 0..num_drafts {
            let off = cursor + i * 4;
            drafts.push(u32::from_le_bytes(
                data[off..off + 4]
                    .try_into()
                    .map_err(|_| SwarmError::Network("Invalid draft token".into()))?,
            ));
        }
        (drafts, flags & 0x01 != 0)
    } else {
        (Vec::new(), false)
    };

    let forward = LayerForward {
        request_id,
        sequence_num,
        index_pos,
        activations: vec![], // Will be filled after decryption
        format,
        model_id,
        layer_range: (layer_start, layer_end),
        tp_meta,
        vision_embeddings: None,
        sender_peer_bytes: None,
        requester_node_id: None,
        pre_embedded: tp_pre_embedded,
        adapter_id: None,
        draft_tokens,
        spec_logits_requested,
    };

    Ok((forward, sealed, aad))
}

// Serde impls for SwarmRequest/SwarmResponse
// Note: TensorPayload variants are never JSON-serialized (handled by binary codec path),
