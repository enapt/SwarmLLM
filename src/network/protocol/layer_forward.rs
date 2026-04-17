use crate::error::SwarmError;
use crate::types::{LayerForward, ModelId, TensorFormat};

use super::{MAX_ACTIVATION_SIZE, TENSOR_TAG_FORWARD};

pub fn encode_layer_forward(forward: &LayerForward) -> Result<Vec<u8>, SwarmError> {
    let data_len = forward.activations.len();
    let model_id_bytes = forward.model_id.0.as_bytes();
    if model_id_bytes.len() > u16::MAX as usize {
        return Err(SwarmError::Network(format!(
            "Model ID too long: {} bytes (max {})",
            model_id_bytes.len(),
            u16::MAX
        )));
    }
    // Header: tag(1) + uuid(16) + seq(4) + index_pos(4) + fmt(1) + data_len(4) = 30
    // Trailer: marker(1) + layer_start(4) + layer_end(4) + model_id_len(2) + model_id(N)
    let trailer_len = 1 + 4 + 4 + 2 + model_id_bytes.len();
    let total = 1 + 29 + data_len + trailer_len;
    let mut buf = Vec::with_capacity(total);

    // Message type tag
    buf.push(TENSOR_TAG_FORWARD);
    // UUID (16 bytes)
    buf.extend_from_slice(forward.request_id.as_bytes());
    // sequence_num (4 bytes LE)
    buf.extend_from_slice(&forward.sequence_num.to_le_bytes());
    // index_pos (4 bytes LE)
    buf.extend_from_slice(&forward.index_pos.to_le_bytes());
    // format tag (1 byte)
    let fmt_tag: u8 = match forward.format {
        TensorFormat::FP16 => 0,
        TensorFormat::FP32 => 1,
        TensorFormat::INT8 => 2,
    };
    buf.push(fmt_tag);
    // data length (4 bytes LE) — guard against >4GiB payloads
    if data_len > u32::MAX as usize {
        return Err(SwarmError::Network(
            "Activation payload exceeds 4GiB wire format limit".into(),
        ));
    }
    buf.extend_from_slice(&(data_len as u32).to_le_bytes());
    // activation data
    buf.extend_from_slice(&forward.activations);

    // Required trailer: layer_range + model_id
    let (layer_start, layer_end) = forward.layer_range;
    buf.push(0x01); // marker byte
    buf.extend_from_slice(&layer_start.to_le_bytes());
    buf.extend_from_slice(&layer_end.to_le_bytes());
    // model_id: 2-byte length prefix + UTF-8 string
    buf.extend_from_slice(&(model_id_bytes.len() as u16).to_le_bytes());
    buf.extend_from_slice(model_id_bytes);

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

    Ok(buf)
}

/// Decode a binary tensor envelope back into a LayerForward.
/// Expects the 1-byte tag prefix to already be stripped (or handles both cases).
pub fn decode_layer_forward(data: &[u8]) -> Result<LayerForward, SwarmError> {
    // Skip the tag byte if present
    let data = if !data.is_empty() && data[0] == TENSOR_TAG_FORWARD {
        &data[1..]
    } else {
        data
    };

    // Header: uuid(16) + seq(4) + index_pos(4) + fmt(1) + data_len(4) = 29
    if data.len() < 29 {
        return Err(SwarmError::Network("Tensor envelope too short".to_string()));
    }

    let request_id = uuid::Uuid::from_bytes(
        data[0..16]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid UUID bytes".into()))?,
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
    let data_len = u32::from_le_bytes(
        data[25..29]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid data_len".into()))?,
    ) as usize;

    // SEC: Cap activation size to prevent memory exhaustion from malicious peers
    if data_len > MAX_ACTIVATION_SIZE {
        return Err(SwarmError::Network(format!(
            "LayerForward activation too large: {} bytes (max {})",
            data_len, MAX_ACTIVATION_SIZE
        )));
    }
    if data.len() < 29 + data_len {
        return Err(SwarmError::Network(format!(
            "Tensor data truncated: expected {} bytes, got {}",
            data_len,
            data.len() - 29
        )));
    }

    let activations = data[29..29 + data_len].to_vec();

    // Read required trailer: marker(1) + layer_start(4) + layer_end(4) + model_id_len(2) + model_id(N)
    let trailer_start = 29 + data_len;
    if data.len() < trailer_start + 9 || data[trailer_start] != 0x01 {
        return Err(SwarmError::Network(
            "LayerForward missing required layer_range/model_id trailer".to_string(),
        ));
    }
    let ls = u32::from_le_bytes(
        data[trailer_start + 1..trailer_start + 5]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid layer_start".into()))?,
    );
    let le = u32::from_le_bytes(
        data[trailer_start + 5..trailer_start + 9]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid layer_end".into()))?,
    );
    let layer_range = (ls, le);

    // Read model_id: 2-byte length prefix + UTF-8 string
    let mid_len_start = trailer_start + 9;
    if data.len() < mid_len_start + 2 {
        return Err(SwarmError::Network(
            "LayerForward missing model_id length".to_string(),
        ));
    }
    let mid_len = u16::from_le_bytes(
        data[mid_len_start..mid_len_start + 2]
            .try_into()
            .map_err(|_| SwarmError::Network("Invalid model_id_len".into()))?,
    ) as usize;
    let mid_start = mid_len_start + 2;
    if data.len() < mid_start + mid_len {
        return Err(SwarmError::Network(
            "LayerForward model_id truncated".to_string(),
        ));
    }
    let model_id_str = std::str::from_utf8(&data[mid_start..mid_start + mid_len])
        .map_err(|_| SwarmError::Network("Invalid model_id UTF-8".into()))?;
    let model_id = ModelId(model_id_str.to_string());

    // Optional: tp_meta trailer (marker 0x02 + tp_rank(1) + tp_size(1) + single_layer(4) + phase(1))
    let tp_meta_start = mid_start + mid_len;
    let (tp_meta, tp_pre_embedded) =
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
            )
        } else {
            (None, false)
        };

    Ok(LayerForward {
        request_id,
        sequence_num,
        index_pos,
        activations,
        format,
        model_id,
        layer_range,
        tp_meta,
        vision_embeddings: None,
        sender_peer_bytes: None,
        requester_node_id: None,
        pre_embedded: tp_pre_embedded,
        adapter_id: None,
    })
}

// Binary layout for LayerResult (v2 with activations):
//   [0..16]      request_id (UUID bytes)
//   [16..20]     num_tokens (u32 LE)
//   [20..20+n*4] token_ids (each u32 LE)
//   [T]          finish_reason tag: 0=None, 1=Stop, 2=MaxTokens, 3=Error
//   [T+1..]      if tag=3: error message (UTF-8 bytes) followed by [4B activations_len][activations]
//                if tag!=3: [4B activations_len][activations data]
