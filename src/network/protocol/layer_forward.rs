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

    // Optional: speculative trailer (marker 0x03 + flags(1) + num_drafts(2 LE) + drafts*4)
    //
    // Emitted when EITHER `draft_tokens` is non-empty OR `spec_logits_requested`
    // is set. Gating on `draft_tokens.is_empty()` alone silently dropped
    // `spec_logits_requested = true` from the wire for the DSD verify path
    // (`build_spec_verify_forward` deliberately leaves `draft_tokens` empty
    // because the IDs are already encoded in `activations`). Decoders ignore
    // unknown trailers, so emitting an empty-drafts trailer for older peers
    // is a no-op extension.
    if !forward.draft_tokens.is_empty() || forward.spec_logits_requested {
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

    // Optional: KV truncation trailer (marker 0x04 + target_len(4 LE)). Used
    // by the speculative coordinator after partial acceptance to discard the
    // trailing γ-k stale draft entries written in the previous round.
    if let Some(target_len) = forward.truncate_kv_to {
        buf.push(0x04);
        buf.extend_from_slice(&target_len.to_le_bytes());
    }

    // Optional: chunk-meta trailer (marker 0x05 + chunk_idx(4 LE) +
    // total_chunks(4 LE)) — Tier 4K STREAM-chunked activation send.
    if let Some(cm) = forward.chunk_meta {
        buf.push(0x05);
        buf.extend_from_slice(&cm.chunk_idx.to_le_bytes());
        buf.extend_from_slice(&cm.total_chunks.to_le_bytes());
    }

    // Optional: next-hop trailer (0x06) and reply-to trailer (0x07) — direct
    // peer chaining. One writer for the plaintext frame, the encrypted frame
    // and the AAD, so the three cannot disagree about these bytes.
    append_chain_trailers(&mut buf, forward);

    Ok(buf)
}

/// Write the chaining trailers: `0x06 | n | n × (node_id(32) | layer_start(4
/// LE) | layer_end(4 LE))` when hops remain, and `0x07 | node_id(32)` — the
/// COORDINATOR the run must answer — whenever the forward names its requester.
///
/// The two are independent on purpose: the LAST hop of a chain receives a
/// forward with NO remaining hops and still has to know whom to answer. (The
/// first cut emitted 0x07 only next to 0x06 and so dropped it on exactly that
/// hop; observed on two machines, 2026-08-21.) The one-hop frame every
/// released node sends and expects stays byte-identical because the
/// coordinator sets `requester_node_id` ONLY on a chained send — see
/// `PipelineExecutor::forward_through_segments` — and every hop copies it
/// onward; each other sender leaves it `None`.
///
/// Shared by `encode_layer_forward`, the encrypted encoder and
/// `build_layer_forward_aad`: all three wrote the 0x06 bytes by hand before
/// 2026-08-21, which is how a fourth copy would have drifted. Both trailers are
/// AAD-bound, so a peer can neither be redirected to another next hop nor told
/// to answer a different coordinator without failing authentication.
///
/// Why 0x07 exists: `requester_node_id` was never on the wire. Every decoder
/// left it `None`, and the serving side fell back to answering the forward's
/// SENDER — correct for one hop, and exactly wrong for a chain, where the
/// sender is the previous hop: the tail answered its predecessor, which had
/// handed the work on and dropped the reply, and the coordinator waited out
/// its whole deadline (observed on two machines). The unit tests built
/// forwards with the field set and never saw it. It is emitted only next to a
/// chain, so the one-hop frame is byte-identical to what every released node
/// sends and expects.
pub(crate) fn append_chain_trailers(buf: &mut Vec<u8>, forward: &LayerForward) {
    if !forward.chain.is_empty() {
        // Bounded by a byte: a pipeline with more than 255 remaining segments
        // is not a routing decision, it is a bug, and truncating is safer than
        // emitting a length nobody can parse.
        let n = forward.chain.len().min(u8::MAX as usize);
        buf.push(0x06);
        buf.push(n as u8);
        for hop in forward.chain.iter().take(n) {
            buf.extend_from_slice(&hop.node_id.0);
            buf.extend_from_slice(&hop.layer_range.0.to_le_bytes());
            buf.extend_from_slice(&hop.layer_range.1.to_le_bytes());
        }
    }
    if let Some(reply_to) = forward.requester_node_id {
        buf.push(0x07);
        buf.extend_from_slice(&reply_to);
    }
}

/// Read the reply-to trailer (`0x07 | node_id(32)`) at `cursor`, if present.
/// It follows the (optional) chain trailer; callers pass the cursor
/// `read_chain_trailer` left. Absent on every released node's frame and on a
/// v1 chain sender's, in which case the receiver keeps the old behaviour
/// (answer the sender) — and the planner does not chain to v1 peers in the
/// first place (`features::PIPELINE_CHAIN_V2`).
pub(crate) fn read_reply_to_trailer(data: &[u8], cursor: &mut usize) -> Option<[u8; 32]> {
    if data.len() < *cursor + 33 || data[*cursor] != 0x07 {
        return None;
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&data[*cursor + 1..*cursor + 33]);
    *cursor += 33;
    Some(id)
}

/// Read a next-hop trailer at `cursor`, if one is present.
///
/// Shared by the plaintext and encrypted decoders so the two cannot disagree
/// about the wire form — the divergence this codebase keeps being caught by.
pub(crate) fn read_chain_trailer(data: &[u8], cursor: &mut usize) -> Vec<crate::types::ChainHop> {
    if data.len() < *cursor + 2 || data[*cursor] != 0x06 {
        return Vec::new();
    }
    let count = data[*cursor + 1] as usize;
    let need = 2 + count * 40;
    if count == 0 || data.len() < *cursor + need {
        return Vec::new();
    }
    let mut hops = Vec::with_capacity(count);
    for i in 0..count {
        let at = *cursor + 2 + i * 40;
        let mut id = [0u8; 32];
        id.copy_from_slice(&data[at..at + 32]);
        let start = u32::from_le_bytes(match data[at + 32..at + 36].try_into() {
            Ok(b) => b,
            Err(_) => return Vec::new(),
        });
        let end = u32::from_le_bytes(match data[at + 36..at + 40].try_into() {
            Ok(b) => b,
            Err(_) => return Vec::new(),
        });
        // A range that cannot be served is worse than no hop at all: the
        // receiver would forward into nothing. Refuse the whole chain and let
        // the result come home, which costs a round trip and always works.
        if end <= start {
            return Vec::new();
        }
        hops.push(crate::types::ChainHop {
            node_id: crate::types::NodeId(id),
            layer_range: (start, end),
        });
    }
    *cursor += need;
    hops
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

    // Optional: speculative trailer (marker 0x03 + flags(1) + num_drafts(2 LE) + drafts*4)
    // Unknown to older decoders — presence is required to be gated by
    // PipelineAssignment.supports_speculative on the sender.
    let (draft_tokens, spec_logits_requested) = if data.len() >= cursor + 4 && data[cursor] == 0x03
    {
        let flags = data[cursor + 1];
        let num_drafts = u16::from_le_bytes(
            data[cursor + 2..cursor + 4]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid num_drafts".into()))?,
        ) as usize;
        cursor += 4;
        // R107/R108: cap peer-controlled num_drafts to keep
        // `Vec::with_capacity` and the subsequent loop bounded. Shared
        // with `encrypted.rs` via `super::MAX_DRAFT_TOKENS` so the
        // plaintext and encrypted decoders cannot drift apart.
        if num_drafts > super::MAX_DRAFT_TOKENS {
            return Err(SwarmError::Network(format!(
                "num_drafts {num_drafts} > {}",
                super::MAX_DRAFT_TOKENS
            )));
        }
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
        cursor += num_drafts * 4;
        (drafts, flags & 0x01 != 0)
    } else {
        (Vec::new(), false)
    };

    // Optional: KV truncation trailer (marker 0x04 + target_len(4 LE))
    let truncate_kv_to = if data.len() >= cursor + 5 && data[cursor] == 0x04 {
        let len = u32::from_le_bytes(
            data[cursor + 1..cursor + 5]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid truncate_kv_to".into()))?,
        );
        cursor += 5;
        Some(len)
    } else {
        None
    };

    // Optional: chunk-meta trailer (marker 0x05 + chunk_idx(4 LE) + total_chunks(4 LE)).
    // Carries STREAM-style chunk binding for Tier 4K daemon-side chunked
    // activation send. Bound into AAD via `build_layer_forward_aad` so
    // chunk reorder / wrong-total attempts fail authentication. Frames
    // without this trailer are treated as `(chunk_idx=0, total_chunks=1)`
    // — single-chunk implicit (the common-case decode wire-form pre-R139).
    let chunk_meta = if data.len() >= cursor + 9 && data[cursor] == 0x05 {
        let chunk_idx = u32::from_le_bytes(
            data[cursor + 1..cursor + 5]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid chunk_idx".into()))?,
        );
        let total_chunks = u32::from_le_bytes(
            data[cursor + 5..cursor + 9]
                .try_into()
                .map_err(|_| SwarmError::Network("Invalid total_chunks".into()))?,
        );
        // SEC: validate self-consistency. A peer sending (idx=5, total=3)
        // would otherwise pass through and crash the assembly state
        // machine.
        if total_chunks == 0 || chunk_idx >= total_chunks {
            return Err(SwarmError::Network(format!(
                "Invalid chunk_meta: chunk_idx={chunk_idx}, total_chunks={total_chunks}"
            )));
        }
        Some(crate::types::ChunkMeta {
            chunk_idx,
            total_chunks,
        })
    } else {
        None
    };
    // The chunk-meta block reads without advancing, because it used to be the
    // last trailer. It no longer is.
    if chunk_meta.is_some() {
        cursor += 9;
    }

    // Optional: next-hop trailer (0x06) — direct peer chaining — and, only
    // after one, the reply-to trailer (0x07) naming the coordinator.
    let chain = read_chain_trailer(data, &mut cursor);
    let requester_node_id = read_reply_to_trailer(data, &mut cursor);
    let _ = cursor;

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
        chain,
        sender_peer_bytes: None,
        requester_node_id,
        pre_embedded: tp_pre_embedded,
        generated_ids: Vec::new(),
        adapter_id: None,
        draft_tokens,
        spec_logits_requested,
        truncate_kv_to,
        chunk_meta,
        sampling: None,
    })
}

// Binary layout for LayerResult (v2 with activations):
//   [0..16]      request_id (UUID bytes)
//   [16..20]     num_tokens (u32 LE)
//   [20..20+n*4] token_ids (each u32 LE)
//   [T]          finish_reason tag: 0=None, 1=Stop, 2=MaxTokens, 3=Error
//   [T+1..]      if tag=3: error message (UTF-8 bytes) followed by [4B activations_len][activations]
//                if tag!=3: [4B activations_len][activations data]

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{TensorParallelMeta, TpPhase};

    fn base_forward() -> LayerForward {
        LayerForward {
            request_id: uuid::Uuid::from_u128(0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00),
            sequence_num: 7,
            index_pos: 13,
            activations: vec![0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04],
            format: TensorFormat::FP16,
            model_id: ModelId("qwen2.5-7b".into()),
            layer_range: (0, 28),
            tp_meta: None,
            vision_embeddings: None,
            chain: Vec::new(),
            sender_peer_bytes: None,
            requester_node_id: None,
            pre_embedded: false,
            generated_ids: Vec::new(),
            adapter_id: None,
            draft_tokens: Vec::new(),
            spec_logits_requested: false,
            truncate_kv_to: None,
            chunk_meta: None,
            sampling: None,
        }
    }

    fn assert_roundtrip_eq(orig: &LayerForward, decoded: &LayerForward) {
        assert_eq!(decoded.request_id, orig.request_id);
        assert_eq!(decoded.sequence_num, orig.sequence_num);
        assert_eq!(decoded.index_pos, orig.index_pos);
        assert_eq!(decoded.activations, orig.activations);
        assert!(matches!(
            (&decoded.format, &orig.format),
            (TensorFormat::FP16, TensorFormat::FP16)
                | (TensorFormat::FP32, TensorFormat::FP32)
                | (TensorFormat::INT8, TensorFormat::INT8)
        ));
        assert_eq!(decoded.model_id, orig.model_id);
        assert_eq!(decoded.layer_range, orig.layer_range);
        assert_eq!(decoded.pre_embedded, orig.pre_embedded);
        assert_eq!(decoded.draft_tokens, orig.draft_tokens);
        assert_eq!(decoded.spec_logits_requested, orig.spec_logits_requested);
        assert_eq!(decoded.truncate_kv_to, orig.truncate_kv_to);
        assert_eq!(decoded.chunk_meta, orig.chunk_meta);
        match (&decoded.tp_meta, &orig.tp_meta) {
            (Some(a), Some(b)) => {
                assert_eq!(a.tp_rank, b.tp_rank);
                assert_eq!(a.tp_size, b.tp_size);
                assert_eq!(a.single_layer, b.single_layer);
                assert_eq!(a.phase, b.phase);
            }
            (None, None) => {}
            _ => panic!("tp_meta presence mismatch"),
        }
    }

    #[test]
    fn roundtrip_minimal_forward() {
        let orig = base_forward();
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_roundtrip_eq(&orig, &decoded);
    }

    #[test]
    fn roundtrip_with_tp_meta_attn_only() {
        let mut orig = base_forward();
        orig.tp_meta = Some(TensorParallelMeta {
            tp_rank: 2,
            tp_size: 4,
            single_layer: 17,
            phase: TpPhase::AttnOnly,
        });
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_roundtrip_eq(&orig, &decoded);
    }

    #[test]
    fn roundtrip_with_tp_meta_ffn_only() {
        let mut orig = base_forward();
        orig.tp_meta = Some(TensorParallelMeta {
            tp_rank: 0,
            tp_size: 2,
            single_layer: 0,
            phase: TpPhase::FfnOnly,
        });
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_roundtrip_eq(&orig, &decoded);
    }

    #[test]
    fn roundtrip_all_tp_phases() {
        for phase in [
            TpPhase::Full,
            TpPhase::AttnOnly,
            TpPhase::FfnOnly,
            TpPhase::EmbedOnly,
        ] {
            let mut orig = base_forward();
            orig.tp_meta = Some(TensorParallelMeta {
                tp_rank: 1,
                tp_size: 4,
                single_layer: 5,
                phase: phase.clone(),
            });
            let bytes = encode_layer_forward(&orig).unwrap();
            let decoded = decode_layer_forward(&bytes).unwrap();
            assert_eq!(decoded.tp_meta.as_ref().unwrap().phase, phase);
        }
    }

    #[test]
    fn roundtrip_with_tp_meta_and_pre_embedded() {
        let mut orig = base_forward();
        orig.pre_embedded = true;
        orig.tp_meta = Some(TensorParallelMeta {
            tp_rank: 3,
            tp_size: 4,
            single_layer: 25,
            phase: TpPhase::EmbedOnly,
        });
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        // pre_embedded is encoded inside the tp_meta trailer (one byte
        // after phase). Without tp_meta, pre_embedded is not on the wire
        // and decode returns false — that's a known limitation.
        assert!(decoded.pre_embedded);
        assert_eq!(decoded.tp_meta.unwrap().phase, TpPhase::EmbedOnly);
    }

    #[test]
    fn roundtrip_with_speculative_drafts() {
        let mut orig = base_forward();
        orig.draft_tokens = vec![100, 101, 102, 103];
        orig.spec_logits_requested = true;
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_eq!(decoded.draft_tokens, orig.draft_tokens);
        assert!(decoded.spec_logits_requested);
    }

    #[test]
    fn roundtrip_spec_logits_requested_with_empty_drafts() {
        // Regression test: build_spec_verify_forward intentionally sets
        // draft_tokens=[] (IDs ride in `activations`). The encoder MUST
        // still emit the 0x03 trailer so spec_logits_requested=true
        // survives the round-trip; otherwise the worker computes
        // want_spec_output=false and never returns spec_logits.
        let mut orig = base_forward();
        orig.draft_tokens = Vec::new();
        orig.spec_logits_requested = true;
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert!(decoded.draft_tokens.is_empty());
        assert!(
            decoded.spec_logits_requested,
            "spec_logits_requested must survive cleartext round-trip when draft_tokens is empty"
        );
    }

    #[test]
    fn roundtrip_with_truncate_kv_to() {
        let mut orig = base_forward();
        orig.truncate_kv_to = Some(42);
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_eq!(decoded.truncate_kv_to, Some(42));
    }

    #[test]
    fn roundtrip_with_all_optional_trailers() {
        // tp_meta + speculative drafts + kv truncation all set. The
        // decoder must scan trailers in marker order and not get
        // confused by adjacency.
        let mut orig = base_forward();
        orig.tp_meta = Some(TensorParallelMeta {
            tp_rank: 1,
            tp_size: 2,
            single_layer: 12,
            phase: TpPhase::AttnOnly,
        });
        orig.pre_embedded = true;
        orig.draft_tokens = vec![1, 2, 3, 4, 5];
        orig.spec_logits_requested = true;
        orig.truncate_kv_to = Some(99);

        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_roundtrip_eq(&orig, &decoded);
    }

    #[test]
    fn decoder_rejects_truncated_envelope() {
        let orig = base_forward();
        let bytes = encode_layer_forward(&orig).unwrap();
        // Truncate to mid-header — decode should error, not panic.
        let result = decode_layer_forward(&bytes[..15]);
        assert!(result.is_err());
    }

    #[test]
    fn decoder_rejects_truncated_activations() {
        let orig = base_forward();
        let mut bytes = encode_layer_forward(&orig).unwrap();
        // Lop off the trailer — decoder should fail on missing
        // layer_range/model_id trailer.
        bytes.truncate(1 + 29 + orig.activations.len() + 3);
        let result = decode_layer_forward(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decoder_skips_optional_tp_trailer_when_absent() {
        let orig = base_forward();
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert!(decoded.tp_meta.is_none());
        assert!(decoded.draft_tokens.is_empty());
        assert!(decoded.truncate_kv_to.is_none());
    }

    #[test]
    fn formats_roundtrip_correctly() {
        for fmt in [TensorFormat::FP16, TensorFormat::FP32, TensorFormat::INT8] {
            let mut orig = base_forward();
            orig.format = fmt.clone();
            let bytes = encode_layer_forward(&orig).unwrap();
            let decoded = decode_layer_forward(&bytes).unwrap();
            // Compare via match since TensorFormat doesn't impl PartialEq.
            let same = matches!(
                (&decoded.format, &fmt),
                (TensorFormat::FP16, TensorFormat::FP16)
                    | (TensorFormat::FP32, TensorFormat::FP32)
                    | (TensorFormat::INT8, TensorFormat::INT8)
            );
            assert!(same, "format roundtrip failed");
        }
    }

    // --- R139 Phase A-rev: chunk-meta trailer (0x05) -----------------------

    #[test]
    fn roundtrip_with_chunk_meta_first_chunk() {
        let mut orig = base_forward();
        orig.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 0,
            total_chunks: 4,
        });
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_roundtrip_eq(&orig, &decoded);
        assert!(!decoded.chunk_meta.unwrap().is_final());
    }

    #[test]
    fn roundtrip_with_chunk_meta_last_chunk_is_final() {
        let mut orig = base_forward();
        orig.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 3,
            total_chunks: 4,
        });
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_roundtrip_eq(&orig, &decoded);
        assert!(decoded.chunk_meta.unwrap().is_final());
    }

    #[test]
    fn chunk_meta_stacks_with_kv_truncate_trailer() {
        // Chunked spec-decode KV-truncate path: 0x04 (kv-truncate) THEN 0x05
        // (chunk-meta) must round-trip without trailer adjacency confusion.
        let mut orig = base_forward();
        orig.truncate_kv_to = Some(42);
        orig.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 2,
            total_chunks: 5,
        });
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert_eq!(decoded.truncate_kv_to, Some(42));
        assert_eq!(
            decoded.chunk_meta,
            Some(crate::types::ChunkMeta {
                chunk_idx: 2,
                total_chunks: 5,
            })
        );
    }

    #[test]
    fn decoder_rejects_invalid_chunk_meta_idx_at_or_above_total() {
        // Build a frame manually with chunk_idx == total_chunks (invalid).
        let orig = base_forward();
        let mut bytes = encode_layer_forward(&orig).unwrap();
        bytes.push(0x05);
        bytes.extend_from_slice(&4u32.to_le_bytes()); // chunk_idx = 4
        bytes.extend_from_slice(&4u32.to_le_bytes()); // total_chunks = 4
        let result = decode_layer_forward(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn decoder_rejects_chunk_meta_with_zero_total() {
        let orig = base_forward();
        let mut bytes = encode_layer_forward(&orig).unwrap();
        bytes.push(0x05);
        bytes.extend_from_slice(&0u32.to_le_bytes()); // chunk_idx = 0
        bytes.extend_from_slice(&0u32.to_le_bytes()); // total_chunks = 0
        let result = decode_layer_forward(&bytes);
        assert!(result.is_err());
    }

    #[test]
    fn legacy_frame_without_chunk_meta_decodes_as_none() {
        // Verify backward-compat: existing peers send frames without the 0x05
        // trailer; receiver sees `chunk_meta: None` and treats as single-chunk
        // implicit (the common-case decode wire-form pre-R139).
        let orig = base_forward();
        let bytes = encode_layer_forward(&orig).unwrap();
        let decoded = decode_layer_forward(&bytes).unwrap();
        assert!(decoded.chunk_meta.is_none());
    }

    /// The next-hop trailer survives a wire round trip, and its absence still
    /// decodes — which is what every node predating chaining sends.
    #[test]
    fn next_hop_survives_a_round_trip_and_is_optional() {
        let mut f = base_forward();
        assert!(f.chain.is_empty(), "the default is no hop");
        let plain = encode_layer_forward(&f).expect("encode");
        assert_eq!(
            decode_layer_forward(&plain).expect("decode").chain,
            Vec::new(),
            "a frame with no trailer decodes to no hop, as an older node sends"
        );

        f.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([7u8; 32]),
            layer_range: (12, 24),
        }];
        let wire = encode_layer_forward(&f).expect("encode");
        let back = decode_layer_forward(&wire).expect("decode");
        assert_eq!(back.chain, f.chain);
    }

    /// A hop must not coexist incorrectly with the trailer that precedes it.
    /// Chunk-meta used to be last and read without advancing the cursor, so a
    /// frame carrying both would have parsed the hop out of the wrong bytes.
    #[test]
    fn a_hop_decodes_correctly_after_a_chunk_meta_trailer() {
        let mut f = base_forward();
        f.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 1,
            total_chunks: 3,
        });
        f.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([9u8; 32]),
            layer_range: (4, 8),
        }];
        let wire = encode_layer_forward(&f).expect("encode");
        let back = decode_layer_forward(&wire).expect("decode");
        assert_eq!(back.chunk_meta, f.chunk_meta, "chunk meta still parses");
        assert_eq!(back.chain, f.chain, "and the hop after it does too");
    }

    /// An empty or inverted layer range is refused rather than forwarded into
    /// nothing. The receiver falls back to replying to the coordinator, which
    /// costs one round trip and always works.
    #[test]
    fn a_nonsensical_hop_range_is_refused() {
        for (a, b) in [(8u32, 8u32), (9, 4)] {
            let mut buf = vec![0x06, 1];
            buf.extend_from_slice(&[3u8; 32]);
            buf.extend_from_slice(&a.to_le_bytes());
            buf.extend_from_slice(&b.to_le_bytes());
            let mut cursor = 0usize;
            assert!(
                read_chain_trailer(&buf, &mut cursor).is_empty(),
                "range ({a},{b}) must be refused"
            );
            assert_eq!(cursor, 0, "a refused trailer must not consume bytes");
        }
    }

    /// A truncated or absurd chain length must not read past the buffer, and
    /// must not consume bytes it did not validate.
    #[test]
    fn a_malformed_chain_length_is_refused_without_reading_past_the_end() {
        // Claims three hops, carries one.
        let mut buf = vec![0x06, 3];
        buf.extend_from_slice(&[1u8; 32]);
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes());
        let mut cursor = 0usize;
        assert!(read_chain_trailer(&buf, &mut cursor).is_empty());
        assert_eq!(cursor, 0);

        // Claims zero hops, which is the same as sending no trailer.
        let mut cursor = 0usize;
        assert!(read_chain_trailer(&[0x06, 0], &mut cursor).is_empty());
        assert_eq!(cursor, 0);
    }

    /// A multi-hop chain round-trips in order. Order is the whole meaning of
    /// the field — each node takes the head and passes the tail on.
    #[test]
    fn a_multi_hop_chain_round_trips_in_order() {
        let mut f = base_forward();
        f.chain = (1u8..=4)
            .map(|i| crate::types::ChainHop {
                node_id: crate::types::NodeId([i; 32]),
                layer_range: (i as u32 * 4, i as u32 * 4 + 4),
            })
            .collect();
        let wire = encode_layer_forward(&f).expect("encode");
        let back = decode_layer_forward(&wire).expect("decode");
        assert_eq!(back.chain, f.chain, "hops must survive in order");
    }

    /// A chained forward names the coordinator the tail must answer, and the
    /// name survives the wire. Without it the serving side falls back to
    /// answering the forward's SENDER — right for one hop, wrong for a chain,
    /// where the sender is the previous hop (observed on two machines,
    /// 2026-08-21: the tail answered the head, which had already handed the
    /// work on, and the coordinator waited out its whole deadline).
    #[test]
    fn a_chained_forward_carries_the_coordinator_it_must_answer() {
        let mut f = base_forward();
        f.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([7u8; 32]),
            layer_range: (12, 28),
        }];
        f.requester_node_id = Some([0xC0; 32]);
        let wire = encode_layer_forward(&f).expect("encode");
        let back = decode_layer_forward(&wire).expect("decode");
        assert_eq!(back.chain, f.chain);
        assert_eq!(
            back.requester_node_id,
            Some([0xC0; 32]),
            "the reply-to trailer must come back as the requester"
        );
    }

    /// The LAST hop of a chain receives a forward with no remaining hops and
    /// still has to know whom to answer — so the reply-to rides on its own,
    /// chain or no chain. (The first cut tied it to the chain trailer and so
    /// dropped it on exactly that hop: the tail answered the head, again.)
    #[test]
    fn the_tail_of_a_chain_still_learns_the_coordinator() {
        let mut f = base_forward();
        f.chain = Vec::new();
        f.requester_node_id = Some([0xC0; 32]);
        let wire = encode_layer_forward(&f).expect("encode");
        let back = decode_layer_forward(&wire).expect("decode");
        assert!(back.chain.is_empty());
        assert_eq!(back.requester_node_id, Some([0xC0; 32]));
    }

    /// A forward that names no requester carries no 0x07 — the frame every
    /// released node sends and expects. Keeping the one-hop wire unchanged is
    /// therefore a SENDER discipline: the coordinator sets the field only on
    /// a chained send, every other constructor leaves it `None`.
    #[test]
    fn a_forward_that_names_no_requester_is_unchanged_on_the_wire() {
        let mut f = base_forward();
        f.requester_node_id = None;
        let wire = encode_layer_forward(&f).expect("encode");
        assert!(
            !wire
                .windows(33)
                .any(|w| w[0] == 0x07 && w.len() == 33 && wire.ends_with(w)),
            "no 0x07 trailer when nothing is named"
        );
        let back = decode_layer_forward(&wire).expect("decode");
        assert_eq!(back.requester_node_id, None);
    }

    /// "Who gets the answer" is a routing decision made by somebody else, so
    /// it is authenticated like "who is next": a different reply-to is a
    /// different AAD, and a tampered one fails Poly1305 rather than quietly
    /// sending the tail's result to an attacker's node.
    #[test]
    fn redirecting_the_reply_changes_the_aad() {
        let mut f = base_forward();
        f.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([7u8; 32]),
            layer_range: (12, 28),
        }];
        f.requester_node_id = Some([0xC0; 32]);
        let honest = super::super::encrypted::build_layer_forward_aad(&f);
        f.requester_node_id = Some([0xBA; 32]);
        let redirected = super::super::encrypted::build_layer_forward_aad(&f);
        assert_ne!(
            honest, redirected,
            "the reply-to must be bound into the AAD"
        );
        // And the AAD is the same bytes the wire carries for these trailers:
        // one writer, three users.
        let mut wire_tail = Vec::new();
        append_chain_trailers(&mut wire_tail, &f);
        assert!(
            redirected.ends_with(&wire_tail),
            "AAD must end with exactly the chain+reply-to bytes the frame carries"
        );
    }
}
