use crate::error::SwarmError;
use crate::types::{LayerForward, ModelId, TensorFormat};

use super::TENSOR_TAG_ENCRYPTED;

/// Build the AAD bytes for sealing/opening a `LayerForward` activation payload.
///
/// Layout (header — always present):
/// `request_id(16) | sequence_num(4 LE) | index_pos(4 LE) | fmt(1)
/// | layer_start(4 LE) | layer_end(4 LE) | model_id_len(2 LE) | model_id`.
///
/// Layout (spec trailer — present iff `!draft_tokens.is_empty() ||
/// spec_logits_requested`):
/// `0x03 marker(1) | spec_flags(1) | num_drafts(2 LE) | drafts(num_drafts × 4 LE)`.
///
/// Layout (kv-truncate trailer — present iff `truncate_kv_to.is_some()`):
/// `0x04 marker(1) | target_len(4 LE)`.
///
/// Trailers are included in the AAD whenever they're emitted on the wire so
/// an active MITM cannot flip `spec_logits_requested` or modify
/// `truncate_kv_to` without invalidating Poly1305. The wire trailers stay
/// cleartext (the worker reads them after decrypt to dispatch correctly),
/// but their values are now authenticated.
///
/// Both the encrypt path (`network/manager/tensors.rs`,
/// `network/pipeline_stream.rs::encode_forward_for_wire`) and the decrypt
/// path (`decode_layer_forward_encrypted`) MUST produce identical bytes;
/// any drift breaks every encrypted forward. Centralising here pins the
/// contract.
///
/// **Wire compatibility:** extending the AAD layout is a protocol bump for
/// encrypted mode. Old↔new mixed clusters running `enable_encryption=true`
/// will fail decrypt with auth-error. Plaintext mode (`enable_encryption=false`)
/// is unaffected. Encrypted mode is opt-in and alpha; this trade-off is
/// documented in `docs/ARCHITECTURE.md`.
pub fn build_layer_forward_aad(forward: &LayerForward) -> Vec<u8> {
    let model_id_bytes = forward.model_id.0.as_bytes();
    let mut aad = Vec::with_capacity(35 + model_id_bytes.len());
    aad.extend_from_slice(forward.request_id.as_bytes());
    aad.extend_from_slice(&forward.sequence_num.to_le_bytes());
    aad.extend_from_slice(&forward.index_pos.to_le_bytes());
    let fmt_tag: u8 = match forward.format {
        TensorFormat::FP16 => 0,
        TensorFormat::FP32 => 1,
        TensorFormat::INT8 => 2,
    };
    aad.push(fmt_tag);
    let (layer_start, layer_end) = forward.layer_range;
    aad.extend_from_slice(&layer_start.to_le_bytes());
    aad.extend_from_slice(&layer_end.to_le_bytes());
    aad.extend_from_slice(&(model_id_bytes.len() as u16).to_le_bytes());
    aad.extend_from_slice(model_id_bytes);

    // Spec trailer fields (mirror `encode_layer_forward[_encrypted]`'s 0x03
    // emission gate exactly — see `protocol/layer_forward.rs`).
    if !forward.draft_tokens.is_empty() || forward.spec_logits_requested {
        aad.push(0x03);
        let flags: u8 = if forward.spec_logits_requested { 1 } else { 0 };
        aad.push(flags);
        // draft_tokens length is u16-bounded by the encoder. The encode
        // helpers reject overlong drafts before AAD is built; we trust
        // that contract here and saturate as a defence-in-depth.
        let n = forward.draft_tokens.len().min(u16::MAX as usize) as u16;
        aad.extend_from_slice(&n.to_le_bytes());
        for t in &forward.draft_tokens {
            aad.extend_from_slice(&t.to_le_bytes());
        }
    }

    // KV-truncate trailer (mirror 0x04 emission gate).
    if let Some(target_len) = forward.truncate_kv_to {
        aad.push(0x04);
        aad.extend_from_slice(&target_len.to_le_bytes());
    }

    // Chunk-meta trailer (mirror 0x05 emission gate). Binds chunk_idx +
    // total_chunks into the AAD so a peer cannot reorder chunks, forge a
    // wrong total, or substitute a chunk from a different transfer without
    // Poly1305 rejecting the open. Mid-chunk frames have `chunk_meta.is_final
    // == false`; the receiver assembles all `total_chunks` frames before
    // dispatching the reassembled activation to the worker.
    if let Some(cm) = forward.chunk_meta {
        aad.push(0x05);
        aad.extend_from_slice(&cm.chunk_idx.to_le_bytes());
        aad.extend_from_slice(&cm.total_chunks.to_le_bytes());
    }

    // Next-hop trailer (mirror 0x06 emission gate). Binds WHO the receiver is
    // being told to forward its output to. Without this in the AAD an active
    // MITM could redirect a segment's activations to a node of its choosing —
    // the activations are sealed, but the routing decision would not be
    // authenticated, and a chained pipeline is exactly a routing decision made
    // by somebody else.
    if !forward.chain.is_empty() {
        let n = forward.chain.len().min(u8::MAX as usize);
        aad.push(0x06);
        aad.push(n as u8);
        for hop in forward.chain.iter().take(n) {
            aad.extend_from_slice(&hop.node_id.0);
            aad.extend_from_slice(&hop.layer_range.0.to_le_bytes());
            aad.extend_from_slice(&hop.layer_range.1.to_le_bytes());
        }
    }

    aad
}

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

    // Optional: KV truncation trailer (marker 0x04 + target_len(4 LE))
    if let Some(target_len) = forward.truncate_kv_to {
        buf.push(0x04);
        buf.extend_from_slice(&target_len.to_le_bytes());
    }

    // Optional: chunk-meta trailer (marker 0x05 + chunk_idx(4 LE) +
    // total_chunks(4 LE)). Mirrors the plaintext encoder.
    if let Some(cm) = forward.chunk_meta {
        buf.push(0x05);
        buf.extend_from_slice(&cm.chunk_idx.to_le_bytes());
        buf.extend_from_slice(&cm.total_chunks.to_le_bytes());
    }

    // Optional: next-hop trailer (marker 0x06 + node_id(32) + layer_start(4 LE)
    // + layer_end(4 LE)). Mirrors the plaintext encoder, and must be emitted
    // here too or the receiver reconstructs a different AAD and every chained
    // encrypted forward fails to open.
    if !forward.chain.is_empty() {
        let n = forward.chain.len().min(u8::MAX as usize);
        buf.push(0x06);
        buf.push(n as u8);
        for hop in forward.chain.iter().take(n) {
            buf.extend_from_slice(&hop.node_id.0);
            buf.extend_from_slice(&hop.layer_range.0.to_le_bytes());
            buf.extend_from_slice(&hop.layer_range.1.to_le_bytes());
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

    // The AAD covers the cleartext header AND the post-payload trailers
    // (spec / kv-truncate). We can't slice the bytes here because the
    // trailers come AFTER the sealed payload — we parse them below first,
    // then reconstruct the AAD via `build_layer_forward_aad` on the parsed
    // forward struct so encrypt and decrypt agree byte-for-byte. See the
    // helper's docstring for the layout contract.
    let sealed_len_start = mid_start + mid_len;
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
        // R107/R108: shared cap via `super::MAX_DRAFT_TOKENS` so plaintext
        // and encrypted decoders enforce the same bound (see rationale in
        // `network/protocol/mod.rs`).
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

    // Optional: chunk-meta trailer (marker 0x05 + chunk_idx(4 LE) +
    // total_chunks(4 LE)). Mirrored from the plaintext decoder.
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

    // Next-hop trailer (0x06). Read through the same helper the plaintext
    // decoder uses, so the two wire readers cannot drift apart.
    let chain = super::layer_forward::read_chain_trailer(data, &mut cursor);
    let _ = cursor;

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
        chain,
        sender_peer_bytes: None,
        requester_node_id: None,
        pre_embedded: tp_pre_embedded,
        generated_ids: Vec::new(),
        adapter_id: None,
        draft_tokens,
        spec_logits_requested,
        truncate_kv_to,
        chunk_meta,
    };

    // Reconstruct AAD from the parsed forward via the helper. This MUST
    // match the bytes the encrypt path passed to `session_manager.seal`.
    // The trailer fields (`spec_logits_requested`, `draft_tokens`,
    // `truncate_kv_to`) are now authenticated — flipping them on the wire
    // invalidates Poly1305 even though they ride as cleartext metadata.
    let aad = build_layer_forward_aad(&forward);

    Ok((forward, sealed, aad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LayerForward, ModelId, TensorFormat, TensorParallelMeta, TpPhase};

    fn base_forward() -> LayerForward {
        LayerForward {
            request_id: uuid::Uuid::from_u128(0xDEAD_BEEF_CAFE_F00D_1122_3344_5566_7788),
            sequence_num: 5,
            index_pos: 11,
            activations: vec![],
            format: TensorFormat::FP32,
            model_id: ModelId("qwen2.5-7b".into()),
            layer_range: (4, 12),
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
        }
    }

    #[test]
    fn encrypted_envelope_preserves_spec_trailers() {
        // Locks down the Task-1 claim: encode_layer_forward_encrypted /
        // decode_layer_forward_encrypted preserve all spec trailers
        // (draft_tokens marker 0x03, truncate_kv_to marker 0x04). Without
        // this, lifting the speculative_common_eligible enable_encryption
        // gate would silently corrupt verify rounds when encryption is on.
        let mut orig = base_forward();
        orig.draft_tokens = vec![100, 200, 300, 400, 500];
        orig.spec_logits_requested = true;
        orig.truncate_kv_to = Some(42);

        let sealed = vec![0u8; 256];
        let bytes = encode_layer_forward_encrypted(&orig, sealed.clone()).unwrap();
        let (decoded, sealed_out, _aad) = decode_layer_forward_encrypted(&bytes).unwrap();

        assert_eq!(decoded.request_id, orig.request_id);
        assert_eq!(decoded.sequence_num, orig.sequence_num);
        assert_eq!(decoded.index_pos, orig.index_pos);
        assert_eq!(decoded.layer_range, orig.layer_range);
        assert_eq!(decoded.model_id, orig.model_id);
        assert_eq!(decoded.draft_tokens, orig.draft_tokens);
        assert!(decoded.spec_logits_requested);
        assert_eq!(decoded.truncate_kv_to, Some(42));
        assert_eq!(sealed_out, sealed);
    }

    #[test]
    fn encrypted_envelope_preserves_tp_meta_alongside_spec() {
        // Belt-and-braces: tp_meta + spec trailers + kv_truncate all set.
        // The decoder scans trailers in marker order; adjacency must not
        // confuse the parser.
        let mut orig = base_forward();
        orig.tp_meta = Some(TensorParallelMeta {
            tp_rank: 2,
            tp_size: 4,
            single_layer: 7,
            phase: TpPhase::AttnOnly,
        });
        orig.pre_embedded = true;
        orig.draft_tokens = vec![1, 2, 3];
        orig.spec_logits_requested = true;
        orig.truncate_kv_to = Some(99);

        let sealed = vec![0xABu8; 64];
        let bytes = encode_layer_forward_encrypted(&orig, sealed).unwrap();
        let (decoded, _sealed, _aad) = decode_layer_forward_encrypted(&bytes).unwrap();

        let tp = decoded.tp_meta.expect("tp_meta should round-trip");
        assert_eq!(tp.tp_rank, 2);
        assert_eq!(tp.tp_size, 4);
        assert_eq!(tp.single_layer, 7);
        assert!(matches!(tp.phase, TpPhase::AttnOnly));
        assert!(decoded.pre_embedded);
        assert_eq!(decoded.draft_tokens, vec![1, 2, 3]);
        assert!(decoded.spec_logits_requested);
        assert_eq!(decoded.truncate_kv_to, Some(99));
    }

    #[test]
    fn aad_helper_matches_inline_layout() {
        // build_layer_forward_aad is the documented single source of truth
        // (see .claude/rules/architecture.md § Centralised Wire-Format
        // Helpers). The encrypt path in network/manager/tensors.rs and the
        // decode path in this file MUST produce byte-identical AAD; pin
        // the layout here so a refactor that subtly drifts the order
        // breaks at unit-test time, not at the wire.
        let mut forward = base_forward();
        forward.sequence_num = 0x1122_3344;
        forward.index_pos = 0x5566_7788;
        forward.format = TensorFormat::INT8;
        forward.layer_range = (0xAABB_CCDD, 0xEEFF_0011);

        let aad = build_layer_forward_aad(&forward);
        // Layout: uuid(16) + seq(4 LE) + idx(4 LE) + fmt(1) + ls(4 LE) + le(4 LE) + mid_len(2 LE) + mid_bytes
        let mid_bytes = forward.model_id.0.as_bytes();
        assert_eq!(aad.len(), 35 + mid_bytes.len());
        assert_eq!(&aad[0..16], forward.request_id.as_bytes());
        assert_eq!(&aad[16..20], &0x1122_3344u32.to_le_bytes());
        assert_eq!(&aad[20..24], &0x5566_7788u32.to_le_bytes());
        assert_eq!(aad[24], 2); // INT8 fmt tag
        assert_eq!(&aad[25..29], &0xAABB_CCDDu32.to_le_bytes());
        assert_eq!(&aad[29..33], &0xEEFF_0011u32.to_le_bytes());
        assert_eq!(&aad[33..35], &(mid_bytes.len() as u16).to_le_bytes());
        assert_eq!(&aad[35..], mid_bytes);
    }

    #[test]
    fn encrypted_envelope_preserves_spec_logits_requested_with_empty_drafts() {
        // Regression test: build_spec_verify_forward intentionally sets
        // draft_tokens=[] (the IDs ride in `activations` for the DSD verify
        // path). The encoder MUST still emit the 0x03 trailer so
        // spec_logits_requested=true survives the round-trip; otherwise
        // the receiver computes want_spec_output=false and the last
        // segment never returns spec_logits, breaking DSD entirely.
        let mut orig = base_forward();
        orig.draft_tokens = Vec::new();
        orig.spec_logits_requested = true;

        let bytes = encode_layer_forward_encrypted(&orig, vec![0u8; 32]).unwrap();
        let (decoded, _sealed, _aad) = decode_layer_forward_encrypted(&bytes).unwrap();
        assert!(decoded.draft_tokens.is_empty());
        assert!(
            decoded.spec_logits_requested,
            "spec_logits_requested must survive encrypted round-trip when draft_tokens is empty"
        );
    }

    #[test]
    fn aad_includes_spec_trailer_when_emitted() {
        // Spec trailer fields MUST be in the AAD whenever they're emitted on
        // the wire. The presence/length of the spec trailer in AAD is gated
        // by the same condition the encoder uses for the 0x03 wire trailer:
        // `!draft_tokens.is_empty() || spec_logits_requested`.
        let mut forward = base_forward();
        forward.draft_tokens = vec![100, 200, 300];
        forward.spec_logits_requested = true;

        let aad = build_layer_forward_aad(&forward);
        let mid_bytes = forward.model_id.0.as_bytes();
        let header_end = 35 + mid_bytes.len();
        // Header byte length: 35 + mid_bytes.len()
        // Spec trailer: 1 (marker) + 1 (flags) + 2 (num_drafts) + 3*4 (drafts) = 16
        assert_eq!(aad.len(), header_end + 16);
        assert_eq!(aad[header_end], 0x03);
        assert_eq!(aad[header_end + 1], 1); // spec_logits_requested flag
        assert_eq!(&aad[header_end + 2..header_end + 4], &3u16.to_le_bytes(),);
        assert_eq!(&aad[header_end + 4..header_end + 8], &100u32.to_le_bytes(),);
    }

    #[test]
    fn aad_includes_kv_truncate_when_set() {
        let mut forward = base_forward();
        forward.truncate_kv_to = Some(0xCAFE_BABE);

        let aad = build_layer_forward_aad(&forward);
        let mid_bytes = forward.model_id.0.as_bytes();
        let header_end = 35 + mid_bytes.len();
        // KV-truncate trailer: 1 (marker) + 4 (target_len) = 5
        assert_eq!(aad.len(), header_end + 5);
        assert_eq!(aad[header_end], 0x04);
        assert_eq!(
            &aad[header_end + 1..header_end + 5],
            &0xCAFE_BABEu32.to_le_bytes(),
        );
    }

    #[test]
    fn aad_omits_trailers_when_absent() {
        // When neither spec nor kv_truncate trailers are emitted, the AAD
        // is just the header bytes (preserves backward-compat for the
        // common no-trailer case).
        let forward = base_forward();
        let aad = build_layer_forward_aad(&forward);
        let mid_bytes = forward.model_id.0.as_bytes();
        assert_eq!(aad.len(), 35 + mid_bytes.len());
    }

    #[test]
    fn aad_authenticates_spec_logits_requested_flip() {
        // Lock down the security claim: an attacker who flips
        // `spec_logits_requested` on the wire MUST produce a different AAD
        // than the one the sender used to seal. The encoder's spec trailer
        // is cleartext (so the worker can read it without decrypting), but
        // the value is now in the AAD, so MITM tampering invalidates the
        // Poly1305 tag.
        //
        // This test compares the helper's output for two LayerForwards
        // that differ only in `spec_logits_requested` and asserts the AAD
        // bytes diverge. Decrypt-side enforcement is exercised by
        // `decode_layer_forward_encrypted` calling `build_layer_forward_aad`
        // on the parsed forward — see the matching round-trip test.
        let mut a = base_forward();
        a.spec_logits_requested = true;
        let aad_a = build_layer_forward_aad(&a);

        let mut b = a.clone();
        b.spec_logits_requested = false;
        let aad_b = build_layer_forward_aad(&b);

        assert_ne!(
            aad_a, aad_b,
            "flipping spec_logits_requested MUST change AAD bytes"
        );
    }

    #[test]
    fn aad_authenticates_truncate_kv_to_change() {
        let mut a = base_forward();
        a.truncate_kv_to = Some(42);
        let aad_a = build_layer_forward_aad(&a);

        let mut b = a.clone();
        b.truncate_kv_to = Some(43);
        let aad_b = build_layer_forward_aad(&b);

        assert_ne!(
            aad_a, aad_b,
            "modifying truncate_kv_to MUST change AAD bytes"
        );
    }

    #[test]
    fn encrypted_envelope_rejects_truncated_input() {
        let orig = base_forward();
        let bytes = encode_layer_forward_encrypted(&orig, vec![0u8; 32]).unwrap();
        // Lop off most of the payload — decoder must error, not panic.
        assert!(decode_layer_forward_encrypted(&bytes[..10]).is_err());
    }

    // --- R139 Phase A-rev: chunk-meta in encrypted envelope ---------------

    #[test]
    fn encrypted_envelope_preserves_chunk_meta() {
        let mut orig = base_forward();
        orig.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 2,
            total_chunks: 5,
        });
        let sealed = vec![0u8; 256];
        let bytes = encode_layer_forward_encrypted(&orig, sealed.clone()).unwrap();
        let (decoded, sealed_out, _aad) = decode_layer_forward_encrypted(&bytes).unwrap();
        assert_eq!(decoded.chunk_meta, orig.chunk_meta);
        assert_eq!(sealed_out, sealed);
    }

    #[test]
    fn encrypted_envelope_preserves_next_hop() {
        let mut orig = base_forward();
        orig.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([5u8; 32]),
            layer_range: (16, 32),
        }];
        let sealed = vec![0u8; 256];
        let bytes = encode_layer_forward_encrypted(&orig, sealed.clone()).unwrap();
        let (decoded, sealed_out, _aad) = decode_layer_forward_encrypted(&bytes).unwrap();
        assert_eq!(decoded.chain, orig.chain);
        assert_eq!(sealed_out, sealed);
    }

    /// A chained pipeline is a routing decision made by somebody else, so WHO a
    /// segment is told to forward to has to be authenticated. Redirecting a
    /// hop must change the AAD, so Poly1305 rejects the open rather than the
    /// activations being delivered to an attacker's node of choice.
    #[test]
    fn aad_authenticates_the_next_hop() {
        let mut a = base_forward();
        a.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([1u8; 32]),
            layer_range: (8, 16),
        }];
        let aad_a = build_layer_forward_aad(&a);

        // Redirected to a different node.
        let mut b = a.clone();
        b.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([2u8; 32]),
            layer_range: (8, 16),
        }];
        assert_ne!(
            aad_a,
            build_layer_forward_aad(&b),
            "redirecting a hop MUST change the AAD"
        );

        // Same node, different layers — also a routing change.
        let mut c = a.clone();
        c.chain = vec![crate::types::ChainHop {
            node_id: crate::types::NodeId([1u8; 32]),
            layer_range: (8, 24),
        }];
        assert_ne!(
            aad_a,
            build_layer_forward_aad(&c),
            "changing the hop's layer range MUST change the AAD"
        );

        // And stripping the hop entirely, which would silently turn a chained
        // forward back into one that replies to the coordinator.
        let mut d = a.clone();
        d.chain = Vec::new();
        assert_ne!(
            aad_a,
            build_layer_forward_aad(&d),
            "removing a hop MUST change the AAD"
        );
    }

    #[test]
    fn aad_authenticates_chunk_idx_flip() {
        // Security claim: an attacker who reorders chunks on the wire (swaps
        // chunk_idx between two captured frames) MUST produce a different AAD
        // than the sender used to seal — Poly1305 rejects the open. This locks
        // the chunk-meta-in-AAD binding from the encoder side.
        let mut a = base_forward();
        a.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 0,
            total_chunks: 4,
        });
        let aad_a = build_layer_forward_aad(&a);

        let mut b = a.clone();
        b.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 1,
            total_chunks: 4,
        });
        let aad_b = build_layer_forward_aad(&b);

        assert_ne!(
            aad_a, aad_b,
            "swapping chunk_idx MUST change AAD bytes (prevents reorder attack)"
        );
    }

    #[test]
    fn aad_authenticates_total_chunks_change() {
        let mut a = base_forward();
        a.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 0,
            total_chunks: 4,
        });
        let aad_a = build_layer_forward_aad(&a);

        let mut b = a.clone();
        b.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 0,
            total_chunks: 5,
        });
        let aad_b = build_layer_forward_aad(&b);

        assert_ne!(
            aad_a, aad_b,
            "modifying total_chunks MUST change AAD bytes (prevents truncation attack)"
        );
    }

    #[test]
    fn aad_omits_chunk_trailer_when_none() {
        // Legacy single-frame forwards have no 0x05 trailer in the AAD —
        // preserves backward compat with peers that don't emit the trailer.
        let forward = base_forward();
        let aad = build_layer_forward_aad(&forward);
        let mid_bytes = forward.model_id.0.as_bytes();
        assert_eq!(aad.len(), 35 + mid_bytes.len());
    }

    #[test]
    fn aad_includes_chunk_trailer_when_set() {
        let mut forward = base_forward();
        forward.chunk_meta = Some(crate::types::ChunkMeta {
            chunk_idx: 7,
            total_chunks: 10,
        });
        let aad = build_layer_forward_aad(&forward);
        let mid_bytes = forward.model_id.0.as_bytes();
        // Chunk-meta trailer: 1 (marker 0x05) + 4 (chunk_idx) + 4 (total_chunks) = 9
        assert_eq!(aad.len(), 35 + mid_bytes.len() + 9);
        let trailer_start = 35 + mid_bytes.len();
        assert_eq!(aad[trailer_start], 0x05);
        assert_eq!(
            &aad[trailer_start + 1..trailer_start + 5],
            &7u32.to_le_bytes()
        );
        assert_eq!(
            &aad[trailer_start + 5..trailer_start + 9],
            &10u32.to_le_bytes()
        );
    }
}

// Serde impls for SwarmRequest/SwarmResponse
// Note: TensorPayload variants are never JSON-serialized (handled by binary codec path),
