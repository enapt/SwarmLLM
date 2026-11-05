//! Layer-aligned shard layout computation.

use std::collections::HashMap;

use super::split::GgufTensorMeta;

// ── Layer-Aligned Sharding ──

/// Describes one layer-aligned shard: which layers it contains and their tensors.
#[derive(Clone, Debug)]
pub struct LayerShardLayout {
    pub index: u32,
    pub layer_start: u32,
    /// Exclusive upper bound of layer range.
    pub layer_end: u32,
    /// Tensor entries: (name, absolute_gguf_offset, size), sorted by offset.
    pub tensors: Vec<(String, u64, u64)>,
    /// Total size of this shard in bytes (sum of tensor sizes).
    pub size_bytes: u64,
}

impl LayerShardLayout {
    /// This part's tensor table: its tensors in layout order, back to back
    /// from byte 0 of the part file. **The one writer of a table** (#334 (2)):
    /// the manifest a publisher builds (`model::manifest::build_shard_infos_from_layouts`)
    /// and the table a holder derives from its header
    /// (`daemon::shard_loader::derive_tensor_entries`) must place every tensor
    /// alike, or the derived table reads the wrong weights and answers
    /// confidently. The last entry ends at `size_bytes`.
    pub fn tensor_entries(&self) -> Vec<crate::types::ShardTensorEntry> {
        let mut shard_offset = 0u64;
        self.tensors
            .iter()
            .map(|(name, gguf_offset, size)| {
                let entry = crate::types::ShardTensorEntry {
                    name: name.clone(),
                    gguf_offset: *gguf_offset,
                    shard_offset,
                    size: *size,
                };
                shard_offset += size;
                entry
            })
            .collect()
    }
}

/// Group layers into `shard_count` shards of roughly equal byte size.
///
/// Non-layer tensors: `token_embd*` → shard 0, `output*`/`output_norm*` → last shard.
/// Each shard contains ONLY complete transformer layers — no layer spans two shards.
pub fn compute_layer_shard_layouts(
    meta: &GgufTensorMeta,
    shard_count: u32,
) -> Vec<LayerShardLayout> {
    if shard_count == 0 {
        return vec![];
    }

    // Classify tensors: per-layer vs prefix (token_embd) vs suffix (output)
    let mut layer_sizes: Vec<(u32, u64)> = Vec::new(); // (layer_idx, total_bytes)
    let mut layer_tensors: HashMap<u32, Vec<(String, u64, u64)>> = HashMap::new();
    let mut prefix_tensors: Vec<(String, u64, u64)> = Vec::new();
    let mut prefix_size: u64 = 0;
    let mut suffix_tensors: Vec<(String, u64, u64)> = Vec::new();
    let mut suffix_size: u64 = 0;

    // Per-layer byte totals
    let mut per_layer_bytes: HashMap<u32, u64> = HashMap::new();

    for (name, loc) in &meta.tensors {
        let abs_offset = meta.tensor_data_offset + loc.offset;
        if name.starts_with("blk.") {
            // Parse layer index: "blk.{N}.suffix"
            if let Some(idx_str) = name.strip_prefix("blk.").and_then(|s| s.split('.').next()) {
                if let Ok(layer_idx) = idx_str.parse::<u32>() {
                    *per_layer_bytes.entry(layer_idx).or_insert(0) += loc.size;
                    layer_tensors.entry(layer_idx).or_default().push((
                        name.clone(),
                        abs_offset,
                        loc.size,
                    ));
                }
            }
        } else if name.starts_with("token_embd") {
            prefix_tensors.push((name.clone(), abs_offset, loc.size));
            prefix_size += loc.size;
        } else if name.starts_with("output") {
            suffix_tensors.push((name.clone(), abs_offset, loc.size));
            suffix_size += loc.size;
        } else {
            // Other tensors (rope_freqs, etc.) go to prefix
            prefix_tensors.push((name.clone(), abs_offset, loc.size));
            prefix_size += loc.size;
        }
    }

    // Sorted layer indices
    let mut layer_indices: Vec<u32> = per_layer_bytes.keys().copied().collect();
    layer_indices.sort();

    // Build (layer_idx, bytes) sorted by layer index
    for &idx in &layer_indices {
        layer_sizes.push((idx, *per_layer_bytes.get(&idx).unwrap_or(&0)));
    }

    let total_layer_bytes: u64 = layer_sizes.iter().map(|(_, s)| s).sum();
    let total_bytes = total_layer_bytes + prefix_size + suffix_size;

    // Single shard: everything in one
    if shard_count == 1 {
        let mut all_tensors = prefix_tensors;
        for &idx in &layer_indices {
            if let Some(t) = layer_tensors.get(&idx) {
                all_tensors.extend(t.iter().cloned());
            }
        }
        all_tensors.extend(suffix_tensors);
        all_tensors.sort_by_key(|(_, off, _)| *off);

        let layer_start = layer_indices.first().copied().unwrap_or(0);
        let layer_end = layer_indices.last().map(|&l| l + 1).unwrap_or(0);

        return vec![LayerShardLayout {
            index: 0,
            layer_start,
            layer_end,
            tensors: all_tensors,
            size_bytes: total_bytes,
        }];
    }

    // Greedily assign layers to shards using a dynamic target that adjusts
    // as shards are emitted. This ensures the algorithm produces the exact
    // requested number of shards instead of underproducing when large prefixes
    // or uneven layer sizes cause the static target to be exceeded early.
    let mut layouts: Vec<LayerShardLayout> = Vec::new();
    let mut current_tensors: Vec<(String, u64, u64)> = Vec::new();
    let mut current_size: u64 = 0;
    let mut current_layer_start: Option<u32> = None;
    let mut current_layer_end: u32 = 0;
    let mut emitted_bytes: u64 = 0;

    // Add prefix tensors to current (will be shard 0)
    current_tensors.extend(prefix_tensors.iter().cloned());
    current_size += prefix_size;

    for (i, &(layer_idx, layer_bytes)) in layer_sizes.iter().enumerate() {
        if current_layer_start.is_none() {
            current_layer_start = Some(layer_idx);
        }
        current_layer_end = layer_idx + 1;

        if let Some(t) = layer_tensors.get(&layer_idx) {
            current_tensors.extend(t.iter().cloned());
        }
        current_size += layer_bytes;

        // Check if this is the last layer going to the last shard
        let is_last_layer = i == layer_sizes.len() - 1;
        let remaining_shards = shard_count as usize - layouts.len() - 1;
        let remaining_layers = layer_sizes.len() - i - 1;

        // Dynamic target: distribute remaining bytes evenly across remaining shard slots.
        // This naturally adjusts when earlier shards are larger (e.g., due to prefix),
        // ensuring later shards are smaller to hit the total shard count.
        let remaining_budget = total_bytes.saturating_sub(emitted_bytes);
        let remaining_slots = (shard_count as usize - layouts.len()).max(1) as u64;
        let dynamic_target = remaining_budget / remaining_slots;

        let should_emit = if is_last_layer || remaining_shards == 0 {
            // Last layer → handled after loop (final shard with suffix).
            // No remaining shards → keep accumulating for final shard.
            false
        } else if remaining_shards > remaining_layers {
            // Must emit now: more shards needed than layers remaining
            true
        } else {
            current_size >= dynamic_target
        };

        if should_emit {
            current_tensors.sort_by_key(|(_, off, _)| *off);
            emitted_bytes += current_size;
            layouts.push(LayerShardLayout {
                index: layouts.len() as u32,
                layer_start: current_layer_start.unwrap_or(0),
                layer_end: current_layer_end,
                tensors: std::mem::take(&mut current_tensors),
                size_bytes: current_size,
            });
            current_size = 0;
            current_layer_start = None;
        }
    }

    // Final shard: add suffix tensors
    current_tensors.extend(suffix_tensors.iter().cloned());
    current_size += suffix_size;
    current_tensors.sort_by_key(|(_, off, _)| *off);

    layouts.push(LayerShardLayout {
        index: layouts.len() as u32,
        layer_start: current_layer_start.unwrap_or(current_layer_end),
        layer_end: current_layer_end,
        tensors: current_tensors,
        size_bytes: current_size,
    });

    layouts
}

/// Does a part whose layers are `part` (`[start, end)`) serve the run of
/// layers `[start, end)` of a model that runs `num_layers`? Where the part
/// holds one of the run's layers — or where the run ENDS the model and the
/// part lies wholly past its last layer.
///
/// A file's next-token-prediction blocks (`nextn_predict_layers`: Ornith-1.5's
/// `blk.32`; the loader runs 32 of its 33, `gguf_meta`) are cut as layers into
/// the LAST part, beside the output head — the end of the model, where
/// llama.cpp places them too (`dev_output`, after `n_layer_all`). Cut so, that
/// part's range is `[num_layers, num_layers + k)` and overlaps no run, so every
/// overlap test skipped the output head: a peer holding `[0, 32)` was planned
/// whole and refused for want of it, and no node fetched that part
/// (2026-11-02). Every "which parts does this run read" over a manifest asks
/// this; the layout stays as it is, since every holder's parts are cut by it.
pub fn part_serves_layers(part: (u32, u32), start: u32, end: u32, num_layers: u32) -> bool {
    (part.0 < end && start < part.1) || (end >= num_layers && part.0 >= num_layers)
}

/// The part that holds the output head: the last, where
/// [`compute_layer_shard_layouts`] cuts every `output*` tensor. Holding it is
/// NOT holding the end of the model — see [`parts_that_finish`].
pub fn head_part(shard_count: u32) -> u32 {
    shard_count.saturating_sub(1)
}

/// **Every part a run that FINISHES the model reads**: the part holding its
/// last layer, every part past it ([`part_serves_layers`]: next-token-prediction
/// blocks) and the head's part — in index order, the vision encoder's never.
///
/// Usually that is the last part alone. Not when the head is cut with a
/// next-token-prediction block and no layer the loader runs: Qwen3.8-27B's part
/// 31 is `[64, 65)`, its layer 63 in part 30, and holding parts 0 and 31 read
/// as holding "both ends" — prompt privacy switched itself on and refused every
/// request, and nothing kept part 30 (2026-11-05, #408). Ornith-1.5 is cut the
/// same way. Every "does this node hold the end" asks this.
pub fn parts_that_finish(manifest: &crate::types::ModelManifest) -> Vec<u32> {
    let n = manifest.num_layers;
    let head = head_part(manifest.shard_count);
    let mut parts: Vec<u32> = manifest
        .shards
        .iter()
        .filter(|s| s.index != crate::types::MMPROJ_SHARD_INDEX)
        .filter(|s| part_serves_layers(s.layer_range, n.saturating_sub(1), n, n))
        .map(|s| s.index)
        .collect();
    // The head's part always, listed or not: a manifest that leaves it out
    // (malformed, a peer's) must read as "this node lacks the head", never as
    // a model whose last layer is enough (security review, 2026-11-05).
    parts.push(head);
    parts.sort_unstable();
    parts.dedup();
    parts
}

/// The parts that must be on THIS node for a reply to start and finish here —
/// part 0 (the token table is cut into it) and [`parts_that_finish`]: what
/// prompt privacy needs, keeps from pruning and fetches.
pub fn parts_at_the_ends(manifest: &crate::types::ModelManifest) -> Vec<u32> {
    let mut parts = parts_that_finish(manifest);
    parts.push(0);
    parts.sort_unstable();
    parts.dedup();
    parts
}

/// Return all contiguous layer ranges from manifest ShardInfo entries.
///
/// Reads layer_range directly from each shard — manifests have accurate
/// layer ranges computed from GGUF tensor metadata. The model's last layer
/// counts only beside every part past it ([`part_serves_layers`]): a run
/// ending there is the last, and needs the output head they hold.
pub fn available_layer_ranges_from_manifest(
    manifest: &crate::types::ModelManifest,
    local_shard_indices: &[u32],
) -> Vec<(usize, usize)> {
    // Collect layer ranges from shards we hold
    let mut layer_bits = vec![false; manifest.num_layers as usize];
    for shard in &manifest.shards {
        if local_shard_indices.contains(&shard.index) {
            let start = shard.layer_range.0 as usize;
            let end = (shard.layer_range.1 as usize).min(layer_bits.len());
            for bit in layer_bits.iter_mut().take(end).skip(start) {
                *bit = true;
            }
        }
    }
    let lacks_a_part_past_the_end = manifest.shards.iter().any(|s| {
        s.index != crate::types::MMPROJ_SHARD_INDEX
            && s.layer_range.0 >= manifest.num_layers
            && !local_shard_indices.contains(&s.index)
    });
    if lacks_a_part_past_the_end {
        if let Some(last) = layer_bits.last_mut() {
            *last = false;
        }
    }

    // Extract contiguous ranges from the bitmap
    let mut ranges = Vec::new();
    let mut run_start = 0;
    let mut in_run = false;
    for (i, &avail) in layer_bits.iter().enumerate() {
        if avail {
            if !in_run {
                run_start = i;
                in_run = true;
            }
        } else if in_run {
            ranges.push((run_start, i));
            in_run = false;
        }
    }
    if in_run {
        ranges.push((run_start, layer_bits.len()));
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::split::TensorLocation;

    /// A header of an embedding table, three layers of unequal sizes and an
    /// output head, its tensor data starting at byte 1,000 of the file.
    fn header() -> GgufTensorMeta {
        let tensors = [
            ("token_embd.weight", 0, 100),
            ("blk.0.attn_q.weight", 100, 50),
            ("blk.0.ffn_up.weight", 150, 70),
            ("blk.1.attn_q.weight", 220, 60),
            ("blk.2.attn_q.weight", 280, 80),
            ("output.weight", 360, 40),
        ]
        .into_iter()
        .map(|(name, offset, size)| (name.to_string(), TensorLocation { offset, size }))
        .collect();
        GgufTensorMeta {
            tensors,
            tensor_data_offset: 1_000,
            model_name: None,
            head_count: 4,
            head_count_kv: 4,
            block_count: 3,
            embedding_length: 64,
            head_dim: 16,
            rope_dim: 16,
            rope_freq_base: 10_000.0,
            rms_norm_eps: 1e-5,
            expert_count: 0,
            expert_used_count: 0,
            architecture: "llama".into(),
            context_length: 2048,
            pooling_type: 0,
            kv_layout: None,
            unbuilt_part: None,
            shared_kv_layers: 0,
        }
    }

    /// **The table a part file is read by** (#334 (2)): the publisher's
    /// manifest and a holder's derived table both come from
    /// `tensor_entries`, so it alone decides where a tensor sits. Each part's
    /// tensors run back to back from byte 0 and end at the part's size — the
    /// size a part on disk must have exactly (`shard::shard_file_is_whole`) —
    /// and each still names where the tensor sits in the original file.
    #[test]
    fn a_parts_tensors_run_back_to_back_and_end_at_its_size() {
        let meta = header();
        for parts in 1..=3 {
            let layouts = compute_layer_shard_layouts(&meta, parts);
            assert!(!layouts.is_empty());
            let mut total = 0;
            for layout in &layouts {
                let entries = layout.tensor_entries();
                let mut next = 0;
                for entry in &entries {
                    assert_eq!(entry.shard_offset, next, "{}", entry.name);
                    assert_eq!(
                        entry.gguf_offset,
                        meta.tensor_data_offset + meta.tensors[&entry.name].offset,
                        "{}",
                        entry.name
                    );
                    next += entry.size;
                }
                assert_eq!(next, layout.size_bytes, "part {} of {parts}", layout.index);
                total += layout.size_bytes;
            }
            assert_eq!(total, 400, "every tensor in exactly one part");
        }
    }

    /// A manifest cut as Ornith-1.5-9B's is: 32 layers the loader runs in
    /// eight parts of four, and a ninth holding the next-token-prediction
    /// block (`blk.32`) beside the output head, its range `[32, 33)`.
    fn manifest_with_a_part_past_the_end(tail_part: bool) -> crate::types::ModelManifest {
        let mut ranges: Vec<(u32, u32)> = (0..8).map(|i| (i * 4, i * 4 + 4)).collect();
        if tail_part {
            ranges.push((32, 33));
        }
        crate::types::ModelManifest {
            id: crate::types::ModelId("ornith-1-5-9b".into()),
            name: "Ornith-1.5-9B".into(),
            architecture: crate::types::ModelArchitecture::Llama,
            num_layers: 32,
            num_params_billions: 9.0,
            quantization: crate::types::Quantization::Q4KM,
            total_size_bytes: 9 * 4,
            shard_count: ranges.len() as u32,
            shards: ranges
                .into_iter()
                .enumerate()
                .map(|(i, layer_range)| crate::types::ShardInfo {
                    index: i as u32,
                    layer_range,
                    size_bytes: 4,
                    hash: [0u8; 32],
                    tensors: Vec::new(),
                })
                .collect(),
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: crate::types::NodeId([0u8; 32]),
            publish_date: chrono::Utc::now(),
            license: "MIT".into(),
            mmproj: None,
            context_length: None,
        }
    }

    /// **A run that ends the model reads the part past its last layer**
    /// (2026-11-02): Ornith-1.5's last part is `[32, 33)` — the nextn block
    /// and the output head — and overlapped no run of a 32-layer model, so a
    /// peer holding parts 0-7 was planned as holding the whole model and
    /// refused ("not the end"), and no node ever fetched part 8.
    #[test]
    fn a_part_past_the_last_layer_serves_only_a_run_that_ends_the_model() {
        // A part past the end serves a run ending the model, wherever it starts.
        assert!(part_serves_layers((32, 33), 0, 32, 32));
        assert!(part_serves_layers((32, 33), 28, 32, 32));
        // ... and no run short of the end.
        assert!(!part_serves_layers((32, 33), 0, 31, 32));
        assert!(!part_serves_layers((32, 33), 16, 28, 32));
        // An ordinary part is the plain overlap test, half-open at both ends.
        assert!(part_serves_layers((28, 32), 31, 32, 32));
        assert!(!part_serves_layers((28, 32), 24, 28, 32));
        assert!(!part_serves_layers((0, 4), 4, 8, 32));
        assert!(part_serves_layers((0, 4), 3, 8, 32));
    }

    /// **The last layer counts only beside the part past it.** Holding parts
    /// 0-7 is layers 0-30 a run may END at — never the model's end; every
    /// part is the whole model; the tail part alone serves no layer.
    #[test]
    fn the_last_layer_is_held_only_with_the_part_past_it() {
        let manifest = manifest_with_a_part_past_the_end(true);
        let all_but_the_tail: Vec<u32> = (0..8).collect();
        assert_eq!(
            available_layer_ranges_from_manifest(&manifest, &all_but_the_tail),
            vec![(0, 31)],
            "without the output head no run can be the model's last"
        );
        let every_part: Vec<u32> = (0..9).collect();
        assert_eq!(
            available_layer_ranges_from_manifest(&manifest, &every_part),
            vec![(0, 32)]
        );
        assert_eq!(
            available_layer_ranges_from_manifest(&manifest, &[8]),
            Vec::<(usize, usize)>::new()
        );
        // A run short of the end is untouched by the tail part's absence.
        assert_eq!(
            available_layer_ranges_from_manifest(&manifest, &[0, 1, 2, 3]),
            vec![(0, 16)]
        );
        assert_eq!(
            available_layer_ranges_from_manifest(&manifest, &[6, 7, 8]),
            vec![(24, 32)]
        );
    }

    /// Qwen3.8-27B as cut on 2026-11-05: 64 layers in 32 parts, part 30 is
    /// `[63, 64)` and part 31 `[64, 65)` — the output head and `blk.64.nextn`.
    fn manifest_cut_as_the_27b_is() -> crate::types::ModelManifest {
        let mut m = manifest_with_a_part_past_the_end(false);
        // The live node's `manifest.json` for qwen3.8-27b-q4-k-m, part by part.
        let ranges: Vec<(u32, u32)> = vec![
            (0, 1),
            (1, 3),
            (3, 5),
            (5, 7),
            (7, 9),
            (9, 11),
            (11, 13),
            (13, 15),
            (15, 18),
            (18, 21),
            (21, 24),
            (24, 27),
            (27, 30),
            (30, 33),
            (33, 35),
            (35, 37),
            (37, 39),
            (39, 41),
            (41, 43),
            (43, 45),
            (45, 47),
            (47, 49),
            (49, 51),
            (51, 53),
            (53, 55),
            (55, 57),
            (57, 59),
            (59, 61),
            (61, 62),
            (62, 63),
            (63, 64),
            (64, 65),
        ];
        m.num_layers = 64;
        m.shard_count = 32;
        m.shards = ranges
            .into_iter()
            .enumerate()
            .map(|(i, layer_range)| crate::types::ShardInfo {
                index: i as u32,
                layer_range,
                size_bytes: 4,
                hash: [0u8; 32],
                tensors: Vec::new(),
            })
            .collect();
        m
    }

    /// **The end of a model is the parts holding its last layer and its head**
    /// (#408): for the 27B that is parts 30 AND 31, so a node holding 0 and 31
    /// does not hold both ends. Each case fails with `[head_part]` alone.
    #[test]
    fn the_parts_that_finish_a_model_are_read_off_its_layer_ranges() {
        let the_27b = manifest_cut_as_the_27b_is();
        assert_eq!(parts_that_finish(&the_27b), vec![30, 31]);
        assert_eq!(parts_at_the_ends(&the_27b), vec![0, 30, 31]);
        let ornith = manifest_with_a_part_past_the_end(true);
        assert_eq!(parts_that_finish(&ornith), vec![7, 8]);
        assert_eq!(parts_at_the_ends(&ornith), vec![0, 7, 8]);
        // Null control: the last part holds the last layer and the head.
        let plain = manifest_with_a_part_past_the_end(false);
        assert_eq!(parts_that_finish(&plain), vec![7]);
        assert_eq!(parts_at_the_ends(&plain), vec![0, 7]);
        assert_eq!(head_part(plain.shard_count), 7);
    }

    /// A one-part model is both ends at once; the vision encoder is neither.
    #[test]
    fn a_single_part_is_both_ends_and_the_image_reader_is_neither() {
        let mut one = manifest_with_a_part_past_the_end(false);
        one.shards.truncate(1);
        one.shards[0].layer_range = (0, 32);
        one.shard_count = 1;
        assert_eq!(parts_at_the_ends(&one), vec![0]);
        let mut with_vision = manifest_with_a_part_past_the_end(false);
        with_vision.shards.push(crate::types::ShardInfo {
            index: crate::types::MMPROJ_SHARD_INDEX,
            layer_range: (32, 33),
            size_bytes: 4,
            hash: [0u8; 32],
            tensors: Vec::new(),
        });
        assert_eq!(parts_that_finish(&with_vision), vec![7]);
    }

    /// A manifest that does not list the head's part (malformed, a peer's)
    /// still needs it: holding the last layer's part alone never reads as
    /// holding the end (security review, 2026-11-05).
    #[test]
    fn the_heads_part_is_needed_even_where_the_manifest_leaves_it_out() {
        let mut cut = manifest_with_a_part_past_the_end(true);
        let head = head_part(cut.shard_count);
        cut.shards.retain(|s| s.index != head);
        let finish = parts_that_finish(&cut);
        assert!(finish.contains(&head), "{finish:?}");
        assert!(finish.len() >= 2, "the last layer's part too: {finish:?}");
    }

    /// Null control: a model whose last part holds its last layer and the
    /// head together reads exactly as before.
    #[test]
    fn a_model_with_no_part_past_its_end_is_read_as_before() {
        let manifest = manifest_with_a_part_past_the_end(false);
        let every_part: Vec<u32> = (0..8).collect();
        assert_eq!(
            available_layer_ranges_from_manifest(&manifest, &every_part),
            vec![(0, 32)]
        );
        assert_eq!(
            available_layer_ranges_from_manifest(&manifest, &[7]),
            vec![(28, 32)]
        );
    }
}
