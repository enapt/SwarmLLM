//! V2 layer-aligned shard layout computation.

use std::collections::HashMap;

use super::split::GgufTensorMeta;

// ── V2 Layer-Aligned Sharding ──

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

/// Return all contiguous layer ranges from manifest ShardInfo entries.
///
/// Reads layer_range directly from each shard — v2 manifests have accurate
/// layer ranges computed from GGUF tensor metadata.
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
