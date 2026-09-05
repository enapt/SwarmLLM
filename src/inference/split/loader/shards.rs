//! Shard-based loading paths for `SplitModel`.
//!
//! These three functions cover the case where a node loads weights from
//! shard files + a separate GGUF header (no full GGUF on disk), which is
//! the standard SwarmLLM deployment mode. They dispatch into the shared
//! [`super::SplitModel::load_model_from_content`] body once a `ShardReader`
//! (or mmap fallback for the single-shard, no-tensor-entries case) is
//! constructed.

use std::io::SeekFrom;
use std::path::{Path, PathBuf};

use candle_core::quantized::gguf_file;
use candle_core::Device;

use crate::error::SwarmError;

use super::super::model::SplitModel;
use super::super::shard_reader::ShardReader;

impl SplitModel {
    /// Load from shards, forcing CPU device (used as GPU OOM fallback).
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_shards_cpu(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        Self::load_from_shards_inner(
            model_dir,
            shard_files,
            tensor_entries,
            total_gguf_size,
            layer_start,
            layer_end,
            is_first,
            is_last,
            true,
        )
    }

    /// Load a partial model from local shard files + GGUF header.
    ///
    /// This is the shard-only alternative to `load_from_gguf`. Instead of needing
    /// the full GGUF file, it reads from:
    /// - `gguf_header.bin`: the raw GGUF header (metadata + tensor info table)
    /// - `shard_NNN.bin` files: layer-aligned shard files with packed tensor data
    ///
    /// The `ShardReader` uses the tensor entries to map virtual GGUF positions
    /// to shard-local offsets, so candle's GGUF parser works unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn load_from_shards(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
    ) -> Result<Self, SwarmError> {
        Self::load_from_shards_inner(
            model_dir,
            shard_files,
            tensor_entries,
            total_gguf_size,
            layer_start,
            layer_end,
            is_first,
            is_last,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn load_from_shards_inner(
        model_dir: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        layer_start: usize,
        layer_end: usize,
        is_first: bool,
        is_last: bool,
        force_cpu: bool,
    ) -> Result<Self, SwarmError> {
        let header_path = model_dir.join(crate::model::shard::HEADER_FILENAME);
        if !header_path.exists() {
            return Err(SwarmError::Internal(format!(
                "GGUF header not found at {}. The originating node must generate this file.",
                header_path.display()
            )));
        }

        // Single shard with no tensor entries = full GGUF file as shard.
        // Load directly via mmap instead of ShardReader.
        let has_tensor_entries = tensor_entries.iter().any(|v| !v.is_empty());
        if shard_files.len() == 1 && !has_tensor_entries {
            let shard_path = &shard_files[0].1;
            tracing::info!(
                model_dir = %model_dir.display(),
                shard_path = %shard_path.display(),
                "Single-shard model with no tensor entries — loading as full GGUF via mmap"
            );
            // Respect force_cpu — don't delegate to load_from_gguf which always picks CUDA
            let file = std::fs::File::open(shard_path).map_err(SwarmError::Io)?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| SwarmError::Internal(format!("Failed to mmap GGUF: {e}")))?;
            let mut cursor = std::io::Cursor::new(mmap.as_ref());
            let ct = gguf_file::Content::read(&mut cursor)
                .map_err(|e| SwarmError::Internal(format!("Failed to read GGUF: {e}")))?;
            let device = if force_cpu {
                Device::Cpu
            } else {
                Device::cuda_if_available(0).unwrap_or(Device::Cpu)
            };
            return Self::load_model_from_content(
                ct,
                &mut cursor,
                device,
                super::SplitLoadOptions {
                    layer_start,
                    layer_end,
                    is_first,
                    is_last,
                    parallel_data: Some(mmap.as_ref()),
                    gpu_layers: crate::inference::split::gpu_layer_limit(),
                },
            );
        }

        // Read header to get tensor_data_offset, and — for a weight-tied model —
        // where its output head lives, so the sidecar can stand in for shard 0.
        let header_bytes = std::fs::read(&header_path).map_err(SwarmError::Io)?;
        let (tensor_data_offset, tied_output) = {
            let mut cursor = std::io::Cursor::new(&header_bytes);
            let ct = gguf_file::Content::read(&mut cursor)
                .map_err(|e| SwarmError::Internal(format!("Failed to parse GGUF header: {e}")))?;
            // Resolution needs the arch metadata block. If that can't be parsed
            // the model won't load anyway, so don't fail *here* — a non-tied
            // model on a node holding shard 0 loads fine without any of this.
            let tied = match crate::inference::split::GgufTensorMeta::from_content(&ct) {
                Ok(meta) => crate::inference::split::resolve_tied_output(model_dir, &meta),
                Err(e) => {
                    tracing::debug!(error = %e, "Could not resolve tied output weight from header");
                    None
                }
            };
            (ct.tensor_data_offset, tied)
        };

        tracing::info!(
            model_dir = %model_dir.display(),
            header_bytes = header_bytes.len(),
            tensor_data_offset,
            shards = shard_files.len(),
            layers = format!("[{layer_start}..{layer_end})"),
            "Loading split model from shard files"
        );

        let mut reader = ShardReader::new(
            &header_path,
            shard_files,
            tensor_entries,
            total_gguf_size,
            tensor_data_offset,
            tied_output,
        )?;

        // Use the same GGUF parsing path as load_from_gguf, but reading from ShardReader
        let ct = gguf_file::Content::read(&mut reader).map_err(|e| {
            SwarmError::Internal(format!("Failed to read GGUF via ShardReader: {e}"))
        })?;

        // Verify tensor_data_offset matches between the two Content::read calls
        if ct.tensor_data_offset != tensor_data_offset {
            tracing::error!(
                expected = tensor_data_offset,
                actual = ct.tensor_data_offset,
                "DIAG: tensor_data_offset MISMATCH between header parse and ShardReader parse!"
            );
        }

        // Diagnostic: log first few tensor offsets from Content vs tensor_map
        for (name, info) in ct.tensor_infos.iter().take(5) {
            let seek_pos = ct.tensor_data_offset + info.offset;
            let size_in_bytes = info.ggml_dtype.type_size() * info.shape.elem_count()
                / info.ggml_dtype.block_size();
            let found = reader.find_shard(seek_pos);
            tracing::info!(
                tensor = %name,
                gguf_seek = seek_pos,
                size = size_in_bytes,
                shard_mapping = ?found,
                "DIAG: tensor mapping check"
            );
        }

        // DIAG: read the first 16 bytes of a tensor THIS SEGMENT holds, to
        // check the shard mapping resolves to real bytes.
        //
        // It used to probe `blk.0.attn_norm.weight` unconditionally. Layer 0
        // lives in the model's first shard, which a node serving a middle
        // segment has no reason to hold — so on every such node the probe
        // read into a gap, and although its failure is ignored here (the
        // `is_ok()`), `ShardReader` reported it at ERROR. A node that went on
        // to load and serve the segment perfectly well was announcing a fault
        // it did not have. Probing layer 0 also verified nothing about the
        // data this worker is about to use.
        let probe_tensor = probe_tensor_name(layer_start, |n| ct.tensor_infos.contains_key(n))
            .and_then(|n| ct.tensor_infos.get(&n).map(|i| (n, i)));
        if let Some((name, norm_info)) = probe_tensor {
            use std::io::{Read as IoReadTrait, Seek as SeekTrait};
            let seek_pos = ct.tensor_data_offset + norm_info.offset;
            reader.seek(SeekFrom::Start(seek_pos)).ok();
            let mut probe = [0u8; 16];
            if reader.read_exact(&mut probe).is_ok() {
                tracing::info!(
                    tensor = %name,
                    seek_pos,
                    first_bytes = ?&probe,
                    "DIAG: first 16 bytes of a held tensor via ShardReader"
                );
            }
        }

        let device = if force_cpu {
            Device::Cpu
        } else {
            Device::cuda_if_available(0).unwrap_or(Device::Cpu)
        };
        if device.is_cuda() {
            tracing::info!(layer_start, layer_end, "Split model using CUDA GPU");
        } else if force_cpu {
            // The worker is TOLD where to run; it does not know why, and there
            // are three possible whys (see `process_pool::CpuReason`). The old
            // wording — "requested: gpu_layers = 0, or GPU OOM fallback" — tried
            // to name them and instead asserted a config value that may never
            // have been set: a tester read it as proof their `gpu_layers = -1`
            // was ignored, when the daemon had honoured it and then refused the
            // model for VRAM (reported 2026-08-10).
            //
            // So state only what this process actually knows, and point at the
            // side that does know. The daemon logs the reason on every spawn.
            tracing::info!(
                layer_start,
                layer_end,
                "Split model using CPU (placement chosen by the daemon — see its \
                 'Model will run on the CPU' line for the reason)"
            );
        } else {
            tracing::info!(
                layer_start,
                layer_end,
                "Split model using CPU (no CUDA available)"
            );
        }

        Self::load_model_from_content(
            ct,
            &mut reader,
            device,
            super::SplitLoadOptions {
                layer_start,
                layer_end,
                is_first,
                is_last,
                // ShardReader can't be shared across threads for parallel loading
                parallel_data: None,
                gpu_layers: crate::inference::split::gpu_layer_limit(),
            },
        )
    }
}

/// Which tensor should the load-time integrity probe read?
///
/// This segment's OWN first layer, falling back to layer 0 only if the model
/// has no such tensor. The probe used to name layer 0 unconditionally, which
/// is in the model's first shard — a node serving a middle segment has no
/// reason to hold it, so the read landed in a gap on exactly the nodes
/// partial holding is designed for, and verified nothing about the data the
/// worker was about to use either way.
///
/// `has` asks whether the model declares that tensor; `None` means there is
/// nothing sensible to probe and the caller skips it.
fn probe_tensor_name(layer_start: usize, has: impl Fn(&str) -> bool) -> Option<String> {
    let own = format!("blk.{layer_start}.attn_norm.weight");
    if has(&own) {
        return Some(own);
    }
    let first = "blk.0.attn_norm.weight".to_string();
    has(&first).then_some(first)
}

#[cfg(test)]
mod probe_tests {
    use super::probe_tensor_name;

    #[test]
    fn a_middle_segment_probes_its_own_first_layer() {
        let declared = |n: &str| n.starts_with("blk.") && n.ends_with(".attn_norm.weight");
        assert_eq!(
            probe_tensor_name(29, declared).as_deref(),
            Some("blk.29.attn_norm.weight"),
            "a node serving [29..32) must probe data it actually holds, not layer 0 \
             — layer 0 is in the first shard, which it has no reason to have"
        );
    }

    #[test]
    fn a_first_segment_probes_layer_zero_as_before() {
        let declared = |n: &str| n.starts_with("blk.") && n.ends_with(".attn_norm.weight");
        assert_eq!(
            probe_tensor_name(0, declared).as_deref(),
            Some("blk.0.attn_norm.weight")
        );
    }

    #[test]
    fn an_architecture_without_that_tensor_is_not_probed() {
        assert_eq!(probe_tensor_name(29, |_| false), None);
    }

    /// A model that declares layer 0 but not this segment's own layer still
    /// gets a probe rather than none — the fallback is a fallback.
    #[test]
    fn the_fallback_still_applies_when_the_segments_tensor_is_absent() {
        assert_eq!(
            probe_tensor_name(29, |n| n == "blk.0.attn_norm.weight").as_deref(),
            Some("blk.0.attn_norm.weight")
        );
    }
}
