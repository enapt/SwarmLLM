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

        // DIAG: Read first 16 bytes of blk.0.attn_norm.weight via ShardReader
        // to verify data integrity
        if let Some(norm_info) = ct.tensor_infos.get("blk.0.attn_norm.weight") {
            use std::io::{Read as IoReadTrait, Seek as SeekTrait};
            let seek_pos = ct.tensor_data_offset + norm_info.offset;
            reader.seek(SeekFrom::Start(seek_pos)).ok();
            let mut probe = [0u8; 16];
            if reader.read_exact(&mut probe).is_ok() {
                tracing::info!(
                    seek_pos,
                    first_bytes = ?&probe,
                    "DIAG: blk.0.attn_norm.weight first 16 bytes via ShardReader"
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
