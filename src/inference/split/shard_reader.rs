// ── ShardReader: virtual GGUF file from header + shard files ──

use std::io::{Read as IoRead, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::error::SwarmError;

/// One tensor's mapping from virtual GGUF position to a shard file.
struct TensorMapEntry {
    /// Absolute byte offset in the virtual GGUF file.
    gguf_offset: u64,
    /// Index into the `shards` vec.
    shard_idx: usize,
    /// Byte offset within the shard file where this tensor's data starts.
    shard_local_offset: u64,
    /// Size of this tensor's data in bytes.
    size: u64,
}

/// Metadata for one shard file.
struct ShardFile {
    path: PathBuf,
    file_len: u64,
}

/// The `tied_output_weight.bin` sidecar, mapped into the virtual GGUF.
///
/// On a weight-tied model the LM head *is* `token_embd.weight`, which lives in
/// shard 0. A node serving the last segment needs that tensor but frequently
/// does not hold shard 0 — so without this the output head is unreachable and
/// the whole pipeline fails. Backing the tensor's gguf byte range with the
/// sidecar makes `ct.tensor(&mut reader, "token_embd.weight", …)` resolve with
/// no change at the call site.
pub struct TiedOutputSource {
    /// Path to the sidecar. Holds the raw tensor bytes at offset 0, nothing else.
    pub path: PathBuf,
    /// Absolute offset of the tensor in the virtual GGUF
    /// (`tensor_data_offset + location.offset`).
    pub gguf_offset: u64,
    /// Tensor size in bytes, from the GGUF header — never the file length, so a
    /// truncated sidecar is caught rather than silently mapped short.
    pub size: u64,
}

/// A reader that presents a GGUF header + layer-aligned shard files as a
/// single contiguous seekable file.  This allows candle's `Content::read()`
/// and `ct.tensor()` to work transparently over shard files.
///
/// Shards contain packed tensor data (not byte-range slices of the GGUF).
/// The `tensor_map` translates virtual GGUF offsets → (shard_idx, shard_local_offset)
/// via binary search.
pub struct ShardReader {
    /// Raw GGUF header bytes (metadata + tensor info table), padded to tensor_data_offset.
    header: Vec<u8>,
    /// Shard files in order by index.
    shards: Vec<ShardFile>,
    /// Sorted by `gguf_offset` for binary search.
    tensor_map: Vec<TensorMapEntry>,
    /// Total size of the virtual GGUF file (header + all tensor data).
    total_size: u64,
    /// Current seek position in the virtual file.
    position: u64,
    /// Currently open shard file handle (cached to avoid repeated opens).
    current_shard: Option<(usize, std::fs::File)>,
}

/// Resolve the tied-output sidecar for a model directory.
///
/// `Some` only when the model is weight-tied AND the sidecar is on disk. A
/// model with a real `output.weight` returns `None` — it has no tied head to
/// map. A weight-tied model whose sidecar is missing also returns `None`: if
/// this node holds shard 0 the load still succeeds from the shard, and if it
/// doesn't, the load fails with the missing-region error naming the offset.
pub fn resolve_tied_output(
    model_dir: &Path,
    meta: &crate::inference::split::GgufTensorMeta,
) -> Option<TiedOutputSource> {
    let loc = meta.tied_output_location()?;
    let path = model_dir.join(crate::inference::split::TIED_OUTPUT_FILENAME);
    if !path.exists() {
        return None;
    }
    Some(TiedOutputSource {
        path,
        gguf_offset: meta.tensor_data_offset + loc.offset,
        size: loc.size,
    })
}

impl ShardReader {
    /// Create a ShardReader from a GGUF header and shard files with tensor maps.
    ///
    /// `shard_files` must be ordered by shard index.  Each shard's tensor entries
    /// describe which virtual-GGUF-offset ranges map to which shard-local offsets.
    ///
    /// `tied_output` is REQUIRED rather than defaulted behind a convenience
    /// wrapper: a caller that silently passes nothing is exactly how a
    /// weight-tied model becomes unservable on any node lacking shard 0. Pass
    /// `None` only when the model ships a real `output.weight`. Resolve it with
    /// [`resolve_tied_output`] rather than assembling one by hand.
    pub fn new(
        header_path: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        tensor_data_offset: u64,
        tied_output: Option<TiedOutputSource>,
    ) -> Result<Self, SwarmError> {
        let header = std::fs::read(header_path).map_err(SwarmError::Io)?;
        // SEC: Cap padding to prevent OOM from malicious tensor_data_offset
        const MAX_GGUF_HEADER_SIZE: usize = 64 * 1024 * 1024; // 64 MB
        let header = if (header.len() as u64) < tensor_data_offset {
            if (tensor_data_offset as usize) > MAX_GGUF_HEADER_SIZE {
                return Err(SwarmError::Internal(format!(
                    "GGUF header offset too large: {} bytes (max {})",
                    tensor_data_offset, MAX_GGUF_HEADER_SIZE
                )));
            }
            let mut padded = header;
            padded.resize(tensor_data_offset as usize, 0);
            padded
        } else {
            header
        };

        let mut shards = Vec::with_capacity(shard_files.len());
        let mut tensor_map = Vec::new();

        for (i, (_idx, path)) in shard_files.iter().enumerate() {
            let file_len = std::fs::metadata(path).map_err(SwarmError::Io)?.len();
            shards.push(ShardFile {
                path: path.clone(),
                file_len,
            });

            // Build tensor map entries from the corresponding tensor_entries
            if let Some(entries) = tensor_entries.get(i) {
                for te in entries {
                    tensor_map.push(TensorMapEntry {
                        gguf_offset: te.gguf_offset,
                        shard_idx: i,
                        shard_local_offset: te.shard_offset,
                        size: te.size,
                    });
                }
            }
        }

        // Map the tied output head from its sidecar, but ONLY where the shards
        // present don't already cover it. A node holding shard 0 reads the
        // tensor from the shard as before; adding a second entry at the same
        // gguf_offset would make the binary search in `find_shard` ambiguous.
        if let Some(tied) = tied_output {
            let already_covered = tensor_map.iter().any(|e| {
                tied.gguf_offset >= e.gguf_offset && tied.gguf_offset < e.gguf_offset + e.size
            });
            if already_covered {
                tracing::debug!(
                    gguf_offset = tied.gguf_offset,
                    "Tied output weight already covered by a local shard — using the shard"
                );
            } else {
                let file_len = std::fs::metadata(&tied.path).map_err(SwarmError::Io)?.len();
                if file_len < tied.size {
                    return Err(SwarmError::Internal(format!(
                        "tied_output_weight.bin is short: {} bytes on disk, header says the tensor is {} \
                         bytes. Delete {} so it can be re-fetched.",
                        file_len,
                        tied.size,
                        tied.path.display()
                    )));
                }
                shards.push(ShardFile {
                    path: tied.path.clone(),
                    file_len,
                });
                tensor_map.push(TensorMapEntry {
                    gguf_offset: tied.gguf_offset,
                    shard_idx: shards.len() - 1,
                    // The sidecar holds this tensor and nothing else.
                    shard_local_offset: 0,
                    size: tied.size,
                });
                tracing::info!(
                    path = %tied.path.display(),
                    gguf_offset = tied.gguf_offset,
                    size = tied.size,
                    "Mapped tied output head from sidecar — shard 0 is not held locally"
                );
            }
        }

        // Sort by gguf_offset for binary search
        tensor_map.sort_by_key(|e| e.gguf_offset);

        Ok(Self {
            header,
            shards,
            tensor_map,
            total_size: total_gguf_size,
            position: 0,
            current_shard: None,
        })
    }

    /// Find which shard (if any) contains the given virtual file position,
    /// returning (shard_vec_index, offset_within_shard_file, remaining_bytes_in_tensor).
    pub(crate) fn find_shard(&self, pos: u64) -> Option<(usize, u64, u64)> {
        // Binary search: find the last entry where gguf_offset <= pos
        let idx = match self
            .tensor_map
            .binary_search_by_key(&pos, |e| e.gguf_offset)
        {
            Ok(i) => i,
            Err(0) => return None,
            Err(i) => i - 1,
        };

        let entry = &self.tensor_map[idx];
        if pos < entry.gguf_offset + entry.size {
            let delta = pos - entry.gguf_offset;
            let remaining_in_tensor = entry.size - delta;
            Some((
                entry.shard_idx,
                entry.shard_local_offset + delta,
                remaining_in_tensor,
            ))
        } else {
            None
        }
    }
}

impl IoRead for ShardReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.position >= self.total_size {
            return Ok(0);
        }

        let header_len = self.header.len() as u64;

        // Reading from header region
        if self.position < header_len {
            let start = self.position as usize;
            let available = (header_len - self.position) as usize;
            let to_read = buf.len().min(available);
            buf[..to_read].copy_from_slice(&self.header[start..start + to_read]);
            self.position += to_read as u64;
            return Ok(to_read);
        }

        // Reading from shard region via tensor map
        if let Some((shard_idx, offset_in_shard, remaining_in_tensor)) =
            self.find_shard(self.position)
        {
            if tracing::enabled!(tracing::Level::TRACE) {
                tracing::trace!(
                    pos = self.position,
                    shard_idx,
                    offset_in_shard,
                    remaining_in_tensor,
                    buf_len = buf.len(),
                    "ShardReader::read"
                );
            }
            // Open the shard file if not already open
            let need_open = match &self.current_shard {
                Some((idx, _)) => *idx != shard_idx,
                None => true,
            };
            if need_open {
                let file = std::fs::File::open(&self.shards[shard_idx].path)
                    .map_err(std::io::Error::other)?;
                self.current_shard = Some((shard_idx, file));
            }

            let shard_file_len = self.shards[shard_idx].file_len;
            let (_, ref mut file) = self.current_shard.as_mut().expect("shard opened above");
            file.seek(SeekFrom::Start(offset_in_shard))?;
            let available_in_shard = shard_file_len.saturating_sub(offset_in_shard) as usize;
            // Cap read to tensor boundary — never bleed into adjacent tensor data
            let available = available_in_shard.min(remaining_in_tensor as usize);
            let to_read = buf.len().min(available);
            if to_read == 0 {
                tracing::error!(
                    pos = self.position,
                    shard_idx,
                    offset_in_shard,
                    shard_file_len,
                    buf_len = buf.len(),
                    "ShardReader: 0 bytes available at offset in shard"
                );
                return Ok(0);
            }
            let n = file.read(&mut buf[..to_read])?;
            self.position += n as u64;
            Ok(n)
        } else {
            // Position is in a gap (missing tensor / missing shard)
            let map_info: Vec<String> = self
                .tensor_map
                .iter()
                .take(5)
                .map(|e| {
                    format!(
                        "shard[{}]@gguf[{}..{})",
                        e.shard_idx,
                        e.gguf_offset,
                        e.gguf_offset + e.size
                    )
                })
                .collect();
            tracing::error!(
                pos = self.position,
                total_size = self.total_size,
                header_len = self.header.len(),
                buf_len = buf.len(),
                tensor_map_sample = ?map_info,
                "ShardReader: position is in a missing shard region"
            );
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!(
                    "ShardReader: position {} is in a missing region (total_size={})",
                    self.position, self.total_size
                ),
            ))
        }
    }
}

impl Seek for ShardReader {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let new_pos = match pos {
            SeekFrom::Start(p) => p as i64,
            SeekFrom::End(p) => self.total_size as i64 + p,
            SeekFrom::Current(p) => self.position as i64 + p,
        };
        if new_pos < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Seek before start",
            ));
        }
        self.position = new_pos as u64;
        Ok(self.position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inference::split::{GgufTensorMeta, TIED_OUTPUT_FILENAME};
    use std::io::Read;

    const HEADER_LEN: u64 = 64;
    const EMBD_SIZE: u64 = 32;
    /// Where `token_embd.weight` sits in the virtual GGUF. Matches the real
    /// layout: the embedding is the first sizeable tensor after the header.
    const EMBD_GGUF_OFFSET: u64 = HEADER_LEN;

    /// A model dir holding a header and a `tied_output_weight.bin` full of `0xAA`.
    fn model_dir_with_sidecar(sidecar_len: u64) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("gguf_header.bin"),
            vec![0u8; HEADER_LEN as usize],
        )
        .unwrap();
        std::fs::write(
            dir.path().join(TIED_OUTPUT_FILENAME),
            vec![0xAAu8; sidecar_len as usize],
        )
        .unwrap();
        dir
    }

    fn tied_source(dir: &tempfile::TempDir, size: u64) -> TiedOutputSource {
        TiedOutputSource {
            path: dir.path().join(TIED_OUTPUT_FILENAME),
            gguf_offset: EMBD_GGUF_OFFSET,
            size,
        }
    }

    fn meta(tied: bool) -> GgufTensorMeta {
        let mut tensors = serde_json::json!({
            "token_embd.weight": { "offset": 0, "size": EMBD_SIZE },
        });
        if !tied {
            tensors["output.weight"] = serde_json::json!({ "offset": 512, "size": EMBD_SIZE });
        }
        serde_json::from_value(serde_json::json!({
            "tensors": tensors,
            "tensor_data_offset": HEADER_LEN,
            "model_name": null,
            "head_count": 8,
            "head_count_kv": 8,
            "block_count": 4,
            "embedding_length": 64,
            "rope_dim": 8,
            "rope_freq_base": 10000.0,
            "rms_norm_eps": 1e-5,
        }))
        .unwrap()
    }

    fn read_at(reader: &mut ShardReader, pos: u64, len: usize) -> std::io::Result<Vec<u8>> {
        reader.seek(SeekFrom::Start(pos))?;
        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// The regression this whole change exists for: a node holding only the LAST
    /// shard must still be able to read the tied output head.
    #[test]
    fn tied_output_readable_when_shard_zero_is_absent() {
        let dir = model_dir_with_sidecar(EMBD_SIZE);
        // This node holds only a late shard, which covers a *different* range.
        let late_shard = dir.path().join("shard_002.bin");
        std::fs::write(&late_shard, vec![0x11u8; 16]).unwrap();
        let entries = vec![vec![crate::types::ShardTensorEntry {
            name: "blk.3.attn_norm.weight".to_string(),
            gguf_offset: 4096,
            shard_offset: 0,
            size: 16,
        }]];

        let mut reader = ShardReader::new(
            &dir.path().join("gguf_header.bin"),
            vec![(2, late_shard)],
            &entries,
            8192,
            HEADER_LEN,
            Some(tied_source(&dir, EMBD_SIZE)),
        )
        .unwrap();

        let got = read_at(&mut reader, EMBD_GGUF_OFFSET, EMBD_SIZE as usize)
            .expect("tied output head must be readable without shard 0");
        assert_eq!(got, vec![0xAAu8; EMBD_SIZE as usize]);
    }

    /// Without the sidecar the read still fails — proving the test above passes
    /// because of the mapping and not because the range was reachable anyway.
    #[test]
    fn tied_output_unreadable_without_sidecar() {
        let dir = model_dir_with_sidecar(EMBD_SIZE);
        let late_shard = dir.path().join("shard_002.bin");
        std::fs::write(&late_shard, vec![0x11u8; 16]).unwrap();
        let entries = vec![vec![crate::types::ShardTensorEntry {
            name: "blk.3.attn_norm.weight".to_string(),
            gguf_offset: 4096,
            shard_offset: 0,
            size: 16,
        }]];

        let mut reader = ShardReader::new(
            &dir.path().join("gguf_header.bin"),
            vec![(2, late_shard)],
            &entries,
            8192,
            HEADER_LEN,
            None,
        )
        .unwrap();

        assert!(read_at(&mut reader, EMBD_GGUF_OFFSET, EMBD_SIZE as usize).is_err());
    }

    /// A node that DOES hold shard 0 keeps reading from the shard. The sidecar
    /// must not shadow it, or a duplicate gguf_offset would make the binary
    /// search in `find_shard` ambiguous.
    #[test]
    fn local_shard_wins_over_sidecar() {
        let dir = model_dir_with_sidecar(EMBD_SIZE);
        let shard0 = dir.path().join("shard_000.bin");
        std::fs::write(&shard0, vec![0x77u8; EMBD_SIZE as usize]).unwrap();
        let entries = vec![vec![crate::types::ShardTensorEntry {
            name: "token_embd.weight".to_string(),
            gguf_offset: EMBD_GGUF_OFFSET,
            shard_offset: 0,
            size: EMBD_SIZE,
        }]];

        let mut reader = ShardReader::new(
            &dir.path().join("gguf_header.bin"),
            vec![(0, shard0)],
            &entries,
            8192,
            HEADER_LEN,
            Some(tied_source(&dir, EMBD_SIZE)),
        )
        .unwrap();

        let got = read_at(&mut reader, EMBD_GGUF_OFFSET, EMBD_SIZE as usize).unwrap();
        assert_eq!(
            got,
            vec![0x77u8; EMBD_SIZE as usize],
            "shard bytes, not sidecar"
        );
    }

    /// A short sidecar is a corrupt download. Fail loudly at construction rather
    /// than mapping a truncated range and producing garbage logits.
    #[test]
    fn truncated_sidecar_is_rejected() {
        let dir = model_dir_with_sidecar(EMBD_SIZE - 8);
        let result = ShardReader::new(
            &dir.path().join("gguf_header.bin"),
            vec![],
            &[],
            8192,
            HEADER_LEN,
            Some(tied_source(&dir, EMBD_SIZE)),
        );
        let err = match result {
            Ok(_) => panic!("a short sidecar must not be accepted"),
            Err(e) => e,
        };
        assert!(err.to_string().contains("short"), "got: {err}");
    }

    #[test]
    fn resolve_skips_models_with_a_real_output_weight() {
        let dir = model_dir_with_sidecar(EMBD_SIZE);
        assert!(resolve_tied_output(dir.path(), &meta(false)).is_none());
    }

    #[test]
    fn resolve_finds_offset_and_size_for_a_tied_model() {
        let dir = model_dir_with_sidecar(EMBD_SIZE);
        let got = resolve_tied_output(dir.path(), &meta(true)).expect("tied model with sidecar");
        assert_eq!(got.gguf_offset, EMBD_GGUF_OFFSET);
        assert_eq!(got.size, EMBD_SIZE);
    }

    #[test]
    fn resolve_returns_none_when_sidecar_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        assert!(resolve_tied_output(dir.path(), &meta(true)).is_none());
    }
}
