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
    _index: u32,
    path: PathBuf,
    file_len: u64,
}

/// A reader that presents a GGUF header + v2 layer-aligned shard files as a
/// single contiguous seekable file.  This allows candle's `Content::read()`
/// and `ct.tensor()` to work transparently over shard files.
///
/// V2 shards contain packed tensor data (not byte-range slices of the GGUF).
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

impl ShardReader {
    /// Create a ShardReader from a GGUF header and v2 shard files with tensor maps.
    ///
    /// `shard_files` must be ordered by shard index.  Each shard's tensor entries
    /// describe which virtual-GGUF-offset ranges map to which shard-local offsets.
    pub fn new(
        header_path: &Path,
        shard_files: Vec<(u32, PathBuf)>,
        tensor_entries: &[Vec<crate::types::ShardTensorEntry>],
        total_gguf_size: u64,
        tensor_data_offset: u64,
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

        for (i, (idx, path)) in shard_files.iter().enumerate() {
            let file_len = std::fs::metadata(path).map_err(SwarmError::Io)?.len();
            shards.push(ShardFile {
                _index: *idx,
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
