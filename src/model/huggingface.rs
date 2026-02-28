use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::time::SystemTime;

/// A GGUF model file discovered on HuggingFace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfModelResult {
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub downloads: u64,
}

/// Search HuggingFace for GGUF model files.
///
/// Uses the HF API to find repos, then checks for GGUF files in each.
/// Returns a flat list of downloadable GGUF files with repo + filename.
pub async fn search_gguf_models(query: &str) -> Result<Vec<HfModelResult>, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| e.to_string())?;

    // Search HF API for models matching query, filtered to gguf tag
    let url = format!(
        "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit=20",
        urlencoding::encode(query)
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "SwarmLLM/0.1")
        .send()
        .await
        .map_err(|e| format!("HF API request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("HF API returned {}", resp.status()));
    }

    let repos: Vec<HfRepoInfo> = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse HF response: {e}"))?;

    let mut results = Vec::new();

    // For each repo, look up the file tree to find GGUF files
    for repo in repos.iter().take(10) {
        match fetch_gguf_files(&client, &repo.id).await {
            Ok(files) => {
                for file in files {
                    results.push(HfModelResult {
                        repo_id: repo.id.clone(),
                        filename: file.rfilename.clone(),
                        size_bytes: file.size.unwrap_or(0),
                        downloads: repo.downloads.unwrap_or(0),
                    });
                }
            }
            Err(e) => {
                tracing::debug!(repo = %repo.id, error = %e, "Failed to list files for repo");
            }
        }
    }

    Ok(results)
}

/// Fetch the list of GGUF files in a HuggingFace repo.
async fn fetch_gguf_files(
    client: &reqwest::Client,
    repo_id: &str,
) -> Result<Vec<HfFileInfo>, String> {
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");

    let resp = client
        .get(&url)
        .header("User-Agent", "SwarmLLM/0.1")
        .send()
        .await
        .map_err(|e| format!("File list request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("File list returned {}", resp.status()));
    }

    let detail: HfModelDetail = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse file list: {e}"))?;

    let gguf_files: Vec<HfFileInfo> = detail
        .siblings
        .unwrap_or_default()
        .into_iter()
        .filter(|f| f.rfilename.ends_with(".gguf"))
        .collect();

    Ok(gguf_files)
}

/// Get the direct download URL for a file in a HuggingFace repo.
pub fn download_url(repo_id: &str, filename: &str) -> String {
    format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    )
}

/// Download a GGUF file from HuggingFace with progress reporting.
///
/// Supports HTTP range requests for partial/shard downloads.
/// Returns the path to the downloaded file.
pub async fn download_model(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
) -> Result<std::path::PathBuf, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);

    let resp = client
        .get(&url)
        .header("User-Agent", "SwarmLLM/0.1")
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Download returned {}", resp.status()));
    }

    let total_size = resp.content_length().unwrap_or(0);

    // Create destination directory
    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;

    // SECURITY: Strip directory components from filename to prevent path traversal
    let safe_filename = std::path::Path::new(filename)
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));
    let dest_path = dest_dir.join(safe_filename);
    let mut file = tokio::fs::File::create(&dest_path)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    use tokio::io::AsyncWriteExt;

    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();

    use futures::StreamExt;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download chunk error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;
        downloaded += chunk.len() as u64;

        if let Some(ref tx) = progress_tx {
            let _ = tx.try_send(DownloadProgress {
                downloaded_bytes: downloaded,
                total_bytes: total_size,
            });
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("Flush error: {e}"))?;

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        size = downloaded,
        "HuggingFace download complete"
    );

    Ok(dest_path)
}

/// Progress update for an in-flight download.
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

// ── Shard-level download via HTTP Range requests ──

/// Information about a remote GGUF file needed to plan shard downloads.
#[derive(Debug, Clone)]
pub struct GgufFileInfo {
    pub total_size: u64,
    pub header_size: u64,
    pub tensor_meta: crate::inference::split::GgufTensorMeta,
    pub layouts: Vec<crate::inference::split::LayerShardLayout>,
}

impl GgufFileInfo {
    /// Number of shards (derived from layouts).
    pub fn shard_count(&self) -> u32 {
        self.layouts.len() as u32
    }
}

/// Probe a remote GGUF file on HuggingFace to determine its size, tensor layout,
/// and layer-aligned shard plan.
///
/// Uses a HEAD request for total size, then a Range GET for the first 16MB to parse
/// the GGUF header and extract tensor metadata. Computes layer-aligned shard layouts.
pub async fn probe_gguf_file(
    repo_id: &str,
    filename: &str,
    shard_size: u64,
) -> Result<GgufFileInfo, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);

    // HEAD request to get total file size
    let head_resp = client
        .head(&url)
        .header("User-Agent", "SwarmLLM/0.1")
        .send()
        .await
        .map_err(|e| format!("HEAD request failed: {e}"))?;

    if !head_resp.status().is_success() {
        return Err(format!("HEAD returned {}", head_resp.status()));
    }

    let total_size = head_resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<u64>().ok())
        .ok_or("Server did not return Content-Length")?;

    // Download the first 16MB to parse the GGUF header.
    // Most GGUF headers are <10MB; 16MB gives margin for large vocab models.
    let probe_size: u64 = 16 * 1024 * 1024;
    let range_end = (probe_size - 1).min(total_size - 1);

    let probe_resp = client
        .get(&url)
        .header("User-Agent", "SwarmLLM/0.1")
        .header("Range", format!("bytes=0-{range_end}"))
        .send()
        .await
        .map_err(|e| format!("Range probe request failed: {e}"))?;

    // 206 Partial Content means Range requests are supported
    if probe_resp.status().as_u16() != 206 && !probe_resp.status().is_success() {
        return Err(format!(
            "Range probe returned {} (server may not support Range requests)",
            probe_resp.status()
        ));
    }

    let probe_bytes = probe_resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read probe bytes: {e}"))?;

    // Parse the GGUF header to get tensor_data_offset and tensor metadata
    let mut cursor = Cursor::new(&probe_bytes[..]);
    let ct = candle_core::quantized::gguf_file::Content::read(&mut cursor)
        .map_err(|e| format!("Failed to parse GGUF header from probe: {e}"))?;

    let header_size = ct.tensor_data_offset;

    // Build GgufTensorMeta from the parsed GGUF content
    let tensor_meta = build_tensor_meta_from_content(&ct)
        .map_err(|e| format!("Failed to extract tensor metadata from probe: {e}"))?;

    let shard_count = total_size.div_ceil(shard_size).max(1) as u32;
    let layouts = crate::inference::split::compute_layer_shard_layouts(&tensor_meta, shard_count);

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        total_size,
        header_size,
        shard_count = layouts.len(),
        shard_size_mb = shard_size / (1024 * 1024),
        "Probed remote GGUF file"
    );

    Ok(GgufFileInfo {
        total_size,
        header_size,
        tensor_meta,
        layouts,
    })
}

/// Build `GgufTensorMeta` from a candle `Content` that was parsed from a GGUF header.
fn build_tensor_meta_from_content(
    ct: &candle_core::quantized::gguf_file::Content,
) -> Result<crate::inference::split::GgufTensorMeta, String> {
    use std::collections::HashMap;

    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| "llama".to_string());

    let md_get = |suffix: &str| -> Result<&candle_core::quantized::gguf_file::Value, String> {
        let key = format!("{arch}.{suffix}");
        ct.metadata
            .get(&key)
            .ok_or_else(|| format!("Missing GGUF metadata: {key}"))
    };

    let head_count = md_get("attention.head_count")?
        .to_u32()
        .map_err(|e| e.to_string())? as usize;
    let head_count_kv = md_get("attention.head_count_kv")?
        .to_u32()
        .map_err(|e| e.to_string())? as usize;
    let block_count = md_get("block_count")?.to_u32().map_err(|e| e.to_string())? as usize;
    let embedding_length = md_get("embedding_length")?
        .to_u32()
        .map_err(|e| e.to_string())? as usize;
    let rope_dim = md_get("rope.dimension_count")
        .and_then(|v| v.to_u32().map_err(|e| e.to_string()))
        .unwrap_or((embedding_length / head_count) as u32) as usize;
    let rms_norm_eps = md_get("attention.layer_norm_rms_epsilon")?
        .to_f32()
        .map_err(|e| e.to_string())? as f64;
    let rope_freq_base = ct
        .metadata
        .get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(10000f32);
    let model_name = ct
        .metadata
        .get("general.name")
        .and_then(|v| v.to_string().ok().cloned());

    let mut tensors = HashMap::new();
    for (name, info) in &ct.tensor_infos {
        let size =
            info.ggml_dtype.type_size() * info.shape.elem_count() / info.ggml_dtype.block_size();
        tensors.insert(
            name.clone(),
            crate::inference::split::TensorLocation {
                offset: info.offset,
                size: size as u64,
            },
        );
    }

    Ok(crate::inference::split::GgufTensorMeta {
        tensors,
        tensor_data_offset: ct.tensor_data_offset,
        model_name,
        head_count,
        head_count_kv,
        block_count,
        embedding_length,
        rope_dim,
        rope_freq_base,
        rms_norm_eps,
    })
}

/// Download the GGUF header (metadata + tensor info table) from a remote GGUF file.
///
/// Returns the path to the saved `gguf_header.bin` file.
pub async fn download_gguf_header(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    header_size: u64,
) -> Result<std::path::PathBuf, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);
    let range_end = header_size - 1;

    let resp = client
        .get(&url)
        .header("User-Agent", "SwarmLLM/0.1")
        .header("Range", format!("bytes=0-{range_end}"))
        .send()
        .await
        .map_err(|e| format!("Header download failed: {e}"))?;

    if resp.status().as_u16() != 206 && !resp.status().is_success() {
        return Err(format!("Header download returned {}", resp.status()));
    }

    let header_bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read header bytes: {e}"))?;

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;
    let dest_path = dest_dir.join("gguf_header.bin");
    std::fs::write(&dest_path, &header_bytes)
        .map_err(|e| format!("Failed to write gguf_header.bin: {e}"))?;

    tracing::info!(
        size = header_bytes.len(),
        path = %dest_path.display(),
        "Downloaded GGUF header from HuggingFace"
    );

    Ok(dest_path)
}

/// Parse the `Retry-After` header value.
///
/// Supports two formats per RFC 7231:
/// - Delta-seconds: e.g. "120" → 120 seconds
/// - HTTP-date: e.g. "Fri, 28 Feb 2026 04:00:00 GMT" → seconds until that time
///
/// Returns `None` if the header cannot be parsed in either format.
pub fn parse_retry_after(value: &str) -> Option<u64> {
    // Try delta-seconds first (most common for APIs)
    if let Ok(secs) = value.trim().parse::<u64>() {
        return Some(secs);
    }

    // Try HTTP-date format: "Day, DD Mon YYYY HH:MM:SS GMT"
    parse_http_date(value.trim()).and_then(|target| {
        let now = SystemTime::now();
        target.duration_since(now).ok().map(|d| d.as_secs().max(1))
    })
}

/// Parse an HTTP-date string into a SystemTime.
///
/// Supports the preferred format: "Day, DD Mon YYYY HH:MM:SS GMT"
fn parse_http_date(s: &str) -> Option<SystemTime> {
    // Format: "Fri, 28 Feb 2026 04:00:00 GMT"
    let parts: Vec<&str> = s.split_whitespace().collect();
    if parts.len() != 6 || parts[5] != "GMT" {
        return None;
    }

    let day: i64 = parts[1].trim_end_matches(',').parse().ok()?;
    let month: i64 = match parts[2] {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts[3].parse().ok()?;
    let time_parts: Vec<&str> = parts[4].split(':').collect();
    if time_parts.len() != 3 {
        return None;
    }
    let hour: i64 = time_parts[0].parse().ok()?;
    let min: i64 = time_parts[1].parse().ok()?;
    let sec: i64 = time_parts[2].parse().ok()?;

    // Convert to Unix timestamp using Rata Die algorithm
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let days = 365 * y + y / 4 - y / 100 + y / 400 + (m * 306 + 5) / 10 + (day - 1) - 719468; // Unix epoch offset
    let timestamp = days * 86400 + hour * 3600 + min * 60 + sec;
    if timestamp < 0 {
        return None;
    }

    Some(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(timestamp as u64))
}

/// Coalesce nearby byte ranges (within `max_gap` bytes) into fewer requests.
///
/// Input: sorted list of (offset, size) pairs.
/// Output: merged (start, end_exclusive) ranges.
pub fn coalesce_byte_ranges(ranges: &[(u64, u64)], max_gap: u64) -> Vec<(u64, u64)> {
    if ranges.is_empty() {
        return vec![];
    }
    let mut sorted: Vec<(u64, u64)> = ranges.iter().map(|&(off, sz)| (off, off + sz)).collect();
    sorted.sort_by_key(|r| r.0);

    let mut merged = vec![sorted[0]];
    for &(start, end) in &sorted[1..] {
        let last = merged.last_mut().unwrap();
        if start <= last.1 + max_gap {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    merged
}

/// Download a v2 layer-aligned shard from HuggingFace.
///
/// Takes a `LayerShardLayout` describing which tensors belong to this shard.
/// Coalesces nearby byte ranges into fewer HTTP Range requests, then packs
/// the tensor data sequentially into `shard_{idx:03}.bin`.
///
/// Connection errors are retried up to 3 times with backoff.
pub async fn download_shard_v2(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    layout: &crate::inference::split::LayerShardLayout,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
) -> Result<std::path::PathBuf, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);
    let shard_index = layout.index;

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;
    let dest_path = dest_dir.join(format!("shard_{shard_index:03}.bin"));
    let tmp_path = dest_dir.join(format!("shard_{shard_index:03}.bin.tmp"));

    // Build byte ranges from tensor locations (gguf_offset, size)
    let tensor_ranges: Vec<(u64, u64)> = layout
        .tensors
        .iter()
        .map(|(_, offset, size)| (*offset, *size))
        .collect();

    // Coalesce nearby ranges (4MB gap tolerance) to reduce HTTP requests
    let coalesced = coalesce_byte_ranges(&tensor_ranges, 4 * 1024 * 1024);
    let total_download_bytes: u64 = coalesced.iter().map(|(s, e)| e - s).sum();

    use tokio::io::AsyncWriteExt;
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create tmp file: {e}"))?;

    let mut downloaded: u64 = 0;

    // Download each coalesced range and write to file
    for (range_start, range_end) in &coalesced {
        let http_retry_delays = [5u64, 30, 120];
        let mut resp = None;

        for attempt in 0..=3u32 {
            let result = client
                .get(&url)
                .header("User-Agent", "SwarmLLM/0.1")
                .header("Range", format!("bytes={}-{}", range_start, range_end - 1))
                .send()
                .await;

            match result {
                Ok(r) => {
                    let status = r.status().as_u16();
                    if status == 429 || status == 503 {
                        if attempt < 3 {
                            let retry_secs = r
                                .headers()
                                .get("retry-after")
                                .and_then(|v| v.to_str().ok())
                                .and_then(parse_retry_after)
                                .unwrap_or(http_retry_delays[attempt as usize])
                                .min(600);
                            tracing::warn!(
                                status,
                                retry_after_secs = retry_secs,
                                attempt = attempt + 1,
                                "HF rate limited, retrying"
                            );
                            tokio::time::sleep(std::time::Duration::from_secs(retry_secs)).await;
                            continue;
                        }
                        return Err(format!("Shard download returned {} after retries", status));
                    }
                    resp = Some(r);
                    break;
                }
                Err(e) => {
                    if attempt < 3 {
                        tracing::warn!(error = %e, attempt = attempt + 1, "Shard range request failed, retrying");
                        tokio::time::sleep(std::time::Duration::from_secs(
                            http_retry_delays[attempt as usize],
                        ))
                        .await;
                        continue;
                    }
                    return Err(format!("Shard download failed after retries: {e}"));
                }
            }
        }
        let resp = resp.ok_or("Shard download failed: no response")?;

        if resp.status().as_u16() != 206 && !resp.status().is_success() {
            return Err(format!("Shard download returned {}", resp.status()));
        }

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            let data = chunk.map_err(|e| format!("Stream error: {e}"))?;
            file.write_all(&data)
                .await
                .map_err(|e| format!("Write error: {e}"))?;
            downloaded += data.len() as u64;

            if let Some(ref tx) = progress_tx {
                let _ = tx.try_send(DownloadProgress {
                    downloaded_bytes: downloaded,
                    total_bytes: total_download_bytes,
                });
            }
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("Flush error: {e}"))?;

    // Atomic rename .tmp → .bin
    std::fs::rename(&tmp_path, &dest_path)
        .map_err(|e| format!("Failed to rename tmp to final shard file: {e}"))?;

    tracing::info!(
        shard = shard_index,
        size = downloaded,
        path = %dest_path.display(),
        "Downloaded v2 layer-aligned shard from HuggingFace"
    );

    Ok(dest_path)
}

/// Download header + specified v2 shards from HuggingFace.
///
/// Main entry point for shard-level downloads:
/// 1. Probes the remote file to get tensor metadata and layer-aligned shard layouts
/// 2. Downloads the GGUF header
/// 3. Downloads each requested shard via coalesced Range requests
/// 4. Returns the model directory and file info
pub async fn download_shards_v2(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    shard_indices: &[u32],
    info: &GgufFileInfo,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
) -> Result<std::path::PathBuf, String> {
    // Download the GGUF header
    download_gguf_header(repo_id, filename, dest_dir, info.header_size).await?;

    let total_shard_bytes: u64 = shard_indices
        .iter()
        .filter_map(|&idx| info.layouts.get(idx as usize))
        .map(|layout| layout.size_bytes)
        .sum();

    let mut cumulative_downloaded: u64 = 0;

    for &shard_idx in shard_indices {
        let layout = info.layouts.get(shard_idx as usize).ok_or_else(|| {
            format!(
                "Shard index {} out of range (max {})",
                shard_idx,
                info.layouts.len().saturating_sub(1)
            )
        })?;

        // Per-shard progress mapping to cumulative
        let (shard_tx, mut shard_rx) = tokio::sync::mpsc::channel::<DownloadProgress>(64);
        let progress_tx_clone = progress_tx.clone();
        let base = cumulative_downloaded;
        let total = total_shard_bytes;
        let progress_task = tokio::spawn(async move {
            while let Some(prog) = shard_rx.recv().await {
                if let Some(ref tx) = progress_tx_clone {
                    let _ = tx.try_send(DownloadProgress {
                        downloaded_bytes: base + prog.downloaded_bytes,
                        total_bytes: total,
                    });
                }
            }
        });

        download_shard_v2(repo_id, filename, dest_dir, layout, Some(shard_tx)).await?;
        let _ = progress_task.await;

        cumulative_downloaded += layout.size_bytes;
    }

    Ok(dest_dir.to_path_buf())
}

// ---- HF API response types ----

#[derive(Deserialize)]
struct HfRepoInfo {
    id: String,
    downloads: Option<u64>,
}

#[derive(Deserialize)]
struct HfModelDetail {
    siblings: Option<Vec<HfFileInfo>>,
}

#[derive(Deserialize, Clone)]
struct HfFileInfo {
    rfilename: String,
    size: Option<u64>,
}

/// URL-encode a string for use in query parameters.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        let mut result = String::new();
        for b in s.bytes() {
            match b {
                b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                    result.push(b as char);
                }
                _ => {
                    result.push('%');
                    result.push_str(&format!("{:02X}", b));
                }
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_url_format() {
        let url = download_url("TheBloke/Llama-2-7B-GGUF", "llama-2-7b.Q4_K_M.gguf");
        assert_eq!(
            url,
            "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
        );
    }

    #[test]
    fn urlencoding_basic() {
        assert_eq!(urlencoding::encode("hello world"), "hello%20world");
        assert_eq!(urlencoding::encode("foo+bar"), "foo%2Bbar");
        assert_eq!(urlencoding::encode("simple"), "simple");
    }

    #[test]
    fn shard_count_from_size() {
        let total_size: u64 = 4_683_074_048; // Qwen2.5-Coder-7B Q4
        let shard_size: u64 = 512 * 1024 * 1024;
        let shard_count = total_size.div_ceil(shard_size).max(1) as u32;
        assert_eq!(shard_count, 9);
    }

    #[test]
    fn coalesce_empty() {
        assert!(coalesce_byte_ranges(&[], 0).is_empty());
    }

    #[test]
    fn coalesce_no_merge() {
        // Two ranges far apart
        let ranges = vec![(0, 100), (1000, 100)];
        let merged = coalesce_byte_ranges(&ranges, 50);
        assert_eq!(merged, vec![(0, 100), (1000, 1100)]);
    }

    #[test]
    fn coalesce_adjacent() {
        // Two adjacent ranges
        let ranges = vec![(0, 100), (100, 100)];
        let merged = coalesce_byte_ranges(&ranges, 0);
        assert_eq!(merged, vec![(0, 200)]);
    }

    #[test]
    fn coalesce_with_gap() {
        // Two ranges with a small gap within max_gap
        let ranges = vec![(0, 100), (110, 100)];
        let merged = coalesce_byte_ranges(&ranges, 20);
        assert_eq!(merged, vec![(0, 210)]);

        // Same gap but max_gap is too small
        let merged = coalesce_byte_ranges(&ranges, 5);
        assert_eq!(merged, vec![(0, 100), (110, 210)]);
    }

    #[test]
    fn coalesce_unsorted() {
        // Input not sorted — should still work
        let ranges = vec![(200, 50), (0, 100), (100, 100)];
        let merged = coalesce_byte_ranges(&ranges, 0);
        assert_eq!(merged, vec![(0, 250)]);
    }

    #[test]
    fn parse_retry_after_delta_seconds() {
        assert_eq!(parse_retry_after("120"), Some(120));
        assert_eq!(parse_retry_after("0"), Some(0));
        assert_eq!(parse_retry_after("1"), Some(1));
        assert_eq!(parse_retry_after(" 60 "), Some(60));
    }

    #[test]
    fn parse_retry_after_http_date() {
        // Use a date far in the future to ensure it's always > now
        let result = parse_retry_after("Fri, 01 Jan 2100 00:00:00 GMT");
        assert!(result.is_some());
        let secs = result.unwrap();
        // Should be many years in the future
        assert!(secs > 365 * 24 * 3600);
    }

    #[test]
    fn parse_retry_after_http_date_past_returns_small() {
        // A date in the past should return at least 1 second (clamped)
        // or None since duration_since would fail
        let result = parse_retry_after("Mon, 01 Jan 2001 00:00:00 GMT");
        // Past date: duration_since(now) fails, returns None
        assert!(result.is_none());
    }

    #[test]
    fn parse_retry_after_invalid() {
        assert_eq!(parse_retry_after("not-a-number"), None);
        assert_eq!(parse_retry_after(""), None);
        assert_eq!(parse_retry_after("abc def"), None);
    }

    #[test]
    fn parse_retry_after_all_months() {
        let months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ];
        for month in months {
            let date = format!("Mon, 15 {} 2100 12:00:00 GMT", month);
            let result = parse_retry_after(&date);
            assert!(result.is_some(), "Failed to parse month: {}", month);
        }
    }

    #[test]
    fn coalesce_real_world() {
        // Simulate tensor ranges with small gaps (like GGUF alignment padding)
        let ranges = vec![
            (1000, 5000),      // tensor 1: [1000, 6000)
            (6032, 5000),      // tensor 2: [6032, 11032) — 32-byte gap
            (11064, 5000),     // tensor 3: [11064, 16064) — 32-byte gap
            (5_000_000, 5000), // tensor 4: [5000000, 5005000) — ~5MB gap
        ];
        let merged = coalesce_byte_ranges(&ranges, 4 * 1024 * 1024);
        // First 3 should merge (within 4MB gap), tensor 4 is separate (>4MB gap)
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0], (1000, 16064));
        assert_eq!(merged[1], (5_000_000, 5_005_000));
    }

    #[test]
    fn parse_http_date_basic() {
        let dt = parse_http_date("Fri, 28 Feb 2026 04:00:00 GMT");
        assert!(dt.is_some());
        // Verify it's a reasonable timestamp (after 2025)
        let since_epoch = dt
            .unwrap()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        // 2026-02-28 should be ~1772 million seconds since epoch
        assert!(since_epoch > 1_770_000_000);
        assert!(since_epoch < 1_780_000_000);
    }

    #[test]
    fn parse_http_date_invalid() {
        assert!(parse_http_date("not a date").is_none());
        assert!(parse_http_date("").is_none());
        assert!(parse_http_date("Fri, 28 Xxx 2026 04:00:00 GMT").is_none());
        // Wrong timezone
        assert!(parse_http_date("Fri, 28 Feb 2026 04:00:00 EST").is_none());
    }
}
