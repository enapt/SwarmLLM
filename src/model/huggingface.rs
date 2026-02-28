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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GgufFileInfo {
    pub total_size: u64,
    pub header_size: u64,
    pub shard_count: u32,
    pub shard_size: u64,
    /// Tensor metadata extracted from the GGUF header (v2 layer-aligned sharding).
    /// Populated during probe for use by repack_to_layer_shards.
    #[serde(skip)]
    pub tensor_meta: Option<crate::inference::split::GgufTensorMeta>,
}

/// Default shard size in bytes (512MB) — used when no config is available.
const DEFAULT_SHARD_SIZE: u64 = 512 * 1024 * 1024;

/// Probe a remote GGUF file on HuggingFace to determine its size and shard layout.
///
/// Uses a HEAD request for total size, then a Range GET for the first 16MB to parse
/// the GGUF header and extract `tensor_data_offset`.
/// Uses the default 512MB shard size. For custom shard sizes, use `probe_gguf_file_with_shard_size`.
pub async fn probe_gguf_file(repo_id: &str, filename: &str) -> Result<GgufFileInfo, String> {
    probe_gguf_file_with_shard_size(repo_id, filename, DEFAULT_SHARD_SIZE).await
}

/// Probe a remote GGUF file with a specific shard size.
pub async fn probe_gguf_file_with_shard_size(
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
    let shard_count = total_size.div_ceil(shard_size).max(1) as u32;

    // Extract tensor metadata for v2 layer-aligned sharding
    let tensor_meta = extract_tensor_meta_from_content(&ct);

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        total_size,
        header_size,
        shard_count,
        shard_size_mb = shard_size / (1024 * 1024),
        "Probed remote GGUF file"
    );

    Ok(GgufFileInfo {
        total_size,
        header_size,
        shard_count,
        shard_size,
        tensor_meta,
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

/// Extract GgufTensorMeta from a parsed GGUF Content struct.
///
/// This is used during the probe phase to capture tensor layout information
/// without needing a file path (we already have the Content from the probe bytes).
fn extract_tensor_meta_from_content(
    ct: &candle_core::quantized::gguf_file::Content,
) -> Option<crate::inference::split::GgufTensorMeta> {
    let arch = ct
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().cloned())
        .unwrap_or_else(|| "llama".to_string());

    let md_get_u32 = |suffix: &str| -> Option<u32> {
        let key = format!("{arch}.{suffix}");
        ct.metadata.get(&key).and_then(|v| v.to_u32().ok())
    };

    let head_count = md_get_u32("attention.head_count")? as usize;
    let head_count_kv = md_get_u32("attention.head_count_kv")? as usize;
    let block_count = md_get_u32("block_count")? as usize;
    let embedding_length = md_get_u32("embedding_length")? as usize;
    let rope_dim = md_get_u32("rope.dimension_count")
        .unwrap_or((embedding_length / head_count) as u32) as usize;
    let rms_norm_eps = ct
        .metadata
        .get(&format!("{arch}.attention.layer_norm_rms_epsilon"))
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(1e-5) as f64;
    let rope_freq_base = ct
        .metadata
        .get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.to_f32().ok())
        .unwrap_or(10000f32);
    let model_name = ct
        .metadata
        .get("general.name")
        .and_then(|v| v.to_string().ok().cloned());

    let mut tensors = std::collections::HashMap::new();
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

    Some(crate::inference::split::GgufTensorMeta {
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

/// Download the full GGUF file from HuggingFace with progress reporting.
///
/// Unlike `download_model` which saves with the original filename, this saves
/// to a deterministic `full.gguf` path for subsequent repacking.
pub async fn download_full_gguf(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
) -> Result<std::path::PathBuf, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(7200))
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

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;

    let tmp_path = dest_dir.join("full.gguf.tmp");
    let final_path = dest_dir.join("full.gguf");
    let mut file = tokio::fs::File::create(&tmp_path)
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

    // Atomic rename
    tokio::fs::rename(&tmp_path, &final_path)
        .await
        .map_err(|e| format!("Rename error: {e}"))?;

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        size = downloaded,
        "Full GGUF download complete for repacking"
    );

    Ok(final_path)
}

/// Download a specific shard (byte range) of a remote GGUF file from HuggingFace.
///
/// Each shard is a slice of the GGUF file. The byte range is computed from the
/// shard index and the shard size stored in `GgufFileInfo`.
///
/// Supports resuming from partial `.tmp` files. If a `.tmp` file exists from a
/// previous interrupted download, it will attempt to resume from the last byte
/// using HTTP `Range` headers, falling back to a full re-download if the server
/// doesn't support Range requests.
///
/// Connection errors during streaming are retried up to 3 times with backoff.
pub async fn download_shard(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    shard_index: u32,
    total_file_size: u64,
    shard_size: u64,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
) -> Result<std::path::PathBuf, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);

    // Shards are byte ranges of the FULL GGUF file (not just tensor data),
    // because that's how the shard hashes and the ShardReader work.
    let range_start = (shard_index as u64) * shard_size;
    let range_end = ((shard_index as u64 + 1) * shard_size - 1).min(total_file_size - 1);
    let expected_size = range_end - range_start + 1;

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;
    let dest_path = dest_dir.join(format!("shard_{shard_index:03}.bin"));
    let tmp_path = dest_dir.join(format!("shard_{shard_index:03}.bin.tmp"));

    // Check for partial download to resume from
    let existing_bytes = if tmp_path.exists() {
        let meta = std::fs::metadata(&tmp_path)
            .map_err(|e| format!("Failed to read tmp file metadata: {e}"))?;
        let len = meta.len();
        if len >= expected_size {
            // tmp file is already complete (or larger) — start fresh to be safe
            tracing::info!(
                shard = shard_index,
                "Tmp file already >= expected size, re-downloading"
            );
            0
        } else {
            tracing::info!(
                shard = shard_index,
                existing_bytes = len,
                expected_size,
                "Found partial download, will attempt resume"
            );
            len
        }
    } else {
        0
    };

    // Connection-level retry: wraps the entire request+stream cycle
    let stream_retry_delays = [2u64, 5, 10];
    let mut total_downloaded: u64 = existing_bytes;

    for stream_attempt in 0..=3u32 {
        // Calculate the actual range start accounting for already-downloaded bytes
        let actual_range_start = range_start + total_downloaded;
        if total_downloaded >= expected_size {
            break; // Already have all bytes
        }

        // HTTP-level retry for 429/503
        let mut resp = None;
        let http_retry_delays = [5u64, 30, 120];
        for attempt in 0..=3u32 {
            let result = client
                .get(&url)
                .header("User-Agent", "SwarmLLM/0.1")
                .header("Range", format!("bytes={actual_range_start}-{range_end}"))
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
                                .unwrap_or(http_retry_delays[attempt as usize]);
                            // Cap retry delay to 10 minutes to avoid indefinite waits
                            let retry_secs = retry_secs.min(600);
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

                    // If server returned 200 instead of 206, it doesn't support Range resume.
                    // We need to re-download from scratch.
                    if status == 200 && total_downloaded > 0 {
                        tracing::warn!(
                            shard = shard_index,
                            "Server returned 200 instead of 206 — Range not supported, restarting download"
                        );
                        total_downloaded = 0;
                        // Delete the tmp file since we're starting over
                        let _ = std::fs::remove_file(&tmp_path);
                    }

                    resp = Some(r);
                    break;
                }
                Err(e) => {
                    if attempt < 3 {
                        tracing::warn!(error = %e, attempt = attempt + 1, "Shard download request failed, retrying");
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

        // Open file in append mode if resuming, create mode if starting fresh
        use tokio::io::AsyncWriteExt;
        let mut file = if total_downloaded > 0 {
            tokio::fs::OpenOptions::new()
                .append(true)
                .open(&tmp_path)
                .await
                .map_err(|e| format!("Failed to open tmp file for append: {e}"))?
        } else {
            tokio::fs::File::create(&tmp_path)
                .await
                .map_err(|e| format!("Failed to create tmp file: {e}"))?
        };

        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut stream_error = false;

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(data) => {
                    file.write_all(&data)
                        .await
                        .map_err(|e| format!("Shard write error: {e}"))?;
                    total_downloaded += data.len() as u64;

                    if let Some(ref tx) = progress_tx {
                        let _ = tx.try_send(DownloadProgress {
                            downloaded_bytes: total_downloaded,
                            total_bytes: expected_size,
                        });
                    }
                }
                Err(e) => {
                    // Connection error during streaming — retry
                    tracing::warn!(
                        error = %e,
                        shard = shard_index,
                        downloaded = total_downloaded,
                        stream_attempt = stream_attempt + 1,
                        "Stream error during shard download"
                    );
                    stream_error = true;
                    break;
                }
            }
        }

        file.flush()
            .await
            .map_err(|e| format!("Shard flush error: {e}"))?;

        if !stream_error {
            // Stream completed successfully
            break;
        }

        // Stream was interrupted — retry with resume
        if stream_attempt < 3 {
            let delay = stream_retry_delays[stream_attempt as usize];
            tracing::info!(
                shard = shard_index,
                delay_secs = delay,
                downloaded = total_downloaded,
                "Retrying shard download from byte {}",
                total_downloaded
            );
            tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
        } else {
            return Err(format!(
                "Shard {} download failed after {} stream retries (got {}/{} bytes)",
                shard_index, stream_attempt, total_downloaded, expected_size
            ));
        }
    }

    // Rename .tmp → .bin atomically
    std::fs::rename(&tmp_path, &dest_path)
        .map_err(|e| format!("Failed to rename tmp to final shard file: {e}"))?;

    tracing::info!(
        shard = shard_index,
        size = total_downloaded,
        path = %dest_path.display(),
        "Downloaded shard from HuggingFace"
    );

    Ok(dest_path)
}

/// Download only the specified shards of a GGUF model from HuggingFace.
///
/// This is the main entry point for shard-level downloads. It:
/// 1. Probes the remote file to get size and GGUF header offset
/// 2. Downloads the GGUF header (~6MB)
/// 3. Downloads each requested shard via Range requests
/// 4. Returns the model directory containing header + shard files
///
/// The `shard_size_bytes` parameter controls the shard granularity. Pass `None`
/// to use the default 512MB size.
pub async fn download_shards(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    shard_indices: &[u32],
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    shard_size_bytes: Option<u64>,
) -> Result<(std::path::PathBuf, GgufFileInfo), String> {
    let shard_sz = shard_size_bytes.unwrap_or(DEFAULT_SHARD_SIZE);

    // Step 1: Probe the remote GGUF with the configured shard size
    let info = probe_gguf_file_with_shard_size(repo_id, filename, shard_sz).await?;

    // Step 2: Download the GGUF header
    download_gguf_header(repo_id, filename, dest_dir, info.header_size).await?;

    // V2 layer-aligned path: if we have tensor metadata and are downloading ALL
    // shards, download the full GGUF and repack into layer-aligned shard files.
    if let Some(ref tensor_meta) = info.tensor_meta {
        let all_shards: Vec<u32> = (0..info.shard_count).collect();
        let downloading_all = shard_indices == all_shards.as_slice()
            || (shard_indices.len() == info.shard_count as usize);

        if downloading_all {
            tracing::info!(
                repo = %repo_id,
                shard_count = info.shard_count,
                "V2 layer-aligned path: downloading full GGUF for repacking"
            );

            // Download full GGUF
            let gguf_path =
                download_full_gguf(repo_id, filename, dest_dir, progress_tx.clone()).await?;

            // Repack in a blocking task (4GB+ sequential I/O)
            let meta_clone = tensor_meta.clone();
            let sc = info.shard_count;
            let dd = dest_dir.to_path_buf();
            let gp = gguf_path.clone();
            let repack_result = tokio::task::spawn_blocking(move || {
                crate::inference::split::repack_to_layer_shards(&gp, &meta_clone, sc, &dd)
            })
            .await
            .map_err(|e| format!("Repack task panicked: {e}"))?
            .map_err(|e| format!("Repack failed: {e}"))?;

            // Clean up the full GGUF (we have the shard files now)
            if let Err(e) = std::fs::remove_file(&gguf_path) {
                tracing::warn!(error = %e, "Failed to remove temp full GGUF after repack");
            }

            tracing::info!(
                shards = repack_result.len(),
                "V2 layer-aligned repack complete"
            );

            return Ok((dest_dir.to_path_buf(), info));
        }
    }

    // V1 fallback: byte-range shard download (for partial downloads or missing tensor_meta)
    let total_shard_bytes: u64 = shard_indices
        .iter()
        .map(|&idx| {
            let start = (idx as u64) * info.shard_size;
            let end = ((idx as u64 + 1) * info.shard_size).min(info.total_size);
            end - start
        })
        .sum();

    let mut cumulative_downloaded: u64 = 0;

    for &shard_idx in shard_indices {
        if shard_idx >= info.shard_count {
            return Err(format!(
                "Shard index {} out of range (max {})",
                shard_idx,
                info.shard_count - 1
            ));
        }

        // Create a per-shard progress sender that maps to cumulative progress
        let (shard_tx, mut shard_rx) = tokio::sync::mpsc::channel::<DownloadProgress>(64);

        let progress_tx_clone = progress_tx.clone();
        let base_downloaded = cumulative_downloaded;
        let total = total_shard_bytes;
        let progress_task = tokio::spawn(async move {
            while let Some(prog) = shard_rx.recv().await {
                if let Some(ref tx) = progress_tx_clone {
                    let _ = tx.try_send(DownloadProgress {
                        downloaded_bytes: base_downloaded + prog.downloaded_bytes,
                        total_bytes: total,
                    });
                }
            }
        });

        download_shard(
            repo_id,
            filename,
            dest_dir,
            shard_idx,
            info.total_size,
            info.shard_size,
            Some(shard_tx),
        )
        .await?;

        // shard_tx was moved into download_shard and is now dropped,
        // so shard_rx.recv() will return None and the progress task exits gracefully.
        let _ = progress_task.await;

        // Update cumulative for next shard
        let start = (shard_idx as u64) * info.shard_size;
        let end = ((shard_idx as u64 + 1) * info.shard_size).min(info.total_size);
        cumulative_downloaded += end - start;
    }

    Ok((dest_dir.to_path_buf(), info))
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
    fn default_shard_size_constant() {
        assert_eq!(DEFAULT_SHARD_SIZE, 512 * 1024 * 1024);
    }

    #[test]
    fn gguf_file_info_shard_count() {
        let total_size: u64 = 4_683_074_048; // Qwen2.5-Coder-7B Q4
        let shard_count = total_size.div_ceil(DEFAULT_SHARD_SIZE).max(1) as u32;
        assert_eq!(shard_count, 9);
    }

    #[test]
    fn gguf_file_info_custom_shard_size() {
        let total_size: u64 = 4_683_074_048;
        // 256MB shards should produce more shards
        let small_shard: u64 = 256 * 1024 * 1024;
        let shard_count = total_size.div_ceil(small_shard).max(1) as u32;
        assert_eq!(shard_count, 18);
        // 1024MB shards should produce fewer shards
        let big_shard: u64 = 1024 * 1024 * 1024;
        let shard_count = total_size.div_ceil(big_shard).max(1) as u32;
        assert_eq!(shard_count, 5);
    }

    #[test]
    fn gguf_file_info_serde() {
        let info = GgufFileInfo {
            total_size: 4_000_000_000,
            header_size: 5_954_048,
            shard_count: 8,
            shard_size: DEFAULT_SHARD_SIZE,
            tensor_meta: None,
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: GgufFileInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_size, 4_000_000_000);
        assert_eq!(parsed.header_size, 5_954_048);
        assert_eq!(parsed.shard_count, 8);
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
    fn resume_byte_offset_calculation() {
        // Simulate: shard 3 of a 4GB file with 512MB shards
        let shard_index: u32 = 3;
        let shard_size: u64 = 512 * 1024 * 1024;
        let total_file_size: u64 = 4_000_000_000;

        let range_start = (shard_index as u64) * shard_size;
        let range_end = ((shard_index as u64 + 1) * shard_size - 1).min(total_file_size - 1);
        let expected_size = range_end - range_start + 1;

        assert_eq!(range_start, 1_610_612_736); // 3 * 512MB
        assert_eq!(expected_size, shard_size); // Full 512MB shard

        // If we've downloaded 100MB already, the resume range starts at range_start + 100MB
        let existing_bytes: u64 = 100 * 1024 * 1024;
        let resume_start = range_start + existing_bytes;
        assert_eq!(resume_start, range_start + existing_bytes);

        // The Range header should be: bytes={resume_start}-{range_end}
        let range_header = format!("bytes={resume_start}-{range_end}");
        assert!(range_header.starts_with("bytes="));
        assert!(range_header.contains('-'));
    }

    #[test]
    fn resume_byte_offset_last_shard() {
        // Last shard may be smaller than shard_size
        let total_file_size: u64 = 4_683_074_048; // Qwen2.5 size
        let shard_size: u64 = 512 * 1024 * 1024;
        let shard_count = total_file_size.div_ceil(shard_size) as u32;
        let last_shard = shard_count - 1; // shard 8

        let range_start = (last_shard as u64) * shard_size;
        let range_end = ((last_shard as u64 + 1) * shard_size - 1).min(total_file_size - 1);
        let expected_size = range_end - range_start + 1;

        // Last shard should be smaller than shard_size
        assert!(expected_size < shard_size);
        assert_eq!(expected_size, total_file_size - range_start);

        // Resume from halfway
        let existing = expected_size / 2;
        let resume_start = range_start + existing;
        assert!(resume_start < total_file_size);
        assert_eq!(total_file_size - resume_start, expected_size - existing);
    }

    #[test]
    fn range_header_construction_from_partial() {
        // Verify that range header is correctly constructed when resuming
        let shard_index: u32 = 0;
        let shard_size: u64 = 512 * 1024 * 1024;
        let total_file_size: u64 = 1_000_000_000;

        let range_start = (shard_index as u64) * shard_size;
        let range_end = ((shard_index as u64 + 1) * shard_size - 1).min(total_file_size - 1);
        let expected_size = range_end - range_start + 1;

        // No existing bytes — full range
        let header_full = format!("bytes={}-{}", range_start, range_end);
        assert_eq!(header_full, "bytes=0-536870911");

        // With 200MB already downloaded — resume range
        let existing: u64 = 200 * 1024 * 1024;
        let actual_start = range_start + existing;
        let header_resume = format!("bytes={}-{}", actual_start, range_end);
        assert_eq!(
            header_resume,
            format!("bytes={}-{}", existing, shard_size - 1)
        );

        // Remaining bytes should be expected_size - existing
        assert_eq!(range_end - actual_start + 1, expected_size - existing);
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
