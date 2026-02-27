use serde::{Deserialize, Serialize};
use std::io::Cursor;

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

    // Parse the GGUF header to get tensor_data_offset
    let mut cursor = Cursor::new(&probe_bytes[..]);
    let ct = candle_core::quantized::gguf_file::Content::read(&mut cursor)
        .map_err(|e| format!("Failed to parse GGUF header from probe: {e}"))?;

    let header_size = ct.tensor_data_offset;
    let shard_count = total_size.div_ceil(shard_size).max(1) as u32;

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

/// Download a specific shard (byte range) of a remote GGUF file from HuggingFace.
///
/// Each shard is a slice of the GGUF file. The byte range is computed from the
/// shard index and the shard size stored in `GgufFileInfo`.
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

    let resp = client
        .get(&url)
        .header("User-Agent", "SwarmLLM/0.1")
        .header("Range", format!("bytes={range_start}-{range_end}"))
        .send()
        .await
        .map_err(|e| format!("Shard download failed: {e}"))?;

    if resp.status().as_u16() != 206 && !resp.status().is_success() {
        return Err(format!("Shard download returned {}", resp.status()));
    }

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;
    let dest_path = dest_dir.join(format!("shard_{shard_index:03}.bin"));
    let mut file = tokio::fs::File::create(&dest_path)
        .await
        .map_err(|e| format!("Failed to create shard file: {e}"))?;

    use futures::StreamExt;
    use tokio::io::AsyncWriteExt;

    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Shard chunk error: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Shard write error: {e}"))?;
        downloaded += chunk.len() as u64;

        if let Some(ref tx) = progress_tx {
            let _ = tx.try_send(DownloadProgress {
                downloaded_bytes: downloaded,
                total_bytes: expected_size,
            });
        }
    }

    file.flush()
        .await
        .map_err(|e| format!("Shard flush error: {e}"))?;

    tracing::info!(
        shard = shard_index,
        size = downloaded,
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

    // Step 3: Download each requested shard
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
        let (shard_tx, mut shard_rx) =
            tokio::sync::mpsc::channel::<DownloadProgress>(64);

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

        progress_task.abort();

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
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: GgufFileInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_size, 4_000_000_000);
        assert_eq!(parsed.header_size, 5_954_048);
        assert_eq!(parsed.shard_count, 8);
    }
}
