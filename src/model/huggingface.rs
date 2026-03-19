use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::time::SystemTime;

/// Size of the GGUF header probe download (16 MB).
/// Most GGUF headers are <10MB; 16MB gives margin for large vocab models.
const GGUF_HEADER_PROBE_SIZE: u64 = 16 * 1024 * 1024;
/// Gap tolerance for coalescing byte-range requests (4MB).
const BYTE_RANGE_COALESCE_GAP: u64 = 4 * 1024 * 1024;

/// SEC: Validate HuggingFace repo ID format to prevent SSRF via crafted repo_id
/// from gossip (e.g., "../../internal-service"). Only allows `owner/repo-name`.
fn validate_hf_repo_id(repo_id: &str) -> Result<(), String> {
    let re_valid = |s: &str| {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
    };
    let parts: Vec<&str> = repo_id.split('/').collect();
    if parts.len() != 2 || !re_valid(parts[0]) || !re_valid(parts[1]) {
        return Err(format!("Invalid HuggingFace repo ID: {repo_id}"));
    }
    if parts[0] == ".." || parts[1] == ".." || parts[0] == "." || parts[1] == "." {
        return Err(format!("Invalid HuggingFace repo ID: {repo_id}"));
    }
    Ok(())
}

/// Read HuggingFace API token from `HF_TOKEN` env var (or `HUGGING_FACE_HUB_TOKEN` fallback).
fn hf_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty())
}

/// Apply auth + user-agent headers to a reqwest builder.
fn hf_headers(builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    let b = builder.header("User-Agent", "SwarmLLM/0.1");
    if let Some(token) = hf_token() {
        b.header("Authorization", format!("Bearer {token}"))
    } else {
        b
    }
}

/// A GGUF model file discovered on HuggingFace.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HfModelResult {
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub downloads: u64,
    pub likes: u64,
}

/// Extract quantization tag from a GGUF filename (e.g. "Q4_K_M", "Q8_0", "F16", "IQ4_XS").
pub fn extract_quant_tag(filename: &str) -> Option<String> {
    // Common GGUF quant patterns: Q4_K_M, Q8_0, Q5_K_S, Q6_K, IQ4_XS, F16, F32, BF16
    let stem = filename.trim_end_matches(".gguf");
    // Split by common delimiters and look for known quant patterns
    for part in stem.split(&['-', '.', '_'][..]) {
        let up = part.to_uppercase();
        // Match direct hits: F16, F32, BF16
        if matches!(up.as_str(), "F16" | "F32" | "BF16") {
            return Some(up);
        }
    }
    // Try regex-like pattern matching for Q/IQ patterns with underscore-joined parts
    // e.g. "model-Q4_K_M.gguf" or "model.Q8_0.gguf"
    for segment in stem.split(&['-', '.'][..]) {
        let up = segment.to_uppercase();
        if (up.starts_with('Q') || up.starts_with("IQ"))
            && up.len() >= 3
            && up
                .chars()
                .nth(if up.starts_with("IQ") { 2 } else { 1 })
                .is_some_and(|c| c.is_ascii_digit())
        {
            return Some(up);
        }
    }
    None
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

    let resp = hf_headers(client.get(&url))
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
                        likes: repo.likes.unwrap_or(0),
                    });
                }
            }
            Err(e) => {
                tracing::debug!(repo = %repo.id, error = %e, "Failed to list files for repo");
            }
        }
    }

    tracing::debug!(
        query,
        repos_count = repos.len(),
        gguf_files_found = results.len(),
        "DIAG: search_gguf_models complete"
    );

    Ok(results)
}

/// Fetch the list of GGUF files in a HuggingFace repo.
async fn fetch_gguf_files(
    client: &reqwest::Client,
    repo_id: &str,
) -> Result<Vec<HfFileInfo>, String> {
    validate_hf_repo_id(repo_id)?;
    let url = format!("https://huggingface.co/api/models/{repo_id}?blobs=true");

    let resp = hf_headers(client.get(&url))
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

    tracing::debug!(
        repo_id,
        file_count = gguf_files.len(),
        "DIAG: fetch_gguf_files complete"
    );

    Ok(gguf_files)
}

/// Get the direct download URL for a file in a HuggingFace repo.
pub fn download_url(repo_id: &str, filename: &str) -> String {
    // SEC: validate repo_id to prevent SSRF. If invalid, return a safe dummy URL
    // that will fail on download. Callers should validate before reaching this point.
    if validate_hf_repo_id(repo_id).is_err() {
        return format!("https://huggingface.co/INVALID_REPO/resolve/main/{filename}");
    }
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

    let resp = hf_headers(client.get(&url))
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Download returned {}", resp.status()));
    }

    let total_size = resp.content_length().unwrap_or(0);

    // Create destination directory
    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;

    // SECURITY: Reject null bytes and strip directory components to prevent path traversal
    if filename.contains('\0') {
        return Err("Filename contains null byte".into());
    }
    let safe_filename = std::path::Path::new(filename)
        .file_name()
        .unwrap_or_else(|| std::ffi::OsStr::new("model.gguf"));
    let dest_path = dest_dir.join(safe_filename);
    let tmp_path = dest_path.with_extension("gguf.tmp");
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create temp file: {e}"))?;

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
    drop(file);

    // Atomic rename: tmp → final (prevents partial downloads from blocking re-downloads)
    tokio::fs::rename(&tmp_path, &dest_path)
        .await
        .map_err(|e| format!("Failed to rename temp file: {e}"))?;

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
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);

    // HEAD request to get total file size
    let head_resp = hf_headers(client.head(&url))
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

    if total_size == 0 {
        return Err("Server returned Content-Length: 0 (empty file)".to_string());
    }
    let probe_size: u64 = GGUF_HEADER_PROBE_SIZE;
    let range_end = (probe_size - 1).min(total_size - 1);

    let probe_resp = hf_headers(client.get(&url))
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
    if head_count == 0 {
        return Err("attention.head_count is zero".into());
    }
    let head_count_kv = md_get("attention.head_count_kv")?
        .to_u32()
        .map_err(|e| e.to_string())? as usize;
    let block_count = md_get("block_count")?.to_u32().map_err(|e| e.to_string())? as usize;
    let embedding_length = md_get("embedding_length")?
        .to_u32()
        .map_err(|e| e.to_string())? as usize;
    // head_dim: prefer attention.key_length (Qwen3 uses 128 vs embed/heads=64)
    let head_dim_actual = ct
        .metadata
        .get(&format!("{arch}.attention.key_length"))
        .and_then(|v| v.to_u32().ok())
        .map(|v| v as usize)
        .unwrap_or(embedding_length / head_count);
    let rope_dim = md_get("rope.dimension_count")
        .and_then(|v| v.to_u32().map_err(|e| e.to_string()))
        .unwrap_or(head_dim_actual as u32) as usize;
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
        // SEC: Use checked arithmetic to prevent integer overflow from crafted GGUF headers.
        // A malicious GGUF with huge elem_count can wrap around in release mode.
        // Cap elem_count to 2^40 (~1 trillion) — no legitimate tensor exceeds this.
        let block_size = info.ggml_dtype.block_size();
        let elem_count = info.shape.elem_count();
        const MAX_ELEM_COUNT: usize = 1 << 40;
        let size = if block_size == 0 || elem_count > MAX_ELEM_COUNT {
            0u64
        } else {
            info.ggml_dtype
                .type_size()
                .checked_mul(elem_count)
                .map(|v| (v / block_size) as u64)
                .unwrap_or(0)
        };
        tensors.insert(
            name.clone(),
            crate::inference::split::TensorLocation {
                offset: info.offset,
                size,
            },
        );
    }

    // Read expert count for MoE models (DeepSeek-V2/V3)
    let expert_count = ct
        .metadata
        .get(&format!("{arch}.expert_count"))
        .and_then(|v| v.to_u32().ok())
        .unwrap_or(0) as usize;

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
        expert_count,
        architecture: arch.clone(),
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
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);
    if header_size == 0 {
        return Err("header_size is 0 — invalid GGUF metadata".to_string());
    }
    let range_end = header_size - 1;

    let resp = hf_headers(client.get(&url))
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
    let tmp_path = dest_dir.join("gguf_header.bin.tmp");
    std::fs::write(&tmp_path, &header_bytes)
        .map_err(|e| format!("Failed to write gguf_header.bin.tmp: {e}"))?;
    std::fs::rename(&tmp_path, &dest_path)
        .map_err(|e| format!("Failed to rename gguf_header.bin: {e}"))?;

    tracing::info!(
        size = header_bytes.len(),
        path = %dest_path.display(),
        "Downloaded GGUF header from HuggingFace"
    );

    Ok(dest_path)
}

/// Download `token_embd.weight` for weight-tied models (no separate `output.weight`).
///
/// Weight-tied models reuse the embedding table as the output head. When shards are
/// distributed across nodes, the last node needs this tensor but may not have shard 0.
/// This downloads the tensor data separately and saves it as `tied_output_weight.bin`.
///
/// Returns `Ok(path)` if downloaded, `Ok(None)` if the model has a separate output.weight.
pub async fn download_tied_output_weight(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    tensor_meta: &crate::inference::split::GgufTensorMeta,
) -> Result<Option<std::path::PathBuf>, String> {
    // Check if model is weight-tied: has token_embd.weight but no output.weight
    let has_output = tensor_meta.tensors.contains_key("output.weight");
    let embd = tensor_meta.tensors.get("token_embd.weight");

    if has_output || embd.is_none() {
        return Ok(None); // Not weight-tied, or no embedding — nothing to do
    }

    // Safety: guarded by `embd.is_none()` return above
    let embd_loc = embd.expect("embd checked non-None above");
    let abs_offset = tensor_meta.tensor_data_offset + embd_loc.offset;
    let size = embd_loc.size;

    tracing::info!(
        repo = %repo_id,
        size_mb = size / (1024 * 1024),
        "Downloading tied output weight (token_embd.weight) for weight-tied model"
    );

    let client = reqwest::Client::builder()
        .connect_timeout(std::time::Duration::from_secs(15))
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;

    let url = download_url(repo_id, filename);
    let range_end = abs_offset + size - 1;

    let resp = hf_headers(client.get(&url))
        .header("Range", format!("bytes={abs_offset}-{range_end}"))
        .send()
        .await
        .map_err(|e| format!("Tied output weight download failed: {e}"))?;

    if resp.status().as_u16() != 206 && !resp.status().is_success() {
        return Err(format!(
            "Tied output weight download returned {}",
            resp.status()
        ));
    }

    let data = resp
        .bytes()
        .await
        .map_err(|e| format!("Failed to read tied output weight bytes: {e}"))?;

    std::fs::create_dir_all(dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;
    let dest_path = dest_dir.join("tied_output_weight.bin");
    let tmp_path = dest_dir.join("tied_output_weight.bin.tmp");
    std::fs::write(&tmp_path, &data)
        .map_err(|e| format!("Failed to write tied_output_weight.bin.tmp: {e}"))?;
    std::fs::rename(&tmp_path, &dest_path)
        .map_err(|e| format!("Failed to rename tied_output_weight.bin: {e}"))?;

    tracing::info!(
        size = data.len(),
        path = %dest_path.display(),
        "Downloaded tied output weight from HuggingFace"
    );

    Ok(Some(dest_path))
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
        // Safety: merged always has at least one element (seeded above)
        let last = merged.last_mut().expect("merged is non-empty");
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
    cancel_flag: Option<&std::sync::atomic::AtomicBool>,
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
    let coalesced = coalesce_byte_ranges(&tensor_ranges, BYTE_RANGE_COALESCE_GAP);
    let total_download_bytes: u64 = coalesced.iter().map(|(s, e)| e - s).sum();
    let expected_tensor_bytes: u64 = layout.tensors.iter().map(|(_, _, sz)| sz).sum();

    tracing::info!(
        shard = shard_index,
        tensors = layout.tensors.len(),
        ranges = coalesced.len(),
        download_bytes = total_download_bytes,
        tensor_bytes = expected_tensor_bytes,
        "Starting shard download"
    );

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
            let result = hf_headers(client.get(&url))
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

        // Buffer the coalesced range so we can extract only tensor data (skip gap bytes).
        // Coalesced ranges merge nearby tensors with up to 4MB gaps between them.
        // Writing the raw HTTP response would include those gap bytes, causing shard_offset
        // mismatches in ShardReader.
        use futures::StreamExt;
        let mut range_buf: Vec<u8> = Vec::new();
        let mut stream = resp.bytes_stream();
        while let Some(chunk) = stream.next().await {
            // Check cancel flag every chunk
            if let Some(flag) = cancel_flag {
                if flag.load(std::sync::atomic::Ordering::Acquire) {
                    // Clean up tmp file
                    drop(file);
                    let _ = tokio::fs::remove_file(&tmp_path).await;
                    return Err("Download cancelled".to_string());
                }
            }
            let data = chunk.map_err(|e| format!("Stream error: {e}"))?;
            range_buf.extend_from_slice(&data);
            downloaded += data.len() as u64;

            if let Some(ref tx) = progress_tx {
                let _ = tx.try_send(DownloadProgress {
                    downloaded_bytes: downloaded,
                    total_bytes: total_download_bytes,
                });
            }
        }

        // Verify we received the expected number of bytes for this range
        let expected_range_bytes = *range_end - *range_start;
        if range_buf.len() as u64 != expected_range_bytes {
            tracing::warn!(
                shard = shard_index,
                expected = expected_range_bytes,
                received = range_buf.len(),
                range_start = range_start,
                range_end = range_end,
                "Coalesced range received incomplete data, retrying"
            );
            // Retry this range once — adjust progress counter for the failed bytes
            downloaded = downloaded.saturating_sub(range_buf.len() as u64);
            range_buf.clear();
            let retry_resp = hf_headers(client.get(&url))
                .header("Range", format!("bytes={}-{}", range_start, range_end - 1))
                .send()
                .await
                .map_err(|e| format!("Range retry failed: {e}"))?;
            if !retry_resp.status().is_success() {
                return Err(format!(
                    "Range retry returned HTTP {} for shard {}",
                    retry_resp.status(),
                    shard_index
                ));
            }
            let mut stream = retry_resp.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let data = chunk.map_err(|e| format!("Stream error on retry: {e}"))?;
                range_buf.extend_from_slice(&data);
                downloaded += data.len() as u64;
            }
            if range_buf.len() as u64 != expected_range_bytes {
                return Err(format!(
                    "Shard {} range {}-{}: expected {} bytes but got {} after retry",
                    shard_index,
                    range_start,
                    range_end,
                    expected_range_bytes,
                    range_buf.len()
                ));
            }
        }

        // Extract only tensor data from the buffered range, skipping inter-tensor gaps
        let mut tensors_extracted = 0u32;
        let mut bytes_written = 0u64;
        for (name, tensor_offset, tensor_size) in &layout.tensors {
            if *tensor_offset >= *range_start && *tensor_offset + *tensor_size <= *range_end {
                let buf_offset = (*tensor_offset - *range_start) as usize;
                let buf_end = buf_offset + *tensor_size as usize;
                if buf_end <= range_buf.len() {
                    file.write_all(&range_buf[buf_offset..buf_end])
                        .await
                        .map_err(|e| format!("Write error: {e}"))?;
                    tensors_extracted += 1;
                    bytes_written += *tensor_size;
                } else {
                    return Err(format!(
                        "Shard {}: tensor {} buffer overflow (need {} bytes, have {})",
                        shard_index,
                        name,
                        buf_end,
                        range_buf.len()
                    ));
                }
            }
        }
        tracing::debug!(
            shard = shard_index,
            tensors_extracted,
            bytes_written,
            "Range extraction complete"
        );
    }

    file.flush()
        .await
        .map_err(|e| format!("Flush error: {e}"))?;
    drop(file);

    // Verify file size matches expected tensor data total
    let expected_size: u64 = layout.tensors.iter().map(|(_, _, sz)| sz).sum();
    let actual_size = tokio::fs::metadata(&tmp_path)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    if actual_size != expected_size {
        // Clean up the incomplete file
        let _ = tokio::fs::remove_file(&tmp_path).await;
        return Err(format!(
            "Shard {} size mismatch: expected {} bytes (from {} tensors) but wrote {} bytes",
            shard_index,
            expected_size,
            layout.tensors.len(),
            actual_size
        ));
    }

    // Atomic rename .tmp → .bin
    std::fs::rename(&tmp_path, &dest_path)
        .map_err(|e| format!("Failed to rename tmp to final shard file: {e}"))?;

    tracing::info!(
        shard = shard_index,
        size = actual_size,
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

    // Download tied output weight for weight-tied models (no separate output.weight)
    if let Err(e) =
        download_tied_output_weight(repo_id, filename, dest_dir, &info.tensor_meta).await
    {
        tracing::warn!(error = %e, "Tied output weight download failed (non-fatal)");
    }

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
        let progress_tx = progress_tx.clone();
        let base = cumulative_downloaded;
        let total = total_shard_bytes;
        let progress_task = tokio::spawn(async move {
            while let Some(prog) = shard_rx.recv().await {
                if let Some(ref tx) = progress_tx {
                    let _ = tx.try_send(DownloadProgress {
                        downloaded_bytes: base + prog.downloaded_bytes,
                        total_bytes: total,
                    });
                }
            }
        });

        download_shard_v2(repo_id, filename, dest_dir, layout, Some(shard_tx), None).await?;
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
    likes: Option<u64>,
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
