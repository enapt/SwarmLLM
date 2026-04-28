use std::io::Cursor;

use super::{
    download_url, hf_headers, GgufFileInfo, GGUF_HEADER_PROBE_SIZE, HF_DOWNLOAD_CLIENT,
    HF_META_CLIENT,
};

/// Run a fallible HTTP probe with the standard 3-attempt exponential backoff
/// (`NETWORK_RETRY_DELAYS = [5, 30, 120] s`). Permanent 4xx errors do not
/// retry; transient connection / 5xx / 429 errors do.
async fn retry_hf<T, F, Fut>(label: &str, mut op: F) -> Result<T, String>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<T, (String, bool /* transient */)>>,
{
    let delays = crate::config::NETWORK_RETRY_DELAYS;
    let mut last_err: Option<String> = None;
    for attempt in 0..=delays.len() {
        match op().await {
            Ok(v) => return Ok(v),
            Err((e, transient)) => {
                if !transient {
                    return Err(e);
                }
                last_err = Some(e);
                if attempt < delays.len() {
                    let delay = delays[attempt];
                    tracing::debug!(
                        label,
                        attempt = attempt + 1,
                        delay_secs = delay,
                        "HF probe transient error — retrying"
                    );
                    tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
                }
            }
        }
    }
    Err(last_err.unwrap_or_else(|| "exhausted retries".into()))
}

pub async fn probe_gguf_file(
    repo_id: &str,
    filename: &str,
    shard_size: u64,
) -> Result<GgufFileInfo, String> {
    let client = &*HF_META_CLIENT;

    let url = download_url(repo_id, filename);

    // HEAD request to get total file size — retry on transient errors.
    let total_size = retry_hf("HEAD", || async {
        let head_resp = hf_headers(client.head(&url))
            .send()
            .await
            .map_err(|e| (format!("HEAD request failed: {e}"), true))?;
        let status = head_resp.status();
        if !status.is_success() {
            // 5xx / 429 are transient; 4xx (other than 429) are permanent.
            let transient = status.is_server_error() || status.as_u16() == 429;
            return Err((format!("HEAD returned {status}"), transient));
        }
        let total_size = head_resp
            .headers()
            .get("content-length")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<u64>().ok())
            .ok_or_else(|| ("Server did not return Content-Length".to_string(), false))?;
        Ok(total_size)
    })
    .await?;

    if total_size == 0 {
        return Err("Server returned Content-Length: 0 (empty file)".to_string());
    }
    let probe_size: u64 = GGUF_HEADER_PROBE_SIZE;
    let range_end = (probe_size - 1).min(total_size - 1);

    let probe_bytes = retry_hf("Range-probe", || async {
        let probe_resp = hf_headers(client.get(&url))
            .header("Range", format!("bytes=0-{range_end}"))
            .send()
            .await
            .map_err(|e| (format!("Range probe request failed: {e}"), true))?;
        // 206 Partial Content means Range requests are supported
        let status = probe_resp.status();
        if status.as_u16() != 206 && !status.is_success() {
            let transient = status.is_server_error() || status.as_u16() == 429;
            return Err((
                format!("Range probe returned {status} (server may not support Range requests)"),
                transient,
            ));
        }
        probe_resp
            .bytes()
            .await
            .map_err(|e| (format!("Failed to read probe bytes: {e}"), true))
    })
    .await?;

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

/// Build `GgufTensorMeta` from a pre-parsed candle `Content`.
/// Thin adapter over [`GgufTensorMeta::from_content`] that maps `SwarmError`
/// to `String` for the HF-download error channel.
fn build_tensor_meta_from_content(
    ct: &candle_core::quantized::gguf_file::Content,
) -> Result<crate::inference::split::GgufTensorMeta, String> {
    crate::inference::split::GgufTensorMeta::from_content(ct).map_err(|e| e.to_string())
}

/// Download the GGUF header (metadata + tensor info table) from a remote GGUF file.
///
/// Returns the path to the saved `gguf_header.bin` file.
/// Write `data` atomically to `dest_dir/filename` by staging to a `.tmp`
/// sibling and renaming into place. Runs the blocking I/O on a dedicated
/// tokio worker thread so the async runtime isn't stalled.
async fn atomic_write_blocking(
    dest_dir: std::path::PathBuf,
    filename: &'static str,
    data: Vec<u8>,
) -> Result<std::path::PathBuf, String> {
    let filename = filename.to_string();
    tokio::task::spawn_blocking(move || -> Result<std::path::PathBuf, String> {
        std::fs::create_dir_all(&dest_dir).map_err(|e| format!("Failed to create dir: {e}"))?;
        let dest_path = dest_dir.join(&filename);
        let tmp_path = dest_dir.join(format!("{filename}.tmp"));
        std::fs::write(&tmp_path, &data)
            .map_err(|e| format!("Failed to write {filename}.tmp: {e}"))?;
        std::fs::rename(&tmp_path, &dest_path)
            .map_err(|e| format!("Failed to rename {filename}: {e}"))?;
        Ok(dest_path)
    })
    .await
    .map_err(|e| format!("spawn_blocking: {e}"))?
}

pub async fn download_gguf_header(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    header_size: u64,
) -> Result<std::path::PathBuf, String> {
    let client = &*HF_DOWNLOAD_CLIENT;

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

    let dest_path = atomic_write_blocking(
        dest_dir.to_path_buf(),
        crate::model::shard::HEADER_FILENAME,
        header_bytes.to_vec(),
    )
    .await?;

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

    let client = &*HF_DOWNLOAD_CLIENT;

    let url = download_url(repo_id, filename);
    if size == 0 {
        return Err("token_embd.weight has zero size — cannot download tied output weight".into());
    }
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

    let dest_path = atomic_write_blocking(
        dest_dir.to_path_buf(),
        "tied_output_weight.bin",
        data.to_vec(),
    )
    .await?;

    tracing::info!(
        size = data.len(),
        path = %dest_path.display(),
        "Downloaded tied output weight from HuggingFace"
    );

    Ok(Some(dest_path))
}
