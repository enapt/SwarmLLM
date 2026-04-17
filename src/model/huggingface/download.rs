use super::{hf_headers, validate_hf_repo_id, DownloadProgress, HF_DOWNLOAD_CLIENT};

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
    let client = &*HF_DOWNLOAD_CLIENT;

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
