use super::{hf_headers, validate_hf_repo_id, DownloadProgress, HF_DOWNLOAD_CLIENT};

/// SEC: cap mmproj / model file size at 16 GiB. Without this, a misconfigured
/// or hostile HF repo could advertise a multi-hundred-GB file and stream it
/// until disk is exhausted (`download_shard` has the disk preflight;
/// `download_model` previously did not). 16 GiB covers every real mmproj
/// (largest seen: ~3 GB for vision encoders on 70B models) plus generous
/// headroom; anything larger is a misconfigured sentinel and refused.
const MAX_MODEL_FILE_BYTES: u64 = 16 * 1024 * 1024 * 1024;

/// Validate an HF filename (the `rfilename` from the API). Allows only
/// safe characters and rejects path traversal / NUL / control chars.
/// Without this, an attacker who controls a typosquat HF repo could supply
/// `rfilename: "../../evil.gguf"` (passes the `.ends_with(".gguf")` check
/// in search) and the URL `https://huggingface.co/org/name/resolve/main/../../evil.gguf`
/// gets handed to the CDN, which on redirect resolves the `..` segments
/// and serves a different file. Also reject percent-encoded traversal
/// (`%2F%2F`) which would survive a URL-percent-decoder on the CDN side.
fn validate_hf_filename(filename: &str) -> Result<(), String> {
    if filename.is_empty() || filename.len() > 512 {
        return Err(format!("filename length {} out of range", filename.len()));
    }
    if filename.contains('\0')
        || filename.contains("..")
        || filename.starts_with('/')
        || filename.starts_with('\\')
    {
        return Err(format!("filename '{filename}' contains unsafe components"));
    }
    // Reject percent-encoded slashes — these survive raw insertion into a
    // URL path and are decoded by the upstream server, enabling traversal.
    let lower = filename.to_ascii_lowercase();
    if lower.contains("%2f") || lower.contains("%5c") {
        return Err(format!(
            "filename '{filename}' contains percent-encoded path separator"
        ));
    }
    if !filename
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '/'))
    {
        return Err(format!(
            "filename '{filename}' contains characters outside [a-zA-Z0-9._\\-/]"
        ));
    }
    Ok(())
}

pub fn download_url(repo_id: &str, filename: &str) -> Result<String, String> {
    // SEC: validate both repo_id AND filename. Returning a "dummy" URL on
    // failure (the previous behavior) issued a real HTTP request to
    // huggingface.co/INVALID_REPO/... — wasting bandwidth and embedding
    // attacker-controlled `filename` in the URL. Now: hard error to caller.
    validate_hf_repo_id(repo_id).map_err(|e| format!("invalid repo_id: {e}"))?;
    validate_hf_filename(filename)?;
    Ok(format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    ))
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

    let url = download_url(repo_id, filename)?;

    let resp = hf_headers(client.get(&url))
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("Download returned {}", resp.status()));
    }

    let total_size = resp.content_length().unwrap_or(0);

    if total_size > MAX_MODEL_FILE_BYTES {
        return Err(format!(
            "HuggingFace file size {total_size} bytes exceeds cap {MAX_MODEL_FILE_BYTES}"
        ));
    }
    if total_size > 0 {
        crate::model::check_disk_space(dest_dir, total_size).map_err(|e| e.to_string())?;
    }

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
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        // SEC: enforce the file-size cap on every chunk so a response
        // missing `Content-Length` (CDN redirect, MITM, malicious repo
        // configuration) still can't stream us to disk exhaustion. The
        // pre-flight check at the top of this function only triggers
        // when `total_size > 0`; this is the streaming-time guard.
        if downloaded > MAX_MODEL_FILE_BYTES {
            drop(file);
            let _ = tokio::fs::remove_file(&tmp_path).await;
            return Err(format!(
                "Download exceeded size cap {MAX_MODEL_FILE_BYTES} (at {downloaded} bytes) — aborting"
            ));
        }
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;

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
