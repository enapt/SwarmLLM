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

/// A download's staging file, removed when this goes out of scope unless the
/// file `landed` (was renamed into place) — so an error, a cancel or the size
/// cap takes its partial file with it on every return path, and no sweep has to
/// guess whose it is (#419). Prior art: tempfile's `TempPath`, removed on drop
/// unless kept or persisted.
struct StagingFile {
    path: std::path::PathBuf,
    armed: bool,
}

impl StagingFile {
    fn new(path: std::path::PathBuf) -> Self {
        Self { path, armed: true }
    }

    fn landed(mut self) {
        self.armed = false;
    }
}

impl Drop for StagingFile {
    fn drop(&mut self) {
        if self.armed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Download a vision encoder whole into `model_dir/mmproj.gguf` (#422), held to
/// the size and SHA-256 HuggingFace lists for it: anything else is refused and
/// its staging file goes. Answers the file's BLAKE3, computed on the way —
/// what every copy is checked against from then on.
pub(super) async fn download_vision_encoder(
    repo_id: &str,
    listed: &super::vision_encoder::ListedEncoder,
    model_dir: &std::path::Path,
) -> Result<crate::types::Blake3Hash, String> {
    download_listed_file(
        repo_id,
        &listed.filename,
        listed.size_bytes,
        &listed.sha256,
        crate::model::manifest::MAX_VISION_ENCODER_BYTES,
        model_dir,
        crate::model::shard::MMPROJ_FILENAME,
    )
    .await
}

/// Download `filename` from `repo_id` whole into `model_dir/dest_name`, held
/// to `size_bytes` (at most `most_bytes`) and `sha256` — the figures
/// HuggingFace lists, or a pin made from them (a block drafter, #428):
/// anything else is refused and its staging file goes. Answers the file's
/// BLAKE3, computed on the way.
pub(super) async fn download_listed_file(
    repo_id: &str,
    filename: &str,
    size_bytes: u64,
    sha256_listed: &[u8; 32],
    most_bytes: u64,
    model_dir: &std::path::Path,
    dest_name: &str,
) -> Result<crate::types::Blake3Hash, String> {
    use futures::StreamExt;
    use sha2::Digest;
    use tokio::io::AsyncWriteExt;

    let expected = size_bytes;
    if expected == 0 || expected > most_bytes {
        return Err(format!("listed size {expected} bytes is out of range"));
    }
    std::fs::create_dir_all(model_dir).map_err(|e| format!("Failed to create dir: {e}"))?;
    crate::model::check_disk_space(model_dir, expected).map_err(|e| e.to_string())?;
    let url = download_url(repo_id, filename)?;
    let resp = hf_headers(HF_DOWNLOAD_CLIENT.get(&url))
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Download returned {}", resp.status()));
    }

    let dest = model_dir.join(dest_name);
    let tmp = model_dir.join(format!("{dest_name}.tmp"));
    // Declared before the file, so it drops after it.
    let staging = StagingFile::new(tmp.clone());
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .map_err(|e| format!("Failed to create temp file: {e}"))?;
    let mut sha256 = sha2::Sha256::new();
    let mut blake3 = blake3::Hasher::new();
    let mut received: u64 = 0;
    let mut stream = resp.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Download chunk error: {e}"))?;
        received = received.saturating_add(chunk.len() as u64);
        if received > expected {
            return Err(format!("longer than the {expected} bytes listed"));
        }
        sha256.update(&chunk);
        blake3.update(&chunk);
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Write error: {e}"))?;
    }
    if received != expected {
        return Err(format!("{received} bytes arrived, {expected} listed"));
    }
    let sha256: [u8; 32] = sha256.finalize().into();
    if &sha256 != sha256_listed {
        return Err("the file's SHA-256 is not the one HuggingFace lists".to_string());
    }
    // The disk's word before the rename (fsyncgate, FUTURE_WORK #217).
    file.sync_all()
        .await
        .map_err(|e| format!("Sync error: {e}"))?;
    drop(file);
    tokio::fs::rename(&tmp, &dest)
        .await
        .map_err(|e| format!("Failed to rename temp file: {e}"))?;
    staging.landed();
    Ok(*blake3.finalize().as_bytes())
}

/// Download a GGUF file from HuggingFace with progress reporting.
///
/// Supports HTTP range requests for partial/shard downloads.
/// Returns the path to the downloaded file. `cancel` is read per chunk; a set
/// flag ends the download and its staging file goes (the caller reads the
/// flag, never this error's text, to tell a cancel from a failure).
pub async fn download_model(
    repo_id: &str,
    filename: &str,
    dest_dir: &std::path::Path,
    progress_tx: Option<tokio::sync::mpsc::Sender<DownloadProgress>>,
    cancel: Option<&std::sync::atomic::AtomicBool>,
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
    // Declared before the file, so it drops after it: the file is closed
    // before this removes it.
    let staging = StagingFile::new(tmp_path.clone());
    let mut file = tokio::fs::File::create(&tmp_path)
        .await
        .map_err(|e| format!("Failed to create temp file: {e}"))?;

    use tokio::io::AsyncWriteExt;

    let mut downloaded: u64 = 0;
    let mut stream = resp.bytes_stream();

    use futures::StreamExt;

    while let Some(chunk) = stream.next().await {
        if cancel.is_some_and(|c| c.load(std::sync::atomic::Ordering::Acquire)) {
            return Err("Download cancelled".into());
        }
        let chunk = chunk.map_err(|e| format!("Download chunk error: {e}"))?;
        downloaded = downloaded.saturating_add(chunk.len() as u64);
        // SEC: enforce the file-size cap on every chunk so a response
        // missing `Content-Length` (CDN redirect, MITM, malicious repo
        // configuration) still can't stream us to disk exhaustion. The
        // pre-flight check at the top of this function only triggers
        // when `total_size > 0`; this is the streaming-time guard.
        if downloaded > MAX_MODEL_FILE_BYTES {
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
    staging.landed();

    tracing::info!(
        repo = %repo_id,
        file = %filename,
        size = downloaded,
        "HuggingFace download complete"
    );

    Ok(dest_path)
}

#[cfg(test)]
mod staging_file_tests {
    use super::StagingFile;

    /// #419: every early return of a download takes its partial file with it.
    #[test]
    fn a_staging_file_goes_unless_it_landed() {
        let dir = tempfile::tempdir().unwrap();
        let failed = dir.path().join("a.gguf.tmp");
        std::fs::write(&failed, b"partial").unwrap();
        drop(StagingFile::new(failed.clone()));
        assert!(
            !failed.exists(),
            "a download that did not land leaves no partial file"
        );

        let landed = dir.path().join("b.gguf.tmp");
        std::fs::write(&landed, b"whole").unwrap();
        StagingFile::new(landed.clone()).landed();
        assert!(landed.exists(), "a landed file is the caller's");
    }
}
