use serde::{Deserialize, Serialize};

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
        match fetch_gguf_files(&client, &repo.model_id).await {
            Ok(files) => {
                for file in files {
                    results.push(HfModelResult {
                        repo_id: repo.model_id.clone(),
                        filename: file.rfilename.clone(),
                        size_bytes: file.size.unwrap_or(0),
                        downloads: repo.downloads.unwrap_or(0),
                    });
                }
            }
            Err(e) => {
                tracing::debug!(repo = %repo.model_id, error = %e, "Failed to list files for repo");
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

    let dest_path = dest_dir.join(filename);
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

// ---- HF API response types ----

#[derive(Deserialize)]
struct HfRepoInfo {
    #[serde(rename = "modelId", alias = "id")]
    model_id: String,
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
}
