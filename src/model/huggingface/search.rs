use super::private_types::{HfFileInfo, HfModelDetail, HfRepoInfo};
use super::urlencoding;
use super::{hf_headers, validate_hf_repo_id, HfModelResult, HF_META_CLIENT};

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
    let client = &*HF_META_CLIENT;

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
        match fetch_gguf_files(client, &repo.id).await {
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
