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
        let two_letters = up.starts_with("IQ") || up.starts_with("TQ");
        if (up.starts_with('Q') || two_letters)
            && up.len() >= 3
            && up
                .chars()
                .nth(if two_letters { 2 } else { 1 })
                .is_some_and(|c| c.is_ascii_digit())
        {
            return Some(up);
        }
        // A type no pattern above shapes (`NVFP4`), named as the refused list
        // names it, so a file of it is known by its name too.
        if crate::inference::split::UNREADABLE_GGML_TYPES
            .iter()
            .any(|&(_, name)| name == up)
        {
            return Some(up);
        }
    }
    None
}

/// Why this node could not host a HuggingFace file, judged by its NAME — the
/// one rule every path that offers a file to download asks before any header
/// is fetched: the search's recommendation (`api::admin_hf::search`), the
/// daily list of what people run (`popular::pick_file`) and the trending
/// list's size (`offered_quant`). A tester's search recommended an IQ2_XS file
/// in a repository folder that no node could take (field report 2026-11-03).
/// The header check stays the real answer — a K-quant's name can hide IQ
/// tensors (unsloth's Q2_K keeps IQ4_NL down experts) — and passes a file over
/// when it refuses one (#394). `None`: nothing in the name says it cannot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CannotHost {
    /// An `IQ*` / `TQ*` quantization ([`quant_is_unreadable`]): candle reads
    /// none of them, and the header check refuses each.
    UnreadableQuantization,
    /// One file of an upload split across several (`-00001-of-00003`): a model
    /// is cut into parts from ONE file.
    SplitAcrossFiles,
    /// A file in a folder of the repository: a download names a bare file
    /// (`api::admin_hf::is_valid_hf_filename`).
    InAFolder,
}

impl CannotHost {
    /// The reason as the search reports it to the browser, which words it.
    pub fn as_str(self) -> &'static str {
        match self {
            CannotHost::UnreadableQuantization => "unreadable_quantization",
            CannotHost::SplitAcrossFiles => "split_across_files",
            CannotHost::InAFolder => "in_a_folder",
        }
    }
}

/// [`CannotHost`] for `filename`, or `None` where its name raises no reason.
pub fn cannot_host(filename: &str) -> Option<CannotHost> {
    if filename.contains('/') || filename.contains('\\') {
        return Some(CannotHost::InAFolder);
    }
    if is_one_file_of_several(filename) {
        return Some(CannotHost::SplitAcrossFiles);
    }
    extract_quant_tag(filename)
        .filter(|tag| quant_is_unreadable(tag))
        .map(|_| CannotHost::UnreadableQuantization)
}

/// A quantization tag this engine cannot read: a type
/// `split::UNREADABLE_GGML_TYPES` lists, or any `IQ*` / `TQ*` — a file name
/// carries an IQ mix's name (`IQ2_M`, `IQ3_XS`) as often as a type's.
pub fn quant_is_unreadable(tag: &str) -> bool {
    let up = tag.to_ascii_uppercase();
    up.starts_with("IQ")
        || up.starts_with("TQ")
        || crate::inference::split::UNREADABLE_GGML_TYPES
            .iter()
            .any(|&(_, name)| name == up)
}

/// `name-00001-of-00003.gguf`: digits, `-of-`, digits.
fn is_one_file_of_several(filename: &str) -> bool {
    let lower = filename.to_ascii_lowercase();
    lower.match_indices("-of-").any(|(at, _)| {
        lower[..at].ends_with(|c: char| c.is_ascii_digit())
            && lower[at + 4..].starts_with(|c: char| c.is_ascii_digit())
    })
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
        let embedding = super::is_embedding_repo(repo.pipeline_tag.as_deref(), &repo.tags);
        match fetch_gguf_files(client, &repo.id).await {
            Ok((files, architecture)) => {
                for file in files {
                    results.push(HfModelResult {
                        repo_id: repo.id.clone(),
                        filename: file.rfilename.clone(),
                        size_bytes: file.size.unwrap_or(0),
                        downloads: repo.downloads.unwrap_or(0),
                        likes: repo.likes.unwrap_or(0),
                        embedding,
                        architecture: architecture.clone(),
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

/// Fetch the list of GGUF files in a HuggingFace repo, and the model family
/// HuggingFace read off its header (`gguf.architecture`, in the same reply).
async fn fetch_gguf_files(
    client: &reqwest::Client,
    repo_id: &str,
) -> Result<(Vec<HfFileInfo>, Option<String>), String> {
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
    let architecture = detail.gguf.and_then(|g| g.architecture);

    tracing::debug!(
        repo_id,
        file_count = gguf_files.len(),
        architecture = architecture.as_deref().unwrap_or("-"),
        "DIAG: fetch_gguf_files complete"
    );

    Ok((gguf_files, architecture))
}
