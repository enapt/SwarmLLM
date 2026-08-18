//! HuggingFace API client: search, download, probe, byte-range shard fetching.

mod download;
mod private_types;
mod probe;
mod search;
mod shards;
pub mod watcher;

pub use download::{download_model, download_url};
pub use probe::{
    download_gguf_header, download_tied_output_weight, hf_failure_log_level,
    probe_failure_is_user_fixable, probe_gguf_file,
};
pub use search::{extract_quant_tag, search_gguf_models};
pub use shards::{download_shard, download_shards, parse_retry_after};
pub use watcher::{is_trusted_publisher, HfTrendingEntry, HfTrendingSnapshot, HfWatcher};

/// Size of the GGUF header probe download (16 MB).
/// Most GGUF headers are <10MB; 16MB gives margin for large vocab models.
const GGUF_HEADER_PROBE_SIZE: u64 = 16 * 1024 * 1024;
/// Gap tolerance for coalescing byte-range requests (4MB).
const BYTE_RANGE_COALESCE_GAP: u64 = 4 * 1024 * 1024;
/// HTTP connect timeout for HuggingFace API/download requests.
const HF_CONNECT_TIMEOUT_SECS: u64 = 15;
/// HTTP total timeout for metadata/probe requests.
const HF_METADATA_TIMEOUT_SECS: u64 = 120;
/// How long a shard download may go SILENT before we give up.
///
/// Was a *total* one-hour timeout, which sets an implicit minimum connection
/// speed: a 512 MB shard has to average ~145 KB/s to finish inside it, and a
/// user below that could never complete the download no matter how many times
/// they retried. Measuring silence bounds a stalled transfer without punishing
/// a slow one. Per-shard backoff already handles a source that keeps failing.
const HF_DOWNLOAD_STALL_SECS: u64 = 120;

/// Shared HTTP client for metadata/search/probe requests (short timeout).
static HF_META_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    crate::http::build_client(|b| {
        b.connect_timeout(std::time::Duration::from_secs(HF_CONNECT_TIMEOUT_SECS))
            .timeout(std::time::Duration::from_secs(HF_METADATA_TIMEOUT_SECS))
    })
});

/// Shared HTTP client for large file downloads (long timeout, connection pooling).
static HF_DOWNLOAD_CLIENT: std::sync::LazyLock<reqwest::Client> = std::sync::LazyLock::new(|| {
    crate::http::build_client(|b| {
        b.connect_timeout(std::time::Duration::from_secs(HF_CONNECT_TIMEOUT_SECS))
            .read_timeout(std::time::Duration::from_secs(HF_DOWNLOAD_STALL_SECS))
    })
});

/// SEC: Validate HuggingFace repo ID format to prevent SSRF via crafted repo_id.
pub(crate) fn validate_hf_repo_id(repo_id: &str) -> Result<(), String> {
    let parts: Vec<&str> = repo_id.split('/').collect();
    if parts.len() != 2 {
        return Err(format!("Invalid HuggingFace repo ID: {repo_id}"));
    }
    for p in &parts {
        if p.is_empty()
            || p.len() > 96
            || !p
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
        {
            return Err(format!("Invalid HuggingFace repo ID: {repo_id}"));
        }
        if *p == ".." || *p == "." {
            return Err(format!("Invalid HuggingFace repo ID: {repo_id}"));
        }
    }
    Ok(())
}

/// Apply auth + user-agent headers to a reqwest builder.
fn hf_headers(builder: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    let b = builder.header("User-Agent", "SwarmLLM/0.1");
    if let Some(token) = crate::config::hf_api_token() {
        b.header("Authorization", format!("Bearer {token}"))
    } else {
        b
    }
}

#[derive(Debug, Clone)]
pub struct HfModelResult {
    pub repo_id: String,
    pub filename: String,
    pub size_bytes: u64,
    pub downloads: u64,
    pub likes: u64,
}

#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
}

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

/// URL-encode a string for use in query parameters.
pub(super) mod urlencoding {
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
mod tests;
