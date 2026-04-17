/// Tracks the HuggingFace origin of a model for re-downloading shards.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HfSource {
    pub repo_id: String,
    pub filename: String,
    /// Filename of the mmproj GGUF on HuggingFace (for VLM models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj_filename: Option<String>,
}

/// Cached result from probing a HuggingFace GGUF file.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct HfProbeInfo {
    pub repo_id: String,
    pub filename: String,
    pub shard_count: u32,
    pub total_size_bytes: u64,
    pub probed_at: chrono::DateTime<chrono::Utc>,
}
