use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct HfRepoInfo {
    pub id: String,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    /// HuggingFace's task for the repo (`text-generation`,
    /// `feature-extraction`, …); absent on many GGUF re-uploads.
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Deserialize)]
pub(super) struct HfModelDetail {
    pub siblings: Option<Vec<HfFileInfo>>,
    /// What HuggingFace read off the repo's GGUF header; absent for a repo
    /// it did not parse.
    #[serde(default)]
    pub gguf: Option<HfGgufSummary>,
}

#[derive(Deserialize)]
pub(super) struct HfGgufSummary {
    /// `general.architecture` (`qwen2`, `qwen4exp`, …).
    #[serde(default)]
    pub architecture: Option<String>,
}

#[derive(Deserialize, Clone)]
pub(super) struct HfFileInfo {
    pub rfilename: String,
    pub size: Option<u64>,
}
