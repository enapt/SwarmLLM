use crate::error::ApiError;

mod cancel;
mod download;
mod probe;
mod progress;
mod search;
mod shards;
mod source;

pub use cancel::cancel_download;
pub use download::{hf_download, HfDownloadRequest};
pub use probe::{hf_probe, HfProbeParams};
pub use search::{hf_search, HfSearchParams};
pub use shards::{hf_download_shards, HfShardDownloadRequest};
pub use source::hf_source;

/// Count unique peers holding shards of the given model IDs.
pub(super) fn count_unique_shard_holders(
    registry: &crate::model::registry::ModelRegistry,
    model_ids: &[crate::types::ModelId],
) -> usize {
    let mut unique = std::collections::HashSet::new();
    for (shard_id, holders) in registry.all_shard_entries() {
        if model_ids.contains(&shard_id.model_id) {
            for h in &holders {
                unique.insert(h.clone());
            }
        }
    }
    unique.len()
}

/// SEC: Validate HuggingFace repo_id format — delegates to the canonical validator.
pub(super) fn is_valid_hf_repo_id(repo_id: &str) -> bool {
    crate::model::huggingface::validate_hf_repo_id(repo_id).is_ok()
}

/// SEC: Validate HuggingFace filename format.
/// Only allows alphanumeric, hyphens, dots, underscores. Must end with .gguf.
pub(super) fn is_valid_hf_filename(filename: &str) -> bool {
    !filename.is_empty()
        && filename.len() <= 256
        && filename.ends_with(".gguf")
        && filename
            .chars()
            .all(|c| c.is_alphanumeric() || c == '-' || c == '_' || c == '.')
        && !filename.contains("..")
}

/// Convert a GGUF filename to a model ID slug.
/// Strips .gguf suffix, lowercases, replaces non-alphanumeric chars with hyphens,
/// and collapses consecutive hyphens.
pub(super) fn gguf_filename_to_model_id(filename: &str) -> String {
    filename
        .trim_end_matches(".gguf")
        .to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Validate HF repo_id and filename inputs, returning ApiError on failure.
///
/// Path-traversal safety: rejects `..`, `/` outside the single owner/repo
/// separator, and any non-allowlisted character. Callers MUST invoke this
/// before passing either string to filesystem APIs (`state.model_dir()`,
/// fs::write, etc.) — every handler in `admin_hf/` is structured this way.
pub(super) fn validate_hf_inputs(repo_id: &str, filename: &str) -> Result<(), ApiError> {
    if repo_id.is_empty() || filename.is_empty() {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "repo_id and filename are required".into(),
        )));
    }
    if !is_valid_hf_repo_id(repo_id) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invalid repo_id format. Expected: owner/repo (alphanumeric, hyphens, dots, underscores)"
                .into(),
        )));
    }
    if !is_valid_hf_filename(filename) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Invalid filename. Must be alphanumeric with hyphens, dots, underscores, ending in .gguf"
                .into(),
        )));
    }
    Ok(())
}

/// Extract EOS token IDs from a GGUF file, with architecture-specific fallbacks.
pub(super) fn extract_eos_token_ids(path: &std::path::Path, arch: &str) -> Vec<u32> {
    match crate::inference::split::GgufTokenizerMeta::from_gguf_file(path) {
        Ok(tok) => tok.eos_tokens_with_arch_fallback(arch),
        Err(_) => vec![crate::inference::pipeline::LLAMA_FALLBACK_EOS_TOKEN],
    }
}
