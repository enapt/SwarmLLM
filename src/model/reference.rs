//! Pinned reference models for testing a swarm.
//!
//! These exist so results from different machines can be compared. A name like
//! "Llama-3.2-3B Q4_K_M" is not one artifact — several publishers ship
//! different quantizations of it, and numbers from different quants do not
//! belong next to each other. Pinning repo + filename makes a benchmark
//! reproducible.
//!
//! This list is the single source of truth: served to the dashboard by
//! `GET /api/admin/reference-models`, read directly by the `swarmllm get-model`
//! CLI (`src/cli/get_model.rs`), and mirrored in
//! `examples/fetch_reference_model.sh`. Rationale and tier guidance live in
//! `docs/REFERENCE_MODELS.md`.
//!
//! Nothing here is ever fetched automatically. Testing the swarm is not a good
//! enough reason to spend someone's bandwidth and disk without them choosing
//! to — every acquisition path runs because a person asked for it.

use serde::Serialize;

/// A pinned model used for benchmarking and network testing.
#[derive(Debug, Clone, Serialize)]
pub struct ReferenceModel {
    /// Stable tier key. Also the i18n suffix the dashboard uses.
    pub tier: &'static str,
    /// Model id the daemon will register this under once acquired — lowercase
    /// filename with `_` normalised to `-` and the extension dropped. Lets the
    /// dashboard mark an already-held model as a reference model without
    /// re-deriving the rule.
    pub model_id: &'static str,
    pub repo_id: &'static str,
    pub filename: &'static str,
    /// Approximate download size. Shown before the user commits to it.
    pub size_mb: u64,
    /// Shard count at the default `shard_size_mb` of 512.
    pub shards: u32,
}

/// The pinned set, cheapest first.
pub const REFERENCE_MODELS: &[ReferenceModel] = &[
    ReferenceModel {
        tier: "smoke",
        model_id: "tinyllama-1.1b-chat-v1.0.q4-k-m",
        repo_id: "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
        filename: "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        size_mb: 638,
        shards: 2,
    },
    ReferenceModel {
        tier: "standard",
        model_id: "llama-3.2-3b-instruct-q4-k-m",
        repo_id: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        size_mb: 1925,
        shards: 4,
    },
    ReferenceModel {
        tier: "stress",
        model_id: "meta-llama-3.1-8b-instruct-q4-k-m",
        repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        size_mb: 4692,
        shards: 10,
    },
];

/// Whether a model id is one of the pinned reference models.
pub fn is_reference_model(model_id: &str) -> bool {
    REFERENCE_MODELS.iter().any(|m| m.model_id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiers_are_unique() {
        let mut seen = std::collections::HashSet::new();
        for m in REFERENCE_MODELS {
            assert!(seen.insert(m.tier), "duplicate tier {}", m.tier);
        }
    }

    /// `model_id` must match what the daemon derives from the filename —
    /// lowercase, `_` to `-`, extension dropped. If this drifts, the dashboard
    /// silently stops badging an acquired reference model.
    #[test]
    fn model_ids_match_the_filename_derivation() {
        for m in REFERENCE_MODELS {
            let derived = m
                .filename
                .trim_end_matches(".gguf")
                .to_lowercase()
                .replace('_', "-");
            assert_eq!(
                derived, m.model_id,
                "{} derives to {derived}, not {}",
                m.filename, m.model_id
            );
        }
    }

    #[test]
    fn is_reference_model_only_matches_pinned_ids() {
        assert!(is_reference_model("llama-3.2-3b-instruct-q4-k-m"));
        assert!(!is_reference_model("some-other-model"));
        assert!(!is_reference_model(""));
    }
}
