//! Re-export all types from the swarmllm-types crate.
//!
//! Also re-exports extension traits for types that have methods
//! defined in the main crate.

pub use swarmllm_types::*;

/// Turn a model's human display name into its model id.
///
/// **This is the only derivation of an id from a display name.** It is what
/// `daemon::manifest::generate_and_register_local_manifest` registers, persists
/// and gossips, so every other surface — resolving a name a user typed, building
/// the model's directory path, announcing which models this node hosts — has to
/// arrive at the same string or it is looking for a model nobody published.
///
/// There used to be three derivations. Two were near-identical slugifiers that
/// diverged on any character that is neither alphanumeric nor `-`/`.`: this one
/// DELETED it while the manifest generator REPLACED it with `-`, so
/// `Model (Q4_K_M)` registered as `model-q4-k-m` and resolved as `model-q4km`.
/// Quantisation suffixes carry underscores, so that is an ordinary GGUF name,
/// not a corner case. The third "derivation" was no derivation at all — the
/// capability announcement in `health::monitor` sent the raw display name, so a
/// node that loaded a model with `--model` advertised holdings under an id no
/// peer could match to a manifest. It was therefore invisible as a holder, and
/// every other node grew a phantom `Llama 3.2 3B Instruct` entry beside the real
/// `llama-3.2-3b-instruct-q4-k-m` (gotcha #310).
///
/// The replace-and-collapse behaviour is canonical because it is the one that
/// produced the ids already on disk and in the DHT; changing it renames models.
/// Lowercase, every other character → `-`, then collapse runs and trim.
pub fn slugify_model_name(name: &str) -> String {
    name.to_lowercase()
        .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
        .split('-')
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Current time as Unix milliseconds.
#[inline]
pub fn unix_now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Current time as Unix seconds.
#[inline]
pub fn unix_now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// BLAKE3 over the concatenation of `parts`, folded to a `u32` from the first
/// four bytes (little-endian).
///
/// This is the project's deterministic-placement primitive: consistent-hash
/// ring positions, replica-to-node assignment, and fair-share seed-shard
/// selection all need the same node/model/index inputs to map to the same
/// number on every node, or peers disagree about who should hold what.
///
/// Callers pass their inputs in order; BLAKE3's `update` is streaming, so
/// `hash_parts_to_u32(&[a, b])` is byte-identical to `update(a); update(b)`.
/// Changing the byte order, the number of bytes taken, or the endianness here
/// silently re-shuffles shard placement across the entire swarm — treat this
/// function as a wire format, not an implementation detail.
#[inline]
pub fn hash_parts_to_u32(parts: &[&[u8]]) -> u32 {
    let mut hasher = blake3::Hasher::new();
    for part in parts {
        hasher.update(part);
    }
    let hash = hasher.finalize();
    let b = hash.as_bytes();
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

// Extension traits for types defined in swarmllm-types
pub use crate::identity::nickname::NicknameRecordExt;
pub use crate::model::manifest::ModelManifestExt;
pub use crate::pool::types::BlindedPoolInvitationExt;

#[cfg(test)]
mod slug_tests {
    use super::slugify_model_name;

    /// The manifest generator's algorithm, reproduced verbatim from what it did
    /// before it delegated. The ids on disk and in the DHT were made by this, so
    /// the shared helper must keep agreeing with it — a change here renames
    /// every model a user already has.
    fn manifest_algorithm(name: &str) -> String {
        name.to_lowercase()
            .replace(|c: char| !c.is_alphanumeric() && c != '-' && c != '.', "-")
            .split('-')
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("-")
    }

    #[test]
    fn the_shared_helper_still_produces_the_ids_already_on_disk() {
        for name in [
            "Llama 3.2 3B Instruct",
            "Meta Llama 3.1 8B Instruct",
            "Phi-3.5-mini-instruct",
            "TinyLlama_v1.1 Chat",
            "Qwen2.5-Coder-7B",
            "gemma-2-2b-it",
            "LLaVA v1.5 7B",
            "Model (Q4_K_M)",
        ] {
            assert_eq!(
                slugify_model_name(name),
                manifest_algorithm(name),
                "renaming an existing model: {name}"
            );
        }
    }

    /// The live failure: what a peer announced vs what its manifest said.
    #[test]
    fn a_display_name_resolves_to_the_id_its_manifest_uses() {
        assert_eq!(
            slugify_model_name("Llama 3.2 3B Instruct"),
            "llama-3.2-3b-instruct"
        );
    }

    /// The divergence that made an underscore unresolvable. The old resolver
    /// deleted the `_`, giving `model-q4km`, which no manifest ever used.
    #[test]
    fn characters_that_are_not_separators_do_not_glue_words_together() {
        assert_eq!(slugify_model_name("Model (Q4_K_M)"), "model-q4-k-m");
        assert_eq!(slugify_model_name("foo_bar baz"), "foo-bar-baz");
        assert_eq!(
            slugify_model_name("TinyLlama_v1.1 Chat"),
            "tinyllama-v1.1-chat"
        );
    }

    /// Applying it to an id must be a no-op, since resolution slugifies whatever
    /// it is handed — including a string that is already an id.
    #[test]
    fn slugifying_an_id_leaves_it_alone() {
        for id in [
            "llama-3.2-3b-instruct-q4-k-m",
            "phi-3.5-mini-instruct.q4-k-m",
            "tinyllama-1.1b-chat-v1.0.q4-k-m",
        ] {
            assert_eq!(slugify_model_name(id), id);
        }
    }

    #[test]
    fn no_leading_trailing_or_doubled_separators() {
        assert_eq!(slugify_model_name("  Spaced  Out  "), "spaced-out");
        assert_eq!(slugify_model_name("(parens)"), "parens");
    }
}
