//! Split inference engine using candle for layer-range execution.
//!
//! This module enables true distributed inference where each node processes
//! only the transformer layers it holds, forwarding hidden-state activations
//! between nodes. Uses candle for direct tensor computation with quantized
//! GGUF weights.

mod entry;
mod executor;
mod gguf_meta;
mod kv_budget;
mod kv_cache;
mod loader;
mod model;
mod prefix_cache;
mod rope;
mod shard_reader;
#[cfg(test)]
mod tests;

// Re-export types from extracted modules so external `crate::inference::split::X` paths still work.
pub(crate) use super::layers::{
    DeepSeekMeta, DeltaNetWeights, FfnVariant, LayerVariant, LayerWeights, MlaWeights, Mlp, MoeFfn,
    MoeGatingFunc, MoeRoutingConfig, QMatMul, Qwen35AttnWeights, SsmState,
};
pub(crate) use super::model_arch::Activation;
pub use super::model_arch::ModelArch;
pub use super::shard_layout::{
    available_layer_ranges_from_manifest, compute_layer_shard_layouts, LayerShardLayout,
};
pub use super::tensor_util::{
    bytes_to_tensor, raw_f32_to_tensor_bytes, sample_token, sample_token_with_params,
    sample_token_with_params_history, tensor_bytes_add, tensor_to_bytes, tensor_to_bytes_q8_0,
    tensor_to_raw_f32,
};
pub use super::tokenizer::{BpeTokenizer, SplitTokenizer, SpmTokenizer};

// Re-export from submodules so that `crate::inference::split::SplitModel` etc. continue to work.
pub use self::entry::BatchItem;
pub use self::entry::{evict_split_models_lru, SplitModelEntry, SplitModelKey};
pub use self::gguf_meta::{
    ensure_gguf_header, gguf_arch_str, save_gguf_header, GgufTensorMeta, GgufTokenizerMeta,
    TensorLocation, TIED_OUTPUT_FILENAME,
};
pub use self::kv_cache::{KvCacheStore, KvOccupancy};
pub use self::model::SplitModel;
pub use self::prefix_cache::{
    compute_block_hashes, deserialize_snapshot, deserialize_snapshot_full,
    serialize_snapshot_with_block_size, snapshot_is_finite, verify_token_hash_chain, KvSnapshot,
    PrefixCache, KV_SNAPSHOT_MAGIC, KV_SNAPSHOT_VERSION,
};
pub use self::shard_reader::{resolve_tied_output, TiedOutputSource};

pub(crate) const DEFAULT_MAX_SEQ_LEN: usize = 4096;

/// How long a conversation this node will actually serve for a model that
/// declares `declared` tokens of context.
///
/// **The single answer to "what is this model's context here".** The rule is
/// two lines and was written out three times — the loader (which sizes the KV
/// cache and RoPE tables), the VRAM estimator (which charges for it) and now
/// `/v1/models` (which tells clients). Three copies of a number that has to
/// agree is how a node ends up charging for one context and serving another.
///
/// An explicit `inference.max_seq_len_override` wins; otherwise the shipped
/// default caps it, the way llama.cpp's `-c` defaults to 4096 rather than to
/// whatever the model advertises. Neither can raise the figure above what the
/// model itself declares.
///
/// Reported 2026-08-10: a client registered a 32k-capable model at its declared
/// length, the node served 4096, and the mismatch surfaced as "400 tokens of
/// prompt plus 4096 reserved" with no room for a prompt. The number was only
/// ever in the daemon's log, so nothing the client read could have known it —
/// which is why this now also feeds `max_model_len` on `/v1/models`.
pub fn effective_context_length(declared: usize) -> usize {
    effective_context_length_with(declared, max_seq_len_override())
}

/// [`effective_context_length`] against an explicitly supplied override.
///
/// The daemon keeps the override in its own atomic — the process-global above
/// belongs to the worker and is set at spawn — so it needs to apply the same
/// rule to a different source. Pure, so the rule is testable without either.
pub fn effective_context_length_with(declared: usize, override_cap: Option<usize>) -> usize {
    match override_cap {
        Some(cap) if cap > 0 => declared.min(cap),
        _ => declared.min(DEFAULT_MAX_SEQ_LEN),
    }
}

/// Process-global override for the GGUF `context_length` value. When non-zero,
/// the loader clamps `context_length` to this number before allocating the
/// KV cache and RoPE tables. Set at worker startup from `--max-seq-len-override`.
/// 0 = use the GGUF value verbatim.
pub static MAX_SEQ_LEN_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Read the current override (None when 0 — unset).
pub fn max_seq_len_override() -> Option<usize> {
    let v = MAX_SEQ_LEN_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if v == 0 {
        None
    } else {
        Some(v)
    }
}

#[cfg(test)]
mod effective_context_tests {
    use super::{effective_context_length_with, DEFAULT_MAX_SEQ_LEN};

    /// The shipped default caps a model that declares more, the way llama.cpp's
    /// `-c` does. A 32k-capable model is served at 4096 unless asked otherwise.
    #[test]
    fn a_long_context_model_is_capped_by_default() {
        assert_eq!(
            effective_context_length_with(32768, None),
            DEFAULT_MAX_SEQ_LEN
        );
    }

    /// A model that declares LESS than the default keeps its own figure —
    /// the cap must never invent context the model does not have.
    #[test]
    fn a_short_context_model_keeps_its_own_limit() {
        assert_eq!(effective_context_length_with(2048, None), 2048);
        assert_eq!(
            effective_context_length_with(2048, Some(32768)),
            2048,
            "and an override cannot raise it past what the model declares"
        );
    }

    /// An explicit override wins over the default, which is the documented way
    /// to serve a model's full context. Reported 2026-08-10: setting it to
    /// 32768 was what made a 32k model usable.
    #[test]
    fn an_override_replaces_the_default_cap() {
        assert_eq!(effective_context_length_with(32768, Some(32768)), 32768);
        assert_eq!(effective_context_length_with(32768, Some(8192)), 8192);
    }

    /// Zero means unset, not "no context" — the daemon stores the override in
    /// an atomic where 0 is the sentinel, and reading it as a real cap would
    /// serve every model a zero-length conversation.
    #[test]
    fn zero_means_unset_not_zero_context() {
        assert_eq!(
            effective_context_length_with(32768, Some(0)),
            DEFAULT_MAX_SEQ_LEN
        );
    }
}
