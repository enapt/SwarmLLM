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
pub use self::kv_cache::KvCacheStore;
pub use self::model::SplitModel;
pub use self::prefix_cache::{
    compute_block_hashes, deserialize_snapshot, deserialize_snapshot_full,
    serialize_snapshot_with_block_size, snapshot_is_finite, verify_token_hash_chain, KvSnapshot,
    PrefixCache, KV_SNAPSHOT_MAGIC, KV_SNAPSHOT_VERSION,
};
pub use self::shard_reader::{resolve_tied_output, TiedOutputSource};

pub(crate) const DEFAULT_MAX_SEQ_LEN: usize = 4096;

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
