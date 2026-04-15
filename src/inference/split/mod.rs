//! Split inference engine using candle for layer-range execution.
//!
//! This module enables true distributed inference where each node processes
//! only the transformer layers it holds, forwarding hidden-state activations
//! between nodes. Uses candle for direct tensor computation with quantized
//! GGUF weights.

mod entry;
mod executor;
mod gguf_meta;
mod kv_cache;
mod loader;
mod model;
mod rope;
mod shard_reader;
#[cfg(test)]
mod tests;

// Re-export types from extracted modules so external `crate::inference::split::X` paths still work.
pub(crate) use super::layers::{
    DeepSeekMeta, DeltaNetWeights, FfnVariant, LayerVariant, LayerWeights, MlaWeights, Mlp, MoeFfn,
    QMatMul, Qwen35AttnWeights, SsmState,
};
pub(crate) use super::model_arch::Activation;
pub use super::model_arch::ModelArch;
pub use super::shard_layout::{
    available_layer_ranges_from_manifest, compute_layer_shard_layouts, LayerShardLayout,
};
pub use super::tensor_util::{
    bytes_to_tensor, raw_f32_to_tensor_bytes, sample_token, sample_token_with_params,
    tensor_bytes_add, tensor_to_bytes, tensor_to_raw_f32,
};
pub use super::tokenizer::{BpeTokenizer, SplitTokenizer, SpmTokenizer};

// Re-export from submodules so that `crate::inference::split::SplitModel` etc. continue to work.
#[cfg(test)]
pub use self::entry::BatchItem;
pub use self::entry::{evict_split_models_lru, SplitModelEntry, SplitModelKey};
pub use self::gguf_meta::{
    ensure_gguf_header, save_gguf_header, GgufTensorMeta, GgufTokenizerMeta, TensorLocation,
};
pub use self::kv_cache::KvCacheStore;
pub use self::model::SplitModel;

pub(crate) const DEFAULT_MAX_SEQ_LEN: usize = 4096;
