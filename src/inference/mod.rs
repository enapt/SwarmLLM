pub mod allreduce;
pub mod attn_kernel;
pub mod chat_template;
pub mod dsd_controller;
pub mod executor;
pub mod hedging;
pub mod kv_cache;
pub(crate) mod layers;
pub mod local_embedder;
pub(crate) mod model_arch;
pub mod model_worker;
pub mod ngram_lookup;
pub mod pipeline;
pub mod process_pool;
pub mod quant;
pub mod router;
pub mod sampling;
pub mod scheduler;
pub(crate) mod shard_layout;
pub mod slot_table;
pub mod speculative;
pub mod split;
pub mod swift;
pub(crate) mod tensor_util;
pub(crate) mod tokenizer;
pub mod vision;
pub mod worker_ipc;

/// Strip a trailing partial stop-string suffix from `text` in place.
///
/// Token-by-token stop-string checking only catches complete matches, so a
/// partial stop string at the very end of generation can leak into the output
/// (e.g. "<|user" when the stop is "<|user|>"). This trims at most one such
/// prefix — once a trim happens we return immediately so later stops can't
/// cascade across the already-truncated text.
pub(crate) fn trim_trailing_partial_stop(text: &mut String, stops: &[String]) {
    for stop in stops {
        for end_len in (1..stop.len()).rev() {
            let prefix = &stop[..end_len];
            if text.ends_with(prefix) {
                text.truncate(text.len() - end_len);
                return;
            }
        }
    }
}
