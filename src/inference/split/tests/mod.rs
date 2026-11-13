//! Test modules for the split inference engine. Originally one ~3000-line file,
//! split into per-domain submodules:
//!
//! - `common` — shared test fixtures (model builders, tensor assertions)
//! - `core` — tensor roundtrips, sampling, KV cache, LRU, forward_batch, flash_attn,
//!   model_arch, MLP activations, and decode/prefill bench timings
//! - `gqa` — GQA attention behavior (Llama 3 / MQA / Qwen 2 ratios) plus KV cache shapes
//! - `gemma2` — Gemma-2 specific behavior (gelu, attn softcap) and real-GGUF cross-checks
//! - `moe_mla` — MoE / shared-expert / MLA / DeepSeek mixed-layer + meta parsing
//! - `llama4_glm4` — Llama 4 + GLM4 specifics: partial RoPE, NoPE skip, iRoPE, MoE
//! - `per_layer_and_shared_kv` — Gemma 4 E2B/E4B: shared-KV layers, per-layer inputs
//!
//! All test files share the `common` module which exposes `pub(super)` builders.

mod common;
mod core;
mod draft_trees;
mod drafting;
pub(crate) mod gemma2;
mod gqa;
mod image_spans;
mod kv_refresh;
mod llama4_glm4;
mod moe_mla;
mod per_layer_and_shared_kv;
mod views;

/// A small model of `num_layers` real layers on the processor, no token table
/// or head — a middle segment — for a test outside `split` (the worker's).
pub(crate) fn plain_test_model(num_layers: usize, hidden_dim: usize) -> super::SplitModel {
    common::make_test_split_model(num_layers, hidden_dim)
}
