//! Split inference engine using candle for layer-range execution.
//!
//! This module enables true distributed inference where each node processes
//! only the transformer layers it holds, forwarding hidden-state activations
//! between nodes. Uses candle for direct tensor computation with quantized
//! GGUF weights.

mod draft;
mod entry;
mod executor;
mod gguf_meta;
pub(crate) mod hybrid;
pub(crate) mod kv_budget;
pub(crate) mod kv_cache;
mod kv_write;
mod loader;
mod model;
mod per_layer;
mod prefix_cache;
mod rope;
mod shard_reader;
#[cfg(test)]
pub(crate) mod tests;
mod token_embedding;

// Re-export types from extracted modules so external `crate::inference::split::X` paths still work.
pub(crate) use super::layers::{
    DeepSeekMeta, DeltaNetWeights, DenseBeside, ExpertFfn, ExpertScales, FfnVariant, LayerVariant,
    LayerWeights, MlaWeights, Mlp, MoeFfn, MoeGatingFunc, MoeRoutingConfig, PerLayerEmbedding,
    QMatMul, Qwen35AttnWeights, SsmState, StackedExperts,
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
pub use self::draft::{FlowGrowth, FlowStep, TreeGrowth};
pub use self::entry::BatchItem;
pub use self::entry::{trim_split_model_cache, SplitModelEntry, SplitModelKey};
pub use self::executor::SkipBridge;
pub(crate) use self::gguf_meta::tensor_byte_size;
pub use self::gguf_meta::{
    ensure_gguf_header, explain_gguf_parse_error, gguf_arch_str, read_gguf_header,
    save_gguf_header, CachedHeader, GgufTensorMeta, GgufTokenizerMeta, HeaderCache, ModelRefusal,
    SidecarSpec, TensorLocation, ROPE_FREQS_FILENAME, TIED_OUTPUT_FILENAME,
};
pub(crate) use self::gguf_meta::{
    host_resident, kv_layers, kv_sources, layer_is_recurrent, query_scale, routed_expert_bytes,
    segment_tensor, SegmentTensor, SlidingWindow, PER_LAYER_TOKEN_TABLE,
};
pub use self::kv_cache::{KvCacheStore, KvOccupancy};
pub(crate) use self::loader::card_free_for_a_load;
pub use self::model::SplitModel;
pub(crate) use self::per_layer::PerLayerInputs;
pub use self::prefix_cache::{
    compute_block_hashes, deserialize_snapshot, deserialize_snapshot_full,
    serialize_snapshot_with_block_size, snapshot_is_finite, verify_token_hash_chain, KvSnapshot,
    PrefixCache, SnapshotHome, SnapshotRoom, KV_SNAPSHOT_MAGIC, KV_SNAPSHOT_VERSION,
};
pub use self::shard_reader::{resolve_sidecars, SidecarTensor};
pub(crate) use self::token_embedding::table_supports_row_gather;

/// Context length served by default, for a model that declares at least this
/// much. A model declaring less keeps its own figure.
///
/// **8192 since 2026-08-18, up from 4096, because 4096 could not hold an
/// agentic client's opening message.** Those send their whole tool schema as a
/// system prompt before the user has said anything — one measured at ~5000
/// tokens — so the very first request failed with "Sequence length exceeds
/// model context window", advising the caller to shorten a prompt that is not
/// theirs to shorten. This project ships an MCP server, an Anthropic-compatible
/// surface and a Claude Code integration, so that was the flagship path failing
/// out of the box.
///
/// The obvious objection is memory, and it is answered in two places rather
/// than by keeping the default small:
///
/// - A KV cache grows on demand (`layers::new_kv_cache`), so raising the
///   ceiling costs nothing until a conversation actually gets long.
/// - GPU admission is capped independently at
///   `model::auto_manage::vram::ADMISSION_KV_CONTEXT`, so this raise does
///   not re-price what a card must have free to load a model at all. That
///   decoupling is deliberate: tying them would mean raising the default
///   silently pushed models off GPUs, which is the exact failure the cap was
///   added to end.
///
/// It DOES raise the CPU admission estimate, which prices the whole ceiling
/// because nothing bounds a CPU worker at runtime. That is the safe direction —
/// it refuses rather than swaps — and is why the raise is to 8192 and not
/// higher. Users needing more set `inference.max_seq_len_override`.
pub(crate) const DEFAULT_MAX_SEQ_LEN: usize = 8192;

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
/// default caps it, the way llama.cpp's `-c` defaults to a fixed figure rather
/// than to whatever the model advertises. Neither can raise the figure above what the
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

/// KV-cache budget handed to a CPU worker by the daemon at spawn
/// (`--kv-budget-bytes`), process-global like the override above. `0` = none
/// (no budget → no runtime guard, the pre-2026-08-21 behaviour). Computed by
/// `ModelProcessPool` at admission as the model's typical-context KV charge
/// plus whatever of the node's RAM budget is still uncommitted, so the cache
/// may grow into free memory and is refused — with a 503 that re-routes —
/// before it would swap.
pub static CPU_KV_BUDGET_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// How many of a segment's layers this worker should place on the graphics
/// card, when the daemon has decided on a partial split.
///
/// `0` means no partial limit — place the segment wherever the device says,
/// which is the all-or-nothing behaviour that predates hybrid offload. It is
/// not ambiguous with "zero layers on the card", because `gpu_layers = 0` sets
/// `force_cpu` and the loader never reaches a card at all.
///
/// Set once from `--gpu-layers`, the same way [`CPU_KV_BUDGET_BYTES`] is set
/// from `--kv-budget-bytes`: the daemon decides, the worker obeys. Graphics
/// memory has one owner and it is not this process.
pub static GPU_LAYER_LIMIT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The partial-split limit, or `None` when the segment is not being split.
pub(crate) fn gpu_layer_limit() -> Option<usize> {
    match GPU_LAYER_LIMIT.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// How many of a segment's card-held mixture-of-experts layers, from the
/// start, keep their routed experts in the machine's memory while the rest of
/// each layer stays on the card (FUTURE_WORK #138 (3), llama.cpp's
/// `--n-cpu-moe`; `hybrid::LayerPlacement::with_experts_on_the_processor`).
/// `0` = none. Set once from `--cpu-moe-layers`, like [`GPU_LAYER_LIMIT`].
pub static CPU_EXPERT_LAYERS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// `inference.prefill_chunk_tokens` as this worker was started with — the
/// configured ceiling of a prompt forward
/// (`prefill_pacer::prompt_chunk_ceiling`), which a sliding-window layer's
/// cache holds beside its window (`SplitModel::widest_prompt_forward`,
/// FUTURE_WORK #281). Set once in `run_worker`, like [`CPU_EXPERT_LAYERS`];
/// the setting's default until then. 0 reads a prompt in one forward.
pub static PROMPT_CHUNK_TOKENS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(128);

/// [`CPU_EXPERT_LAYERS`], or `None` when every card-held layer keeps its experts.
pub(crate) fn cpu_expert_layers() -> Option<usize> {
    match CPU_EXPERT_LAYERS.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        n => Some(n),
    }
}

/// Where ONE range's layers go — the two numbers the loader obeys
/// (`SplitLoadOptions::gpu_layers` / `cpu_expert_layers`), for a range the
/// daemon placed on its own (FUTURE_WORK #265).
///
/// `--gpu-layers` and `--cpu-moe-layers` were sized at spawn for the FIRST
/// range against the room the card had then, and the loader applied them to
/// every range the worker loaded after it, while the daemon charged each later
/// range to the card whole: a range the loader would have put mostly in RAM
/// was refused for card memory it never needed (report #037: "8 layers 40..48
/// … need about 3016 MB more"), and one the card could take whole was split
/// anyway. A later range is now placed by the daemon from the room each device
/// has, and the worker is told before its first forward
/// (`DaemonMsg::PlaceRange`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RangePlacement {
    /// The range's first N layers on the card and the rest on the processor;
    /// `None` puts every one on the card.
    pub gpu_layers: Option<usize>,
    /// Of the card-held layers, the first N keep their routed experts in the
    /// machine's memory; `None` keeps none there.
    pub cpu_expert_layers: Option<usize>,
}

impl RangePlacement {
    /// Every layer of the range on the card, its experts with it.
    pub const ON_THE_CARD: Self = Self {
        gpu_layers: None,
        cpu_expert_layers: None,
    };
}

/// The placements the daemon sent for ranges after the spawn's, by
/// `[layer_start, layer_end)`. A range with none is placed as the spawn said.
static RANGE_PLACEMENTS: std::sync::Mutex<Vec<((usize, usize), RangePlacement)>> =
    std::sync::Mutex::new(Vec::new());

/// Record where `[layer_start, layer_end)` goes, or with `None` forget it so
/// the range is placed as the spawn said. Called from the worker's message
/// loop, which handles messages in order, so it lands before the forward that
/// loads the range.
pub fn place_range(layer_start: usize, layer_end: usize, placement: Option<RangePlacement>) {
    let mut placements = RANGE_PLACEMENTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    placements.retain(|(range, _)| *range != (layer_start, layer_end));
    if let Some(placement) = placement {
        placements.push(((layer_start, layer_end), placement));
    }
}

/// Where the loader puts `[layer_start, layer_end)`: the daemon's placement
/// for that range when it sent one, else the spawn's (`--gpu-layers`,
/// `--cpu-moe-layers`). The ONE answer every load path asks.
pub(crate) fn placement_of(layer_start: usize, layer_end: usize) -> RangePlacement {
    RANGE_PLACEMENTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .iter()
        .find(|(range, _)| *range == (layer_start, layer_end))
        .map(|&(_, placement)| placement)
        .unwrap_or(RangePlacement {
            gpu_layers: gpu_layer_limit(),
            cpu_expert_layers: cpu_expert_layers(),
        })
}

/// Read the CPU worker's KV-cache budget (None when 0 — unset).
pub fn cpu_kv_budget_bytes() -> Option<u64> {
    match CPU_KV_BUDGET_BYTES.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        v => Some(v),
    }
}

/// Read the current context override (None when 0 — unset).
pub fn max_seq_len_override() -> Option<usize> {
    let v = MAX_SEQ_LEN_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if v == 0 {
        None
    } else {
        Some(v)
    }
}

#[cfg(test)]
mod range_placement_tests {
    use super::{cpu_expert_layers, gpu_layer_limit, place_range, placement_of, RangePlacement};

    /// A range the daemon placed loads as it was told; any other range — and
    /// one whose placement was withdrawn — loads as the spawn said (#265).
    /// Ranges far past any real model's, so no other test's can collide in
    /// this process-wide table.
    #[test]
    fn a_range_loads_where_the_daemon_placed_it_and_any_other_as_the_spawn_said() {
        let as_spawned = || RangePlacement {
            gpu_layers: gpu_layer_limit(),
            cpu_expert_layers: cpu_expert_layers(),
        };
        let split = RangePlacement {
            gpu_layers: None,
            cpu_expert_layers: Some(7),
        };
        place_range(9_000, 9_020, Some(split));
        assert_eq!(placement_of(9_000, 9_020), split);
        assert_eq!(
            placement_of(9_000, 9_019),
            as_spawned(),
            "only that exact range"
        );
        place_range(9_000, 9_020, Some(RangePlacement::ON_THE_CARD));
        assert_eq!(
            placement_of(9_000, 9_020),
            RangePlacement::ON_THE_CARD,
            "a later word replaces the first"
        );
        place_range(9_000, 9_020, None);
        assert_eq!(placement_of(9_000, 9_020), as_spawned());
    }
}

#[cfg(test)]
mod effective_context_tests {
    use super::{effective_context_length_with, DEFAULT_MAX_SEQ_LEN};

    /// The shipped default caps a model that declares more, the way llama.cpp's
    /// `-c` does — a 32k-capable model is served at the default unless asked
    /// otherwise. Asserted against the constant rather than a literal, so
    /// changing the default is a one-line change and not a test hunt.
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
