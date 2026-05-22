//! Inference + auto-shard-management + model-storage config.
//!
//! Hosts the largest sub-config — `InferenceConfig` (gpu_layers, KV/prefix
//! caches, batching, speculative decoding, SWIFT, parallax routing,
//! distributed pipeline knobs, encryption, etc.) — plus the related
//! `AutoManageConfig` + `ModelAutoManagePolicy` (background shard
//! acquisition / pruning), `ModelConfig` (shard_size_mb +
//! shard_size_bytes/validate methods), and the public network-retry +
//! shard-size bound constants used by HF download / admin validation.

use super::default_true;
use crate::error::SwarmError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceConfig {
    #[serde(default)]
    pub default_model: String,
    #[serde(default = "default_session_timeout")]
    pub session_timeout_seconds: u64,
    /// Maximum concurrent inference requests scheduled across ALL loaded models.
    /// Note: requests for the same model are serialized at the worker IPC socket
    /// (one subprocess per ModelId, locked per request) — this setting does not
    /// enable intra-model parallelism. Use `max_batch_size` to amortize the
    /// per-model lock across grouped requests.
    #[serde(default = "default_max_concurrent")]
    pub max_concurrent_requests: u32,
    #[serde(default)]
    pub model_path: Option<PathBuf>,
    #[serde(default = "default_gpu_layers")]
    pub gpu_layers: u32,
    /// KV-cache session TTL in seconds (default 600 = 10 minutes).
    #[serde(default = "default_kv_cache_ttl")]
    pub kv_cache_ttl_secs: Option<u64>,
    /// Enable speculative decoding when a valid draft-target model pair is available.
    #[serde(default)]
    pub speculative_decoding: bool,
    /// Use a persistent libp2p bidirectional stream per pipeline session for
    /// distributed inference instead of the per-token request_response path.
    /// Off by default until validated. See `docs/plans/archive/distributed_inference_speedup.md`.
    #[serde(default)]
    pub persistent_pipeline_stream: bool,
    /// R139 Tier 4K — daemon-side STREAM-chunked activation send. When on AND
    /// the segment-boundary activation exceeds
    /// `streaming_min_activation_bytes`, the coordinator splits the activation
    /// into K = ceil(size / streaming_chunk_size_bytes) chunks, encrypts each
    /// with its own session-nonce + chunk-meta-bound AAD, and ships them
    /// sequentially. Receiver assembles by request_id before dispatching to
    /// the worker. **Default `false` until WAN-bench evidence justifies.** On
    /// LAN/loopback the activation send is already sub-millisecond — chunking
    /// adds per-chunk fixed cost with near-zero compute-send overlap; the
    /// win only materializes when wire transfer dominates encrypt time
    /// (typically <30 Mbps WAN per current research). See
    /// `docs/FUTURE_WORK.md § Tier 4K`.
    #[serde(default)]
    pub streaming_chunked_send: bool,
    /// Chunk size in bytes for `streaming_chunked_send`. Defaults to 256 KiB
    /// matching age STREAM construction guidance + TokenWeave (MLSys 2026)
    /// K=2-4 sweet spot. Set lower to widen overlap window at the cost of
    /// per-chunk encrypt/encode fixed overhead.
    #[serde(default = "default_streaming_chunk_size_bytes")]
    pub streaming_chunk_size_bytes: u32,
    /// Activation-size floor for `streaming_chunked_send`. Activations below
    /// this threshold ship as a single (un-chunked) frame regardless of the
    /// flag — chunking overhead exceeds benefit at small sizes. Default
    /// 64 KiB matches age + RFC 9771 STREAM guidance.
    #[serde(default = "default_streaming_min_activation_bytes")]
    pub streaming_min_activation_bytes: u32,
    /// TTL for incomplete chunk assemblies on the receiver side. A stuck or
    /// abandoned sender would otherwise leak `pending_activation_chunks`
    /// entries; the periodic sweep evicts assemblies whose last chunk
    /// arrived more than this many seconds ago.
    #[serde(default = "default_streaming_chunk_assembly_ttl_secs")]
    pub streaming_chunk_assembly_ttl_secs: u64,
    /// Enable speculative decoding for the distributed inference path. Requires
    /// `speculative_decoding = true` AND a loaded draft model. Off by default.
    #[serde(default)]
    pub speculative_distributed: bool,
    /// Enable continuous batching on the remote segment holder. When on,
    /// multiple concurrent decode requests for the same model are batched
    /// into a single worker-subprocess forward call, amortizing IPC +
    /// compute setup across requests. **Default on as of 2026-04-19.** Worker
    /// falls back to sequential on CPU (measured neutral-to-loss on CPU, see
    /// `docs/plans/benchmarks/round3.md`); delivers 1.34–1.55× on GPU at
    /// batch 2–8. Single-request workloads are unaffected. Set to `false`
    /// to bypass the scheduler entirely.
    #[serde(default = "default_continuous_batching")]
    pub continuous_batching: bool,
    /// Maximum number of concurrent decode requests to batch into a single
    /// worker forward. Only consulted when `continuous_batching = true`.
    #[serde(default = "default_max_decode_batch")]
    pub max_concurrent_decode_batch: u32,
    /// Time window (ms) the batch scheduler waits after the first request
    /// arrives before dispatching, to allow additional concurrent requests to
    /// coalesce. On WSL2 timer resolution is ~15 ms so anything below that
    /// effectively dispatches immediately.
    #[serde(default = "default_batch_collection_ms")]
    pub batch_collection_ms: u64,
    /// Item 7 Phase 2: Sarathi-style chunked prefill chunk size (in prompt
    /// tokens). When `continuous_batching = true`, every admitted slot
    /// advances by this many prompt tokens per decode tick before its first
    /// token is sampled — bounding the latency a long admission can impose
    /// on already-active decode slots. Smaller = lower decode interruption
    /// at the cost of more prefill ticks; larger = the opposite. 0 / 1
    /// degenerate to one-token-per-tick prefill.
    #[serde(default = "default_prefill_chunk_tokens")]
    pub prefill_chunk_tokens: u32,
    /// Item 7 Phase 4: fuse concurrent same-shape Prefilling slots into one
    /// `forward_batch` call inside `step_decode_pool`'s Phase A. Groups are
    /// formed by `(chunk_len, index_pos)`; non-matching chunks fall back to
    /// sequential forwards automatically. **Default on as of 2026-04-19.**
    /// Set to `false` to disable the grouping so Phase A always runs
    /// singleton-per-slot — useful for A/B benchmarks that want to isolate
    /// Phase 4 from Phases 1+2 (toggling `continuous_batching` would disable
    /// both). When `continuous_batching = false`, this flag has no effect
    /// because the SlotTable never activates.
    #[serde(default = "default_batched_prefill_forward")]
    pub batched_prefill_forward: bool,
    /// Number of draft tokens to propose per verification step (default: 4).
    #[serde(default = "default_speculative_gamma")]
    pub speculative_gamma: u32,
    /// SWARM-SPEC Layer 2: enable adaptive pipeline hedging — when a
    /// forward exceeds `hedge_after_factor × p99_estimate` for the
    /// (model, segment, holder) triple, dispatch a duplicate forward
    /// to the second-best holder. Whichever Response arrives first
    /// wins; the loser is cancelled. Cuts p95-p99 latency 30-50% on
    /// flaky P2P links; costs ~`hedge_max_rate` extra bandwidth.
    /// Default `false` because loopback RTT is too consistent to
    /// exceed `1.5 × p99` — needs a WAN deployment to validate the
    /// win before flipping defaults. Single-segment dispatch is fully
    /// wired (R136); multi-segment requires alt-pipeline assembly
    /// (still deferred — see docs/FUTURE_WORK.md).
    #[serde(default)]
    pub hedge_enabled: bool,
    /// Hedge when elapsed > `factor × p99_estimate`. Default 1.5.
    #[serde(default = "default_hedge_after_factor")]
    pub hedge_after_factor: f32,
    /// Max fraction of decisions that fire a hedge. Default 0.05 (5%).
    #[serde(default = "default_hedge_max_rate")]
    pub hedge_max_rate: f32,
    /// Minimum samples before EWMA is trusted. Default 20 (bumped from
    /// 5 in R136 L2 review after the warm-up period over-fired).
    #[serde(default = "default_hedge_min_samples")]
    pub hedge_min_samples: u32,
    /// SWARM-SPEC Layer 3: enable conversation-level predictive
    /// prefetch — use peer idle time between user turns to
    /// pre-compute activations for likely next-message first tokens.
    /// Cuts TTFT by ~50-200 ms on multi-turn chat workloads.
    /// Default `false`. Observability-complete dispatch shipped in
    /// R136 (records decision + emits ActivityEvent); the K-layer
    /// activation prefetch compute itself is workload-dependent and
    /// deferred (small models on fast hardware see negligible savings).
    #[serde(default)]
    pub prefetch_enabled: bool,
    /// Minimum idle ms before prefetch fires. Default 2000.
    #[serde(default = "default_prefetch_min_idle_ms")]
    pub prefetch_min_idle_ms: u64,
    /// Minimum user turns of history before prediction is trusted.
    /// Default 2.
    #[serde(default = "default_prefetch_min_turns")]
    pub prefetch_min_turns_for_prediction: u32,
    /// Max prefetch candidates per cycle. Default 3.
    #[serde(default = "default_prefetch_max_candidates")]
    pub prefetch_max_candidates: u32,
    /// SWARM-SPEC Layer 1.1: enable n-gram prompt-lookup as the first
    /// source for speculative drafts. When enabled, each spec round first
    /// tries to find a continuation by matching the recent context tail
    /// against the prompt + recent generation (zero-cost lookup). On
    /// miss, falls back to the draft-model path. Massive speedup on
    /// input-grounded workloads (Claude Code, MCP tool use, RAG, code
    /// completion) — published benchmarks show 2.4-4.2× per token.
    /// Default `true` — pure additive over existing spec, no quality
    /// regression possible (drafts are still verified by the target).
    #[serde(default = "default_ngram_lookup_enabled")]
    pub ngram_lookup_enabled: bool,
    /// Maximum n-gram size to try during prompt lookup. Falls back to
    /// smaller n if larger doesn't match. Default 4 matches HuggingFace
    /// `max_matching_ngram_size`.
    #[serde(default = "default_ngram_max_size")]
    pub ngram_max_size: u32,
    /// Number of candidate tokens to emit per n-gram match. Default 10
    /// matches HuggingFace `prompt_lookup_num_tokens`. Capped at
    /// `speculative_gamma + 1` at runtime (no point proposing more
    /// drafts than the spec wire format will verify).
    #[serde(default = "default_ngram_num_pred_tokens")]
    pub ngram_num_pred_tokens: u32,
    /// Path to a smaller draft model for speculative decoding.
    /// Must be a GGUF file. The draft model should be much smaller than the
    /// main model (ideally <1/10th parameters) and share the same vocabulary.
    #[serde(default)]
    pub draft_model_path: Option<PathBuf>,
    /// GPU layers to offload for the draft model (default: same as main model).
    #[serde(default)]
    pub draft_gpu_layers: Option<u32>,
    /// Optional shard range for split inference (e.g. "0-4").
    /// When set, the node only claims these shard indices instead of all shards.
    #[serde(default)]
    pub shard_range: Option<(u32, u32)>,
    /// Maximum number of requests to batch together for inference.
    /// Default 1 means no batching (sequential, backward-compatible).
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: u32,
    /// How long (ms) to wait for additional requests before dispatching a partial batch.
    #[serde(default = "default_batch_timeout_ms")]
    pub batch_timeout_ms: u64,
    /// Maximum GPU memory (MB) for cached split models. When exceeded, the
    /// least-recently-used models are evicted. Default: None (unlimited).
    #[serde(default)]
    pub max_split_model_memory_mb: Option<u64>,
    /// When true, KV-cache multi-turn sessions do NOT persist the `cached_prompt`
    /// field to the database — prompts stay in-memory only and are lost on restart.
    /// This prevents user prompts from being written to disk. Default: false.
    #[serde(default)]
    pub privacy_mode: bool,
    /// When true, the requesting node performs token→embedding locally before sending
    /// activations to the first pipeline segment. Remote nodes never see raw token IDs,
    /// only hidden-state activation tensors (which are harder to invert).
    /// Requires the embedding table to be available locally (auto-extracted from shard_000).
    /// Default: false.
    #[serde(default)]
    pub local_embedding_privacy: bool,
    /// When true, forces the requesting node to hold the final shard and perform
    /// token sampling locally. Combined with local_embedding_privacy (auto-enabled),
    /// this ensures no remote node ever sees plaintext — only intermediate activations.
    /// The pipeline "boomerangs" through remote nodes and returns to the requester.
    /// Requires the requester to hold both shard 0 (embedding) and the final shard (output head).
    /// Only useful for models with 3+ shards (2-shard = fully local, no distribution).
    /// Default: false.
    #[serde(default)]
    pub encrypted_pipeline: bool,
    /// Maximum peer RTT (ms) to consider for tensor parallelism AllReduce.
    /// Peers with measured latency above this threshold are excluded from TP groups.
    /// Default: 10ms (LAN-only).
    #[serde(default = "default_tp_max_latency_ms")]
    pub tp_max_latency_ms: u32,
    /// Enable cross-request prefix KV-cache on worker subprocesses. When on,
    /// the worker keeps a small LRU cache of prefill KV snapshots keyed by
    /// the prompt's token prefix and reuses them on subsequent requests
    /// that share the prefix. Covers same-session multi-turn AND
    /// cross-request reuse (same system prompt from different users).
    /// Default: true.
    #[serde(default = "default_true")]
    pub prefix_cache_enabled: bool,
    /// Maximum cached prefix snapshots retained per model on a worker.
    /// Each entry stores the full KV state up to its prefix boundary, so
    /// memory scales as `entries × prefix_tokens × hidden × layers`. Keep
    /// this low if you have many distinct models loaded. Default 16.
    #[serde(default = "default_prefix_cache_max_entries")]
    pub prefix_cache_max_entries: u32,
    /// Prompts longer than this many tokens are NOT inserted into the
    /// prefix cache (too memory-heavy). Lookups still run against the
    /// existing cache. Default 8192.
    #[serde(default = "default_prefix_cache_max_prompt_tokens")]
    pub prefix_cache_max_prompt_tokens: u32,
    /// Insertion block granularity. When a prefill completes with N tokens,
    /// snapshots are inserted at positions `block, 2*block, ..., N` so that
    /// later prompts sharing only a shorter prefix (e.g. same system prompt,
    /// different user turn) can still hit at a block boundary. Set to 0 to
    /// only insert at the full-prompt tail. Default 64.
    #[serde(default = "default_prefix_cache_block_tokens")]
    pub prefix_cache_block_tokens: u32,
    /// Minimum prefix length (in tokens) for which the cache is active.
    /// Prompts shorter than this aren't worth caching. Default 32.
    #[serde(default = "default_prefix_cache_min_tokens")]
    pub prefix_cache_min_tokens: u32,
    /// Item 8 Phase 3: minimum peer trust score to accept a cross-node
    /// prefix-KV fetch from. Peers below this threshold are skipped
    /// entirely at probe-time (no wire round-trip). On a successful
    /// fetch the bytes are still sanity-checked (no NaN/Inf) before
    /// hydration — any check failure triggers
    /// `TrustEvent::SpotCheckFail` on the sender. Default 0.5 (the
    /// DEFAULT_TRUST level for a freshly-seen peer — any peer that has
    /// misbehaved drops below and is locked out).
    #[serde(default = "default_cross_node_prefix_trust_min")]
    pub cross_node_prefix_trust_min: f32,
    /// Enable SWIFT (arxiv 2410.06916) self-speculative decoding inside
    /// `handle_generate`. The target model itself acts as its own draft by
    /// skipping a contiguous range of intermediate layers. No external draft
    /// model needed. Off by default until validated.
    #[serde(default)]
    pub swift_self_speculative: bool,
    /// Number of warmup tokens during which SWIFT runs the full target plus a
    /// rotating set of skip-pattern candidates to pick the best. After this,
    /// the winning pattern is used for the rest of the request. Default 32.
    #[serde(default = "default_swift_calibration_tokens")]
    pub swift_calibration_tokens: u32,
    /// Number of draft tokens proposed per verification round. Default 4.
    #[serde(default = "default_swift_gamma")]
    pub swift_gamma: u32,
    /// Fraction of layers to skip in the draft pass (0.0–0.95). Skip range is
    /// always a contiguous block centered in the model's middle layers — the
    /// outer layers are most sensitive to perturbation. Default 0.45 (skip ~45%).
    #[serde(default = "default_swift_skip_ratio")]
    pub swift_skip_ratio: f32,
    /// Force every attention call (prefill, decode, verify) through
    /// `standard_attention` instead of letting candle pick `cpu_flash_attention`
    /// or GPU `flash_attn` for multi-position forwards. Required for
    /// speculative paths (SWIFT, classic spec) so draft and verify produce
    /// numerically identical logits — otherwise even `skip_ratio=0` yields
    /// < 100% accept due to softmax differences. Auto-enabled while SWIFT is
    /// active; setting this manually applies it to non-SWIFT requests too
    /// (useful for fair-comparison benchmarking). Default false.
    #[serde(default)]
    pub force_standard_attn: bool,
    /// Override the GGUF-reported `context_length` when constructing the KV
    /// cache. candle pre-allocates `[B, H, max_seq_len, D]` zeros tensors per
    /// layer at first forward, so models with a 128K context (Phi-3.5) OOM
    /// instantly on small VRAM. Set this to e.g. 4096 to make those models
    /// fit at the cost of rejecting prompts longer than the override. None
    /// = use the GGUF value.
    #[serde(default)]
    pub max_seq_len_override: Option<u32>,
    /// Enable Decentralized Speculative Decoding (DSD, arxiv 2511.11733) for
    /// multi-segment distributed inference. Coordinator drafts γ tokens
    /// locally, pushes the whole γ-window through every pipeline segment in a
    /// single round trip, then accept-rejects on the returned γ+1 logits.
    /// Eliminates `(N-1)·t1·(γ-1)/γ` of inter-node round-trip latency at the
    /// cost of N·γ× the per-link payload (still small after Item 13's Q8_0).
    /// Single-segment workloads continue to use the Item 4 fast path. Off by
    /// default until the coordinator loop and adaptive γ controller land
    /// (DSD Phases 2–4 — see `docs/plans/archive/distributed_inference_speedup.md`
    /// Item 12). Worker (Phase 1) accepts the γ-token wire format already.
    #[serde(default)]
    pub decentralized_spec_decoding: bool,
    /// Quantize intermediate-segment hidden state activations to Q8_0
    /// (group-32 symmetric) before sending them to the next pipeline peer.
    /// Compresses ~3.76× vs raw f32 with negligible quality loss (PPL drift
    /// well under 1% on standard benchmarks — see
    /// `docs/plans/archive/distributed_inference_speedup.md` Item 13). Receivers
    /// auto-dispatch on the dtype tag, so enabling this on one peer does not
    /// require all peers to upgrade — uncompressed peers still send raw f32
    /// and receive correctly-dequantized inputs. Doesn't affect single-segment
    /// fast-path (Item 4) since that bypasses hidden state transfer entirely.
    ///
    /// **Default ON as of SWARM-SPEC Layer 0 (R136).** The Q8_0 implementation
    /// uses per-block (group-32) f16 scales — same algorithm as llama.cpp Q8_0
    /// weight quant. Per-block scale isolates outliers within a group so
    /// activation spikes (GLU-style FFN intermediates) don't degrade
    /// neighbouring values. Published perplexity delta vs. FP16: < 1% on
    /// standard benchmarks (Wikitext-2: 7.49 baseline). If you observe
    /// quality regression on a specific model, override to `false` via
    /// config.toml or `--no-activation-compression` CLI flag.
    #[serde(default = "default_activation_compression")]
    pub activation_compression: bool,
    /// Replace the greedy pipeline assembler with a Parallax-inspired
    /// shortest-path DP over (node, layer_range) vertices. Picks the chain
    /// minimising total `2*rtt + compute + load_penalty` rather than greedy
    /// next-hop coverage — see `docs/plans/archive/distributed_inference_speedup.md`
    /// Item 16. **Default on as of 2026-04-18.** Falls back to greedy when
    /// the DP has no valid source→sink path or candidate list is empty, so
    /// routing never regresses below the greedy baseline. Uses the same
    /// candidate signals (latency, load, region, est_tokens_per_sec) plus
    /// Phase B observed per-layer latency. Set to `false` in config to
    /// revert to pure greedy.
    #[serde(default = "default_parallax_routing")]
    pub parallax_routing: bool,
}

fn default_parallax_routing() -> bool {
    true
}

fn default_continuous_batching() -> bool {
    true
}

fn default_batched_prefill_forward() -> bool {
    true
}

fn default_tp_max_latency_ms() -> u32 {
    10
}

fn default_prefix_cache_max_entries() -> u32 {
    16
}

fn default_prefix_cache_max_prompt_tokens() -> u32 {
    8192
}

fn default_prefix_cache_block_tokens() -> u32 {
    64
}

fn default_prefix_cache_min_tokens() -> u32 {
    32
}

fn default_cross_node_prefix_trust_min() -> f32 {
    0.5
}

fn default_swift_calibration_tokens() -> u32 {
    32
}

fn default_swift_gamma() -> u32 {
    4
}

fn default_swift_skip_ratio() -> f32 {
    0.45
}

/// Configuration for automatic shard management.
///
/// When enabled, the node periodically evaluates network shard coverage
/// and downloads rarest shards for popular models — filling gaps to
/// improve overall network availability.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutoManageConfig {
    /// Master toggle for auto shard management.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum disk space (MB) the auto-manager may use for shard storage.
    /// Defaults to the global `max_disk_mb` if 0.
    #[serde(default)]
    pub max_storage_mb: u64,
    /// How often (in minutes) the auto-manager evaluates and downloads.
    #[serde(default = "default_auto_manage_interval")]
    pub interval_minutes: u32,
    /// Maximum number of shards to hold at once (0 = unlimited within disk budget).
    #[serde(default)]
    pub max_shards: u32,
    /// Override interval in seconds (for testing). Takes precedence over `interval_minutes`.
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    /// Maximum number of concurrent shard downloads (default 3).
    #[serde(default = "default_max_concurrent_downloads")]
    pub max_concurrent_downloads: usize,
    /// Default cap on auto-managed shards per model (0 = unlimited).
    /// Prevents auto-manage from downloading ALL shards of a single model.
    #[serde(default)]
    pub default_model_shard_cap: u32,
    /// Per-model auto-manage overrides keyed by model ID.
    #[serde(default)]
    pub model_policies: HashMap<String, ModelAutoManagePolicy>,
    /// Enable automatic shard pruning (removal of over-replicated shards).
    #[serde(default = "default_true")]
    pub prune_enabled: bool,
    /// Minimum number of replicas to maintain per shard across the network.
    #[serde(default = "default_min_replicas")]
    pub min_replicas: u32,
    /// Cooldown in seconds between prune actions on the same model.
    #[serde(default = "default_prune_cooldown_secs")]
    pub prune_cooldown_secs: u64,
    /// Block pruning if remaining holders have avg load above this threshold.
    #[serde(default = "default_max_holder_load_for_prune")]
    pub max_holder_load_for_prune: u32,
    /// Phase C.2 (Parallax) auto-rebalance: when enabled, the auto-manage
    /// loop runs `PipelineScheduler::allocate_offline` each cycle and
    /// biases shard acquire / prune scores toward the allocator's
    /// recommendation. Requires `PARALLAX_STABILITY_THRESHOLD` consecutive
    /// ticks of consistent signal before the bias kicks in — a single
    /// noisy tick can't flip anything. Respects all existing trust,
    /// credit, locked-shard, pin, encrypted-pipeline, and configured-range
    /// constraints (the bias is purely additive on score).
    #[serde(default = "default_true")]
    pub parallax_auto_rebalance: bool,
    /// R112: enable the background HfWatcher that polls HuggingFace's
    /// trending GGUF feed every hour and seeds the wishlist with
    /// candidate models the swarm could host. Off-by-default for
    /// air-gapped / privacy-sensitive deployments. Single hourly fetch
    /// per node, well below HF's anonymous rate limits.
    #[serde(default = "default_true")]
    pub hf_watcher_enabled: bool,
    /// R130: opt-in cross-pool wishlist gossip. When on, this node
    /// periodically broadcasts the top-N entries of its local wishlist
    /// (model_id + a coarse 0..100 score) on the regions topic. Inbound
    /// announcements always feed `state.models.foreign_wishlist` and
    /// boost scoring regardless of this flag — the flag only gates
    /// *publishing*, so privacy-conscious nodes can still benefit from
    /// the swarm-wide signal without leaking their own interests.
    /// Default off. Publishes "we want this model" at model granularity;
    /// does not expose pool composition, region, or per-shard interest.
    #[serde(default)]
    pub wishlist_gossip_publish: bool,
    /// R134.6: auto-action for the quant recommendation surface (R133).
    /// When on, auto-manage promotes the recommended quant variant's
    /// trust level to `DemandVerified` for any model family where the
    /// user currently hosts a *different* quant — letting the normal
    /// scoring/download path opportunistically acquire the better
    /// variant. The OLD variant is NOT proactively pruned; the standard
    /// prune cycle handles deduplication when VRAM pressure hits, so
    /// there's no in-flight inference disruption window.
    ///
    /// Default **true** (R141 — non-tech-user UX): a recommendation
    /// surface that requires the user to read it and click a button
    /// isn't a recommendation, it's a chore. Trust + prune cooldown
    /// already guard the bandwidth cost. Operators on metered links
    /// can flip this off.
    #[serde(default = "default_true")]
    pub auto_switch_quants: bool,
}

/// Per-model auto-manage policy controlling whether a model participates
/// in automatic shard downloads and how many shards to acquire.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelAutoManagePolicy {
    /// Whether auto-manage may download shards for this model.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum shards auto-manage will acquire for this model (0 = unlimited / use global default).
    #[serde(default)]
    pub max_shards: u32,
    /// Whether auto-manage may prune (delete) over-replicated shards for this model.
    #[serde(default = "default_true")]
    pub prune_enabled: bool,
}

impl Default for AutoManageConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_storage_mb: 0,
            interval_minutes: default_auto_manage_interval(),
            max_shards: 0,
            interval_seconds: None,
            max_concurrent_downloads: default_max_concurrent_downloads(),
            default_model_shard_cap: 0,
            model_policies: HashMap::new(),
            prune_enabled: true,
            min_replicas: default_min_replicas(),
            prune_cooldown_secs: default_prune_cooldown_secs(),
            max_holder_load_for_prune: default_max_holder_load_for_prune(),
            parallax_auto_rebalance: true,
            hf_watcher_enabled: true,
            wishlist_gossip_publish: false,
            auto_switch_quants: true,
        }
    }
}

fn default_max_concurrent_downloads() -> usize {
    3
}

fn default_min_replicas() -> u32 {
    2
}

fn default_prune_cooldown_secs() -> u64 {
    300
}

fn default_max_holder_load_for_prune() -> u32 {
    3
}

fn default_auto_manage_interval() -> u32 {
    5
}

/// Configuration for model storage and sharding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Size of each shard in megabytes when splitting a model for distribution.
    /// Must be between 64 and 2048 (inclusive). Default: 512.
    /// Changing this only affects newly created shards — existing shards keep their original size.
    #[serde(default = "default_shard_size_mb")]
    pub shard_size_mb: u64,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            shard_size_mb: default_shard_size_mb(),
        }
    }
}

fn default_shard_size_mb() -> u64 {
    512
}

/// Network retry backoff delays in seconds (3 attempts: 5s, 30s, 120s).
pub const NETWORK_RETRY_DELAYS: [u64; 3] = [5, 30, 120];

/// Minimum allowed shard size in MB.
pub const SHARD_SIZE_MIN_MB: u64 = 64;
/// Maximum allowed shard size in MB.
pub const SHARD_SIZE_MAX_MB: u64 = 2048;

impl ModelConfig {
    /// Return the configured shard size in bytes.
    pub fn shard_size_bytes(&self) -> u64 {
        self.shard_size_mb * 1024 * 1024
    }

    /// Validate and clamp shard_size_mb to allowed range.
    pub fn validate(&self) -> Result<(), SwarmError> {
        if self.shard_size_mb < SHARD_SIZE_MIN_MB || self.shard_size_mb > SHARD_SIZE_MAX_MB {
            return Err(SwarmError::Config(format!(
                "shard_size_mb must be between {} and {} (got {})",
                SHARD_SIZE_MIN_MB, SHARD_SIZE_MAX_MB, self.shard_size_mb
            )));
        }
        if !self.shard_size_mb.is_power_of_two() {
            tracing::warn!(
                shard_size_mb = self.shard_size_mb,
                "shard_size_mb is not a power of 2 — this may cause suboptimal alignment"
            );
        }
        Ok(())
    }
}

fn default_session_timeout() -> u64 {
    600
}

fn default_max_concurrent() -> u32 {
    10
}

fn default_gpu_layers() -> u32 {
    0
}

fn default_kv_cache_ttl() -> Option<u64> {
    Some(600)
}

fn default_speculative_gamma() -> u32 {
    4
}

/// SWARM-SPEC Layer 0: default activation compression to ON. Saves
/// ~50-70% wire bandwidth on multi-segment pipelines with negligible
/// quality impact (group-32 Q8_0 — see `inference/quant.rs`).
fn default_activation_compression() -> bool {
    true
}

fn default_hedge_after_factor() -> f32 {
    1.5
}

fn default_hedge_max_rate() -> f32 {
    0.05
}

fn default_hedge_min_samples() -> u32 {
    20
}

fn default_prefetch_min_idle_ms() -> u64 {
    2_000
}

fn default_prefetch_min_turns() -> u32 {
    2
}

fn default_prefetch_max_candidates() -> u32 {
    3
}

fn default_ngram_lookup_enabled() -> bool {
    // SWARM-SPEC Layer 1.1: ON by default. N-gram lookup is purely
    // additive — drafts are verified by the target so quality cannot
    // regress. The only cost is a sub-millisecond hash-table lookup
    // per spec round, paid only when speculative decoding is itself on.
    true
}

fn default_ngram_max_size() -> u32 {
    crate::inference::ngram_lookup::DEFAULT_MAX_NGRAM_SIZE as u32
}

fn default_ngram_num_pred_tokens() -> u32 {
    crate::inference::ngram_lookup::DEFAULT_NUM_PRED_TOKENS as u32
}

fn default_max_batch_size() -> u32 {
    1
}

fn default_batch_timeout_ms() -> u64 {
    50
}

fn default_max_decode_batch() -> u32 {
    8
}

fn default_prefill_chunk_tokens() -> u32 {
    // Sized to keep a single chunk's compute well under typical per-decode
    // latency on small models (TinyLlama Q4 ~ a few ms per token at 128
    // chunk on CPU). Bigger models / GPU users can raise via config.
    128
}

fn default_batch_collection_ms() -> u64 {
    5
}

fn default_streaming_chunk_size_bytes() -> u32 {
    262_144 // 256 KiB — age STREAM construction default, TokenWeave K=2-4 sweet spot
}

fn default_streaming_min_activation_bytes() -> u32 {
    65_536 // 64 KiB — below this, per-chunk overhead exceeds benefit
}

fn default_streaming_chunk_assembly_ttl_secs() -> u64 {
    30 // 30s — matches existing pipeline timeouts; stuck assemblies evicted
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            default_model: String::new(),
            session_timeout_seconds: default_session_timeout(),
            max_concurrent_requests: default_max_concurrent(),
            model_path: None,
            gpu_layers: default_gpu_layers(),
            kv_cache_ttl_secs: default_kv_cache_ttl(),
            speculative_decoding: false,
            persistent_pipeline_stream: false,
            streaming_chunked_send: false,
            streaming_chunk_size_bytes: default_streaming_chunk_size_bytes(),
            streaming_min_activation_bytes: default_streaming_min_activation_bytes(),
            streaming_chunk_assembly_ttl_secs: default_streaming_chunk_assembly_ttl_secs(),
            speculative_distributed: false,
            continuous_batching: default_continuous_batching(),
            max_concurrent_decode_batch: default_max_decode_batch(),
            batch_collection_ms: default_batch_collection_ms(),
            prefill_chunk_tokens: default_prefill_chunk_tokens(),
            batched_prefill_forward: default_batched_prefill_forward(),
            speculative_gamma: default_speculative_gamma(),
            hedge_enabled: false,
            hedge_after_factor: default_hedge_after_factor(),
            hedge_max_rate: default_hedge_max_rate(),
            hedge_min_samples: default_hedge_min_samples(),
            prefetch_enabled: false,
            prefetch_min_idle_ms: default_prefetch_min_idle_ms(),
            prefetch_min_turns_for_prediction: default_prefetch_min_turns(),
            prefetch_max_candidates: default_prefetch_max_candidates(),
            ngram_lookup_enabled: default_ngram_lookup_enabled(),
            ngram_max_size: default_ngram_max_size(),
            ngram_num_pred_tokens: default_ngram_num_pred_tokens(),
            draft_model_path: None,
            draft_gpu_layers: None,
            shard_range: None,
            max_batch_size: default_max_batch_size(),
            batch_timeout_ms: default_batch_timeout_ms(),
            max_split_model_memory_mb: None,
            privacy_mode: false,
            local_embedding_privacy: false,
            encrypted_pipeline: false,
            tp_max_latency_ms: default_tp_max_latency_ms(),
            prefix_cache_enabled: true,
            prefix_cache_max_entries: default_prefix_cache_max_entries(),
            prefix_cache_max_prompt_tokens: default_prefix_cache_max_prompt_tokens(),
            prefix_cache_block_tokens: default_prefix_cache_block_tokens(),
            prefix_cache_min_tokens: default_prefix_cache_min_tokens(),
            cross_node_prefix_trust_min: default_cross_node_prefix_trust_min(),
            swift_self_speculative: false,
            swift_calibration_tokens: default_swift_calibration_tokens(),
            swift_gamma: default_swift_gamma(),
            swift_skip_ratio: default_swift_skip_ratio(),
            force_standard_attn: false,
            max_seq_len_override: None,
            decentralized_spec_decoding: false,
            activation_compression: default_activation_compression(),
            parallax_routing: default_parallax_routing(),
        }
    }
}
