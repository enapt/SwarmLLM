use crate::daemon::SharedState;
use crate::types::ModelArchitecture;

/// Estimate VRAM required to run a model based on size and quantization.
///
/// Rule of thumb for quantized GGUF models:
/// - The model weights need ~model_size_bytes of VRAM (already quantized)
/// - KV cache overhead adds ~10-20% on top depending on context length
/// - We use 1.15x multiplier as a conservative estimate
pub fn estimate_model_vram_mb(total_size_bytes: u64) -> u64 {
    // Quantized model weights are already compressed; VRAM ~= file size + ~15% overhead
    (total_size_bytes as f64 * 1.15 / (1024.0 * 1024.0)) as u64
}

/// The pieces a worker's GPU footprint is actually made of.
///
/// [`estimate_model_vram_mb`] is `file_size * 1.15`, a flat 15% for "KV cache
/// overhead" with no term for vocabulary, context length, layer count or
/// KV-head geometry. Measured against real steady-state loads on an RTX 3070 it
/// is **56% low on phi-3.5-mini-q4**, which is fine for the prune scoring it was
/// written for and
/// useless as an admission decision: a gate fed those numbers admits models
/// that cannot fit and the worker dies with `CUDA_ERROR_OUT_OF_MEMORY` anyway.
///
/// The dominant missing term is the **dequantized embedding table**, on any
/// worker that still materialises one. `Embedding::new` takes a dense tensor, so
/// a 128k-vocabulary model carries `128256 * 2048 * 2 = 501 MB` that no
/// file-size multiple can see — a large fraction of the entire quantized
/// checkpoint for a 1B model, and on Gemma 2's 256k vocabulary larger than the
/// checkpoint outright. It is resident at
/// `inference::split::loader::EMBEDDING_DTYPE`; the two MUST agree, and
/// `EMBEDDING_TABLE_BYTES_PER_ELEMENT` below is the copy of that fact used here.
///
/// Since 2026-08-18 neither device materialises it: both keep the quantized
/// table and read rows as they are looked up
/// (`inference::split::token_embedding`), so the term is absent from both
/// estimators whenever `embedding_gatherable` holds. It is still priced here
/// because a table that cannot be gathered — unquantized, or a row that is not
/// block-aligned — is still dequantized in full.
///
/// The KV cache is the other large term, bounded by
/// `inference.max_seq_len_override` and the shipped default — and, for the GPU
/// estimator only, capped again at [`ADMISSION_KV_CONTEXT`] (see
/// `inference::split::kv_budget`).
#[derive(Debug, Clone, Copy)]
pub struct VramFootprintInputs {
    /// Sum of the shard bytes this worker will map. On-disk size.
    pub quantized_weight_bytes: u64,
    /// Bytes per element of the weight tensors when the checkpoint is NOT
    /// quantized (F16/BF16 → 2, F32 → 4); `None` for a quantized checkpoint.
    ///
    /// candle keeps a quantized tensor quantized on the device, but materialises
    /// an unquantized one as dense f32 (`QMatMul::from_arc` dequantizes F16 /
    /// BF16 / F32 eagerly rather than keeping a `QTensor`). So an F16 checkpoint
    /// occupies TWICE its file size in VRAM, and reading the on-disk figure as
    /// the resident one understates the dominant term by 2x for precisely the
    /// models this gate most needs to refuse.
    pub unquantized_bytes_per_element: Option<u64>,
    /// Rows in the embedding table — `{arch}.vocab_size`, or the token count.
    pub vocab_size: u64,
    /// `{arch}.embedding_length` (hidden dim).
    pub embedding_length: u64,
    /// Layers in THIS segment, not the whole model.
    pub segment_layers: u64,
    /// `{arch}.attention.head_count_kv`.
    pub head_count_kv: u64,
    /// `{arch}.attention.key_length`, or `embedding_length / head_count`.
    pub head_dim: u64,
    /// `{arch}.rope.dimension_count`.
    pub rope_dim: u64,
    /// The context length the KV cache will actually be sized to — i.e. AFTER
    /// the `DEFAULT_MAX_SEQ_LEN` cap and any override, not the raw GGUF value.
    pub effective_context: u64,
    /// Does this segment hold the embedding table (layer 0)?
    pub is_first: bool,
    /// Can `token_embd.weight` have its rows read on demand instead of being
    /// dequantized whole to [`EMBEDDING_TABLE_BYTES_PER_ELEMENT`]?
    ///
    /// Shape and dtype only. The DEVICE half of that question lives in
    /// `rows_on_demand_eligible` — CPU and CUDA each have a gather that stays
    /// on their own device, Metal does not — and both estimators here answer
    /// the same way, so this is NOT a device-specific term. Decided by
    /// `inference::split::token_embedding::table_supports_row_gather`; do not
    /// re-derive it here. A disagreement between this figure and what the
    /// loader allocates is invisible until a node either refuses a model that
    /// would have fitted or is admitted and then runs out of memory — the same
    /// trap `EMBEDDING_TABLE_BYTES_PER_ELEMENT` already carries a test for.
    pub embedding_gatherable: bool,
}

/// Bytes a CUDA worker process costs beyond its tensors: driver context, cuBLAS
/// handles, activation and prefill-chunk buffers.
///
/// **Calibrated against measured steady state, 2026-07-30.** phi-3.5-mini-q4 on
/// an RTX 3070: weights 2282 + f32 embedding 376 + KV 3072 + RoPE 1.5 = 5731 MB
/// of tensors, against 6037 MiB measured once a request had completed — so ~306
/// MB of everything else. 320 MB reproduces the total to **+0.2%**, where the
/// file-size estimator is 56.5% low on the same model.
///
/// It is a fixed constant for a term that is partly fixed (driver context, ~120
/// MB) and partly scaled by hidden size and prefill chunk. On a small model that
/// over-charges by ~190 MB, which is the safe direction for admission.
///
/// **Do not calibrate against the worker's `vram_after_load_mb`.** That is
/// sampled when loading finishes, and candle allocates the KV cache lazily on
/// the *first append* — i.e. during the first forward, after that sample. Load
/// time and steady state differed by 3265 MB on phi-3.5; comparing the estimate
/// to the load figure makes it look ~2x high when it is not.
pub const CUDA_PROCESS_OVERHEAD_BYTES: u64 = 320 * 1024 * 1024;

/// Baseline resident set of a worker subprocess that is NOT using CUDA: the
/// binary, its runtime, the tokenizer and the IPC buffers. The CPU counterpart
/// of [`CUDA_PROCESS_OVERHEAD_BYTES`], and much smaller because there is no
/// device context to establish.
///
/// Errs high for the same reason the VRAM estimate does, but the failure it
/// guards against is worse: under-estimating here means swapping, which
/// degrades every request on the machine rather than just this model's.
pub const CPU_PROCESS_OVERHEAD_BYTES: u64 = 256 * 1024 * 1024;

/// A CPU worker establishes no device context, so it must never be charged
/// more than a CUDA one. Enforced at build time: inverting these would silently
/// make the CPU budget the stricter of the two, refusing loads that fit.
const _: () = assert!(CPU_PROCESS_OVERHEAD_BYTES < CUDA_PROCESS_OVERHEAD_BYTES);

/// Context length the GPU admission check charges KV cache for, however long a
/// conversation the user has configured.
///
/// **Why admission may charge less than the worst case, and only here.** A KV
/// cache no longer reserves its whole `max_seq_len` on first use — since
/// 2026-08-07 it grows in quanta — and since 2026-08-08 every forward checks the
/// positions it is about to claim against real free VRAM and refuses with a 503
/// that re-routes to a peer (`inference::split::kv_budget`). So on a GPU the
/// worst case is *enforced at the moment memory is taken*, and pre-paying for it
/// at load buys nothing.
///
/// It costs a great deal, though. `inference.max_seq_len_override` is the only
/// way to hold an agentic client's system prompt — those run to ~5000 tokens of
/// tool schema before the conversation starts — and raising it used to raise
/// this charge in step, so the model stopped fitting the card and was loaded on
/// the CPU instead. An external report on 2026-08-17 measured that as 396
/// seconds of prompt processing and a thermal warning, for a model that had been
/// asked to support long conversations, not to reserve for them permanently.
///
/// 4096 was the shipped default context until 2026-08-18, so a user who never
/// touched the setting is charged exactly what they were charged before: this
/// changes admission ONLY for people who raised it, which is precisely the
/// population it was breaking.
///
/// **Deliberately not [`crate::inference::split::DEFAULT_MAX_SEQ_LEN`].** That
/// is a product default and moves when the product's audience changes; this is a
/// statement about a typical working conversation. Tying them would mean raising
/// the default silently re-broke the case above.
///
/// **The CPU estimator does NOT apply this cap**, because there is no runtime
/// head-room check on a CPU worker to catch the difference — under-charging
/// there would mean swapping, which degrades every request on the machine.
pub const ADMISSION_KV_CONTEXT: u64 = 4096;

/// Bytes per element of the token-embedding table, when it is held dequantized.
///
/// **This must match what the loader actually allocates.** Where the loader
/// dequantizes `token_embd.weight` it does so via `SplitModel::EMBEDDING_DTYPE`
/// — the two are checked against each other by
/// `embedding_dtype_matches_the_vram_estimate` in `inference::split::loader`,
/// because a disagreement here is invisible until a node either refuses a model
/// that fits or OOMs on one that does not.
///
/// Whether the table is held dequantized *at all* is a separate question,
/// answered by `token_embedding::rows_on_demand_eligible`. This constant prices
/// it; it does not decide it.
///
/// f16 rather than f32 because the values come from a quantized tensor whose
/// own block scales are f16 — the wider type stores no additional information,
/// and on a large-vocabulary model it was costing more memory than the entire
/// rest of the model. Reported 2026-08-01: a 6 GB card refused Gemma 2 2B
/// (1629 MB of weights) at an estimated 5447 MB, of which 2250 MB was this
/// table at f32.
pub const EMBEDDING_TABLE_BYTES_PER_ELEMENT: u64 = 2;

/// A worker's tensor footprint: weights, the embedding table, the KV cache and
/// the RoPE tables. A model's shape costs the same in system RAM as it does in
/// VRAM, so the two public estimators below are this plus a different
/// per-process constant.
///
/// `rows_on_demand` is the one term that is NOT shape: a CPU worker reads
/// embedding rows out of the quantized table and never materialises it, and a
/// CUDA worker cannot. Passed in rather than read off the inputs so each
/// estimator states its own device's answer at the point of use.
fn estimate_model_resident_bytes(
    i: &VramFootprintInputs,
    rows_on_demand: bool,
    kv_admission_context: u64,
) -> u64 {
    resident_footprint(i, rows_on_demand, kv_admission_context).total_bytes()
}

/// The terms a worker's resident memory is made of, kept apart so a refusal
/// can SAY what it is refusing. A user reading "phi-3.5 needs about 27125 MB"
/// for a 2.3 GB file called it a 10x estimate; it was 2.3 GB of weights plus
/// 24.6 GB of f32 KV cache for the 32768-token context they had configured —
/// correct, and invisible (external report, 2026-08-21). `kv_context` is the
/// token count the KV term was priced at, for the same reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentFootprint {
    pub weights_bytes: u64,
    pub embedding_bytes: u64,
    pub kv_bytes: u64,
    pub rope_bytes: u64,
    pub kv_context: u64,
}

impl ResidentFootprint {
    pub fn total_bytes(&self) -> u64 {
        self.weights_bytes
            .saturating_add(self.embedding_bytes)
            .saturating_add(self.kv_bytes)
            .saturating_add(self.rope_bytes)
    }
}

/// Per-term estimate; see [`estimate_model_resident_bytes`] for the rules.
fn resident_footprint(
    i: &VramFootprintInputs,
    rows_on_demand: bool,
    kv_admission_context: u64,
) -> ResidentFootprint {
    const F32: u64 = 4;
    // Weights. A quantized checkpoint stays quantized on the device, so its
    // on-disk size is what it costs. An UNQUANTIZED one does not: candle's
    // `QMatMul::from_arc` dequantizes F16 / BF16 / F32 to a dense f32 tensor
    // eagerly, so an F16 file costs twice its bytes. See
    // `unquantized_bytes_per_element`.
    let weights_bytes = match i.unquantized_bytes_per_element {
        Some(bpe) if bpe > 0 && bpe < F32 => i
            .quantized_weight_bytes
            .saturating_mul(F32)
            .saturating_div(bpe),
        _ => i.quantized_weight_bytes,
    };

    // Embedding table. First segment only.
    //
    // On a modern large-vocabulary model this is the single largest term —
    // larger than the quantized weights themselves. Gemma 2 2B carries a
    // 256,000-token vocabulary at hidden size 2304: 1125 MB at f16, against
    // 1629 MB for the entire rest of the model. Llama 3.1's 128,256-token
    // vocabulary at hidden size 4096 is the same order.
    //
    // Where the loader reads rows on demand it never materialises that table,
    // and the quantized bytes are already counted in `quantized_weight_bytes`
    // (token_embd lives in shard 0, whose file size is summed above) — so the
    // term is simply absent rather than replaced. Charging it anyway is what
    // refuses a model that would have fitted.
    let embedding_bytes = if i.is_first && !rows_on_demand {
        i.vocab_size
            .saturating_mul(i.embedding_length)
            .saturating_mul(EMBEDDING_TABLE_BYTES_PER_ELEMENT)
    } else {
        0
    };

    // KV cache — [B, H, ctx, D] per layer, for K and V, as f32, at
    // `kv_admission_context`.
    //
    // That is the full effective context on a CPU worker and CAPPED on a GPU
    // one; see [`ADMISSION_KV_CONTEXT`] for why the two differ. The cap is
    // the only place this estimator deliberately charges less than the worst
    // case, and it is allowed to because a GPU worker has a runtime check that
    // catches the difference.
    let kv_bytes = i
        .segment_layers
        .saturating_mul(2)
        .saturating_mul(i.head_count_kv)
        .saturating_mul(i.head_dim)
        .saturating_mul(kv_admission_context)
        .saturating_mul(F32);

    // RoPE cos/sin tables. Charged at the FULL context regardless, because
    // unlike the KV cache these really are precomputed in full at load
    // (`rope::precompute_freqs_cis`) — nothing grows them on demand and no
    // runtime check bounds them. They are small (34 MB at 32k) but the reason
    // they are not capped is the reason the KV cache can be.
    let rope_bytes = i
        .effective_context
        .saturating_mul(i.rope_dim.max(2) / 2)
        .saturating_mul(F32)
        .saturating_mul(2);

    ResidentFootprint {
        weights_bytes,
        embedding_bytes,
        kv_bytes,
        rope_bytes,
        kv_context: kv_admission_context,
    }
}

/// Estimate a worker's GPU footprint in MB from the model's real geometry.
///
/// Deliberately errs HIGH: for an admission decision, over-estimating costs a
/// model that would have fitted, while under-estimating costs a hard OOM and —
/// until this release — a permanent fall back to the CPU.
pub fn estimate_worker_vram_mb(i: &VramFootprintInputs) -> u64 {
    // A CUDA worker reads embedding rows on demand too, through an on-device
    // gather, so it no longer materialises the table either — 720 MB on
    // meta-llama-3.1-8b. See `token_embedding`.
    //
    // The KV cache is charged at a capped context because a GPU worker has a
    // runtime head-room check that refuses gracefully if a conversation
    // outgrows it — see `ADMISSION_KV_CONTEXT`.
    estimate_model_resident_bytes(
        i,
        i.embedding_gatherable,
        i.effective_context.min(ADMISSION_KV_CONTEXT),
    )
    .saturating_add(CUDA_PROCESS_OVERHEAD_BYTES)
        / (1024 * 1024)
}

/// Estimate a worker's system-RAM footprint in MB from the same geometry.
///
/// Identical to [`estimate_worker_vram_mb`] but for the per-process overhead —
/// a CPU worker establishes no device context. Used for CPU admission, which
/// exists because the GPU path's own fallback is "load it in system RAM
/// instead": the more often that fires, the more weight lands here. KV is
/// charged at [`ADMISSION_KV_CONTEXT`]; see [`cpu_footprint`].
pub fn estimate_worker_ram_mb(i: &VramFootprintInputs) -> u64 {
    cpu_footprint(i)
        .total_bytes()
        .saturating_add(CPU_PROCESS_OVERHEAD_BYTES)
        / (1024 * 1024)
}

/// The CPU worker's resident terms, itemised.
pub fn cpu_footprint(i: &VramFootprintInputs) -> ResidentFootprint {
    // The SAME admission context the GPU uses, for the same reason: since
    // 2026-08-21 a CPU worker is handed a KV budget (`--kv-budget-bytes`) and
    // refuses at run time to grow past it, so admission no longer has to price
    // the whole ceiling. Pricing it was what turned a 2.3 GB phi-3.5 into a
    // 27 GB refusal at a 32k override — the model never loaded, where now it
    // loads and only a conversation that outgrows the room is refused (and
    // re-routed to a peer).
    resident_footprint(
        i,
        i.embedding_gatherable,
        i.effective_context.min(ADMISSION_KV_CONTEXT),
    )
}

/// Where the admission context came from, for the refusal message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextSource {
    /// `inference.max_seq_len_override` is set and won.
    Override,
    /// The model's declared context, capped at the shipped default.
    DeclaredOrDefault,
}

/// The human-readable account of a CPU refusal: what the estimate is made of,
/// and what the budget is limited by. Pure, so the wording is testable and the
/// two numbers in the message can no longer surprise the reader in silence.
pub fn describe_cpu_refusal(
    model: &str,
    f: &ResidentFootprint,
    effective_context: u64,
    ctx_source: ContextSource,
    budget_mb: u64,
    budget_note: &str,
    in_use_mb: u64,
) -> String {
    const MB: u64 = 1024 * 1024;
    let total_mb = f.total_bytes().saturating_add(CPU_PROCESS_OVERHEAD_BYTES) / MB;
    let other_mb = f
        .embedding_bytes
        .saturating_add(f.rope_bytes)
        .saturating_add(CPU_PROCESS_OVERHEAD_BYTES)
        / MB;
    let ctx_why = match ctx_source {
        ContextSource::Override => "set by `inference.max_seq_len_override`",
        ContextSource::DeclaredOrDefault => "its declared context, capped at the default",
    };
    let kv_clause = if f.kv_context < effective_context {
        format!(
            "{} MB of KV cache for the {}-token conversation it is admitted at (its full context \
             is {effective_context} tokens, {ctx_why}; longer conversations are allowed to grow \
             while memory lasts and are refused and re-routed when it runs out)",
            f.kv_bytes / MB,
            f.kv_context,
        )
    } else {
        format!(
            "{} MB of KV cache for a {}-token context ({ctx_why})",
            f.kv_bytes / MB,
            f.kv_context,
        )
    };
    let mut s = format!(
        "{model} needs about {total_mb} MB of memory: {} MB of weights, {kv_clause}, \
         {other_mb} MB of embeddings, tables and overhead. This node's budget allows {budget_mb} MB",
        f.weights_bytes / MB,
    );
    if !budget_note.is_empty() {
        s.push_str(" (");
        s.push_str(budget_note);
        s.push(')');
    }
    s.push_str(&format!(
        " and {in_use_mb} MB is already in use. Raise `resources.max_ram_mb`, or free memory by \
         unloading another model."
    ));
    s
}

/// The system-RAM budget a CPU model is judged against, as a snapshot taken
/// AT ADMISSION — never frozen at startup.
///
/// Two limits, both live: the cap the owner (or the auto default) set, read
/// through `cfg()` so a Settings change applies at once, and an anti-swap
/// clamp against what is free on the machine right now. Until 2026-08-21 the
/// clamp was folded into the cap once at startup ("70% of the memory free when
/// the node started"), so a daemon restarted while memory was busy carried the
/// smaller figure for the rest of its life — the same `max_ram_mb = 18000`
/// reported "budget allows 13370 MB" one day and "10500 MB" the next, with
/// 14773 MB actually free at the time of the refusal (external report).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RamBudget {
    /// The configured or auto-sized ceiling, MB.
    pub cap_mb: u64,
    /// `resources.max_ram_mb` as configured (0 = auto).
    pub configured_mb: u64,
    /// Machine total, MB (0 = unreadable).
    pub total_mb: u64,
    /// Memory available right now, MB (0 = unreadable → not judged).
    pub available_mb: u64,
    /// How much a NEW model may take right now without swapping:
    /// `max(70% of available, total/4)`, or `u64::MAX` when unreadable.
    pub live_headroom_mb: u64,
}

impl RamBudget {
    pub fn from_machine(cap_mb: u64, configured_mb: u64, total_mb: u64, available_mb: u64) -> Self {
        let live_headroom_mb = if available_mb == 0 {
            u64::MAX
        } else {
            (available_mb / 100 * FREE_RAM_HEADROOM_PCT).max(total_mb / 4)
        };
        Self {
            cap_mb,
            configured_mb,
            total_mb,
            available_mb,
            live_headroom_mb,
        }
    }

    /// A cap with no knowledge of the machine (tests, and the fallback when no
    /// provider is installed): only the ceiling is enforced.
    pub fn cap_only(cap_mb: u64) -> Self {
        Self {
            cap_mb,
            configured_mb: cap_mb,
            total_mb: 0,
            available_mb: 0,
            live_headroom_mb: u64::MAX,
        }
    }

    /// May a model estimated at `estimate_mb` be loaded while `committed_mb`
    /// is already charged? Both limits must agree.
    pub fn allows(&self, committed_mb: u64, estimate_mb: u64) -> bool {
        committed_mb.saturating_add(estimate_mb) <= self.cap_mb
            && estimate_mb <= self.live_headroom_mb
    }

    /// What is left for the admitted model's KV cache to grow into: the
    /// smaller of the uncommitted cap and the live headroom beyond the model's
    /// own charge.
    pub fn headroom_after(&self, committed_mb_after: u64, this_estimate_mb: u64) -> u64 {
        self.cap_mb
            .saturating_sub(committed_mb_after)
            .min(self.live_headroom_mb.saturating_sub(this_estimate_mb))
    }

    /// Where the cap came from, for messages.
    pub fn cap_source(&self) -> String {
        if self.configured_mb > 0 {
            format!("`resources.max_ram_mb` is {} MB", self.configured_mb)
        } else {
            format!(
                "auto-sized to {} MB from {} MB of RAM",
                self.cap_mb, self.total_mb
            )
        }
    }

    /// The figure a refused model was actually judged against, and why —
    /// the cap when the cap refused it, otherwise the live headroom.
    pub fn limiting_figure(&self, committed_mb: u64, estimate_mb: u64) -> (u64, String) {
        if committed_mb.saturating_add(estimate_mb) > self.cap_mb {
            return (self.cap_mb, self.cap_source());
        }
        (
            self.live_headroom_mb,
            format!(
                "{}; right now {} MB of memory is free and SwarmLLM uses at most \
                 {FREE_RAM_HEADROOM_PCT}% of that ({} MB) so loading a model cannot push \
                 the machine into swap",
                self.cap_source(),
                self.available_mb,
                self.live_headroom_mb
            ),
        )
    }
}

/// MoE-aware VRAM estimation. For Mixture-of-Experts models, only a fraction
/// of expert weights are active at any time, so the actual VRAM requirement
/// is much lower than the total file size.
///
/// Formula:
///   active_fraction = 0.40 (attention/embeddings, always loaded)
///                   + 0.60 * (experts_per_token / num_experts)
///   effective_vram = total_size * active_fraction * 1.15 (KV overhead)
pub fn estimate_model_vram_mb_arch(total_size_bytes: u64, arch: &ModelArchitecture) -> u64 {
    let (num_experts, experts_per_token) = match arch {
        ModelArchitecture::Mixtral {
            num_experts,
            experts_per_token,
        }
        | ModelArchitecture::DeepSeek {
            num_experts,
            experts_per_token,
        }
        | ModelArchitecture::Llama4 {
            num_experts,
            experts_per_token,
        }
        | ModelArchitecture::Qwen35Moe {
            num_experts,
            experts_per_token,
        } => (*num_experts, *experts_per_token),
        // Dense architectures — all weights active
        _ => return estimate_model_vram_mb(total_size_bytes),
    };

    if num_experts == 0 || experts_per_token >= num_experts {
        return estimate_model_vram_mb(total_size_bytes);
    }

    let active_fraction = 0.40 + 0.60 * (experts_per_token as f64 / num_experts as f64);
    (total_size_bytes as f64 * active_fraction * 1.15 / (1024.0 * 1024.0)) as u64
}

/// This node's memory bandwidth in GB/s — the figure every "how fast is this
/// machine" answer is derived from.
///
/// A graphics card is looked up by name; a processor-only node reports what
/// `mem_bandwidth::measured_gbps` actually measured on it, because generating a
/// token is bandwidth-bound.
///
/// `None` means the measurement could not be taken, which is a different fact
/// from "slow" and must not be reported as one. What to do about it is the
/// CALLER's policy and the two callers differ on purpose: the capability gossip
/// falls back to a nominal figure so a memory-starved node still advertises
/// something rather than nothing, while `GET /api/admin/stats` reports unknown
/// rather than state a number it did not derive.
///
/// **Why this is one function.** `health::monitor` measured the CPU case and
/// gossiped it, so every peer on the swarm was told a real speed for a
/// processor-only node — while that node's own `/api/admin/stats` answered
/// `null`, because its `match` on `gpu_info` had no `None` arm that asked. The
/// swarm knew a number about the machine that the machine would not state, and
/// anyone diagnosing a slow CPU node from its own dashboard had nothing to read
/// (observed 2026-08-25 on a node 44 minutes into a run). A new surface that
/// wants this figure calls this rather than re-deriving it.
/// Takes the card's NAME rather than a struct, because there are two `GpuInfo`
/// types in this codebase — `inference::executor::GpuInfo` on `SharedState` and
/// `swarmllm_types::GpuInfo` on the wire — and the name is the only thing this
/// needs from either.
pub fn node_memory_bandwidth_gbps(gpu_name: Option<&str>) -> Option<f32> {
    match gpu_name {
        Some(name) => Some(gpu_memory_bandwidth_gbps(name)),
        None => crate::inference::mem_bandwidth::measured_gbps(),
    }
}

/// Lookup table for GPU memory bandwidth in GB/s.
/// Used for bandwidth-based speed estimation (tokens/s ≈ bandwidth / model_size * efficiency).
pub fn gpu_memory_bandwidth_gbps(name: &str) -> f32 {
    let name_upper = name.to_uppercase();
    // Match most specific first
    match () {
        // NVIDIA data center
        _ if name_upper.contains("H100") => 3352.0,
        _ if name_upper.contains("H200") => 4800.0,
        _ if name_upper.contains("A100") && name_upper.contains("80") => 2039.0,
        _ if name_upper.contains("A100") => 1555.0,
        _ if name_upper.contains("A6000") => 768.0,
        _ if name_upper.contains("A5000") => 768.0,
        _ if name_upper.contains("A4000") => 448.0,
        _ if name_upper.contains("V100") => 900.0,
        _ if name_upper.contains("L40S") => 864.0,
        _ if name_upper.contains("L40") => 864.0,
        _ if name_upper.contains("L4") => 300.0,
        _ if name_upper.contains("T4") => 300.0,
        // NVIDIA consumer - RTX 40 series
        _ if name_upper.contains("4090") => 1008.0,
        _ if name_upper.contains("4080") && name_upper.contains("SUPER") => 736.0,
        _ if name_upper.contains("4080") => 717.0,
        _ if name_upper.contains("4070") && name_upper.contains("SUPER") => 504.0,
        _ if name_upper.contains("4070") => 504.0,
        _ if name_upper.contains("4060") && name_upper.contains("TI") => 288.0,
        _ if name_upper.contains("4060") => 272.0,
        // NVIDIA consumer - RTX 30 series
        _ if name_upper.contains("3090") && name_upper.contains("TI") => 1008.0,
        _ if name_upper.contains("3090") => 936.0,
        _ if name_upper.contains("3080") && name_upper.contains("TI") => 912.0,
        _ if name_upper.contains("3080") => 760.0,
        _ if name_upper.contains("3070") && name_upper.contains("TI") => 608.0,
        _ if name_upper.contains("3070") => 448.0,
        _ if name_upper.contains("3060") && name_upper.contains("TI") => 448.0,
        _ if name_upper.contains("3060") => 360.0,
        // NVIDIA consumer - RTX 20 series
        _ if name_upper.contains("2080") && name_upper.contains("TI") => 616.0,
        _ if name_upper.contains("2080") => 448.0,
        _ if name_upper.contains("2070") => 448.0,
        _ if name_upper.contains("2060") => 336.0,
        // Apple Silicon
        _ if name_upper.contains("M4 MAX") => 546.0,
        _ if name_upper.contains("M4 PRO") => 273.0,
        _ if name_upper.contains("M4") => 120.0,
        _ if name_upper.contains("M3 MAX") => 400.0,
        _ if name_upper.contains("M3 PRO") => 150.0,
        _ if name_upper.contains("M3") => 100.0,
        _ if name_upper.contains("M2 MAX") => 400.0,
        _ if name_upper.contains("M2 PRO") => 200.0,
        _ if name_upper.contains("M2") => 100.0,
        _ if name_upper.contains("M1 MAX") => 400.0,
        _ if name_upper.contains("M1 PRO") => 200.0,
        _ if name_upper.contains("M1") => 68.0,
        // AMD
        _ if name_upper.contains("7900 XTX") => 960.0,
        _ if name_upper.contains("7900 XT") => 800.0,
        _ if name_upper.contains("MI300") => 5300.0,
        _ if name_upper.contains("MI250") => 3277.0,
        // Default: conservative estimate for unknown GPU
        _ => 300.0,
    }
}

/// Estimate tokens/s for a 7B model based on GPU memory bandwidth.
///
/// Formula: tokens/s = memory_bandwidth_GB_s / model_size_GB * efficiency
/// The efficiencies are **measured**, not assumed, through
/// `examples/prefill_bench` — which drives `SplitModel::forward` directly, so
/// there is no HTTP, no speculation and no prefix cache between the number and
/// the thing being measured. Same model, prompt and KV depth on both sides
/// (`qwen2.5-coder-7b-instruct-q4-k-m`, 896-token prompt, ~912 KV, 2026-09-01):
///
/// | | measured | previously advertised |
/// |---|---|---|
/// | Ryzen 7 5800H, 29.9 GB/s measured | **5.26 tok/s** | 1.02 |
/// | RTX 3070 Laptop, 448 GB/s from the table | **35.32 tok/s** | 30.5 |
///
/// **The processor constant was 0.15 and is 5.2x wrong.** A processor reaches
/// ~82% of its memory roofline on decode — corroborated by the pinned 3B
/// baseline (10.44 tok/s, the same 82%) and by the "69% of roofline" figure
/// recorded for CPU decode elsewhere — while 0.15 asserts 15%. So every
/// processor-only node in the swarm advertised about a fifth of what it does,
/// and an Apple M4 Mac mini (69.8 GB/s measured by its own node) claimed 2.38
/// tok/s where it should claim ~12.
///
/// The graphics constant was 0.30 and is close: 0.347 reproduces the
/// measurement. Note the direction — the card was *understated* too, not
/// overstated. An earlier reading of this taken over HTTP said the opposite and
/// was wrong: that path is not a decode measurement, because the n-gram cascade
/// drafts out of the prompt and the prefix cache removes the prefill, and three
/// attempts on one machine gave 20.75, ~50 and ~27 tok/s.
///
/// **Confidence differs between the two.** The processor figure has two models
/// on one machine agreeing, plus a prediction that matched it to 1% before it
/// was taken. The graphics figure is **one card**; treat its third digit as
/// unearned. Both are rounded slightly down, since a machine serving the swarm
/// is not an idle one.
///
/// Re-measure with:
/// ```text
/// SWARM_BENCH_MODEL=<7B Q4 dir> SWARM_BENCH_PROMPT=896 SWARM_BENCH_DECODE=32 \
/// SWARM_BENCH_REPS=3 [SWARM_BENCH_DEVICE=cuda] ./target/release/examples/prefill_bench
/// ```
/// A release build is required on both sides — an unoptimised one measures its
/// own loop (gotcha #427) — and the CUDA arm needs `--features flash-attn`,
/// which the harness enforces rather than silently reporting processor timings.
pub fn estimate_tokens_per_sec_7b(bandwidth_gbps: f32, is_gpu: bool) -> f32 {
    const MODEL_SIZE_7B_Q4: f32 = 4.4; // ~4.4 GB for 7B Q4_K_M
                                       // Fraction of the memory roofline decode actually achieves. A card is
                                       // *less* efficient per byte than a processor at batch 1, because one query
                                       // row cannot fill it and the per-layer kernel launches dominate — which is
                                       // the opposite of what the old pair of constants asserted.
    let efficiency = if is_gpu { 0.35 } else { 0.75 };
    bandwidth_gbps / MODEL_SIZE_7B_Q4 * efficiency
}

/// What THIS node would manage on a 7B Q4 — the same figure every peer gossips
/// about itself in `NodeCapability.est_tokens_per_sec_7b`, derived the same way.
///
/// `None` means the bandwidth could not be established, which is a different
/// fact from "slow"; what to do about it stays the caller's policy, exactly as
/// it is for [`node_memory_bandwidth_gbps`] (the capability gossip substitutes
/// a nominal figure so a node still advertises something; `/api/admin/stats`
/// reports unknown rather than state a number it did not derive).
///
/// **Why this exists.** Two scheduler sites derived the local node's speed from
/// `gpu_info` alone — `.map(|g| …gpu_memory_bandwidth_gbps…).unwrap_or(0.0)` —
/// so a processor-only node reported its OWN speed as zero. Both consumers read
/// zero as *unknown* and substitute a generic constant (`UNKNOWN_COMPUTE_MS`;
/// the parallax allocator documents 0 as "treats as average"), so the one node
/// whose speed we can actually measure was priced with a guess while every
/// remote peer was priced with a real gossiped figure. That is the same missing
/// `None` arm [`node_memory_bandwidth_gbps`] was written to close for
/// `/api/admin/stats`; these two sites were not updated with it.
pub fn node_tokens_per_sec_7b(gpu_name: Option<&str>) -> Option<f32> {
    node_memory_bandwidth_gbps(gpu_name)
        .map(|bw| estimate_tokens_per_sec_7b(bw, gpu_name.is_some()))
}

/// Compute the optimal shard window for a model given VRAM budget.
///
/// Returns `(start, end)` inclusive shard indices that fit within the budget.
/// Always prefers shard 0 (embeddings) and the last shard (output head) for
/// "boomerang" local inference. Fills remaining budget with contiguous middle shards.
///
/// Returns None if even 2 shards don't fit.
pub fn compute_optimal_shard_window(
    shard_count: u32,
    shard_vram_each_mb: u64,
    vram_budget_mb: u64,
) -> Option<Vec<u32>> {
    if shard_count == 0 || shard_vram_each_mb == 0 {
        return None;
    }

    let shards_that_fit = (vram_budget_mb / shard_vram_each_mb) as u32;
    if shards_that_fit == 0 {
        return None;
    }

    // If all shards fit, load everything
    if shards_that_fit >= shard_count {
        return Some((0..shard_count).collect());
    }

    let last_shard = shard_count - 1;
    let mut selected = Vec::new();

    if shard_count == 1 {
        return Some(vec![0]);
    }

    // Always include shard 0 (embeddings) and last shard (output head)
    selected.push(0);
    if shards_that_fit >= 2 {
        selected.push(last_shard);
    }

    // Fill remaining budget with contiguous middle shards starting from shard 1
    for i in 1..last_shard {
        if selected.len() as u32 >= shards_that_fit {
            break;
        }
        selected.push(i);
    }

    selected.sort();
    Some(selected)
}

/// Compute the total VRAM available across the entire network (all peers + local node).
pub fn global_pool_vram_mb(shared: &SharedState) -> u64 {
    let mut total = 0u64;

    // Local GPU -- use gpu_info if available, fallback to nvidia-smi
    total += local_vram_mb(shared);

    // Private mode: only count allowed nodes' VRAM
    let allowed_set = crate::pool::scope::allowed_node_set(shared);

    for peer in shared.peer_registry.iter() {
        // In private mode, skip peers outside the allowed set
        if let Some(ref allowed) = allowed_set {
            if !allowed.contains(peer.key()) {
                continue;
            }
        }
        if let Some(ref cap) = peer.capability {
            if let Some(ref gpu) = cap.gpu {
                total += gpu.vram_total_mb;
            }
        }
    }

    total
}

/// Get local VRAM in MB, with nvidia-smi fallback when gpu_info is None.
pub fn local_vram_mb(shared: &SharedState) -> u64 {
    if let Some(ref gpu) = shared.gpu_info {
        return gpu.vram_total_mb;
    }
    // Fallback: detect via nvidia-smi
    detect_vram_nvidia_smi().unwrap_or(0)
}

/// Fallback GPU VRAM detection via nvidia-smi.
fn detect_vram_nvidia_smi() -> Option<u64> {
    detect_gpu_nvidia_smi().1
}

/// Detect GPU name and total VRAM via nvidia-smi.
pub(crate) fn detect_gpu_nvidia_smi() -> (Option<String>, Option<u64>) {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    match output {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout);
            let line = text.trim();
            if let Some((name, vram_str)) = line.split_once(',') {
                let name = name.trim().to_string();
                let vram_mb = vram_str.trim().parse::<u64>().ok();
                (Some(name), vram_mb)
            } else {
                (None, None)
            }
        }
        _ => (None, None),
    }
}

/// Detect the GPU's CUDA compute capability via nvidia-smi.
///
/// Separate from [`detect_gpu_nvidia_smi`] because it is asked exactly once, at
/// startup, and only by CUDA builds — folding it into the VRAM query would make
/// every caller of that pay for a field they do not use.
///
/// `None` means "could not tell", never "too old": nvidia-smi missing, an NVML
/// version mismatch, or an unrecognised format all land here, and
/// [`crate::daemon::gpu_support`]'s callers must leave the GPU alone in that
/// case. A working card sent to the CPU because a subprocess misbehaved would
/// be a worse bug than the one the probe exists to prevent.
///
/// Only compiled for CUDA builds: a build with no CUDA kernels has no
/// capability floor to compare against, so asking would be meaningless work.
#[cfg(feature = "candle-cuda")]
pub(crate) fn detect_gpu_compute_cap() -> Option<(u32, u32)> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    // Multi-GPU hosts print one line per card. We only ever bind device 0.
    crate::daemon::gpu_support::parse_compute_cap(text.lines().next()?)
}

/// Query live GPU VRAM usage in MB via nvidia-smi.
///
/// Called on each auto-manage tick (~5 min) for accurate VRAM pressure.
/// Returns None if nvidia-smi is unavailable or fails.
pub(crate) fn query_gpu_vram_used() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    text.trim().parse::<u64>().ok()
}

/// Query live free GPU VRAM in MB via a single `nvidia-smi` call.
///
/// `memory.total - memory.used`, so it accounts for everything already
/// resident — other processes, and any model this daemon already loaded.
/// One combined query rather than [`detect_gpu_nvidia_smi`] +
/// [`query_gpu_vram_used`] because the two would be sampled at different
/// instants, and the subtraction of two racing samples can go negative.
///
/// Called once per model load by the split loader to size the KV cache
/// (`inference::split::kv_budget`). Returns None when nvidia-smi is
/// unavailable, which the caller must treat as "unknown", never as "zero".
pub(crate) fn query_gpu_vram_free_mb() -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let line = text.trim().lines().next()?;
    let (total, used) = line.split_once(',')?;
    let total = total.trim().parse::<u64>().ok()?;
    let used = used.trim().parse::<u64>().ok()?;
    Some(total.saturating_sub(used))
}

/// Estimate VRAM for a segment (layer range) by scaling the full-model estimate
/// by the fraction of layers covered.
pub(super) fn estimate_segment_vram_mb(
    manifest: &crate::types::ModelManifest,
    layer_start: usize,
    layer_end: usize,
) -> u64 {
    let total_layers = manifest.num_layers as usize;
    if total_layers == 0 {
        return estimate_model_vram_mb(manifest.total_size_bytes);
    }
    let fraction = (layer_end - layer_start) as f64 / total_layers as f64;
    let full_vram = estimate_model_vram_mb(manifest.total_size_bytes);
    (full_vram as f64 * fraction).ceil() as u64
}

/// Compute the VRAM budget from SharedState for passing to `check_and_load_model`.
pub fn compute_vram_budget(shared: &crate::daemon::SharedState) -> Option<u64> {
    let gpu_total = shared
        .gpu_info
        .as_ref()
        .map(|g| g.vram_total_mb)
        .unwrap_or(0);
    // The LIVE config, not the boot snapshot. `max_gpu_vram_mb` is the escape
    // hatch for exactly the situation where this matters — a model that does
    // not fit the default fraction of the card — and reading the startup value
    // meant raising it in Settings saved, answered "ok", wrote the new number
    // to disk, and changed nothing until the daemon restarted.
    //
    // Measured on this machine 2026-08-24: `max_gpu_vram_mb = 7000` on disk,
    // the running daemon still reporting a 4095 MB budget, an 8B model needing
    // 6033 MB refused, and the card 88% empty (832 MB used of 8192). The model
    // ran on the processor at 1.0 tok/s instead.
    //
    // Same lesson as gotcha #281, and the sibling `ram_budget_now` was given
    // exactly this treatment in August (gotcha #362) while this one was left
    // on the snapshot.
    // What OTHER programs are holding on the card right now: the used total,
    // less what our own workers have already reserved. Passing our own models'
    // memory through here would charge for them twice, since the caller weighs
    // this budget against `committed + estimated`.
    let other_process_mb = query_gpu_vram_free_mb().map(|free| {
        let used_total = gpu_total.saturating_sub(free);
        used_total.saturating_sub(shared.model_process_pool.vram_committed_mb())
    });
    shared.cfg().resources.inference_vram_budget_mb(
        gpu_total,
        other_process_mb,
        shared.contribution(),
    )
}

/// Percentage of *currently free* system RAM a configured budget may claim.
///
/// Mirrors `FREE_DISK_HEADROOM_PCT`: a configured ceiling is a ceiling, not a
/// promise the memory exists. Lower than the disk figure because the operating
/// system and every other process on the box need headroom to keep running,
/// and the failure mode when they do not get it is swapping — which degrades
/// the whole machine rather than just this daemon.
pub const FREE_RAM_HEADROOM_PCT: u64 = 70;

/// Effective system-RAM budget for CPU model loading, in MB.
///
/// The cap and its source, for the startup log and the no-provider fallback.
/// Admission itself uses [`ram_budget_now`] every time, so the anti-swap clamp
/// is judged against memory free NOW rather than at startup. `None` means "do
/// not judge": no cap could be derived — a limit must never be invented from a
/// failed measurement.
pub fn compute_ram_budget(shared: &crate::daemon::SharedState) -> Option<(u64, String)> {
    let b = ram_budget_now(shared)?;
    Some((b.cap_mb, b.cap_source()))
}

/// Machine memory right now: `(total_mb, available_mb)`. Reads `/proc/meminfo`
/// (or the platform equivalent); cheap enough to call at every admission.
pub fn system_memory_mb() -> (u64, u64) {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    (
        sys.total_memory() / (1024 * 1024),
        sys.available_memory() / (1024 * 1024),
    )
}

/// The live RAM budget: the cap from the CURRENT config (so a Settings change
/// applies at once) and the anti-swap headroom from what is free right now.
/// `None` means "do not judge": no cap could be derived (unreadable total with
/// no configured value).
pub fn ram_budget_now(shared: &crate::daemon::SharedState) -> Option<RamBudget> {
    let (total_mb, available_mb) = system_memory_mb();
    // "Has a GPU" here must mean "the GPU will actually run the models", not
    // merely that one is installed. A node with `inference.gpu_layers = 0` runs
    // everything on the CPU regardless of its hardware, so it is in exactly the
    // CPU-only situation the higher fraction exists for. Routed through the
    // canonical placement mapping so this cannot drift from where models are
    // really loaded.
    let has_gpu = shared.gpu_info.is_some()
        && !crate::daemon::shard_loader::force_cpu_for(shared.config.inference.gpu_layers);

    let configured = shared.cfg().resources.max_ram_mb;
    let by_config =
        shared
            .cfg()
            .resources
            .inference_ram_budget_mb(total_mb, has_gpu, shared.contribution())?;

    Some(RamBudget::from_machine(
        by_config,
        configured,
        total_mb,
        available_mb,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moe_vram_lower_than_dense() {
        // Mixtral 8x7B ≈ 47GB on disk
        let total_bytes = 47u64 * 1024 * 1024 * 1024;
        let dense_vram = estimate_model_vram_mb(total_bytes);
        let moe_vram = estimate_model_vram_mb_arch(
            total_bytes,
            &ModelArchitecture::Mixtral {
                num_experts: 8,
                experts_per_token: 2,
            },
        );
        // MoE should be significantly less than dense
        assert!(
            moe_vram < dense_vram,
            "MoE={moe_vram} should be < dense={dense_vram}"
        );
        // Mixtral 8x7B with 2/8 experts: active_fraction = 0.40 + 0.60*0.25 = 0.55
        // So MoE VRAM should be ~55% of dense
        // active_fraction = 0.40 + 0.60*0.25 = 0.55, so ~55% of dense ≈ 30GB
        assert!(
            moe_vram < 31 * 1024,
            "MoE 8x7B should fit in <31GB, got {moe_vram}MB"
        );
    }

    #[test]
    fn dense_arch_unchanged() {
        let total_bytes = 4u64 * 1024 * 1024 * 1024;
        let dense = estimate_model_vram_mb(total_bytes);
        let llama = estimate_model_vram_mb_arch(total_bytes, &ModelArchitecture::Llama);
        assert_eq!(dense, llama);
    }

    #[test]
    fn gpu_bandwidth_known_gpus() {
        assert_eq!(gpu_memory_bandwidth_gbps("NVIDIA RTX 4090"), 1008.0);
        assert_eq!(gpu_memory_bandwidth_gbps("NVIDIA GeForce RTX 3070"), 448.0);
        assert_eq!(gpu_memory_bandwidth_gbps("NVIDIA A100 80GB"), 2039.0);
        assert_eq!(gpu_memory_bandwidth_gbps("Apple M3 Max"), 400.0);
    }

    #[test]
    fn gpu_bandwidth_unknown_returns_default() {
        assert_eq!(gpu_memory_bandwidth_gbps("Unknown GPU XYZ"), 300.0);
    }

    /// A graphics card's figure comes from the table, by name.
    #[test]
    fn a_card_reports_its_table_bandwidth() {
        assert_eq!(
            node_memory_bandwidth_gbps(Some("NVIDIA GeForce RTX 3070")),
            Some(448.0)
        );
    }

    /// A processor-only node reports what its memory actually delivers, not
    /// nothing. This is the case `/api/admin/stats` answered `null` for while
    /// the same measurement was being gossiped to every peer.
    #[test]
    fn a_processor_only_node_still_reports_a_bandwidth() {
        let measured = crate::inference::mem_bandwidth::measured_gbps();
        assert_eq!(node_memory_bandwidth_gbps(None), measured);
        if let Some(bw) = measured {
            assert!(bw > 0.0, "a measurement of zero would be a broken probe");
            assert!(
                estimate_tokens_per_sec_7b(bw, false) > 0.0,
                "a real bandwidth must yield a real speed estimate"
            );
        }
    }

    /// The scheduler asked for the local node's speed and got zero on any
    /// machine without a card, because both sites derived it from `gpu_info`
    /// alone. Zero is read downstream as *unknown* and replaced with a generic
    /// constant, so the one node whose speed we can actually measure was the
    /// only one priced with a guess.
    #[test]
    fn a_processor_only_node_states_a_real_speed_rather_than_zero() {
        let Some(measured) = crate::inference::mem_bandwidth::measured_gbps() else {
            return; // bandwidth unmeasurable here; the None policy is the caller's
        };
        let tps = node_tokens_per_sec_7b(None).expect("measurable bandwidth means a speed");
        assert!(tps > 0.0, "a processor-only node reported {tps} tok/s");
        // It must be priced as a processor, not as a card.
        assert_eq!(tps, estimate_tokens_per_sec_7b(measured, false));
        assert_ne!(
            tps,
            estimate_tokens_per_sec_7b(measured, true),
            "the two efficiencies must not have collapsed into one"
        );
    }

    /// A card is priced from its name, never from host memory bandwidth — the
    /// name determines the hardware, and the two figures differ by an order of
    /// magnitude.
    #[test]
    fn a_card_is_priced_from_its_name() {
        const NAME: &str = "NVIDIA GeForce RTX 3070";
        assert_eq!(
            node_tokens_per_sec_7b(Some(NAME)),
            Some(estimate_tokens_per_sec_7b(
                gpu_memory_bandwidth_gbps(NAME),
                true
            ))
        );
    }

    #[test]
    fn speed_estimation_sanity() {
        // RTX 3070: 448 GB/s, 7B Q4 ≈ 4.4GB
        let tps = estimate_tokens_per_sec_7b(448.0, true);
        assert!(tps > 20.0 && tps < 50.0, "Expected ~35 t/s, got {tps}");
    }

    /// The constants are calibrated against `prefill_bench` on one machine, at
    /// the same model, prompt and KV depth on both sides (2026-09-01). If a
    /// re-measurement moves either of these, change the constant — do not widen
    /// the test, or the estimate stops meaning anything.
    #[test]
    fn the_efficiencies_reproduce_the_measurements_they_were_taken_from() {
        // Ryzen 7 5800H, 29.9 GB/s measured, 5.26 tok/s observed.
        let cpu = estimate_tokens_per_sec_7b(29.9, false);
        assert!(
            (4.7..=5.7).contains(&cpu),
            "processor estimate {cpu} no longer reproduces the measured 5.26 tok/s"
        );
        // RTX 3070 Laptop, 448 GB/s table figure, 35.32 tok/s observed.
        let gpu = estimate_tokens_per_sec_7b(448.0, true);
        assert!(
            (32.0..=38.0).contains(&gpu),
            "card estimate {gpu} no longer reproduces the measured 35.32 tok/s"
        );
    }

    /// The defect these constants replaced: a processor was priced at 15% of
    /// its memory roofline while reaching ~82%, so every processor-only node
    /// advertised about a fifth of what it does. An Apple M4 Mac mini measures
    /// 69.8 GB/s and was claiming 2.38 tok/s.
    #[test]
    fn a_processor_is_no_longer_priced_at_a_fifth_of_what_it_does() {
        let m4 = estimate_tokens_per_sec_7b(69.8, false);
        assert!(
            m4 > 8.0,
            "an M4-class machine still advertises only {m4} tok/s"
        );
        // The old pair had the ratio backwards: a card is LESS efficient per
        // byte than a processor at batch 1. Pin the direction, not the values.
        let per_byte_cpu = estimate_tokens_per_sec_7b(100.0, false);
        let per_byte_gpu = estimate_tokens_per_sec_7b(100.0, true);
        assert!(
            per_byte_cpu > per_byte_gpu,
            "at equal bandwidth a processor must not be priced below a card"
        );
    }

    #[test]
    fn shard_window_all_fit() {
        let result = compute_optimal_shard_window(4, 500, 3000);
        assert_eq!(result, Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn shard_window_partial_prefers_first_last() {
        // 8 shards, 500MB each, budget for 3
        let result = compute_optimal_shard_window(8, 500, 1500).unwrap();
        assert!(result.contains(&0), "Must include shard 0");
        assert!(result.contains(&7), "Must include last shard");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn shard_window_too_small() {
        let result = compute_optimal_shard_window(8, 500, 100);
        assert_eq!(result, None);
    }

    #[test]
    fn shard_window_single_shard() {
        let result = compute_optimal_shard_window(1, 500, 1000);
        assert_eq!(result, Some(vec![0]));
    }
}

#[cfg(test)]
mod footprint_tests {
    use super::*;

    /// tinyllama-1.1b-q4 as it actually sits on disk: 636 MB of shards, 32k
    /// vocabulary, 22 layers, 4 KV heads, head_dim 64, 2048 context.
    fn tinyllama() -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 667_078_656,
            unquantized_bytes_per_element: None,
            vocab_size: 32_000,
            embedding_length: 2048,
            segment_layers: 22,
            head_count_kv: 4,
            head_dim: 64,
            rope_dim: 64,
            effective_context: 2048,
            is_first: true,
            embedding_gatherable: true,
        }
    }

    /// llama-3.2-1b-q8: only 821 MB of weights but a 128k vocabulary, so the
    /// f32 embedding table alone is larger than the checkpoint.
    fn llama32_1b() -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 860_807_296,
            unquantized_bytes_per_element: None,
            vocab_size: 128_256,
            embedding_length: 2048,
            segment_layers: 16,
            head_count_kv: 8,
            head_dim: 64,
            rope_dim: 64,
            effective_context: 4096,
            is_first: true,
            embedding_gatherable: true,
        }
    }

    /// phi-3.5-mini-q4: 32 layers, full MHA (32 KV heads), head_dim 96, 32k
    /// vocabulary, 131072 declared context capped to 4096.
    fn phi35() -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 2_392_493_952,
            unquantized_bytes_per_element: None,
            vocab_size: 32_064,
            embedding_length: 3072,
            segment_layers: 32,
            head_count_kv: 32,
            head_dim: 96,
            rope_dim: 96,
            effective_context: 4096,
            is_first: true,
            embedding_gatherable: true,
        }
    }

    /// The calibration case, measured on an RTX 3070 once a request had
    /// completed (so the KV cache was actually allocated): **6037 MiB**.
    ///
    /// This is the test that justifies `CUDA_PROCESS_OVERHEAD_BYTES`. The
    /// file-size estimator is 56% low on the same model, which is what made it
    /// useless for admission.
    #[test]
    fn matches_measured_steady_state_on_phi35() {
        // 6037 MiB was measured while the embedding table was resident at f32.
        // The table is now f16, so the REAL steady state falls by exactly the
        // same amount the estimate does — the calibration still holds, but it
        // has to be compared against the adjusted figure or it would be
        // validating the estimate against a measurement of different code.
        const MEASURED_WITH_F32_EMBEDDING: u64 = 6037;
        let embedding_elems = phi35().vocab_size * phi35().embedding_length;
        let saved_mb = embedding_elems * (4 - EMBEDDING_TABLE_BYTES_PER_ELEMENT) / (1024 * 1024);
        let measured = MEASURED_WITH_F32_EMBEDDING - saved_mb;

        let new = estimate_worker_vram_mb(&phi35());
        let err_pct = 100.0 * (new as f64 - measured as f64) / measured as f64;
        assert!(
            err_pct.abs() < 5.0,
            "estimate {new} vs adjusted measurement {measured} is {err_pct:+.1}% — outside 5%"
        );
        const MEASURED: u64 = 6037;
        let old = estimate_model_vram_mb(phi35().quantized_weight_bytes);
        assert!(
            old < MEASURED * 2 / 3,
            "the file-size estimator should be far low: {old} vs {MEASURED}"
        );
    }

    /// A large vocabulary is where the file-size multiple fails worst, because
    /// the dequantized embedding table is invisible to it — on this model that
    /// table is larger than the whole checkpoint.
    #[test]
    fn beats_the_file_size_multiple_on_a_large_vocabulary_model() {
        let i = llama32_1b();
        let old = estimate_model_vram_mb(i.quantized_weight_bytes);
        let new = estimate_worker_vram_mb(&i);
        assert!(new > old, "new {new} must exceed old {old}");

        // The claim being pinned is about a worker that MATERIALISES the table:
        // the f16 embedding alone (128256 * 2048 * 2 = 501 MB) is a large
        // fraction of the 821 MB checkpoint and entirely invisible to a
        // file-size multiple. Asserted against the dense path, because since
        // 2026-08-18 a gathering worker genuinely does not hold it — the gap
        // narrowing there is the optimisation working, not the estimator
        // regressing.
        let dense = estimate_model_resident_bytes(&i, false, i.effective_context)
            .saturating_add(CUDA_PROCESS_OVERHEAD_BYTES)
            / (1024 * 1024);
        assert!(
            dense - old > 500,
            "the embedding term alone should dominate: {old} -> {dense}"
        );
    }

    /// The embedding term dominates on a large vocabulary and must only be
    /// charged to the segment that actually holds it.
    #[test]
    fn the_embedding_table_is_charged_to_the_first_segment_only() {
        // The dense path, where there is a table to charge. 128256*2048*2 = 501 MB.
        let deq = |i: &VramFootprintInputs| {
            estimate_model_resident_bytes(i, false, i.effective_context) / (1024 * 1024)
        };
        let first = deq(&llama32_1b());
        let mut middle = llama32_1b();
        middle.is_first = false;
        let mid = deq(&middle);
        assert!(
            (first - mid) > 450 && (first - mid) < 550,
            "first {first} minus middle {mid} should be ~501 MB"
        );

        // And with rows read on demand there is nothing extra to charge the
        // first segment at all: the quantized bytes are already inside
        // `quantized_weight_bytes`. That the two segments now cost the SAME is
        // the saving, asserted so it cannot silently come back.
        let g = |i: &VramFootprintInputs| {
            estimate_model_resident_bytes(i, true, i.effective_context) / (1024 * 1024)
        };
        assert_eq!(
            g(&llama32_1b()),
            g(&middle),
            "a gathering worker holds no dense table, so holding layer 0 costs no extra"
        );
    }

    /// Context length no longer drives the CPU estimate: since 2026-08-21 a
    /// CPU worker is handed a KV budget and refuses a conversation that would
    /// outgrow it (`CPU_KV_BUDGET_BYTES`), so admission charges the same
    /// typical context the GPU does. Until then the estimator priced the
    /// whole ceiling — correctly, for a worker nothing bounded — which turned
    /// a 2.3 GB model into a "needs 27 GB" refusal at a 32k override.
    ///
    /// **If this test starts failing because the estimate moved with context
    /// again, check that the runtime guard still exists before "fixing" it**:
    /// a typical-context charge with no guard means swapping.
    #[test]
    fn context_length_no_longer_moves_the_cpu_estimate_because_the_worker_is_budgeted() {
        let short = estimate_worker_ram_mb(&llama32_1b());
        let mut long = llama32_1b();
        long.effective_context = 131_072;
        let full = estimate_worker_ram_mb(&long);
        // Only the RoPE table grows with the ceiling — tens of MB, not GB.
        assert!(
            full < short + 200,
            "131k context must not change CPU admission by more than the RoPE table: {short} -> {full}"
        );
        // The ceiling is still computable, for the refusal message to report.
        let ceiling = resident_footprint(&long, true, long.effective_context);
        assert!(
            ceiling.kv_bytes / (1024 * 1024) > 7_000,
            "the whole-ceiling KV figure must still be available on request"
        );
    }

    /// The GPU estimator caps the KV term at [`ADMISSION_KV_CONTEXT`], so
    /// raising `max_seq_len_override` no longer costs a model its place on the
    /// card. What remains context-dependent there is the RoPE table, which
    /// really is precomputed in full — so the difference must be small and
    /// non-zero, not zero.
    ///
    /// The failure this prevents is specific: an agentic client needs ~5000
    /// tokens of context for its tool schema alone, raising it re-priced the
    /// whole KV ceiling, the model no longer fitted and was loaded on the CPU,
    /// and prompt processing went to 396 seconds (reported 2026-08-17).
    #[test]
    fn raising_the_context_no_longer_costs_a_model_its_place_on_the_gpu() {
        let short = estimate_worker_vram_mb(&llama32_1b());
        let mut long = llama32_1b();
        long.effective_context = 131_072;
        let full = estimate_worker_vram_mb(&long);

        // KV at 131k for this model would be ~8 GB. It must not appear.
        assert!(
            full < short + 200,
            "GPU estimate must not re-price the KV ceiling: {short} -> {full}"
        );
        // But RoPE genuinely does scale, so this is not simply ignoring context.
        assert!(
            full > short,
            "the RoPE table still scales with context: {short} -> {full}"
        );
    }

    /// A user who never touched `max_seq_len_override` must be charged exactly
    /// what they were charged before the cap existed — the cap is only allowed
    /// to change admission for people who raised the setting.
    #[test]
    fn the_cap_is_inert_at_the_context_it_was_derived_from() {
        let mut at_cap = llama32_1b();
        at_cap.effective_context = ADMISSION_KV_CONTEXT;
        let capped = estimate_worker_vram_mb(&at_cap);

        let uncapped_bytes = estimate_model_resident_bytes(
            &at_cap,
            at_cap.embedding_gatherable,
            at_cap.effective_context,
        )
        .saturating_add(CUDA_PROCESS_OVERHEAD_BYTES)
            / (1024 * 1024);
        assert_eq!(
            capped, uncapped_bytes,
            "at or below the cap the estimate must be untouched"
        );
    }

    /// A short-context, small-vocabulary model should not be wildly
    /// over-estimated either — over-estimating costs admissions.
    #[test]
    fn a_small_model_is_not_grossly_over_estimated() {
        const MEASURED: u64 = 1145;
        // Measured 2026-07-30 against a worker holding the DEQUANTIZED table,
        // so the calibration is checked against that same path. A gathering
        // worker really does use ~90 MB less on this model (125 MB f16 table
        // against 35 MB quantized), so holding it to this constant would be
        // holding it to a figure for different behaviour.
        let i = tinyllama();
        let est = estimate_model_resident_bytes(&i, false, i.effective_context)
            .saturating_add(CUDA_PROCESS_OVERHEAD_BYTES)
            / (1024 * 1024);
        assert!(
            est >= MEASURED,
            "must not under-estimate ({est} < {MEASURED})"
        );
        assert!(
            est < MEASURED * 2,
            "must not be more than 2x over ({est} vs {MEASURED})"
        );
    }

    /// gemma-2-2b-it-q4_k_m: 1629 MB of shards but a **256,000**-token
    /// vocabulary at hidden size 2304 — the case reported on 2026-08-01.
    fn gemma2_2b(effective_context: u64) -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 1629 * 1024 * 1024,
            unquantized_bytes_per_element: None,
            vocab_size: 256_000,
            embedding_length: 2304,
            segment_layers: 26,
            head_count_kv: 4,
            head_dim: 256,
            rope_dim: 256,
            effective_context,
            is_first: true,
            embedding_gatherable: true,
        }
    }

    /// The reported failure: a 6 GB card (4600 MB budget) refused Gemma 2 2B
    /// with `estimated_mb=5447` and `committed_mb=0` — nothing else was using
    /// the GPU at all.
    ///
    /// The estimate was not wrong; the allocation genuinely was that large,
    /// and 2250 MB of it was one f32 embedding table — more than the 1629 MB
    /// the rest of the model weighs. At f16 the same model fits, which is the
    /// whole point: a modest laptop GPU can serve it again.
    #[test]
    fn gemma2_2b_fits_a_6gb_card_now_that_the_embedding_is_f16() {
        const BUDGET_MB: u64 = 4600;
        for ctx in [2048, 6144] {
            let est = estimate_worker_vram_mb(&gemma2_2b(ctx));
            assert!(
                est <= BUDGET_MB,
                "gemma-2-2b at ctx {ctx} estimates {est} MB, over the {BUDGET_MB} MB budget \
                 that made it fall back to the CPU"
            );
        }
    }

    /// Guards the arithmetic itself against a silent regression: the f32
    /// table was 2250 MB, which alone exceeded half the card.
    #[test]
    fn the_gemma2_embedding_table_is_no_longer_the_largest_single_term() {
        let i = gemma2_2b(6144);
        let embedding_mb =
            i.vocab_size * i.embedding_length * EMBEDDING_TABLE_BYTES_PER_ELEMENT / (1024 * 1024);
        assert_eq!(embedding_mb, 1125, "256000 x 2304 x 2 bytes");
        assert!(
            embedding_mb < i.quantized_weight_bytes / (1024 * 1024),
            "the embedding table must no longer outweigh the model's own weights"
        );
    }

    /// Degenerate geometry must not panic or overflow.
    #[test]
    fn degenerate_inputs_are_safe() {
        let z = VramFootprintInputs {
            quantized_weight_bytes: 0,
            unquantized_bytes_per_element: None,
            vocab_size: 0,
            embedding_length: 0,
            segment_layers: 0,
            head_count_kv: 0,
            head_dim: 0,
            rope_dim: 0,
            effective_context: 0,
            embedding_gatherable: false,
            is_first: true,
        };
        // Just the process overhead.
        assert_eq!(
            estimate_worker_vram_mb(&z),
            CUDA_PROCESS_OVERHEAD_BYTES / (1024 * 1024)
        );
        let huge = VramFootprintInputs {
            quantized_weight_bytes: u64::MAX,
            unquantized_bytes_per_element: None,
            vocab_size: u64::MAX,
            embedding_length: u64::MAX,
            segment_layers: u64::MAX,
            head_count_kv: u64::MAX,
            head_dim: u64::MAX,
            rope_dim: u64::MAX,
            effective_context: u64::MAX,
            embedding_gatherable: false,
            is_first: true,
        };
        let _ = estimate_worker_vram_mb(&huge); // must not panic
        let _ = estimate_worker_ram_mb(&huge); // the CPU sibling likewise
    }

    /// The two estimators must agree about the model and differ only by terms
    /// that are genuinely device-specific. If they disagree about anything
    /// else, one of the two budgets is wrong.
    ///
    /// There are exactly TWO such terms:
    ///
    /// 1. Per-process overhead — a CPU worker establishes no device context.
    /// 2. The KV ceiling. A GPU worker charges a capped context because it has
    ///    a runtime head-room check that refuses gracefully when a conversation
    ///    outgrows the card; a CPU worker has none, so it prices the whole
    ///    ceiling. See [`ADMISSION_KV_CONTEXT`].
    ///
    /// The embedding table is NOT one of them: both devices now read its rows
    /// on demand. It was briefly device-specific, between the CPU gather
    /// landing and the CUDA one, and that is exactly the kind of asymmetry this
    /// test exists to force someone to declare.
    ///
    /// Stated as an equation rather than a tolerance so that adding a third
    /// device-specific term fails here and has to be declared.
    #[test]
    fn ram_and_vram_estimates_differ_only_by_declared_device_specific_terms() {
        let overhead_mb =
            (CUDA_PROCESS_OVERHEAD_BYTES - CPU_PROCESS_OVERHEAD_BYTES) / (1024 * 1024);
        for i in [tinyllama(), llama32_1b(), phi35(), qwen05b_f16()] {
            let vram = estimate_worker_vram_mb(&i);
            let ram = estimate_worker_ram_mb(&i);
            // KV the CPU estimator prices and the GPU one does not, i.e. the
            // context above the cap. Zero for every fixture at or below it,
            // which is what makes those fixtures a control on the overhead
            // term alone.
            let uncapped = i
                .effective_context
                .saturating_sub(i.effective_context.min(ADMISSION_KV_CONTEXT));
            let kv_gap_mb =
                i.segment_layers * 2 * i.head_count_kv * i.head_dim * uncapped * 4 / (1024 * 1024);
            // Each estimator truncates its own total to MB, so the difference
            // of two rounded figures can sit 1 MB off the rounded difference.
            // The terms being pinned are tens to hundreds of MB, so a missing
            // one is never mistaken for this.
            let expected = overhead_mb + kv_gap_mb;
            assert!(
                vram.abs_diff(ram).abs_diff(expected) <= 1,
                "the two may differ only by process overhead and the uncapped \
                 KV ceiling: vram {vram} - ram {ram} vs expected {expected}"
            );
        }
    }

    /// qwen2.5-0.5b as HuggingFace ships its "F16" variant: 948 MB of shards
    /// that are NOT quantized, 151k vocabulary, 24 layers.
    fn qwen05b_f16() -> VramFootprintInputs {
        VramFootprintInputs {
            quantized_weight_bytes: 994_156_864,
            unquantized_bytes_per_element: Some(2),
            vocab_size: 151_936,
            embedding_length: 896,
            segment_layers: 24,
            head_count_kv: 2,
            head_dim: 64,
            rope_dim: 64,
            effective_context: 4096,
            is_first: true,
            embedding_gatherable: false,
        }
    }

    /// An unquantized checkpoint costs TWICE its file size on the device.
    ///
    /// candle's `QMatMul::from_arc` dequantizes an F16 / BF16 / F32 tensor to a
    /// dense f32 one rather than keeping it quantized, so the on-disk figure —
    /// which is the right answer for every Q* checkpoint — understates the
    /// dominant term by 2x here. This is an ADMISSION estimate, so being low is
    /// the direction that ends in `CUDA_ERROR_OUT_OF_MEMORY`.
    ///
    /// Latent until 2026-08-10: an unquantized GGUF could not load on a CUDA
    /// node at all (`unsupported dtype for dequantize F16`), so nothing ever
    /// reached the gate with one.
    #[test]
    fn an_unquantized_checkpoint_is_charged_twice_its_file_size() {
        let f16 = qwen05b_f16();
        let as_if_quantized = VramFootprintInputs {
            unquantized_bytes_per_element: None,
            ..f16
        };

        let with = estimate_worker_vram_mb(&f16);
        let without = estimate_worker_vram_mb(&as_if_quantized);
        let file_mb = f16.quantized_weight_bytes / (1024 * 1024);

        assert_eq!(
            with - without,
            file_mb,
            "an f16 checkpoint must be charged one extra copy of its file size"
        );
        assert!(
            with > file_mb * 2,
            "estimate {with} MB must exceed twice the {file_mb} MB checkpoint \
             (weights are doubled, and the embedding table is on top)"
        );
    }

    /// F32 on disk is already the dtype candle materialises, so it must NOT be
    /// scaled — only the narrower types expand. Getting this wrong would refuse
    /// models that fit.
    #[test]
    fn an_f32_checkpoint_is_not_charged_twice() {
        let f32_ckpt = VramFootprintInputs {
            unquantized_bytes_per_element: Some(4),
            ..qwen05b_f16()
        };
        let quantized = VramFootprintInputs {
            unquantized_bytes_per_element: None,
            ..qwen05b_f16()
        };
        assert_eq!(
            estimate_worker_vram_mb(&f32_ckpt),
            estimate_worker_vram_mb(&quantized),
            "f32 weights are already dense f32 on the device"
        );
    }

    /// A CPU worker establishes no device context, so its baseline is lower —
    /// but still counted, because the process itself is not free.
    #[test]
    fn the_cpu_estimate_still_counts_a_process_baseline() {
        let est = estimate_worker_ram_mb(&tinyllama());
        let weights_only = tinyllama().quantized_weight_bytes / (1024 * 1024);
        assert!(
            est > weights_only,
            "estimate {est} MB must exceed the raw weights {weights_only} MB"
        );
    }

    /// The number a user called "10x the model's size": phi-3.5-mini (32
    /// layers, 32 KV heads — MHA, no GQA — head_dim 96) at the 32768-token
    /// context they had configured is 0.75 MB of f32 KV per token, 24 GB in
    /// all, on top of 2.3 GB of weights. Correct, and the refusal must SAY so.
    #[test]
    fn a_cpu_refusal_itemises_weights_kv_and_context() {
        let i = VramFootprintInputs {
            quantized_weight_bytes: 2284 * 1024 * 1024,
            unquantized_bytes_per_element: None,
            vocab_size: 32064,
            embedding_length: 3072,
            segment_layers: 32,
            head_count_kv: 32,
            head_dim: 96,
            rope_dim: 96,
            effective_context: 32768,
            is_first: true,
            embedding_gatherable: true,
        };
        let f = cpu_footprint(&i);
        assert_eq!(f.weights_bytes / (1024 * 1024), 2284);
        // Admission now charges the SAME typical context the GPU does; the
        // 32k ceiling is bounded at run time by the worker's KV budget.
        assert_eq!(f.kv_context, ADMISSION_KV_CONTEXT);
        assert_eq!(
            f.kv_bytes / (1024 * 1024),
            32 * 2 * 32 * 96 * ADMISSION_KV_CONTEXT * 4 / (1024 * 1024)
        );
        let total_mb = (f.total_bytes() + CPU_PROCESS_OVERHEAD_BYTES) / (1024 * 1024);
        assert!(
            total_mb < 13370,
            "the model that was refused at 27 GB is admitted: {total_mb} MB"
        );
        let msg = describe_cpu_refusal(
            "phi-3.5-mini-instruct.q4-k-m",
            &f,
            32768,
            ContextSource::Override,
            13370,
            "`resources.max_ram_mb` is 18000 MB, limited to 70% of the 19100 MB that was free when the node started, so loading a model cannot push it into swap",
            0,
        );
        assert!(msg.contains("2284 MB of weights"), "{msg}");
        assert!(
            msg.contains(&format!(
                "{}-token conversation it is admitted at (its full context is 32768 tokens, set by `inference.max_seq_len_override`",
                ADMISSION_KV_CONTEXT
            )),
            "{msg}"
        );
        assert!(
            msg.contains(
                "budget allows 13370 MB (`resources.max_ram_mb` is 18000 MB, limited to 70%"
            ),
            "{msg}"
        );
        assert!(
            !msg.contains("Lower `inference.max_seq_len_override`"),
            "the override is no longer what stops the model fitting: {msg}"
        );
        // What the old message priced — the whole ceiling — and why it read as 10x.
        let ceiling = resident_footprint(&i, true, 32768);
        assert_eq!(ceiling.kv_bytes / (1024 * 1024), 24576);
    }

    #[test]
    fn the_budget_note_names_what_limited_it() {
        // The reporter's machine: max_ram_mb 18000, 14773 MB free at the time.
        let b = RamBudget::from_machine(18000, 18000, 32000, 14773);
        assert_eq!(b.live_headroom_mb, 14773 / 100 * 70);
        // Refused by the LIVE headroom, not the cap — and the note says so.
        let (figure, note) = b.limiting_figure(0, 13149);
        assert_eq!(figure, b.live_headroom_mb);
        // 14773 / 100 * 70 = 10290 (integer arithmetic — which is also why the
        // reporter's second figure was a round 10500: 15000 MB free at boot).
        assert_eq!(b.live_headroom_mb, 10290);
        assert!(
            note.starts_with("`resources.max_ram_mb` is 18000 MB; right now 14773 MB of memory is free and SwarmLLM uses at most 70% of that (10290 MB)"),
            "{note}"
        );
        // Refused by the cap when the cap is what it hit.
        let (figure, note) = b.limiting_figure(12000, 7000);
        assert_eq!(figure, 18000);
        assert_eq!(note, "`resources.max_ram_mb` is 18000 MB");
        // Auto-sized cap names its derivation.
        assert_eq!(
            RamBudget::from_machine(12800, 0, 16000, 15000).cap_source(),
            "auto-sized to 12800 MB from 16000 MB of RAM"
        );
    }

    /// The budget is a snapshot taken at admission: the same cap judges
    /// differently as free memory changes, which is the point — a restart
    /// while memory was busy must not set the verdict for the rest of the
    /// daemon's life (external report, 2026-08-21: 13370 MB one day, 10500 MB
    /// the next, 14773 MB actually free).
    #[test]
    fn the_live_budget_follows_free_memory_not_the_moment_the_node_started() {
        let busy = RamBudget::from_machine(18000, 18000, 32000, 15000);
        let quiet = RamBudget::from_machine(18000, 18000, 32000, 26000);
        assert!(
            !busy.allows(0, 13149),
            "10500 MB of headroom refuses a 13 GB model"
        );
        assert!(quiet.allows(0, 13149), "18200 MB of headroom admits it");
        // The cap still binds regardless of how much is free.
        assert!(!quiet.allows(6000, 13149));
        // Unreadable free memory judges by the cap alone — never invents a limit.
        let unknown = RamBudget::from_machine(18000, 18000, 0, 0);
        assert!(unknown.allows(0, 17999));
        assert!(!unknown.allows(0, 18001));
        assert_eq!(RamBudget::cap_only(6000).live_headroom_mb, u64::MAX);
        // KV headroom for the admitted model: the tighter of the two limits —
        // here the 18000 cap, just under the 18200 MB live headroom…
        assert_eq!(quiet.headroom_after(13149, 13149), 18000 - 13149);
        // …and the live term when THAT is tighter.
        assert_eq!(busy.headroom_after(5000, 5000), 10500 - 5000);
    }
}
