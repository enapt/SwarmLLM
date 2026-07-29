//! VRAM-aware sizing of the KV cache and RoPE tables.
//!
//! candle's `KvCache` allocates its whole `[B, H, max_seq_len, D]` buffer on
//! the first append — it does not grow with the conversation. So `max_seq_len`
//! is not a limit, it is an *allocation*, and sizing it from the GGUF's
//! declared `context_length` charges every user the worst case up front.
//!
//! That is ruinous on modern long-context models. Measured on an RTX 3070
//! (8 GB) against `llama-3.2-1b-instruct-q8-0` — 1.3 GB of weights, but
//! `context_length = 131072`:
//!
//! | | KV sized to declared ctx | KV sized to 4096 |
//! |---|---|---|
//! | VRAM | 7958 MiB (97% of card) | 3110 MiB |
//! | Throughput | 3.7 tok/s | 46–59 tok/s |
//!
//! Same model, same GPU, ~14x. Two models of near-identical size can differ by
//! 93x in KV cache purely because one declares 2048 and the other 131072
//! (tinyllama-1.1b: 88 MiB; llama-3.2-1b: 8 GiB).
//!
//! Where it does not simply OOM, it silently degrades: the driver's
//! system-memory fallback (WSL2 and Windows) absorbs the overflow, so the
//! model still answers, `device=Cuda` still appears in the log, and only
//! external GPU profiling shows the card sitting idle. Reported 2026-07-29
//! as "GPU loads the model but never computes on it".
//!
//! So the loader asks this module what actually fits. The answer is capped to
//! the declared context (never raised above what the model supports) and
//! floored at [`MIN_AUTO_CONTEXT`] so a pathological budget cannot produce an
//! unusable model. `inference.max_seq_len_override` still wins outright when
//! set — this only supplies a sane default where there was none.

/// Fraction of free VRAM the weights plus KV cache may claim, in percent.
///
/// The remainder absorbs activations, cuBLAS workspaces, allocator
/// fragmentation, and — the case that motivated the margin — a *second* model
/// loading alongside this one. The reported failure had a resident TinyLlama
/// turn a 3B load into a hard OOM, after which the model was pinned to CPU for
/// the rest of the daemon's run.
pub(crate) const VRAM_HEADROOM_PCT: u64 = 85;

/// Never auto-cap below this many tokens. A model that cannot hold even this
/// is not usefully loadable, and a tiny context produces confusing truncation
/// errors rather than an honest failure — better to let the load OOM and fall
/// back to CPU, which the caller already handles.
pub(crate) const MIN_AUTO_CONTEXT: usize = 512;

/// KV-cache bytes one sequence position costs across a whole segment.
///
/// K and V are cached *before* the GQA repeat (see `LayerWeights::forward`,
/// which reshapes to `n_kv_head` and appends that), so the per-token cost is
/// driven by `n_kv_head`, not `n_head`. Both are stored f32.
pub(crate) fn kv_bytes_per_token(layers: usize, k_elems: usize, v_elems: usize) -> u64 {
    const F32: u64 = std::mem::size_of::<f32>() as u64;
    (layers as u64)
        .saturating_mul((k_elems as u64).saturating_add(v_elems as u64))
        .saturating_mul(F32)
}

/// Per-token K and V element counts for the standard (MHA/GQA) attention path.
pub(crate) fn standard_kv_elems(head_count_kv: usize, head_dim: usize) -> (usize, usize) {
    let per = head_count_kv.saturating_mul(head_dim);
    (per, per)
}

/// Per-token K and V element counts for DeepSeek-style MLA, which caches the
/// *decompressed* K and V at full head count with asymmetric widths
/// (`key_length` != `value_length`) — see `MlaWeights::forward_mla`.
pub(crate) fn mla_kv_elems(
    head_count: usize,
    key_length: usize,
    value_length: usize,
) -> (usize, usize) {
    (
        head_count.saturating_mul(key_length),
        head_count.saturating_mul(value_length),
    )
}

/// The largest context length whose KV cache fits alongside `weight_bytes` in
/// `free_vram_bytes`, or `None` when the declared context already fits.
///
/// `None` means "change nothing" — the caller keeps the GGUF value. Returning
/// `Some` always indicates a reduction, so callers can log unconditionally on
/// `Some` without re-comparing.
pub(crate) fn fit_context_to_budget(
    declared_context: usize,
    kv_bytes_per_token: u64,
    weight_bytes: u64,
    free_vram_bytes: u64,
) -> Option<usize> {
    if declared_context == 0 || kv_bytes_per_token == 0 {
        return None;
    }

    let usable = free_vram_bytes / 100 * VRAM_HEADROOM_PCT;

    // Weights alone overflow the budget. Capping context cannot rescue this
    // load — it will OOM and the caller falls back to CPU — but shrink anyway
    // so the KV cache is not what tips an otherwise-recoverable margin over.
    let kv_budget = usable.saturating_sub(weight_bytes);

    let fits = (kv_budget / kv_bytes_per_token) as usize;
    // The floor can never exceed the ceiling: a model declaring a context
    // below MIN_AUTO_CONTEXT would otherwise make `clamp` panic, and this runs
    // inside the model worker where a panic takes the whole subprocess down.
    let floor = MIN_AUTO_CONTEXT.min(declared_context);
    let capped = fits.clamp(floor, declared_context);

    (capped < declared_context).then_some(capped)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The real geometry behind the 2026-07-29 report: 8 GB card, a 1.3 GB
    /// model whose GGUF declares a 128K context. Sizing the KV cache to the
    /// declared value asks for 8 GiB on top of the weights.
    #[test]
    fn long_context_model_on_a_small_card_is_capped() {
        // llama-3.2-1b-instruct-q8-0: 16 layers, 8 KV heads, head_dim 64.
        let (k, v) = standard_kv_elems(8, 64);
        let per_token = kv_bytes_per_token(16, k, v);
        assert_eq!(per_token, 16 * (512 + 512) * 4);

        let free = 7_000u64 * 1024 * 1024; // ~7 GiB free on an 8 GiB card
        let weights = 1_300u64 * 1024 * 1024;
        let capped = fit_context_to_budget(131_072, per_token, weights, free)
            .expect("128K must not fit beside 1.3 GB of weights on a 7 GiB budget");

        assert!(capped < 131_072);
        // Whatever we pick must actually fit inside the headroom.
        assert!(weights + per_token * capped as u64 <= free / 100 * VRAM_HEADROOM_PCT);
    }

    /// The control from the same investigation: tinyllama-1.1b declares 2048,
    /// so its KV cache is 88 MiB and nothing should change. A cap here would
    /// be a regression for every short-context model.
    #[test]
    fn short_context_model_is_left_alone() {
        let (k, v) = standard_kv_elems(4, 64);
        let per_token = kv_bytes_per_token(22, k, v);
        let free = 7_000u64 * 1024 * 1024;
        let weights = 700u64 * 1024 * 1024;
        assert_eq!(fit_context_to_budget(2048, per_token, weights, free), None);
    }

    /// A big card should serve the full declared context untouched — the cap
    /// exists for constrained hardware, not as a blanket ceiling.
    #[test]
    fn large_card_keeps_full_declared_context() {
        let (k, v) = standard_kv_elems(8, 128);
        let per_token = kv_bytes_per_token(28, k, v);
        let free = 80_000u64 * 1024 * 1024; // 80 GiB
        let weights = 2_000u64 * 1024 * 1024;
        assert_eq!(
            fit_context_to_budget(131_072, per_token, weights, free),
            None
        );
    }

    /// Never hand back something unusable, even when the weights have already
    /// eaten the entire budget.
    #[test]
    fn floors_at_min_rather_than_zero() {
        let (k, v) = standard_kv_elems(8, 128);
        let per_token = kv_bytes_per_token(28, k, v);
        let free = 2_000u64 * 1024 * 1024;
        let weights = 4_000u64 * 1024 * 1024; // weights alone exceed the card
        assert_eq!(
            fit_context_to_budget(131_072, per_token, weights, free),
            Some(MIN_AUTO_CONTEXT)
        );
    }

    /// The cap is a reduction or nothing: it must never raise a model's
    /// context above what its GGUF declares, however much VRAM is free.
    ///
    /// The `declared < MIN_AUTO_CONTEXT` rows are not hypothetical padding —
    /// they panicked `clamp` (min > max) before the floor was itself clamped,
    /// which inside the model worker means the subprocess dies on load.
    #[test]
    fn never_raises_above_declared() {
        let (k, v) = standard_kv_elems(2, 64);
        let per_token = kv_bytes_per_token(4, k, v);
        for free in [0u64, 1 << 20, 80_000u64 * 1024 * 1024] {
            for declared in [1usize, 128, 511, 512, 2048, 8192] {
                match fit_context_to_budget(declared, per_token, 0, free) {
                    None => {}
                    Some(c) => assert!(c <= declared, "raised {declared} to {c}"),
                }
            }
        }
    }

    /// MLA caches decompressed K/V at full head count with asymmetric widths,
    /// so it is far more expensive per token than GQA at the same head_dim.
    /// Getting this wrong would under-estimate DeepSeek's cache and re-create
    /// the exact bug on those models.
    #[test]
    fn mla_costs_more_per_token_than_gqa() {
        let (gk, gv) = standard_kv_elems(8, 128);
        let (mk, mv) = mla_kv_elems(128, 192, 128);
        assert!(
            kv_bytes_per_token(1, mk, mv) > kv_bytes_per_token(1, gk, gv),
            "MLA per-token cost must exceed GQA's"
        );
    }

    /// Degenerate inputs must not panic or divide by zero.
    #[test]
    fn zero_inputs_are_inert() {
        assert_eq!(fit_context_to_budget(0, 1024, 0, 1 << 30), None);
        assert_eq!(fit_context_to_budget(4096, 0, 0, 1 << 30), None);
        // No GPU information at all → caller passes 0 free; we still must not
        // panic, and must not silently claim the full context fits.
        assert_eq!(
            fit_context_to_budget(131_072, 4096, 0, 0),
            Some(MIN_AUTO_CONTEXT)
        );
    }
}
