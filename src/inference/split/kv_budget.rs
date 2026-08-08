//! VRAM-aware sizing of the KV cache and RoPE tables.
//!
//! `max_seq_len` bounds how large a KV cache can GROW, and the conversation
//! guard in `forward_inner_impl` is what holds it there. So sizing it from the
//! GGUF's declared `context_length` lets a long conversation grow until it
//! exhausts the card.
//!
//! **This module's premise changed on 2026-08-07 and its conclusion did not.**
//! Until then a cache allocated its whole `[B, H, max_seq_len, D]` buffer on
//! the first append, so `max_seq_len` was not a limit but an immediate
//! *allocation* — every user charged the worst case from token one.
//! `layers::new_kv_cache` now passes a growth quantum instead, so the
//! allocation tracks the conversation. That makes the numbers below a *ceiling*
//! rather than a reservation, and makes this module MORE load-bearing, not
//! less: it is now the only thing standing between on-demand growth and an OOM
//! part-way through a long conversation.
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

/// Bytes available for KV cache once the weights are resident.
///
/// The same headroom arithmetic [`fit_context_to_budget`] uses, exposed on its
/// own because it is now the RUNTIME budget rather than only a load-time
/// sizing input: the loader records it on the model and every forward checks
/// against it before claiming another growth quantum.
pub(crate) fn kv_headroom_bytes(weight_bytes: u64, free_vram_bytes: u64) -> u64 {
    (free_vram_bytes / 100 * VRAM_HEADROOM_PCT).saturating_sub(weight_bytes)
}

/// Whether this forward will claim another growth quantum of KV cache.
///
/// A cache grows only when a conversation crosses a quantum boundary, so this
/// is the ONLY moment a headroom check can matter — and checking on every
/// token instead would walk the whole cache store per generated token for an
/// answer that is almost always "no".
///
/// `index_pos == 0` counts as growth: a request with nothing cached claims its
/// first quantum.
pub(crate) fn forward_claims_new_quantum(
    index_pos: usize,
    total_seq: usize,
    quantum: usize,
) -> bool {
    let q = quantum.max(1);
    total_seq.div_ceil(q) > index_pos.div_ceil(q)
}

/// Would claiming another quantum push total KV occupancy past the budget?
///
/// `in_use_bytes` is what the whole store already holds — every request on
/// this worker, not just the one asking. That is the point: the load-time
/// clamp this replaces sized for ONE conversation at full length and so did
/// not bound concurrency at all, while a growth quantum claimed by the fourth
/// concurrent request is exactly as real as one claimed by the first.
///
/// This is Head-Room Admission (vLLM). The difference here is what happens on
/// refusal: vLLM must preempt and recompute because it has nowhere else to
/// send the work, whereas a refused request in a swarm can be served by a
/// peer — so refusing early is cheap and correct.
pub(crate) fn quantum_exceeds_headroom(
    budget_bytes: u64,
    in_use_bytes: u64,
    kv_bytes_per_token: u64,
    quantum: usize,
) -> bool {
    let claim = kv_bytes_per_token.saturating_mul(quantum as u64);
    in_use_bytes.saturating_add(claim) > budget_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// MLA caches decompressed K/V at full head count with asymmetric widths,
    /// so it is far more expensive per token than GQA at the same head_dim.
    /// Getting this wrong would under-estimate DeepSeek's cache.
    #[test]
    fn mla_costs_more_per_token_than_gqa() {
        let (gk, gv) = standard_kv_elems(8, 128);
        let (mk, mv) = mla_kv_elems(128, 192, 128);
        assert!(
            kv_bytes_per_token(1, mk, mv) > kv_bytes_per_token(1, gk, gv),
            "MLA per-token cost must exceed GQA's"
        );
    }

    /// The real geometry behind the 2026-07-29 report: an 8 GB card and a
    /// 1.3 GB model declaring a 128K context. The budget must come out well
    /// short of that context — which is now a warning at load and a refusal
    /// only if a conversation actually gets that long, where it used to be a
    /// permanent cut to everyone's context.
    #[test]
    fn a_long_context_model_on_a_small_card_gets_a_short_budget() {
        let (k, v) = standard_kv_elems(8, 64);
        let per_token = kv_bytes_per_token(16, k, v);
        let free = 7_000u64 * 1024 * 1024;
        let weights = 1_300u64 * 1024 * 1024;
        let budget = kv_headroom_bytes(weights, free);
        let affordable = budget / per_token;
        assert!(affordable > 0, "the card must afford SOME conversation");
        assert!(
            affordable < 131_072,
            "a 128K conversation must not fit: affordable {affordable}"
        );
        // And the budget must respect the headroom margin.
        assert!(budget + weights <= free / 100 * VRAM_HEADROOM_PCT + weights);
    }

    /// Weights that already exceed the headroom leave a zero budget rather
    /// than wrapping around to an enormous one.
    #[test]
    fn weights_larger_than_the_card_give_a_zero_budget() {
        assert_eq!(kv_headroom_bytes(8 << 30, 4 << 30), 0);
    }

    /// The check must fire exactly when a conversation crosses a quantum
    /// boundary, and not otherwise — that is what keeps it off the per-token
    /// path.
    #[test]
    fn growth_is_detected_only_at_quantum_boundaries() {
        let q = 512;
        // A fresh request claims its first quantum.
        assert!(forward_claims_new_quantum(0, 1, q));
        assert!(forward_claims_new_quantum(0, 512, q));
        // Decoding inside the first quantum claims nothing more.
        assert!(!forward_claims_new_quantum(1, 2, q));
        assert!(!forward_claims_new_quantum(510, 511, q));
        assert!(!forward_claims_new_quantum(511, 512, q));
        // Crossing into the second does.
        assert!(forward_claims_new_quantum(512, 513, q));
        assert!(!forward_claims_new_quantum(513, 514, q));
        // A prefill spanning several quanta counts as growth.
        assert!(forward_claims_new_quantum(0, 2000, q));
    }

    /// A zero quantum must not divide by zero.
    #[test]
    fn a_zero_quantum_is_treated_as_one() {
        assert!(forward_claims_new_quantum(0, 1, 0));
    }

    /// The budget is against the WHOLE store, not one request — the case the
    /// load-time clamp it replaces could not see at all.
    #[test]
    fn headroom_counts_every_request_not_just_this_one() {
        let per_token = 1_000u64;
        let q = 512usize;
        let claim = per_token * q as u64; // 512 KB per quantum
        let budget = claim * 4; // room for four quanta in total

        // Nothing in use: fine.
        assert!(!quantum_exceeds_headroom(budget, 0, per_token, q));
        // Three already held: the fourth still fits.
        assert!(!quantum_exceeds_headroom(budget, claim * 3, per_token, q));
        // Four already held — by any mix of requests — and the fifth does not.
        assert!(quantum_exceeds_headroom(budget, claim * 4, per_token, q));
    }

    /// Arithmetic near the limits must saturate rather than wrap: wrapping
    /// would turn "no memory" into "unlimited memory".
    #[test]
    fn headroom_arithmetic_saturates() {
        assert!(quantum_exceeds_headroom(0, u64::MAX, 1, 1));
        assert!(quantum_exceeds_headroom(10, u64::MAX, u64::MAX, usize::MAX));
        // A zero per-token cost cannot claim anything, so it never refuses.
        assert!(!quantum_exceeds_headroom(0, 0, 0, 512));
    }
}
