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
//! **This module no longer shortens anyone's context, and must not start
//! again.** It once did: a load-time clamp sized the context to what the card
//! could hold, which shrank every user's conversation so that a single
//! full-length one would fit, and bounded concurrency not at all. That clamp
//! (`fit_context_to_budget`, `MIN_AUTO_CONTEXT`) was deleted on 2026-08-08 and
//! this doc described it for ten days afterwards.
//!
//! What replaced it is [`claim_exceeds_headroom`]: the loader records
//! [`kv_headroom_bytes`] on the model, and every forward checks the positions it
//! is about to claim against that budget before claiming them, refusing with a
//! `ServiceUnavailable` (503) that re-routes to a peer rather than growing into
//! an OOM. So the ceiling is enforced per REQUEST, at the moment memory is
//! actually taken, instead of being pre-paid by everyone at load.

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
///
/// `mirrored` adds the f16 BSHD mirror `LayerKv` keeps for the CUDA flash
/// kernel — the same elements again at half the width, so 1.5x in total.
/// **It has to be counted here.** This figure is what the runtime head-room
/// check charges a request for, so omitting the mirror would let a GQA model
/// on a GPU claim 50% more VRAM than the budget believes and then run out for
/// real — trading a clean 503 that reroutes to a peer for an OOM. See
/// `layers::model_wants_kv_mirror` for which models carry one.
pub(crate) fn kv_bytes_per_token(
    layers: usize,
    k_elems: usize,
    v_elems: usize,
    mirrored: bool,
) -> u64 {
    const F32: u64 = std::mem::size_of::<f32>() as u64;
    const F16: u64 = std::mem::size_of::<half::f16>() as u64;
    let per_elem = if mirrored { F32 + F16 } else { F32 };
    (layers as u64)
        .saturating_mul((k_elems as u64).saturating_add(v_elems as u64))
        .saturating_mul(per_elem)
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

/// Sequence positions this forward will NEWLY reserve, or 0 if it fits inside
/// what the cache already has.
///
/// A cache grows only when a conversation crosses a quantum boundary, so this
/// is 0 for almost every decode step — which is what keeps the headroom check
/// off the per-token path.
///
/// **It returns positions, not "one quantum".** A prefill jumps many quanta at
/// once: 0 -> 5000 tokens at a 512 quantum reserves ten of them in a single
/// forward. Charging one would under-count the largest claim any request
/// makes by an order of magnitude, which is precisely the case the budget
/// exists to catch.
pub(crate) fn positions_claimed(index_pos: usize, total_seq: usize, quantum: usize) -> usize {
    let q = quantum.max(1);
    total_seq
        .div_ceil(q)
        .saturating_sub(index_pos.div_ceil(q))
        .saturating_mul(q)
}

/// Would reserving `positions` more push total KV occupancy past the budget?
///
/// `in_use_bytes` is what the whole store already holds — every request on
/// this worker, not just the one asking. That is the point: the load-time
/// clamp this replaces sized for ONE conversation at full length and so did
/// not bound concurrency at all.
///
/// This is Head-Room Admission (vLLM). The difference here is what happens on
/// refusal: vLLM must preempt and recompute because it has nowhere else to
/// send the work, whereas a refused request in a swarm can be served by a
/// peer — so refusing early is cheap and correct.
pub(crate) fn claim_exceeds_headroom(
    budget_bytes: u64,
    in_use_bytes: u64,
    kv_bytes_per_token: u64,
    positions: usize,
) -> bool {
    let claim = kv_bytes_per_token.saturating_mul(positions as u64);
    in_use_bytes.saturating_add(claim) > budget_bytes
}

/// What to do about a prompt BEFORE its prefill starts (gotcha #440).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptAdmission {
    /// Live cache plus cached prompts plus this prompt fit the budget.
    Fits,
    /// It fits once this many bytes of cached prompts are evicted.
    EvictBytes(u64),
    /// It does not fit even with every cached prompt gone; refuse now,
    /// short by this many bytes.
    Refuse { short_by: u64 },
}

/// Decide once, for the whole prompt, whether its KV can live on the device.
///
/// **Why a whole-prompt decision exists beside the per-chunk guard.** The
/// per-chunk guard (`claim_exceeds_headroom`) sees only what the KV store
/// holds. The prefix cache's snapshots are the same device memory — up to
/// the cache's byte cap, and one entry may exceed it by design — and were
/// charged nowhere, while the budget was sized from free memory at LOAD,
/// before any snapshot existed. Measured on an 8 GB card (gotcha #439/#440):
/// the second 6.4k-token request found ~300 MB free, its 2.6 GB cache was
/// allocated anyway — WSL2 hands out host-backed memory rather than failing —
/// and every decode step then read the cache over PCIe: 3-5 tok/s where an
/// empty card did 19-33. Nothing refused, nothing logged.
///
/// The cached prompts are a cache: reconstructible, and worth less than the
/// request in hand. So the order is fit → evict cached prompts → refuse,
/// and the refusal comes at token 0 rather than at chunk N with a partial
/// cache already built (the ratchet of gotcha #387, one level up).
pub(crate) fn admit_prompt(
    budget_bytes: u64,
    live_bytes: u64,
    cached_prompt_bytes: u64,
    kv_bytes_per_token: u64,
    positions: usize,
) -> PromptAdmission {
    let claim = kv_bytes_per_token.saturating_mul(positions as u64);
    let total = live_bytes
        .saturating_add(cached_prompt_bytes)
        .saturating_add(claim);
    if total <= budget_bytes {
        return PromptAdmission::Fits;
    }
    let excess = total - budget_bytes;
    if excess <= cached_prompt_bytes {
        PromptAdmission::EvictBytes(excess)
    } else {
        PromptAdmission::Refuse {
            short_by: excess - cached_prompt_bytes,
        }
    }
}

/// How much of a finished prompt to keep as a prefix-cache snapshot, so that
/// the snapshot fits BESIDE the request's own live cache (gotcha #440).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SnapshotPlan {
    /// Bytes of older cached prompts to evict first.
    pub(crate) evict_bytes: u64,
    /// Positions to snapshot after that; 0 means do not snapshot.
    pub(crate) positions: usize,
}

/// A snapshot is a COPY of the request's cache, so on a device where the live
/// cache alone takes a third of the budget the whole-prompt snapshot cannot
/// fit beside it — measured: admission evicted the previous snapshot, the
/// request started warm, and the snapshot taken after its prefill put the
/// card over the top again, so the next growth quantum was refused
/// mid-reply and the tokens before it decoded at 3-6 tok/s. Older cached
/// prompts go first; then the snapshot is cut to what fits, because a
/// partial prefix still saves its length of prefill on the next turn
/// (`lookup` narrows a longer entry to the shared length).
pub(crate) fn plan_snapshot(
    budget_bytes: u64,
    live_bytes: u64,
    cached_prompt_bytes: u64,
    kv_bytes_per_token: u64,
    prompt_positions: usize,
) -> SnapshotPlan {
    if kv_bytes_per_token == 0 {
        return SnapshotPlan {
            evict_bytes: 0,
            positions: prompt_positions,
        };
    }
    let needed = kv_bytes_per_token.saturating_mul(prompt_positions as u64);
    let room = budget_bytes.saturating_sub(live_bytes.saturating_add(cached_prompt_bytes));
    if room >= needed {
        return SnapshotPlan {
            evict_bytes: 0,
            positions: prompt_positions,
        };
    }
    let evict_bytes = (needed - room).min(cached_prompt_bytes);
    let room_after = room.saturating_add(evict_bytes);
    SnapshotPlan {
        evict_bytes,
        positions: (room_after / kv_bytes_per_token).min(prompt_positions as u64) as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The snapshot is cut to the room beside the live cache, after older
    /// cached prompts have been offered up — and is skipped outright when the
    /// live cache leaves nothing.
    #[test]
    fn a_snapshot_is_cut_to_what_fits_beside_the_live_cache() {
        // 1000-byte budget, 10 bytes a token, a 40-token prompt (400 bytes).
        assert_eq!(
            plan_snapshot(1000, 400, 0, 10, 40),
            SnapshotPlan {
                evict_bytes: 0,
                positions: 40
            }
        );
        // Live 400 + cached 500: room 100 → evict all 500 cached, room 600 → whole prompt.
        assert_eq!(
            plan_snapshot(1000, 400, 500, 10, 40),
            SnapshotPlan {
                evict_bytes: 300,
                positions: 40
            }
        );
        // Live 800, cached 100: room 100, evict 100 → room 200 → 20 positions.
        assert_eq!(
            plan_snapshot(1000, 800, 100, 10, 40),
            SnapshotPlan {
                evict_bytes: 100,
                positions: 20
            }
        );
        // Live cache already at the budget: nothing to snapshot.
        assert_eq!(
            plan_snapshot(1000, 1000, 0, 10, 40),
            SnapshotPlan {
                evict_bytes: 0,
                positions: 0
            }
        );
        // Unknown per-token cost: no judgement, snapshot everything.
        assert_eq!(
            plan_snapshot(1000, 1000, 0, 0, 40),
            SnapshotPlan {
                evict_bytes: 0,
                positions: 40
            }
        );
    }

    /// The measured case: a 6.4k-token prompt whose live cache is 2.6 GB of a
    /// 4.5 GB budget cannot keep a full copy of itself; it keeps what fits.
    #[test]
    fn the_long_prompts_own_snapshot_is_cut_rather_than_overfilling_the_card() {
        let mb = 1024 * 1024u64;
        let per_token = 344 * 1024u64;
        let live = per_token * 7680;
        let plan = plan_snapshot(4523 * mb, live, 0, per_token, 7649);
        assert_eq!(plan.evict_bytes, 0);
        assert!(plan.positions > 0 && plan.positions < 7649, "{plan:?}");
        assert!(live + per_token * plan.positions as u64 <= 4523 * mb);
    }

    /// The three outcomes, and the boundary between them: cached prompts are
    /// the ONLY thing eviction can recover, so a shortfall past them is a
    /// refusal however large the cache is.
    #[test]
    fn a_prompt_is_admitted_evicts_cached_prompts_or_is_refused_at_token_zero() {
        // 1000-byte budget, 10 bytes a token.
        assert_eq!(admit_prompt(1000, 0, 0, 10, 100), PromptAdmission::Fits);
        assert_eq!(admit_prompt(1000, 500, 0, 10, 50), PromptAdmission::Fits);
        // Live 500 + cached 300 + claim 300 = 1100: 100 over, cache can cover it.
        assert_eq!(
            admit_prompt(1000, 500, 300, 10, 30),
            PromptAdmission::EvictBytes(100)
        );
        // Live 500 + cached 300 + claim 600 = 1400: 400 over, cache covers 300.
        assert_eq!(
            admit_prompt(1000, 500, 300, 10, 60),
            PromptAdmission::Refuse { short_by: 100 }
        );
        // Exactly at the budget fits; one byte past it evicts one byte.
        assert_eq!(admit_prompt(1000, 700, 100, 10, 20), PromptAdmission::Fits);
        assert_eq!(
            admit_prompt(999, 700, 100, 10, 20),
            PromptAdmission::EvictBytes(1)
        );
    }

    /// The measured case: a 3.1 GB model on an 8 GB card, one 6.4k-token
    /// prompt's snapshot already cached, the same prompt again. Before this
    /// existed the answer was "fits" and the cache spilled to host memory.
    #[test]
    fn the_second_long_prompt_evicts_the_first_ones_snapshot_rather_than_spilling() {
        let mb = 1024 * 1024u64;
        let budget = 4523 * mb;
        let per_token = 344 * 1024u64;
        // The first prompt's snapshot: 7680 positions (6.4k tokens rounded
        // up to the growth quantum) at 344 KB each, ~2.6 GB.
        let cached = per_token * 7680;
        assert!(cached < budget && 2 * cached > budget);
        assert_eq!(
            admit_prompt(budget, 0, cached, per_token, 7680),
            PromptAdmission::EvictBytes(2 * cached - budget)
        );
    }

    /// MLA caches decompressed K/V at full head count with asymmetric widths,
    /// so it is far more expensive per token than GQA at the same head_dim.
    /// Getting this wrong would under-estimate DeepSeek's cache.
    #[test]
    fn mla_costs_more_per_token_than_gqa() {
        let (gk, gv) = standard_kv_elems(8, 128);
        let (mk, mv) = mla_kv_elems(128, 192, 128);
        assert!(
            kv_bytes_per_token(1, mk, mv, false) > kv_bytes_per_token(1, gk, gv, false),
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
        let per_token = kv_bytes_per_token(16, k, v, false);
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

    /// Growth is claimed exactly at quantum boundaries and nowhere else —
    /// that is what keeps the check off the per-token path.
    #[test]
    fn growth_is_claimed_only_at_quantum_boundaries() {
        let q = 512;
        assert_eq!(
            positions_claimed(0, 1, q),
            512,
            "a fresh request claims one"
        );
        assert_eq!(positions_claimed(0, 512, q), 512);
        // Decoding inside the current quantum claims nothing.
        assert_eq!(positions_claimed(1, 2, q), 0);
        assert_eq!(positions_claimed(510, 511, q), 0);
        assert_eq!(positions_claimed(511, 512, q), 0);
        // Crossing into the next claims one.
        assert_eq!(positions_claimed(512, 513, q), 512);
        assert_eq!(positions_claimed(513, 514, q), 0);
    }

    /// **A prefill claims MANY quanta in one forward.** Charging it one — which
    /// the first version of this did — under-counts the single largest claim a
    /// request ever makes, letting exactly the allocation the budget exists to
    /// refuse go straight through.
    #[test]
    fn a_large_prefill_claims_every_quantum_it_spans() {
        let q = 512;
        assert_eq!(positions_claimed(0, 5000, q), 5120, "10 quanta, not 1");
        assert_eq!(positions_claimed(0, 2000, q), 2048);
        // Continuing a conversation charges only the new span.
        assert_eq!(positions_claimed(1000, 5000, q), 5120 - 1024);
    }

    /// And the budget must see that full claim, not a single quantum's worth.
    #[test]
    fn a_large_prefill_is_refused_when_it_does_not_fit() {
        let per_token = 1_000u64;
        let q = 512usize;
        // Room for two quanta only.
        let budget = per_token * (2 * q) as u64;
        // One quantum fits...
        assert!(!claim_exceeds_headroom(
            budget,
            0,
            per_token,
            positions_claimed(0, 1, q)
        ));
        // ...but a ten-quantum prefill must not.
        assert!(
            claim_exceeds_headroom(budget, 0, per_token, positions_claimed(0, 5000, q)),
            "a prefill spanning 10 quanta must be refused against a 2-quantum budget"
        );
    }

    /// A zero quantum must not divide by zero.
    #[test]
    fn a_zero_quantum_is_treated_as_one() {
        assert!(positions_claimed(0, 1, 0) > 0);
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
        assert!(!claim_exceeds_headroom(budget, 0, per_token, q));
        // Three already held: the fourth still fits.
        assert!(!claim_exceeds_headroom(budget, claim * 3, per_token, q));
        // Four already held — by any mix of requests — and the fifth does not.
        assert!(claim_exceeds_headroom(budget, claim * 4, per_token, q));
    }

    /// Arithmetic near the limits must saturate rather than wrap: wrapping
    /// would turn "no memory" into "unlimited memory".
    #[test]
    fn headroom_arithmetic_saturates() {
        assert!(claim_exceeds_headroom(0, u64::MAX, 1, 1));
        assert!(claim_exceeds_headroom(10, u64::MAX, u64::MAX, usize::MAX));
        // A zero per-token cost cannot claim anything, so it never refuses.
        assert!(!claim_exceeds_headroom(0, 0, 0, 512));
    }

    /// The f16 flash mirror is real VRAM and the head-room check must charge
    /// for it.
    ///
    /// `LayerKv` keeps an f16 BSHD copy of K and V for GQA models on CUDA, so
    /// those caches cost 1.5x what the f32 figure alone says. If this were left
    /// uncounted the admission check would let a model claim half again as much
    /// VRAM as the budget believes, and the clean 503 that reroutes the request
    /// to a peer would be replaced by an actual out-of-memory.
    #[test]
    fn a_mirrored_cache_is_charged_for_the_mirror() {
        let (k, v) = standard_kv_elems(8, 128);
        let plain = kv_bytes_per_token(28, k, v, false);
        let mirrored = kv_bytes_per_token(28, k, v, true);
        assert_eq!(
            mirrored,
            plain + plain / 2,
            "the mirror is the same elements at half the width, so 1.5x"
        );
        // Stated as a claim about admission, not just arithmetic: the same
        // headroom must admit fewer positions once a mirror is in play.
        let headroom = plain * 1000;
        let fits_plain = headroom / plain;
        let fits_mirrored = headroom / mirrored;
        assert!(
            fits_mirrored < fits_plain,
            "a mirrored cache must exhaust the budget sooner ({fits_mirrored} vs {fits_plain})"
        );
    }
}
