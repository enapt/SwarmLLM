//! Deciding how many of a segment's layers go on the graphics card.
//!
//! **Why this exists.** Placement was all-or-nothing: `force_cpu_for` is
//! `gpu_layers == 0`, so a model needing slightly more graphics memory than is
//! free lost the card *entirely*. Measured on this swarm that is a ~24x cliff,
//! the largest multiplier in the system, and it has now been reported by three
//! different machines — most recently a 14B needing 9347 MB against a 4990 MB
//! budget, which ran on the processor with **5151 MB of the card free and
//! unused**, pinning several cores and pushing the machine past its thermal
//! threshold (gotcha #431, `docs/FUTURE_WORK.md`).
//!
//! **Prior art, and where we deliberately differ** (diagnosis rule 0).
//! llama.cpp's `--n-gpu-layers` is a count, not a switch, and its auto mode
//! (PR #17485) reserves a flat **800 MB** for "KV cache and compute buffers"
//! then divides the remaining VRAM by an estimated per-layer size. Reviewers on
//! that PR pushed toward #16653's approach instead — real trial allocations
//! rather than an a-priori formula — because a formula that disagrees with
//! reality is exactly the trap.
//!
//! We keep the formula but **do not need the flat reserve**, because the thing
//! it stands in for is a number we already compute. KV cache is charged per
//! layer per token by [`kv_bytes_per_token`], so KV belongs *inside* the
//! per-layer cost rather than in a lump beside it — a layer placed on the card
//! brings its own KV with it, and one placed on the processor does not. That
//! turns a guess into arithmetic, and it is why this takes a context length
//! rather than a magic constant.
//!
//! What survives as a reserve is only what genuinely is not per-layer: the
//! transient activation and attention-score buffers a forward pass allocates,
//! which scale with the prefill chunk and the head count, not with depth.
//!
//! **The split is contiguous and card-first**, matching llama.cpp: layers
//! `[0, n)` of the segment on the card, `[n, end)` on the processor. Contiguous
//! is what keeps the cost at **one** device transition per forward — the same
//! reasoning as #425's shard contiguity, one level down. Which end holds the
//! card does not affect compute (dense layers cost the same), so matching the
//! convention everyone else uses is worth more than a private optimum.

use crate::inference::split::kv_budget::kv_bytes_per_token;
use candle_core::{Device, Tensor};

/// Held back for the transient buffers a forward pass allocates.
///
/// Activations and attention scores, which scale with the prefill chunk and the
/// head count rather than with depth — so unlike KV they cannot be folded into
/// a per-layer cost. At a 128-token chunk and a 4k context a score matrix runs
/// to tens of MB and several are live at once.
///
/// Deliberately smaller than llama.cpp's 800 MB, because that figure covers KV
/// as well and ours does not. Erring high here costs a layer or two on the
/// card; erring low costs an out-of-memory failure mid-request, so it is not a
/// symmetric trade and this is set generously.
pub(crate) const FORWARD_BUFFER_RESERVE_BYTES: u64 = 384 * 1024 * 1024;

/// How many of `total_layers` fit on the card, given what a layer costs.
///
/// Returns `0..=total_layers`. Zero means the processor takes the whole
/// segment, which is the old behaviour and still correct when nothing fits.
///
/// `context_tokens` is what KV is charged for. It has to be the same figure
/// admission uses, or a model is placed by one number and then runs out of
/// memory against another — the disagreement that made
/// `estimate_gpu_footprint_mb` and `estimate_vram_from_shard_dir` reach
/// different conclusions about the same model (gotcha #388).
pub(crate) fn layers_that_fit(
    budget_bytes: u64,
    weights_bytes_per_layer: u64,
    kv_bytes_per_layer_per_token: u64,
    context_tokens: u64,
    total_layers: usize,
) -> usize {
    if total_layers == 0 || weights_bytes_per_layer == 0 {
        return 0;
    }
    let usable = budget_bytes.saturating_sub(FORWARD_BUFFER_RESERVE_BYTES);
    // A layer on the card brings its KV with it; one on the processor does not.
    // Folding KV in here is what removes the need for a flat reserve.
    let per_layer = weights_bytes_per_layer
        .saturating_add(kv_bytes_per_layer_per_token.saturating_mul(context_tokens));
    if per_layer == 0 {
        return 0;
    }
    ((usable / per_layer) as usize).min(total_layers)
}

/// The same decision expressed in the units the loader actually has.
///
/// `k_elems`/`v_elems` are per-token per-layer element counts, as
/// [`kv_bytes_per_token`] takes them — asking for one layer's worth is what
/// makes the per-layer cost above meaningful.
pub(crate) fn plan_gpu_layers(
    budget_bytes: u64,
    segment_weights_bytes: u64,
    total_layers: usize,
    k_elems: usize,
    v_elems: usize,
    mirrored: bool,
    context_tokens: u64,
) -> usize {
    if total_layers == 0 {
        return 0;
    }
    let weights_per_layer = segment_weights_bytes / total_layers as u64;
    let kv_per_layer_per_token = kv_bytes_per_token(1, k_elems, v_elems, mirrored);
    layers_that_fit(
        budget_bytes,
        weights_per_layer,
        kv_per_layer_per_token,
        context_tokens,
        total_layers,
    )
}

/// May this architecture's layers be split across two devices?
///
/// **Deny by default, and a new architecture is denied until someone checks
/// it.** The loader applies placement by shadowing `device`, `cos` and `sin`
/// at the head of each per-layer loop, which is safe exactly when the loop's
/// tensor loads go through those names. Qwen 3.5 does not: it builds its own
/// `q35_cos`/`q35_sin` outside the loop, so a split model would put a
/// card-resident layer's RoPE tables on the processor.
///
/// That failure is invisible until runtime — candle only objects when two
/// tensors from different devices meet inside an op — which is precisely the
/// hazard `docs/FUTURE_WORK.md` flagged as where the real risk lives. A
/// compiler warning about an unused shadow is what surfaced it here, and
/// nothing would have surfaced it on a machine without that architecture.
///
/// So the list is an allowlist of architectures whose loops have been read,
/// not a denylist of known-bad ones. Adding one means checking that every
/// tensor its loop loads comes from the shadowed names — and ideally running
/// it split on a real card.
pub(crate) fn arch_supports_hybrid(arch: &crate::inference::model_arch::ModelArch) -> bool {
    use crate::inference::model_arch::ModelArch as A;
    match arch {
        // Dense-family: the per-layer loop loads everything through the
        // shadowed `device`, and takes its RoPE from the shared `cos`/`sin`.
        A::Llama
        | A::Qwen2
        | A::Gemma
        | A::Gemma2
        | A::Phi3
        | A::Mistral
        | A::Starcoder2
        | A::Glm4
        | A::Llama4 => true,
        // Own RoPE tables (`q35_cos`/`q35_sin`) built outside the loop, plus a
        // state-space path whose buffers have not been checked.
        A::Qwen35 | A::Qwen35Moe => false,
        // MLA attention with its own projection set; unverified.
        A::DeepSeek2 => false,
        // An architecture we do not recognise is one whose loop nobody has
        // read. Deny, for the same reason the list is an allowlist.
        A::Unknown(_) => false,
    }
}

/// Which device each layer of a segment loads onto, and the RoPE tables that
/// go with it.
///
/// **Applied by shadowing, not by threading.** The loader has ~128 `&device`
/// references across four per-architecture loops, and editing each one is how
/// a path gets missed — which for device placement fails at *runtime*, not at
/// compile time, because candle only complains when two tensors from different
/// devices meet in an op. Instead each loop shadows `device`, `cos` and `sin`
/// with this type's answers on its first line, so every load inside the body
/// picks up the right device with no further edit and a new architecture
/// inherits the behaviour by construction.
pub(crate) struct LayerPlacement {
    /// Absolute index of this segment's first layer.
    layer_start: usize,
    /// How many of the segment's layers, from the start, go on the card.
    gpu_layers: usize,
    /// The card, when any layer is going there.
    gpu: Option<Device>,
    /// Where everything else goes.
    cpu: Device,
    /// RoPE tables per device. Built by moving, not recomputing: `to_device`
    /// on a plain tensor is a copy, and these are small next to the weights.
    rope_gpu: Option<(Tensor, Tensor)>,
    rope_cpu: (Tensor, Tensor),
}

impl LayerPlacement {
    /// Everything on one device — the behaviour before hybrid placement, and
    /// still what a model that fits (or one with no card) gets.
    pub(crate) fn uniform(
        layer_start: usize,
        device: Device,
        cos: Tensor,
        sin: Tensor,
    ) -> candle_core::Result<Self> {
        let on_gpu = device.is_cuda();
        Ok(Self {
            layer_start,
            gpu_layers: if on_gpu { usize::MAX } else { 0 },
            gpu: if on_gpu { Some(device.clone()) } else { None },
            cpu: if on_gpu { Device::Cpu } else { device },
            rope_gpu: if on_gpu {
                Some((cos.clone(), sin.clone()))
            } else {
                None
            },
            rope_cpu: (cos, sin),
        })
    }

    /// Split: the first `gpu_layers` on the card, the rest on the processor.
    pub(crate) fn split(
        layer_start: usize,
        gpu: Device,
        gpu_layers: usize,
        cos: Tensor,
        sin: Tensor,
    ) -> candle_core::Result<Self> {
        let rope_cpu = (cos.to_device(&Device::Cpu)?, sin.to_device(&Device::Cpu)?);
        Ok(Self {
            layer_start,
            gpu_layers,
            gpu: Some(gpu),
            cpu: Device::Cpu,
            rope_gpu: Some((cos, sin)),
            rope_cpu,
        })
    }

    fn on_gpu(&self, abs_layer: usize) -> bool {
        // `checked_sub`, not `saturating_sub`: an index below this segment's
        // start is not in the segment at all, and saturating it to 0 reads as
        // "the first layer", i.e. on the card. Caught by
        // `the_split_puts_the_first_layers_on_the_card_and_the_rest_on_the_processor`,
        // which is the difference between a guard and a comment.
        let Some(offset) = abs_layer.checked_sub(self.layer_start) else {
            return false;
        };
        self.gpu.is_some() && offset < self.gpu_layers
    }

    /// The device layer `abs_layer` loads onto.
    pub(crate) fn device_for(&self, abs_layer: usize) -> Device {
        match (&self.gpu, self.on_gpu(abs_layer)) {
            (Some(gpu), true) => gpu.clone(),
            _ => self.cpu.clone(),
        }
    }

    /// The RoPE tables for that layer, already on its device.
    pub(crate) fn rope_for(&self, abs_layer: usize) -> (Tensor, Tensor) {
        match (&self.rope_gpu, self.on_gpu(abs_layer)) {
            (Some((c, s)), true) => (c.clone(), s.clone()),
            _ => (self.rope_cpu.0.clone(), self.rope_cpu.1.clone()),
        }
    }

    /// Per-layer devices, in segment order, for the model to keep.
    pub(crate) fn devices(&self, layer_count: usize) -> Vec<Device> {
        (0..layer_count)
            .map(|i| self.device_for(self.layer_start + i))
            .collect()
    }

    /// Is this segment actually split across two devices?
    pub(crate) fn is_hybrid(&self, layer_count: usize) -> bool {
        self.gpu.is_some() && self.gpu_layers < layer_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;

    /// The reported case: a 14B segment wanting 9347 MB against a 4990 MB
    /// budget, on a card with 5151 MB free. It used to get **nothing**.
    #[test]
    fn the_reported_case_now_puts_something_on_the_card() {
        let total = 48;
        let weights = 9347 * MB;
        let n = plan_gpu_layers(4990 * MB, weights, total, 1024, 1024, false, 4096);
        assert!(n > 0, "still placing nothing on a card with room");
        assert!(n < total, "claimed to fit a segment that does not fit");
        // And what it placed must actually fit inside the budget.
        let per_layer = weights / total as u64;
        let kv = kv_bytes_per_token(1, 1024, 1024, false) * 4096;
        assert!(
            n as u64 * (per_layer + kv) + FORWARD_BUFFER_RESERVE_BYTES <= 4990 * MB,
            "the plan overcommits the card"
        );
    }

    #[test]
    fn a_segment_that_fits_entirely_goes_entirely_to_the_card() {
        // 2 GB of weights, 8 GB budget: everything, and never more than exists.
        let n = plan_gpu_layers(8 * GB, 2 * GB, 28, 512, 512, false, 4096);
        assert_eq!(n, 28);
    }

    #[test]
    fn a_card_with_no_room_still_gets_nothing() {
        // The whole budget is swallowed by the forward-buffer reserve.
        assert_eq!(
            plan_gpu_layers(256 * MB, 8 * GB, 32, 1024, 1024, false, 4096),
            0
        );
        // And a zero budget cannot go negative or wrap.
        assert_eq!(plan_gpu_layers(0, 8 * GB, 32, 1024, 1024, false, 4096), 0);
    }

    /// KV is charged per layer placed, not as a lump — so asking for a longer
    /// conversation must move layers off the card, never leave the plan
    /// unchanged. This is the property that replaces llama.cpp's flat reserve.
    #[test]
    fn a_longer_context_costs_layers_on_the_card() {
        let args = |ctx| plan_gpu_layers(6 * GB, 4 * GB, 32, 1024, 1024, false, ctx);
        let short = args(2048);
        let long = args(32768);
        assert!(
            long < short,
            "context length did not affect placement ({short} at 2k, {long} at 32k) — \
             KV is not being charged per layer"
        );
    }

    #[test]
    fn the_mirror_is_charged_for() {
        // The f16 flash mirror is real graphics memory (see `LayerKv`), so a
        // model maintaining one must not be planned as though it were free.
        let plain = plan_gpu_layers(6 * GB, 4 * GB, 32, 1024, 1024, false, 8192);
        let mirrored = plan_gpu_layers(6 * GB, 4 * GB, 32, 1024, 1024, true, 8192);
        assert!(mirrored <= plain, "mirroring must not increase the plan");
    }

    /// The boundary is the whole contract: layers before it on the card,
    /// layers after it on the processor, and the RoPE tables following their
    /// layer so no op ever meets tensors from two devices.
    #[test]
    fn the_split_puts_the_first_layers_on_the_card_and_the_rest_on_the_processor() {
        let cos = Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let sin = cos.clone();
        // `Device::Cpu` stands in for the card here: the placement logic is
        // about indices, and a machine running these tests has no GPU.
        let p = LayerPlacement::split(10, Device::Cpu, 3, cos, sin).unwrap();
        // Absolute indices 10,11,12 are the segment's first three.
        assert!(p.on_gpu(10) && p.on_gpu(11) && p.on_gpu(12));
        assert!(!p.on_gpu(13), "boundary is off by one");
        // And a layer before this segment's start must never read as on-card.
        assert!(!p.on_gpu(9), "index below layer_start wrapped");
        assert_eq!(p.devices(5).len(), 5);
    }

    #[test]
    fn a_uniform_placement_is_never_hybrid() {
        let cos = Tensor::zeros((4, 2), candle_core::DType::F32, &Device::Cpu).unwrap();
        let p = LayerPlacement::uniform(0, Device::Cpu, cos.clone(), cos).unwrap();
        assert!(
            !p.is_hybrid(32),
            "a single-device model must not take the transition path"
        );
        assert!(p.devices(32).iter().all(|d| !d.is_cuda()));
    }

    #[test]
    fn degenerate_inputs_do_not_panic_or_wrap() {
        assert_eq!(layers_that_fit(u64::MAX, 0, 0, 0, 8), 0);
        assert_eq!(layers_that_fit(0, 1, 1, 1, 0), 0);
        // Never more layers than the segment has, however large the budget.
        assert_eq!(layers_that_fit(u64::MAX, 1, 0, 0, 8), 8);
    }
}
