//! Attention-kernel selection override.
//!
//! candle dispatches attention based on tensor shape — single-position decode
//! and multi-position prefill/verify go through different kernels (matmul vs
//! flash). For speculative-decoding paths (SWIFT, classic spec) the draft and
//! verify forwards must produce numerically identical logits; otherwise even
//! `skip_ratio = 0` (draft = full target) yields < 100 % accept because of
//! tiny softmax differences.
//!
//! This module exposes a thread-local override that forces every attention
//! call to use `standard_attention` regardless of shape. The SWIFT decode
//! loop wraps each forward in a `ForceStandardAttnGuard` so prefill, draft,
//! and verify all share the matmul path. The guard restores the previous
//! flag value on drop, composing safely with nested calls.

//! ## `SWARMLLM_FORCE_STANDARD_ATTN=1`
//!
//! Sets the *initial* value of the override on every thread, which makes the
//! whole process take `standard_attention` and nothing else change. That is the
//! only honest way to price an attention kernel: two separately-built binaries
//! differ in more than the kernel (link order, inlining, codegen), so a
//! difference between them is not attributable. Same binary, same weights, one
//! branch — the comparison the CPU-side crossovers in
//! [`crate::inference::layers::run_attention`] were measured with, and the one
//! that priced flash-attention-2 on CUDA when it was re-enabled.
//!
//! It sets the initial value rather than OR-ing into the result so that
//! [`ForceStandardAttnGuard`]'s nested `false` still works — a debugging switch
//! that quietly changed the guard's semantics would be its own bug.

use std::cell::Cell;

/// Whether `SWARMLLM_FORCE_STANDARD_ATTN` asks every thread to start forced.
///
/// Read once per process: the environment cannot change under a running daemon,
/// and this is consulted once per attention call.
fn env_forces_standard() -> bool {
    use std::sync::OnceLock;
    static FORCED: OnceLock<bool> = OnceLock::new();
    *FORCED.get_or_init(|| {
        matches!(
            std::env::var("SWARMLLM_FORCE_STANDARD_ATTN").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

thread_local! {
    static FORCE_STANDARD_ATTN: Cell<bool> = Cell::new(env_forces_standard());
}

/// Returns `true` if the current thread has the standard-attention override
/// active. Read once at the top of the dispatch in
/// `crate::inference::layers::run_attention`.
pub fn is_force_standard_attn() -> bool {
    FORCE_STANDARD_ATTN.with(|f| f.get())
}

/// RAII guard: while this lives, [`is_force_standard_attn`] returns the
/// requested value. Restores the previous value on drop so nested guards
/// behave correctly. Must be created on the same thread that runs the
/// forward pass — for tokio code, this means inside the `block_in_place`
/// closure (not in the surrounding async fn).
pub struct ForceStandardAttnGuard {
    prev: bool,
}

impl ForceStandardAttnGuard {
    pub fn new(force: bool) -> Self {
        let prev = FORCE_STANDARD_ATTN.with(|f| f.replace(force));
        Self { prev }
    }
}

impl Drop for ForceStandardAttnGuard {
    fn drop(&mut self) {
        let prev = self.prev;
        FORCE_STANDARD_ATTN.with(|f| f.set(prev));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_false() {
        assert!(!is_force_standard_attn());
    }

    #[test]
    fn guard_sets_and_restores() {
        assert!(!is_force_standard_attn());
        {
            let _g = ForceStandardAttnGuard::new(true);
            assert!(is_force_standard_attn());
            {
                let _inner = ForceStandardAttnGuard::new(false);
                assert!(!is_force_standard_attn());
            }
            assert!(is_force_standard_attn());
        }
        assert!(!is_force_standard_attn());
    }
}
