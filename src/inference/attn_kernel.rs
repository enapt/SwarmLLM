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

use std::cell::Cell;

thread_local! {
    static FORCE_STANDARD_ATTN: Cell<bool> = const { Cell::new(false) };
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
