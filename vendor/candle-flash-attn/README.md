# candle-flash-attn (vendored, patched)

FlashAttention for candle, vendored from upstream 0.10.x with three local
patches: `cudart` linked statically, the bf16 kernels removed, and `run_mha`
launching on the caller's CUDA stream instead of stream 0 (the cause of
v0.3.199-alpha's garbage output). Re-apply all three after any re-vendor —
`docs/ARCHITECTURE.md` describes each, and
`the_vendored_attention_kernels_launch_on_the_devices_stream` in
`tests/repo_consistency.rs` fails if the stream patch is lost.
