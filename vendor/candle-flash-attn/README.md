# candle-flash-attn (vendored, patched)

FlashAttention for candle, vendored from upstream 0.10.x with five local
patches: `cudart` linked statically, the bf16 kernels removed, `run_mha`
launching on the caller's CUDA stream instead of stream 0 (the cause of
v0.3.199-alpha's garbage output), the library carrying compute_80 PTX with
`run_mha` returning its launch status (#928), and `flash_attn_key_runs` (a
tree's context read in runs that share one set of queries, #134). Re-apply all
five after any re-vendor —
`docs/ARCHITECTURE.md` describes each, and
`the_vendored_attention_kernels_launch_on_the_devices_stream` in
`tests/repo_consistency.rs` fails if the stream patch is lost.
