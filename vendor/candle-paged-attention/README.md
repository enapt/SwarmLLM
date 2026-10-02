# Candle Paged Attention (vendored, not used)

PagedAttention CUDA kernels for candle. Nothing in SwarmLLM depends on this
crate yet — PagedAttention was never wired in (gotcha #257) — and its kernels
still launch on CUDA stream 0, so they must be moved to the caller's stream
before anything uses them (see `vendor/candle-flash-attn/README.md`).

All files in `kernels` are adapted from https://github.com/vllm-project/vllm/tree/main/csrc and are under the vLLM
Project copyright.
