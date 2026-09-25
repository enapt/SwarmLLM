# Qwen 3.5 support (FUTURE_WORK #117) — the plan, read off llama.cpp

Written 2026-09-25 night, when Qwen 3.5 was REFUSED because no real file could
load: the loader had been written against a guessed layout, and several of its
tensor names are llama.cpp's C++ **member** names (`attn_post_norm`, `ssm_dt`)
rather than the GGUF **tensor** names those members are loaded from
(`post_attention_norm`, `ssm_dt.bias`). Everything below is from llama.cpp
master `4b1a27f` (`src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`,
`llama_model_rope_type`) and the real header of `unsloth/Qwen3.5-4B-GGUF` /
`ggml-org/Qwen3.5-0.8B-GGUF`. Re-read those files before implementing — this is
a map, not a substitute.

## The reference is ready

- llama.cpp master, CPU-only: `~/llama.cpp-ref` (`build/bin/libllama.so`).
  llama-cpp-python 0.3.16 (the usual reference here) predates `qwen35`.
- `~/llama.cpp-ref/dump_logits MODEL OUT.f32 id,id,...` — `[n, vocab]` f32, the
  layout `examples/logits_reference_probe.rs` writes;
  `~/llama.cpp-ref/compare_f32_logits.py <probe prefix> OUT.f32` compares.
- Fixture with REAL weights: `~/swarmllm-ref/qwen35/Qwen3.5-0.8B-Q8_0.gguf`
  (25 blocks = 24 + one next-token-prediction layer; vocab 248,320). llama.cpp's
  logits for the probe's default 24 ids are in the night round's scratchpad —
  regenerate with `dump_logits`.
- ⚠ The probe loads through the split loader, which refuses `qwen35`: flip
  `ModelArch::is_supported` on the working branch FIRST, and back only when the
  comparison agrees.

## Tensors (GGUF names, Qwen3.5-0.8B shapes)

Globals: `token_embd`, `output_norm`, `output` (may be tied). Metadata:
`full_attention_interval` 4, `rope.dimension_sections` [11, 11, 10, 0],
`rope.dimension_count` 64 (of key length 256), `rope.freq_base` 1e7,
`ssm.{conv_kernel 4, state_size 128, group_count 16, time_step_rank 16,
inner_size 2048}`, `nextn_predict_layers` 1 (the LAST block is the MTP layer —
skip it for ordinary decoding, as llama.cpp's main graph does).

Every layer: `attn_norm`, `post_attention_norm` (the pre-FFN norm — llama.cpp
member `attn_post_norm`), `ffn_gate/up/down` (dense) or the MoE set.

DeltaNet ("linear attention") layers — 3 of every 4:
`attn_qkv` [n_embd → 2·k_heads·k_dim + v_heads·v_dim] (member `wqkv`),
`attn_gate` [n_embd → d_inner] (the z gate; member `wqkv_gate`), `ssm_beta`,
`ssm_alpha` [n_embd → v_heads], `ssm_dt.bias` [v_heads] (member `ssm_dt`),
`ssm_a` [v_heads] (already `-exp(A_log)`), `ssm_conv1d` [kernel, channels],
`ssm_norm` [v_dim], `ssm_out` [d_inner → n_embd].

Full-attention layers — every 4th (`(il + 1) % 4 == 0`): `attn_q` projects to
`2 · head_dim · n_head` — Q and an output GATE interleaved PER HEAD
(`[q_h0 | g_h0 | q_h1 | g_h1 …]`); `attn_k`, `attn_v`, `attn_q_norm`,
`attn_k_norm`, `attn_output`. There is NO separate `attn_gate` here.

## The forward, op by op

Block: `x → attn_norm → (DeltaNet | attention) → + x → post_attention_norm → FFN → + residual`.

**Full attention** (`build_layer_attn`): Q_full = attn_q·x; Q = the even
head-halves, gate = the odd ones; Q = RMSNorm(Q, attn_q_norm) per head; K =
RMSNorm(attn_k·x, attn_k_norm) per head; V = attn_v·x; RoPE both with
`ggml_rope_multi`, rope type **IMROPE**, sections [11, 11, 10, 0] over 64 of
256 dims — for TEXT all position components are the token index, so check
whether that reduces to plain partial NEOX RoPE (compare one layer's Q against
the reference before assuming); softmax attention, scale 1/√head_dim; output ×
sigmoid(gate); attn_output.

**DeltaNet** (`build_layer_attn_linear`):
1. `qkv_mixed = attn_qkv·x`, `z = attn_gate·x`.
2. `beta = sigmoid(ssm_beta·x)` per v-head.
3. `g = softplus(ssm_alpha·x + ssm_dt.bias) · ssm_a` per v-head (a NEGATIVE
   log-decay, since `ssm_a = -exp(A_log)`).
4. causal depthwise conv over `qkv_mixed` with `ssm_conv1d` (kernel 4, carried
   conv state across steps), then SiLU.
5. split the conv output into q [k_heads·k_dim], k [same], v [v_heads·v_dim];
   **L2-normalise q and k** per head (`build_gdn_l2_norm`, eps = rms eps);
   repeat q/k heads to v_heads when they differ.
6. gated delta rule over the sequence (`build_recurrent_attn` →
   `delta-net-base.cpp`: chunked for prompts, autoregressive for decode — the
   two must agree; read `build_delta_net_autoregressive` for the exact update
   and where q is scaled).
7. `out = RMSNorm(o, ssm_norm) · silu(z)` per head (`build_norm_gated`), then
   `ssm_out`.

**The recurrence** (`build_delta_net_autoregressive`, one token; the chunked
form must equal it), per v-head with state `S` [v_dim × k_dim]:

```
q  = q / sqrt(k_dim)                 (after the L2 norm)
S  = S * exp(g)                      (g ≤ 0: softplus(α + dt_bias) · ssm_a)
e  = (v − S·k) * beta                (beta = sigmoid(ssm_beta·x), a scalar per head)
S  = S + e ⊗ k                       (row j gains k · e_j)
o  = S·q
```

⚠ The description of the current `layers/qwen35.rs` (ARCHITECTURE § Qwen 3.5)
differs from this in three places: it decays by `exp(-softplus(α + dt))` with no
`ssm_a` factor, it applies beta to k AND v separately (`β_v·v − g·S@(β_k·k)`)
where llama.cpp scales the error once, and it names no `1/sqrt(k_dim)` on q.
Check each against the code, not the description.

State per sequence: the conv tail (kernel−1 rows of `qkv_mixed`) and the
[v_dim × v_dim] recurrent state per v-head — the existing `SsmState` is the
place, but check its shapes against these.

## What the current code has, and what to keep

`layers/qwen35.rs` + the loader's hybrid-SSM branch implement a delta net with
per-step alpha/beta gating, conv state and gated norm — the SHAPE of the right
thing, written without a file to test against. Treat every formula in it as
unverified: rebuild the loader against the tensor list above, then bring
`layers/qwen35.rs` into agreement op by op, checking intermediate tensors
against llama.cpp (the `cb(…)` names in `qwen35.cpp` are what
`GGML_SCHED`/eval-callback tools print).

## Done means

1. `logits_reference_probe` on Qwen3.5-0.8B agrees with `dump_logits` at every
   position (cosine ≥ 0.999; Q8_0, so expect ~0.9994-class noise — also build
   an F16/BF16 run to separate rounding from arithmetic), whole AND split
   (the recurrent state crosses a segment boundary as a hidden state only —
   check the split point falls where llama.cpp would be happy too).
2. Decode steps (the probe's last 4) agree too — the autoregressive path.
3. `is_supported` flipped, `supported_list` restored, FUTURE_WORK #117 closed,
   and a real Qwen3.5-4B reply scored against llama.cpp on the CUDA build.
4. Then `qwen35moe` (same attention/DeltaNet, MoE FFN with a sigmoid-gated
   shared expert — `ffn_gate_inp_shexp`).
