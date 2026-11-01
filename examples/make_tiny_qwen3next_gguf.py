"""Build a tiny random-weight Qwen3-Next (`qwen3next`) GGUF for checking our
loader against llama.cpp MASTER (`~/llama.cpp-ref/dump_logits`, REF_BIN=) with
examples/logits_reference_probe.rs + compare_logits_reference.py. The real
Qwen3-Coder-Next is 48.5 GB; a random-weight file is all a logits comparison
needs, because both engines read the same file.

Shape follows the real Qwen3-Coder-Next header (unsloth / bartowski Q4_K_M, read
2026-11-01; FUTURE_WORK #390), scaled down, keeping every ratio that decides the
arithmetic:

- DeltaNet layers with FEWER key heads than value heads (2 and 4 here, 16 and 32
  there), so the value heads' GROUPING by key head is exercised — llama.cpp's
  `[k0_v0, k0_v1, k1_v2, k1_v3]`, where Qwen 3.5's files tile them
  (`llama-model.cpp` `get_split_segments`). With one value head per key head the
  two orders are the same and a wrong one would pass.
- beta and alpha in ONE tensor, `ssm_ba`: per key head, its r beta rows then its
  r alpha rows (`qwen3next.cpp` `build_layer_attn_linear`).
- A full-attention layer every INTERVAL-th (default 2): Q and its output gate
  interleaved per head in `attn_q`, Q/K RMS norms, partial NEOX RoPE over a
  quarter of each head.
- Every layer a mixture of experts plus a shared expert gated by
  `sigmoid(ffn_gate_inp_shexp · x)`.

LAYOUT=ssm_in writes the older files' fused `ssm_in` (per key head: q, k, its
values, its gates) in place of `attn_qkv` + `attn_gate` — llama.cpp's "legacy
(slower) path", which every Qwen3-Next file converted before the split tensors
existed carries.

Usage:

    PYTHONPATH=~/llama.cpp-ref/gguf-py python3 examples/make_tiny_qwen3next_gguf.py OUT.gguf

Environment: QUANT=f32 (default), q8, or q8-deltanet (Q8_0 only where the
loader reorders rows); DEQUANTIZED=1 (with a q8 mode) writes those blocks back
as f32, the rounding control; N_LAYER (default 4); INTERVAL (default 2);
LAYOUT=qkv (default) or ssm_in. Then:

    SWARMLLM_UNCHECKED_FAMILIES=qwen3next LOGITS_PROBE_GGUF=OUT.gguf LOGITS_PROBE_OUT=/tmp/q3n \\
      cargo run --no-default-features --features dev,claude-subscription \\
      --example logits_reference_probe
    REF_BIN=~/llama.cpp-ref/dump_logits python3 examples/compare_logits_reference.py OUT.gguf /tmp/q3n

The family is refused until a real file is checked (`ModelArch::is_supported`, #390), so the
probe needs the switch. Results and why a Q8_0 file is not the test: docs/invariants/inference.md
§ "Qwen3-Next: value heads grouped by key head".
"""
import os
import sys

import numpy as np
import gguf

out = sys.argv[1]
rng = np.random.default_rng(20261101)

n_embd, n_head, n_head_kv, head_dim = 64, 4, 2, 32
rope_dim = head_dim // 4
n_layer = int(os.environ.get("N_LAYER", 4))
interval = int(os.environ.get("INTERVAL", 2))
layout = os.environ.get("LAYOUT", "qkv")
n_expert, n_expert_used, n_ff_exp, n_ff_shexp = 4, 2, 32, 32
# DeltaNet: key heads, value heads (r = 2 per key head), state width.
n_k, n_v, d_state, d_conv = 2, 4, 16, 4
r = n_v // n_k
key_dim, value_dim = n_k * d_state, n_v * d_state
conv_dim = 2 * key_dim + value_dim
n_ctx = 256

quant = os.environ.get("QUANT", "f32")
F32 = gguf.GGMLQuantizationType.F32
Q8 = gguf.GGMLQuantizationType.Q8_0 if quant == "q8" else F32
# QUANT=q8-deltanet: Q8_0 only for the DeltaNet projections whose rows the
# loader reorders (attn_qkv / attn_gate / ssm_in, ssm_ba, ssm_out), everything
# else f32 — checks the reorder on a block type without the activation rounding
# that makes a whole tiny Q8 file part from llama.cpp on its own.
Q8_SSM = gguf.GGMLQuantizationType.Q8_0 if quant in ("q8", "q8-deltanet") else F32

# SentencePiece-style vocabulary: specials, the 256 byte tokens, then pieces.
pieces = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
words = ["▁the", "▁a", "▁is", "▁of", "▁and", "▁to", "▁in", "▁it", "▁that", "▁was"]
# logits_reference_probe picks ids up to ~5100, so the vocabulary must reach past it.
words += [f"▁w{i}" for i in range(5800)]
pieces += words
n_vocab = len(pieces)
types = [2, 3, 3] + [6] * 256 + [1] * len(words)  # UNKNOWN, CONTROL, BYTE, NORMAL
scores = [0.0, 0.0, 0.0] + [0.0] * 256 + [-float(i) for i in range(len(words))]

w = gguf.GGUFWriter(out, "qwen3next")
w.add_name("tiny-qwen3next-random")
w.add_block_count(n_layer)
w.add_context_length(n_ctx)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(n_ff_exp * 2)
w.add_head_count(n_head)
w.add_head_count_kv(n_head_kv)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_rope_dimension_count(rope_dim)
w.add_rope_freq_base(5000000.0)
w.add_layer_norm_rms_eps(1e-6)
w.add_expert_count(n_expert)
w.add_expert_used_count(n_expert_used)
w.add_expert_feed_forward_length(n_ff_exp)
w.add_expert_shared_feed_forward_length(n_ff_shexp)
w.add_ssm_conv_kernel(d_conv)
w.add_ssm_state_size(d_state)
w.add_ssm_group_count(n_k)
w.add_ssm_time_step_rank(n_v)
w.add_ssm_inner_size(value_dim)
w.add_full_attention_interval(interval)
w.add_vocab_size(n_vocab)
w.add_tokenizer_model("llama")
w.add_token_list(pieces)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_bos_token_id(1)
w.add_eos_token_id(2)
w.add_unk_token_id(0)
w.add_add_bos_token(True)
w.add_add_eos_token(False)


def t(shape, scale=0.08):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


# DEQUANTIZED=1 writes each Q8_0 tensor as the f32 values its blocks hold — the
# same weights with no activation rounding: the control that says whether a Q8
# file's distance from llama.cpp is rounding or the loader.
dequantized = os.environ.get("DEQUANTIZED") == "1"


def put(name, arr, q):
    if q == gguf.GGMLQuantizationType.Q8_0 and dequantized:
        w.add_tensor(name, gguf.quants.dequantize(gguf.quants.quantize(arr, q), q).astype(np.float32).reshape(arr.shape))
    elif q == gguf.GGMLQuantizationType.Q8_0:
        w.add_tensor(name, gguf.quants.quantize(arr, q), raw_shape=gguf.quants.quant_shape_to_byte_shape(arr.shape, q), raw_dtype=q)
    else:
        w.add_tensor(name, arr)


# numpy shape is ggml's ne reversed: {n_embd, n_vocab} -> (n_vocab, n_embd).
put("token_embd.weight", t((n_vocab, n_embd), 0.5), Q8)
put("output_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
put("output.weight", t((n_vocab, n_embd), 0.3), Q8)
for i in range(n_layer):
    p = f"blk.{i}"
    put(f"{p}.attn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    put(f"{p}.post_attention_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    if (i + 1) % interval != 0:
        # Gated DeltaNet. Rows of each projection are in the file's order: value
        # heads grouped by key head (head h reads key head h // r).
        q_rows, k_rows = t((key_dim, n_embd), 0.2), t((key_dim, n_embd), 0.2)
        v_rows, z_rows = t((value_dim, n_embd), 0.2), t((value_dim, n_embd), 0.2)
        if layout == "ssm_in":
            per_head = []
            for kh in range(n_k):
                per_head += [q_rows[kh * d_state:(kh + 1) * d_state],
                             k_rows[kh * d_state:(kh + 1) * d_state],
                             v_rows[kh * r * d_state:(kh + 1) * r * d_state],
                             z_rows[kh * r * d_state:(kh + 1) * r * d_state]]
            put(f"{p}.ssm_in.weight", np.concatenate(per_head), Q8_SSM)
        else:
            put(f"{p}.attn_qkv.weight", np.concatenate([q_rows, k_rows, v_rows]), Q8_SSM)
            put(f"{p}.attn_gate.weight", z_rows, Q8_SSM)
        # Per key head: r beta rows, then r alpha rows.
        put(f"{p}.ssm_ba.weight", t((2 * n_v, n_embd), 0.3), Q8_SSM)
        put(f"{p}.ssm_conv1d.weight", t((conv_dim, d_conv), 0.4), F32)
        put(f"{p}.ssm_dt.bias", t((n_v,), 0.5), F32)
        # -exp(A_log): negative, so every head decays.
        put(f"{p}.ssm_a", -np.exp(t((n_v,), 0.5)).astype(np.float32), F32)
        put(f"{p}.ssm_norm.weight", 1.0 + t((d_state,), 0.1), F32)
        put(f"{p}.ssm_out.weight", t((n_embd, value_dim), 0.2), Q8_SSM)
    else:
        # Per head: its head_dim query rows, then its head_dim gate rows.
        put(f"{p}.attn_q.weight", t((2 * n_head * head_dim, n_embd), 0.2), Q8)
        put(f"{p}.attn_k.weight", t((n_head_kv * head_dim, n_embd), 0.2), Q8)
        put(f"{p}.attn_v.weight", t((n_head_kv * head_dim, n_embd), 0.2), Q8)
        put(f"{p}.attn_output.weight", t((n_embd, n_head * head_dim), 0.2), Q8)
        put(f"{p}.attn_q_norm.weight", 1.0 + t((head_dim,), 0.1), F32)
        put(f"{p}.attn_k_norm.weight", 1.0 + t((head_dim,), 0.1), F32)
    # Router logits spread wide enough that experts are chosen decisively.
    put(f"{p}.ffn_gate_inp.weight", t((n_expert, n_embd), 0.6), F32)
    put(f"{p}.ffn_gate_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_up_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_down_exps.weight", t((n_expert, n_embd, n_ff_exp), 0.2), Q8)
    put(f"{p}.ffn_gate_inp_shexp.weight", t((n_embd,), 0.3), F32)
    put(f"{p}.ffn_gate_shexp.weight", t((n_ff_shexp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_up_shexp.weight", t((n_ff_shexp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_down_shexp.weight", t((n_embd, n_ff_shexp), 0.2), Q8)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, "vocab", n_vocab, "layers", n_layer, "interval", interval, "layout", layout)
