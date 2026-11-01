"""Build a tiny random-weight GLM-4.5 (`glm4moe`) GGUF for checking our loader
against llama.cpp MASTER (`~/llama.cpp-ref/dump_logits`, REF_BIN=) with
examples/logits_reference_probe.rs + compare_logits_reference.py. The real
GLM-4.5-Air is 45-68 GB; a random-weight file is all a logits comparison needs,
because both engines read the same file.

Shape follows the real GLM-4.5-Air header (unsloth Q2_K, read 2026-11-02;
FUTURE_WORK #390), scaled down, keeping every ratio that decides the arithmetic:

- Attention with biases on Q, K and V, more query heads than key/value heads,
  and NEOX RoPE over HALF of each head (`rope.dimension_count` 64 of 128).
  QK_NORM=1 adds the per-head Q/K RMS norms the full GLM-4.5 carries.
- `post_attention_norm` is the norm BEFORE the feed-forward (HF's
  `post_attention_layernorm`) — GLM-4-0414's dense files use the same name for a
  norm of the attention's OUTPUT, so a loader reading it the dense way is wrong
  here.
- The first `leading_dense_block_count` layers (1) are a dense SwiGLU; the rest
  route each token to N_EXPERT_USED of N_EXPERT experts by SIGMOID scores plus a
  per-expert selection bias (`exp_probs_b`), which decides the CHOICE only — the
  weights are the unbiased scores, renormalised over the chosen and scaled by
  `expert_weights_scale` (W_SCALE, 2.5 here as in the full GLM-4.5; Air's is 1.0)
  — plus an always-on shared expert with no gate. The bias is large beside the
  scores, so a router that ignores it, or weighs with it, picks or weighs
  different experts.
- One next-token-prediction block after the model's own (`nextn_predict_layers`
  1, `blk.<N_LAYER>.*` with its `nextn.*` tensors), which an ordinary forward
  never runs.

Usage:

    PYTHONPATH=~/llama.cpp-ref/gguf-py python3 examples/make_tiny_glm4moe_gguf.py OUT.gguf

Environment: QUANT=f32 (default) or q8; DEQUANTIZED=1 (with q8) writes the Q8_0
blocks back as f32, the rounding control; N_LAYER (default 4, the model's own);
QK_NORM=1; W_SCALE (default 2.5); N_EXPERT (8), N_EXPERT_USED (3); EXPERT_GROUPS and EXPERT_GROUPS_USED
(default 1 and EXPERT_GROUPS: a file limiting its experts to the best groups is refused). Then:

    SWARMLLM_UNCHECKED_FAMILIES=glm4moe LOGITS_PROBE_GGUF=OUT.gguf LOGITS_PROBE_OUT=/tmp/glm \\
      cargo run --no-default-features --features dev,claude-subscription \\
      --example logits_reference_probe
    REF_BIN=~/llama.cpp-ref/dump_logits python3 examples/compare_logits_reference.py OUT.gguf /tmp/glm

The family is refused until a real file is checked (`ModelArch::is_supported`, #390), so the
probe needs the switch. Results: docs/invariants/inference.md § "GLM-4.5".
"""
import os
import sys

import numpy as np
import gguf

out = sys.argv[1]
rng = np.random.default_rng(20261102)

n_embd, n_head, n_head_kv, head_dim = 64, 4, 2, 32
rope_dim = head_dim // 2
n_layer = int(os.environ.get("N_LAYER", 4))
n_nextn = 1
n_dense_lead = 1
n_expert = int(os.environ.get("N_EXPERT", 8))
n_expert_used = int(os.environ.get("N_EXPERT_USED", 3))
n_ff, n_ff_exp = 96, 32
n_ff_shexp = n_ff_exp  # expert_shared_count 1
w_scale = float(os.environ.get("W_SCALE", 2.5))
qk_norm = os.environ.get("QK_NORM") == "1"
n_ctx = 256

quant = os.environ.get("QUANT", "f32")
F32 = gguf.GGMLQuantizationType.F32
Q8 = gguf.GGMLQuantizationType.Q8_0 if quant == "q8" else F32
dequantized = os.environ.get("DEQUANTIZED") == "1"

# SentencePiece-style vocabulary: specials, the 256 byte tokens, then pieces.
pieces = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
words = ["▁the", "▁a", "▁is", "▁of", "▁and", "▁to", "▁in", "▁it", "▁that", "▁was"]
# logits_reference_probe picks ids up to ~5100, so the vocabulary must reach past it.
words += [f"▁w{i}" for i in range(5800)]
pieces += words
n_vocab = len(pieces)
types = [2, 3, 3] + [6] * 256 + [1] * len(words)  # UNKNOWN, CONTROL, BYTE, NORMAL
scores = [0.0, 0.0, 0.0] + [0.0] * 256 + [-float(i) for i in range(len(words))]

w = gguf.GGUFWriter(out, "glm4moe")
w.add_name("tiny-glm4moe-random")
w.add_block_count(n_layer + n_nextn)
w.add_nextn_predict_layers(n_nextn)
w.add_context_length(n_ctx)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head_kv)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_rope_dimension_count(rope_dim)
w.add_rope_freq_base(1000000.0)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(n_expert)
w.add_expert_used_count(n_expert_used)
w.add_expert_group_count(int(os.environ.get("EXPERT_GROUPS", 1)))
w.add_expert_group_used_count(int(os.environ.get("EXPERT_GROUPS_USED", os.environ.get("EXPERT_GROUPS", 1))))
w.add_expert_feed_forward_length(n_ff_exp)
w.add_expert_shared_count(1)
w.add_leading_dense_block_count(n_dense_lead)
w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
w.add_expert_weights_scale(w_scale)
w.add_expert_weights_norm(True)
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


def put(name, arr, q):
    if q == gguf.GGMLQuantizationType.Q8_0 and dequantized:
        w.add_tensor(name, gguf.quants.dequantize(gguf.quants.quantize(arr, q), q).astype(np.float32).reshape(arr.shape))
    elif q == gguf.GGMLQuantizationType.Q8_0:
        w.add_tensor(name, gguf.quants.quantize(arr, q), raw_shape=gguf.quants.quant_shape_to_byte_shape(arr.shape, q), raw_dtype=q)
    else:
        w.add_tensor(name, arr)


def attention_and_norms(p):
    put(f"{p}.attn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    put(f"{p}.attn_q.weight", t((n_head * head_dim, n_embd), 0.2), Q8)
    put(f"{p}.attn_k.weight", t((n_head_kv * head_dim, n_embd), 0.2), Q8)
    put(f"{p}.attn_v.weight", t((n_head_kv * head_dim, n_embd), 0.2), Q8)
    put(f"{p}.attn_q.bias", t((n_head * head_dim,), 0.3), F32)
    put(f"{p}.attn_k.bias", t((n_head_kv * head_dim,), 0.3), F32)
    put(f"{p}.attn_v.bias", t((n_head_kv * head_dim,), 0.3), F32)
    put(f"{p}.attn_output.weight", t((n_embd, n_head * head_dim), 0.2), Q8)
    if qk_norm:
        put(f"{p}.attn_q_norm.weight", 1.0 + t((head_dim,), 0.1), F32)
        put(f"{p}.attn_k_norm.weight", 1.0 + t((head_dim,), 0.1), F32)
    put(f"{p}.post_attention_norm.weight", 1.0 + t((n_embd,), 0.1), F32)


def experts(p):
    # Router logits spread wide enough that experts are chosen decisively, and a
    # selection bias as large as the sigmoid scores' own spread.
    put(f"{p}.ffn_gate_inp.weight", t((n_expert, n_embd), 0.6), F32)
    put(f"{p}.exp_probs_b.bias", t((n_expert,), 0.4), F32)
    put(f"{p}.ffn_gate_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_up_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_down_exps.weight", t((n_expert, n_embd, n_ff_exp), 0.2), Q8)
    put(f"{p}.ffn_gate_shexp.weight", t((n_ff_shexp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_up_shexp.weight", t((n_ff_shexp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_down_shexp.weight", t((n_embd, n_ff_shexp), 0.2), Q8)


# numpy shape is ggml's ne reversed: {n_embd, n_vocab} -> (n_vocab, n_embd).
put("token_embd.weight", t((n_vocab, n_embd), 0.5), Q8)
put("output_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
put("output.weight", t((n_vocab, n_embd), 0.3), Q8)
for i in range(n_layer):
    p = f"blk.{i}"
    attention_and_norms(p)
    if i < n_dense_lead:
        put(f"{p}.ffn_gate.weight", t((n_ff, n_embd), 0.2), Q8)
        put(f"{p}.ffn_up.weight", t((n_ff, n_embd), 0.2), Q8)
        put(f"{p}.ffn_down.weight", t((n_embd, n_ff), 0.2), Q8)
    else:
        experts(p)
# The next-token-prediction block: a whole MoE layer plus its own tensors. Random,
# so a forward that ran it would part from llama.cpp's at once.
p = f"blk.{n_layer}"
attention_and_norms(p)
experts(p)
put(f"{p}.nextn.eh_proj.weight", t((n_embd, 2 * n_embd), 0.2), Q8)
put(f"{p}.nextn.embed_tokens.weight", t((n_vocab, n_embd), 0.5), Q8)
put(f"{p}.nextn.enorm.weight", 1.0 + t((n_embd,), 0.1), F32)
put(f"{p}.nextn.hnorm.weight", 1.0 + t((n_embd,), 0.1), F32)
put(f"{p}.nextn.shared_head_head.weight", t((n_vocab, n_embd), 0.3), Q8)
put(f"{p}.nextn.shared_head_norm.weight", 1.0 + t((n_embd,), 0.1), F32)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, "vocab", n_vocab, "layers", n_layer, "+ nextn", n_nextn, "qk_norm", qk_norm, "w_scale", w_scale)
