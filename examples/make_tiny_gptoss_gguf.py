"""Build a tiny random-weight gpt-oss GGUF for checking our loader against
llama.cpp MASTER (`~/llama.cpp-ref/dump_logits`, REF_BIN=) with
examples/logits_reference_probe.rs + compare_logits_reference.py. The real
gpt-oss-20b is 11.5 GB of MXFP4; a random-weight file is all a logits
comparison needs, because both engines read the same file.

Shape follows the real gpt-oss-20b header (ggml-org/gpt-oss-20b-GGUF
`gpt-oss-20b-MXFP4.gguf`, read 2026-11-02; FUTURE_WORK #390 item 2), scaled
down, keeping every ratio and every tensor that decides the arithmetic:

- More query heads than key/value heads, and query heads x head size WIDER than
  the embedding (64 x 64 = 4096 against 2880 there; 8 x 16 = 128 against 64
  here). Biases on Q, K, V and on the attention's output.
- One attention SINK per query head (`attn_sinks.weight`): an extra score in
  each head's softmax that takes a share of the weight and goes to no key. Drawn
  wide (about +-2) so a path that drops it is far off.
- A sliding window on EVEN layers, full attention on odd ones (llama.cpp's
  `load_swa_pattern(ml, 2)`); the window (SWA, default 8) is shorter than the
  probe's sequence so it decides keys.
- YaRN with the real file's settings (factor 32 from 4,096, beta 32 / 1,
  base 150,000), NEOX rotation over the whole head.
- `post_attention_norm` is the norm BEFORE the experts (there is no ffn_norm).
- Every layer routes each token to N_EXPERT_USED of N_EXPERT experts: the
  router's logits PLUS its bias choose them, and a softmax over the chosen
  logits weighs them. Each expert has gate/up/down biases, and the activation
  is gpt-oss's clamped SwiGLU (`swiglu_oai`, limit 7, alpha 1.702, up + 1),
  so the up and gate projections are drawn wide enough to reach the clamps.
- Experts in MXFP4 (type 39) as in the real file; attention, embeddings and
  output in Q8_0; router, norms, biases and sinks in F32. Untied output.

Usage:

    PYTHONPATH=~/llama.cpp-ref/gguf-py python3 examples/make_tiny_gptoss_gguf.py OUT.gguf

Environment: QUANT=real (default: the real file's types) or f32 (every tensor
f32 — the control for "is the residual rounding"); DEQUANTIZED=1 (with real)
writes the quantized weights back as f32: the same numbers, read exactly — the
answer both engines' quantized runs approximate; N_LAYER (default 4); SWA
(default 8, 0 = no window); SINKS=0 (every sink -inf: none); N_EXPERT (8),
N_EXPERT_USED (4). Then:

    SWARMLLM_UNCHECKED_FAMILIES=gpt-oss LOGITS_PROBE_GGUF=OUT.gguf LOGITS_PROBE_OUT=/tmp/gptoss \\
      cargo run --no-default-features --features dev,claude-subscription \\
      --example logits_reference_probe
    REF_BIN=~/llama.cpp-ref/dump_logits python3 examples/compare_logits_reference.py OUT.gguf /tmp/gptoss

Results: docs/invariants/inference.md § "gpt-oss: the whole forward against llama.cpp's".
"""
import os
import sys

import numpy as np
import gguf

out = sys.argv[1]
rng = np.random.default_rng(20261102)

n_embd, n_head, n_head_kv, head_dim = 64, 8, 2, 16
n_layer = int(os.environ.get("N_LAYER", 4))
n_expert = int(os.environ.get("N_EXPERT", 8))
n_expert_used = int(os.environ.get("N_EXPERT_USED", 4))
# Not equal to n_embd (the real file's are), so a stack read transposed fails;
# a multiple of 32, which MXFP4's blocks need along each row.
n_ff_exp = 96
swa = int(os.environ.get("SWA", 8))
sinks_on = os.environ.get("SINKS", "1") != "0"
n_ctx = 131072

quant = os.environ.get("QUANT", "real")
F32 = gguf.GGMLQuantizationType.F32
Q8 = gguf.GGMLQuantizationType.Q8_0 if quant == "real" else F32
MXFP4 = gguf.GGMLQuantizationType.MXFP4 if quant == "real" else F32
dequantized = os.environ.get("DEQUANTIZED") == "1"

# SentencePiece-style vocabulary: specials, the 256 byte tokens, then pieces.
# The probe feeds token ids, so the real file's gpt-4o BPE is not needed here.
pieces = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
words = ["▁the", "▁a", "▁is", "▁of", "▁and", "▁to", "▁in", "▁it", "▁that", "▁was"]
# logits_reference_probe picks ids up to ~5100, so the vocabulary must reach past it.
words += [f"▁w{i}" for i in range(5800)]
pieces += words
n_vocab = len(pieces)
types = [2, 3, 3] + [6] * 256 + [1] * len(words)  # UNKNOWN, CONTROL, BYTE, NORMAL
scores = [0.0, 0.0, 0.0] + [0.0] * 256 + [-float(i) for i in range(len(words))]

w = gguf.GGUFWriter(out, "gpt-oss")
w.add_name("tiny-gpt-oss-random")
w.add_block_count(n_layer)
w.add_context_length(n_ctx)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(n_ff_exp)
w.add_head_count(n_head)
w.add_head_count_kv(n_head_kv)
w.add_rope_scaling_type(gguf.RopeScalingType.YARN)
w.add_rope_scaling_factor(32.0)
w.add_rope_scaling_orig_ctx_len(4096)
w.add_rope_scaling_yarn_beta_fast(32.0)
w.add_rope_scaling_yarn_beta_slow(1.0)
w.add_rope_freq_base(150000.0)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(n_expert)
w.add_expert_used_count(n_expert_used)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
# llama.cpp requires the key; a window as long as the context is no window.
w.add_sliding_window(swa if swa > 0 else n_ctx)
w.add_expert_feed_forward_length(n_ff_exp)
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
    if q == F32:
        w.add_tensor(name, arr)
    elif dequantized:
        w.add_tensor(name, gguf.quants.dequantize(gguf.quants.quantize(arr, q), q).astype(np.float32).reshape(arr.shape))
    else:
        w.add_tensor(name, gguf.quants.quantize(arr, q), raw_shape=gguf.quants.quant_shape_to_byte_shape(arr.shape, q), raw_dtype=q)


# numpy shape is ggml's ne reversed: {n_embd, n_vocab} -> (n_vocab, n_embd).
put("token_embd.weight", t((n_vocab, n_embd), 0.5), Q8)
put("output_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
put("output.weight", t((n_vocab, n_embd), 0.3), Q8)
for i in range(n_layer):
    p = f"blk.{i}"
    put(f"{p}.attn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    put(f"{p}.attn_q.weight", t((n_head * head_dim, n_embd), 0.25), Q8)
    put(f"{p}.attn_k.weight", t((n_head_kv * head_dim, n_embd), 0.25), Q8)
    put(f"{p}.attn_v.weight", t((n_head_kv * head_dim, n_embd), 0.25), Q8)
    put(f"{p}.attn_q.bias", t((n_head * head_dim,), 0.3), F32)
    put(f"{p}.attn_k.bias", t((n_head_kv * head_dim,), 0.3), F32)
    put(f"{p}.attn_v.bias", t((n_head_kv * head_dim,), 0.3), F32)
    put(f"{p}.attn_output.weight", t((n_embd, n_head * head_dim), 0.15), Q8)
    put(f"{p}.attn_output.bias", t((n_embd,), 0.3), F32)
    sinks = t((n_head,), 1.0) if sinks_on else np.full((n_head,), -np.inf, dtype=np.float32)
    put(f"{p}.attn_sinks.weight", sinks, F32)
    put(f"{p}.post_attention_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    # Router logits spread wide enough to choose decisively, and a bias as wide
    # as their spread, so a router that ignores it picks other experts.
    put(f"{p}.ffn_gate_inp.weight", t((n_expert, n_embd), 0.6), F32)
    put(f"{p}.ffn_gate_inp.bias", t((n_expert,), 1.0), F32)
    # Gate and up wide enough that some products pass the +-7 clamps.
    put(f"{p}.ffn_gate_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.6), MXFP4)
    put(f"{p}.ffn_up_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.6), MXFP4)
    put(f"{p}.ffn_down_exps.weight", t((n_expert, n_embd, n_ff_exp), 0.1), MXFP4)
    put(f"{p}.ffn_gate_exps.bias", t((n_expert, n_ff_exp), 0.5), F32)
    put(f"{p}.ffn_up_exps.bias", t((n_expert, n_ff_exp), 0.5), F32)
    put(f"{p}.ffn_down_exps.bias", t((n_expert, n_embd), 0.3), F32)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, "vocab", n_vocab, "layers", n_layer, "window", swa, "sinks", sinks_on, "quant", quant)
