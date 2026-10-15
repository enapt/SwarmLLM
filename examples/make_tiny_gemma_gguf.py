"""Build a tiny random-weight Gemma 2 or Gemma 3 GGUF for checking our Gemma path
against llama.cpp (llama-cpp-python 0.3.16) with examples/logits_reference_probe.rs +
compare_logits_reference.py, with f32 weights so quantization noise cannot hide a
logic error. What a real file cannot show quickly:

- the sliding window (FUTURE_WORK #278): SWA=N sets `gemma2.attention.sliding_window`
  to N (default 8), so a 40-token prompt crosses it on every even layer — the real
  2B's window is 4,096 positions;
- the 27B attention scale (#279): llama.cpp takes a 46-layer Gemma 2 for the 27B
  and scales queries by 1/sqrt(n_embd / n_head) instead of 1/sqrt(head_dim)
  (`llama_model_gemma2::load_arch_hparams`). Here n_embd / n_head = 16 and
  head_dim = 32, so a wrong scale is far outside quantization noise;
- ARCH=gemma3 (#115 (1)): per-head Q/K norms, no soft-capping, the window on five
  layers in six, the windowed layers' RoPE at base 10,000 against the file's
  1,000,000, and linear scaling ROPE_FACTOR (default 8) on the global layers —
  the real 4B's shape at 40 tokens.

Usage:

    PYTHONPATH=~/llama.cpp-ref/gguf-py python3 examples/make_tiny_gemma_gguf.py OUT.gguf

Environment: ARCH=gemma2 (default) or gemma3; N_LAYER=46 for gemma2 (26 is the 2B
type, whose scale is 1/sqrt(head_dim)), 12 for gemma3 (62 is its 27B); SWA=8 (0
writes a window of 0 — no window; Gemma 2 only); QUANT=f32 (default; q8 for Q8_0).
Then:

    LOGITS_PROBE_GGUF=OUT.gguf LOGITS_PROBE_OUT=/tmp/g2 \\
      cargo run --no-default-features --features dev,claude-subscription \\
      --example logits_reference_probe
    python3 examples/compare_logits_reference.py OUT.gguf /tmp/g2

Verified 2026-10-15 (#278), 40 tokens, window 8, N_LAYER=26: worst cosine 0.99928
whole, in 5-token pieces (LOGITS_PROBE_CHUNK=5) and split at layer 3; 0.334 with
SWARMLLM_SLIDING_WINDOW=0. llama.cpp keeps its KV cache in f16, which is the
remaining gap.

Verified 2026-10-15 (#279), N_LAYER=46: with QK_STD=0.25 worst cosine 0.99987
(0.727 with SWARMLLM_QUERY_SCALE=0). With the default QK_STD=0.5 only 0.966:
rounding amplified over 46 decisive layers, not logic (llama.cpp with an f32 KV
cache: 0.992).
"""
import os
import sys

import gguf
import numpy as np

out = sys.argv[1]
rng = np.random.default_rng(20261015)

n_embd, n_head, n_head_kv, head_dim, n_ff, n_ctx = 64, 4, 2, 32, 128, 512
arch = os.environ.get("ARCH", "gemma2")
assert arch in ("gemma2", "gemma3"), arch
n_layer = int(os.environ.get("N_LAYER", 46 if arch == "gemma2" else 12))
rope_factor = float(os.environ.get("ROPE_FACTOR", 8.0))
swa = int(os.environ.get("SWA", 8))
# QK_STD: the spread of the query and key weights. 0.5 makes attention decisive, so a wrong
# scale shows at once; it also amplifies rounding over 46 layers (llama.cpp keeps its KV
# cache in f16), so a softer file separates the two.
qk_std = float(os.environ.get("QK_STD", 0.5))
Q = gguf.GGMLQuantizationType.Q8_0 if os.environ.get("QUANT", "f32") == "q8" else gguf.GGMLQuantizationType.F32
F32 = gguf.GGMLQuantizationType.F32

# SentencePiece-style vocabulary: specials, the 256 byte tokens, then pieces.
pieces = ["<pad>", "<eos>", "<bos>", "<unk>"] + [f"<0x{b:02X}>" for b in range(256)]
words = ["▁the", "▁a", "▁is", "▁of", "▁and", "▁to", "▁in", "▁it", "▁that", "▁was"]
# logits_reference_probe picks ids up to ~5100, so the vocabulary must reach past it.
words += [f"▁w{i}" for i in range(5800)]
pieces += words
n_vocab = len(pieces)
types = [3, 3, 3, 2] + [6] * 256 + [1] * len(words)  # CONTROL, UNKNOWN, BYTE, NORMAL
scores = [0.0] * 4 + [0.0] * 256 + [-float(i) for i in range(len(words))]

w = gguf.GGUFWriter(out, arch)
w.add_name(f"tiny-{arch}-random")
w.add_block_count(n_layer)
w.add_context_length(n_ctx)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(n_ff)
w.add_head_count(n_head)
w.add_head_count_kv(n_head_kv)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_layer_norm_rms_eps(1e-6)
w.add_sliding_window(swa)
if arch == "gemma2":
    w.add_rope_freq_base(10000.0)
    w.add_attn_logit_softcapping(50.0)
    w.add_final_logit_softcapping(30.0)
else:
    w.add_rope_freq_base(1_000_000.0)
    w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
    w.add_rope_scaling_factor(rope_factor)
w.add_vocab_size(n_vocab)
w.add_tokenizer_model("llama")
w.add_token_list(pieces)
w.add_token_scores(scores)
w.add_token_types(types)
w.add_pad_token_id(0)
w.add_eos_token_id(1)
w.add_bos_token_id(2)
w.add_unk_token_id(3)
w.add_add_bos_token(True)
w.add_add_eos_token(False)


def t(shape, scale):
    return (rng.standard_normal(shape) * scale).astype(np.float32)


def put(name, arr, q):
    if q == gguf.GGMLQuantizationType.Q8_0:
        w.add_tensor(name, gguf.quants.quantize(arr, q), raw_shape=gguf.quants.quant_shape_to_byte_shape(arr.shape, q), raw_dtype=q)
    else:
        w.add_tensor(name, arr)


# numpy shape is ggml's ne reversed: {n_embd, n_vocab} -> (n_vocab, n_embd).
# The output head is the embedding table (tied), as in every Gemma 2 file.
put("token_embd.weight", t((n_vocab, n_embd), 0.5), Q)
put("output_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
for i in range(n_layer):
    p = f"blk.{i}"
    put(f"{p}.attn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    # Wide enough that attention is decisive: a wrong scale or a missing window
    # then moves the output well past rounding.
    put(f"{p}.attn_q.weight", t((n_head * head_dim, n_embd), qk_std), Q)
    put(f"{p}.attn_k.weight", t((n_head_kv * head_dim, n_embd), qk_std), Q)
    put(f"{p}.attn_v.weight", t((n_head_kv * head_dim, n_embd), 0.3), Q)
    put(f"{p}.attn_output.weight", t((n_embd, n_head * head_dim), 0.15), Q)
    if arch == "gemma3":
        put(f"{p}.attn_q_norm.weight", 1.0 + t((head_dim,), 0.1), F32)
        put(f"{p}.attn_k_norm.weight", 1.0 + t((head_dim,), 0.1), F32)
    put(f"{p}.post_attention_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    put(f"{p}.ffn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    put(f"{p}.ffn_gate.weight", t((n_ff, n_embd), 0.2), Q)
    put(f"{p}.ffn_up.weight", t((n_ff, n_embd), 0.2), Q)
    put(f"{p}.ffn_down.weight", t((n_embd, n_ff), 0.15), Q)
    put(f"{p}.post_ffw_norm.weight", 1.0 + t((n_embd,), 0.1), F32)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, arch, "layers", n_layer, "window", swa, "vocab", n_vocab)
