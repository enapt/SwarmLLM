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
  the real 4B's shape at 40 tokens; ROPE_TYPE=yarn (#280) declares YaRN instead,
  over an original context of ORIG_CTX (default 16), so the ramp turns inside a
  40-token prompt;
- ARCH=gemma4 (#115 (3)): the 12B/31B shape — head width 32 on windowed layers
  and 64 on global ones, where K doubles as V; an unweighted norm on V; no query
  scaling; "proportional" RoPE (a `rope_freqs` table turning only the first quarter
  of each global head); a per-layer output scale; a final softcap. MOE=1 adds the
  26B-A4B's experts beside the dense MLP: the router's own norm and scale, a fused
  gate/up table and a per-expert output scale.

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

Verified 2026-10-16 (#115 (1), #280), ARCH=gemma3 QK_STD=0.25: worst cosine
1.000000 whole, in 5-token pieces and split at layer 7 (0.987 with
SWARMLLM_ROPE_SCALING=0, 0.892 with SWARMLLM_LAYER_ROPE=0, 0.601 with
SWARMLLM_SLIDING_WINDOW=0); with ROPE_TYPE=yarn 1.000000 at ORIG_CTX 16 and 4096.
"""
import os
import sys

import gguf
import numpy as np

out = sys.argv[1]
rng = np.random.default_rng(20261015)

n_embd, n_head, n_head_kv, head_dim, n_ff, n_ctx = 64, 4, 2, 32, 128, 512
arch = os.environ.get("ARCH", "gemma2")
assert arch in ("gemma2", "gemma3", "gemma4"), arch
n_layer = int(os.environ.get("N_LAYER", 46 if arch == "gemma2" else 12))
# Gemma 4: global layers are wider and carry fewer K/V heads, as in every real file.
head_dim_full, n_head_kv_full = 64, 1
moe = arch == "gemma4" and os.environ.get("MOE", "0") == "1"
# N_EXPERT_USED=8 routes every token through every expert: no top-k pick, so no
# router near-tie can flip, and what is left of a gap is the arithmetic.
n_expert, n_expert_used, n_ff_exp = 8, int(os.environ.get("N_EXPERT_USED", 2)), 32
is_swa = [i % 6 != 5 for i in range(n_layer)]
rope_factor = float(os.environ.get("ROPE_FACTOR", 8.0))
rope_type = os.environ.get("ROPE_TYPE", "linear")
assert rope_type in ("linear", "yarn"), rope_type
swa = int(os.environ.get("SWA", 8))
# QK_STD: the spread of the query and key weights. 0.5 makes attention decisive, so a wrong
# scale shows at once; it also amplifies rounding over 46 layers (llama.cpp keeps its KV
# cache in f16), so a softer file separates the two.
qk_std = float(os.environ.get("QK_STD", 0.5))
# Gemma 4 norms Q and K and applies no scale, so there QK_STD sets the norm gain: 0.5 is
# a gain of 1 (scores of order sqrt(head_dim)), 0.25 halves both and quarters the scores.
qk_gain = 2 * qk_std
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
if arch == "gemma4":
    w.add_head_count_kv([n_head_kv if s else n_head_kv_full for s in is_swa])
    w.add_key_length(head_dim_full)
    w.add_value_length(head_dim_full)
    w.add_key_length_swa(head_dim)
    w.add_value_length_swa(head_dim)
    w.add_rope_dimension_count(head_dim_full)
    w.add_rope_dimension_count_swa(head_dim)
    w.add_sliding_window_pattern(is_swa)
    w.add_shared_kv_layers(0)
    w.add_embedding_length_per_layer_input(0)
    w.add_final_logit_softcapping(30.0)
    if moe:
        w.add_expert_count(n_expert)
        w.add_expert_used_count(n_expert_used)
        w.add_expert_feed_forward_length(n_ff_exp)
else:
    w.add_head_count_kv(n_head_kv)
    w.add_key_length(head_dim)
    w.add_value_length(head_dim)
w.add_layer_norm_rms_eps(1e-6)
w.add_sliding_window(swa)
if arch == "gemma2":
    w.add_rope_freq_base(10000.0)
    w.add_attn_logit_softcapping(50.0)
    w.add_final_logit_softcapping(30.0)
elif arch == "gemma4":
    w.add_rope_freq_base(1_000_000.0)
    w.add_rope_freq_base_swa(10_000.0)
else:
    w.add_rope_freq_base(1_000_000.0)
    w.add_rope_scaling_type(gguf.RopeScalingType.YARN if rope_type == "yarn" else gguf.RopeScalingType.LINEAR)
    w.add_rope_scaling_factor(rope_factor)
    if rope_type == "yarn":
        w.add_rope_scaling_orig_ctx_len(int(os.environ.get("ORIG_CTX", 16)))
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
if arch == "gemma4":
    # "Proportional" RoPE, as the converter writes it: the first quarter of each global
    # head's pairs turn, the rest are held still by a divisor of 1e30.
    n_rot_pairs = head_dim_full // 2 // 4
    put("rope_freqs.weight", np.array([1.0] * n_rot_pairs + [1e30] * (head_dim_full // 2 - n_rot_pairs), dtype=np.float32), F32)
for i in range(n_layer):
    p = f"blk.{i}"
    put(f"{p}.attn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    if arch == "gemma4":
        hd, kv = (head_dim, n_head_kv) if is_swa[i] else (head_dim_full, n_head_kv_full)
        put(f"{p}.attn_q.weight", t((n_head * hd, n_embd), qk_std), Q)
        put(f"{p}.attn_k.weight", t((kv * hd, n_embd), qk_std), Q)
        if is_swa[i]:
            # Global layers have no V: their K is used as V too.
            put(f"{p}.attn_v.weight", t((kv * hd, n_embd), 0.3), Q)
        put(f"{p}.attn_output.weight", t((n_embd, n_head * hd), 0.15), Q)
        # Q and K are normed before the product and nothing scales it, so their norm
        # weights, not QK_STD's projections, set how decisive attention is.
        put(f"{p}.attn_q_norm.weight", qk_gain * (1.0 + t((hd,), 0.1)), F32)
        put(f"{p}.attn_k_norm.weight", qk_gain * (1.0 + t((hd,), 0.1)), F32)
        put(f"{p}.post_attention_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
        put(f"{p}.ffn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
        put(f"{p}.ffn_gate.weight", t((n_ff, n_embd), 0.2), Q)
        put(f"{p}.ffn_up.weight", t((n_ff, n_embd), 0.2), Q)
        put(f"{p}.ffn_down.weight", t((n_embd, n_ff), 0.15), Q)
        put(f"{p}.post_ffw_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
        put(f"{p}.layer_output_scale.weight", np.array([0.5 + 0.5 * rng.random()], dtype=np.float32), F32)
        if moe:
            put(f"{p}.ffn_gate_inp.weight", t((n_expert, n_embd), 0.5), F32)
            put(f"{p}.ffn_gate_inp.scale", 1.0 + t((n_embd,), 0.1), F32)
            put(f"{p}.pre_ffw_norm_2.weight", 1.0 + t((n_embd,), 0.1), F32)
            put(f"{p}.post_ffw_norm_1.weight", 1.0 + t((n_embd,), 0.1), F32)
            put(f"{p}.post_ffw_norm_2.weight", 1.0 + t((n_embd,), 0.1), F32)
            # ggml ne {n_embd, 2 * n_ff_exp, n_expert}: gate rows first, then up.
            put(f"{p}.ffn_gate_up_exps.weight", t((n_expert, 2 * n_ff_exp, n_embd), 0.2), Q)
            put(f"{p}.ffn_down_exps.weight", t((n_expert, n_embd, n_ff_exp), 0.15), Q)
            put(f"{p}.ffn_down_exps.scale", (0.5 + rng.random(n_expert)).astype(np.float32), F32)
        continue
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
