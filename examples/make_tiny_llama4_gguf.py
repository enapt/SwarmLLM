"""Build a tiny random-weight Llama-4 GGUF for checking our Llama-4 path against
llama.cpp (llama-cpp-python 0.3.16) with examples/logits_reference_probe.rs +
compare_logits_reference.py. No tiny Llama-4 GGUF exists on HF (2026-09-25),
and a random-weight model is all a logits comparison needs: both engines read
the same file.

Shape follows the real Llama-4-Scout header (unsloth Q2_K, read 2026-09-25):
every layer MoE (`interleave_moe_layer_step` 1), a shared expert of the routed
experts' width, `attn_q/k/v/output`, router `ffn_gate_inp` in F32. Layer 3 is a
NoPE layer ((il + 1) % 4 == 0). Usage:

    python3 examples/make_tiny_llama4_gguf.py OUT.gguf [TOPK]

Environment: QUANT=f32 (unquantized — the check that separates arithmetic
from rounding; with Q8_0 both engines quantize activations and a top-1
router near-tie can flip), MOE_STEP=99 (every layer dense — isolates
attention), N_EXPERT=128 (llama.cpp then turns Q/K norm OFF — Maverick's type).
Then:

    LOGITS_PROBE_GGUF=OUT.gguf LOGITS_PROBE_OUT=/tmp/l4 \
      cargo run --no-default-features --features dev,claude-subscription \
      --example logits_reference_probe
    python3 examples/compare_logits_reference.py OUT.gguf /tmp/l4

Verified 2026-09-25 (FUTURE_WORK #114): QUANT=f32, top-1 and top-2, whole and
split at layer 2, all 24/24 top-1 at cosine >= 0.999996.
"""
import sys
import os
import numpy as np
import gguf

out = sys.argv[1]
topk = int(sys.argv[2]) if len(sys.argv) > 2 else 1
rng = np.random.default_rng(20260925)

n_embd, n_head, n_head_kv, head_dim = 64, 4, 2, 16
import os
n_layer, n_expert, n_ff_exp, n_ctx = int(os.environ.get("N_LAYER", 4)), int(os.environ.get("N_EXPERT", 4)), 64, 256
# MOE_STEP > n_layer makes every layer dense ((il + 1) % step == 0 is MoE).
moe_step = int(os.environ.get("MOE_STEP", 1))
# QUANT=f32 writes every weight unquantized: neither engine then quantizes the
# activations, so what remains between them is arithmetic, not rounding.
Q8 = gguf.GGMLQuantizationType.Q8_0 if os.environ.get("QUANT", "q8") == "q8" else gguf.GGMLQuantizationType.F32
F32 = gguf.GGMLQuantizationType.F32

# SentencePiece-style vocabulary: specials, the 256 byte tokens, then pieces.
pieces = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
words = ["▁the", "▁a", "▁is", "▁of", "▁and", "▁to", "▁in", "▁it", "▁that", "▁was",
         "▁he", "▁she", "▁for", "▁on", "▁are", "▁with", "▁as", "▁I", "▁his", "▁they",
         "▁be", "▁at", "▁one", "▁have", "▁this", "▁from", "▁or", "▁had", "▁by", "▁hot",
         "e", "t", "a", "o", "i", "n", "s", "r", "h", "l"]
# logits_reference_probe picks ids up to ~5100, so the vocabulary must reach past it.
words += [f"▁w{i}" for i in range(5800)]
pieces += words
n_vocab = len(pieces)
types = [2, 3, 3] + [6] * 256 + [1] * len(words)  # UNKNOWN, CONTROL, BYTE, NORMAL
scores = [0.0, 0.0, 0.0] + [0.0] * 256 + [-float(i) for i in range(len(words))]

w = gguf.GGUFWriter(out, "llama4")
w.add_name("tiny-llama4-random")
w.add_block_count(n_layer)
w.add_context_length(n_ctx)
w.add_embedding_length(n_embd)
w.add_feed_forward_length(n_ff_exp * 2)
w.add_head_count(n_head)
w.add_head_count_kv(n_head_kv)
w.add_rope_freq_base(500000.0)
w.add_layer_norm_rms_eps(1e-5)
w.add_expert_count(n_expert)
w.add_expert_used_count(topk)
w.add_key_length(head_dim)
w.add_value_length(head_dim)
w.add_rope_dimension_count(head_dim)
w.add_vocab_size(n_vocab)
w.add_uint32("llama4.interleave_moe_layer_step", moe_step)
w.add_expert_feed_forward_length(n_ff_exp)
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
    if q == gguf.GGMLQuantizationType.Q8_0:
        w.add_tensor(name, gguf.quants.quantize(arr, Q8), raw_shape=gguf.quants.quant_shape_to_byte_shape(arr.shape, Q8), raw_dtype=Q8)
    else:
        w.add_tensor(name, arr)


# numpy shape is ggml's ne reversed: {n_embd, n_vocab} -> (n_vocab, n_embd).
put("token_embd.weight", t((n_vocab, n_embd), 0.5), Q8)
put("output_norm.weight", (1.0 + t((n_embd,), 0.1)), F32)
put("output.weight", t((n_vocab, n_embd), 0.3), Q8)
for i in range(n_layer):
    p = f"blk.{i}"
    put(f"{p}.attn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    put(f"{p}.attn_q.weight", t((n_head * head_dim, n_embd), 0.2), Q8)
    put(f"{p}.attn_k.weight", t((n_head_kv * head_dim, n_embd), 0.2), Q8)
    put(f"{p}.attn_v.weight", t((n_head_kv * head_dim, n_embd), 0.2), Q8)
    put(f"{p}.attn_output.weight", t((n_embd, n_head * head_dim), 0.2), Q8)
    put(f"{p}.ffn_norm.weight", 1.0 + t((n_embd,), 0.1), F32)
    if (i + 1) % moe_step != 0:
        put(f"{p}.ffn_gate.weight", t((n_ff_exp * 2, n_embd), 0.2), Q8)
        put(f"{p}.ffn_up.weight", t((n_ff_exp * 2, n_embd), 0.2), Q8)
        put(f"{p}.ffn_down.weight", t((n_embd, n_ff_exp * 2), 0.2), Q8)
        continue
    # Router logits spread wide enough that experts are chosen decisively.
    put(f"{p}.ffn_gate_inp.weight", t((n_expert, n_embd), 0.6), F32)
    put(f"{p}.ffn_gate_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_up_exps.weight", t((n_expert, n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_down_exps.weight", t((n_expert, n_embd, n_ff_exp), 0.2), Q8)
    put(f"{p}.ffn_gate_shexp.weight", t((n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_up_shexp.weight", t((n_ff_exp, n_embd), 0.2), Q8)
    put(f"{p}.ffn_down_shexp.weight", t((n_embd, n_ff_exp), 0.2), Q8)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("wrote", out, "vocab", n_vocab, "topk", topk)
