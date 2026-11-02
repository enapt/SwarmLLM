"""Exact (float64) forward of a tiny gpt-oss GGUF (examples/make_tiny_gptoss_gguf.py),
as llama.cpp's openai-moe.cpp computes it: YaRN NEOX rotation, a sliding window on even
layers, one attention sink per query head, biases on Q/K/V/O, a router whose bias
weighs, per-expert gate/up/down biases and the clamped SwiGLU. It says which engine
sits nearer the exact answer where a logits comparison of the two leaves it open.

    PYTHONPATH=~/llama.cpp-ref/gguf-py python3 examples/gptoss_exact_reference.py \\
        MODEL.gguf OURS.json OURS.f32 LLAMA_CPP.f32

OURS.* are logits_reference_probe's output; LLAMA_CPP.f32 is
`~/llama.cpp-ref/dump_logits MODEL.gguf LLAMA_CPP.f32 <OURS.json's tokens, comma-joined>`.
F16_TOKEN_TABLE=1 rounds the token table to f16, as this engine holds it
(`split::token_embedding`). Only f32 files: the weights are read as stored, so a
quantized file's control is its DEQUANTIZED=1 twin. Results: docs/invariants/inference.md
§ "gpt-oss: the whole forward against llama.cpp's"."""
import json
import math
import os
import sys

import numpy as np
import gguf

path, ids_json, ours_path, ref_path = sys.argv[1:5]
r = gguf.GGUFReader(path)
T = {t.name: np.asarray(t.data, dtype=np.float64).reshape([int(d) for d in reversed(t.shape)]) for t in r.tensors}


def kv(key):
    f = r.fields[key]
    return f.parts[f.data[0]].tolist()[0] if len(f.data) == 1 else None


A = "gpt-oss"
n_layer = kv(f"{A}.block_count")
n_head, n_kv = kv(f"{A}.attention.head_count"), kv(f"{A}.attention.head_count_kv")
hd = kv(f"{A}.attention.key_length")
base = kv(f"{A}.rope.freq_base")
eps = kv(f"{A}.attention.layer_norm_rms_epsilon")
k_used = kv(f"{A}.expert_used_count")
n_swa = kv(f"{A}.attention.sliding_window")
factor = kv(f"{A}.rope.scaling.factor")
n_ctx_orig = kv(f"{A}.rope.scaling.original_context_length")
beta_fast = kv(f"{A}.rope.scaling.yarn_beta_fast")
beta_slow = kv(f"{A}.rope.scaling.yarn_beta_slow")

meta = json.load(open(ids_json))
ids = meta["tokens"]
N = len(ids)
# With LOGITS_PROBE_KEEP_LAST the probe keeps rows from `first_position` on (the
# prompt before it read in LOGITS_PROBE_CHUNK-sized chunks for its cache only).
first = meta.get("first_position", 0)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def yarn_angles(pos):
    """ggml's rope_yarn (ggml-cpu/ops.cpp): per pair, interpolated and extrapolated
    angles mixed by a ramp over the correction dims, and the magnitude scaled by
    1 + 0.1 ln(factor)."""
    n_dims = hd
    freq_scale = 1.0 / factor

    def corr(beta):
        return n_dims * math.log(n_ctx_orig / (beta * 2 * math.pi)) / (2 * math.log(base))

    low = max(0.0, math.floor(corr(beta_fast)))
    high = min(n_dims - 1.0, math.ceil(corr(beta_slow)))
    half = n_dims // 2
    i0 = 2 * np.arange(half, dtype=np.float64)
    theta_extrap = pos[:, None] * base ** (-i0 / n_dims)[None, :]
    theta_interp = freq_scale * theta_extrap
    y = (i0 / 2 - low) / max(0.001, high - low)
    ramp = 1 - np.clip(y, 0, 1)  # ext_factor 1
    theta = theta_interp * (1 - ramp) + theta_extrap * ramp
    mscale = 1.0 + 0.1 * math.log(1.0 / freq_scale)
    return np.cos(theta) * mscale, np.sin(theta) * mscale


def neox(x, c, s):  # x [N, heads, hd]
    half = hd // 2
    a, b = x[..., :half], x[..., half:]
    c, s = c[:, None, :], s[:, None, :]
    return np.concatenate([a * c - b * s, a * s + b * c], -1)


def swiglu_oai(g, u):
    x = np.minimum(g, 7.0)
    y = np.clip(u, -7.0, 7.0)
    return x / (1 + np.exp(-1.702 * x)) * (y + 1)


x = T["token_embd.weight"][ids]
if os.environ.get("F16_TOKEN_TABLE") == "1":
    x = x.astype(np.float16).astype(np.float64)
pos = np.arange(N, dtype=np.float64)
c, s = yarn_angles(pos)
causal = np.triu(np.full((N, N), -np.inf), 1)
dist = pos[:, None] - pos[None, :]
windowed = np.where(dist >= n_swa, -np.inf, causal)
margins = []
for il in range(n_layer):
    p = f"blk.{il}"
    h = rms(x, T[f"{p}.attn_norm.weight"])
    q = (h @ T[f"{p}.attn_q.weight"].T + T[f"{p}.attn_q.bias"]).reshape(N, n_head, hd)
    k = (h @ T[f"{p}.attn_k.weight"].T + T[f"{p}.attn_k.bias"]).reshape(N, n_kv, hd)
    v = (h @ T[f"{p}.attn_v.weight"].T + T[f"{p}.attn_v.bias"]).reshape(N, n_kv, hd)
    q, k = neox(q, c, s), neox(k, c, s)
    mask = windowed if il % 2 == 0 else causal
    sinks = T[f"{p}.attn_sinks.weight"]
    g = n_head // n_kv
    out = np.zeros((N, n_head, hd))
    for hh in range(n_head):
        sc = q[:, hh] @ k[:, hh // g].T / math.sqrt(hd) + mask
        m = np.maximum(sc.max(1, keepdims=True), sinks[hh])
        e = np.exp(sc - m)
        den = e.sum(1, keepdims=True) + np.exp(sinks[hh] - m)
        out[:, hh] = (e / den) @ v[:, hh // g]
    x = x + out.reshape(N, -1) @ T[f"{p}.attn_output.weight"].T + T[f"{p}.attn_output.bias"]
    h = rms(x, T[f"{p}.post_attention_norm.weight"])
    logits = h @ T[f"{p}.ffn_gate_inp.weight"].T + T[f"{p}.ffn_gate_inp.bias"]
    y = np.zeros_like(x)
    for t in range(N):
        order = np.argsort(-logits[t], kind="stable")
        picks = order[:k_used]
        if k_used < len(order):
            margins.append((logits[t][order[k_used - 1]] - logits[t][order[k_used]], t, il))
        w = np.exp(logits[t][picks] - logits[t][picks].max())
        w = w / w.sum()
        for e_, we in zip(picks, w):
            gate = T[f"{p}.ffn_gate_exps.weight"][e_] @ h[t] + T[f"{p}.ffn_gate_exps.bias"][e_]
            up = T[f"{p}.ffn_up_exps.weight"][e_] @ h[t] + T[f"{p}.ffn_up_exps.bias"][e_]
            down = T[f"{p}.ffn_down_exps.weight"][e_] @ swiglu_oai(gate, up) + T[f"{p}.ffn_down_exps.bias"][e_]
            y[t] += we * down
    x = x + y
logits = rms(x, T["output_norm.weight"]) @ T["output.weight"].T


def cos(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


logits = logits[first:]
ours = np.fromfile(ours_path, dtype=np.float32).reshape(N - first, -1)
ref = np.fromfile(ref_path, dtype=np.float32).reshape(N, -1)[first:]
for name, other in [("ours", ours), ("llama.cpp", ref)]:
    cs = cos(logits, other.astype(np.float64))
    print(f"exact vs {name:9s} median {np.median(cs):.6f} worst {cs.min():.6f} at {first + cs.argmin()}  "
          f"top1 {np.sum(logits.argmax(1) == other.argmax(1))}/{N - first}  (positions {first}..{N - 1})")
    print("   ", np.round(cs, 5).tolist())
margins.sort()
print("closest picks (margin, position, layer):", [(f"{m:.2e}", t, l) for m, t, l in margins[:5]])
