"""Exact (float64) forward of a tiny `glm4moe` GGUF (examples/make_tiny_glm4moe_gguf.py),
as llama.cpp's glm4-moe.cpp computes it, and each MoE pick's margin: the gap between
the k-th and (k+1)-th selection score. Which engine sits on the exact answer's side
of a near-tie is the question a logits comparison cannot answer by itself.

    PYTHONPATH=~/llama.cpp-ref/gguf-py python3 -I examples/glm4moe_exact_reference.py \\
        MODEL.gguf OURS.json OURS.f32 LLAMA_CPP.f32

OURS.* are logits_reference_probe's output; LLAMA_CPP.f32 is
`~/llama.cpp-ref/dump_logits MODEL.gguf LLAMA_CPP.f32 <OURS.json's tokens, comma-joined>`.
F16_TOKEN_TABLE=1 rounds the token table to f16, as this engine holds it
(`split::token_embedding`). Only f32 files: the weights are read as stored, so a
Q8_0 file's control is its DEQUANTIZED=1 twin. Results: docs/invariants/inference.md
§ "GLM-4.5"."""
import json
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


A = "glm4moe"
n_layer = kv(f"{A}.block_count") - kv(f"{A}.nextn_predict_layers")
n_head, n_kv = kv(f"{A}.attention.head_count"), kv(f"{A}.attention.head_count_kv")
hd = kv(f"{A}.attention.key_length")
rope = kv(f"{A}.rope.dimension_count")
base = kv(f"{A}.rope.freq_base")
eps = kv(f"{A}.attention.layer_norm_rms_epsilon")
k_used = kv(f"{A}.expert_used_count")
w_scale = kv(f"{A}.expert_weights_scale")
dense_lead = kv(f"{A}.leading_dense_block_count")

ids = json.load(open(ids_json))["tokens"]
N = len(ids)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def neox(x, pos):  # x [N, heads, hd]
    half = rope // 2
    inv = base ** (-np.arange(half) * 2.0 / rope)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    a, b = x[..., :half], x[..., half:rope]
    out = x.copy()
    out[..., :half] = a * c - b * s
    out[..., half:rope] = a * s + b * c
    return out


def silu(x):
    return x / (1 + np.exp(-x))


def swiglu(x, g, u, d):
    return (silu(x @ g.T) * (x @ u.T)) @ d.T


x = T["token_embd.weight"][ids]
if os.environ.get("F16_TOKEN_TABLE") == "1":
    x = x.astype(np.float16).astype(np.float64)
pos = np.arange(N, dtype=np.float64)
mask = np.triu(np.full((N, N), -np.inf), 1)
margins = []
for il in range(n_layer):
    p = f"blk.{il}"
    h = rms(x, T[f"{p}.attn_norm.weight"])
    q = (h @ T[f"{p}.attn_q.weight"].T + T[f"{p}.attn_q.bias"]).reshape(N, n_head, hd)
    k = (h @ T[f"{p}.attn_k.weight"].T + T[f"{p}.attn_k.bias"]).reshape(N, n_kv, hd)
    v = (h @ T[f"{p}.attn_v.weight"].T + T[f"{p}.attn_v.bias"]).reshape(N, n_kv, hd)
    if f"{p}.attn_q_norm.weight" in T:
        q = rms(q, T[f"{p}.attn_q_norm.weight"])
        k = rms(k, T[f"{p}.attn_k_norm.weight"])
    q, k = neox(q, pos), neox(k, pos)
    g = n_head // n_kv
    out = np.zeros((N, n_head, hd))
    for hh in range(n_head):
        s = q[:, hh] @ k[:, hh // g].T / np.sqrt(hd) + mask
        s = np.exp(s - s.max(1, keepdims=True))
        out[:, hh] = (s / s.sum(1, keepdims=True)) @ v[:, hh // g]
    x = x + out.reshape(N, -1) @ T[f"{p}.attn_output.weight"].T
    h = rms(x, T[f"{p}.post_attention_norm.weight"])
    if il < dense_lead:
        x = x + swiglu(h, T[f"{p}.ffn_gate.weight"], T[f"{p}.ffn_up.weight"], T[f"{p}.ffn_down.weight"])
        continue
    probs = 1 / (1 + np.exp(-(h @ T[f"{p}.ffn_gate_inp.weight"].T)))
    sel = probs + T[f"{p}.exp_probs_b.bias"]
    y = np.zeros_like(x)
    for t in range(N):
        order = np.argsort(-sel[t], kind="stable")
        picks = order[:k_used]
        margins.append((sel[t][order[k_used - 1]] - sel[t][order[k_used]], t, il))
        w = probs[t][picks]
        w = w / w.sum() * w_scale
        for e, we in zip(picks, w):
            y[t] += we * swiglu(h[t], T[f"{p}.ffn_gate_exps.weight"][e], T[f"{p}.ffn_up_exps.weight"][e], T[f"{p}.ffn_down_exps.weight"][e])
    y += swiglu(h, T[f"{p}.ffn_gate_shexp.weight"], T[f"{p}.ffn_up_shexp.weight"], T[f"{p}.ffn_down_shexp.weight"])
    x = x + y
logits = rms(x, T["output_norm.weight"]) @ T["output.weight"].T

cos = lambda a, b: (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)
ours = np.fromfile(ours_path, dtype=np.float32).reshape(N, -1)
ref = np.fromfile(ref_path, dtype=np.float32).reshape(N, -1)
print("exact vs ours     ", np.round(cos(logits, ours), 5).tolist())
print("exact vs llama.cpp", np.round(cos(logits, ref), 5).tolist())
margins.sort()
print("closest picks (margin, position, layer):", [(f"{m:.2e}", t, l) for m, t, l in margins[:5]])
