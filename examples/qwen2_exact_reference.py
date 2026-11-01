"""Exact (float64) forward of a Llama-shaped GGUF (`qwen2`, `qwen2vl` read for text, `llama`,
`granite` with its attention scale and the other multipliers at 1), its
weights dequantized one layer at a time, and which engine sits closer to the exact answer at each
position. A logits comparison says two engines differ; this says which one is off.

    PYTHONPATH=~/llama.cpp-ref/gguf-py python3 -P examples/qwen2_exact_reference.py \\
        MODEL.gguf OURS.json OURS.f32 LLAMA_CPP.f32

OURS.* are logits_reference_probe's output (`LOGITS_PROBE_IDS` for a real prompt); LLAMA_CPP.f32 is
`~/llama.cpp-ref/dump_logits MODEL.gguf LLAMA_CPP.f32 <OURS.json's tokens, comma-joined>`.
F16_TOKEN_TABLE=1 rounds the token table to f16, as this engine holds it (`split::token_embedding`).
EXACT_HIDDEN=1 also writes every layer's residual state to OURS.f32.exact_hidden.npy
([layers, positions, width]), for the probe's `LOGITS_PROBE_SPLIT=L LOGITS_PROBE_HIDDEN=1`.

Where a gap enters, and whether it is a defect at all:
- SAVE_EXACT=E.npy keeps the exact logits; EXACT_LOGITS=E.npy then compares a variant run against them
  instead of against the engines.
- START_LAYER=L START_HIDDEN=OURS_L.hidden.f32 finishes the forward exactly from OUR state entering
  layer L (the probe's `LOGITS_PROBE_SPLIT=L LOGITS_PROBE_HIDDEN=1 LOGITS_PROBE_DECODE=0`): where the
  finished logits stop recovering the exact ones, our layers before L let the error in.
- Q8_ACT=even|away rounds every matmul's input to Q8_0 blocks of 32, as both engines' processor
  matmuls do (ties to even, or away from zero): if that simulation lands as far from exact as an
  engine does, the gap is 8-bit activation noise, not a defect (Granite 4.2-3B, 2026-11-02).
Memory: one layer's weights in float64 plus the token table in float32 (a 3B: ~3 GB).
Results: docs/invariants/inference.md § "Qwen2.5-VL"."""
import json
import os
import sys

import numpy as np
import gguf
from gguf.quants import dequantize

path, ids_json, ours_path, ref_path = sys.argv[1:5]
r = gguf.GGUFReader(path)
TI = {t.name: t for t in r.tensors}


def W(name, dtype=np.float64):
    t = TI[name]
    return dequantize(t.data, t.tensor_type).reshape([int(d) for d in reversed(t.shape)]).astype(dtype)


def kv(key):
    f = r.fields.get(key)
    return f.parts[f.data[0]].tolist()[0] if f is not None and len(f.data) == 1 else None


A = bytes(r.fields["general.architecture"].parts[-1]).decode()
n_layer = kv(f"{A}.block_count")
n_head, n_kv = kv(f"{A}.attention.head_count"), kv(f"{A}.attention.head_count_kv")
emb = kv(f"{A}.embedding_length")
hd = emb // n_head
base = kv(f"{A}.rope.freq_base")
eps = kv(f"{A}.attention.layer_norm_rms_epsilon")
NORM = A in ("llama", "granite")
scale = 1 / np.sqrt(hd)
if A == "granite":  # granite.cpp: kq_scale = attention.scale (else 1/sqrt(d)); other multipliers must be 1
    scale = kv("granite.attention.scale") or scale
    assert all(kv(f"granite.{k}") in (None, 0.0, 1.0) for k in ("embedding_scale", "residual_scale")), "multipliers"
    assert kv("granite.logit_scale") == 1.0, "logit scale"
print(f"{A}: {n_layer} layers, {n_head}/{n_kv} heads of {hd}, rope base {base}, norm eps {eps}, attention scale {scale}, {'NORM' if NORM else 'NEOX'} rotation")

ids = json.load(open(ids_json))["tokens"]
N = len(ids)
pos = np.arange(N, dtype=np.float64)


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x):  # [N, heads, hd]; MROPE with a text position in every section is NEOX
    half = hd // 2
    ang = pos[:, None] * (base ** (-np.arange(half) * 2.0 / hd))[None, :]
    c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    if NORM:  # adjacent pairs (llama.cpp's NORM: Llama, Granite)
        a, b = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x)
        out[..., 0::2], out[..., 1::2] = a * c - b * s, a * s + b * c
        return out
    a, b = x[..., :half], x[..., half:]
    return np.concatenate([a * c - b * s, a * s + b * c], axis=-1)


Q8_ACT = os.environ.get("Q8_ACT")


def act(x):  # a matmul's input as an engine hands it over (Q8_ACT), else exact
    if not Q8_ACT:
        return x
    blocks = x.reshape(-1, 32)
    amax = np.abs(blocks).max(-1, keepdims=True)
    inv = np.where(amax > 0, 127.0 / np.where(amax > 0, amax, 1), 0.0)
    v = (blocks * inv).astype(np.float32).astype(np.float64)
    q = np.round(v) if Q8_ACT == "even" else np.sign(v) * np.floor(np.abs(v) + 0.5)
    return (q * (amax / 127.0).astype(np.float16).astype(np.float64)).reshape(x.shape)


def proj(x, name):  # weight, plus its bias where the file has one
    y = act(x) @ W(name + ".weight").T
    return y + W(name + ".bias") if name + ".bias" in TI else y


tok = W("token_embd.weight", np.float32)
table = np.float16 if os.environ.get("F16_TOKEN_TABLE") == "1" else np.float64
h = tok[ids].astype(table).astype(np.float64)
mask = np.triu(np.full((N, N), -np.inf), 1)
hidden = []
START = int(os.environ.get("START_LAYER", "0"))
if START:
    h = np.fromfile(os.environ["START_HIDDEN"], dtype="<f4").reshape(N, emb).astype(np.float64)
for il in range(START, n_layer):
    p = f"blk.{il}."
    x = rms(h, W(p + "attn_norm.weight"))
    q = rope(proj(x, p + "attn_q").reshape(N, n_head, hd))
    k = rope(proj(x, p + "attn_k").reshape(N, n_kv, hd))
    v = proj(x, p + "attn_v").reshape(N, n_kv, hd)
    group = n_head // n_kv
    out = np.empty((N, n_head, hd))
    for hh in range(n_head):
        s = q[:, hh] @ k[:, hh // group].T * scale + mask
        s = np.exp(s - s.max(-1, keepdims=True))
        out[:, hh] = (s / s.sum(-1, keepdims=True)) @ v[:, hh // group]
    h = h + proj(out.reshape(N, emb), p + "attn_output")
    x = rms(h, W(p + "ffn_norm.weight"))
    gate, up = act(x) @ W(p + "ffn_gate.weight").T, act(x) @ W(p + "ffn_up.weight").T
    h = h + act(gate / (1 + np.exp(-gate)) * up) @ W(p + "ffn_down.weight").T
    hidden.append(h.astype(np.float32))
    spike = np.unravel_index(np.abs(h).argmax(), h.shape)
    print(f"layer {il:2d}: largest |state| {np.abs(h).max():8.1f} at position {spike[0]}, dimension {spike[1]}")
if os.environ.get("EXACT_HIDDEN") == "1":
    np.save(ours_path + ".exact_hidden.npy", np.stack(hidden))
h = rms(h, W("output_norm.weight"))
head = W("output.weight", np.float32) if "output.weight" in TI else tok
h = act(h)
exact = np.concatenate([h @ head[i:i + 16384].astype(np.float64).T for i in range(0, head.shape[0], 16384)], axis=1)

ours = np.fromfile(ours_path, dtype="<f4").reshape(N, -1)
ref = np.fromfile(ref_path, dtype="<f4").reshape(N, -1)


def cos(a, b):
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


if os.environ.get("SAVE_EXACT"):
    np.save(os.environ["SAVE_EXACT"], exact)
if os.environ.get("EXACT_LOGITS"):
    full = np.load(os.environ["EXACT_LOGITS"])
    for name, got in (("this run", exact), ("ours", ours), ("llama.cpp", ref)):
        cs = [cos(got[i], full[i]) for i in range(N)]
        top = sum(int(got[i].argmax()) == int(full[i].argmax()) for i in range(N))
        print(f"{name:>9} ~ exact: worst {min(cs):.6f} at {int(np.argmin(cs))}  median {np.median(cs):.6f}  top-1 {top}/{N}")
    sys.exit(0)

co, cr = [cos(ours[i], exact[i]) for i in range(N)], [cos(ref[i], exact[i]) for i in range(N)]
for i in range(N):
    e, o, f = int(exact[i].argmax()), int(ours[i].argmax()), int(ref[i].argmax())
    flag = "  <--" if min(co[i], cr[i]) < 0.999 or o != e or f != e else ""
    print(f"pos {i:3d}  ours~exact {co[i]:.6f}  llama.cpp~exact {cr[i]:.6f}  top-1 exact {e} ours {o} llama.cpp {f}{flag}")
top_o = sum(int(ours[i].argmax()) == int(exact[i].argmax()) for i in range(N))
top_r = sum(int(ref[i].argmax()) == int(exact[i].argmax()) for i in range(N))
print(f"ours:      worst {min(co):.6f}  median {np.median(co):.6f}  exact's top-1 {top_o}/{N}")
print(f"llama.cpp: worst {min(cr):.6f}  median {np.median(cr):.6f}  exact's top-1 {top_r}/{N}")
