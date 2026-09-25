#!/usr/bin/env python3
"""Turn a PEFT LoRA adapter into a llama.cpp GGUF LoRA — the REFERENCE for ours.

A node applies a PEFT `.safetensors` adapter itself (`src/model/lora.rs`). To
check it against an independent implementation, llama.cpp has to apply the same
adapter, and llama.cpp takes adapters as GGUF. Its own converter
(`convert_lora_to_gguf.py`) needs torch and the base model's HF config, neither
of which is here, so this does the part of it that matters, in numpy:

  * names: `...layers.N.self_attn.q_proj.lora_A.weight` → `blk.N.attn_q.weight.lora_a`
  * shapes: A stays `[rank, in]`, B stays `[out, rank]` (llama.cpp checks
    `a.ne[0] == in`, `b.ne[1] == out`)
  * **Llama/Mistral q and k: B's rows reordered by `LlamaModel.permute`**,
    copied verbatim from `convert_hf_to_gguf.py` — the converter runs the base
    model's `modify_tensors` over the adapter, and its `LoraTorchTensor` routes
    that reshape-and-swap onto B. This is the step a node gets wrong silently.
  * metadata: `general.type = adapter`, `adapter.type = lora`,
    `adapter.lora.alpha`.

usage: peft_lora_to_gguf.py <adapter.safetensors> <out.gguf> --arch llama|qwen2|... \
           --alpha A [--heads N --kv-heads M]   (heads required for llama/mistral)
       --no-permute   write q/k in checkpoint order (a deliberately WRONG reference,
                      to show a comparison can tell the two apart)
Then: `score_against_reference.py --lora out.gguf ...`.
"""
import argparse
import json
import struct

import numpy as np
import gguf

PROJ = {"q_proj": "attn_q", "k_proj": "attn_k", "v_proj": "attn_v", "o_proj": "attn_output",
        "gate_proj": "ffn_gate", "up_proj": "ffn_up", "down_proj": "ffn_down"}


def llama_permute(weights, n_head, n_head_kv):
    # convert_hf_to_gguf.py, LlamaModel.permute — verbatim but for np.
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


def read_safetensors(path):
    raw = open(path, "rb").read()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    header.pop("__metadata__", None)
    body = raw[8 + n:]
    out = {}
    for name, v in header.items():
        s, e = v["data_offsets"]
        b = body[s:e]
        if v["dtype"] == "BF16":
            a = (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        elif v["dtype"] == "F16":
            a = np.frombuffer(b, dtype=np.float16).astype(np.float32)
        elif v["dtype"] == "F32":
            a = np.frombuffer(b, dtype=np.float32)
        else:
            raise SystemExit(f"{name}: dtype {v['dtype']} not handled")
        out[name] = a.reshape(v["shape"]).copy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("adapter")
    ap.add_argument("out")
    ap.add_argument("--arch", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--heads", type=int)
    ap.add_argument("--kv-heads", type=int)
    ap.add_argument("--no-permute", action="store_true")
    a = ap.parse_args()
    permute = a.arch in ("llama", "mistral") and not a.no_permute
    if permute and not (a.heads and a.kv_heads):
        raise SystemExit("--heads and --kv-heads are required for llama/mistral")

    w = gguf.GGUFWriter(a.out, arch=a.arch)
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, a.alpha)
    n = 0
    for name, t in sorted(read_safetensors(a.adapter).items()):
        parts = name.split(".")
        layer = int(parts[parts.index("layers") + 1])
        proj = next(PROJ[p] for p in parts if p in PROJ)
        ab = "lora_a" if "lora_A" in name else "lora_b"
        if ab == "lora_b" and permute and proj == "attn_q":
            t = llama_permute(t, a.heads, a.heads)
        if ab == "lora_b" and permute and proj == "attn_k":
            t = llama_permute(t, a.heads, a.kv_heads)
        w.add_tensor(f"blk.{layer}.{proj}.weight.{ab}", np.ascontiguousarray(t, dtype=np.float32))
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"{a.out}: {n} tensors, arch {a.arch}, alpha {a.alpha}, q/k permuted: {permute}")


if __name__ == "__main__":
    main()
