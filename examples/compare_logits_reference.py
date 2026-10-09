#!/usr/bin/env python3
"""Compare examples/logits_reference_probe.rs's logits with llama.cpp's.

    python3 examples/compare_logits_reference.py <model.gguf> <probe OUT prefix>

llama-cpp-python (0.3.16 here) evaluates the same token ids on the processor
with logits kept at every position, and each position is compared on the
VALUES, not only the argmax: a random-weight test model has near-flat
distributions, where a correct and a broken implementation can both miss the
argmax by a hair. What a wrong architecture looks like is a cosine far from 1
and a max |diff| of the order of the logits themselves; a right one agrees to
quantization noise.

Exit 0 when every position clears COS_MIN (default 0.999), or all but
MAX_OUTLIERS of them (default 0). A mixture-of-experts router can meet a
near-tie between two experts at a token, where a last-bit difference picks the
other one for a layer: that position then sits apart while every other one
agrees to quantization noise (qwen3moe's tiny test model does it at ONE of 100
positions, 2026-09-25). A systematic error moves every position; an isolated
one is a tie. The median is printed so the two can be told apart.
"""
import json
import os
import sys

import numpy as np
from llama_cpp import Llama

gguf, prefix = sys.argv[1], sys.argv[2]
meta = json.load(open(prefix + ".json"))
ours = np.fromfile(prefix + ".f32", dtype="<f4").reshape(meta["positions"], meta["vocab"])
tokens = meta["tokens"]
# The probe may keep only a long prompt's last positions (`LOGITS_PROBE_KEEP_LAST`):
# its rows are tokens[first:], and the positions are numbered as in the sequence.
first = meta.get("first_position", 0)
# N_CTX: llama.cpp's context size. A LongRoPE model (Phi-3.5, Phi-4-mini) picks its
# rotary factors by it — give it the node's served window, as score_against_reference.py's
# --n-ctx explains.
n_ctx = int(os.environ.get("N_CTX", max(64, len(tokens) + 8)))

llm = Llama(model_path=gguf, n_ctx=n_ctx, logits_all=True,
            n_gpu_layers=0, n_threads=4, verbose=False)
llm.eval(tokens)
ref = np.array(llm.scores[first: len(tokens)], dtype=np.float32)
if ref.shape[1] != ours.shape[1]:
    sys.exit(f"vocab differs: llama.cpp {ref.shape[1]} vs ours {ours.shape[1]}")
if ref.shape[0] != ours.shape[0]:
    sys.exit(f"positions differ: llama.cpp {ref.shape[0]} from {first} vs ours {ours.shape[0]}")

cos_min = float(os.environ.get("COS_MIN", "0.999"))
cosines, top1 = [], 0
for i in range(len(ours)):
    a, b = ours[i], ref[i]
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    cosines.append(cos)
    same = int(a.argmax()) == int(b.argmax())
    top1 += same
    top5 = len(set(np.argsort(-a)[:5]) & set(np.argsort(-b)[:5]))
    p = first + i
    phase = "prefill" if p < meta["prefill"] else "decode"
    print(f"pos {p:3d} {phase:7s} cos={cos:.6f} max|d|={np.abs(a - b).max():.4f} "
          f"|ref|max={np.abs(b).max():.3f} top1={'=' if same else 'x'} top5={top5}/5")
outliers = [(first + i, round(c, 6)) for i, c in enumerate(cosines) if c < cos_min]
max_outliers = int(os.environ.get("MAX_OUTLIERS", "0"))
print(f"worst cosine {min(cosines):.6f}, median {float(np.median(cosines)):.6f}, "
      f"top-1 agreement {top1}/{len(ours)}, split={meta['split']}, "
      f"positions below {cos_min}: {outliers}")
ok = len(outliers) <= max_outliers
print("AGREES with llama.cpp" if ok else "DISAGREES with llama.cpp")
sys.exit(0 if ok else 1)
