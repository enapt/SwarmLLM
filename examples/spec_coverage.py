#!/usr/bin/env python3
"""How many tokens per network round trip could speculation buy on a WAN split?

For a target model and a small drafter sharing its tokenizer, on real target replies:
  1. the target writes a greedy reply (llama.cpp, CPU) — the TRUE path;
  2. both models are teacher-forced over prompt+reply;
  3. per reply position we record the rank of the true token among the drafter's
     guesses, and (for temperature sampling) the target probability mass the
     drafter's top-K covers.
Then we replay the reply as a sequence of network rounds under several schemes:
  - chain(g):  linear speculation of g drafts (what DSD does): a round yields the run
               of consecutive rank-0 drafts, capped at g, plus one token;
  - walk(K):   the tail-walk tree — the head precomputes, per level, the drafter's top-K
               children and keeps extending depth; the tail walks while the true token
               is among the children, and a round yields that run plus the fall-off
               token (the tail sampled it; it just has no hidden state for it);
  - tree(w):   the same with a finite budget: breadth w[d] at depth d;
  - ngram:     draft-free prompt lookup (the existing n-gram path's idea).
usage: spec_coverage.py qwen|llama [n_gen]   (models under ~/swarmllm-ref/, llama-cpp-python 0.3.16)
"""
import json, sys, time
import numpy as np
from llama_cpp import Llama

REF = "/home/user/swarmllm-ref/"
PAIRS = {
    "qwen": (REF + "qwen2.5-coder-7b-instruct-q4-k-m.gguf", REF + "spec/qwen2.5-coder-0.5b-instruct-q8_0.gguf",
             lambda u: f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
                       f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n"),
    "llama": (REF + "llama-3.2-3b-instruct-q4-k-m.gguf", REF + "spec/llama-3.2-1b-instruct-q8_0.gguf",
              lambda u: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\n"
                        f"Today Date: 27 Sep 2026\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n"),
}
PROMPTS = [
    "Explain how a city's water supply works, from the reservoir to the tap. Use several paragraphs.",
    "Write a Python function that parses a CSV file of transactions (date, description, amount) and returns the total spent per month, with a short docstring and error handling.",
    "What are the main differences between TCP and UDP? Give concrete examples of when to use each.",
    "My laptop gets very hot when I play games. What can I do about it? Give practical steps.",
    "Here is a function:\n\ndef area(r):\n    pi = 3.14159\n    result = pi * r * r\n    return result\n\ndef perimeter(r):\n    pi = 3.14159\n    result = 2 * pi * r\n    return result\n\nRewrite both functions to use math.pi instead of the local constant, keeping everything else the same.",
    "Summarize the causes and consequences of the 2008 financial crisis for a high-school student.",
]

which = sys.argv[1]
n_gen = int(sys.argv[2]) if len(sys.argv) > 2 else 160
tpath, dpath, fmt = PAIRS[which]
kw = dict(n_ctx=1536, n_threads=8, logits_all=True, verbose=False)
tgt, dft = Llama(model_path=tpath, **kw), Llama(model_path=dpath, **kw)
eos = {tgt.token_eos()}
for marker in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"):
    ids = tgt.tokenize(marker.encode(), add_bos=False, special=True)
    if len(ids) == 1:
        eos.add(ids[0])
V = min(tgt.n_vocab(), dft.n_vocab())

def softmax(x):
    x = x - x.max(); e = np.exp(x); return e / e.sum()

dump = {"dq_top32": [], "dq_true": [], "rank": [], "tcover4": [], "reply": []}
records = []  # per reply: list of (rank_in_draft, {K: target mass covered at T=0.7}, ngram_hit)
t0 = time.time()
for pi, prompt in enumerate(PROMPTS):
    ptoks = tgt.tokenize(fmt(prompt).encode(), add_bos=False, special=True)
    out = []
    for tok in tgt.generate(ptoks, temp=0.0, top_k=1, reset=True):
        if tok in eos or len(out) >= n_gen:
            break
        out.append(tok)
    seq = ptoks + out
    tgt.reset(); tgt.eval(seq); T = np.array(tgt.scores[: len(seq), :V], dtype=np.float32)
    dft.reset(); dft.eval(seq); D = np.array(dft.scores[: len(seq), :V], dtype=np.float32)
    rows = []
    for i in range(len(ptoks) - 1, len(seq) - 1):
        true = seq[i + 1]
        d = D[i]
        rank = int((d > d[true]).sum())
        order = np.argsort(-d)[:8]
        pt = softmax(T[i] / 0.7)
        cover = {k: float(pt[order[:k]].sum()) for k in (1, 2, 4, 8)}
        # prompt lookup: last 2 tokens seen earlier in the context, and what followed then == true?
        ctx = seq[: i + 1]
        key = tuple(ctx[-2:])
        ng = None
        for j in range(len(ctx) - 3, -1, -1):
            if tuple(ctx[j:j + 2]) == key:
                ng = ctx[j + 2] if j + 2 < len(ctx) else None
                break
        rows.append((rank, cover, ng == true))
        qd = softmax(d)
        top = np.sort(qd)[::-1][:32]
        dump["dq_top32"].append(top); dump["dq_true"].append(qd[true]); dump["rank"].append(rank)
        dump["tcover4"].append(cover[4]); dump["reply"].append(pi)
    records.append(rows)
    print(f"  prompt {pi}: {len(out)} tokens, draft top-1 {np.mean([r[0] == 0 for r in rows]):.2f} "
          f"top-4 {np.mean([r[0] < 4 for r in rows]):.2f} ({time.time() - t0:.0f}s)", flush=True)

def rounds(hit, cap=10**9):
    """Replay: a round yields the run of consecutive hits (capped) plus one token."""
    toks = rnds = 0
    for rows_hits in hit:
        pos, n = 0, len(rows_hits)
        while pos < n:
            run = 0
            while pos + run < n and run < cap and rows_hits[pos + run]:
                run += 1
            got = min(run + 1, n - pos)
            toks += got; rnds += 1; pos += got
    return toks / rnds

def tree_rounds(schedule):
    toks = rnds = 0
    for rows in records:
        pos, n = 0, len(rows)
        while pos < n:
            run = 0
            while pos + run < n and run < len(schedule) and rows[pos + run][0] < schedule[run]:
                run += 1
            got = min(run + 1, n - pos)
            toks += got; rnds += 1; pos += got
    return toks / rnds

def sampled_walk(K, trials=400, rng=np.random.default_rng(0)):
    """Temperature 0.7: each level continues with probability = target mass the top-K covers."""
    toks = rnds = 0
    for rows in records:
        h = np.array([r[1][K] for r in rows])
        for _ in range(trials // len(records)):
            pos, n = 0, len(h)
            while pos < n:
                run = 0
                while pos + run < n and rng.random() < h[pos + run]:
                    run += 1
                got = min(run + 1, n - pos); toks += got; rnds += 1; pos += got
    return toks / rnds

allr = [r for rows in records for r in rows]
res = {"pair": which, "positions": len(allr),
       "coverage_greedy": {k: float(np.mean([r[0] < k for r in allr])) for k in (1, 2, 4, 8)},
       "coverage_T0.7": {k: float(np.mean([r[1][k] for r in allr])) for k in (1, 2, 4, 8)},
       "ngram_hit": float(np.mean([r[2] for r in allr]))}
res["tokens_per_round"] = {
    "plain (today)": 1.0,
    "ngram lookup, chain 10": rounds([[r[2] for r in rows] for rows in records], cap=10),
    "chain g=4 (DSD)": rounds([[r[0] == 0 for r in rows] for rows in records], cap=4),
    "chain g=8": rounds([[r[0] == 0 for r in rows] for rows in records], cap=8),
    "walk K=2 (unbounded depth)": rounds([[r[0] < 2 for r in rows] for rows in records]),
    "walk K=4 (unbounded depth)": rounds([[r[0] < 4 for r in rows] for rows in records]),
    "walk K=8 (unbounded depth)": rounds([[r[0] < 8 for r in rows] for rows in records]),
    "tree 4,2,2,1,1,1,1,1 (108 nodes)": tree_rounds([4, 2, 2, 1, 1, 1, 1, 1]),
    "tree 4,3,2,2,1,1,1,1 (280 nodes)": tree_rounds([4, 3, 2, 2, 1, 1, 1, 1]),
    "walk K=4, sampled T=0.7": sampled_walk(4),
    "walk K=8, sampled T=0.7": sampled_walk(8),
}
print(json.dumps(res, indent=1))
json.dump(res, open(REF + f"spec/coverage_{which}.json", "w"), indent=1)
np.savez(REF + f"spec/dump_{which}.npz", **{k: np.array(v) for k, v in dump.items()})
