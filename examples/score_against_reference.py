#!/usr/bin/env python3
"""Is a greedy reply the MODEL's, or only plausible text? Ask llama.cpp.

Teacher-forces each reply through llama.cpp (llama-cpp-python, the independent
reference this repo judges replies against) and reports where every emitted
token ranks among the reference model's next-token choices, given the reply's
own prefix.

WHY: byte-equality between two runs is the wrong test for a split reply. Two
greedy runs over the SAME topology split at the same near-tie on 2026-09-25
(`split_rig.sh failover`: one with cold stand-ins and the prompt relayed through
the coordinator, one warm and chained), so "!=" said nothing about whether the
failover had worked. What a broken cache state produces is tokens the reference
ranks nowhere near the top — `examples/failover_kv_probe.rs` measured the right
token's probability at 0.005 for a stand-in missing half the model's context —
while a correct one is rank 1 almost everywhere, with the odd rank-2 pick by a
small margin. Compare replies by their SCORES: a takeover should score like a
same-topology control.

usage: score_against_reference.py [--lora adapter.gguf] [--system TEXT] <model.gguf> <replies.jsonl> <prompt-file> [label ...]
  --lora         score against the model WITH this adapter applied by llama.cpp
                 (`peft_lora_to_gguf.py` makes one from a PEFT adapter).
  --system       render a system turn first. The node supplies
                 `chat_template::DEFAULT_SYSTEM_PROMPT` ("You are a helpful
                 assistant.") to a template that expects a system turn and writes
                 none of its own — TinyLlama, Phi-3.5 — so scoring their replies
                 without it scores them against a prompt they were never given.
                 Check: "prompt tokens" must equal the node's `usage.prompt_tokens`.

A reply is scored as llama.cpp re-tokenizes its TEXT. For a SentencePiece model
(TinyLlama, Mistral, Phi-3.5) llama.cpp's tokenizer is no authority on whitespace
and word splits: on TinyLlama it splits "Yellow" into " Ye"+"ll"+"ow" and puts a
leading space marker on the first word, so those positions rank in the hundreds
for ANY reply, correct or not (measured 2026-09-25). Judge a SentencePiece
family by rank-2 near-ties only, or use a BPE model (Llama-3.x, Qwen) where the
re-tokenization is exact.
  replies.jsonl  one JSON object per line with a "content" field (what
                 `split_rig.sh` writes); labels default to the line number.
  The prompt is rendered with the GGUF's own chat template, as the node renders
  it (jinja2 standing in for minijinja; `strftime_now` is today's date).

Needs llama-cpp-python and jinja2, and the model as a whole GGUF (the node keeps
shards; `~/swarmllm-ref/rebuild_gguf.py` rebuilds one). CPU, a few seconds per
reply for a 3B model.
"""
import datetime
import json
import sys

import numpy as np
from jinja2 import Environment
from llama_cpp import Llama


def main():
    args = sys.argv[1:]
    lora = None
    if "--lora" in args:
        i = args.index("--lora")
        lora = args[i + 1]
        del args[i:i + 2]
    system = None
    if "--system" in args:
        i = args.index("--system")
        system = args[i + 1]
        del args[i:i + 2]
    if len(args) < 3:
        sys.exit(__doc__)
    gguf, replies_path, prompt_path = args[:3]
    prompt = open(prompt_path).read()
    rows = [json.loads(line) for line in open(replies_path) if line.strip()]
    labels = args[3:] or [f"reply {i + 1}" for i in range(len(rows))]

    llm = Llama(model_path=gguf, n_ctx=4096, n_gpu_layers=0, n_threads=8,
                logits_all=True, verbose=False, seed=0, lora_path=lora)
    env = Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["strftime_now"] = lambda f: datetime.datetime.now().strftime(f)

    def raise_exception(message):
        raise Exception(message)

    env.globals["raise_exception"] = raise_exception
    rendered = env.from_string(llm.metadata["tokenizer.chat_template"]).render(
        messages=([{"role": "system", "content": system}] if system else [])
        + [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        bos_token=llm.detokenize([llm.token_bos()], special=True).decode(),
        eos_token=llm.detokenize([llm.token_eos()], special=True).decode())
    ptoks = llm.tokenize(rendered.encode(), add_bos=False, special=True)
    print(f"prompt tokens: {len(ptoks)}")

    for label, row in zip(labels, rows):
        reply = row.get("content") or ""
        rtoks = llm.tokenize(reply.encode(), add_bos=False, special=False)
        if not rtoks:
            print(f"--- {label}: empty reply")
            continue
        llm.reset()
        llm.eval(ptoks + rtoks)
        logits = np.array(llm.scores[len(ptoks) - 1: len(ptoks) - 1 + len(rtoks)])
        ranks, gaps = [], []
        for i, tok in enumerate(rtoks):
            step = logits[i]
            ranks.append(int((step > step[tok]).sum()) + 1)
            gaps.append(float(step.max() - step[tok]))
        off = [(i, r, g) for i, (r, g) in enumerate(zip(ranks, gaps)) if r > 1]
        print(f"--- {label}: {len(rtoks)} tokens, rank-1 {ranks.count(1)}, "
              f"not rank-1 {len(off)}, worst rank {max(ranks)}, "
              f"largest logit gap {max(gaps):.3f}")
        for i, r, g in off[:10]:
            piece = llm.detokenize([rtoks[i]]).decode(errors="replace")
            print(f"    token {i:3d} {piece!r:>16} rank {r} gap {g:.3f}")


if __name__ == "__main__":
    main()
