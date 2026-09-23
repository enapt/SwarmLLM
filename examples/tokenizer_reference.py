#!/usr/bin/env python3
"""Tokenize every local model's chat prompts with llama.cpp — the independent
reference for SwarmLLM's own tokenizer.

    python3 examples/tokenizer_reference.py /tmp/cases.jsonl [models_dir]
    SWARM_TOKENIZER_CASES=/tmp/cases.jsonl \\
        cargo test --lib -- --ignored tokenizer_agrees_with_llama_cpp --nocapture

Why it exists (FUTURE_WORK #97): every tokenizer test compared the encoder to
itself, and conformance judges replies, not prompts — so GLM-4's `[gMASK]` and
Mistral v0.3's `[INST]` were spelled out as ordinary text in every prompt, and a
doubled BOS reached every Llama-3 / Gemma / Mistral prompt, with nothing going
red.

Needs `llama-cpp-python` and `jinja2`. No weights are read or copied: for each
model a SPARSE GGUF (its `gguf_header.bin` plus a hole out to the real file
size, sized from `manifest.json`) is written to a temp dir, because llama.cpp
checks tensor bounds even with `vocab_only=True`. The hole costs no disk.

For each model: its OWN chat template (from the GGUF) rendered with jinja2 over
three conversations, then tokenized the way llama.cpp's chat path does — ONE
BOS: `common/chat.cpp` strips the template's leading BOS when the tokenizer will
add one — plus raw strings that stress the pre-tokenizer."""
import glob, json, os, shutil, sys, tempfile

import jinja2
from llama_cpp import Llama

OUT = sys.argv[1] if len(sys.argv) > 1 else "cases.jsonl"
MODELS = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.local/share/swarmllm/models")

CONVERSATIONS = [
    [{"role": "user", "content": "Write a short Python function that returns the factorial of n."}],
    [{"role": "system", "content": "You are a helpful assistant."},
     {"role": "user", "content": "What is the capital of France? Answer in one sentence."}],
    [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello! How can I help?"},
     {"role": "user", "content": "List three colours:\n\n1. red\n2.   green\n\n\n3. blue"}],
]
RAW = [
    "fn main() { let xs: Vec<u32> = (0..10).map(|i| i * 2).collect(); }",
    "Отправь это сообщение — 你好，世界! \U0001F600 café naïve",
    "   leading and    irregular\twhitespace\n\nand blank lines   ",
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)\n",
    "<div class=\"x\">a<br>b</div> [INST] [/INST] [gMASK] <|im_start|>",
]


def sparse_gguf(model_dir, dst):
    m = json.load(open(os.path.join(model_dir, "manifest.json")))
    end = max((t["gguf_offset"] + t["size"] for s in m["shards"] for t in s.get("tensors", [])),
              default=0)
    shutil.copyfile(os.path.join(model_dir, "gguf_header.bin"), dst)
    if end > os.path.getsize(dst):
        os.truncate(dst, end)


def render(template, messages, bos, eos):
    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True, loader=jinja2.BaseLoader())

    def raise_exception(msg):
        raise jinja2.exceptions.TemplateError(msg)

    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: "26 Jul 2024"
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True, bos_token=bos, eos_token=eos, tools=None)


tmp = tempfile.mkdtemp(prefix="swarm-vocab-")
try:
    with open(OUT, "w") as out:
        for model_dir in sorted(glob.glob(os.path.join(MODELS, "*/"))):
            name = os.path.basename(model_dir.rstrip("/"))
            header = os.path.join(model_dir, "gguf_header.bin")
            if not (os.path.exists(header) and os.path.exists(os.path.join(model_dir, "manifest.json"))):
                continue
            vocab = os.path.join(tmp, name + ".gguf")
            sparse_gguf(model_dir, vocab)
            llm = Llama(model_path=vocab, vocab_only=True, verbose=False)
            md = llm.metadata
            bos_id = int(md["tokenizer.ggml.bos_token_id"]) if "tokenizer.ggml.bos_token_id" in md else None
            piece = lambda i: llm.detokenize([i], special=True).decode(errors="replace")
            bos = piece(bos_id) if bos_id is not None else ""
            eos = piece(int(md["tokenizer.ggml.eos_token_id"])) if "tokenizer.ggml.eos_token_id" in md else ""
            texts = []
            template = md.get("tokenizer.chat_template")
            if template:
                for conv in CONVERSATIONS:
                    try:
                        texts.append(("chat", render(template, conv, bos, eos)))
                    except Exception as e:  # e.g. Mistral's strict role alternation
                        print(f"{name}: template refused a conversation: {e}", file=sys.stderr)
            texts += [("raw", t) for t in RAW]
            for kind, text in texts:
                ids = llm.tokenize(text.encode(), add_bos=True, special=True)
                # llama.cpp's chat path: the template's own BOS is the one BOS.
                if kind == "chat" and bos_id is not None and ids[:2] == [bos_id, bos_id]:
                    ids = ids[1:]
                out.write(json.dumps({"model": name, "header": header, "kind": kind,
                                      "text": text, "ref_ids": ids}) + "\n")
            print(name, len(texts), "cases")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
