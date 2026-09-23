#!/usr/bin/env python3
"""Re-reference tokenizer cases against Hugging Face `tokenizers` — the model's
OWN tokenizer — where llama.cpp cannot settle a question.

`examples/tokenizer_reference.py` writes cases whose `ref_ids` come from
llama.cpp. For SentencePiece families llama.cpp is not an authority on
whitespace: it inserts a `▁` after every special token (HF's `legacy` style,
which Mistral v0.3's tokenizer does not use), applies Phi-3's strip-after-marker
rule only when `general.name` contains "phi-3"/"phi3", and segments a
merges-carrying SentencePiece vocabulary (TinyLlama) by score. Measured
2026-09-24 against each model's tokenizer.json: llama.cpp agreed with HF in
3/7 (Mistral), 5/8 (Phi-3.5) and 3/8 (TinyLlama) cases.

This rewrites each case's `ref_ids` to HF's ids — tokenized as
`apply_chat_template` tokenizes a rendered prompt (`add_special_tokens=False`;
the template carries its own specials) and given the SAME leading BOS the
llama.cpp case had, so BOS is not what is compared — and prints how often
llama.cpp and HF agreed. Feed the output to the ignored test:

    curl -sL -o /tmp/phi35.json \\
      https://huggingface.co/microsoft/Phi-3.5-mini-instruct/resolve/main/tokenizer.json
    python3 examples/tokenizer_reference.py /tmp/cases.jsonl
    python3 examples/tokenizer_hf_reference.py /tmp/cases.jsonl /tmp/hf.jsonl \\
        phi-3.5-mini-instruct.q4-k-m=/tmp/phi35.json
    SWARM_TOKENIZER_CASES=/tmp/hf.jsonl SWARM_TOKENIZER_VERBOSE=1 \\
        cargo test --lib -- --ignored tokenizer_agrees_with_llama_cpp --nocapture

Only the models named on the command line are written out. Needs
`pip install tokenizers`.

usage: tokenizer_hf_reference.py <cases_in> <cases_out> MODEL=tokenizer.json... [-v]
"""
import json
import sys

from tokenizers import Tokenizer


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "-v"]
    verbose = "-v" in sys.argv
    if len(args) < 3:
        sys.exit(__doc__)
    src, dst, mappings = args[0], args[1], args[2:]
    toks = {}
    for m in mappings:
        model, _, path = m.partition("=")
        toks[model] = Tokenizer.from_file(path)
    agree = {}
    with open(dst, "w") as out:
        for line in open(src):
            case = json.loads(line)
            model = case["model"]
            tok = toks.get(model)
            if tok is None:
                continue
            hf = tok.encode(case["text"], add_special_tokens=False).ids
            llama = case["ref_ids"]
            # The same leading BOS the llama.cpp case has: segmentation, not BOS,
            # is the question here. BOS has its own rule and its own assertion.
            if llama and (not hf or hf[0] != llama[0]) and tok.id_to_token(llama[0]) in ("<s>", "<bos>"):
                hf = [llama[0]] + hf
            n, same = agree.get(model, (0, 0))
            agree[model] = (n + 1, same + (hf == llama))
            if verbose and hf != llama:
                print(f"llama.cpp != HF  {model}: {case['text'][:70]!r}\n  llama {llama}\n  hf    {hf}")
            case["ref_ids"] = hf
            out.write(json.dumps(case) + "\n")
    for model, (n, same) in sorted(agree.items()):
        print(f"{model}: llama.cpp == HF in {same}/{n}")


if __name__ == "__main__":
    main()
