#!/usr/bin/env python3
"""How often a drafter's guesses hold the target's choice, read OFFLINE, and the flow priced from it.

The flow (`pipeline::dsd_flow`) on a long link is held by the drafter's guesses, not by compute
(`docs/invariants/scheduling.md` § "The flow's speed is the drafter's ranks"). A greedy reply's
far-end draws are the target's first choice at each position of that reply, so teacher forcing one
reply through the target and through each drafter gives every draw's rank in the drafter's list —
in minutes, on the processor, with no link. `flow_rows_sim.run(trace=...)` replays the flow on those
ranks; on the refrigerator prompt it reproduced the real Thailand-Australia link within 0-8% for
four drafters (2026-10-24).

  1. ids <dir> <name> <prompt file>: the target's greedy reply from the node on localhost:8800
     (`SWARM_RANKS_MODEL`, default qwen2.5-coder-7b-instruct-q4-k-m; `SWARM_RANKS_MAX_TOKENS`,
     default 400), chat-templated by the template of a GGUF of the target's family
     (`SWARM_RANKS_VOCAB`, read vocabulary-only), tokenized by it -> <dir>/<name>.ids.json + .meta.json
  2. per model, examples/logits_reference_probe.rs on those ids: LOGITS_PROBE_MODEL=<model dir>
     LOGITS_PROBE_IDS=<dir>/<name>.ids.json LOGITS_PROBE_DECODE=0 LOGITS_PROBE_CHUNK=512
     LOGITS_PROBE_KEEP_LAST=<reply tokens + 1> LOGITS_PROBE_OUT=<dir>/<name>_<tag> (no dot in the
     tag: the probe replaces the extension), then
     reduce <dir> <name> <tag> [truth tag] — the target first, with no truth tag; deletes the .f32.
     Truth tag `ids`: the reply's own tokens are the target's choices (a greedy reply is its first
     choice at every position) — for a target no one computer may hold whole, such as a 70B,
     whose logits cannot be read here. Probe the drafter with KEEP_LAST = reply tokens + 1.
  3. report <dir> <tag,...> <name,...>: P[rank], calibration bins, the log-loss by temperature
  4. sim <dir> <tag> <call ms> <name,...> [key=value ...]: tok/s, misses and tokens a forward on
     the trace (keys: run()'s levels, rows, live_most; temp=<one of TEMPS>; perfect=1 makes every
     draw the drafter's first choice). <call ms> is a 3-level drafter call as the real flow logs it.
  5. chain <dir> <tag,...> <name,...> [draft=<ms a guess> fixed=<ms a check> row=<ms a row>]: the
     tokens a check of a CHAIN of g guesses yields, g = 1..8, replayed on the ranks (a hit is rank
     0), and tok/s for checks that wait for the drafter — what the continuous stream and the rounds
     get where trees are off (a drafter or a segment off the card, e.g. the 70B).
Needs numpy (and llama_cpp for `ids`).
"""
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow_rows_sim  # noqa: E402

TEMPS = (0.5, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)
QWEN25_REAL_VOCAB = 151665  # the rest of a Qwen2.5 output row is padding
QWEN25_PADDED_VOCABS = (151936, 152064)


def real_vocab(vocab):
    """The columns of an output row that are tokens — a Qwen2.5 row is padded past them."""
    return QWEN25_REAL_VOCAB if vocab in QWEN25_PADDED_VOCABS else vocab


def chat_head(tok, prompt):
    """The prompt as the target's chat template renders it, from the vocabulary GGUF's own
    template (HF's conventions: bos_token, add_generation_prompt); Qwen2.5's literal head where the
    file carries none, as before."""
    template = tok.metadata.get("tokenizer.chat_template")
    if not template:
        return ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful "
                f"assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    import datetime
    import jinja2

    def raise_exception(message):
        raise ValueError(message)

    env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.datetime.now().strftime(fmt)
    bos = tok.metadata.get("tokenizer.ggml.bos_token_id")
    bos_text = tok.detokenize([int(bos)], special=True).decode() if bos is not None else ""
    return env.from_string(template).render(
        messages=[{"role": "user", "content": prompt}], add_generation_prompt=True,
        bos_token=bos_text)


def ids(d, name, prompt_file):
    from llama_cpp import Llama

    model = os.environ.get("SWARM_RANKS_MODEL", "qwen2.5-coder-7b-instruct-q4-k-m")
    vocab = os.path.expanduser(os.environ.get(
        "SWARM_RANKS_VOCAB", "~/swarmllm-ref/qwen2.5-coder-7b-instruct-q4-k-m.gguf"))
    key = open(os.path.expanduser("~/.local/share/swarmllm/api_key")).read().strip()
    prompt = open(prompt_file).read()
    max_tokens = int(os.environ.get("SWARM_RANKS_MAX_TOKENS", "400"))
    body = json.dumps({"model": model, "max_tokens": max_tokens, "temperature": 0,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("http://localhost:8800/v1/chat/completions", data=body, headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    reply = json.load(urllib.request.urlopen(req, timeout=1800))["choices"][0]["message"]["content"]
    tok = Llama(model_path=vocab, vocab_only=True, verbose=False)
    # The template with no system message of the caller's.
    head = chat_head(tok, prompt)
    p_ids = tok.tokenize(head.encode(), add_bos=False, special=True)
    r_ids = tok.tokenize(reply.encode(), add_bos=False, special=False)
    json.dump(p_ids + r_ids, open(f"{d}/{name}.ids.json", "w"))
    json.dump({"prompt_tokens": len(p_ids), "reply_tokens": len(r_ids), "reply": reply},
              open(f"{d}/{name}.meta.json", "w"))
    print(f"{name}: {len(p_ids)} prompt + {len(r_ids)} reply tokens; keep {len(r_ids) + 1}")


def reduce(d, name, tag, truth_tag=None):
    meta = json.load(open(f"{d}/{name}_{tag}.json"))
    n, vocab = meta["positions"], meta["vocab"]
    rows = np.fromfile(f"{d}/{name}_{tag}.f32", dtype="<f4").reshape(n, vocab)
    rows = rows[:, :real_vocab(vocab)].astype(np.float64)
    if truth_tag == "ids":
        # Position i predicts token i + 1: the kept rows begin at the prompt's last token, and the
        # last one predicts past the reply, where the reply's tokens say nothing.
        reply = json.load(open(f"{d}/{name}.meta.json"))["reply_tokens"]
        assert n == reply + 1, f"probe with KEEP_LAST = {reply + 1}, not {n}"
        n, rows = reply, rows[:reply]
    # A flat row ranks every token 0 and reads as a perfect drafter: the sparse "vocab" GGUFs in
    # ~/swarmllm-ref/vocab/ are 4.6 GB on paper, a few MB on disk, every weight zero (2026-10-29).
    flat = int((rows.max(axis=1) == rows.min(axis=1)).sum())
    assert flat == 0, f"{name}_{tag}: {flat} of {n} rows are flat — does the model file hold its weights?"
    top = np.argsort(-rows, axis=1)[:, :16]
    m = rows.max(axis=1, keepdims=True)
    out = {"top": top}
    if truth_tag == "ids":
        truth = np.array(json.load(open(f"{d}/{name}.ids.json"))[-n:])
    elif truth_tag:
        truth = np.load(f"{d}/{name}_{truth_tag}.npz")["top"][:, 0]
    if truth_tag:
        assert len(truth) == n, "the drafter and the target read different positions"
        t_logit = rows[np.arange(n), truth]
        out["rank"] = (rows > t_logit[:, None]).sum(axis=1)
        lse_T = [m[:, 0] / T + np.log(np.exp((rows - m) / T).sum(axis=1)) for T in TEMPS]
        out["truth_lp_T"] = np.stack([t_logit / T - lse for T, lse in zip(TEMPS, lse_T)], 1)
        out["top_lp_T"] = np.stack([np.take_along_axis(rows, top[:, :8], 1) / T - lse[:, None]
                                    for T, lse in zip(TEMPS, lse_T)], 1)  # [n, temps, 8]
    np.savez(f"{d}/{name}_{tag}.npz", **out)
    os.remove(f"{d}/{name}_{tag}.f32")
    print(f"{name}_{tag}: {n} positions")


def load(d, tag, names):
    return {p: np.load(f"{d}/{p}_{tag}.npz") for p in names if os.path.exists(f"{d}/{p}_{tag}.npz")}


def report(d, tags, names):
    i1 = TEMPS.index(1.0)
    for tag in tags:
        zs = load(d, tag, names)
        if not zs:
            continue
        r = np.concatenate([z["rank"] for z in zs.values()])
        lp = np.concatenate([z["truth_lp_T"] for z in zs.values()])
        tl = np.concatenate([z["top_lp_T"] for z in zs.values()])
        P = [float((r == k).mean()) for k in range(8)]
        print(f"\n== {tag}: {len(r)} positions ({','.join(zs)})")
        print("  P[rank] " + " ".join(f"{x:.3f}" for x in P) + f"  outside 8 {float((r >= 8).mean()):.3f}")
        print(f"  mean first-choice probability {np.exp(tl[:, i1, 0]).mean():.3f} against its hits {P[0]:.3f}")
        bins = (0, .3, .5, .7, .9, .97, 1.0001)
        for k in (0, 1):
            pk = np.exp(tl[:, i1, k])
            cells = [f"[{a:.2f},{b:.2f}) n={s.sum()} p={pk[s].mean():.2f} hit={(r[s] == k).mean():.2f}"
                     for a, b in zip(bins, bins[1:]) for s in [(pk >= a) & (pk < b)] if s.sum() >= 5]
            print(f"  rank {k}: " + "; ".join(cells))
        print("  log-loss of the target's choice: " + " ".join(
            f"T{T}={v:.3f}" for T, v in zip(TEMPS, -lp.mean(axis=0))))


def sim(d, tag, call_ms, names, *opts):
    kw = {k: float(v) if k == "temp" else int(v) for k, v in (o.split("=", 1) for o in opts)}
    ti = TEMPS.index(kw.pop("temp", 1.0))
    perfect = kw.pop("perfect", 0)
    zs = load(d, tag, names)
    pool = [list(row) for z in zs.values() for row in z["top_lp_T"][:, ti, :]]
    for name, z in zs.items():
        rank = np.zeros_like(z["rank"]) if perfect else z["rank"]
        trace = (rank, [list(row) for row in z["top_lp_T"][:, ti, :]], pool)
        rs = [flow_rows_sim.run(None, s, call_ms=float(call_ms), trace=trace, **kw) for s in range(3)]
        m = {k: float(np.mean([x[k] for x in rs])) for k in ("tps", "misses", "tok_fw", "wait")}
        print(f"{tag} {name}: {m['tps']:.2f} tok/s  misses {m['misses']:.1f}  "
              f"tokens a forward {m['tok_fw']:.2f}  waiting {m['wait']:.2f} s")


def chain_tokens(rank, guesses):
    """Tokens a chain check of `guesses` yields over one reply, replayed in order: the guesses kept
    up to the first miss, plus the target's own token there (or after the last guess)."""
    pos, checks = 0, 0
    while pos < len(rank):
        kept = 0
        while kept < guesses and pos + kept < len(rank) and rank[pos + kept] == 0:
            kept += 1
        pos += kept + 1
        checks += 1
    return len(rank) / max(checks, 1)


def chain(d, tags, names, *opts):
    kw = {k: float(v) for k, v in (o.split("=", 1) for o in opts)}
    for tag in tags:
        zs = load(d, tag, names)
        if not zs:
            continue
        hits = float(np.mean(np.concatenate([z["rank"] for z in zs.values()]) == 0))
        print(f"\n== {tag}: alpha (first choice is the target's) {hits:.3f} over {','.join(zs)}")
        for g in range(1, 9):
            per = float(np.mean([chain_tokens(z["rank"], g) for z in zs.values()]))
            line = f"  {g} guesses: {per:.2f} tokens a check"
            if {"draft", "fixed", "row"} <= kw.keys():
                ms = kw["fixed"] + kw["row"] * (g + 1) + kw["draft"] * g
                line += f"  {1000 * per / ms:.2f} tok/s at {ms:.0f} ms a check"
            print(line)


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "ids":
        ids(*args)
    elif cmd == "reduce":
        reduce(*args)
    elif cmd == "report":
        report(args[0], args[1].split(","), args[2].split(","))
    elif cmd == "sim":
        sim(args[0], args[1], args[2], args[3].split(","), *args[4:])
    elif cmd == "chain":
        chain(args[0], args[1].split(","), args[2].split(","), *args[3:])
    else:
        sys.exit(__doc__)
