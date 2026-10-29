#!/usr/bin/env python3
"""Rewrite the README's "On one computer" speeds from a release check's own records.

The release check (`examples/release_gate.sh`, step 7) times four placements on the
DOWNLOADED release and on the one before it, and writes every sample to
`<gate dir>/out/speed.jsonl`. This takes the samples of the release being checked
(`<place>/cand/<n>`) and rewrites the block between the README's
`<!-- release-check:begin -->` and `<!-- release-check:end -->` markers, so these
figures are copied by a program, never typed. Run it in each release's bump commit
on the PREVIOUS release's records — the newest the gate has measured
(.claude/rules/completeness.md § "README speeds"; guard
`the_readme_speeds_are_reviewed_at_every_release`).

usage: python3 -I examples/readme_speeds.py ~/swarmllm-gate/v0.3.260-alpha/out/speed.jsonl [README.md]

The version is read from the path: the gate writes under `v<version>/`.
"""

import json
import re
import statistics
import sys
from pathlib import Path

# The gate's placements, in the order the README lists them.
PLACES = {
    "card": "the graphics card",
    "split": "card and processor together (too big for the card)",
    "moe": "card and processor together (a mixture of experts, too big for the card)",
    "cpu": "the processor only",
}
NAMES = {
    "qwen2.5-coder-7b-instruct-q4-k-m": "Qwen2.5 Coder 7B",
    "qwen2.5-14b-instruct-q4-k-m": "Qwen2.5 14B",
    "qwen3-30b-a3b-instruct-2507-q2-k": "Qwen3 30B-A3B (2-bit)",
    "llama-3.2-3b-instruct-q4-k-m": "Llama 3.2 3B",
}
BEGIN = "<!-- release-check:begin -->"
END = "<!-- release-check:end -->"


def span(values, step):
    """`lo–hi` rounded to `step` (1 → whole numbers, 0.1 → one decimal, 10 → tens)."""
    def fmt(v):
        r = round(v / step) * step
        return f"{r:,.1f}" if step < 1 else f"{int(r):,}"
    lo, hi = fmt(min(values)), fmt(max(values))
    return lo if lo == hi else f"{lo}–{hi}"


def block(records, version):
    arms = {}
    for r in records:
        place, who, _ = (r["label"].split("/") + ["", ""])[:3]
        if who != "cand" or r.get("error") or not r.get("decode_tps"):
            continue
        arms.setdefault(place, []).append(r)
    missing = [p for p in PLACES if p not in arms]
    if missing:
        sys.exit(f"no candidate samples for {missing}: is this a release check's speed.jsonl?")
    every = [r for p in PLACES for r in arms[p]]
    prompt = round(statistics.median(r["prompt_tokens"] for r in every), -2)
    written = int(statistics.median(r["completion_tokens"] for r in every))
    lines = [
        BEGIN,
        f"**On one computer**: the v{version} release check, run on the downloaded release. Each",
        f"model reads a {int(prompt):,}-token document and writes {written} tokens about it.",
        "",
        "| Model | Runs on | Writing (tokens a second) | Reading (tokens a second) |",
        "|---|---|---|---|",
    ]
    for place, where in PLACES.items():
        rows = arms[place]
        # Each run's best repeat, then the range across runs: the gate's own
        # reading (its SUMMARY "best"), min-of-N as CLAUDE.md § Measurement asks.
        # A run's first repeat is slower on the processor while it warms.
        def best(field):
            runs = {}
            for r in rows:
                if r.get(field):
                    runs[r["label"]] = max(runs.get(r["label"], 0), r[field])
            return list(runs.values())
        decode, reading = best("decode_tps"), best("prompt_tps")
        name = NAMES.get(rows[0]["model"], rows[0]["model"])
        write = span(decode, 1 if max(decode) >= 30 else 0.1)
        read = span(reading, 10) if reading else "—"
        lines.append(f"| {name} | {where} | {write} | {read} |")
    lines.append(END)
    return "\n".join(lines)


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    source = Path(sys.argv[1]).expanduser()
    found = re.search(r"v(\d+\.\d+\.\d+)", str(source))
    if not found:
        sys.exit(f"{source}: no v<version> in the path (the gate writes under v<version>/)")
    readme = Path(sys.argv[2]) if len(sys.argv) == 3 else Path(__file__).resolve().parent.parent / "README.md"
    records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    text = readme.read_text()
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < start:
        sys.exit(f"{readme}: the {BEGIN} … {END} markers are missing")
    new = text[:start] + block(records, found.group(1)) + text[end + len(END):]
    readme.write_text(new)
    print(block(records, found.group(1)))


if __name__ == "__main__":
    main()
