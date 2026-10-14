#!/bin/bash
# The release gate: what only the DOWNLOADED artifact can show, in under an hour.
#
# usage: examples/release_gate.sh <candidate binary> <previous release binary> <outdir>
#   env BASELINE=<dir>  the previous release's gate outdir (its conformance.log and
#                       ab.jsonl). Missing files are taken now off <previous> (~20 min).
#   env SINCE=<rev>     the previous release's tag, for the diff-scoped steps
#                       (default: the newest v*-alpha tag that is not HEAD's)
#   env SPEED=0         skip the speed comparison
# Exit: 0 = no FAIL. REVIEW lines need a reader: $OUT/summary.txt lists every verdict.
# Run it with NO other SwarmLLM process on the machine (a rig shares the machine with
# nothing, #708) and, on a machine that has one, under its safety kit.
#
# WHY IT IS THIS SHORT (2026-10-14)
# ---------------------------------
# The gate grew from ~24 min (v0.3.207) to ~5 h (v0.3.234-.236), one step per fix: about
# 35 of its ~50 steps re-checked a single fix, and none of those ever failed on a release
# candidate. What did catch defects before they shipped: real generation on the downloaded
# artifact (v0.3.199 shipped garbage without it), reply A/B against the previous release
# and the split/failover rigs (#93, #767, #247), the heal rig (#776), every sample of a
# speed comparison instead of a best-of (#257 hid behind one), and a review of every commit
# since the last release (run beside CI, not here). So this gate checks the BUILD and
# looks for REGRESSIONS; it does not re-verify each fix. A fix is verified when it lands,
# and its commit carries the measurement. A rig that covers one subsystem runs when the
# release touches that subsystem (the diff-scoped steps below).
set -u
CAND=$(readlink -f "${1:?usage: release_gate.sh <candidate> <previous> <outdir>}")
PREV=$(readlink -f "${2:?previous release binary}")
OUT=$(mkdir -p "${3:?outdir}" && readlink -f "$3")
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(dirname "$HERE")
BASELINE=${BASELINE:-}
SINCE=${SINCE:-$(git -C "$REPO" describe --tags --abbrev=0 --match 'v*-alpha' --exclude "$(git -C "$REPO" describe --tags --exact-match 2>/dev/null || echo none)" HEAD 2>/dev/null)}
SUMMARY=$OUT/summary.txt
: > "$SUMMARY"
for b in "$CAND" "$PREV"; do [ -x "$b" ] || { echo "not executable: $b"; exit 2; }; done
if ps -eo comm | grep -q '^swarmllm'; then
  echo "another SwarmLLM process is running; stop it first (a rig shares the machine with nothing)"; exit 2
fi

# A machine's safety kit (SAFETY_KIT=<file to source>, with SAFETY_NAME and SAFETY_GPU_BASE exported by
# the caller that started it) settles between steps: the card's memory back to where the gate began.
[ -n "${SAFETY_KIT:-}" ] && source "$SAFETY_KIT"
verdict() { echo "$1: $2${3:+ — $3}" | tee -a "$SUMMARY"; }
step() { echo; echo "== $1 $(date -u +%T)"; }
settle() { if declare -F safety_settle > /dev/null; then safety_settle; fi; sleep 3; }

echo "candidate: $("$CAND" --version)   previous: $("$PREV" --version)   since: ${SINCE:-?}"

step "1 smoke"
SWARM_SMOKE_MODEL=${SWARM_SMOKE_MODEL:-llama-3.2-3b-instruct-q4-k-m} "$HERE/smoke_test.sh" "$CAND" 8809 > "$OUT/smoke.log" 2>&1
rc=$?; tail -3 "$OUT/smoke.log"
[ $rc = 0 ] && verdict smoke PASS || verdict smoke FAIL "exit $rc, $OUT/smoke.log"
settle

step "2 shapes"
"$HERE/release_shapes.sh" "$CAND" 8819 llama-3.2-3b-instruct-q4-k-m > "$OUT/shapes.log" 2>&1
rc=$?; tail -3 "$OUT/shapes.log"
[ $rc = 0 ] && verdict shapes PASS || verdict shapes FAIL "exit $rc, $OUT/shapes.log"
settle

# The two comparisons below diff against the previous release's own run; take it if missing.
norm() { sed -E 's/[0-9]+\.[0-9]+\.[0-9]+-alpha/VER/g; s/port [0-9]+/port N/g; s/[0-9]+(\.[0-9]+)? ?s\b/T/g' "$1"; }
if [ -z "$BASELINE" ] || [ ! -s "$BASELINE/conformance.log" ] || [ ! -s "$BASELINE/ab.jsonl" ]; then
  BASELINE=$OUT/baseline; mkdir -p "$BASELINE"
  step "2b baseline off the previous release"
  [ -s "$BASELINE/conformance.log" ] || "$HERE/family_conformance.sh" "$PREV" 8821 > "$BASELINE/conformance.log" 2>&1
  settle
  [ -s "$BASELINE/ab.jsonl" ] || "$HERE/reply_ab.sh" "$PREV" 8823 "$BASELINE/ab.jsonl" > "$BASELINE/ab.log" 2>&1
  settle
fi

step "3 conformance"
"$HERE/family_conformance.sh" "$CAND" 8821 > "$OUT/conformance.log" 2>&1
rc=$?; tail -4 "$OUT/conformance.log"
nd=$(diff <(norm "$BASELINE/conformance.log") <(norm "$OUT/conformance.log") | grep -c '^[<>]')
if [ $rc != 0 ]; then verdict conformance FAIL "exit $rc, $OUT/conformance.log"
elif [ "$nd" != 0 ]; then verdict conformance REVIEW "every family passed; $nd lines differ from the previous release"
else verdict conformance PASS "0 lines differ from the previous release"; fi
settle

step "4 reply A/B on the card"
"$HERE/reply_ab.sh" "$CAND" 8823 "$OUT/ab.jsonl" > "$OUT/ab.log" 2>&1
python3 - "$BASELINE/ab.jsonl" "$OUT/ab.jsonl" <<'PY' | tee "$OUT/ab.verdict"
import json, sys
load = lambda p: {(r["model"], r["prompt"]): r for r in map(json.loads, open(p))}
a, b = load(sys.argv[1]), load(sys.argv[2])
same = [k for k in b if k in a and a[k].get("content") == b[k].get("content")]
errs = [k for k in b if "error" in b[k] or not (b[k].get("content") or b[k].get("reasoning"))]
diff = [k for k in b if k not in same and k not in errs]
for k in diff:
    print(f"  differs: {k[0]} / {k[1]!r}\n    prev: {a.get(k, {}).get('content', '<absent>')[:160]!r}\n    cand: {b[k].get('content', '')[:160]!r}")
for k in errs:
    print(f"  ERROR: {k[0]} / {k[1]!r}: {b[k].get('error') or 'empty reply'}")
v = "FAIL" if errs or len(b) < len(a) else ("REVIEW" if diff else "PASS")
print(f"ab {v} {len(same)}/{len(b)} identical, {len(diff)} differ (judge against llama.cpp: examples/score_against_reference.py), {len(errs)} errors")
PY
read -r _ v rest < <(tail -1 "$OUT/ab.verdict"); verdict reply_ab "$v" "$rest"
settle

step "5 split across two nodes"
OUT="$OUT/split" "$HERE/split_rig.sh" split "$CAND" > "$OUT/split.log" 2>&1
python3 - "$OUT/split/replies.jsonl" <<'PY' > "$OUT/split.verdict" 2>&1
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
# Two nodes really took part: the route left this node and the plan had two segments or more.
ok = len(rows) == 2 and all(r["status"].endswith("200 ok") and r.get("content") and r.get("route") not in (None, "local")
                            and int(r.get("segments") or 0) >= 2 for r in rows)
print("PASS" if ok else "FAIL", "; ".join(f'{r["status"]} route={r.get("route")} segments={r.get("segments")}' for r in rows))
PY
read -r v rest < "$OUT/split.verdict"; verdict split "${v:-FAIL}" "${rest:-no replies, $OUT/split.log}"
settle

step "6 failover (a segment's node killed mid-prompt)"
OUT="$OUT/failover" "$HERE/split_rig.sh" failover "$CAND" > "$OUT/failover.log" 2>&1
grep -h '^failover:' "$OUT/failover.log" | tail -3
grep -q '^failover: PASS' "$OUT/failover.log" && verdict failover PASS || verdict failover FAIL "$OUT/failover.log"
settle

# Every sample printed, each prompt unique (#257): candidate against the previous release per
# placement. Below 85% of the previous median is REVIEW (re-run that placement before blaming
# the build: a dip in both arms is the machine).
if [ "${SPEED:-1}" != 0 ]; then
  step "7 speed per placement, candidate vs previous"
  PROMPT_FILE=$OUT/speed_prompt.txt
  { echo "Summarise the following document in three short paragraphs."; echo; head -c 7000 "$REPO/docs/ARCHITECTURE.md"; } > "$PROMPT_FILE"
  arm() { # binary label model extra_toml
    local D; D=$(mktemp -d); ln -s "$HOME/.local/share/swarmllm/models" "$D/models"
    printf '[auto_manage]\nenabled = false\nprune_enabled = false\n\n[network]\nbootstrap_peers = []\ndisable_default_bootstrap = true\nenable_mdns = false\ngossip_network_id = "swarmllm-gate-speed"\n\n[ui]\nopen_browser_on_start = false\n%b\n' "$4" > "$D/config.toml"
    SWARMLLM_NODE_DATA_DIR="$D" "$1" run -p 8825 > "$D/node.log" 2>&1 &
    local P=$!
    for _ in $(seq 1 90); do [ -f "$D/api_key" ] && curl -s -m 3 localhost:8825/health > /dev/null && break; sleep 2; done
    SWARMLLM_API_KEY_FILE="$D/api_key" python3 "$HERE/stream_bench.py" "$3" --port 8825 --unique --reps 3 --warmup 1 \
      --max-tokens 128 --prompt-file "$PROMPT_FILE" --label "$2" --json-out "$OUT/speed.jsonl" 2>&1 | grep -v '^    text'
    kill $P; wait $P 2>/dev/null; rm -f "$D/models"; rm -rf "$D"; settle
  }
  : > "$OUT/speed.jsonl"
  # name|model|config lines (\n in the config is a newline): whole on an 8 GB card, split between
  # card and processor, a mixture of experts too big for the card, processor only.
  PLACEMENTS=${PLACEMENTS:-'card|qwen2.5-coder-7b-instruct-q4-k-m|
split|qwen2.5-14b-instruct-q4-k-m|
moe|qwen3-30b-a3b-instruct-2507-q2-k|
cpu|llama-3.2-3b-instruct-q4-k-m|[inference]\ngpu_layers = 0'}
  while IFS='|' read -r name model extra; do
    [ -n "$name" ] || continue
    arm "$PREV" "$name/prev" "$model" "$extra"
    arm "$CAND" "$name/cand" "$model" "$extra"
  done <<< "$PLACEMENTS"
  python3 - "$OUT/speed.jsonl" <<'PY' | tee "$OUT/speed.verdict"
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
worst = "PASS"
for name in dict.fromkeys(r["label"].split("/")[0] for r in rows):
    get = lambda arm, k: [r[k] for r in rows if r["label"] == f"{name}/{arm}" and r.get(k)]
    line = []
    for k in ("prompt_tps", "decode_tps"):
        p, c = get("prev", k), get("cand", k)
        if not p or not c:
            line.append(f"{k} MISSING prev {p} cand {c}"); worst = "FAIL"; continue
        ratio = statistics.median(c) / statistics.median(p)
        if ratio < 0.85 and worst == "PASS": worst = "REVIEW"
        line.append(f"{k} prev {p} cand {c} ({ratio:.2f}x)")
    print(f"  {name}: " + " | ".join(line))
print(f"speed {worst}")
PY
  verdict speed "$(tail -1 "$OUT/speed.verdict" | cut -d' ' -f2)" "every sample in $OUT/speed.verdict"
fi

# Diff-scoped: a rig that covers one subsystem, when the release touched it.
step "8 diff-scoped since ${SINCE:-?}"
CHANGED=$(git -C "$REPO" diff --name-only "${SINCE:-HEAD}"..HEAD 2>/dev/null)
if grep -qE '^src/model/(canonical|acquisition|manifest|shard)\.rs|^src/storage/' <<< "$CHANGED"; then
  OUT="$OUT/outvoted" "$HERE/outvoted_rig.sh" "$CAND" > "$OUT/outvoted.log" 2>&1
  grep -q ': PASS' "$OUT/outvoted.log" && verdict outvoted PASS || verdict outvoted FAIL "$OUT/outvoted.log"
  settle
else
  verdict outvoted SKIPPED "no heal/canonical/storage change"
fi
if grep -qiE 'windows|update_restart|job_object|launcher' <<< "$CHANGED" \
   || git -C "$REPO" diff "${SINCE:-HEAD}"..HEAD -- src | grep -qE '^[+-].*cfg\((target_os = "windows"|windows)'; then
  verdict windows REVIEW "Windows code changed: wincheck the draft's .exe files now, a real update after publishing"
else
  verdict windows SKIPPED "no Windows code changed"
fi

echo; echo "== summary $(date -u +%T)"; cat "$SUMMARY"
! grep -q ': FAIL' "$SUMMARY"
