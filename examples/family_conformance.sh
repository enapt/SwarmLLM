#!/usr/bin/env bash
# Does each model FAMILY produce a sane reply, or only bytes?
#
# Why this exists
# ---------------
# Four consecutive releases fixed field-reported bugs of one shape: a
# family-specific defect in the prompt, the stop condition, or the tool framing.
#
#   v0.3.169  every Qwen3 request reached the model with the question MISSING
#   v0.3.170  the chat-template renderer half-rendered the official template
#   v0.3.171  `tools` was never passed to the template, on every request ever made
#   (this)    Phi declares one end-of-turn token and emits another, so replies
#             never stopped — 120/120 tokens of invented conversation
#
# 2566 unit tests stayed green through all four, and so did the release gate:
# `release_shapes.sh` runs ONE family (`llama-3.2-3b`) and asserts that more than
# three tokens came back. **All four bugs pass that.** Its tools check is aimed
# at exactly this feature and asserts the wrong property — it sends twelve
# schemas with "Name three primary colours", a prompt that should produce no
# call, and verifies a REPLY exists, so a node where tool calling is completely
# broken passes.
#
# The unit tests cannot cover it either: they render templates and compare
# strings, which is llama.cpp's `test-chat-template.cpp` shape and catches
# rendering bugs. Every bug above was a GENERATION-level defect — the reply was
# produced, and was wrong. Only running a real model finds those.
#
# So this asserts PROPERTIES OF THE REPLY, per family:
#
#   1. it answers a question with a checkable answer   (question reached the model)
#   2. it stops by itself                              (the end-of-turn token is right)
#   3. it carries no control markers                   (nothing leaked into content)
#   4. a tool call is parsed, not returned as text     (the extractor knows the format)
#   5. the model's own template rendered                (no silent fallback)
#   6. nothing was logged as an ERROR
#
# Check 4 deliberately fails ONLY when the model tried and we did not parse it —
# `content` looks like a call while `tool_calls` is null. A small model that
# answers in prose instead is not a bug in this daemon, and asserting it must
# call would make this harness flaky on model temperament rather than on our code.
#
# Getting the models
# -------------------
# These are ordinary models in the local store, not a fixture. Add one with
# `swarmllm get-model`, the dashboard, or directly:
#
#   curl -sH "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
#     -d '{"repo_id":"bartowski/microsoft_Phi-4-mini-instruct-GGUF",
#          "filename":"microsoft_Phi-4-mini-instruct-Q4_K_M.gguf","shards":[0,1,2,3,4]}' \
#     localhost:8800/api/admin/hf/download-shards
#
# `GET /api/admin/hf/probe?repo_id=…&filename=…` answers `shard_count` first.
#
# Usage:  examples/family_conformance.sh [binary] [port] [model-id ...]
# Exit:   0 = every present family passed, N = N failed checks, 2 = nothing ran
#
# Models absent from the local store are reported as COULD NOT RUN, separately
# and loudly. A harness that counts a skipped family as a pass is the thing
# `smoke_test.sh` was fixed for in 2026-08-25.
set -u

BIN="${1:-./target/release/swarmllm}"
PORT="${2:-8821}"
shift 2 2>/dev/null || true
# Add a model here when its family is not already represented — the point is
# family COVERAGE, not model count.
# One per template family we can hold locally, and BOTH Phi tokenizer families.
# That pair is not redundant: Phi-3.5's vocabulary is SentencePiece and Phi-4's is
# GPT-2 BPE, and the same missing end-of-turn token was INVISIBLE on the first and
# LEAKED into the reply on the second. One cause, two symptoms, and only one of
# them is what a user reports. The `<|end|>` token also sits at id 32007 in one and
# 200020 in the other, which is why it is found by name.
#
# `gemma2` and `tinyllama` are here for their own reasons, not for size. Gemma's
# template REFUSES a system role with `raise_exception` and its attention carries a
# logit soft-cap, so it exercises the two places the renderer and the attention
# tail behave differently from everyone else. TinyLlama is the `zephyr` fallback
# family — its name contains "llama" while its format is not Llama's, which is the
# case `fallback_by_model_name` has a dedicated early branch for (gotcha #169).
DEFAULT_MODELS="llama-3.2-3b-instruct-q4-k-m qwen3-1.7b-q8-0 phi-3.5-mini-instruct.q4-k-m microsoft-phi-4-mini-instruct-q4-k-m qwen2.5-coder-7b-instruct-q4-k-m gemma-2-2b-it-q4-k-m tinyllama-1.1b-chat-v1.0.q4-k-m"
MODELS="${*:-$DEFAULT_MODELS}"
MODELS_DIR="${SWARM_CONFORMANCE_MODELS_DIR:-$HOME/.local/share/swarmllm/models}"
[ -x "$BIN" ] || { echo "not executable: $BIN"; exit 2; }

D=$(mktemp -d)
# Keep the log when something failed: the check names the fault, the log explains
# it, and a harness that deletes its own evidence sends you back to re-run a
# twenty-minute job to learn what it already knew.
cleanup() {
  [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null
  rm -f "$D/models"
  if [ "${fails:-0}" -gt 0 ] && [ -f "$D/node.log" ]; then
    keep="${TMPDIR:-/tmp}/conformance-failed-$$.log"
    cp "$D/node.log" "$keep" 2>/dev/null && echo "node log kept at $keep"
  fi
  rm -rf "$D"
}
trap cleanup EXIT
[ -d "$MODELS_DIR" ] && ln -s "$MODELS_DIR" "$D/models"

# auto-manage OFF: this throwaway node shares the real models directory via the
# symlink above and would otherwise prune the shards of the node you actually
# run. And off the public swarm: a private `gossip_network_id` is what isolates
# it — an empty bootstrap list does NOT, because cached peers and DHT discovery
# do not care about it (gotcha #536). Every check here must be answered by THIS
# binary, not by a peer.
cat > "$D/config.toml" <<'TOML'
[auto_manage]
enabled = false
prune_enabled = false

[network]
bootstrap_peers = []
disable_default_bootstrap = true
enable_mdns = false
gossip_network_id = "swarmllm-conformance"

[pool]
private_mode = true
private_mode_allow_lan = false
offline_mode = true

# Processor only, deliberately. Every property asserted here — did the question
# arrive, did the model stop, did a marker leak, was the call parsed — is decided
# by the prompt, the token ids and the parser, none of which depend on the device.
# Pinning it makes a run on a CUDA release artifact comparable with a run on a
# processor-only build, and stops the harness competing for the card with a live
# node (which refuses the load outright, turning a conformance question into a
# memory question). `smoke_test.sh` and `release_shapes.sh` already exercise an
# artifact on its native device.
[inference]
gpu_layers = 0
TOML

echo "conformance: $("$BIN" --version) on port $PORT"
SWARMLLM_NODE_DATA_DIR="$D" "$BIN" run -p "$PORT" -c "$D/config.toml" > "$D/node.log" 2>&1 &
PID=$!
API="http://localhost:$PORT"
for _ in $(seq 1 60); do
  curl -sf -m 2 "$API/health/ready" 2>/dev/null | grep -q '"ready":true' && break
  sleep 1
done
K=$(cat "$D/api_key" 2>/dev/null || true)
[ -n "$K" ] || { echo "node never became ready; see $D/node.log"; exit 2; }

fails=0; ran=0; skipped=""
check() { if [ "$2" = "0" ]; then printf '    %-46s OK\n' "$1"; else printf '    %-46s FAIL  %s\n' "$1" "${3:-}"; fails=$((fails+1)); fi; }

post() { curl -s -m 600 -H "Authorization: Bearer $K" -H 'Content-Type: application/json' \
           -d "$1" "$API/v1/chat/completions"; }

for M in $MODELS; do
  if [ ! -f "$MODELS_DIR/$M/manifest.json" ]; then
    skipped="$skipped $M(absent)"; continue
  fi
  # A metadata-only model directory cannot serve anything; saying "skipped"
  # is honest, saying "passed" is not (gotcha #311).
  want=$(python3 -c "import json;print(len(json.load(open('$MODELS_DIR/$M/manifest.json'))['shards']))" 2>/dev/null || echo 0)
  have=$(ls "$MODELS_DIR/$M"/shard_*.bin 2>/dev/null | wc -l)
  if [ "$want" = "0" ] || [ "$have" -lt "$want" ]; then
    skipped="$skipped $M($have/$want-shards)"; continue
  fi

  echo; echo "  $M"
  ran=$((ran+1))
  logmark=$(wc -l < "$D/node.log")

  # 1-3. A question with a checkable answer. A model that never stops runs to the
  # cap here; a model that never saw the question answers something else fluently.
  #
  # "in one short sentence" rather than "only the number" deliberately: a
  # one-token reply is CORRECT for the latter and trips the daemon's own
  # "ended its turn immediately, well inside the token budget" heuristic, which
  # then sits in the log of every clean run looking like a fault.
  #
  # The budget is generous because a REASONING model spends it before it answers:
  # Qwen3-1.7B replied `<think>\nOkay, the user is asking for 2 plus 2. Let me
  # think.` and ran out at 80, failing both checks for no fault of the daemon's.
  # A model that genuinely never stops still hits this ceiling — the Phi defect
  # this harness exists for produced 120 of 120 — so raising it costs nothing in
  # detection and removes a false alarm on every reasoning model.
  R=$(post "$(python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "max_tokens": 600, "temperature": 0,
 "messages": [{"role": "user",
               "content": "What is 2 plus 2? Answer in one short sentence."}]}))' "$M")")
  # A response with no `choices` means the request never produced a reply — a
  # refusal, an auth failure, or a harness bug. Every check below would then be
  # measuring nothing, and two of them (the log scans) would PASS, because a
  # request that never ran logs neither a warning nor an error. Reporting that as
  # a pass is the defect `smoke_test.sh` was fixed for; report the family as
  # COULD NOT RUN and move on.
  if ! grep -q '"choices"' <<<"$R" 2>/dev/null; then
    ran=$((ran-1))
    skipped="$skipped $M(no-reply)"
    echo "    COULD NOT RUN — no reply. Server said:"
    printf '      %.200s\n' "$R"
    continue
  fi
  FIN=$(python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('finish_reason','<none>'))" <<<"$R" 2>/dev/null || echo '<parse-error>')
  TXT=$(python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('choices',[{}])[0].get('message',{}).get('content') or ''))" <<<"$R" 2>/dev/null || echo '')

  # The word counts. Phi-4-mini answers "Four." and a digit-only match failed it,
  # which says nothing about this daemon — a model that never received the
  # question cannot produce either spelling, so accepting both costs the check
  # none of its power. Same class as the two harness defects this file's first
  # run produced (gotcha #548): the assertion, not the daemon, was wrong.
  case "$(printf '%s' "$TXT" | tr '[:upper:]' '[:lower:]')" in
    *4*|*four*) r=0;;
    *) r=1;;
  esac
  check "answers a checkable question" "$r" "got: $(printf '%.60s' "$TXT")"

  r=0; [ "$FIN" = "stop" ] || r=1
  check "stops by itself" "$r" "finish_reason=$FIN (length = never stopped)"

  r=0
  for mark in '<|' '<start_of_turn>' '<end_of_turn>' '</s>' '[INST]' '[/INST]'; do
    case "$TXT" in *"$mark"*) r=1;; esac
  done
  check "no control marker in the reply" "$r" "reply: $(printf '%.60s' "$TXT")"

  # 4. Tool framing. Fails only when the model produced a call we failed to read.
  RT=$(post "$(python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "max_tokens": 160, "temperature": 0,
 "tools": [{"type": "function", "function": {"name": "get_time",
   "description": "Get the current time",
   "parameters": {"type": "object", "properties": {"zone": {"type": "string"}},
                  "required": ["zone"]}}}],
 "messages": [{"role": "user",
               "content": "Call get_time for zone UTC. Do not explain."}]}))' "$M")")
  TC=$(python3 -c "import json,sys;d=json.load(sys.stdin);print('yes' if (d.get('choices',[{}])[0].get('message',{}).get('tool_calls')) else 'no')" <<<"$RT" 2>/dev/null || echo 'no')
  CT=$(python3 -c "import json,sys;d=json.load(sys.stdin);print((d.get('choices',[{}])[0].get('message',{}).get('content') or ''))" <<<"$RT" 2>/dev/null || echo '')
  looks=0
  for mark in 'tool_call' 'functools' '"arguments"' "'arguments'" '[TOOL_CALLS]'; do
    case "$CT" in *"$mark"*) looks=1;; esac
  done
  if [ "$TC" = "yes" ]; then
    check "tool call parsed into tool_calls" 0
  elif [ "$looks" = "1" ]; then
    check "tool call parsed into tool_calls" 1 "call left in content: $(printf '%.70s' "$CT")"
  else
    printf '    %-46s n/a   model answered in prose, no call attempted\n' "tool call parsed into tool_calls"
  fi

  # 5-6. What the daemon said about this model while serving it.
  NEW=$(tail -n "+$((logmark+1))" "$D/node.log")
  # Anchored on `DIAG: `, because the bare phrases appear inside OTHER messages
  # as advice to the reader: the daemon's "ended its turn immediately" warning
  # ends with "(grep `chat template failed`)", and the unanchored pattern matched
  # that — reporting a template failure on a model whose template rendered
  # perfectly. Same trap as gotcha #521.
  BAD=$(grep -acE "DIAG: chat template failed|DIAG: chat template rendered a prompt|DIAG: no chat template for a request|rendered none of them" <<<"$NEW" || true)
  r=0; [ "${BAD:-0}" -eq 0 ] || r=1
  check "the model's own template rendered" "$r" "$BAD template warning(s) — grep node.log"
  ERRS=$(grep -ac " ERROR " <<<"$NEW" || true)
  r=0; [ "${ERRS:-0}" -eq 0 ] || r=1
  check "nothing logged as ERROR" "$r" "$ERRS ERROR line(s)"

  curl -s -m 60 -X POST -H "Authorization: Bearer $K" "$API/api/admin/models/$M/unload" >/dev/null 2>&1 || true
  sleep 2
done

echo
if [ -n "$skipped" ]; then
  echo "COULD NOT RUN (not a pass):$skipped"
  echo "  Fetch with examples/fetch_reference_model.sh, or pass model ids as arguments."
fi
if [ "$ran" = "0" ]; then
  echo "conformance: NOTHING RAN — no complete model found. This is not a pass."
  exit 2
fi
if [ "$fails" -gt 0 ]; then
  echo "conformance: $fails check(s) FAILED across $ran family/families"
  exit "$fails"
fi
echo "conformance: every check passed on $ran family/families (see COULD NOT RUN above, if any)"
