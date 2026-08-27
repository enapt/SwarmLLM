#!/usr/bin/env bash
# Exercise the REQUEST SHAPES that smoke_test.sh cannot.
#
# Why this exists: every defect found on 2026-08-26 — foreign peers adopted as
# swarm members, a model taking the GPU then refusing everything, and the
# caller's sampling parameters being discarded — was found by RUNNING the node
# with a real request, not by tests. `cargo test` and `clippy` were green for
# all three, and `smoke_test.sh` passed, because that script asks only whether
# the node starts and answers a short prompt on a warm model.
#
# The shapes below are the ones that were actually broken:
#   COLD START  — an unloaded model declines the split fast path and goes
#                 through the pipeline, which is a different code path with
#                 different bugs. A `curl` retry is WARM and never sees it.
#   LONG PROMPT — an agentic client sends thousands of tokens of system prompt
#                 before the user speaks.
#   TOOLS       — 10+ tool schemas change how the prompt is built.
#   GREEDY      — top_k=1 must be deterministic; if it is not, the caller's
#                 sampling parameters are not reaching the sampler.
#
# Run against the DOWNLOADED release artifact, before tagging.
#   examples/release_shapes.sh ./swarmllm-linux-x86_64-cuda 8819 <model-id>
set -u
BIN="${1:-./target/release/swarmllm}"
PORT="${2:-8819}"
MODEL="${3:-llama-3.2-3b-instruct-q4-k-m}"
MODELS_DIR="${SWARM_SHAPES_MODELS_DIR:-$HOME/.local/share/swarmllm/models}"
[ -x "$BIN" ] || { echo "not executable: $BIN"; exit 1; }

D=$(mktemp -d)
cleanup() { [ -n "${PID:-}" ] && kill "$PID" 2>/dev/null; rm -f "$D/models"; rm -rf "$D"; }
trap cleanup EXIT
[ -d "$MODELS_DIR" ] && ln -s "$MODELS_DIR" "$D/models"
# auto-manage OFF: this throwaway node shares the real models directory and
# would otherwise prune the shards of the node you actually run.
printf '[auto_manage]\nenabled = false\nprune_enabled = false\n' > "$D/config.toml"

echo "shapes: $("$BIN" --version) on port $PORT"
SWARMLLM_NODE_DATA_DIR="$D" "$BIN" run -p "$PORT" > "$D/node.log" 2>&1 &
PID=$!
sleep 12
K=$(cat "$D/api_key" 2>/dev/null || true)
API="http://localhost:$PORT"
fails=0
skipped=0
check() { if [ "$2" = "0" ]; then printf '  %-40s OK\n' "$1"; else printf '  %-40s FAIL\n' "$1"; fails=$((fails+1)); fi; }

curl -s -m 10 -o /dev/null "$API/health" || { echo "  node did not start"; exit 1; }

if ! curl -s -m 15 -H "Authorization: Bearer $K" "$API/v1/models" 2>/dev/null | grep -q "$MODEL"; then
  echo "  (model $MODEL not present — shape checks skipped)"
  exit 0
fi

post() { curl -s -m 600 -H "Authorization: Bearer $K" -H "Content-Type: application/json" \
          -X POST "$API/v1/chat/completions" -d "$1"; }
unload() { curl -s -m 60 -X POST -H "Authorization: Bearer $K" \
            "$API/api/admin/models/$MODEL/unload" >/dev/null 2>&1; }
ntok() { python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('completion_tokens',0) if 'error' not in d else 0)"; }
ptok() { python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('usage',{}).get('prompt_tokens',0) if 'error' not in d else 0)"; }
text() { python3 -c "import sys,json;d=json.load(sys.stdin);print('' if 'error' in d else d['choices'][0]['message']['content'])"; }

# 1. COLD START — the path a first request actually takes.
unload; sleep 2
N=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colours.\"}],\"max_tokens\":40}" | ntok)
[ "${N:-0}" -gt 3 ]; check "cold start returns a real reply" $?

# 2. LONG PROMPT, COLD — the agentic-client shape.
#
# The length matters, and 1400 tokens is not enough. This check existed, and
# passed, all through v0.3.128 while a cold long request was answering with a
# single token: the coordinator measured the prompt as `chars / 4` and used
# that as the position the reply continues from, so the error scaled with
# prompt length and a short prompt still read correctly (gotcha #400). The
# shape that actually breaks is an agentic client's — tens of KB of system
# prompt, several thousand tokens.
# 460 repeats is ~24 KB and ~5500 tokens: large enough that the old estimate
# missed by ~460 positions, and still inside the shipped 8192-token default so
# the request is answered rather than refused. A refusal would fail this check
# for the wrong reason, which is how the first attempt at it went.
LONG=$(python3 -c "print('You are a coding assistant. Be precise and complete. ' * 460)")
BODY=$(LONG="$LONG" MODEL="$MODEL" python3 -c "
import json,os
print(json.dumps({'model':os.environ['MODEL'],'max_tokens':40,'messages':[
 {'role':'system','content':os.environ['LONG']},
 {'role':'user','content':'Name three primary colours.'}]}))")
unload; sleep 2
COLD=$(post "$BODY")
N=$(printf '%s' "$COLD" | ntok)
[ "${N:-0}" -gt 3 ]; check "long system prompt, cold, returns a reply" $?

# 2b. The SAME request, warm, must report the SAME prompt length.
#
# This is the check that discriminates, and reply length is not. The same fault
# also produces one token repeated to the limit, which passes "more than 3
# tokens" comfortably. But the number of tokens in a prompt is a property of
# the prompt: cold and warm must agree exactly. They disagreed by 524 on the
# request that was failing, and that disagreement IS the bug rather than a
# symptom of it — the estimate was being handed to the model as a position.
# No tokenizer is needed here; the node is asked the same question twice.
WARM=$(post "$BODY")
PCOLD=$(printf '%s' "$COLD" | ptok)
PWARM=$(printf '%s' "$WARM" | ptok)
if [ "${PCOLD:-0}" -eq 0 ] || [ "${PWARM:-0}" -eq 0 ]; then
  printf '  %-40s COULD NOT RUN (no usage reported)\n' "prompt length agrees cold and warm"
  skipped=$((skipped+1))
else
  [ "$PCOLD" = "$PWARM" ]; check "prompt length agrees cold and warm" $?
  [ "$PCOLD" = "$PWARM" ] || echo "      cold=$PCOLD warm=$PWARM"
fi

# 3. GREEDY DETERMINISM — proves the caller's sampling reaches the sampler.
unload; sleep 2
A=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Invent a two-line poem about rain.\"}],\"max_tokens\":24,\"top_k\":1,\"temperature\":0}" | text)
unload; sleep 2
B=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Invent a two-line poem about rain.\"}],\"max_tokens\":24,\"top_k\":1,\"temperature\":0}" | text)
[ -n "$A" ] && [ "$A" = "$B" ]; check "greedy is deterministic across cold starts" $?
# Control: without it, an always-deterministic prompt would pass check 3 vacuously.
C=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Invent a two-line poem about rain.\"}],\"max_tokens\":24}" | text)
Dv=$(post "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Invent a two-line poem about rain.\"}],\"max_tokens\":24}" | text)
if [ "$C" = "$Dv" ]; then
  printf '  %-40s COULD NOT RUN (prompt is deterministic anyway)\n' "greedy check has a live control"
  skipped=$((skipped+1))
else
  check "greedy check has a live control" 0
fi

# 4. TOOLS — 12 schemas change how the prompt is built.
unload; sleep 2
N=$(post "$(python3 -c "
import json
tools=[{'type':'function','function':{'name':n,'description':f'The {n} tool.',
 'parameters':{'type':'object','properties':{'path':{'type':'string'}},'required':['path']}}}
 for n in ['read','bash','edit','write','grep','find','ls','fetch','search','run','test','plan']]
print(json.dumps({'model':'$MODEL','max_tokens':40,'tools':tools,
 'messages':[{'role':'user','content':'Name three primary colours.'}]}))")" | ntok)
[ "${N:-0}" -gt 3 ]; check "tool-heavy request returns a reply" $?

ERRS=$(grep -cE " ERROR " "$D/node.log" || true)
[ "${ERRS:-0}" -eq 0 ]; check "no errors logged" $?

echo
if [ "$fails" -gt 0 ]; then echo "shapes: $fails check(s) FAILED"; exit "$fails"; fi
if [ "$skipped" -gt 0 ]; then echo "shapes: $skipped check(s) COULD NOT RUN; the rest passed"; exit 0; fi
echo "shapes: all checks passed"
