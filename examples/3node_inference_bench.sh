#!/bin/bash
# SWARM-SPEC real-inference benchmark across 3 local nodes.

set -e
API_KEY_A=$(cat /tmp/swarm_bench_a/api_key)
PORT=8800
# Model under test. Defaults to the Smoke tier; see docs/REFERENCE_MODELS.md
# for the Standard/Stress pins and why the choice matters for comparability.
MODEL="${SWARM_BENCH_MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"

json_escape() {
    python3 -c 'import sys, json; print(json.dumps(sys.stdin.read()), end="")'
}

parse_field() {
    python3 -c "import sys, json; d=json.loads(sys.stdin.read()); print(d.get('usage', {}).get('$1', 0))"
}

bench_one() {
    local label=$1
    local prompt=$2
    local max_tok=$3
    echo
    echo "=== $label ==="
    echo "Prompt: $(echo "$prompt" | head -c 80)..."
    echo "Max tokens: $max_tok"
    PROMPT_JSON=$(printf '%s' "$prompt" | json_escape)
    for trial in 1 2 3; do
        START=$(date +%s%N)
        RESPONSE=$(curl -s -m 180 -H "Authorization: Bearer $API_KEY_A" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"messages\": [{\"role\":\"user\",\"content\":$PROMPT_JSON}],
                \"max_tokens\": $max_tok,
                \"temperature\": 0.0
            }" \
            http://localhost:$PORT/v1/chat/completions 2>&1)
        END=$(date +%s%N)
        ELAPSED_MS=$(( (END - START) / 1000000 ))
        PT=$(echo "$RESPONSE" | parse_field prompt_tokens 2>/dev/null || echo 0)
        CT=$(echo "$RESPONSE" | parse_field completion_tokens 2>/dev/null || echo 0)
        TPS="n/a"
        if [ "$CT" -gt 0 ] 2>/dev/null && [ "$ELAPSED_MS" -gt 0 ]; then
            TPS=$(awk "BEGIN {printf \"%.1f\", $CT * 1000.0 / $ELAPSED_MS}")
        fi
        echo "  Trial $trial: ${ELAPSED_MS}ms total, prompt=${PT} tok, completion=${CT} tok, ${TPS} tok/s"
    done
}

bench_one "code-completion" \
"Complete this Python function:

def fibonacci(n):
    if n <= 1:
        return n
    return " \
60

bench_one "summarisation" \
"Summarize this passage in one sentence: The Industrial Revolution was a period of major industrialization that took place during the late 18th and early 19th centuries. It began in Great Britain and quickly spread throughout Western Europe and North America. New manufacturing processes, the rise of factory systems, and the development of steam power transformed economies and societies. The Industrial Revolution marked a major turning point in history." \
50

bench_one "free-form chat" \
"What are three interesting things about octopuses?" \
80

echo
echo "=== Hedge tracker dry-run + prefetch metrics ==="
curl -s -H "Authorization: Bearer $API_KEY_A" \
    "http://localhost:$PORT/api/admin/stats" 2>&1 | \
    python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    ss = d.get('swarm_spec', {})
    print(json.dumps(ss, indent=2))
except Exception as e:
    print('parse error:', e)
"

echo
echo "Done."
