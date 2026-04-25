#!/bin/bash
# Run both the old (pipeline-time) and new (TTFB) streaming bench against
# the daemon at $PORT, and dump a four-line summary.
set -u
PORT="${PORT:-8830}"
API_KEY="$(cat /tmp/resp_final/api_key)"
MODEL="tinyllama-1.1b-chat-v1.0.q4-k-m"
ITERS="${ITERS:-5}"
LABEL="${LABEL:-unknown}"

# Warm up
curl -s -o /dev/null --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/chat/completions -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0.1}' > /dev/null
curl -s -o /dev/null --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":3,"temperature":0.1,"store":false}' > /dev/null

# Old methodology — `time -p` of (curl|grep -m1 "data:") subshell.
old_metric() {
    local endpoint="$1"
    local payload="$2"
    local samples=()
    for i in $(seq 1 $ITERS); do
        local t=$( { time -p (curl -s -N -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" --max-time 30 http://localhost:$PORT$endpoint -d "$payload" 2>/dev/null | grep -m1 "^data: " > /dev/null) 2>&1; } 2>&1 | grep "^real" | awk '{print $2}')
        samples+=("$t")
    done
    python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(round(statistics.median(s)*1000, 1))
"
}

# New methodology — curl --time_starttransfer (TTFB).
new_metric() {
    local endpoint="$1"
    local payload="$2"
    local samples=()
    for i in $(seq 1 $ITERS); do
        local t=$(curl -s -N -o /dev/null -w "%{time_starttransfer}" --max-time 30 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT$endpoint -d "$payload")
        samples+=("$t")
    done
    python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(round(statistics.median(s)*1000, 1))
"
}

CHAT_PAYLOAD='{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0.1,"stream":true}'
RESP_PAYLOAD='{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1,"stream":true,"store":false}'

OLD_CHAT=$(old_metric "/v1/chat/completions" "$CHAT_PAYLOAD")
OLD_RESP=$(old_metric "/v1/responses" "$RESP_PAYLOAD")
NEW_CHAT=$(new_metric "/v1/chat/completions" "$CHAT_PAYLOAD")
NEW_RESP=$(new_metric "/v1/responses" "$RESP_PAYLOAD")

echo "[$LABEL] old(pipeline-time):  chat=${OLD_CHAT}ms  resp=${OLD_RESP}ms  gap=$(python3 -c "print(round(abs($OLD_RESP - $OLD_CHAT), 1))")ms"
echo "[$LABEL] new(TTFB):          chat=${NEW_CHAT}ms  resp=${NEW_RESP}ms  gap=$(python3 -c "print(round(abs($NEW_RESP - $NEW_CHAT), 1))")ms"
