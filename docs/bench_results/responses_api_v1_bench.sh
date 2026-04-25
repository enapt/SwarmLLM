#!/bin/bash
# Micro-benchmarks for /v1/responses vs /v1/chat/completions.
#
# We measure wall-clock end-to-end HTTP latency for small max_output_tokens
# so the translation overhead shows up against the inference floor.
set -u
PORT=8830
API_KEY=$(cat /tmp/resp_final/api_key)
MODEL="tinyllama-1.1b-chat-v1.0.q4-k-m"
ITERS=10

# Warm up the worker subprocess so the first-token latency of a cold
# inference session doesn't dominate the first sample.
echo "warmup..."
curl -s -o /dev/null -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" --max-time 60 http://localhost:$PORT/v1/chat/completions -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0.1}' > /dev/null
curl -s -o /dev/null -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" --max-time 60 http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":3,"temperature":0.1,"store":false}' > /dev/null

measure() {
    local label="$1"
    local endpoint="$2"
    local payload="$3"
    local -a samples=()
    for i in $(seq 1 $ITERS); do
        local t=$(curl -s -o /dev/null -w "%{time_total}" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" --max-time 60 http://localhost:$PORT$endpoint -d "$payload")
        samples+=("$t")
    done
    python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"$label\":<40} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  p95={sorted(s)[int(len(s)*0.95)-1]*1000:7.1f}ms  (n={len(s)})')
"
}

echo ""
echo "=== Non-streaming latency (same 5-token generation) ==="
measure "chat_completions (baseline)" "/v1/chat/completions" '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0.1}'
measure "responses (store=false)" "/v1/responses" '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1,"store":false}'
measure "responses (store=true, default)" "/v1/responses" '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1}'
measure "responses + instructions" "/v1/responses" '{"model":"'$MODEL'","input":"hi","instructions":"Be terse.","max_output_tokens":5,"temperature":0.1,"store":false}'
measure "responses + array input" "/v1/responses" '{"model":"'$MODEL'","input":[{"type":"message","role":"user","content":"hi"}],"max_output_tokens":5,"temperature":0.1,"store":false}'
measure "responses + function tool" "/v1/responses" '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1,"store":false,"tools":[{"type":"function","name":"f","parameters":{"type":"object"}}]}'

echo ""
echo "=== Streaming first-token latency (time to first data: line) ==="
for label_pair in "chat:chat_completions:/v1/chat/completions:messages=[{role:user,content:hi}]" "resp:responses:/v1/responses:input=hi"; do
    IFS=':' read -r key label endpoint shape <<< "$label_pair"
    if [ "$key" = "chat" ]; then
        payload='{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0.1,"stream":true}'
    else
        payload='{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1,"stream":true,"store":false}'
    fi
    samples=()
    for i in $(seq 1 5); do
        # time to first non-event-line data bytes
        t=$( { time -p (curl -s -N -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" --max-time 30 http://localhost:$PORT$endpoint -d "$payload" 2>/dev/null | grep -m1 "^data: " > /dev/null) 2>&1; } 2>&1 | grep "^real" | awk '{print $2}')
        samples+=("$t")
    done
    python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"$label (first data: line)\":<40} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"
done

echo ""
echo "=== M7 storage overhead (GET + DELETE) ==="
# Create one response, then measure GET + DELETE latency.
curl -s -o /tmp/resp_final/bench_create.json --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":3,"temperature":0.1}' > /dev/null
BENCH_ID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/bench_create.json'))['id'])")

samples=()
for i in $(seq 1 20); do
    t=$(curl -s -o /dev/null -w "%{time_total}" -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$BENCH_ID)
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"GET /v1/responses/:id (cache hit)\":<40} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"

samples=()
for i in $(seq 1 20); do
    curl -s -o /tmp/resp_final/bench_mk.json -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"x","max_output_tokens":1,"store":true}' > /dev/null
    MKID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/bench_mk.json'))['id'])")
    t=$(curl -s -o /dev/null -w "%{time_total}" -X DELETE -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$MKID)
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"DELETE /v1/responses/:id\":<40} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"

echo ""
echo "=== M9 background return latency (should be near-zero) ==="
samples=()
for i in $(seq 1 10); do
    t=$(curl -s -o /dev/null -w "%{time_total}" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","background":true,"max_output_tokens":300}')
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"POST background=true (queued return)\":<40} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"
