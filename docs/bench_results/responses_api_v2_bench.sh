#!/bin/bash
# v2 bench — supersedes the v1 bench script for the V1 streaming first-byte
# checkpoint. The v1 version measured `(curl | grep -m1 'data:') time` which
# includes pipeline tear-down latency and gave misleading results post-V1.
# Here we use curl's built-in `time_starttransfer` (TTFB) which measures
# the moment the server's HTTP response head + first body byte arrive — that
# is what the V1 fix targets.

set -u
PORT=8830
API_KEY=$(cat /tmp/resp_final/api_key)
MODEL="tinyllama-1.1b-chat-v1.0.q4-k-m"
ITERS=10

# Warm up
echo "warmup..."
curl -s -o /dev/null --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/chat/completions -d '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":3,"temperature":0.1}' >/dev/null
curl -s -o /dev/null --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":3,"temperature":0.1,"store":false}' >/dev/null

ttfb_total() {
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
print(f'{\"$3\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  p95={sorted(s)[int(len(s)*0.95)-1]*1000:7.1f}ms  (n={len(s)})')
"
}

end_to_end() {
    local endpoint="$1"
    local payload="$2"
    local samples=()
    for i in $(seq 1 $ITERS); do
        local t=$(curl -s -o /dev/null -w "%{time_total}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT$endpoint -d "$payload")
        samples+=("$t")
    done
    python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"$3\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  p95={sorted(s)[int(len(s)*0.95)-1]*1000:7.1f}ms  (n={len(s)})')
"
}

echo ""
echo "=== Streaming first-byte (TTFB — what V1 targets) ==="
ttfb_total "/v1/chat/completions" '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0.1,"stream":true}' "chat_completions stream"
ttfb_total "/v1/responses" '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1,"stream":true,"store":false}' "responses stream"

echo ""
echo "=== Non-streaming end-to-end ==="
end_to_end "/v1/chat/completions" '{"model":"'$MODEL'","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"temperature":0.1}' "chat_completions baseline"
end_to_end "/v1/responses" '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1,"store":false}' "responses store=false"
end_to_end "/v1/responses" '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1}' "responses store=true (default)"
end_to_end "/v1/responses" '{"model":"'$MODEL'","input":"hi","instructions":"Be terse.","max_output_tokens":5,"temperature":0.1,"store":false}' "responses + instructions"
end_to_end "/v1/responses" '{"model":"'$MODEL'","input":[{"type":"message","role":"user","content":"hi"}],"max_output_tokens":5,"temperature":0.1,"store":false}' "responses + array input"

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
print(f'{\"POST background=true (queued return)\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"

echo ""
echo "=== V8 background+stream 202 return latency ==="
samples=()
for i in $(seq 1 10); do
    t=$(curl -s -o /dev/null -w "%{time_total}" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","background":true,"stream":true,"max_output_tokens":5}')
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"POST background+stream (202 + Location)\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"

echo ""
echo "=== Storage overhead ==="
samples=()
for i in $(seq 1 20); do
    curl -s -o /tmp/resp_final/bench_mk.json -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"x","max_output_tokens":1,"store":true}' >/dev/null
    MKID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/bench_mk.json'))['id'])")
    t=$(curl -s -o /dev/null -w "%{time_total}" -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$MKID)
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"GET /v1/responses/:id (cache hit)\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"

samples=()
for i in $(seq 1 20); do
    curl -s -o /tmp/resp_final/bench_mk.json -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"x","max_output_tokens":1,"store":true}' >/dev/null
    MKID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/bench_mk.json'))['id'])")
    t=$(curl -s -o /dev/null -w "%{time_total}" -X DELETE -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$MKID)
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"DELETE /v1/responses/:id\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"

echo ""
echo "=== V4 input_items endpoint ==="
curl -s -o /tmp/resp_final/bench_v4.json -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":[{"type":"message","role":"user","content":"a"},{"type":"message","role":"user","content":"b"},{"type":"message","role":"user","content":"c"}],"max_output_tokens":1,"store":true}' >/dev/null
V4ID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/bench_v4.json'))['id'])")
samples=()
for i in $(seq 1 20); do
    t=$(curl -s -o /dev/null -w "%{time_total}" -H "Authorization: Bearer $API_KEY" "http://localhost:$PORT/v1/responses/$V4ID/input_items?limit=20")
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"GET /v1/responses/:id/input_items\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"

echo ""
echo "=== V6 admin /api/admin/responses ==="
samples=()
for i in $(seq 1 20); do
    t=$(curl -s -o /dev/null -w "%{time_total}" -H "Authorization: Bearer $API_KEY" "http://localhost:$PORT/api/admin/responses?limit=100")
    samples+=("$t")
done
python3 -c "
import statistics
s = [$(IFS=,; echo "${samples[*]}")]
print(f'{\"GET /api/admin/responses?limit=100\":<46} median={statistics.median(s)*1000:7.1f}ms  mean={statistics.mean(s)*1000:7.1f}ms  (n={len(s)})')
"
