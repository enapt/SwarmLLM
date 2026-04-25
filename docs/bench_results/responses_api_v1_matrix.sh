#!/bin/bash
# Comprehensive end-to-end matrix for /v1/responses (M1-M9).
# Exits 0 if everything passes, non-zero on any mismatch.
set -u
PORT=8830
API_KEY=$(cat /tmp/resp_final/api_key)
MODEL="tinyllama-1.1b-chat-v1.0.q4-k-m"
PASS=0
FAIL=0

check() {
    local name="$1"
    local expected="$2"
    local actual="$3"
    if [ "$actual" = "$expected" ]; then
        echo "PASS | $name"
        PASS=$((PASS+1))
    else
        echo "FAIL | $name: expected=$expected got=$actual"
        FAIL=$((FAIL+1))
    fi
}

h() { echo "--- $1 ---"; }

# ----- M2: built-in tool rejection -----
h "M2: web_search → 400"
CODE=$(curl -s -o /tmp/resp_final/m2_1.json -w "%{http_code}" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"x","tools":[{"type":"web_search"}]}')
check "M2.web_search.status" "400" "$CODE"
MSG=$(python3 -c "import json;print('web_search' in json.load(open('/tmp/resp_final/m2_1.json'))['error']['message'])")
check "M2.web_search.msg_contains" "True" "$MSG"

for t in file_search computer_use_preview code_interpreter image_generation mcp custom; do
    CODE=$(curl -s -o /tmp/resp_final/m2_$t.json -w "%{http_code}" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"x","tools":[{"type":"'$t'"}]}')
    check "M2.$t.status" "400" "$CODE"
done

# ----- M3: plain text local inference -----
h "M3: plain string input"
CODE=$(curl -s -o /tmp/resp_final/m3_1.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":10,"temperature":0.1}')
check "M3.string.status" "200" "$CODE"
OBJ=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m3_1.json'))['object'])")
check "M3.string.object" "response" "$OBJ"
HAS_OUTPUT=$(python3 -c "import json;d=json.load(open('/tmp/resp_final/m3_1.json'));print(bool(d.get('output_text')) and d['output'][0]['type']=='message')")
check "M3.string.output" "True" "$HAS_OUTPUT"

h "M3: array input with roles"
CODE=$(curl -s -o /tmp/resp_final/m3_2.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":[{"type":"message","role":"system","content":"Be terse."},{"type":"message","role":"user","content":"hi"}],"max_output_tokens":10,"temperature":0.1}')
check "M3.array.status" "200" "$CODE"

h "M3: instructions mapped to system message"
CODE=$(curl -s -o /tmp/resp_final/m3_3.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","instructions":"Be very brief.","max_output_tokens":10,"temperature":0.1}')
check "M3.instructions.status" "200" "$CODE"

# ----- M4: function tools -----
h "M4: function tool accepted"
CODE=$(curl -s -o /tmp/resp_final/m4_1.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":10,"tools":[{"type":"function","name":"get_x","parameters":{"type":"object"}}]}')
check "M4.function.status" "200" "$CODE"

h "M4: tool_choice auto/object forms"
for tc in '"auto"' '"none"' '"required"' '{"type":"function","name":"get_x"}'; do
    CODE=$(curl -s -o /tmp/resp_final/m4_tc.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"tools":[{"type":"function","name":"get_x","parameters":{"type":"object"}}],"tool_choice":'"$tc"'}')
    check "M4.tool_choice.$(echo $tc | tr -d '"{}:,')" "200" "$CODE"
done

h "M4: function_call + function_call_output input items"
CODE=$(curl -s -o /tmp/resp_final/m4_fc.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":[{"type":"message","role":"user","content":"weather?"},{"type":"function_call","call_id":"c1","name":"get_weather","arguments":"{\"city\":\"NYC\"}"},{"type":"function_call_output","call_id":"c1","output":"{\"temp\":72}"}],"max_output_tokens":5,"tools":[{"type":"function","name":"get_weather","parameters":{"type":"object"}}]}')
check "M4.function_chain.status" "200" "$CODE"

# ----- M5: cloud routing -----
# Post-V3 (responses_api_v2) update: claude-* no longer 400s with "use
# /v1/messages". The Responses → Anthropic Messages bridge translates and
# forwards. With claude-subscription feature on we expect 200 (subprocess);
# with no Anthropic credentials at all the upstream call should error 4xx/5xx
# but the local request is well-formed.
h "M5: claude model is accepted (V3: translates to Anthropic Messages)"
CODE=$(curl -s -o /tmp/resp_final/m5_1.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"claude-opus-4-7","input":"hi","max_output_tokens":3}')
if [ "$CODE" = "200" ] || [ "$CODE" = "401" ] || [ "$CODE" = "402" ] || [ "$CODE" = "403" ] || [ "$CODE" = "404" ] || [ "$CODE" = "504" ]; then
    echo "PASS | M5.claude.status=$CODE (200 if subscription/key configured, upstream 4xx/5xx otherwise — no longer 400)"
    PASS=$((PASS+1))
else
    echo "FAIL | M5.claude.status=$CODE (expected 200 or upstream 4xx/5xx after V3)"
    FAIL=$((FAIL+1))
fi

# ----- M6: streaming SSE -----
h "M6: stream=true → 23+ events"
curl -s -N -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" --max-time 30 http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"stream":true}' > /tmp/resp_final/m6_stream.sse
EVENT_COUNT=$(grep -c '"type":"response\.' /tmp/resp_final/m6_stream.sse || echo 0)
# Expect at least: created + in_progress + item.added + part.added + 1+ deltas + text.done + part.done + item.done + terminal = 9+
if [ "$EVENT_COUNT" -ge 9 ]; then
    echo "PASS | M6.stream.count=$EVENT_COUNT (>= 9)"
    PASS=$((PASS+1))
else
    echo "FAIL | M6.stream.count=$EVENT_COUNT (expected >= 9)"
    FAIL=$((FAIL+1))
fi
HAS_CREATED=$(grep -c '"type":"response.created"' /tmp/resp_final/m6_stream.sse || echo 0)
check "M6.stream.created" "1" "$HAS_CREATED"
HAS_TERMINAL=$(grep -cE '"type":"response\.(completed|incomplete|failed)"' /tmp/resp_final/m6_stream.sse || echo 0)
check "M6.stream.terminal" "1" "$HAS_TERMINAL"
# Monotonic sequence
MONOTONIC=$(python3 -c "
import re
seqs = [int(m) for m in re.findall(r'\"sequence_number\":(\d+)', open('/tmp/resp_final/m6_stream.sse').read())]
print(seqs == list(range(len(seqs))))")
check "M6.stream.monotonic" "True" "$MONOTONIC"

# ----- M7: store + retrieve + delete -----
h "M7: create → GET → DELETE round-trip"
curl -s -o /tmp/resp_final/m7_1.json --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1}' > /dev/null
RID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m7_1.json'))['id'])")
CODE=$(curl -s -o /tmp/resp_final/m7_get.json -w "%{http_code}" -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$RID)
check "M7.get.status" "200" "$CODE"
GOT_ID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m7_get.json'))['id'])")
check "M7.get.id_match" "$RID" "$GOT_ID"

CODE=$(curl -s -o /tmp/resp_final/m7_del.json -w "%{http_code}" -X DELETE -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$RID)
check "M7.delete.status" "200" "$CODE"
DELETED=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m7_del.json'))['deleted'])")
check "M7.delete.flag" "True" "$DELETED"

CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$RID)
check "M7.get_after_delete.status" "400" "$CODE"

h "M7: store=false does NOT persist"
curl -s -o /tmp/resp_final/m7_sf.json --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":3,"store":false}' > /dev/null
RID2=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m7_sf.json'))['id'])")
CODE=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$RID2)
check "M7.store_false.get_status" "400" "$CODE"

# ----- M8: previous_response_id chaining -----
h "M8: previous_response_id chaining"
curl -s -o /tmp/resp_final/m8_1.json --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","max_output_tokens":5,"temperature":0.1}' > /dev/null
PID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m8_1.json'))['id'])")

CODE=$(curl -s -o /tmp/resp_final/m8_2.json -w "%{http_code}" --max-time 60 -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"how are you","previous_response_id":"'$PID'","max_output_tokens":5,"temperature":0.1}')
check "M8.chain.status" "200" "$CODE"
ECHO_PID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m8_2.json'))['previous_response_id'])")
check "M8.chain.prev_id_echoed" "$PID" "$ECHO_PID"

CODE=$(curl -s -o /tmp/resp_final/m8_err.json -w "%{http_code}" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","previous_response_id":"resp_missing"}')
check "M8.invalid_prev.status" "400" "$CODE"

# ----- M9: background + cancel -----
h "M9: background + cancel"
curl -s -o /tmp/resp_final/m9_1.json -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"write a long essay about compilers","background":true,"max_output_tokens":300,"temperature":0.1}' > /dev/null
BGID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m9_1.json'))['id'])")
STATUS=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m9_1.json'))['status'])")
check "M9.bg.initial_status" "queued" "$STATUS"

sleep 1
CODE=$(curl -s -o /tmp/resp_final/m9_cancel.json -w "%{http_code}" -X POST -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$BGID/cancel)
check "M9.cancel.status" "200" "$CODE"
ST=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m9_cancel.json'))['status'])")
check "M9.cancel.state" "cancelled" "$ST"

# wait for background task to finish, confirm cancel-wins
sleep 15
FINAL=$(curl -s -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$BGID | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
check "M9.cancel.persists_after_task_finish" "cancelled" "$FINAL"

# Short background to completion
curl -s -o /tmp/resp_final/m9_short.json -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" http://localhost:$PORT/v1/responses -d '{"model":"'$MODEL'","input":"hi","background":true,"max_output_tokens":5,"temperature":0.1}' > /dev/null
SID=$(python3 -c "import json;print(json.load(open('/tmp/resp_final/m9_short.json'))['id'])")
sleep 15
FINAL_SHORT=$(curl -s -H "Authorization: Bearer $API_KEY" http://localhost:$PORT/v1/responses/$SID | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['status'])")
# Either completed or incomplete (depending on finish_reason) is fine
if [ "$FINAL_SHORT" = "completed" ] || [ "$FINAL_SHORT" = "incomplete" ]; then
    echo "PASS | M9.bg_completion.final=$FINAL_SHORT"
    PASS=$((PASS+1))
else
    echo "FAIL | M9.bg_completion.final=$FINAL_SHORT (expected completed/incomplete)"
    FAIL=$((FAIL+1))
fi

echo ""
echo "============================"
echo "TOTAL: $PASS pass, $FAIL fail"
echo "============================"
exit $FAIL
