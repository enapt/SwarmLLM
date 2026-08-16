#!/bin/bash
# Long-running soak: drive an ISOLATED node with sustained inference and sample
# the things that leak, so growth over hours is visible rather than inferred.
#
# Why an isolated node and not the live one: a soak makes its target busy for
# hours, and the live node is somebody's. This copies a model into a throwaway
# data dir — never symlinks it, because a test node's auto-manage prunes what it
# finds and has already deleted a live node's shard that way (gotcha, 2026-08-10).
#
# WHAT IT SAMPLES, and why each one is here rather than "memory":
#   rss_kb        daemon resident set. Coarse — see the KV note below.
#   worker_rss_kb the inference worker, which is where the model actually lives.
#   threads/fds   a leak here is a leak of the thing that is hardest to notice.
#   kv_used_mb    KvCacheStore occupancy. **Reason about KV from this, NEVER from
#                 RSS** — the reservation is lazily-faulted zero pages, so a 4-8x
#                 change in reserved bytes moved RSS ~5% and in both directions.
#                 Two conclusions drawn from RSS about this cache were wrong.
#   workers       a respawn is invisible in RSS but means a worker died.
#   log_lines     log growth rate; a run that starts logging per-request is a
#                 regression a memory graph will not show.
#   ok/fail       request outcomes. A soak that silently stops serving is the
#                 failure this is looking for.
#
# Usage:
#   HOURS=6 MODEL=tinyllama-1.1b-chat-v1.0.q4-k-m ./examples/soak_test.sh [binary]
#
# Env: HOURS (default 4), PORT (default 8895), MODEL, CONCURRENCY (default 2),
#      SAMPLE_SECS (default 60), PROMPT_TOKENS_HINT (rough prompt size).
#
# Output: $DATA/soak.csv (one row per sample) + $DATA/soak.log.
# Analyse with: examples/soak_report.sh $DATA/soak.csv

set -u

BINARY="${1:-$(cd "$(dirname "$0")/.." && pwd)/target/debug/swarmllm}"
HOURS="${HOURS:-4}"
PORT="${PORT:-8895}"
MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
CONCURRENCY="${CONCURRENCY:-2}"
SAMPLE_SECS="${SAMPLE_SECS:-60}"
SRC=~/.local/share/swarmllm/models/$MODEL
DATA=/tmp/swarm_soak

if [ ! -f "$SRC/manifest.json" ]; then
    echo "No model at $SRC" >&2
    ls -1 ~/.local/share/swarmllm/models/ 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi
# Metadata alone is not a model. A dir can hold manifest + header with no shard
# files, and a soak against it "works" perfectly: the node registers the model,
# holds nothing, and the scheduler routes every request to whichever swarm peer
# really holds the shards — so the run samples an idle daemon while putting
# hours of load on somebody else's machine, and none of the code under test
# executes. Observed 2026-08-16 (every request went to a LAN peer).
if ! ls "$SRC"/shard_*.bin >/dev/null 2>&1; then
    echo "Model at $SRC has metadata but NO shard files — nothing to soak locally." >&2
    echo "Models with real shards on this machine:" >&2
    for d in ~/.local/share/swarmllm/models/*/; do
        ls "$d"shard_*.bin >/dev/null 2>&1 && basename "$d" | sed 's/^/  /' >&2
    done
    exit 1
fi
[ -x "$BINARY" ] || { echo "No binary at $BINARY" >&2; exit 1; }

# Stop only OUR node, matched on its data dir in /proc/<pid>/environ. Never a
# bare `pkill -x swarmllm` — that killed the user's live node (gotcha #283).
stop_ours() {
    for p in $(pgrep -x swarmllm 2>/dev/null); do
        if tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | grep -q "^SWARMLLM_NODE_DATA_DIR=$DATA$"; then
            kill "$p" 2>/dev/null || true
        fi
    done
}
trap 'echo; echo "stopping..."; STOP=1' INT TERM

stop_ours; sleep 2
rm -rf "$DATA"; mkdir -p "$DATA/models/$MODEL"
cp "$SRC"/* "$DATA/models/$MODEL/" 2>/dev/null

# auto-manage off so the node does not acquire or prune underneath the run —
# either would change what is being measured halfway through.
cat > "$DATA/config.toml" <<CFG
[auto_manage]
enabled = false
prune_enabled = false

[inference]
gpu_layers = 0

[network]
bootstrap_peers = []
disable_default_bootstrap = true
enable_mdns = false
# No bootstrap and no mDNS is NOT isolation on a machine with another node:
# loopback discovery is unconditional (it probes 127.0.0.1 ports and exists
# precisely for the mDNS-off case), so this node WILL connect to a live node
# and, through it, the swarm. The private gossip id is what keeps it deaf to
# the swarm's shard-holder gossip, so the scheduler never learns a remote
# holder to route to. Same recipe as two_node_test.sh.
gossip_network_id = "swarmllm-soak"

[ui]
# Headless run — and the spawned browser opener was a defunct child the worker
# sampler miscounted as a model worker.
open_browser_on_start = false
CFG

export SWARMLLM_NODE_DATA_DIR="$DATA"
# -v so the KV-cache occupancy debug lines are emitted; that counter is the
# only correct way to reason about KV memory and it is not on any API.
# Note this makes log_lines a DEBUG-level growth rate, not a production one.
nohup "$BINARY" run --port "$PORT" --data-dir "$DATA" --config "$DATA/config.toml" -v \
    > "$DATA/soak.log" 2>&1 &
DAEMON_PID=$!

echo "soak: $MODEL on port $PORT for ${HOURS}h, concurrency $CONCURRENCY"
for i in $(seq 1 60); do
    curl -s -m 3 -o /dev/null "http://localhost:$PORT/health" 2>/dev/null && break
    sleep 2
done
KEY=$(cat "$DATA/api_key" 2>/dev/null)
[ -n "$KEY" ] || { echo "no api key — did the node start?" >&2; tail -5 "$DATA/soak.log" >&2; exit 1; }
# One request. Varies the prompt by index so a cache cannot make later requests
# artificially cheap and hide a leak behind a fast path.
fire() {
    local n=$1
    local body
    body=$(printf '{"model":"%s","messages":[{"role":"user","content":"Question %d: name one fact about the number %d."}],"max_tokens":24}' "$MODEL" "$n" "$n")
    curl -s -m 180 -o /dev/null -w "%{http_code}" \
        -X POST "http://localhost:$PORT/v1/chat/completions" \
        -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
        -d "$body" 2>/dev/null
}

sample() {
    local now elapsed rss wrss threads fds workers kv lines wpid
    now=$(date +%s); elapsed=$((now-START))
    rss=$(awk '/VmRSS/{print $2}' /proc/$DAEMON_PID/status 2>/dev/null)
    threads=$(awk '/Threads/{print $2}' /proc/$DAEMON_PID/status 2>/dev/null)
    fds=$(ls /proc/$DAEMON_PID/fd 2>/dev/null | wc -l)
    # Workers are children of the daemon carrying "model-worker" in their argv.
    # The ppid match alone is not enough — the daemon has other children (a
    # defunct browser-opener sat there for a whole run and was counted as a
    # worker, RSS 0). The ppid keeps it from picking up another node's worker
    # on the same machine; the argv match keeps it to actual workers.
    workers=0; wrss=0
    for wpid in $(pgrep -P $DAEMON_PID -f "model-worker" 2>/dev/null); do
        workers=$((workers+1))
        # A worker can exit between the pgrep and the read, and then awk prints
        # NOTHING while still exiting 0 — so `|| echo 0` never fires and the
        # arithmetic gets an empty operand. Default after the fact instead.
        local one
        one=$(awk '/VmRSS/{print $2}' "/proc/$wpid/status" 2>/dev/null)
        wrss=$((wrss + ${one:-0}))
    done
    # KV occupancy is not on any API surface — it is emitted as a debug line by
    # whichever store is the meaningful one (see the comments at both sites in
    # model_worker.rs and router/mod.rs). A single node answering its own
    # requests reports through the WORKER line; the distributed-path line stays
    # empty there. So the node runs at -v and we read the last such line.
    #
    # Deliberately not process RSS: the reservation is lazily-faulted zero
    # pages, so RSS moved ~5% for a 4-8x change in reserved bytes, in both
    # directions. rss_kb is still recorded, but it answers a different question.
    kv=$(grep "KV-cache occupancy" "$DATA/soak.log" 2>/dev/null | tail -1 \
         | grep -oE "used_mb=[0-9]+" | cut -d= -f2)
    lines=$(wc -l < "$DATA/soak.log" 2>/dev/null)
    echo "$(date -Iseconds),$elapsed,${rss:-},${wrss:-},${threads:-},${fds:-},$workers,${kv:-},${lines:-},$OK,$FAIL" >> "$DATA/soak.csv"
    printf "  %5ss  rss=%sMB worker=%sMB thr=%s fd=%s workers=%s kv=%sMB ok=%s fail=%s\n" \
        "$elapsed" "$((${rss:-0}/1024))" "$((${wrss:-0}/1024))" "${threads:-?}" "${fds:-?}" "$workers" "${kv:-?}" "$OK" "$FAIL"
}

OK=0; FAIL=0; STOP=0

# Warm-up: wait until the model has loaded and answers a real request, so the
# counters and the clock measure the soak rather than the load, and so the
# guard below has a served request to judge.
echo "waiting for the model to load and answer (a big model takes a minute or two)..."
warm=""
for _ in $(seq 1 100); do
    [ "$STOP" -ne 0 ] && break
    rc=$(fire 0)
    [ "$rc" = "200" ] && { warm=1; break; }
    sleep 3
done
if [ "$STOP" -ne 0 ]; then echo "stopped during warm-up"; stop_ours; exit 130; fi
[ -n "$warm" ] || { echo "model never answered 200 during warm-up" >&2; tail -5 "$DATA/soak.log" >&2; stop_ours; exit 1; }

# The failure this guards against is silent and total: a request served by a
# peer returns 200, samples cleanly, and exercises none of the code under
# test — while loading a machine that is somebody's. A Pipeline segment line
# naming a foreign node is that failure happening. A locally-served request
# takes the split fast path and assembles no pipeline at all, so on a healthy
# soak this grep matches nothing.
LOCAL_ID=$(curl -s -m 5 -H "Authorization: Bearer $KEY" "http://localhost:$PORT/v1/status" \
           | grep -oE '"node_id" *: *"[0-9a-f]{16}"' | grep -oE '[0-9a-f]{16}')
offtarget_lines() {
    grep "Pipeline segment" "$DATA/soak.log" 2>/dev/null | grep -v "node=${LOCAL_ID:-__unknown__}"
}
if [ -n "$(offtarget_lines)" ]; then
    echo "OFF-TARGET: requests are being served by a peer, not this node:" >&2
    offtarget_lines | tail -3 >&2
    stop_ours; exit 1
fi

END=$(( $(date +%s) + HOURS*3600 ))
echo "ts,elapsed_s,rss_kb,worker_rss_kb,threads,fds,workers,kv_used_mb,log_lines,ok,fail" > "$DATA/soak.csv"
START=$(date +%s)
sample || echo '  (first sample failed, continuing)' >&2
NEXT_SAMPLE=$(( $(date +%s) + SAMPLE_SECS ))
N=0
while [ "$(date +%s)" -lt "$END" ] && [ "$STOP" -eq 0 ]; do
    pids=""
    for _ in $(seq 1 "$CONCURRENCY"); do
        N=$((N+1)); fire "$N" > "$DATA/.rc.$N" & pids="$pids $!"
    done
    for p in $pids; do wait "$p" 2>/dev/null; done
    for f in "$DATA"/.rc.*; do
        [ -f "$f" ] || continue
        if [ "$(cat "$f")" = "200" ]; then OK=$((OK+1)); else FAIL=$((FAIL+1)); fi
        rm -f "$f"
    done
    if [ "$(date +%s)" -ge "$NEXT_SAMPLE" ]; then
        sample || echo "  (sample failed, continuing)" >&2
        # Re-check each sample: a mid-run failover to a peer is the same
        # invalidation as starting off-target, just later.
        if [ -n "$(offtarget_lines)" ]; then
            echo "OFF-TARGET: a peer started serving these requests mid-run:" >&2
            offtarget_lines | tail -3 >&2
            STOP=2
        fi
        NEXT_SAMPLE=$(( $(date +%s) + SAMPLE_SECS ))
    fi
done
if [ "$STOP" = "2" ]; then
    echo "loop ended: ABORTED — requests were being served by a peer (see above)" >&2
elif [ "$STOP" -ne 0 ]; then
    echo "loop ended: received a stop signal"
elif [ "$(date +%s)" -ge "$END" ]; then
    echo "loop ended: reached the ${HOURS}h deadline"
else
    echo "loop ended: UNEXPECTED — neither deadline nor signal" >&2
fi

sample
echo "soak done: $OK ok, $FAIL failed over $(( ($(date +%s)-START)/60 )) min"
echo "csv: $DATA/soak.csv"
stop_ours
[ "$STOP" = "2" ] && exit 1
exit 0
