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

END=$(( $(date +%s) + HOURS*3600 ))
OK=0; FAIL=0; STOP=0
echo "ts,elapsed_s,rss_kb,worker_rss_kb,threads,fds,workers,kv_used_mb,log_lines,ok,fail" > "$DATA/soak.csv"
START=$(date +%s)

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
    # Workers are children of the daemon; match by ppid, not by name, so this
    # cannot pick up another node's worker on the same machine.
    workers=0; wrss=0
    for wpid in $(pgrep -P $DAEMON_PID 2>/dev/null); do
        workers=$((workers+1))
        wrss=$((wrss + $(awk '/VmRSS/{print $2}' /proc/$wpid/status 2>/dev/null || echo 0)))
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

sample
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
        sample; NEXT_SAMPLE=$(( $(date +%s) + SAMPLE_SECS ))
    fi
done

sample
echo "soak done: $OK ok, $FAIL failed over $(( ($(date +%s)-START)/60 )) min"
echo "csv: $DATA/soak.csv"
stop_ours
