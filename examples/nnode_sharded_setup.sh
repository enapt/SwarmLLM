#!/bin/bash
# N-node sharded cluster — forces a pipeline with N distinct segment holders.
#
# Generalises examples/3node_sharded_setup.sh, which splits a model across
# exactly two holders. Use this when the question is "does an N-segment chain
# assemble and produce a correct answer", not just "does a split work at all".
#
# Layout (HOLDERS=n):
#   Node A  (BASE_PORT):     manifest + gguf_header only, ZERO shards.
#                            Pure coordinator — every request it serves must be
#                            assembled from the holders below.
#   Nodes 1..n (BASE_PORT+i): one contiguous, DISJOINT slice of the shard list.
#
# A request to A therefore forces an n-segment pipeline across n distinct nodes.
#
# KNOWN LIMITATION — read before believing a failure here.
#
# Every shard has exactly ONE holder, so there is no standby anywhere. That is
# the same-host, zero-redundancy case docs/FUTURE_WORK.md § "Connection churn on
# multi-interface hosts" documents as NOT covered: one loopback host advertising
# several interfaces (WSL2's NAT gateway, link-local, Docker bridge, LAN) can
# form a stale connection, and with no standby the pipeline fails with
# "Segment N failed with no standby available".
#
# So a failure HERE is not evidence that N-node inference is broken. Confirm on
# separate hosts before concluding that. Redundancy cannot be added without
# defeating the test: give any shard a second holder and the scheduler correctly
# collapses the chain to fewer segments, because minimising hops is its job.
#
# Usage:
#   HOLDERS=4 SWARM_BENCH_MODEL=llama-3.2-3b-instruct-q4-k-m ./examples/nnode_sharded_setup.sh
#
# Env: HOLDERS (default 3), BASE_PORT (default 8890), BINARY, SWARM_BENCH_MODEL,
#      GPU_LAYERS (default 0 = CPU; these are throwaway nodes and must not
#      contend for VRAM with a real node on the same box).

set -e

HOLDERS="${HOLDERS:-3}"
BASE_PORT="${BASE_PORT:-8890}"
GPU_LAYERS="${GPU_LAYERS:-0}"
MODEL_NAME="${SWARM_BENCH_MODEL:-llama-3.2-3b-instruct-q4-k-m}"
SRC_MODEL=~/.local/share/swarmllm/models/$MODEL_NAME
BINARY="${BINARY:-$(cd "$(dirname "$0")/.." && pwd)/target/release/swarmllm}"
PREFIX=/tmp/swarm_nnode

if [ ! -f "$SRC_MODEL/manifest.json" ]; then
    echo "No model at $SRC_MODEL" >&2
    ls -1 ~/.local/share/swarmllm/models/ 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi

SHARDS=($(ls -1 "$SRC_MODEL"/shard_*.bin 2>/dev/null | sort))
if [ "${#SHARDS[@]}" -lt "$HOLDERS" ]; then
    echo "Need >= HOLDERS shards to give each holder one; model has ${#SHARDS[@]}, HOLDERS=$HOLDERS" >&2
    exit 1
fi

# Stop only the nodes THIS script owns — by port. A broad `pkill swarmllm`
# takes down any other node on the machine, including a production one
# (gotcha #283).
stop_by_port() {
    for p in $(pgrep -x swarmllm 2>/dev/null); do
        if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE -- "(--port|-p) $1( |$)"; then
            kill "$p" 2>/dev/null || true
        fi
    done
}
for i in $(seq 0 "$HOLDERS"); do stop_by_port $((BASE_PORT + i)); done
sleep 2

rm -rf ${PREFIX}_*
for i in $(seq 0 "$HOLDERS"); do
    mkdir -p "${PREFIX}_$i/models/$MODEL_NAME"
    cp "$SRC_MODEL/manifest.json"    "${PREFIX}_$i/models/$MODEL_NAME/"
    cp "$SRC_MODEL/gguf_header.bin"  "${PREFIX}_$i/models/$MODEL_NAME/"
    cp "$SRC_MODEL/hf_source.json"   "${PREFIX}_$i/models/$MODEL_NAME/" 2>/dev/null || true
    # Weight-tied models keep the LM head in shard 0; a node serving the LAST
    # segment needs it even though it does not hold that shard (gotcha #178).
    cp "$SRC_MODEL/tied_output_weight.bin" "${PREFIX}_$i/models/$MODEL_NAME/" 2>/dev/null || true
    cat > "${PREFIX}_$i/config.toml" <<CFG
[auto_manage]
enabled = false
prune_enabled = false

[inference]
gpu_layers = $GPU_LAYERS

[network]
bootstrap_peers = []
disable_default_bootstrap = true
CFG
done

# Deal the shards round-robin-free: contiguous disjoint slices, holder i gets
# slice i. Contiguous matters — a pipeline segment is a layer RANGE, so a
# holder with non-adjacent shards cannot serve one segment.
N=${#SHARDS[@]}
PER=$(( (N + HOLDERS - 1) / HOLDERS ))
for idx in "${!SHARDS[@]}"; do
    holder=$(( idx / PER + 1 ))
    [ "$holder" -gt "$HOLDERS" ] && holder=$HOLDERS
    cp "${SHARDS[$idx]}" "${PREFIX}_$holder/models/$MODEL_NAME/"
done

echo "Model $MODEL_NAME: $N shards across $HOLDERS holders (coordinator holds none)"
for i in $(seq 0 "$HOLDERS"); do
    role=$([ "$i" = 0 ] && echo "coordinator" || echo "holder $i")
    got=$(ls -1 "${PREFIX}_$i/models/$MODEL_NAME"/shard_*.bin 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
    echo "  port $((BASE_PORT + i))  $role: ${got:-<none>}"
done

for i in $(seq 0 "$HOLDERS"); do
    port=$((BASE_PORT + i))
    nohup "$BINARY" run --port "$port" --data-dir "${PREFIX}_$i" \
        --config "${PREFIX}_$i/config.toml" > "${PREFIX}_$i/log.txt" 2>&1 &
done

echo "Waiting 20s for mDNS discovery + shard announcements..."
sleep 20
echo
echo "Coordinator: port $BASE_PORT  api_key=$(cat ${PREFIX}_0/api_key 2>/dev/null)"
echo "Stop with: for i in \$(seq 0 $HOLDERS); do for p in \$(pgrep -x swarmllm); do tr '\\0' ' ' < /proc/\$p/cmdline | grep -q -- \"-p \$((${BASE_PORT}+i))\" && kill \$p; done; done"
