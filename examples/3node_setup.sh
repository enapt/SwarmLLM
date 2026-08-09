#!/bin/bash
# SWARM-SPEC 3-node local cluster setup for benchmarking.

set -e

SRC_MODEL=~/.local/share/swarmllm/models/tinyllama-1.1b-chat-v1.0.q4-k-m
# Resolve binary relative to this script (repo_root/target/release/swarmllm)
# so the cluster works regardless of clone path. Override BINARY=... to
# point at a different build (e.g. CUDA, llama feature).
BINARY="${BINARY:-$(cd "$(dirname "$0")/.." && pwd)/target/release/swarmllm}"

# Ports for the throwaway cluster. Deliberately NOT 8800: that is the default
# port a real node listens on, and this script used to stop and replace whatever
# was there. Override if these collide with something.
BENCH_PORT_A="${BENCH_PORT_A:-8890}"
BENCH_PORT_B="${BENCH_PORT_B:-8891}"
BENCH_PORT_C="${BENCH_PORT_C:-8892}"

# Stop only the nodes THIS script owns — the ones on its own ports and data
# directories. A broad `killall swarmllm` / `pkill -f swarmllm` takes down any
# other node on the machine, including a production one: that happened on
# 2026-08-09 and cost a live node serving the swarm (gotcha #283). Development
# machines run more than one instance, always.
stop_bench_node() {
    local port="$1"
    for p in $(pgrep -x swarmllm 2>/dev/null; pgrep -f '[s]warmllm-[a-z0-9_.-]* run' 2>/dev/null); do
        if tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -qE -- "(--port|-p) $port( |$)"; then
            kill "$p" 2>/dev/null || true
        fi
    done
}

stop_bench_node "$BENCH_PORT_A"
stop_bench_node "$BENCH_PORT_B"
stop_bench_node "$BENCH_PORT_C"
sleep 1

for label in a b c; do
    DIR=/tmp/swarm_bench_$label
    rm -rf "$DIR"
    mkdir -p "$DIR/models"
    cp -r "$SRC_MODEL" "$DIR/models/"
done

# mDNS discovery on loopback. Each daemon picks its own ephemeral
# P2P port (port+10 by default).
for label_port in a:$BENCH_PORT_A b:$BENCH_PORT_B c:$BENCH_PORT_C; do
    label=${label_port%:*}
    port=${label_port#*:}
    DIR=/tmp/swarm_bench_$label
    LOG=$DIR/log.txt
    echo "Starting node $label on port $port (data: $DIR)..."
    nohup $BINARY run \
        --port $port \
        --data-dir $DIR \
        > $LOG 2>&1 &
    echo "  PID=$!"
done

sleep 3
echo
echo "Nodes started. Waiting 12s for mDNS discovery + key generation..."
sleep 12

for label_port in a:$BENCH_PORT_A b:$BENCH_PORT_B c:$BENCH_PORT_C; do
    label=${label_port%:*}
    port=${label_port#*:}
    DIR=/tmp/swarm_bench_$label
    API_KEY=$(cat $DIR/api_key 2>/dev/null || echo "MISSING")
    echo
    echo "Node $label (port $port):"
    echo "  api_key=$API_KEY"
    echo "  log: $DIR/log.txt"
    if [ "$API_KEY" != "MISSING" ]; then
        PEERS=$(curl -s -m 5 -H "Authorization: Bearer $API_KEY" \
            http://localhost:$port/api/admin/peers 2>&1 | head -c 300)
        echo "  peers: $PEERS"
    fi
done

echo
echo "Done. To stop these nodes only: for p in $BENCH_PORT_A $BENCH_PORT_B $BENCH_PORT_C; do
  for pid in \$(pgrep -x swarmllm); do tr '\\0' ' ' < /proc/\$pid/cmdline | grep -q -- \"-p \$p\" && kill \$pid; done
done"
