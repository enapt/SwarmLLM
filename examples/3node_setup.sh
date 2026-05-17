#!/bin/bash
# SWARM-SPEC 3-node local cluster setup for benchmarking.

set -e

SRC_MODEL=~/.local/share/swarmllm/models/tinyllama-1.1b-chat-v1.0.q4-k-m
BINARY=/home/user/SwarmLLM/target/release/swarmllm

killall -9 swarmllm 2>/dev/null || true
sleep 1

for label in a b c; do
    DIR=/tmp/swarm_bench_$label
    rm -rf "$DIR"
    mkdir -p "$DIR/models"
    cp -r "$SRC_MODEL" "$DIR/models/"
done

# mDNS discovery on loopback. Each daemon picks its own ephemeral
# P2P port (port+10 by default).
for label_port in a:8800 b:8801 c:8802; do
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

for label_port in a:8800 b:8801 c:8802; do
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
echo "Done. To stop: killall -9 swarmllm"
