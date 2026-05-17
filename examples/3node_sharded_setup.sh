#!/bin/bash
# SWARM-SPEC 3-node sharded cluster — forces distributed pipeline.
#
# Layout:
#   Node A (port 8800): NO model. Pure coordinator/requester.
#   Node B (port 8801): shard_000 only (layers 0-11). First segment holder.
#   Node C (port 8802): shard_001 only (layers 12-21). Last segment holder.
#
# Inference requests to A force a 2-segment distributed pipeline
# (B → C → result). This exercises Layer 0 (Q8_0 activation
# compression on the wire) and the full distributed forward path.

set -e

SRC_MODEL=~/.local/share/swarmllm/models/tinyllama-1.1b-chat-v1.0.q4-k-m
MODEL_NAME=tinyllama-1.1b-chat-v1.0.q4-k-m
BINARY=/home/user/SwarmLLM/target/release/swarmllm

killall -9 swarmllm 2>/dev/null || true
sleep 1

for label in a b c; do
    DIR=/tmp/swarm_bench_$label
    rm -rf "$DIR"
    mkdir -p "$DIR/models/$MODEL_NAME"
done

# Node A: just the manifest + gguf_header so the daemon knows about
# the model but has zero shards locally. The manifest is needed to
# resolve the layer→shard mapping.
cp "$SRC_MODEL/manifest.json" "/tmp/swarm_bench_a/models/$MODEL_NAME/"
cp "$SRC_MODEL/gguf_header.bin" "/tmp/swarm_bench_a/models/$MODEL_NAME/"
cp "$SRC_MODEL/hf_source.json" "/tmp/swarm_bench_a/models/$MODEL_NAME/" 2>/dev/null || true

# Node B: shard_000 (layers 0-11) + manifest
cp "$SRC_MODEL/manifest.json" "/tmp/swarm_bench_b/models/$MODEL_NAME/"
cp "$SRC_MODEL/gguf_header.bin" "/tmp/swarm_bench_b/models/$MODEL_NAME/"
cp "$SRC_MODEL/shard_000.bin" "/tmp/swarm_bench_b/models/$MODEL_NAME/"
cp "$SRC_MODEL/hf_source.json" "/tmp/swarm_bench_b/models/$MODEL_NAME/" 2>/dev/null || true

# Node C: shard_001 (layers 12-21) + manifest
cp "$SRC_MODEL/manifest.json" "/tmp/swarm_bench_c/models/$MODEL_NAME/"
cp "$SRC_MODEL/gguf_header.bin" "/tmp/swarm_bench_c/models/$MODEL_NAME/"
cp "$SRC_MODEL/shard_001.bin" "/tmp/swarm_bench_c/models/$MODEL_NAME/"
cp "$SRC_MODEL/hf_source.json" "/tmp/swarm_bench_c/models/$MODEL_NAME/" 2>/dev/null || true

echo "Shard placement:"
for label in a b c; do
    DIR=/tmp/swarm_bench_$label
    echo "  Node $label:"
    ls -la "$DIR/models/$MODEL_NAME/" | grep -E "shard|manifest|gguf" | awk '{print "    " $9 " (" $5 " bytes)"}'
done

# Spawn nodes
for label_port in a:8800 b:8801 c:8802; do
    label=${label_port%:*}
    port=${label_port#*:}
    DIR=/tmp/swarm_bench_$label
    LOG=$DIR/log.txt
    echo "Starting node $label on port $port..."
    nohup $BINARY run --port $port --data-dir $DIR > $LOG 2>&1 &
done

sleep 3
echo
echo "Waiting 15s for mDNS discovery + shard announcements..."
sleep 15

for label_port in a:8800 b:8801 c:8802; do
    label=${label_port%:*}
    port=${label_port#*:}
    DIR=/tmp/swarm_bench_$label
    API_KEY=$(cat $DIR/api_key 2>/dev/null || echo "MISSING")
    echo
    echo "Node $label (port $port):"
    echo "  api_key=$API_KEY"
done

echo
echo "Done. To stop: killall -9 swarmllm"
