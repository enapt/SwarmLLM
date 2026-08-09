#!/bin/bash
# SWARM-SPEC 3-node sharded cluster — forces distributed pipeline.
#
# Layout:
#   Node A (BENCH_PORT_A, default 8890): NO model. Pure coordinator/requester.
#   Node B (BENCH_PORT_B, default 8891): shard_000 only (layers 0-11). First segment holder.
#   Node C (BENCH_PORT_C, default 8892): shard_001 only (layers 12-21). Last segment holder.
#
# Inference requests to A force a 2-segment distributed pipeline
# (B → C → result). This exercises Layer 0 (Q8_0 activation
# compression on the wire) and the full distributed forward path.
#
# KNOWN LIMITATION — read before believing a failure here.
#
# This layout is the same-host, zero-redundancy case that
# docs/FUTURE_WORK.md § "Connection churn on multi-interface hosts" documents
# as NOT covered: one loopback host advertising several interfaces (WSL2's
# NAT gateway, link-local, Docker bridge, LAN) can still form a stale
# connection, and with exactly one holder per shard there is no standby to
# fail over to. Inference then fails with "Segment N failed with no standby
# available" — reproduced on v0.3.28, 2026-07-26.
#
# Real deployments do not hit this: distinct hosts with min_replicas >= 2
# have somewhere to fail over. So a failure HERE is not evidence that
# distributed inference is broken — confirm on two machines before concluding
# that. Use this script for shard-splitting and scheduling behaviour; use two
# real hosts to validate the forward path end to end.

set -e

# Model under test. Defaults to the Smoke tier (2 shards); override with
# SWARM_BENCH_MODEL for a deeper split. See docs/REFERENCE_MODELS.md.
MODEL_NAME="${SWARM_BENCH_MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
SRC_MODEL=~/.local/share/swarmllm/models/$MODEL_NAME

if [ ! -f "$SRC_MODEL/manifest.json" ]; then
    echo "No model at $SRC_MODEL" >&2
    echo "Acquire it first, or set SWARM_BENCH_MODEL to one you have." >&2
    echo "Available:" >&2
    ls -1 ~/.local/share/swarmllm/models/ 2>/dev/null | sed 's/^/  /' >&2
    exit 1
fi
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
    mkdir -p "$DIR/models/$MODEL_NAME"
done

# Node A: just the manifest + gguf_header so the daemon knows about
# the model but has zero shards locally. The manifest is needed to
# resolve the layer→shard mapping.
cp "$SRC_MODEL/manifest.json" "/tmp/swarm_bench_a/models/$MODEL_NAME/"
cp "$SRC_MODEL/gguf_header.bin" "/tmp/swarm_bench_a/models/$MODEL_NAME/"
cp "$SRC_MODEL/hf_source.json" "/tmp/swarm_bench_a/models/$MODEL_NAME/" 2>/dev/null || true

# Nodes B and C: split the shard list in half, B taking the earlier layers.
# Done by enumeration rather than naming shard_000/shard_001 so the script
# works for any shard count — a Standard-tier model at shard_size_mb=256 has
# eight shards, not two.
SHARDS=($(ls -1 "$SRC_MODEL"/shard_*.bin 2>/dev/null | sort))
if [ ${#SHARDS[@]} -lt 2 ]; then
    echo "Need at least 2 shards to force a distributed pipeline, found ${#SHARDS[@]}" >&2
    exit 1
fi
HALF=$(( (${#SHARDS[@]} + 1) / 2 ))

for label in b c; do
    cp "$SRC_MODEL/manifest.json" "/tmp/swarm_bench_$label/models/$MODEL_NAME/"
    cp "$SRC_MODEL/gguf_header.bin" "/tmp/swarm_bench_$label/models/$MODEL_NAME/"
    cp "$SRC_MODEL/hf_source.json" "/tmp/swarm_bench_$label/models/$MODEL_NAME/" 2>/dev/null || true
done
for i in "${!SHARDS[@]}"; do
    if [ "$i" -lt "$HALF" ]; then target=b; else target=c; fi
    cp "${SHARDS[$i]}" "/tmp/swarm_bench_$target/models/$MODEL_NAME/"
done
echo "Split ${#SHARDS[@]} shards: B gets first $HALF, C gets the rest"

echo "Shard placement:"
for label in a b c; do
    DIR=/tmp/swarm_bench_$label
    echo "  Node $label:"
    ls -la "$DIR/models/$MODEL_NAME/" | grep -E "shard|manifest|gguf" | awk '{print "    " $9 " (" $5 " bytes)"}'
done

# Per-node config. Two settings are what make this a *sharded* test rather
# than three nodes that quietly converge on holding everything:
#
#   auto_manage.enabled = false — otherwise each node downloads the shards it
#   is missing from its peers within a minute or two, every node ends up with
#   the whole model, and inference runs locally. The split is then gone and the
#   run silently measures the wrong thing. (This was documented as a manual
#   prerequisite and not actually applied, so the script did not do what it
#   claimed — observed 2026-07-26: node B had both shards within 30s.)
#
#   bootstrap_peers = [] + disable_default_bootstrap = true — keeps the cluster
#   to itself. With the defaults these nodes join the public swarm, learn shard
#   holders from it, and can schedule a segment onto a node that is not part of
#   the test. The second line is load-bearing: an empty list on its own falls
#   back to the built-in anchors, because that is what a pre-2026-07-21 config
#   looks like and those nodes must not be stranded.
for label in a b c; do
    cat > /tmp/swarm_bench_$label/config.toml <<'CFG'
[auto_manage]
enabled = false

[network]
bootstrap_peers = []
disable_default_bootstrap = true
CFG
done

# Spawn nodes
for label_port in a:$BENCH_PORT_A b:$BENCH_PORT_B c:$BENCH_PORT_C; do
    label=${label_port%:*}
    port=${label_port#*:}
    DIR=/tmp/swarm_bench_$label
    LOG=$DIR/log.txt
    echo "Starting node $label on port $port..."
    nohup $BINARY run --port $port --data-dir $DIR --config $DIR/config.toml > $LOG 2>&1 &
done

sleep 3
echo
echo "Waiting 15s for mDNS discovery + shard announcements..."
sleep 15

for label_port in a:$BENCH_PORT_A b:$BENCH_PORT_B c:$BENCH_PORT_C; do
    label=${label_port%:*}
    port=${label_port#*:}
    DIR=/tmp/swarm_bench_$label
    API_KEY=$(cat $DIR/api_key 2>/dev/null || echo "MISSING")
    echo
    echo "Node $label (port $port):"
    echo "  api_key=$API_KEY"
done

echo
echo "Done. To stop these nodes only: for p in $BENCH_PORT_A $BENCH_PORT_B $BENCH_PORT_C; do
  for pid in \$(pgrep -x swarmllm); do tr '\\0' ' ' < /proc/\$pid/cmdline | grep -q -- \"-p \$p\" && kill \$pid; done
done"
