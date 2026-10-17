#!/bin/bash
# FUTURE_WORK #91 (2): a newcomer advertising `features::MANIFEST_DIGESTS` is
# caught up with a `ManifestDigest` (the version of each manifest, ~200 bytes a
# model) and pulls (`ManifestPull`) only the versions it lacks, where it was
# sent every manifest in full (~35 KB each; 84 of them, ~2.9 MB, reached one
# probe node from six peers in 20 s on 2026-10-16).
#
# Three nodes in a private network namespace (re-runs itself under `unshare
# -rn`, like capability_gate_rig.sh), seeing each other on loopback and nothing
# else. A holds TinyLlama (its parts HARD-LINKED from MODEL_DIR, so no copy and
# no download; the small files are copied, so nothing A writes reaches the
# source). B joins first and takes A's first broadcast — the network loop
# buffers a publish that found no peer and flushes it to the first that
# subscribes — so C, joining later, is a newcomer as it is in a real swarm:
# nothing gossiped reaches it until a manifest changes, and the catch-up is
# what tells it.
#
#   1. C learns the model. With A and C new: A logs "caught a new peer up on
#      manifests with a digest", C logs one "pulling the manifest versions", and
#      the manifest reaches C point to point.
#   2. A restarts (B and C now hold the version): A catches both up with a
#      digest again and neither pulls — the reconnect that used to cost every
#      manifest.
#
# Mixed versions: BIN_A=<older> or BIN_C=<older> — no bit on one side, so the
# manifests go in full as before, and C still learns the model.
#
# ✅ 2026-10-17 (dev build, re-run on the reviewed code): C pulled 1, got 1
# manifest direct, 0 by gossip;
# after A's restart 0 pulls. BIN_C=v0.3.240: A caught C up in full, C learned
# it; BIN_A=v0.3.240: full catch-ups both ways, C learned it.
#
# Usage: examples/manifest_digest_rig.sh <binary> [out_dir]
#   MODEL_DIR=~/.local/share/swarmllm/models/tinyllama-1.1b-chat-v1.0.q4-k-m
#   BIN_A=, BIN_C=<binary for that node> (default: <binary>)
set -u
if [ -z "${DIGEST_RIG_IN_NETNS:-}" ]; then
  exec env DIGEST_RIG_IN_NETNS=1 unshare -rn bash "$0" "$@"
fi
BIN="$(readlink -f "$1")"; OUT="${2:-./manifest_digest_out}"
BIN_A="$(readlink -f "${BIN_A:-$BIN}")"; BIN_C="$(readlink -f "${BIN_C:-$BIN}")"
MODEL_DIR="${MODEL_DIR:-$HOME/.local/share/swarmllm/models/tinyllama-1.1b-chat-v1.0.q4-k-m}"
MODEL="$(basename "$MODEL_DIR")"
mkdir -p "$OUT"
ip link set lo up || { echo "cannot bring up loopback"; exit 2; }
BASE=$(mktemp -d "$HOME/.digest-rig.XXXXXX")
make_node() { # dir bootstrap
  mkdir -p "$1"
  cat > "$1/config.toml" <<CFG
[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-digest-rig"
enable_mdns = false
enable_upnp = false
relay_forwarding_auto = false

[auto_manage]
enabled = false
prune_enabled = false

[updates]
mode = "off"

[ui]
open_browser_on_start = false
CFG
}
start() { # dir port binary
  env SWARMLLM_NODE_DATA_DIR="$1" SWARMLLM_INFERENCE_GPU_LAYERS=0 "$3" run -p "$2" -v >> "$1/node.log" 2>&1 & echo $!
}
up() {
  for _ in $(seq 1 120); do
    [ -f "$1/api_key" ] && curl -s -m 3 "localhost:$2/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "node on $2 never came up — see $1/node.log"; return 1
}
addr_of() {
  local id
  for _ in $(seq 1 30); do
    id=$(grep -a -m1 -oE "Initializing network peer_id=12D3Koo[A-Za-z0-9]+" "$1/node.log" | cut -d= -f2)
    [ -n "$id" ] && grep -aq "New listen address address=/ip4/127.0.0.1/tcp/$(( $2 + 10 ))" "$1/node.log" && break
    sleep 1
  done
  [ -n "$id" ] && echo "/ip4/127.0.0.1/tcp/$(( $2 + 10 ))/p2p/$id"
}
count() { grep -ac "$2" "$1" 2>/dev/null; }
# The node has registered the model's manifest (the registry's own line).
knows_model() { grep -a "DIAG: register_manifest" "$BASE/$1/node.log" | grep -aq "model=$MODEL"; }
received() { grep -a 'DIAG: manifest received' "$BASE/$1/node.log" | grep -ac "transport=$2"; }
report() { # label
  local a="$BASE/A/node.log"
  echo "[$1] A catch-ups by digest: $(count "$a" 'caught a new peer up on manifests with a digest')," \
    "in full: $(count "$a" 'caught a new peer up on manifests directly')"
  for n in B C; do
    echo "      $n pulls: $(count "$BASE/$n/node.log" 'pulling the manifest versions')," \
      "manifests received direct: $(received $n Direct), by gossip: $(received $n Gossip)"
  done
}
stop() { # pid — the nodes are not this shell's children, so `wait` cannot see them
  kill "$1" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null
}
cleanup() {
  for p in ${PA:-} ${PB:-} ${PC:-}; do stop "$p"; done
  for n in A B C; do cp "$BASE/$n/node.log" "$OUT/$n.log" 2>/dev/null; done
  rm -rf "$BASE"
}
trap cleanup EXIT

[ -f "$MODEL_DIR/manifest.json" ] || { echo "no manifest in $MODEL_DIR"; exit 2; }
mkdir -p "$BASE/A/models/$MODEL"
for f in "$MODEL_DIR"/*; do
  case "$(basename "$f")" in
    shard_*.bin) ln "$f" "$BASE/A/models/$MODEL/" || exit 2 ;;
    *) cp "$f" "$BASE/A/models/$MODEL/" ;;
  esac
done
make_node "$BASE/A" ""
PA=$(start "$BASE/A" 8960 "$BIN_A"); up "$BASE/A" 8960 || exit 1
AADDR=$(addr_of "$BASE/A" 8960); [ -n "$AADDR" ] || { echo "A published no listen address"; exit 1; }
make_node "$BASE/B" "\"$AADDR\""
PB=$(start "$BASE/B" 8980 "$BIN"); up "$BASE/B" 8980 || exit 1
echo "A: $("$BIN_A" --version)   B: $("$BIN" --version)   C: $("$BIN_C" --version)   model: $MODEL"
for _ in $(seq 1 60); do knows_model B && break; sleep 2; done
sleep 40   # past A's next round: its buffered first broadcast has gone to B

make_node "$BASE/C" "\"$AADDR\""
PC=$(start "$BASE/C" 9000 "$BIN_C"); up "$BASE/C" 9000 || exit 1
for _ in $(seq 1 90); do knows_model C && break; sleep 2; done
knows_model C && echo "step 1: C learned $MODEL" || echo "step 1: C never learned $MODEL"
sleep 40   # a round or more past the catch-up, so a stray repeat would show
report "after C joined"

pulls() { echo $(( $(count "$BASE/B/node.log" 'pulling the manifest versions') + $(count "$BASE/C/node.log" 'pulling the manifest versions') )); }
before=$(pulls); caught=$(count "$BASE/A/node.log" 'caught a new peer up on manifests')
stop "$PA"
PA=$(start "$BASE/A" 8960 "$BIN_A"); up "$BASE/A" 8960 || exit 1
for _ in $(seq 1 60); do
  [ "$(count "$BASE/A/node.log" 'caught a new peer up on manifests')" -ge $(( caught + 2 )) ] && break
  sleep 2
done
sleep 20
report "after A restarted"
echo "RESULT: B and C pulled $(( $(pulls) - before )) time(s) after A restarted (expect 0 when A and C are new)"
exit 0
