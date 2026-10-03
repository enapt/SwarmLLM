#!/bin/bash
# A node that cannot reach HuggingFace, holding a part that is not the upload's
# bytes, beside two nodes that can — does it replace the part from them?
#
# WHY THIS EXISTS
# ---------------
# Peer `9594e1ff` never replaced a single wrong part on v0.3.221, .222 or .223
# while every other peer's heal acted within the hour (FUTURE_WORK #160). The
# heal needed HuggingFace for everything — which upload is the swarm's, whether
# a part is its bytes, where the replacement comes from — and a node in offline
# mode skipped it outright. Since v0.3.224 a node whose heal checked its copy
# says so (`ShardAnnounce::origin_checked_models`), and a node with no origin to
# ask replaces a part on the word of `CHECKED_QUORUM` (2) such holders that
# agree, fetching it from them (`auto_manage::canonical::settle_by_checked_holders`).
#
# THE SHAPE
# ---------
#   A, C  hold every part of MODEL, reach HuggingFace, run the heal: they check
#         their copies and announce them as checked.
#   B     holds every part, its LAST part zeroed (a copy, never the hard link)
#         under a manifest rewritten to vouch for the zeroed bytes — the
#         spliced-copy shape (`split_rig.sh spliced`) — and CANNOT reach
#         HuggingFace:
#           HFLESS=proxy    (default) every HTTP request goes to a dead proxy —
#                           the "no route to HuggingFace" case;
#           HFLESS=offline  `[pool] offline_mode = true` — B dials nobody, so
#                           A and C dial B.
# PASS = B logs the checked holders' verdict, deletes the part, fetches it from
# A or C, and holds the upload's bytes again (BLAKE3-identical to the live
# node's part). The null control is v0.3.223 as B: the part stays zeroed.
#
# usage: outvoted_rig.sh <binary for A and C> [<binary for B>]
#   MODEL     default tinyllama-1.1b-chat-v1.0.q4-k-m (needs >= 2 parts here)
#   WAIT_SECS how long B gets (default 900)
#   OUT       where logs go (default: a temp dir, printed at the end)
# The live node must be STOPPED: every node on this machine finds the others
# through its loopback probe (#708), and the rig's nodes would find it.
set -u
BIN_AC="${1:?usage: outvoted_rig.sh <binary for A and C> [<binary for B>]}"
BIN_B="${2:-$BIN_AC}"
HFLESS="${HFLESS:-proxy}"
case "$HFLESS" in proxy|offline) ;; *) echo "HFLESS must be proxy or offline"; exit 2 ;; esac
[ -x "$BIN_AC" ] && [ -x "$BIN_B" ] || { echo "binary not executable"; exit 2; }
MODEL="${MODEL:-tinyllama-1.1b-chat-v1.0.q4-k-m}"
MODELS_DIR="${MODELS_DIR:-$HOME/.local/share/swarmllm/models}"
SRC="$MODELS_DIR/$MODEL"
[ -f "$SRC/manifest.json" ] && [ -f "$SRC/hf_source.json" ] || { echo "no manifest/hf_source for $MODEL in $MODELS_DIR"; exit 2; }
SHARDS=$(ls "$SRC" | sed -n 's/^shard_\([0-9]*\)\.bin$/\1/p' | sed 's/^0*\([0-9]\)/\1/' | sort -n)
LAST=$(echo "$SHARDS" | tail -1)
[ "$(echo "$SHARDS" | wc -l)" -ge 2 ] || { echo "$MODEL needs at least 2 part files here"; exit 2; }
PARTF=$(printf 'shard_%03d.bin' "$LAST")
OTHERS=$(ps -eo pid,comm | awk '$2 ~ /swarmllm/ {print $1}' | paste -sd' ')
[ -z "$OTHERS" ] || { echo "rig: other SwarmLLM processes are running (pids: $OTHERS) — stop them first"; exit 2; }
BASE=$(mktemp -d "$HOME/.outvoted-rig.XXXXXX")
OUT="${OUT:-$BASE/out}"
mkdir -p "$OUT"
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib

make_node() { # dir bootstrap extra-toml
  local d="$1" m="$1/models/$MODEL"
  mkdir -p "$m"
  # Parts as hard links (nothing writes into a part; a replacement is renamed
  # over it), everything the heal may REWRITE as copies — writing through a
  # link would change the live node's file.
  for i in $SHARDS; do ln "$SRC/$(printf 'shard_%03d.bin' "$i")" "$m/" || { echo "cannot link part $i"; exit 2; }; done
  for f in gguf_header.bin hf_source.json manifest.json tied_output_weight.bin rope_freqs.bin; do
    [ -f "$SRC/$f" ] && cp "$SRC/$f" "$m/$f"
  done
  cat > "$d/config.toml" <<CFG
[network]
bootstrap_peers = [$2]
disable_default_bootstrap = true
gossip_network_id = "swarmllm-outvoted-rig"
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

[resources]
max_bandwidth_mbps = 10000
CFG
  [ -n "$3" ] && printf '\n%s\n' "$3" >> "$d/config.toml"
  return 0
}
start() { # dir port bin [env...]
  local d="$1" p="$2" b="$3"; shift 3
  env SWARMLLM_NODE_DATA_DIR="$d" SWARMLLM_INFERENCE_GPU_LAYERS=0 "$@" "$b" run -p "$p" > "$d/node.log" 2>&1 &
  echo $!
}
up() { # dir port
  for _ in $(seq 1 90); do
    [ -f "$1/api_key" ] && curl -s -m 3 "localhost:$2/health" >/dev/null 2>&1 && return 0
    sleep 2
  done
  echo "node on $2 never came up — see $1/node.log"; return 1
}
addr_of() { # dir port — a direct TCP address the node publishes
  local k; k=$(cat "$1/api_key")
  curl -s -m 8 -H "Authorization: Bearer $k" "localhost:$2/api/admin/diagnostics?full=1" \
    | grep -oE "/ip4/[0-9.]+/tcp/[0-9]+/p2p/[A-Za-z0-9]+" | grep -v "p2p-circuit" \
    | grep -v "10\.255\.255\.254" | head -1
}
holding() { # dir port
  local k; k=$(cat "$1/api_key")
  curl -s -m 10 -H "Authorization: Bearer $k" "localhost:$2/api/admin/models" \
    | python3 -c "import sys,json; m=next((m for m in json.load(sys.stdin) if m['id']=='$MODEL'), {}); print(((m.get('shared_copy') or {}).get('this_computer') or {}).get('state','?'))" 2>/dev/null
}
cleanup() {
  kill ${PA:-} ${PB:-} ${PC:-} 2>/dev/null; sleep 3
  for n in A B C; do cp "$BASE/$n/node.log" "$OUT/$n.log" 2>/dev/null; done
  rm -rf "$BASE/A" "$BASE/B" "$BASE/C"
  [ "$OUT" = "$BASE/out" ] || rmdir "$BASE" 2>/dev/null
  echo "rig: logs in $OUT"
}
trap cleanup EXIT

# B first: in offline mode it dials nobody, so A and C must dial it.
B_EXTRA=""; B_ENV=()
if [ "$HFLESS" = offline ]; then
  B_EXTRA=$'[pool]\noffline_mode = true'
else
  B_ENV=(HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 ALL_PROXY=http://127.0.0.1:9 NO_PROXY=)
fi
make_node "$BASE/B" "" "$B_EXTRA"
M="$BASE/B/models/$MODEL"
rm -f "$M/$PARTF"
cp "$SRC/$PARTF" "$M/$PARTF" || exit 2
SZ=$(stat -c %s "$M/$PARTF")
dd if=/dev/zero of="$M/$PARTF" bs=1M count=$(( SZ / 1048576 )) conv=notrunc status=none || exit 2
python3 - "$M/manifest.json" "$M/$PARTF" "$LAST" <<'PY' || exit 2
import json, sys, blake3
path, part, last = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = json.load(open(path))
h = blake3.blake3(open(part, 'rb').read()).digest()
for s in m['shards']:
    if s['index'] == last:
        s['hash'] = list(h)
m['manifest_hash'] = [0] * 32
json.dump(m, open(path, 'w'))
print(f"rig: B's part {last} zeroed, its manifest vouches for {h.hex()[:16]}")
PY
PB=$(start "$BASE/B" 8920 "$BIN_B" "${B_ENV[@]}")
up "$BASE/B" 8920 || exit 1
ADDR_B=$(addr_of "$BASE/B" 8920)
[ -n "$ADDR_B" ] || { echo "B published no address to dial"; exit 1; }
make_node "$BASE/A" "\"$ADDR_B\"" ""
make_node "$BASE/C" "\"$ADDR_B\"" ""
PA=$(start "$BASE/A" 8900 "$BIN_AC")
PC=$(start "$BASE/C" 8940 "$BIN_AC")
up "$BASE/A" 8900 || exit 1
up "$BASE/C" 8940 || exit 1
echo "rig: B ($HFLESS, $("$BIN_B" --version 2>/dev/null)) and A, C ($("$BIN_AC" --version 2>/dev/null)) up"

WANT=$(python3 -c 'import sys,blake3; print(blake3.blake3(open(sys.argv[1],"rb").read()).hexdigest())' "$SRC/$PARTF")
T0=$(date +%s); same=0; last_report=""
while [ $(( $(date +%s) - T0 )) -lt "${WAIT_SECS:-900}" ]; do
  sz=$(stat -c %s "$M/$PARTF" 2>/dev/null || echo gone)
  report="A=$(holding "$BASE/A" 8900) C=$(holding "$BASE/C" 8940) B's part $LAST=$sz verdict=$(grep -ac "differ from what the holders that checked theirs" "$BASE/B/node.log") deleted=$(grep -ac "deleted this node's parts that are not the canonical upload" "$BASE/B/node.log")"
  [ "$report" != "$last_report" ] && echo "rig: +$(( $(date +%s) - T0 ))s $report"
  last_report=$report
  if [ "$sz" = "$SZ" ] && [ "$(grep -ac "deleted this node's parts" "$BASE/B/node.log")" -ge 1 ]; then
    got=$(python3 -c 'import sys,blake3; print(blake3.blake3(open(sys.argv[1],"rb").read()).hexdigest())' "$M/$PARTF" 2>/dev/null)
    [ "$got" = "$WANT" ] && { same=1; break; }
  fi
  sleep 10
done
echo "--- B's heal ---"
grep -aE "checked theirs|deleted this node's parts|P2P shard download complete|Fetching from the model's origin|Fetching a part queued for repair|could not be checked on HuggingFace|Could not compare parts" "$BASE/B/node.log" | cut -c1-240 | head -20
echo "--- A and C ---"
grep -aE "canonical upload's$|this node's parts are the canonical upload's" "$BASE/A/node.log" "$BASE/C/node.log" | cut -c1-200 | head -4
if [ "$same" = 1 ]; then
  echo "outvoted ($HFLESS): PASS — B replaced its part from the checked holders after $(( $(date +%s) - T0 )) s"
  exit 0
fi
echo "outvoted ($HFLESS): FAIL — B's part $LAST is not the upload's after ${WAIT_SECS:-900} s"
exit 1
