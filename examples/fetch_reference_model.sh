#!/bin/bash
# Opt in to hosting a reference test model.
#
#   ./fetch_reference_model.sh smoke              # host your fair share
#   ./fetch_reference_model.sh standard --all     # host every shard
#   ./fetch_reference_model.sh --list
#
# Nothing here runs on its own. Reference models are only ever fetched because
# someone ran this — they are for testing the swarm, and spending a user's
# bandwidth and disk on that without asking would not be reasonable.
#
# Pins and rationale: docs/REFERENCE_MODELS.md
set -euo pipefail

PORT="${SWARMLLM_PORT:-8800}"
DATA_DIR="${SWARMLLM_NODE_DATA_DIR:-$HOME/.local/share/swarmllm}"

# tier|repo_id|filename|approx MB|shards at the default 512 MB shard size
TIERS=(
  "smoke|TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF|tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf|638|2"
  "standard|bartowski/Llama-3.2-3B-Instruct-GGUF|Llama-3.2-3B-Instruct-Q4_K_M.gguf|1925|4"
  "stress|bartowski/Meta-Llama-3.1-8B-Instruct-GGUF|Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|4692|10"
)

list_tiers() {
    printf '%-10s %-12s %-8s %s\n' TIER DOWNLOAD SHARDS REPO
    for row in "${TIERS[@]}"; do
        IFS='|' read -r tier repo file mb shards <<< "$row"
        printf '%-10s %-12s %-8s %s\n' "$tier" "${mb} MB" "$shards" "$repo"
    done
}

if [ $# -eq 0 ] || [ "${1:-}" = "--list" ] || [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    list_tiers
    echo
    echo "Usage: $0 <tier> [--all]"
    echo "  default    fetch only this node's fair share of the shards"
    echo "  --all      fetch every shard (needs room for the whole model)"
    exit 0
fi

TIER="$1"
ALL="${2:-}"

ROW=""
for row in "${TIERS[@]}"; do
    [ "${row%%|*}" = "$TIER" ] && ROW="$row" && break
done
if [ -z "$ROW" ]; then
    echo "Unknown tier: $TIER" >&2
    echo >&2
    list_tiers >&2
    exit 1
fi
IFS='|' read -r _ REPO FILE MB SHARDS <<< "$ROW"

KEY_FILE="$DATA_DIR/api_key"
if [ ! -f "$KEY_FILE" ]; then
    echo "No API key at $KEY_FILE — is the daemon running with this data dir?" >&2
    echo "Set SWARMLLM_NODE_DATA_DIR if it lives elsewhere." >&2
    exit 1
fi
API_KEY=$(cat "$KEY_FILE")

if [ "$ALL" = "--all" ]; then
    # Explicit index list: the endpoint requires either shards[] or fair-share,
    # and the two are mutually exclusive.
    INDICES=$(seq -s, 0 $((SHARDS - 1)))
    PAYLOAD="{\"repo_id\":\"$REPO\",\"filename\":\"$FILE\",\"shards\":[$INDICES]}"
    WHAT="all $SHARDS shards (~${MB} MB)"
else
    PAYLOAD="{\"repo_id\":\"$REPO\",\"filename\":\"$FILE\",\"shards\":[],\"peer_fair_share\":true}"
    WHAT="this node's fair share of $SHARDS shards"
fi

echo "Tier:   $TIER"
echo "Repo:   $REPO"
echo "File:   $FILE"
echo "Fetch:  $WHAT"
echo

RESPONSE=$(curl -s -m 30 -X POST "http://127.0.0.1:$PORT/api/admin/hf/download-shards" \
    -H "Authorization: Bearer $API_KEY" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD")

echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo
echo "Download runs in the background. Watch it on the dashboard, or:"
echo "  curl -s -H \"Authorization: Bearer \$(cat $KEY_FILE)\" \\"
echo "    http://127.0.0.1:$PORT/api/admin/models | python3 -m json.tool"
