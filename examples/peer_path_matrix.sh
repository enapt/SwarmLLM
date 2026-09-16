#!/bin/bash
# Sweep one model across the routing shapes the swarm can produce, and report
# which computers actually served each.
#
#   examples/peer_path_matrix.sh                               # defaults below
#   examples/peer_path_matrix.sh http://localhost:8800 llama-3.2-3b-instruct-q4-k-m
#
# WHY THIS EXISTS
#
# The case this project is for — a model whose layers live on several machines —
# is the one a development swarm stops exercising DELIBERATELY. Auto-manage
# replicates, which is its job, so nodes converge on holding whole models and
# every interesting split stops being a thing you can arrange on purpose.
# `examples/*_sharded_setup.sh` build a split swarm from scratch to get around
# that, at the cost of a full model distribution per scenario.
#
# This uses the `swarm_route` request field instead: each request is PLANNED as
# though this node held less, or as though a named computer were not there.
# Nothing moves on disk and nothing restarts, so a whole matrix runs in the time
# it takes to answer a few short prompts.
#
# It reports the route from the response headers — `x-swarm-route`,
# `x-swarm-nodes`, `x-swarm-regions` — which is the discriminator gotcha #375
# was written about: "for any report from a multi-node setup, establish which
# node served the failing request before reproducing anything". This asks that
# question for every shape at once, and needs no access to the node's log.
#
# WHAT IT CANNOT TELL YOU
#
# The override only ever makes this node's candidate set SMALLER. It cannot
# invent a holder, and it cannot make a peer accept work it would otherwise
# refuse — so a scenario that fails here may be saying the swarm genuinely has
# no route, not that routing is broken. Read the ROUTE and NODES columns, not
# just the RESULT one.
set -u

BASE="${1:-http://localhost:8800}"
MODEL="${2:-llama-3.2-3b-instruct-q4-k-m}"
# Same lookup order the CLI uses: an explicit key wins, else the running node's.
KEY="${SWARMLLM_API_KEY:-$(cat "${SWARM_KEY_FILE:-$HOME/.local/share/swarmllm/api_key}" 2>/dev/null)}"
PROMPT="${SWARM_MATRIX_PROMPT:-Name the capital of France in one short sentence.}"
MAX_TOKENS="${SWARM_MATRIX_MAX_TOKENS:-24}"

[ -n "$KEY" ] || { echo "no API key — set SWARMLLM_API_KEY or SWARM_KEY_FILE"; exit 1; }

hdrs=$(mktemp); body=$(mktemp)
cleanup() { rm -f "$hdrs" "$body"; }
trap cleanup EXIT

# Run one scenario. $1 is the label, $2 the `swarm_route` object — or empty for
# no override at all, which is NOT the same as an empty object and is the
# baseline every other row is read against.
run_case() {
    local label="$1" route="$2" extra=""
    [ -n "$route" ] && extra=",\"swarm_route\":$route"
    local t0 t1
    t0=$(date +%s.%N)
    curl -s -S -m 300 -D "$hdrs" -o "$body" \
        -X POST "$BASE/v1/chat/completions" \
        -H "Authorization: Bearer $KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS,\"temperature\":0$extra}" \
        2>/dev/null
    t1=$(date +%s.%N)

    local code secs r nodes regions segs
    code=$(awk 'NR==1{print $2}' "$hdrs")
    secs=$(awk "BEGIN{printf \"%.1f\", $t1-$t0}")
    # Header names are case-insensitive on the wire, and curl keeps the CR.
    r=$(tr -d '\r' < "$hdrs" | awk 'BEGIN{IGNORECASE=1}/^x-swarm-route:/{print $2}')
    segs=$(tr -d '\r' < "$hdrs" | awk 'BEGIN{IGNORECASE=1}/^x-swarm-segments:/{print $2}')
    nodes=$(tr -d '\r' < "$hdrs" | awk 'BEGIN{IGNORECASE=1}/^x-swarm-nodes:/{print $2}')
    regions=$(tr -d '\r' < "$hdrs" | awk 'BEGIN{IGNORECASE=1}/^x-swarm-regions:/{print $2}')

    local verdict
    if [ "$code" = "200" ]; then
        # A 200 carrying no text is a failure wearing a success, so the verdict
        # comes from the reply rather than from the status.
        if python3 -c "
import json,sys
d=json.load(open('$body'))
c=(d.get('choices') or [{}])[0]
sys.exit(0 if ((c.get('message') or {}).get('content') or '').strip() else 1)
" 2>/dev/null; then verdict="ok"; else verdict="EMPTY REPLY"; fi
    else
        verdict="HTTP $code"
        # Say why, briefly — a refusal here is often the honest answer.
        local why
        why=$(python3 -c "
import json
try: print(json.load(open('$body'))['error']['message'][:70])
except Exception: pass
" 2>/dev/null)
        [ -n "$why" ] && verdict="$verdict: $why"
    fi

    printf '%-34s %-7s %-12s %-5s %-26s %-8s %s\n' \
        "$label" "${secs}s" "${r:--}" "${segs:--}" "${nodes:--}" "${regions:--}" "$verdict"
}

echo "peer-path matrix: $MODEL via $BASE"
echo
printf '%-34s %-7s %-12s %-5s %-26s %-8s %s\n' SCENARIO TIME ROUTE SEGS NODES REGIONS RESULT
printf '%.0s-' {1..125}; echo

# 1. No instruction at all — every other row is read against this one.
#    ⚠ Do NOT expect `local` here even on a node holding every layer.
#    `gather_candidates` PRICES the local candidate against peer chains rather
#    than excluding it (.claude/rules/arch-scheduling.md), so a peer chain can
#    legitimately win. Measured 2026-09-17 on a node holding the whole model:
#    the baseline came back `distributed` over three segments. That is the
#    router working, not a fault.
run_case "baseline (no override)" ""

# 2. The case the project exists for: this node holds none of it.
run_case "every layer elsewhere" '{"pretend_local_holds":"none"}'

# 3. Keep only the first part here — a partial holding, which is the state most
#    real nodes are in for a model they have not finished fetching.
#    ⚠ This does NOT reliably produce a boomerang (first and last segments
#    local). Measured 2026-09-17: the search priced a single remote segment
#    below any chain through this node's one part and used it. Prompt privacy is
#    what forces both ends home, and it stands down for a model whose ends this
#    node does not hold — which releasing them is exactly what did.
run_case "first part here, rest elsewhere" '{"pretend_local_holds":"0"}'

# 4. Exclude each connected peer in turn, with everything else released. This is
#    what forces a DIFFERENT chain each time rather than the same best one: a
#    peer that holds the whole model wins every route until it is taken away.
peers=$(curl -s -m 15 -H "Authorization: Bearer $KEY" "$BASE/api/admin/peers" 2>/dev/null | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
for p in (d if isinstance(d,list) else d.get('peers',[])):
    n=p.get('node_id')
    if n: print(n[:16])
" 2>/dev/null)

if [ -z "$peers" ]; then
    echo
    echo "(no connected peers — rows 2-4 can only answer locally or fail, which is"
    echo " information about the swarm rather than about routing)"
else
    for p in $peers; do
        run_case "all remote, without ${p:0:8}" \
            "{\"pretend_local_holds\":\"none\",\"exclude_nodes\":[\"$p\"]}"
    done
fi

echo
echo "Reading this: NODES names who actually ran the layers, and it is the only"
echo "column that answers 'did this request go where I asked'. Expect the chosen"
echo "peer to vary between rows even where the excluded one was not serving — the"
echo "cost model re-prices on live latency, so run-to-run variance is normal and a"
echo "row differing from its neighbour is not on its own a finding."
