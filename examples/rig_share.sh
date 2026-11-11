# shellcheck shell=bash
# Sourced by the rigs that start several nodes and ask through one of them.
#
# Since v0.3.271 a computer has other computers run its requests only if it
# SHARES at least 2 model parts (`daemon::state::swarm_use`,
# `PARTS_TO_USE_THE_SWARM`) — every model's parts together. A rig's asking node
# is often exactly what that refuses: a coordinator holding 1 part of the model,
# only its header, or nothing, which answers 503 `sharing_required_error`
# before any peer is asked (the release gate's split step did, 2026-11-04).
#
# So the asking node also holds SHARE_MODEL whole: a small model of ANOTHER
# vocabulary (no split takes it for a drafter) that nothing asks for. The node
# shares as a real user's computer does and the topology under test is
# unchanged. SHARE_MODEL=<id> picks another; SHARE_MODEL=none leaves it out, to
# see the refusal itself.
#
#   rig_share_init <model under test> <models dir>   # once; exits 2 if unusable
#   rig_share <node data dir>                        # links SHARE_MODEL whole into its models/

# Test nodes: no heat hold refuses their work (#440) — the safety kit guards this machine.
export SWARMLLM_HEAT_GUARD="${SWARMLLM_HEAT_GUARD:-0}"

rig_share_init() {
  local target="$1"
  RIG_SHARE_MODELS_DIR="$2"
  if [ -z "${SHARE_MODEL:-}" ]; then
    case "$target" in
      tinyllama*) SHARE_MODEL=qwen2.5-coder-0.5b-instruct-q8-0 ;;
      *) SHARE_MODEL=tinyllama-1.1b-chat-v1.0.q4-k-m ;;
    esac
  fi
  [ "$SHARE_MODEL" = none ] && return 0
  if [ "$SHARE_MODEL" = "$target" ]; then
    echo "rig: SHARE_MODEL must differ from the model under test ($target)" >&2
    exit 2
  fi
  local n
  n=$(ls "$RIG_SHARE_MODELS_DIR/$SHARE_MODEL" 2>/dev/null | command grep -c '^shard_' || true)
  if [ "${n:-0}" -lt 2 ]; then
    echo "rig: SHARE_MODEL $SHARE_MODEL is not held whole (2+ parts) in $RIG_SHARE_MODELS_DIR — name another, or SHARE_MODEL=none" >&2
    exit 2
  fi
}

rig_share() {
  [ "${SHARE_MODEL:-none}" = none ] && return 0
  local src="$RIG_SHARE_MODELS_DIR/$SHARE_MODEL" dst="$1/models/$SHARE_MODEL" f
  mkdir -p "$dst"
  for f in "$src"/*; do
    # A hard link where the filesystem allows it (nothing writes into a model
    # held whole with downloads off); a copy where it does not.
    ln "$f" "$dst/" 2>/dev/null || cp "$f" "$dst/" || { echo "rig: cannot place $f" >&2; exit 2; }
  done
  echo "rig: $(basename "$1") holds $SHARE_MODEL whole as well, so it shares 2+ parts (swarm_use)" >&2
}
