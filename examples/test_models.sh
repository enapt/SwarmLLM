# shellcheck shell=bash
# Sourced by the test scripts: which folder holds a model's parts.
#
# A developer's own node folder (~/.local/share/swarmllm/models) holds what that
# node hosts, and a node with its storage full gives back what nobody runs. The
# models the release check and the rigs are written against (Llama 3.2 3B,
# TinyLlama, Qwen2.5-Coder 7B, …) are older than what the swarm runs now, so they
# can be kept apart from the node instead, in SWARMLLM_TEST_MODELS
# (~/.local/share/swarmllm-test-models by default, the same layout:
# <model id>/shard_*.bin, manifest.json, gguf_header.bin). Hard links put a
# model there without a second copy: `cp -al ~/.local/share/swarmllm/models/<id>
# ~/.local/share/swarmllm-test-models/` — then the node may delete its own.
#
#   models_dir_for <model id>   # the test folder if it holds the model, else the node's
#
# Without a test folder every script reads the node's folder, as before.

SWARMLLM_TEST_MODELS="${SWARMLLM_TEST_MODELS:-$HOME/.local/share/swarmllm-test-models}"
SWARMLLM_NODE_MODELS="${SWARMLLM_NODE_MODELS:-$HOME/.local/share/swarmllm/models}"

models_dir_for() {
  if [ -n "${1:-}" ] && [ -d "$SWARMLLM_TEST_MODELS/$1" ]; then
    printf '%s\n' "$SWARMLLM_TEST_MODELS"
  else
    printf '%s\n' "$SWARMLLM_NODE_MODELS"
  fi
}
