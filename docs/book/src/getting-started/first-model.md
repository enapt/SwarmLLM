# Your First Model

You need at least one AI model before you can chat.

## Easiest path: use what the swarm already runs (R141)

When the daemon starts and connects to peers, the Chat tab's empty state
shows three rows of models you can use **without setting anything up**:

- **"Available right now on the swarm"** — models the network can route
  inference to today. Click any chip → that model is selected and a
  fresh chat opens. Start typing.
- **"The swarm is gathering these"** — models with partial shard
  coverage; ready as the missing parts download.
- **"Popular models the swarm could adopt"** — popular HuggingFace
  releases the swarm hasn't picked up yet. Click one to open the HF
  browse pre-filtered to that repo so you can pick the quant variant
  and download it locally.

This is the recommended path for most users — the swarm handles the rest.

## Manual download via Dashboard

If you want to add a specific model the swarm doesn't already run:

1. Open the Dashboard at `http://localhost:8800`
2. Click **Browse HuggingFace** in the Models section
3. Search for a model (try `TinyLlama` for a small, fast model)
4. Choose a quantization variant (Q4_K_M recommended for most hardware)
5. Click **Add to node** — the node downloads its fair share of shards, and peers with auto-manage enabled auto-acquire the rest
6. The dashboard auto-refreshes when downloads complete (no page reload needed)

> **Heads-up**: models from trusted curators (meta-llama, mistralai,
> Qwen, bartowski, unsloth, …) replicate across the swarm faster than
> obscure publishers because R141 lowered their auto-promotion
> threshold. If you pick an unpopular model, expect weaker distribution.

## Download via CLI

```bash
# Smart distribution: node downloads its fair share, peers get the rest
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct.Q4_K_M.gguf", "peer_fair_share": true}'

# Or download specific shards manually:
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct.Q4_K_M.gguf", "shards": [0, 1, 2]}'
```

## Recommended Models by Hardware

| Hardware | Model | Size |
|---|---|---|
| Any (testing) | TinyLlama 1.1B Q4_K_M | ~700 MB |
| 8 GB RAM, no GPU | Qwen2.5-3B Q4_K_M | ~2 GB |
| 8 GB VRAM | Qwen2.5-7B Q4_K_M | ~4.5 GB |
| 16+ GB VRAM | Llama-3-13B Q4_K_M | ~7 GB |

## On-Demand Loading

You do not need to pre-load models into VRAM. When you send an inference request for a model whose shards are on disk but not loaded, SwarmLLM automatically loads the model on the fly. If VRAM is full, the least-recently-used model is evicted to make room. The first request to a cold model may take a few extra seconds while loading completes.

## Start Chatting

**Web UI:**
1. Click the **Chat** tab
2. Select your model from the dropdown
3. Type a message and press Enter

**CLI:**
```bash
./swarmllm chat
# Or with a specific model:
./swarmllm chat --model-name "qwen2.5-coder-7b"
```

**API:**
```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## What Are Shards?

Large AI models are split into smaller pieces called **shards** (~512 MB each) so they can be distributed across the network. Each shard contains a subset of the model's transformer layers. SwarmLLM handles this automatically — you just pick a model and download.

A node never needs all shards of a model. In distributed inference, each node loads only the layers it's responsible for.
