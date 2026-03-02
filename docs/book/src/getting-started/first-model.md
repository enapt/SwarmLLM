# Your First Model

You need at least one AI model before you can chat.

## Download via Dashboard

1. Open the Dashboard at `http://localhost:8800`
2. Click **Browse HuggingFace** in the Models section
3. Search for a model (try `TinyLlama` for a small, fast model)
4. Click **Download** — shards turn green as they complete

## Download via CLI

```bash
# Find your API key in the daemon startup logs
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"}'
```

## Recommended Models by Hardware

| Hardware | Model | Size |
|---|---|---|
| Any (testing) | TinyLlama 1.1B Q4_K_M | ~700 MB |
| 8 GB RAM, no GPU | Qwen2.5-3B Q4_K_M | ~2 GB |
| 8 GB VRAM | Qwen2.5-7B Q4_K_M | ~4.5 GB |
| 16+ GB VRAM | Llama-3-13B Q4_K_M | ~7 GB |

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
