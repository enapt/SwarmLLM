# Your First Model

You need at least one AI model before you can chat.

## Easiest path: use what the swarm already runs

When SwarmLLM starts and connects to other computers, the Chat tab shows
models you can use **without downloading anything**:

- **"Available right now on the swarm"** — click one and a fresh chat opens.
  Start typing.
- **"The swarm is gathering these"** — the swarm has some of the parts;
  they'll be ready once the rest arrive.
- **"Popular models the swarm could adopt"** — popular models nobody is
  running yet. Clicking one opens the model search so you can pick a version.

When the Chat tab says **"No models available yet"**, its **Get shared test
model** button downloads a small model everyone can use, so you can start
chatting straight away.

## Download a model yourself

**Recommended: a shared test model.** In **Settings → Testing & Diagnostics**,
click **Get my share** (this computer downloads only the parts it should hold)
or **Get all of it** (so this computer can answer on its own). From a
terminal, the same thing is:

```bash
./swarmllm get-model             # list the test models
./swarmllm get-model standard    # Llama 3.2 3B
```

**Any other model:** open the **Models** tab and choose **Search HuggingFace**
(or click **Find Models** on the Dashboard), search for a model — try
`TinyLlama` for a small, fast one — and check its badge: **✓ Runs locally**,
**○ Swarm computers only** or **⚠ Too large for your swarm**. The **Download**
button starts the download.

> **Heads-up**: models from well-known publishers (meta-llama, mistralai,
> Qwen, bartowski, unsloth, …) spread across the swarm faster than models
> from unknown publishers. If you pick an unpopular model, fewer other
> computers will pick up its parts.

## Download via CLI

```bash
# Every part, so this computer can answer on its own
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "all_shards": true}'

# Or one part as this computer's share; other computers may pick up the rest
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "peer_fair_share": true}'

# Or download specific parts (shards) by number:
curl -X POST http://localhost:8800/api/admin/hf/download-shards \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF", "filename": "qwen2.5-coder-7b-instruct-q4_k_m.gguf", "shards": [0, 1, 2]}'
```

## Recommended Models by Hardware

| Hardware | Model | Size |
|---|---|---|
| Any (testing) | TinyLlama 1.1B Q4_K_M | ~700 MB |
| 8 GB RAM, no GPU | Qwen2.5-3B Q4_K_M | ~2 GB |
| 8 GB VRAM | Qwen2.5-7B Q4_K_M | ~4.5 GB |
| 16+ GB VRAM | Qwen2.5-14B Q4_K_M | ~9 GB |

## On-Demand Loading

You do not need to pre-load models into VRAM. When you send a request for a model whose parts are on disk but not loaded, SwarmLLM automatically loads the model on the fly. If VRAM is full, the least-recently-used model is evicted to make room. The first request to a cold model may take a few extra seconds while loading completes.

## Start Chatting

**Web UI:**
1. Click the **Chat** tab
2. Select your model from the dropdown
3. Type a message and press Enter

**CLI:**
```bash
./swarmllm chat
# Or with a specific model:
./swarmllm chat --model qwen2.5-coder-7b-instruct-q4-k-m
```

**API:**
```bash
curl http://localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder-7b-instruct-q4-k-m",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## What Are Model Parts?

Large AI models are split into smaller **parts** (called *shards* in settings and logs, ~512 MB each) so they can be spread across the network. Each part holds some of the model's layers. SwarmLLM handles this automatically — you just pick a model and download.

A computer never needs every part of a model. When a model is split across computers, each one loads only the layers it's responsible for.
