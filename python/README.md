# swarmllm-client

Python client SDK for [SwarmLLM](https://github.com/enapt/SwarmLLM) — decentralized P2P LLM inference.

Provides both synchronous and asynchronous clients for the OpenAI-compatible API and SwarmLLM-specific admin/identity/pool endpoints.

## Installation

```bash
pip install swarmllm-client
```

Or from source:

```bash
cd python/
pip install -e .
```

## Quick Start

### Synchronous Client

```python
from swarmllm_client import SwarmLLM

client = SwarmLLM("http://localhost:8800", api_key="your-key")

# Chat completion (auto-selects model if omitted)
response = client.chat("Hello!", model="qwen2.5-coder-7b")
print(response.content)

# Streaming
for chunk in client.chat("Tell me a story", stream=True):
    print(chunk, end="", flush=True)

# Multi-turn with ChatMessage objects
from swarmllm_client import ChatMessage

messages = [
    ChatMessage(role="system", content="You are a helpful assistant."),
    ChatMessage(role="user", content="What is SwarmLLM?"),
]
response = client.chat_completion(messages, model="qwen2.5-coder-7b")
print(response.content)
print(f"Tokens used: {response.usage.total_tokens}")

# KV-cache sessions for multi-turn conversations
r1 = client.chat("Hello", model="m")
r2 = client.chat_completion(
    [{"role": "user", "content": "Follow up question"}],
    model="m",
    session_id=r1.session_id,
)
```

### Async Client

```python
import asyncio
from swarmllm_client import AsyncSwarmLLM

async def main():
    async with AsyncSwarmLLM("http://localhost:8800", api_key="your-key") as client:
        response = await client.chat("Hello!")
        print(response.content)

        # Streaming
        stream = await client.chat("Tell me a story", stream=True)
        async for chunk in stream:
            print(chunk, end="", flush=True)

asyncio.run(main())
```

### Admin Endpoints

```python
client = SwarmLLM("http://localhost:8800", api_key="your-key")

# Node stats
stats = client.admin.stats()
print(f"Node: {stats.node_id}, Peers: {stats.peers_connected}")

# Connected peers
for peer in client.admin.peers():
    print(f"{peer.node_id} healthy={peer.healthy} gpu={peer.gpu}")

# Credit balance
credits = client.admin.credits()
print(f"Balance: {credits.balance} ({credits.tier})")

# Shard storage
storage = client.admin.shard_storage()
print(f"Disk: {storage.disk_usage_bytes / 1e9:.1f} GB")

# Download shards from HuggingFace
client.admin.hf_download_shards(
    repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
    filename="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    shards=[0, 1],
)

# Lock a shard from auto-pruning
client.admin.lock_shard("my-model", shard_index=0, locked=True)
```

### Identity & Pool

```python
# Nicknames
client.identity.set_nickname("my-cool-node")
print(client.identity.get_nickname())

# Leaderboard
for entry in client.identity.leaderboard():
    print(entry)

# Device pools
client.pool.create("my-gpu-pool")
client.pool.invite("node-id-abc123")
```

### Embeddings

```python
result = client.embeddings("Hello world", model="embedding-model")
print(result.data)  # [[0.1, 0.2, ...]]
```

## Using the OpenAI SDK Directly

SwarmLLM exposes an OpenAI-compatible API, so you can use the official `openai` Python package directly:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8800/v1",
    api_key="your-swarmllm-api-key",
)

response = client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)

# Streaming works too
for chunk in client.chat.completions.create(
    model="qwen2.5-coder-7b",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

The `swarmllm-client` package adds value beyond the OpenAI SDK with:
- Admin API access (node stats, shard management, downloads, pools)
- Identity and pool management
- Typed response objects for SwarmLLM-specific data
- KV-cache session tracking

## Error Handling

```python
from swarmllm_client import SwarmLLM, SwarmLLMError

client = SwarmLLM("http://localhost:8800")
try:
    client.chat("Hello")
except SwarmLLMError as e:
    print(f"API error {e.status_code}: {e.message}")
```

## API Reference

### SwarmLLM / AsyncSwarmLLM

| Method | Description |
|---|---|
| `chat(prompt, ...)` | Single-turn chat (convenience) |
| `chat_completion(messages, ...)` | Full chat completion with all parameters |
| `embeddings(input, ...)` | Create text embeddings |
| `models()` | List available models |
| `status()` | Node status |
| `health()` | Health check |
| `health_ready()` | Readiness probe |
| `metrics()` | Prometheus metrics |

### admin

| Method | Description |
|---|---|
| `stats()` | Node statistics and hardware info |
| `peers()` | Connected peers |
| `credits()` | Credit balance and tier |
| `api_key()` | Retrieve current API key |
| `models()` | Models with shard status |
| `model_status(model_id)` | Model acquisition progress |
| `add_model(model_id)` | Trigger model acquisition from network |
| `model_metadata(model_id)` | GGUF metadata browser |
| `shard_storage()` | Per-model storage breakdown |
| `config()` / `update_config(...)` | Read/update daemon configuration |
| `reload_config()` | Hot-reload operational parameters |
| `hf_search(query)` | Search HuggingFace for GGUF models |
| `hf_probe(repo_id, filename)` | Probe remote GGUF for shard info |
| `hf_download_shards(...)` | Download specific shards |
| `hf_source(model_id)` | HF source info for a model |
| `downloads()` | Active download queue |
| `cancel_download(model_id)` | Cancel a download |
| `lock_shard(model_id, index)` | Lock shard from auto-pruning |
| `delete_shard(model_id, index)` | Delete a single shard |
| `delete_model(model_id)` | Remove a model and all shards |
| `get_auto_manage(model_id)` | Per-model auto-manage policy |
| `set_auto_manage(model_id, policy)` | Update auto-manage policy |
| `network_map()` | Network topology heatmap data |
| `network_code()` | Get shareable invite code |
| `join_network(code)` | Join network via invite code |
| `schedule()` / `update_schedule(...)` | Resource schedule |
| `prune_history()` | Recent auto-prune events |
| `shutdown()` | Gracefully shut down the node |

### identity

| Method | Description |
|---|---|
| `get_nickname()` | Get node nickname |
| `set_nickname(name)` | Set node nickname |
| `delete_nickname()` | Remove nickname |
| `leaderboard()` | Network credit leaderboard |
| `peers()` | Peer identity directory |

### pool

| Method | Description |
|---|---|
| `state()` | Current pool membership |
| `create(name)` | Create a device pool |
| `invite(node_id)` | Invite a node |
| `accept(invitation_id)` | Accept an invitation |
| `leave()` | Leave the pool |

## Development

```bash
pip install -e ".[dev]"
pytest
```

## Requirements

- Python 3.9+
- `requests` (sync client)
- `aiohttp` (async client)

## License

MIT
