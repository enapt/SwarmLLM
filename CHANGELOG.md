# Changelog

All notable changes to SwarmLLM are documented here.

## [Unreleased]

### Claude Code Integration (Phase 13)
- **Full Anthropic Messages API** (`POST /v1/messages`) — complete Claude Code compatibility
  - `tools`, `tool_choice`, `metadata`, `thinking` (extended thinking) request fields
  - `tool_use`, `tool_result`, `thinking`, `redacted_thinking` content blocks
  - `cache_control` on system blocks (Anthropic prompt caching)
  - Full pass-through to Anthropic cloud (all fields preserved including tools and thinking)
  - Anthropic→OpenAI translation proxy for non-Claude cloud models (GPT-4o, DeepSeek, etc.)
  - Tool calls and thinking blocks converted to text for local GGUF inference
  - `ResponseContentBlock` refactored from struct to enum (Text, ToolUse, Thinking variants)
- **MCP `compare` tool** — send same prompt to multiple models concurrently (up to 10)
  - Returns side-by-side results with `content`, `latency_ms`, `input_tokens`, `output_tokens`, `status`
  - Supports local, network, and cloud models in same comparison
  - Routes through `/v1/messages` for consistent routing logic
- **Claude Code as client**: `ANTHROPIC_BASE_URL=http://localhost:8800 claude --model qwen2.5-coder-7b`
- **Model Compare dashboard page** — side-by-side multi-model comparison UI with streaming
- 6 new unit tests (tool_use, tool_result, thinking, tools request, response serialization, internal conversion)
- 665 tests passing (597 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E)

### Published Benchmark Data
- **GPU (RTX 3070 8GB):** TinyLlama 1.1B 27.2 tok/s, Gemma-2 2B 20.6 tok/s, Phi-3.5 3.8B 46.4 tok/s, Qwen2.5 7B 29.0 tok/s
- **CPU (Ryzen 7 5800H):** TinyLlama 4.2 tok/s, Gemma-2 3.5 tok/s, Phi-3.5 1.8 tok/s, Qwen2.5 2.4 tok/s
- GPU speedups: 6.5x to 25.8x depending on architecture
- Methodology: 100 output tokens, 3-run average, single model loaded, Q4_K_M quantization

## [0.1.0-alpha.1] — 2026-03-07

First public release. Single Rust binary (~31MB) for decentralized P2P LLM inference.

### Inference Engine
- **11 model architectures**: Llama, Llama 4, Qwen2, Qwen 3.5 (hybrid SSM+attention), Gemma/2, Phi-3, Mistral, Starcoder2, DeepSeek-V2/V3 (MoE+MLA), GLM-4
- **4 architectures verified** with real models: Llama (TinyLlama-1.1B), Qwen2 (Qwen2.5-Coder-7B), Phi-3 (Phi-3.5-mini), Gemma2 (Gemma-2-2B-IT)
- **Distributed inference** verified on 2-node real LAN (WSL2 laptop + Proxmox server) with 5 models, crash recovery, auto-reconnect
- **Tensor parallelism** via AllReduce (star topology) with RTT-based LAN peer detection
- **VLM support**: LLaVA-v1.5-7B verified end-to-end (CLIP vision encoder + correct fine-tuned text model from second-state/Llava-v1.5-7B-GGUF), distributed mmproj, chat UI image upload (camera button, paste, drag-drop)
- **LoRA adapters**: per-request loading, verified with Qwen2.5-Coder-7B + rank-16 adapter
- **Speculative decoding** with draft model + rejection sampling
- **Cross-request batching** (GPU batch tensors, configurable `max_batch_size`)
- **Multi-turn KV-cache** with session reuse, cross-request prefix caching, chunked prefill
- **Flash attention** (CPU + GPU) and **paged attention** (CUDA block pool)
- **Structured output**: ResponseFormat API with JSON grammar state machine + schema validation
- **Sampling**: temperature, top-k, top-p, frequency/presence penalty, stop sequences

### API & Compatibility
- **OpenAI-compatible API**: `POST /v1/chat/completions` with streaming (SSE), `tool_calls`, `tool_choice`, `logprobs`, `top_logprobs`, Tool role
- **Anthropic Messages API**: `POST /v1/messages` — full Claude Code compatibility (tools, tool_choice, thinking, cache_control, metadata)
- **MCP server** at `/mcp` — `chat`, `models`, and `compare` (multi-model comparison) tools for Claude Code, Cursor, and MCP-compatible agents
- **12 cloud provider fallback**: OpenAI, Anthropic, DeepSeek, Mistral, Groq, NVIDIA NIM, Cerebras, SambaNova, Fireworks, Together, DeepInfra, Moonshot/Kimi
- **Hidden states API**: `/v1/internal/hidden-states` for research (activation inspection, adapter insertion)
- **Embeddings**: `POST /v1/embeddings`
- **~62 admin REST routes** for dashboard, config, model management, downloads, providers
- **WebSocket** live updates (2s stats + prune event notifications)
- **Prometheus metrics** at `/metrics` (6 gauges + histogram)

### SDKs & Integrations
- **Python SDK**: `pip install swarmllm-client` — sync + async clients, streaming
- **JavaScript/TypeScript SDK**: zero-dependency, streaming support
- **LangChain integration**: `ChatSwarmLLM` provider
- **LlamaIndex integration**: `SwarmLLM` provider
- **Benchmark CLI**: `swarmllm bench` — sequential latency + concurrent throughput, JSON output

### Networking
- **P2P**: libp2p 0.55 with TCP+Yamux (primary) and QUIC transport
- **5-layer discovery**: mDNS (LAN), persistent peer cache (redb), encrypted invite codes, peer exchange (PEX), Kademlia DHT
- **NAT traversal**: libp2p relay circuits + DCUtR hole punching
- **GossipSub**: 6 topics for shard announcements, credit gossip, health, governance
- **Unified protocol**: `/swarmllm/1.0.0` — JSON control messages + binary tensor payloads (type-tag byte)
- **Wire compression**: zstd for tensor payloads

### Security
- **E2E encryption**: X25519 key exchange + ChaCha20-Poly1305 symmetric encryption
- **Forward secrecy**: ephemeral re-keying with key rotation
- **Sealed gossip**: all gossip messages authenticated (no plaintext fallback)
- **Replay protection**: nonce tracking + rejection
- **Shard integrity**: BLAKE3 content hash verified on every load
- **API auth**: Bearer token with auto-generation, loopback-only key retrieval
- **Provider key security**: at-rest encryption (AES-GCM), zeroize on drop, log scrubbing
- **Content-Security-Policy** header, IP-based rate limiting, CORS lockdown, SSRF protection
- **KV-cache privacy mode**: configurable per-session data isolation

### Model Management
- **Shard-only operation**: nodes download individual shards (~512MB each), never need a full model
- **HuggingFace integration**: search, browse, byte-range shard downloads with resume/retry
- **VRAM-aware auto shard management**: rarity-scored acquisition, popularity-based scoring
- **Smart shard pruning**: auto-remove over-replicated shards based on demand, resource pressure, and region diversity
- **Per-shard lock/pin** and per-model prune toggle
- **BLAKE3 integrity verification** on every shard load

### Credit System
- **Credit ledger**: earn credits by serving inference, hosting shards, seeding data
- **4 priority tiers**: Platinum (top 10%), Gold (top 30%), Silver (positive), Bronze (zero/negative)
- **Dual-signed transactions**: Ed25519 signatures from both parties
- **Credit escrow** for large requests
- **Anti-gaming**: rate limits, spot-check verification, subnet clustering detection
- **Sybil resistance**: trust scoring with decay, reputation tracking

### Identity & Pools
- **Ed25519 cryptographic identity** per node
- **Nicknames** with leaderboard
- **Device pools**: multi-device credit pooling with dual-signature invitation protocol

### Frontend
- **Embedded web dashboard** (vanilla HTML/CSS/JS, no build step, < 200KB)
- **4-step setup wizard** for first-run experience
- **Chat interface**: multi-turn + streaming, switchable Linear/Messenger layout, image upload (camera button, paste, drag-drop) for VLM models
- **Model browser**: HuggingFace search, shard grid visualization, download progress
- **Network map**: peer visualization with region grouping
- **Mobile-responsive** layout with dark theme
- **Reasoning model support**: DeepSeek R1 think token rendering

### Operations
- **Single binary**: ~31MB, zero runtime dependencies
- **CLI**: `run`, `status`, `chat`, `bench`, `peers`, `test-split`, `version`
- **Config priority**: CLI flags > env vars (`SWARMLLM_` prefix) > config.toml > defaults
- **Config hot-reload** via SIGHUP or API
- **Graceful shutdown**: SIGTERM handler with subsystem drain
- **Auto-updater**: checks GitHub Releases, downloads + self-replaces with restart prompt
- **JoinSet task supervisor**: automatic restart-on-crash for all 10 subsystems
- **Database**: redb v3 (embedded, ACID, ~15% faster than v2)

### Platform Support
- Linux x86_64 (CPU + CUDA + ROCm)
- macOS aarch64 Apple Silicon (Metal)
- macOS x86_64 Intel (CPU)
- Windows x86_64 (CPU + CUDA)

### Model Loading
- **Auto-extract gguf_header.bin** from shard_000.bin when header is missing (daemon pre-pass)
- **Single-GGUF manifest**: Full GGUF files stored as shard_000.bin generate 1-shard manifests (not split into logical shards)
- **Single-shard mmap fallback**: Models with 1 shard and no tensor entries load via mmap instead of ShardReader
- **Probed flag fix**: Models with gguf_header.bin correctly show as probed in admin API

### Test Suite
- 659 tests: 591 unit + 22 integration + 31 module + 14 yamux + 1 VLM E2E
- All passing, clippy clean, rustfmt clean
- CI: GitHub Actions (fmt → clippy → test → build)

[0.1.0-alpha.1]: https://github.com/enapt/SwarmLLM/releases/tag/v0.1.0-alpha.1
