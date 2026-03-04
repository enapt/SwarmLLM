# SwarmLLM Diagnostic Instrumentation Guide

All diagnostic log lines are prefixed with `DIAG:` for easy filtering.

## Quick Start — Filtering Diagnostic Logs

```bash
# Run with debug logging, filter to DIAG lines only
cargo run -- run -vv 2>&1 | grep "DIAG:"

# Full trace (very verbose) — includes encryption nonce details
cargo run -- run -vvv 2>&1 | grep "DIAG:"

# Filter to specific subsystem
cargo run -- run -vv 2>&1 | grep "DIAG:.*encrypt"    # Encryption issues
cargo run -- run -vv 2>&1 | grep "DIAG:.*segment"     # Pipeline segment timing
cargo run -- run -vv 2>&1 | grep "DIAG:.*connection"   # Connection lifecycle
cargo run -- run -vv 2>&1 | grep "DIAG:.*LayerForward" # Tensor forward path
cargo run -- run -vv 2>&1 | grep "DIAG:.*SSE"          # SSE streaming path
cargo run -- run -vv 2>&1 | grep "DIAG:.*KV-cache"     # KV-cache hit/miss
cargo run -- run -vv 2>&1 | grep "DIAG:.*split stream"  # Split model decode loop
cargo run -- run -vv 2>&1 | grep "DIAG:.*execute_request" # End-to-end request timing
cargo run -- run -vv 2>&1 | grep "DIAG:.*decompress"    # Tensor compression
```

## End-to-End Request Trace

Every inference request gets a `request_id` (UUID) that appears in logs across all subsystems. To trace a single request:

```bash
cargo run -- run -vv 2>&1 | grep "request_id=<UUID>"
```

### Request Lifecycle (log points)

1. **API entry** → `Queued inference request` (router.rs)
2. **Dispatch** → `DIAG: dispatch_single starting inference` (router.rs)
3. **Pipeline assembly** → `DIAG: pipeline assembled` with `segments`, `standbys`, `schedule_ms` (router.rs)
4. **Forward start** → `DIAG: starting forward_through_segments` with `seq_num`, `index_pos`, `activation_bytes` (pipeline.rs)
5. **Tensor forward send** → `Sent tensor forward` with `is_connected`, `total_connections`, `pending_tensor_count`, `outbound_id` (manager.rs)
6. **Codec write** → `DIAG: codec write_request start/done` with `frame_len` (protocol.rs)
7. **Encryption (if enabled)** → `DIAG: encrypting tensor forward` with `aad_len`, `has_session` (manager.rs)
8. **Remote receive** → `DIAG: codec read_request header` with `tag`, `len` (protocol.rs)
9. **Inbound dispatch** → `DIAG: inbound TensorPayload request` → `DIAG: stored ResponseChannel` (manager.rs)
10. **Dispatcher** → `DIAG: dispatcher received LayerForward` with `seq`, `layer_range`, `activation_bytes` (daemon.rs)
11. **Local execution** → `DIAG: processing LayerForward locally` with `elapsed_ms` (daemon.rs)
12. **Split model forward** → `DIAG: SplitModel forward pass complete` with `forward_ms`, `seq_len`, `num_layers` (split.rs)
13. **Result send** → `DIAG: LayerForward processed, sending result back` (daemon.rs)
14. **Response write** → `DIAG: codec write_response start/done` with `frame_len` (protocol.rs)
15. **ResponseSent event** → `DIAG: ResponseSent event — response written to wire` (manager.rs)
16. **Response read** → `DIAG: codec read_response header` with `tag`, `len` (protocol.rs)
17. **Response received** → `DIAG: received response` with `kind`, `was_tensor_forward`, `pending_tensor_out` (manager.rs)
18. **Response dispatch** → `DIAG: received TensorPayload response` (manager.rs)
19. **Result delivery** → `DIAG: dispatcher received LayerResult` → `DIAG: LayerResult delivered to pipeline` (daemon.rs)
20. **Forward complete** → `DIAG: forward_through_segments returned OK` with `fwd_ms`, `tokens`, `activations_bytes` (pipeline.rs)
21. **Local segment** → `DIAG: local segment complete` with `segment_ms`, `activation_bytes` (pipeline.rs)
22. **Remote segment** → `DIAG: remote segment complete` with `segment_ms`, `activation_bytes` (pipeline.rs)
23. **Segment result** → `DIAG: segment result received` with `elapsed_ms` (pipeline.rs)
24. **Pipeline complete** → `DIAG: forward_through_segments completed` with `pipeline_ms` (pipeline.rs)
25. **Execute complete** → `DIAG: execute_request completed successfully` with `schedule_ms`, `execute_ms`, `total_ms` (router.rs)
26. **Completion** → `DIAG: inference completed` with `elapsed_ms`, `prompt_tokens`, `completion_tokens` (router.rs)

### Network Event Diagnostics

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: processing swarm event` — event type name for every swarm event | manager.rs |
| DEBUG | `DIAG: handling outbound command` — command type for every outbound command | manager.rs |
| INFO  | `DIAG: OutboundFailure` — `is_connected`, `pending_tensor_out`, `pending_channels` | manager.rs |
| WARN  | `DIAG: InboundFailure` — `pending_channels` | manager.rs |
| INFO  | `DIAG: ResponseSent event` — confirms response written to wire | manager.rs |

### Failure Paths

- **Timeout** → `DIAG: segment TIMED OUT after 30s` (pipeline.rs)
- **Outbound failure** → `DIAG: OutboundFailure` → `Tensor forward OutboundFailure — notifying pipeline` (manager.rs)
- **Inbound failure** → `DIAG: InboundFailure — response send may have failed` (manager.rs)
- **Decryption fail** → `DIAG: decrypt FAILED — possible AAD mismatch` (manager.rs)
- **No standby** → `DIAG: NO standby available for failed segment` (pipeline.rs)
- **Client disconnect** → `DIAG: result_tx receiver dropped` (router.rs)
- **Channel drop** → `DIAG: LayerResult delivered but pipeline receiver DROPPED` (daemon.rs)
- **No pending channel** → `DIAG: No pending channel for LayerResult — already timed out or duplicate` (daemon.rs)
- **Streaming done event** → `DIAG: streaming done_event send failed` (router.rs)

## SSE Streaming Diagnostics

All three streaming paths are instrumented with timing and error reporting:

### Distributed Pipeline Streaming (split_non_stream_response)

| Level | What | Where |
|-------|------|-------|
| WARN  | `DIAG: SSE role delta send failed` — client disconnected before stream started | openai.rs |
| WARN  | `DIAG: SSE final text delta send failed` — client disconnected on last token | openai.rs |
| WARN  | `DIAG: SSE finish delta send failed` — client disconnected at finish | openai.rs |
| WARN  | `DIAG: SSE token delta send failed` — client disconnected mid-stream | openai.rs |
| DEBUG | `DIAG: SSE stream no finish event from pipeline` — falling back to result_rx | openai.rs |
| WARN  | `DIAG: SSE fallback content/finish/error send failed` — various fallback failures | openai.rs |
| WARN  | `DIAG: SSE result_rx channel dropped` — pipeline task died | openai.rs |
| INFO  | `DIAG: SSE distributed stream completed` — `elapsed_ms`, `token_count` | openai.rs |

### Split Model Streaming (split_stream_response)

| Level | What | Where |
|-------|------|-------|
| WARN  | `DIAG: split stream model not found` — model evicted during request | openai.rs |
| ERROR | `DIAG: split stream tokenize/prefill/forward/sample failed` — compute error | openai.rs |
| INFO  | `DIAG: split stream prefill complete` — `prefill_ms`, `prompt_tokens` | openai.rs |
| INFO  | `DIAG: split stream decode loop complete` — `decode_ms`, `tok_per_sec`, `prefill_ms` | openai.rs |
| WARN  | `DIAG: split stream client disconnected mid-decode` — `token_count`, `elapsed_ms` | openai.rs |
| INFO  | `DIAG: split stream completed` — `elapsed_ms`, `token_count` | openai.rs |

### Local Executor Streaming (stream_response)

| Level | What | Where |
|-------|------|-------|
| WARN  | `DIAG: local stream role delta send failed` — client disconnected early | openai.rs |
| WARN  | `DIAG: local stream token send failed` — channel full or client disconnected | openai.rs |
| ERROR | `DIAG: local stream generate_stream error` — executor error | openai.rs |
| INFO  | `DIAG: local stream completed` — `elapsed_ms`, `token_count` | openai.rs |

## Encryption Diagnostics

The encrypted tensor path logs at multiple levels:

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: encrypting tensor forward` — AAD length, session state | manager.rs |
| DEBUG | `DIAG: decrypting tensor` — AAD length, sealed length, session existence | manager.rs |
| TRACE | `DIAG: seal() success` — nonce counter, ciphertext length | session.rs |
| TRACE | `DIAG: open() decryption success` — nonce, plaintext length | session.rs |
| ERROR | `DIAG: seal() failed` — full context on encryption failure | manager.rs |
| ERROR | `DIAG: decrypt FAILED` — AAD mismatch, key mismatch, or corruption | manager.rs, session.rs |
| ERROR | `DIAG: open() decryption FAILED` — nonce state, AAD/sealed lengths | session.rs |

### Common Encryption Failures

**AAD Mismatch**: The sender and receiver construct AAD from the cleartext header fields (uuid + seq + idx_pos + fmt + layer_range + model_id). If these don't match byte-for-byte, decryption fails. Look for `aad_len` differences between send and receive logs.

**No Session**: The sender has an encryption session but the receiver doesn't (or vice versa). Check `has_session` in logs. Sessions are established via ECDH key exchange during peer discovery.

**Nonce Replay**: If `Rejecting replayed nonce` appears, a duplicate or out-of-order message was received. This can happen with connection flapping.

## Transport Layer

SwarmLLM uses dual transport: **TCP** (primary, Noise+Yamux) and **QUIC** (fallback).

### Port Layout

| Service | Port | Protocol |
|---------|------|----------|
| HTTP API (Axum) | `port` (default 8800) | TCP |
| P2P TCP (Noise+Yamux) | `port + 10` (default 8810) | TCP |
| P2P QUIC | `port` (default 8800) | UDP |

TCP P2P uses `port+10` to avoid conflicting with the Axum HTTP server on the same TCP port.

### Why TCP Primary

QUIC substream negotiation on WSL2 (and potentially other virtualized networks) can take **14-25 seconds per substream**. Since `request_response` serializes outbound requests through a single substream at a time, this creates a fatal bottleneck — tensor forwards queue behind health pings and never reach the codec before the 30-second pipeline timeout.

TCP+Yamux substream opening is sub-millisecond, enabling per-token round trips of ~20-26ms for distributed inference.

### Bootstrap with TCP

When connecting nodes, use TCP addresses for bootstrap:

```bash
# Node 1 on port 8800 (TCP P2P on 8810)
swarmllm run -p 8800

# Node 2 bootstraps to Node 1's TCP P2P address
swarmllm run -p 8801 --bootstrap /ip4/<node1-ip>/tcp/8810
```

## Connection Diagnostics

### Local Multi-Node Testing

When running multiple nodes on the same machine (localhost), connection management is more complex:

- mDNS discovers the local node on multiple interfaces (loopback, LAN, WSL)
- Both sides dial simultaneously, creating multiple connection attempts
- `max_established_per_peer=1` — prevents request_response round-robin routing to dead connections
- Identify handler adds only the **connected** address to Kademlia (not all listen_addrs)
- `connection_addrs: HashMap<ConnectionId, Multiaddr>` tracks which address each connection uses

Look for `is_loopback=true` in `DIAG: connection established` logs to confirm same-machine connections.

### Connection Lifecycle

```
DIAG: connection established  — peer_id, connection_id, count, remote_addr, is_loopback, is_dialer, total_established, total_peers, pending_tensor_forwards
DIAG: connection closed        — peer_id, cause, remaining, pending_tensor_forwards, affected_request_ids, total_peers
```

If `pending_tensor_forwards > 0` when a connection closes, those requests will get `OutboundFailure` and the pipeline will attempt failover.

## Tensor Compression Diagnostics

| Level | What | Where |
|-------|------|-------|
| ERROR | `DIAG: request tensor decompression failed` — zstd decompress error | protocol.rs |
| ERROR | `DIAG: response tensor decompression failed` — zstd decompress error | protocol.rs |
| DEBUG | `DIAG: request tensor decompressed` — `compressed_len`, `decompressed_len`, `ratio` | protocol.rs |
| DEBUG | `DIAG: response tensor decompressed` — `compressed_len`, `decompressed_len`, `ratio` | protocol.rs |

## KV-Cache Diagnostics

### Multi-turn Session Cache (kv_cache.rs)

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: KV-cache MISS — no multi-turn session found` — `total_sessions`, `total_multi_turn` | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — internal session evicted` — session removed from store | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — session expired` — `elapsed_secs`, `ttl_secs` | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — pipeline degraded` — `missing` nodes, `total_holders` | kv_cache.rs |
| INFO  | `DIAG: KV-cache MISS — prompt prefix mismatch` — `cached_prompt_len`, `new_prompt_len` | kv_cache.rs |
| INFO  | `DIAG: KV-cache HIT — skipping prefill` — `start_pos`, `cached_tokens`, `cache_holders` | kv_cache.rs |

### Per-Request KV-Cache Store (split.rs)

| Level | What | Where |
|-------|------|-------|
| INFO  | `DIAG: KV-cache store cleanup — expired entries removed` — `removed`, `remaining` | split.rs |

## Split Model Forward Pass Diagnostics

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: SplitModel forward pass complete` — `forward_ms`, `seq_len`, `num_layers`, `is_first`, `is_last`, `kv_offset` | split.rs |

For per-token decode analysis, combine the forward pass timing with the decode loop timing from `DIAG: split stream decode loop complete` which reports `tok_per_sec`.

## Performance Diagnostics

### Identifying Slow Requests

The `elapsed_ms` field appears at multiple points:

1. `DIAG: SplitModel forward pass complete` — time for a single forward pass (compute only)
2. `DIAG: local segment complete` — time for a local pipeline segment
3. `DIAG: remote segment complete` — time for a remote pipeline segment (network + compute)
4. `DIAG: segment result received` — time for a single segment (network + compute)
5. `DIAG: forward_through_segments completed` — total pipeline forwarding time
6. `DIAG: execute_request completed successfully` — `schedule_ms` (pipeline assembly) + `execute_ms` (pipeline execution)
7. `DIAG: split stream prefill complete` — time for prefill only
8. `DIAG: split stream decode loop complete` — decode time with `tok_per_sec`
9. `DIAG: inference completed` — total end-to-end time

If `schedule_ms` is high, the bottleneck is pipeline assembly. If `execute_ms` is high but individual `segment_ms` values are low, the bottleneck is inter-segment overhead. If a single segment is slow, check that node's compute or network latency.

### 27-Second Response Times

Common causes:
- **Timeout-then-failover**: A segment times out at 30s, then failover succeeds quickly → ~30s total. Check for `DIAG: segment TIMED OUT` followed by `DIAG: failing over to standby`.
- **Connection not established**: Tensor sent to a peer that's not connected. Check `is_connected=false` in `Sent tensor forward` logs.
- **Encryption failure + fallback**: Encrypted send fails, falls back to plaintext, which also fails. Check for `DIAG: seal() failed` logs.
- **Channel backpressure**: Result arrives but the dispatcher channel is full. Check for `Outbound channel full, dropping tensor result`.
- **SSE fallback path**: If `DIAG: SSE stream no finish event from pipeline` appears, the streaming token channel broke and the system fell back to waiting for the full result — check pipeline errors above.

## Health Monitor Diagnostics

```
DIAG: removing stale peers      — stale_count, total_peers, active_pipelines
DIAG: cleaning up stale pending_layer_results — count, total_pending, request_ids
DIAG: cleaning up stale streaming_token_txs   — count, total_streaming
```

If stale channel cleanup is happening frequently, requests are timing out or being abandoned before results arrive.

## Network Subsystem Diagnostics

### Gossip Decryption Fallback

```
DIAG: gossip decryption failed, plaintext fallback succeeded
```

This is normal during bootstrapping (new nodes don't have the gossip seal key yet). If it persists after the network is established, it indicates a key rotation issue.

### Bootstrap Failures

```
DIAG: bootstrap dial failed            — addr, peer_id, error
DIAG: Kademlia bootstrap failed        — connected_peers
```

Promoted from DEBUG to WARN so they're visible in production. A bootstrap failure with 0 connected peers means the node is isolated.

### Shard Download Failures

```
DIAG: shard download OutboundFailure   — model, shard_index, error, bytes_downloaded
```

Shows exactly which shard download failed, how far it got, and why.

## Credit Ledger Diagnostics

```
DIAG: failed to read credit balance from database — starting at zero
```

Only logged on startup if the database is corrupted. The node will function but starts with 0 credits.

## WSL2 Mitigations

WSL2's Hyper-V Networking Stack (HNS) introduces TCP stalls and protocol negotiation failures that can starve libp2p's outbound substream requests. Two mitigations are available via config:

### Disable autonat/dcutr

AutoNAT and DCUtR generate protocol negotiation traffic that can starve `poll_outbound` in libp2p's `Connection::poll` loop (handler NotifyBehaviour events have priority over `poll_outbound`). On WSL2, where protocol negotiations are already slow, this can prevent tensor forwards from ever reaching the codec.

```toml
# config/default.toml or ~/.local/share/swarmllm/config.toml
[network]
enable_autonat = false
enable_dcutr = false
```

Both protocols use `Toggle<T>` wrappers — when disabled, no events are emitted and no network traffic is generated. NAT detection and hole-punching are not needed for loopback/LAN testing.

### Yamux window size

The yamux receive window and max buffer are set to 16MB (up from the default 256KB). This prevents flow control stalls when forwarding large tensor activations over TCP+Yamux on WSL2:

```
cfg.set_receive_window_size(16 * 1024 * 1024);  // 16MB
cfg.set_max_buffer_size(16 * 1024 * 1024);       // 16MB
```

### WSL2 networking mode

For best results, use mirrored networking in `~/.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

This avoids the virtual NAT layer that causes additional latency and routing issues.

### Recommendation

For production testing, use native Linux (dual boot or bare metal). WSL2 is suitable for single-node development and basic multi-node testing with the above mitigations, but production distributed inference should run on native networking.

## Inference Subsystem Diagnostics

### Scheduler (scheduler.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: assemble_pipeline_for` | `candidates_count`, `segments`, `standbys`, `elapsed_ms` |
| DEBUG | `DIAG: gather_candidates` | `candidates_count` |
| DEBUG | `DIAG: find_standbys` | `segment_count`, `standby_count` |

### Executor (executor.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: load_model` | `path`, `backend_type`, `elapsed_ms` |
| DEBUG | `DIAG: generate_stream starting` | `prompt_len`, `temperature`, `max_tokens` |

### Speculative Decoding (speculative.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: speculative record_batch` | `drafted`, `accepted`, `acceptance_rate` |

### Vision (vision.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: encode_images` | `image_count`, `elapsed_ms` |

### Paged KV-Cache (paged_kv.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: paged_kv allocate` | `blocks_needed`, `free_blocks` |
| WARN  | `DIAG: paged_kv allocate OOM` | `blocks_needed`, `free_blocks` |

### Prefix Cache (prefix_cache.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: prefix_cache lookup` | `cache_entries`, `hit` (true/false) |

### Chat Template (chat_template.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: build_prompt` | `template_matched`, `fallback` |

## Model Subsystem Diagnostics

### Shard Store (shard.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: verify_shard FAILED` | `model`, `shard` |
| INFO  | `DIAG: load_all_local complete` | `model_count`, `total_shards`, `rejected_count` |

### Model Registry (registry.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: register_manifest` | `model_id`, `schema_version`, `shard_count` |
| INFO  | `DIAG: load_from_db complete` | `manifests_loaded_count` |

### HuggingFace (huggingface.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: search_gguf_models` | `query`, `repos_found` |

### Manifest (manifest.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: load_from_dir` | `dir`, `schema_version`, `shard_count` |

### LoRA (lora.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: load_adapter` | `adapter_path`, `rank`, `alpha`, `target_modules` |

### Acquisition (acquisition.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: handle_acquire` | `model`, `needed_shards` |
| DEBUG | `DIAG: select_best_peer` | `eligible_peers`, `selected_peer` |

### Auto-Manage (auto_manage.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: evaluate_and_prune` | `resource_pressure`, `pressure_urgent` |
| DEBUG | `DIAG: register_local_shard` | `model`, `shard_index` |
| INFO  | `DIAG: check_and_load_model` | `model_id`, `available_shards`, `missing_shards`, `ready` |

## API Subsystem Diagnostics

### Server (server.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: server startup` | `addr` |

### Admin (admin.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: hf_download_shards` | `model_id`, `variant` |

### Providers (providers.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: resolve_provider` | `model_id`, `resolved_provider` |

### WebSocket (websocket.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: websocket connected` | `addr` |

### Middleware (middleware.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: auth failure` | `path`, `auth_present` |

## Credit Subsystem Diagnostics

### Ledger (ledger.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: record_transaction` | `tx_type`, `delta`, `new_balance` |

### Escrow (escrow.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: create_escrow` / `DIAG: release_escrow` | `tx_id`, `amount`, `state` |

### Trust (trust.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: update_trust` | `node`, `score_delta`, `new_score` |

## Crypto Subsystem Diagnostics

### Key Rotation (key_rotation.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: key rotation eviction tick` | `active_sessions`, `stale_evicted` |
| DEBUG | `DIAG: key rotation re-keying tick` | `active_sessions`, `rekey_initiated` |

### Key Exchange (manager.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: key exchange initiated` | `peer` |
| INFO  | `DIAG: key exchange completed` | `peer`, `elapsed_ms` |

## Infrastructure Diagnostics

### Database (db.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: db_open` | `path` |

### Identity (keypair.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: load identity` | `path` |

### Peer Cache (peer_cache.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: peer cache saved` | `count` |

### Relay (relay.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: relay reservation` | `peer` |

## Files Modified

| File | Diagnostics Added |
|------|-------------------|
| `src/crypto/session.rs` | seal/open success+failure logging with nonce, AAD, key state |
| `src/crypto/key_rotation.rs` | Eviction tick, re-keying tick with session counts |
| `src/network/manager.rs` | Encrypted tensor send/receive, connection lifecycle, gossip audit, outbound tracking, key exchange |
| `src/network/behaviour.rs` | Connection limits, autonat/dcutr toggle state |
| `src/network/discovery.rs` | Bootstrap failures promoted to WARN with peer counts |
| `src/network/protocol.rs` | Tensor decompression success/failure with sizes and compression ratios |
| `src/network/relay.rs` | Relay reservation logging |
| `src/network/peer_cache.rs` | Peer cache save count |
| `src/inference/pipeline.rs` | Segment timing (local + remote), pipeline total timing, failover details, wait_for_result context |
| `src/inference/router.rs` | Pipeline schedule vs execute timing breakdown, result channel delivery, streaming done event |
| `src/inference/split.rs` | Per-forward-pass timing with seq_len/layer/kv_offset context, KV-cache cleanup logging |
| `src/inference/kv_cache.rs` | KV-cache hit/miss with detailed miss reasons (expired, degraded, prefix mismatch, evicted) |
| `src/inference/scheduler.rs` | Pipeline assembly timing, candidate counts, standby counts |
| `src/inference/executor.rs` | Model load timing with backend type, generate_stream params |
| `src/inference/speculative.rs` | Batch acceptance rate tracking |
| `src/inference/vision.rs` | Image encoding timing |
| `src/inference/paged_kv.rs` | Block allocation success/failure |
| `src/inference/prefix_cache.rs` | Cache hit/miss with entry counts |
| `src/inference/chat_template.rs` | Template matching and fallback detection |
| `src/model/shard.rs` | Shard verification failures, load_all_local summary |
| `src/model/registry.rs` | Manifest registration with schema version, DB load counts |
| `src/model/huggingface.rs` | Search result counts, HF_TOKEN auth |
| `src/model/manifest.rs` | Manifest load with schema version and shard count |
| `src/model/lora.rs` | Adapter load with rank, alpha, target modules |
| `src/model/acquisition.rs` | Acquisition requests, peer selection |
| `src/model/auto_manage.rs` | Prune evaluation, shard registration, model readiness |
| `src/api/server.rs` | Server startup with bind address |
| `src/api/openai.rs` | All 3 streaming paths with per-token timing, client disconnect detection, fallback path logging |
| `src/api/admin.rs` | HF shard download initiation |
| `src/api/providers.rs` | Provider resolution |
| `src/api/websocket.rs` | WebSocket connection lifecycle |
| `src/api/middleware.rs` | Auth failure with path context |
| `src/credit/ledger.rs` | Transaction recording with balance changes, DB restore failure |
| `src/credit/escrow.rs` | Escrow create/release with state |
| `src/credit/trust.rs` | Trust score updates |
| `src/storage/db.rs` | Database open with path |
| `src/identity/keypair.rs` | Identity key load |
| `src/daemon.rs` | LayerForward timing, LayerResult delivery, pending channel state |
| `src/health/monitor.rs` | Broadcast failures, stale peer counts, channel cleanup details |

## Coverage Statistics (2026-03-04)

**197 DIAG lines across 53/79 source files (67%).**

The 26 uncovered files are `mod.rs` re-exports (13), `types.rs` / `error.rs` data definitions (3), `main.rs` / `lib.rs` / `config.rs` entry points (3), `assets.rs` / `ui/mod.rs` static files (2), and utility files with no decision/timing points (5).

### Coverage by Subsystem

| Subsystem | Files | DIAG Lines | Key Log Points |
|-----------|-------|------------|----------------|
| Network (manager, behaviour, protocol, discovery, relay, peer_cache) | 6 | ~50 | Connection lifecycle, codec read/write, encryption, swarm events |
| Inference (router, pipeline, scheduler, executor, split, sampling, speculative, vision, kv_cache, prefix_cache, chat_template) | 11 | ~40 | Request dispatch, pipeline assembly, forward pass, token sampling |
| API (server, openai, admin, websocket, middleware, providers, anthropic) | 7 | ~35 | Server startup, SSE streaming, auth, schedule updates, WebSocket |
| Model (shard, manifest, huggingface, acquisition, auto_manage, registry, distribution, governance, lora) | 9 | ~25 | Shard verification, HF search/download, model loading, pruning |
| Credit (ledger, transaction, priority, anti_gaming, trust, escrow) | 6 | ~15 | Transaction verification, tier calculation, trust updates, escrow |
| Crypto (session, key_rotation, gossip_seal, pipeline_seal) | 4 | ~10 | Key exchange, session management, encryption seal/open |
| Daemon (daemon.rs) | 1 | ~10 | LayerForward processing, result delivery, block_in_place |
| Pool (manager, crypto, forward) | 3 | ~5 | Pool commands, invitations, credit forwarding |
| Identity (keypair, keystore, nickname) | 3 | ~3 | Key generation, keystore save/load, nickname records |
| Health (monitor, rebalancer) | 2 | ~4 | Rebalance events, health monitoring |

## Testing Results (2026-03-04)

### Single-Node Inference — Verified

Full DIAG trace confirmed working lifecycle:

```
DIAG: db_open → DIAG: load identity → DIAG: server startup →
DIAG: load_all_local → DIAG: register_manifest → DIAG: check_and_load_model →
DIAG: assemble_pipeline_for → DIAG: SplitModel forward pass complete →
DIAG: split stream prefill complete → DIAG: split stream decode loop complete →
DIAG: inference completed
```

All subsystem DIAG points fire correctly during a chat request. Filter with `grep "DIAG:"` to trace the full lifecycle.

### Multi-Node Distributed Inference — Partial (WSL2 Issue)

Two-node TinyLlama split (shard 0 on Node 1, shard 1 on Node 2):

- **1st token round-trip: WORKS** (~700ms including model forward on both nodes)
- **2nd outbound substream: STALLS** — never reaches codec, 30s pipeline timeout fires
- **Not caused by**: encryption, autonat/dcutr, mDNS, yamux window size, network interface
- **Cause**: Intermittent WSL2 HNS (Hyper-V Networking Stack) issue — same binary confirmed working hours earlier with ~20-26ms per-token latency

The DIAG trace for the successful 1st token shows the full distributed path:

```
[Node 1] DIAG: assemble_pipeline_for → DIAG: starting forward_through_segments
[Node 1] DIAG: codec write_request → DIAG: encrypting tensor forward (if enabled)
[Node 2] DIAG: codec read_request → DIAG: inbound TensorPayload request
[Node 2] DIAG: dispatcher received LayerForward → DIAG: SplitModel forward pass complete
[Node 2] DIAG: codec write_response → DIAG: ResponseSent event
[Node 1] DIAG: codec read_response → DIAG: received TensorPayload response
[Node 1] DIAG: forward_through_segments completed
```

### Config Note for Multi-Node Testing

Config loads from `~/.local/share/swarmllm/config.toml` (the default `data_dir`), not from `SWARMLLM_NODE_DATA_DIR`. To change network settings for multi-node testing, edit the global config file:

```bash
# Edit the global config (affects all nodes on this machine)
vim ~/.local/share/swarmllm/config.toml

# Per-node data dirs only affect data storage, not config
SWARMLLM_NODE_DATA_DIR=/tmp/swarm_a1 cargo run --release -- run -p 8800
SWARMLLM_NODE_DATA_DIR=/tmp/swarm_n2 cargo run --release -- run -p 8801 --bootstrap /ip4/127.0.0.1/tcp/8810
```

### Substream Stalling Root Cause Investigation

14 isolated tests in `tests/integration/test_yamux_substream.rs` proved that:

1. **Yamux works perfectly** — 10 sequential round-trips with 4MB payloads, ~1.6ms each (small), ~92ms each (large)
2. **QUIC works perfectly** — no yamux needed, native multiplexing, ~2.4ms each (small), ~193ms each (large)
3. **Full libp2p behaviours work** — kad+gossipsub+identify+request_response, 10 rapid rounds, no stalls
4. **Channel-driven requests work** — external mpsc channel driving requests (simulates real daemon flow)
5. **Gossipsub heartbeats don't interfere** — 5 topic subscriptions with 1s heartbeat interval
6. **CPU-blocking responder works** — 700ms `std::thread::sleep` on the responder, all 5 rounds complete

**Root cause:** `split_model.forward()` in `handle_layer_forward()` (daemon.rs) ran synchronously on a Tokio worker thread for 708ms+, blocking the runtime. Fixed with `tokio::task::block_in_place()`.

Run the yamux isolation tests:
```bash
cargo test --test yamux_substream -- --nocapture --test-threads=1
```
