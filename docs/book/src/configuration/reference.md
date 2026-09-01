# Config File Reference

Every configuration option, organized by section.

## When a change takes effect

Settings changed through the dashboard (or by editing this file and calling
`POST /api/admin/config/reload`) apply to the running node immediately. You do
not need to restart for a resource limit, a contribution level, an auto-manage
threshold, or a batching setting to matter.

Three things are decided once, when the node starts, and keep their startup
value until it starts again:

| Setting | Why |
|---|---|
| `[network] max_peers` (and the contribution level it derives from) | Connection limits are fixed when the peer-to-peer network is built |
| `[resources] max_cpu_threads` for an **already-loaded** model | Threads are handed to a model's worker when it loads; taking them back would interrupt whatever it is answering. The next model to load uses the new value |
| `[updates] mode` | The update checker is started, or not, at launch |

Everything else is live. Before v0.3.88 none of it was: a setting would save,
report success, show its new value, and leave the running node on the value it
booted with.

## `[node]` — Basic Node Settings

| Option | Type | Default | Description |
|---|---|---|---|
| `listen_port` | integer | `8800` | Port for web dashboard and P2P networking |
| `data_dir` | path | Platform-specific | Where SwarmLLM stores data |
| `contribution` | string | `"minimal"` | Resource contribution: `"minimal"`, `"moderate"`, `"maximum"`. Applies without a restart, except for the two startup-fixed items noted above |
| `contribution_auto` | boolean | `true` | Auto-scale contribution at swarm saturation. Applies without a restart |
| `anchor_mode` | boolean | `false` | Run as a pure bootstrap / relay / reachability-probe node. Skips all inference — no models load, no HuggingFace polling, no shard acquisition, no auto-manage — and binds the dashboard to loopback only. The node still participates fully in the peer-to-peer network |

## `[resources]` — Resource Limits

| Option | Type | Default | Description |
|---|---|---|---|
| `max_gpu_vram_mb` | integer | `0` | Max GPU memory in MB. `0` = auto-detect |
| `max_ram_mb` | integer | `0` | Max system RAM in MB for models loaded on the CPU. `0` = auto: 50% of system RAM on a machine with a GPU, 80% on a CPU-only node (where serving models is the machine's purpose). An explicitly configured value is clamped to what is actually free, so a figure larger than the machine cannot push it into swap. A model that does not fit is refused with a 503 rather than loaded |
| `max_disk_mb` | integer | `50000` | Max disk space in MB for model storage |
| `max_bandwidth_mbps` | integer | `0` | Max upload bandwidth for serving model files to peers. `0` = auto, from `[node] contribution` — 10 Mbps at minimal, 50 at moderate, unlimited at maximum |
| `max_cpu_threads` | integer | `0` | CPU threads used for inference. `0` = auto, from `[node] contribution` — half / three-quarters / all of the **physical** cores. More threads than physical cores makes this *slower*: quantised inference is limited by memory bandwidth, so two threads sharing a core contend rather than add (measured on an 8-core/16-thread laptop: 6 threads 2.36 tok/s, 16 threads 1.49). An explicit value is honoured up to the logical core count. |

## `[resources.schedule]` — Usage Schedule

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `false` | Enable scheduled resource reduction |
| `reduced_hours_start` | integer | `22` | Hour (0-23) to start reduced mode |
| `reduced_hours_end` | integer | `8` | Hour (0-23) to end reduced mode |
| `reduced_contribution` | string | `"minimal"` | Contribution level during reduced hours |
| `prune_aggressiveness` | string | `"normal"` | Shard pruning during reduced hours: `"normal"`, `"aggressive"`, `"conservative"` |

## `[network]` — Networking

| Option | Type | Default | Description |
|---|---|---|---|
| `bootstrap_peers` | list | built-in anchor | Peer addresses to dial on startup. An empty list means "not configured" and falls back to the built-in anchors |
| `disable_default_bootstrap` | boolean | `false` | Genuinely start with no bootstrap peers (private / air-gapped swarm). Implied by `node.anchor_mode` |
| `enable_mdns` | boolean | `true` | LAN peer discovery |
| `gossip_network_id` | string | none | Run a private network. Nodes sharing this value form their own gossip network: they use their own topics and their own encryption key, so their traffic never reaches the public swarm and vice versa. Leave unset to join the public network |
| `peer_exchange` | boolean | `true` | Share peer lists with connected nodes |
| `enable_relay` | boolean | `true` | Act as relay for peers behind firewalls |
| `enable_relay_client` | boolean | `true` | Let THIS node route its own traffic through someone else's relay when it is not directly reachable. Turning it off leaves the node able to reach others but generally unreachable from the internet. The opposite role — relaying for others — is `enable_relay` |
| `max_peers` | integer | *(from `contribution`)* | Max simultaneous peer connections. Unset it follows `[node] contribution` — minimal `150`, moderate `300`, maximum `500`. An explicit value wins in either direction. |
| `auto_relay` | boolean | `true` | Auto-use relay when NAT detected |
| `relay_max_circuit_duration_secs` | integer | `3600` | Max relay circuit duration |
| `relay_max_circuits` | integer | `16` | Max relay circuits to serve |
| `enable_encryption` | boolean | `true` | E2E encryption for tensor forwards and control messages |
| `enable_autonat` | boolean | `true` | NAT detection. Disable on WSL2 to reduce noise |
| `enable_dcutr` | boolean | `true` | Hole punching. Disable on WSL2 to reduce noise |
| `tensor_compression` | boolean | `true` | Zstd compression for tensor payloads |
| `prefix_kv_compression` | boolean | `false` | Zstd compression for cross-node prefix-KV snapshot wire frames. Default off — meaningful win on WAN where wire size is the bottleneck; roughly neutral on localhost. Receivers always decompress regardless of this flag. |
| `tensor_compress_level` | integer | `1` | Zstd compression level (1-22, 1 = fastest). Shared between tensor and prefix-KV. |
| `tensor_compress_threshold` | integer | `1024` | Min payload bytes before compression. Shared between tensor and prefix-KV. |
| `listen_address` | string | `"0.0.0.0"` | Address to bind peer-to-peer listeners on. Set to `127.0.0.1` on WSL2 to avoid binding unreliable NAT adapters |
| `enable_quic` | boolean | `true` | QUIC transport. Disabling it on WSL2 avoids a race where the faster QUIC handshake displaces the TCP connection |
| `enable_upnp` | boolean | `true` | Ask the home router to open the peer-to-peer ports automatically, and confirm the resulting public address with the swarm. The zero-configuration path to being reachable from the internet |
| `max_connections_per_peer` | integer | `2` | Simultaneous connections to a single peer. **Must be at least 2 for NAT hole punching**: the upgrade dials a direct connection while the relayed one is still open, so `1` kills one of them |
| `relay_forwarding` | boolean | `false` | Carry *inference* messages between two peers that cannot reach each other. Distinct from `enable_relay`, which carries the connection itself |
| `relay_forwarding_auto` | boolean | `true` | Donate that relay capacity automatically once this node is confirmed reachable from the open internet |

## `[inference]` — AI Model Inference

| Option | Type | Default | Description |
|---|---|---|---|
| `default_model` | string | `""` | Default model. Empty = first available |
| `session_timeout_seconds` | integer | `600` | Chat session memory lifetime (10 min) |
| `max_concurrent_requests` | integer | `10` | Max parallel requests |
| `model_path` | path | none | Path to a GGUF model file |
| `gpu_layers` | integer | `-1` | Device placement. `-1` = auto: the whole model on the card when it fits, and when it does not, as many of its first layers as do with the rest on the processor (since v0.3.145; `SWARMLLM_HYBRID_OFFLOAD=0` turns the split off and falls back to processor-only). `0` = processor only. `>0` = put that many layers on the card and the rest on the processor. Architectures verified for the split: Llama, Llama 4, Qwen2, Gemma, Gemma 2, Phi-3, Mistral, Starcoder2, GLM-4; others load on one device as before |
| `kv_cache_ttl_secs` | integer | `600` | KV-cache lifetime |
| `max_batch_size` | integer | `8` | How many requests run through the model together, amortising the weight reads that dominate a decode step. Measured on an RTX 3070 with llama-3.2-3b using per-request rates: no cost at any concurrency, about 3% at eight concurrent requests and about 16% at twelve — the gain grows with load. Neutral on a processor (6-core i5-10500T, llama-3.2-1b). Set to `1` to disable |
| `batch_timeout_ms` | integer | `50` | Ms to wait for additional requests before dispatching a partial batch. `0` = dispatch immediately (purely opportunistic batching) |
| `speculative_decoding` | boolean | `false` | Enable speculative decoding |
| `speculative_gamma` | integer | `4` | Draft tokens per verification step |
| `draft_model_path` | path | none | Path to draft model |
| `max_split_model_memory_mb` | integer | none | Max GPU memory for split model cache |
| `tensor_parallel` | boolean | `false` | Split single layers across LAN peers via per-layer AllReduce. Off by default — over Ethernet the two round trips per layer cost more than the compute they split, and a node that holds every layer never forms a group regardless |
| `tp_max_latency_ms` | integer | `10` | Max peer latency (ms) for tensor parallelism groups (only consulted when `tensor_parallel = true`) |
| `local_embedding_privacy` | boolean | `false` | Embed tokens locally before sending to first segment. Remote nodes never see raw token IDs |
| `encrypted_pipeline` | boolean | `false` | Force first+last segment to local node (boomerang topology). No remote sees plaintext. Adds ~1 RTT/token. Per-model override via API. Requires shard 0 + final shard locally |
| `privacy_mode` | boolean | `false` | Never write user prompts to disk — KV-cache sessions stay in memory only |
| `parallax_routing` | boolean | `true` | Use Parallax shortest-path DP for segment assignment; falls back to greedy on any failure |
| `persistent_pipeline_stream` | boolean | `false` | One long-lived libp2p stream per pipeline session instead of per-token request/response |
| `max_seq_len_override` | integer | none | Cap the GGUF `context_length` when sizing the KV cache, so long-context models fit small VRAM. Unset = use the GGUF value. Raising it no longer costs a model its place on a node: admission on both CPU and GPU charges KV cache for a typical 4096-token conversation, and a worker refuses a conversation that would outgrow its memory budget at run time (a 503 that re-routes to a peer) rather than swapping. A model without grouped-query attention such as phi-3.5-mini still costs 0.75 MB per token of KV cache, so how long a conversation a CPU node can hold is set by its free RAM (`resources.max_ram_mb`), and the node's log says how many tokens fit when the model loads. |
| `draft_gpu_layers` | integer | none | Device placement for the draft model. Unset = inherit `gpu_layers` |
| `force_standard_attn` | boolean | `false` | Route every attention call through `standard_attention` instead of the fused kernel. Diagnostic; auto-enabled while SWIFT is on |
| `shard_range` | tuple | none | Advanced/dev: claim only this shard index range for split inference. Normal nodes auto-detect their local shards |

### `[inference]` — batching

| Option | Type | Default | Description |
|---|---|---|---|
| `continuous_batching` | boolean | `true` | Coalesce concurrent decode requests for the same model into one fused worker forward. 1.34–1.55× on GPU at batch 2–8; neutral-to-loss on CPU, where the worker falls back to sequential |
| `max_concurrent_decode_batch` | integer | `8` | Maximum decode slots fused into one batch |
| `batch_collection_ms` | integer | `5` | How long the scheduler waits for more arrivals after the first request lands in an empty batch. WSL2 timer resolution is ~15 ms, so smaller values dispatch immediately there |
| `prefill_chunk_tokens` | integer | `128` | Sarathi-style chunked prefill size. Each `Prefilling` slot advances by this many prompt tokens per decode tick, bounding how long one admission can stall active decodes |
| `batched_prefill_forward` | boolean | `true` | Fuse concurrent same-shape prefill chunks into one forward. Set `false` to isolate this from continuous batching in A/B benchmarks |

### `[inference]` — prefix cache

| Option | Type | Default | Description |
|---|---|---|---|
| `prefix_cache_enabled` | boolean | `true` | Reuse prefill KV across requests sharing a prompt prefix — covers both multi-turn and the same system prompt from different users |
| `prefix_cache_max_entries` | integer | `16` | Cached prefix snapshots retained per model |
| `prefix_cache_max_mb` | integer | `2048` | Max megabytes of prefix KV retained per model, `0` for no byte bound. This is the bound that expresses memory: `prefix_cache_max_entries` caps how many entries are kept, which says nothing about their weight, since an entry scales with its prompt. At the insert ceiling on a 3B model one entry can reach ~1.9 GB, so sixteen distinct long conversations could retain ~30 GB without ever exceeding the count cap |
| `prefix_cache_max_prompt_tokens` | integer | `8192` | Prompts longer than this are not inserted |
| `prefix_cache_block_tokens` | integer | `64` | Block alignment for the chained-hash manifest, and the granularity of a partial hit |
| `prefix_cache_min_tokens` | integer | `32` | Shortest prefix worth caching |
| `cross_node_prefix_trust_min` | float | `0.5` | Minimum peer trust score before accepting a prefix-KV snapshot fetched from that peer |
| `share_prefix_cache_with_peers` | bool | `false` | Offer this node's prompt prefix cache to other nodes. Off by default: turning it on announces hashes of the prompts this node caches to the whole swarm, and serves those prompts' token IDs and key/value cache to any peer that asks. Your own local prefix cache works either way |

### `[inference]` — speculative decoding

| Option | Type | Default | Description |
|---|---|---|---|
| `speculative_distributed` | boolean | `false` | Speculative decoding on the distributed path. Needs `speculative_decoding` + a loaded draft model |
| `decentralized_spec_decoding` | boolean | `false` | DSD: draft and target split across nodes |
| `swift_self_speculative` | boolean | `false` | SWIFT ([arXiv 2410.06916](https://arxiv.org/abs/2410.06916)) — draft by skipping layers of the target model, so no separate draft model is needed |
| `swift_calibration_tokens` | integer | `32` | Warm-up tokens before SWIFT's calibrator pins a skip pattern |
| `swift_gamma` | integer | `4` | Draft tokens proposed per SWIFT verification round |
| `swift_skip_ratio` | float | `0.45` | Fraction of layers skipped in the SWIFT draft pass |
| `ngram_lookup_enabled` | boolean | `true` | SWARM-SPEC Layer 1: draft from n-grams already present in the prompt, no draft model required. Large win on input-grounded workloads (RAG, coding, summarisation) — measured +45% at a 77% hit rate |
| `ngram_max_size` | integer | `4` | Longest n-gram matched against the prompt |
| `ngram_num_pred_tokens` | integer | `10` | Tokens proposed per n-gram hit |

### `[inference]` — SWARM-SPEC hedging and prefetch

| Option | Type | Default | Description |
|---|---|---|---|
| `hedge_enabled` | boolean | `false` | Layer 2: race a duplicate forward to an alternate shard holder when the primary looks slow, take the winner, discard the loser. Costs bandwidth to cut tail latency |
| `hedge_after_factor` | float | `1.5` | Fire the duplicate once elapsed time exceeds this multiple of the estimated p99 for that (model, segment, holder) |
| `hedge_min_samples` | integer | `20` | Latency samples required before hedging engages. At α=0.2 the variance EWMA only reaches ~90% of its true value by 20 samples; lower values collapse the p99 estimate toward the mean and over-fire after a restart |
| `hedge_max_rate` | float | `0.05` | Ceiling on the fraction of forwards that may be hedged |
| `prefetch_enabled` | boolean | `false` | Layer 3: predict the next turn in a conversation and warm state during idle time |
| `prefetch_min_turns_for_prediction` | integer | `2` | Turns observed before a session is predictable enough to prefetch for |
| `prefetch_min_idle_ms` | integer | `2000` | Idle time before prefetch may use the device |
| `prefetch_max_candidates` | integer | `3` | Candidate continuations considered per prediction |

### `[inference]` — activation transfer

| Option | Type | Default | Description |
|---|---|---|---|
| `activation_compression` | boolean | `true` | Quantise intermediate hidden states to Q8_0 before sending to the next peer (~3.76× smaller, group-32 + f16 scale). Receivers auto-dispatch on the dtype tag, so this is safe to toggle per node |
| `streaming_chunked_send` | boolean | `false` | Split a segment-boundary activation into K chunks sent on one stream, overlapping encrypt with transfer. Off by default: on LAN the send is already sub-millisecond and per-chunk cost dominates. The win is WAN-only (roughly <30 Mbps). Requires `persistent_pipeline_stream` |
| `streaming_chunk_size_bytes` | integer | `262144` | Chunk size for the above. 256 KiB matches the age STREAM default and the TokenWeave K=2–4 sweet spot |
| `streaming_min_activation_bytes` | integer | `65536` | Activations below this ship as a single frame regardless of the flag |
| `streaming_chunk_assembly_ttl_secs` | integer | `30` | Receiver-side TTL for an incomplete chunk assembly before it is swept |
| `prefill_target_ms` | integer | `100` | Wall-time budget for one tick of prompt processing **while more than one request is active** — the same limit as `prefill_chunk_tokens` but expressed in what a waiting request actually feels |
| `encrypted_pipeline_auto` | boolean | `true` | Turn prompt privacy on automatically for any model where this node holds both the first and last shard, which is the only condition under which it can work. An explicit per-model or global `encrypted_pipeline` setting always wins |
| `parallax_partial_ranges` | boolean | `false` | Let the router use only PART of a node's shard range, so a node holding a whole model can serve just one end of it. Off by default: the cost model charges a remote hop once per segment rather than per token, so it cannot see the round-trip penalty and over-splits |
| `pipeline_chaining` | boolean | `true` | Chain consecutive remote pipeline segments peer to peer instead of returning every hop to the coordinator. Unchained, an N-segment pipeline costs N coordinator round trips per token; a chained run costs one trip out and one back however long it is, so per-token network cost stops scaling with shard count. A local segment ends a chain, so prompt privacy is unaffected — with `encrypted_pipeline` the first and last segments stay local and only the middle is chained. On by default since v0.3.109, validated end to end on two machines; every reason not to chain falls back to the old behaviour rather than failing, a chained run that fails is re-run unchained for that request, and peers on earlier versions are never chained to. Set to `false` to send every segment through the coordinator as before. See `docs/plans/direct_peer_chaining.md` |
| `max_chain_hops` | integer | `8` | Longest run of peer-to-peer hops in one chain. Bounds how much of a request rides on nobody checking in: the coordinator hears nothing between handing a chain over and its tail reporting back, so a longer chain means a slower failure |
| `shed_load_when_busy` | boolean | `false` | Let a busy node hand a request it could serve locally to the router, so a peer can take it. A node holding a complete model otherwise serves it locally without consulting the router at all: measured 2026-08-20, eight concurrent requests all stayed on one GPU (36 → 7.5 tok/s per request) while a peer advertising 20 tok/s sat idle. Off by default because the benefit is unproven — aggregate throughput kept RISING with concurrency on that GPU, so degrading per-request is not the same as needing to stop taking work |
| `shed_load_threshold` | integer | `4` | How many requests must already be in flight before `shed_load_when_busy` offers one to the router. One in flight is not busy: the second request shares a GPU that is mostly idle between token launches, which is what batching exploits |

## `[logging]` — Log Output

| Option | Type | Default | Description |
|---|---|---|---|
| `level` | string | `"info"` | Log level: `"error"`, `"warn"`, `"info"`, `"debug"`, `"trace"` |
| `format` | string | `"pretty"` | Log format: `"pretty"` or `"json"` |
| `file` | path | none | Write logs to file |

## `[ui]` — Web Interface

| Option | Type | Default | Description |
|---|---|---|---|
| `open_browser_on_start` | boolean | `true` | Open dashboard on launch |

## `[api]` — API Authentication

| Option | Type | Default | Description |
|---|---|---|---|
| `api_key` | string | none | Bearer token. Empty = auto-generated |
| `rate_limit_rpm` | integer | `60` | Rate limit for `/v1/` endpoints (requests/min) |
| `rate_limit_admin_rpm` | integer | `200` | Rate limit for `/api/admin/` endpoints (requests/min) |
| `metrics_auth_required` | boolean | `false` | Require Bearer auth on `/metrics` even from loopback |
| `dashboard_trust_overlay` | boolean | `true` | Hand the dashboard its access key over a Tailscale-style overlay, when this node is on one too |
| `dashboard_trust_lan` | boolean | `false` | Hand the dashboard its access key to any private/LAN address |

The dashboard fetches its own access key on page load. These two options decide
which networks that happens on — loopback always does, and anywhere else the
page asks you to paste the key once instead of failing silently.

`dashboard_trust_overlay` only takes effect when this node itself holds an
overlay address. The IPv4 range Tailscale uses (`100.64.0.0/10`) is shared
carrier-grade NAT space that some ISPs also hand out, so a peer's address alone
does not prove a tailnet.

`dashboard_trust_lan` exists for the case where you reach a node through a
Tailscale **subnet router** or a container publish. Those rewrite the source
address by default, so the request arrives from the router's private address and
is indistinguishable from any other LAN client. It can be toggled from Settings →
Identity & Access and applies immediately, without restarting the node.

On a network this node trusts, anything that can reach the API port can obtain
the access key, and with it admin and inference. That is the intended bargain for
a tailnet — devices you authorised — which is why the LAN case stays opt-in. See
[Tailscale / WAN](../operations/tailscale-wan.md).

## `[model]` — Model Storage

| Option | Type | Default | Description |
|---|---|---|---|
| `shard_size_mb` | integer | `512` | Shard size in MB. Range: 64-2048 |

## `[auto_manage]` — Automatic Shard Management

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `true` | Auto-download popular shards (only for models at DemandVerified+ or Pinned trust level) |
| `max_storage_mb` | integer | `0` | Max disk for auto-downloads. `0` = 50% of max_disk_mb |
| `interval_minutes` | integer | `5` | Check interval for new shards |
| `interval_seconds` | integer | none | Testing override for `interval_minutes`. Takes precedence when set |
| `model_policies` | table | `{}` | Per-model overrides keyed by model id, e.g. `[auto_manage.model_policies."llama-3.1-8b"]` |
| `max_shards` | integer | `0` | Max shards. `0` = unlimited |
| `max_concurrent_downloads` | integer | `3` | Max parallel downloads |
| `prune_enabled` | boolean | `true` | Auto-remove over-replicated shards |
| `min_replicas` | integer | `2` | Min network replicas before pruning |
| `prune_cooldown_secs` | integer | `300` | Seconds between prune actions per model |
| `max_holder_load_for_prune` | integer | `3` | Block pruning if holders are busy |
| `hf_watcher_enabled` | boolean | `true` | Background poll of HF trending GGUF feed (hourly). Disable for air-gapped / bandwidth-constrained nodes |
| `wishlist_gossip_publish` | boolean | `false` | Opt-in: publish your wishlist as cross-pool demand gossip (R130) |
| `auto_switch_quants` | boolean | `true` | **R141 default flip**: auto-acquire the recommended quant variant when the recommender (R133) suggests a better one. Set `false` on metered links to keep the current quant |
| `parallax_auto_rebalance` | boolean | `true` | Bias scoring toward Parallax allocator recommendations (C.2) |
| `default_model_shard_cap` | integer | `0` | Max shards auto-manage acquires per model. `0` = unlimited. **Set this if you want to add capacity rather than redundancy** — see below |
| `idle_unload_secs` | integer | `900` | Free a loaded model's GPU memory after this long with no local requests AND little demand for it elsewhere in the network. Shards stay on disk, so the model reloads on the next request |


### Contributing capacity vs. contributing redundancy

`default_model_shard_cap = 0` (the default) lets auto-manage take every shard of
a model it decides to host. With `min_replicas = 2` and a small number of
participants, the usual outcome is that **every node ends up holding every model
in full** — measured on a two-node setup as ~20 GB and ~18 GB of near-identical
content.

That is the right default: it maximises availability, and a model nobody holds
completely cannot be served at all.

It is not always what an operator wants. Holding a full copy of a model your
neighbour already holds adds a spare, not throughput. If you are joining to add
capacity — more machines able to serve *different* parts of more models — set a
cap:

```toml
[auto_manage]
default_model_shard_cap = 2   # at most 2 shards of any one model
```

Rough guidance:

| Goal | Setting | Effect |
|---|---|---|
| Survive nodes going offline | `0` (default) | Full copies everywhere; any node can serve alone |
| Host more distinct models on limited disk | small cap (1-3) | Partial coverage, requests span several nodes |
| Dedicated always-on node | `0` | Be the copy others fall back to |

A cap only limits what auto-manage acquires **on its own**. Shards you download
explicitly, and shards pinned to this node by a pool owner, are unaffected.

One caveat worth knowing before you tune this: holding fewer shards spreads
storage, not necessarily load. A node that holds a model in full still serves
requests for it locally rather than sharing them out, so capping shards is about
what gets *stored* where, not about how work is distributed.

## `[pool]` — Device Pool

| Option | Type | Default | Description |
|---|---|---|---|
| `max_pool_size` | integer | `10` | Max devices in a pool |
| `invitation_ttl_hours` | integer | `24` | Invitation validity period |
| `rate_limit_per_hour` | integer | `10` | Max pool operations per hour |
| `gossip_interval_secs` | integer | `600` | Pool state gossip interval |
| `private_mode` | bool | `false` | Restrict inference to pool members only. Toggleable at runtime via API/UI |
| `private_mode_allow_lan` | bool | `true` | Also allow LAN peers (mDNS-discovered) when private mode is on |
| `offline_mode` | bool | `false` | Air-gapped: no bootstrap peers, no HF downloads, mDNS-only discovery |
| `state_diff_gossip` | boolean | `false` | Send pool membership changes as signed differences between full broadcasts, instead of always resending the whole member list |
| `share_model_catalog` | boolean | `false` | Let the pool owner advertise which models the pool can serve, so other pools can discover it |
| `share_model_catalog_min_members` | integer | `3` | Anonymity floor for the above: a pool smaller than this never publishes its catalog, whatever the flag says |
| `allow_cross_pool_inference` | boolean | `false` | With `private_mode` also on, allow routing to another pool's members for a model that pool has advertised. Off means private mode stays strictly within your own devices |

## `[pool.credit_rates]` — Credit Rates

| Option | Type | Default | Description |
|---|---|---|---|
| `inference_serve` | integer | `10` | Credits earned per layer per token served |
| `inference_consume` | integer | `10` | Credits spent per layer per token consumed |
| `shard_hosting` | integer | `1` | Credits per GB per hour hosting |
| `shard_seeding` | integer | `5` | Credits per GB seeding |
| `relay_service` | integer | `2` | Credits per connection hour relaying |
| `penalty_serve_failure` | integer | `50` | Credits deducted per failure |

## `[updates]` — Auto-Update

| Option | Type | Default | Description |
|---|---|---|---|
| `auto_update` | string | `"disabled"` | Policy: `"disabled"`, `"stable"`, `"all"`. Default flipped to disabled in R88 (security — users opt in via `[updates] auto_update = "stable"`). |
| `check_interval_hours` | integer | `6` | Update check frequency |
| `include_prereleases` | boolean | `true` | Offer pre-release builds. Defaults on because every release so far is tagged `-alpha`, so excluding them would mean never seeing an update at all |

## `[identity]` — Your Identity

| Option | Type | Default | Description |
|---|---|---|---|
| `region` | string | none | Country code for network map (e.g., `"US"`) |

## `[providers.claude_subscription]` — Claude Subscription (feature-gated)

> Requires `--features claude-subscription` at build time. Managed via the dashboard or `PUT /api/admin/providers`.

| Option | Type | Default | Description |
|---|---|---|---|
| `enabled` | boolean | `false` | Route `claude-*` model requests through the local CLI |
| `claude_binary` | string | `"claude"` | Path to the `claude` binary |
| `default_model` | string | none | Override model for all requests |
| `max_concurrent` | integer | `3` | Maximum concurrent subprocess invocations |
| `timeout_secs` | integer | `300` | Per-request timeout in seconds |
| `working_dir` | string | *(temp dir)* | Working directory for the subprocess. Empty or `"none"` uses system temp dir (recommended for API proxy use). Set to a project path for context-aware responses. |
