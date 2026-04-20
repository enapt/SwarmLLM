# Round 6 Benchmarks — Item 8 cross-node prefix KV sharing

> **TL;DR**: Phases 1 + 2a + 2b + 3 of Item 8 are all in-tree. This doc
> is the **bench recipe** for Phase 4 — measuring TTFT reduction when a
> prompt's prefix KV is fetched from a peer instead of re-prefilled
> locally. Numbers need real hardware; the doc is written so a user on
> the RTX 3070 can run the two-daemon loopback and fill them in.

## What we're measuring

After the full Phase 1–3 stack landed, a node B receiving a long prompt
whose prefix has already been prefilled by peer A should see:

1. Node A's `PrefixCacheAnnounce` has propagated to B's
   `cross_node_prefix_index` (Phase 1).
2. B's worker tokenizes the prompt, local `PrefixCache.lookup` misses,
   sends `WorkerMsg::PrefixFetchProbe` to its daemon (Phase 2b).
3. B's daemon walks the index, picks A as the only holder (and
   trust-gates: A must have `trust_score >= cross_node_prefix_trust_min`,
   default 0.5) (Phase 3).
4. B's daemon dispatches `NetworkCommand::SendPrefixKvFetch` → A's
   daemon → A's worker pulls the snapshot from its local PrefixCache →
   bytes travel back over `WIRE_TAG_PREFIX_KV` → B's daemon
   BLAKE3-verifies + NaN/Inf-checks (Phase 3) → hands bytes back to
   B's worker → hydrate KV → prefill suffix only.
5. B's TTFT should drop from "full prompt prefill" to "suffix prefill
   + 1 round trip + KV transfer bandwidth".

The **cross-over point** is when `prompt_prefill_ms > RTT + kv_transfer_ms`.
For TinyLlama-1.1B with a ~500-token cached prefix, prefill is ~300 ms
on GPU; RTT on loopback is ~1 ms; KV snapshot for 22 layers × 4 kv_heads
× 500 tokens × 64 head_dim × 4 bytes (f32) ≈ 22 MB uncompressed. At
localhost bandwidth (several GB/s) the transfer is <10 ms. So the
fetched path should win by ~200 ms on a 500-token prefix cache hit.

## Recipe

### 0. Build

```bash
cd ~/SwarmLLM
cargo build --release --no-default-features --features dev,candle-cuda,claude-subscription
```

### 1. Spin up two daemons on loopback

Use separate data dirs so they don't fight over DBs / model registries,
and non-overlapping ports:

```bash
# Node A on 8800 (HTTP) + 8810/8800 (P2P)
SWARMLLM_NODE_DATA_DIR=/tmp/swarm_a ./target/release/swarmllm run \
    -p 8800 -v 2>&1 | tee /tmp/swarm_a.log &
# Wait for A's API to be ready
until curl -s http://localhost:8800/healthz >/dev/null; do sleep 0.2; done

# Node B on 8900 — bootstrap off of A via its local p2p address.
# Identify A's multiaddr from /api/admin/peers or the log:
A_ADDR=$(grep -oE '/ip4/127\.0\.0\.1/udp/8800/quic-v1/p2p/[A-Za-z0-9]+' /tmp/swarm_a.log | head -1)

SWARMLLM_NODE_DATA_DIR=/tmp/swarm_b ./target/release/swarmllm run \
    -p 8900 -v --bootstrap "$A_ADDR" 2>&1 | tee /tmp/swarm_b.log &
until curl -s http://localhost:8900/healthz >/dev/null; do sleep 0.2; done
```

Both daemons should connect within a few seconds. Verify via
`curl http://localhost:8800/api/admin/peers | jq` — you should see
node-B in node-A's peer list (and vice versa).

### 2. Load the same model on both

The fastest path is to copy the pre-staged shards from your primary
data dir (`~/.local/share/swarmllm/models/`) into both test dirs:

```bash
cp -r ~/.local/share/swarmllm/models/<tinyllama-model-id> /tmp/swarm_a/models/
cp -r ~/.local/share/swarmllm/models/<tinyllama-model-id> /tmp/swarm_b/models/

# Force both daemons to re-scan
curl -X POST http://localhost:8800/api/admin/models/rescan -H "Authorization: Bearer $(cat /tmp/swarm_a/api_key)"
curl -X POST http://localhost:8900/api/admin/models/rescan -H "Authorization: Bearer $(cat /tmp/swarm_b/api_key)"
```

Replace `<tinyllama-model-id>` with the actual slug under your models
directory (e.g., `tinyllama-1.1b-chat-v1.0.q4_k_m`). Check
`memory/local_model_shards.md` for the canonical path.

### 3. Warm up node A with a long system-prompt request

Use a ~500-token "agent scaffold" prompt so the prefix-cache block
hashes actually hit `min_tokens` (default 32). Example:

```bash
PROMPT='You are a meticulous Rust systems programmer working on SwarmLLM, a decentralized LLM inference network. The codebase uses libp2p, tokio, candle-core for tensor compute, and a custom shard-based model distribution system. When answering, prefer direct code references over prose. Cite file paths + line numbers. Never fabricate APIs. (... pad to ~500 tokens with realistic context ...)'

./target/release/swarmllm bench \
    -p 8800 \
    --iterations 3 --max-tokens 100 --stream \
    --prompt "$PROMPT" \
    --model-id <tinyllama-model-id>
```

The first iteration warms the prefix cache on A. Expected log on A:

```
DIAG: prefix-cache inserted snapshot model_key="0-22-24" entries=<N>
DIAG: prefix-cache loopback indexed (self) model=<tinyllama-model-id> ...
```

### 4. Wait for announce to reach B

Gossip propagates through the `swarm/models` topic. A full health-ping
cycle is 30s; the initial `spawn_initial_announcements` fires 5s after
daemon start and every insert triggers a broadcast. So 5–10s should be
plenty. Verify:

```bash
grep "PrefixCacheAnnounce indexed" /tmp/swarm_b.log | tail -3
# Expected: "DIAG: PrefixCacheAnnounce indexed node_id=<A's hex> model=... blocks=<N>"
```

### 5. Measure node B TTFT with the same prompt

```bash
./target/release/swarmllm bench \
    -p 8900 \
    --iterations 3 --max-tokens 100 --stream \
    --prompt "$PROMPT" \
    --model-id <tinyllama-model-id> \
    --json > /tmp/bench_b_with_fetch.json
```

Expected log on B (watch `/tmp/swarm_b.log`):

```
DIAG: cross-node prefix HIT — hydrated KV matched_tokens=448 total_tokens=513 bytes=<K>
DIAG: handle_generate prefix-cache HIT — prefilling suffix only matched_tokens=448 ...
```

On A (the serving side):

```
DIAG: PrefixKvFetch: serving inbound fetch via worker IPC ...
DIAG: served PrefixKvFetch ticket=<uuid> age_ms=<ms> hit=true
```

### 6. Control measurement: disable cross-node fetch

To isolate the win vs. full local prefill, re-run step 5 with the probe
disabled. Easiest way: set the trust threshold above 1.0 so every peer
is gated out:

```bash
# Edit /tmp/swarm_b/config.toml, add under [inference]:
#   cross_node_prefix_trust_min = 2.0
# Then restart node B and re-run step 5.
```

Save TTFT distribution to `/tmp/bench_b_control.json`.

### 7. Report

Fill in:

| Scenario | TTFT mean (ms) | TTFT min / max | Decode tok/s | DIAG `cross-node prefix HIT` count |
|---|---|---|---|---|
| B with cross-node fetch (step 5) | TODO | TODO | TODO | 3 |
| B without cross-node fetch (step 6) | TODO | TODO | TODO | 0 |
| A single-node baseline (step 3 iter 2+) | TODO | TODO | TODO | 0 |

Expected win: step 5 ≈ step 3 iter 2+ (both serve from a cached prefix),
step 6 ≈ full prefill latency (baseline miss).

## Troubleshooting

- **Probe times out**: `PREFIX_FETCH_TIMEOUT_MS = 500`. Bump the
  `InferenceConfig` or check that the serving-side worker's IPC is
  healthy (look for `fetch_local_snapshot: timed out` on A).
- **B gets rejection**: look for
  `"prefix-probe: rejected KV snapshot — penalizing peer trust"`.
  Three rejection reasons:
  - `hash_chain_mismatch`: sender's block_size differed from what we
    expected AND wasn't in the common-defaults list (32, 64, 128).
    This should be impossible with matching configs; double-check
    `prefix_cache_block_tokens` is equal on both daemons.
  - `non_finite_tensors`: GPU overflow on serialization side (rare on
    CPU f32 wire, but possible on badly-calibrated GPU models).
  - `deserialize_failed`: wire corruption — open an issue.
- **No announce reaches B**: verify both daemons joined the same
  `/swarm/models` gossipsub topic. Check `grep "subscribed" /tmp/swarm_a.log /tmp/swarm_b.log`.
- **Trust floor blocks the fetch**: new peers start at
  `DEFAULT_TRUST = 0.5`, which equals the default
  `cross_node_prefix_trust_min = 0.5`, so a freshly-connected peer
  should just barely pass. If B's log shows `"prefix-probe"` never
  firing, run `curl http://localhost:8900/api/admin/peers | jq '.[] | {id, trust_score}'`
  and confirm A's score is ≥ 0.5.

## Deferred for a future bench

- **Larger model numbers** (Phi-3.5-mini, Qwen2.5-7B). TinyLlama's
  prefix-cache tensors are small enough that wire transfer + dtype
  cast overhead is barely a win on localhost. The real payoff is WAN
  where the round trip is dominated by the network, not the compute.
- **WAN bench**: two daemons on different machines, different regions.
  Expected 50–150 ms RTT per fetch — still a win vs. seconds of
  prefill on long prompts, but a different shape of curve.
- **Concurrency**: N concurrent fetches on B all pointing at the same
  block hash. The `pending_prefix_kv_inbound` map caps at 256 on A;
  beyond that A replies `None` immediately (see Phase 2b).
- **Compression**: the `WIRE_TAG_PREFIX_KV` frame doesn't zstd-compress
  today (unlike `WIRE_TAG_TENSOR_COMPRESSED`). Prefix KV blocks are
  f32 with some zero-ish regions; a rough estimate is 30–50% wire
  reduction. Add compression in a follow-up once the base case is
  measured.

## What this bench tells us

A successful run validates that the end-to-end cross-node prefix-KV
pipeline **works correctly** under realistic conditions — announce →
index → probe → fetch → verify → hydrate → suffix prefill. The win
size (step 5 vs step 6 TTFT delta) sizes how much latency this path
actually saves.

Even a modest win on TinyLlama-1.1B implies a much larger win on a
7B+ model because prefill cost grows super-linearly with hidden_dim
while wire size scales with hidden_dim × kv_heads. The fetch path
scales *better* than local prefill does.
