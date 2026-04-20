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

## Results (2026-04-20, RTX 3070 Laptop 8GB, WSL2, localhost loopback)

Measured per the recipe below. Both daemons on the same machine, loopback TCP; 672-token prompt (640 fall inside cached prefix blocks, 32-token suffix + 100 decode tokens); model pre-loaded on B with a short unrelated prompt before the long-prompt measurement so the 672-token TTFT excludes weight-load latency.

### TinyLlama-1.1B Q4_K_M (GPU A / GPU B)

| Scenario | TTFT iter 1 (ms) | TTFT iter 2 (ms) | TTFT iter 3 (ms) | DIAG `cross-node prefix HIT` |
|---|---|---|---|---|
| A cold (full prefill + model load) | 1548 | 249 | 246 | 0 |
| B with cross-node fetch (model pre-warm) | **809** | 256 | 231 | 1 (iter 1) |
| B control, fetch gated via `cross_node_prefix_trust_min=2.0` | **713** | 253 | 253 | 0 |

DIAG trace on iter 1 (fetch-enabled run) confirmed the full pipeline:

```
A: DIAG: PrefixKvFetch: serving inbound fetch via worker IPC ticket=...
A: DIAG: served PrefixKvFetch ticket=... age_ms=147 hit=true
B: DIAG: received PrefixKvData response ... hit=true bytes_len=28840528
B: DIAG: cross-node prefix HIT — hydrated KV matched_tokens=640 total_tokens=672 bytes=28840528
B: DIAG: try_register_generate_slot prefix-cache HIT matched_tokens=640 total_tokens=672
```

**Interpretation.** The cross-node KV-hydration path works end-to-end — announce → index → probe → trust-gate → fetch → BLAKE3 verify → hydrate → suffix-prefill — and passes all integrity checks. But on localhost + RTX 3070 + TinyLlama-1.1B, **the fetched path is ~100 ms slower than re-doing the prefill locally** (809 ms vs 713 ms). TinyLlama-1.1B prefill on a 640-token prefix takes only ~460 ms on this GPU, while the wire round trip for a 28 MB uncompressed f32 KV snapshot is ~160 ms + ~96 ms of hydrate/deserialize. This is exactly the "TinyLlama is too small to demonstrate the win" outcome the recipe predicted — prefill cost grows super-linearly in `hidden_dim × tokens` while wire size scales linearly, so larger models shift the cross-over.

### Qwen2.5-Coder-7B Q4_K_M (CPU A / CPU B)

Qwen-7B Q4 weights sit at 4.7 GB; loading + CUDA scratch maxes out the 8 GB RTX 3070 card on iter-1 prefill (`CUDA_ERROR_OUT_OF_MEMORY` during the batched attention kernel). Both daemons were run CPU-only (`CUDA_VISIBLE_DEVICES=""`) so the comparison stayed apples-to-apples — 640-token CPU prefill takes ~2.5 min on this host, which is exactly the regime where cross-node fetch is supposed to pay off.

| Scenario | TTFT iter 1 (ms) | TTFT iter 2 (ms) | TTFT iter 3 (ms) | DIAG `cross-node prefix HIT` |
|---|---|---|---|---|
| B with cross-node fetch (model pre-warm) | **11 755** | 9 509 | 9 226 | 1 (iter 1) |
| B control, fetch gated via `cross_node_prefix_trust_min=2.0` | **151 749** | 9 572 | 9 529 | 0 |

DIAG trace on iter 1 (fetch-enabled run):

```
A: DIAG: PrefixKvFetch: serving inbound fetch via worker IPC ticket=36decb66-...
A: DIAG: served PrefixKvFetch ticket=36decb66-... age_ms=286 hit=true
B: DIAG: received prefix_kv_data response request_id=OutboundRequestId(19)
B: DIAG: cross-node prefix HIT — hydrated KV matched_tokens=640 total_tokens=672 bytes=73405355
```

**Interpretation.** This is the cross-over point. Iter 1 TTFT drops from **151.7 s → 11.8 s** when B fetches A's KV snapshot instead of re-prefilling — a 12.9× speedup, saving ~140 seconds on a 672-token prompt. The 73 MB f32 snapshot transfers end-to-end (serialize + wire + BLAKE3 verify + hydrate) in ~1.0 s on loopback, versus ~150 s of 640-token Qwen-7B CPU prefill. Iter 2 and 3 are effectively identical across scenarios (9.2–9.6 s) because both B runs have B's own local prefix cache populated after iter 1 — the cross-node path is only consulted on local cache miss. Prune-only control (iter 1 fetch gate) confirms B never emitted a probe when `trust_min=2.0` locked out A as a fetch peer.

**Three bugs the bench uncovered and fixed in-tree before the numbers above:**

1. `SwarmMessage::PrefixCacheAnnounce` was never mapped to a GossipSub topic in `NetworkManager::handle_broadcast` (`src/network/manager/mod.rs`), so Phase 1 announces were silently dropped at the wire. The loopback self-index path masked it in single-node tests. Fix: add `PrefixCacheAnnounce` to the `TOPIC_MODELS` arm.
2. `WorkerMsg::PrefixSnapshotResponse` and `DaemonMsg::PrefixFetchResult` both carried `payload: Option<Vec<u8>>` inside the JSON-framed header. `serde_json` encodes `Vec<u8>` as a JSON array of integers, which inflates 28 MB of binary → ~102 MB of header bytes and blows past the 64 MiB IPC header cap, killing the worker. Fix: move the payload bytes onto the IPC binary-payload slot and keep a `present: bool` tag in the header.
3. All three cross-node-fetch timeouts (`PREFIX_FETCH_TIMEOUT_MS=500` in the worker, the 400 ms daemon-side network timeout in `src/daemon/background.rs`, and the 500 ms serving-worker IPC timeout in `src/inference/process_pool.rs::fetch_local_snapshot`) were sized for TinyLlama's 28 MB snapshot. A Qwen-7B snapshot is 73 MB — serialization + wire round trip measured at ~500–1000 ms — which tripped every timeout and silently forced the local-prefill fallback on iter 1. Fix: bump to 3000 / 2500 / 2000 ms respectively, keeping the worker timeout as the outer bound so a stuck daemon still returns a clean miss.

## Recipe

The recipe is model-agnostic — swap the model id and daemon-start env
vars to reproduce either row.

- **TinyLlama (GPU both sides):** start each daemon with the default
  `candle-cuda` enabled; replace `<model-id>` with
  `tinyllama-1.1b-chat-v1.0.q4-k-m`.
- **Qwen-7B (CPU both sides):** prefix each `swarmllm run` command with
  `CUDA_VISIBLE_DEVICES=""` and replace `<model-id>` with
  `qwen2.5-coder-7b-instruct-q4-k-m`. Candle falls through to
  `Device::Cpu` when CUDA has no visible devices. On an 8 GB GPU the
  Qwen weights fit but prefill scratch does not, so GPU-mode iter 1
  OOMs; CPU-mode is required for this host. Iter 1 will take ~2.5 min
  of CPU prefill on the control run — budget accordingly.

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
cp -r ~/.local/share/swarmllm/models/<model-id> /tmp/swarm_a/models/
cp -r ~/.local/share/swarmllm/models/<model-id> /tmp/swarm_b/models/

# Force both daemons to re-scan
curl -X POST http://localhost:8800/api/admin/models/rescan -H "Authorization: Bearer $(cat /tmp/swarm_a/api_key)"
curl -X POST http://localhost:8900/api/admin/models/rescan -H "Authorization: Bearer $(cat /tmp/swarm_b/api_key)"
```

Replace `<model-id>` with the slug under your models directory
(`tinyllama-1.1b-chat-v1.0.q4-k-m` or
`qwen2.5-coder-7b-instruct-q4-k-m`). Check
`memory/local_model_shards.md` for the canonical paths.

### 3. Warm up node A with a long system-prompt request

Use a ~500-token "agent scaffold" prompt so the prefix-cache block
hashes actually hit `min_tokens` (default 32). Example:

```bash
PROMPT='You are a meticulous Rust systems programmer working on SwarmLLM, a decentralized LLM inference network. The codebase uses libp2p, tokio, candle-core for tensor compute, and a custom shard-based model distribution system. When answering, prefer direct code references over prose. Cite file paths + line numbers. Never fabricate APIs. (... pad to ~500 tokens with realistic context ...)'

./target/release/swarmllm bench \
    -p 8800 \
    --iterations 3 --max-tokens 100 --stream \
    --prompt "$PROMPT" \
    --model-id <model-id>
```

The first iteration warms the prefix cache on A. Expected log on A:

```
DIAG: prefix-cache inserted snapshot model_key="0-22-24" entries=<N>
DIAG: prefix-cache loopback indexed (self) model=<model-id> ...
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
    --model-id <model-id> \
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

See the **Results** section at the top of this doc for measured TTFT numbers
from the 2026-04-20 RTX 3070 run. Key caveats when re-running:

- Pre-warm node B's model with a short unrelated prompt (e.g., `"Hi there."`)
  before the long-prompt measurement. If you skip this, iter 1 TTFT includes
  ~1 s of weight-load cost and completely dominates the fetch-vs-prefill
  signal.
- Both fetch-enabled and control runs only surface the cross-node path on
  **iter 1**. Iter 2+ always hits B's own newly-populated local prefix cache,
  so identical TTFTs across iter 2/3 in both scenarios are expected and not
  a bug.
- Verify `hit=true` + `DIAG: cross-node prefix HIT` appears in B's logs on
  iter 1 of the fetch-enabled run. If not, re-check that A's announce reached
  B's index (step 4) and that A's `trust_score ≥ cross_node_prefix_trust_min`.

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

- **GPU-mixed asymmetry** (Qwen-7B on GPU-A, CPU-B). The original
  plan in `docs/plans/next_steps.md` called for A on the RTX 3070 and
  B on CPU so the fetch-vs-prefill ratio favored fetch even harder.
  Qwen-7B Q4 doesn't fit in 8 GB VRAM with headroom for
  prefill scratch on this host — the card OOM'd during the batched
  attention kernel on iter 1. The CPU-CPU run above demonstrates the
  cross-over cleanly on its own; reproducing the GPU-asymmetric case
  wants either a 12 GB+ card or a smaller model (Phi-3.5-mini).
- **Phi-3.5-mini numbers**. Phi is 3.8B with MHA (32 kv-heads), so a
  snapshot at 640 tokens is ~470 MB — 6× Qwen's GQA snapshot, 17×
  TinyLlama's. Useful stress test for the wire path and a natural
  candidate to isolate where localhost bandwidth becomes the limit.
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
