# Troubleshooting

## Start here: why was that request slow, or where did it fail?

Every completed request writes **one** summary line. Read it before anything
else — it usually identifies the problem on its own:

```bash
grep "DIAG: request complete" node.log | tail -5
```

```
DIAG: request complete request_id=1ddd2912-… route=distributed segments=2
  nodes=0718d8b9,96842635 regions=TH,TH queue_ms=3 sched_ms=1 ttft_ms=180
  decode_ms=1420 total_ms=1604 tokens=48 tok_per_sec=33.8
  seg0_ms=520 seg1_ms=900 outcome=ok
```

| What you see | What it means |
|---|---|
| `queue_ms` large | this node is saturated — raise `max_concurrent_requests`, or your credit tier is capping you |
| `sched_ms` large | the scheduler is struggling to find holders — check the peer table below |
| `ttft_ms` large, `decode_ms` small | prefill or a cold model load. Not the network |
| `decode_ms` large | per-token cost — find the slow hop in the `segN_ms` values |
| one `segN_ms` dominates | that peer is the bottleneck |
| `route=relayed` | no direct path to a holder, so traffic takes an extra hop each way |
| `outcome=error error_type=…` | the name points at the subsystem that failed |

A missing field means "not measured", never zero. `ttft_ms` and `decode_ms` are
absent on requests that did not stream, because there is no honest way to split
decode out of the total there.

### No log file to hand?

```bash
curl -s -H "Authorization: Bearer $(cat ~/.local/share/swarmllm/api_key)" \
  localhost:8800/api/admin/diagnostics
```

This is the single most useful thing to attach to a bug report. It includes
whether your machine is reachable from the internet, the last 50 requests with
their routes, **per-peer serving performance** (ping, ms per layer, latency,
region — slowest first), what your node has served for others, and recent
failures with *which machine served each one*. That last detail is what separates
"my node has a problem" from "one peer has a problem".

### Diagnosing from the client side

You do not need server access. Every response carries its route:

```bash
curl -i -X POST localhost:8800/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"…","messages":[{"role":"user","content":"hi"}]}' \
  | grep -i '^x-swarm-\|^server-timing'
```

```
x-swarm-route: distributed
x-swarm-peers: 1
x-swarm-nodes: 0718d8b9,96842635
server-timing: queue;dur=3, sched;dur=1, ttft;dur=180, decode;dur=1420
```

On a **streaming** response `Server-Timing` carries only what is known before the
body starts — queue and scheduling. Token-level figures arrive at the end of the
stream, because a header cannot be revised once sent.

## Can't Connect to Peers

**Check the bootstrap address format:**
```
/ip4/203.0.113.50/udp/8800/quic-v1/p2p/12D3KooW...
```

**Firewall:** SwarmLLM needs **TCP port 8810** (P2P) and optionally **UDP port 8800** (QUIC) open.
- **Linux:** `sudo ufw allow 8810/tcp && sudo ufw allow 8800/udp`
- **Windows:** Windows Defender Firewall > Inbound Rules > New > Port > TCP 8810 + UDP 8800
- **macOS:** System Settings > Network > Firewall > allow SwarmLLM

**Same LAN?** Use local IP (e.g., `192.168.1.x`). LAN peers should be found automatically via mDNS.

## Model Download Stuck

1. Check disk space — a 7B model needs ~4-5 GB free
2. Verify internet access to `https://huggingface.co`
3. Cancel and retry from the Dashboard
4. Start with `-v` for verbose logs: `./swarmllm run -v`
5. Try a smaller model first (TinyLlama, ~700 MB)

## GPU Not Detected

1. Verify GPU works: `nvidia-smi`
2. Install NVIDIA drivers if needed
3. Enable GPU offloading: `./swarmllm run --gpu-layers 99`

**WSL2 users:** The CUDA driver comes from your Windows NVIDIA driver. Check that `/usr/lib/wsl/lib/libcuda.so.1` exists and add to your `~/.bashrc`:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

## Port Already in Use

```bash
./swarmllm run --port 9000    # Use a different port
lsof -i :8800                 # Find what's using 8800
./swarmllm status             # Check if another instance is running
```

## Dashboard opens but nothing saves

Symptoms: the page loads from another device, but the setup wizard's "Start
SwarmLLM" button appears to do nothing, settings won't save, and panels sit
empty. The hardware panel may say "CPU only" on a machine that has a GPU.

Every admin call is returning 401 because the page was never handed an access
key. The daemon only hands it out automatically over networks it trusts:
loopback always, a Tailscale-style overlay when this node is on one too, and a
private/LAN address only if you opted in.

A banner at the top of the page states this, and — importantly — names the
address the daemon actually saw for you. Behind a NAT, a container publish, or a
Tailscale **subnet router** that is *not* the address in your browser's address
bar, because those rewrite the source address by default.

Two ways through:

- Paste your access key into the banner. It's in the `api_key` file in the data
  directory (read it from inside the container, if that's where SwarmLLM runs),
  and is remembered per node so this is a one-time step per browser.
- Turn on **Allow access from my local network** in Settings → Identity &
  Access. This applies immediately without restarting the node, which matters
  when the node you can't reach is the one you'd have to restart.

Inference and the OpenAI/Anthropic APIs are unaffected — they accept the key as
a Bearer token from any address. Full detail in
[Tailscale / WAN](operations/tailscale-wan.md).

## Slow First Request

If the first inference request to a model takes noticeably longer than subsequent ones, this is expected. SwarmLLM uses **on-demand model loading** — models whose shards are on disk but not loaded into VRAM are loaded when first requested. If VRAM is full, an LRU eviction occurs first. Subsequent requests to the same model will be fast.

## Slow Inference

1. **GPU vs CPU:** CPU is 5-20x slower. Check Dashboard for GPU status.
2. **Model too large:** Use Q4 quantization, match model size to VRAM.
3. **Enable batching:** Set `max_batch_size = 4` in config.

## Database Corrupted

```bash
# Back up first
cp -r ~/.local/share/swarmllm ~/.local/share/swarmllm-backup
# Delete database (models and config are preserved)
rm ~/.local/share/swarmllm/db.redb
# Restart
./swarmllm run
```

## GPU Out of Memory

If a model exceeds your GPU's VRAM, SwarmLLM automatically falls back to CPU inference. You'll see this in the logs:

```
WARN GPU OOM detected, retrying on CPU
```

CPU inference is 5-20x slower but works for any model size. To avoid OOM:
- Use smaller quantizations (Q4 instead of Q8)
- Use a model that fits in VRAM (check model size vs available VRAM in the dashboard)
- For models too large for one GPU, use distributed inference across multiple nodes

## GPU Memory Stays Full When Idle

Seeing high VRAM use with little activity is usually **expected, not a leak**. Your
node keeps models loaded in GPU memory so it can serve the swarm without a cold
start. How much it commits is set by your **contribution** level
(`[node] contribution` — minimal / moderate / maximum, also on the dashboard).

The daemon reclaims VRAM in two ways:

- **Demand-driven (before pressure):** a model with no local requests for
  `[auto_manage] idle_unload_secs` (default 5 min) **and** low network demand is
  unloaded from GPU memory automatically. Its shards stay on disk, so it reloads
  (one cold start) on the next request — your holder status never changes. Set
  `idle_unload_secs = 0` to keep every loaded model resident. Deliberately-held
  models are never idle-unloaded — reference/test models (`swarmllm get-model`),
  pinned or locked models, and encrypted-pipeline models stay resident.
- **Pressure-driven (automatic):** above **70%** VRAM the daemon narrows loaded
  models to fewer shards; above **95%** it fully unloads a model. Both keep shards
  on disk.

To free VRAM immediately, restart the daemon, or lower `contribution`. There is no
leak here — the `model-worker` subprocess holding a model is killed (freeing all
its GPU memory) whenever the daemon unloads it by either path above.

## Distributed Inference Issues

**Peers visible but inference fails:**
1. Ensure both nodes have the required shards loaded (check Dashboard > Models)
2. Verify P2P TCP connectivity: port `<base_port> + 10` must be reachable
3. Run with `-vv` and filter: `./swarmllm run -vv 2>&1 | grep "DIAG:"`
4. Check for `DIAG: segment TIMED OUT` — indicates network or compute bottleneck

**High latency per token:**
- Distributed inference adds ~20-130ms per token for network round-trips
- Use TCP bootstrap addresses (not QUIC) for lowest latency
- Ensure nodes are on the same LAN for tensor parallelism

**Pipeline assembly fails:**
- The scheduler needs enough shard coverage to build a complete pipeline
- Check `DIAG: assemble_pipeline_for` for candidate counts

**Inference fails with "peer never acknowledged" or "silent drop":**
- A `SendDirectMessage` was issued but neither a Response nor an
  `OutboundFailure` event arrived from libp2p within 10s
  (`RR_ACK_TIMEOUT_SECS`). Treated as a transient failure: the router
  automatically retries once with a fresh pipeline assembly that
  filters out the unreachable peer. If retry also fails, the user
  sees the error within ~20s (vs the 120s `FIRST_TOKEN_TIMEOUT`).
- Most common cause: the target peer was killed or partitioned and
  the local libp2p connection state hasn't yet caught up.
- Look for `DIAG: rr ACK timeout — closing streaming caller` in
  the logs to confirm the fast-fail path engaged.

**Concurrent requests stall when only some get dispatched:**
- Per-tier concurrency caps come from `inference.max_concurrent_requests`
  (default 10): Bronze=2, Silver=5, Gold=10, Platinum=20. Excess
  requests queue until prior ones complete. To raise: bump the config
  knob or earn credits to climb tiers.
- If queued requests don't dispatch even after others complete,
  check for a missed `queue_notify.notify_one()` after
  `active_count.fetch_sub(1)` (should never happen on `main`; was a
  real regression fixed in `da6f485`).

## Cross-Node Prefix-KV Sharing

The cross-node prefix fetch is default-on. Expected logs on a successful
first hit of a peer's cached prefix:

```
B: DIAG: cross-node prefix HIT — hydrated KV matched_tokens=N total_tokens=M
A: DIAG: served PrefixKvFetch ... hit=true
```

**I never see `cross-node prefix HIT`:**
- Only fires on iter 1 of a prompt whose prefix your local node hasn't
  prefilled yet. Iter 2/3 hit the local cache (populated by iter 1).
- Check the peer even announced the prefix: look for
  `DIAG: PrefixCacheAnnounce indexed node_id=... blocks=N` in your log.
  No announce → peer's gossip never reached you (check
  `grep 'Published message to GossipSub' | grep 'swarm/models'`).
- Check the peer passes the trust gate: default
  `cross_node_prefix_trust_min = 0.5` equals `DEFAULT_TRUST`, so a
  freshly-seen peer should just barely pass. Any misbehavior drops it
  below.

**I see `prefix-probe: fetch timed out`:**
- The peer didn't return a snapshot inside the worker-probe window
  (3000 ms by default). On a large model (7B+) with cold CPU this can
  happen if the snapshot is >100 MB. The path degrades to local prefill
  — no worse than not having the feature. The current 3000/2500/2000 ms
  chained timeouts are sized for 7B-class snapshots; the older
  500/400/500 ms values were TinyLlama-sized and forced a fallback to
  local prefill on larger models.

**I see `rejected KV snapshot — penalizing peer trust`:**
- The returned snapshot failed BLAKE3 reverification or contained
  NaN/Inf. Three rejection reasons:
  - `hash_chain_mismatch` → `prefix_cache_block_tokens` differs between
    nodes (default 64, common alternatives 32/128)
  - `non_finite_tensors` → GPU overflow on the serving side
  - `deserialize_failed` → wire corruption — open an issue

**Disable cross-node fetch entirely:**
Set `inference.cross_node_prefix_trust_min = 2.0` in `config.toml`. The
probe never fires because no peer passes the trust gate.

## Running the Test Suite

SwarmLLM ships 1158 lib tests + 75 integration tests + VLM E2E.

```bash
# Run all tests (release, used in CI)
cargo test --release

# Unit tests only (fastest feedback loop)
cargo test --lib

# Integration tests only
cargo test --test '*'

# A specific test by name substring
cargo test --release prefix_cache

# With CUDA features on (requires NVIDIA GPU)
cargo test --release --features candle-cuda
```

If a test fails, the release build shows the name + line; rerun with
`--nocapture` to see its stderr:

```bash
cargo test failing_test_name -- --nocapture
```

Integration tests under `tests/integration/` simulate multi-node P2P on
loopback — they're the slow ones, and CI runs them with
`--test-threads=1` to avoid port contention.

See [Benchmarking](./operations/benchmarking.md) for reproducing the
performance benchmarks and [Performance](./operations/performance.md)
for which knobs turn each speedup on/off.

## Model Trust

Models go through trust levels: Discovered → Pinned → DemandVerified → NetworkPopular. Auto-manage only downloads shards for models at sufficient trust levels.

**Model stuck at "Discovered":**
- Pin it manually from the Dashboard to promote to "Pinned"
- Models reach "DemandVerified" after receiving inference requests
- Models reach "NetworkPopular" when enough peers host them
- **R141**: HfWatcher auto-promotes `Discovered` → `DemandVerified` for trending HF models above the per-publisher download floor + 24h age:
  - **Trusted curators** (meta-llama, mistralai, Qwen, google, microsoft, deepseek-ai, bartowski, TheBloke, unsloth, etc. — full list in `src/model/huggingface/watcher.rs::TRUSTED_HF_PUBLISHERS`) promote at **10k** downloads
  - **Unknown publishers** promote at **100k** downloads
  - Both tiers respect the 24h age gate (defeats download-pump attacks)
- Failed promotions accrue strikes that exponentially extend the cooldown — 4 strikes blocks auto-promotion until you pin it manually

## Chat dropdown shows "No models available yet"

This is the cold-start state. R141 surfaces actionable swarm-available
models directly in the chat empty state — the dashboard renders three
rows when no model is selected:

- **"Available right now on the swarm"** — Hosting + Serveable wishlist
  entries the swarm can route inference to today. Click any chip to
  select that model and open a fresh chat.
- **"The swarm is gathering these"** — Aspirational entries (partial
  shard coverage on the network). Will be ready as the missing parts
  finish downloading.
- **"Popular models the swarm could adopt"** — HF trending Candidate
  entries the swarm doesn't have yet. Click to route to the HF browse
  pre-filtered to the repo so you can pick the quant variant.

If none of these appear: your node hasn't received any peer gossip yet
AND HfWatcher hasn't returned a snapshot. Check **Settings → Connection**
to verify the daemon found bootstrap peers; an air-gapped node with
`hf_watcher_enabled = false` won't see Candidate entries by design.

## Still Stuck?

- Run with full diagnostics: `./swarmllm run -vv 2>&1 | grep "DIAG:"`
- See the [Diagnostics Guide](../../DIAGNOSTICS.md) for detailed log instrumentation
- Check [GitHub Issues](https://github.com/enapt/SwarmLLM/issues)
- Open a new issue with: OS, hardware, `./swarmllm version`, and logs from `-vv`
