# Troubleshooting

## Start here: why was that request slow, or where did it fail?

SwarmLLM logs to the window it runs in. To keep a file on Linux or macOS,
start it with `./swarmllm run >> node.log 2>&1`. Docker: `docker compose logs`;
.deb package: `journalctl -u swarmllm`.

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
| `queue_ms` large | this node is saturated — raise `max_concurrent_requests` |
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
swarmllm diagnostics
```

This is the single most useful thing to attach to a bug report, and it is safe
to post in public: no API key, no invite code, no file paths, and every network
address — yours and your peers' — replaced by a placeholder naming only its kind.
Add `--full` if you are debugging your own machine and need the addresses back. It includes
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
3. Check you downloaded a GPU build (`-gpu` on Windows, `-cuda` on Linux)
4. On Linux the card must be an NVIDIA RTX 30-series or newer; older cards use the processor automatically and say so in the log
5. Macs run on the processor only for now

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

## A worker keeps working after the client gave up

Symptoms: a client timed out or was closed, the log says the request was
cancelled, and `top` still shows a `swarmllm model-worker` process at full CPU
minutes later — on a processor-only machine, sometimes long enough to trip a
thermal warning.

Since v0.3.152 a cancel reaches every place a request can wait, so this should
stop within one prompt chunk (about 128 tokens' worth of work, seconds on a
processor). To see what the node is computing and stop a worker without
`kill -9`:

```bash
./swarmllm status                     # "Workers:" lists each model worker, its pid,
                                      # card or processor, and requests in flight
./swarmllm unload <model-id>          # retire that worker; the files stay, it
                                      # loads again on the next request
```

If a worker stays busy longer than a chunk after the client has gone, the cancel
did not reach it — `docs/DIAGNOSTICS.md` § "The client left — did the work
stop?" lists the log lines to check, in order.

## Slow First Request

If the first inference request to a model takes noticeably longer than subsequent ones, this is expected. SwarmLLM uses **on-demand model loading** — models whose shards are on disk but not loaded into VRAM are loaded when first requested. If VRAM is full, an LRU eviction occurs first. Subsequent requests to the same model will be fast.

## Slow Inference

1. **GPU vs CPU:** CPU is 5-20x slower. Check Dashboard for GPU status.
2. **Model too large:** Use Q4 quantization, match model size to VRAM.
3. **Far-apart helpers:** a model split across distant computers is slow, because each word of the reply passes through every one of them. `swarmllm diagnostics` lists the slowest computers first.

## Database Corrupted

```bash
# Stop SwarmLLM first, then back up just the database
cp ~/.local/share/swarmllm/db.redb ~/swarmllm-db-backup.redb
rm ~/.local/share/swarmllm/db.redb
./swarmllm run
```

Models and `config.toml` are kept, but your **Access Token changes** (apps
using the old one need the new one) and this device leaves its My Devices
group.

## GPU Out of Memory

If a model is bigger than your graphics card, SwarmLLM puts as many of its
layers on the card as fit and runs the rest on the processor. If loading on
the card still runs out of memory, it retries on the processor and logs:

```
GPU OOM — retrying model load on CPU
```

The processor is 5-20x slower, and a model that doesn't fit in the memory
SwarmLLM may use is refused rather than swapped. To avoid OOM:
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
- Splitting a model adds one network round trip per computer for every word of the reply — fast on a local network, slow across continents
- Tensor parallelism is off by default (`inference.tensor_parallel`) and only helps on a fast local network

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
- Concurrency is capped by `inference.max_concurrent_requests` (default
  10), and any one requester may use at most half of it — the same for
  everyone, since credits are dormant and gate nothing. Excess requests queue
  until earlier ones finish; raise the setting to allow more.

## Cross-Node Prefix-KV Sharing

Off by default: a computer only shares its prefix cache if its owner sets
`inference.share_prefix_cache_with_peers = true`, because announcing it
reveals hashes of the prompts it has cached. Expected logs on a successful
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

**Stop sharing your own prefix cache:**
Leave `inference.share_prefix_cache_with_peers = false` (the default).

## Running the Test Suite

Building SwarmLLM and running its tests is covered in
[CONTRIBUTING.md](https://github.com/enapt/SwarmLLM/blob/main/CONTRIBUTING.md).
See [Benchmarking](./operations/benchmarking.md) for reproducing the
performance benchmarks and [Performance](./operations/performance.md)
for which knobs turn each speedup on/off.

## A model the swarm never picks up

Computers only download parts of a model on their own once it has earned
some trust, so a model from an unknown source can't make everyone download
gigabytes. The trust levels you may see in the app are "Manually approved by
you", "Has received real inference requests" and "Widely hosted across the
network".

- **Download any part of it yourself** (see [First Model](./getting-started/first-model.md)) —
  that marks it as approved by you on this computer.
- Popular models on HuggingFace are approved automatically once they are at
  least a day old and have enough downloads: 10,000 for well-known publishers
  (meta-llama, mistralai, Qwen, google, microsoft, deepseek-ai, bartowski,
  TheBloke, unsloth and others), 100,000 for everyone else.
- A model also earns trust when people send it requests, or when many
  computers host it.

## Chat dropdown shows "No models available yet"

This is the cold-start state. Click **Get shared test model** in the Chat tab
to download a small model everyone can use and start chatting straight away.
Once your computer hears from others, the Chat tab also shows three rows:

- **"Available right now on the swarm"** — models the swarm can run for you
  today. Click one to select it and open a fresh chat.
- **"The swarm is gathering these"** — the swarm has some of the parts; they
  will be ready once the rest finish downloading.
- **"Popular models the swarm could adopt"** — popular models nobody is
  running yet. Clicking one opens the model search.

If none of these appear, your computer hasn't heard from any others yet.
Check the **Computers** panel on the Dashboard, or run `swarmllm peers`, to
see whether it found other computers. A computer with
`hf_watcher_enabled = false` won't see the "could adopt" row by design.

## Still Stuck?

- Run `./swarmllm diagnostics` and paste its output into your report — it is
  safe to post publicly
- For more detail in the log: `./swarmllm run -vv 2>&1 | grep "DIAG:"`
- See the [Diagnostics Guide](https://github.com/enapt/SwarmLLM/blob/main/docs/DIAGNOSTICS.md) for detailed log instrumentation
- Check [GitHub Issues](https://github.com/enapt/SwarmLLM/issues)
- Open a new issue with your OS, hardware, `./swarmllm version`, and the
  `swarmllm diagnostics` output
