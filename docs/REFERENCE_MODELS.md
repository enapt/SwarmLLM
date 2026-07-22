# Reference test models

Standard models for testing SwarmLLM across a swarm. Pinned so that results
from different machines are comparable — "Llama-3.2-3B Q4_K_M" is not one
artifact, and bartowski, unsloth and lmstudio-community all publish different
quantizations of it. Use these exact repo/filename pairs or the numbers do not
mean anything next to each other.

## The three tiers

| Tier | Model | Download | `shard_size_mb` | Shards | What it is for |
|---|---|---|---|---|---|
| **Smoke** | TinyLlama-1.1B-Chat v1.0 Q4_K_M | 638 MB | 512 (default) | 2 | "Is this node routing at all" |
| **Standard** | Llama-3.2-3B-Instruct Q4_K_M | 1925 MB | 512 (default) | 4 | Performance + networking. The default choice. |
| **Stress** | Meta-Llama-3.1-8B-Instruct Q4_K_M | 4692 MB | 512 (default) | ~10 | VRAM pressure, deep pipelines, OOM paths |

```
Smoke     TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
          tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

Standard  bartowski/Llama-3.2-3B-Instruct-GGUF
          Llama-3.2-3B-Instruct-Q4_K_M.gguf

Stress    bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
          Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

All three publishers are on the `TRUSTED_HF_PUBLISHERS` allowlist
(`src/model/huggingface/watcher.rs`), so auto-manage will adopt them at the
10k-download threshold rather than the 100k one.

## Why Standard is the default

1925 MB fits a 6 GB card with room for the KV cache, runs at roughly 10 tok/s
on CPU so no-GPU nodes stay in the comparison, and splits four ways. Every node
can also hold it whole, so local-only and distributed runs are comparable on
identical weights.

It is a `llama`-architecture model on purpose. This codebase has separate paths
for `phi3` (fused QKV/FFN), `gemma2`, `qwen35`, and MoE/MLA; a test model on one
of those means every anomaly has to be ruled out as an architecture bug before
it can be blamed on the network. `llama` with GQA is the best-covered path,
which is what a control should be.

## Shard count, hop count, and why they are not the same thing

The obvious worry about more shards is more network hops. That is not how it
works, and the distinction matters when choosing a shard size.

`gather_candidates` does not route per shard. It builds a **layer bitmap** from
everything a node holds and extracts contiguous runs
(`inference/shard_layout.rs::available_layer_ranges_from_manifest`). A node
holding shards 0-3 contributes **one** segment covering all their layers, not
four. So:

> Hop count is the number of *nodes* in the assembled pipeline. Shard count only
> decides how finely work *can* be divided, not how finely it *is*.

Eight shards across two nodes is two segments and one hop — identical to two
shards across two nodes.

`shard_count = ceil(total_size / shard_size_mb)` (`huggingface/probe.rs`), and
`model.shard_size_mb` is configurable (64 MB minimum, 512 default), so depth is
tunable independently of model size:

| `shard_size_mb` | Shards for Llama-3.2-3B (1925 MB) |
|---|---|
| 512 (default) | 4 |
| 256 | 8 |
| 128 | 16 |

**The default is still the right choice, for two reasons that only show up
under auto-manage.**

*Fragmentation.* Contiguous runs merge, but non-contiguous ones do not. Nothing
in acquisition scoring prefers contiguity — `auto_manage/scoring.rs` ranks
shards by rarity and demand, so a node can end up holding 0, 1, 4, 5. The
bitmap turns that into **two** ranges, and a pipeline that could have been one
hop becomes two. Smaller shards give that more opportunities to happen, and the
effect is invisible until you look at segment counts.

*Bookkeeping.* Every shard carries a holder record, a DHT provider record, and
an entry in each announce. `MAX_SHARDS_PER_ANNOUNCE` is 512, so 16-shard models
put a real ceiling on how many models one node can announce at once.

Lower `shard_size_mb` when you are placing shards **deliberately** — the
`3node_sharded_setup.sh` split assigns contiguous halves by hand, so
fragmentation cannot occur and finer granularity is pure benefit. Leave it at
the default when auto-manage is deciding placement.

## Pitfall: everyone must agree on `shard_size_mb`

Shard layout is decided by whichever node first probes the model, and travels
with the manifest. Two nodes that independently download the same model with
different `shard_size_mb` values produce **different manifests for the same
weights**, and their shards will not interoperate.

If you do change it for a deliberate split, set it on every node *before* the
first download, or let one node acquire the model and allow the manifest to
propagate.

## Running a comparable test

The bench scripts take the model from the environment:

```bash
# Smoke (default)
examples/3node_setup.sh && examples/3node_inference_bench.sh

# Standard
SWARM_BENCH_MODEL=llama-3.2-3b-instruct-q4-k-m examples/3node_inference_bench.sh

# Forced distributed split, any shard count
SWARM_BENCH_MODEL=llama-3.2-3b-instruct-q4-k-m examples/3node_sharded_setup.sh
```

For results to be comparable, hold these fixed: the same prompt, the same
`max_tokens`, and `temperature: 0.0` (the bench script already sets the last
two — sampling at any other temperature makes runs incomparable by design).

**Turn auto-manage off before building a deliberate split.** With
`auto_manage.min_replicas = 2` a node re-downloads shards you removed within
seconds, and the split you were testing dissolves under you. Set
`auto_manage.enabled = false` on every node in the test.

**An anchor node cannot take part.** `anchor_mode` skips the inference
subsystems entirely — it contributes bootstrap and relay, not compute.

## Opting in

Reference models are never fetched automatically. They exist to test the swarm,
and quietly spending a user's bandwidth and disk for that is not a reasonable
default — so acquiring one is always something a person chose to do.

```bash
examples/fetch_reference_model.sh --list            # what is available
examples/fetch_reference_model.sh standard          # host your fair share
examples/fetch_reference_model.sh standard --all    # host every shard
```

The default uses the `peer_fair_share` mode of
`POST /api/admin/hf/download-shards`, which gives this node a slice sized
against how many peers are participating rather than the whole model. That is
usually what you want for a swarm test: the model ends up distributed across
the machines taking part, which is the thing being tested. Use `--all` when you
want one node able to serve the model alone, for a local-vs-distributed
comparison.

Either way it is the same path any model takes — nothing about a reference
model is special to the daemon.

## What each tier will and will not show you

- **Smoke** cannot measure throughput. Two shards is one hop, and at 1.1B the
  compute is small enough that the numbers are dominated by overhead. It answers
  "does routing work", nothing more.
- **Standard** is the one to quote. Deep enough to exercise multi-hop routing,
  failover to a standby, and hedging; small enough that a CPU-only node still
  participates at roughly 10 tok/s rather than dropping out of the comparison.
- **Stress** is for the paths only pressure reveals: VRAM exhaustion, the CPU
  fallback after a GPU OOM, and pipelines deeper than most swarms will build.
  It will not fit comfortably on a 6 GB card, which is the point.
