# As fast as local, or faster: what distance allows, and the plan

**Written 2026-09-27 from a deep dive**: three research surveys (limits of split
decode, big models on one machine, where many machines beat one), the replayed
speculation measurements in `split_speculation.md`, and the v0.3.210 gate. The
question was: *a split across two GPUs decodes at 2.85 tok/s where each GPU
alone does 11-44 — how does the swarm become as fast as local, or faster?*

## 1. The limit, and why no scheduling trick removes it

Token t+1 enters the first layer only after token t has left the last. When
the layers are on two machines, every token crosses A→B and back:

    time per token  ≥  compute of every layer  +  one round trip

Nobody has published a theorem for this, because it is simply the dependency.
Petals measured it to the millisecond: Llama-2-70B on three T4s lost exactly
200 ms per token when a 50 ms one-way delay was added to each of its four hops
(arXiv 2312.08361, Table 2). With a 206 ms ICMP round trip Thailand↔Belgium,
**exact decoding of an unmodified model tops out near 4 tok/s**, and nothing
about the GPUs changes that.

**Where splitting a model IS faster than one device:** Groq (~40 µs per layer
across ~576 chips), Cerebras (~110 µs per token of link latency across four
wafers), TPU tensor parallelism (Pope et al., 2211.05102), and Exo over
Thunderbolt RDMA (<50 µs). All of them add up memory bandwidth across chips
over MICROSECOND links. Our RTX 3070 spends ~0.8 ms per layer (22.7 ms for 28),
so a layer split stays within 10% of local only below **~2 ms** round trip, and
tensor parallelism needs ~56 syncs per token each well under 400 µs. A 206 ms
link is two to three orders of magnitude past both. That is LAN territory, and
nothing else.

The only three ways to break the cycle for an off-the-shelf model:

1. **Do not cross the network per token** — keep decode on one machine.
2. **Shorten the round trip** — machines close to each other; our own overhead gone.
3. **Compute a token before the one before it is known** — verified prediction
   (speculative decoding, Lookahead). Output provably identical (Leviathan
   2211.17192, Chen 2302.01318); only the speed depends on how often it is right.

Everything else found needs a retrained model (StagFormer 2501.15665, Parallel
Loop Transformer, Ladder Residual, CLLM, parallel attention+MLP) or µs RDMA
plus large batches (Lamina, MegaScale-Infer, Step-3 AFD). Layer Parallelism
(2502.02790) is training-free but lossy (Llama-2-7B perplexity 6.26 → ~9.1) and
still needs NVLink-class links.

## 2. What that means, case by case

| The user's situation | Fastest honest answer | Expected |
|---|---|---|
| Model fits their GPU | Run it locally. The swarm speeds up what is NOT per-token: prompts already read elsewhere, independent requests fanned out | Decode = local; agent turns 5-20× faster to first token (§3.1) |
| Model is an MoE that fits GPU + RAM | Local, experts in system RAM, everything else on the card | Qwen3-30B-A3B class: **30-45 tok/s** on an 8 GB card + 32 GB RAM (measured by others, §3.2) |
| Dense model a little too big for the card | Local: a quantization that fits the card, or card + processor with the processor part fast | 14B: 3.35 → ~8 tok/s by fixing our CPU half (§3.3) |
| CPU-only user, long prompt | Hand the prompt to a GPU peer | 30-50× faster to first token (§3.1) |
| Model fits nowhere, machines close (≤ 25 ms) | Split + verified prediction with a small drafter | ~local speed (`split_speculation.md` projection: 38-50 tok/s at 25 ms) |
| Model fits nowhere, machines far (~300 ms) | Split + verified prediction with a SHADOW of the far layers | 8-15 tok/s with a small drafter; 25-35 with a shadow (§3.5) |

## 3. The levers, ranked

### 3.1 Where many machines genuinely beat one (no per-token network cost)

- **Prefix affinity.** Agent traffic is ~100:1 input to output, and 85-97% of a
  turn's tokens repeat the last turn (Preble 2407.00023, Manus). Route a request
  to the node that already holds its prefix's cache instead of re-reading it:
  Prompt Cache 8× on GPU and 60× on CPU (2311.04934); Preble 1.5-14.5×. A request
  is kilobytes and its cache is hundreds of megabytes, so move the request. This
  node already prices ITS OWN warm prefix (`scheduler::cached_prefix`); the gap is
  peers' prefixes (gossiped digests), a TTL beyond the 10-minute idle expiry
  (agent sessions pause longer), and a RAM/disk tier.
- **CPU users hand long prompts to GPU peers.** A 10K-token prompt on a 7B:
  ~14 s on an RTX 3070 against 7-14 minutes on a laptop CPU. The delegation
  path exists (`delegation_target`); the cost model must compare the requester's
  measured prompt rate with round trip + the peer's rate.
- **Fan-out.** Independent requests (subagents, parallel tool branches, best-of-n)
  to distinct warm peers: ~2.8-3× over one 3070 batching four, bounded by how much
  of a task is parallel (research agents: a lot; coding agents: little).
- **Prompt reading pipelined across a split** (`split_speculation.md` Phase 3):
  large where the alternative is CPU offload; ≤ 1.7× for a model that fits.

Ruled out for our links: ring/sequence/tensor parallelism (needs ~460K-token
blocks at 100 Mbit/s to hide transfer), shipping KV caches between machines for
a GPU user (573 MB fp16 for 10K tokens of a 7B; wins only for CPU users).

### 3.2 Run MoE models the way consumer machines can

Decode speed ≈ memory bandwidth ÷ bytes read per token, and an MoE reads only its
active experts. llama.cpp's `-ot exps=CPU` / `--n-cpu-moe` keeps attention, the
router and shared weights on the card and the experts in system RAM:
Qwen3-Coder-30B-A3B at **32.5 tok/s** on an 8 GB RTX 3060 Ti, a 35B-A3B at
**44.4** on an 8 GB RTX 2070S; gpt-oss-20b at 56-64 on a 12 GB card. Our hybrid
placement is per LAYER (`split::hybrid::LayerPlacement`), so a large MoE today
goes processor-heavy for everything. **Per-tensor placement — experts on the
processor, the rest on the card — is the change**, with MoE loading already
quantized per expert (`split::loader::load_moe_ffn`).

### 3.3 Make our own card + processor path as fast as the memory allows

Our 14B at 3.35 tok/s with 28 of 48 layers on the processor reads ~4.9 GB per
token from RAM: ~17 GB/s effective, about a third of dual-channel DDR4. The first
suspects: the decode thread count for the OWNER's request on a hybrid worker (the
contribution level caps decode at half the cores; `cpu_pools::in_phase_pool`
widens only prompt reading), then our quantized matvec against llama.cpp's.
Rule of thumb from the limit above: at ~50 GB/s, one 206 ms round trip reads
~10 GB of weights, so **local offload beats a far split whenever less than ~10 GB
sits off the card** — which is every 14B.

### 3.4 Remove our own overhead from every round trip (exact, no prediction)

- **Protocol negotiation costs a full round trip per token on the default path.**
  libp2p opens a substream per request-response message and negotiates it with
  multistream-select `Version::V1`, which is "always at least one dedicated
  round-trip message exchange before application data" (multistream-select
  0.13.0 docs). `Version::V1Lazy` makes it 0-RTT when the dialer offers one
  protocol. The request-response crate is vendored, so the tensor protocol's
  outbound substreams can take V1Lazy alone, leaving gossip/kad/identify as they
  are. Measured per token today: 838 ms on request-response vs 351 ms on the
  persistent stream, at a 412 ms application-level round trip.
- **The persistent stream** (2.4×, measured) waits on #133 (a peer stalling on a
  large prompt frame; no receipt ACK). Chunking the prompt pass (Phase 3)
  removes the large frame.
- **~110 ms per token is still unexplained** on the stream path (351 ms against
  206 ms ICMP + ~35 ms compute). Candidates: relay vs direct, libp2p ping vs ICMP
  on the same connection, `tcp_slow_start_after_idle` resetting the window below
  one fp32 frame. Measure before fixing.

### 3.5 Verified prediction for the split that cannot be avoided

`split_speculation.md` has the measurements. In one line: time per token ≈
1/v + m·RTT, where v is the drafting side's local speed and m how often the far
side disagrees. A small same-family drafter (m ≈ 26%) gives 8-15 tok/s at 300 ms
and near-local at 25 ms. A **shadow** of the far layers (a 2-3-bit copy on the
near machine; its errors cost speed, never correctness) is estimated at
α ≈ 0.95-0.97 for a Q3 remote half (ML-SpecQD measured Qwen2.5-Coder-7B at 91% for
an MXFP4 whole-model copy against 62% for its 0.5B sibling) → 25-35 tok/s at
300 ms. **Shared randomness** (Gumbel-max with a position-keyed seed, "drafter-
invariant speculative decoding", Daliri et al. 2408.07978) keeps that agreement
at the temperatures clients actually send. Measurement: `~/swarmllm-ref/spec/`
`shadow.py` + `coupling.py`, results below when run.

## 4. The order of work

Status 2026-10-02: items 1, 2 and 6 have shipped; 3-5 and 7 are open (the
persistent stream is still off by default). `docs/FUTURE_WORK.md` carries them: item 3 is #137,
item 4 is #138, item 5 is #10, item 7 is #133 (with V1Lazy, request-response is as fast, so
there is no speed reason to default the stream on). The shadow of §3.5 is decided against as a
default (#144, § "Decided"): no design may need a user to hold a whole model, even at low bits.

1. ✅ **Ship v0.3.211**: the card/processor batched-forward fix found at the .210
   gate (every concurrent request to a model split between card and processor
   failed on .209 and earlier).
2. ✅ **V1Lazy for the tensor protocol** (§3.4) — small, exact, every split user.
   Measure on the real link, never loopback (gotcha #739).
3. **Our processor half** (§3.3) — measure the owner's decode threads first.
4. **MoE experts on the processor** (§3.2).
5. **Prefix affinity across peers** (§3.1), then CPU users' prompt hand-off pricing.
6. ✅ **Split speculation Phase 1** (accept at the tail — shipped; the stream of checks followed in v0.3.216),
   then the shadow + coupling measurement decides whether Phase 4 is built.
7. **#133 + chunked prompt pass**, then the persistent stream by default.

## Sources

Petals 2312.08361 · Pope et al. 2211.05102 · StagFormer 2501.15665 · Layer
Parallelism 2502.02790 · Lookahead 2402.02057 · CLLM 2403.00835 · Lamina
2405.01814 · MegaScale-Infer 2504.02263 · BloomBee 2604.21072 · Prompt Cache
2311.04934 · Preble 2407.00023 · SkyWalker 2505.24095 · CacheGen 2310.07240 ·
LLMCompiler 2312.04511 · Skeleton-of-Thought 2307.15337 · SpecExec 2406.02532 ·
Sequoia 2402.12374 · ML-SpecQD 2503.13565 · QuantSpec 2502.10424 · SPEQ
2510.18525 · Daliri et al. 2408.07978 · multistream-select 0.13.0 `Version`
docs · llama.cpp `--n-cpu-moe` reports (discussions #15396, dev.to 8 GB
Qwen3-Coder-30B) · Leviathan 2211.17192 · Chen 2302.01318.
