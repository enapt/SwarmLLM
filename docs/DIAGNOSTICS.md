# SwarmLLM Diagnostic Instrumentation Guide

> **For contributors and developers.** This guide covers the internal diagnostic logging system used for debugging distributed inference, networking, and pipeline issues.

> **Accuracy is enforced.** Every `DIAG:` marker listed here is checked against
> the source by `every_documented_diag_line_exists_in_the_source` in
> `tests/repo_consistency.rs` — a guide whose greps come back empty is worse than
> no guide, because a failed grep looks exactly like the thing not happening.
> 28 markers had been renamed or deleted out of the code before that check
> existed (2026-08-09); rename or remove the entry when you change a log line.
>
> **Per-message network events are `debug`, not `info`** — `Received request`,
> `DIAG: received response`, `DIAG: ResponseSent event` and `DIAG: rr_ping sent`
> fire once per request_response message, which means once per streamed token
> under load. Run with `-v` to see them. Failures stay at info/warn.

All diagnostic log lines are prefixed with `DIAG:` for easy filtering.

## Start here: one line per request

Before tracing anything through the 26 lifecycle points below, read the
completion summary. Every finished inference emits **one** line carrying the
whole route and where the time went:

```bash
grep "DIAG: request complete" node.log
```

```
DIAG: request complete request_id=1ddd2912-… route=distributed segments=2
  model=llama-3.2-1b-instruct-q8-0 nodes=0718d8b9,96842635 regions=TH,TH
  queue_ms=3 sched_ms=1 ttft_ms=180 decode_ms=1420 total_ms=1604
  prompt_tokens=22 tokens=48 tok_per_sec=33.8 tpot_ms=30.2
  seg0_ms=520 seg1_ms=900 activation_bytes=39188 outcome=ok
```

This answers most questions on its own:

| Symptom in the line | Where to look next |
|---|---|
| `queue_ms` large | node is saturated — tier caps in `router/mod.rs`, or `max_concurrent_requests` |
| `sched_ms` large | scheduler struggling to find holders — check `-- peer serving performance --` |
| `assemblies=2` present | the request FAILED once and retried. Whatever else the line says, start here: the first attempt's cause is in the log just above |
| `ttft_ms` large, `decode_ms` small | prefill or a cold model load, not the network |
| `decode_ms` large, `tpot_ms` high | per-token cost — find the slow hop via `segN_ms` |
| one `segN_ms` dominates | that peer is the bottleneck; cross-check its row in the peer table |
| `route=relayed` | no direct path to a holder; ~1 extra RTT each way, see NAT section |
| `outcome=error error_type=…` | the variant name points at the subsystem |

`sched_ms` is time spent *assembling*, summed across attempts — not
time-since-dequeue, which would charge a failed attempt's whole execution to
"scheduling". `assemblies` appears only when it is >1.

Absent fields mean "not measured", never zero. `ttft_ms` and `decode_ms` are
omitted on a path that never emitted an incremental token, because there is no
honest way to split decode out of the total there.

## "The reply came back empty or one token"

`finish_reason: "stop"` cannot tell you why, and the OpenAI schema has no field
for it — a reply cut off by the caller's own stop sequence and a model that
ended its turn immediately look identical to a client. Since v0.3.116 the node
says which happened, so this is one grep rather than a testing session
(gotcha #372, from an external report that took several rounds to narrow):

```bash
# A stop sequence in the REQUEST matched almost at once — names the culprit.
grep "stop sequence in the request matched" node.log

# The model emitted end-of-turn straight away, nothing cut it off.
# Points at the prompt: check the chat template for this model.
grep "ended its turn immediately" node.log

# Finalisation removed everything the model generated (leaked markers, a stop
# matching at position 0). Older, and covers the fully-empty case only.
grep "empty after finalisation" node.log
```

Neither of the first two fires when the caller legitimately asked for a short
reply — `max_tokens: 1` yielding one token is not a fault and is deliberately
silent, so the warning stays meaningful.

**"ended its turn immediately" points at the prompt, and the prompt is not
always the text.** That warning's own advice — check the chat template, check
that the rendered prompt ends where the model expects to answer — was followed
for seven releases against an external report and was a dead end every time:
the template was fine and the prompt was well-formed (gotcha #400). What was
wrong was WHERE the model was told to continue from.

Two numbers must be equal, and they are printed in adjacent lines at `-v`:

```bash
grep -E "starting forward_through_segments|SplitModel forward pass complete" node.log
```

```
seq_num=0  index_pos=0     seq_len=5529   kv_offset=0        <- prefill wrote 5529 positions
seq_num=1  index_pos=6053  seq_len=1      kv_offset=5529     <- decode asked for 6053
```

`index_pos` is the rotary position the next token is computed at; `kv_offset` is
where the model's cache actually ends. **A gap means the model is being asked to
continue from somewhere its own memory of the prompt does not reach**, and the
logits are noise — which surfaces either as end-of-turn at once, or as one token
repeated to `max_tokens`. Both shapes, one cause.

The check that needs no log at all, and works against a node you cannot see:

```bash
# Same body twice: once with the model unloaded, once warm.
curl -X POST .../api/admin/models/$MODEL/unload -H "Authorization: Bearer $K"
COLD=$(curl -s ... -d @body.json); WARM=$(curl -s ... -d @body.json)
# usage.prompt_tokens MUST be identical. It is a property of the prompt.
```

A disagreement is the fault itself rather than a symptom of it, and it is what
`examples/release_shapes.sh` now asserts. Reply length cannot substitute: the
repetition shape passes any "more than N tokens" check comfortably.

**Three things that make this class of bug look intermittent, all of which cost
time on that report.** It fires only on the PIPELINE path, taken while the model
is not loaded — so the same request fails cold and succeeds warm, and every
retry is warm. The error scales with prompt length, so a minimal reproduction is
below the threshold at which anything goes wrong. And `chars / 4` lands in the
right ballpark, so an estimate reads as a plausible token count to anyone
eyeballing it. Compare it against something, never against your expectations.


## "Why is my model on the processor?"

The whole placement decision is greppable, in the order it is taken. On one
request against a node whose card is occupied:

```
DIAG: admitting model to GPU  model=X estimated_mb= committed_mb= budget_mb= headroom_mb=
DIAG: GPU admission refused — not enough budget at this moment          (DEBUG)
Freeing graphics memory from an idle model ... reclaimed_mb= for_model=  (reclaim fired)
No idle model could be reclaimed to fit this one                        (DEBUG; nothing eligible)
Model will run on the CPU  model=X reason=  configured_gpu_layers= estimated_vram_mb=
model-worker: Model loaded ... device=Cuda(...) | device=Cpu  vram_after_load_mb=
```

`reason=` is one of `not_enough_vram`, `configured_cpu_only`,
`gpu_too_old_for_this_build` — three completely different situations that all
produce `--gpu-layers 0` and were indistinguishable before v0.3.x.
**`device=` in the worker's own line is the answer**, not the daemon's intent.

Coming back the other way (v0.3.130+):

```
Graphics memory has freed up — retiring this model's processor worker ...
Model worker stopped and its memory budget released  device="cpu"
Freeing graphics memory from an idle model ... for_model=X
DIAG: admitting model to GPU  model=X
model-worker: Model loaded ... device=Cuda(...)
```

**Things that are NOT the explanation, each having cost a session:**

- **An eviction is not necessarily the reclaim.** `try_idle_vram_unload` (timer,
  keeps a model the swarm wants for up to an hour) and
  `free_vram_for_admission` (on demand, 5 s idle floor, plans first) both log
  about freeing memory. An external tester read the first as the second and
  concluded the floor was broken; it was a third mechanism entirely (#402).
- **A model on the processor does not occupy the card.** Anything summing
  `split_models[*].estimated_vram_mb` without filtering by device is answering
  a different question — that is what `MemoryScope` exists for.
- **`cpu_reason` is a prediction about the next spawn, not a fact about what is
  running.** For a resident worker, ask `placed_on_cpu_because` /
  `cpu_placement_reason` (#401).

## "Why is this node talking to a stranger?"

```
Ignoring a peer that does not speak SwarmLLM ... protocol_version= agent=   (INFO, once/peer)
Not dialling a peer that does not speak SwarmLLM  peer_id= site=           (DEBUG)
```

The second names the dial site (`pex`, `mdns`, `relay_providers`,
`connection_race`, `invite_code`). If foreign peers keep reconnecting and that
line never appears, **the dials are not coming from our code** — that null
result is what identified #404. Count connections per peer rather than trusting
the peer list, which has been clean since v0.3.125 while the reconnections
continued:

```bash
for p in $(grep -a "does not speak SwarmLLM" node.log | grep -ao "peer_id=12D3KooW[A-Za-z0-9]*" | cut -d= -f2 | sort -u); do
  echo "$p: $(grep -ac "connection established peer_id=$p" node.log)"
done   # 1 each is correct — a node must be spoken to before it can be identified
```

**Was it served locally or by a peer?** That changes which code path to suspect
entirely, and it is a response header rather than a log line:

```bash
curl -sD- -o /dev/null http://localhost:8800/v1/chat/completions ... | grep x-swarm
# x-swarm-route: local | x-swarm-segments: 1 | x-swarm-peers: 0 | x-swarm-nodes: …
```

Ask for that FIRST on any report from a multi-node setup — model, request body
and version were all obtained for the report above and none of them
discriminated, while this header would have (gotcha #374).

## Is speculative decoding actually helping?

A local model drafts ahead of itself when the reply repeats something already in
the context. One line per request, at debug:

```bash
grep "local n-gram speculation complete" node.log
# rounds=6 drafted=53 accepted=47 paused_rounds=0 tokens_per_round=8.83
```

`tokens_per_round` is the number that matters: ~8.8 means it is working, ~1.0
means this workload has nothing to copy. `paused_rounds` counts rounds where the
backoff suppressed drafting — high is CORRECT on prose, not a fault. A request
that is not alone on the worker joins the batch instead and logs nothing.

## "Something is slow" — split user time from kernel time FIRST

Before theorising about which code is slow, spend one command finding out
whether the CPU is running your code at all. Fields 14 and 15 of
`/proc/<pid>/stat` are the process's user and system ticks (100 per second):

```bash
read u1 s1 <<< "$(awk '{print $14, $15}' /proc/<pid>/stat)"
curl -s -m 60 -o /dev/null -w "wall=%{time_total}s\n" -H "Authorization: Bearer $KEY" <url>
read u2 s2 <<< "$(awk '{print $14, $15}' /proc/<pid>/stat)"
echo "user=$((u2-u1)) system=$((s2-s1))"
```

Read it like this:

- **System ticks dominate** → syscalls. Something is doing an enormous number of
  small operations against the kernel. This is what found gotcha #410:
  `/api/admin/models` spent 962 of 1192 ticks in the kernel because GGUF headers
  were being parsed off an unbuffered `File`, one syscall per tiny read.
- **User ticks dominate** → your code. Parsing, allocation, arithmetic.
  Optimisation and algorithms are on the table.
- **Neither, but the wall clock is long** → waiting. A lock, a peer, a timeout.

**Two readings that come free with it.** A *stable* wall time across runs
(11.30 / 11.11 / 11.14 s) is a fixed amount of work, not contention — contention
is noisy, so go and find the count. And a request whose CPU time is close to its
wall time is running on ONE thread the whole way, which for an async handler
means a worker thread is blocked for that long.

The trap it saves you from is real: "parses a lot of metadata" and "makes 820k
read calls" both fit the symptom, look identical in the source, and no amount of
reading distinguishes them. It is also why the optimised release binary was no
faster than a debug build on that path — optimisation cannot remove a syscall.

## No log file? Use the endpoint

`GET /api/admin/diagnostics` renders plain text for a shell, and includes the
last 50 completed requests, the failure ring, per-peer serving performance
(round-trip time, ms/layer, EWMA latency, sample count, region) and what this
node has served for others. One command instead of a log excerpt:

**`-- this machine --`** — CPU, GPU, measured memory bandwidth, and the
`advertised speed` derived from it. That last figure is what every other node's
scheduler ranks this one on, so it is the first thing to read when someone asks
why work is never routed to their machine, or why their fast box loses to a
slower one. A GPU node takes its bandwidth from the card's spec table; a
processor-only node reports what `inference::mem_bandwidth` actually measured.
"Could not be measured" is a distinct answer from a low number and is printed
as one.

**`in_flight: N traces, M pipelines`** — both should be `0` on an idle node.
Non-zero with no traffic means bookkeeping has been left behind, and the trace
count is the one that bites: it is the oracle behind `model_is_in_use`, so a
stale entry makes deleting that model fail with "in use" **permanently**, on a
node serving nobody. There is no sweep behind the RAII cleanup, so this number
is the only way to see it.

```bash
swarmllm diagnostics          # safe to paste in public
swarmllm diagnostics --full   # keeps network addresses, for your own machine
```

Or straight from the endpoint, which the command is a wrapper over:

```bash
curl -s -H "Authorization: Bearer $(cat ~/.local/share/swarmllm/api_key)" \
  'localhost:8800/api/admin/diagnostics?full=1'
```

**Without `?full=1` every network address is replaced** by a placeholder naming
its kind — `<public-ip-a3f1>`, `<private-ip-…>`, `<host-…>` — while transport,
port, peer id and `/p2p-circuit` structure survive, so the report still answers
"is this node public?" and "is that hop relayed?". Two occurrences of one host
share a tag, so "ten cache entries, all the same machine" is still visible; the
tag is salted per report and means nothing across two of them. The project's own
bootstrap anchor is exempt, since it ships in every binary. The default is
redacted because the dashboard's **Copy diagnostics** button is this endpoint's
main consumer and its output gets pasted into public channels.

Per-request routing is also on **every response**, so a failing client can be
diagnosed without server access at all:

```bash
curl -i -X POST localhost:8800/v1/chat/completions -H '…' -d '…' | grep -i '^x-swarm-\|^server-timing'
```

```
x-swarm-route: distributed
x-swarm-segments: 2
x-swarm-nodes: 0718d8b9,96842635
server-timing: queue;dur=3, sched;dur=1, ttft;dur=180, decode;dur=1420
```

On a streaming response the `Server-Timing` header carries only what is known
before the body flushes (queue, schedule); the token-level figures arrive in the
final SSE usage event.

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
cargo run -- run -vv 2>&1 | grep "DIAG:.*codec"           # Wire-protocol codec frames
```

## End-to-End Request Trace

Every inference request gets a `request_id` (UUID) that appears in logs across all subsystems. To trace a single request:

```bash
cargo run -- run -vv 2>&1 | grep "request_id=<UUID>"
```

### Request Lifecycle (log points)

1. **API entry** → `Queued inference request` (router/mod.rs)
2. **Dispatch** → `DIAG: dispatch_single starting inference` (router/mod.rs)
3. **Pipeline assembly** → `DIAG: pipeline assembled` with `segments`, `standbys`, `schedule_ms` (router/distributed_exec.rs)
4. **Forward start** → `DIAG: starting forward_through_segments` with `seq_num`, `index_pos`, `activation_bytes` (pipeline/distributed.rs)
5. **Tensor forward send** → `DIAG: sent tensor forward via send_request` with `is_connected`, `total_connections`, `pending_tensor_count`, `outbound_id` (manager/tensors.rs)
6. **Codec write** → `DIAG: codec write_request start/done` with `frame_len` (protocol.rs)
7. **Encryption (if enabled, R139)** → encrypt offloaded from event loop via `tokio::spawn`. Failure: `DIAG: tensor encrypt+encode failed — dropping forward` (manager/tensors.rs). On success the spawn task posts `NetworkCommand::SendEncodedTensor` back through `internal_cmd_tx`; the critical task then performs only the `send_request` step. Decode/decrypt offloaded symmetrically in the inbound path; failures log `DIAG: decrypt FAILED — possible AAD mismatch, key mismatch, or corruption`
9. **Inbound dispatch** → `DIAG: inbound TensorPayload request` → `DIAG: acknowledged tensor forward on receipt` (manager/requests.rs) — the request is ACKed immediately; the result travels back as its own request (`features::FORWARD_ACK`, 2026-08-21). A coordinator that sees no ACK within the RTT-scaled deadline logs `DIAG: tensor forward not acknowledged within the ACK deadline` (manager/mod.rs, the stale sweep) and the pipeline fails over; an un-ACKed forward no longer waits out the segment deadline.
10. **Dispatcher** → `DIAG: dispatcher received LayerForward, spawning handler` with `seq`, `layer_range`, `activation_bytes` (daemon/dispatch/mod.rs)
11. **Local execution** → `DIAG: processing LayerForward locally` with `elapsed_ms` (daemon/dispatch/layer_forward.rs)
12. **Split model forward** → `DIAG: SplitModel forward pass complete` with `forward_ms`, `seq_len`, `num_layers` (split/executor.rs)
13. **Result send** → `DIAG: LayerForward processed via worker subprocess` with `tokens`, `activations_bytes`, `elapsed_ms`, `layer_start`, `layer_end` (daemon/dispatch/layer_forward.rs)
14. **Response write** → `DIAG: codec write_response start/done` with `frame_len` (protocol.rs)
15. **ResponseSent event** → `DIAG: ResponseSent event — response written to wire` (manager/events.rs)
16. **Response read** → `DIAG: codec read_response done` with `tag`, `len` (protocol.rs)
17. **Response received** → `DIAG: received response` with `kind`, `was_tensor_forward`, `pending_tensor_out` (manager/events.rs)
18. **Response dispatch** → `DIAG: received TensorPayload response` (manager/requests.rs)
19. **Result delivery** → `DIAG: dispatcher received LayerResult` → `DIAG: LayerResult delivered to pipeline` (daemon/dispatch/mod.rs)
20. **Forward complete** → `DIAG: forward_through_segments returned OK` with `fwd_ms`, `tokens`, `activations_bytes` (pipeline/distributed.rs)
21. **Local segment** → `DIAG: local segment complete` with `segment_ms`, `activation_bytes` (pipeline/distributed.rs)
22. **Remote segment** → `DIAG: remote segment complete` with `segment_ms`, `activation_bytes` (pipeline/distributed.rs)
23. **Segment result** → `DIAG: segment result received` with `elapsed_ms` (pipeline/local.rs)
24. **Pipeline complete** → `DIAG: forward_through_segments completed` with `pipeline_ms` (pipeline/distributed.rs)
25. **Execute complete** → `DIAG: execute_request completed successfully` with `schedule_ms`, `execute_ms`, `total_ms` (router/distributed_exec.rs)
26. **Completion** → `DIAG: request complete` — the single summary line described at the top of this guide, carrying route, nodes, regions, per-phase timings, per-segment timings, tok/s and outcome (`daemon/state/relay.rs::publish_request_trace`, called from router/mod.rs)

All 26 points are built from one `RequestTrace` (`inference/trace.rs`), which is
also what feeds the response headers, the diagnostics ring and the Prometheus
histograms. Adding a field means adding it there once, not at each surface.

### Network Event Diagnostics

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: processing swarm event` — event type name for every swarm event | manager/events.rs |
| DEBUG | `DIAG: handling outbound command` — command type for every outbound command | manager/commands.rs |
| INFO  | `DIAG: OutboundFailure` — `is_connected`, `pending_tensor_out`, `pending_channels` | manager/events.rs |
| WARN  | `DIAG: InboundFailure` — `pending_channels` | manager/events.rs |
| DEBUG | `DIAG: remote-generate stream complete` — `streamed_count`, the number the done token carries so the coordinator can tell a finished stream from one whose end overtook its middle | daemon/dispatch/remote_generate.rs |
| DEBUG | `DIAG: ResponseSent event` — confirms response written to wire. Per-message, so `-v`: at info these were three quarters of an idle node's log, and one line per streamed token under load | manager/events.rs |

### Failure Paths

- **Timeout** → `DIAG: segment TIMED OUT — no result received` (pipeline/local.rs)
- **Outbound failure** → `DIAG: OutboundFailure` → `Tensor forward OutboundFailure — notifying pipeline` (manager/events.rs)
- **Inbound failure** → `DIAG: InboundFailure — response send may have failed` (manager/events.rs)
- **Decryption fail** → `DIAG: decrypt FAILED — possible AAD mismatch` (manager/tensors.rs)
- **No standby** → `DIAG: NO standby available for failed segment` (pipeline/distributed.rs)
- **Client disconnect** → `DIAG: result_tx receiver dropped` (router/mod.rs)
- **Channel drop** → `DIAG: LayerResult delivered but pipeline receiver DROPPED` (daemon/dispatch/mod.rs)
- **No pending channel** → `DIAG: No pending channel for LayerResult — timed out, duplicate, or hedge loser` (daemon/dispatch/mod.rs)
- **Streaming done event** → `DIAG: streaming done_event send failed` (router/mod.rs or router/distributed_exec.rs)

## Comparing a change against the released binary (null control)

The cheapest way to prove a change caused a difference — rather than something
ambient (peer set, model state, load) — is to run the **same data dir and the
same request** under both binaries.

```bash
T=/tmp/nullctl; rm -rf "$T"; mkdir -p "$T/models"
cp -r ~/.local/share/swarmllm/models/<model> "$T/models/"
printf '[auto_manage]\nenabled = false\nprune_enabled = false\n' > "$T/config.toml"

# candidate
SWARMLLM_NODE_DATA_DIR="$T" ./target/release/swarmllm run -p 8872 &
# ...run the probe, record the output...

# stop ONLY this node: match the data dir, never a bare pkill (gotcha #283)
for p in $(pgrep -x swarmllm); do
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "SWARMLLM_NODE_DATA_DIR=$T" && kill $p
done

# control — the RELEASED binary, same dir, same probe
SWARMLLM_NODE_DATA_DIR="$T" ~/.local/bin/swarmllm run -p 8872 &
```

The match also catches that node's `model-worker` child, so kill the daemon, not
just the first hit. Peer-dependent probes need ~45 s after a restart for the peer
set to settle — a different failure right after start is usually the swarm, not
the change.

## Measuring cancellation (what NOT to use)

`active_requests` from `/api/admin/stats` reads **0 even mid-stream** on the
local split fast path, because that path bypasses the router. It cannot measure
whether a client walking away stopped the work. Use the worker's CPU instead:

```bash
# worker pid for a given data dir
pgrep -x swarmllm | while read p; do
  tr '\0' ' ' < /proc/$p/cmdline | grep -q "model-worker.*$DATA_DIR" && echo $p
done
# utime+stime from /proc/<pid>/stat fields 14,15, sampled over N seconds
```

Healthy cancellation on this box: ~330% of one core during generation → ~30% 4 s
after the client closes (the in-flight forward finishing) → 0% by 8 s.

## SSE Streaming Diagnostics

All three streaming paths are instrumented with timing and error reporting:

### Distributed Pipeline Streaming (split_non_stream_response)

| Level | What | Where |
|-------|------|-------|
| WARN  | `DIAG: SSE role delta send failed` — client disconnected before stream started | api/openai/streaming.rs |
| WARN  | `DIAG: SSE final text delta send failed` — client disconnected on last token | api/openai/streaming.rs |
| WARN  | `DIAG: SSE finish delta send failed` — client disconnected at finish | api/openai/streaming.rs |
| DEBUG | `DIAG: SSE stream no finish event from pipeline` — falling back to result_rx | api/openai/streaming.rs |
| WARN  | `DIAG: SSE result_rx channel dropped` — pipeline task died | api/openai/streaming.rs |
| INFO  | `DIAG: SSE distributed stream completed` — `elapsed_ms`, `token_count` | api/openai/streaming.rs |

### Split Model Streaming (split_stream_response)

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: split stream model not found` — model evicted during request | api/openai/streaming.rs |
| DEBUG | `DIAG: split stream decode loop complete (subprocess)` — `decode_ms`, `tok_per_sec` | api/openai/streaming.rs |
| WARN  | `DIAG: split stream client disconnected (connection closed) — cancelling decode` — `token_count`, `elapsed_ms` | api/openai/streaming.rs |
| INFO  | `DIAG: split stream completed` — `elapsed_ms`, `token_count` | api/openai/streaming.rs |

### Local Executor Streaming (stream_response)

| Level | What | Where |
|-------|------|-------|
| WARN  | `DIAG: local stream role delta send failed` — client disconnected early | api/openai/streaming.rs |
| WARN  | `DIAG: local stream token send failed` — channel full or client disconnected | api/openai/streaming.rs |
| ERROR | `DIAG: local stream generate_stream error` — executor error | api/openai/streaming.rs |
| INFO  | `DIAG: local stream completed` — `elapsed_ms`, `token_count` | api/openai/streaming.rs |

## Encryption Diagnostics

The encrypted tensor path logs at multiple levels:

| Level | What | Where |
|-------|------|-------|
| DEBUG | `DIAG: decrypting tensor` — AAD length, sealed length, session existence | manager/tensors.rs |
| TRACE | `DIAG: seal() success` — nonce counter, ciphertext length | session.rs |
| TRACE | `DIAG: open() decryption success` — nonce, plaintext length | session.rs |
| ERROR | `DIAG: seal() encryption failed` — full context on encryption failure | manager/tensors.rs |
| ERROR | `DIAG: decrypt FAILED` — AAD mismatch, key mismatch, or corruption | manager/tensors.rs, session.rs |
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
| ERROR | `DIAG: {label} tensor decompression failed` — zstd decompress error | protocol.rs |
| ERROR | `DIAG: {label} tensor decompression failed` — zstd decompress error | protocol.rs |
| DEBUG | `DIAG: {label} tensor decompressed` — `compressed_len`, `decompressed_len`, `ratio` | protocol.rs |
| DEBUG | `DIAG: {label} tensor decompressed` — `compressed_len`, `decompressed_len`, `ratio` | protocol.rs |

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

### Per-Request KV-Cache Store (split/kv_cache.rs)

| Level | What | Where |
|-------|------|-------|
| INFO  | `DIAG: KV-cache store cleanup — expired entries removed` — `removed`, `remaining` | split/kv_cache.rs |

## Split Model Forward Pass Diagnostics

| Level | What | Where |
|-------|------|-------|
| TRACE | `DIAG: layer forward complete` — `layer`, `layer_ms` (per-layer timing) | split/executor.rs |
| DEBUG | `DIAG: SplitModel forward pass complete` — `forward_ms`, `seq_len`, `num_layers`, `is_first`, `is_last`, `kv_offset` | split/executor.rs |

For per-token decode analysis, combine the forward pass timing with the decode loop timing from `DIAG: split stream decode loop complete` which reports `tok_per_sec`. Use `-vvv` (trace) to see per-layer timing.

## Performance Diagnostics

### Identifying Slow Requests

The `elapsed_ms` field appears at multiple points:

1. `DIAG: SplitModel forward pass complete` — time for a single forward pass (compute only)
2. `DIAG: local segment complete` — time for a local pipeline segment
3. `DIAG: remote segment complete` — time for a remote pipeline segment (network + compute)
4. `DIAG: segment result received` — time for a single segment (network + compute)
5. `DIAG: forward_through_segments completed` — total pipeline forwarding time
6. `DIAG: execute_request completed successfully` — `schedule_ms` (pipeline assembly) + `execute_ms` (pipeline execution)
7. `DIAG: split stream decode loop complete (subprocess)` — decode time with `tok_per_sec`
8. `DIAG: request complete` — total end-to-end time

If `schedule_ms` is high, the bottleneck is pipeline assembly. If `execute_ms` is high but individual `segment_ms` values are low, the bottleneck is inter-segment overhead. If a single segment is slow, check that node's compute or network latency.

### 27-Second Response Times

Common causes:
- **Timeout-then-failover**: A segment times out at 30s, then failover succeeds quickly → ~30s total. Check for `DIAG: segment TIMED OUT` followed by `DIAG: failing over to standby`.
- **Connection not established**: Tensor sent to a peer that's not connected. Check `is_connected=false` in `Sent tensor forward` logs.
- **Encryption failure + fallback**: Encrypted send fails, falls back to plaintext, which also fails. Check for `DIAG: seal() encryption failed` logs.
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

WSL2's Hyper-V Networking Stack (HNS) causes multi-address connection races when autonat/mDNS discover the WSL2 NAT adapter (10.255.255.254). With `max_established_per_peer=1`, both nodes simultaneously establish connections via multiple interfaces, sending mutual yamux GoAway frames that kill ALL connections. Two mitigations are available via config:

### Disable autonat/dcutr

AutoNAT and DCUtR trigger mDNS multi-address discovery on WSL2 (loopback + LAN + NAT adapter), causing connection races. Disable for WSL2 testing. `NetworkConfig::default()` already sets these to `false` (the serde default of `true` only applies when loading from a config file).

```toml
# config/default.toml or ~/.local/share/swarmllm/config.toml
[network]
enable_autonat = false
enable_dcutr = false
```

Both protocols use `Toggle<T>` wrappers — when disabled, no events are emitted and no network traffic is generated. NAT detection and hole-punching are not needed for loopback/LAN testing.

### Yamux configuration

Yamux uses 0.13 defaults with auto-tuned windows (1 GiB max connection window). Do NOT call the deprecated `set_receive_window_size` or `set_max_buffer_size` methods — they silently downgrade to yamux 0.12 which has severe substream opening delays (~30s between successful outbound requests).

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

### Sampling (sampling.rs)

| Level | What | Fields |
|-------|------|--------|
| TRACE | `DIAG: sample_token complete` | `token`, `vocab_size`, `mode` (`greedy`/`stochastic`), `temperature`, `top_k`, `top_p` |
| WARN  | `DIAG: sampling fallback` | `vocab_size`, `sum` (cumulative probability rounding) |

### Speculative Decoding (speculative.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: speculative batch` | `drafted`, `accepted`, `acceptance_rate` |

### Vision (vision.rs + pipeline/mod.rs + daemon/dispatch/mod.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: encode_images` | `image_count`, `patch_count`, `elapsed_ms` |
| DEBUG | `DIAG: merge_vision_text_embeddings` | `text_seq`, `num_vision`, `hidden`, `positions` |
| INFO  | `DIAG: precompute_vision_embeddings local` | `image_count`, `compressed_bytes` |
| INFO  | `DIAG: precompute_vision_embeddings remote` | `remote_node` |

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
| INFO  | `DIAG: register_manifest` — **only when new or changed** (`manifest_hash` differs) | `model_id`, `shard_count`, `publisher` |
| DEBUG | `DIAG: register_manifest (unchanged)` — a re-gossip of a manifest we already hold | `model_id` |
| INFO  | `DIAG: load_from_db complete` | `manifests_loaded_count` |

### HuggingFace (huggingface.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: search_gguf_models` | `query`, `repos_found` |

### Manifest (manifest.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: load_from_dir` | `dir`, `shard_count` |

### LoRA (lora.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: lora adapter loaded` | `adapter_path`, `rank`, `alpha`, `target_modules` |

### Acquisition (acquisition.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: handle_acquire` | `model`, `needed_shards` |

### Auto-Manage (auto_manage/)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: evaluate_and_prune` | `resource_pressure`, `pressure_urgent` |
| DEBUG | `DIAG: register_local_shard` | `model`, `shard_index` |
| INFO  | `DIAG: check_and_load_model` | `model_id`, `available_shards`, `missing_shards`, `ready` |
| DEBUG | `Skipping model — insufficient trust for auto-manage` | `model`, `trust` |
| INFO  | `Model promoted to NetworkPopular` | `model`, `holders` |
| INFO  | `HfWatcher: promoted to DemandVerified` | `model`, `repo`, `downloads` (R141 — fires at 10k for trusted publishers, 100k for unknown) |
| DEBUG | `HfWatcher: re-promotion blocked by failed-promotion cooldown` | `model`, `repo` |
| WARN  | `HfSourceGossip dropped — hf_sources at capacity` | `model`, `cap` (R141 — fires alongside `activity.hf_sources_cap_reached`) |
| WARN  | `Auto-manage: released stalled P2P download permit; HF fallback will fire next cycle` | `model`, `shard`, `stall_secs` (R141 — `P2P_PERMIT_STALL_SECS = 180`) |
| INFO  | `On-demand loading: model has shards on disk but not loaded` | `request_id`, `model` |

## API Subsystem Diagnostics

### Server (server.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: server startup` | `addr` |

### Admin HF (admin_hf/shards.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: hf_download_shards` | `model_id`, `variant` |

### Providers (providers.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: provider resolution` | `model_id`, `resolved_provider` |

### WebSocket (websocket.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: client connected` | `subsystem = "websocket"` |
| DEBUG | `DIAG: client disconnected` | `subsystem = "websocket"` |
| DEBUG | `DIAG: push_task exited first` / `DIAG: receiver loop exited first` | `subsystem = "websocket"` |

### Middleware (middleware.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: auth failure` | `path`, `auth_present` |

### Anthropic (anthropic/mod.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: anthropic messages request` | `request_id`, `model`, `messages`, `stream`, `max_tokens` |
| DEBUG | `DIAG: anthropic connectivity probe` | `request_id` |
| DEBUG | `DIAG: anthropic inference path resolution` | `request_id`, `has_local_split_model`, `network_available` |
| INFO  | `DIAG: anthropic proxying to cloud API` | `model` |

### Identity (identity.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: set_nickname persisted` | `nickname` |
| DEBUG | `DIAG: leaderboard query` | `peer_count`, `limit` |

### Metrics (metrics.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: metrics scrape` | — |
| DEBUG | `DIAG: health_ready probe` | `ready` |

### Pool (pool.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: pool_create request` | `name` |
| INFO  | `DIAG: pool_invite request` | — |
| INFO  | `DIAG: pool_rates_set request` | `pool_id` |

## Config Diagnostics

### Config (config/mod.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: config load_or_create starting` | `config_path`, `cli_port`, `cli_data_dir` |
| DEBUG | `DIAG: config load_or_create complete` | `port`, `data_dir` |

## Update Diagnostics

### Update Checker (update.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: check_for_update starting` | — |
| DEBUG | `DIAG: check_for_update version compare` | `current`, `latest` |
| INFO  | `DIAG: apply_update starting` | `path` |

## Daemon Startup Diagnostics

### Main (main.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: daemon starting` | `version` |

## Credit Subsystem Diagnostics

### Ledger (ledger.rs)

| Level | What | Fields |
|-------|------|--------|

### Escrow (escrow.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: escrow created` / `DIAG: escrow release` | `tx_id`, `amount`, `state` |

### Trust (trust.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: trust score update` | `node`, `score_delta`, `new_score` |

## Crypto Subsystem Diagnostics

### Key Rotation (key_rotation.rs)

| Level | What | Fields |
|-------|------|--------|
| DEBUG | `DIAG: key rotation tick (eviction)` | `active_sessions`, `stale_evicted` |
| DEBUG | `DIAG: key rotation tick (re-keying)` | `active_sessions`, `rekey_initiated` |

### Key Exchange (manager/identify.rs)

| Level | What | Fields |
|-------|------|--------|

## Infrastructure Diagnostics

### Database (db.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: db_open` | `path` |

### Identity (keypair.rs)

| Level | What | Fields |
|-------|------|--------|
| INFO  | `DIAG: identity key loaded from disk` | `path` |

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
| `src/network/manager/` | Encrypted tensor send/receive, connection lifecycle, gossip audit, outbound tracking, key exchange |
| `src/network/behaviour.rs` | Connection limits, autonat/dcutr toggle state |
| `src/network/discovery.rs` | Bootstrap failures promoted to WARN with peer counts |
| `src/network/protocol/mod.rs` | Tensor decompression success/failure with sizes and compression ratios |
| `src/network/relay.rs` | Relay reservation logging |
| `src/network/peer_cache.rs` | Peer cache save count |
| `src/inference/pipeline/` | Segment timing (local + remote), pipeline total timing, failover details, wait_for_result context |
| `src/inference/router/` | Pipeline schedule vs execute timing breakdown (distributed_exec.rs), result channel delivery, streaming done event (mod.rs) |
| `src/inference/split/` | Per-forward-pass timing (executor.rs), KV-cache cleanup (kv_cache.rs) |
| `src/inference/kv_cache.rs` | KV-cache hit/miss with detailed miss reasons (expired, degraded, prefix mismatch, evicted) |
| `src/inference/scheduler/mod.rs` | Pipeline assembly timing, candidate counts, standby counts |
| `src/inference/executor.rs` | Model load timing with backend type, generate_stream params |
| `src/inference/speculative.rs` | Batch acceptance rate tracking |
| `src/inference/vision.rs` | Image encoding timing |
| `src/inference/chat_template/mod.rs` | Template matching and fallback detection |
| `src/model/shard.rs` | Shard verification failures, load_all_local summary |
| `src/model/registry.rs` | Manifest registration, DB load counts |
| `src/model/huggingface/search.rs` | Search result counts, HF_TOKEN auth |
| `src/model/manifest.rs` | Manifest load with shard count |
| `src/model/lora.rs` | Adapter load with rank, alpha, target modules |
| `src/model/acquisition.rs` | Acquisition requests, peer selection |
| `src/model/auto_manage/` | Prune evaluation (prune.rs), shard registration (download.rs), model readiness (scan.rs) |
| `src/api/server.rs` | Server startup with bind address |
| `src/api/openai/streaming.rs` | All 3 streaming paths with per-token timing, client disconnect detection, fallback path logging |
| `src/api/admin_hf/shards.rs` | HF shard download initiation |
| `src/api/providers.rs` | Provider resolution |
| `src/api/websocket.rs` | WebSocket connection lifecycle |
| `src/api/middleware.rs` | Auth failure with path context |
| `src/credit/ledger.rs` | Transaction recording with balance changes, DB restore failure |
| `src/credit/escrow.rs` | Escrow create/release with state |
| `src/credit/trust.rs` | Trust score updates |
| `src/storage/db.rs` | Database open with path |
| `src/identity/keypair.rs` | Identity key load |
| `src/daemon/dispatch/mod.rs` | LayerForward timing, LayerResult delivery, pending channel state |
| `src/health/monitor.rs` | Broadcast failures, stale peer counts, channel cleanup details |
| `src/api/anthropic/` | Messages API request entry, connectivity probe fast-path, inference path resolution, cloud proxy |
| `src/api/identity.rs` | Nickname set/gossip, leaderboard query with peer filtering |
| `src/api/metrics.rs` | Metrics scrape, health readiness probe |
| `src/api/pool.rs` | Pool create, invite, rate set operations |
| `src/config/mod.rs` | Config load source, data_dir resolution, validation complete |
| `src/update.rs` | Update check start, version compare, apply start |
| `src/main.rs` | Daemon startup |

## Coverage Statistics (2026-03-08)

**~250 DIAG lines across 61/79 source files (100% of actionable files).**

All 61 files containing runtime decision/timing/error logic are instrumented. The 18 uninstrumented files are:
- `mod.rs` re-exports (11): no logic, just `pub mod` declarations
- Type definitions (3): `types.rs`, `pool/types.rs`, `error.rs` — struct/enum definitions only
- Static assets (2): `ui/assets.rs`, `ui/mod.rs` — embedded file serving
- Pure functions (1): `network/transport.rs` (keypair conversion)
- `lib.rs` (1): module declarations only
- `inference/json_grammar.rs` (1): pure state machine with no I/O

### Coverage by Subsystem

| Subsystem | Files | DIAG Lines | Key Log Points |
|-----------|-------|------------|----------------|
| Network (manager, behaviour, protocol, discovery, relay, peer_cache) | 6 | ~50 | Connection lifecycle, codec read/write, encryption, swarm events |
| Inference (router, pipeline, scheduler, executor, split, sampling, speculative, vision, kv_cache, chat_template) | 10 | ~38 | Request dispatch, pipeline assembly, forward pass, token sampling |
| API (server, openai, admin, websocket, middleware, providers, anthropic, identity, internal, metrics, pool) | 12 | ~55 | Server startup, SSE streaming, auth, Anthropic proxy, pool ops, metrics scrape |
| Model (shard, manifest, huggingface, acquisition, auto_manage, registry, distribution, lora) | 8 | ~23 | Shard verification, HF search/download, model loading, pruning |
| Credit (ledger, transaction, priority, anti_gaming, trust, escrow) | 6 | ~15 | Transaction verification, tier calculation, trust updates, escrow |
| Crypto (session, key_rotation, gossip_seal, pipeline_seal) | 4 | ~10 | Key exchange, session management, encryption seal/open |
| Daemon + Main (daemon/, main.rs) | 5 | ~11 | Daemon startup, LayerForward processing, result delivery |
| Config (config/) | 1 | ~2 | Config load source, WSL2 detection, validation |
| Update (update.rs) | 1 | ~3 | Update check, version compare, apply |
| Pool (manager, crypto, forward) | 3 | ~5 | Pool commands, invitations, credit forwarding |
| Identity (keypair, keystore, nickname) | 3 | ~3 | Key generation, keystore save/load, nickname records |
| Health (monitor, rebalancer) | 2 | ~4 | Rebalance events, health monitoring |


## Stage profiler — where a forward pass actually spends its time

`SWARMLLM_PROFILE=1` makes every forward pass print a per-stage breakdown to
stderr and reset. Stages are wall-clock and non-overlapping, so they sum to
roughly the block time; the report also prints what they do NOT account for,
which is as informative as the stages.

```
SWARMLLM_PROFILE=1 swarmllm run -p 8899
...
PROF seq_len=128 index_pos=384 layers=28 — total 10045 ms
   4571.7 ms   45.5%  attention scores + softmax + AV
   2558.2 ms   25.5%  ffn up + gate        (quantized matmul)
   1330.7 ms   13.2%  ffn down             (quantized matmul)
    848.0 ms    8.4%  qkv projections      (quantized matmul)
    ...
     30.4 ms    0.3%  unattributed (allocation, copies, dispatch)
```

Accumulation is unconditional — `Instant::now` is ~25 ns against stages that run
for milliseconds — so only the dump is gated. Implementation in
`src/inference/prof.rs`; add a stage by extending the `stages!` macro and wrapping
the call site in `timed!`.

**This is what found the CPU attention kernel** (2026-08-06): attention was 2.3%
of the arithmetic but 45% of prompt-processing time, i.e. running 37x slower per
MAC than the quantized matmul beside it. Reach for it before optimising anything
in the forward path — the previous round tuned a matmul that turned out to be a
quarter of the cost.

## Attention-kernel A/B — `SWARMLLM_FORCE_STANDARD_ATTN=1`

Forces every attention call in the process onto `standard_attention`, so the
whole daemon runs without the fused kernel and nothing else changes. Pair a run
with it against a run without it to price the fused path end to end:

```
SWARMLLM_FORCE_STANDARD_ATTN=1 swarmllm run -p 8899   # A: standard everywhere
swarmllm run -p 8899                                  # B: normal dispatch
```

It sets the *initial* value of the per-thread override that
`ForceStandardAttnGuard` manipulates, so the speculative-decoding paths that
deliberately nest a `false` guard still behave correctly — a debug switch that
changed the guard's semantics would be its own bug.

**Why this exists rather than two builds.** Two separately-built binaries differ
in link order, inlining and codegen, so a difference between them is not
attributable to the kernel (diagnosis rule 4 — prove the mechanism fired, not
just that the number moved). One binary, one branch, identical weights is the
only comparison that isolates it. This is how the CPU prefill/decode crossovers
in `run_attention` were measured, and how flash-attention-2 was priced on CUDA
when it was re-enabled.

For the kernel in isolation, without a daemon or a model, there is a microbench
at the bottom of `src/inference/layers/mod.rs`:

```
CUDA_COMPUTE_CAP=86 cargo test --release \
  --no-default-features --features dev,claude-subscription,flash-attn \
  flash_vs_standard -- --ignored --nocapture
```

It sweeps prefill and decode shapes for an MHA and a GQA model, and asserts the
two kernels agree numerically before reporting any speed figure — flash runs in
F16 where standard runs in F32, and a fast wrong answer is not an optimisation.

## Network event loop stalls (2026-08-21)

Every arm of `NetworkManager::run`'s `select!` is timed; an iteration over
100 ms logs `DIAG: network event loop stalled` with `arm=` (the interval or
queue that was being serviced, or `swarm_event:<kind>`) and `took_ms`. Every
latency this node measures — the PEX ping that becomes `latency_ms`, per-hop
pipeline timings, ACK deadlines — is measured across this loop, so a stall
here is added to every number routing uses. Zero lines on a live node is the
normal reading; it is how the relay-carried-inbound-connection bug (#356) was
separated from a loop problem in four minutes. `grep "loop stalled" node.log |
grep -oE "arm=[^ ]+ took_ms=[0-9]+" | sort | uniq -c | sort -rn` names the
culprit when there is one.

## "no receipt acknowledgement within Ns" — late, or missing? (2026-08-25)

A peer that fails every distributed request with this looks dead. It may simply
be busy: the ACK is emitted by the network event loop, so it arrives late exactly
when that loop is loaded, and a ping RTT cannot see that.

**The sender logs ACK receipt at `debug`**, so an info-level log shows nothing
either way — absence there is not evidence. Run the node with `-v` and look:

```bash
grep 'DIAG: received response' node.log | grep 'kind="ack"'
```

ACKs present, including from the failing peer, means late-not-missing, and the
deadline is the thing to look at rather than the peer. Measured 2026-08-25: a
peer failed for about an hour, then served the same request in 6.1 s, with its
ACKs arriving the whole time. Since v0.3.124 the deadline is per-peer
(`AckRttEstimator`, RFC 6298) and backs off on a miss, so this should
self-correct — if it does not, that estimator is the place to look.

**A cheap way to reproduce without touching the live node**: start a throwaway
node (`SWARMLLM_NODE_DATA_DIR=$(mktemp -d)`, its own port, `[auto_manage]
enabled=false`) with `-v` and issue the same request to it. It joins the swarm
from gossip within about a minute.

## Is a shard actually corrupt? Ask the origin, not the swarm (2026-08-25)

**Peer agreement is not evidence in a network that copies from itself.** Two
independent peers served byte-identical bytes that failed verification here, which
reads as "our expected hash must be wrong" — it was not; the corruption had
spread. Only the model's ORIGIN settles it.

A shard is a byte range of the upstream GGUF, so fetch exactly that range and hash
it. The ranges come from `manifest.json`:

```python
# coalesce the shard's tensors into contiguous GGUF runs, in shard_offset order
for t in sorted(shard["tensors"], key=lambda t: t["shard_offset"]):
    if runs and t["gguf_offset"] == runs[-1][1]: runs[-1][1] = t["gguf_offset"] + t["size"]
    else: runs.append([t["gguf_offset"], t["gguf_offset"] + t["size"]])
# then Range-GET each run in order into one blake3 hasher
```

**⚠ A shard is NOT always one contiguous range.** One llama-3.1-8b shard has two
runs separated by a 122 MB gap; reconstructing it as a single span from
`min(gguf_offset)` to `max(gguf_offset+size)` produced 646021120 bytes against a
declared 523304960 — and a confident FALSE mismatch on a healthy file. **Assert
the reconstructed byte count equals the manifest's `size_bytes` before believing
any verdict**; that one check catches it.

Recovery, once the origin has spoken: `POST /api/admin/hf/download-shards` with
`{"model_id": …, "shards": [n]}` refetches from the origin and (since v0.3.123)
records the hash as origin-verified, so no peer's claim can displace it.

## Benchmarks

Every harness below runs against an ISOLATED node or no daemon at all. None of
them touch a running node; several used to, and that is where most of the traps
in this section came from. **The two Python harnesses are the deliberate
exception**: they measure a LIVE node over its own API, because that is what a
user gets, and they change nothing on it — they are how #432 and #433 were found
on the released binary when a tester's own node ruled out test nodes.

**Before quoting a GPU number, check WHICH DEVICE the model is actually on.**
`GET /api/admin/models` reports `cpu_placement_reason` per model, read from the
worker's recorded placement rather than re-predicted, so it stays truthful even
after the memory frees. A model demoted at admission — because another model
took the budget first — runs perhaps 5x slower and nothing about the request
says so. Measured 2026-08-31: llama-3.2-3b read a stable 9.0-9.5 tok/s against
41.0 on the same box and binary, purely because phi-3.5 had taken 5676 MB of a
6616 MB budget.

**And a back-to-back benchmark loop can prevent the recovery it is measuring.**
`worker_should_return_to_gpu` refuses to promote a worker used within
`VRAM_MAKE_ROOM_MIN_IDLE_SECS` (5 s), so a loop that issues requests with no gap
holds the model on the processor indefinitely. Inserting a 12 s gap between reps
produced 9.7 -> 15.6 -> 33.1 tok/s as it walked back onto the card. Note the
shape: **a large change with a TIGHT spread is a different configuration, not
noise** — the opposite of the contention signature, where the mean moves and the
spread widens with it. See gotcha #422.

| harness | what it measures | notes |
|---|---|---|
| `examples/prefill_bench.rs` | prompt processing + decode, driving `SplitModel::forward` directly | no daemon, no scheduler, no API in the way. `SWARM_BENCH_MODEL` (a model dir holding every shard), `SWARM_BENCH_PROMPT` (896), `SWARM_BENCH_DECODE` (32), `SWARM_BENCH_REPS` (3), `SWARM_BENCH_DEVICE=cuda`. Pair with `SWARMLLM_PROFILE=1` for the per-stage breakdown |
| `examples/qmatmul_bench.rs` | the quantized matmul against batch size | ALSO asserts the tiled path is bit-identical to the upstream ordering — run it after touching either kernel |
| `examples/tokenizer_scaling.rs` | `SplitTokenizer::encode` against prompt length | tells an O(n) tokenizer from an O(n²) one — point `SWARM_TOK_HEADER` at a model's `gguf_header.bin`. It prints `tokenizer_model` / `merges` / `scores`, which is what decides WHICH encode path a GGUF takes (#420); a doubling that quadruples the time is the signature |
| `examples/attn_bench.rs` | attention ops in isolation | ⚠ an isolated call is not a forward pass (#255/#266) |
| `examples/sysinfo_probe.rs` | what it costs to describe this machine — `System::new_all()`+`refresh_all()` against a targeted refresh, and that both report the SAME facts | the admin `stats` endpoint spent 182 ms of its 273 ms here (#417). Prints a value comparison first: a cheaper call that answers `Unknown` for the CPU name is a regression, not a win |
| `examples/stream_bench.py MODEL [--reps N --max-tokens N --port P]` | what a user gets from a running node: streaming TTFT, decode tok/s (`(n-1)/(t_last-t_first)`, a client-side window — #312), whole-request tok/s, the card's memory before/after | reads the API key from the data dir. Compare arms WITHIN one session only (decode spreads ~9-19% on this box); verify the mechanism per arm — placement log lines, `vram_after_load_mb`, `cpu_placement_reason` — not just the number. Found #432 |
| `examples/remote_checks.py [MODEL...]` | remote inference through the real swarm: route headers (`x-swarm-nodes`, `Server-Timing` per segment), one finish per stream (#414), multi-byte replies whole (#416) | non-streaming for the headers, streaming for the finish/duplication checks. ⚠ Run at steady state — ~60 s after a restart everything peer-held 503s "insufficient capacity" (rule 3). **Check the FAILURE paths too** (a model nobody holds, streaming): that is where #433 was |
| `examples/smoke_test.sh [binary] [port]` | 9 end-to-end checks on an isolated node | run it on the DOWNLOADED release artifact, not a local build (#268) |
| `examples/release_shapes.sh [binary] [port]` | 7 pre-release shape checks — cold start, long cold prompt, `prompt_tokens` agreeing cold and warm (#400), greedy determinism WITH a live control, tool-heavy | also on the DOWNLOADED artifact, BEFORE tagging. Local verification used to be a strict subset of CI's |
| `examples/swap_patience.sh` | what the GPU swap floor costs, in CONVERSATION | two models that each fit the card but not together, alternating multi-turn so a warm prefix is worth something. Arms switched by `SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS`, never by rebuilding. Measured 2026-08-28: floor 60 s → 299 s, floor 0 → 82 s, floor 5 s → 89 s (#403) |
| `examples/soak_test.sh [binary]` | sustained inference, sampling worker RSS / KV / threads / fds / ok-fail | `HOURS=` must be a WHOLE number (shell arithmetic); data dir is `/tmp/swarm_soak-$PORT`, per-port so two soaks cannot kill each other; analyse with `soak_report.sh` |
| `examples/two_node_test.sh`, `3node_setup.sh`, `3node_sharded_setup.sh` | cross-node paths | EXPECTED to fail on a single multi-interface host — that is the documented connection-churn case, not a regression. Validate on two real machines |

### Current baseline — 2026-08-29, v0.3.132-alpha

**Re-take with the same command before claiming a delta.** These were taken on
an idle box (AMD Ryzen 7 5800H / RTX 3070 Laptop, WSL2) with the live node
running but idle.

```bash
SWARM_BENCH_MODEL=~/.local/share/swarmllm/models/llama-3.2-3b-instruct-q4-k-m \
RAYON_NUM_THREADS=4 SWARM_BENCH_REPS=3 \
./target/release/examples/prefill_bench     # --no-default-features --features dev
```

| metric | 2026-08-15 (v0.3.97) | **2026-08-29 (v0.3.132)** | |
|---|---|---|---|
| CPU prompt processing | 20.97 tok/s | **32.58 tok/s** (896 tok in 27.51 s) | 1.55x |
| CPU decode | 4.71 tok/s | **10.44 tok/s** (95.7 ms/tok @ ~912 KV) | 2.21x |
| model load, 28 layers | 14.0 s | **5.3 s** warm | |
| KV cache | 235 MB alloc / 213 used / 91% | unchanged | |
| GPU, end-to-end via the API | — | **45.4 tok/s** warm (17.7 cold) | |

`RAYON_NUM_THREADS=4` is half the physical cores and is kept only for
comparability with the 0815 series — a run at 8 threads is a different
configuration, not an improvement.

The GPU row is **end-to-end through `/v1/chat/completions`** (200 tokens,
`temperature=0`), so it includes prefill, templating and HTTP. It is a
user-visible number and is **not** comparable with the CPU rows, which drive
`SplitModel::forward` with no daemon in the way.

Two things that will otherwise be misread:

- **Decode spread is now 8.8%** (104.1 / 95.7 / 99.1 / 96.3 ms across four runs),
  against the ~3.5% recorded in the 0815 baseline. Prefill is still tight
  (2.4%). So this box currently cannot resolve a decode change below ~10% —
  quoting the old 3.5% would license a false positive. Re-check the spread
  before trusting a small delta, rather than assuming the recorded one still
  holds.
- **A first run after a build reports model load at ~68 s, not ~5 s.** That is
  cold page cache — a release build evicts it and the shards are ~2 GB on the
  WSL vhdx. An immediate re-run loaded in 5.3 s with prefill and decode
  reproducing to within 2.4% and 0.6%. **Do not report a load-time regression
  without a warm second run.**

### Traps that have cost real time

- **The box must be idle.** The same unchanged code path measured 0.42 ms and
  0.97 ms here. A run taken while a build or another bench is going is worthless,
  and it will not look wrong — it will look like a result.
- **min-of-N is for BENCHMARKS, not live measurement** (#367). Controlled
  environment, every error adds time → the minimum is the least contaminated.
  Samples taken from live traffic are different tokens at different cache
  lengths on a busy machine → the minimum is the LUCKIEST one.
- **A/B inside ONE binary**, via an env switch — `SWARMLLM_DECODE_CALIBRATE=0`,
  `SWARMLLM_DECODE_ATTN=standard`, `SWARMLLM_FORCE_STANDARD_ATTN`,
  `SWARMLLM_DECODE_THREADS=0`, `SWARMLLM_VRAM_SWAP_MIN_IDLE_SECS`. Comparing two
  builds compares two builds.
- **A one-shot benchmark cannot see a cost that only appears across turns.** The
  GPU swap floor was defended on the grounds that eviction discards a model's
  warm prefix cache — true, and it still lost 3.65x once measured in
  conversation, because the processor is slower at *every* turn than the reload
  it spares (#403). If the mechanism you are arguing about only bites on the
  second request, the benchmark has to make a second request.
- **Do not start a node and run a long request in ONE bash call.** The 2-minute
  harness timeout SIGTERMs the process group, which includes a node launched
  with `nohup … &` in the same invocation — it kills the daemon mid-load and the
  resulting "early eof / worker closed connection before reply" reads exactly
  like a crash. Use `setsid`, and keep requests in separate calls.
- **Verify the mechanism fired.** An outcome can improve for unrelated reasons;
  assert on the log line or counter the change emits.
- **A short run magnifies a one-time cost** into what looks like a standing
  loss. Vary the length it should amortise against before believing it.
- **The benches have no tracing subscriber**, so `tracing::info!` from the code
  under test goes nowhere. If a decision needs to be observed, give it an
  explicit `eprintln!` behind an env var.
