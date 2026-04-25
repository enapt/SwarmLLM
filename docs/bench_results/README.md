# Responses API benchmark artifacts

Test rig: TinyLlama-1.1B Q4_K_M on CPU (WSL2), daemon on `localhost:8830`,
data dir `/tmp/resp_final/`. Captured 2026-04-25.

## Files

### Matrix scripts (correctness)
- `responses_api_v1_matrix.sh` — M1–M9 cases (built-in tool rejection,
  plain text, function tools, cloud routing, SSE, persistence,
  chaining, background, cancel). The M5 case ("`claude-*` → 400") was
  *updated* after the v2 plan landed: V3 now translates `claude-*`
  models to Anthropic Messages instead of returning 400, so the test
  accepts 200 (subscription/key configured) or upstream 4xx/5xx.
- `responses_api_v2_matrix.sh` — V1–V8 cases on top of M1–M9. Multimodal
  rejection paths, UTF-8 file inline, input_items pagination, V8
  202+Location handshake + SSE replay + cursor skip, V5 completed-
  record replay, V6 admin list + status filter, V1 first-byte timing.

Last green run: 38/38 M1–M9 + 27/27 V1–V8 (commit `5a138cb`).

### Bench scripts (latency)
- `responses_api_v1_bench.sh` — original v1 streaming bench. **Has a
  measurement bug** described under "Methodology pitfalls" below.
  Kept here for historical comparison; do not use to validate V1.
- `responses_api_v2_bench.sh` — v2 bench replacing v1's misleading
  pipeline-time measure with curl's `time_starttransfer` (TTFB).
  Adds M9 / V8 / V4 / V6 endpoint timings. Output captured in
  `responses_api_v2_bench_2026-04-25.txt`.
- `v1_compare_bench.sh` — same harness run against an arbitrary daemon
  with both methodologies (old + TTFB) for direct apples-to-apples
  comparison. `LABEL=… ITERS=… PORT=… bash v1_compare_bench.sh`.
- `v1_event_timer.py` — precise Python timer for the SSE first-event
  arrival (the metric V1 actually targets — see below).

## Methodology pitfalls (the "did V1 work?" question)

The v2 plan opened with the claim:

> Streaming first-byte latency fix — bench showed +400-700 ms vs Chat —
> same hop, different path

That number came from `responses_api_v1_bench.sh` line:

```bash
t=$( { time -p (curl ... | grep -m1 "^data: " > /dev/null) 2>&1; } \
        2>&1 | grep "^real" | awk '{print $2}')
```

This `time -p` of the subshell does **not** measure when the first
SSE `data:` line arrived. It measures when the subshell exits. The
subshell exits when both `curl` and `grep` exit:

- `grep -m1` exits as soon as one match is seen.
- `curl` keeps running until SIGPIPE on its next write.
- Curl's next write is the next SSE chunk from the server.
- On TinyLlama CPU each chunk is ~150–600 ms apart while the model
  is generating, and seconds apart between final-event and stream
  close.

So `time -p` is bounded **below** by inference completion + curl
SIGPIPE notice, not by first `data:` arrival. The "+400–700 ms gap"
was variance in inference completion time between consecutive runs,
not a real difference in when `response.created` reached the client.

### TTFB is also wrong (different reason)
`curl --time_starttransfer` clocks the HTTP response head, not the
first body byte content. Axum returns the SSE Response object as soon
as `run_streaming` returns; pre-V1 that already happened in <2 ms
because the inner `chat_completions().await` returned its (empty-body)
SSE Response head fast. So TTFB is invariant pre/post-V1. Useful for
"is the response head delayed?" but not for "when did the first
event arrive?".

### The right metric: timing to first SSE `data:` line
`v1_event_timer.py` opens a raw HTTP connection, posts, then reads
bytes from the body until it sees a complete `data:` line. Records
the elapsed wall-clock from request start to that line.

## Pre/post V1 results (precise event-arrival timing)

| build | iter | chat first `data:` median | resp first `data:` median | gap |
|---|---|---|---|---|
| pre-V1  (`c5d9659`) | 20 | 1.9 ms | 2.5 ms | **0.6 ms** |
| post-V1 (`8c1e3c2…`) | 20 | 1.8 ms | 2.6 ms | **0.8 ms** |

The two builds are statistically indistinguishable at warmed-up steady
state on TinyLlama CPU. The 0.2 ms difference between gap medians is
well inside the per-iteration noise (max samples differ by 1–2 ms).

Full output: `event_timing_pre_v1_2026-04-25.txt`,
`event_timing_post_v1_2026-04-25.txt`.

### Why does V1 still matter then?
V1 is structurally correct: post-V1, `response.created` is yielded
*before* `chat_completions()` is awaited, so any future preflight
work in chat_completions (cold worker probe, slow template build,
queue wait) cannot block the lifecycle event. The conditions that
made the original "400 ms gap" claim reproducible were almost
certainly cold-start sensitive (worker subprocess spin-up before
warmup ran). At steady state, both builds emit `response.created`
within ~2 ms; V1 just guarantees that property holds even when chat
preflight is slow.

The fix is also cleaner code: the SSE generator owns the lifecycle
events, doesn't have to interleave them with chat-stream consumption.

## Both-methodology side-by-side

`v1_compare_bench.sh` runs both old (pipeline-time) and new (TTFB)
metrics in one pass against any daemon:

```
[pre-V1]  old(pipeline-time): chat=2795 ms  resp=3015 ms  gap=220 ms
[pre-V1]  new(TTFB):          chat=0.8 ms   resp=1.0 ms   gap=0.2 ms
[post-V1] old(pipeline-time): chat=3110 ms  resp=3560 ms  gap=450 ms
[post-V1] new(TTFB):          chat=0.8 ms   resp=0.9 ms   gap=0.1 ms
```

The "old" rows show that the v1 bench script's gap is a function of
inference variance, not stream lifecycle. The "new" rows show TTFB
is invariant pre/post-V1. Neither metric on its own validates V1 —
that takes the precise event timer in `v1_event_timer.py`.

## How to reproduce

```bash
# 1. Build a daemon. From a worktree at the commit you want to test:
RUSTFLAGS="" cargo build --release --no-default-features \
    --features dev,claude-subscription

# 2. Start it. /tmp/resp_final must contain a TinyLlama model dir +
#    api_key file (re-use the one this repo's harness leaves there).
SWARMLLM_NODE_DATA_DIR=/tmp/resp_final SWARMLLM_FRONTEND_DIR=./frontend \
    ./target/release/swarmllm run -p 8830 > /tmp/daemon.log 2>&1 &

# 3. Run the matrix(es).
bash docs/bench_results/responses_api_v1_matrix.sh
bash docs/bench_results/responses_api_v2_matrix.sh

# 4. Run the bench(es).
bash docs/bench_results/responses_api_v2_bench.sh
LABEL=$(git rev-parse --short HEAD) ITERS=20 \
    bash docs/bench_results/v1_compare_bench.sh
python3 docs/bench_results/v1_event_timer.py --iters 20 \
    --label $(git rev-parse --short HEAD)
```
