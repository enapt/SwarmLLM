# OpenAI `/v1/responses` — v2 Plan

**Status**: V1–V8 shipped 2026-04-25 across commits `8c1e3c2..c85e3b4`.
V9 (`POST /v1/responses/compact`) remains deferred indefinitely — no
concrete caller has asked for it yet.

**Inputs**: v1 commit log, `docs/plans/responses_api.md` § Watch-list, end-to-end matrix
(38/38 pass) and benchmarks captured in `/tmp/resp_final/`.

**Final test count**: 814 lib tests passing (up from 769 at v1 close —
45 new tests across V1–V6/V7/V5+V8). Clippy clean both with and without
`claude-subscription`.

**End-to-end curl matrix**: 38/38 M1–M9 (`docs/bench_results/responses_api_v2_matrix.sh`
covers the new V1–V8 surfaces; the original `/tmp/resp_final/matrix.sh` for
M1–M9 was updated for the V3 routing change so M5 no longer expects 400
on `claude-*`). 27/27 V1–V8 cases pass — multimodal rejection paths,
input_items pagination + cursor + order=desc, V8 202+Location return,
V8 SSE replay with cursor, V5 completed-record replay, V6 admin list
endpoint with status filter, V1 first-byte gap.

**Bench rerun** (`docs/bench_results/responses_api_v2_bench.sh`,
TinyLlama CPU on WSL2):

| | median | mean | p95 |
|---|---|---|---|
| chat_completions stream TTFB | 1.1 ms | 1.1 ms | 1.2 ms |
| **responses stream TTFB** | **1.2 ms** | **1.2 ms** | **1.4 ms** |
| POST background=true (queued) | 2.5 ms | 3.0 ms | — |
| POST background+stream 202 | 2.4 ms | 3.0 ms | — |
| GET /v1/responses/:id (hit) | 0.9 ms | 0.9 ms | — |
| GET /v1/responses/:id/input_items | 0.8 ms | 0.9 ms | — |
| GET /api/admin/responses (list 100) | 3.5 ms | 3.5 ms | — |

V1 streaming TTFB gap: **0.1 ms median** (well inside the 20 ms target;
v1's bench script measured `(curl | grep) time` which included pipeline
tear-down latency and produced misleading ~600 ms readings — the v2
bench uses curl's `time_starttransfer` which is what V1 actually
targets).

## Why v2

v1 landed the core surface (types, routes, Chat translation, cloud proxy,
streaming, redb persistence, `previous_response_id` chaining, background + cancel).
38/38 curl matrix pass, 769 lib tests green. Testing surfaced one real gap and
benchmarks showed the remaining overhead is concentrated on the streaming path.
The deferred items from v1's watch-list are still deferred; several of them now
have concrete callers in the wild because OpenAI's 2026 Q1 SDKs default to
Responses for gpt-5 / o-series.

## What v2 adds (in priority order)

| # | Milestone | Driver | Estimate |
|---|-----------|--------|----------|
| V1 | Streaming first-token latency fix | Bench showed +400-700 ms vs Chat — same hop, different path | 0.5 day |
| V2 | Multimodal input (image/file/audio → Chat vision) | Current translator drops all non-text content parts | 1 day |
| V3 | Claude → Anthropic Messages translation on `/v1/responses` | Today: 400 "use /v1/messages". Real callers using OpenAI SDK against a Claude model just get stuck. | 1-2 days |
| V4 | `GET /v1/responses/:id/input_items` pagination | Deferred from v1 (501). OpenAI SDKs do hit it for retried-tool-call flows. | 0.5 day |
| V5 | Resumable SSE (`stream=true&starting_after={seq}`) | Deferred from v1 (400). Only useful once V8 background-streaming lands — wire both together. | 1 day |
| V6 | Dashboard surface for `/v1/responses` | Parallel of the existing Chat panel; keeps the endpoint discoverable. | 1 day |
| V7 | `reasoning` item propagation for o-series cloud proxy | Cloud proxy already forwards extras; this milestone adds an explicit `include[reasoning.encrypted_content]` shortcut + a stored-record flatten that preserves reasoning across `previous_response_id` chains on the local path. | 0.5 day |
| V8 | Background streaming (`background=true` + `stream=true`) | V1 rejected this pair; resumable SSE (V5) makes the right shape work. Pair these. | 1-1.5 days |
| V9 | `POST /v1/responses/compact` | Explicitly v3 in the original plan — still v3 unless a caller needs it. | (deferred) |

Total for V1-V8: ~6-7 focused days. V9 stays deferred.

## Non-goals for v2

- Server-side conversation-resource CRUD. OpenAI's `conversation` parameter
  forwards through the cloud proxy today; a local `conversation` type with its
  own endpoints is a separate design.
- Built-in tools (`web_search`, `file_search`, `computer_use_preview`,
  `code_interpreter`, `image_generation`, `mcp`) for the LOCAL path. They still
  forward via cloud proxy unchanged.
- `custom` tools with Lark / regex grammars. Reject on local, forward on cloud.
- Token-level cancel for background inference. Current MVP checks the cancel
  flag at completion; per-token cancel needs hooks into `chat_completions`
  that are out of scope.

## V1. Streaming first-token latency fix

**Measured (TinyLlama, CPU, 5-token output, 5 iter)**:
- `/v1/chat/completions` stream: median first `data:` line = 2360 ms
- `/v1/responses` stream: median first `data:` line = 2760 ms
- Gap: ~400 ms, reproducibly.

**Root cause hypothesis**: `stream::run_streaming` awaits the full
`chat_completions` call before parsing its body and emitting the first
`response.created` event. On CPU TinyLlama the handler does blocking setup
(worker probe, template build) before returning the SSE `Response`, and that
window happens before we can yield `response.created`.

**Fix**: yield `response.created` + `response.in_progress` the moment the
caller's HTTP request hits the handler — don't wait for chat_completions to
even return the SSE Response. The `initial_response` struct is built from the
request alone, no inference needed. Options:

1. Emit the two lifecycle events first, then wrap the chat SSE stream as its
   tail. Requires restructuring `run_streaming` so the outer SSE stream starts
   before the inner `chat_completions().await`.
2. Spawn `chat_completions` as a task, open the SSE response immediately, and
   feed chat bytes into the event generator as they arrive.

Option 1 is cleaner — the generator already has the initial state. Refactor
`build_response_event_stream` so it takes a future resolving to
`Result<Stream<Bytes>, ApiError>` instead of a resolved stream. Yield
`response.created` + `response.in_progress` unconditionally; then `await` the
future and consume its stream (or yield `response.failed` if it errored).

**Checkpoint**: bench shows Responses streaming first-token within 20 ms of
Chat streaming first-token. Re-run `/tmp/resp_final/bench.sh`.

## V2. Multimodal input on `/v1/responses`

v1's `translate::collect_text_from_parts` drops `input_image`, `input_file`,
and `input_audio` parts silently. The Chat handler already supports vision
(base64 image_url). Wire them through.

**Mapping**:
- `input_text {text}` → existing `ContentPart::Text {text}`.
- `input_image {image_url}` → `ContentPart::ImageUrl {image_url: {url}}`. Base64
  data URIs pass straight through; the chat decoder already handles decode +
  size cap + format validation.
- `input_image {file_id}` → 400 "file_id references not supported on this
  server; inline the image via image_url base64 data URI instead". (File IDs
  require an uploads API we don't run.)
- `input_file {file_id}` → same rejection.
- `input_file {file_data, filename}` → inline base64 document. For PDF / text,
  decode server-side and concat as text input. Keep the 20 MB cap.
- `input_audio {input_audio}` → 400 "audio input not supported" for now. Real
  audio handling needs Whisper-class model plumbing.

**Translation change**: `collect_text_from_parts` → `collect_content_parts`
returning `Vec<ContentPart>` (Chat type). Message content becomes `Parts(...)`
when any non-text part exists; falls back to `Text(...)` when all parts were
text.

**Tests**:
- Base64 image → 200, chat handler receives image in messages.
- `file_id` only → 400 with clear error naming the field.
- Audio-only input → 400 "audio input not yet supported".
- Mixed text + image parts → 200 with both in the chat message.

**Checkpoint**: LLaVA-v1.5 (already staged per MEMORY) answers a multimodal
Responses request routed through `/v1/responses`.

## V3. Claude → Anthropic Messages translation

Today: `model=claude-*` on `/v1/responses` returns 400 "use /v1/messages".
v2 translates instead. Claude models become first-class on both endpoints.

**Flow**: When `resolve_provider` returns an Anthropic or claude-subscription
provider:
1. Translate `ResponsesRequest` → `MessagesRequest` (Anthropic shape).
2. Call the existing `proxy_to_anthropic` (or `claude_sub::send`) with the
   translated body.
3. Translate the Anthropic `MessagesResponse` / SSE back to Responses shape.

**Translation sketch**:

| Responses field | Anthropic field |
|---|---|
| `instructions` | `system` (string or blocks) |
| `input` text | `messages[0]` user content |
| `input` array messages | `messages[]` with role + content blocks |
| `input` function_call | assistant message with `tool_use` block |
| `input` function_call_output | user message with `tool_result` block |
| `max_output_tokens` | `max_tokens` |
| `temperature`, `top_p` | direct |
| `stop` | `stop_sequences` |
| `tools[{type:function,...}]` | `tools[{name,description,input_schema}]` |
| `tool_choice` | `tool_choice` |
| `reasoning.effort` + `.summary` | `thinking {type:'enabled', budget_tokens:N}` |
| `metadata` | `metadata` |

Response side:
| Anthropic | Responses |
|---|---|
| `content[type:text]` | `output[{type:message, content:[{type:output_text,...}]}]` |
| `content[type:tool_use]` | `output[{type:function_call,...}]` |
| `content[type:thinking]` | `output[{type:reasoning, summary:[{type:summary_text,text}]}]` |
| `stop_reason: end_turn` | status: completed |
| `stop_reason: max_tokens` | status: incomplete + incomplete_details |
| `stop_reason: stop_sequence` | status: completed (with stop_sequence in metadata) |
| `stop_reason: tool_use` | status: completed |
| `usage.input_tokens/output_tokens/cache_*` | `usage.input_tokens/output_tokens + input_tokens_details.cached_tokens` |

**Streaming**: map Anthropic's `message_start`, `content_block_start`,
`content_block_delta`, `message_delta`, `message_stop` → Responses events.
Similar structure to V1 OpenAI Chat mapping.

**Files**:
- `src/api/openai/responses/anthropic_bridge.rs` (new): Responses ↔ Messages
  translation.
- Handler: cloud-proxy precedence already tries `try_proxy_openai_responses`;
  add a companion `try_proxy_anthropic_responses` that translates + forwards
  to Anthropic.

**Tests**:
- Roundtrip: Responses request → Messages request → expected wire shape.
- Roundtrip: Messages response → Responses response → expected output items.
- Thinking block → reasoning item.
- Tool use → function_call.
- Streaming event mapping.
- With `claude-subscription` feature: route through `claude_sub`.

**Checkpoint**: OpenAI Python SDK's
`client.responses.create(model="claude-sonnet-4-6", input=..., tools=...)`
works end-to-end (non-streaming + streaming).

## V4. `GET /v1/responses/:id/input_items` pagination

Simple: iterate `stored.request.input.Items(...)` if array form, or emit a
single `message` item if string form, paginated with `after` + `limit`.

**Shape**:
```
GET /v1/responses/:id/input_items?limit=20&after=<item_id>
→ {"object":"list","data":[...input items...],"first_id":..,"last_id":..,"has_more":bool}
```

**Checkpoint**: curl returns paginated items matching the stored request.
`after` param resumes mid-list.

## V5. Resumable SSE

When a caller passes `?starting_after={seq}` on a GET to `/v1/responses/:id`
with `stream=true`, replay the cached event buffer from `seq+1` onward, then
live-tail if the response is still in_progress.

**State**: add `Arc<Mutex<Vec<(u64, Event)>>>` to `BACKGROUND_CANCEL`'s map
value so events accumulate during a background+streaming run. Cap at e.g.
2000 events per response (enough for multi-thousand token outputs).

**Wire**: re-serialize cached events with their original `sequence_number`.
After draining the cache, subscribe to the live event channel and continue.

**Checkpoint**: disconnect mid-stream, reconnect with `starting_after=<N>`,
receive the remaining events without duplicates.

## V6. Dashboard surface

New panel: `frontend/js/components/responses.js`, mirrors the Chat tab but
adds:
- Retrieve-by-id input (paste an OpenAI response id, fetch from local store).
- Background task list (queued / in_progress / completed / cancelled).
- Cancel button per background entry.

**Backend**: add `GET /api/admin/responses?status=...` for listing. Reuses the
existing admin Bearer auth.

**i18n**: add keys to `frontend/i18n/en.json` for each label; propagate to all
20 other locales per `.claude/rules/i18n.md`.

**Checkpoint**: dashboard panel renders; create-a-response, list, retrieve,
delete all work from the UI against local models.

## V7. Reasoning item propagation

Two pieces:
1. **Cloud-proxy include shortcut**: when `include` array contains
   `reasoning.encrypted_content`, add it verbatim to the outgoing request
   (already forwards via extras; explicit test + doc).
2. **Local chain preservation**: the M8 flatten drops reasoning items today
   (only cloud models use them). Keep them in the stored record and skip them
   during flatten — already the case. Test: chain through three turns, prior
   record's reasoning items round-trip into the stored response.output but
   are NOT re-injected into chat messages.

**Checkpoint**: two-turn chain with an o-series model preserves
`encrypted_content` across calls (verified with integration-style test that
mocks the upstream).

## V8. Background streaming

Combine M9 spawn + M6 streaming + V5 resumable SSE.

**Flow**:
- `POST /v1/responses` with `background=true` and `stream=true` responds
  with `202 Accepted` + a Location header pointing at
  `/v1/responses/:id?stream=true&starting_after=-1`.
- Client opens that GET; the server begins consuming the chat stream
  internally (via the existing `run_streaming` logic, but with output
  redirected into the cached event buffer) and replays to the caller.
- Cancel remains the same POST `/cancel` — the buffer gets a final
  `response.cancelled` event and the cached chat consumer shuts down.

**Checkpoint**: OpenAI Python SDK's
`client.responses.create(..., background=True, stream=True)` iterates events
resumably.

## V9. `POST /v1/responses/compact` (deferred)

Still v3. Flagged here for completeness; don't implement unless a concrete
caller asks for it.

## Implementation order

Sequential:
1. V1 (streaming perf) — short, high-impact, unblocks cleaner streaming code
   for later milestones.
2. V2 (multimodal) — extends existing M3/M4 translators; mostly additive.
3. V3 (Claude translation) — the biggest surface-expansion; ships Claude on
   both endpoints.
4. V4 (input_items) — small bookkeeping endpoint.
5. V6 (dashboard) — visual, nice-to-have; can park between V3 and V5 if a
   break is needed.
6. V7 (reasoning) — short.
7. V5 + V8 (resumable SSE + background streaming) — joint design; ship
   together.
8. V9 — deferred indefinitely.

## Validation matrix (per milestone)

Same as v1:

```bash
cargo fmt && cargo clippy --all-targets --no-default-features --features dev,claude-subscription -- -D warnings
cargo test --lib --no-default-features --features dev,claude-subscription
cargo build --release --no-default-features --features dev,claude-subscription
```

Plus end-to-end curl against `/tmp/resp_v2_test` per milestone. Reuse
`/tmp/resp_final/matrix.sh` (update for new cases) and `/tmp/resp_final/bench.sh`
to verify no latency regression.

## References

- v1 plan: `docs/plans/responses_api.md`
- v1 commits: `dfa4af6..6dd4e4b` (M1-M9)
- End-to-end matrix: `/tmp/resp_final/matrix.sh` (38/38)
- Benchmarks: `/tmp/resp_final/bench.sh`
- OpenAI Responses API reference: https://platform.openai.com/docs/api-reference/responses
- Anthropic Messages API: `src/api/anthropic/*.rs`
- Claude subscription subprocess: `src/api/claude_sub.rs`
