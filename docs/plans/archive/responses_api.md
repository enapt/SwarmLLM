# OpenAI `/v1/responses` — Design Plan

**Status**: V1 (Milestones 1–9) shipped 2026-04-24. See `responses_api_v2.md`
for the V1–V8 follow-on work (also shipped 2026-04-25). This file is preserved
as the original design rationale.
**Target**: SwarmLLM exposes a proxy-compatible `/v1/responses` endpoint that covers the ~80% of real-world use (plain generation + function calling + reasoning) and explicitly surfaces what it can't translate (built-in tools, server-side conversation state, compaction).
**Non-goal**: feature parity with OpenAI's managed Responses API. Built-in tools (`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`) stay out of scope for v1 — they require backing infra we don't run.

## Why this exists

Responses is the default API for o-series and gpt-5-series in 2026. Current OpenAI SDKs default `client.responses.create(...)` for those model families. Without this endpoint, SDKs fall back to Chat Completions or outright 404 for reasoning-sensitive flows. The Assistants API sunsets 2026-08-26; Responses is the replacement.

SwarmLLM already proxies `/v1/chat/completions` and `/v1/messages` (Anthropic). Adding `/v1/responses` closes the OpenAI SDK compatibility gap for reasoning-era models and gets ahead of the deprecation wave.

## Endpoint surface

| Method | Path | Status |
|---|---|---|
| POST | `/v1/responses` | v1 scope — core generation path |
| GET | `/v1/responses/{id}` | v1 scope — requires `store=true` round trip |
| DELETE | `/v1/responses/{id}` | v1 scope — redb-backed persistence |
| POST | `/v1/responses/{id}/cancel` | v1 — idempotent, for `background=true` |
| GET | `/v1/responses/{id}/input_items` | v2 — paginated input replay |
| GET | `/v1/responses/{id}?stream=true&starting_after={seq}` | v2 — resumable SSE |
| POST | `/v1/responses/compact` | **out of scope** — server-side compaction needs separate planning |

Persistence backing for `store=true`: new redb tree `responses` keyed by `resp_id`, value = serialized `ResponsesRecord`. Entries expire after 30 days (matches OpenAI retention default) via a background sweep reusing the existing acquisition-cleanup pattern.

## Request body

Define `ResponsesRequest` in `src/api/openai/responses.rs` mirroring the spec:

```rust
pub struct ResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,          // string | Vec<InputItem>
    pub instructions: Option<String>,
    pub previous_response_id: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    pub stop: Option<StopField>,
    pub seed: Option<u64>,
    pub user: Option<String>,
    pub metadata: Option<HashMap<String, Value>>,
    pub stream: Option<bool>,
    pub store: Option<bool>,            // defaults to true upstream
    pub background: Option<bool>,
    pub parallel_tool_calls: Option<bool>,
    pub truncation: Option<String>,
    pub service_tier: Option<String>,
    pub modalities: Option<Vec<String>>,
    pub include: Option<Vec<String>>,
    pub tools: Option<Vec<ToolDef>>,    // type-discriminated
    pub tool_choice: Option<ToolChoice>,
    pub reasoning: Option<ReasoningOpts>,  // {effort, summary, encrypted_content}
    pub text: Option<TextFormat>,       // {format, verbosity}
    pub conversation: Option<ConversationRef>,
    pub context_management: Option<Vec<ContextMgmt>>, // deferred
    #[serde(flatten)]
    pub extras: HashMap<String, Value>, // preserve unknown fields
}
```

Input items (array form): `message`, `function_call`, `function_call_output`, `reasoning`, `mcp_approval_response`, `computer_call_output`, `item_reference`. Content parts: `input_text`, `input_image`, `input_file`, `input_audio`.

Tool types we accept and pass through to local inference as `function`: just the `function` tool. Other tool types (`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`, `custom` with grammar) → fast-fail with a clear error message listing what's not supported, same pattern as the Anthropic server-tool rejection in `aadd368`.

## Response body

```rust
pub struct ResponsesResponse {
    pub id: String,              // resp_<blake3>
    pub object: &'static str,    // "response"
    pub created_at: i64,
    pub status: ResponseStatus,  // queued | in_progress | completed | failed | incomplete | cancelled
    pub model: String,
    pub output: Vec<OutputItem>,
    pub output_text: Option<String>,  // convenience concat
    pub usage: ResponsesUsage,        // {input_tokens, output_tokens, total_tokens, *_details}
    pub error: Option<ApiError>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub previous_response_id: Option<String>,
    pub instructions: Option<String>,
    pub tools: Option<Vec<ToolDef>>,
    pub tool_choice: Option<ToolChoice>,
    pub parallel_tool_calls: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_output_tokens: Option<u32>,
    pub truncation: Option<String>,
    pub metadata: Option<HashMap<String, Value>>,
    pub user: Option<String>,
    pub reasoning: Option<ReasoningOpts>,
    pub text: Option<TextFormat>,
    pub modalities: Option<Vec<String>>,
    pub service_tier: Option<String>,
    pub background: Option<bool>,
}
```

Output item types in scope: `message` (with `output_text` + `refusal` parts), `function_call`, `reasoning` (when the backing model exposes it — Claude-via-Anthropic-proxy, gpt-5-via-cloud-proxy).

Out of scope in v1: `file_search_call`, `web_search_call`, `computer_call`, `code_interpreter_call`, `image_generation_call`, `mcp_*`, `custom_tool_call`, `compaction`. Request-time rejection for these; don't emit them in outputs.

## Streaming events

Full SSE event catalogue is ~53 types. v1 emits:

**Lifecycle (all 7)**: `response.created`, `response.queued`, `response.in_progress`, `response.completed`, `response.failed`, `response.incomplete`, `error`.

**Output item / content / text (9)**: `response.output_item.added`, `response.output_item.done`, `response.content_part.added`, `response.content_part.done`, `response.output_text.delta`, `response.output_text.done`, `response.output_text.annotation.added`, `response.refusal.delta`, `response.refusal.done`.

**Function tool args (2)**: `response.function_call_arguments.delta`, `response.function_call_arguments.done`.

**Reasoning (when backed model exposes it) (4)**: `response.reasoning_summary_text.delta`, `response.reasoning_summary_text.done`, `response.reasoning_text.delta`, `response.reasoning_text.done`.

Every event carries `sequence_number` (monotonic). Resume-from-seq is v2.

## Chat Completions ↔ Responses translation

Proxy shape for local inference:

| Responses field | Chat field | Notes |
|---|---|---|
| `input` (array of `message` items) | `messages` | Map `input_text` → `text`; `input_image` → `image_url`; `input_file` → translate or reject |
| `max_output_tokens` | `max_tokens` | Direct |
| `text.format.json_schema` | `response_format.json_schema` | Name/schema/strict direct |
| `tools[{type:function, ...flat}]` | `tools[{type:function, function:{...}}]` | Flatten/unflatten |
| `tool_choice{type:function, name}` | `tool_choice{type:function, function:{name}}` | Flatten/unflatten |
| `instructions` | system `messages[0]` | Or developer message, whichever matches backing model |
| `output[*].content[type:output_text].text` concat | `choices[0].message.content` | Reverse on proxy response |
| `output[*][type:function_call]` | `choices[0].message.tool_calls` | `call_id` ↔ `id` |
| `usage.input_tokens` | `prompt_tokens` | |
| `usage.output_tokens` | `completion_tokens` | |
| `usage.output_tokens_details.reasoning_tokens` | `completion_tokens_details.reasoning_tokens` | |

## Irreducible incompatibilities (document + fail loud)

1. **Reasoning-item chaining**: gpt-5 / o-series require prior `reasoning` items (optionally `encrypted_content`) re-fed on next call, immediately before the next tool call, or the upstream 400s with *"reasoning item provided without its required following item"*. Proxying to Chat Completions drops reasoning — acceptable for simple generation, breaks multi-turn tool loops. Surface via `include:["reasoning.encrypted_content"]` when routing to cloud providers that support it; fail with 400 + clear message for local inference models that don't.

2. **Built-in tools** (`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`): no backing infra. Reject at parse time.

3. **`previous_response_id` chaining**: server-side conversation state. For v1, back it with a redb store keyed by `resp_id`; on follow-up, re-expand stored `output` into `messages[]` and prepend to the new `input`. For cloud-provider-backed models, pass `previous_response_id` through verbatim and let the provider handle it.

4. **Multi-item output**: Chat assumes a single assistant turn. Responses can emit message + reasoning + 5 web_search_calls + message in one response. Local inference path outputs at most one `message` item (no tool orchestration beyond user-provided function tools) + one `function_call` item if the model chose a tool. Anything richer requires the cloud path.

5. **`text.verbosity`, `reasoning.effort:"minimal"`, `service_tier:"flex"/"priority"`, `background`+resume, compaction, `custom` grammar tools**: preserved in `extras: HashMap<String, Value>` and forwarded on the cloud-proxy path; ignored on local inference with a one-line `tracing::debug!` noting the skipped fields.

## Implementation sketch (files to create / modify)

**New files**:
- `src/api/openai/responses/mod.rs` — endpoint handlers + route wiring
- `src/api/openai/responses/types.rs` — request + response + streaming-event structs
- `src/api/openai/responses/translate.rs` — Responses ↔ Chat translation pair
- `src/api/openai/responses/stream.rs` — SSE event emitter (maps from existing StreamingToken)
- `src/api/openai/responses/store.rs` — redb-backed Responses record store
- `docs/plans/responses_api.md` — this doc

**Modified**:
- `src/api/server.rs` — route wiring for `/v1/responses`, `/v1/responses/:id`, cancel, input_items
- `src/api/openai/mod.rs` — re-export responses
- `src/storage/db.rs` — new tree `responses`, TTL sweep
- `docs/ARCHITECTURE.md` — OpenAI-compatible API section update

**Estimated effort**: 3–5 days of focused work for v1 scope (plain + function-call + basic reasoning, no built-in tools, `previous_response_id` redb-backed). Streaming adds ~1 day. Each tool type beyond `function` adds ~0.5–2 days of integration depending on whether a stub "not supported" response is enough or real emulation is wanted.

## Watch-list (2026 spec churn)

- `context_management` / server-side compaction — 2026 Q1 addition; full design in separate RFC when v1 lands.
- `conversation` parameter — Q1 2026, partially overlaps `previous_response_id`. For v1, accept as input (forward through) but don't implement conversation-resource CRUD.
- `custom` tools with Lark / regex grammars — reject with clear error for local inference; forward through on cloud proxy.
- `service_tier:"priority"`, `reasoning.effort:"minimal"` — new values. Accept + pass through.
- `include` keys (`message.output_text.logprobs`, `web_search_call.action.sources`, `computer_call_output.output.image_url`) — preserve via `include` array forwarding.
- MCP auth tightening — already out of scope, but note.

## Session-1 pickup guide

This section gives a fresh Claude Code session the exact starting sequence so no rediscovery is needed. Read the top of this file for the design decisions, then follow the milestones below.

### Code patterns to mirror

- **Router registration**: `src/api/server.rs` lines 96–104 show the existing `/v1/*` pattern. Add Responses routes directly under the `/v1/messages` block. Use the same `JsonBody<T>` extractor + `ApiError` return type that `anthropic::messages` uses.
- **Handler skeleton**: `src/api/openai/mod.rs::chat_completions` (line 149) and `src/api/anthropic/mod.rs::messages` (line 34) are the two closest handlers — structure the Responses handler the same way: validate → activity event → route decision (local vs cloud proxy) → emit.
- **Cloud-proxy pattern**: `src/api/anthropic/proxy.rs::proxy_to_anthropic` is the most up-to-date (has the `#[serde(flatten)] extras` pattern, `anthropic-beta` forwarding, and unknown-field preservation from commits `35a0191` / `0ecd38e` / `4347992`). Copy the `extras: HashMap<String, Value>` catch-all idiom into `ResponsesRequest` and `ResponsesResponse` at the top level and inside every content-part / tool-def / output-item struct that OpenAI might extend in-place.
- **Streaming SSE**: `src/api/anthropic/sse.rs` and `src/api/openai/streaming.rs` show the two conventions. Responses uses a new event schema closer to Anthropic's (`type: "response.output_text.delta"` etc.) than Chat's bare choices delta. The Anthropic sse module is a better template.
- **Activity events**: use `state.emit_activity(ActivityEvent::new("inference", "...", msg).with_model(...).with_toast("info", ...))` — see the existing chat_completions handler for the exact shape. Do NOT create a new event category; reuse `"inference"`.
- **Cancel**: `background=true` + `POST /v1/responses/{id}/cancel` maps to our existing cancel-shutdown flow. Spawn an `Arc<AtomicBool>` cancel flag per stored response; the cancel handler sets it, the inference path checks it at token boundaries (same pattern as HF download cancel flags in `state.models.download_cancel_flags`).

### redb persistence

- Tree name: `"responses"` (convention: lowercase, plural, matches `"models"`, `"responses"`, `"transactions"`). The database uses composite keys via `put_json(tree, key, value)` — see `src/storage/db.rs::put_json`.
- Record shape: `ResponsesRecord { id, created_at, request: ResponsesRequest, response: ResponsesResponse, expires_at }`.
- TTL: 30-day sweep. Add to the existing `stale_tensor_interval` sweep in `src/daemon/background.rs` rather than a new interval — it already runs every 10s and iterates quickly.
- Do NOT put ResponsesRecord in SharedState as a DashMap. redb is the source of truth; hitting disk is fine at Responses-create cadence. Cache only the "currently-streaming-a-background-response" map in memory, keyed by `resp_id → Arc<AtomicBool>` cancel flag + `Arc<RwLock<Vec<StreamEvent>>>` resume buffer (for the resume-from-seq path in v2).

### Milestones — each gets its own commit

**Milestone 1 — request parsing (no handler yet).** Create `src/api/openai/responses/types.rs` with all the request/response structs. Add 5+ serde roundtrip tests covering: string `input`, array `input` with mixed message + function_call + reasoning items, nested content parts, tool array with 3+ tool types, all numeric edge cases. `cargo test` green. No routes wired. **Checkpoint**: `cargo test --lib api::openai::responses::types` passes + 10+ tests green.

**Milestone 2 — reject-and-400 for built-in tools.** Wire the `POST /v1/responses` route to a handler that parses the request, rejects `tools[*].type` ∈ {`web_search`, `file_search`, `computer_use_preview`, `code_interpreter`, `image_generation`, `mcp`, `custom`} with a clear 400 message naming which tool is unsupported, and 501 for everything else. Curl test. **Checkpoint**: built-in tool rejection returns 400 with specific tool name in body; plain text request returns 501 (not yet implemented).

**Milestone 3 — Responses → Chat translation (local inference path).** Implement `translate::request_to_chat(responses_request) -> ChatCompletionRequest` and `translate::chat_to_response(chat_response) -> ResponsesResponse`. Plain text in/out, no tools, no streaming. Wire handler: translate, call existing `chat_completions` path, translate back. **Checkpoint**: `curl POST /v1/responses` with `{"model":"<local>","input":"Hello"}` returns a non-streaming ResponsesResponse with `output[0].content[0].text` set.

**Milestone 4 — function tools.** Translate `tools[{type:function, name, description, parameters, strict}]` ↔ Chat's `tools[{type:function, function:{...}}]`. Translate `tool_choice` both directions. Translate assistant `function_call` output item ↔ Chat's `tool_calls`. **Checkpoint**: OpenAI Python SDK's `client.responses.create(..., tools=[...])` round-trips a function call correctly.

**Milestone 5 — cloud proxy path.** For Claude / gpt-5 / o-series model IDs, bypass local translation and proxy verbatim to the upstream provider. Use `#[serde(flatten)] extras` to preserve unknown fields. Forward `anthropic-beta` header when targeting Claude. Add matching `ProviderError` status preservation. **Checkpoint**: `reasoning.effort`, `service_tier`, `include`, `text.verbosity` all round-trip verbatim end-to-end (test with a real upstream if available, or with a local echo server).

**Milestone 6 — streaming (SSE).** Map the internal `StreamingToken` stream to the shortlist of event types in the "Streaming events" section above. Emit `sequence_number` monotonically starting at 0 per response. Handle `response.completed` / `response.incomplete` / `response.failed` terminal states. **Checkpoint**: OpenAI Python SDK's `async for event in await client.responses.create(..., stream=True)` iterates events in the correct order.

**Milestone 7 — `store=true` + retrieve + delete.** redb-backed `ResponsesRecord` storage, `GET /v1/responses/{id}` deserializes and returns, `DELETE /v1/responses/{id}` removes. 30-day TTL sweep. **Checkpoint**: create → retrieve → delete round trip green; stored record survives daemon restart; expired records pruned.

**Milestone 8 — `previous_response_id` chaining (local).** On follow-up call, if `previous_response_id` is set, fetch the stored record, flatten its `output` items back into `messages[]`, prepend to the new `input`, and call the chat path as normal. Cloud proxy path forwards `previous_response_id` verbatim. **Checkpoint**: two-turn conversation via `previous_response_id` produces the same behavior as passing the full messages array.

**Milestone 9 — `background=true` + `cancel`.** Spawn inference in a tokio task keyed by `resp_id`. Return `{status:"queued"}` immediately. `GET /v1/responses/{id}` polls status. `POST /v1/responses/{id}/cancel` sets the cancel flag. **Checkpoint**: curl matrix covering: create background → poll status → cancel mid-stream → status flips to `cancelled`.

### Scope out of v1 (explicit deferrals)

These are tracked in the "Watch-list" section; explicitly return 400/501 with a clear error message rather than silently accepting:

- `GET /v1/responses/{id}/input_items` pagination (return 501)
- `GET /v1/responses/{id}?stream=true&starting_after={seq}` resumable streaming (return 400 — only `stream=true` on create is supported)
- `POST /v1/responses/compact` server-side compaction (return 501)
- `conversation` parameter — accept but log a one-line warn that conversation-resource CRUD is not implemented (pass through for cloud proxy; ignore for local inference)
- Built-in tools (every type listed above) — reject at parse time with a clear error naming the tool

### Gotchas carried over from this session

1. **Unknown field preservation is load-bearing**: the OpenAI proxy fix in `0ecd38e` added `#[serde(flatten)] extras: HashMap<String, Value>` specifically because dropping unknown fields broke real SDKs' passthrough of `reasoning_effort` / `service_tier` / etc. Responses will have 10× more field churn than Chat — default-apply flatten+extras to every struct at top-level and in any nested type OpenAI might extend.
2. **Large payloads go in the IPC binary payload, not JSON header** (gotcha #24 in MEMORY.md). If Responses ever carries images or audio input items, decode them to the internal representation at handler time — don't pass `Vec<u8>` through `WorkerMsg` JSON.
3. **i18n error messages**: any user-facing error message must have a key in `frontend/i18n/en.json` and be propagated to all 20 non-English locales. For Responses this matters for error responses the dashboard might surface (frontend chat component renders upstream error messages).
4. **Per-session Claude Code interaction**: `src/api/claude_sub.rs` (feature `claude-subscription`) routes Claude models through a local subprocess instead of the cloud API. Responses hitting Claude models should check this feature flag and route through `claude_sub` when enabled — mirror the pattern in `anthropic::messages`.
5. **CPU decode ≥ 2048 KV with GQA** now uses fused flash attention (commit `16ed1e8`) — Responses doesn't need to care, but if you add perf tests against long-context Responses calls, that's where the speedup comes from.
6. **Frontend authFetch pattern**: `App.authFetch` already handles Bearer auth for any `/v1/*` call; the Responses endpoint works from the dashboard without frontend changes, as long as the new routes are registered on the server before the catch-all.

### Validation matrix (run before each milestone commit)

```bash
cargo fmt && cargo clippy --all-targets --no-default-features --features dev,claude-subscription -- -D warnings
cargo test --lib --no-default-features --features dev,claude-subscription
cargo build --release --no-default-features --features dev,claude-subscription
```

And for the end-to-end curl smoke test (reuse the pattern from the auth-hardening commit `5a19acc`):

```bash
# start daemon on a test port
mkdir -p /tmp/resp_test
SWARMLLM_NODE_DATA_DIR=/tmp/resp_test SWARMLLM_FRONTEND_DIR=$PWD/frontend \
  ./target/release/swarmllm run -p 8821 >/tmp/resp_test/log 2>&1 &
sleep 3
API_KEY=$(grep "Generated new API key" /tmp/resp_test/log | sed 's/.*: //')

# plain generation
curl -s -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  http://localhost:8821/v1/responses \
  -d '{"model":"<local-model-id>","input":"Hello"}'

# built-in tool rejection (expect 400)
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $API_KEY" -H "Content-Type: application/json" \
  http://localhost:8821/v1/responses \
  -d '{"model":"<local-model-id>","input":"x","tools":[{"type":"web_search"}]}'
```

### Anchors this plan relies on (verify before starting)

- `src/api/openai/mod.rs::chat_completions` (line 149) — handler template.
- `src/api/anthropic/mod.rs::messages` (line 34) — second handler template, closer in structure.
- `src/api/anthropic/proxy.rs::proxy_to_anthropic` — cloud-proxy pattern with serde flatten.
- `src/api/server.rs` lines 96–104 — `/v1/*` route registration style.
- `src/storage/db.rs::put_json` — redb helper convention.
- `src/daemon/background.rs::stale_tensor_interval` — sweep interval to reuse for TTL.
- MEMORY.md gotchas #18 (stop sequences), #24 (JSON header size), #25 (cross-node prefix-fetch timeouts) — general discipline that Responses implementation should follow.

## References

- [Create a model response — OpenAI API Reference](https://platform.openai.com/docs/api-reference/responses/create)
- [Streaming events — OpenAI API Reference](https://platform.openai.com/docs/api-reference/responses-streaming)
- [Migrate to the Responses API — OpenAI](https://platform.openai.com/docs/guides/migrate-to-responses)
- [Responses overview — developers.openai.com](https://developers.openai.com/api/reference/resources/responses)
- [Better performance from reasoning models using the Responses API — OpenAI Cookbook](https://cookbook.openai.com/examples/responses_api/reasoning_items)
- [Azure OpenAI Responses API — Microsoft Learn (mirror, updated 2026-04-14)](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses)
