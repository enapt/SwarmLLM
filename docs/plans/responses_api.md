# OpenAI `/v1/responses` — Design Plan

**Status**: research + scoping. No implementation landed yet.
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

## References

- [Create a model response — OpenAI API Reference](https://platform.openai.com/docs/api-reference/responses/create)
- [Streaming events — OpenAI API Reference](https://platform.openai.com/docs/api-reference/responses-streaming)
- [Migrate to the Responses API — OpenAI](https://platform.openai.com/docs/guides/migrate-to-responses)
- [Responses overview — developers.openai.com](https://developers.openai.com/api/reference/resources/responses)
- [Better performance from reasoning models using the Responses API — OpenAI Cookbook](https://cookbook.openai.com/examples/responses_api/reasoning_items)
- [Azure OpenAI Responses API — Microsoft Learn (mirror, updated 2026-04-14)](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses)
