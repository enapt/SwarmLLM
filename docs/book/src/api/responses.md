# Responses API

OpenAI's `/v1/responses` is the default API for o-series and gpt-5-series
models in 2026 and the replacement for the sunsetting Assistants API
(2026-08-26). SwarmLLM exposes the full v1 surface plus follow-on
features such as resumable streams, async background runs, MCP tool
integration, and conversation chaining via `previous_response_id`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/responses` | Create a response (streaming or not, foreground or background) |
| `GET` | `/v1/responses/{id}` | Fetch a stored response. With `?stream=true&starting_after=N`, resume the SSE stream from event `N` (V5). |
| `DELETE` | `/v1/responses/{id}` | Delete a stored response. |
| `POST` | `/v1/responses/{id}/cancel` | Cancel a background response (M9). The cancel flag is checked at completion time; per-token interruption is deferred. |
| `GET` | `/v1/responses/{id}/input_items` | Paginated input-item listing (V4) for chained `previous_response_id` flows. |
| `GET` | `/api/admin/responses` | Admin: list all stored response records (used by the dashboard). |

All endpoints accept the same Bearer-auth header as the rest of the API.

## Routing

`POST /v1/responses` picks one of three execution paths in this order:

1. **Cloud proxy** — when the requested `model` resolves to an OpenAI-routed
   provider, the request is serialized verbatim and forwarded to the
   upstream `/v1/responses` endpoint. Built-in tools, streaming, background,
   reasoning effort, `text.verbosity`, `include[]`, `previous_response_id`,
   and any future field round-trip via `#[serde(flatten)]` extras.
2. **Anthropic-Messages bridge (V3)** — when the model resolves to an
   Anthropic provider (or the local `claude-subscription` subprocess), the
   Responses request is translated to an Anthropic Messages request,
   forwarded, and translated back. This lets Claude Code clients drive
   `/v1/responses` end-to-end without losing tool-call or streaming
   semantics.
3. **Local inference** — translates to `/v1/chat/completions` and runs on
   the local model. Function tools and `tool_choice` translate through;
   built-in tools (`web_search`, `file_search`, `computer_use_preview`,
   `code_interpreter`, `image_generation`, `mcp`, `custom`) are rejected
   with HTTP 400 because they require backing infrastructure SwarmLLM
   does not run.

## Capabilities

- **Multimodal input (V2)** — `input_image` and `input_file` (UTF-8 only)
  parts in the structured `input` array. Binary file payloads (PDF, docx,
  image bytes via `file_data`) are rejected with a clear hint pointing at
  `input_image`.
- **Function tools** — `tools` definitions and `tool_choice` translate to
  OpenAI Chat Completions tool semantics; assistant `tool_calls` map back
  to `function_call` output items.
- **Streaming SSE (M6 + V1)** — `stream=true` emits the full Responses
  event sequence (`response.created` → `response.in_progress` →
  `response.output_item.added` → `response.content_part.added` →
  per-delta `response.output_text.delta` → `response.output_text.done` →
  `response.content_part.done` → `response.output_item.done` →
  `response.completed`). The V1 fix shipped in 2026-04-25 cuts first-token
  latency by emitting `created` and `in_progress` before model warmup
  instead of after.
- **Persistence (M7)** — `store=true` (the OpenAI default) writes the
  full response object to redb with a 30-day TTL. `previous_response_id`
  (M8) chains follow-up requests by prepending the prior turn's messages
  before the new input.
- **Background mode (M9 + V8)** — `background=true` returns HTTP 202 with
  a `Location: /v1/responses/{id}` header; the client polls or, with
  `background=true && stream=true`, opens a resumable SSE connection at
  `GET /v1/responses/{id}?stream=true` that replays buffered events and
  then tails the live producer.

  A background run that fails stores `status: "failed"` with an `error.code`
  describing **why**, not merely that something went wrong — the same code the
  foreground request would have returned for the identical input
  (`not_found_error` for a model that does not exist, `invalid_request_error`
  for a prompt longer than the context, `upstream_error` for a provider
  failure, `server_error` for an actual fault). Since a background caller never
  sees a status code, that field is the only thing distinguishing "fix the
  request" from "retry later".

## Validation (ingress)

The handler runs `validate_responses_ingress` BEFORE any routing decision
so the cloud-proxy and Anthropic-bridge paths can't forward attacker-sized
strings to upstream providers (where they'd burn quota or land in log
lines). Caps:

| Field | Limit |
|---|---|
| `model` | 1..=256 chars |
| `previous_response_id` | ≤64 ASCII alphanumeric (`_` / `-` allowed); generation format is `resp_<32-hex>` |
| `instructions` | ≤2 MB |
| `user` | ≤256 chars |
| `truncation`, `service_tier` | ≤64 chars each |
| `metadata` | ≤64 KB total (keys + values) |

Stop / temperature / top_p / max_tokens are clamped or validated at the
sampling-params layer.

## Dashboard

The admin dashboard exposes a Responses tab (`/admin/responses`) backed by
`GET /api/admin/responses`. It shows the most-recent stored response
records with status, model, input snippet, and per-record cancel/delete
actions.

## Deferred

- `POST /v1/responses/compact` (V9) — no concrete caller has asked for it.
- Token-level cancel for background inference — current cancel flips a
  flag checked at completion time; per-token interruption needs hooks in
  `chat_completions` that are out of v2 plan scope.
- Server-side `conversation` resource CRUD — OpenAI's `conversation`
  parameter forwards through cloud proxy verbatim today; a local
  conversation type with its own endpoints is a separate design.
- Built-in tools on the local path — see "Local inference" above.
- `custom` tools with Lark / regex grammars — rejected on local, forwarded
  on cloud. Local grammar-constrained generation is a candle-side project.
- Audio input on `/v1/responses` — `input_audio` returns 400; needs a
  Whisper-class transcription model SwarmLLM doesn't currently expose.
- Binary file inputs in `input_file{file_data}` — UTF-8 only; PDF/docx/
  image-bytes payloads are rejected with a clear hint pointing at
  `input_image` (for images) or server-side text extraction.
