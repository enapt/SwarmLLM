---
paths:
  - "src/api/**"
  - "src/error.rs"
  - "src/http.rs"
  - "src/inference/chat_template/**"
  - "src/api/tool_parse.rs"
  - "src/cli/**"
---

# API surfaces, errors and streaming

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## A reasoning model's scratchpad is not the reply

`inference::take_leading_reasoning_block` (`finalize_reply_text`) and `tool_parse::StreamingToolText` strip a leading `<think>…</think>` on all four API paths. `tool_parse::ReasoningOut` is a REQUIRED constructor argument, no `Default`: OpenAI `Separate` (`delta.reasoning_content`), Anthropic `Withheld`. Search for tool calls only AFTER a closed scratchpad. `inference::REPLY_LEADING_WHITESPACE` is the one rule for what a reply may not open with. Guard: `streamed_and_unstreamed_replies_strip_the_same_scratchpad` (both `detect_tools` modes).

→ `docs/invariants/api-surfaces.md` § "A reasoning model's scratchpad is not the reply"

## A reply budget the caller did not choose is a ceiling, not a demand

**`model_worker::resolve_max_new_tokens`** is the single answer to "how many tokens may this reply use"; both tokenization sites call it AND assign the result back to `gen.sampling.max_tokens`. An absent `max_tokens` is lowered to fit, never raised or refused; an EXPLICIT one is honoured or refused with the number that fits. `SamplingParams.max_tokens_explicit` is a `#[serde(default)]` bool, not `Option<u32>`.

→ `docs/invariants/api-surfaces.md` § "A reply budget the caller did not choose is a ceiling, not a demand"

## A rendered prompt that lost the question is a FAILED render

`chat_template::render_kept_the_last_question` is a post-condition inside `build_prompt_inner`: a render without the last user message is discarded and replaced by the fallback chain. Assert on CONTENT, never on the frame.

→ `docs/invariants/api-surfaces.md` § "A rendered prompt that lost the question is a FAILED render"

## A model is told about its tools the way it was trained to be

**`chat_template::build_prompt` is the ONE place that decides how a model learns its tools** (`template_renders_tools`, else `describe_tools_in_prose`); `tools` is a REQUIRED parameter, never flattened into a system message at the API edge. `tojson` is OURS (`chat_template::tojson`); `serde_json` and `minijinja` keep `preserve_order`.

→ `docs/invariants/api-surfaces.md` § "A model is told about its tools the way it was trained to be"

## A template that refuses a system role is still told what the system turn said

**`chat_template::fold_system_into_first_user`** retries a render a template declined for a system turn (Gemma, Mistral) with the system text moved into the first user turn; a template that renders a system turn is untouched.

→ `docs/invariants/api-surfaces.md` § "A template that refuses a system role is still told what the system turn said"

## Chat templates render on minijinja, and its settings are part of the contract

`minijinja` + `minijinja-contrib` `pycompat`, never a hand-rolled subset. Keep `trim_blocks`, `lstrip_blocks`, `keep_trailing_newline` and the `pycompat` callback; `raise_exception` fails the render, a bad `strftime_now` does not. A template is untrusted input AND a program: output, instructions and source are bounded.

→ `docs/invariants/api-surfaces.md` § "Chat templates render on minijinja, not on a subset of our own"

## A prompt that closed someone else's turn is finished for the model

`chat_template::open_the_models_turn_if_the_prompt_closed_it` runs in `build_prompt_with_model`, the choke point, and appends the family's generation prompt when the render ends on a turn-CLOSING marker.

→ `docs/invariants/api-surfaces.md` § "A prompt that closed someone else's turn is finished for the model"

## What part of a tool-carrying reply is content is decided in one place

**`tool_parse::leading_content`** is the single answer, for both non-streaming surfaces, to what part of a tool-carrying reply is content.

→ `docs/invariants/api-surfaces.md` § "What part of a tool-carrying reply is content is decided in one place"

## A tool-carrying reply streams the part that cannot be a tool call

**`tool_parse::content_prefix_len`** is the single answer to how much of a reply is certainly content; **`tool_parse::StreamingToolText`** is the buffer all four API paths share.

→ `docs/invariants/api-surfaces.md` § "A tool-carrying reply streams the part that cannot be a tool call"

## A ticker merged into a response stream is a termination condition

`api::sse::progress_ticker` is the ONE keep-alive ticker for both SSE encoders (`watch` finish signal, comments from `api::sse::keep_alive_lines`). A surface passes a REQUIRED `Option<api::sse::IdleKeepAlive>`: a comment keeps a chunk-counting client alive for nothing. A status is DERIVED from `RequestTrace::live_status`, never cycled on a timer.

→ `docs/invariants/api-surfaces.md` § "A ticker merged into a response stream is a termination condition"

## Active-Pipeline Guard on Manual Mutations

Anything that removes a shard file or model MUST first check `active_pipelines` and refuse with `SwarmError::ServiceUnavailable` (503): `api/admin_models/shards.rs::delete_shard`, `api/admin_models/lifecycle.rs::delete_model`, auto-manage prune. A new delete handler adds it; memory-only unloads are out of scope.

→ `docs/invariants/api-surfaces.md` § "Active-Pipeline Guard on Manual Mutations"

## API errors must be readable by the caller

Every failure is `{"error": {"message", "type", "param", "code"}}`: take bodies with `JsonBody<T>`, never axum's `Json<T>`; keep the router `.fallback()` (`unknown_route`). Choose the STATUS from the cause (`probe_failure_is_user_fixable`).

→ `docs/invariants/api-surfaces.md` § "API errors must be readable by the caller"

## A generation loop that blocks its thread must be told to, and a full buffer is not a departed client

**`inference::executor::without_starving_the_runtime`** wraps every in-process `executor.generate*`; **`api::sse_send_live_blocking`** is what a generation callback sends with — `try_send(..).is_ok()` reads a FULL channel as a departed client.

→ `docs/invariants/api-surfaces.md` § "A generation loop that blocks its thread must be told to, and a full buffer is not a departed client"

## A streaming path must announce that it finished

`api/openai/streaming.rs` reads "no finish event" as "never streamed" and re-emits the whole `InferenceOutput.content`, so a coordinator that streams and returns without a terminal `finish_reason: Some(..)` DUPLICATES the reply.

→ `docs/invariants/api-surfaces.md` § "A streaming path must announce that it finished"

## A generation that reaches the context window has FINISHED, not failed

`SwarmError::ContextWindowReached` (pre-flight in `split/executor.rs`, decode steps only: `window_overflow_is_a_finished_reply`) becomes `finish_reason: "length"` via `pipeline::distributed::length_finish_or_error`; a prompt too long to START keeps its 400. Both decode-loop halves AND the choke point over the five alternative coordinators must ask.

→ `docs/invariants/api-surfaces.md` § "A generation that reaches the context window has finished, not failed"

## A stream that fails must say so (2026-09-01)

A failure is `StreamEvent::Error` typed through `classify_error` with no finish delta (Anthropic: `AnthropicSseEvent::Error`) — never a natural `stop` / `end_turn`. Guard: `a_stream_that_fails_never_pretends_the_model_chose_to_stop`, on BOTH surfaces.

→ `docs/invariants/api-surfaces.md` § "A stream that fails must say so (2026-09-01)"

## Two counters both called "tokens" — write down which event each one counts

`StreamReassembler::truncated()` is the single answer to "did tokens the peer SENT fail to arrive?" — NOT `usage.completion_tokens > emitted()`.

→ `docs/invariants/api-surfaces.md` § "Two counters both called "tokens" — write down which event each one counts"

## `PipelineError` is the route planner's signal to ITSELF

`PipelineError` is route-planner control flow, never a user-facing error; a failure raised while EXECUTING is `Internal` or gets its own variant. Guard: `an_execution_failure_is_never_the_route_planners_internal_signal`. ⚠ `ServiceUnavailable` is wrong for a failure of OUR OWN machinery (`router::remote_peer_could_not_serve` bars the peer). `PromptPrivacyUnavailable` and `PromptPrivacyNeedsFinalShard` share a 503 and `error_type` but stay two variants.

→ `docs/invariants/api-surfaces.md` § "`PipelineError` is the route planner's signal to ITSELF"

## `auto` names a model that can be SERVED, and keeps naming it (2026-09-26)

**`api::openai::resolver::resolve_auto` is the single answer to "which model is `auto`"**, for both surfaces. Never read `loaded_model_info` for it (#120).

→ `docs/invariants/api-surfaces.md` § "`auto` names a model that can be served, and keeps naming it (2026-09-26, #120)"

## Single-source-of-truth helpers — API surfaces, errors and streaming

A second implementation of any of these is this codebase's most-repeated defect (`architecture.md` § "One invariant, N paths").

- **`api::metrics::network_traffic_json`** — traffic in every status payload (THREE surfaces; `every_stats_surface_carries_the_traffic_figure`).
- **`api::mcp::dispatch::spawn_model_call_task`** — whether a fan-out call **answered**.
- **`inference::model_worker::resolve_max_new_tokens`** — a reply's token budget.
- **`crate::error::reclassify_flattened_error`** — an error's CLASS after a boundary.
- **`crate::error::classify_error`** — `(StatusCode, message, error type)`.
- **`crate::error::failure_log_level` + `log_failure!`** — log loudness.
- **`crate::error::error_hint_with_key`** — the hint as `(key, english)`.
- **`AnthropicSseEvent::Error`** — the ONLY Anthropic streaming failure.

→ `docs/invariants/api-surfaces.md` § "Single-source-of-truth helpers — API surfaces, errors and streaming"
