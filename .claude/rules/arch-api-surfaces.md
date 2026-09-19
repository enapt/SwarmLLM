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

`inference::take_leading_reasoning_block` removes a leading `<think>…</think>`
in `finalize_reply_text` (the non-streaming choke point), and
`tool_parse::StreamingToolText` withholds the same block while streaming — the
buffer both encoders already share, so all four API paths inherit it.

**`StreamingToolText` has no `Default`, and that is load-bearing.** The buffer
does two jobs — hold text back while it could still be a tool call, AND withhold
the reasoning preamble — and all four streaming surfaces wrapped `push` in
`if tools_requested`, so on an ordinary chat message the filter was never
reached and the whole scratchpad streamed to the user as the answer. The mode is
a REQUIRED constructor argument (`new(detect_tools)`) so no caller can express
"skip the buffer"; `push` always runs the filter and only withholds for tool
detection when asked. The two flush helpers return early with `pending_all()`
when `!detects_tools()` — they must still run, but must not find a call in prose.

**Two implementations of one rule, so the property asserted is that they
AGREE** (`streamed_and_unstreamed_replies_strip_the_same_scratchpad`), **and it
is asserted with `detect_tools` both ways** — the axis the production gate
actually varied. While each was tested only against itself they diverged on the
input neither used: a first chunk that is pure whitespace. In the streaming decision an empty
remainder is the WEAKEST evidence about a `<think>` block, not the strongest —
nothing decisive has been seen yet — and treating it as decisive latched
"no scratchpad here" on zero characters and streamed the whole thing to the
user (report #031). A lone leading space as its own chunk is ordinary: a BPE
tokenizer decodes its word-boundary marker to one.

→ `docs/invariants/api-surfaces.md`

## A reply budget the caller did not choose is a ceiling, not a demand

**`model_worker::resolve_max_new_tokens`** is the single answer to "how many
tokens may this reply use", called from both tokenization sites, and both
**assign what it returns back to `gen.sampling.max_tokens`** so every downstream
reader sees the budget granted, not the one asked for.

An absent `max_tokens` is NOT a value to check against the window — it is our
own fallback, and checking it refused every non-empty prompt on every
2048-context model, TinyLlama included (report #001). Non-explicit is **lowered
to fit, never raised to fill** (vLLM's `max_model_len - prompt_len` is the
verl#5504 over-reservation trap, and raising it would also override MCP's
deliberate 512). An **explicit** budget is honoured exactly or refused with the
number that would fit — never silently shortened.

`SamplingParams.max_tokens_explicit` is a `#[serde(default)]` **bool beside the
existing `u32`, deliberately not `Option<u32>`** — an `Option` serialises `null`,
which an older peer cannot deserialise. No `PROTOCOL_VERSION` bump; compatible
both directions, pinned by two tests.

⚠ `pipeline/distributed.rs` consults no window at all — `docs/FUTURE_WORK.md` #85.

→ `docs/invariants/api-surfaces.md`

## A rendered prompt that lost the question is a FAILED render

`chat_template::render_kept_the_last_question` is a post-condition on
`apply_chat_template`, inside `build_prompt_inner`: a render that does not
contain the last user message's text is discarded, logged, and replaced by the
fallback chain.

Asserting on the FRAME cannot see this. The Qwen3 render test asserted the
prompt ends on `<|im_start|>assistant`, which a prompt that dropped every
message also does, and it was green while every Qwen3 request in the field
arrived with no question in it.

→ `docs/invariants/api-surfaces.md`

## A model is told about its tools the way it was trained to be

**`chat_template::build_prompt` is the ONE place that decides how a model learns
what tools it has** — it needs the tool definitions AND the model's own
template, and nothing else holds both. `template_renders_tools` chooses: a
template that reads `tools` renders them itself; one that never mentions them
gets `describe_tools_in_prose`.

Flattening tools into a system message at the API edge is what left Qwen3's own
`{%- if tools %}` branch unreachable on every request ever made. `tools` is a
REQUIRED parameter of `build_prompt` and `InferenceRequest::local`, and rides on
the request beside `messages` — the router builds its prompt long after the API
surface is gone. The Anthropic surface translates its `input_schema` shape into
the `{"type": "function", "function": {...}}` one templates are written against.

**`tojson` is a minijinja feature (`json`), and an unknown filter fails the
WHOLE render.** Every real tool-rendering template calls it.

**And the `tojson` a template gets is OURS, not minijinja's** —
`chat_template::tojson` implements the signature `transformers` defines
(`ensure_ascii`, `indent`, `separators`, `sort_keys`, Python's separator
defaults) and does not escape HTML. minijinja's builtin does both wrong for this
use: it rewrites `<`, `>`, `&` and `'` for a web page, and it rejects every
keyword but `indent` — which fails the whole render.

**And a schema's keys reach the model in the order its author wrote them.**
`serde_json` and `minijinja` are both built with `preserve_order`; drop either
and every tool schema is alphabetised on its way through `tojson`, which is
what happened on every tool-carrying request until 2026-09-12. Every other
serving stack delivers the author's order, and formatting is not cosmetic to
a small model.

→ `docs/invariants/api-surfaces.md`

## A template that refuses a system role is still told what the system turn said

Gemma and Mistral `raise_exception` on a system turn, which fails the whole
render — and the tool description IS a system message, so every such request
rendered through a FALLBACK instead of the model's own template.
**`chat_template::fold_system_into_first_user`** moves the system text into the
first user turn, as a RETRY after the render has already declined, so a template
that renders a system turn today is untouched.

→ `docs/invariants/api-surfaces.md`

## Chat templates render on minijinja, and its settings are part of the contract

Rendering is `minijinja` + `minijinja-contrib`'s `pycompat` — the engine
HuggingFace's TGI and SGLang use — NOT a subset of our own. A thousand lines of
hand-rolled Jinja were deleted on 2026-09-10 because a subset does not decline
on a template past its edge, it HALF-renders.

Four settings are load-bearing and must not be dropped: `trim_blocks`,
`lstrip_blocks` and `keep_trailing_newline` (what `transformers` renders with,
so a template's own indentation is not part of the prompt), and the `pycompat`
unknown-method callback (templates call `split` / `lstrip` / `startswith`, which
minijinja does not implement natively). `raise_exception` must fail the render;
a bad `strftime_now` specifier must not.

A template arrives inside a downloaded GGUF, so it is untrusted input AND a
program: output size, an instruction budget, and the template source are all
bounded.

→ `docs/invariants/api-surfaces.md`

## A prompt that closed someone else's turn is finished for the model

`chat_template::open_the_models_turn_if_the_prompt_closed_it` runs in
`build_prompt_with_model` — the choke point every prompt passes through — and
appends the family's own generation prompt when the rendered prompt ends on a
turn-CLOSING marker.

→ `docs/invariants/api-surfaces.md`

## What part of a tool-carrying reply is content is decided in one place

**`tool_parse::leading_content`** is the single answer for both non-streaming
surfaces, which each computed `text[..content_prefix_len(text)].trim()`
themselves. It adds the one rule that only makes sense once a call has been
found: **a reasoning block ended by a tool call rather than by `</think>` is
still a reasoning block.** `inference::take_leading_reasoning_block` requires
the closing tag and is right to — without one it cannot know where the
scratchpad stops. Here the call answers that.

→ `docs/invariants/api-surfaces.md`

## A tool-carrying reply streams the part that cannot be a tool call

**`tool_parse::content_prefix_len`** is the single answer to "how much of this
reply so far is certainly ordinary content", and **`tool_parse::StreamingToolText`**
is the buffer all four API paths share — OpenAI streaming and not, Anthropic
streaming and not.

→ `docs/invariants/api-surfaces.md`

## A ticker merged into a response stream is a termination condition

`api::sse::progress_ticker` is the ONE keep-alive/progress ticker for both SSE
encoders. Its wait is cancellable — `tokio::select!` on the interval versus a
`tokio::sync::watch` finish signal, which is why the signal is a `watch` and not
an `AtomicBool`: the ticker has to *wait* on it, not merely read it.

It carries TWO comment lines, built by `api::sse::keep_alive_lines`: prose for a
person watching a terminal, and `swarmllm-status ` + JSON
(`inference::trace::LiveStatus`) for a client that can show what is happening.
**Comments, because this rides `/v1/chat/completions`** — every conforming SSE
reader drops a `:` line, while a `data:` frame carrying a non-chat-completion
object is not safe and an `event:` name is invisible outside `EventSource`.

**A status is DERIVED from what the trace recorded, never cycled on a timer.**
`RequestTrace::live_status` reads the marks — `mark_dequeued`, `mark_assembled`,
`set_progress`, `mark_first_token` — so queued, planning and contacting-nodes
are facts, not plausible-sounding filler. A label that advances whether or not
anything is happening looks like information and cannot be told from a hang,
which is the problem it would be there to solve.

→ `docs/invariants/api-surfaces.md`

→ `docs/invariants/api-surfaces.md`

## Active-Pipeline Guard on Manual Mutations

Anything that removes a shard file or model from a node MUST first
check whether `active_pipelines` references it, and refuse with
`SwarmError::ServiceUnavailable(...)` (mapped to HTTP 503) if so —
yanking a shard file out from under an in-flight token loop surfaces
as `ShardNotFound` mid-stream, which is unrecoverable. The
auto-manage prune path already does this via `active_pipeline_shards`
in `model/auto_manage/prune.rs`. The same guard MUST live in:

- `api/admin_models/shards.rs::delete_shard` — checks
  `seg.shard_id.model_id == mid && seg.shard_id.index == shard_index`.
- `api/admin_models/lifecycle.rs::delete_model` — checks
  `seg.shard_id.model_id == mid`.

New "delete" or "evict-from-disk" admin handlers MUST add the guard
before the destructive operation. Note that `unload_model` /
`unload_shard` (memory-only eviction) are NOT in scope — the worker
will simply re-load on next request.

## API errors must be readable by the caller

Every failure the API can produce has to come back as
`{"error": {"message", "type", "param", "code"}}`. Two ways to break that, both
of which shipped:

- **Using axum's `Json<T>` as a request extractor.** Its rejection is raw text
  with a 422. Nine handlers used the `JsonBody<T>` wrapper and 27 did not, so
  most admin, model and pool endpoints returned something the dashboard could
  not read. `getApiErrorMessage` does `await resp.json()` inside a try/catch, so
  raw text throws, the catch swallows it, and the user gets the generic fallback
  with the real reason discarded — every one of those endpoints could only ever
  say "action failed". Use `JsonBody<T>` in the request-body position.
- **No `.fallback()` on the router.** An unrouted path returned a bare 404 with
  an empty body. `/v1/completions` is the case that matters: OpenAI deprecated it
  but plenty of tooling still calls it, and an empty 404 gives no hint that
  `/v1/chat/completions` exists. `unknown_route` now answers in the envelope and
  names the replacement.

Choose the STATUS from the cause, not from where the error came from.
`probe_failure_is_user_fixable` is the pattern: a mistyped HuggingFace repo is a
404 the caller can act on, while a rate limit or an upstream outage stays a 502.
Reporting a typo as `502 Bad Gateway` says this server is broken about something
in the caller's own input.

## A generation loop that blocks its thread must be told to, and a full buffer is not a departed client

**`inference::executor::without_starving_the_runtime`** wraps every in-process
`executor.generate*` call: that loop never yields, and on a Tokio worker it stops
the runtime draining the response, so a streamed reply does not stream at all.
**`api::sse_send_live_blocking`** is what a generation callback sends with —
`try_send(..).is_ok()` reads a FULL channel as a departed client and ends the
reply at the buffer's capacity, reported as a natural `stop`. Terminal
`finish_reason` events go through it too.

→ `docs/invariants/api-surfaces.md`

## A streaming path must announce that it finished

`api/openai/streaming.rs` treats "no finish event arrived" as "this path never
streamed" and falls back to emitting the whole `InferenceOutput.content` as one
delta. So a coordinator that streams tokens and then returns without sending a
terminal `finish_reason: Some(..)` does not merely omit a marker — it
**duplicates the entire reply**.

→ `docs/invariants/api-surfaces.md`

## A generation that reaches the context window has FINISHED, not failed

`SwarmError::ContextWindowReached { used, window }` is raised at the
`split/executor.rs` pre-flight ONLY for a decode step into an existing
conversation (`window_overflow_is_a_finished_reply(seq_len, index_pos)`), and
`pipeline::distributed::length_finish_or_error` turns it into
`finish_reason: "length"` wherever tokens were produced. A prompt too long to
START keeps its 400 and its wording, which is correct there.

**Both halves of the decode loop and the choke point wrapping the five
alternative coordinators must ask.** The default distributed path is one of
those five, so a fix written only in the standard loop is inert for every node
that holds no model — which is every new user.

→ `docs/invariants/api-surfaces.md`

## A stream that fails must say so (2026-09-01)

`finish_reason` carries only what the OpenAI spec defines — `stop`, `length`,
`tool_calls` — and none of them means "something went wrong". A failure on the
OpenAI streaming surface is `StreamEvent::Error`, typed through
`classify_error`, and no finish delta at all; the Anthropic sibling is
`AnthropicSseEvent::Error`. `a_stream_that_fails_never_pretends_the_model_chose_to_stop`
in `tests/repo_consistency.rs` fails the build on an `Err` arm that produces the
literal for a natural finish within a few lines — **on BOTH surfaces**:
`"stop"` in `api/openai/streaming.rs`, `"end_turn"` in `api/anthropic/{sse,
handlers}.rs`. It scanned only the OpenAI one until 2026-09-14; the Anthropic
sibling was correct by hand-inspection and unguarded, which is how the OpenAI
one came to need the guard in the first place.

→ `docs/invariants/api-surfaces.md`

## Two counters both called "tokens" — write down which event each one counts

`StreamReassembler::truncated()` is the single answer to "did tokens the peer
SENT fail to arrive?". It is deliberately NOT `usage.completion_tokens >
emitted()`.

→ `docs/invariants/api-surfaces.md`

## `PipelineError` is the route planner's signal to ITSELF

It is control flow, not a user-facing error. The capacity-rung walker in
`assemble_pipeline_for` catches each one and tries the next `CapacityBound`;
`greedy_assign` catches its own capacity refusal and re-runs unbounded. Those
producers never reach a caller and are correct as they are.

**A failure raised while EXECUTING is ours, or it gets its own variant** —
never this one. Three execution failures were filed under it ("Pipeline has no
segments", "Pipeline completed without producing a result", "Response channel
dropped") and each inherited its hint: *fetch the model part that is missing*.
None of them is that, so the advice could not help — gotcha #295's family. They
are `Internal` now, which carries no hint by design. Guard:
`an_execution_failure_is_never_the_route_planners_internal_signal`, drawn at the
DIRECTORY because that is where the meaning changes — `scheduler/` plans a
route, `pipeline/` runs one.

⚠ **`ServiceUnavailable` is the wrong home for a failure of OUR OWN machinery**:
`router::remote_peer_could_not_serve` reads it as "a peer could not serve" and
bars that peer from the retry.

**Both ends of the prompt-privacy refusal are named separately** —
`PromptPrivacyUnavailable` (shard 0, the embedding table) and
`PromptPrivacyNeedsFinalShard` (the output head). Same 503 and same
`error_type`, because to a caller it is one situation; two variants because the
two send the reader after DIFFERENT parts, and one hint cannot do that.

→ `docs/invariants/api-surfaces.md`

## Single-source-of-truth helpers — API surfaces, errors and streaming

Each names the ONE place a decision is made. A second implementation of any of
them is this codebase's most-repeated defect — see `.claude/rules/architecture.md`
§ "One invariant, N paths". **Read the topic file before changing one.**

Full evidence: `docs/invariants/api-surfaces.md`

- **`api::metrics::network_traffic_json` — every status payload reports traffic the same way, and there are THREE.** `/api/admin/stats`, the WebSocket `stats_update` tick, and `/v1/status` (which `swarmllm status` reads). One builder is not one surface: the figure reached two of them and the CLI printed nothing, while its formatter passed against a hand-made object (gotcha #569). `every_stats_surface_carries_the_traffic_figure` in `tests/repo_consistency.rs` names all three.

- **`api::mcp::dispatch::spawn_model_call_task`** — the single place that decides whether a fan-out model call actually **answered**, as opposed to merely not erroring.
- **`inference::model_worker::resolve_max_new_tokens`** — the single answer to "how many tokens may this reply use", against the model's real context window. Both tokenization sites call it AND use what it returns. A budget the caller did not name is a ceiling to lower, never a value to check and refuse.
- **`crate::error::reclassify_flattened_error`** — recovers an error's CLASS from a message that crossed a boundary carrying no types.
- **`crate::error::classify_error`** — the single answer to "what is this failure, to a caller": `(StatusCode, client-safe message, error type)`.
- **`crate::error::failure_log_level` + the `log_failure!` macro** — the single answer to "how loudly should this failure be recorded in THIS node's log".
- **`crate::error::error_hint_with_key`** — returns the actionable hint as a stable `(key, english)` pair, from ONE match arm.
- **`AnthropicSseEvent::Error`** — the ONLY way the Anthropic streaming surface reports a failure.
