# API surfaces, errors and streaming

The evidence behind the rules in `.claude/rules/architecture.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## A reasoning model's scratchpad is not the reply

`inference::take_leading_reasoning_block` removes a leading `<think>…</think>`
in `finalize_reply_text` (the non-streaming choke point), and
`tool_parse::StreamingToolText` withholds the same block while streaming — the
buffer both encoders already share, so all four API paths inherit it.

**Why both.** A reply's content must not depend on whether the caller asked for
`stream`; a fork on a presentation flag is a fork in policy. Fixing only the
finaliser would mean the same question answered by a reasoning model returns the
scratchpad when streamed and the answer when not.

**Why it is removed at all.** Qwen3, QwQ and DeepSeek-R1 write their working out
first and the answer after it. Every serving stack separates the two — vLLM
extracts it with a `--reasoning-parser` into `reasoning_content`, llama.cpp
defaults Qwen3.5 to non-thinking so it is never produced — and we did neither,
so someone asking "Say OK" got several hundred tokens of deliberation with the
answer at the end. Same class as gotcha #169: content that is not the reply
reaching the user as the reply.

Four things a change here must keep:

- **Only at the START, and only CLOSED.** A `<think>` later in the text is the
  model writing about thinking. An unterminated one means the reply was cut off
  mid-reasoning, so the whole thing is scratchpad — the non-streaming path
  leaves it rather than returning an empty message the caller cannot tell from a
  real blank answer, and the streaming path releases it through `pending_all`.
- **The streaming side waits for the answer to START.** The blank line between
  scratchpad and answer arrives in a LATER token than `</think>`, so deciding on
  first sight of the closer emits it and the reply opens with a stray newline.
- **The silence is covered.** `sse::progress_ticker` is already merged into both
  encoders precisely so a slow reply does not look dead — which is what makes
  withholding during reasoning safe rather than a hang.
- **The reasoning is DISCARDED, not surfaced.** Exposing it wants a field on
  both surfaces (`reasoning_content` on OpenAI, a thinking block on Anthropic)
  and is worth doing; returning it glued to the answer is not a smaller version
  of that, it is the bug.

## A prompt that closed someone else's turn is finished for the model

`chat_template::open_the_models_turn_if_the_prompt_closed_it` runs in
`build_prompt_with_model` — the choke point every prompt passes through — and
appends the family's own generation prompt when the rendered prompt ends on a
turn-CLOSING marker.

**What it replaced.** The condition was already DETECTED there, named precisely,
logged at WARN, and then sent anyway. Its comment justified that as "being wrong
about this must never cost someone an answer", which is the right instinct about
warning-versus-erroring and skips the third option: the outcome is not in doubt.
A model shown a finished conversation ends its turn at once — one token,
`finish_reason: "stop"` — so proceeding costs the answer just as surely, while
explaining it.

**The evidence is positive, which is what makes repair safe.** The prompt ENDS
with one of the six `TURN_ENDING_MARKERS`; that is the presence of a closer, not
the absence of a recognised opener, so a correct-but-unusual prompt cannot be
mistaken for a broken one.

**Four of the six name one family; two do not.** `<|im_end|>`, `<|eot_id|>`,
`<|end|>` and `<end_of_turn>` each map to exactly one opener, and appending it
produces the same string that family's own template would have. `</s>` is
Llama-2, Mistral AND vicuna, and `<|endoftext|>` spans unrelated vocabularies —
those keep the warning and nothing else, because a confidently wrong opener is
worse than a diagnosable prompt. Reaching ChatML for a non-ChatML model is the
failure that put stray `<|im_end|>` in Llama-3 replies for several releases
(gotcha #169); this must never become "fall back to ChatML".

Reported against a Qwen3-8B (2026-09-05). At the time this renderer was a
hand-rolled Jinja subset and DECLINED that template — `namespace()`, a reversed
slice `messages[::-1]`, `loop.index0`, `tojson` and string methods were all past
its edge — so every request rendered a prompt ending on `<|im_end|>`. Pinned by
`a_prompt_left_on_a_closed_turn_gets_the_models_turn_opened`, with the official
template kept as a fixture.

**Rendering moved to minijinja on 2026-09-10 and Qwen3 now renders natively**,
so this particular template no longer reaches the repair. The repair stays: it
guards every template that ends a conversation without opening the model's turn,
which is a property of the template rather than of the engine.

**A test that fixtures a turn-closing prompt must now close on `</s>`**, or the
repair fixes it and the test asserts nothing — which is what
`both_prompt_entry_points_go_through_the_same_renderer` caught about itself, via
the guard it already carried.

## A tool-carrying reply streams the part that cannot be a tool call

**`tool_parse::content_prefix_len`** is the single answer to "how much of this
reply so far is certainly ordinary content", and **`tool_parse::StreamingToolText`**
is the buffer all four API paths share — OpenAI streaming and not, Anthropic
streaming and not.

**What it replaced.** A local model can only express a tool call as text, so
`parse_tool_calls` needs the whole reply to recognise one, and both streaming
encoders therefore appended every token to a `String` and flushed once at the
end. Measured on llama-3.2-3b, identical prompt: **120 content deltas without
`tools`, 1 with them.** Every agentic client sends `tools`, so every agentic
client got a single lump after a generation that can run for minutes — which is
indistinguishable from a hang and is what a client timeout actually fires on.
Reported as OpenClaw "takes ages and times out".

**Why the obvious guard is wrong, and what makes this one right.** "If it does
not start with `{`, stream freely" fails because `parse_tool_calls`
deliberately finds a call EMBEDDED after prose, and the non-streaming path used
to DISCARD that prose. Streaming it first would have made the two surfaces
disagree about the same reply.

The resolution is not a cleverer guard, it is **matching the reference
implementation**: vLLM streams text before the marker as content, and its
non-streaming path keeps that text too (`content = model_output[:start]`, null
only when empty). `content` and `tool_calls` coexist in one OpenAI message, and
in Anthropic a text block precedes the `tool_use` blocks. So the non-streaming
paths now keep the preamble as well, and the surfaces agree by being the shape
clients already expect. Discarding it was throwing away text the model
produced.

Four things a change here must keep:

- **A bare `{` is a marker.** `try_generic` and `try_llama3` accept an object
  with no marker at all, so an unadorned brace begins a possible call. The bare
  word `tool_call` deliberately is NOT one — every parser keying on it also
  needs an object, so the brace is reached first, and leaving it out lets prose
  that merely mentions tool calls keep streaming.
- **Hold back a suffix that could be half a marker.** A marker arrives a token
  at a time, so `<tool` must not go out. `partial_marker_overlap` withholds
  exactly the ambiguous tail and no more; vLLM calls the same thing
  `partial_tag_overlap`. Never emit a cut that lands inside a character.
- **The emitter owns BOTH outcomes.** `emit_openai_tool_calls` /
  `emit_anthropic_tool_blocks` take the buffer by `&mut` and flush either the
  remaining prose (call found) or the whole remainder (no call). Splitting that
  across the caller is how one of them gets forgotten — the shape this codebase
  keeps being caught by.
- **Nothing emitted twice, nothing lost.** `emitted + pending == text` is the
  invariant every caller depends on, pinned by
  `nothing_is_ever_emitted_twice_or_lost` over six reply shapes.

Verified end to end against the released binary as the control, same model and
prompt: 1 delta → 99 (OpenAI), 1 → 55 (Anthropic), tool calls still emitted and
no marker character leaked.

## A ticker merged into a response stream is a termination condition

`api::sse::progress_ticker` is the ONE keep-alive/progress ticker for both SSE
encoders. Its wait is cancellable — `tokio::select!` on the interval versus a
`tokio::sync::watch` finish signal, which is why the signal is a `watch` and not
an `AtomicBool`: the ticker has to *wait* on it, not merely read it.

**Why it is shared.** OpenAI and Anthropic each had a byte-identical copy, and
both carried the same defect: sleep the whole interval, THEN check whether the
response had finished. `Stream::merge` ends only when both halves end, so every
streamed reply stayed open for the remainder of that sleep — measured, an
8-token reply delivered in 0.5 s held its connection to 15.0 s, while the same
request answered non-streaming in 0.56 s (gotcha #390). Clients that stop at
`[DONE]` never noticed; anything reading to end-of-stream waited, and the server
held a task and a connection per stream either way.

The comment above the old copy said *"the ticker MUST terminate"* and was right
about the hazard it had in mind — an unbounded ticker holds the response open
for ever. **Terminating late is the same bug with a bound on it.**

Three things a change here must keep:

- **A dropped sender ends the ticker.** The token stream is gone; there is
  nothing left to keep alive. Without this, `changed()` returning `Err` in a
  loop that ignored it would spin.
- **An unfinished response still gets keep-alives**, or a slow request looks dead
  to the client and to any intermediary. That is the hazard the original comment
  was written about and it is still covered by a test.
- **The interval is a `Duration`, not a count of seconds.** That is what lets a
  test drive it at millisecond scale and assert exactly, with no `tokio`
  `test-util` dev-dependency: an hour-long interval means a ticker that waits it
  out cannot possibly answer inside the timeout.

**A note on the observed period.** Keep-alive comments arrive at alternating
gaps of 12.52 s and 15.00 s, not a flat 15 s, and end-of-stream used to land on
whichever boundary came next — which is why the pre-fix measurements clustered at
those same two values and why they looked inexplicable until the ticker itself
was timed. The likely cause is two sources beating against each other: the merged
ticker on its own interval, and axum's `KeepAlive`, which resets on every write.
That explanation is *unverified* — it fits the numbers and nothing depends on it,
since the hold it produced is gone.

Ask of any periodic task merged into a response stream: when the thing it is
keeping alive finishes, how long until this notices?

## A streaming path must announce that it finished

`api/openai/streaming.rs` treats "no finish event arrived" as "this path never
streamed" and falls back to emitting the whole `InferenceOutput.content` as one
delta. So a coordinator that streams tokens and then returns without sending a
terminal `finish_reason: Some(..)` does not merely omit a marker — it
**duplicates the entire reply**.

`ngram_only_spec.rs` was the only coordinator missing it;
`distributed.rs`, `remote_generate.rs`, `dsd.rs` and `speculative.rs` all had
it, which is why nothing else showed the fault. Measured on the released
v0.3.135: "Count 1 to 3, digits only" came back as `1\n2\n3<|eot_id|>1\n2\n3`
from every peer-held model on default settings (gotcha #414).

The same file also streamed the EOS token as reply text at two sites, while
`finish_speculative` filters it out of the non-streaming content — so one reply
differed by transport. **End-of-turn is a control token: it ends the reply, it
is not part of it.** Keep it in the accumulator so the loop still stops on it,
and exclude it from what is streamed.

`a_streaming_pipeline_path_sends_its_terminal_finish_event` in
`tests/repo_consistency.rs` fails the build on a streaming coordinator with no
terminal send. `pipeline/mod.rs` is excluded because it holds the shared emit
helpers — ending the stream is the coordinator's job, since only it knows why
generation stopped.

Ask of any "did this happen?" flag what its consumer does when it stays false.

## A stream that fails must say so (2026-09-01)

`finish_reason` carries only what the OpenAI spec defines — `stop`, `length`,
`tool_calls` — and none of them means "something went wrong". A failure on the
OpenAI streaming surface is `StreamEvent::Error`, typed through
`classify_error`, and no finish delta at all; the Anthropic sibling is
`AnthropicSseEvent::Error`. `a_stream_that_fails_never_pretends_the_model_chose_to_stop`
in `tests/repo_consistency.rs` fails the build on an `Err` arm that produces the
literal `"stop"` within a few lines.

**Why**: measured on the released v0.3.145 (gotcha #433), a streamed request for
a model no node held answered `200`, an empty assistant delta, `finish_reason:
"stop"`, `[DONE]` — while the identical request without `stream` answered 503
with the hint naming the cause. The streaming branch of the no-coverage case in
`api/openai/mod.rs` went to the legacy in-process `stream_response` (from before
the router could stream), whose error arm mapped every failure to `"stop"` with
a comment asserting the caller surfaced errors another way. Nothing did.

Two rules follow. **Two branches that differ only by `stream` must reach the
same decision-maker** — the router owns the in-process executor AND the
distributed pipeline, streaming or not, so both forms of a request go through
it and are refused by it identically; a fork on a presentation flag is a fork in
policy. And **a peer's "I do not hold it" is a stale claim at whichever hop it
arrives**: `dispatch::remote_generate::REMOTE_GENERATE_NOT_HOSTED` is the fast
path's refusal, named so `remote_error_means_missing_shard` can match it and
retract, blacklist and retry — the same handling a mid-pipeline missing-shard
error has had since July. Unmatched, one honest refusal failed a request four
other peers could have served.

## Two counters both called "tokens" — write down which event each one counts

`StreamReassembler::truncated()` is the single answer to "did tokens the peer
SENT fail to arrive?". It is deliberately NOT `usage.completion_tokens >
emitted()`.

`decode_token` accumulates bytes in a `carry` buffer and returns EMPTY until a
multi-byte codepoint completes, and the serving node skips forwarding an
empty-text event without numbering it. So `streamed_count` — what the done token
carries and what `token_id` densely numbers — counts NETWORK SENDS, while
`usage.completion_tokens` counts MODEL STEPS. They are equal only for pure
ASCII, which is what everything got tested with.

Comparing the two therefore refused correct replies: on the released v0.3.135,
"one sentence in Chinese" answered `503 Reply truncated in transit: 1 of 3
tokens arrived` and five emoji answered `9 of 12`, both having arrived complete,
with a hint telling the user to try a different machine (gotcha #416). The
product ships 21 locales and most are multi-byte.

Three properties a change here must keep. **`missing()` is derived from the done
token**, which is the only figure comparable to what arrived. **An unsequenced
peer is never judged truncated** — it cannot say what to expect, the same
degradation `is_complete` makes for mixed-version swarms. And **a genuine loss
must still be caught**: `a_genuinely_lost_token_is_still_truncated` is the
control, because "never report truncation" would silently reintroduce the
truncated replies of gotcha #282.

The clamp of `completion_tokens` down to `delivered` now happens only on a real
truncation, so usage stops under-reporting every multi-byte reply.

## `api::mcp::dispatch::spawn_model_call_task`

(2026-08-10) — the single place
that decides whether a fan-out model call actually **answered**, as opposed to
merely not erroring. Every real model call in `compare` / `research` /
`batch_prompts` passes through it, so it stamps `"empty": true` onto any result
whose call succeeded with blank text, and `count_answered` downstream only reads
that flag. Do NOT re-derive blankness by inspecting the collected JSON.
**The three tools deliberately name the answer field differently** — `content`
for compare and batch, `response` for research — so a downstream check has to
know every one of those names and silently mis-reports the moment a fourth tool
picks a new one. That is not hypothetical: the first cut of this fix did exactly
that, checked `content` only, and would have flagged every successful research
answer as blank (gotcha #291). The verdict belongs where the text is, before any
tool names it. A new fan-out tool inherits the flag with no author action.
Note `status` is deliberately NOT changed for a blank answer — clients already
branch on `"ok"`, and reclassifying a success to fix a reporting gap would break
them.

## `crate::error::reclassify_flattened_error`

(2026-08-12) — recovers an
error's CLASS from a message that crossed a boundary carrying no types.
`SwarmError` survives neither the worker IPC hop nor the network hop; both
deliver a `String`, and whatever is left is re-wrapped as `Inference` → HTTP
500. Call it at any such boundary before falling back to `Inference`.
Two boundaries had the identical problem and only the worker one had a
remedy (three private helpers in `process_pool.rs`, now folded into this).
A prompt too long for a peer-held model answered `500 server_error` carrying
the words "Validation error", while the same request on a local model
answered `400 invalid_request_error` — so whose fault a mistake was depended
on which machine held the model (gotcha #304). It also mis-attributed blame:
`failure_is_penalty_worthy` exempts `Validation` but never saw one, so the
peer was docked for the caller's mistake. **Matching on prose is #295's trap
and this is the exception** — the markers are `SwarmError`'s own
`#[error(...)]` Display prefixes, i.e. part of the type, not wording written
for a human that gets rewritten. Adding a variant means adding its marker
here; nothing else may re-derive a class from a message.

## `crate::error::classify_error`

(2026-08-12) — the single answer to "what is
this failure, to a caller": `(StatusCode, client-safe message, error type)`.
`ApiError::into_response` is one caller; the SSE encoders are the others.
**Never choose an error type at a call site.** It used to live inside
`into_response`, so streaming could not reach it and both encoders hardcoded
one: the same over-long prompt was a `400 invalid_request_error` when the
client did not stream and a `"server_error"` inside a `200` when it did — the
user's own mistake reported as this server breaking, and monitoring told this
node has a bug (gotcha #301). Classify where the typed error still exists:
`StreamFailure::from_error` does it at the site that previously discarded it
with `e.to_string()`. Do NOT re-derive a type by matching on the message —
that is #295's substring-matching-prose trap, and the wording is what changes.
`a_streamed_error_names_the_same_failure_as_its_non_streaming_sibling` fails
the build on a new literal.

## `crate::error::failure_log_level` + the `log_failure!` macro

(2026-08-17) —
the single answer to "how loudly should this failure be recorded in THIS
node's log". **Never pick `error!` vs `warn!` at a site that logs a
`SwarmError`.** The level is derived from the status `classify_error` already
had to choose, because that status IS the answer to whose mistake it was: 4xx
→ Info, `501` → Info, other 5xx → Warn, 500 → Error. A new variant therefore
inherits a sensible level with no second decision to forget.
**Why it exists**: an over-long prompt produced three `ERROR` lines when the
model happened to be peer-held and one `WARN` when it was local — the same
user mistake at a different severity, decided by which machine held the model
— and a `501` for embeddings (deliberate, documented, answered with what to
use instead) logged `ERROR Server error`. `ERROR` means "this node is broken",
so the product was reporting its users' typos as its own faults (gotcha #316).
This is the logging-layer survivor of #300-#305: the HTTP surface had already
been taught to classify, and every site that *logged* still hardcoded a level.
**`classify_error` is pure and must stay pure** — it used to `tracing::error!`
from its catch-all, which meant merely *asking* it what level to use emitted
an ERROR of its own (gotcha #315). The full error is logged by whoever reports
the failure, from the original `SwarmError` rather than the genericised
message. Pin that behaviourally by counting emitted events, not by scanning
source for `tracing::` — the first attempt did the latter and tripped over the
comment explaining the removal.

## `crate::error::error_hint_with_key`

(2026-08-17) — returns the actionable
hint as a stable `(key, english)` pair, from ONE match arm. `error_hint` is a
thin view over it. The envelope carries `hint_key` beside the unchanged
English `hint`, and the dashboard looks up `error_hint.<key>`, falling back to
the English it was sent so nothing can ever render as a raw key name.
A separate `error_hint_key` function would be a second decision to keep in
step — this codebase's most-repeated defect — so they cannot drift here.
Adding a hint means adding its translation in all 21 locales;
`every_backend_hint_key_has_a_translation` fails the build both ways (a key
with no entry, and an entry no variant can emit).

## `AnthropicSseEvent::Error`

(2026-08-12) — the ONLY way the Anthropic
streaming surface reports a failure. Emit `event: error`; never write the
reason into assistant content, and never invent a `stop_reason` for it.
Before it existed this surface could not say "that went wrong" at all, so each
path improvised: the router arm reported every failure as `stop_reason:
"end_turn"` with an empty body (a `PromptPrivacyUnavailable` refusal — the
thing #295 exists to explain — reached the client as the model choosing to say
nothing), and the split path wrote `[inference failed: …]` into the message,
where a client cannot tell it from a real reply and it persists as an
assistant turn, alongside `stop_reason: "error"`, which the API does not
define (gotcha #300). Three invariants a new caller must keep: the frame is
**terminal** (`build_anthropic_sse_response` ends its keepalive ticker on it,
as it does on `message_stop` — a terminal frame that does not stop the ticker
hangs the connection); close any open content block first; and translate the
type through `anthropic_error_type`, because our canonical types are
OpenAI-flavoured and Anthropic clients match on Anthropic's own set (#302).

## A rendered prompt that lost the question is a FAILED render

**Rule:** `.claude/rules/architecture.md` § "A rendered prompt that lost the
question is a FAILED render".

### What it replaced

`build_prompt_inner` took any `Some(..)` from `apply_chat_template` as success.
The renderer was then a hand-rolled Jinja subset, and the official Qwen3
template uses `messages[::-1]`, `namespace()`, `loop.index0`/`first`/`last`,
`tojson`, and `startswith`/`split`/`rstrip`. Given that template it did not
decline — it produced:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>assistant
```

Every user message gone. The model received a well-formed request to answer
nothing and answered something else, fluently, which is why it was field-reported
as a broken forward pass ("RoPE, KV heads, or rotation dimensions") rather than
a broken prompt.

Measured here on `Qwen/Qwen3-1.7B-GGUF`: three prompts of 3, 6 and ~540 words
all reported `prompt_tokens=14` and returned byte-identical replies. Control
`llama-3.2-3b-instruct-q4-k-m` on the same node reported 42 and answered
correctly. After the fix: 25 / 31 / 42, every reply on topic.

### Why the existing test could not see it

`the_official_qwen3_template_opens_the_assistants_turn` asserts the render ends
on `<|im_start|>assistant`. A render that dropped every message ends that way
too. **A test on the frame cannot see the content going missing** — which is
the general lesson, not a Qwen3 one.

### What a change must keep

- **The check is a post-condition on the render, not a template allowlist.** Any
  template the engine cannot fully run falls back loudly rather than silently
  dropping the conversation. This still matters with a real engine behind it:
  the guard is about the OUTCOME, not about which constructs are implemented.
- **Falling back is the safe outcome.** The fallback chain (gemma → model-name →
  ChatML) carries the question and the turn markers. For Qwen3 it reaches
  ChatML, which is the format Qwen3 actually uses.
- **Only the LAST user message is required to survive.** Templates legitimately
  transform or truncate history; none legitimately drops the question being
  asked. An empty or absent user message passes, because there is nothing to
  check.
- **Working templates must still render natively.**
  `the_official_llama3_template_renders_exactly_as_jinja2_does` is the guard on
  that, and it passes unchanged.

### Resolved 2026-09-10 — the engine underneath was replaced

Qwen3 no longer falls back: both revisions of its template render natively. See
"Chat templates render on minijinja" below for what changed and what a change
must keep. This guard is unchanged and still load-bearing — it is what makes a
future template the engine cannot run fail safely instead of silently.

## Chat templates render on minijinja, not on a subset of our own

**Rule:** `.claude/rules/architecture.md` § "A rendered prompt that lost the
question is a FAILED render" — the post-condition above sits on top of this.

### What it replaced

About a thousand lines of hand-rolled Jinja (`chat_template/parser.rs` +
`eval.rs`, both deleted 2026-09-10). A subset is not the wrong idea — llama.cpp,
Jan and GPT4All all use `minja`, a C++ subset written for exactly this — but
**ours failed in the worst available way**: a template past its edge did not
decline, it HALF-rendered, emitting a well-formed prompt with every user message
dropped.

The decision was researched rather than assumed (diagnosis rule 0), and the
research is what settled it: **HuggingFace's own Rust inference server (TGI) and
SGLang both use `minijinja`**, and the `pycompat` shim TGI needs was contributed
by minijinja's author. A spike proved it rendered both Qwen3 revisions —
including the one an actual GGUF ships, which our subset could not — for three
new crates, everything else already being in the tree.

### What a change must keep

- **The environment must match what model authors write against.**
  `transformers` renders chat templates with
  `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` and
  `keep_trailing_newline`, so a template's own indentation and the newline after
  a block tag are NOT part of the prompt. Rendering with Jinja's defaults puts
  the author's layout into the text the model reads. Invisible on a template
  that marks every block `{%-` (Qwen3 does, which is why it agreed either way);
  the whole difference on one that does not.
- **`pycompat` is required, not optional.** Chat templates call Python string
  methods — `split`, `lstrip`, `startswith`. minijinja implements no Python
  methods natively; the shim is what makes real templates work.
- **Undefined must be falsy, not an error.** Templates guard optional fields
  (`message.reasoning_content`) and probe for callables with `is defined`.
- **`raise_exception` must fail the render.** It is how Gemma and Mistral
  templates say they cannot accept a message; the caller then falls back. The
  old engine skipped it silently and rendered a turn those models were never
  trained on.
- **A bad `strftime_now` specifier must not fail the render.** `chrono`'s
  `Display` PANICS on an unknown specifier so it has to be caught, but the
  format string comes from model metadata and one bad `%Q` should not discard an
  otherwise fine prompt. It yields an empty string.
- **The three amplification bounds stay.** A template arrives inside a
  downloaded GGUF — untrusted input, and a program. R101 capped rendered output
  at 4 MiB because a recursion cap does not stop `{% set x = x + x %}`; that
  guard lived in the deleted evaluator. It is now output size + an instruction
  budget (`fuel`) + a bound on the template SOURCE, because doubling a value is
  one cheap instruction that costs a lot of memory. All three are tested by
  planting the attack, with controls just under each ceiling so a refusal is
  attributable to the cap rather than to any render failure.

### What this did NOT change

Tool definitions are not passed to the renderer, so `{% if tools %}` is always
false here and native rendering does not add tool-call framing to any model.
That is handled separately by the API layer and `tool_parse`.

### Known divergence from Jinja2: none currently

The hand-rolled engine kept `<think>…</think>` in assistant HISTORY where jinja2
strips it. minijinja + pycompat matches jinja2 there. If a future divergence is
found, pin it explicitly in the test the way that one was, rather than leaving it
in a comment.
