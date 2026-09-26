# API surfaces, errors and streaming

The evidence behind the rules in `.claude/rules/arch-api-surfaces.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## A generation that reaches the context window has finished, not failed

**The rule.** Reaching the model's context window mid-DECODE is a reply that has
finished for length. `SwarmError::ContextWindowReached { used, window }` carries
it out of `split/executor.rs`, and `length_finish_or_error` converts it to
`finish_reason: "length"` wherever tokens were produced.

**What it replaced.** The same pre-flight answered `SwarmError::Validation` for
both cases, so a reply that ran to the wall was reported as the caller's
mistake: *"This conversation is 260 tokens, longer than the 256 this model is
currently set to serve … send a shorter prompt"* — for a 38-token prompt, after
40 seconds of work.

**Measured on a two-node rig, same request, before and after** (v0.3.188
released vs the fix). Non-streaming: `finish_reason: "error"` → **`"length"`**.
Streaming: 198 deltas followed by an **error event** → 198 deltas followed by a
terminal `"length"` chunk and **zero error events**. `total_tokens: 257` against
a 256 window on every run is the mechanism firing exactly at the wall.

⚠ **The entry that reported this (`docs/FUTURE_WORK.md` #85) said the reply was
DISCARDED and the status was 400. That was true on .187 and is not on .188** —
#88's salvage already ships. What remained was the label, and the streaming
error event. Re-measure before repeating a severity from an older entry.

**What a change must keep.**

- **Prefill overflow stays a 400 with the existing wording.** The discriminator
  is `seq_len == 1 && index_pos > 0` — one position into a conversation that has
  already started. A CHUNK of a prompt is still a prompt.
- **With nothing produced it stays an error**, rewritten to the old `Validation`
  message: the window can only be hit at the first decode step if the
  conversation already filled it, and there that wording is exactly true. This
  is why the fix needed **no new user-facing string and no new i18n key**.
- **The variant carries NUMBERS, not prose.** It must survive the worker IPC hop
  and the network hop, neither of which keeps types;
  `reclassify_flattened_error` recovers it from its Display form, so the wording
  is part of the TYPE (gotcha #295).
- **Every failure arm asks, and so does the choke point.** The decode loop has
  two arms; `keeping_the_partial` wraps the five alternative coordinators, none
  of which asks for itself. The first version of this fix covered only the loop
  and was inert on `try_ngram_only_distributed`, the DEFAULT path for a node
  holding nothing. Guard:
  `every_failure_arm_of_the_decode_loop_asks_whether_the_reply_finished`, which
  compares COUNTS rather than looking back N lines — its first version looked
  back 25 and the real code sat at 26.
- **A streamed reply gets its terminal event** at the choke point, or
  `api::openai::streaming` reads the missing finish as "never streamed" and
  re-emits the whole reply as one delta (gotcha #414).

## A reply is finalised on the coordinator, including one a peer generated

(2026-09-17, gotcha #634.)

**`inference::finalize_reply_text` is the single place reply text is finalised**
— it scrubs control-token artifacts, removes a leading `<think>` reasoning
block, truncates at a stop sequence and drops the newlines that step strands.
Five paths produce a reply. Four called it: `router::local_exec`,
`process_pool`, `executor`, and `pipeline::distributed`. The fifth,
**`pipeline::remote_generate` — the path taken whenever ONE peer holds the whole
model, which is the commonest distributed shape there is — called nothing.**

⚠ **"Five" was wrong too** — there were six, and the sixth is the subsection
below (2026-09-18). The count is left as it was written because being wrong
twice in the same paragraph is the lesson: a census in prose is stale as soon
as a path is added, which is why the guard asserts the property instead.

**What it cost.** A reasoning model asked over the swarm answered with its raw
`<think>...</think>` scratchpad as the reply, while the identical request
answered locally came back clean. Reproduced 3/3 on qwen3-1.7b at
`max_tokens: 500`, `finish_reason: stop`, with both tags present in `content`.

**The discriminator that ruled out "an older peer did not strip it."** The
obvious explanation is a peer on a build that predates the strip. It was ruled
out by forcing the request onto THIS node's own peer — same binary, known to
strip — using `swarm_route.exclude_nodes` to eliminate every other candidate.
Asked directly that node stripped the block; asked as a peer it did not. Same
node, same model, same build, opposite results: it is the path, not the build.

**Why the coordinator and not the serving node.** The coordinator is the only
place that covers every peer, including ones running a build that never learned
to strip anything. Fixing the serving side would leave every already-deployed
peer leaking. The helper documents itself idempotent, so a peer that already
finalised loses nothing by it running twice.

**Why an EMPTY stop set, which is the part most likely to be "corrected" later.**
The peer generated the text and already applied both the caller's stop sequences
and its own template's, reporting the result in `matched_stop_seq`. Re-running
that decision here, against stops this node would derive for a model it may not
even hold, could truncate a reply the peer correctly kept. What remains — the
control-token scrub, the reasoning block, the stranded newlines — is exactly the
part no peer can have done on our behalf.

**What a change must keep.** An unclosed `<think>` is still shown, and that is
correct: `take_leading_reasoning_block` requires the closing tag because without
one it cannot know where the scratchpad ends. Reproduced at `max_tokens: 30`
(shown) and clean at 500 (stripped). Do not "fix" this by guessing the end.

### And a second time, on the DEFAULT distributed path (2026-09-18, gotcha #643)

"Five paths produce a reply" was the wrong count. The three speculative
coordinators — `ngram_only_spec`, `dsd` and `speculative` — are reply sources
too, they share one finaliser (`finish_speculative`), and it called nothing. It
filtered EOS **ids** out of the token list and returned the decode verbatim.

**Filtering EOS ids is not finalising**, and the gap is not cosmetic:

- a control marker the tokenizer never declared as EOS — `<|im_end|>` on a
  model that declares only `<|endoftext|>`, `<|end|>` on Phi — is an id no
  filter catches, so it reached the user as visible text;
- a `<think>…</think>` block came back as the answer, i.e. #634 again;
- **a caller's `stop` was ignored outright.** `finish_speculative` took no
  stops, set `matched_stop_sequence: None` unconditionally, and nothing
  downstream applies them: `grep -n "sampling_params.stop" src/inference/`
  matched only `executor.rs`.

**Who is on that path.** `try_ngram_only_distributed` is the DEFAULT — it needs
only `ngram_lookup_enabled` (true), no draft model configured (the default) and
one remote segment, so a node holding nothing takes it for every request. That
is every new user.

**And the standard loop had the other half of the same bug.**
`pipeline::distributed` derived its stop set from `extract_stop_strings(template)`
and never read `sampling_params.stop` at all, so a caller's `stop` was ignored
on *every* distributed request, speculative or not — while the local path
applied it twice over (in the executor, then again with template stops). One
path implementing half an invariant and another implementing the other half is
`.claude/rules/architecture.md` § "One invariant, N paths" in its purest form.

**The fix is a value, not a parameter.** `PipelineExecutor::reply_stops` answers
"what stops end this reply" once per request — caller's ∪ template's, via the
existing `chat_template::with_template_stops` — and `build_prompt_with_header`
warms it with the template it actually built the prompt from, in all three of
its branches including the `loaded_info_describes` filter (#294). A parameter
was the obvious design and is the wrong one: there are seven `finish_speculative`
call sites, and `&[]` is always spellable. Warming at the prompt choke point is
also what makes it free — the standard loop had already parsed the header, and a
second `GgufTokenizerMeta::from_gguf_file` allocates the whole vocabulary.

**How it was proved before it was fixed.** Two tests written against the
unchanged code, both red:
`a_speculative_reply_is_finalised_like_every_other_reply` got
`"<think>thinking</think>\n\nAnswer<|im_end|>"` where `"Answer"` was expected,
and `a_callers_stop_sequence_truncates_a_speculative_reply` got
`"One Two Three"` for `stop: ["Two"]`. Asserted on `finish_speculative`, the
shared finaliser, rather than once per path — so a fourth speculative
coordinator inherits the coverage.

**And verified on a real request, with the discriminating control.** Two
throwaway nodes, TinyLlama's two shards split one each, a genuine 2-segment
route (`x-swarm-route: distributed`, `x-swarm-segments: 2`) confirmed on the
n-gram coordinator (`try_ngram_only_distributed ELIGIBLE`,
`num_segments=2`). The same request — `stop: ["5"]`, "count from 1 to 10" —
against the deployed **v0.3.187-alpha** and against the fixed build:

| | v0.3.187-alpha | fixed |
|---|---|---|
| content | `1. One … 5. Five … 16. Sixteen` | `1. One\n2. Two\n3. Three\n4. Four\n` |
| `finish_reason` | `length` | `stop` |

The released binary ignored the stop outright, ran to `max_tokens` and invented
numbers past ten. Unit tests cannot reach this: they prove `finish_speculative`
finalises against whatever `reply_stops` returns, not that `reply_stops` is
populated from a real header on a real route.

⚠ **Reproduction trap: the n-gram path self-disables after one request.**
`payoff_justifies_the_wire` lets `seen_x100 == 0` (unknown) through and this
workload then scores **106** against a bar of **130**, so a second request on
the same process takes the STANDARD loop instead — and the figure is a
per-process static. **A probe on this path needs a freshly started
coordinator**, not merely a fresh request. A first attempt at the salvage probe
below silently measured `pipeline/distributed.rs` for exactly this reason, and
only the `ELIGIBLE` line in the log said so.

**What a change must keep.** `remote_generate` remains the one caller passing an
empty stop set, for the reason above: the PEER ran that decode. A coordinator
that sampled the tokens itself has no peer to have done it and must pass
`reply_stops`. The distinction is "who decided which token came next", not
"is this request distributed".

**And the census is now a guard.** `every_reply_source_finalises_its_text` in
`tests/repo_consistency.rs` resolves the function enclosing each
`InferenceOutput` construction (whole body, never a character window — see
`.claude/rules/arch-guards-and-tests.md`) and fails the build when one carries
content without its function calling `finalize_reply_text`. Exempt:
`content: String::new()`, anything inside a `#[cfg(test)]` span, and
`from_gen_result`, whose producers finalise first. Verified by restoring #643's
real pre-fix line in `speculative.rs` and watching it name
`src/inference/pipeline/speculative.rs:603`; its self-test
`the_finalisation_guard_catches_a_reply_source_that_skips_the_finaliser` keeps
the planted violation, plus the two exemptions, a neighbour case (a finalising
function must not vouch for the one after it) and the end of the `#[cfg(test)]`
span — because the first version of the exemption looked for the attribute
inside the function body, where it never is, and reported four fixtures as
offenders.

## A reply budget the caller did not choose is a ceiling, not a demand

**`inference::model_worker::resolve_max_new_tokens` is the single answer to "how
many tokens may this reply use?"**, called from both tokenization sites, and
both must USE what it returns — they assign it back to
`gen.sampling.max_tokens` so the generation loop, the off-by-one guard and the
finish-reason check all see the budget that was granted rather than the one that
was asked for.

**What it replaced.** `prompt_fits_window` only ever refused. It compared
`params.max_tokens` against the window with no regard for where that number came
from, and for an absent `max_tokens` that number was a flat serde default of
2048 — chosen with no knowledge of any model. On every model whose context
window is also 2048, `prompt_tokens + 2048 > 2048` for any non-empty prompt
whatsoever, so **every chat turn was refused**, a 34-token one included.
TinyLlama-1.1B-Chat has exactly that window and is a shipped reference model,
i.e. the obvious first pick on a weak machine. Reported as #001 against
v0.3.181-alpha, reproduced on v0.3.182-alpha's released binary 2026-09-16 with
the discriminating control: the same prompt and model with an explicit
`max_tokens: 32` was served.

The advice compounded it and is its own lesson (the gotcha #295 family): the
message told a 39-token prompt to shorten itself "by about 39 tokens", which is
the whole prompt. **Advice that cannot be followed is worse than no advice**, so
the zero-room case now names the real overage and says to start a new
conversation, and the too-large case names the budget that WOULD fit.

**One invariant, two paths — and the other path was already right.** The
llama.cpp executor had `params.max_tokens.min(n_ctx.saturating_sub(prompt_tokens))`
(`executor.rs:571`, and again at 891) since long before. Only the candle path
refused rather than clamping. Same rule, one helper now.

**Why the distinction between absent and explicit is carried on the WIRE.**
`SamplingParams.max_tokens_explicit` exists because the effective context window
is a LOAD-time property — the GGUF's `context_length` after
`effective_context_length`, `MAX_SEQ_LEN_OVERRIDE` and the memory budget have
each had a say — so no coordinator can resolve the default before dispatch. Only
the node that loaded the model knows.

**It is a bool beside the existing `u32`, NOT `Option<u32>`, and that is the
compatibility argument.** An `Option` serialises `null` when absent, which an
older peer cannot deserialise into `u32` — it would fail the whole request, in
the new→old direction, which `.claude/rules/architecture.md` § "Additive
Protocol Evolution" forbids. `max_tokens` therefore stays a concrete number on
the wire and the flag rides beside it with `#[serde(default)]`:

- old → new: field absent, reads `false`, so that peer's request is CLAMPED
  rather than refused — more permissive, never less.
- new → old: the extra field is ignored and `max_tokens` is read exactly as
  today.

No message becomes undecodable either way, so this needs no `features` bit and
does **not** bump `PROTOCOL_VERSION`. Pinned by
`a_peer_without_the_explicit_flag_is_read_as_not_explicit` and
`the_wire_still_carries_a_concrete_max_tokens_for_older_peers`.

**Lowered to fit, never raised to fill.** vLLM defaults a missing `max_tokens`
to `max_model_len - prompt_tokens`; on a long-context model that reserves far
more KV than the reply needs, which is the preemption storm reported as
verl#5504. `DEFAULT_REPLY_BUDGET` (2048) is therefore a ceiling the serving node
may only lower. This also preserves each surface's own default — MCP asks for
512 and still gets 512 on a 128k model.

**An explicit budget is honoured or refused, never shortened.** Serving a
quarter of what was asked for cannot be diagnosed from outside the server.

⚠ **The check is not on every path.** It lives in the worker, so it covers local
generation and the `remote_generate` fast path (that peer is a worker too, and
the flag reaches it). `pipeline/distributed.rs` runs its own loop over
`sampling_params.max_tokens` and never consults a window — see
`docs/FUTURE_WORK.md` #85. That is pre-existing, not a regression, and is
recorded as observed rather than reproduced.

## Every status payload reports traffic the same way, and there are THREE

**`api::metrics::network_traffic_json`** builds the figure; three surfaces serve
it, and all three must call it:

| surface | file | who reads it |
|---|---|---|
| `GET /api/admin/stats` | `api/admin.rs` | the dashboard's initial load |
| the WebSocket `stats_update` tick | `api/websocket.rs` | the dashboard, every 2 s |
| `GET /v1/status` | `api/openai/mod.rs` | **`swarmllm status`** |

**One builder is not one surface.** The figure was written as a single builder
and called from the first two; the third is the one a person is told to run when
they are working out whether this program is saturating their connection, and it
printed no Traffic line at all while the API served the numbers (2026-09-12).
`describe_traffic` in `cli/status.rs` was unit-tested against a hand-made JSON
object, so it passed while nothing supplied it.

That is the third instance in one day of the same shape — the unit passed, the
connection was missing — and the one that reached a released artifact, because
it was found by DEPLOYING and running the command a user would run. See gotchas
#567 (an unwatched cancel), #568 (a frontend scope error), #569 (this).

`every_stats_surface_carries_the_traffic_figure` fails by name when a surface
drops it.

**Absent, never zero.** `network_traffic_json` returns `Value::Null` when
`BandwidthMeter::current()` has nothing, every surface omits the figure rather
than printing 0, and `/metrics` emits no counter at all. A confident `0.0 Mbps`
cannot be told apart from a silent node, which is the reading the whole feature
exists to correct (gotcha #565).

## A model is told about its tools the way it was trained to be

(2026-09-10). **`chat_template::build_prompt` is the ONE place that decides how
a model learns what tools it has**, because the choice needs both the tool
definitions and the model's own template, and nothing else holds both.
`template_renders_tools` picks: a template that reads `tools` renders them
itself; one that never mentions them gets `describe_tools_in_prose`, the
hand-written JSON instruction that used to be the only option.

Before this, the two API surfaces flattened tools into a system message at the
edge — `ChatCompletionRequest::to_chat_messages` and `anthropic::convert` —
where the template is not known. So **Qwen3's own `{%- if tools %}` branch was
unreachable on every request ever made**: `apply_chat_template` had no `tools`
parameter at all, the variable was always undefined, and a model trained to emit
`<tool_call>{"name": …}</tool_call>` was instead handed a `{"tool_calls": [...]}`
format it had never seen. Reported from the field against v0.3.170 on Qwen3-8B,
where the model emitted repeated malformed special tokens and made no call at
all, through two independent clients.

Three things a change here must keep:

- **`tools` is a REQUIRED parameter of `build_prompt` and of
  `InferenceRequest::local`, with no shorter form that omits it.** That exact
  convenience wrapper is how the template fallback was disabled on six of seven
  paths (gotcha #171); making it required is what made the compiler enumerate
  all eleven call sites here instead of leaving the router path silently
  tool-less.
- **Tools ride on `InferenceRequest` beside `messages`, not inside them.** The
  router path builds its prompt deep inside `pipeline/prompt.rs` and
  `router/{local,distributed}_exec.rs`, long after the API surface is gone.
- **The Anthropic surface translates.** Anthropic says `{"name",
  "description", "input_schema"}`; every HuggingFace template is written
  against `{"type": "function", "function": {"name", "description",
  "parameters"}}` and dumps it verbatim — Qwen3 does `{{- tool | tojson }}`
  straight into `<tools>`. `convert::tool_definitions_for_template` is that
  translation.

`tool_choice: "none"` is enforced by returning no definitions at all, on both
surfaces: a local model knows its tools only because the prompt describes them,
so not describing them is the only place the choice can be held.

The reading half already worked — `tool_parse::try_hermes` has always parsed
`<tool_call>`. Rendering native framing without it would have produced a reply
that looked like prose containing XML, so the round trip is pinned by
`qwen3_native_framing` tests rather than assumed.

## `tojson` is a minijinja FEATURE, and without it the whole render fails

(2026-09-10). This crate builds minijinja with `default-features = false`, and
the feature list did not include `json`. **Every real chat template that renders
tools calls `tojson`** — Qwen3, Llama 3.1 and the Qwen3 GGUF-shipped variant all
do — and an unknown filter fails the ENTIRE render, not that one expression. The
model then silently gets a fallback template.

It was invisible because the only `tojson` calls in those templates sit inside
the tool branch, which nothing could reach until tools were passed. So the fix
above would have shipped INERT: the first end-to-end attempt rendered Qwen3's
non-tools branch and the test failed on a template that was, by then, being
handed its tools correctly.

Pinned as behaviour by `the_tojson_filter_is_available_to_templates` rather than
as a line in `Cargo.toml` — what matters is that the filter resolves. Dropping
the feature fails that test and `qwen3_renders_its_own_tool_framing`, both
verified by removing it.

### …and the filter itself is ours, because minijinja's answers a different question

(2026-09-11). Having the filter resolve is not the same as it behaving the way
the model's author saw it behave. minijinja's `tojson` is written for embedding
JSON in a web page, and it differs from `transformers`' in two ways that both
reached the model:

- **It escapes HTML.** `<`, `>`, `&` and `'` come out as `\u003c`, `\u003e`,
  `\u0026` and `\u0027`. A tool described as "the user's location" was handed to
  the model as `the user\u0027s location`, and so was any schema mentioning `<`
  — on every tool-carrying request to every model whose template renders tools
  through the filter. `transformers` overrides the builtin for exactly this
  reason, and its source says so in a comment. llama.cpp's minja does not escape
  either, so we were the only one of the three that did.
- **It takes `indent` and nothing else**, then calls `Kwargs::assert_all_used`,
  so any other keyword raises `unknown keyword argument` — and an error inside a
  filter fails the WHOLE render. GLM-4's template asks for
  `tojson(indent=4, ensure_ascii=False)`, so every GLM-4 request carrying tools
  was answered through a fallback with none of the model's own `# 可用工具`
  framing. minja rejects that keyword too (`Unknown argument ensure_ascii`), so
  llama.cpp has the same bug: **the reference is where to start, not where to
  stop.**

`chat_template::tojson` implements `transformers`' signature —
`tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False)` —
with Python's separator defaults (`", "` / `": "` with no indent, `","` / `": "`
with one), so a bare `{{ x | tojson }}` now produces the same bytes HuggingFace
produces. The single POSITIONAL argument stays `indent`, as in Jinja2's builtin,
minijinja and minja; `transformers` reads that slot as `ensure_ascii`, but no
chat template passes it positionally and reinterpreting an indent as a flag is
the worse failure.

What a change here must keep:

- `the_glm4_template_renders_exactly_as_transformers_does` compares byte-for-byte
  against `jinja2` driven the way `transformers` drives it — key ORDER included,
  since 2026-09-12. Until then the one deliberate difference was that keys
  arrived alphabetised: a tool definition is a `serde_json::Value` by the time
  it reaches the renderer, and `serde_json` was built without `preserve_order`,
  so its map was a `BTreeMap`; minijinja's map was one too. Both crates now
  carry `preserve_order` (`Cargo.toml` says why at each), so a schema written
  `{"type", "function": {"name", "description", "parameters"}}` reaches the
  model in that order — what `transformers`, vLLM (Python dicts) and llama.cpp
  (minja's `nlohmann::ordered_json`) all deliver, and what the model was
  validated against. `a_tool_schema_reaches_the_model_in_the_order_its_author_wrote_it`
  parses a tool from text, the shape a request body has, and pins the Qwen3
  `<tools>` line jinja2 produces; it fails with the feature dropped from EITHER
  crate (verified both ways). `sort_keys=True` is now a real sort
  (`Value::sort_all_objects`), not a no-op that happened to be true.
  The blast radius was checked before switching: no signature or hash is
  computed over a serialised `Value` map (every `identity.sign` site hashes
  explicit fields), `TransactionReason` carries no `Value`, redb values are
  re-parsed, and `Value` equality is order-insensitive under `IndexMap`. Two
  things did change shape and are accepted: `json!` literals in API responses
  now serialise in literal order rather than alphabetical, and
  `claude_sub::session_key` hashes a message's text so is now sensitive to a
  client reordering keys between turns (a cache miss, nothing worse).
- `a_tool_schema_reaches_the_model_unescaped` is the regression guard for the
  escaping half; it fails on an apostrophe alone.
- All four tests were verified to fail with the filter registration removed.

## A reasoning model's scratchpad is not the reply

**Report #031 (2026-09-13): one whitespace-only first chunk disabled the
streaming filter for the entire reply.** `withholding_reasoning`'s
"can this still become `<think>`?" test read
`THINK_OPEN.starts_with(rest) && !rest.is_empty()`. When every token so far
trims away to nothing, `rest` is `""` — which means *no character has yet said
anything either way*, the weakest possible evidence — and the `!is_empty()`
clause carved exactly that case out of "undecidable". `Reasoning::Absent`
latched, and since it short-circuits the whole match and never re-reads the
text, the model's complete scratchpad streamed to the user as the answer.
Confirmed in the field on qwen3-1.7b through the dashboard chat.

Three things a change here must keep:

- **An empty remainder waits.** Withholding for ever is not the risk it looks
  like: `pending_all` releases everything at the end of the stream, and both
  streaming surfaces already call it — the same way a reply that opens with a
  bare `<` and stops has always been released.
- **A leading whitespace chunk is ordinary input, not an edge case.** A BPE
  tokenizer decodes its word-boundary marker to a literal space, so a great
  many replies begin with one as a standalone chunk. Every one of the seven
  tests that existed here started its stream with the literal `"<think>"` —
  which is precisely the case that worked.
- **Assert that the two paths AGREE, not that each is right.** The
  non-streaming sibling was correct throughout, because it runs once over the
  complete text and so never has to decide what "nothing but whitespace so far"
  means. Two implementations tested only against themselves diverge on the
  input neither author thought of;
  `streamed_and_unstreamed_replies_strip_the_same_scratchpad` feeds one reply
  through both and requires the same answer, so a new shape of input covers
  both at once.

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

### A comment keeps a SOCKET alive, not a client that counts chunks (2026-09-26)

**Field report** (v0.3.206, CPU-only Ryzen 7 5700U): nanobot 0.3.5 against a
9,235-token agent prompt read at ~13.6 tok/s. The progress comments went out
every 15 s for the whole 11-minute prefill, and nanobot hung up at 90 s:
`Error calling LLM: stream stalled for more than 90 seconds`. The daemon side
then read `Request abandoned by the client`.

**Mechanism, read from source.** nanobot (`providers/openai_compat_provider.py`)
wraps each `stream_iter.__anext__()` of the OpenAI Python SDK in
`asyncio.wait_for(timeout=90)` (`NANOBOT_STREAM_IDLE_TIMEOUT_S`). The SDK's
`SSEDecoder.decode` (`openai/_streaming.py`) returns `None` for a line starting
with `:`, so a comment never becomes a chunk and never ends the wait. A byte-level
timer (Node's undici `bodyTimeout`, the `pi` harness) IS reset by comments;
a chunk-level one is not. Both kinds exist in agent harnesses.

**What others do.** llama.cpp's `--sse-ping-interval` and OpenRouter's
`: OPENROUTER PROCESSING` are comments, with the same blind spot; llama.cpp's
`return_progress` puts `prompt_progress` inside data chunks but only on request;
vLLM declined it (#40362); nobody was found sending an empty data chunk as a
prefill keep-alive.

**The shape chosen, and why it is safe.** OpenAI's own stream opens with
`delta: {"role": "assistant", "content": ""}`, so every OpenAI-compatible client
already parses exactly that; `ChoiceDelta` has every field optional, nanobot
guards `if text:`, and our dashboard guards `if (delta.content)`. Anthropic's
counterpart is its own `ping` event, sent after `message_start` (our preamble
goes out before the prompt pass). Ollama's empty-`content` deltas DURING
generation broke a tool-call decoder (vllm-project/semantic-router#4166), which
is why this fires only after half an interval with no data at all — a stream
producing tokens never carries one. The ticker checks the finish signal with no
await before building the event, so a keep-alive cannot follow `[DONE]`.

Tests: `a_silent_stream_is_sent_data_a_chunk_counting_client_can_see` and
`a_stream_carrying_data_is_sent_no_data_keep_alive`, each red with its half of
the condition inverted.

## A generation loop that blocks its thread must be told to, and a full buffer is not a departed client

(2026-09-11, gotchas #555/#556, FUTURE_WORK #45.) Field-reported on v0.3.172:
every streamed reply stopped at **exactly 65 completion tokens** with
`finish_reason: "stop"` — on two models, two devices, with and without tools,
purely from setting `"stream": true`, while the same request non-streaming ran to
1000 tokens. Two defects in series, and the second one hid the first.

**`inference::executor::without_starving_the_runtime` is the outer half.** The
in-process llama.cpp executor — the `-m` whole-file path, which only a GPU build
has, since `cuda` and `windows-gpu` are the feature sets that pull in `llama` —
generates on the calling thread and never yields. Called straight from a router
task, that thread is a Tokio worker, and a worker inside a multi-second C++ loop
cannot poll the task draining this request's tokens into the response. So
**nothing streamed at all**: every delta arrived at the instant generation
finished. `block_in_place` hands the worker's other tasks to a replacement thread
for the duration. All seven in-process `executor.generate*` calls go through the
helper, streaming and not — a blocking loop on a worker also stalls the libp2p
event loop, which has a tripwire of its own.

**`api::sse_send_live_blocking` is the inner half**, and it is what turned that
stall into silent data loss. `try_send` reports `Full` and `Closed` as one `Err`,
and the generation callbacks read `try_send(..).is_ok()` as "keep generating".
With nothing draining, the 64-slot channel filled and the reply ended at 65 —
and `GenerationResult` computes `finish_reason` from `completion_tokens >=
max_tokens` alone, so a generation *told* to stop reports the same `"stop"` as
one that ended its own turn. It is the blocking sibling of `sse_send_live`, with
the same two "consumer gone" conditions and the same
`SSE_CONSUMER_STALL_TIMEOUT`. Terminal `finish_reason` events go through it too:
the OpenAI encoder reads "no finish event arrived" as "this path never streamed"
and re-emits the whole reply as one delta.

**Measured on a 0.5B, one node, same prompt**: before, 65 tokens in 121 s
(`tpot_ms=1893`); after, 390 deltas ~20 ms apart, matching the non-streaming
baseline's 390 tokens at 47 tok/s. The shard path was never affected — it sends
tokens from an async task with `.send().await`, which is why it measured 0.126 s
median gaps throughout and why nothing reproduced on a CPU build, which refuses
`-m` outright.

⚠ **Fixing only the inner half makes it worse, and that is how the outer half was
found.** With backpressure handled but the runtime still starved, the generator
waited out the 60 s stall limit on the 65th token and another 60 s on the
terminal event: the same 65 tokens, now after 121 s instead of 1.4 s. A
truncation that gets slower is not a truncation that is fixed.

⚠ **Two null controls had to be repaired before either test could fail.** The
first ran the blocking loop from the test body, where it blocks `block_on`'s own
thread and starves no worker; the second gave the runtime two workers, so the
consumer was simply picked up by the other one. The test needs **one worker AND
the loop inside a spawned task** — the smallest arrangement in which the thread
running generation is the only thread that could be draining the response. It
then stalls at exactly the channel's capacity, which is the 65-on-64 shape.

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

**Rule:** `.claude/rules/arch-api-surfaces.md` § "A rendered prompt that lost the
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

## A template that refuses a system role is still told what the system turn said

(2026-09-11, gotcha #554, FUTURE_WORK #44.) Gemma and Mistral declare no system
role, and their templates do not ignore one — they `raise_exception` on it, which
by contract fails the whole render. The tool description IS a system message
(`describe_tools_in_prose`), so **every Gemma-2 request carrying tools rendered
through `gemma_fallback` rather than the template that shipped with the model**,
and so did any request carrying a caller's own system message. Found by
`examples/family_conformance.sh`, whose "the model's own template rendered" check
exists for exactly this: a fallback answers, plausibly, and nothing else says the
real template never ran.

**`chat_template::fold_system_into_first_user`** moves every system turn into the
first user turn — the remedy both publishers document — and it runs as a **RETRY
on the failure path**, after `apply_chat_template` has already declined. That
placement is the point: the template has answered the question for itself, so a
model that renders a system turn today cannot have its prompt changed by this.

**It is deliberately not keyed on `template_expects_system`.** That predicate
answers false for ANY template containing `raise_exception`, including ones that
raise only on role alternation. Conservative is right when deciding whether to
ADD a default system prompt — the question it was written for — and wrong as
grounds for rewriting a prompt that already renders.

A conversation with no user turn answers `None` and takes the fallback, which
does render a system turn: losing what the system message said is worse than
rendering it in a shape the model was not trained on.

⚠ **A test that asserts the turn markers cannot tell the template from the
fallback** — `gemma_fallback` emits the same `<start_of_turn>` markers and the
same generation prompt. The regression test puts a marker in its template that
only the template can produce, because the first draft's identity assertion
passed on the fallback it existed to rule out.

## Chat templates render on minijinja, not on a subset of our own

**Rule:** `.claude/rules/arch-api-surfaces.md` § "A rendered prompt that lost the
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

## The scratchpad filter is not a tool-calling feature, and must not be gated like one

`StreamingToolText::new(detect_tools: bool)` is the only constructor — **there is
no `Default`** — and `push` runs the reasoning filter unconditionally.
`releasable_len` withholds text for tool detection only when `detect_tools`, so
an ordinary reply streams token by token exactly as before. The two flush
helpers (`emit_openai_tool_calls`, `emit_anthropic_tool_blocks`) return early
with `pending_all()` when `!detects_tools()`, and both end-of-stream flushes run
unconditionally.

**What this replaced.** The buffer was built for tool detection, and the
reasoning filter was added to it later as a shared side effect — the rule file
recorded that approvingly, as "the buffer both encoders already share, so all
four API paths inherit it". What they inherited was a buffer that only existed
inside `if tools_requested`:

```rust
if tools_requested {
    if let Some(safe) = buffered.push(&event.text) { …send safe… }
    continue;
}
// …otherwise send event.text RAW…
```

An ordinary chat message carries no `tools`, so `push` — the only thing that runs
the filter — was never called. Reproduced on the released v0.3.179 with
qwen3-1.7b, same prompt, one field different: without `tools` the reply opened
`{"delta":{"content":"<think>"}}` and streamed the whole scratchpad; with them it
was `"C" "iao" "!"`.

**What a change must keep.** Removing `Default` is what makes the mistake
unrepresentable — a caller must say which kind of reply it is reading, and the
compiler finds every site. And the equivalence test runs `detect_tools` both
ways: the tests shipped with the previous fix all called `push` directly, which
is what production did *only on the branch that was not taken*, so they could
not have caught this. **A test that exercises a helper cannot tell you the
helper is called.**

→ gotcha #601

## The pre-token window is reported, and reported from evidence

**Rule:** `.claude/rules/arch-api-surfaces.md` § "A ticker merged into a response
stream is a termination condition".

### What this replaced

The chat tab displayed a fixed `Thinking...` from the moment a question was sent
until the first token. On a cold request that window is model loading plus
prefill — prefill alone is linear in prompt length and ~99% of a long request —
so on a modest machine the label sat there for minutes, saying the same thing
whether the node was working, waiting on a peer, or wedged. It was also simply
wrong for the many models that do no "thinking" at all.

The information existed the whole time. `progress_ticker` had been interleaving
`format_progress_comment` — phase, percent, tokens done, and an ETA measured
from that request's own rate — into every streamed response. It reached the
browser and `utils.js::readSseStream` dropped it on its first line, which
skipped everything that was not `data:` (gotcha #605).

### What a change must keep

- **Comments, not frames.** This rides an OpenAI-compatible stream. A `data:`
  frame carrying a non-chat-completion object breaks clients that deserialise
  every frame strictly; an `event:` name is invisible to anything not using
  `EventSource`. A `:` line is dropped by every conforming reader, so the
  feature costs nothing to anyone who does not want it. `STATUS_COMMENT_PREFIX`
  is the marker that separates the machine half from the prose half; it is a
  wire contract, not a private handshake.
- **Both halves.** The prose line is for a person watching `curl` and was the
  original reason the ticker existed. Do not replace it with JSON.
- **Derived, never cycled.** `LiveStatus.phase` comes from which marks the trace
  carries. There is no timer walking through stages. This is the difference
  between a status line and decoration: a label that moves regardless of what
  the node is doing is indistinguishable from a hang, which is precisely the
  complaint being answered.
- **An unknown phase must not blank the line.** A newer worker naming a phase
  this build does not know maps to `Working`, and the frontend falls back to the
  generic label when a key is missing. A mixed-version swarm is ordinary.
- **Tokens outrank the status.** `_onStatus` returns early once `live.cleared`
  is set, so a late keep-alive cannot repaint over an answer being written.
- **Node-level detail is opt-in** (`App.NODE_DETAIL_KEY`, off by default). The
  default line answers "is it working, and how long"; which computer holds which
  layers is a node operator's question, and putting it in everyone's chat bubble
  is how a chat box starts reading like a log. It is a browser preference, not
  node config — it changes only what is drawn, so it must not depend on the
  daemon being reachable, and it must not ride the Settings panel's config save.


---

## One invariant, N paths — the full path enumeration

> Rule statement: `.claude/rules/architecture.md` § "One invariant, N paths".
> This section is the evidence and the complete list of paths.

The single most repeated defect here is a **shared invariant implemented per
path**, where fixing the path in the bug report leaves the others broken. It
recurred *seven times* on 2026-07-25/26 alone: stop-string application, tool-call
buffering (twice), `include_usage` emission (twice), control-token scrubbing, and
`strip_provider_prefix`. In every case a correct helper already existed and one
consumer didn't call it.

**Before fixing anything in the request/response path, enumerate the paths.**
There are more than you expect:

- **Inference text sources (THREE)** — `inference/executor.rs` (in-process),
  `inference/process_pool.rs` (worker subprocess), `inference/pipeline/
  distributed.rs` (assembled from remote segments). A reply-content rule belongs
  at all three. Note the cold-start request takes the *distributed* path while
  later ones take the split path, so a per-path bug can look fixed five times
  and leak on the sixth.
- **OpenAI response paths** — `router_inference` + `split_non_stream_response`
  (non-streaming), `router_inference_stream` + `split_stream_response`
  (streaming).
- **Anthropic response paths** — `anthropic_non_stream` +
  `anthropic_split_non_stream`, `anthropic_stream` + `anthropic_split_stream`.
  The `_split_` variants are the local-complete fast path; the others go via the
  router.
- **Responses API** — `run_streaming` (foreground) and the background task's own
  chat request in `responses/background.rs`. They share the event loop but build
  their chat requests separately, so an opt-in set on one is absent on the other.

**A shared helper is not enough — put it where the caller cannot skip it.**
This was the standing advice here, and it kept failing: `with_template_stops`,
`emit_openai_tool_calls`, `emit_anthropic_tool_blocks` and
`strip_control_token_artifacts` all existed, were documented, and were still
missed by a sibling path. A helper nobody is *obliged* to call will eventually
not be called. Three escalating ways to make it obligatory, best first:

1. **Do it at the choke point, not in the callers.** Find the single place the
   value crosses the boundary and transform it there.
   `providers::strip_prefix_in_body` now runs inside `try_proxy_openai`,
   `proxy_to_anthropic` and `proxy_via_subprocess_anthropic` — the three
   functions that actually send — so a new proxy path is correct with no
   author action. Same shape for `inference::finalize_reply_text`: the three
   reply-text sources call one finaliser that owns the whole ordered sequence
   (scrub → truncate → trim → newline cleanup), instead of each composing those
   steps itself, which is how they silently diverged.
2. **Make the wrong call unrepresentable.** If context is needed to be correct,
   make it a required parameter rather than an `Option` with a convenience
   wrapper that passes `None` — that wrapper is how `build_prompt` disabled the
   template fallback on 6 of 7 paths (gotcha #171).
3. **Assert the property on the shared helper**, not once per path, so a new
   path inherits the coverage instead of needing its own test.

Only when none of those fit should you fall back to a doc comment saying
forgetting it is the bug.

**Verify by running the request, not by reading the diff.** Every one of the
seven passed review. The ones caught early were caught by executing the actual
path — and where a report names a specific model, that model is part of the
reproduction (gotcha #168).

**Bad reply content is evidence about the PROMPT first, the output second.**
The `<|im_end|>` leak was chased across four releases as an output-scrubbing
problem. It was a prompt problem: `apply_chat_template` returned `None` for
every official Llama-3.x template, and the fallback chain reached ChatML, so a
Llama-3 model was asked a ChatML question and answered in ChatML (gotcha #169).
Before touching `strip_control_token_artifacts` or the stop-string list, check
`grep "chat template failed" node.log` — that WARN names the real fault and had
been firing on every request for several releases. `build_prompt_with_model`
falling back at all is a bug report, not a safety net: the fallbacks
(gemma/vicuna/llava/ChatML) exist for models that ship no template, and any
model that DOES ship one should be rendering it.

---

## Timeouts — the five instances that produced the rule

> Rule statement: `.claude/rules/architecture.md` § "Timeouts: bound what actually varies".

A fixed deadline is only correct when the work behind it has a fixed size.
Where it does not, the constant silently becomes a **minimum-capability
requirement for the user** that nobody chose deliberately. Five instances were
found in one night (2026-07-27, gotcha #190):

- `UPDATE_DOWNLOAD_TIMEOUT_SECS = 300` against a ~933 MB GPU build required a
  sustained ~3.1 MB/s. Anyone slower could **never** complete an update.
- `HF_DOWNLOAD_TIMEOUT_SECS = 3600` required ~145 KB/s for a 512 MB shard.
- `INFERENCE_FORWARD_TIMEOUT_SECS = 120` capped a question forwarded to a peer
  regardless of prompt length.
- `PROVIDER_PROXY_TIMEOUT_SECS = 300` was documented as being about time to the
  first token but enforced on the whole exchange, cutting off cloud replies that
  were still streaming.
- `REQUEST_TIMEOUT_SECS = 300` capped every HTTP request, generation included —
  and so silently capped the prompt-scaled first-token budget at 300s no matter
  what it was raised to.

Rules that follow:

1. **Prefer an inactivity timeout to a total one.** `reqwest`'s `read_timeout`
   (0.12+) catches a stalled transfer just as fast while leaving a slow healthy
   one alone, and requires no guess about size or bandwidth. Use it for every
   download and every streamed proxy response.
2. **Where inactivity does not apply, scale the budget by the input and cap it** —
   `pipeline::remote_generate::first_token_timeout(prompt_tokens)` is the shared
   helper; call it rather than inventing another rule. Prefill is linear in
   prompt length and is ~99% of a long request.
3. **Generation gets no blanket deadline.** Routes that can run a model are
   merged into the router OUTSIDE the `TimeoutLayer` (`generation_routes` in
   `api/server.rs`). The merge MUST stay before the auth layer or those
   endpoints answer without a key — pinned by
   `generation_routes_still_require_a_key`.
4. **When you change a limit, grep the whole path for other limits.** A budget
   is only as generous as the tightest ceiling above it, and that ceiling is
   usually in another file, in middleware, behind a comment that went stale
   before the code did.
5. **Read the comment against the code.** In four of the five, the comment
   reasoned about one quantity ("before the first token") while the constant
   bounded another (the total). A stale comment asserting an invariant reads as
   verification and stops anyone re-deriving it.
6. **The frontend is part of "the whole path".** The five instances above were
   all in Rust, and the tightest ceiling on a comparison request turned out to
   be a hardcoded 45 s `AbortController` in `frontend/js/components/compare.js`
   — on the very requests the daemon deliberately serves outside its own
   `TimeoutLayer`. It discarded replies the daemon had finished computing
   (report #009: `execute_ms=44886`, `finish_reason=stop`, aborted a fraction of
   a second earlier), and the duration was baked into the translated string in
   all 21 locales, so it was not even greppable as a number. A generation
   request from the browser gets no client-invented deadline either; where one
   is unavoidable it is derived from what the daemon permits and says something
   true when it fires — the node may still be working, and the reply was not
   necessarily lost.
   **Streaming is what makes rule 1 available to a browser**: a non-streaming
   `fetch` has no intermediate bytes, so it cannot have an inactivity timeout.
   The chat tab streams, which is why its 30 s `authFetch` default bounds only
   time-to-headers and is harmless there.

## "At this machine" is decided once, from the socket AND the headers (2026-09-23)

**What it replaced.** Seven privileges were granted to "the person at this
machine", each decided with `addr.ip().is_loopback()`: the automatic API-key
handout (`middleware.rs`, through `dashboard_trust::classify`), keyless
`/metrics`, `POST /api/admin/update/check` and `/update/apply`, `/shutdown`,
prompt previews in `GET /api/admin/responses`, and the admin rate-limit
exemption. Loopback means only that the last TCP hop began in this network
namespace — and a reverse proxy on the same host is exactly that, for everyone
who can reach it. The book's own nginx example (`proxy_pass
http://127.0.0.1:8800`) therefore gave every visitor admin: the page served them
a nonce, the nonce plus loopback bought the key. `tailscale serve` and Funnel do
the same by design. Found by a docs audit reading the example against
`dashboard_trust.rs`, whose own header had described the proxy case since it was
written — the fix there changed the classifier's other branches and left the
loopback branch as it was.

**What was researched.** Jupyter (`check_host`, `ServerApp.local_hostnames`)
treats a request as local only if the `Host` header is `localhost` or a
loopback IP — it was written against DNS rebinding and covers proxies for the
same reason: both carry the name the visitor used. Home Assistant
(`trusted_proxies`) refuses forwarding headers from a proxy nobody declared.
`tailscale serve` adds `X-Forwarded-For/Host/Proto` and `Tailscale-User-Login`;
Caddy adds `X-Forwarded-*` by default; nginx adds nothing by default.

**The rule.** `api::origin::RequestOrigin` is computed from the socket and the
headers; `is_this_machine()` = loopback AND a local `Host` (or none) AND no
forwarding header (`Forwarded`, `X-Forwarded-*`, `X-Real-IP`, `Via`,
`Tailscale-User-Login`). `classify` takes the whole origin — never a bare
address — and answers `DashboardTrust::Proxied` for any proxied request, so a
LAN or tailnet proxy cannot carry LAN/overlay trust to whoever is behind it.
`is_trusted` and the frontend's `isTrustedOrigin` are ALLOWLISTS: the frontend
check had been `!== 'untrusted'`, which would have read the new label as trusted.

**What it does not do.** It is defence in depth. A proxy configured to send no
forwarding header and `Host: 127.0.0.1` (nginx's defaults) is still loopback and
still trusted — which is why the docs keep saying never to proxy over loopback.
Every signal only REMOVES a privilege, so none of them has to be trusted: a
forged header costs its sender the automatic key and grants nothing. A person
browsing their own machine by its hostname (Debian's `127.0.1.1`) now pastes the
key once; failing closed there is the intended direction.

Guard: `no_request_privilege_is_decided_by_a_bare_loopback_check` (with a
planted self-test). Tests: `api::origin` (truth table incl. DNS rebinding and
look-alike hosts), `a_proxied_request_is_never_trusted`.
