pub mod allreduce;
pub mod attn_kernel;
pub(crate) mod attn_rows;
pub(crate) mod attn_softmax;
pub(crate) mod cancel;
pub(crate) mod card_pace;
pub mod chat_template;
pub mod coupled_noise;
pub(crate) mod cpu_pools;
pub(crate) mod cuda_graph;
pub(crate) mod cuda_pool;
pub mod decode_attn;
pub mod draft_tree;
pub mod dsd_controller;
pub mod executor;
pub mod fast_math;
pub(crate) mod harmony;
#[cfg(feature = "candle-cuda")]
pub(crate) mod heads_first;
pub mod kv_cache;
pub(crate) mod layers;
pub mod local_embedder;
pub mod mem_bandwidth;
pub(crate) mod model_arch;
pub mod model_worker;
pub mod ngram_lookup;
pub mod owner_replies;
#[cfg(feature = "candle-cuda")]
pub(crate) mod partial_rope;
pub mod pipeline;
pub mod prefetch;
pub mod prefill_attn;
pub mod prefill_pacer;
pub mod process_pool;
pub(crate) mod prof;
pub mod quant;
pub(crate) mod reply_grammar;
pub mod residual_norm;
pub mod route_override;
pub mod router;
pub mod sampling;
pub mod scheduler;
pub mod segment_latency;
pub(crate) mod shard_layout;
pub mod slot_table;
pub mod speculative;
pub mod split;
pub mod swift;
pub(crate) mod tensor_util;
pub mod thermal;
pub(crate) mod tokenizer;
pub mod trace;
#[cfg(feature = "candle-cuda")]
pub(crate) mod tree_rank;
pub mod vision;
pub mod worker_ipc;

/// The `finish_reason` on a reply that was cut short by a failure rather than
/// by the model, a limit, or the caller.
///
/// **`"error"` is vLLM's own value for this** (`FinishReason::ERROR`, beside
/// `ABORT`), which is the precedent worth following: the OpenAI schema defines
/// only `stop` / `length` / `tool_calls` / `content_filter` / `function_call`,
/// and none of them means "the machinery gave up part-way". A serving stack
/// that has to say so goes outside that set, and the alternative — reusing a
/// defined value — is the gotcha #433 lie in a new place: `"stop"` claims the
/// model chose to end, `"length"` claims a limit was reached.
///
/// It is deliberately NOT a hint that the caller should retry. A reply carrying
/// it is real output that really was generated; what it is not is finished.
///
/// The Anthropic surface cannot pass it through — that vocabulary has no member
/// for an interrupted turn and an undefined `stop_reason` was removed once
/// already (gotcha #300) — so `api::anthropic::convert::map_finish_reason`
/// translates it to `max_tokens`, the only value Anthropic defines that means
/// "incomplete, cut off". See that function for why the catch-all is not enough.
pub const FINISH_REASON_INTERRUPTED: &str = "error";

/// Strip a trailing partial stop-string suffix from `text` in place.
///
/// Token-by-token stop-string checking only catches complete matches, so a
/// partial stop string at the very end of generation can leak into the output
/// (e.g. "<|user" when the stop is "<|user|>"). This trims at most one such
/// prefix — once a trim happens we return immediately so later stops can't
/// cascade across the already-truncated text.
///
/// A prefix ends on a CHARACTER boundary, never a byte offset. The byte form
/// panicked on any stop ending in a multi-byte character — `"。"`, `"\n用户"` —
/// and every reply on every path is finalised through here, so a caller's own
/// non-ASCII `stop` killed the request after its whole reply had been
/// generated. A byte-level partial could not match anyway: `text` is valid
/// UTF-8 and cannot end halfway through a character.
pub(crate) fn trim_trailing_partial_stop(text: &mut String, stops: &[String]) {
    for stop in stops {
        // Byte lengths of every non-empty proper prefix, longest first: the
        // start of each character after the first is where a prefix ends.
        for end_len in stop.char_indices().map(|(i, _)| i).filter(|&i| i > 0).rev() {
            let prefix = &stop[..end_len];
            if text.ends_with(prefix) {
                text.truncate(text.len() - end_len);
                return;
            }
        }
    }
}

/// Control-token names that appear in the chat templates of the model families
/// we support. Used by [`strip_control_token_artifacts`].
///
/// Deliberately a NAME list rather than a list of full markers: the whole point
/// is to catch mangled spellings, so `<|im_end|>`, `<|im_end|`, `<|im_end>|`
/// and `<|im_end` all reduce to the same name.
const CONTROL_TOKEN_NAMES: &[&str] = &[
    "im_end",
    "im_start",
    "eot_id",
    "eom_id",
    "start_header_id",
    "end_header_id",
    "endoftext",
    "python_tag",
    "end_of_turn",
    "start_of_turn",
    // Phi-3/3.5/4: `<|end|>` closes a turn and the turn markers open one. A
    // GPT-2-BPE vocab decodes all four to their literal characters, so they
    // reached users as visible text (reported in the field on Phi-4-mini,
    // 2026-09-10). Scrubbing is safe even for the harmony / solar-open formats
    // where `<|end|>` must NOT stop generation: removing a control marker from
    // visible text and ending the reply at it are different decisions, and only
    // the second one truncates.
    "end",
    "user",
    "assistant",
    "system",
];

/// What a reply may not OPEN with: newlines, and spaces.
///
/// The newlines are what a control marker removed in front of the answer
/// leaves behind. The space is a SentencePiece model's first token: after a
/// turn marker it generates `▁The`, which decodes to " The", so Mistral's
/// replies — and Phi-3.5's, once its prompt matched its own tokenizer — began
/// with a space (FUTURE_WORK #103). Hugging Face's decoder drops it
/// (`Strip(start=1)`); llama.cpp, vLLM and TGI pass it through; OpenAI's API
/// never returns one. A reply here always opens a fresh assistant turn — there
/// is no continuation of a partial assistant message, where a leading space
/// would carry meaning — so it is always noise.
///
/// ALL of them, not exactly one as Hugging Face strips: `finalize_reply_text`
/// must stay idempotent, and stripping one would take another on each call.
///
/// **The one rule for both halves**: `finalize_reply_text` (non-streamed) and
/// `api::tool_parse::StreamingToolText` (every streamed reply, on all four API
/// surfaces) both read it, so a reply cannot open differently depending on
/// whether it was streamed.
pub(crate) const REPLY_LEADING_WHITESPACE: [char; 3] = ['\n', '\r', ' '];

/// Put generated text into its final, user-facing form.
///
/// **Every source of reply text must end by calling this, and must not perform
/// these steps itself.** There are three sources — the in-process
/// [`executor`](crate::inference::executor), the worker subprocess
/// ([`process_pool`](crate::inference::process_pool)), and the reply assembled
/// from remote segments ([`pipeline::distributed`]) — and a control token
/// reached users across four releases because each was fixed separately.
///
/// The steps are ordered, and the order is the part that keeps being got
/// wrong, so it lives here rather than in three comment blocks:
///
/// 0. **Rewrite a harmony reply** ([`PromptOpened::Harmony`], gpt-oss) into a
///    leading `<think>` block, content and Hermes calls ([`harmony`]) — before
///    the scrub, which removes the `<|end|>` that ends each of its messages.
/// 1. **Scrub control-token artifacts.** Before anything truncates, because
///    some models emit a stray end-of-turn marker *before* their answer
///    (`<|im_end|>hello`) — truncating at that match first would discard the
///    answer, turning a visible leak into an empty reply.
/// 2. **Truncate at the first genuine stop sequence.** Only caller-supplied
///    stops, on what survived step 1.
/// 3. **Trim a trailing partial stop.** A marker decodes across several tokens
///    and the leading pieces match nothing, so they are already emitted.
/// 4. **Drop the whitespace a reply opens with** — newlines stranded by a
///    marker removed in step 1, and the space a SentencePiece reply starts
///    with ([`REPLY_LEADING_WHITESPACE`]).
///
/// Before this existed the three sources ran different subsets in different
/// orders: the distributed path trimmed before scrubbing and never did step 4,
/// so it could still return an answer-less reply after the scrub was
/// "everywhere". Pass an empty `stops` slice when the caller has none — steps
/// 2 and 3 then do nothing and the rest still applies.
///
/// Returns what it took out besides trimming ([`FinishedReply`]): the stop
/// sequence that truncated the text, so a caller can set `finish_reason` from
/// it, and a reasoning model's closed scratchpad, which the caller hands on in
/// `InferenceOutput::reasoning` (#141). It is safe to call more than once:
/// every step is idempotent, which lets a caller run it again with a different
/// stop set (the router applies template-derived stops that the executor never
/// sees) — a second call finds no scratchpad, so a caller keeps the FIRST
/// call's.
///
/// `opened` says whether the prompt left the reply inside a reasoning block
/// ([`PromptOpened`], #392) — REQUIRED, so no source can forget it. Only the
/// FIRST call on raw generated text passes the prompt's verdict; a second pass
/// over text this already finalised passes [`PromptOpened::Nothing`], or the
/// answer would be read as an unfinished scratchpad.
pub(crate) fn finalize_reply_text(
    text: &mut String,
    stops: &[String],
    opened: PromptOpened,
) -> FinishedReply {
    // Snapshot enough of the incoming text to identify it if the steps below
    // consume all of it. Costs one small allocation per completion — this runs
    // once per reply, never per token — and nothing at all when the model
    // produced no text to begin with.
    let had_content = !text.trim().is_empty();
    let before: String = if had_content {
        text.chars().take(EMPTIED_REPLY_LOG_CHARS).collect()
    } else {
        String::new()
    };

    // A harmony reply's channels, rewritten before the scrub — which removes
    // `<|end|>`, the marker that ends each message.
    if opened == PromptOpened::Harmony {
        *text = harmony::rewrite(text);
    }
    strip_control_token_artifacts(text);
    // After the scrub, which is for what the MODEL wrote; the opener is the
    // template's.
    opened.put_in_front_of(text);
    let reasoning = take_leading_reasoning_block(text);
    let mut matched = None;
    for stop in stops {
        if let Some(pos) = text.find(stop.as_str()) {
            text.truncate(pos);
            matched = Some(stop.clone());
            break;
        }
    }
    trim_trailing_partial_stop(text, stops);
    let start = text.len() - text.trim_start_matches(REPLY_LEADING_WHITESPACE).len();
    if start > 0 {
        text.drain(..start);
    }

    // The model generated something and finalisation removed all of it. The
    // caller will return `content: ""` with `finish_reason: "stop"` and HTTP
    // 200, which is indistinguishable from a real answer — so without this line
    // the only way to notice is for a human to see a blank reply. Every blank
    // reply diagnosed here so far (a missing system turn, a template from the
    // wrong model family, byte-fallback decoding) began that way.
    //
    // What was removed is the evidence: a leaked marker points at the template,
    // a stop sequence matching at position 0 points at the prompt. Logged at
    // WARN because a user is looking at an empty answer either way.
    if let Some(evidence) = emptied_reply_evidence(had_content, text, &before) {
        EMPTY_REPLIES_TOTAL.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        tracing::warn!(
            matched_stop = ?matched,
            generated = %evidence,
            "reply is empty after finalisation — the model generated text and all of it was \
             removed as control tokens or stop sequences; check the prompt and chat template \
             for this model before looking at sampling"
        );
    }

    FinishedReply {
        stop: matched,
        reasoning,
    }
}

/// What [`finalize_reply_text`] took out of a reply besides trimming it.
///
/// `must_use` because dropping it is the bug it exists for: the closed
/// scratchpad is gone from the text once the call returns, so a source that
/// ignores `reasoning` loses it for every surface (#141 — every non-streamed
/// reply did, while the stream carried it as `delta.reasoning_content`).
#[must_use = "a reply's reasoning goes in `InferenceOutput::reasoning`, or it is lost (#141)"]
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub(crate) struct FinishedReply {
    /// The caller's stop sequence that truncated the text, if any.
    pub stop: Option<String>,
    /// A reasoning model's CLOSED scratchpad from the start of the reply,
    /// trimmed as the stream trims what it sends; `None` when there was none.
    /// An unclosed one stays in the text (see [`take_leading_reasoning_block`]).
    pub reasoning: Option<String>,
}

/// Replies the model generated and finalisation then removed entirely —
/// `swarmllm_empty_replies_total` on `/metrics`.
///
/// The warning beside it is for a person reading the log; this is for a graph.
/// An emptied reply reaches the client as `content: ""`, `finish_reason:
/// "stop"`, HTTP 200 — indistinguishable from a real answer — so nothing in the
/// response, the request counters or the latency histogram moves when it
/// happens, and how OFTEN it happens on a node was unknowable except by reading
/// its log. An emptied reply stays a 200, as OpenAI, vLLM and llama.cpp answer
/// an empty generation — an error would fail legitimate ones too (FUTURE_WORK
/// § "Decided", #186); this figure is what would reopen that. A `static` because `finalize_reply_text` is a free function reached
/// from all three reply-text sources and has no state to hang a counter on.
pub static EMPTY_REPLIES_TOTAL: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Longest reply still treated as "a stop sequence ate the answer".
///
/// Two tokens is not an answer to anything. Deliberately tiny: the point is to
/// catch a stop string matching the model's opening words, not to second-guess
/// a genuinely terse reply.
const STOP_ATE_THE_REPLY_TOKENS: u32 = 2;

/// The token budget above which a near-empty reply is worth reporting.
///
/// Below this the caller plainly wanted something short — a classification, a
/// yes/no, one word — and getting it back is not evidence of anything. Agentic
/// clients, which is where the real fault was seen, budget thousands.
///
/// Chosen from the false positives an external tester actually hit: their short
/// probes used a budget of 10, the genuine failure used 4096. Set at 64 — above
/// any "one word please" request, and below the 100-token budget the existing
/// test already documents as worth reporting. A warning that cries wolf on
/// ordinary traffic is the one nobody reads when it matters.
const BUDGET_THAT_EXPECTED_PROSE: u32 = 64;

/// Did the model end its own turn immediately, well inside the token budget?
///
/// No stop sequence matched, so nothing external cut the reply short — the model
/// emitted end-of-turn at once. That points at the prompt (a chat template from
/// the wrong family, a rendered prompt that does not end where the model expects
/// to answer) rather than at sampling.
///
/// `max_tokens` is what separates it from a caller who asked for one token and
/// got exactly one. Warning on that would fire on every deliberate
/// single-token request, which is how a useful warning becomes noise.
///
/// **`completion_tokens < max_tokens` is not that separation**, which is what
/// the first version used: it is true of almost every request, including
/// "say OK in one word" with a budget of 10. A tester running down a real
/// single-token bug reported this firing twice on their own short test requests
/// where one token WAS the right answer, and asked whether the false-positive
/// rate was worth tuning — it was. The budget has to be large enough that a
/// near-empty reply is genuinely surprising, so the warning means something when
/// it does fire.
fn reply_ended_itself_early(
    completion_tokens: u32,
    max_tokens: u32,
    matched: Option<&str>,
) -> bool {
    matched.is_none()
        && completion_tokens <= STOP_ATE_THE_REPLY_TOKENS
        && max_tokens >= BUDGET_THAT_EXPECTED_PROSE
}

/// Did a stop sequence leave the caller with essentially no reply?
///
/// Split from the logging so the threshold can be tested without capturing log
/// output — the same reason `emptied_reply_evidence` is split out.
pub(crate) fn stop_sequence_ate_the_reply(completion_tokens: u32, matched: Option<&str>) -> bool {
    matched.is_some() && completion_tokens <= STOP_ATE_THE_REPLY_TOKENS
}

/// Say why a reply came back with (almost) nothing.
///
/// Two causes, and until this existed the node reported neither: a stop sequence
/// in the request matched immediately, or the model emitted end-of-turn at once.
/// Both surface to an OpenAI client identically — `finish_reason: "stop"`,
/// HTTP 200, a blank or one-word answer — and the distinction is exactly what
/// tells a user whether to look at their client's `stop` array or at the prompt.
///
/// **The OpenAI surface cannot say this in the response.** `finish_reason` is
/// `"stop"` whether the model chose to end its turn or a stop sequence cut it
/// off, and the schema has no field for which one fired. (The Anthropic surface
/// does carry `stop_sequence`, and does report it.) So a reply truncated to
/// nothing is indistinguishable, to the client, from a model with nothing to
/// say — and the node said nothing either.
///
/// That gap has a measured cost. An external report on 2026-08-23 described a
/// coding agent getting `completion_tokens: 1` from a large tool-using request
/// with no error, and a testing session went into narrowing it; plain `curl`
/// could not reproduce it because the attempts left the client's `stop` array
/// out. Reproduced locally afterwards: a stop string matching the model's
/// second token yields exactly `completion_tokens: 1, finish_reason: "stop"`,
/// silently. Whether that was the reported cause is unconfirmed; that the
/// product could not say so is not.
///
/// **Call this wherever a generation finishes with a `matched_stop_sequence`.**
/// It deliberately does NOT live in `finalize_reply_text`, which looks like the
/// choke point and is the wrong place: every generator applies the caller's stop
/// sequences DURING generation and stops before the matching text is kept, so by
/// the time finalisation runs the text no longer contains the stop and there is
/// nothing left to notice. That was tried, and it silently never fired —
/// caught by running a request rather than by reading the diff.
pub(crate) fn report_short_reply(
    request_id: &uuid::Uuid,
    completion_tokens: u32,
    max_tokens: u32,
    matched: Option<&str>,
) {
    if let Some(stop) = matched {
        if !stop_sequence_ate_the_reply(completion_tokens, matched) {
            tracing::debug!(
                %request_id,
                completion_tokens,
                stop_sequence = %stop.escape_debug(),
                "DIAG: reply ended on a caller-supplied stop sequence"
            );
            return;
        }
        tracing::warn!(
            %request_id,
            completion_tokens,
            stop_sequence = %stop.escape_debug(),
            "a stop sequence in the request matched almost immediately, so the reply came back \
             with next to nothing — the caller sees finish_reason \"stop\" and cannot tell that \
             from the model choosing to say nothing; check the `stop` values the client sends"
        );
        return;
    }

    if reply_ended_itself_early(completion_tokens, max_tokens, matched) {
        tracing::warn!(
            %request_id,
            completion_tokens,
            max_tokens,
            "the model ended its turn immediately, well inside the token budget, and no stop \
             sequence matched — it emitted end-of-turn straight away. That is usually the prompt \
             rather than sampling: check the chat template for this model (grep `chat template \
             failed`) and whether the rendered prompt ends where the model expects to answer"
        );
    }
}

/// Whether an emptied reply is worth reporting, and the evidence to report with
/// it — `None` means stay silent.
///
/// **Extracted so the decision can be tested without capturing logs.** The
/// tests for this used to install a global `tracing` subscriber and assert on
/// what it captured, and that arrangement failed intermittently on CI three
/// times across two attempted fixes, each time reporting `got: ` with nothing
/// after it on a commit that had touched none of it. The cause was never
/// established: on the third occurrence the two documented mechanisms were both
/// falsified by direct experiment (a callsite hit before the subscriber exists
/// is NOT left permanently silenced, and neither racing the install from eight
/// threads nor 3,600 concurrent captures against a faithful copy of the helper
/// reproduced a single empty capture).
///
/// So the dependence is removed rather than the race chased further. What is
/// pinned here is what the tests were really for: that it fires only when
/// generated text was wholly removed, and that the evidence names what went —
/// a leaked marker points at the template, a stop matching at position 0 points
/// at the prompt. That the adjacent `warn!` is reached is a single line in plain
/// view, which is the trade being made, and it follows the pattern this codebase
/// already uses for `inbound_warning_decision`: pin the decision, not the log.
pub(crate) fn emptied_reply_evidence(
    had_content: bool,
    finalised: &str,
    before: &str,
) -> Option<String> {
    (had_content && finalised.trim().is_empty()).then(|| before.escape_debug().to_string())
}

/// How much of an emptied reply to put in the log. Enough to recognise a leaked
/// template or a stray marker, short enough not to dump a whole generation.
const EMPTIED_REPLY_LOG_CHARS: usize = 240;

/// Remove control-token artifacts from generated text, in whatever spelling
/// they arrive in.
///
/// Stop-string matching handles the well-formed case, and
/// [`trim_trailing_partial_stop`] handles a marker split across tokens. Neither
/// helps when a model file's template disagrees with the weights it was
/// quantised from: such a model emits another family's markers, and decoding
/// them produces genuinely malformed spellings. Observed live 2026-07-25 on a
/// Llama-3.2 Q8_0 whose declared EOS is `128009` (`<|eot_id|>`) but which
/// emitted `<|im_end|>`, `<|im_end|` and `<|im_end>|` — the last with `>` and
/// `|` transposed, so it is not a prefix of anything and can appear mid-string.
///
/// Only spans naming a KNOWN control token are removed, so a user's own
/// `<|something_custom|>` survives. That is the line between sanitising our own
/// leakage and editing someone's content — a reply that genuinely discusses
/// `<|im_end|>` will lose it, which is the accepted cost of not showing control
/// tokens to every user of a mismatched model file.
/// Remove a reasoning model's `<think>…</think>` preamble from the reply (or
/// any block in [`REASONING_BLOCKS`]).
///
/// A reasoning model — Qwen3, QwQ, DeepSeek-R1 — writes its working out first
/// and the answer after it, and every serving stack separates the two: vLLM
/// extracts it with a `--reasoning-parser` into `reasoning_content`, and
/// llama.cpp defaults Qwen3.5 to non-thinking so it is never produced. We did
/// neither, so the whole scratchpad was returned as the assistant's message —
/// someone asking "Say OK" gets several hundred tokens of deliberation with the
/// answer somewhere at the end. Same class as gotcha #169: content that is not
/// the reply reaching the user as the reply.
///
/// Three conditions, and all of them matter:
///
/// - **It must be at the START.** A `<think>` later in the text is the model
///   writing about thinking, not thinking.
/// - **The block must be CLOSED.** An unterminated one means the reply was cut
///   off mid-reasoning (`max_tokens`), so the whole thing is scratchpad —
///   removing it returns an empty message, and the caller cannot tell that from
///   a real blank answer. Leaving it is the lesser harm, and the emptied-reply
///   warning below is the thing that would otherwise fire on every such reply.
/// - **Only the block, plus the whitespace after it.** The answer follows.
///
/// The reasoning itself is RETURNED, trimmed (`None` when it was blank), for
/// the surface that has a field for it: OpenAI's `message.reasoning_content`,
/// as llama.cpp, SGLang and DeepSeek name it and as our own stream sends it
/// (vLLM has since renamed its field `reasoning`). Anthropic withholds it, as
/// its stream does. Returning it glued to the answer is the bug, not a smaller
/// version of the field.
pub(crate) fn take_leading_reasoning_block(text: &mut String) -> Option<String> {
    let lead = text.len() - text.trim_start().len();
    let block = opening_reasoning_block(&text[lead..])?;
    let rel = text[lead..].find(block.close)?;
    let inner = text[lead + block.open.len()..lead + rel].trim();
    let reasoning = (!inner.is_empty()).then(|| inner.to_string());
    let mut end = lead + rel + block.close.len();
    end += text[end..].len() - text[end..].trim_start().len();
    text.drain(..end);
    reasoning
}

/// The markers of one kind of reasoning block.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ReasoningBlock {
    pub open: &'static str,
    pub close: &'static str,
}

/// Every reasoning block a reply may open with — the ONE list
/// [`take_leading_reasoning_block`] and `api::tool_parse`'s streaming buffer
/// both read. `<think>` is Qwen3's, QwQ's and DeepSeek-R1's; Gemma 4 thinks in
/// a channel, `<|channel>thought` … `<channel|>` (llama.cpp master's
/// `thinking_start_tag` / `thinking_end_tags`, `common/parsers/gemma4.cpp`).
pub(crate) const REASONING_BLOCKS: [ReasoningBlock; 2] = [
    ReasoningBlock {
        open: "<think>",
        close: "</think>",
    },
    ReasoningBlock {
        open: "<|channel>thought",
        close: "<channel|>",
    },
];

/// The block `text` opens with, if it opens with one. `text` starts where the
/// reply's leading whitespace ends.
pub(crate) fn opening_reasoning_block(text: &str) -> Option<ReasoningBlock> {
    REASONING_BLOCKS
        .into_iter()
        .find(|block| text.starts_with(block.open))
}

/// Did the rendered prompt leave the model's reply INSIDE a reasoning block —
/// or hand it a harmony turn, whose reply is channels ([`Self::Harmony`])?
///
/// A template can open the scratchpad itself: Qwen 3.5's generation prompt
/// ends `<|im_start|>assistant\n<think>\n`, so the reply starts mid-thought
/// with no opening tag of its own, and every splitter here — which recognise a
/// block only when the REPLY opens it — handed the whole scratchpad, `</think>`
/// and all, to the reader as the answer (#392: the network's most-run family,
/// every surface, unasked).
///
/// llama.cpp's answer (`common/chat.cpp`, master): take the generation prompt
/// as the render with `add_generation_prompt` less the render without, and
/// parse `generation_prompt + output`. Ours is the same in one step: the
/// verdict is a pure function of the rendered text, decided where it is
/// rendered, and the opener is put back in front of the reply — by
/// [`finalize_reply_text`] for a finished one, by `api::tool_parse::
/// StreamingToolText` for a streamed one — so every splitter reads the reply
/// as if the model had opened the block itself. A template's CLOSED empty
/// block (`<think>\n\n</think>\n\n`, thinking turned off) is [`Nothing`](Self::Nothing).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PromptOpened {
    /// The reply starts outside any block — the model opens one if it thinks.
    Nothing,
    /// The prompt ends on this block's opener: the reply starts inside it.
    Inside(ReasoningBlock),
    /// The prompt handed the model a harmony turn (`<|start|>assistant`,
    /// gpt-oss): the reply is channels, rewritten by [`harmony::Reply`] into
    /// a leading `<think>` block, content and Hermes calls BEFORE anything
    /// else reads it (#390) — by [`finalize_reply_text`] and by
    /// `api::tool_parse::StreamingToolText` alike.
    Harmony,
}

impl PromptOpened {
    /// Read off a rendered chat prompt: its end, less trailing whitespace, is
    /// an opener of [`REASONING_BLOCKS`].
    ///
    /// Only ever the text a template rendered for the model's own turn — never
    /// a prompt with a continued reply appended (`build_prompt_with_header`
    /// reads it before the append), which ends mid-reply, not on the opener.
    pub fn of(prompt: &str) -> Self {
        let end = prompt.trim_end();
        if end.ends_with(harmony::TURN) {
            return Self::Harmony;
        }
        REASONING_BLOCKS
            .into_iter()
            .find(|block| end.ends_with(block.open))
            .map_or(Self::Nothing, Self::Inside)
    }

    /// Put the opener back in front of `reply`, as llama.cpp parses
    /// `generation_prompt + output` — unless the reply is blank: a model that
    /// generated nothing answered nothing, and an opener alone would read as an
    /// unfinished scratchpad.
    pub(crate) fn put_in_front_of(self, reply: &mut String) {
        if let Self::Inside(block) = self {
            if !reply.trim().is_empty() {
                reply.insert_str(0, block.open);
            }
        }
    }
}

/// A [`PromptOpened`] shared between the code that renders a request's prompt
/// and the code that reads its reply as it streams.
///
/// A streamed reply's buffer exists BEFORE anything renders: the API opens the
/// stream, the router plans, and the prompt is rendered on whichever path
/// serves it. So the verdict travels beside the tokens — set once where the
/// prompt is rendered (`StreamingTokenTx::note_prompt_opened`; a second
/// setting is ignored, so a continued reply keeps its first attempt's), read
/// when the reply's first character arrives. Never set reads as
/// [`PromptOpened::Nothing`], the behaviour before #392.
#[derive(Clone, Default, Debug)]
pub(crate) struct PromptOpenedSlot(std::sync::Arc<std::sync::OnceLock<PromptOpened>>);

impl PromptOpenedSlot {
    /// A slot that already holds `verdict`.
    #[cfg(test)]
    pub(crate) fn known(verdict: PromptOpened) -> Self {
        let slot = Self::default();
        slot.note(verdict);
        slot
    }

    /// Record the verdict. The first one stands.
    pub(crate) fn note(&self, verdict: PromptOpened) {
        let _ = self.0.set(verdict);
    }

    /// The verdict, or [`PromptOpened::Nothing`] where none was recorded.
    pub(crate) fn get(&self) -> PromptOpened {
        self.0.get().copied().unwrap_or(PromptOpened::Nothing)
    }
}

pub(crate) fn strip_control_token_artifacts(text: &mut String) {
    if !text.contains("<|") {
        return;
    }
    let mut out = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'<' && i + 1 < bytes.len() && bytes[i + 1] == b'|' {
            // Consume the name: word characters after the opening `<|`.
            let name_start = i + 2;
            let mut j = name_start;
            while j < bytes.len() && (bytes[j].is_ascii_alphanumeric() || bytes[j] == b'_') {
                j += 1;
            }
            let name = &text[name_start..j];
            if CONTROL_TOKEN_NAMES.contains(&name) {
                // Consume the trailing terminator run in any order and length:
                // `|>`, `>|`, `|`, `>`, `|>>`, or nothing at all (truncated).
                // Not capped at two characters — a mangled marker can carry a
                // third, and capping left a stray `>` in the middle of an
                // otherwise clean sentence ("The ocean> is a vast body...").
                // A legitimate `>` is separated from the marker by other text,
                // so it stops the run naturally.
                let mut k = j;
                while k < bytes.len() && (bytes[k] == b'|' || bytes[k] == b'>') {
                    k += 1;
                }
                i = k;
                continue;
            }
        }
        // Not an artifact — copy this byte through. Safe because we only ever
        // skip whole ASCII spans, so we never split a multi-byte character.
        let ch_len = utf8_char_len(bytes[i]);
        out.push_str(&text[i..(i + ch_len).min(text.len())]);
        i += ch_len;
    }
    *text = out;
}

/// Length in bytes of the UTF-8 character starting with `b`.
fn utf8_char_len(b: u8) -> usize {
    match b {
        0x00..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    }
}

/// Roughly how many tokens a set of chat messages will become.
///
/// **An estimate, and deliberately a cheap one.** It exists for the routing
/// decision in `scheduler::parallax::vertex_cost`, which needs to tell a
/// ten-token request from a ten-thousand-token one and does not care about the
/// difference between 4900 and 5000. Tokenising properly here would mean
/// loading a tokenizer the coordinator may not have (a network-only model has
/// no local `gguf_header.bin`) and doing the work twice, on the latency path,
/// for a number that only has to land in the right order of magnitude.
///
/// Four bytes per token is the usual rule of thumb for English prose through a
/// BPE vocabulary; code and non-Latin scripts run denser, so this UNDER-counts
/// them, which biases toward the old behaviour rather than toward a surprise.
/// The per-message constant covers the role markers and separators every chat
/// template adds.
///
/// Images are counted at a fixed per-image cost: a CLIP-style encoder emits a
/// fixed number of embeddings regardless of the picture's size: 576 for LLaVA
/// (one per patch, `inference::vision`), the most of the families read here.
///
/// **Tool definitions are part of the prompt.** Chat templates render them into
/// it — as JSON, one per tool — and an agent harness sends dozens, so they can
/// be the larger half of what the model reads. They were not counted until
/// 2026-09-26, which priced every agent request as shorter than it is: its
/// prefill, and the KV cache a peer is asked to hold for it. Measured at the
/// same four bytes per token as the text, which is a fair rate for JSON.
pub fn estimate_prompt_tokens(
    messages: &[crate::types::ChatMessage],
    tools: Option<&[serde_json::Value]>,
) -> u32 {
    const BYTES_PER_TOKEN: usize = 4;
    const PER_MESSAGE_OVERHEAD_TOKENS: usize = 4;
    const TOKENS_PER_IMAGE: usize = 576;

    let mut total = 0usize;
    for m in messages {
        total += m.content.len() / BYTES_PER_TOKEN;
        total += PER_MESSAGE_OVERHEAD_TOKENS;
        total += m.images.len() * TOKENS_PER_IMAGE;
    }
    for tool in tools.unwrap_or_default() {
        total += tool.to_string().len() / BYTES_PER_TOKEN;
    }
    total.min(u32::MAX as usize) as u32
}

#[cfg(test)]
mod prompt_estimate_tests {
    use super::estimate_prompt_tokens;
    use crate::types::{ChatMessage, Role};

    fn msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            content: content.to_string(),
            images: vec![],
            tool: Default::default(),
        }
    }

    /// The only property routing needs: the bands are far apart and ordered.
    /// Pinning an exact count would pin the rule of thumb rather than the
    /// behaviour that matters.
    #[test]
    fn a_long_prompt_is_ordered_far_above_a_short_one() {
        let short = estimate_prompt_tokens(&[msg("hi")], None);
        let agentic = estimate_prompt_tokens(&[msg(&"x".repeat(20_000))], None);
        assert!(short < 20, "a two-character prompt should be tiny: {short}");
        assert!(
            agentic > 20 * short,
            "a 20 KB prompt must land in a different band from a 2-byte one: \
             {agentic} vs {short}"
        );
    }

    /// Empty input must be zero, not a per-message constant applied to nothing
    /// — `vertex_cost` reads zero as "no prefill to price".
    #[test]
    fn no_messages_is_zero() {
        assert_eq!(estimate_prompt_tokens(&[], None), 0);
    }

    /// An image carries real prefill cost even when its message has no text.
    #[test]
    fn an_image_costs_more_than_its_empty_caption() {
        let mut with_image = msg("");
        with_image.images.push(crate::types::ImageData {
            rgb_bytes: vec![],
            width: 1,
            height: 1,
        });
        assert!(estimate_prompt_tokens(&[with_image], None) > 500);
    }

    /// An agent's tool definitions are rendered into its prompt, so they are
    /// counted — the field report's harnesses sent 4,100-14,400 tokens a turn,
    /// much of it tool schemas the estimate used to ignore entirely.
    #[test]
    fn tool_definitions_count_toward_the_prompt() {
        let tool = serde_json::json!({
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "x".repeat(4_000),
                "parameters": {"type": "object", "properties": {}}
            }
        });
        let bare = estimate_prompt_tokens(&[msg("hi")], None);
        let with_tools = estimate_prompt_tokens(&[msg("hi")], Some(&[tool.clone(), tool]));
        assert!(
            with_tools >= bare + 2_000,
            "two 4 KB tool schemas are ~2,000 tokens of prompt: {with_tools} vs {bare}"
        );
        assert_eq!(
            estimate_prompt_tokens(&[msg("hi")], Some(&[])),
            bare,
            "no tools and an empty list mean the same thing"
        );
    }
}

#[cfg(test)]
mod stop_trim_tests {
    use super::trim_trailing_partial_stop;

    /// A marker decoded across several tokens leaks its earlier pieces: the
    /// worker withholds only the token that COMPLETES the match, so the
    /// consumer is left holding e.g. `<|im_end|` when the stop is `<|im_end|>`.
    /// Reported three releases running against a Llama-3.2 GGUF that emits
    /// ChatML markers (external report 2026-07-25).
    #[test]
    fn trims_a_partial_marker_left_by_multi_token_decoding() {
        let stops = vec!["<|im_end|>".to_string()];
        for partial in ["<|im_end|", "<|im_end", "<|im_", "<|"] {
            let mut text = format!("Hello{partial}");
            trim_trailing_partial_stop(&mut text, &stops);
            assert_eq!(text, "Hello", "failed to trim {partial:?}");
        }
    }

    /// A bare partial with no preceding content must trim to empty rather than
    /// returning the marker fragment as the whole reply — the exact symptom
    /// reported (`content: "<|im_end|"`).
    #[test]
    fn a_reply_that_is_only_a_partial_marker_becomes_empty() {
        let stops = vec!["<|im_end|>".to_string()];
        let mut text = "<|im_end|".to_string();
        trim_trailing_partial_stop(&mut text, &stops);
        assert_eq!(text, "");
    }

    /// Ordinary text ending in characters that merely START a stop marker must
    /// not be over-trimmed beyond the marker prefix itself.
    #[test]
    fn does_not_touch_text_without_a_marker_prefix() {
        let stops = vec!["<|im_end|>".to_string()];
        for keep in ["Hello there", "2 < 3", "a|b", ""] {
            let mut text = keep.to_string();
            trim_trailing_partial_stop(&mut text, &stops);
            assert_eq!(text, keep, "over-trimmed {keep:?}");
        }
    }

    /// No stop strings configured must be a no-op.
    #[test]
    fn empty_stop_list_is_a_no_op() {
        let mut text = "<|im_end|".to_string();
        trim_trailing_partial_stop(&mut text, &[]);
        assert_eq!(text, "<|im_end|");
    }

    /// A caller's stop in any script is trimmed as a partial by CHARACTERS,
    /// and a reply that merely shares no prefix with it is left alone. The byte
    /// version sliced `"。"` (3 bytes) at 2 and panicked, on every path, after
    /// the whole reply had been generated.
    #[test]
    fn a_stop_ending_in_a_multibyte_character_neither_panics_nor_over_trims() {
        let stops = vec!["\n用户".to_string(), "。".to_string(), "Ende💬".to_string()];
        for (reply, want) in [
            ("回答\n用", "回答"),
            ("回答\n", "回答"),
            ("Tschüss Ende", "Tschüss "),
            ("回答完成", "回答完成"),
            ("plain ascii", "plain ascii"),
        ] {
            let mut text = reply.to_string();
            trim_trailing_partial_stop(&mut text, &stops);
            assert_eq!(text, want, "reply {reply:?}");
        }
    }
}

#[cfg(test)]
mod control_token_strip_tests {
    use super::strip_control_token_artifacts;

    /// The reported spellings, including the transposed one that no
    /// prefix-based trim can catch because `<|im_end>` differs from
    /// `<|im_end|` at position 8.
    #[test]
    fn strips_every_reported_spelling() {
        for (input, want) in [
            ("<|im_end|>", ""),
            ("<|im_end|", ""),
            ("<|im_end>|", ""),
            ("<|im_end", ""),
            ("<|im_end>|Hello", "Hello"),
            ("<|im_end|>hello\n<|im_start|", "hello\n"),
            ("Hello<|eot_id|>", "Hello"),
            ("<|python_tag|>{\"a\":1}", "{\"a\":1}"),
            // Observed live: a mangled marker carrying a third terminator
            // char left a stray `>` mid-sentence.
            ("The ocean<|im_end|>> is vast", "The ocean is vast"),
            ("<|im_end|>>>x", "x"),
        ] {
            let mut got = input.to_string();
            strip_control_token_artifacts(&mut got);
            assert_eq!(got, want, "input {input:?}");
        }
    }

    /// Ordinary content must survive untouched — including text that merely
    /// contains `<` or `|`, and a user's own angle-pipe construct that does not
    /// name a known control token.
    #[test]
    fn leaves_ordinary_content_alone() {
        for keep in [
            "Hello there",
            "2 < 3 and 4 | 5",
            "if (a<|b) {}",
            "<|my_custom_thing|>",
            "a <|not_a_marker|> b",
            "",
        ] {
            let mut got = keep.to_string();
            strip_control_token_artifacts(&mut got);
            assert_eq!(got, keep, "over-stripped {keep:?}");
        }
    }

    /// Phi-3/3.5/4 markers. On a GPT-2-BPE vocabulary these decode to their
    /// literal characters, so a reply carried a visible `<|end|>` and fabricated
    /// `<|user|>` / `<|assistant|>` turns (reported in the field on Phi-4-mini,
    /// 2026-09-10). Scrubbing them is safe even for the harmony formats where
    /// `<|end|>` must NOT stop generation: removing a control marker from
    /// visible text and ending the reply at it are different decisions.
    #[test]
    fn strips_phi_turn_markers() {
        let cases = [
            ("Bonjour<|end|>", "Bonjour"),
            (
                "answer<|end|>\n<|user|>next question",
                "answer\nnext question",
            ),
            ("a<|assistant|>b<|system|>c", "abc"),
        ];
        for (input, want) in cases {
            let mut got = input.to_string();
            strip_control_token_artifacts(&mut got);
            assert_eq!(got, want, "input {input:?}");
        }
    }

    /// The scrubber matches a whole marker name, so the longer markers that
    /// share a prefix with `end` must keep working.
    #[test]
    fn a_short_marker_name_does_not_shadow_a_longer_one() {
        let mut got = "a<|end_header_id|>b<|endoftext|>c<|end_of_turn|>d".to_string();
        strip_control_token_artifacts(&mut got);
        assert_eq!(got, "abcd");
    }

    /// Multi-byte characters must not be split when scanning byte-wise.
    #[test]
    fn preserves_multibyte_text() {
        for keep in ["héllo wörld", "日本語のテキスト", "emoji 🎉 here"] {
            let mut got = keep.to_string();
            strip_control_token_artifacts(&mut got);
            assert_eq!(got, keep);
        }
        let mut mixed = "日本<|im_end|>語".to_string();
        strip_control_token_artifacts(&mut mixed);
        assert_eq!(mixed, "日本語");
    }
}

#[cfg(test)]
mod finalize_reply_text_tests {
    use super::{
        emptied_reply_evidence, finalize_reply_text, reply_ended_itself_early,
        stop_sequence_ate_the_reply, EMPTIED_REPLY_LOG_CHARS, STOP_ATE_THE_REPLY_TOKENS,
    };

    fn stops() -> Vec<String> {
        vec!["<|im_end|>".to_string()]
    }

    /// The ordering rule, asserted once. A marker emitted BEFORE the answer
    /// must be removed without taking the answer with it — scrubbing has to
    /// happen before truncation. Getting this backwards turned a visible leak
    /// into an empty reply, and the distributed path was still doing it in
    /// that order after the scrub was added "everywhere".
    #[test]
    fn marker_before_the_answer_keeps_the_answer() {
        let mut t = "<|im_end|>Red, blue and yellow.".to_string();
        let matched =
            finalize_reply_text(&mut t, &stops(), crate::inference::PromptOpened::Nothing).stop;
        assert_eq!(t, "Red, blue and yellow.");
        assert_eq!(matched, None, "a scrubbed artifact is not a stop match");
    }

    /// The condition the "reply is empty after finalisation" warning keys on:
    /// the model DID generate text and every character of it was removed. The
    /// caller then returns HTTP 200 with empty content and
    /// `finish_reason: "stop"`, which no client can tell from a real answer —
    /// so this case has to be recognisable in the log.
    /// The emptied-reply counter moves when a reply is emptied. Delta-based,
    /// because the counter is process-wide and other tests finalise text in
    /// parallel; that the counter does NOT move on an ordinary reply is pinned
    /// through `emptied_reply_evidence`, which is the one condition it keys on.
    #[test]
    fn an_emptied_reply_is_counted() {
        use super::EMPTY_REPLIES_TOTAL;
        use std::sync::atomic::Ordering;
        let before = EMPTY_REPLIES_TOTAL.load(Ordering::Relaxed);
        let mut t = String::from("<|im_end|>");
        let _ = finalize_reply_text(&mut t, &stops(), crate::inference::PromptOpened::Nothing);
        assert!(t.trim().is_empty(), "control: the reply was emptied");
        assert!(
            EMPTY_REPLIES_TOTAL.load(Ordering::Relaxed) > before,
            "an emptied reply must move swarmllm_empty_replies_total"
        );
    }

    #[test]
    fn a_reply_of_nothing_but_a_marker_finalises_to_empty() {
        let mut t = "<|im_end|>".to_string();
        let _ = finalize_reply_text(&mut t, &stops(), crate::inference::PromptOpened::Nothing);
        assert!(t.trim().is_empty(), "expected the marker to leave nothing");

        // ...and a stop matching at position 0 does the same thing by a
        // different route, which is the prompt-fault shape rather than the
        // template-fault one.
        let mut t = "\nUser: who asked?".to_string();
        let _ = finalize_reply_text(
            &mut t,
            &["\nUser:".to_string()],
            crate::inference::PromptOpened::Nothing,
        );
        assert!(t.trim().is_empty());
    }
    /// The report has to name what was removed — that text is the entire
    /// diagnostic value, since it distinguishes a template fault (a leaked
    /// marker) from a prompt fault (a stop matching at position 0).
    #[test]
    fn a_stop_that_ate_the_reply_is_reported() {
        // The reported shape: one token, then a stop sequence ended it.
        assert!(stop_sequence_ate_the_reply(1, Some(" tests")));
        assert!(stop_sequence_ate_the_reply(0, Some("Unit")));
        assert!(stop_sequence_ate_the_reply(
            STOP_ATE_THE_REPLY_TOKENS,
            Some("x")
        ));
    }

    #[test]
    fn a_real_answer_cut_by_a_stop_is_not_reported() {
        // A stop trimming the tail of a genuine answer is the feature working.
        // Warning about it would train operators to ignore the warning, which is
        // how the case that matters gets missed.
        assert!(!stop_sequence_ate_the_reply(
            STOP_ATE_THE_REPLY_TOKENS + 1,
            Some("\nUser:")
        ));
        assert!(!stop_sequence_ate_the_reply(80, Some("\nUser:")));
    }

    #[test]
    fn a_caller_who_asked_for_one_token_is_not_a_fault() {
        // `max_tokens: 1` legitimately yields one token. Warning there would
        // fire on every deliberate single-token request — the classic way a
        // useful warning becomes noise and stops being read.
        assert!(!reply_ended_itself_early(1, 1, None));
        assert!(!reply_ended_itself_early(0, 0, None));
        assert!(!reply_ended_itself_early(2, 2, None));
    }

    #[test]
    fn a_model_stopping_well_inside_its_budget_is_reported() {
        // The reported shape: one token out of a hundred asked for, nothing cut
        // it off, so the model emitted end-of-turn straight away.
        assert!(reply_ended_itself_early(1, 100, None));
        assert!(reply_ended_itself_early(0, 100, None));
    }

    #[test]
    fn a_caller_who_asked_for_something_short_is_not_reported() {
        // Reported by an external tester chasing a real single-token bug: this
        // warning fired twice on their own probes ("say OK in one word",
        // max_tokens 10) where one token WAS the answer. The old rule only
        // required completion < max, which is true of nearly every request, so
        // the signal was diluted by exactly the traffic it had to be trusted
        // against.
        assert!(!reply_ended_itself_early(1, 10, None));
        assert!(!reply_ended_itself_early(1, 32, None));
        assert!(!reply_ended_itself_early(1, 1, None));

        // ...while the shape that was the genuine fault still reports: an
        // agentic client budgeting thousands and getting one token back.
        assert!(reply_ended_itself_early(1, 4096, None));
    }

    #[test]
    fn a_stop_match_is_the_other_reports_business() {
        // When a stop sequence matched, that is the explanation and it is
        // reported separately — this branch must stay quiet so one event never
        // produces two different warnings.
        assert!(!reply_ended_itself_early(1, 100, Some(" tests")));
    }

    #[test]
    fn a_full_length_reply_is_not_reported() {
        assert!(!reply_ended_itself_early(100, 100, None));
        assert!(!reply_ended_itself_early(64, 100, None));
    }

    #[test]
    fn a_short_reply_that_nothing_cut_off_is_not_reported() {
        // Terse but voluntary. `max_tokens: 1` is a legitimate request, and a
        // model answering "Yes." is not a fault to report.
        assert!(!stop_sequence_ate_the_reply(1, None));
        assert!(!stop_sequence_ate_the_reply(0, None));
    }

    #[test]
    fn an_emptied_reply_is_reported_with_what_was_removed() {
        let evidence = emptied_reply_evidence(true, "", "<|im_end|>")
            .expect("generated text that was wholly removed must be reported");
        assert!(
            evidence.contains("im_end"),
            "the report must name what was removed, got: {evidence}"
        );
    }

    /// The counterpart, and the reason the report is conditional: an ordinary
    /// reply must stay silent. A diagnostic that fires on healthy traffic is
    /// noise, and this runs on every completion.
    #[test]
    fn an_ordinary_reply_reports_nothing() {
        assert_eq!(
            emptied_reply_evidence(
                true,
                "Rivers carve landscapes.",
                "Rivers carve landscapes.<|im_end|>"
            ),
            None,
            "a normal reply must not be reported"
        );
        // A model that generated nothing at all is a different condition and is
        // already visible to the caller as zero tokens — it must not report here.
        assert_eq!(
            emptied_reply_evidence(false, "", ""),
            None,
            "no generation must not be reported"
        );
    }

    /// The whole path, not just the decision: a reply that is nothing but a
    /// marker must finalise to empty AND be reportable, which is what ties the
    /// pure helper above to what `finalize_reply_text` actually does.
    #[test]
    fn the_finalised_reply_and_the_report_agree() {
        let mut t = "<|im_end|>".to_string();
        let before: String = t.chars().take(EMPTIED_REPLY_LOG_CHARS).collect();
        let had_content = !t.trim().is_empty();
        let _ = finalize_reply_text(&mut t, &stops(), crate::inference::PromptOpened::Nothing);
        assert!(t.is_empty(), "the marker-only reply must finalise to empty");
        assert!(
            emptied_reply_evidence(had_content, &t, &before).is_some(),
            "and that must be a reportable condition"
        );
    }

    /// The snapshot taken for that warning is built with `chars().take(..)`, so
    /// it must not split a multi-byte character. A panic here would abort a
    /// perfectly good reply on any non-ASCII generation.
    #[test]
    fn finalising_multibyte_text_does_not_panic() {
        let mut t = "日本語のテキストです。".repeat(40);
        let _ = finalize_reply_text(&mut t, &stops(), crate::inference::PromptOpened::Nothing);
        assert!(t.starts_with("日本語"), "content must survive untouched");

        let mut emoji = "🙂🙃".repeat(300);
        emoji.push_str("<|im_end|>");
        let _ = finalize_reply_text(
            &mut emoji,
            &stops(),
            crate::inference::PromptOpened::Nothing,
        );
        assert!(emoji.starts_with("🙂"));
    }

    /// A genuine stop still ends the reply, and is reported so the caller can
    /// set finish_reason.
    #[test]
    fn genuine_stop_truncates_and_is_reported() {
        let mut t = "Answer.<|im_end|>trailing".to_string();
        // Not a scrubbable artifact case: prove truncation via a stop the
        // scrubber does not know about.
        let s = vec!["\nUser:".to_string()];
        let mut t2 = "Answer.\nUser: next question".to_string();
        assert_eq!(
            finalize_reply_text(&mut t2, &s, crate::inference::PromptOpened::Nothing).stop,
            Some("\nUser:".to_string())
        );
        assert_eq!(t2, "Answer.");
        // The control-token form is scrubbed rather than truncated.
        let _ = finalize_reply_text(&mut t, &stops(), crate::inference::PromptOpened::Nothing);
        assert_eq!(t, "Answer.trailing");
    }

    /// Newlines stranded by a removed marker are dropped — a step the
    /// distributed path never had.
    #[test]
    fn strands_no_leading_newlines() {
        let mut t = "<|im_end|>\n\nHello".to_string();
        let _ = finalize_reply_text(&mut t, &stops(), crate::inference::PromptOpened::Nothing);
        assert_eq!(t, "Hello");
    }

    /// Safe to run twice with different stop sets — the router applies
    /// template-derived stops after the executor has applied the caller's.
    #[test]
    fn is_idempotent_across_repeated_calls() {
        let mut once = "<|im_end|>\nHi there".to_string();
        let _ = finalize_reply_text(&mut once, &[], crate::inference::PromptOpened::Nothing);
        let mut twice = once.clone();
        let _ = finalize_reply_text(
            &mut twice,
            &stops(),
            crate::inference::PromptOpened::Nothing,
        );
        assert_eq!(once, twice);
        assert_eq!(twice, "Hi there");
    }

    /// No stops configured still scrubs and cleans up.
    #[test]
    fn empty_stops_still_scrubs() {
        let mut t = "<|eot_id|>\nAnswer".to_string();
        assert_eq!(
            finalize_reply_text(&mut t, &[], crate::inference::PromptOpened::Nothing).stop,
            None
        );
        assert_eq!(t, "Answer");
    }
}

/// The vendored multi-row quantized matmul kernels (`GgmlType::vec_dot_rows`,
/// Q4_K, Q6_K, Q2_K and Q3_K four rows per pass) promise bit-identical results to the
/// per-row upstream ordering — each row keeps its own accumulators and sees
/// the single-row kernel's operations in the same order. `examples/qmatmul_bench`
/// asserts the same thing, but an example is not run by CI; this is.
#[cfg(test)]
mod qmatmul_exactness_tests {
    use candle_core::quantized::k_quants::{
        matmul, BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ6K,
        BlockQ8_0, GgmlType,
    };

    /// The upstream algorithm: batch row outer, one `vec_dot` per element.
    fn reference<T: GgmlType>(
        (m, k, n): (usize, usize, usize),
        lhs: &[f32],
        rhs_t: &[T],
    ) -> Vec<f32> {
        let kb = k / T::BLCK_SIZE;
        let mut lhs_b = vec![T::VecDotType::zeros(); m * kb];
        for r in 0..m {
            T::VecDotType::from_float(&lhs[r * k..(r + 1) * k], &mut lhs_b[r * kb..(r + 1) * kb]);
        }
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            let row = &lhs_b[r * kb..(r + 1) * kb];
            for c in 0..n {
                out[r * n + c] = T::vec_dot(k, &rhs_t[c * kb..(c + 1) * kb], row);
            }
        }
        out
    }

    fn check<T: GgmlType>(label: &str) {
        let (k, n) = (512usize, 96usize);
        let kb = k / T::BLCK_SIZE;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0
        };
        let src: Vec<f32> = (0..n * k).map(|_| noise()).collect();
        let mut rhs = vec![T::zeros(); n * kb];
        T::from_float(&src, &mut rhs);
        // Row counts that exercise the multi-row passes, their tails, the m=1
        // decode path and a whole row block.
        for m in [1usize, 2, 3, 4, 5, 7, 8, 9, 13, 130] {
            let lhs: Vec<f32> = (0..m * k).map(|_| noise()).collect();
            let mut got = vec![0f32; m * n];
            matmul((m, k, n), &lhs, &rhs, &mut got).unwrap();
            let want = reference((m, k, n), &lhs, &rhs);
            let bad = got
                .iter()
                .zip(&want)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                bad,
                0,
                "{label} m={m}: {bad}/{} elements differ from the per-row ordering",
                m * n
            );
        }
    }

    #[test]
    fn multi_row_kernels_are_bit_identical_to_the_per_row_ordering() {
        check::<BlockQ4K>("Q4_K");
        check::<BlockQ6K>("Q6_K");
        check::<BlockQ2K>("Q2_K");
        check::<BlockQ3K>("Q3_K");
        check::<BlockQ8_0>("Q8_0");
        check::<BlockQ4_0>("Q4_0");
        check::<BlockQ5_0>("Q5_0");
        check::<BlockQ4_1>("Q4_1");
        check::<BlockQ5_1>("Q5_1");
    }

    /// The 32-value formats' SIMD dot products (Q5_0, Q4_1 and Q5_1 had none
    /// before 2026-10-20) against the scalar loop over the same blocks. Not
    /// bit-identical — eight lanes sum in another order — but a misread fifth
    /// bit or offset misses by whole units of the block scale, not by rounding.
    fn agrees_with_the_scalar_loop<T: GgmlType>(label: &str) {
        let k = 896usize; // 28 blocks: the 0.5B drafter's width
        let kb = k / T::BLCK_SIZE;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut noise = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0
        };
        for trial in 0..64 {
            let w: Vec<f32> = (0..k).map(|_| noise()).collect();
            let a: Vec<f32> = (0..k)
                .map(|_| noise() + if trial % 2 == 0 { 0.3 } else { 0.0 })
                .collect();
            let mut col = vec![T::zeros(); kb];
            T::from_float(&w, &mut col);
            let mut row = vec![T::VecDotType::zeros(); kb];
            T::VecDotType::from_float(&a, &mut row);
            let (got, want) = (T::vec_dot(k, &col, &row), T::vec_dot_unopt(k, &col, &row));
            assert!(
                (got - want).abs() <= 1e-4 * (1.0 + want.abs()),
                "{label} trial {trial}: SIMD {got} against scalar {want}"
            );
        }
    }

    #[test]
    fn the_32_value_dot_products_agree_with_the_scalar_loop() {
        agrees_with_the_scalar_loop::<BlockQ8_0>("Q8_0");
        agrees_with_the_scalar_loop::<BlockQ4_0>("Q4_0");
        agrees_with_the_scalar_loop::<BlockQ5_0>("Q5_0");
        agrees_with_the_scalar_loop::<BlockQ4_1>("Q4_1");
        agrees_with_the_scalar_loop::<BlockQ5_1>("Q5_1");
    }
}

/// MXFP4 (GGML type 39, gpt-oss's experts) in the vendored candle, checked
/// against llama.cpp's own code: `examples/mxfp4_reference.c` runs
/// `dequantize_row_mxfp4`, the CPU `vec_dot` and `quantize_row_mxfp4_ref` on the
/// inputs [`reference_inputs`] makes (the same xorshift32, step for step) and
/// printed the figures asserted here (llama.cpp master, AVX2 build, 2026-11-02).
#[cfg(test)]
mod mxfp4_tests {
    use candle_core::quantized::k_quants::{BlockMxfp4, BlockQ8_0, GgmlType};
    use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QTensor};
    use candle_core::{Device, Module, Tensor};

    struct XorShift(u32);
    impl XorShift {
        fn next(&mut self) -> u32 {
            let mut s = self.0;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            self.0 = s;
            s
        }
    }

    /// Blocks from the bytes a GGUF holds them as.
    fn blocks<T: GgmlType>(bytes: &[u8]) -> Vec<T> {
        let size = std::mem::size_of::<T>();
        assert_eq!(bytes.len() % size, 0);
        let n = bytes.len() / size;
        let mut out = vec![T::zeros(); n];
        // SAFETY: both types are `repr(C)` plain data, and `out` holds exactly
        // `bytes.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len())
        };
        out
    }

    fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Nine MXFP4 blocks (the first two with the scales below the normal range),
    /// nine Q8_0 blocks, and 192 values to quantize.
    fn reference_inputs() -> (Vec<BlockMxfp4>, Vec<BlockQ8_0>, Vec<f32>) {
        let mut r = XorShift(0x1234_5678);
        let mut x = Vec::new();
        for b in 0..9 {
            let e = match b {
                0 => 0u8,
                1 => 1,
                _ => (110 + r.next() % 30) as u8,
            };
            x.push(e);
            x.extend((0..16).map(|_| (r.next() & 0xFF) as u8));
        }
        let mut y = Vec::new();
        for _ in 0..9 {
            y.extend_from_slice(&((0x2000 + r.next() % 0x2000) as u16).to_le_bytes());
            y.extend((0..32).map(|_| (r.next() & 0xFF) as u8));
        }
        let mut src: Vec<f32> = (0..128)
            .map(|_| ((r.next() % 20001) as i32 - 10000) as f32 / 997.0)
            .collect();
        src.extend(std::iter::repeat_n(0.0, 32));
        src.extend((0..32).map(|_| ((r.next() % 2001) as i32 - 1000) as f32 / 65536.0));
        (blocks(&x), blocks(&y), src)
    }

    #[test]
    fn mxfp4_reads_multiplies_and_quantizes_as_llama_cpp_does() {
        let (x, y, src) = reference_inputs();

        let mut deq = vec![0f32; 9 * 32];
        BlockMxfp4::to_float(&x, &mut deq);
        let bits = |v: &[f32]| fnv1a(v.iter().flat_map(|f| f.to_bits().to_le_bytes()));
        assert_eq!(bits(&deq), 0x62d0_a978_220d_a17a, "dequantized values");
        // Spot values: a scale below the normal range (block 0) reads exactly.
        assert_eq!(deq[0].to_bits(), 0x00c0_0000);

        let mut q = vec![BlockMxfp4::zeros(); 6];
        BlockMxfp4::from_float(&src, &mut q);
        let q_bytes: Vec<u8> = q
            .iter()
            .flat_map(|b| {
                let mut v = vec![0u8; 17];
                // SAFETY: a `repr(C)` block of 17 bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        b as *const BlockMxfp4 as *const u8,
                        v.as_mut_ptr(),
                        17,
                    )
                };
                v
            })
            .collect();
        assert_eq!(
            fnv1a(q_bytes.iter().copied()),
            0x27ff_17c1_8192_1ddc,
            "quantized bytes"
        );

        // The dot product. llama.cpp's figures are its AVX2 kernel's, whose
        // order this build's AVX2 kernel follows; the scalar loop sums in
        // another order, so it is held to the rounding only.
        let (want8, want9) = (f32::from_bits(0xc560_81d8), f32::from_bits(0x448f_3078));
        let dot8 = BlockMxfp4::vec_dot(8 * 32, &x[..8], &y[..8]);
        let dot9 = BlockMxfp4::vec_dot(9 * 32, &x, &y);
        #[cfg(target_feature = "avx2")]
        assert_eq!(
            (dot8.to_bits(), dot9.to_bits()),
            (want8.to_bits(), want9.to_bits()),
            "{dot8} {dot9}"
        );
        for (got, want) in [
            (dot8, want8),
            (dot9, want9),
            (BlockMxfp4::vec_dot_unopt(8 * 32, &x[..8], &y[..8]), want8),
            (BlockMxfp4::vec_dot_unopt(9 * 32, &x, &y), want9),
        ] {
            assert!(
                (got - want).abs() <= 1e-5 * want.abs(),
                "{got} against llama.cpp's {want}"
            );
        }
    }

    /// The low nibble of byte `j` is element `j`, the high one element `j + 16`,
    /// and a value is the doubled E2M1 table entry times half the scale.
    #[test]
    fn an_mxfp4_block_reads_its_nibbles_in_llama_cpps_order() {
        // e = 127: a scale of 1, so half of it is 0.5.
        let mut bytes = vec![127u8];
        bytes.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let x: Vec<BlockMxfp4> = blocks(&bytes);
        let mut out = [0f32; 32];
        BlockMxfp4::to_float(&x, &mut out);
        let table = [
            0., 1., 2., 3., 4., 6., 8., 12., 0., -1., -2., -3., -4., -6., -8., -12.,
        ];
        for j in 0..16 {
            assert_eq!(out[j], table[j] * 0.5, "element {j}");
            assert_eq!(out[j + 16], table[15 - j] * 0.5, "element {}", j + 16);
        }
    }

    /// A GGUF names MXFP4 by type 39, and a weight stored so multiplies on the
    /// processor as its dequantized values do.
    #[test]
    fn an_mxfp4_weight_survives_a_gguf_and_multiplies_like_its_values() {
        let dev = Device::Cpu;
        let (n, k) = (48usize, 256usize);
        let mut r = XorShift(0x0bad_cafe);
        let w: Vec<f32> = (0..n * k)
            .map(|_| (r.next() % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let w = Tensor::from_vec(w, (n, k), &dev).unwrap();
        let q = QTensor::quantize(&w, GgmlDType::Mxfp4).unwrap();
        assert_eq!(q.storage_size_in_bytes(), n * k / 32 * 17);

        let mut file = std::io::Cursor::new(Vec::new());
        gguf_file::write(&mut file, &[], &[("w", &q)]).unwrap();
        // ggml's tensor-info entry: name, dims, then the type as a u32.
        let raw = file.get_ref();
        let at = raw.windows(1).position(|s| s == b"w").unwrap();
        let type_at = at + 1 + 4 + 2 * 8;
        assert_eq!(
            u32::from_le_bytes(raw[type_at..type_at + 4].try_into().unwrap()),
            39
        );
        file.set_position(0);
        let content = gguf_file::Content::read(&mut file).unwrap();
        assert_eq!(content.tensor_infos["w"].ggml_dtype, GgmlDType::Mxfp4);
        let back = content.tensor(&mut file, "w", &dev).unwrap();
        let values = back.dequantize(&dev).unwrap();
        assert_eq!(
            values.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            q.dequantize(&dev)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );

        for rows in [1usize, 3, 4, 9] {
            let a: Vec<f32> = (0..rows * k)
                .map(|_| (r.next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let a = Tensor::from_vec(a, (rows, k), &dev).unwrap();
            let got = QMatMul::from_qtensor(content.tensor(&mut file, "w", &dev).unwrap())
                .unwrap()
                .forward(&a)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let want = a
                .matmul(&values.t().unwrap())
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            // The activations are rounded to Q8_0 on the way in: about 0.4% a
            // value, summed over 256 products of unit size.
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= 0.25, "{rows} rows: {g} against {w}");
            }
        }
    }

    /// The real file. `SWARMLLM_GPT_OSS_HEADER` names the opening bytes of
    /// ggml-org's `gpt-oss-20b-MXFP4.gguf` (its header ends 13.0 MB in; a range
    /// read of 16 MB holds it): the header parses, and exactly the three expert
    /// stacks of each of its 24 layers are MXFP4, 2,880 values (90 blocks) a row.
    #[test]
    #[ignore = "needs SWARMLLM_GPT_OSS_HEADER"]
    fn a_real_gpt_oss_header_parses_with_its_experts_mxfp4() {
        let path = std::env::var("SWARMLLM_GPT_OSS_HEADER").expect("SWARMLLM_GPT_OSS_HEADER");
        let content =
            crate::inference::split::read_gguf_header(std::path::Path::new(&path)).unwrap();
        let mxfp4: Vec<_> = content
            .tensor_infos
            .iter()
            .filter(|(_, t)| t.ggml_dtype == GgmlDType::Mxfp4)
            .collect();
        assert_eq!(mxfp4.len(), 24 * 3);
        for (name, t) in mxfp4 {
            assert!(
                ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                    .iter()
                    .any(|kind| name.ends_with(&format!(".{kind}.weight"))),
                "{name}"
            );
            assert_eq!(t.shape.dims(), &[32, 2880, 2880], "{name}");
        }
    }
}

/// IQ4_NL and IQ4_XS (GGML types 20 and 23, #404) in the vendored candle,
/// checked against llama.cpp's own code: `examples/iq4_reference.c` runs each
/// type's `to_float`, the CPU `vec_dot` and `from_float_ref` on the inputs
/// [`reference_inputs`] makes (the same xorshift32, step for step) and printed
/// the figures asserted here (llama.cpp master 4b1a27f, AVX2 build, 2026-11-04).
#[cfg(test)]
mod iq4_tests {
    use candle_core::quantized::k_quants::{BlockIq4Nl, BlockIq4Xs, BlockQ8K, BlockQ8_0, GgmlType};
    use candle_core::quantized::{gguf_file, GgmlDType, QMatMul, QTensor};
    use candle_core::{Device, Module, Tensor};

    struct XorShift(u32);
    impl XorShift {
        fn next(&mut self) -> u32 {
            let mut s = self.0;
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            self.0 = s;
            s
        }
        /// A Q8 byte as a quantizer writes one: within +-127.
        fn q8(&mut self) -> u8 {
            ((self.next() % 255) as i32 - 127) as i8 as u8
        }
    }

    /// Blocks from the bytes a GGUF holds them as.
    fn blocks<T: GgmlType>(bytes: &[u8]) -> Vec<T> {
        let size = std::mem::size_of::<T>();
        assert_eq!(bytes.len() % size, 0);
        let n = bytes.len() / size;
        let mut out = vec![T::zeros(); n];
        // SAFETY: every block type is `repr(C)` plain data, and `out` holds
        // exactly `bytes.len()` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len())
        };
        out
    }

    /// The bytes a GGUF would hold `blocks` as.
    fn bytes_of<T: GgmlType>(blocks: &[T]) -> Vec<u8> {
        let n = std::mem::size_of_val(blocks);
        let mut out = vec![0u8; n];
        // SAFETY: `repr(C)` plain data of exactly `n` bytes.
        unsafe { std::ptr::copy_nonoverlapping(blocks.as_ptr() as *const u8, out.as_mut_ptr(), n) };
        out
    }

    fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
        bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    fn bits(v: &[f32]) -> u64 {
        fnv1a(v.iter().flat_map(|f| f.to_bits().to_le_bytes()))
    }

    struct Inputs {
        nl: Vec<BlockIq4Nl>,
        q8_0: Vec<BlockQ8_0>,
        xs: Vec<BlockIq4Xs>,
        q8k: Vec<BlockQ8K>,
        nl_src: Vec<f32>,
        xs_src: Vec<f32>,
    }

    /// `n` values to quantize: a quarter spread, a quarter zeros, a quarter
    /// small, a quarter with one large outlier.
    fn to_quantize(r: &mut XorShift, n: usize) -> Vec<f32> {
        let quarter = n / 4;
        let mut src: Vec<f32> = (0..quarter)
            .map(|_| ((r.next() % 20001) as i32 - 10000) as f32 / 997.0)
            .collect();
        src.extend(std::iter::repeat_n(0.0, quarter));
        src.extend((0..quarter).map(|_| ((r.next() % 2001) as i32 - 1000) as f32 / 65536.0));
        src.extend((0..n - 3 * quarter).map(|_| ((r.next() % 2001) as i32 - 1000) as f32 / 500.0));
        src[3 * quarter + 5] = 37.5;
        src
    }

    /// Nine IQ4_NL blocks (the third with a negative scale) and nine Q8_0
    /// blocks; three IQ4_XS super-blocks (the second negative) and three Q8_K;
    /// then 256 values for IQ4_NL's quantizer and 1,024 for IQ4_XS's.
    fn reference_inputs() -> Inputs {
        let mut r = XorShift(0x1234_5678);
        let mut nl = Vec::new();
        for b in 0..9 {
            let mut d = (0x2000 + r.next() % 0x2000) as u16;
            if b == 2 {
                d |= 0x8000;
            }
            nl.extend_from_slice(&d.to_le_bytes());
            nl.extend((0..16).map(|_| (r.next() & 0xFF) as u8));
        }
        let mut q8_0 = Vec::new();
        for _ in 0..9 {
            q8_0.extend_from_slice(&((0x2000 + r.next() % 0x2000) as u16).to_le_bytes());
            q8_0.extend((0..32).map(|_| r.q8()));
        }
        let mut xs = Vec::new();
        for b in 0..3 {
            let mut d = (0x2000 + r.next() % 0x2000) as u16;
            if b == 1 {
                d |= 0x8000;
            }
            xs.extend_from_slice(&d.to_le_bytes());
            xs.extend_from_slice(&((r.next() & 0xFFFF) as u16).to_le_bytes());
            xs.extend((0..4 + 128).map(|_| (r.next() & 0xFF) as u8));
        }
        let mut q8k = Vec::new();
        for _ in 0..3 {
            let d = (1 + r.next() % 1000) as f32 / 16384.0;
            q8k.extend_from_slice(&d.to_le_bytes());
            let qs: Vec<u8> = (0..256).map(|_| r.q8()).collect();
            q8k.extend_from_slice(&qs);
            for g in 0..16 {
                let sum: i16 = qs[16 * g..16 * g + 16]
                    .iter()
                    .map(|&q| q as i8 as i16)
                    .sum();
                q8k.extend_from_slice(&sum.to_le_bytes());
            }
        }
        let nl_src = to_quantize(&mut r, 8 * 32);
        let xs_src = to_quantize(&mut r, 4 * 256);
        Inputs {
            nl: blocks(&nl),
            q8_0: blocks(&q8_0),
            xs: blocks(&xs),
            q8k: blocks(&q8k),
            nl_src,
            xs_src,
        }
    }

    /// `got` is llama.cpp's figure: bit for bit where this build runs the AVX2
    /// kernel, whose order llama.cpp's AVX2 kernel is; the scalar loop sums in
    /// another order, so it is held to the rounding only.
    fn agrees(got_avx: f32, got_scalar: f32, want_bits: u32) {
        let want = f32::from_bits(want_bits);
        #[cfg(target_feature = "avx2")]
        assert_eq!(
            got_avx.to_bits(),
            want_bits,
            "{got_avx} against llama.cpp's {want}"
        );
        for got in [got_avx, got_scalar] {
            assert!(
                (got - want).abs() <= 1e-5 * want.abs(),
                "{got} against llama.cpp's {want}"
            );
        }
    }

    #[test]
    fn iq4_nl_reads_multiplies_and_quantizes_as_llama_cpp_does() {
        let i = reference_inputs();
        let mut deq = vec![0f32; 9 * 32];
        BlockIq4Nl::to_float(&i.nl, &mut deq);
        assert_eq!(bits(&deq), 0xc2c8_7282_434c_70d0, "dequantized values");

        agrees(
            BlockIq4Nl::vec_dot(8 * 32, &i.nl[..8], &i.q8_0[..8]),
            BlockIq4Nl::vec_dot_unopt(8 * 32, &i.nl[..8], &i.q8_0[..8]),
            0xc697_e81a,
        );
        // An odd last block takes the scalar rule beside the pairs.
        agrees(
            BlockIq4Nl::vec_dot(9 * 32, &i.nl, &i.q8_0),
            BlockIq4Nl::vec_dot_unopt(9 * 32, &i.nl, &i.q8_0),
            0xc697_049b,
        );

        let mut q = vec![BlockIq4Nl::zeros(); 8];
        BlockIq4Nl::from_float(&i.nl_src, &mut q);
        let q = bytes_of(&q);
        assert_eq!(
            fnv1a(q.iter().copied()),
            0x22c4_2f21_de27_b589,
            "quantized bytes {:02x?}",
            &q[..18]
        );
    }

    #[test]
    fn iq4_xs_reads_multiplies_and_quantizes_as_llama_cpp_does() {
        let i = reference_inputs();
        let mut deq = vec![0f32; 3 * 256];
        BlockIq4Xs::to_float(&i.xs, &mut deq);
        assert_eq!(bits(&deq), 0xdf94_72b2_45e1_a527, "dequantized values");

        agrees(
            BlockIq4Xs::vec_dot(256, &i.xs[..1], &i.q8k[..1]),
            BlockIq4Xs::vec_dot_unopt(256, &i.xs[..1], &i.q8k[..1]),
            0x4383_bf94,
        );
        agrees(
            BlockIq4Xs::vec_dot(3 * 256, &i.xs, &i.q8k),
            BlockIq4Xs::vec_dot_unopt(3 * 256, &i.xs, &i.q8k),
            0x46d3_1cb6,
        );

        let mut q = vec![BlockIq4Xs::zeros(); 4];
        BlockIq4Xs::from_float(&i.xs_src, &mut q);
        let q = bytes_of(&q);
        assert_eq!(
            fnv1a(q.iter().copied()),
            0xaccd_4877_5f81_aaa7,
            "quantized bytes {:02x?}",
            &q[..18]
        );
    }

    /// The low nibble of byte `j` is element `j`, the high one element `j + 16`,
    /// and a value is the scale times the non-linear table's entry.
    #[test]
    fn an_iq4_nl_block_reads_its_nibbles_in_llama_cpps_order() {
        // d = 1.0 (f16 0x3c00).
        let mut bytes = 0x3c00u16.to_le_bytes().to_vec();
        bytes.extend((0..16u8).map(|j| j | ((15 - j) << 4)));
        let x: Vec<BlockIq4Nl> = blocks(&bytes);
        let mut out = [0f32; 32];
        BlockIq4Nl::to_float(&x, &mut out);
        let table = [
            -127., -104., -83., -65., -49., -35., -22., -10., 1., 13., 25., 38., 53., 69., 89.,
            113.,
        ];
        for j in 0..16 {
            assert_eq!(out[j], table[j], "element {j}");
            assert_eq!(out[j + 16], table[15 - j], "element {}", j + 16);
        }
    }

    /// An IQ4_XS super-block's eight runs of 32 each read their own 6-bit
    /// scale: the low four bits from `scales_l` (two runs a byte, the even
    /// run's in the low nibble), the high two from `scales_h` (two bits a run,
    /// run 0's lowest), less 32.
    #[test]
    fn an_iq4_xs_run_reads_its_own_scale() {
        // Run r's scale is 33 + 3r: the multiplier (scale − 32) is 1 + 3r.
        let ls: Vec<u16> = (0..8).map(|r| 33 + 3 * r).collect();
        let mut bytes = 0x3c00u16.to_le_bytes().to_vec(); // d = 1.0
        let h: u16 = (0..8).map(|r| (ls[r] >> 4) << (2 * r)).sum();
        bytes.extend_from_slice(&h.to_le_bytes());
        bytes.extend((0..4).map(|b| ((ls[2 * b] & 0xF) | ((ls[2 * b + 1] & 0xF) << 4)) as u8));
        // Every nibble 8: the table's entry 1.
        bytes.extend(std::iter::repeat_n(0x88u8, 128));
        let x: Vec<BlockIq4Xs> = blocks(&bytes);
        let mut out = [0f32; 256];
        BlockIq4Xs::to_float(&x, &mut out);
        for (j, v) in out.iter().enumerate() {
            assert_eq!(*v, (1 + 3 * (j / 32)) as f32, "element {j}");
        }
    }

    /// **A Q8_K block writes its largest activation at 127, never -128** — llama.cpp's
    /// `quantize_row_q8_K_ref` (`iscale = -127/max`). The IQ kernels multiply through
    /// `_mm256_sign_epi8`, which cannot negate -128: candle's older -128/max sent an
    /// IQ4_XS product 30% off beside a negative weight (the matmul test below).
    #[test]
    fn a_q8k_block_writes_its_largest_activation_at_127_as_llama_cpp_does() {
        for big in [1.0f32, -1.0] {
            let mut xs = vec![0.25f32; 256];
            xs[7] = big;
            xs[8] = -0.75;
            let mut q = vec![BlockQ8K::zeros(); 1];
            BlockQ8K::from_float(&xs, &mut q);
            let raw = bytes_of(&q);
            let d = f32::from_le_bytes(raw[0..4].try_into().unwrap());
            let qs: Vec<i8> = raw[4..260].iter().map(|b| *b as i8).collect();
            // iscale = -127 / max: the largest is always written -127 (the scale carries
            // the sign), so the old -128 / max wrote -128 in every block.
            assert_eq!(qs[7], -127, "max {big}");
            assert!(qs.iter().all(|v| *v != -128), "max {big}");
            assert_eq!(d, -big / 127.0);
            assert_eq!(qs[8], (127.0f32 * 0.75 * big.signum()).round() as i8);
        }
    }

    /// A GGUF names IQ4_NL by type 20 and IQ4_XS by 23, and a weight stored in
    /// either multiplies on the processor as its dequantized values do.
    #[test]
    fn an_iq4_weight_survives_a_gguf_and_multiplies_like_its_values() {
        let dev = Device::Cpu;
        let (n, k) = (48usize, 512usize);
        for (dtype, id, block_bytes, block) in [
            (GgmlDType::Iq4Nl, 20u32, 18, 32),
            (GgmlDType::Iq4Xs, 23, 136, 256),
        ] {
            let mut r = XorShift(0x0bad_cafe);
            let w: Vec<f32> = (0..n * k)
                .map(|_| (r.next() % 2001) as f32 / 1000.0 - 1.0)
                .collect();
            let w = Tensor::from_vec(w, (n, k), &dev).unwrap();
            let q = QTensor::quantize(&w, dtype).unwrap();
            assert_eq!(q.storage_size_in_bytes(), n * k / block * block_bytes);

            let mut file = std::io::Cursor::new(Vec::new());
            gguf_file::write(&mut file, &[], &[("w", &q)]).unwrap();
            // ggml's tensor-info entry: name, dims, then the type as a u32.
            let raw = file.get_ref();
            let at = raw.windows(1).position(|s| s == b"w").unwrap();
            let type_at = at + 1 + 4 + 2 * 8;
            assert_eq!(
                u32::from_le_bytes(raw[type_at..type_at + 4].try_into().unwrap()),
                id
            );
            file.set_position(0);
            let content = gguf_file::Content::read(&mut file).unwrap();
            assert_eq!(content.tensor_infos["w"].ggml_dtype, dtype);
            let values = content
                .tensor(&mut file, "w", &dev)
                .unwrap()
                .dequantize(&dev)
                .unwrap();
            let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            assert_eq!(
                flat(&values),
                flat(&q.dequantize(&dev).unwrap()),
                "{dtype:?}"
            );
            // The weights quantize close to what they were: 4.25-4.5 bits a value.
            let err = (&values - &w).unwrap().abs().unwrap().mean_all().unwrap();
            assert!(err.to_scalar::<f32>().unwrap() < 0.05, "{dtype:?}");

            for rows in [1usize, 3, 4, 9] {
                let a: Vec<f32> = (0..rows * k)
                    .map(|_| (r.next() % 2001) as f32 / 1000.0 - 1.0)
                    .collect();
                let a = Tensor::from_vec(a, (rows, k), &dev).unwrap();
                let got = QMatMul::from_qtensor(content.tensor(&mut file, "w", &dev).unwrap())
                    .unwrap()
                    .forward(&a)
                    .unwrap();
                let want = a.matmul(&values.t().unwrap()).unwrap();
                // The activations are rounded to Q8 on the way in: about 0.4% a
                // value, summed over 512 products of unit size.
                for (g, w) in flat(&got).iter().zip(flat(&want)) {
                    assert!(
                        (g - w).abs() <= 0.35,
                        "{dtype:?}, {rows} rows: {g} against {w}"
                    );
                }
            }
        }
    }

    /// A model holding IQ4 weights is one no card kernel of this build reads,
    /// so it is kept on the processor (`CpuReason::WeightFormatNeedsTheProcessor`);
    /// a model of K-quants is not.
    #[test]
    fn a_header_holding_iq4_is_one_the_card_cannot_read() {
        let dev = Device::Cpu;
        let w = Tensor::zeros((4, 256), candle_core::DType::F32, &dev).unwrap();
        let header = |dtype| {
            let q = QTensor::quantize(&w, dtype).unwrap();
            let mut file = std::io::Cursor::new(Vec::new());
            gguf_file::write(&mut file, &[], &[("blk.0.ffn_down.weight", &q)]).unwrap();
            file.set_position(0);
            gguf_file::Content::read(&mut file).unwrap()
        };
        use crate::inference::split::holds_weights_the_card_cannot_read as cannot;
        assert!(cannot(&header(GgmlDType::Iq4Nl)));
        assert!(cannot(&header(GgmlDType::Iq4Xs)));
        assert!(!cannot(&header(GgmlDType::Q4K)));
        assert!(!cannot(&header(GgmlDType::Mxfp4)));
    }
}

#[cfg(test)]
mod reasoning_block_tests {
    use super::{finalize_reply_text, take_leading_reasoning_block};

    /// A reasoning model writes its working out first. The answer is the
    /// answer; the scratchpad is not part of it.
    #[test]
    fn a_closed_reasoning_block_is_removed_and_the_answer_kept() {
        let mut t =
            "<think>\nThe user wants a greeting. Keep it short.\n</think>\n\nOK".to_string();
        take_leading_reasoning_block(&mut t);
        assert_eq!(t, "OK");
    }

    /// An UNTERMINATED block means the reply was cut off mid-reasoning, so the
    /// whole thing is scratchpad. Removing it returns an empty message, which
    /// the caller cannot tell from a genuine blank answer — the worse failure.
    #[test]
    fn an_unterminated_block_is_left_alone() {
        let mut t = "<think>\nStill working through this and then I ran out of room".to_string();
        let before = t.clone();
        take_leading_reasoning_block(&mut t);
        assert_eq!(t, before, "a truncated reply must not be emptied");
    }

    /// A model writing ABOUT thinking is not thinking. Only a block at the very
    /// start is the preamble.
    #[test]
    fn a_think_tag_later_in_the_reply_is_content() {
        let mut t = "You can wrap reasoning in <think>like this</think> if you want.".to_string();
        let before = t.clone();
        take_leading_reasoning_block(&mut t);
        assert_eq!(t, before);
    }

    /// Gemma 4 thinks in a channel, not a `<think>` block (#282): with thinking
    /// on, and after a tool's answer, its reply opens with one. An empty
    /// channel — what its template closes by default — goes the same way.
    #[test]
    fn gemma_4s_thought_channel_is_a_reasoning_block() {
        let mut t = "<|channel>thought\nThe user wants Paris.\n<channel|>It is 14 °C.".to_string();
        take_leading_reasoning_block(&mut t);
        assert_eq!(t, "It is 14 °C.");
        let mut t = "<|channel>thought\n<channel|>Light rain.".to_string();
        take_leading_reasoning_block(&mut t);
        assert_eq!(t, "Light rain.");
        // Unclosed: left alone, as `<think>` is.
        let mut t = "<|channel>thought\nStill weighing".to_string();
        take_leading_reasoning_block(&mut t);
        assert_eq!(t, "<|channel>thought\nStill weighing");
    }

    /// **#392: a template that opens the scratchpad in the PROMPT.** Qwen 3.5's
    /// generation prompt ends `<think>\n`, so the reply starts mid-thought with
    /// no tag of its own; read as if the reply opened it, the whole scratchpad
    /// and its `</think>` were the answer. Told what the prompt opened, the
    /// finaliser reads it from inside the block, as llama.cpp parses
    /// `generation_prompt + output`.
    #[test]
    fn a_reply_its_prompt_opened_inside_the_scratchpad_is_read_from_inside_it() {
        use super::{PromptOpened, REASONING_BLOCKS};
        let think = PromptOpened::Inside(REASONING_BLOCKS[0]);
        let reply = "Thinking Process:\n\n1. 17 × 3 = 51.\n</think>\n\n51";

        let mut t = reply.to_string();
        let finished = finalize_reply_text(&mut t, &[], think);
        assert_eq!(t, "51");
        assert_eq!(
            finished.reasoning.as_deref(),
            Some("Thinking Process:\n\n1. 17 × 3 = 51.")
        );

        // Cut short inside the block: the same shape as a reply that opened its
        // own block and ran out — the opener is in front, so the surface with a
        // field shows it as reasoning and the answer as empty, as before #392.
        let mut cut = "Thinking Process:\n\n1. 17".to_string();
        assert_eq!(finalize_reply_text(&mut cut, &[], think).reasoning, None);
        assert_eq!(cut, "<think>Thinking Process:\n\n1. 17");
        assert_eq!(
            crate::api::tool_parse::split_unfinished_reasoning(cut),
            (
                String::new(),
                Some("Thinking Process:\n\n1. 17".to_string())
            )
        );

        // A reply of nothing stays nothing — an opener alone is no scratchpad.
        for blank in ["", " ", "\n\n"] {
            let mut t = blank.to_string();
            let _ = finalize_reply_text(&mut t, &[], think);
            assert_eq!(t, "", "{blank:?}");
        }

        // A second pass over a finalised reply passes `Nothing`, and keeps it.
        let mut t = reply.to_string();
        let _ = finalize_reply_text(&mut t, &[], think);
        let _ = finalize_reply_text(&mut t, &[], PromptOpened::Nothing);
        assert_eq!(t, "51");

        // Gemma 4's thought channel, opened by the prompt the same way.
        let mut t = "\nweighing it\n<channel|>Sure.".to_string();
        let finished = finalize_reply_text(&mut t, &[], PromptOpened::Inside(REASONING_BLOCKS[1]));
        assert_eq!(
            (t.as_str(), finished.reasoning.as_deref()),
            ("Sure.", Some("weighing it"))
        );
    }

    /// **gpt-oss (#390): a harmony reply is channels**, every marker a control
    /// token a byte-level decode writes out as text. Told the prompt handed the
    /// model a harmony turn, the finaliser reads the analysis channel as the
    /// scratchpad, the final channel as the answer, and a `to=functions.NAME`
    /// message as a call the tool parser reads — no marker reaches the reply.
    #[test]
    fn a_harmony_reply_is_read_as_scratchpad_answer_and_call() {
        use super::PromptOpened;
        let harmony = PromptOpened::Harmony;
        let mut t = "<|channel|>analysis<|message|>The user greets.<|end|>\
                     <|start|>assistant<|channel|>final<|message|>Hello!"
            .to_string();
        let finished = finalize_reply_text(&mut t, &[], harmony);
        assert_eq!(t, "Hello!");
        assert_eq!(finished.reasoning.as_deref(), Some("The user greets."));

        let mut t = "<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant\
                     <|channel|>commentary to=functions.get_weather <|constrain|>json\
                     <|message|>{\"city\":\"Paris\"}"
            .to_string();
        let finished = finalize_reply_text(&mut t, &[], harmony);
        assert_eq!(finished.reasoning.as_deref(), Some("Need weather."));
        let tools = crate::api::tool_parse::RequestTools::untyped(true);
        let calls = crate::api::tool_parse::parse_tool_calls(&t, &tools).expect("a call");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, "{\"city\":\"Paris\"}");

        // A stop sequence inside the scratchpad leaves the answer whole, as for
        // any other reasoning model.
        let mut t = "<|channel|>analysis<|message|>Do not say STOP.<|end|>\
                     <|start|>assistant<|channel|>final<|message|>Fine."
            .to_string();
        let finished = finalize_reply_text(&mut t, &["STOP".to_string()], harmony);
        assert_eq!((t.as_str(), finished.stop), ("Fine.", None));

        // The same text from a prompt that was NOT a harmony turn is not read
        // as channels — only its markers are scrubbed, as before.
        let mut t = "<|channel|>final<|message|>Hello!".to_string();
        let _ = finalize_reply_text(&mut t, &[], PromptOpened::Nothing);
        assert_ne!(t, "Hello!");

        // A second pass over a finalised reply passes `Nothing`, and keeps it.
        let mut t = "<|channel|>final<|message|>Hi.".to_string();
        let _ = finalize_reply_text(&mut t, &[], harmony);
        let _ = finalize_reply_text(&mut t, &[], PromptOpened::Nothing);
        assert_eq!(t, "Hi.");
    }

    /// The verdict is read off the END of the rendered prompt, less trailing
    /// whitespace: an opener there is a block the reply starts inside; a
    /// CLOSED empty block (thinking turned off) or an opener anywhere else is
    /// not.
    #[test]
    fn a_prompt_opens_a_scratchpad_only_where_it_ends_on_an_opener() {
        use super::{PromptOpened, REASONING_BLOCKS};
        let think = PromptOpened::Inside(REASONING_BLOCKS[0]);
        let of = PromptOpened::of;
        assert_eq!(of("<|im_start|>assistant\n<think>\n"), think);
        assert_eq!(of("<|im_start|>assistant\n<think>"), think);
        assert_eq!(
            of("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
            PromptOpened::Nothing
        );
        assert_eq!(of("<|im_start|>assistant\n"), PromptOpened::Nothing);
        assert_eq!(
            of("<|im_start|>user\nWhat is <think>?<|im_end|>\n<|im_start|>assistant\n"),
            PromptOpened::Nothing
        );
        assert_eq!(
            of("<start_of_turn>model\n<|channel>thought\n"),
            PromptOpened::Inside(REASONING_BLOCKS[1])
        );
        assert_eq!(of(""), PromptOpened::Nothing);
        // gpt-oss's turn, and a harmony marker anywhere else.
        assert_eq!(
            of("<|start|>user<|message|>Hi<|end|><|start|>assistant"),
            PromptOpened::Harmony
        );
        assert_eq!(
            of("<|start|>assistant<|channel|>final<|message|>Hi<|end|>"),
            PromptOpened::Nothing
        );
    }

    /// The ordinary case must cost nothing and change nothing.
    #[test]
    fn a_reply_without_reasoning_is_untouched() {
        let mut t = "OK".to_string();
        take_leading_reasoning_block(&mut t);
        assert_eq!(t, "OK");
    }

    /// It runs inside the one finaliser every reply path goes through, so no
    /// caller can be exempt — and it must run BEFORE stop-string truncation, or
    /// a stop sequence occurring inside the reasoning cuts the reply there and
    /// the real answer is lost with it.
    #[test]
    fn the_shared_finaliser_applies_it_before_stop_sequences() {
        let mut t =
            "<think>\nI should not say STOP here.\n</think>\n\nThe answer is 4.".to_string();
        let finished = finalize_reply_text(
            &mut t,
            &["STOP".to_string()],
            crate::inference::PromptOpened::Nothing,
        );
        let matched = finished.stop;
        assert_eq!(t, "The answer is 4.");
        assert_eq!(
            matched, None,
            "a stop sequence inside the scratchpad must not truncate the answer"
        );
        assert_eq!(
            finished.reasoning.as_deref(),
            Some("I should not say STOP here."),
            "the scratchpad is handed back, trimmed, for the surface with a field for it (#141)"
        );
    }

    /// The scratchpad is reported ONCE: a second call finds none, which is why
    /// a caller that finalises again keeps the first call's (#141).
    #[test]
    fn the_scratchpad_is_reported_once_and_a_blank_one_not_at_all() {
        let mut t = "<think>\n  weighing it  \n</think>\n\nYes.".to_string();
        assert_eq!(
            finalize_reply_text(&mut t, &[], crate::inference::PromptOpened::Nothing)
                .reasoning
                .as_deref(),
            Some("weighing it")
        );
        assert_eq!(
            finalize_reply_text(&mut t, &[], crate::inference::PromptOpened::Nothing).reasoning,
            None
        );
        assert_eq!(t, "Yes.");

        let mut blank = "<think>\n\n</think>\n\nYes.".to_string();
        assert_eq!(
            finalize_reply_text(&mut blank, &[], crate::inference::PromptOpened::Nothing).reasoning,
            None
        );
        assert_eq!(blank, "Yes.");

        let mut unclosed = "<think>still going".to_string();
        assert_eq!(
            finalize_reply_text(&mut unclosed, &[], crate::inference::PromptOpened::Nothing)
                .reasoning,
            None
        );
        assert_eq!(
            unclosed, "<think>still going",
            "an unclosed block stays in the text"
        );

        let mut gemma = "<|channel>thought\nhmm\n<channel|>Sure.".to_string();
        assert_eq!(
            finalize_reply_text(&mut gemma, &[], crate::inference::PromptOpened::Nothing)
                .reasoning
                .as_deref(),
            Some("hmm")
        );
        assert_eq!(gemma, "Sure.");
    }

    /// Idempotent, like every other step in the finaliser — the router runs it
    /// again with template-derived stops the executor never saw.
    #[test]
    fn running_it_twice_changes_nothing() {
        let mut t = "<think>\nreasoning\n</think>\n\nanswer".to_string();
        let _ = finalize_reply_text(&mut t, &[], crate::inference::PromptOpened::Nothing);
        let once = t.clone();
        let _ = finalize_reply_text(&mut t, &[], crate::inference::PromptOpened::Nothing);
        assert_eq!(t, once);
    }
}
