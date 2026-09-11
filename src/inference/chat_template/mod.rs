//! Jinja2-style chat template engine for GGUF models.
//!
//! GGUF files store a `tokenizer.chat_template` metadata field containing a
//! Jinja2-format template string. This module implements enough of Jinja2 to
//! handle the common patterns used by popular model families:
//! ChatML, Llama, Mistral, Qwen, Gemma, Phi, TinyLlama, etc.

use crate::types::{ChatMessage, Role};

mod fallbacks;
mod tojson;

pub use fallbacks::chatml_fallback;

use fallbacks::{
    gemma_fallback, llama3_fallback, mistral_fallback, vicuna_fallback, zephyr_fallback,
};

/// Apply a Jinja2-style chat template to a list of messages.
///
/// Returns `None` if the template could not be applied, so callers can fall
/// back to ChatML.
///
/// That includes a template which parses and evaluates but renders *nothing*
/// from a non-empty message list. Only structural token errors (an unclosed
/// `{% for %}`, a bare `{{`) actually fail evaluation here; an unknown filter,
/// an unknown variable, a stray `{% endfor %}` and an unclosed `{% if %}` all
/// evaluate quietly to the empty string. Reporting those as success handed the
/// model an empty prompt — no system message, no user turn, and for a VLM no
/// `<image>` placeholder, so vision embeddings were prepended instead of being
/// inserted at the right token position.
/// Ceiling on rendered prompt size (R101). A template is untrusted input and
/// `{% set x = x + x %}` doubles a value in one statement, so output is bounded
/// rather than trusted.
const MAX_TEMPLATE_OUTPUT: usize = 4 * 1024 * 1024;

/// Ceiling on the template SOURCE. Bounds how many value-doubling statements a
/// straight-line template can contain, which fuel does not (a doubling is one
/// cheap instruction that costs a lot of memory). Real chat templates are a few
/// kilobytes; the largest fixture here is 4.2 KB.
const MAX_TEMPLATE_BYTES: usize = 200 * 1024;

/// Instruction budget for one render. Bounds a runaway loop without tripping on
/// a long conversation: a 100-message chat through a branch-heavy template is
/// comfortably under this.
const TEMPLATE_FUEL: u64 = 5_000_000;

pub fn apply_chat_template(
    template: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
    tools: Option<&[serde_json::Value]>,
) -> Option<String> {
    let mut env = minijinja::Environment::new();

    // Chat templates lean on Python string methods — `split`, `lstrip`,
    // `startswith` — which minijinja does not implement natively. This is the
    // shim its own author added to HuggingFace's TGI for exactly this.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function("strftime_now", strftime_now);
    env.add_function("raise_exception", raise_exception);

    // `tojson` as `transformers` defines it, replacing minijinja's builtin.
    // The builtin escapes `<`, `>`, `&` and `'` for the benefit of a web page,
    // and rejects every keyword but `indent` — which fails the WHOLE render.
    // See [`tojson`] for what each of those cost.
    env.add_filter("tojson", tojson::tojson);

    // Undefined must be FALSY rather than an error. Templates guard optional
    // fields with `{% if message.reasoning_content %}` and probe for callables
    // with `is defined`, and erroring on those would decline templates that
    // work.
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);

    // Match the environment model authors actually write against. HuggingFace
    // `transformers` renders chat templates with an
    // `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`
    // and `keep_trailing_newline`, so a template's indentation and its
    // newlines after a block tag are NOT part of the prompt. Rendering the
    // same template with Jinja's defaults instead puts the author's
    // indentation into the text the model sees.
    //
    // It is invisible on a template that marks every block with `{%-` / `-%}`
    // — Qwen3 does, which is why it agreed either way — and it is the whole
    // difference on one that does not.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_keep_trailing_newline(true);

    // A chat template is UNTRUSTED INPUT: it arrives inside a GGUF downloaded
    // from the network, and it is a program. R101 found that a recursion cap
    // alone does not stop `{% set x = x + x %}` doubling a value, and capped
    // rendered output at 4 MiB. That guard lived in the evaluator this
    // replaced, so it is re-established here rather than lost in the swap.
    //
    // Three bounds, because they stop different things: `fuel` bounds how many
    // instructions a loop may run, the writer bounds how much a template may
    // EMIT, and the length check below bounds how many doublings a
    // straight-line template can even contain — 200 KB is already two orders of
    // magnitude above any real chat template (the largest seen here is 4.2 KB).
    if template.len() > MAX_TEMPLATE_BYTES {
        tracing::warn!(
            bytes = template.len(),
            limit = MAX_TEMPLATE_BYTES,
            "chat template is implausibly large — declining to render it"
        );
        return None;
    }
    env.set_fuel(Some(TEMPLATE_FUEL));

    env.add_template("chat", template).ok()?;
    let tmpl = env.get_template("chat").ok()?;
    let rendered = tmpl
        .render(minijinja::context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_token,
            eos_token => eos_token,
            // Undefined when there are none, NOT an empty list: every template
            // gates its tool section on `{%- if tools %}`, and both are falsy
            // there, but some also do `{{ tools | length }}` or index into it
            // once inside. Undefined is what `transformers` passes, so it is
            // what templates are written against.
            tools => tools,
        })
        .map_err(|e| {
            // Not a warning: declining is a supported outcome that the caller
            // handles by falling back, and Gemma-family templates decline ON
            // PURPOSE via `raise_exception` when handed a system turn.
            tracing::debug!(error = %e, "chat template did not render");
        })
        .ok()?;

    if rendered.len() > MAX_TEMPLATE_OUTPUT {
        tracing::warn!(
            bytes = rendered.len(),
            limit = MAX_TEMPLATE_OUTPUT,
            "chat template rendered implausibly large output — discarding it"
        );
        return None;
    }

    // A template that swallowed a whole conversation is not a render. Kept
    // from the previous engine; `render_kept_the_last_question` in
    // `build_prompt_inner` is the stronger form of the same idea.
    if !messages.is_empty() && rendered.trim().is_empty() {
        return None;
    }
    Some(rendered)
}

/// `strftime_now("%d %b %Y")` — the current date, as Llama-3.x templates ask
/// for it.
///
/// Llama-3.x renders today's date into the system block guarded by
/// `{% if strftime_now is defined %}`, with a HARDCODED date in the `else`.
/// With no implementation the guard was false and every Llama-3 model on the
/// network was told the date was 26 Jul 2024.
///
/// An unparseable format is an error rather than a panic: the format string
/// arrives from model metadata, and `chrono`'s `Display` panics on a bad
/// specifier instead of erroring.
fn strftime_now(fmt: String) -> Result<String, minijinja::Error> {
    // An unrenderable specifier yields an empty string rather than failing the
    // render. `chrono`'s `Display` PANICS on an unknown specifier instead of
    // erroring, so it must be caught — but the format string comes from model
    // metadata, and discarding a whole template over one bad `%Q` would throw
    // away a prompt that is otherwise fine.
    if chrono::format::StrftimeItems::new(&fmt)
        .any(|item| matches!(item, chrono::format::Item::Error))
    {
        tracing::debug!(format = %fmt, "chat template: unsupported strftime format");
        return Ok(String::new());
    }
    Ok(chrono::Local::now().format(&fmt).to_string())
}

/// `raise_exception("...")` — how Gemma and Mistral templates say they do not
/// support the message they were given, usually a system turn.
///
/// Failing the render is the correct answer: the caller falls back, and
/// `template_expects_system` already refuses to inject a system message into a
/// template containing this call. The previous engine treated it as a silent
/// skip, which rendered a turn the model was never trained on.
fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(
        minijinja::ErrorKind::InvalidOperation,
        msg,
    ))
}

/// Markers that are NEVER legitimate assistant output, whatever a model's
/// template contains.
///
/// A GGUF whose chat template says one family but whose weights were tuned on
/// another will emit the other family's markers — reported live 2026-07-25: a
/// Llama-3.2 q8_0 returned `<|im_end|>hello\n</im_start>` even though its
/// template is not ChatML, so scanning the template alone could never catch it.
///
/// Only the `<|...|>` / `<...>` special-token forms belong here. `[INST]` and
/// `</s>` stay template-gated because they can plausibly appear in real prose
/// (code, XML), whereas these cannot. `<|end|>` is also deliberately absent —
/// it is template-gated with an exclusion, because for harmony / solar-open
/// models it separates messages inside a reply that is still being written.
fn always_unsafe_markers() -> Vec<String> {
    [
        "<|im_end|>",
        "<|im_start|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|start_header_id|>",
        "<|endoftext|>",
        "<end_of_turn>",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

/// Extract stop strings from a chat template.
///
/// These are role markers that, when generated by the model, indicate the start
/// of a new turn and should terminate generation. Common patterns:
/// - Zephyr/TinyLlama: `<|user|>`, `<|system|>`
/// - ChatML: `<|im_end|>`
/// - Llama: `[INST]`
pub fn extract_stop_strings(template: Option<&str>) -> Vec<String> {
    let tmpl = match template {
        Some(t) => t,
        // No template does NOT mean ChatML. `build_prompt_inner`'s no-template
        // branch asks `fallback_by_model_name` first, so the prompt may be
        // zephyr, llama3, gemma, mistral or vicuna — and returning ChatML's one
        // marker left every one of those with no stop for the marker it will
        // actually emit. The always-on list below is the right answer here for
        // the same reason it is always-on: none of those markers is ever
        // legitimate assistant output, whatever format the prompt turned out to
        // be.
        None => return always_unsafe_markers(),
    };

    let mut stops = always_unsafe_markers();

    // Scan the template for turn-boundary markers. Anything present here is a
    // marker this model's own template uses, so seeing it mid-generation means
    // the model has ended its turn (or started hallucinating another one) and we
    // should stop rather than pass the marker through as visible text.
    //
    // This list must cover what models actually EMIT, which is not always the
    // token the tokenizer declares as EOS. Observed live 2026-07-25: a
    // Llama-3.2 model emitted `<|eom_id|><|start_header_id|>assistant<|end_header_id|>`
    // into user-visible content because only `<|eot_id|>` was listed —
    // Llama 3.1+ uses `<|eom_id|>` ("end of message") when it believes it is
    // making a tool call, and neither the header markers nor `<|eom_id|>` were
    // being caught.
    for marker in &[
        // ChatML (Qwen, Hermes, many finetunes)
        "<|im_end|>",
        "<|im_start|>",
        // Zephyr / TinyLlama style
        "<|user|>",
        "<|system|>",
        // Llama 2 / Mistral
        "[INST]",
        "</s>",
        // Llama 3.x — `eot` ends a turn, `eom` ends a message (tool calls), and
        // a header marker mid-stream is the model opening a turn it shouldn't.
        "<|eot_id|>",
        "<|eom_id|>",
        "<|start_header_id|>",
        // Gemma
        "<end_of_turn>",
        // GPT-2 style vocabularies (Qwen base, several small models)
        "<|endoftext|>",
    ] {
        if tmpl.contains(marker) && !stops.iter().any(|s| s == marker) {
            stops.push(marker.to_string());
        }
    }

    // Phi-3/3.5/4 close every turn with `<|end|>`, and their GGUFs declare a
    // DIFFERENT token as EOS (`<|endoftext|>`), so nothing stopped the reply
    // there: on a GPT-2-BPE vocab the marker leaked into visible content, on a
    // SentencePiece vocab it decoded to nothing and the model silently carried
    // on inventing further turns. The end-of-generation token id resolved in
    // `split::gguf_meta` is the primary fix; this is the second line of defence,
    // and the only one that can also REMOVE a leaked marker from the text.
    //
    // Excluded for harmony (gpt-oss) and solar-open, where `<|end|>` separates
    // messages within one reply rather than ending it — stopping there truncates
    // every such reply at its first message. llama.cpp carries the same
    // exclusion (`llama_vocab::impl::load`), which is why it is here and not in
    // `always_unsafe_markers`.
    let end_separates_messages = ["<|return|>", "<|call|>", "<|calls|>", "<|flush|>"]
        .iter()
        .any(|m| tmpl.contains(m));
    if tmpl.contains("<|end|>") && !end_separates_messages && !stops.iter().any(|s| s == "<|end|>")
    {
        stops.push("<|end|>".to_string());
    }

    // For Zephyr-style templates, `<|assistant|>` in the middle of generation
    // means the model is hallucinating a new assistant turn after ending its own.
    if tmpl.contains("<|assistant|>") && tmpl.contains("<|user|>") {
        // Already have <|user|>, also stop on <|assistant|> if model re-emits it
        if !stops.contains(&"<|assistant|>".to_string()) {
            stops.push("<|assistant|>".to_string());
        }
    }

    stops
}

/// Add the stop strings implied by a model's chat template to `params`.
///
/// A model signals the end of its turn with a template marker rather than only
/// the tokenizer's EOS id, so without these it can run to `max_tokens` emitting
/// `<|im_end|>`, `<|user|>` and friends as visible text — and then keep going,
/// inventing the next turn of a conversation the user never had.
///
/// **This exists as a shared helper because forgetting it is the recurring
/// bug**, and it has now been missed three times: the streaming split path, its
/// non-streaming sibling (a tester saw `<|im_end|>` in a reply with
/// `finish_reason: "length"` — ran to the cap, never matched a stop), and the
/// router/remote path, which built a templated prompt and forwarded the
/// caller's params untouched. That third one is the fast path a node takes
/// whenever ONE peer holds the whole model, i.e. the normal case for a machine
/// that stores nothing itself.
///
/// It lives here, next to `extract_stop_strings`, rather than in one API
/// module: the inference paths that need it cannot reach into `api::openai`,
/// which is why the router paths went without for as long as they did. New
/// prompt-building paths should prefer
/// `PipelineExecutor::build_prompt_and_stops`, which returns the prompt and
/// these stops together so they cannot be separated.
pub fn with_template_stops(
    mut params: swarmllm_types::inference::SamplingParams,
    chat_template: Option<&str>,
) -> swarmllm_types::inference::SamplingParams {
    for stop in extract_stop_strings(chat_template) {
        if !params.stop.contains(&stop) {
            params.stop.push(stop);
        }
    }
    params
}

/// Build a chat prompt using the given template, falling back to ChatML.
///
/// This is the main entry point for chat prompt construction.
///
/// `model_name` is REQUIRED, not optional, and passing `None` is a real choice
/// with a real cost: it is the only thing that lets a failed template degrade
/// to the right family format instead of ChatML. This used to be a
/// four-argument convenience wrapper that hardcoded `None`, and six of the
/// seven production call sites used it — so the gemma / LLaVA fallbacks could
/// never fire on the OpenAI, Anthropic, streaming, or router paths, and a
/// Llama-3 model whose template failed was asked to speak ChatML (which is
/// where the stray `<|im_end|>` markers came from). Pass the model id unless
/// you genuinely do not have one.
/// Markers that CLOSE a turn. A rendered chat prompt that ends on one of these
/// has handed the model a finished conversation.
///
/// Only closers appear here, never openers: `<|im_start|>assistant\n` and
/// `<|start_header_id|>assistant<|end_header_id|>` are how a correct template
/// ends, and both must pass.
const TURN_ENDING_MARKERS: &[&str] = &[
    "<|im_end|>",
    "<|eot_id|>",
    "<|end|>",
    "<|endoftext|>",
    "<end_of_turn>",
    "</s>",
];

/// Does this rendered prompt hand the turn to the model?
///
/// A chat prompt is supposed to end where the model is meant to start writing —
/// the generation prompt. When it ends on a turn-CLOSING marker instead, the
/// model is looking at a completed conversation and the correct thing for it to
/// do is end its turn, which it does: one token, `finish_reason: "stop"`, no
/// stop sequence involved and nothing wrong with sampling.
///
/// **Why this is checked at render time rather than after a short reply.** That
/// symptom is intermittent and maddening to chase from the outside — a tester
/// spent three releases on a version of it, could not build a minimal repro,
/// and neither could we. But the condition itself is deterministic and visible
/// the moment the prompt is built, before a single token is generated. Checking
/// the postcondition where it is established turns an unreproducible symptom
/// into a line in the log on the request that caused it.
///
/// Returns `true` for a prompt with no control markers at all: that is an
/// ordinary completion-style prompt and none of this applies to it.
pub(crate) fn prompt_hands_over_to_the_model(prompt: &str) -> bool {
    let tail = prompt.trim_end();
    !TURN_ENDING_MARKERS.iter().any(|m| tail.ends_with(m))
}

/// `tools` is a REQUIRED parameter with no convenience wrapper that passes
/// `None`, deliberately. That exact shape is how the template fallback was
/// disabled on six of seven paths (gotcha #171): a caller reaches for the
/// short form, and the context it silently drops is the context that made the
/// call correct.
pub fn build_prompt(
    messages: &[ChatMessage],
    template: Option<&str>,
    bos_token: &str,
    eos_token: &str,
    model_name: Option<&str>,
    tools: Option<&[serde_json::Value]>,
) -> String {
    build_prompt_with_model(messages, template, bos_token, eos_token, model_name, tools)
}

/// Pick a fallback prompt format from the model name alone.
///
/// Returns the rendered prompt plus the name of the format chosen, or `None`
/// when the name carries no usable signal and the caller should use ChatML.
/// Shared by both `build_prompt_with_model` branches: a model with no template
/// at all and a model whose template failed to evaluate want the same answer.
/// Reaching ChatML for a model that is not a ChatML model is the failure mode
/// that produced stray `<|im_end|>` markers in Llama-3 replies for several
/// releases: the model answers in whatever format it was asked in. Every family
/// we can recognise by name gets its own format here, so a template we cannot
/// evaluate degrades to the right shape instead of a foreign one.
///
/// Checked before the generic families because a name can match more than one
/// substring — `llava-v1.6-mistral-7b` is a LLaVA model that must use the
/// vicuna/LLaVA prompt, not Mistral's.
fn fallback_by_model_name(
    messages: &[ChatMessage],
    model_name: Option<&str>,
) -> Option<(String, &'static str)> {
    let name_lower = model_name?.to_lowercase();
    if name_lower.contains("llava") || name_lower.contains("vicuna") {
        return Some((vicuna_fallback(messages), "vicuna"));
    }
    // Checked before the llama families: "tinyllama" contains "llama" and this
    // is a Zephyr-format model, not a Llama-chat one.
    if name_lower.contains("tinyllama") || name_lower.contains("zephyr") {
        return Some((zephyr_fallback(messages), "zephyr"));
    }
    if name_lower.contains("gemma") {
        return Some((gemma_fallback(messages), "gemma"));
    }
    if name_lower.contains("llama-3")
        || name_lower.contains("llama3")
        || name_lower.contains("llama_3")
    {
        return Some((llama3_fallback(messages), "llama3"));
    }
    // `mixtral` and `ministral`/`magistral` share the [INST] convention.
    if name_lower.contains("mistral")
        || name_lower.contains("mixtral")
        || name_lower.contains("ministral")
        || name_lower.contains("magistral")
    {
        return Some((mistral_fallback(messages), "mistral"));
    }
    None
}

/// System message supplied when the caller sends none and the model's template
/// shows it expects one.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are a helpful assistant.";

/// Whether this template has a system-role branch it will actually honour.
///
/// Two families must be told apart:
///
/// - Zephyr/TinyLlama, Phi and Llama-3 branch on `role == 'system'` and emit a
///   distinct marker for it. These models are trained with a system turn and
///   behave badly without one.
/// - Gemma and Mistral do NOT support a system role and say so with
///   `raise_exception`. Our evaluator deliberately treats `raise_exception` as
///   a silent skip, so injecting a system message for them does not fail
///   loudly — it renders a turn the model was never trained on (Gemma would
///   emit `<start_of_turn>system`). The presence of `raise_exception` is
///   therefore a hard veto.
///
/// Absent positive evidence, this returns false: not injecting is always safe,
/// injecting into the wrong template is not.
fn template_expects_system(template: &str) -> bool {
    if template.contains("raise_exception") {
        return false;
    }
    template.contains("'system'") || template.contains("\"system\"")
}

/// Move every system turn into the first user turn, for a template that
/// refuses the role.
///
/// Gemma and Mistral declare no system role and their templates
/// `raise_exception` on one, so the render fails outright rather than ignoring
/// it. Both publishers document the same remedy: prepend the system text to the
/// first user message. `template_expects_system` is the static sibling of this
/// and is deliberately not reused — it answers false for ANY template
/// containing `raise_exception`, including ones that raise only on role
/// alternation, which is the right conservatism for deciding whether to ADD a
/// default system prompt and the wrong basis for rewriting a prompt that
/// already renders.
///
/// Answers `None` when there is nothing to move, or nothing to move it into: a
/// conversation with no user turn keeps its system message and takes the
/// fallback, which does render one.
fn fold_system_into_first_user(messages: &[ChatMessage]) -> Option<Vec<ChatMessage>> {
    if !messages.iter().any(|m| matches!(m.role, Role::System)) {
        return None;
    }
    let system_text = messages
        .iter()
        .filter(|m| matches!(m.role, Role::System))
        .map(|m| m.content.trim())
        .filter(|c| !c.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    let mut out: Vec<ChatMessage> = messages
        .iter()
        .filter(|m| !matches!(m.role, Role::System))
        .cloned()
        .collect();
    if !system_text.is_empty() {
        let first_user = out.iter_mut().find(|m| matches!(m.role, Role::User))?;
        first_user.content = format!("{system_text}\n\n{}", first_user.content);
    }
    Some(out)
}

/// Ensure a usable system turn, returning an owned list only when one is needed.
///
/// A blank system message is treated as absent — it renders an empty system
/// turn, which for TinyLlama reproduces the exact failure this guards against
/// (verified: an empty system message still yields a blank reply).
fn with_system_message(messages: &[ChatMessage]) -> Option<Vec<ChatMessage>> {
    if messages.is_empty() {
        return None;
    }
    match messages.iter().position(|m| matches!(m.role, Role::System)) {
        // Caller supplied a real system message — never override it.
        Some(i) if !messages[i].content.trim().is_empty() => None,
        Some(i) => {
            let mut v = messages.to_vec();
            v[i].content = DEFAULT_SYSTEM_PROMPT.to_string();
            Some(v)
        }
        None => {
            let mut v = Vec::with_capacity(messages.len() + 1);
            v.push(ChatMessage {
                role: Role::System,
                content: DEFAULT_SYSTEM_PROMPT.to_string(),
                images: Vec::new(),
            });
            v.extend_from_slice(messages);
            Some(v)
        }
    }
}

/// Build prompt with optional model name hint for fallback template selection.
///
/// When the model's template shows it expects a system turn and the caller sent
/// none, a neutral default is supplied. TinyLlama-1.1B-Chat answers a bare user
/// question with nothing but a `<|user|>` turn marker — stop-truncation strips
/// it and the user sees a blank reply reported as a successful completion. The
/// same question with a system message is answered normally.
pub fn build_prompt_with_model(
    messages: &[ChatMessage],
    template: Option<&str>,
    bos_token: &str,
    eos_token: &str,
    model_name: Option<&str>,
    tools: Option<&[serde_json::Value]>,
) -> String {
    let mut prompt =
        build_prompt_inner(messages, template, bos_token, eos_token, model_name, tools);
    open_the_models_turn_if_the_prompt_closed_it(&mut prompt, model_name);
    warn_if_the_prompt_closes_the_turn(&prompt, model_name);
    prompt
}

/// The token sequence that opens the model's turn, for a prompt that ended on
/// `closer`.
///
/// `None` where the closer does not name one family: `</s>` is Llama-2,
/// Mistral and vicuna, and `<|endoftext|>` is used by several unrelated
/// vocabularies. Guessing there would replace a diagnosable prompt with a
/// confidently wrong one, which is worse — those keep the warning and nothing
/// else.
fn generation_prompt_after(closer: &str) -> Option<&'static str> {
    match closer {
        "<|im_end|>" => Some("<|im_start|>assistant\n"),
        "<|eot_id|>" => Some("<|start_header_id|>assistant<|end_header_id|>\n\n"),
        "<|end|>" => Some("<|assistant|>\n"),
        "<end_of_turn>" => Some("<start_of_turn>model\n"),
        _ => None,
    }
}

/// Finish a prompt that stopped at the end of someone else's turn.
///
/// **We already knew this prompt was broken and sent it anyway.** The condition
/// is established before a single token is generated, it is deterministic, and
/// it guarantees the reply: the model is shown a finished conversation, so it
/// ends its turn at once — one token, `finish_reason: "stop"`. Logging that and
/// proceeding explains the failure without preventing it.
///
/// The evidence is positive, not an absence: the prompt ENDS with one of the
/// six markers in [`TURN_ENDING_MARKERS`], so it demonstrably closed a turn.
/// Four of those name exactly one family, and appending that family's opener is
/// the same string its own template would have added. The two ambiguous ones
/// are left alone.
///
/// Reported against a Qwen3-8B whose template this renderer declines (it uses
/// `namespace()`, a reversed slice, `loop.index0`, `tojson` and string methods),
/// leaving a prompt ending on `<|im_end|>` on every request.
fn open_the_models_turn_if_the_prompt_closed_it(prompt: &mut String, model_name: Option<&str>) {
    if prompt_hands_over_to_the_model(prompt) {
        return;
    }
    let Some(closer) = TURN_ENDING_MARKERS
        .iter()
        .find(|m| prompt.trim_end().ends_with(*m))
        .copied()
    else {
        return;
    };
    let Some(opener) = generation_prompt_after(closer) else {
        return;
    };
    // Trailing whitespace between the closed turn and the opener is what the
    // templates themselves emit, so normalise to exactly one newline.
    while prompt.ends_with('\n') || prompt.ends_with(' ') {
        prompt.pop();
    }
    prompt.push('\n');
    prompt.push_str(opener);
    tracing::info!(
        model = model_name.unwrap_or("<unknown>"),
        closed_with = %closer,
        "the chat template left the prompt at the end of a finished turn, so the model's own \
         turn was opened for it — without this the reply is a single end-of-turn token"
    );
}

/// Report a rendered prompt that ends by closing a turn instead of opening the
/// model's. See [`prompt_hands_over_to_the_model`].
///
/// **Called from `build_prompt_with_model`, which is the choke point every
/// prompt passes through — the local fast path via `build_prompt`, and the
/// router/distributed paths via `pipeline::prompt::build_prompt_with_header`.**
/// It lived in the `build_prompt` wrapper for one commit, which the router path
/// does not call, so the distributed paths were silently exempt: the exact
/// one-invariant-N-paths defect this repo keeps hitting, caught by tracing the
/// second caller rather than by re-reading the diff.
fn warn_if_the_prompt_closes_the_turn(prompt: &str, model_name: Option<&str>) {
    if prompt_hands_over_to_the_model(prompt) {
        return;
    }
    // WARN, not an error: the request still runs. Being wrong about this must
    // never cost someone an answer; what it buys is that the next occurrence
    // explains itself instead of looking like the model refusing to speak.
    let closer = TURN_ENDING_MARKERS
        .iter()
        .find(|m| prompt.trim_end().ends_with(*m))
        .copied()
        .unwrap_or("");
    tracing::warn!(
        model = model_name.unwrap_or("<unknown>"),
        ends_with = %closer,
        prompt_chars = prompt.len(),
        "the rendered prompt ends by CLOSING a turn rather than opening the model's, so the \
         model is being shown a finished conversation and will most likely end its turn at \
         once — one token, finish_reason \"stop\". This is the prompt, not sampling: the chat \
         template for this model did not append its generation prompt"
    );
}

/// Did this render keep the question in it?
///
/// A chat template is a program, and this renderer is a documented SUBSET of
/// Jinja. A subset running a program it does not fully implement can produce
/// something that LOOKS like a prompt — the system preamble present, the
/// model's turn correctly opened — with every user message missing. That
/// reaches the model as a well-formed request to answer nothing, and the model
/// duly answers something else, fluently. It reads like a broken forward pass;
/// it is a broken prompt.
///
/// Observed on the official Qwen3 template, which uses `messages[::-1]` and
/// other constructs past this renderer's edge: three different questions all
/// arrived as the same 14 tokens of scaffolding and produced the same reply.
/// The existing render test passed throughout, because it asserted only that
/// the prompt ends by opening the model's turn — which a prompt that dropped
/// every message also does.
///
/// Declining is safe: `build_prompt_inner` falls back, and a fallback that
/// carries the question beats a faithful-looking render that does not.
fn render_kept_the_last_question(rendered: &str, messages: &[ChatMessage]) -> bool {
    let Some(asked) = messages
        .iter()
        .rev()
        .find(|m| matches!(m.role, Role::User))
        .map(|m| m.content.trim())
    else {
        return true;
    };
    asked.is_empty() || rendered.contains(asked)
}

/// Does this template render tool definitions itself?
///
/// Every HuggingFace template that supports tools reads the `tools` variable —
/// `{%- if tools %}` is the near-universal opening — so referencing it at all
/// is the signal. A template that never mentions it cannot render tools no
/// matter what it is handed, and its model must be told about them in prose
/// instead.
///
/// Deliberately a cheap textual check rather than a trial render. A trial
/// render answers a different question — "did this template produce anything
/// when given tools" — which is true for every template, since one that
/// ignores `tools` still renders the conversation perfectly well.
pub fn template_renders_tools(template: &str) -> bool {
    template.contains("tools")
}

/// Describe tools to a model whose template cannot, as a system message.
///
/// The wording is `api::tool_parse::format_tool_prompt`, shared with the
/// surfaces that used to do this themselves, because
/// `api::tool_parse::parse_tool_calls` tries the format it asks for first.
fn describe_tools_in_prose(
    messages: &[ChatMessage],
    tools: &[serde_json::Value],
) -> Vec<ChatMessage> {
    let specs: Vec<(String, Option<String>, Option<String>)> = tools
        .iter()
        .filter_map(|t| {
            let f = t.get("function")?;
            Some((
                f.get("name")?.as_str()?.to_string(),
                f.get("description")
                    .and_then(|d| d.as_str())
                    .map(str::to_string),
                f.get("parameters").map(|p| p.to_string()),
            ))
        })
        .collect();
    if specs.is_empty() {
        return messages.to_vec();
    }
    let mut out = Vec::with_capacity(messages.len() + 1);
    out.push(ChatMessage {
        role: Role::System,
        content: crate::api::tool_parse::format_tool_prompt(&specs),
        images: Vec::new(),
    });
    out.extend_from_slice(messages);
    out
}

/// A render that was HANDED tools and did not put them in the prompt is a
/// FAILED render, the same way one that lost the question is.
///
/// `template_renders_tools` can only read the template's text, and a template
/// may mention `tools` without ever consulting the variable it was given.
/// Phi-4-mini-instruct is that template: its tool branch tests
/// `'tools' in message and message['tools'] is not none` — a per-MESSAGE field,
/// not the top-level variable — so the substring check passes, the prose
/// description is suppressed as redundant, the branch never fires, and the model
/// is told **nothing at all** about its tools. That is worse than either branch
/// alone; it is the gap between them. Reported from the field 2026-09-10: "no
/// engagement with the `tools` array whatsoever, despite one being present and
/// directly relevant", from a model trained by its publisher for function
/// calling.
///
/// Checked on the tool NAMES, because a name is what the model needs in order to
/// call anything and what every rendering of a tool necessarily contains.
fn render_shows_the_tools(rendered: &str, tools: &[serde_json::Value]) -> bool {
    tools
        .iter()
        .filter_map(|t| {
            t.get("function")
                .and_then(|f| f.get("name"))
                .or_else(|| t.get("name"))
                .and_then(|n| n.as_str())
        })
        .all(|name| rendered.contains(name))
}

fn build_prompt_inner(
    messages: &[ChatMessage],
    template: Option<&str>,
    bos_token: &str,
    eos_token: &str,
    model_name: Option<&str>,
    tools: Option<&[serde_json::Value]>,
) -> String {
    let tools = tools.filter(|t| !t.is_empty());
    // At most two attempts, and only ever two: the model's own template, then
    // prose if that template turned out not to use the tools it was handed.
    // `force_prose` is the ONLY thing that differs between them.
    let first = build_prompt_attempt(
        messages, template, bos_token, eos_token, model_name, tools, false,
    );
    let Some(t) = tools else { return first };
    if render_shows_the_tools(&first, t) {
        return first;
    }
    tracing::warn!(
        model_name = model_name,
        tool_count = t.len(),
        "DIAG: the chat template was handed this request's tools and rendered none of          them — it mentions `tools` but does not read the variable, so the model would          be told nothing about them. Describing them in prose instead."
    );
    build_prompt_attempt(
        messages, template, bos_token, eos_token, model_name, tools, true,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_prompt_attempt(
    messages: &[ChatMessage],
    template: Option<&str>,
    bos_token: &str,
    eos_token: &str,
    model_name: Option<&str>,
    tools: Option<&[serde_json::Value]>,
    force_prose: bool,
) -> String {
    // A model whose template renders tools is told about them the way it was
    // trained to be; every other model is told in prose. This is the ONE place
    // that choice is made — it needs both the tools and the template, and the
    // two API surfaces that used to flatten tools into a system message had
    // only the first, so they described Qwen3's tools in a JSON format it had
    // never seen while its own `{%- if tools %}` branch sat unreachable.
    let renders_tools =
        !force_prose && template.is_some_and(template_renders_tools) && tools.is_some();
    let described = match tools {
        Some(t) if !renders_tools => Some(describe_tools_in_prose(messages, t)),
        _ => None,
    };
    let messages: &[ChatMessage] = described.as_deref().unwrap_or(messages);
    let tools_for_render = if renders_tools { tools } else { None };

    let injected = template
        .filter(|t| template_expects_system(t))
        .and_then(|_| with_system_message(messages));
    let messages: &[ChatMessage] = injected.as_deref().unwrap_or(messages);

    if let Some(tmpl) = template {
        if let Some(result) =
            apply_chat_template(tmpl, messages, bos_token, eos_token, true, tools_for_render)
        {
            if render_kept_the_last_question(&result, messages) {
                tracing::debug!(template_matched = true, "DIAG: chat template applied");
                return result;
            }
            tracing::warn!(
                model_name = model_name,
                rendered_len = result.len(),
                "DIAG: chat template rendered a prompt with the user's question missing — \
                 discarding the render and falling back, because a prompt without the \
                 question gets a fluent answer to something else"
            );
        }
        // A template that declares no system role does not ignore one, it
        // RAISES on it — Gemma and Mistral both do — and the convention their
        // publishers document is to prepend the system text to the first user
        // turn. Retried here, on the failure path, rather than decided from the
        // template's text: the template has just answered the question for
        // itself, so a model that renders a system turn today cannot have its
        // prompt changed by this.
        //
        // The tool description IS a system message (`describe_tools_in_prose`),
        // so without this every Gemma-2 request carrying tools rendered through
        // the fallback below instead of the model's own template — and a
        // caller's own system message did the same on any model that refuses
        // the role.
        if let Some(folded) = fold_system_into_first_user(messages) {
            if let Some(result) =
                apply_chat_template(tmpl, &folded, bos_token, eos_token, true, tools_for_render)
            {
                if render_kept_the_last_question(&result, &folded) {
                    tracing::debug!(
                        template_matched = true,
                        "DIAG: chat template applied with the system turn folded into the first user turn"
                    );
                    return result;
                }
            }
        }
        // Template failed. Prefer evidence from the template body itself — it
        // describes the model that shipped it — then fall back to the same
        // model-name heuristic the no-template branch uses. Reaching ChatML
        // here would drop LLaVA's `<image>` placeholder, which prepends the
        // vision embeddings instead of inserting them at the right position.
        if tmpl.contains("start_of_turn") {
            tracing::warn!(
                fallback = "gemma",
                "DIAG: chat template failed, using gemma fallback"
            );
            return gemma_fallback(messages);
        }
        if let Some((prompt, fallback)) = fallback_by_model_name(messages, model_name) {
            tracing::warn!(
                model_name = model_name,
                fallback,
                "DIAG: chat template failed, using model-name fallback"
            );
            return prompt;
        }
        tracing::warn!(
            fallback = "chatml",
            "DIAG: chat template failed, using fallback"
        );
    } else {
        // A model we could not read a template for cannot be told about its
        // tools the way it was trained to be — `renders_tools` is false above,
        // so it gets the prose description instead and will answer in a format
        // it was never trained on. That is worth a WARN rather than the debug
        // line below, because the visible symptom is a malformed tool call and
        // nothing else names this as the reason. Two field reports on 2026-09-10
        // showed exactly our prose example's shape, `"id": "call_12345"` and
        // all, from models whose own templates render tools perfectly well.
        if tools.is_some() {
            tracing::warn!(
                model_name = model_name,
                "DIAG: no chat template for a request carrying tools — the model's own                  tool-call format is unreachable, describing tools in prose instead.                  A node with the model's gguf_header.bin renders the real template."
            );
        }
        // No template: pick fallback based on model name heuristic
        if let Some((prompt, fallback)) = fallback_by_model_name(messages, model_name) {
            tracing::debug!(
                model_name = model_name,
                fallback,
                "DIAG: no chat template, using model-name fallback"
            );
            return prompt;
        }
        tracing::debug!(
            template_matched = false,
            fallback = "chatml",
            "DIAG: no chat template, using fallback"
        );
    }
    chatml_fallback(messages)
}

#[cfg(test)]
mod tests;
