//! gpt-oss's "harmony" replies, rewritten into the shapes every reply reader
//! here already takes (#390 item 2, step 6).
//!
//! A gpt-oss reply is a run of messages, each a header and a body:
//! `<|channel|>analysis<|message|>…<|end|><|start|>assistant<|channel|>final<|message|>…`.
//! The prompt's own `<|start|>assistant` opens the first header, and the reply
//! ends on `<|return|>` or `<|call|>` — stop tokens, never in the text. Every
//! marker is a CONTROL token that a byte-level decode writes out as its text.
//!
//! llama.cpp master (`common/parsers/gpt-oss.cpp`, read 2026-11-02) parses
//! `<|start|>assistant` + the reply: an `analysis` body is reasoning; a
//! `commentary` body with no recipient (a preamble) and the `final` body are
//! content; a header naming ` to=functions.NAME` — in the role part or after
//! the channel — is a call whose body is its JSON arguments; a message to any
//! other recipient (a built-in browser or python tool) is consumed unseen; a
//! stray `<|channel|>commentary` before the real channel is tolerated.
//!
//! [`Reply`] rewrites that ONCE into what the readers already understand: the
//! reasoning as one leading `<think>…</think>` block (`REASONING_BLOCKS[0]`),
//! content as it is, a call as Hermes' `<tool_call>{"name": …, "arguments":
//! …}</tool_call>`. It is a choke point, not a reader of its own:
//! `finalize_reply_text` and `api::tool_parse::StreamingToolText` both pass a
//! reply through it when the prompt handed the model a harmony turn
//! ([`PromptOpened::Harmony`](super::PromptOpened::Harmony)), so a streamed and
//! an unstreamed reply split the same way on every surface. Text that does not
//! open with a harmony header passes untouched — a model that ignored its
//! format, or text this already rewrote — so a second pass changes nothing.

/// The turn a harmony prompt hands the model: what [`super::PromptOpened::of`]
/// recognises at a rendered prompt's end.
pub(crate) const TURN: &str = "<|start|>assistant";

const START: &str = "<|start|>";
const END: &str = "<|end|>";
const MESSAGE: &str = "<|message|>";
const CHANNEL: &str = "<|channel|>";
const CONSTRAIN: &str = "<|constrain|>";
const RETURN: &str = "<|return|>";
const CALL: &str = "<|call|>";

/// What ends a message's body. `<|start|>` without an `<|end|>` before it
/// opens the next header all the same.
const BODY_ENDS: [&str; 4] = [END, START, RETURN, CALL];

/// What a header can open with: after the prompt's `<|start|>assistant`, a
/// channel or a recipient; after an `<|end|>`, the next `<|start|>`.
const HEADER_OPENERS: [&str; 5] = [CHANNEL, START, " to=", CONSTRAIN, MESSAGE];

/// A "header" longer than this with no `<|message|>` is not one. Real headers
/// are a few dozen characters (`<|start|>assistant to=functions.get_weather
/// <|channel|>commentary <|constrain|>json`); this only stops a reply that
/// lost its format from being held back to its end.
const HEADER_LIMIT: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Body {
    /// `analysis`: the scratchpad.
    Reasoning,
    /// `final`, or a preamble on `commentary`: the answer.
    Content,
    /// A call's JSON arguments, gathered until the call ends.
    Call,
    /// A message to a recipient that is not a function the caller offered, or
    /// reasoning after the answer began: consumed, as llama.cpp does.
    Unseen,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    /// Reading a header, up to its `<|message|>`.
    Header,
    Body(Body),
    /// The reply did not open as harmony: everything passes as it is.
    Plain,
}

/// One harmony reply, rewritten as it arrives — see the module docs.
pub(crate) struct Reply {
    state: State,
    /// Text not yet rewritten: a header being read, or a body's tail that
    /// may be the start of a marker.
    held: String,
    /// Still before the first header: whether the reply is harmony at all is
    /// decided there.
    first: bool,
    /// The call being read: its function's name and its arguments so far.
    call: Option<(String, String)>,
    /// `<think>` written, `</think>` not yet.
    thinking: bool,
    /// The last reasoning message ended with `<|end|>` — the scratchpad is
    /// whole, so the reply's end may close it.
    reasoning_ended: bool,
    /// Content or a call has begun: later reasoning is not shown, since the
    /// readers take one leading block, and an answer never carries it.
    answered: bool,
    /// Any content written: a later content message is set apart from it.
    wrote_content: bool,
    /// The current message has written nothing yet.
    fresh: bool,
    finished: bool,
}

impl Reply {
    pub(crate) fn new() -> Self {
        Self {
            state: State::Header,
            held: String::new(),
            first: true,
            call: None,
            thinking: false,
            reasoning_ended: false,
            answered: false,
            wrote_content: false,
            fresh: true,
            finished: false,
        }
    }

    /// Take the next piece of the reply; returns the rewritten text it made
    /// ready (possibly none — a header, or what may begin a marker, waits).
    pub(crate) fn push(&mut self, raw: &str) -> String {
        let mut out = String::new();
        self.held.push_str(raw);
        loop {
            match self.state {
                State::Plain => {
                    out.push_str(&std::mem::take(&mut self.held));
                    return out;
                }
                State::Header => {
                    if let Some(at) = self.held.find(MESSAGE) {
                        let header = self.held[..at].to_string();
                        self.held.drain(..at + MESSAGE.len());
                        self.first = false;
                        self.open(&header, &mut out);
                        continue;
                    }
                    // As written (` to=` opens with its space) and past any
                    // whitespace before a header.
                    let raw = self.held.as_str();
                    let h = raw.trim_start_matches(['\n', '\r', ' ']);
                    let opens = HEADER_OPENERS
                        .iter()
                        .any(|o| raw.starts_with(o) || h.starts_with(o));
                    let may_open = HEADER_OPENERS
                        .iter()
                        .any(|o| o.starts_with(raw) || o.starts_with(h));
                    if (opens && self.held.len() <= HEADER_LIMIT) || may_open {
                        return out; // wait for the rest of the header
                    }
                    if self.first {
                        self.state = State::Plain;
                    } else {
                        // Text after an `<|end|>` with no header: the model
                        // went on writing — read it as content.
                        self.open("", &mut out);
                    }
                }
                State::Body(kind) => {
                    let end = BODY_ENDS
                        .iter()
                        .filter_map(|m| self.held.find(m).map(|at| (at, *m)))
                        .min();
                    match end {
                        Some((at, marker)) => {
                            let body = self.held[..at].to_string();
                            // `<|start|>` stays: it opens the next header.
                            let consumed = if marker == START {
                                at
                            } else {
                                at + marker.len()
                            };
                            self.held.drain(..consumed);
                            self.write(kind, &body, &mut out);
                            self.close(kind, marker == END, &mut out);
                            self.state = State::Header;
                        }
                        None => {
                            let keep = marker_prefix_len(&self.held);
                            let ready = self.held.len() - keep;
                            let body = self.held[..ready].to_string();
                            self.held.drain(..ready);
                            self.write(kind, &body, &mut out);
                            return out;
                        }
                    }
                }
            }
        }
    }

    /// The reply has ended: everything still held, rewritten. A call is
    /// closed — it ended on `<|call|>`, which never reaches the text — and a
    /// scratchpad its own `<|end|>` closed is closed; one the reply was cut
    /// off inside stays open, as every reader expects of an unfinished block.
    /// A second call returns nothing.
    pub(crate) fn finish(&mut self) -> String {
        let mut out = String::new();
        if std::mem::replace(&mut self.finished, true) {
            return out;
        }
        match self.state {
            State::Plain => out.push_str(&std::mem::take(&mut self.held)),
            // Never decided: what is held is whitespace or the start of a
            // marker, and the reply is not harmony after all.
            State::Header if self.first => out.push_str(&std::mem::take(&mut self.held)),
            // An unfinished header is markup.
            State::Header => self.held.clear(),
            State::Body(kind) => {
                let rest = std::mem::take(&mut self.held);
                self.write(kind, &rest, &mut out);
                self.close(kind, false, &mut out);
            }
        }
        if self.thinking && self.reasoning_ended {
            out.push_str("</think>");
            self.thinking = false;
        }
        out
    }

    /// A header has ended at its `<|message|>`: begin its body.
    fn open(&mut self, header: &str, out: &mut String) {
        self.fresh = true;
        let kind = match read_header(header) {
            Header::Reasoning if self.answered => Body::Unseen,
            Header::Reasoning => {
                self.reasoning_ended = false;
                Body::Reasoning
            }
            Header::Content => {
                self.end_thinking(out);
                self.answered = true;
                Body::Content
            }
            Header::Call(name) => {
                self.end_thinking(out);
                self.answered = true;
                self.call = Some((name, String::new()));
                Body::Call
            }
            Header::Unseen => Body::Unseen,
        };
        self.state = State::Body(kind);
    }

    fn write(&mut self, kind: Body, text: &str, out: &mut String) {
        if text.is_empty() {
            return;
        }
        match kind {
            Body::Reasoning => {
                if !self.thinking {
                    out.push_str("<think>");
                    self.thinking = true;
                } else if self.fresh {
                    out.push_str("\n\n");
                }
                out.push_str(text);
            }
            Body::Content => {
                if self.fresh && self.wrote_content {
                    out.push_str("\n\n");
                }
                out.push_str(text);
                self.wrote_content = true;
            }
            Body::Call => {
                if let Some((_, args)) = self.call.as_mut() {
                    args.push_str(text);
                }
            }
            Body::Unseen => {}
        }
        self.fresh = false;
    }

    /// A body has ended — at `<|end|>` when `ended` says so.
    fn close(&mut self, kind: Body, ended: bool, out: &mut String) {
        match kind {
            Body::Reasoning => self.reasoning_ended = ended,
            Body::Call => self.write_call(out),
            Body::Content | Body::Unseen => {}
        }
    }

    fn end_thinking(&mut self, out: &mut String) {
        if self.thinking {
            out.push_str("</think>\n\n");
            self.thinking = false;
        }
    }

    /// The call as Hermes writes one. Arguments that are not JSON (the
    /// header's `<|constrain|>json` makes them JSON) are passed as a string,
    /// so the call still reaches the caller rather than its markup.
    fn write_call(&mut self, out: &mut String) {
        let Some((name, args)) = self.call.take() else {
            return;
        };
        let args = args.trim();
        let arguments = match serde_json::from_str::<serde_json::Value>(args) {
            Ok(_) => args.to_string(),
            Err(_) => serde_json::Value::String(args.to_string()).to_string(),
        };
        let name = serde_json::Value::String(name).to_string();
        out.push_str(&format!(
            "<tool_call>\n{{\"name\": {name}, \"arguments\": {arguments}}}\n</tool_call>"
        ));
    }
}

/// What a header says its body is.
#[derive(Debug, PartialEq, Eq)]
enum Header {
    Reasoning,
    Content,
    Call(String),
    Unseen,
}

/// Read one header — everything between the previous body's end and
/// `<|message|>`: an optional `<|start|>ROLE`, a recipient (` to=…`) in the
/// role part or after the channel, the channel (the LAST `<|channel|>`, past a
/// stray `<|channel|>commentary`), an optional `<|constrain|>TYPE`.
fn read_header(header: &str) -> Header {
    if let Some(after) = header.strip_prefix(START) {
        let role_end = after
            .find(|c: char| c.is_whitespace() || c == '<')
            .unwrap_or(after.len());
        if &after[..role_end] != "assistant" {
            return Header::Unseen;
        }
    }
    let word = |s: &str| -> String {
        let end = s
            .find(|c: char| c.is_whitespace() || c == '<')
            .unwrap_or(s.len());
        s[..end].to_string()
    };
    // `to=assistant` is a stray commentary's, addressed back to the model.
    let recipient = header
        .match_indices("to=")
        .map(|(at, _)| word(&header[at + 3..]))
        .filter(|r| !r.is_empty() && r != "assistant")
        .last();
    if let Some(recipient) = recipient {
        return match recipient.strip_prefix("functions.") {
            Some(name) if !name.is_empty() => Header::Call(name.to_string()),
            _ => Header::Unseen,
        };
    }
    let channel = header
        .rfind(CHANNEL)
        .map(|at| word(&header[at + CHANNEL.len()..]))
        .unwrap_or_default();
    if channel == "analysis" {
        Header::Reasoning
    } else {
        Header::Content
    }
}

/// How much of `text`'s end may be the start of a marker that ends a body.
fn marker_prefix_len(text: &str) -> usize {
    let longest = BODY_ENDS.iter().map(|m| m.len()).max().unwrap_or(0);
    (1..longest.min(text.len() + 1))
        .rev()
        .find(|&k| {
            let at = text.len() - k;
            text.is_char_boundary(at) && BODY_ENDS.iter().any(|m| m.starts_with(&text[at..]))
        })
        .unwrap_or(0)
}

/// A whole reply, rewritten.
pub(crate) fn rewrite(raw: &str) -> String {
    let mut reply = Reply::new();
    let mut out = reply.push(raw);
    out.push_str(&reply.finish());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every reply, fed whole and fed one character at a time, rewrites the
    /// same — a stream splits a reply where it likes.
    fn both_ways(raw: &str) -> String {
        let whole = rewrite(raw);
        let mut reply = Reply::new();
        let mut bits = String::new();
        for c in raw.chars() {
            bits.push_str(&reply.push(&c.to_string()));
        }
        bits.push_str(&reply.finish());
        assert_eq!(whole, bits, "chunking changed the rewrite of {raw:?}");
        whole
    }

    #[test]
    fn reasoning_then_an_answer() {
        assert_eq!(
            both_ways(
                "<|channel|>analysis<|message|>The user greets.<|end|>\
                 <|start|>assistant<|channel|>final<|message|>Hello!"
            ),
            "<think>The user greets.</think>\n\nHello!"
        );
        // No reasoning at all.
        assert_eq!(both_ways("<|channel|>final<|message|>Hi."), "Hi.");
    }

    /// A call, in either place gpt-oss names its recipient, ends on
    /// `<|call|>` — a stop token, so the text just ends — and becomes the
    /// Hermes call the readers parse.
    #[test]
    fn a_call_in_either_header_shape() {
        let call = "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\":\"Paris\"}}\n</tool_call>";
        assert_eq!(
            both_ways(
                "<|channel|>analysis<|message|>Need weather.<|end|><|start|>assistant\
                 <|channel|>commentary to=functions.get_weather <|constrain|>json\
                 <|message|>{\"city\":\"Paris\"}"
            ),
            format!("<think>Need weather.</think>\n\n{call}")
        );
        assert_eq!(
            both_ways(
                " to=functions.get_weather<|channel|>commentary json<|message|>{\"city\":\"Paris\"}"
            ),
            call
        );
        // Arguments that are not JSON still make a call, as a string.
        assert_eq!(
            both_ways("<|channel|>commentary to=functions.run<|message|>ls -la"),
            "<tool_call>\n{\"name\": \"run\", \"arguments\": \"ls -la\"}\n</tool_call>"
        );
    }

    /// A preamble on the commentary channel is content, set apart from the
    /// answer after it; a stray `<|channel|>commentary` before the real
    /// channel is passed over; a built-in tool's message is not shown.
    #[test]
    fn preambles_strays_and_built_in_tools() {
        assert_eq!(
            both_ways(
                "<|channel|>commentary<|message|>Let me check.<|end|>\
                 <|start|>assistant<|channel|>final<|message|>It is 4."
            ),
            "Let me check.\n\nIt is 4."
        );
        assert_eq!(
            both_ways(
                "<|channel|>commentary to=assistant<|channel|>analysis<|message|>Hmm.<|end|>\
                 <|start|>assistant<|channel|>final<|message|>Yes."
            ),
            "<think>Hmm.</think>\n\nYes."
        );
        assert_eq!(
            both_ways(
                "<|channel|>analysis<|message|>Search.<|end|><|start|>assistant to=browser.search\
                 <|channel|>commentary json<|message|>{\"query\":\"x\"}<|end|>\
                 <|start|>assistant<|channel|>final<|message|>Done."
            ),
            "<think>Search.</think>\n\nDone."
        );
    }

    /// Two analysis messages are one scratchpad; reasoning after the answer
    /// began is never shown as the answer.
    #[test]
    fn reasoning_is_one_leading_block() {
        assert_eq!(
            both_ways(
                "<|channel|>analysis<|message|>One.<|end|><|start|>assistant<|channel|>analysis\
                 <|message|>Two.<|end|><|start|>assistant<|channel|>final<|message|>Three."
            ),
            "<think>One.\n\nTwo.</think>\n\nThree."
        );
        assert_eq!(
            both_ways(
                "<|channel|>final<|message|>Answer.<|end|><|start|>assistant<|channel|>analysis\
                 <|message|>Afterthought."
            ),
            "Answer."
        );
    }

    /// A reply cut off inside its reasoning leaves the block open, as every
    /// reader expects of an unfinished scratchpad; one whose reasoning ended
    /// and nothing followed closes it.
    #[test]
    fn where_a_reply_ends() {
        assert_eq!(
            both_ways("<|channel|>analysis<|message|>Still thinking"),
            "<think>Still thinking"
        );
        assert_eq!(
            both_ways("<|channel|>analysis<|message|>Done thinking.<|end|>"),
            "<think>Done thinking.</think>"
        );
        // Cut off inside the next header: the header is markup.
        assert_eq!(
            both_ways("<|channel|>analysis<|message|>A.<|end|><|start|>assistant<|chan"),
            "<think>A.</think>"
        );
    }

    /// Text that does not open as harmony passes untouched — a model that
    /// ignored its format, a reply this already rewrote — so a second pass
    /// changes nothing.
    #[test]
    fn what_is_not_harmony_passes() {
        for text in [
            "Plain answer.",
            "<think>x</think>\n\nAnswer",
            "<|im_end|>odd",
            "",
            "  ",
        ] {
            assert_eq!(both_ways(text), text);
        }
        let once = rewrite(
            "<|channel|>analysis<|message|>R.<|end|><|start|>assistant<|channel|>final<|message|>A.",
        );
        assert_eq!(rewrite(&once), once);
        // Text after an `<|end|>` with no header is still the answer's.
        assert_eq!(
            both_ways("<|channel|>final<|message|>A.<|end|>B."),
            "A.\n\nB."
        );
        // A body keeps what only looks like the start of a marker.
        assert_eq!(both_ways("<|channel|>final<|message|>a <| b"), "a <| b");
    }
}
