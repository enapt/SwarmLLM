# Frontend

The evidence behind the rules in `.claude/rules/architecture.md`: what each
rule replaced, what it was measured at, and what a change must keep.

Every entry here was paid for. **Read the entry before changing the code it
names** — the rule statement in `architecture.md` is the summary, this is the
reasoning, and several of these describe a fix that looked obviously correct
and was not.

## "Is the empty chat state showing?" is a question about the DOM

**`App.chat.refreshEmptyState()`** rebuilds `#chat-empty` in place when that is
what is on screen, and no-ops otherwise. Every caller that needs the empty state
to reflect changed data goes through it: the `stats_update` tick, the model list
loading, a model being picked, and entering the Chat tab.

**Why.** The empty state is a live view — it names the picked model and renders
the swarm catalogue out of `App.data.cache.stats` — but it is built once, from a
cache that is still empty at first render. Four separate callers refreshed it,
and all four asked whether to by the SESSION: `currentSessionId` set, the
session exists, `messages.length === 0`. That proxy is false in the commonest
case there is — the very first render, before any session has been created — so
opening the app on Chat gave a state frozen at page load. Measured on a node
with 6 peers, 11 ready models and 4h50m of uptime: "no models available yet ·
looking for other computers", indefinitely (report #027). One of those callers
carried a comment naming that precise failure as the thing it existed to
prevent.

Three things a change here must keep.

- **A hidden `#chat-empty` means a conversation is on screen.**
  `appendMessageToDOM` hides it rather than removing it, so rebuilding it
  unconditionally puts a fresh, visible "type a message below to start" above a
  live reply — verified by running the unguarded version against a streaming
  conversation.
- **`chat.js` may still ask a session question.** `newSession` asks whether to
  reuse an empty session rather than create a second one, which is not about
  what is rendered. That is why the guard exempts the file rather than the
  pattern.
- **Entering the tab refreshes.** Every other tab reloads something on entry and
  chat reloaded nothing, so the WS tick was the only repair path and it only ran
  while the tab was already open.

`the_chat_empty_state_is_not_refreshed_on_a_session_shaped_guard` in
`tests/repo_consistency.rs` fails the build on a fifth occurrence, with a
self-test that plants the multi-line form all four real ones were written in.

## Every surface that shows a model's reply renders it the same way

**`utils.renderReplyInto(el, text, opts)`** is the one place a reply becomes
rendered HTML: it adds `md-body`, runs `renderMarkdown`, and keeps the source on
`el._rawText` so Copy hands back what the model actually wrote rather than the
markup stripped of its markdown. `chat.js::_renderReply` is now a thin wrapper
over it.

**Why.** It lived on `App.chat`, so the Compare tab did not have it and wrote
replies with `.textContent`: bold, lists, code blocks and tables all arrived as
literal asterisks, dashes and pipes — on the one screen built for judging
replies side by side (report #026). The same tab also forced `stream: false`
and sat on a spinner until the whole reply landed, so the model running on a
processor-only peer, exactly the one a comparison exists to identify, was the
one that showed nothing for 30-60 s and could not be told apart from a stall.
Both were omissions rather than decisions: no comment weighed either, and the
long comment above the request is about the backstop timeout, which has nothing
to do with whether the reply streams while it is produced.

Four things a change here must keep.

- **A stream is re-assembled into the shape the renderer already reads.**
  Compare accumulates `content_block_delta` text and `message_delta` usage into
  the same `{content:[{type:'text',text}],usage}` object the non-streaming reply
  produced — what the official SDKs' `.accumulate()` does, and what
  `renderHistory` was already building by hand. One result shape, whichever way
  the text arrived.
- **`flush` for a final render.** The rAF coalescing that keeps a streaming
  reply from re-rendering the document per token is suspended in a backgrounded
  tab (gotcha #471), so a reply that finished while the user was elsewhere would
  sit on its last frame.
- **An error is not markdown.** It is a message from this node, so it stays
  `textContent` with `.error` and never goes through the renderer.
- **An unknown token count is not zero.** The Anthropic encoder omits
  `input_tokens` rather than sending a confident zero; the card omits the chip
  for the same reason, and `null` survives into the stored history entry so a
  restored card says what the live one said.
