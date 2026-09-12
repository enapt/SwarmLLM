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

## A control in persistent chrome earns its seat, and says its own name

The header carried seven icon-only buttons — share peer address, auto-manage
status, private mode, Settings, language, theme, node id, stop — on every page,
chat included. Each explained itself only through `title=`.

**The research is the reason this is a rule and not a preference.** NN/g's icon
guidance is explicit: *"Don't rely on hover to reveal text labels: not only does
it increase the interaction cost, but it also fails to translate well on touch
devices."* It also puts the set of near-universally recognised icons at roughly
home, print and the magnifying glass — none of the seven. Microsoft's
icon-only Outlook toolbar is the worked example of what that costs: users could
not tell what the icons did until labels were added. Separately, the guidance on
destructive controls is that they get separation and friction rather than
sharing a row of same-weight buttons with everything else — the stop control sat
flush against the node id in a row of identical ghost buttons.

**What changed, and the rule each move follows.**

- **Group by where a thing belongs, never by how expert you must be to want
  it** — the same principle as the nav change, and the reason there is still no
  "Advanced mode" (Home Assistant is removing theirs; see the nav entry).
  Node operation went to the Dashboard, which is the page that answers "how is
  my node doing": node id and **Stop** into the Node panel header, auto-manage
  status beside the models it manages, "Connect a node" into the Network panel.
- **A duplicated control has one home.** Language was already in Settings and is
  now only there. Theme was NOT — it existed solely as a header toggle cycling
  dark → light → system, so the glyph was the only reading of which of three
  states was in force. It is a named select in Settings → Preferences now, which
  also makes `welcome.card4_body` true: the tour had been promising "the gear
  icon opens Settings — language, theme, contribution caps" for months.
- **A state indicator renders when the state is on, not always.** Private mode
  was a padlock that was always present and so said nothing about whether
  anything was locked. It is a labelled chip shown only while private mode is
  on, and clicking it navigates to My Devices rather than toggling — a one-click
  disable of a privacy guarantee from permanent chrome is exactly the accidental
  destructive click. Same reasoning as the trust column's em dash: a default
  rendered as a measurement is worse than nothing.

**Three things a change here must keep**, each of which broke during the move
and was caught only by driving the page:

1. **`#auto-manage-dot`'s base styling lives on the ID selector.**
   `auto-manage-status.js::render` does `dot.className = state`, which REPLACES
   the class list — any base styling carried by a class is wiped on the first
   render. Its geometry used to be inline because the dot was absolutely
   positioned over an icon; it is inline-before-a-label now.
2. **A control inside a `data-collapse` panel header must not collapse it.**
   The Network header now carries the connect popover, so typing an address
   would have folded the panel away mid-edit. The handler ignores clicks that
   land on anything interactive — and a click on the header's own empty area
   must still collapse, which is the null control for that guard.
3. **Opening a popover from a delegated click handler happens on the next
   tick.** The `data-goto-network-code` CTA and the "close the share popover
   when the click landed outside `.share-btn-wrap`" listener are both document
   click handlers on the SAME event, and the CTA is always outside that
   wrapper — so opening it inline opened and immediately closed it. The visible
   symptom was simply that nothing happened.

**Method note.** Every one of those three was invisible in the diff and green in
`node -c`; all three were found by driving the real page. And the first attempt
to verify the third read a STALE frontend: `cargo test` and `cargo clippy` run
with default features and overwrite `target/debug/swarmllm` with an `embedded`
build, so the dev node silently went back to serving a baked-in snapshot of the
frontend (gotcha #573). Confirm with
`curl -s :PORT/static/js/<file> | grep <your edit>` before doubting the browser.
