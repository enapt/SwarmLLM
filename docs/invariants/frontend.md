# Frontend

The evidence behind the rules in `.claude/rules/arch-frontend.md`: what each
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

## Frequency decides the nav — and frequency is a question about real users

The nav went from seven tabs to three plus a "More" menu on 2026-09-12, on the
principle that a destination earns a permanent seat by how OFTEN it is wanted,
never by how expert you must be to want it. That principle is right and stands.
**The frequency judgement inside it was wrong**, and only a user could say so:

> "the 2 link i use the most is ranking and the map to see what happening,
> who's there, and now there are burried."

Both had gone into the overflow. The trap is that the ranking was made for a
NEW user's first hour — the audience the round was explicitly optimising for —
and then applied to everybody, including the people who keep a node running and
are the reason the swarm has capacity at all. Two real audiences, one ordering.

**What fixed it was merging rather than promoting.** The map and the
leaderboard are one question asked twice — who is on this network, and what are
they contributing — so they became one primary destination ("Network") instead
of two competing for a slot. The nav is four wide, not five, and nothing went
back into a menu.

Rules that follow:

- **A destination removed from the nav keeps its URL.** `/admin/leaderboard`
  resolves to the Network view; it is in people's bookmarks and history, and
  landing on the dashboard instead reads as "the page was deleted".
- **Before promoting two things, ask whether they are one thing.** A nav slot
  is the scarcest surface in the product; merging costs none.
- **A frequency claim about users is checkable by asking them.** This one was
  wrong for four days and cost a daily user their two most-used links.

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

## Advice on an empty state must be advice the reader can take

`utils.js`'s chat empty state chooses its privacy line from
`modelData.has_first_shard && modelData.has_last_shard`, because the two
audiences need opposite sentences and only one of them can act.

**What it replaced.** It said, unconditionally: *Use the "Enable prompt privacy"
button in the bar above to encrypt your prompts end-to-end.* On a new node's
very first screen that was wrong three separate ways:

1. **The bar is not there.** "The bar above" is the session header's encryption
   banner, built by `chat.js::_renderSessionHeader` — which needs a SESSION. On
   the empty state no chat has been started, so the element it names does not
   exist yet.
2. **The button is not called that.** The control in that banner is
   `enc.enable_privacy` — "Turn on end-to-end encryption". Nothing in the
   product has ever been labelled "Enable prompt privacy".
3. **The reader usually cannot have it at all.** The enable button requires
   this device to hold the model's FIRST and LAST pieces (the boomerang's two
   ends). A new node holds nothing, so the action was unreachable — the first
   screen a new user reads told them to do something impossible.

The condition is the fix, not the wording: with both ends present the line
points at the Models tab (a control that exists on every page); without them it
states the requirement instead — *"this device needs the model's first and last
pieces. It does not have them yet."* Both were driven and asserted to differ,
which is the null control: a condition that changes nothing is the same bug in
a new coat.

**The general rule.** An instruction naming a control names three things that
can each be false independently — that the control exists, that it is called
that, and that it is reachable in this state. A string that hard-codes all
three goes stale silently, because nothing compiles it against the UI. Prefer
naming a PLACE that always exists over a button that sometimes does.

## A key must cover every colour the thing it explains can paint

`dashboard-shards.js::buildShardLegend` lists all six `data-loc` values the
route strip can emit — `live`, `disk`, `swarm`, `thin`, `moving`, `absent` —
and its swatches ARE `.avail-seg` elements, so they inherit the strip's own
colour rules and cannot drift from it.

The first cut left `moving` out as "transient". That is the original defect in
miniature: a pulsing segment with no entry in the key leaves the reader exactly
where the hover-only titles did, for that one state. If a colour can appear, it
is in the key.

Two things a change must keep: the swatch inherits `data-loc` rather than
restating any colour (only geometry and the pulse are overridden), and
`absent` — which has no background COLOUR at all, being a 45° hatch plus a
border — keeps its legend-size override, or it renders as an invisible swatch
next to the words "No computer has this".

The key renders in the EXPANDED view only. Details on demand (Shneiderman
1996): collapsed rows keep their density, and anyone who opens a model to look
at its pieces gets the key beside them.

## A panel that could not READ its settings must not be able to WRITE them

`App.settings._loadedConfig` is the baseline `save` diffs against, and `null`
means the form was never populated from a real config. Three states, set by
`_setSaveable`: `'loading'` (disabled, nothing said), `'ok'` (enabled), and
`'unreadable'` (disabled, `settings.unreadable` shown beside the button). The
Save button carries `disabled` in the markup so the gap between opening the
panel and the config arriving is not a window in which it can be pressed.

**What this replaced.** `load` did `if (!data) return;` — on a 401 from a
rotated key or a 503 from a node still starting, the panel was left showing its
static HTML defaults (10 concurrent requests, unlimited bandwidth, 50 GB of
disk) and `save` read all eight fields out of the DOM and PUT every one. Change
one setting during a hiccup and the other seven were overwritten with values the
user never chose, under a toast saying "Settings saved". Measured on an isolated
dev node: the node held `max_disk_mb = 120000`, and the old save would have
written `50000`.

**What a change must keep.** `save` sends only fields that DIFFER from the
baseline. This is not an optimisation — `api/admin.rs::update_config` builds on
the live config and applies only the fields a request names, so an omitted field
is deliberately left alone. That is what stops this panel reverting a setting
changed through another endpoint while the modal sat open, and it is the
handler's own design: its comment records that building from the boot snapshot
instead made two sequential saves undo each other. A client that always sends a
full document throws that away. `_readForm` is shared by the baseline and the
send so a field cannot be read one way in one and another way in the other.

→ gotcha #595

## A periodic refresher may repaint a display; it may not repaint an input

`dashboard.js::loadInitial` does NOT populate the Settings form.
`App.settings.load()` owns those fields and runs on every open.

**What this replaced.** `loadInitial` wrote four settings fields, and
`notifications.js` runs it on a **30-second poll** — so it fired while the panel
was open and in use, reverting the fields to the node's stored values. Measured:
drag Max Disk from 120 GB to 300 GB, wait one poll, and the slider reads 120 GB
again with nothing said. Every settings change not saved within thirty seconds
was silently discarded, which reads to the user as a setting that will not
stick. The dashboard copy was pure redundancy — there were no readers of those
elements outside `settings.js`, and the panel still opens showing the node's
real values without it.

**What a change must keep.** Before adding a poll or a live updater that touches
the DOM, ask which of the elements it writes are INPUTS. Repainting a display
costs nothing; repainting an input discards what the user was in the middle of
doing, and leaves no trace.

→ gotcha #596

## A write whose response is never read reports failure as success

`U.apiAction(url, opts, onSuccess, { fallback })` is how the dashboard performs
a write. It checks `resp.ok`, shows the daemon's own reason through
`getApiErrorMessage`, and returns a boolean the caller can act on.

**What this replaced.** `App.authFetch` resolves for a 401 and a 500 exactly as
it does for a 200 — `fetch` rejects only on a network-level failure — so
`await authFetch(...)` followed by a success banner announced every refusal as a
success. Five paths did it: provider API keys (which also **cleared the input**,
leaving nothing to retry with), the provider key-source dropdown, the Claude
subscription toggle, the first-run nickname (swallowed entirely), and Shut Down
— which replaced the whole page with "shutting down…" against a 403 from a
still-running node, turning the dashboard into a dead end. That last one is the
`LocalOnly` split of gotcha #309 seen from the other side: the variant exists so
the message can name the machine the command must be run from, and the frontend
discarded it by never reading the body.

**What a change must keep.** A control that stays where the user put it is a
claim the node agreed: on failure the subscription toggle reverts and the
key-source dropdown reloads from the daemon, so neither shows a state the node
is not in. To find a regression of this class, grep for
`method: '(PUT|POST|DELETE|PATCH)'` with no `.ok`, `apiAction`,
`getApiErrorMessage` or `throw` within about fourteen lines.

→ gotcha #597

## Text that sits next to a control goes in a span, never on its wrapper

`I18n.translatePage` does `el.textContent = t(key)` for every `[data-i18n]`
element, and `textContent` replaces everything inside that element. So a
translated element that CONTAINS a control deletes the control on every page
load, in every language including English.
`a_translated_element_never_wraps_a_control_it_would_delete` in
`tests/repo_consistency.rs` fails the build on one.

**What this replaced.** `<label data-i18n="settings.key_source_label">` wrapped
the `provider-key-source` `<select>`, so the Cloud Providers "Key source"
setting (Auto / .env only / Dashboard only) was not in the DOM for any user:
`init.js` bound its change handler to `null`, `settings.js` wrote its value
behind an `if (sel)` that was never true, and three translated options across 21
locales were strings nobody could see. The markup was valid and the translations
correct; the control was simply absent, and no test that asks whether a feature
*works* can see a feature that is not *there*. The whole-file scan found exactly
one instance, which is why it survived so long.

**What a change must keep.** The scan's self-test plants the defect and requires
it to fire, and also asserts the corrected shape does not — a guard that goes
off on correct markup is one somebody will delete rather than satisfy.

→ gotcha #598

## A partial writer writes its part, never a shorter whole

`shardWhereText` is the one place a shard row's "where" sentence is composed —
locality plus, for a part this computer holds, how many others have a copy.
`_patchShardRow` renders through it whenever the caller passes the shard object,
sets the swatch from `loc` independently of the text (the swatch is the locality;
the sentence is locality *plus* replica count), and **leaves the sentence alone
when the caller passes neither** — which is how a caller says "I know where this
part is, but not how many copies exist".

**What this replaced.** `updateShardsLive` has two patch sources. The
shardRegistry branch passed the shard object and was correct. The acquisitions
branch open-coded `whereText: I18n.t('shard.loc.disk')` on completion — but
`shard_details` carries index, state, progress and bytes and **no `holders`**, so
it could not know the replica count and wrote the sentence without it. The
registry branch normally repaired that on the same tick, except it is skipped
when the registry has not changed, while the acquisition keeps reporting
`complete`. Measured: the tick that completed a download said "· also on 4 other
computers", the next tick dropped it, and on a quiet swarm it never came back —
losing precisely the fact the docstring calls "the fact that decides whether
losing this machine loses the model".

**What a change must keep.** The acquisitions branch still sets the row STATE on
completion: leaving the row on `downloading` is what makes the registry branch
skip it entirely (`if (current === 'downloading') return;`). The tell for this
class of bug is a caller passing a pre-rendered string where a sibling passes the
object — that caller has less information and is about to flatten it.

**Known and left:** the role badges ("Reads your prompt", "Removed",
"Disagrees") are build-time only. The live tick's source does not carry
`disputed` or `removed_by_user`, so a badge can outlive its fact for one refresh
cycle.

→ gotcha #599

## The app shell never page-scrolls, and on iOS that needs both halves

**Rule:** `.claude/rules/arch-frontend.md` § "The app shell never page-scrolls".

### What this replaced

`body` already said `overflow: hidden` and `.app-layout` was sized to the
viewport, so the design was always "the page does not scroll; each long region
scrolls inside itself". That held in every desktop browser and nowhere on
iPad/iPhone Safari, where the whole dashboard scrolled as one block — sidebar,
chat header and message list drifting together, the input reachable only by
scrolling past everything above it (report #034, 2026-09-14).

### Why neither half works alone

1. **`overflow: hidden` on `body` is not enough on iOS.** The scrolling element
   there is the documentElement, so it is set on `html` too — and the height is
   FIXED rather than a minimum, because a minimum lets content grow the page
   out from under the lock.
2. **`100vh` on iOS is the chrome-COLLAPSED viewport**, taller than what is on
   screen, so the shell rendered larger than the space it had.

And the order is not free: **Safari measures `dvh` against that same larger
viewport for as long as the page is itself scrollable.** So switching units
alone changes nothing — the lock is what makes the unit correct. The report that
found this said the two fixes were both needed; what it did not have is *why*,
which is also what says the lock is the load-bearing half.

### What a change must keep

- Every `vh` length paired with a `dvh` one, `vh` first so a browser that does
  not know the unit keeps today's behaviour. Guarded by
  `every_viewport_height_in_css_has_a_dynamic_fallback_beside_it`, which fails
  on a new bare `vh` AND on a removed `dvh` sibling.
- **Check the inner scrollers before locking anything.** `.container`,
  `.chat-messages` and `.session-list` all scroll internally; locking a page
  where some long region does not makes it unreachable, which is far worse than
  the bug being fixed.
- A `touchmove` guard in JS is the other commonly-cited remedy and is
  deliberately unused: it is easy to write one that also kills the inner
  scrolling this layout depends on, and with nothing taller than the viewport
  there is nowhere left to scroll to.

⚠ **Verified by mechanism, not on the device** — no WebKit engine here, and this
machine's window manager ignores resize, so narrow-viewport rendering is still
unverified. The null control reproduces the symptom in Chrome by restoring
exactly the two conditions and shows it gone with them. FUTURE_WORK #82 is the
open follow-up: the on-screen keyboard has never been exercised against the
now-locked shell.

## One word per thing, in the UI, in every language

**Rule:** `.claude/rules/arch-frontend.md` § "One word per thing".

### What this replaced

The interface called a piece of a model a "shard", a "part" AND a "piece", and a
machine a "peer", a "node", a "computer" AND a "device" — not by area, but
mixed: `dashboard.info_shards` read "Parts" while the tip beside it read "All
shards available", and one activity feed carried two spellings two lines apart.
Every locale had inherited the split, several having added a third word of their
own (Spanish used *fragmento*, *parte* and *pieza* for one thing).

### The five surfaces

Only the first is findable by searching for the old word, which is why the job
was repeatedly estimated as smaller than it was:

1. `frontend/i18n/*.json` — 21 locales.
2. `frontend/index.html` fallback text inside `data-i18n` elements, which had
   also DRIFTED from `en.json` independently.
3. Rust `ActivityEvent` messages — shown verbatim when a `kind` has no
   `activity.*` key.
4. Labels assembled in a variable and only later interpolated into one of those
   (`scan.rs`'s `shard_label`, four in `admin_models/shards.rs`).
5. A helper that RETURNS the word: `ShardId::display_index` → `"part 15"`. This
   one also explained a defect on screen the whole time and never reported —
   "P2P: downloading shard shard 15 from peer", two callers writing the noun the
   helper already supplied.

### What a change must keep

- The guard checks `en.json` only. Translations are prose in another language;
  pinning their vocabulary from a Rust test would be guessing.
- Two strings mean units of WORK rather than parts of a model
  (`perf.served_detail`, `dashboard.stat_forwards_tip`) and keep "pieces"; they
  are allowlisted by name.
- **Identifiers, `shard_NNN.bin`, `ShardId`, `hosted_shards` and wire fields are
  NOT renamed.** This is a change to what the UI says.
- A **duplicated-word scan** is the detector for the failure a leftover scan
  cannot see: when two source words map to one target nothing old remains, so
  "is the old word gone?" answers yes while the sentence reads "computers
  computers". Exclude grammatical reduplication ("vous vous").
