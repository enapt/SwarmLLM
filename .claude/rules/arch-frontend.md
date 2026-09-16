---
paths:
  - "frontend/**"
---

# Frontend

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## The app shell never page-scrolls, and on iOS that needs both halves

`html` AND `body` are locked (`height: 100dvh` with a `100vh` fallback first,
`overflow: hidden`), because every long region already scrolls inside itself —
`.container`, `.chat-messages`, `.session-list`. Two WebKit behaviours defeat
the lock and they are **entangled**: `overflow: hidden` on `body` alone does not
stop the page scrolling on iOS, and `100vh` there is the chrome-COLLAPSED
viewport. Safari measures `dvh` against that larger viewport for as long as the
page is itself scrollable, so **the lock is what makes the unit correct** — a
`dvh`-only change does nothing.

Every `vh` length is paired with a `dvh` one, guarded by
`every_viewport_height_in_css_has_a_dynamic_fallback_beside_it`. Before locking
a page, check every long region scrolls internally: locking one that does not
makes it unreachable, which is worse than the bug.

→ `docs/invariants/frontend.md`

## One word per thing, in the UI, in every language

A piece of a model is a **part**; a machine is a **computer**. Not "shard",
"piece", "peer" or "node" — all four were in use at once, sometimes two in one
sentence. The pool feature is the single exception and keeps **device**:
machines you own and link are a different idea from anyone's computer on the
swarm, and "My Devices" is a nav destination named for the first.

`the_english_ui_uses_one_word_for_a_model_part_and_one_for_a_machine` fails the
build on either old word in `en.json`. **The vocabulary lives on five surfaces**
and only the first is findable by searching for the old word: the locale files,
the markup's `data-i18n` fallback text, Rust `ActivityEvent` messages, labels
assembled in a variable and only later interpolated into one, and a helper that
RETURNS the word (`ShardId::display_index`). Identifiers, `shard_NNN.bin` and
wire fields are deliberately NOT renamed.

→ `docs/invariants/frontend.md`

## "Is the empty chat state showing?" is a question about the DOM

**`App.chat.refreshEmptyState()`** rebuilds `#chat-empty` in place when that is
what is on screen, and no-ops otherwise. Every caller that needs the empty state
to reflect changed data goes through it: the `stats_update` tick, the model list
loading, a model being picked, and entering the Chat tab.

→ `docs/invariants/frontend.md`

## Frequency decides the nav, and a frequency claim is checkable by asking

A destination earns a permanent seat by how OFTEN it is wanted, never by how
expert you must be to want it. The ranking made for a new user's first hour was
applied to everybody and buried the two links a daily node operator used most.
Fixed by MERGING rather than promoting — the map and the leaderboard are one
question asked twice, so they are one destination. A destination removed from
the nav keeps its URL.

→ `docs/invariants/frontend.md`

## A control in persistent chrome earns its seat, and says its own name

Anything that sits in the header is on EVERY page. Node operation (node id,
Stop, connect-a-node, auto-manage status) lives on the Dashboard; a preference
set once (language, appearance) lives in Settings, once; a state indicator
renders only while its state is on (the private-mode chip). An icon whose only
explanation is `title=` is not labelled — NN/g rules out hover for this, and
only home/print/search are near-universal.

Three things that broke during the move and must be kept: `#auto-manage-dot`'s
base styling is on the ID selector because `render()` replaces `className`; a
control inside a `data-collapse` header must not collapse it (and the header's
own empty area must still collapse); and a popover opened from a delegated click
handler opens on the NEXT TICK, or the same event's outside-click listener
closes it immediately.

→ `docs/invariants/frontend.md`

## A panel that could not READ its settings must not be able to WRITE them

`App.settings._loadedConfig` is the baseline `save` diffs against; `null` means
the form was never populated from a real config and Save is refused. Three
states via `_setSaveable`: `'loading'`, `'ok'`, `'unreadable'`. The Save button
carries `disabled` in the markup, so the gap before the config arrives is not a
window in which it can be pressed.

**`save` sends only the fields that CHANGED.** `api/admin.rs::update_config`
builds on the live config and applies only the fields a request names — an
omitted field is deliberately left alone. A client that always sends a full
document throws that away, and turns a failed read into a write of the markup's
defaults over the user's real settings.

**And a periodic refresher may repaint a display, never an input.**
`dashboard.js::loadInitial` does not populate the Settings form;
`App.settings.load()` owns it. `loadInitial` is on a 30-second poll, so its copy
discarded every settings change not saved within thirty seconds.

→ `docs/invariants/frontend.md`

## A write whose response is never read reports failure as success

**`U.apiAction(url, opts, onSuccess, { fallback })`** is how the dashboard
performs a write. `App.authFetch` resolves for a 401 and a 500 exactly as for a
200 — `fetch` rejects only on a network failure — so `await authFetch(...)`
followed by a success banner announces every refusal as a success. Five paths
did it, including one that cleared the API-key input the daemon had just
rejected. A control that stays where the user put it is a claim the node agreed:
revert it, or reload it, when the write fails.

→ `docs/invariants/frontend.md`

## Text that sits next to a control goes in a span, never on its wrapper

`I18n.translatePage` sets `textContent` on every `[data-i18n]` element, which
deletes whatever that element contains. A translated element wrapping a control
removes it on every page load, in every language — that is how the "Key source"
setting came to exist in the markup and in 21 locales but in no user's DOM.
`a_translated_element_never_wraps_a_control_it_would_delete` in
`tests/repo_consistency.rs` fails the build on one.

→ `docs/invariants/frontend.md`

## A partial writer writes its part, never a shorter whole

`shardWhereText` composes a shard row's "where" sentence — locality plus, for a
part this computer holds, how many others have a copy. `_patchShardRow` renders
through it when given the shard object, sets the swatch from `loc`
independently of the text, and leaves the sentence alone when given neither.
The acquisitions branch of `updateShardsLive` knows the locality but not the
replica count, so it writes no sentence on completion rather than a truncated
one — it had been dropping "· also on N other computers" permanently on a quiet
swarm. The tell is a caller passing a pre-rendered string where a sibling passes
the object.

→ `docs/invariants/frontend.md`

## Advice on an empty state must be advice the reader can take

An instruction naming a control asserts three things that can each be false on
their own: that the control exists, that it is called that, and that it is
reachable in this state. Nothing compiles a string against the UI, so all three
go stale silently. The chat empty state now picks its privacy line from
`has_first_shard && has_last_shard`, and a key/legend lists EVERY value the
thing it explains can paint (`buildShardLegend`, whose swatches are
`.avail-seg` so they cannot drift from the strip).

→ `docs/invariants/frontend.md`

## Every surface that shows a model's reply renders it the same way

**`utils.renderReplyInto(el, text, opts)`** is the one place a reply becomes
rendered HTML: it adds `md-body`, runs `renderMarkdown`, and keeps the source on
`el._rawText` so Copy hands back what the model actually wrote rather than the
markup stripped of its markdown. `chat.js::_renderReply` is now a thin wrapper
over it.

→ `docs/invariants/frontend.md`

## Frontend Event Handling

All WS events are handled by `_handleActivityEvent()` in notifications.js. Do NOT:
- Add new WS message types (use activity_event with a new `kind`)
- Add direct `showToast()` calls for backend events (set `toast_level` on the ActivityEvent instead)
- Add direct `logActivity()` calls from WS handlers (everything goes through `_handleActivityEvent`)

## Frontend Storage

All storage keys are registered as constants on `App` in state.js (e.g., `App.MODEL_SORT_KEY`). Do NOT use raw string literals for localStorage/sessionStorage keys.

## Frontend Data Fetching

Use `App.data.loadModels()` and `App.data.loadStats()` for model/stats data. Do NOT make independent `authFetch('/api/admin/models')` calls from components — this bypasses the dedup cache.

## Frontend Component IIFE Boilerplate

Every `frontend/js/components/*.js` file opens with the same boilerplate
inside its IIFE:

```js
(function () {
  if (!window.App) return;
  var U = App.utils;   // <-- mandatory if the component calls escapeHtml / formatBytes / etc.
  // ...
})();
```

`U.escapeHtml`, `U.formatBytes`, `U.formatMB`, etc. are pulled off
`App.utils`, which is populated by `core/utils.js`. Components that
reference `U.*` without declaring `var U = App.utils` first will throw
`ReferenceError: U is not defined` at the call site — the R111
swarm-tab regression hid behind this until the Capacity Plan view
rendered for the first time. When adding a new component, copy the
existing boilerplate from a sibling file (e.g. `chat.js`).
