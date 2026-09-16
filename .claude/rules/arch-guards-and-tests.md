---
paths:
  - "tests/**"
  - "examples/**"
---

# Repo-consistency guards and tests

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — what the rule replaced, what it was
measured at, and what a change must keep — lives in `docs/invariants/`.

**This file loads only when you touch the code it governs.** Read the linked
`docs/invariants/` topic file before changing code a rule names.

## A source-scanning guard is only as good as the spellings it knows (2026-08-30)

`tests/repo_consistency.rs` is where this project stops known past mistakes from
coming back. Five of those guards were tested by **planting the violation each
one exists to catch. Four did not notice** — all had been reporting success for
months (gotcha #413).

Three rules follow, and they are cheap.

**Scan statements, not lines.** `statements(text)` joins continuation lines and
closes the gap a wrapped chain leaves before its `.`, so
`s.metrics\n    .node_stats` reads back as `s.metrics.node_stats`. Every guard
matching a dotted path or a field-plus-operation must use it.
`self.shared_state.metrics.node_stats.requests_served_atomic.fetch_add(1,
Ordering::Relaxed)` is 99 characters at two levels of indentation, so **one more
nesting level and rustfmt splits it across four lines** — which is the ordinary
shape, not an edge case. It blinded the serving-accounting guard, the
per-request-state guard, the live-config guard (the one against #281's fourth
recurrence), the update-reporting guard, the advertised-load guard and the dial
guard.

**Take the whole body, never a character window.** The VRAM guard read
`src[start..start + 1600]` of a function 1698 characters long and was blind to
its last 100, where a planted boot-snapshot read passed. It was brittle in both
directions: the `cfg()` call its positive assertion depended on sat at offset
1568, **twenty characters inside the cap**, so twenty characters of unrelated
growth would have failed it on correct code. `fn_body(src, signature)` takes to
the closing brace in column zero. For the same reason, never match a literal
carrying indentation (`"shared\n        .config\n        .resources"`) — that is
pinned to whatever rustfmt produced the day it was written.

**A file-level `contains` is not a site-level claim.** The prompt-privacy guard
asserted each FILE mentions the send somewhere, the setting somewhere and
`cfg()` somewhere. Both files make unrelated `cfg()` calls, so the third
assertion was satisfied unconditionally and none of the three established the
gate was on the path; a second ungated send in either file kept it green. Use a
proximity window sized from the real code.

**And give every scan a self-test that plants the violation.** A repo-wide scan
that finds nothing is indistinguishable from one that *cannot* find anything, so
the scanner's reach has to be pinned the way any other behaviour is —
`the_statement_scanner_sees_a_chain_rustfmt_has_wrapped`,
`the_unbuffered_gguf_guard_catches_every_form_of_the_defect`,
`the_boot_snapshot_check_is_not_pinned_to_one_formatting`,
`the_prefix_sharing_guard_catches_an_ungated_send`.

**How to test one by hand**: a stray `.rs` under `src/` that no `mod`
references. The scanner walks the directory and finds it; the compiler never
sees it, so it need not even compile. Delete it afterwards.

**A guard too weak to fire is also too weak to be checked for correctness.**
Strengthening the #281 guard is what surfaced a real contradiction in its own
field list — it forbade all seventeen `.config.auto_manage.` fields while its
own comment stated the principle that only the four the Settings panel exposes
are live-settable, the rest being config-file/CLI where the boot value is
CORRECT.
