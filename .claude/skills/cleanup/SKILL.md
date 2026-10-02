---
name: cleanup
description: Verify recent SwarmLLM changes are complete — no stale references, no broken tests, no deferred items lost, docs updated. Run after changes to SharedState fields, API endpoints, JS file structure, broadcast channels, WS message formats or error-to-status mappings.
allowed-tools: Read, Grep, Glob, Bash, Edit, Write
effort: high
---

# Post-Change Cleanup Verification

Execute every item; fix what fails, then report PASS/FAIL per item.

1. **Build + tests**: `cargo fmt --check && cargo lint`, then `cargo dev-test 2>&1 | grep "test result:"`.
   `tests/repo_consistency.rs` already guards i18n parity, doubled sub-struct paths, console output, counts
   across documents, the MSRV and the payload budget — do not re-check those by hand.
2. **Stale references**: for every symbol, field, endpoint, channel or file this change removed or renamed,
   `grep -rn` across `src/ tests/ crates/ frontend/ docs/ examples/ python/ .claude/` — not just `src/`.
   Read any `// NOTE:` near the change for claims it made false.
3. **Frontend** (if `frontend/js/` changed): `for f in frontend/js/**/*.js; do node -c "$f"; done` and
   `node examples/frontend_load_check.js` (a syntax check alone misses an IIFE that throws — gotcha #568).
   A new user-visible string is translated into all 21 locales.
4. **Docs**: `docs/ARCHITECTURE.md` (source tree, endpoints, channels, WS types) matches the code; a changed
   rule is in its `arch-*.md` with evidence in `docs/invariants/`. Test counts are NOT refreshed here — that
   happens at release.
5. **Git**: everything committed and pushed; commit subjects stand alone for the public Discord feed.
