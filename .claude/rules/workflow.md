# Workflow Rules

## Commit and Push After Each Task

This project requires `git push` after every logical unit of work — don't batch to end of session. Long sessions and compactions can lose uncommitted work.

Sequence (always run together, in this order):
```
cargo fmt && cargo clippy --all-targets -- -D warnings && git add -A && git commit -m "..." && git push origin main
```

`cargo fmt` actually applies formatting (not `--check`). `cargo clippy` must be zero warnings — fix before pushing.

## Pushes are public-facing (2026-07-22)

The repo is public AND a GitHub webhook relays activity to the project
**Discord**. Every commit and push is broadcast to real users — including
non-technical ones evaluating whether to run this software.

Write for that audience without dumbing anything down:

- **Commit subjects must stand alone.** They appear in a feed with no
  surrounding context. `fix(scheduler): never form a TP group the request
  does not need` reads fine cold; `fix bug 1` does not.
- **Lead with user-visible impact, then mechanism.** A reader in Discord
  wants to know whether this affects them before they care how it works.
- **Never reference a person by name** in a commit message, and never
  paste inbound bug reports / private correspondence into the repo — see
  the `user_bug_report_*.txt` gitignore entry. Cite an issue number
  instead.
- **No alarming shorthand without context.** "security fix", "data loss",
  "broken" in a subject line will be read literally by users deciding
  whether to upgrade. If severity is real, state scope and affected
  versions in the body.
- **Announce disruptive git operations before doing them.** Force-pushes,
  history rewrites, and tag deletions all surface in the feed and look
  like something went wrong. Get explicit sign-off, and say why in the
  commit or a Discord note.
- **Don't push half-finished work to main** expecting to fix it in the
  next commit — the intermediate state is visible.

## Memory Management Around Compaction

The PreCompact hook blocks compaction if anything is uncommitted or `cargo check` fails. Before ~70% context usage, proactively update `memory/MEMORY.md` with anything worth carrying forward.

After compaction, before doing anything else: read `memory/MEMORY.md`, then `git log --oneline -10` and `git diff HEAD~3 --stat`. Don't re-do committed work.
