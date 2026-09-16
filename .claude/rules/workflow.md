# Workflow Rules

## Research EVERY task before touching code (user rule, reaffirmed 2026-09-12)

Not only the unfamiliar or non-trivial ones — every task, every time, before
the first edit. `.claude/rules/diagnosis.md` § 0 gives the method and the
evidence; this is the scope: **there is no task small enough to skip it.**

What "research" means here, in order:

1. **How do the systems with more scars than ours do it?** vLLM, llama.cpp,
   TGI/SGLang, libp2p, WireGuard — the failure mode usually has a NAME there,
   and the one detail you would otherwise get wrong is usually in the
   rationale. Fetch the actual source when the reference behaviour is a
   line of code (minja's `ordered_json`, WireGuard's per-keypair counter).
2. **What is the CURRENT API of the crate you are about to call?** Read the
   registry source under `~/.cargo/registry/src/`, not memory of it — the
   method may be feature-gated, renamed, or absent in the pinned version.
3. **What does this repo already know?** `memory/gotchas.md`,
   `docs/FUTURE_WORK.md` (the entry BODY), `docs/invariants/<topic>.md`,
   `closed_findings.md`. Half the "new" defects here were re-derivations.

Write down what the research changed — in the code comment, the round log or
the FUTURE_WORK entry — so the next reader can check the reasoning rather than
redo it. Research that changed nothing is still worth one line saying so.

Why it is a rule: v0.3.175 shipped with a defect found within hours, and the
user has asked for this twice. Both times research was skipped on 2026-08-03
the work went worse; both times it was done, it changed the implementation.

## What the hooks enforce (2026-09-16)

The three rules above were prose for months and were skipped anyway, so the
checkable parts now run as `PreToolUse` hooks. **They gate MUTATIONS only** —
reading is never blocked, so every denial is resolvable by reading something.

`research-gate.sh` refuses to let a file be changed when:

1. **its subsystem rules never loaded.** `arch-*.md` files auto-load on the
   **Read tool only**; `cat`/`sed`/`grep` through Bash do not trigger them.
   Measured, not assumed: a session that had read `repo_consistency.rs`,
   `CLAUDE.md` and several `src/` files entirely through Bash had logged zero
   `path_glob_match` events, and one `Read` logged one immediately. In
   bypass-permissions mode, where Bash is the default for reading and editing,
   this silently disabled the whole path-scoped rules architecture **and**
   Claude Code's read-before-edit check, which is attached to the Edit tool.
2. **the file exists and this session has never looked at it.** Restores
   read-before-edit for the Bash path.
3. **the task consulted nothing this repo already knows.** Scoped per task via
   `prompt_id`, satisfied by `gotchas.md`, `closed_findings.md`,
   `docs/invariants/`, `FUTURE_WORK.md`, the sweep log, or a web search — item
   3 of the research rule, the only one of the three a machine can check.

`commit-gate.sh` runs `cargo test --test repo_consistency` before a `git commit`
that touches a file whose figures another document restates (`CLAUDE.md`,
`README.md`, `docs/ARCHITECTURE.md`, `Cargo.toml`, `frontend/i18n/*.json`). A
stamp file was the obvious design and is the wrong one — it records that the
test ran, not that it ran against THIS content. ~25 s, and only on those files.

**All of them fail OPEN** — unparseable payload, missing transcript, missing
cargo or a timeout lets the call through. A gate that cannot read its input must
not become a gate that blocks everything, which is exactly what
`pre-edit-check.sh` became when it read the wrong payload key and sat inert for
months. **Verify a hook by its OUTPUT, never its exit code** (#614): both gates
were written, looked right, and did nothing until a planted violation proved
otherwise — `research-gate.sh`'s first version accepted a one-line `grep` of a
rules file as having loaded it.

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

The PreCompact hook blocks compaction while anything is uncommitted, and names the
files. Commit, and compaction proceeds. (It no longer runs `cargo check`: that
arm could not fail and only added ~30s to every compaction. And the hook itself
was missing between 2026-04-08 and 2026-09-09 — deleted by an unrelated commit
while this line went on promising it, so treat a documented safety net as a
claim to verify rather than a fact.)

Before ~70% context usage, proactively update `memory/MEMORY.md` with anything
worth carrying forward.

After compaction, before doing anything else: read `memory/MEMORY.md`, then `git log --oneline -10` and `git diff HEAD~3 --stat`. Don't re-do committed work.
