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
   read-before-edit for the Bash path. A subagent's own reads count: its hook
   input carries `agent_id`, and the gate follows that to the subagent's
   transcript. Until 2026-09-23 it read only the parent's, and denied every
   subagent edit of an existing file (gotcha #688).
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

**`python3 examples/research_gate_probe.py` is that planted violation, kept.**
One per Bash mutation form, printing which are caught, plus a null control that
fails if the gate denies nothing at all. Run it after touching the gate's
patterns: a regex that silently stops matching reads exactly like a rule nobody
breaks. Its first run (2026-09-16) found the gate catching redirects, `tee`,
`cp`/`mv`, `truncate` and quoted `sed -i`, while missing `sed -i -e`, every
python in-place edit, `perl -pi`, `git checkout --`, `git apply`, `patch <` and
`rm` — most of the path it exists to close, including the multi-edit python
script this repo reaches for routinely. All 20 forms are covered now, and the
six read-only forms stay unblocked.

## I do the mechanics. The user does what needs their hands. (2026-09-22)

**The user has never run a git command on this project. Not one.** Every
commit, push, tag, branch and force-push in the history is mine. So "should I
tag?" is not a question — tagging is my job, and asking it hands back work the
user cannot do and has never done.

**Theirs is only what physically requires them**: a password or private key
(`sign_release.sh`), credentials I do not hold, a machine I cannot reach, or a
physical act (un-maximising a browser window). **Everything else is mine**,
including every step of the release gate up to signing, and the deploy after it
(`feedback_deploy_without_asking.md`, said in 2026-08-31's words: *"stop waiting
for me to tell u to update local and proxmox nodes"*).

⚠ **"Confirm before outward-facing actions" means ANNOUNCE AND DO, not hand
back.** Sign-off is for the genuinely disruptive and irreversible — force-push,
history rewrite, tag deletion, publishing something public. A normal tag on a
release that CI deliberately leaves as a draft is none of those.

⚠ **A blocked CHECK is not a blocked TASK.** A denial that stops me inspecting
an artifact almost never stops the work; diagnose the denial (#675) instead of
parking. An overnight run was abandoned this way while the build it could not
`ls` had in fact succeeded.

**The test before asking**: *could the user even do this themselves?* If no, it
is mine and asking is noise. If yes but they have delegated it before, still
mine. Ask only where their answer changes what gets built — and then ask once.

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

**Nothing may block compaction.** A `SessionStart` hook (matcher `compact`,
`post-compact-status.sh`) runs AFTER a compaction and lists what is still
uncommitted, so the resumed session knows the work is pending. It replaced a
PreCompact hook that refused to compact while anything was uncommitted — and
that hook killed a whole session on 2026-09-23 (gotcha #687): work sat
uncommitted for hours, the context filled, `/compact` was refused, and "commit
and compact" then failed with "Prompt is too long", because there was no room
left to run the commit it demanded. Compaction never touches the disk; what it
can lose is only the knowledge that work is pending, and that is what the new
hook restores.

**So commit as you go — it is the only protection.** A session with hours of
uncommitted work across a dozen agents is one bad compaction from losing the
thread of it, whatever the hooks do.

Before ~70% context usage, proactively update `memory/MEMORY.md` with anything
worth carrying forward.

After compaction, before doing anything else: read `memory/MEMORY.md`, then `git log --oneline -10` and `git diff HEAD~3 --stat`. Don't re-do committed work.
