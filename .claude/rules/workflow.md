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

The checkable parts of the research rule run as `PreToolUse` hooks. **They gate
MUTATIONS only** — reading is never blocked, so every denial is resolved by
reading something. `research-gate.sh` refuses a change to a file when:

1. **its subsystem rules never loaded** — `arch-*.md` load on the **Read tool
   only**; `cat`/`sed`/`grep` through Bash do not trigger them (measured
   2026-09-16, gotcha #617). Read the file, or its rules file, first.
2. **the file exists and this session never looked at it** — read-before-edit
   for the Bash path. A subagent's own reads count (#688).
3. **the task consulted nothing this repo already knows** — per task
   (`prompt_id`): `gotchas.md`, `closed_findings.md`, `docs/invariants/`,
   `FUTURE_WORK.md`, the sweep log, or a web search satisfies it.

⚠ It reads any path NAMED in a mutating Bash command as a target, heredoc
bodies included. For a multi-file edit script, write it to the scratchpad and
run it by path.

`commit-gate.sh` runs `cargo dev-test --test repo_consistency` (~2 s once built)
before a `git commit` touching a file whose figures another document restates
(`CLAUDE.md`, `README.md`, `docs/ARCHITECTURE.md`, `Cargo.toml`,
`frontend/i18n/*.json`, the book's installation page) — it tests THIS content,
which a stamp file could not.

**Both fail OPEN** (unparseable payload, missing cargo, timeout), and **a hook
is verified by its OUTPUT, never its exit code** (#614) — three hooks here sat
inert for months looking healthy. After touching the gate's patterns run
`python3 examples/research_gate_probe.py`: one planted violation per Bash
mutation form plus a null control.

There is **no compile-after-edit hook** (removed 2026-10-02): it ran a blocking
`cargo check` on a second feature set after every `.rs` edit (~13 s each) and
duplicated the one `cargo lint` that matters. Run `cargo lint` when a change is
complete.

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
cargo fmt && cargo lint && git add -A && git commit -m "..." && git push origin main
```

`cargo fmt` applies formatting (not `--check`); `cargo lint` is clippy on the one
local feature set with `-D warnings`. The commit gate and pre-push hook reuse
that build, so nothing compiles twice. Give `git push` up to 20 minutes after a
version bump — the hook rebuilds the test binary (#698).

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

**Nothing may block compaction, so commit as you go — it is the only
protection** (#687). A PreCompact hook that refused to compact while work was
uncommitted killed a whole session on 2026-09-23: the context filled, `/compact`
was refused, and the commit it demanded no longer fit. Compaction never touches
the disk; `post-compact-status.sh` (SessionStart, matcher `compact`) lists what
is still uncommitted afterwards. CLAUDE.md § "Compact instructions" says what a
summary must keep.

Before ~70% context, update `memory/MEMORY.md` with anything worth carrying
forward. After a compaction, first read `MEMORY.md`, `git log --oneline -10` and
`git diff HEAD~3 --stat`; don't re-do committed work.
