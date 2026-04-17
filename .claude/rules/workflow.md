# Workflow Rules

## Commit and Push After Each Task

This project requires `git push` after every logical unit of work — don't batch to end of session. Long sessions and compactions can lose uncommitted work.

Sequence (always run together, in this order):
```
cargo fmt && cargo clippy --all-targets -- -D warnings && git add -A && git commit -m "..." && git push origin main
```

`cargo fmt` actually applies formatting (not `--check`). `cargo clippy` must be zero warnings — fix before pushing.

## Memory Management Around Compaction

The PreCompact hook blocks compaction if anything is uncommitted or `cargo check` fails. Before ~70% context usage, proactively update `memory/MEMORY.md` with anything worth carrying forward.

After compaction, before doing anything else: read `memory/MEMORY.md`, then `git log --oneline -10` and `git diff HEAD~3 --stat`. Don't re-do committed work.
