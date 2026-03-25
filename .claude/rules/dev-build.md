# Dev Build Rule

## Always use dev frontend for non-release builds

When building SwarmLLM for testing, development, or debugging:

```bash
cargo build --no-default-features --features dev
```

This serves frontend files from disk (`frontend/`) so CSS/JS/HTML changes are instant without recompiling.

**Only use `cargo build --release` for actual production releases.**

Never use `cargo build` (bare) or `cargo build --release` during development — the embedded frontend requires a full rebuild for every frontend change.
