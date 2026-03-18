# Contributing to SwarmLLM

Thanks for your interest in SwarmLLM. This document covers the basics for getting a contribution merged.

## Building from Source

Requires Rust 1.80+.

```bash
git clone https://github.com/enapt/SwarmLLM.git
cd SwarmLLM

# CPU-only
cargo build --release

# With CUDA GPU acceleration
cargo build --release --features candle-cuda
```

## Running Tests

```bash
# Unit and module tests
cargo test

# Integration tests (must run single-threaded)
cargo test --test integration -- --test-threads=1
```

## Code Quality

Every PR must pass these checks locally before submission:

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
```

Zero clippy warnings. No exceptions.

## Submitting a PR

1. Fork the repo and create a branch off `main`.
2. Make your changes. Keep commits focused — one logical change per commit.
3. Write commit messages in imperative mood ("Add retry logic for shard downloads", not "Added retry logic").
4. Run `cargo fmt`, `cargo clippy`, and `cargo test` before pushing.
5. Open a PR against `main`.

## What We Look For

- **Does it compile and pass CI?** This is the minimum bar.
- **Is it tested?** New functionality should have unit tests. Bug fixes should have a regression test where practical.
- **Is it focused?** PRs that do one thing well get reviewed faster than sprawling changes.
- **Does it match existing patterns?** Follow the conventions in `CLAUDE.md` — error handling with `thiserror`, `DashMap` for shared state, `mpsc` channels between subsystems, structured `tracing` logging.
- **No unnecessary dependencies.** The binary is ~31MB. We want to keep it lean.

## Reporting Bugs

Open a [GitHub Issue](https://github.com/enapt/SwarmLLM/issues) using the bug report template. Include:

- Steps to reproduce
- Expected vs actual behavior
- OS, Rust version, GPU (if relevant)
- Log output with `-vv` flag if applicable

## Requesting Features

Open a [GitHub Issue](https://github.com/enapt/SwarmLLM/issues) using the feature request template. Describe the use case, not just the solution.

## Docker

```bash
# CPU image
docker build -t swarmllm .
docker run -p 8800:8800 -v swarmllm-data:/data swarmllm

# CUDA GPU image
docker build -f Dockerfile.cuda -t swarmllm:cuda .
docker run --gpus all -p 8800:8800 -v swarmllm-data:/data swarmllm:cuda

# 3-node test cluster
docker compose up
```

## Project Structure

The codebase is a Cargo workspace with three crates:

- **`swarmllm`** (root) — main binary and all subsystem logic
- **`crates/swarmllm-types/`** — shared data types (69 types: NodeId, ModelManifest, SwarmMessage, etc.)
- **`crates/swarmllm-frontend/`** — embedded or dev-mode frontend asset serving

Key directories:
- `src/daemon/` — startup, shared state, message dispatch
- `src/network/` — libp2p networking, peer discovery, transport
- `src/inference/` — router, pipeline, executor, split inference
- `src/api/` — HTTP server, OpenAI/Anthropic endpoints, admin dashboard
- `src/credit/` — credit system, transactions, anti-gaming
- `frontend/` — vanilla HTML/CSS/JS dashboard (no build step): `js/core/` (state, utils, data), `js/components/` (8 UI modules), `js/init.js`, 12 HTML `<template>` elements

## Security Issues

Do **not** open a public issue for security vulnerabilities. See [SECURITY.md](SECURITY.md) for responsible disclosure instructions.

## License

By contributing, you agree that your contributions will be dual-licensed under MIT and Apache 2.0, consistent with the project license.
