# Where the documentation is

One page, so nothing needs hunting for. Each document is listed once, under the
reader it is written for. **Current** documents describe the code as it is now;
**history** is kept only to explain why, and is never the place to look up how
something works today.

## Using SwarmLLM

| Document | What it answers |
|---|---|
| [The book](https://enapt.github.io/SwarmLLM/) (source: [`book/src/`](book/src/SUMMARY.md)) | Installing, the first model, joining the network, every setting, the APIs, running a node, troubleshooting |
| [TESTING.md](TESTING.md) | Something broke: what to send, where the log file is, how to capture a problem you can reproduce |
| [NETWORKING.md](NETWORKING.md) | Making a computer reachable from outside, and running an anchor |
| [RELEASE_SIGNING.md](RELEASE_SIGNING.md) | How releases are signed, and how to check a download yourself |
| [../SECURITY.md](../SECURITY.md) | Reporting a security problem, and the advisories knowingly accepted |

## Working on the code

| Document | What it answers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | **Start here**: the subsystems, the source tree, the protocols, the security model, § "HTTP API Routes" and § "Deferred Items" |
| [invariants/](invariants/) | The evidence behind each rule (`.claude/rules/arch-*.md`): what it replaced, what it was measured at, what a change must keep. Seven topics: `api-surfaces`, `frontend`, `inference`, `memory`, `network`, `scheduling`, `state-and-config` |
| [DIAGNOSTICS.md](DIAGNOSTICS.md) | Every `DIAG:` log line and what it means; benchmark traps |
| [FUTURE_WORK.md](FUTURE_WORK.md) | Open work, ranked (§ "PRIORITIES"). A fix moves its entry to § "Closed" in the same commit |
| [REFERENCE_MODELS.md](REFERENCE_MODELS.md) | The exact model files to test with (three tiers), so results from different computers compare |
| [CREDITS_DESIGN.md](CREDITS_DESIGN.md) | The credit system — dormant: it gates nothing. Read before touching credits |
| [plans/](plans/) | One design note per piece of work in progress (split speed, regional pipelines, decode submissions, …) |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Building each release variant from source, running the tests, the checks a PR must pass |

Agents working in this repo also read `../CLAUDE.md` and `../.claude/rules/` — the
rules load by path, so they arrive with the files they govern.

## History — read to learn why, never how it works now

| Document | What it is |
|---|---|
| [FUTURE_WORK_ARCHIVE.md](FUTURE_WORK_ARCHIVE.md) | Every closed entry, in full, and the re-ranking log |
| [NETWORKING_PLAN.md](NETWORKING_PLAN.md) | The July 2026 plan for reaching computers behind NAT — implemented; the code cites its sections |
| [plans/archive/](plans/archive/) | Plans that finished or were superseded (April 2026 speed-up plan, its next steps, the Responses API plans) |
| [plans/benchmarks/](plans/benchmarks/) | Benchmark rounds 3-8, April-May 2026 |
| [bench_results/](bench_results/README.md) | Responses API measurements and scripts, 2026-04-25 |
| [../CHANGELOG.md](../CHANGELOG.md) | What each release changed |
