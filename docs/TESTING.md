# Testing SwarmLLM (Alpha)

You're running an **alpha** build. Things will work; some things won't. This page tells you what we need from you when something breaks.

## Before you file anything

1. Confirm you're on the latest release: `./swarmllm version` (Windows: `swarmllm.exe version`) and compare against the [GitHub releases page](https://github.com/enapt/SwarmLLM/releases). By default SwarmLLM checks hourly and installs a new build by itself once it is idle; `./swarmllm update` does it straight away. If your install cannot update itself — a packaged `.deb` under a read-only unit, for instance — it says so and names the folder to replace by hand.
2. Copy a diagnostics report: in the app, **Settings → Testing & Diagnostics → Copy diagnostics**, or run `./swarmllm diagnostics`. It is safe to post publicly: no keys, no invite codes, and network addresses are replaced with placeholders.
3. Re-run the failing action with verbose logging: `./swarmllm run -vv 2>&1 | tee /tmp/swarmllm.log`. Verbose adds `DIAG:` instrumentation that traces every step of the request lifecycle. SwarmLLM writes its log only to its own window or terminal, not to a file, so capture it this way.
4. Search the [open issues](https://github.com/enapt/SwarmLLM/issues) — a one-line confirmation on an existing issue is more useful than a duplicate.

## Where to file

- **Bugs and crashes:** [GitHub Issues](https://github.com/enapt/SwarmLLM/issues/new).
- **Security issues:** see `SECURITY.md` — please don't open a public issue for these.
- **Quick questions:** ask on [Discord](https://discord.gg/nq9be3u828) or in [Discussions](https://github.com/enapt/SwarmLLM/discussions).

## What to include in a bug report

Copy-paste this template:

```
**What I did:** <one or two lines>

**What I expected:** <one line>

**What happened:** <one line>

**Platform:** <Linux/macOS/Windows> + <CPU/GPU model> + <RAM/VRAM>
**Version:** <output of ./swarmllm version>
**Network mode:** <Global/Pool/LAN/Offline — visible in the dashboard's Network Status panel>

**Logs:**
<paste the last ~50 lines of swarmllm.log, especially anything tagged DIAG: or ERROR>

**Reproduction:**
1. ...
2. ...
3. ...
```

Stripping IPs and tokens is fine — the daemon's `node_id` (visible in Settings → Identity & Access) is enough for us to correlate cross-tester reports.

## Areas we especially want testing on

- **Cold start.** Does your dashboard show models within ~30 seconds of first launch? If you see "No models available" with no actionable chips, that's a bug — please file it with how many computers you are connected to, the network mode, and a diagnostics report (Settings → Testing & Diagnostics → Copy diagnostics).
- **Distributed inference latency.** If a 7B+ model runs slow when more than 2 nodes hold its shards, capture: model id, hosted_shards / shard_count, peer count, region, prompt length, time-to-first-token, tokens/sec. The `DIAG:` log lines around `pipeline_forward` and `forward_through_segments` are what we'll need.
- **Pool invite codes.** Both `swarmpool://...` codes and legacy 8-character codes should work. Tell us which one you tried, what platform the inviter / joiner are on, and whether they're on the same LAN or across the internet.
- **The setup wizard.** Hardware autodetect, contribution slider, peer/cloud setup. If anything looks wrong for your hardware (wrong VRAM detected, weird recommendation), screenshot the wizard plus paste `./swarmllm status --json`.
- **Prompt privacy** (the "End-to-end encryption" toggle in the Models tab; `encrypted_pipeline` in config). It keeps the first and last parts of the model on your own computer, so other computers never receive your prompt or the reply as text. They do still receive the model's intermediate numbers for their share of the work, and those can be partly turned back into text — so it does not protect you from a computer that is doing the work. Tell us whether it switches on when you hold both end parts, how much slower replies get, and whether the status shown during a reply matches what you set.
- **Translations.** SwarmLLM ships 21 languages. If a translation reads wrong or English text leaks through, switch to that language in Settings, screenshot the broken screen, and file with the locale code (e.g. `i18n: de`).

## What's known broken

Tracked in the [open GitHub issues](https://github.com/enapt/SwarmLLM/issues). Before you file, please skim them — a one-line confirmation on an existing issue is more useful than a duplicate.

## Privacy notes

- SwarmLLM sends no telemetry to the project, and nothing crash-reports automatically — we rely on you to share logs. It does contact other services: GitHub (the hourly update check), Hugging Face (model downloads and popularity lists), and ip-api.com once at start-up to learn your country (skipped if you set `region` under `[identity]` in `config.toml`). Other computers in the swarm are told your computer's hardware summary, country, and which model parts it holds.
- Bug reports on GitHub are public. Strip API keys, server addresses, and personal-data prompts before pasting.
- Your peer-id is public anyway (gossip), so including it is fine and helps us correlate reports.

## Updating

Updates install themselves. SwarmLLM checks GitHub every hour, refuses anything
not signed with the SwarmLLM release key (releases are signed since
v0.3.191-alpha), and installs and restarts once it has finished any work in
progress. To change that, open **Settings → Software updates** (or set
`mode = "notify"` under `[updates]` in `config.toml`). `./swarmllm update` checks
and installs straight away.

To update by hand — for example a packaged install that cannot replace itself:
stop SwarmLLM, download the new file from the
[releases page](https://github.com/enapt/SwarmLLM/releases), replace the old one,
and start it again. To verify a download yourself, see
[RELEASE_SIGNING.md](RELEASE_SIGNING.md#verifying-a-release-independently) — the
bare binaries carry a `.sha256` and a signature; the `.zip` and `.tar.gz`
archives do not.

## Thank you

You're letting us catch the bugs that don't show up in CI. Genuinely appreciated.
