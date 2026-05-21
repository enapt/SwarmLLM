# Testing SwarmLLM (Alpha)

You're running an **alpha** build. Things will work; some things won't. This page tells you what we need from you when something breaks.

## Before you file anything

1. Confirm you're on the latest release: `./swarmllm version` and compare against the [GitHub releases page](https://github.com/enapt/SwarmLLM/releases). Auto-update is disabled in alpha — you must download new builds manually.
2. Re-run the failing action with verbose logging: `./swarmllm run -vv 2>&1 | tee /tmp/swarmllm.log`. Verbose adds `DIAG:` instrumentation that traces every step of the request lifecycle.
3. Search the [open issues](https://github.com/enapt/SwarmLLM/issues) — a one-line confirmation on an existing issue is more useful than a duplicate.

## Where to file

- **Bugs and crashes:** [GitHub Issues](https://github.com/enapt/SwarmLLM/issues/new).
- **Security issues:** see `SECURITY.md` — please don't open a public issue for these.
- **Quick questions / vibes:** issue with the `question` label is fine; we read them.

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

Stripping IPs and tokens is fine — the daemon's `node_id` (visible in Settings → Identity) is enough for us to correlate cross-tester reports.

## Areas we especially want testing on

- **Cold start.** Does your dashboard show models within ~30 seconds of first launch? If you see "No models available" with no actionable chips, that's a bug — please file it with peer count, network mode, and `~/.local/share/swarmllm/swarmllm.log`.
- **Distributed inference latency.** If a 7B+ model runs slow when more than 2 nodes hold its shards, capture: model id, hosted_shards / shard_count, peer count, region, prompt length, time-to-first-token, tokens/sec. The `DIAG:` log lines around `pipeline_forward` and `forward_through_segments` are what we'll need.
- **Pool invite codes.** Both v2 `swarmpool://...` blobs (R140) and legacy 8-character codes should work. Tell us which one you tried, what platform the inviter / joiner are on, and whether they're on the same LAN or across the internet.
- **The setup wizard.** Hardware autodetect, contribution slider, peer/cloud setup. If anything looks wrong for your hardware (wrong VRAM detected, weird recommendation), screenshot the wizard plus paste `./swarmllm status --json`.
- **Encryption modes.** `encrypted_pipeline = true` (per-model toggle in the Models tab) sends every activation through a per-request sealed channel. We want to know if it works, what the latency overhead feels like, and whether the "End-to-end encrypted" banner appears correctly during inference.
- **Translations.** SwarmLLM ships 21 languages. If a translation reads wrong or English text leaks through, switch to that language in Settings, screenshot the broken screen, and file with the locale code (e.g. `i18n: de`).

## What's known broken

Maintained in [`docs/KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) (if present) and the open GitHub issues. Before you file, please skim both.

## Privacy notes

- The daemon does not send telemetry. Nothing crash-reports automatically. We rely on you to share logs.
- Bug reports are public unless you mark them otherwise. Strip API keys, server addresses, and personal-data prompts before pasting.
- Your peer-id is public anyway (gossip), so including it is fine and helps us correlate reports.

## Updating

```bash
# Stop the daemon
killall swarmllm   # or systemctl stop swarmllm

# Download the new binary from the GitHub releases page
# Verify the SHA256 against the release notes
# Replace your existing binary
# Restart

./swarmllm run
```

In a future release we'll re-enable auto-update once binary signing is in place. Until then, please update by hand whenever a new release lands.

## Thank you

You're letting us catch the bugs that don't show up in CI. Genuinely appreciated.
