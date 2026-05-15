# Next Steps (post-Round-6)

> Status snapshot taken 2026-04-20 after Item 8 cross-over demo on
> RTX 3070 + Qwen2.5-Coder-7B. Last landed commit before this plan:
> `badde4a Item 8 Phase 4: two-daemon bench recipe + probe-resolver sanity tests`.

## What's done

- All 20 build phases.
- Distributed-inference speedup arc Items 1–7, 8 (all phases), 12, 13, 16.
- Round 6 bench run: cross-node prefix-KV pipeline validated end-to-end
  on both TinyLlama (fast-prefill corner case) and Qwen-7B
  (cross-over demonstrated: **12.9× iter-1 TTFT speedup**, 151.7 s →
  11.8 s on 640-token CPU prefill). Three wire bugs caught + fixed
  in-tree (see `round6.md` Results):
  1. `PrefixCacheAnnounce` missing from `handle_broadcast` topic match.
  2. IPC JSON `Vec<u8>` bloat on `PrefixSnapshotResponse` +
     `PrefixFetchResult` → moved bytes to the binary-payload slot.
  3. All three cross-node-fetch timeouts (500 ms worker probe,
     400 ms daemon network, 500 ms serving IPC) were TinyLlama-sized;
     bumped to 3000 / 2500 / 2000 ms to handle 7B-class snapshots.
- Docs brought current: `round6.md`, `distributed_inference_speedup.md`
  top-of-doc, `docs/ARCHITECTURE.md § Prefix-Cache KV Sharing`,
  `memory/MEMORY.md` gotchas #23 + #24.

## What's left — ordered by signal-to-effort

### 1. zstd compression on `WIRE_TAG_PREFIX_KV` ✅ LANDED 2026-04-25 (flag-gated)

Implemented as `NetworkConfig::prefix_kv_compression: bool` (default off).
Reuses the existing `compression::compress_tensor` / `decompress_tensor`
helpers and the same level + threshold knobs as tensor compression.

Wire format: `WIRE_TAG_PREFIX_KV` frame's flag byte gained a third value
(flag=2 = zstd-compressed payload). flag=0 (miss) and flag=1 (raw) are
unchanged. Receivers always decompress regardless of the flag, so flipping
it on a single peer doesn't require a coordinated upgrade.

Send-side falls back to flag=1 when the compressed form isn't smaller than
the raw form, so we never make the wire larger by accident.

Tests: 5 new round-trip tests in `network::protocol::tests::prefix_kv_*`
cover flag-on, flag-off, below-threshold, larger-when-compressed, and miss.
**Awaiting WAN bench (item 2 below) to decide default-on.**

### 2. WAN bench *(real hardware, 1 d)*

**Why:** Localhost is the pathological case for KV-fetch — RTT is ~1 ms,
so even a perfect fetch path only buys you back the prefill cost minus
the narrow + serialize overhead. On WAN (50–150 ms RTT) the picture
inverts: a single 150 ms RTT still beats multiple seconds of remote
prefill on any non-trivial prompt.

Needs two machines on different networks (or a VPN between two
cloud regions). The Qwen-7B CPU-CPU run already showed a clear
cross-over on localhost, so the WAN question is sharper: does the
win hold when RTT becomes a meaningful fraction of the fetch total?

### 3. Item 14 / 17 / 18 research candidates

Item 14 (Mirror Speculative Decoding, Apple), Item 17 (disaggregated
prefill/decode, Mooncake-style), and Item 18 (per-token early-exit
with adaptive depth, HELIOS / TIDE / DREX) are still in the
"research, then decide if worth building" bucket. Don't pull any in
until (a) Qwen + WAN numbers are in (§ 2 above) and (b) we've asked
whether the next bottleneck is compute, wire, or multi-segment tail
latency — the answer flips which item is highest-leverage.

Per-item assessment + sequencing recommendation written 2026-04-26 in
[`items_14_17_18_research.md`](./items_14_17_18_research.md).
Headline ranking by signal-to-effort for SwarmLLM today: **17 > 18 > 14**.
Item 14 inherits Item 6 (SWIFT) blocker — defer until SWIFT lands
measurable wins.

### 4. Release hygiene

- ✅ `v0.1.0` tag cut 2026-04-25 (816 tests). Subsequent commits are
  in CHANGELOG `[Unreleased] — post-v0.1.0`. Two black-hat audits
  (2026-04-28 + 2026-04-29) plus the autonomous R76→R91 sweep arc
  (2026-04-30 → 2026-05-01, 16 rounds) have landed since, plus the
  R92→R120 arc through 2026-05-12 plus R121 contribution-mode landing;
  current head sits at 936 lib + 75 integration tests, clippy clean,
  both default and `--features llama` feature sets compile.
- ✅ macOS CI matrix is live (clippy + test + build on `macos-15` in
  `.github/workflows/ci.yml` — clippy default features only, integration
  tests Linux-only by design until macOS multi-process IPC is exercised).
- ✅ Cargo metadata + workspace alignment (license, repository,
  homepage, MSRV) verified across all three workspace crates 2026-04-29.
- ✅ `cargo audit` known/accepted advisories documented in
  `SECURITY.md` (core2 yanked, paste unmaintained, rand custom-logger
  unsoundness — none of which we trip).
- ⏳ Item 12 multi-segment DSD bench still pending (needs 3+ daemons
  + draft model + WAN RTT — `docs/plans/benchmarks/round7.md` § Item
  12 for recipe). Q8_0 activation compression (Item 13) measured
  2026-04-20 in `round7.md`: 3.15× wire reduction, localhost decode
  neutral-to-slightly-negative as predicted; WAN bench will decide
  default-on.
- ✅ Worker crash-loop backoff, per-token cancellation observation
  (`/v1/responses/{id}/cancel` end-to-end), stop-sequence KV truncate
  to remote peers on session requests, escrow refund on inference
  failure, and the end-to-end daemon-lifecycle integration test
  (`tests/integration/end_to_end.rs`, `#[ignore]`'d) all landed
  2026-04-29.
- 🔓 **C1 binary signing** still open — three-option write-up in
  `memory/signing_options.md` (recommendation: **minisign**). Needs
  a key-custody decision (hardware token / encrypted file / GitHub
  Actions secret) from the maintainer before the next-tag cut.
  Tracked in `docs/ARCHITECTURE.md` § Deferred Items.

## Anti-goals

- **Don't batch release hygiene behind research items.** v0.1.0 should
  cut whenever the tree is green and stable, not whenever the research
  backlog is clear.
