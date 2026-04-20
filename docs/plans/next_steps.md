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

### 1. zstd compression on `WIRE_TAG_PREFIX_KV` *(1 d)*

**Why next:** Prefix KV blocks are f32 with wide zero-ish regions (the
seq dim beyond `token_count` is zero-padded, and attention patterns
often cluster). A rough estimate from existing `WIRE_TAG_TENSOR_COMPRESSED`
behavior is 30–50% wire reduction on localhost with ~5–15 ms
compress/decompress overhead. On WAN this is a clear win; on localhost
it probably roughly neutralizes.

Plumbing already exists for the tensor path — reuse the same
zstd-level + magic-byte framing convention rather than inventing a new
one. Gate behind an `InferenceConfig` boolean defaulted to off until
the larger-model bench confirms the benefit on WAN.

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

Items 14 (sequence-level speculative decoding), 17 (overlapped comm +
compute), and 18 (gradient-free adaptive batch sizing) are still in the
"research, then decide if worth building" bucket per
`docs/plans/distributed_inference_speedup.md`. Don't pull these in
until (a) Qwen + WAN numbers are in, and (b) we've asked whether the
next bottleneck is compute or wire. Do the bench first, pick the
lever second.

### 4. Release hygiene *(2–4 h mechanical)*

- Tag `v0.1.0`. All 20 phases complete + 775 tests passing + Item 8
  validated (including cross-over demo) is a reasonable version cut.
- macOS CI matrix: currently Linux-only. macOS build is believed to
  work (uses Metal via candle's default) but untested in CI.
- Benchmarks for already-shipped Items still under flags that we
  haven't measured: multi-segment DSD (Item 12), Q8_0 activation
  compression (Item 13).

## Anti-goals

- **Don't add the zstd path yet.** The Qwen-7B cross-over showed a
  140 s absolute win without compression, so zstd is optional polish
  on the localhost case. It may still matter for WAN (where a smaller
  wire shrinks RTT-dominated windows), but land the WAN measurement
  first to know which direction to optimize.
- **Don't batch release hygiene behind research items.** v0.1.0 should
  cut whenever the tree is green and stable, not whenever the research
  backlog is clear.
