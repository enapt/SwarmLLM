# Next Steps (post-Round-6)

> Status snapshot taken 2026-04-20 after Item 8 end-to-end validation on
> RTX 3070 + TinyLlama-1.1B. Last landed commit before this plan:
> `badde4a Item 8 Phase 4: two-daemon bench recipe + probe-resolver sanity tests`.

## What's done

- All 20 build phases.
- Distributed-inference speedup arc Items 1–7, 8 (all phases), 12, 13, 16.
- Round 6 bench run: cross-node prefix-KV pipeline validated end-to-end;
  two wire bugs caught + fixed in-tree (see `round6.md` Results for detail):
  1. `PrefixCacheAnnounce` missing from `handle_broadcast` topic match.
  2. IPC JSON `Vec<u8>` bloat on `PrefixSnapshotResponse` +
     `PrefixFetchResult` → moved bytes to the binary-payload slot.
- Docs brought current: `round6.md`, `distributed_inference_speedup.md`
  top-of-doc, `docs/ARCHITECTURE.md § Prefix-Cache KV Sharing`,
  `memory/MEMORY.md` gotchas #23 + #24.

## What's left — ordered by signal-to-effort

### 1. Item 8 cross-over demo on a larger model *(2–3 h hands-on)*

**Why first:** Round 6 validated the *architecture* but couldn't show a
measurable localhost win because TinyLlama-1.1B prefill is faster than a
28 MB KV wire transfer on this GPU. A Qwen2.5-7B Q4 run (already staged
at `~/.local/share/swarmllm/models/qwen2.5-coder-7b-instruct-q4-k-m/`,
28 layers, 8 shards) should flip the sign: prefill grows as
`hidden_dim² × tokens`, wire as `hidden_dim × tokens`.

VRAM budget: Qwen 7B Q4 ≈ 4.7 GB; two copies + KV snapshots + CUDA
workspace is likely **over** the 8 GB ceiling for two colocated daemons.
Run A on GPU and B pinned to CPU (`gpu_layers = 0` in B's config) —
that also makes the prefill-vs-fetch ratio more favorable for the fetch
path because CPU prefill is much slower.

Recipe: copy the round6.md script, swap in the Qwen model ID and CPU
config for B, re-run. Update `round6.md` § Results with a "Qwen-7B,
GPU A / CPU B" row.

### 2. Optional: zstd compression on `WIRE_TAG_PREFIX_KV` *(1 d)*

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

### 3. WAN bench *(real hardware, 1 d)*

**Why:** Localhost is the pathological case for KV-fetch — RTT is ~1 ms,
so even a perfect fetch path only buys you back the prefill cost minus
the narrow + serialize overhead. On WAN (50–150 ms RTT) the picture
inverts: a single 150 ms RTT still beats multiple seconds of remote
prefill on any non-trivial prompt.

Needs two machines on different networks (or a VPN between two
cloud regions). Probably pair with the Qwen-7B run so both a
larger-model *and* a WAN measurement land together.

### 4. Item 14 / 17 / 18 research candidates

Items 14 (sequence-level speculative decoding), 17 (overlapped comm +
compute), and 18 (gradient-free adaptive batch sizing) are still in the
"research, then decide if worth building" bucket per
`docs/plans/distributed_inference_speedup.md`. Don't pull these in
until (a) Qwen + WAN numbers are in, and (b) we've asked whether the
next bottleneck is compute or wire. Do the bench first, pick the
lever second.

### 5. Release hygiene *(2–4 h mechanical)*

- Tag `v0.1.0`. All 20 phases complete + 775 tests passing + Item 8
  validated is a reasonable version cut.
- macOS CI matrix: currently Linux-only. macOS build is believed to
  work (uses Metal via candle's default) but untested in CI.
- Benchmarks for already-shipped Items still under flags that we
  haven't measured: multi-segment DSD (Item 12), Q8_0 activation
  compression (Item 13).

## Anti-goals

- **Don't defer Item 8 win demonstration as "too hard."** It'll sit on
  the shelf and stale. A same-session Qwen-7B run is the right next
  move even though it requires CPU-binding B, reloading models, and
  another ~10 minutes of daemon restart choreography.
- **Don't add the zstd path before the Qwen bench.** If Qwen shows the
  win is already ≥300 ms, compression is optional polish. If it shows
  the win is ≤100 ms, compression might not be enough to matter and we
  should question the premise.
- **Don't batch release hygiene behind research items.** v0.1.0 should
  cut whenever the tree is green and stable, not whenever the research
  backlog is clear.
