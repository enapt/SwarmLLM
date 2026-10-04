---
paths:
  - "src/**"
---

# Errors: one classifier, one variant per meaning

Invariants this codebase has paid to learn. Each heading is the rule; the text
under it is what to do. The evidence — the incidents behind each line — is in
`docs/invariants/api-surfaces.md` § "Error type discipline".

**This file loads when you read any file under `src/`** (moved out of the
always-loaded `completeness.md` on 2026-10-02).

## Never choose an error type at a call site

`crate::error::classify_error` returns `(StatusCode, client-safe message,
error_type)` and is the single answer for every surface — HTTP, both SSE
encoders, the Responses API, MCP. A literal next to `"type":` is the bug
(gotchas #300-#305); `a_streamed_error_names_the_same_failure_as_its_non_streaming_sibling`
fails the build on a new one.

A surface may **refine** — Responses' `upstream_error`, MCP's
`RESOURCE_UNAVAILABLE`, Anthropic's own vocabulary — only when it names
something more precisely, commented at its definition. The same meaning under a
different word is a divergence, and a bug.

**A type that crossed a boundary is already lost.** The worker IPC hop and the
network hop both deliver a `String`, re-wrapped as `Inference` → 500.
`reclassify_flattened_error` is the ONLY sanctioned place to derive a class from
a message; it matches `SwarmError`'s own `#[error(...)]` prefixes, which are
part of the type, not prose (#295).

## The variant → status contract

- `Validation` → 400, API input errors. Never `Config` or `Internal` for request validation.
- `ModelNotAvailable` / `ShardNotFound` → 404.
- `Config` → startup / config file only.
- `ServiceUnavailable` → 503, *this server* can't serve: missing local binary, subprocess spawn/I/O failure, broken pipe, init timeout — every subprocess lifecycle failure.
- `LocalMemoryUnavailable` → 503, same message shape and `error_type` as the line above. It exists for ONE decision: the only local failure the router re-plans without a remote segment, because the re-plan is handed a new fact (`note_local_memory_refusal`). Never widen `ServiceUnavailable`'s retry to match it; never emit it where a re-plan cannot help.
- `ProviderError {status, body}` → upstream returned an error OR its reply could not be parsed (`api/openai/responses/translate.rs`); preserves the upstream status. Not `Internal`, not `Validation`.
- `Internal` → 500, actual bugs, only when no external party can be blamed (our own well-typed struct failing to serialize; NOT a subprocess crash).
- `PeerUnresponsive` → 503, a peer took the request and went silent (ACK sweep, first-token deadline, `pipeline/local.rs::segment_timeout_error`). Penalty-ELIGIBLE, retried by TYPE (`router::peer_went_silent`, only after a remote segment was involved). ⚠ Every producer MUST call `blacklist_holder_for_request` on the silent peer FIRST, or the re-plan re-picks it and waits the deadline twice.
- `MixedModelCopy` → 503, THIS node's header and tensor table describe two uploads; refused at load, never quarantined (the parts may be fine). Local-only for penalties: a peer's crosses the wire as missing shards — `sanitize_peer_facing_error` (layer forward), `worker_failure_for_the_coordinator` (whole-model hand-off) — which retracts that peer. A new path returning a worker's error to another node translates it too.
- `SegmentFailoverExhausted` → 503, mid-pipeline holder failure with no standby. Not `ModelIncompleteInSwarm` (`assembly_failed_for_lack_of_holders` → a pointless DHT wait), not `ServiceUnavailable` (the peer-blacklist retry); penalty-exempt, it names no culprit.
- `SwarmShortOfMemory` / `HoldersDeclined` → 503, written ONLY by `router::report_after_a_replan` after a holder refused and the re-plan found nothing (#218): no room among the holders (retrying cannot help) vs the holders said no (retry later). Never `ModelIncompleteInSwarm` ("has gone") for a holder that is still there; penalty-exempt.

When unsure between `Internal`, `ProviderError` and `ServiceUnavailable`, follow
the pattern the surrounding function already uses.

## A policy refusal is 503, and must never be told to retry

A request this node declines *by configuration* (private mode, prompt privacy) is
`ServiceUnavailable`-shaped (503), never `Internal` (500), with its own variant
and `error_type` — `PrivateModeUnavailable`, `PromptPrivacyUnavailable`.

- **A permanent failure gets its own variant**, never `PipelineError(String)`,
  whose hint is chosen by substring-matching prose and defaults to "a peer went
  offline, try again" — advice three failures got for conditions a retry can
  never fix (#295).
- **One variant carrying two causes that need opposite advice is two
  variants** — `Unauthorized` vs `LocalOnly` → 403 `permission_error` (#309).
  Ask: would both situations get the same next step?

A new `SwarmError` variant must also:

- join the local-only list of **`failure_is_penalty_worthy`**
  (`router/distributed_exec.rs`, default `_ => true` = blame the peer) when it
  describes OUR config or a local fault, or it penalises innocent peers;
- get an **`error_hint`** that says so when retrying cannot help. Test the
  ADVICE, not the wording.

→ `docs/invariants/api-surfaces.md` § "Error type discipline"
