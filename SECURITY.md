# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in SwarmLLM, please report it responsibly.

**Email:** security@enapt.dev

**Do not** open a public GitHub issue for security vulnerabilities.

## What to Include

- Description of the vulnerability
- Steps to reproduce
- Affected versions/components
- Potential impact

## Response Timeline

- **48 hours** — acknowledgment of your report
- **7 days** — initial assessment and severity classification
- **90 days** — coordinated disclosure timeline (we ask that you do not publish details before this period or before a fix is released, whichever comes first)

## Scope

SwarmLLM's security model includes:

- **Node identity** — Ed25519 keypairs for authentication and transaction signing
- **E2E encryption** — X25519 ECDH + ChaCha20-Poly1305 for peer communication
- **Shard integrity** — BLAKE3 content hashing on every load
- **API authentication** — Bearer token with constant-time comparison
- **Credit system** — dual-signed transactions to prevent forgery

Issues in any of these areas, as well as path traversal, injection, authentication bypass, or denial of service, are in scope.

## Recognition

We credit security researchers in release notes (unless you prefer to remain anonymous).

## Known accepted advisories

These advisories show up in `cargo audit` and are accepted for the
following reasons. Re-evaluate when the upstream ecosystem moves.

| ID | Crate | Reason accepted |
|---|---|---|
| RUSTSEC-2024-0436 | `paste` (unmaintained) | Transitive via `tokenizers → candle`. Compile-time macro only, not in the runtime trust boundary. |
| RUSTSEC-2026-0097 | `rand 0.8.x / 0.9.x / 0.10.x` (unsound with custom logger) | Triggered only when a consumer installs a custom `rand` logger. We do not. Cryptographic randomness uses `OsRng`, not `thread_rng()`. |
| RUSTSEC-2026-0118 | `hickory-proto 0.25.2` (NSEC3 unbounded loop) | Transitive via `libp2p-mdns` and `libp2p-dns`. mDNS path is link-local without DNSSEC; DNS resolver only resolves bootstrap multiaddrs at startup. No upstream fix yet (waiting on `libp2p` to bump `hickory ≥ 0.26`). Re-evaluate on next libp2p release. |
| RUSTSEC-2026-0119 | `hickory-proto 0.25.2` (O(n²) name-compression CPU exhaustion) | Same dep paths as 0118. Attacker would need to inject DNS responses into the daemon's resolver — link-local mDNS or bootstrap-time only. Fix requires `hickory ≥ 0.26.1`, not yet adopted by libp2p 0.56. |
| RUSTSEC-2026-0221 | `event-listener 5.4.1` (unsound: `!Send` tags can cross thread boundaries via `StackSlot`) | Transitive via `libp2p-gossipsub → async-channel → event-listener-strategy`. Unsoundness is reachable only through the tagged-listener API, which neither we nor `async-channel` use — `async-channel` uses the untagged listener. Nothing in this tree can construct a `!Send` tag. Fix requires `libp2p` to carry a newer `async-channel`. |

The auto-update integrity-binding finding **C1** (audit_2026-04-29) —
SHA256 sidecar fetched from the same GitHub release as the binary —
is tracked in `docs/ARCHITECTURE.md` § Deferred Items and remains open
until an offline signing keypair is in place.
