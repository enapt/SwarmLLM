# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in SwarmLLM, please report it responsibly.

Report it privately through GitHub: open this repository's **Security** tab and
choose **Report a vulnerability**
(https://github.com/enapt/SwarmLLM/security/advisories/new). Only the
maintainer can see the report.

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
- **Encryption in transit** — every peer connection is encrypted (Noise over
  TCP, TLS over QUIC). Model activations forwarded to the next stage of a split
  are additionally sealed with X25519 + ChaCha20-Poly1305; results returning to
  the requester, and some fast paths, rely on the connection encryption alone.
  This is not end-to-end:
  a peer computing part of a request sees the data it computes on
  (`docs/ARCHITECTURE.md` § Pipeline Privacy Model).
- **Shard integrity** — BLAKE3 content hashing on every load
- **API authentication** — Bearer token with constant-time comparison
- **Update authenticity** — auto-updates install only if the release's checksum
  file carries a valid minisign signature from the offline release key
  (`docs/RELEASE_SIGNING.md`)
- **Credit system** — dual-signed transactions (dormant: credits currently gate
  nothing)

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

The auto-update integrity finding **C1** (audit_2026-04-29) — a SHA256
sidecar fetched from the same GitHub release as the binary — was closed on
2026-09-19 by release signing (`docs/RELEASE_SIGNING.md`). One exception: the
anchor updater in `deploy/anchor/` still checks only the SHA256 file. Signing
cannot protect against a build pipeline that was already compromised when it
produced a binary; that needs reproducible builds and remains open.
