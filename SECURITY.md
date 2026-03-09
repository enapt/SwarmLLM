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
