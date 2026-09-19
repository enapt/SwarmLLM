# Release signing

Every release asset the auto-updater can download carries a detached
[minisign](https://jedisct1.github.io/minisign/) signature, made by a key that
exists only on the maintainer's machine. A node refuses to install an update it
cannot verify against the public key compiled into it.

This closes audit item **C1** (`audit_2026-04-29`), which is why the procedure
below is not optional and why the private key must not move.

## What is signed

Not the binaries — their **`.sha256` sidecars**.

The updater already verifies a downloaded binary against its sidecar, so
signing the sidecar authenticates the binary in two hops: the signature proves
the hash came from the key holder, and the download proves the bytes match the
hash. Debian's `Release.gpg` over a checksum list is the same construction.

The reason to prefer it here is practical, and it is the reason the key can stay
offline at all: signing binaries would mean moving ~1.5 GB per release (the CUDA
asset alone is ~1 GB) to wherever the key lives. That cost is exactly what
pushes projects into putting a signing key in CI, where a compromised workflow
can use it. Signing sidecars moves about 700 bytes.

The list of assets that must be signed is `release_signed_assets.txt`, read by
both the signer and CI. `an_asset_the_updater_can_ask_for_is_an_asset_we_sign`
fails the build if the updater learns to request something not on it.

## What it does and does not defend against

**Does:** an attacker who can replace release assets — a stolen account, a
leaked token with release scope — cannot produce a signature, so a swapped
binary is refused by every node. That is precisely C1's threat model.

**Does not:** a build pipeline that was already compromised when it produced the
binary. Signing a hash cannot tell you the hash is of something good. Defending
that needs reproducible builds, and it is a separate problem — do not let the
presence of signatures suggest otherwise.

## One-time setup

Generate the keypair on the machine that will cut releases. Do this once.

```bash
cargo install rsign2                      # only needed for this one step
rsign generate -p release_pubkey.pub -s ~/.swarmllm-release-secret.key
```

`rsign` is used here and nowhere else. Signing a release goes through
`examples/sign_release.rs`, built from this repo, because it unlocks the key
ONCE for the whole release — `rsign` reads its password straight from
`/dev/tty` per invocation, which meant one prompt per asset.

It will ask for a password. Use one — the file is the whole security boundary.

Then put the public half into the repo:

```bash
grep -v '^untrusted comment:' release_pubkey.pub > release_pubkey.txt
rm release_pubkey.pub
git add release_pubkey.txt && git commit -m "chore: add the release signing key"
```

`release_pubkey.txt` is compiled into every binary via `include_str!`. **A build
whose copy is empty cannot verify anything and will refuse every update** —
deliberately, because a node that cannot check authenticity must not replace its
own binary.

### Key custody

The secret key stays on the maintainer's machine and nowhere else. In particular
**never a GitHub Actions secret**: a workflow that can read the key is a
workflow an attacker who compromises CI can sign with, which removes most of
what this buys. GitHub's own guidance on build-system security makes the same
point — build jobs must not be able to reach signing material.

Back it up somewhere offline. Losing it does not endanger anyone, but it does
mean a key rotation, and rotation is slow (see below).

## Cutting a signed release

CI builds the assets and leaves the release a **draft**. It cannot publish,
because it cannot sign. One command finishes the job:

```bash
examples/sign_release.sh v0.3.191-alpha
```

That downloads every sidecar and checks the set is complete **before**
asking for anything, then prompts for the password **once** and signs them
all. Each signature is verified against the *public* key in the repo before
it can be uploaded, and the signer refuses outright if the secret key is not
the one `release_pubkey.txt` publishes — signing with the wrong key produces
a release that looks perfect and that every node rejects. It refuses to
sign a release with a missing sidecar, because publishing a partially signed
release would silently freeze every node on the missing platform.

Add `--verify-artifacts` to also download each binary and re-check its hash
before signing. Off by default because it moves ~1.5 GB to re-establish
something the updater checks anyway; worth it when an asset has been re-uploaded
by hand.

## Verifying a release independently

Anyone can check a download without running our code:

```bash
rsign verify swarmllm-linux-x86_64.sha256 \
  -x swarmllm-linux-x86_64.sha256.minisig \
  -P "$(grep -v '^#' release_pubkey.txt | tail -1)"
sha256sum -c swarmllm-linux-x86_64.sha256
```

The trusted comment names the asset and version the signature was minted for.
That is not decoration: `update_signature.rs` checks it, so a genuine signature
for one asset cannot be replayed as another's.

## Rotating the key

Replace `release_pubkey.txt` and ship a release **signed with the old key**. A
node only trusts the key compiled into it, so the new key reaches the field
through an update the old key vouches for. Skipping that step strands every node
on its current version with no way back other than a manual reinstall.

So: rotate over two releases. Release N is signed with the old key and carries
the new public key. Release N+1 is signed with the new one.
