#!/usr/bin/env bash
#
# Sign a release and publish it — the one step of the release gate that needs
# the offline key, and the reason a release is not published by CI.
#
#   examples/sign_release.sh v0.3.191-alpha
#
# WHAT THIS SIGNS, AND WHY IT IS NOT THE BINARIES
#
# The auto-updater verifies a downloaded binary against its `.sha256` sidecar.
# Signing that sidecar therefore authenticates the binary in two hops — the
# signature proves the hash came from the key holder, and the updater proves
# the bytes match the hash — while moving ~100 bytes per asset instead of
# ~1.5 GB. That is what lets the private key stay off GitHub without the
# release gate becoming a half-hour chore. Debian signs a checksum list for
# the same reason.
#
# WHAT IT DEFENDS AGAINST (audit item C1)
#
# Until this existed, the sidecar came from the same GitHub release as the
# binary, so whoever could replace one could replace the other and our check
# passed. A signature made by a key that never exists in CI defeats that.
#
# It does NOT defend against a build pipeline that was already compromised when
# it produced the binary — signing a hash cannot tell you the hash is of
# something good. That is a build-integrity problem (reproducible builds), and
# it is out of C1's scope. Do not let this script's existence suggest otherwise.
#
# KEY CUSTODY
#
# The secret key lives on the maintainer's machine and nowhere else. Never a
# GitHub Actions secret: a workflow that can read it is a workflow that can
# sign, which is most of what this is trying to prevent.
#
set -euo pipefail

TAG="${1:-}"
if [ -z "$TAG" ]; then
  echo "usage: $0 <tag>            e.g. $0 v0.3.191-alpha" >&2
  echo "  env SWARMLLM_RELEASE_SECRET_KEY   path to the offline secret key" >&2
  echo "      --verify-artifacts            also download each binary and" >&2
  echo "                                    re-check its hash before signing" >&2
  exit 2
fi
shift || true

VERIFY_ARTIFACTS=0
for arg in "$@"; do
  case "$arg" in
    --verify-artifacts) VERIFY_ARTIFACTS=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

REPO="${SWARMLLM_RELEASE_REPO:-enapt/SwarmLLM}"
SECRET_KEY="${SWARMLLM_RELEASE_SECRET_KEY:-$HOME/.swarmllm-release-secret.key}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSET_LIST="$ROOT/release_signed_assets.txt"
PUBKEY_FILE="$ROOT/release_pubkey.txt"

# The version the trusted comment records. `update.rs` compares it with a
# leading `v` stripped from either side, but write the same spelling the crate
# uses so a human reading a signature sees what they expect.
VERSION="${TAG#v}"

# The signer is `examples/sign_release.rs`, built from this repo, because it
# unlocks the key ONCE for the whole release. The `rsign` CLI cannot: it reads
# the password straight from /dev/tty per invocation, so seven assets meant
# seven prompts. `rsign` is still the tool anyone else uses to VERIFY a release.
command -v cargo >/dev/null 2>&1 || { echo "cargo not found (needed to build the signer)" >&2; exit 1; }
command -v gh >/dev/null 2>&1 || { echo "gh CLI not found" >&2; exit 1; }
[ -f "$SECRET_KEY" ] || {
  echo "No secret key at $SECRET_KEY" >&2
  echo "Generate one ONCE, and keep it off this repository and off GitHub:" >&2
  echo "    rsign generate -p release_pubkey.pub -s $SECRET_KEY" >&2
  echo "  (install that one-off generator with: cargo install rsign2)" >&2
  echo "then put the RW... line from release_pubkey.pub into release_pubkey.txt" >&2
  exit 1
}

# The public key compiled into the binaries being signed. Checking against it
# at the end is what catches the mistake this script cannot otherwise see:
# signing with a key nothing in the field trusts, which produces a release that
# looks perfect here and is refused by every node.
PUBKEY=$(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$PUBKEY_FILE" | tail -1 | tr -d '[:space:]')
if [ -z "$PUBKEY" ]; then
  echo "release_pubkey.txt holds no key — the builds cannot verify anything." >&2
  echo "Set it before cutting a release." >&2
  exit 1
fi

mapfile -t ASSETS < <(grep -vE '^[[:space:]]*#|^[[:space:]]*$' "$ASSET_LIST")
[ "${#ASSETS[@]}" -gt 0 ] || { echo "no assets listed in $ASSET_LIST" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Signing $TAG (${#ASSETS[@]} assets) with $SECRET_KEY"
echo "Public key the field trusts: $PUBKEY"
echo

# Fetch every sidecar and check the set is complete BEFORE asking for the
# password. A release missing a platform is not signable, and finding that out
# after someone has typed their passphrase is a poor way to learn it.
MISSING=()
for asset in "${ASSETS[@]}"; do
  sidecar="${asset}.sha256"
  if ! gh release download "$TAG" --repo "$REPO" --pattern "$sidecar" --dir "$WORK" >/dev/null 2>&1; then
    MISSING+=("$sidecar")
    continue
  fi

  if [ "$VERIFY_ARTIFACTS" = "1" ]; then
    # Off by default because it moves ~1.5 GB to re-establish something the
    # updater checks anyway: if the sidecar and the binary disagree, every node
    # refuses the download. Worth running when an asset has been re-uploaded by
    # hand, which is exactly when that assumption is weakest.
    echo "  [$asset] downloading binary to re-check its hash…"
    gh release download "$TAG" --repo "$REPO" --pattern "$asset" --dir "$WORK" >/dev/null
    want=$(awk '{print $1}' "$WORK/$sidecar")
    got=$(sha256sum "$WORK/$asset" | awk '{print $1}')
    if [ "$want" != "$got" ]; then
      echo "REFUSING TO SIGN $asset — its sidecar does not describe it." >&2
      echo "  sidecar says: $want" >&2
      echo "  file hashes:  $got" >&2
      exit 1
    fi
    rm -f "$WORK/$asset"
  fi

  echo "  [$asset] checksum file fetched"
done

if [ "${#MISSING[@]}" -gt 0 ]; then
  echo >&2
  echo "Refusing to sign $TAG — these sidecars are not in the release:" >&2
  printf '  %s\n' "${MISSING[@]}" >&2
  echo >&2
  echo "A missing sidecar means that platform's build did not finish. Signing" >&2
  echo "the rest would publish a release that silently freezes those nodes at" >&2
  echo "their current version, because an unsigned asset is refused." >&2
  exit 1
fi

# One password prompt for the whole release. The signer refuses before writing
# anything if this key is not the one `release_pubkey.txt` publishes, and
# verifies each signature it produces against that key before it can be
# uploaded. Both the trusted-comment format and the prehashed mode are its
# contract with `src/update_signature.rs`.
echo
echo "Signing ${#ASSETS[@]} checksum files — one password for all of them."
# Pinned to the feature set the gate has already built (CLAUDE.md's standard
# build command), and to this repo's manifest so the script works from any
# directory. The signer needs none of the daemon, but an example links the lib,
# so matching the flags means this reuses the existing artifacts instead of
# compiling the tree a second way. `--no-default-features` alone does NOT
# compile: the frontend needs `embedded` or `dev`.
cargo run --quiet --manifest-path "$ROOT/Cargo.toml" \
  --no-default-features --features dev,claude-subscription \
  --example sign_release -- "$VERSION" "$WORK" "${ASSETS[@]}"

echo
echo "Uploading ${#ASSETS[@]} signatures…"
gh release upload "$TAG" --repo "$REPO" --clobber "$WORK"/*.minisig

echo
echo "Publishing $TAG"
gh release edit "$TAG" --repo "$REPO" --draft=false

echo
echo "Done. $TAG is signed and published."
echo "Anyone can verify an asset independently with:"
echo "    rsign verify <asset>.sha256 -x <asset>.sha256.minisig -P $PUBKEY"
echo "  (or the minisign CLI; both read this format)"
