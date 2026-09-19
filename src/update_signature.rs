//! Offline signature verification for release artifacts (audit item C1).
//!
//! # What this closes
//!
//! Until this landed, `update.rs` verified a downloaded binary against a
//! `.sha256` sidecar **fetched from the same GitHub release as the binary**.
//! That detects corruption and nothing else: whoever can replace the binary can
//! replace the sidecar beside it, so a compromised release-asset uploader — a
//! stolen account, a leaked CI token — publishes a matching pair and every
//! node accepts it. `docs/ARCHITECTURE.md` § Deferred Items carried this as C1.
//!
//! A signature made by a key that **never exists on GitHub** defeats that
//! class. The private half lives only on the maintainer's machine and signs at
//! the release gate; nothing in this binary can sign, only verify.
//!
//! # Why the CHECKSUM file is signed, and not the binary
//!
//! Signing the binaries would mean moving ~1.5 GB per release (the CUDA asset
//! alone is ~1 GB) to wherever the key is, which is precisely the pressure that
//! pushes a signing key into CI — where it is reachable from the build jobs and
//! stops being worth much. Signing the sidecar moves ~100 bytes, so the key can
//! stay offline without the release gate becoming a chore.
//!
//! The guarantee is the same, in two hops: the signature proves the *hash* came
//! from the key holder, and `download_update` already proves the *bytes* match
//! that hash. Debian (`Release.gpg` over a checksum list) and Node.js ship the
//! same construction.
//!
//! # The binding, which is the part that is easy to get wrong
//!
//! A signature over `<hash>  swarmllm-linux-x86_64` is a valid signature over
//! *that text* no matter which asset it is served beside. Without a further
//! check, an attacker replays the Linux sidecar and its genuine signature as
//! the CUDA one, and verification passes on a file the maintainer never
//! intended for that slot.
//!
//! So the asset name and release version are bound into minisign's **trusted
//! comment**, which is covered by the signature's second (global) signature —
//! and [`verify_release_sidecar`] *checks* that comment rather than displaying
//! it. Most tooling only prints it, which is what makes this worth spelling
//! out. Downgrade-by-replay of an older signed pair stays covered by the
//! strictly-newer version check in `UpdateChecker::apply_update_with_version`.
//!
//! # Mode
//!
//! Signatures must be **prehashed** (minisign's `ED` algorithm, the default in
//! current minisign and the only thing `rsign2` emits). Legacy `Ed`
//! signatures are refused: [`minisign_verify::PublicKey::verify`] takes
//! `allow_legacy` as its third argument and this module always passes `false`.
//! That parameter reads like "prehashed" in some summaries of the crate — it is
//! the opposite, and passing `true` would widen what we accept.

use minisign_verify::{PublicKey, Signature};

/// Public half of the release-signing key, embedded at compile time.
///
/// The matching secret key is generated and held offline by the maintainer and
/// is never present in CI, in this repository, or in a released binary. See
/// `docs/RELEASE_SIGNING.md` for generation and rotation.
///
/// Replacing this constant is what rotates the key, and a node can only trust
/// releases signed by the key compiled into *it* — so a rotation reaches a node
/// only through an update signed by the key it already has.
pub const RELEASE_PUBLIC_KEY: &str = include_str!("../release_pubkey.txt");

/// Why a release signature was not accepted.
///
/// Deliberately granular: "this build has no usable key" and "this signature is
/// forged" are the same outcome for the user but completely different for
/// whoever is debugging it, and collapsing them is how a misconfigured build
/// gets reported as an attack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureError {
    /// The key compiled into this build is missing or malformed. This is a
    /// build-configuration fault, not a bad release.
    KeyNotConfigured(String),
    /// The `.minisig` body could not be parsed.
    Malformed(String),
    /// Parsed, but not a valid signature over these bytes by our key. Covers a
    /// wrong key id, a legacy (non-prehashed) signature, and a forgery.
    NotOurSignature(String),
    /// Valid signature, but its trusted comment describes a different artifact
    /// than the one being installed — a replay across assets or versions.
    WrongArtifact {
        expected_asset: String,
        expected_version: String,
        signed_comment: String,
    },
}

impl std::fmt::Display for SignatureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignatureError::KeyNotConfigured(e) => write!(
                f,
                "this build has no usable release-signing key compiled in ({e}) — \
                 it cannot verify updates, so it will not install one"
            ),
            SignatureError::Malformed(e) => {
                write!(f, "the release signature could not be read ({e})")
            }
            SignatureError::NotOurSignature(e) => write!(
                f,
                "the release signature was not made by this project's signing key ({e})"
            ),
            SignatureError::WrongArtifact {
                expected_asset,
                expected_version,
                signed_comment,
            } => write!(
                f,
                "the release signature is genuine but describes a different download — \
                 expected {expected_asset} of version {expected_version}, \
                 signature says \"{signed_comment}\""
            ),
        }
    }
}

/// Parse the embedded public key.
///
/// Separate from verification so a misconfigured build reports itself as such
/// rather than as a stream of rejected releases.
pub fn release_public_key() -> Result<PublicKey, SignatureError> {
    // `include_str!` keeps the file's trailing newline, and a minisign public
    // key file also carries an untrusted-comment line above the key. Take the
    // last non-empty line so both a bare key and a full `.pub` file work.
    let key_line = RELEASE_PUBLIC_KEY
        .lines()
        .map(str::trim)
        .rev()
        .find(|l| !l.is_empty() && !l.starts_with("untrusted comment:"))
        .unwrap_or("");

    if key_line.is_empty() || key_line.starts_with('#') {
        return Err(SignatureError::KeyNotConfigured(
            "release_pubkey.txt holds no key".to_string(),
        ));
    }

    PublicKey::from_base64(key_line)
        .map_err(|e| SignatureError::KeyNotConfigured(format!("{key_line:.12}…: {e}")))
}

/// Verify a detached signature over a release's `.sha256` sidecar.
///
/// `sidecar_bytes` must be the sidecar **exactly as downloaded** — the
/// signature covers the bytes, so a trimmed or re-joined string will not
/// verify. `asset_name` is the binary this sidecar is supposed to describe
/// (e.g. `swarmllm-linux-x86_64-cuda`) and `version` the release it belongs to;
/// both are checked against the signed trusted comment.
pub fn verify_release_sidecar(
    sidecar_bytes: &[u8],
    signature_text: &str,
    asset_name: &str,
    version: &str,
) -> Result<(), SignatureError> {
    let public_key = release_public_key()?;
    verify_release_sidecar_with(
        &public_key,
        sidecar_bytes,
        signature_text,
        asset_name,
        version,
    )
}

/// [`verify_release_sidecar`], against a caller-supplied key.
///
/// Exists so the rule can be exercised without depending on which key a given
/// build embeds. There is exactly ONE implementation of the check — this one —
/// and the function above only resolves the key before delegating here.
pub fn verify_release_sidecar_with(
    public_key: &PublicKey,
    sidecar_bytes: &[u8],
    signature_text: &str,
    asset_name: &str,
    version: &str,
) -> Result<(), SignatureError> {
    let signature =
        Signature::decode(signature_text).map_err(|e| SignatureError::Malformed(e.to_string()))?;

    // `false` = do NOT accept legacy, non-prehashed signatures. See the module
    // docs: this argument is `allow_legacy`, not `prehashed`.
    //
    // This call also verifies the global signature covering the trusted
    // comment, which is what makes the binding check below meaningful. Reading
    // the comment before this point would be reading attacker-controlled text.
    public_key
        .verify(sidecar_bytes, &signature, false)
        .map_err(|e| SignatureError::NotOurSignature(e.to_string()))?;

    let comment = signature.trusted_comment();
    if !comment_describes(comment, asset_name, version) {
        return Err(SignatureError::WrongArtifact {
            expected_asset: asset_name.to_string(),
            expected_version: normalize_version(version).to_string(),
            signed_comment: comment.to_string(),
        });
    }

    Ok(())
}

/// Does this (already authenticated) trusted comment describe exactly this
/// asset at this version?
///
/// The comment the release gate writes looks like:
///
/// ```text
/// swarmllm-release asset:swarmllm-linux-x86_64-cuda version:0.3.191-alpha
/// ```
///
/// Both fields are REQUIRED. A comment missing either — including minisign's
/// own default `timestamp:… file:…` comment — does not describe our artifact
/// and is refused, so an artifact signed by hand without the gate's arguments
/// fails loudly instead of being accepted on the strength of the key alone.
fn comment_describes(comment: &str, asset_name: &str, version: &str) -> bool {
    let want_version = normalize_version(version);
    let mut saw_asset = false;
    let mut saw_version = false;

    for token in comment.split_whitespace() {
        if let Some(value) = token.strip_prefix("asset:") {
            // Any `asset:` token that disagrees is fatal rather than skipped —
            // otherwise a second, matching token could be appended to a
            // signature minted for something else.
            if value != asset_name {
                return false;
            }
            saw_asset = true;
        } else if let Some(value) = token.strip_prefix("version:") {
            if normalize_version(value) != want_version {
                return false;
            }
            saw_version = true;
        }
    }

    saw_asset && saw_version
}

/// Release tags carry a leading `v` and `CARGO_PKG_VERSION` does not, and both
/// spellings reach this code. Compare without it rather than making every
/// caller remember which one it holds.
fn normalize_version(v: &str) -> &str {
    v.strip_prefix('v').unwrap_or(v)
}

/// Signed fixtures for the crate's own tests.
///
/// The keypair here is a throwaway generated for the suite and has nothing to
/// do with the release key in `release_pubkey.txt` — its secret half is not in
/// this repository and could not sign a release anyone would accept. The whole
/// module is `#[cfg(test)]`, so none of it exists in a shipped binary.
///
/// `update.rs` uses these too: its download and apply paths now refuse an
/// unsigned release, so exercising them needs a genuinely signed one.
#[cfg(test)]
pub(crate) mod test_fixtures {
    use super::PublicKey;

    pub const PUBLIC_KEY: &str = "RWSnNTJFJlsboE9gR5LofeHMEgdryLMBqp5RzTvulyMYY1opHNzpQFDz";

    pub fn public_key() -> PublicKey {
        PublicKey::from_base64(PUBLIC_KEY).expect("test key parses")
    }

    /// A realistic sidecar — the genuine v0.3.190-alpha Linux hash.
    pub const SIDECAR: &[u8] =
        b"7155ba3ac5faf9f02b3a3a0ae60b43c3967a6a416cdac2544d18465712a243e8  swarmllm-linux-x86_64\n";

    /// Over [`SIDECAR`], trusted comment
    /// `swarmllm-release asset:swarmllm-linux-x86_64 version:0.3.190-alpha`.
    pub const SIGNATURE: &str = "untrusted comment: SwarmLLM release signature\n\
RUSnNTJFJlsboI3tSBmtyw4f3BF1tCAg/0thmNG+Hl9Hci79jClnIF+cU6gFNkIcj+NwzSN+astsgLpf8GeGE+nXwlrSd3SitA8=\n\
trusted comment: swarmllm-release asset:swarmllm-linux-x86_64 version:0.3.190-alpha\n\
gyAvz0gfc6wp8IW1wjk9Xc7ALfTk5gy6LsJZDvaiLz3F5QyZnkK8uLVjiNaGTAXeufPh24UZ/5QTIAI9jFvPCA==\n";

    /// The asset name `update.rs`'s staging fixtures use.
    pub const STAGED_ASSET: &str = "swarmllm-test-asset";
    /// Version those fixtures claim, matching `info_for` in `update.rs`.
    pub const STAGED_VERSION: &str = "0.9.9";

    /// Sidecar for a staged file whose contents are exactly `b"pretend binary"`.
    pub const STAGED_SIDECAR: &str =
        "9f05ef97cc90b003959b48cd01b637658368bdfc78fe54c1a061d5eff0c46104  swarmllm-test-asset\n";

    /// Over [`STAGED_SIDECAR`], bound to [`STAGED_ASSET`] / [`STAGED_VERSION`].
    pub const STAGED_SIGNATURE: &str = "untrusted comment: SwarmLLM release signature\n\
RUSnNTJFJlsboO5DvBvv4gE+W18fgt5I5dWbeYdA9L8lyT8aBzKrKFdnKbqOM2U58sPZenF9cOcUuWlxFmV0qMXEYQeXLPD0VAY=\n\
trusted comment: swarmllm-release asset:swarmllm-test-asset version:0.9.9\n\
VSYp578prAyRqLvXpPwGG6NLxG0TfGPZFsJ7nxbZVaTI8uY0UCJ0bj67ieL2he46wjhieFTrkZMmy+s2mQVUAQ==\n";
}

#[cfg(test)]
mod tests {
    use super::test_fixtures::{public_key as test_public_key, SIDECAR, SIGNATURE};
    use super::*;

    /// Verification against the TEST key, so the logic is exercised without
    /// depending on the production key being set in this checkout.
    ///
    /// Deliberately delegates to the REAL function rather than re-implementing
    /// the steps — a second copy of the check here would be a test that passes
    /// while production is broken, which is the defect this whole rules file
    /// calls "one invariant, N paths".
    fn verify_with_test_key(
        sidecar: &[u8],
        sig_text: &str,
        asset: &str,
        version: &str,
    ) -> Result<(), SignatureError> {
        verify_release_sidecar_with(&test_public_key(), sidecar, sig_text, asset, version)
    }

    #[test]
    fn a_genuine_signature_over_the_right_asset_is_accepted() {
        verify_with_test_key(SIDECAR, SIGNATURE, "swarmllm-linux-x86_64", "0.3.190-alpha")
            .expect("the fixture should verify");
    }

    #[test]
    fn the_tag_spelling_with_a_leading_v_is_the_same_version() {
        verify_with_test_key(
            SIDECAR,
            SIGNATURE,
            "swarmllm-linux-x86_64",
            "v0.3.190-alpha",
        )
        .expect("a `v`-prefixed tag names the same release");
    }

    /// The replay this module exists to stop: a genuine signature, genuine
    /// bytes, served in the slot of a DIFFERENT asset.
    #[test]
    fn a_genuine_signature_cannot_be_replayed_onto_another_asset() {
        let err = verify_with_test_key(
            SIDECAR,
            SIGNATURE,
            "swarmllm-linux-x86_64-cuda",
            "0.3.190-alpha",
        )
        .expect_err("a signature for the plain linux asset must not pass for the CUDA one");
        assert!(matches!(err, SignatureError::WrongArtifact { .. }), "{err}");
    }

    /// The same replay across releases — an old signed pair re-served as a new
    /// one. `apply_update_with_version` also refuses a downgrade; this makes
    /// the signature itself carry the release it belongs to.
    #[test]
    fn a_genuine_signature_cannot_be_replayed_onto_another_version() {
        let err =
            verify_with_test_key(SIDECAR, SIGNATURE, "swarmllm-linux-x86_64", "0.3.191-alpha")
                .expect_err("a signature for .190 must not pass for .191");
        assert!(matches!(err, SignatureError::WrongArtifact { .. }), "{err}");
    }

    #[test]
    fn a_tampered_sidecar_does_not_verify() {
        let mut tampered = SIDECAR.to_vec();
        tampered[0] = b'8';
        let err = verify_with_test_key(
            &tampered,
            SIGNATURE,
            "swarmllm-linux-x86_64",
            "0.3.190-alpha",
        )
        .expect_err("a changed hash must not verify");
        assert!(matches!(err, SignatureError::NotOurSignature(_)), "{err}");
    }

    /// A signature is over BYTES. Re-serialising the sidecar without its
    /// trailing newline is the most likely way a caller breaks this, so pin it:
    /// the failure must be a clean error, never a pass.
    #[test]
    fn the_sidecar_must_be_the_bytes_that_were_signed() {
        let trimmed = SIDECAR
            .strip_suffix(b"\n")
            .expect("fixture ends in a newline");
        let err =
            verify_with_test_key(trimmed, SIGNATURE, "swarmllm-linux-x86_64", "0.3.190-alpha")
                .expect_err("trimming the sidecar changes the signed bytes");
        assert!(matches!(err, SignatureError::NotOurSignature(_)), "{err}");
    }

    #[test]
    fn a_signature_from_another_key_is_refused() {
        // Same fixture, verified against a different (production) key id.
        let other =
            PublicKey::from_base64("RWQf6LRCGA9i53mlYecO4IzT51TGPpvWucNSCh1CBM0QTaLn73Y7GFO3")
                .expect("well-formed key");
        let signature = Signature::decode(SIGNATURE).expect("fixture parses");
        assert!(
            other.verify(SIDECAR, &signature, false).is_err(),
            "a signature must not verify under an unrelated key"
        );
    }

    #[test]
    fn a_malformed_signature_is_reported_as_malformed() {
        let err = verify_with_test_key(
            SIDECAR,
            "not a signature",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha",
        )
        .expect_err("garbage must not parse");
        assert!(matches!(err, SignatureError::Malformed(_)), "{err}");
    }

    /// minisign's own default trusted comment (`timestamp:… file:…`) names no
    /// `asset:`/`version:`, so a release signed by hand without the gate's
    /// arguments must be refused rather than trusted on the key alone.
    #[test]
    fn a_comment_without_the_binding_fields_is_not_enough() {
        assert!(!comment_describes(
            "timestamp:1789799107\tfile:swarmllm-linux-x86_64.sha256",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha"
        ));
        assert!(!comment_describes(
            "swarmllm-release asset:swarmllm-linux-x86_64",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha"
        ));
        assert!(!comment_describes(
            "swarmllm-release version:0.3.190-alpha",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha"
        ));
    }

    /// Appending a matching token must not rescue a comment that also names
    /// something else — otherwise the check is "mentions us somewhere".
    #[test]
    fn a_second_contradicting_field_still_fails() {
        assert!(!comment_describes(
            "swarmllm-release asset:swarmllm-macos-aarch64 asset:swarmllm-linux-x86_64 version:0.3.190-alpha",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha"
        ));
        assert!(!comment_describes(
            "swarmllm-release asset:swarmllm-linux-x86_64 version:0.3.1-alpha version:0.3.190-alpha",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha"
        ));
    }

    /// An asset name that merely starts with ours is a different file.
    #[test]
    fn asset_matching_is_exact_not_a_prefix() {
        assert!(!comment_describes(
            "swarmllm-release asset:swarmllm-linux-x86_64-cuda version:0.3.190-alpha",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha"
        ));
        assert!(comment_describes(
            "swarmllm-release asset:swarmllm-linux-x86_64 version:0.3.190-alpha",
            "swarmllm-linux-x86_64",
            "0.3.190-alpha"
        ));
    }
}
