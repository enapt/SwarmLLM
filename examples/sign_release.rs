//! Sign a release's checksum files with the offline release key — one password
//! prompt for the whole release.
//!
//! `examples/sign_release.sh` drives this; it does the GitHub side (downloading
//! each `.sha256`, uploading each `.minisig`, publishing) and hands the signing
//! to this. Run directly only when debugging that.
//!
//! ```text
//! cargo run --example sign_release -- <version> <dir> <asset>...
//! ```
//!
//! # Why this exists rather than the `rsign` CLI
//!
//! `rsign sign` unlocks the secret key once per invocation and reads the
//! password straight from `/dev/tty` through `rpassword` — there is no env var
//! and no stdin path (a piped password fails with `os error 6`). Seven release
//! assets therefore meant seven identical password prompts at the gate, which
//! is not dangerous but is exactly the kind of friction that gets a signing
//! step skipped or automated badly. Here the key is decrypted once and used
//! seven times.
//!
//! # What it guarantees before it writes anything
//!
//! **The key must be the one the shipped binaries trust.** `release_pubkey.txt`
//! is compiled into every build, and a node verifies against that and nothing
//! else — so signing with the wrong key produces a release that looks perfect
//! and is refused by the entire field. The public key is derived from the
//! secret and compared before any file is signed.
//!
//! Every signature is then verified against that same published key before
//! being written out, so a bad signature cannot reach the upload step.
//!
//! # Password
//!
//! Prompted on a terminal. When stdin is not a terminal the first line is read
//! from it instead, which is what makes this testable — deliberately piping a
//! password in is a choice, not an accident.

use std::error::Error;
use std::fs;
use std::io::{BufRead, IsTerminal, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use minisign::{PublicKey, SecretKeyBox, SignatureBox};
// The ONE definition of the comment format, shared with the verifier that
// checks it. Formatting our own string here is how a signer and a verifier
// drift apart without any test noticing.
use swarmllm::update_signature::trusted_comment;

/// The key the shipped binaries verify against.
///
/// `SWARMLLM_RELEASE_PUBKEY_FILE` overrides the location, for exercising this
/// tool against a throwaway keypair. It is not a way to weaken anything: the
/// only thing it changes is which key this program REFUSES to sign with, and a
/// signature made by any other key is still rejected by every node, because a
/// node reads the key compiled into it and nothing else.
fn published_public_key(repo_root: &Path) -> Result<PublicKey, Box<dyn Error>> {
    let path = std::env::var("SWARMLLM_RELEASE_PUBKEY_FILE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("release_pubkey.txt"));
    let raw =
        fs::read_to_string(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let line = raw
        .lines()
        .map(str::trim)
        .rev()
        .find(|l| !l.is_empty() && !l.starts_with('#') && !l.starts_with("untrusted comment:"))
        .ok_or("release_pubkey.txt holds no key")?;
    Ok(PublicKey::from_base64(line)?)
}

fn read_password() -> Result<String, Box<dyn Error>> {
    if std::io::stdin().is_terminal() {
        Ok(rpassword::prompt_password(
            "Password for the release signing key: ",
        )?)
    } else {
        let mut line = String::new();
        // EOF with NOTHING read means no terminal and nothing piped — not a
        // password at all. Passed on as "", it came back from the key as
        // "Wrong password", which sent the one person who can sign to re-type a
        // correct password twice: a wrapper had run this in the background,
        // and a background job's stdin is /dev/null (gotcha #691). A piped
        // empty password still arrives as a newline and is unaffected.
        if std::io::stdin().lock().read_line(&mut line)? == 0 {
            return Err(
                "no terminal to ask for the password on, and nothing on standard \
                        input — run this from a terminal. If you did, something between \
                        the terminal and this program is swallowing its input (a wrapper \
                        that backgrounds the command gives it /dev/null)"
                    .into(),
            );
        }
        Ok(line.trim_end_matches(['\r', '\n']).to_string())
    }
}

fn main() {
    // Printed with Display, not Debug. Returning `Result` from `main` formats
    // the error with `{:?}`, which turns a multi-line explanation into one line
    // of literal `\n` escapes — and the message that matters most here (signing
    // with a key the field does not trust) is three lines long.
    if let Err(e) = run() {
        eprintln!("\nsign_release: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        eprintln!("usage: sign_release <version> <dir> <asset>...");
        eprintln!("  env SWARMLLM_RELEASE_SECRET_KEY  (default ~/.swarmllm-release-secret.key)");
        std::process::exit(2);
    }
    let version = args[0].trim_start_matches('v').to_string();
    let dir = PathBuf::from(&args[1]);
    let assets = &args[2..];

    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let secret_path = std::env::var("SWARMLLM_RELEASE_SECRET_KEY")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::var("HOME").unwrap_or_default())
                .join(".swarmllm-release-secret.key")
        });

    let published = published_public_key(&repo_root)?;
    let sk_box = SecretKeyBox::from_string(&fs::read_to_string(&secret_path).map_err(|e| {
        format!(
            "cannot read the secret key at {}: {e}",
            secret_path.display()
        )
    })?)?;

    // One prompt, one decryption, however many assets follow.
    let secret = sk_box.into_secret_key(Some(read_password()?))?;

    // Refuse before writing anything if this is not the key the field trusts.
    // A release signed by the wrong key is not a recoverable mistake — it looks
    // correct everywhere except on the nodes, all of which reject it.
    let derived = PublicKey::from_secret_key(&secret)?;
    if derived.to_base64() != published.to_base64() {
        return Err(format!(
            "the secret key at {} is NOT the key the binaries trust.\n  \
             release_pubkey.txt: {}\n  that key derives:   {}\n\
             Signing with it would produce a release every node refuses.",
            secret_path.display(),
            published.to_base64(),
            derived.to_base64()
        )
        .into());
    }
    eprintln!("Key matches release_pubkey.txt ({})", published.to_base64());

    for asset in assets {
        let sidecar = dir.join(format!("{asset}.sha256"));
        let sig_path = dir.join(format!("{asset}.sha256.minisig"));
        let comment = trusted_comment(asset, &version);

        let mut f = fs::File::open(&sidecar)
            .map_err(|e| format!("cannot read {}: {e}", sidecar.display()))?;
        let signature: SignatureBox = minisign::sign(
            Some(&published),
            &secret,
            &mut f,
            Some(&comment),
            Some("SwarmLLM release signature"),
        )?;

        // Verify what was just produced, against the PUBLISHED key, before it
        // can be uploaded. Cheap, and it means a broken signature cannot leave
        // this machine.
        f.seek(SeekFrom::Start(0))?;
        minisign::verify(&published, &signature, &mut f, true, false, false)
            .map_err(|e| format!("{asset}: signature did not verify after signing: {e}"))?;

        fs::write(&sig_path, signature.into_string())?;
        eprintln!("  [{asset}] signed and verified");
    }

    eprintln!("Signed {} asset(s) for {}", assets.len(), version);
    Ok(())
}
