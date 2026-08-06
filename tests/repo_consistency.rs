//! Consistency checks between the repo's files and the docs describing them.
//!
//! These exist because a claim in prose has no compiler. The i18n entry count
//! is stated independently in `CLAUDE.md` and `docs/ARCHITECTURE.md`, and a
//! round that adds keys tends to update one and not the other — sweep rounds
//! have caught that same drift roughly twenty times. Remembering has been
//! tried; a test is cheaper.

use std::collections::BTreeSet;
use std::path::PathBuf;

/// Locale files carry two metadata entries that are not translatable strings.
const NON_TRANSLATION_KEYS: usize = 2; // `_lang`, `_dir`

/// Every language listed in `.claude/rules/i18n.md`.
const EXPECTED_LOCALES: usize = 21;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn locale_files() -> Vec<PathBuf> {
    let dir = repo_root().join("frontend/i18n");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    files
}

fn keys_of(path: &PathBuf) -> BTreeSet<String> {
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let map: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&raw)
        .unwrap_or_else(|e| panic!("{} is not a JSON object: {e}", path.display()));
    map.keys().cloned().collect()
}

/// Read the `<N> entries per locale` / `<N> translation keys` figures a doc
/// claims. Panics rather than silently skipping if the wording changed — a
/// check that quietly stops checking is worse than no check.
fn claimed_count(text: &str, needle: &str, doc: &str) -> usize {
    let idx = text.find(needle).unwrap_or_else(|| {
        panic!("{doc}: no '<N>{needle}' claim found — wording changed? Update this test with it.")
    });
    let digits: String = text[..idx]
        .chars()
        .rev()
        .take_while(char::is_ascii_digit)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    digits
        .parse()
        .unwrap_or_else(|_| panic!("{doc}: no number immediately before '{needle}'"))
}

#[test]
fn every_locale_has_the_same_key_set() {
    let files = locale_files();
    assert_eq!(
        files.len(),
        EXPECTED_LOCALES,
        "expected {EXPECTED_LOCALES} locale files, found {}",
        files.len()
    );

    let en = repo_root().join("frontend/i18n/en.json");
    let reference = keys_of(&en);

    for path in &files {
        if *path == en {
            continue;
        }
        let keys = keys_of(path);
        let missing: Vec<_> = reference.difference(&keys).cloned().collect();
        let extra: Vec<_> = keys.difference(&reference).cloned().collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "{} is out of parity with en.json\n  missing {} key(s): {:?}\n  extra {} key(s): {:?}",
            path.file_name().unwrap().to_string_lossy(),
            missing.len(),
            missing,
            extra.len(),
            extra
        );
    }
}

#[test]
fn docs_report_the_actual_i18n_counts() {
    let entries = keys_of(&repo_root().join("frontend/i18n/en.json")).len();
    let translation_keys = entries - NON_TRANSLATION_KEYS;

    // Both docs state these independently, which is exactly why they drift.
    for doc in ["CLAUDE.md", "docs/ARCHITECTURE.md"] {
        let text = std::fs::read_to_string(repo_root().join(doc))
            .unwrap_or_else(|e| panic!("cannot read {doc}: {e}"));

        assert_eq!(
            claimed_count(&text, " entries per locale", doc),
            entries,
            "{doc} claims the wrong per-locale entry count (actual {entries}). \
             Adding or removing i18n keys means updating BOTH {doc} and its sibling doc."
        );
        assert_eq!(
            claimed_count(&text, " translation keys", doc),
            translation_keys,
            "{doc} claims the wrong translation-key count \
             (actual {translation_keys} = {entries} entries - {NON_TRANSLATION_KEYS} metadata keys)."
        );
    }
}

/// Every CLI command that sends an authenticated request must translate a 401
/// into `cli::exit_api_key_rejected`'s explanation, via
/// `cli::exit_if_api_key_rejected`.
///
/// The daemon answers a stale or mismatched key with 401, and each command used
/// to render that in its own words — "Download request failed (401
/// Unauthorized)", "Could not remove <model>: request failed", reqwest's raw
/// `error_for_status` text, or, worst, `discover_model`'s "No models available
/// — load a model first", which sends the user to download a model to fix an
/// auth problem. None of them named the cause, which is almost always the CLI
/// and the daemon disagreeing about the data directory.
///
/// This is the codebase's signature defect — one invariant, N paths, with a
/// correct helper that two of eight callers used. A grep test is the cheapest
/// thing that makes a NEW command inherit the requirement rather than having to
/// remember it.
#[test]
fn cli_commands_explain_a_rejected_key() {
    let cli_dir = repo_root().join("src/cli");
    let mut offenders = Vec::new();

    for entry in std::fs::read_dir(&cli_dir).expect("src/cli must exist") {
        let path = entry.expect("readable dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        // `mod.rs` defines the helpers; `run.rs` starts the daemon rather than
        // calling it; `update.rs` talks to GitHub, not to our own API.
        if matches!(name.as_str(), "mod.rs" | "run.rs" | "update.rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("readable source");

        // "Sends an authenticated request" = builds an Authorization header, by
        // either spelling reqwest offers.
        let authenticates = src.contains("Bearer ") || src.contains("bearer_auth");
        if !authenticates {
            continue;
        }
        // Commands that delegate every authenticated call to a shared helper
        // (`discover_model`, `pool_post`) inherit the check from it.
        let handles_it =
            src.contains("exit_if_api_key_rejected") || src.contains("exit_api_key_rejected");
        if !handles_it {
            offenders.push(name);
        }
    }

    assert!(
        offenders.is_empty(),
        "these CLI commands send authenticated requests but never explain a rejected key: {offenders:?}\n\
         Call `super::exit_if_api_key_rejected(status, data_dir, port)` on the response status \
         before interpreting the body — otherwise a stale api_key file surfaces as a confusing \
         message about models, downloads or the transport."
    );
}

/// Remove `{placeholder}` names so they are not mistaken for words.
fn strip_placeholders(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut depth = 0usize;
    for c in s.chars() {
        match c {
            '{' => depth += 1,
            '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// English strings that are meant to read identically in every language, so a
/// locale repeating them verbatim is correct rather than untranslated.
///
/// Keep this list SHORT and specific. Every entry is a promise that the string
/// carries no translatable words; adding one to silence the check below is how
/// the panel this test exists for came to ship in English.
fn identical_by_nature(english: &str) -> bool {
    // A string with no run of three or more letters is punctuation, digits,
    // units or symbols: "≤ 25%", "#", "{n}k ctx", "A–Z".
    let has_word = english
        .split(|c: char| !c.is_alphabetic())
        .any(|w| w.chars().count() >= 3);
    if !has_word {
        return true;
    }
    // Acronyms and proper nouns that every locale keeps.
    const KEEP: &[&str] = &[
        "LAN",
        "CPU",
        "GPU",
        "RAM",
        "VRAM",
        "API",
        "ID",
        "HuggingFace",
        "SwarmLLM",
        "AI",
        "Ping",
        "tok/s",
        "Offline",
        "ctx",
        "KB",
        "MB",
        "GB",
    ];
    english
        .split(|c: char| !c.is_alphanumeric() && c != '/')
        .filter(|w| w.chars().any(char::is_alphabetic))
        .all(|w| KEEP.iter().any(|k| k.eq_ignore_ascii_case(w)))
}

/// A locale must carry its own words, not English ones.
///
/// Key parity — which `every_locale_has_the_same_key_set` already checks — says
/// nothing about VALUES, and that gap shipped: the whole network-status panel,
/// the first thing a user reads when something looks wrong, was extended with
/// 25 new strings that were added to all 21 files with their ENGLISH text. Every
/// key was present, so every existing check passed, and 20 languages showed
/// sentences like "Just your computer — share your peer address to invite
/// others" untranslated.
///
/// The rule is enforced against a whole phrase rather than a single word,
/// because a one-word label genuinely does coincide across languages often
/// enough that flagging it would train people to ignore this test.
#[test]
fn locales_do_not_fall_back_to_english_prose() {
    let root = repo_root().join("frontend/i18n");
    let en_raw = std::fs::read_to_string(root.join("en.json")).expect("read en.json");
    let en: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&en_raw).expect("en.json is a JSON object");

    let mut offenders: Vec<String> = Vec::new();
    for path in locale_files() {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if name == "en" {
            continue;
        }
        let raw = std::fs::read_to_string(&path).expect("read locale");
        let loc: serde_json::Map<String, serde_json::Value> =
            serde_json::from_str(&raw).expect("locale is a JSON object");

        for (key, value) in &en {
            if key.starts_with('_') {
                continue;
            }
            let Some(english) = value.as_str() else {
                continue;
            };
            // Only judge real sentences. `{placeholder}` names are code, not
            // words — counting them made "{size} VRAM" look like a phrase.
            let prose = strip_placeholders(english);
            let words = prose
                .split_whitespace()
                .filter(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 3)
                .count();
            if words < 2 || identical_by_nature(&prose) {
                continue;
            }
            if loc.get(key).and_then(|v| v.as_str()) == Some(english) {
                offenders.push(format!("  {name}: {key} = {english:?}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "{} strings are still in English in a non-English locale.\n\
         Translate them in `frontend/i18n/<lang>.json` — see `.claude/rules/i18n.md`.\n\
         If a string genuinely reads the same in every language, say so in \
         `identical_by_nature` rather than leaving it to look untranslated:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}

/// Every setting a user can write must be read by something.
///
/// Three have shipped that were not, each declared, defaulted, documented in
/// the configuration reference, and consulted nowhere: `max_peers` (gotcha
/// #236), `network.enable_relay_client`, and `[ui] theme`. A setting that does
/// nothing is worse than a missing one — the user believes it took effect and
/// configures around a behaviour they never changed.
///
/// The check is deliberately crude: a field name must appear in some source
/// file OTHER than the one declaring it. That catches "nothing reads this at
/// all", which is the failure that keeps happening, without pretending to know
/// whether a read is meaningful. A field consumed only inside its own module
/// (none today) would need an entry here explaining why.
#[test]
fn every_config_setting_is_read_somewhere() {
    let root = repo_root();
    let cfg_dir = root.join("src/config");
    let mut sources: Vec<(PathBuf, String)> = Vec::new();
    let mut stack = vec![root.join("src"), root.join("crates")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                if let Ok(t) = std::fs::read_to_string(&p) {
                    sources.push((p, t));
                }
            }
        }
    }

    let mut dead: Vec<String> = Vec::new();
    for (path, text) in sources.iter().filter(|(p, _)| p.starts_with(&cfg_dir)) {
        for line in text.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("pub ") else {
                continue;
            };
            let Some((name, _)) = rest.split_once(':') else {
                continue;
            };
            let name = name.trim();
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            {
                continue; // types, generics, fn signatures
            }
            let used_elsewhere = sources.iter().any(|(p, t)| {
                p != path
                    && t.split(|c: char| !(c.is_alphanumeric() || c == '_'))
                        .any(|w| w == name)
            });
            if !used_elsewhere {
                dead.push(format!(
                    "  {} (declared in {})",
                    name,
                    path.strip_prefix(&root).unwrap_or(path).display()
                ));
            }
        }
    }
    dead.sort();
    dead.dedup();

    assert!(
        dead.is_empty(),
        "{} configuration setting(s) are read by nothing. Either wire them up or \
         remove them — a setting that silently does nothing is worse than no \
         setting, because the user believes it took effect:\n{}",
        dead.len(),
        dead.join("\n")
    );
}
