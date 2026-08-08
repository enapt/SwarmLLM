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

/// Fields on a workflow matrix cell that feed the build cache key or the
/// compiled output. `rustflags` is the one that actually bit: it is hashed into
/// the `rust-cache` key AND into cargo's own fingerprint.
const CACHE_KEY_FIELDS: &[&str] = &[
    "runner",
    "target",
    "features",
    "rustflags",
    "cuda_linux",
    "cuda_windows",
    "vulkan",
];

/// Parse a GitHub Actions `matrix.include:` list into `name -> {field: value}`.
///
/// Deliberately a small hand parser rather than a YAML dependency: the two
/// files have a fixed, flat shape, and a parse that stops matching is a loud
/// test failure rather than a silent pass.
fn workflow_matrix(file: &str) -> std::collections::BTreeMap<String, Vec<(String, String)>> {
    let path = repo_root().join(".github/workflows").join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {file}: {e}"));
    let mut cells: std::collections::BTreeMap<String, Vec<(String, String)>> = Default::default();
    let mut current: Option<String> = None;
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(rest) = line.strip_prefix("- name: ") {
            let name = rest.trim().trim_matches('"').to_string();
            // Only matrix cells, not job/step names: those sit at a shallower
            // indent and carry no `runner:`/`target:` beneath them. Recording
            // an extra key is harmless — it simply never matches the other file.
            current = Some(name.clone());
            cells.entry(name).or_default();
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some(name) = current.clone() else {
            continue;
        };
        if let Some((k, v)) = line.split_once(": ") {
            let k = k.trim();
            if CACHE_KEY_FIELDS.contains(&k) {
                let v = v.trim().trim_matches('"').to_string();
                cells.entry(name).or_default().push((k.to_string(), v));
            }
        }
    }
    cells
}

/// An omitted matrix flag and an explicit `false` mean the same thing to
/// `${{ matrix.x && ... }}`, so they must not read as divergence. Only values
/// that actually change the build are compared.
fn normalize(value: &str) -> &str {
    match value {
        "false" => "",
        other => other,
    }
}

/// Every cell `cache-warm.yml` warms must match `release.yml`'s cell of the same
/// name on everything that feeds the cache key.
///
/// Adding `-C target-cpu=x86-64-v3` to release.yml in v0.3.79 without mirroring
/// it here made every warmed cache unrestorable, and the CUDA release build went
/// from 12 minutes to 78. It survived two releases because a cache miss still
/// produces a correct binary — it is only slow. cache-warm.yml already carried a
/// comment saying its cells must mirror release.yml; a comment is not a check.
#[test]
fn cache_warm_mirrors_the_release_matrix() {
    let release = workflow_matrix("release.yml");
    let warm = workflow_matrix("cache-warm.yml");

    let warmed: Vec<_> = warm
        .iter()
        .filter(|(_, fields)| fields.iter().any(|(k, _)| k == "runner"))
        .collect();
    assert!(
        !warmed.is_empty(),
        "parsed no matrix cells out of cache-warm.yml — the parser has stopped \
         matching the file's shape, so this test is no longer checking anything"
    );

    let mut problems = Vec::new();
    for (name, warm_fields) in warmed {
        let Some(release_fields) = release.get(name) else {
            problems.push(format!(
                "cache-warm warms `{name}`, which release.yml does not build — \
                 warming a cell nothing restores is pure cost"
            ));
            continue;
        };
        for (key, warm_value) in warm_fields {
            let release_value = release_fields
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            if normalize(release_value) != normalize(warm_value) {
                problems.push(format!(
                    "`{name}` differs on `{key}`: cache-warm has {warm_value:?}, \
                     release.yml has {release_value:?}"
                ));
            }
        }
        // The reverse direction matters just as much: a field present only on
        // the release cell (as `rustflags` was) still diverges the key.
        for (key, release_value) in release_fields {
            if !CACHE_KEY_FIELDS.contains(&key.as_str()) {
                continue;
            }
            let warmed_value = warm_fields.iter().find(|(k, _)| k == key);
            if warmed_value.is_none() && !normalize(release_value).is_empty() {
                problems.push(format!(
                    "`{name}` sets `{key}` = {release_value:?} in release.yml but \
                     not in cache-warm.yml — the warmed cache will not restore"
                ));
            }
        }
    }

    assert!(
        problems.is_empty(),
        "cache-warm.yml and release.yml disagree on {} cache-key input(s). Every \
         release built by an affected cell recompiles from scratch:\n  {}",
        problems.len(),
        problems.join("\n  ")
    );
}

/// Every `CUDA_COMPUTE_CAP: ... '<NN>' ...` value a workflow pins, in file order.
///
/// The workflows set it inside a GitHub expression rather than as a bare value
/// (`${{ (matrix.cuda_linux || matrix.cuda_windows) && '80' || '' }}`), so the
/// digits are pulled out of the quoted branch. An empty result is a hard
/// failure: a check that silently stops finding anything is worse than no check.
fn pinned_compute_caps(file: &str) -> Vec<String> {
    let path = repo_root().join(".github/workflows").join(file);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {file}: {e}"));
    let mut found = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("CUDA_COMPUTE_CAP:") else {
            continue;
        };
        // Either `"80"` / `'80'` directly, or the first quoted run of digits
        // inside the expression. Both reduce to "first quoted number".
        let digits: String = rest
            .split(['\'', '"'])
            .find(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit()))
            .unwrap_or_default()
            .to_string();
        assert!(
            !digits.is_empty(),
            "{file}: found a CUDA_COMPUTE_CAP line with no quoted number in it \
             ({line:?}) — the shape changed, so this test has stopped checking"
        );
        found.push(digits);
    }
    assert!(
        !found.is_empty(),
        "{file}: no CUDA_COMPUTE_CAP pin found at all — either the key was \
         renamed or the build has stopped pinning it, and candle-kernels then \
         auto-detects the BUILDER's GPU (or panics with `ParseIntError` on a \
         runner that has none)"
    );
    found
}

/// The compute-capability floor is stated in four places that cannot disagree.
///
/// `MIN_COMPUTE_CAP` decides at RUNTIME whether to route a card to the CPU;
/// `CUDA_COMPUTE_CAP` decides at BUILD time which kernels exist. If the
/// constant is lower than the build, cards that cannot run a single forward are
/// sent to the GPU anyway and every request fails with
/// `CUDA_ERROR_NO_BINARY_FOR_GPU`. If it is higher, working cards are silently
/// demoted to the CPU and nobody finds out except by wondering why it is slow.
///
/// cache-warm.yml matters for a third reason: it shares one rust-cache key with
/// release.yml, so a divergent value there means the release restores a cache
/// built for a different architecture.
#[test]
fn compute_cap_matches_release_workflow() {
    let expected = format!(
        "{}{}",
        swarmllm::daemon::gpu_support::MIN_COMPUTE_CAP.0,
        swarmllm::daemon::gpu_support::MIN_COMPUTE_CAP.1
    );

    let mut problems = Vec::new();
    for file in ["release.yml", "cache-warm.yml", "ci.yml"] {
        for pinned in pinned_compute_caps(file) {
            if pinned != expected {
                problems.push(format!(
                    "{file} builds kernels for compute capability {pinned}, but \
                     daemon::gpu_support::MIN_COMPUTE_CAP is {expected}"
                ));
            }
        }
    }
    assert!(
        problems.is_empty(),
        "the compute-capability floor disagrees across {} place(s):\n  {}\n\
         Both must move together — see src/daemon/gpu_support.rs.",
        problems.len(),
        problems.join("\n  ")
    );
}

/// FlashAttention is the whole reason the floor is 8.0, so the two must not
/// drift apart in either direction.
///
/// Dropping `flash-attn` from `cuda` while leaving the floor at 80 would keep
/// pre-Ampere cards excluded for nothing — the exact trade this change accepted
/// only because it bought something. Adding it back to a build below 8.0 does
/// not compile: every `candle-flash-attn` kernel source is `_sm80`.
#[test]
fn flash_attn_and_the_compute_cap_floor_agree() {
    let manifest =
        std::fs::read_to_string(repo_root().join("Cargo.toml")).expect("read Cargo.toml");
    let cuda_line = manifest
        .lines()
        .find(|l| l.trim_start().starts_with("cuda = ["))
        .expect("no `cuda = [...]` feature in Cargo.toml — was it renamed?");

    let floor_is_ampere = swarmllm::daemon::gpu_support::MIN_COMPUTE_CAP >= (8, 0);
    let has_flash = cuda_line.contains("flash-attn");

    assert_eq!(
        has_flash,
        floor_is_ampere,
        "the `cuda` feature {} flash-attn but MIN_COMPUTE_CAP is {:?}.\n\
         flash-attn REQUIRES 8.0 (its kernels are all `_sm80`), and a floor of \
         8.0 is only worth paying for BECAUSE of flash-attn. Change both or \
         neither.\n  cuda = {cuda_line}",
        if has_flash { "includes" } else { "omits" },
        swarmllm::daemon::gpu_support::MIN_COMPUTE_CAP,
    );
}

/// Every `[[example]]` in the root manifest must declare an explicit `path`.
///
/// Without one, cargo resolves the example by scanning `examples/` and **fails
/// to parse the manifest entirely** when that directory is absent. The
/// Dockerfile's build context is exactly that case: it copies `src/`,
/// `crates/`, `frontend/` and `config/`, and nothing else. So an example
/// declared without a path breaks the container image while `cargo build`,
/// `cargo clippy --all-targets`, the whole test suite and every CI job stay
/// green — the Docker workflow only runs on a tag, so it surfaces at release.
///
/// That happened on 2026-08-08 and cost v0.3.83-alpha its container images.
/// Declaring the path defers the existence check to the point of building that
/// target, which the image build never does.
#[test]
fn every_declared_example_has_an_explicit_path() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("read root Cargo.toml");

    let mut offenders = Vec::new();
    let mut in_example = false;
    let mut name = String::new();
    let mut saw_path = false;
    // Section-scanned rather than TOML-parsed so the test needs no new
    // dependency; `[[example]]` blocks are flat and end at the next header.
    for line in manifest.lines().map(str::trim) {
        if line.starts_with('[') {
            if in_example && !saw_path {
                offenders.push(name.clone());
            }
            in_example = line == "[[example]]";
            name.clear();
            saw_path = false;
            continue;
        }
        if !in_example {
            continue;
        }
        if let Some(v) = line.strip_prefix("name") {
            name = v
                .trim_start_matches([' ', '='])
                .trim()
                .trim_matches('"')
                .to_string();
        } else if line.starts_with("path") {
            saw_path = true;
        }
    }
    if in_example && !saw_path {
        offenders.push(name);
    }

    assert!(
        offenders.is_empty(),
        "these [[example]] entries have no explicit `path`, which breaks the Docker image \
         build (its context has no examples/ directory) while every other check passes: {offenders:?}"
    );
}
