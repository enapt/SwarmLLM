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

/// The advertised minimum Rust version must be the real one.
///
/// **This was wrong by nine minor versions and nothing noticed.** The repo
/// promised Rust 1.80 in seven places — including a README badge, the first
/// thing anyone evaluating the project sees — while `redb` in the locked
/// dependency tree requires 1.89. CI only ever builds `stable`, so no job on
/// the push path could have caught it: the claim was pure prose with no
/// compiler behind it, which is the exact situation this file exists for.
///
/// A wrong floor is not harmless in either direction. Too low sends someone on
/// a distro toolchain into a dependency error instead of a clear "upgrade
/// Rust"; too high turns people away who could have built it fine.
///
/// Derived from `cargo metadata`, so it tracks reality rather than a second
/// hand-maintained number. When a dependency bump raises the floor, this fails
/// and names the crate responsible.
#[test]
fn msrv_claim_matches_the_dependency_tree() {
    fn parse(v: &str) -> (u64, u64, u64) {
        let mut it = v.split('.').map(|p| p.parse::<u64>().unwrap_or(0));
        (
            it.next().unwrap_or(0),
            it.next().unwrap_or(0),
            it.next().unwrap_or(0),
        )
    }

    let out = std::process::Command::new(env!("CARGO"))
        .args(["metadata", "--format-version", "1"])
        .current_dir(repo_root())
        .output()
        .expect("cargo metadata");
    assert!(out.status.success(), "cargo metadata failed");
    let meta: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("cargo metadata is not JSON");

    // The floor is the highest `rust-version` anywhere in the resolved tree.
    let mut floor = (0, 0, 0);
    let mut blamed = String::new();
    for pkg in meta["packages"].as_array().expect("packages") {
        let Some(rv) = pkg["rust_version"].as_str() else {
            continue;
        };
        let v = parse(rv);
        if v > floor {
            floor = v;
            blamed = format!(
                "{} {} needs {rv}",
                pkg["name"].as_str().unwrap_or("?"),
                pkg["version"].as_str().unwrap_or("?")
            );
        }
    }
    let declared = parse(env!("CARGO_PKG_RUST_VERSION"));
    assert!(
        declared >= floor,
        "Cargo.toml declares rust-version {}.{}.{} but the dependency tree needs \
         {}.{}.{} ({blamed}). Raise rust-version in Cargo.toml AND both crates/*/Cargo.toml, \
         then update the README badge, README install note, CONTRIBUTING.md, \
         docs/book/src/getting-started/installation.md and CLAUDE.md.",
        declared.0,
        declared.1,
        declared.2,
        floor.0,
        floor.1,
        floor.2,
    );

    // Every place the number is written by hand must agree with Cargo.toml.
    let want = env!("CARGO_PKG_RUST_VERSION");
    let claims: [(&str, String); 6] = [
        ("README.md", format!("rust-{want}%2B")),
        ("README.md", format!("# Requires Rust {want}+")),
        ("CONTRIBUTING.md", format!("Requires Rust {want}+")),
        (
            "docs/book/src/getting-started/installation.md",
            format!("Requires Rust {want}+"),
        ),
        ("CLAUDE.md", format!("Minimum Rust Version**: {want}+")),
        ("crates/swarmllm-types/Cargo.toml", format!("\"{want}\"")),
    ];
    for (file, needle) in claims {
        let text = std::fs::read_to_string(repo_root().join(file))
            .unwrap_or_else(|e| panic!("read {file}: {e}"));
        assert!(
            text.contains(&needle),
            "{file} does not state the MSRV as {want} (looked for {needle:?}). \
             Cargo.toml and the docs must agree, or the badge lies to people \
             deciding whether they can build this."
        );
    }
}

/// Every HTTP route the server registers must appear in the architecture doc.
///
/// `docs/ARCHITECTURE.md` calls itself the canonical HTTP API reference, and a
/// reference that silently omits endpoints is worse than one that admits a gap:
/// people conclude the endpoint does not exist. Seven had drifted out of it —
/// the five long-lived Claude-session routes, `enable-privacy`, and
/// `reference-models` — none of which any check could have noticed, because a
/// route works perfectly whether or not anyone wrote it down.
///
/// Matching is on the path with its parameter names normalised, since the doc
/// writes `{model_id}` where the router writes `{id}` and both are correct for
/// a reader. Only `/api/*` paths are covered: the OpenAI and Anthropic surfaces
/// are specified by those vendors and documented as compatibility statements
/// rather than route-by-route.
#[test]
fn every_api_route_is_in_the_architecture_doc() {
    /// `/api/admin/models/{id}/shards/{index}` -> `/api/admin/models/{}/shards/{}`
    fn normalise(path: &str) -> String {
        let mut out = String::with_capacity(path.len());
        let mut in_param = false;
        for c in path.chars() {
            match c {
                '{' => {
                    in_param = true;
                    out.push_str("{}");
                }
                '}' => in_param = false,
                _ if in_param => {}
                _ => out.push(c),
            }
        }
        out.trim_end_matches('/').to_string()
    }

    let server = std::fs::read_to_string(repo_root().join("src/api/server.rs"))
        .expect("read src/api/server.rs");
    let doc = std::fs::read_to_string(repo_root().join("docs/ARCHITECTURE.md"))
        .expect("read docs/ARCHITECTURE.md");

    // Every path REGISTERED as a route, which is not the same as every
    // "/api/..." literal in the file — `unknown_route` matches `"/api/"` as a
    // prefix to decide whether a 404 is a person or a client, and its test
    // names paths that are deliberately not served. Both read as undocumented
    // routes to a scan for bare literals, so the extraction keys on the
    // registration itself. Whitespace is skipped because several routes are
    // registered across two lines.
    let mut routes: BTreeSet<String> = BTreeSet::new();
    for (i, _) in server.match_indices(".route(") {
        let rest = &server[i + ".route(".len()..];
        let Some(open) = rest.find('"') else { continue };
        // Only a literal that follows the paren directly, so `.route(VAR, ..)`
        // cannot pick up a string from the handler expression after it.
        if rest[..open].trim().is_empty() {
            if let Some(end) = rest[open + 1..].find('"') {
                let path = &rest[open + 1..open + 1 + end];
                if path.starts_with("/api/") {
                    routes.insert(normalise(path));
                }
            }
        }
    }
    assert!(
        routes.len() > 40,
        "only found {} routes — the extraction broke, not the docs",
        routes.len()
    );

    // Every "/api/..." mention anywhere in the doc, normalised the same way.
    let mut documented: BTreeSet<String> = BTreeSet::new();
    for (i, _) in doc.match_indices("/api/") {
        let rest = &doc[i..];
        let end = rest
            .find(|c: char| !(c.is_ascii_alphanumeric() || "/_{}-".contains(c)))
            .unwrap_or(rest.len());
        documented.insert(normalise(&rest[..end]));
    }

    let missing: Vec<&String> = routes.difference(&documented).collect();
    assert!(
        missing.is_empty(),
        "these routes are served but absent from docs/ARCHITECTURE.md, which \
         presents itself as the canonical HTTP API reference — a reader \
         concludes they do not exist: {missing:#?}"
    );
}

/// Every translation key the frontend asks for must exist in `en.json`.
///
/// The two i18n checks above compare locales against each other and catch a
/// key that is missing from *some* languages. Neither notices a key that is
/// missing from **all** of them, which is what a typo produces — and the
/// failure mode is visible to exactly the people this project is aimed at: a
/// missing key renders as its own raw name, so a non-technical user reads
/// `dashboard.peer_cont` where a sentence should be.
///
/// Three shapes are deliberately not treated as references:
///
/// - **Concatenation.** `I18n.t('reference.tier_' + m.tier)` builds the key at
///   runtime; the literal is a prefix, not a key. Detected by the `+` that
///   follows the closing quote.
/// - **`i18n.js` itself**, whose header comment documents usage as
///   `I18n.t('key')`. Scanning the implementation for calls finds its own
///   documentation.
/// - **Plurals.** `I18n.t('dashboard.peer_count', {count: n})` resolves through
///   `pluralKey` to `_one` / `_other`, so the bare name is correct precisely
///   when the two suffixed forms exist.
#[test]
fn every_translation_key_the_frontend_uses_exists() {
    let root = repo_root();
    let en: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("frontend/i18n/en.json")).expect("read en.json"),
    )
    .expect("en.json is not JSON");
    let defined: BTreeSet<String> = en
        .as_object()
        .expect("en.json is not an object")
        .keys()
        .cloned()
        .collect();

    /// Pull `I18n.t('...')` keys, skipping any whose literal is concatenated.
    fn js_keys(src: &str, out: &mut BTreeSet<String>) {
        const CALL: &str = "I18n.t(";
        let mut rest = src;
        while let Some(i) = rest.find(CALL) {
            rest = &rest[i + CALL.len()..];
            let after_ws = rest.trim_start();
            let Some(quote) = after_ws.chars().next() else {
                continue;
            };
            if quote != '\'' && quote != '"' {
                continue; // a variable, nothing literal to check
            }
            let body = &after_ws[1..];
            let Some(end) = body.find(quote) else {
                continue;
            };
            let key = &body[..end];
            // A `+` after the closing quote means the literal is a prefix.
            if body[end + 1..].trim_start().starts_with('+') {
                continue;
            }
            out.insert(key.to_string());
        }
    }

    /// Pull `data-i18n`, `data-i18n-placeholder`, `data-i18n-title`, … values.
    fn html_keys(src: &str, out: &mut BTreeSet<String>) {
        let mut rest = src;
        while let Some(i) = rest.find("data-i18n") {
            rest = &rest[i + "data-i18n".len()..];
            let Some(eq) = rest.find('=') else { break };
            // Only an attribute-name suffix may sit between; a space means this
            // was a different attribute entirely.
            if rest[..eq].contains(' ') {
                continue;
            }
            let after = rest[eq + 1..].trim_start();
            if !after.starts_with('"') {
                continue;
            }
            let body = &after[1..];
            let Some(end) = body.find('"') else { continue };
            out.insert(body[..end].to_string());
        }
    }

    let mut used: BTreeSet<String> = BTreeSet::new();
    let mut walk = |dir: &str, ext: &str, f: &dyn Fn(&str, &mut BTreeSet<String>)| {
        let mut stack = vec![root.join(dir)];
        while let Some(d) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&d) else {
                continue;
            };
            for e in entries.flatten() {
                let p = e.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|s| s.to_str()) == Some(ext) {
                    // The implementation's own usage comment is not a call.
                    if p.file_name().and_then(|s| s.to_str()) == Some("i18n.js") {
                        continue;
                    }
                    if let Ok(t) = std::fs::read_to_string(&p) {
                        f(&t, &mut used);
                    }
                }
            }
        }
    };
    walk("frontend/js", "js", &js_keys);
    walk("frontend", "html", &html_keys);

    assert!(
        used.len() > 500,
        "only found {} referenced keys — the scan broke, not the translations",
        used.len()
    );

    let missing: Vec<&String> = used
        .iter()
        .filter(|k| {
            !defined.contains(*k)
                // Plural form: correct when both suffixed variants exist.
                && !(defined.contains(&format!("{k}_one")) && defined.contains(&format!("{k}_other")))
        })
        .collect();
    assert!(
        missing.is_empty(),
        "the frontend asks for translation keys that do not exist in en.json, so these \
         render to the user as their own raw names: {missing:#?}"
    );
}

/// Every asset name the updater can ask for must be one the release actually
/// publishes.
///
/// `update.rs` builds the name it downloads from the host's platform, the build
/// variant, and — for a processor without AVX2 — a `-baseline` redirect.
/// `.github/workflows/release.yml` declares the names it uploads in a matrix.
/// These are two hand-maintained lists that must agree exactly, and **nothing
/// compared them**: `update.rs` has unit tests, but they assert its output
/// against constants written in the same file, so a rename in the workflow
/// leaves them green.
///
/// The consequence is silent and permanent. A node asks for an asset that does
/// not exist, finds nothing, and stops updating — with no error a user would
/// see, because "no matching asset" is indistinguishable from "no new version".
/// The `-baseline` names are the sharp end: they exist for pre-2013 processors,
/// which are precisely the machines nobody is testing on.
///
/// Same shape as `compute_cap_matches_release_workflow` — a constant in the
/// source pinned against the workflow that has to honour it.
#[test]
fn every_asset_the_updater_can_request_is_published_by_the_release_workflow() {
    let workflow = std::fs::read_to_string(repo_root().join(".github/workflows/release.yml"))
        .expect("read release.yml");

    // `bare_asset: swarmllm-linux-x86_64` — the un-archived binary the updater
    // downloads. (The `.tar.gz` / `.zip` archives are for humans.)
    let published: BTreeSet<String> = workflow
        .lines()
        .filter_map(|l| l.trim().strip_prefix("bare_asset:"))
        .map(|v| v.trim().to_string())
        .collect();
    assert!(
        published.len() >= 5,
        "found only {} bare_asset entries in release.yml — the parse broke, not the workflow",
        published.len()
    );

    // Every (os, arch, variant) combination `update.rs` can resolve to, and for
    // x86-64 the `-baseline` sibling a pre-AVX2 host is redirected to.
    let mut wanted: BTreeSet<String> = BTreeSet::new();
    for (os, arch, variant, windows) in [
        ("linux", "x86_64", "", false),
        ("linux", "x86_64", "-cuda", false),
        ("macos", "aarch64", "", false),
        ("windows", "x86_64", "", true),
        ("windows", "x86_64", "-gpu", true),
    ] {
        let ext = if windows { ".exe" } else { "" };
        let name = format!("swarmllm-{os}-{arch}{variant}{ext}");
        // Only the plain x86-64 builds have a baseline sibling: a GPU build
        // already requires far newer hardware, and macOS aarch64 has AVX2's
        // equivalent in its baseline.
        if arch == "x86_64" && variant.is_empty() {
            wanted.insert(match name.rsplit_once('.') {
                Some((stem, e)) => format!("{stem}-baseline.{e}"),
                None => format!("{name}-baseline"),
            });
        }
        wanted.insert(name);
    }

    let missing: Vec<&String> = wanted.difference(&published).collect();
    assert!(
        missing.is_empty(),
        "the updater can ask for these assets and the release workflow does not publish them, \
         so a node on that platform would silently stop updating: {missing:#?}\n\
         published: {published:#?}"
    );
}

/// Logical statements, each paired with the 1-indexed line it starts on, with
/// interior newlines collapsed to spaces.
///
/// Line-by-line matching cannot see a method chain that rustfmt has wrapped,
/// and wrapping is the ordinary shape for the writes the two guards below
/// forbid: `self.shared_state.metrics.node_stats.requests_served_atomic
/// .fetch_add(1, Ordering::Relaxed);` is 99 characters at two levels of
/// indentation, so one more nesting level splits it across four lines and the
/// field name no longer shares a line with `fetch_add`. Both guards were
/// verified blind to exactly that, which mattered because the architecture
/// notes say each of them "fails the build" on a stray write.
///
/// Full-line comments are dropped before joining, so a comment inside a
/// statement's span cannot inject a false match.
/// Collapse a statement's whitespace, closing the gap a wrapped method chain
/// leaves before its `.` — `s.metrics\n    .node_stats` must read back as
/// `s.metrics.node_stats`, or a guard looking for `field.remove(` still misses
/// it and the whole exercise is pointless.
fn join_statement(raw: &str) -> String {
    raw.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace(" .", ".")
}

fn statements(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut start = 0usize;
    let mut started = false;
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.starts_with("//") {
            continue;
        }
        for ch in line.chars() {
            if !started && !ch.is_whitespace() {
                start = i + 1;
                started = true;
            }
            cur.push(ch);
            if matches!(ch, ';' | '{' | '}') {
                if started {
                    out.push((start, join_statement(&cur)));
                }
                cur.clear();
                started = false;
            }
        }
        cur.push(' ');
    }
    if started {
        out.push((start, join_statement(&cur)));
    }
    out
}

/// The scanner both "exactly one place" guards depend on must see a wrapped
/// chain, or those guards assert far less than they claim.
#[test]
fn the_statement_scanner_sees_a_chain_rustfmt_has_wrapped() {
    let src = "fn f() {\n    s.metrics\n        .node_stats\n        .requests_served_atomic\n        .fetch_add(1, Ordering::Relaxed);\n}\n";
    let sts = statements(src);
    let hit = sts
        .iter()
        .find(|(_, t)| t.contains("requests_served_atomic") && t.contains("fetch_add"))
        .expect("a wrapped chain must arrive as one statement");
    assert_eq!(
        hit.0, 2,
        "reported line should be where the statement starts"
    );

    // The same for a wrapped `.remove(` on a guarded map.
    let src2 = "fn f() {\n    s.active_traces\n        .remove(&id);\n}\n";
    assert!(statements(src2)
        .iter()
        .any(|(_, t)| t.contains("active_traces.remove(")));

    // A full-line comment inside the span must not inject a match.
    let src3 = "fn f() {\n    let a = 1;\n    // s.active_traces.remove(&id);\n    let b = 2;\n}\n";
    assert!(!statements(src3)
        .iter()
        .any(|(_, t)| t.contains("active_traces.remove(")));
}

/// Serving must be counted and paid in exactly one place, reached only from the
/// two inbound paths that do work for a peer.
///
/// This is pinned rather than documented because the failure it prevents is
/// silent and was live for several releases. `forwards_served_atomic` and the
/// credit earn used to be bumped by a helper only the multi-segment path
/// called, while the remote-generate fast path — how a machine holding a whole
/// model answers a peer, i.e. the common case — recorded nothing. A node doing
/// real work reported `requests_served = 0` and was paid nothing while the
/// requester was still debited. The mirror-image mistake was equally live: the
/// router's own completion hook and the local-segment path bumped the same
/// counters for work the node did for *itself*, so a user whose only traffic
/// was their own chat was told they had served the swarm.
///
/// So: the two `*_served_atomic` counters and `pending_credit_earn` may be
/// written only inside `record_peer_serve`. Everywhere else reads them.
/// A model's tensor geometry is read through one accessor, which learns it on
/// a miss.
///
/// `SharedState.gguf_meta` was filled at startup, by the admin HuggingFace
/// shard download and by local manifest generation — and by nothing at all on
/// the path a model takes when its shards arrive from the swarm while the
/// daemon is running. Its one reader is the scheduler's per-peer capacity
/// bound, which charges a peer for what the prompt's KV cache will cost per
/// layer and treats an absent geometry as "charge nothing". So the bound was
/// silently inert on the one case it was written for: a model being
/// distributed for the first time. A 6 GB card was handed 28 layers of an
/// 11.2 k-token prompt, four attempts in a row (gotcha #451).
///
/// `SharedState::gguf_meta_for` is now the only read, and reads the local
/// shard header on a miss. `insert` / `remove` / `contains_key` remain open:
/// those are the writers and their idempotence guards, and the accessor is
/// built out of one of them.
#[test]
fn the_model_geometry_is_read_through_one_accessor() {
    let root = repo_root();
    let mut stack = vec![root.join("src")];
    let mut offenders: Vec<String> = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            // The accessor itself lives here, and is the only legitimate read.
            if rel == "src/daemon/state/mod.rs" {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            // Statements, not lines: a chain rustfmt has wrapped across
            // `shared_state` / `.gguf_meta` / `.get(..)` is the ordinary shape
            // here, and a line scan cannot see it.
            for (line_no, l) in statements(&text) {
                if l.contains("gguf_meta.get(") {
                    offenders.push(format!("{rel}:{line_no}: {}", l.trim()));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "read the model's geometry through SharedState::gguf_meta_for, which \
         learns it from the local shard header on a miss — a bare `gguf_meta.get` \
         reports \"unknown\" for every model acquired since the last restart:\n{}",
        offenders.join("\n")
    );
}

/// The scan above must see the wrapped form, or it asserts nothing.
#[test]
fn the_geometry_accessor_guard_catches_a_wrapped_read() {
    let src =
        "fn f() {\n    let m = self.shared_state\n        .gguf_meta\n        .get(&id);\n}\n";
    assert!(
        statements(src)
            .iter()
            .any(|(_, l)| l.contains("gguf_meta.get(")),
        "a chain rustfmt has wrapped must still be caught"
    );
}

#[test]
fn serving_is_counted_and_paid_in_exactly_one_place() {
    let root = repo_root();
    let mut sources: Vec<(PathBuf, String)> = Vec::new();
    let mut stack = vec![root.join("src")];
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

    // The single writer. Everything else must only read.
    let writer = "src/daemon/state/relay.rs";
    let guarded = [
        "requests_served_atomic",
        "forwards_served_atomic",
        "pending_credit_earn",
    ];

    let mut offenders: Vec<String> = Vec::new();
    for (path, text) in &sources {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if rel == writer {
            continue;
        }
        // The ledger owns the flush — it drains `pending_credit_earn` into the
        // persisted balance and restores it if that write fails, which is the
        // other half of the same mechanism, not a competing writer.
        let is_ledger = rel == "src/credit/ledger.rs";
        // Construction of the zeroed initial state is not a write.
        let is_ctor = rel == "src/daemon/state/mod.rs" || rel == "src/daemon/state/credits.rs";
        // Statements, not lines: rustfmt wraps these chains, and a wrapped
        // write was invisible to this guard.
        for (line_no, l) in statements(text) {
            for name in guarded {
                if !l.contains(name) {
                    continue;
                }
                let mutates = l.contains("fetch_add")
                    || l.contains("fetch_sub")
                    || l.contains(".store(")
                    || l.contains(".swap(")
                    || l.contains("compare_exchange");
                if !mutates {
                    continue;
                }
                if (is_ledger || is_ctor) && name == "pending_credit_earn" {
                    continue;
                }
                offenders.push(format!("{rel}:{line_no}: {l}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "serving counters/credits are written outside `SharedState::record_peer_serve`.\n\
         Work done FOR A PEER must go through that helper (it counts and bills in one place);\n\
         work the node does for ITSELF must not be counted as serving at all.\n{}",
        offenders.join("\n")
    );

    // And the helper must still be reachable from both serving paths, or the
    // consolidation has quietly lost one of them again.
    for expected in [
        "src/daemon/dispatch/layer_forward.rs",
        "src/daemon/dispatch/remote_generate.rs",
    ] {
        let text = std::fs::read_to_string(root.join(expected)).expect("readable serving path");
        assert!(
            text.contains("record_peer_serve"),
            "{expected} serves peers but no longer records it — that work would be \
             invisible and unpaid"
        );
    }
}

/// A setting the user can change must be READ from the live config, not the
/// boot-time snapshot — otherwise it saves, reports success, shows its new
/// value, and changes nothing until restart.
///
/// Verified on the released v0.3.87: switching contribution to Maximum left the
/// storage target at 6250 MB. The VRAM, disk, bandwidth, shard-size, batch-
/// timeout and auto-manage storage caps all had the same fault, and
/// `OperationalParams` — whose own doc comment says "can be changed without
/// restart" — carried five fields nothing consumed.
///
/// Whole SECTIONS are checked rather than field names, because the frozen value
/// is just as often reached through a method: `config.resources
/// .shard_upload_mbps(..)` never mentions `max_bandwidth_mbps`, and that is
/// exactly how that one survived a first pass.
#[test]
fn user_settable_config_is_read_live_not_from_the_boot_snapshot() {
    let root = repo_root();

    // Sections the Settings panel can change.
    // Whole sections where every field is a resource cap the user can move...
    let mutable = [
        ".config.resources.",
        // Only the auto-manage fields the Settings panel actually exposes.
        // This was the whole `.config.auto_manage.` section, which contradicted
        // the principle stated for `inference` just below: a field the panel
        // does not offer is config-file/CLI only, and reading its boot value is
        // CORRECT. Thirteen of the seventeen fields in that section are in that
        // category — `max_concurrent_downloads` among them, which sizes a
        // semaphore once at construction and appears nowhere in the dashboard.
        // The contradiction was invisible while the scan was line-based,
        // because the one read it would have flagged is a wrapped chain.
        ".config.auto_manage.enabled",
        ".config.auto_manage.max_shards",
        ".config.auto_manage.max_storage_mb",
        ".config.auto_manage.prune_enabled",
        ".config.node.contribution",
        ".config.model.shard_size_mb",
        // ...and the three `inference` fields the Settings panel exposes. The
        // rest of that section (gpu_layers, shard_range, encrypted_pipeline) is
        // config-file/CLI only, so reading the boot value there is CORRECT —
        // `shard_range` in particular is meant to be fixed for the process.
        ".config.inference.max_concurrent_requests",
        ".config.inference.max_batch_size",
        ".config.inference.batch_timeout_ms",
    ];

    // Startup-only readers: each builds something that cannot be rebuilt live
    // (the swarm's connection limits, the dispatch semaphores), reads or writes
    // the persisted file, or initialises the live config itself.
    let allowed = [
        "src/network/manager/mod.rs",
        "src/daemon/dispatch/mod.rs",
        "src/daemon/state/mod.rs",
        "src/api/admin.rs",
        "src/daemon/mod.rs",
    ];

    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            // Test code builds Config values directly; that is not a runtime read.
            if allowed.contains(&rel.as_str())
                || rel.starts_with("src/config/")
                || rel.ends_with("/tests.rs")
                || rel.contains("/tests/")
            {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            // Everything from the first `#[cfg(test)]` on is test code.
            let cutoff = text
                .find("#[cfg(test)]")
                .map(|i| text[..i].lines().count() + 1)
                .unwrap_or(usize::MAX);
            // Statements, not lines. Every pattern here is a contiguous dotted
            // path, and rustfmt breaks a long chain across lines — so
            // `shared\n    .config\n    .resources\n    .max_gpu_vram_mb`
            // matched nothing at all, which is this guard's own defect wearing
            // the shape of the bug it exists to catch (gotcha #281, three
            // recurrences).
            for (line_no, l) in statements(&text) {
                if line_no >= cutoff {
                    continue;
                }
                if mutable.iter().any(|m| l.contains(m)) {
                    offenders.push(format!("{rel}:{line_no}: {l}"));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these read a user-settable value from the boot-time config.\n\
         Use `shared_state.cfg()` so a change in Settings applies without a restart.\n{}",
        offenders.join("\n")
    );
}

/// The three per-request maps are cleared together, in one place.
///
/// `active_pipelines`, `active_traces` and `request_holder_blacklist` are keyed
/// by request id and share a lifetime. Five call sites used to remove all three
/// by hand, with the invariant held only by three adjacent lines and a comment
/// asserting it. Dropping one is silent and unbounded — `active_traces` is the
/// oracle behind `model_is_in_use`, so a stranded entry refuses to delete that
/// model for the daemon's lifetime.
/// Every long per-request wait in the pipeline watches the cancel flag.
///
/// `inference::cancel`'s module doc has now been wrong about being complete
/// TWICE — it named two waits when there were three (gotcha #459, the model
/// load), and three when there were five (#468: the peer fast path's
/// first-token wait, up to ten minutes, and the remote vision encode's two).
/// A doc comment cannot enforce this and a helper nobody is obliged to call
/// will eventually not be called, so the rule is checked instead.
///
/// The shape it forbids: `tokio::time::timeout(...)` awaiting a channel in
/// `src/inference/pipeline/`, not wrapped in `unless_cancelled`. Those are the
/// waits that can run for minutes with a client on the other end.
#[test]
fn every_long_pipeline_wait_watches_the_cancel_flag() {
    let root = repo_root();
    let dir = root.join("src/inference/pipeline");
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![dir];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            // Test code sets its own deadlines and has no client to abandon it.
            let src = &text[..text.find("#[cfg(test)]").unwrap_or(text.len())];
            for line_no in unwatched_pipeline_waits(src)
                .into_iter()
                .chain(unwatched_pool_generations(src))
            {
                offenders.push(format!(
                    "{}:{}",
                    p.strip_prefix(&root).unwrap_or(&p).display(),
                    line_no
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "long pipeline wait(s) that do not watch the request's cancel flag:\n  {}\n\
         Wrap the wait in `inference::cancel::unless_cancelled(async {{ Ok(..) }}, \
         self.request.cancel.as_ref())`. Checking the flag BEFORE the wait is not \
         enough — that is what left a ten-minute first-token wait unnoticed \
         (gotcha #468).",
        offenders.join("\n  ")
    );
}

/// The generation half must see the shape it forbids, in the form rustfmt
/// actually produces — the call it was written for is eight lines long.
#[test]
fn the_cancel_watch_guard_catches_an_unwatched_generation() {
    // One line.
    assert_eq!(
        unwatched_pool_generations("let out = pool.generate(&m, r, p, s, id, None, tx).await?;\n"),
        vec![1]
    );
    // Wrapped, which is how it is really written.
    let wrapped = "let out = self\n    .shared_state\n    .model_process_pool\n    .generate(\n                           &model_id,\n        layer_range,\n        prompt,\n    )\n    .await?;\n";
    assert_eq!(unwatched_pool_generations(wrapped), vec![1]);
    // Watched: the fix, and it must not be reported.
    let watched = "let out = crate::inference::cancel::unless_cancelled(\n                           pool.generate(&model_id, layer_range, prompt),\n                           self.request.cancel.as_ref(),\n    )\n    .await?;\n";
    assert!(unwatched_pool_generations(watched).is_empty());
    // The in-process executor blocks rather than awaits.
    assert!(unwatched_pool_generations(
        "let (c, g) = without_starving_the_runtime(|| executor.generate(&p, &s))?;\n"
    )
    .is_empty());
}

/// The guard is only worth having if it can see the shape it forbids.
#[test]
fn the_cancel_watch_guard_catches_an_unwatched_wait() {
    let planted = r#"
        let maybe = tokio::time::timeout(timeout_dur, stream_rx.recv()).await;
    "#;
    assert_eq!(
        unwatched_pipeline_waits(planted).len(),
        1,
        "a bare timeout on a channel must be caught"
    );
    let watched = r#"
            let maybe = match crate::inference::cancel::unless_cancelled(
                async { Ok(tokio::time::timeout(timeout_dur, stream_rx.recv()).await) },
                self.request.cancel.as_ref(),
            )
    "#;
    assert!(
        unwatched_pipeline_waits(watched).is_empty(),
        "the wrapped form is the sanctioned one"
    );
}

/// Lines awaiting a channel under `tokio::time::timeout` without
/// `unless_cancelled` on the same line.
/// A whole-reply generation awaited in the pipeline without watching the flag.
///
/// `unwatched_pipeline_waits` above matches `tokio::time::timeout(` — a wait
/// with a deadline on it. This one has none and is the LONGEST wait in a
/// request's life: `ModelProcessPool::generate` returns when the worker has
/// finished the whole reply, which on a processor is minutes. It does not read
/// the cancel flag itself, and a closed token channel does not end it
/// (`let _ = tx.send(..)`), so an unwatched `.await` on it keeps a worker
/// generating for a client that has gone — the exact failure gotchas #441/#445
/// were about.
///
/// Added 2026-09-12, when `pipeline::local_generate` introduced the first such
/// call and the timeout-shaped guard could not see it. Statement-joined,
/// because rustfmt splits a seven-argument call across eight lines.
fn unwatched_pool_generations(src: &str) -> Vec<usize> {
    statements(src)
        .into_iter()
        .filter(|(_, stmt)| {
            stmt.contains(".generate(")
                && stmt.contains(".await")
                && !stmt.contains("unless_cancelled")
                // The in-process executor's synchronous generate, which blocks
                // rather than awaits and carries its own cancellation.
                && !stmt.contains("without_starving_the_runtime")
        })
        .map(|(line, _)| line)
        .collect()
}

fn unwatched_pipeline_waits(src: &str) -> Vec<usize> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim();
            !t.starts_with("//")
                && t.contains("tokio::time::timeout(")
                && !t.contains("unless_cancelled")
                // `Ok(tokio::time::timeout(..))` is the body of the async block
                // handed to `unless_cancelled` on the line above it.
                && !t.starts_with("async { Ok(tokio::time::timeout(")
        })
        .map(|(i, _)| i + 1)
        .collect()
}

/// A worker leaves the pool in exactly two places, and both release what it
/// held.
///
/// Gotcha #461 fixed three sites that removed a dead worker and released
/// nothing — and there were NINE. The other six are the paths a worker actually
/// dies on: a failed IPC send, a closed reader channel on forward /
/// batch-forward / generate, and `classify_worker_error`'s fatal arm, which is
/// the CUDA-OOM path. The health-tick reap cannot cover them, because it scans
/// `workers` and they have already removed the entry — so the memory charge
/// leaked for the daemon's lifetime and refused every later model (#467).
#[test]
fn a_worker_only_leaves_the_pool_where_its_memory_is_released() {
    let root = repo_root();
    let path = root.join("src/inference/process_pool.rs");
    let text = std::fs::read_to_string(&path).expect("read process_pool.rs");
    let src = &text[..text.find("#[cfg(test)]").unwrap_or(text.len())];

    let offenders: Vec<usize> = worker_removal_lines(src);
    assert!(
        offenders.is_empty(),
        "`self.workers.remove(...)` outside the two sanctioned paths, at line(s) {offenders:?}.\n         Removing a worker without releasing its RAM/VRAM charge leaves a phantom \
         reservation that refuses every later model until the daemon restarts \
         (gotchas #461, #467). Go through `evict_this_worker` / \
         `evict_worker_where`, or — if you must drain first, as `unload_model` \
         does — end in `after_worker_gone`."
    );
}

/// The guard above is only worth having if it can see the shape it forbids.
#[test]
fn the_worker_removal_guard_catches_a_bare_remove() {
    let planted = r#"
        fn some_new_path(&self) {
            self.workers.remove(&model_id);
        }
    "#;
    assert_eq!(
        worker_removal_lines(planted).len(),
        1,
        "a bare `workers.remove` must be caught"
    );
    // And the two sanctioned forms are not offenders.
    let ok = r#"
        let Some((_, handle)) = self.workers.remove_if(model_id, |_, cur| pred(cur)) else {
            return false;
        };
        // unload_model drains first, then ends in after_worker_gone.
        if let Some((_, handle)) = self.workers.remove(model_id) {
            self.after_worker_gone(model_id, freed_gpu_memory, "stopped");
        }
    "#;
    assert!(
        worker_removal_lines(ok).is_empty(),
        "remove_if and the drain-then-release form are the sanctioned paths"
    );
}

/// Lines doing a bare `self.workers.remove(` that are not the `unload_model`
/// drain form. `remove_if` is the helper's own call and is always fine.
fn worker_removal_lines(src: &str) -> Vec<usize> {
    src.lines()
        .enumerate()
        .filter(|(_, l)| {
            let t = l.trim();
            !t.starts_with("//")
                && t.contains("self.workers.remove(")
                // `unload_model` binds the handle so it can drain before
                // killing; it ends in `after_worker_gone` like everything else.
                && !t.contains("if let Some((_, handle)) = self.workers.remove(")
        })
        .map(|(i, _)| i + 1)
        .collect()
}

#[test]
fn per_request_state_is_released_in_one_place() {
    let root = repo_root();
    let guarded = [
        "active_pipelines",
        "active_traces",
        "request_holder_blacklist",
        "peer_vram_commitments",
        "salvaged_replies",
        // Released with `.release(`, not `.remove(` — it owns its own store
        // rather than being a bare map. Both spellings are matched below, so
        // adding a name here does not silently produce a guard that cannot
        // fire (gotcha #413).
        "retained_activations",
    ];
    // The ways a caller clears one of these directly. A name is only guarded
    // as well as the spellings listed here.
    let operations = [".remove(", ".release("];
    let allowed = [
        // Owns the helper.
        "src/daemon/state/relay.rs",
        // `TraceGuard` registers a trace for the split fast path, which bypasses
        // the router and never creates a pipeline or a blacklist entry. Its
        // insert and remove are a balanced pair on one map; calling the helper
        // would have it clear state it never owned.
        "src/api/openai/streaming.rs",
    ];

    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            if allowed.contains(&rel.as_str()) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            // Everything from the first `#[cfg(test)]` on is test code.
            let cutoff = text
                .find("#[cfg(test)]")
                .map(|i| text[..i].lines().count() + 1)
                .unwrap_or(usize::MAX);
            // Statements, not lines — see `statements`.
            for (line_no, l) in statements(&text) {
                if line_no >= cutoff {
                    continue;
                }
                for name in guarded {
                    for op in operations {
                        if l.contains(&format!("{name}{op}")) {
                            offenders.push(format!("{rel}:{line_no}: {l}"));
                        }
                    }
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "per-request state is removed outside `SharedState::release_request_state`.\n\
         A path that owns a pipeline must clear them together — clearing a \
         subset strands the rest.\n{}",
        offenders.join("\n")
    );
}

/// Every `DIAG:` line the diagnostics guide tells you to grep for must exist.
///
/// `docs/DIAGNOSTICS.md` is a list of greppable markers; its whole value is that
/// searching for one finds something. 30 of 147 had been renamed or deleted out
/// of the code (measured 2026-08-09), so one lookup in five sent the reader
/// hunting for a string that was not there — and silently, because a failed
/// grep looks exactly like the thing not happening (gotcha #228).
///
/// A `start/done` pair in the doc means two lines in the source; both are
/// checked.
#[test]
fn every_documented_diag_line_exists_in_the_source() {
    let root = repo_root();
    let doc = std::fs::read_to_string(root.join("docs/DIAGNOSTICS.md")).expect("read DIAGNOSTICS");

    let mut src = String::new();
    let mut stack = vec![root.join("src")];
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
                    src.push_str(&t);
                }
            }
        }
    }

    // Pull `DIAG: ...` out of backticks.
    let mut claims: Vec<String> = Vec::new();
    for (i, _) in doc.match_indices("`DIAG: ") {
        let rest = &doc[i + 1..];
        if let Some(end) = rest.find('`') {
            claims.push(rest[..end].to_string());
        }
    }
    claims.sort();
    claims.dedup();
    assert!(
        claims.len() > 50,
        "parsed only {} DIAG markers — the doc format moved and this checks nothing",
        claims.len()
    );

    let mut missing: Vec<String> = Vec::new();
    for claim in &claims {
        // Drop the prose the doc appends after an em dash.
        let core = claim.split('—').next().unwrap_or(claim).trim();
        // `foo start/done` documents two source lines.
        let probes: Vec<String> = match core.rsplit_once(" start/done") {
            Some((head, _)) => vec![format!("{head} start"), format!("{head} done")],
            None => vec![core.to_string()],
        };
        for probe in probes {
            if !probe.is_empty() && !src.contains(&probe) {
                missing.push(probe);
            }
        }
    }

    assert!(
        missing.is_empty(),
        "docs/DIAGNOSTICS.md points at {} log line(s) that no longer exist.\n\
         A guide whose greps come back empty is worse than no guide — rename or \
         remove the entry when the line changes.\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

/// A frontend component that uses `U.` must bind `U` first.
///
/// `U` is the conventional local alias for `App.utils`, established by the
/// boilerplate every component copies. Forgetting the binding is not caught by
/// anything at load time — the file parses, registers, and throws
/// `ReferenceError: U is not defined` the first time that code path renders.
/// The R111 swarm tab shipped that way and only surfaced when a user opened the
/// Capacity Plan view.
#[test]
fn frontend_components_bind_the_utils_alias_they_use() {
    let dir = repo_root().join("frontend/js/components");
    let rd = std::fs::read_dir(&dir).expect("components dir");

    let mut offenders: Vec<String> = Vec::new();
    let mut checked = 0;
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        if !p.extension().is_some_and(|x| x == "js") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        checked += 1;

        let declares = text.contains("var U = App.utils") || text.contains("const U = App.utils");
        if declares {
            continue;
        }
        // `U.` preceded by an identifier char is something else (e.g. `App.U.`).
        let uses = text.match_indices("U.").any(|(i, _)| {
            let prev = text[..i].chars().next_back();
            !matches!(prev, Some(c) if c.is_alphanumeric() || c == '_' || c == '.')
        });
        if uses {
            offenders.push(
                p.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
            );
        }
    }

    assert!(
        checked >= 15,
        "only {checked} components scanned — the directory moved and this checks nothing"
    );
    assert!(
        offenders.is_empty(),
        "these components use `U.` without binding it, and will throw \
         `ReferenceError: U is not defined` the first time that code renders:\n  {}\n\
         Add `var U = App.utils;` inside the IIFE, as the sibling components do.",
        offenders.join("\n  ")
    );
}

/// The frontend payload has a budget, and it is measured rather than hoped for.
///
/// `CLAUDE.md` carried "Total frontend size target: < 200KB" for a long time
/// while the real figure was ~1044 KB — 5.6x out, and nothing ever compared the
/// two. A number nobody checks stops being a budget and becomes decoration.
///
/// This is a regression budget with headroom, not a goal: it should fail when
/// something large is added (a vendored library, an inlined asset), not on
/// ordinary growth. Raise it deliberately, with the new figure in `CLAUDE.md`.
#[test]
fn frontend_payload_stays_within_budget() {
    // What a browser fetches on first paint. Locales are excluded: exactly one
    // is loaded at a time, so the other 20 are not payload.
    const PAYLOAD_BUDGET_KB: u64 = 1400;
    const LOCALE_BUDGET_KB: u64 = 120;

    let root = repo_root().join("frontend");
    let mut payload: u64 = 0;
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                // i18n is measured separately — one file is fetched, not all 21.
                if p.file_name().is_some_and(|n| n == "i18n") {
                    continue;
                }
                stack.push(p);
                continue;
            }
            let is_payload = p
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| matches!(x, "js" | "css" | "html"));
            if is_payload {
                payload += std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
            }
        }
    }
    let payload_kb = payload / 1024;
    assert!(
        payload_kb > 200,
        "measured only {payload_kb} KB — the layout moved and this checks nothing"
    );
    assert!(
        payload_kb <= PAYLOAD_BUDGET_KB,
        "frontend payload is {payload_kb} KB, over the {PAYLOAD_BUDGET_KB} KB budget.\n\
         Something large was added — check for a vendored library or an inlined asset.\n\
         If the growth is intended, raise the budget here AND update the figure in CLAUDE.md."
    );

    let en = root.join("i18n/en.json");
    let locale_kb = std::fs::metadata(&en).map(|m| m.len()).unwrap_or(0) / 1024;
    assert!(
        locale_kb <= LOCALE_BUDGET_KB,
        "en.json is {locale_kb} KB, over the {LOCALE_BUDGET_KB} KB budget — \
         one locale is fetched per page load, so this is payload too."
    );
}

/// Every setting a user can put in `config.toml` appears in the reference.
///
/// 16 were missing when this was first measured (2026-08-09) — reachability
/// controls, the idle-unload timer, the cross-pool sharing flags, anchor mode.
/// A settings reference is only useful if a reader can conclude that a setting
/// they cannot find does not exist, and that conclusion was wrong for one option
/// in nine.
///
/// Only leaf scalars are checked: section structs are documented as headings,
/// and `OperationalParams` is a derived view rather than something anyone writes
/// in a file.
#[test]
fn every_config_setting_is_documented() {
    let root = repo_root();
    let doc = std::fs::read_to_string(root.join("docs/book/src/configuration/reference.md"))
        .expect("read config reference");

    // Structs that are not written by hand into config.toml.
    let skip_structs = ["OperationalParams", "CustomProvider"];

    let mut missing: Vec<String> = Vec::new();
    let mut checked = 0;
    let cfg_dir = root.join("src/config");
    let Ok(rd) = std::fs::read_dir(&cfg_dir) else {
        panic!("src/config unreadable");
    };
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        if !p.extension().is_some_and(|x| x == "rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&p) else {
            continue;
        };
        let mut current = String::new();
        let mut in_tests = false;
        for line in text.lines() {
            let l = line.trim();
            if l.starts_with("#[cfg(test)]") {
                in_tests = true;
            }
            if in_tests {
                continue;
            }
            if let Some(rest) = l.strip_prefix("pub struct ") {
                current = rest
                    .split(|c: char| !c.is_alphanumeric() && c != '_')
                    .next()
                    .unwrap_or("")
                    .to_string();
                continue;
            }
            let Some(rest) = l.strip_prefix("pub ") else {
                continue;
            };
            let Some((name, ty)) = rest.split_once(':') else {
                continue;
            };
            let name = name.trim();
            let ty = ty.trim().trim_end_matches(',');
            if current.is_empty() || skip_structs.contains(&current.as_str()) {
                continue;
            }
            // Leaf scalars only — a struct-typed field is a section heading.
            let core = ty
                .replace("Option<", "")
                .replace("Vec<", "")
                .replace('>', "");
            let is_scalar = matches!(
                core.as_str(),
                "bool"
                    | "String"
                    | "PathBuf"
                    | "usize"
                    | "u8"
                    | "u16"
                    | "u32"
                    | "u64"
                    | "i32"
                    | "i64"
                    | "f32"
                    | "f64"
            );
            if !is_scalar {
                continue;
            }
            checked += 1;
            if !doc.contains(&format!("`{name}`")) {
                missing.push(format!("{current}.{name}"));
            }
        }
    }

    assert!(
        checked > 100,
        "only {checked} settings scanned — the config layout moved and this checks nothing"
    );
    missing.sort();
    missing.dedup();
    assert!(
        missing.is_empty(),
        "{} config setting(s) are not in docs/book/src/configuration/reference.md:\n  {}\n\
         A reference is only useful if a setting a reader cannot find does not exist.",
        missing.len(),
        missing.join("\n  ")
    );
}

/// The async Python client exposes everything the sync one does.
///
/// Two of them drifted: `generate-code` and `join` — the `swarmpool://` invite
/// flow — shipped in the sync client and never reached the async one, so an
/// async user simply could not use invite codes (logged as deferred in the sweep
/// log, found still open 2026-08-09).
///
/// Only parity is checked. Which endpoints the SDK wraps at all is a product
/// decision; the two clients disagreeing is always a mistake.
#[test]
fn the_two_python_clients_wrap_the_same_endpoints() {
    let root = repo_root().join("python/swarmllm_client");
    let Ok(rd) = std::fs::read_dir(&root) else {
        panic!("python client dir unreadable");
    };

    let mut sync = String::new();
    let mut asyncc = String::new();
    for e in rd.filter_map(|e| e.ok()) {
        let p = e.path();
        if !p.extension().is_some_and(|x| x == "py") {
            continue;
        }
        let Ok(t) = std::fs::read_to_string(&p) else {
            continue;
        };
        if p.file_name().is_some_and(|n| n == "async_client.py") {
            asyncc.push_str(&t);
        } else {
            sync.push_str(&t);
        }
    }

    // Endpoint path literals, as the clients write them.
    let paths = |txt: &str| -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for (i, _) in txt.match_indices("\"/api/") {
            let rest = &txt[i + 1..];
            if let Some(end) = rest.find('"') {
                out.insert(rest[..end].to_string());
            }
        }
        out
    };
    let s = paths(&sync);
    let a = paths(&asyncc);
    assert!(
        s.len() > 20,
        "parsed only {} endpoints from the sync client — the layout moved",
        s.len()
    );

    let only_sync: Vec<&String> = s.difference(&a).collect();
    let only_async: Vec<&String> = a.difference(&s).collect();
    assert!(
        only_sync.is_empty() && only_async.is_empty(),
        "the Python clients have drifted.\n  only in sync:  {:?}\n  only in async: {:?}\n\
         Whichever gained an endpoint, the other needs it too.",
        only_sync,
        only_async
    );
}

/// Every way of sending a message to a model counts it.
///
/// `requests_made` backs a dashboard tile reading "messages you've sent to AI
/// models". It was incremented by the chat and Anthropic endpoints only, so the
/// Responses API and MCP — both first-class ways in, and MCP is how the node is
/// driven from Claude Code — left it at zero forever.
///
/// A middleware over the generation routes would be the obvious choke point and
/// is wrong: `/mcp` also carries the SSE channel and `tools/list` handshakes,
/// none of which is a message to a model. So the call belongs at each real
/// entry, and this checks they all have it.
#[test]
fn every_model_facing_entry_point_counts_the_request() {
    let root = repo_root();
    let entries = [
        ("src/api/openai/mod.rs", "chat completions"),
        ("src/api/anthropic/mod.rs", "Anthropic messages"),
        ("src/api/openai/responses/mod.rs", "Responses API"),
        ("src/api/mcp/tools.rs", "MCP tools"),
    ];
    let mut missing: Vec<String> = Vec::new();
    for (rel, what) in entries {
        let Ok(text) = std::fs::read_to_string(root.join(rel)) else {
            missing.push(format!("{rel} ({what}) — unreadable"));
            continue;
        };
        if !text.contains("increment_requests_made") {
            missing.push(format!("{rel} ({what})"));
        }
    }
    assert!(
        missing.is_empty(),
        "these send messages to a model without counting them, so the dashboard \
         under-reports for anyone using them:\n  {}",
        missing.join("\n  ")
    );
}

/// What a node REPORTS about updates must be what it DOES.
///
/// `updates.auto_update` is the legacy field. It defaults to `Disabled`, and
/// `UpdateConfig::effective_mode()` deliberately resolves that to `Notify` —
/// because `disabled` was the shipped default rather than anyone's decision,
/// and honouring it literally left nodes on old builds with nothing ever
/// saying so.
///
/// `GET /api/admin/version` read the legacy field directly, so a stock node
/// answered `"channel": "disabled"` while checking on schedule with a
/// populated `last_checked` in the same response (observed on a live node,
/// 2026-08-10). The endpoint built to answer "will this node tell me about a
/// release?" gave the opposite answer, to essentially every node, because
/// `Disabled` is the default.
///
/// The fix is to report `effective_mode()`. This guard exists because the
/// tempting read — the field literally named `auto_update` — is the wrong one,
/// and nothing else would catch a revert: the config-level unit tests pass
/// either way, since the bug was the handler bypassing them.
#[test]
fn update_reporting_uses_the_effective_mode_not_the_legacy_field() {
    let root = repo_root();
    let api = root.join("src/api");
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![api];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let Ok(src) = std::fs::read_to_string(&p) else {
                    continue;
                };
                // Statements, not lines — a wrapped chain hides a dotted path.
                for (line_no, l) in statements(&src) {
                    // Reading the legacy field anywhere under the HTTP surface
                    // means something user-facing is describing update
                    // behaviour from a value that does not determine it.
                    if l.contains("updates.auto_update") || l.contains("AutoUpdateMode::") {
                        offenders.push(format!(
                            "{}:{}: {}",
                            p.strip_prefix(&root).unwrap_or(&p).display(),
                            line_no,
                            l
                        ));
                    }
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the API reports update behaviour from the legacy `auto_update` field. \
         Use `cfg().updates.effective_mode()` — `auto_update` defaults to \
         Disabled, which resolves to Notify, so reading it directly tells \
         almost every user their updates are off while the node is checking:\n{}",
        offenders.join("\n")
    );
}

/// A path that renders a chat template must send the stops that template
/// implies, not the caller's params untouched.
///
/// A model ends its turn with a marker (`<|user|>`, `<|im_end|>`, `<|eot_id|>`)
/// rather than only the tokenizer's EOS id. A path that builds a templated
/// prompt but forwards `sampling_params` unchanged lets the model run to
/// `max_tokens` emitting those markers as visible text and then invent the next
/// turn. The serving node cannot rescue it — it only truncates the stop list it
/// is handed, so a marker the coordinator never sends is one nothing matches.
///
/// `with_template_stops` was written for this and its own comment named the
/// three paths needing it, yet the router/remote path went unwired for
/// releases: `remote_generate` is the fast path taken whenever ONE peer holds
/// the whole model — the normal case for a node that stores nothing itself.
/// Observed on TinyLlama over the network:
/// `'Count from six to ten.\n<|user|> Can you give me a summary…'`.
///
/// Pinned at source level because the leak only shows when the model happens to
/// emit a marker, so a behavioural test passes most runs with the bug present.
#[test]
fn templated_prompts_carry_their_template_stops() {
    let root = repo_root();
    let path = root.join("src/inference/pipeline/remote_generate.rs");
    let src = std::fs::read_to_string(&path).expect("remote_generate.rs must exist");

    assert!(
        src.contains("build_prompt_and_stops"),
        "remote_generate builds a chat-templated prompt, so it must obtain its \
         stop strings from the same place — use build_prompt_and_stops"
    );
    assert!(
        !src.contains("sampling: self.request.sampling_params.clone()"),
        "remote_generate must not forward the caller's sampling params untouched: \
         the template's stop markers would never reach the serving node"
    );
}

/// Every failure the API can produce must be readable by the caller, including
/// the ones axum generates itself.
///
/// The router already answers an unrouted path with the standard envelope. The
/// right-path/wrong-method case did not: `POST /api/admin/config` (a PUT
/// endpoint) returned 405 and an EMPTY body. The dashboard parses failures with
/// `await resp.json()` in a try/catch, so an empty body throws, the catch
/// swallows it, and the user sees the generic "action failed" with the real
/// reason discarded — which is precisely what the error-envelope rule exists to
/// prevent.
#[test]
fn the_router_answers_bad_methods_in_the_error_envelope() {
    let root = repo_root();
    let src =
        std::fs::read_to_string(root.join("src/api/server.rs")).expect("api/server.rs must exist");
    assert!(
        src.contains(".method_not_allowed_fallback("),
        "the router must install a method-not-allowed fallback, or a wrong-method \
         request returns 405 with an empty body that no client can read"
    );
    assert!(
        src.contains("async fn wrong_method("),
        "the method-not-allowed fallback handler must exist"
    );

    // ORDER, not just presence. A `.layer()` does not wrap a fallback added
    // after it, so registering this one last put it outside the CORS layer,
    // where it answered the browser's OPTIONS preflight 405 with no
    // `access-control-*` headers — failing the preflight and taking the
    // dashboard's API calls with it. The released binary answers that same
    // preflight 200; the regression was caught only by comparing against it.
    let fallback_at = src
        .find(".method_not_allowed_fallback(")
        .expect("checked above");
    let cors_at = src
        .find(".layer(middleware::cors_layer(")
        .expect("the CORS layer must still be applied");
    assert!(
        fallback_at < cors_at,
        "method_not_allowed_fallback must be registered BEFORE the CORS layer, \
         or OPTIONS preflight bypasses CORS and every browser client breaks"
    );
}

/// Both HuggingFace probe call sites must tell a mistyped repo apart from an
/// upstream fault.
///
/// A wrong repo name is the caller's to fix; a rate limit or an outage is not.
/// `probe.rs` classified them and `shards.rs` did not, so the endpoint people
/// actually use to add a model answered a typo with `502 Bad Gateway` and the
/// generic cloud-provider hint ("The cloud provider returned an error. Try
/// again") — naming the wrong system and advising the one thing that cannot
/// help. The stale comment there even claimed to match probe.rs.
#[test]
fn both_huggingface_probe_sites_treat_a_typo_as_the_callers_mistake() {
    let root = repo_root();
    for rel in ["src/api/admin_hf/probe.rs", "src/api/admin_hf/shards.rs"] {
        let src = std::fs::read_to_string(root.join(rel)).expect("call site must exist");
        assert!(
            src.contains("probe_failure_is_user_fixable"),
            "{rel} probes HuggingFace, so it must separate a wrong name from an \
             upstream failure — otherwise a typo is reported as a gateway error"
        );
    }
}

/// A streaming error frame must take its type from `classify_error`, never a
/// literal.
///
/// Both SSE encoders used to hardcode one, so every failure a stream reported
/// was typed `server_error` — an over-long prompt, which the non-streaming
/// sibling answers `400 invalid_request_error`, was reported inside a 200 as
/// this server breaking (measured on the released v0.3.95 binary, 2026-08-12).
/// A literal here is invisible: the frame still looks well-formed, and only a
/// client comparing the two surfaces on the same input can tell.
#[test]
fn a_streamed_error_names_the_same_failure_as_its_non_streaming_sibling() {
    let root = repo_root();
    for rel in [
        "src/api/openai/streaming.rs",
        "src/api/anthropic/handlers.rs",
    ] {
        let src =
            std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        // The encoders build the frame from the event's own `error_type`; a
        // quoted type next to `"type":` means someone chose one locally again.
        for literal in [
            "\"type\": \"server_error\"",
            "\"type\":\"server_error\"",
            "\"type\": \"api_error\"",
        ] {
            assert!(
                !src.contains(literal),
                "{rel} hardcodes {literal} in a streamed error frame — take the \
                 type from crate::error::classify_error so the streaming and \
                 non-streaming surfaces cannot disagree about the same failure"
            );
        }
    }

    // `stop_reason` carries only values the Messages API defines, and "error"
    // is not one of them. The split stream used to send it.
    let anthropic = std::fs::read_to_string(root.join("src/api/anthropic/handlers.rs"))
        .expect("read anthropic handlers");
    assert!(
        !anthropic.contains("stop_reason = \"error\""),
        "src/api/anthropic/handlers.rs sets stop_reason to \"error\", which the \
         Anthropic API does not define — report the failure with an `error` SSE \
         event instead, which is terminal and needs no stop_reason"
    );
}

/// A stream that fails must say so — never end with the model "choosing" to
/// stop.
///
/// Measured on the released v0.3.145 (gotcha #433): a streamed request for a
/// model no node held answered `200`, an empty assistant delta,
/// `finish_reason: "stop"`, `[DONE]`. The identical request without `stream`
/// answered 503 with a hint naming the cause. The legacy in-process stream
/// mapped every execution error to `"stop"` with a comment explaining that the
/// caller would surface the error some other way — which nothing did. And the
/// router stream's channel-dropped arm did the same for a pipeline that died.
///
/// The shape this catches: an `Err` arm that, within a few lines, produces the
/// literal `"stop"`. A failure has `StreamEvent::Error` for exactly this.
/// **Both streaming surfaces are scanned.** This checked only the OpenAI one
/// until 2026-09-14, when an audit pointed out the Anthropic sibling was
/// correct by hand-inspection and unguarded — which is how the OpenAI one got
/// this way in the first place. The terminal literal differs: OpenAI ends a
/// natural reply with `finish_reason: "stop"`, Anthropic with
/// `stop_reason: "end_turn"`. Neither means "something went wrong", and the
/// Anthropic surface has `AnthropicSseEvent::Error` for that.
#[test]
fn a_stream_that_fails_never_pretends_the_model_chose_to_stop() {
    let root = repo_root();
    // (file, the literal that means "the model finished normally")
    let surfaces: &[(&str, &str)] = &[
        ("src/api/openai/streaming.rs", "\"stop\""),
        ("src/api/anthropic/sse.rs", "\"end_turn\""),
        ("src/api/anthropic/handlers.rs", "\"end_turn\""),
    ];
    for (rel, natural_end) in surfaces {
        let src =
            std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            let is_err_arm = trimmed.starts_with("Err(") && trimmed.contains("=>");
            if !is_err_arm {
                continue;
            }
            let window = lines[i..lines.len().min(i + 10)].join("\n");
            assert!(
                !window.contains(natural_end),
                "{rel}:{} — an Err arm ends the stream with {natural_end}, which means the \
                 model chose to stop:\n{window}\n\
                 report the failure as an error event (typed via \
                 crate::error::classify_error) instead, so a streaming client hears the \
                 same failure its non-streaming sibling would",
                i + 1
            );
        }
    }
}

/// Every hint key the backend can emit must have a translation entry.
///
/// The hints are the advice a user acts on when something goes wrong, and until
/// 2026-08-17 they had no translation route at all — the dashboard ships 21
/// locales and showed every one of these paragraphs in English. They travel now
/// as a stable `hint_key` next to the English text, and the frontend looks up
/// `error_hint.<key>`.
///
/// The failure this pins is silent and invisible to anyone testing in English:
/// a new `SwarmError` variant gets a hint in `src/error.rs`, nobody adds the
/// JSON entry, and the lookup falls back to the English the backend sent. It
/// works — in English. Every other locale silently drops back to a language its
/// reader may not speak, and nothing anywhere says so.
///
/// The sibling `all_locales_have_the_same_keys` check then extends this to the
/// other 20 automatically.
#[test]
fn every_backend_hint_key_has_a_translation() {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("src/error.rs")).expect("read error.rs");

    // The keys are the first element of each `Some((` pair in
    // `error_hint_with_key`, which is the only place they are written.
    let body = src
        .split("pub fn error_hint_with_key")
        .nth(1)
        .expect("error_hint_with_key exists");
    let body = body
        .split("\npub fn error_hint")
        .next()
        .expect("error_hint follows it");

    let mut keys: BTreeSet<String> = BTreeSet::new();
    let mut rest = body;
    while let Some(i) = rest.find("Some((") {
        rest = &rest[i + "Some((".len()..];
        let after = rest.trim_start();
        if !after.starts_with('"') {
            continue;
        }
        let inner = &after[1..];
        let Some(end) = inner.find('"') else { continue };
        keys.insert(inner[..end].to_string());
    }
    assert!(
        keys.len() >= 20,
        "expected the full hint set, found {} — has `error_hint_with_key` \
         changed shape? {keys:?}",
        keys.len()
    );

    let en: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("frontend/i18n/en.json")).expect("read en.json"),
    )
    .expect("en.json is not JSON");
    let defined = en.as_object().expect("en.json is not an object");

    let missing: Vec<String> = keys
        .iter()
        .map(|k| format!("error_hint.{k}"))
        .filter(|k| !defined.contains_key(k))
        .collect();
    assert!(
        missing.is_empty(),
        "hint keys emitted by src/error.rs with no entry in frontend/i18n/en.json: \
         {missing:?}\nAdd them to en.json and translate across all 21 locales — \
         see .claude/rules/i18n.md."
    );

    // And the converse: an `error_hint.*` entry no backend variant can produce
    // is dead weight carried in 21 files.
    let orphans: Vec<&String> = defined
        .keys()
        .filter(|k| k.starts_with("error_hint."))
        .filter(|k| !keys.contains(k.trim_start_matches("error_hint.")))
        .collect();
    assert!(
        orphans.is_empty(),
        "frontend/i18n/en.json defines error_hint entries that src/error.rs \
         never emits: {orphans:?}"
    );
}

/// While the credit economy is dormant, nothing may publish or act on a balance.
///
/// Three separate places did, and fixing two of them was not enough: the
/// leaderboard's *self* entry is built by different code from its peer entries,
/// so removing the fields from the peer loop left the node still publishing its
/// own. That was caught by calling the endpoint, not by reading the change —
/// this test is what makes the next one cheap to catch.
///
/// See `docs/CREDITS_DESIGN.md` for what has to be true before any of this
/// comes back.
#[test]
fn credits_stay_dormant() {
    let root = repo_root();

    // 1. No balance floor. A remote requester below it was refused outright.
    let ledger = std::fs::read_to_string(root.join("src/credit/ledger.rs")).expect("read ledger");
    assert!(
        ledger.contains("pub const MIN_BALANCE_FOR_INFERENCE: i64 = 0;"),
        "MIN_BALANCE_FOR_INFERENCE is non-zero — a self-minted balance is \
         refusing somebody inference (docs/CREDITS_DESIGN.md § 4)"
    );

    // 2. The tier is the same for everyone, so a balance buys no throughput.
    let priority =
        std::fs::read_to_string(root.join("src/credit/priority.rs")).expect("read priority");
    assert!(
        priority.contains("pub fn calculate_tier(_balance: i64, _network_percentile: f32)"),
        "calculate_tier reads its arguments again — a balance is buying \
         priority (docs/CREDITS_DESIGN.md § 4)"
    );

    // 3. The leaderboard neither ranks by credits nor publishes them. Checked
    //    on the whole file so a NEW entry-construction path is caught too —
    //    that is precisely how this one escaped the first fix.
    let identity =
        std::fs::read_to_string(root.join("src/api/identity.rs")).expect("read api/identity");
    for field in ["\"credits\":", "\"tier\":", "\"balance_known\":"] {
        assert!(
            !identity.contains(field),
            "src/api/identity.rs publishes {field} — the leaderboard must not \
             expose a self-minted balance (docs/CREDITS_DESIGN.md § 4)"
        );
    }

    // 3b. The diagnostics report does not print a balance. It is written to be
    //     PASTED to somebody, so it is a publishing surface — and
    //     `docs/CREDITS_DESIGN.md` names it in its own list of surfaces to
    //     check. The 2026-09-13 CLI pass removed the CLI's own credit printing
    //     and left the section in the report the CLI prints, which is how a
    //     released binary still answered `swarmllm diagnostics` with
    //     "-- credits --  balance: 6397160" (gotcha #569: one builder is not
    //     one surface).
    let admin = std::fs::read_to_string(root.join("src/api/admin.rs")).expect("read api/admin");
    let report = fn_body(&admin, "pub async fn diagnostics(")
        .expect("the diagnostics report builder must be findable, or this guard checks nothing");
    for marker in ["-- credits --", "credit_balance.read()"] {
        assert!(
            !report.contains(marker),
            "the diagnostics report mentions {marker} — it is pasted to \
             strangers, and a self-minted balance that gates nothing must not \
             be published there (docs/CREDITS_DESIGN.md § 4)"
        );
    }

    // 4. And no dashboard code reads a credit figure. The backend half of this
    //    was checked and the frontend half was not, so an unreachable
    //    `sortKey === 'credits'` branch survived the cleanup that removed every
    //    element which rendered the balance — ranking peers by a self-minted
    //    number, one restored column header away from working again, with
    //    nothing to catch it. It sorted on a `credits` field the peer payload
    //    has not carried for releases, which is why nobody noticed.
    //
    //    Comments may discuss the figure — one of them documents exactly why
    //    nothing renders it — so only code is scanned.
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.join("frontend/js")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "js") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            for (i, line) in text.lines().enumerate() {
                let l = line.trim();
                if l.starts_with("//") || l.starts_with("*") || l.starts_with("/*") {
                    continue;
                }
                if l.contains("credits") {
                    offenders.push(format!(
                        "{}:{}: {l}",
                        p.strip_prefix(&root).unwrap_or(&p).display(),
                        i + 1
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "dashboard code reads a credit figure. Credits are dormant: the balance \
         is self-minted and reconciled with nobody, so nothing may rank, sort or \
         display it (docs/CREDITS_DESIGN.md § 4).\n{}",
        offenders.join("\n")
    );

    // 5. And no COMMAND-LINE output publishes one either.
    //
    //    § 6's exit criteria say credits "do not become visible or enforcing
    //    again" until a credit provably moves between two machines. That was
    //    applied to the dashboard, to the leaderboard API and to the frontend
    //    JS — three surfaces, all reached through a browser — while
    //    `swarmllm pool status` went on printing "Total Credits", a CREDITS
    //    column per device, and, on joining, "All credits earned by this device
    //    will be forwarded to the pool owner": a promise of value transfer
    //    about a figure nobody reconciles. "Every surface" had quietly been
    //    read as "every surface the browser reaches", which is this codebase's
    //    most-repeated defect (`.claude/rules/architecture.md` § "One
    //    invariant, N paths").
    //
    //    Scanned as printed OUTPUT rather than as any mention of the word, so
    //    the flag names, doc comments and the forwarding mechanism itself —
    //    which still runs — are untouched.
    let mut cli_offenders: Vec<String> = Vec::new();
    // `src/main.rs` carries the top-level clap tree, so it is part of the CLI
    // surface even though it is not under `src/cli`. **A single FILE in the
    // stack, which the walk has to handle**: written first as a directory-only
    // walk, `read_dir` on a file simply errored and the entry was skipped, so
    // the guard could not fire on main.rs at all. Caught by planting the
    // violation, which is the only way that shape of hole is ever found
    // (`.claude/rules/architecture.md` § "A source-scanning guard is only as
    // good as the spellings it knows").
    let mut files: Vec<std::path::PathBuf> = Vec::new();
    let mut stack = vec![root.join("src/cli"), root.join("src/main.rs")];
    while let Some(entry) = stack.pop() {
        if entry.is_file() {
            if entry.extension().is_some_and(|x| x == "rs") {
                files.push(entry);
            }
            continue;
        }
        let Ok(rd) = std::fs::read_dir(&entry) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            stack.push(e.path());
        }
    }
    assert!(
        files.iter().any(|p| p.ends_with("main.rs")),
        "the CLI scan must reach src/main.rs — it holds the top-level clap tree"
    );
    {
        for p in &files {
            let p = p.clone();
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            for (line, stmt) in statements(&text) {
                let printing = stmt.contains("println!") || stmt.contains("print!");
                if printing && stmt.to_lowercase().contains("credit") {
                    cli_offenders.push(format!(
                        "{}:{line}: {stmt}",
                        p.strip_prefix(&root).unwrap_or(&p).display()
                    ));
                }
            }
            // A `///` in a clap-derive tree is not a note to the next reader —
            // it is the help text, printed verbatim by `--help`. `statements`
            // drops comments (rightly, for the scan above), so this is a second
            // pass. It is what the first version of this guard missed: the CLI
            // output was cleaned and `swarmllm pool --help` still opened with
            // "Device pool management (combine credits across your devices)",
            // describing the command's PURPOSE as the dormant thing.
            for (i, l) in text.lines().enumerate() {
                let l = l.trim();
                if l.starts_with("///") && l.to_lowercase().contains("credit") {
                    cli_offenders.push(format!(
                        "{}:{}: {l}  (a doc comment here is printed by --help)",
                        p.strip_prefix(&root).unwrap_or(&p).display(),
                        i + 1
                    ));
                }
            }
        }
    }
    assert!(
        cli_offenders.is_empty(),
        "command-line output publishes a credit figure. Credits are dormant and \
         do not become visible again until the exit criteria in \
         docs/CREDITS_DESIGN.md § 6 hold — and that applies to every surface, \
         not only the ones a browser reaches.\n{}",
        cli_offenders.join("\n")
    );
}

/// The CLI half of the guard above must actually be able to fire, on the shape
/// the real code was written in — a `println!` that rustfmt has wrapped across
/// several lines, with the word in the format string rather than on the
/// `println!` line. Planted violation, per `.claude/rules/architecture.md` §
/// "A source-scanning guard is only as good as the spellings it knows".
#[test]
fn the_dormant_credits_guard_sees_a_printed_balance_rustfmt_has_wrapped() {
    let planted = "fn f() {\n\
        \x20   println!(\n\
        \x20       \"Total Credits: {}\",\n\
        \x20       body.get(\"total_lifetime_credits\").unwrap()\n\
        \x20   );\n\
        }\n";
    assert!(
        statements(planted).iter().any(|(_, s)| {
            (s.contains("println!") || s.contains("print!")) && s.to_lowercase().contains("credit")
        }),
        "the scanner must see a wrapped println! whose credit mention is on a \
         later line than the macro itself"
    );

    // And it must NOT fire on the mechanism, which still runs and is still
    // named in flags and doc comments.
    let benign = "/// Forward credits to the pool owner.\nfn forward_credits() {}\n";
    assert!(
        !statements(benign).iter().any(|(_, s)| {
            (s.contains("println!") || s.contains("print!")) && s.to_lowercase().contains("credit")
        }),
        "a doc comment naming the mechanism is not a published balance"
    );
}

/// Nothing outside `chat.js` may decide "is the user looking at the empty chat
/// state?" from the session.
///
/// That question is about the DOM — `#chat-empty` is on screen or it is not —
/// and the session-shaped proxy for it (`currentSessionId` set, session exists,
/// `messages.length === 0`) is false in the commonest case there is: the very
/// first render, before any session has been created. Four separate callers
/// asked it that way and all four were blind to a fresh page load, so an empty
/// state built before the first stats arrived stayed on screen: a node with 6
/// peers and 11 ready models sat on "no models available yet · looking for
/// other computers" indefinitely (report #027). One of those callers carried a
/// comment naming that exact failure as the thing it existed to prevent.
///
/// `App.chat.refreshEmptyState` is the one answer; it reads the DOM and no-ops
/// when a conversation is showing. `chat.js` is exempt because `newSession`
/// legitimately asks a session question — whether to reuse an empty session
/// rather than make a second one — which is not about what is rendered.
#[test]
fn the_chat_empty_state_is_not_refreshed_on_a_session_shaped_guard() {
    let root = repo_root();
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.join("frontend/js")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "js") {
                continue;
            }
            // `chat.js` owns both the refresher and `newSession`.
            if p.file_name().is_some_and(|n| n == "chat.js") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            // Statement-joined so a guard rustfmt-style-wrapped across lines is
            // still seen as one condition (the trap of gotcha #413).
            for (i, stmt) in js_statements(&text) {
                if stmt.contains("messages.length === 0")
                    && (stmt.contains("renderMessages") || stmt.contains("createEmptyState"))
                {
                    offenders.push(format!(
                        "{}:{}: {stmt}",
                        p.strip_prefix(&root).unwrap_or(&p).display(),
                        i
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the chat empty state is being refreshed on a session-shaped guard, which \
         cannot see the first render — the case it is most needed for (report \
         #027). Call `App.chat.refreshEmptyState()`, which asks the DOM.\n{}",
        offenders.join("\n")
    );
}

/// The scanner above must actually be able to see the defect it forbids —
/// including when the condition is spread over several lines, which is how
/// every real instance of it was written.
#[test]
fn the_chat_empty_state_guard_catches_the_defect_it_forbids() {
    let planted = "if (S.activeTab === 'chat' && S.currentSessionId &&\n    \
                   S.sessions[S.currentSessionId] &&\n    \
                   S.sessions[S.currentSessionId].messages.length === 0 &&\n    \
                   App.chat && App.chat.renderMessages) {\n  App.chat.renderMessages();\n}\n";
    let hit = js_statements(planted)
        .into_iter()
        .any(|(_, s)| s.contains("messages.length === 0") && s.contains("renderMessages"));
    assert!(
        hit,
        "the guard cannot see a multi-line session-shaped refresh — it would pass \
         over the very code it exists to forbid"
    );
    // And the control: the shared refresher must NOT be reported.
    let ok = "if (S.activeTab === 'chat' && App.chat && App.chat.refreshEmptyState) {\n  \
              App.chat.refreshEmptyState();\n}\n";
    assert!(
        !js_statements(ok)
            .into_iter()
            .any(|(_, s)| s.contains("messages.length === 0")),
        "the guard reports the correct form"
    );
}

/// Join a JS source into statements, so a condition rustfmt or a human has
/// wrapped across lines is scanned as the one thing it is. Returns
/// (1-based line of the statement's first line, joined text).
fn js_statements(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut start = 1usize;
    for (i, raw) in text.lines().enumerate() {
        let l = raw.trim();
        if l.starts_with("//") || l.starts_with("*") || l.starts_with("/*") {
            continue;
        }
        if cur.is_empty() {
            start = i + 1;
        } else {
            cur.push(' ');
        }
        cur.push_str(l);
        // A statement ends at a terminator or a block open/close.
        if l.ends_with(';') || l.ends_with('{') || l.ends_with('}') {
            out.push((start, std::mem::take(&mut cur)));
        }
    }
    if !cur.is_empty() {
        out.push((start, cur));
    }
    out
}

/// Every `var(--x)` written without a fallback must name a property something
/// actually defines.
///
/// CSS makes this failure silent and total: a `var()` reference to an undefined
/// custom property with no fallback is invalid at computed-value time, so the
/// **whole declaration** is dropped — not just that one value. There is no
/// console warning and no visual clue beyond the styling simply not being there.
///
/// It has now shipped twice. On 2026-08-17 four properties were used and never
/// defined (`--text` 18 times, `--bg-elevated` 12), which is why every Settings
/// toggle knob was invisible and the Swarm tab had never once rendered as
/// designed. On 2026-08-18 `--bg` and `--text` were still being used in four
/// more places: the Master/Linked role badges in the pool list had no text
/// colour, and the auto-manage and private-mode status dots in the header had no
/// ring at all.
///
/// The direction of this check is deliberate. Used-but-undefined is a real bug
/// every time; defined-but-unused is not, and a text search reports live things
/// as orphans (gotcha #332). So this asserts only the safe direction.
///
/// `var(--x, fallback)` is excluded on purpose — it degrades rather than breaks,
/// which is exactly what a fallback is for.
#[test]
fn every_css_custom_property_used_without_a_fallback_is_defined() {
    let root = repo_root();

    fn ident_at(bytes: &[u8], mut i: usize) -> Option<(String, usize)> {
        if bytes.get(i) != Some(&b'-') || bytes.get(i + 1) != Some(&b'-') {
            return None;
        }
        let start = i;
        i += 2;
        while matches!(bytes.get(i), Some(c) if c.is_ascii_alphanumeric() || *c == b'-' || *c == b'_')
        {
            i += 1;
        }
        Some((String::from_utf8_lossy(&bytes[start..i]).into_owned(), i))
    }
    fn skip_ws(bytes: &[u8], mut i: usize) -> usize {
        while matches!(bytes.get(i), Some(c) if c.is_ascii_whitespace()) {
            i += 1;
        }
        i
    }

    let mut defined: BTreeSet<String> = BTreeSet::new();
    let mut used: Vec<(String, String)> = Vec::new();

    let mut stack = vec![root.join("frontend")];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !matches!(
                p.extension().and_then(|s| s.to_str()),
                Some("css") | Some("js") | Some("html")
            ) {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            let b = text.as_bytes();
            let where_ = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .into_owned();

            for i in 0..b.len() {
                // A definition: `--name:` — in a stylesheet rule or an inline style.
                if let Some((name, after)) = ident_at(b, i) {
                    if b.get(skip_ws(b, after)) == Some(&b':') {
                        defined.insert(name);
                        continue;
                    }
                }
                // A definition set from script: setProperty('--name', ...).
                if b[i..].starts_with(b"setProperty(") {
                    let j = skip_ws(b, i + "setProperty(".len());
                    let j = if matches!(b.get(j), Some(b'\'') | Some(b'"')) {
                        j + 1
                    } else {
                        j
                    };
                    if let Some((name, _)) = ident_at(b, j) {
                        defined.insert(name);
                    }
                    continue;
                }
                // A use with no fallback: `var(--name)` and nothing else.
                if b[i..].starts_with(b"var(") {
                    let j = skip_ws(b, i + "var(".len());
                    if let Some((name, after)) = ident_at(b, j) {
                        if b.get(skip_ws(b, after)) == Some(&b')') {
                            used.push((name, where_.clone()));
                        }
                    }
                }
            }
        }
    }

    assert!(
        !used.is_empty() && !defined.is_empty(),
        "the scan found nothing — the frontend layout must have moved, and a test \
         that scans nothing passes for the wrong reason"
    );

    let missing: Vec<String> = used
        .iter()
        .filter(|(name, _)| !defined.contains(name))
        .map(|(name, where_)| format!("{name} used in {where_}"))
        .collect();
    assert!(
        missing.is_empty(),
        "CSS custom properties used with no fallback and never defined — every \
         declaration referencing one is silently dropped in the browser:\n  {}",
        missing.join("\n  ")
    );
}

/// A node's advertised load must come from `SharedState::active_inference_load`,
/// never from `active_pipelines.len()`.
///
/// `active_pipelines` is the *coordinator's* map. `src/daemon/state/mod.rs`
/// documents this explicitly — "a node answering a peer's `RemoteGenerateRequest`
/// or `LayerForward` never appears in it, so anything that consults only
/// `active_pipelines` believes a pure-server node is doing nothing" — and the
/// health ping, the health pong and the scheduler's own local candidate each did
/// exactly that anyway. The doc comment had been there for weeks.
///
/// The effect was the opposite of load balancing, and worst on the most useful
/// node in a swarm: a machine that only serves advertises a load of zero
/// forever, so every coordinator sees it idle, keeps choosing it, and never
/// observes it saturating. Measured 2026-08-20 with one GPU peer present.
///
/// This is enforced rather than documented because documenting it is precisely
/// what failed.
#[test]
fn advertised_load_counts_every_kind_of_work() {
    let root = repo_root();
    let mut sources: Vec<(PathBuf, String)> = Vec::new();
    let mut stack = vec![root.join("src")];
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

    let mut offenders: Vec<String> = Vec::new();
    for (path, text) in &sources {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        // The helper itself is allowed to say why it does NOT use this map, and
        // `update_restart` asks a different question — "is anything in flight at
        // all", where either map answering yes is enough to defer a restart.
        if rel == "src/daemon/state/mod.rs" || rel == "src/update_restart.rs" {
            continue;
        }
        // The diagnostics dump prints BOTH maps side by side on purpose, so an
        // operator can see them disagree. Collapsing it into the helper would
        // destroy the comparison it exists to make.
        let raw_comparison_is_the_point =
            rel == "src/api/admin.rs" && text.contains("in_flight: {} traces, {} pipelines");
        // Statements, not lines — a wrapped chain hides a dotted path.
        for (line_no, l) in statements(text) {
            if l.contains("active_pipelines.len()") {
                if raw_comparison_is_the_point && l.contains("ss.active_pipelines.len()") {
                    continue;
                }
                // Reporting the map under its own name is honest and stays
                // allowed — a `tracing` field spelled `active_pipelines = ...`
                // claims to be the coordinator's map and is. What this test
                // forbids is the map standing in for something broader, which
                // is always spelled as a different word: `load`,
                // `active_request_count`, `active_requests`.
                if l.contains("active_pipelines =") {
                    continue;
                }
                offenders.push(format!("{rel}:{line_no}: {l}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "a load figure is being derived from `active_pipelines.len()`, which \
         cannot see work this node does for peers or on its local fast path.\n\
         Use `SharedState::active_inference_load()`.\n{}",
        offenders.join("\n")
    );
}

/// Everything that puts a prompt, or anything derived from one, on the wire
/// Non-comment lines (1-indexed) in `text` mentioning `marker`.
fn marker_lines(text: &str, marker: &str) -> Vec<usize> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim_start().starts_with("//") && l.contains(marker))
        .map(|(i, _)| i + 1)
        .collect()
}

/// Is there a `share_prefix_cache_with_peers` read, taken from the LIVE config,
/// close enough to line `at` to be gating it?
///
/// Site-level on purpose. This was a whole-FILE check — "the file mentions the
/// marker somewhere, and the words `share_prefix_cache_with_peers` somewhere,
/// and `.cfg()` somewhere" — and the last of those three is satisfied by any of
/// the unrelated `.cfg()` calls these files already make, so it asserted very
/// nearly nothing. A second, ungated send added to either file would have kept
/// it green. Both files are large and the doc comment above is explicit that
/// nothing but this test connects the two paths.
fn live_sharing_gate_near(text: &str, at: usize) -> bool {
    /// How far from the send a gate can sit and still plausibly guard it. The
    /// gate precedes the announce by 9 lines and follows the fetch marker by
    /// 16, so the window has to reach both ways.
    const GATE_WINDOW: usize = 25;
    /// `cfg()` and the field are on one line at one site and five apart at the
    /// other, where the call is a multi-line method chain.
    const CFG_WINDOW: usize = 6;

    let lines: Vec<&str> = text.lines().collect();
    let is_code = |l: &&str| !l.trim_start().starts_with("//");
    let lo = at.saturating_sub(GATE_WINDOW + 1);
    let hi = (at + GATE_WINDOW).min(lines.len());
    (lo..hi).any(|i| {
        if !is_code(&lines[i]) || !lines[i].contains("share_prefix_cache_with_peers") {
            return false;
        }
        // The field alone is not enough: read from the boot snapshot it is a
        // privacy toggle that needs a restart to take effect.
        let clo = i.saturating_sub(CFG_WINDOW);
        let chi = (i + CFG_WINDOW + 1).min(lines.len());
        lines[clo..chi]
            .iter()
            .any(|l| is_code(l) && l.contains("cfg()"))
    })
}

/// The guard must catch an ungated send and a send gated on the boot snapshot,
/// and must not fire on the two shapes the codebase actually uses. Without this
/// the check can silently weaken back into the whole-file version it replaced.
#[test]
fn the_prefix_sharing_guard_catches_an_ungated_send() {
    // Gate on one line, send below — the announce site's shape.
    let ok_before = "if !state.cfg().inference.share_prefix_cache_with_peers {\n  return;\n}\nsend(SwarmMessage::PrefixCacheAnnounce(a));";
    // Multi-line cfg() chain above, marker above that — the fetch site's shape.
    let ok_after = "SwarmRequest::PrefixKvFetch(req) => {\nlet on = self\n.shared_state\n.cfg()\n.inference\n.share_prefix_cache_with_peers;\nif !on { return; }";
    assert!(live_sharing_gate_near(ok_before, 4));
    assert!(live_sharing_gate_near(ok_after, 1));

    // No gate at all.
    assert!(!live_sharing_gate_near(
        "send(SwarmMessage::PrefixCacheAnnounce(a));",
        1
    ));
    // Gated, but read from the boot snapshot — the #281 shape. A privacy
    // toggle that needs a restart is the bug, not the fix.
    assert!(!live_sharing_gate_near(
        "if !state.config.inference.share_prefix_cache_with_peers { return; }\nsend(a);",
        2
    ));
    // A gate far away in the same file must not count — this is exactly what
    // the whole-file check got wrong.
    let far = format!(
        "if !state.cfg().inference.share_prefix_cache_with_peers {{ return; }}\n{}send(a);",
        "\n".repeat(60)
    );
    assert!(!live_sharing_gate_near(&far, 62));
    // A commented-out gate is not a gate.
    assert!(!live_sharing_gate_near(
        "// if !state.cfg().inference.share_prefix_cache_with_peers { return; }\nsend(a);",
        2
    ));
}

/// must be gated on `inference.share_prefix_cache_with_peers`.
///
/// There are two such paths and they are easy to fix one at a time. The
/// announce broadcasts BLAKE3 hashes chained over the prompt's token blocks to
/// the whole gossip mesh; the serve hands back `SnapshotHeader.tokens`, which
/// is the prompt itself as token IDs. Gating only one leaves the other, and the
/// two are in different subsystems (`daemon/background.rs` and
/// `network/manager/requests.rs`) so nothing but this test connects them.
///
/// The gate must be read through `cfg()` — the live config — because a user who
/// turns prompt sharing off expects it to stop, not to stop after a restart.
/// That is gotcha #281, and a privacy toggle is the worst possible place for it.
#[test]
fn nothing_leaves_this_node_with_a_prompt_in_it_unless_sharing_is_on() {
    let root = repo_root();
    let sites = [
        (
            "src/daemon/background.rs",
            "PrefixCacheAnnounce broadcast",
            "SwarmMessage::PrefixCacheAnnounce",
        ),
        (
            "src/network/manager/requests.rs",
            "PrefixKvFetch serve",
            "SwarmRequest::PrefixKvFetch",
        ),
    ];

    for (rel, what, marker) in sites {
        let src = std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let uses = marker_lines(&src, marker);
        assert!(
            !uses.is_empty(),
            "{rel}: expected to find {what} ({marker}). If this path moved, move \
             the gate with it and update this test — do not delete the assertion."
        );
        for line_no in uses {
            assert!(
                live_sharing_gate_near(&src, line_no),
                "{rel}:{line_no}: {what} is not gated on \
                 inference.share_prefix_cache_with_peers read through cfg(). This \
                 path puts prompt-derived data on the wire, so it must be opt-in, \
                 and the gate must come from the LIVE config — turning a privacy \
                 setting off has to take effect without a restart (gotcha #281)."
            );
        }
    }

    // The default is the whole point: a node nobody configured must not publish
    // anything derived from its user's prompts.
    let cfg = swarmllm::config::Config::default();
    assert!(
        !cfg.inference.share_prefix_cache_with_peers,
        "prefix-cache sharing must default to OFF — the dashboard tells every user \
         'No peer can read your prompts or outputs', and that has to be true for \
         someone who changed no settings."
    );
}

/// The scheduler must not re-derive the local node's speed from `gpu_info`.
///
/// Two sites did — `.map(|g| …gpu_memory_bandwidth_gbps…).unwrap_or(0.0)` — so
/// a processor-only node reported its OWN speed as zero. Both consumers read
/// zero as *unknown* and substitute a generic constant (`UNKNOWN_COMPUTE_MS`;
/// the parallax allocator documents 0 as "treats as average"), so the one node
/// whose speed we can actually measure was the only one priced with a guess,
/// while every remote peer got a real gossiped figure.
///
/// `vram::node_tokens_per_sec_7b` is the shared derivation. The card lookup is
/// still legitimate elsewhere in this file (pricing a *peer's* advertised
/// card), so the check is for the specific composition that skips the
/// processor arm.
#[test]
fn the_local_nodes_speed_is_not_re_derived_from_the_graphics_card_alone() {
    let src = std::fs::read_to_string("src/inference/scheduler/mod.rs").expect("read scheduler");
    // Statement-joined so a chain rustfmt wrapped is still visible (gotcha #413).
    for (line, stmt) in statements(&src) {
        let flat: String = stmt.chars().filter(|c| !c.is_whitespace()).collect();
        assert!(
            !(flat.contains("gpu_info") && flat.contains("estimate_tokens_per_sec_7b")),
            "src/inference/scheduler/mod.rs:{line} derives the local node's speed from \
             gpu_info, so a processor-only node reports 0 and is priced as unknown. \
             Use vram::node_tokens_per_sec_7b, which asks the processor when there is \
             no card."
        );
    }
    assert!(
        src.contains("node_tokens_per_sec_7b"),
        "the scheduler no longer asks for the local node's speed at all — if that is \
         deliberate, retire this guard"
    );
}

/// No surface may answer "how much room does this node have" from the graphics
/// card alone.
///
/// The sibling guard above covers SPEED in one file. This covers MEMORY across
/// the repo, because the same mistake reappeared somewhere the file-scoped scan
/// could not see: `src/api/admin_hf/search.rs` computed
/// `gpu_info.map(|g| g.vram_free_mb * ...).unwrap_or(0)` and gated every
/// `fits_*` flag on the result being non-zero — so on a processor-only node
/// nothing ever fitted, and the model browser's "only show what fits" filter,
/// on by default, hid **every result of every search** including the node's own
/// wishlist "Set this up" button (gotcha #483).
///
/// `vram::node_model_budget_bytes` is the shared derivation; it asks system RAM
/// when there is no usable card. Reading a peer's advertised card is still
/// legitimate — the check is for the specific composition that turns THIS
/// node's absent card into a zero capacity.
#[test]
fn no_surface_reads_this_nodes_capacity_from_the_graphics_card_alone() {
    // The two places that legitimately own the distinction: the helper itself,
    // and the RAM budget it delegates to.
    // `vram.rs` owns the decision. `gossip.rs` is exempt for a different and
    // better reason: it announces `gpu_vram_available_mb`, a field that MEANS
    // graphics memory, and it announces RAM separately on the next lines — so
    // zero there is honest rather than a conflation.
    const OWNS_THE_DECISION: &[&str] = &[
        "src/model/auto_manage/vram.rs",
        "src/pool/manager/gossip.rs",
    ];
    let root = repo_root();
    let mut offenders = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().is_none_or(|x| x != "rs") {
                continue;
            }
            let p = path.to_string_lossy().replace('\\', "/");
            if OWNS_THE_DECISION.iter().any(|o| p.ends_with(o)) {
                continue;
            }
            let Ok(src) = std::fs::read_to_string(&path) else {
                continue;
            };
            for (line, stmt) in statements(&src) {
                let flat: String = stmt.chars().filter(|c| !c.is_whitespace()).collect();
                if flat.starts_with("//") {
                    continue;
                }
                if flat.contains("gpu_info")
                    && (flat.contains("vram_free_mb") || flat.contains("vram_total_mb"))
                    && flat.contains("unwrap_or(0")
                {
                    offenders.push(format!("{p}:{line}"));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "these sites turn 'this node has no graphics card' into 'this node has no \
         memory', which is a different fact and describes every processor-only \
         machine: {offenders:?}. Use vram::node_model_budget_bytes, which asks \
         system RAM when there is no usable card."
    );
}

/// The scan above must be able to see the defect it exists for — planted
/// verbatim, in the shape `search.rs` actually had (gotcha #413: four of five
/// guards could not see what they guarded).
#[test]
fn the_node_capacity_scan_catches_the_shape_it_was_written_for() {
    let planted = r#"
    let available_vram_bytes: u64 = state
        .shared_state
        .gpu_info
        .as_ref()
        .map(|g| g.vram_free_mb * 1024 * 1024)
        .unwrap_or(0);
"#;
    let mut seen = false;
    for (_, stmt) in statements(planted) {
        let flat: String = stmt.chars().filter(|c| !c.is_whitespace()).collect();
        if flat.contains("gpu_info")
            && flat.contains("vram_free_mb")
            && flat.contains("unwrap_or(0")
        {
            seen = true;
        }
    }
    assert!(
        seen,
        "the scan cannot see a chain rustfmt wrapped across seven lines, which is \
         exactly how the real one was written"
    );
}

/// Both halves of the guard above must actually work.
#[test]
fn the_local_speed_guard_catches_the_shape_it_is_written_for() {
    let planted = "        let local_tps = self
            .shared_state
            .gpu_info
                               .as_ref()
            .map(|g| estimate_tokens_per_sec_7b(bw(&g.name), true))
                               .unwrap_or(0.0);
";
    let hit = statements(planted).into_iter().any(|(_, stmt)| {
        let flat: String = stmt.chars().filter(|c| !c.is_whitespace()).collect();
        flat.contains("gpu_info") && flat.contains("estimate_tokens_per_sec_7b")
    });
    assert!(
        hit,
        "the scanner cannot see the wrapped chain it exists to catch"
    );
}

/// The diagnostics report is written to be pasted somewhere public — the
/// dashboard has a one-click "Copy diagnostics" button and `swarmllm
/// diagnostics` prints the same thing — and the person doing the pasting is
/// explicitly not expected to read it first. It carries this machine's
/// addresses and up to ten remembered peer multiaddrs, which on a live node
/// are other people's home IP addresses.
///
/// The redaction is one call at the end of one handler. Nothing else in the
/// system would notice if it went away, and the button's own hint tells the
/// user the output is safe to share.
#[test]
fn the_diagnostics_report_hides_addresses_unless_full_is_asked_for() {
    let src = std::fs::read_to_string("src/api/admin.rs").expect("read admin.rs");
    let body = fn_body(&src, "pub async fn diagnostics(")
        .expect("the diagnostics handler was renamed — re-point this guard");
    // Whitespace-stripped so the assertion is not pinned to whatever rustfmt
    // produced on the day it was written (gotcha #413).
    let flat: String = body.chars().filter(|c| !c.is_whitespace()).collect();

    assert!(
        flat.contains("redact::redact_addresses(&out)"),
        "GET /api/admin/diagnostics no longer redacts network addresses. Its \
         output is copied by a dashboard button whose hint promises the text is \
         safe to share, and it contains every peer address this node remembers."
    );
    assert!(
        flat.contains("query.wants_full()"),
        "the unredacted report must stay behind an explicit `?full=1`, not \
         become the default"
    );

    // The other half: the two surfaces that hand the report to a person must
    // ask for the safe form. `?full=1` belongs to an operator debugging their
    // own machine.
    let cli = std::fs::read_to_string("src/cli/diagnostics.rs").expect("read cli/diagnostics.rs");
    let cli_flat: String = cli.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(
        cli_flat.contains(r#"iffull{"?full=1"}else{""}"#),
        "`swarmllm diagnostics` must default to the redacted report"
    );
    let js = std::fs::read_to_string("frontend/js/components/reference-models.js")
        .expect("read reference-models.js");
    assert!(
        !js.contains("diagnostics?full"),
        "the dashboard's copy button must not request the unredacted report"
    );
}

/// A function's whole body, from its signature to the closing brace in column
/// zero. `None` if the signature is not present.
///
/// Replaces a fixed 1600-character window, which did not reach the end of the
/// function it was reading. `compute_vram_budget` runs to 1698 characters, so
/// the guard below was blind to its last 100 — and a `shared.config.resources`
/// read planted there passed the test, which is gotcha #281's exact defect and
/// the one that cost a 26x slowdown. The `cfg()` call it asserts on sat at
/// offset 1568, twenty characters inside the cap, so twenty characters of
/// unrelated growth anywhere above it would also have failed the test on
/// correct code. Take the body, not a guess at how long the body is.
fn fn_body<'a>(src: &'a str, signature: &str) -> Option<&'a str> {
    let start = src.find(signature)?;
    let rest = &src[start..];
    let end = rest.find("\n}\n").map(|i| i + 2).unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Does this code read the boot-time config snapshot rather than the live one?
///
/// Whitespace is stripped first, because the check it replaces matched one
/// exact indentation (`"shared\n        .config\n        .resources"`). That
/// string is pinned to whatever rustfmt produced on the day it was written, so
/// nesting the call one level deeper would have silently retired the
/// assertion.
///
/// `.config.` is the whole marker: nothing in these budget functions has any
/// business reading the snapshot, so there is no legitimate use to exclude, and
/// naming the receiver (`shared`, `shared_state`, `self.shared`) would just be
/// the same brittleness in another form.
fn reads_the_boot_snapshot(body: &str) -> bool {
    let flat: String = body.chars().filter(|c| !c.is_whitespace()).collect();
    flat.contains(".config.")
}

/// Both halves of the guard above must actually work, on the real shapes.
#[test]
fn the_boot_snapshot_check_is_not_pinned_to_one_formatting() {
    assert!(reads_the_boot_snapshot(
        "shared.config.resources.max_gpu_vram_mb"
    ));
    // The multi-line chain, at any indentation — the form the old check froze.
    assert!(reads_the_boot_snapshot(
        "shared\n        .config\n        .resources"
    ));
    assert!(reads_the_boot_snapshot("shared\n  .config\n  .resources"));
    // A different receiver is the same defect.
    assert!(reads_the_boot_snapshot("shared_state.config.resources"));
    // The live read must not be mistaken for it.
    assert!(!reads_the_boot_snapshot(
        "shared.cfg().resources.inference_vram_budget_mb(gpu_total)"
    ));

    // And the body must be taken whole, or a read in its tail is invisible.
    let src = "pub fn f(x: u32) -> u32 {\n    let a = 1;\n    shared.config.resources.y\n}\n";
    let body = fn_body(src, "pub fn f").expect("body");
    assert!(body.contains("shared.config.resources"));
    assert!(reads_the_boot_snapshot(body));
    assert!(fn_body(src, "pub fn absent").is_none());
}

/// The VRAM budget decides whether a model runs on the graphics card or crawls
/// on the processor, so it must be read from the LIVE config.
///
/// It was not. `compute_vram_budget` read `shared.config.resources` — the
/// snapshot taken at startup — so raising `max_gpu_vram_mb` in Settings saved,
/// answered "ok", wrote the new value to disk, and changed nothing until the
/// daemon restarted. Measured 2026-08-24 on a machine with 7187 MB of its card
/// free: config on disk said 7000, the running daemon reported a 4095 MB
/// budget, and a 6033 MB model was pushed to the processor at 1.0 tok/s.
///
/// Its sibling `ram_budget_now` was given exactly this treatment in August
/// (gotcha #362). The two must not drift apart again.
#[test]
fn the_vram_budget_is_read_live_like_the_ram_budget_beside_it() {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("src/model/auto_manage/vram.rs")).expect("vram.rs");

    let body = fn_body(&src, "pub fn compute_vram_budget")
        .expect("compute_vram_budget must exist — if it moved, move this test with it");

    assert!(
        body.contains("cfg()"),
        "compute_vram_budget must read the live config via cfg(), not the boot \
         snapshot — raising max_gpu_vram_mb has to take effect without a restart"
    );
    assert!(
        !reads_the_boot_snapshot(body),
        "compute_vram_budget still reads shared.config (the boot snapshot)"
    );
}

/// Every outbound dial must go through `NetworkManager::dial_checked`, which
/// refuses a peer Identify has shown does not speak SwarmLLM.
///
/// The per-site version of this check shipped and did not hold: two of six
/// sites had it, and the node still opened 17 connections to three foreign
/// nodes in seven minutes — each disconnected 43 ms later by the Identify gate,
/// then dialled again. Working out *which* of the other four was responsible
/// took a `-vv` capture and still did not settle it, which is the argument for
/// a choke point over a checklist: a guard a caller can forget is one a caller
/// will forget.
#[test]
fn every_dial_goes_through_the_foreign_peer_gate() {
    let net = repo_root().join("src/network");
    let mut sites: Vec<String> = Vec::new();
    let mut stack = vec![net];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "rs") {
                let Ok(text) = std::fs::read_to_string(&p) else {
                    continue;
                };
                // Match the FREE-FUNCTION form too, not just `self.swarm`.
                // This test read `self.swarm.dial(` only, and
                // `discovery::bootstrap_peers` took `&mut Swarm` and called
                // `swarm.dial(addr)` — so the one dial site that mattered was
                // invisible to the audit that added this test, and went on
                // dialling every cached address unconditionally and ungated for
                // another release (gotcha #405).
                //
                // Statements rather than lines, and comments are dropped by the
                // scanner — the doc comments that explain this rule tripped the
                // line-based version the first time round.
                for (line_no, call) in statements(&text) {
                    if call.contains("swarm.dial(") {
                        sites.push(format!("{}:{}", p.display(), line_no));
                    }
                }
            }
        }
    }
    // Two sites are permitted, and each is named: `dial_checked`'s own, and
    // the loopback probe, which dials fixed `127.0.0.1` candidate ports that
    // name no peer — nothing for the gate to check and nothing to dedup
    // against. Excluding the whole FILE instead would re-open the hole this
    // test just had, one directory down.
    let loopback: Vec<String> = sites
        .iter()
        .filter(|s| s.contains("discovery.rs"))
        .cloned()
        .collect();
    assert_eq!(
        loopback.len(),
        1,
        "`discovery.rs` may dial exactly once — the loopback probe. Anything \
         else there must go through `dial_checked`.\nFound:\n  {}",
        loopback.join("\n  ")
    );
    assert!(
        std::fs::read_to_string(repo_root().join("src/network/discovery.rs"))
            .is_ok_and(|t| t.contains("pub fn probe_loopback_peers")),
        "the permitted discovery.rs dial is the loopback probe; that function is gone, \
         so re-check what is dialling there"
    );
    sites.retain(|s| !s.contains("discovery.rs"));
    assert_eq!(
        sites.len(),
        1,
        "`self.swarm.dial` may appear exactly once — inside `dial_checked`. \
         Every other dial goes through that helper, or a foreign peer can be \
         re-dialled for the life of the process.\nFound:\n  {}",
        sites.join("\n  ")
    );
    assert!(
        sites[0].contains("connections.rs"),
        "the one permitted call is `dial_checked`'s own, in connections.rs; found {}",
        sites[0]
    );
}

/// Files this repo authors as TEXT, and therefore expects to be greppable.
/// Extension-based rather than content-sniffing on purpose: a content sniff
/// would call a NUL-carrying file "binary" and skip exactly the file the scan
/// exists to catch. Pinned by `the_nul_scan_looks_at_the_files_it_claims_to`.
fn is_text_source_extension(p: &std::path::Path) -> bool {
    p.extension().and_then(|x| x.to_str()).is_some_and(|x| {
        matches!(
            x,
            "md" | "rs"
                | "js"
                | "css"
                | "html"
                | "json"
                | "toml"
                | "sh"
                | "py"
                | "yml"
                | "yaml"
                | "txt"
        )
    })
}

/// The scan above must look at the file types the defect it exists for actually
/// landed in — a scan that cannot reach a file reports success exactly like a
/// clean tree (`.claude/rules/architecture.md` § "A source-scanning guard is
/// only as good as the spellings it knows"). Markdown alone is what let
/// gotcha #513 through in a `.rs` file.
#[test]
fn the_nul_scan_looks_at_the_files_it_claims_to() {
    use std::path::Path;
    for name in [
        "docs/DIAGNOSTICS.md",
        "src/inference/split/kv_cache.rs",
        "frontend/js/core/utils.js",
        "frontend/css/main.css",
        "frontend/index.html",
        "frontend/i18n/en.json",
        "Cargo.toml",
        ".githooks/pre-push.sh",
        "python/client.py",
        ".github/workflows/ci.yml",
    ] {
        assert!(
            is_text_source_extension(Path::new(name)),
            "{name} is a text file this repo writes and must be scanned for NUL bytes"
        );
    }
    // Binary assets are not text and must not be flagged.
    for name in [
        "frontend/fonts/IBMPlexSans.woff2",
        "docs/img/diagram.png",
        "model.gguf",
    ] {
        assert!(
            !is_text_source_extension(Path::new(name)),
            "{name} is a binary asset and must not be scanned as text"
        );
    }
}

/// Documentation must be greppable, which means no NUL bytes.
///
/// A single NUL makes `file` report "data", makes GNU grep print
/// "binary file matches" and suppress every hit, and makes any `grep`
/// configured to skip binaries (ugrep `-I`, ripgrep by default) return **no
/// matches at all, silently**. The failure is invisible: an empty result looks
/// exactly like the thing not being there.
///
/// Found 2026-08-29 in `docs/DIAGNOSTICS.md`, which had carried one since
/// 2026-08-15 — a shell snippet had a literal NUL where `tr '\0' ' '` was
/// meant, so the documented command was broken too (a shell cannot pass a NUL
/// in argv). The whole debugging guide had been ungreppable for two weeks, and
/// that file's own header warns that "a guide whose greps come back empty is
/// worse than no guide". It was caught only because a search for a section that
/// *did* exist came back empty.
///
/// Deliberately covers every tracked doc, not just the one that broke: nothing
/// about the mistake was specific to that file, and the Rust-side checks kept
/// passing throughout because a NUL is valid UTF-8.
///
/// **Widened from markdown to every text source on 2026-09-09.** Eleven days
/// after this guard shipped, the same defect landed in
/// `src/inference/split/kv_cache.rs` — one NUL inside a comment, which hid that
/// whole file from every repo-wide search until it was found by accident
/// (gotcha #513). The guard could not see it because it filtered on `.md`.
/// Nothing about a NUL is specific to markdown: it is a property of TEXT, so
/// the scan is scoped by "is this a text file we wrote", not by one extension.
#[test]
fn documentation_contains_no_nul_bytes() {
    let root = repo_root();
    let mut offenders = Vec::new();
    // Just the root: `read_dir` returns hidden entries, so `.claude/` and
    // `docs/` are reached by recursion. Seeding them explicitly as well walked
    // them twice and reported every offender twice.
    let mut stack = vec![root.clone()];

    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if p.is_dir() {
                // `book/` holds generated output; target/ and .git are noise.
                // `book/` is generated output and `vendor/` is upstream source:
                // a NUL in either is not ours to fix, and both are large.
                if !matches!(name, "target" | ".git" | "node_modules" | "book" | "vendor") {
                    stack.push(p);
                }
                continue;
            }
            if !is_text_source_extension(&p) {
                continue;
            }
            let Ok(bytes) = std::fs::read(&p) else {
                continue;
            };
            if let Some(idx) = bytes.iter().position(|b| *b == 0) {
                let line = bytes[..idx].iter().filter(|b| **b == b'\n').count() + 1;
                offenders.push(format!(
                    "{} (first NUL at byte {idx}, line {line})",
                    p.strip_prefix(&root).unwrap_or(&p).display()
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "these text files contain NUL bytes, which makes grep skip them silently:\n  {}\n\
         Write the two-character escape (\\0) rather than a literal NUL.",
        offenders.join("\n  ")
    );
}

/// A GGUF header is parsed through a buffered reader, never straight off a
/// `File`.
///
/// `gguf_file::Content::read` walks the metadata with many tiny reads — for
/// every string, a length and then its bytes — so a bare `File` turns each of
/// those into a syscall. A 7.8 MB header carrying a 128k-token vocabulary and
/// 280k merges is roughly 820k of them.
///
/// Measured on the live node 2026-08-29 (gotcha #410): `GET /api/admin/models`
/// took 11.2 s, of which 9.6 s was KERNEL time, because it parses every local
/// model's header and seven call sites were handing over an unbuffered `File`.
/// Optimisation cannot touch syscall cost, which is why the release binary was
/// no faster than a debug one on that path.
///
/// The check is proximity, not naming: a `File::open` a few lines above a
/// Lines in `text` (1-indexed) where a GGUF header is parsed straight off a
/// file handle that was opened, unbuffered, within `WINDOW` lines above.
///
/// Split out from the test so the guard's own effectiveness can be asserted on
/// planted violations. A repo-wide scan that finds nothing is indistinguishable
/// from a scan that cannot find anything, and this one had already been silently
/// blind twice — once to a `match File::open` form, once to `OpenOptions`.
fn unbuffered_gguf_parse_lines(text: &str) -> Vec<usize> {
    /// How far above the call an `open` still counts as "and then parsed it".
    const WINDOW: usize = 6;
    let lines: Vec<&str> = text.lines().collect();
    let is_code = |l: &&str| !l.trim_start().starts_with("//");
    let mut hits = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        // Skip comments, or this trips over the doc comments that explain the
        // rule — the trap a sibling test here already hit.
        if !is_code(line) || !line.contains("Content::read(") {
            continue;
        }
        let window = &lines[i.saturating_sub(WINDOW)..=i];
        // `OpenOptions` counts as opening a file, and is not a corner case: it
        // is what anyone reaching for specific flags writes, and it yields
        // exactly the same unbuffered `File`.
        let opens_a_file = window
            .iter()
            .any(|l| is_code(l) && (l.contains("File::open") || l.contains("OpenOptions")));
        let buffers_it = window
            .iter()
            .any(|l| is_code(l) && (l.contains("BufReader") || l.contains("Cursor")));
        if opens_a_file && !buffers_it {
            hits.push(i + 1);
        }
    }
    hits
}

/// The guard must actually catch each way of writing the defect, and must not
/// fire on the buffered forms the codebase legitimately uses. Without this the
/// scan above can quietly become a no-op and still report success.
#[test]
fn the_unbuffered_gguf_guard_catches_every_form_of_the_defect() {
    let caught = |src: &str| !unbuffered_gguf_parse_lines(src).is_empty();

    // The plain form, and the `match` form that a naming-based rule missed.
    assert!(caught(
        "let mut f = File::open(p)?;\nContent::read(&mut f)?;"
    ));
    assert!(caught(
        "match File::open(p) {\n  Ok(mut f) => Content::read(&mut f),\n}"
    ));
    // `OpenOptions` — the same unbuffered handle, missed until 2026-08-30.
    assert!(caught(
        "let mut f = OpenOptions::new().read(true).open(p)?;\nContent::read(&mut f)?;"
    ));

    // Buffered handles are correct and must stay quiet, or the guard becomes
    // noise and gets suppressed rather than fixed.
    assert!(!caught(
        "let f = File::open(p)?;\nlet mut r = BufReader::new(f);\nContent::read(&mut r)?;"
    ));
    assert!(!caught(
        "let mut r = Cursor::new(&bytes);\nContent::read(&mut r)?;"
    ));
    // A reader built further up has no open next to it and is not this defect.
    assert!(!caught("Content::read(&mut reader)?;"));
    // A comment describing the defect is not the defect — the trap a sibling
    // test in this file actually hit.
    assert!(!caught(
        "// let mut f = File::open(p)?;\n// Content::read(&mut f)?;"
    ));
}

/// Lines that ask sysinfo to describe the WHOLE machine.
///
/// `System::new_all()` enumerates every process on the box; `refresh_all()`
/// then does the same scan again. Both are whole-machine sweeps, and a caller
/// wanting four numbers pays for all 105 processes twice.
///
/// Comments are skipped by `statements`, so the paragraph above — which names
/// both calls — is not itself an offender.
fn whole_machine_sysinfo_lines(text: &str) -> Vec<usize> {
    statements(text)
        .into_iter()
        .filter(|(_, st)| {
            st.contains("System::new_all()")
                || st.contains(".refresh_all()")
                || asks_for_every_subsystem(st)
        })
        .map(|(line, _)| line)
        .collect()
}

/// `RefreshKind::everything()` is the same whole-machine sweep spelled through
/// `new_with_specifics`, so the guard has to know it — otherwise it forbids two
/// spellings of one idea and waves the third through.
///
/// It must NOT match `CpuRefreshKind::everything()` or
/// `ProcessRefreshKind::everything()`: those are narrow kinds, and the second is
/// exactly right inside a `refresh_processes` scoped to one pid. The check is
/// therefore on the character BEFORE the match — a letter means it is part of a
/// longer type name.
fn asks_for_every_subsystem(st: &str) -> bool {
    const NEEDLE: &str = "RefreshKind::everything()";
    st.match_indices(NEEDLE).any(|(i, _)| {
        i == 0
            || !st[..i]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

/// No site may ask sysinfo to describe the whole machine.
///
/// `GET /api/admin/stats` did, through `detect_hardware`: `System::new_all()`
/// followed by `refresh_all()`, i.e. a full enumeration of every process on the
/// box, twice, to read four numbers — total RAM, used RAM, this process's RSS,
/// and the CPU name and count. Measured on a 105-process machine: **182 ms**
/// against **0.43 ms** for a targeted refresh, which was essentially the whole
/// cost of the endpoint (273 ms, of which 178 ms was kernel time).
///
/// It was the LONE outlier — `vram.rs`, `health/monitor.rs` and
/// `pool/manager/gossip.rs` all already build a bare `System::new()` and
/// refresh only what they read, one of them commenting that this makes it
/// "cheap enough to call at every admission".
///
/// This is gotcha #410's shape on a second endpoint: the dashboard paying
/// kernel time for data it did not ask for. Splitting utime from stime is what
/// found both.
#[test]
fn sysinfo_is_never_asked_to_describe_the_whole_machine() {
    let root = repo_root();
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            for line_no in whole_machine_sysinfo_lines(&text) {
                offenders.push(format!(
                    "{}:{}",
                    p.strip_prefix(&root).unwrap_or(&p).display(),
                    line_no
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "sysinfo is being asked to describe the whole machine:\n  {}\n\
         `System::new_all()` and `refresh_all()` enumerate every process on the \
         box. Build a bare `System::new()` and refresh only what you read \
         (`refresh_memory`, `refresh_cpu_list`, `refresh_processes` scoped to \
         an explicit pid list) — 182 ms against 0.43 ms when this was last \
         measured. Note that scoping NARROWLY is not the same as scoping to \
         our own pid: see `the_node_memory_figure_counts_the_model_workers`.",
        offenders.join("\n  ")
    );
}

/// The node's memory figure must count the model workers, not just the daemon.
///
/// The weights and the KV cache live in `swarmllm model-worker` subprocesses,
/// so a figure built from `std::process::id()` alone is not a low estimate of
/// this node's memory — it is blind to essentially all of it. The dashboard
/// showed a few hundred MB on a machine whose resident 14B held gigabytes, and
/// the "Your contribution" bar read 0% while the node was busy serving peers
/// (report #011).
///
/// The pull the other way is real and is why this is pinned: the sibling guard
/// above forbids a whole-machine scan because one cost 182 ms on a polled
/// endpoint (gotcha #417), and the cheapest way to obey it is to measure one
/// pid. Both rules hold at once only by naming an explicit pid LIST — this
/// daemon plus `worker_pids()`.
#[test]
fn the_node_memory_figure_counts_the_model_workers() {
    let src = std::fs::read_to_string(repo_root().join("src/api/admin.rs"))
        .expect("cannot read src/api/admin.rs");
    let body = fn_body(
        &src,
        "fn detect_hardware(shared_state: &crate::daemon::SharedState)",
    )
    .expect("detect_hardware not found — has it been renamed?");
    let flat = statements(body)
        .into_iter()
        .map(|(_, l)| l)
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        flat.contains("worker_pids()"),
        "detect_hardware no longer asks the pool for its worker pids, so the \
         memory it reports is the daemon's alone — which excludes the model \
         weights and the KV cache, i.e. nearly all of it (report #011)."
    );
    assert!(
        flat.contains("worker_rss_mb"),
        "detect_hardware must report `worker_rss_mb` beside the total: the \
         dashboard tooltip names the two parts, and a reader who remembers the \
         old daemon-only figure needs to see where the rest came from."
    );
}

/// Self-test for the guard above: it must fail on the shape it forbids.
#[test]
fn the_worker_memory_guard_catches_a_daemon_only_measurement() {
    // The pre-#011 body, verbatim in shape: one pid, no pool involvement.
    let planted =
        "fn detect_hardware(shared_state: &crate::daemon::SharedState) -> serde_json::Value {\n\
         let pid = sysinfo::Pid::from_u32(std::process::id());\n\
         sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);\n\
         let process_rss_mb = sys.process(pid).map(|p| p.memory()).unwrap_or(0);\n}\n";
    let body = fn_body(
        planted,
        "fn detect_hardware(shared_state: &crate::daemon::SharedState)",
    )
    .expect("the planted function must be findable");
    assert!(
        !body.contains("worker_pids()"),
        "the guard would not have caught the defect it exists for"
    );
}

/// A download speed is a RATE, so every figure written into
/// `speed_bytes_per_sec` must be a quantity divided by an elapsed time.
///
/// The P2P shard path assigned `chunk_len` — the size of the chunk that had
/// just arrived — straight into the field. At a 32 MiB chunk taking ~26 s the
/// dashboard read 33.5 MB/s for a 1.26 MB/s transfer, and because
/// `admin_models::helpers` derives `eta_secs` by dividing the remaining bytes
/// by this figure, the time remaining was wrong by the same factor and never
/// converged. The two HuggingFace writers had always divided properly, which
/// is exactly why it went unnoticed: only downloads from peers looked wrong.
///
/// Assigning a literal `0` is the documented way to say "not downloading" and
/// is allowed.
#[test]
fn a_download_speed_is_a_rate_not_a_size() {
    let root = repo_root();
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            let stmts = statements(&text);
            for i in 0..stmts.len() {
                if let Some(bad) = speed_assigned_without_a_division(&stmts, i) {
                    offenders.push(format!(
                        "{}:{} — {bad}",
                        p.strip_prefix(&root).unwrap_or(&p).display(),
                        stmts[i].0
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a byte count is being reported as a speed:\n  {}\n\
         `speed_bytes_per_sec` is read by the dashboard AND divided into the \
         remaining bytes to produce an ETA, so a size here is wrong twice. \
         Divide the bytes moved by the seconds they took (see \
         `model::acquisition`'s rolling window for the reference shape), or \
         assign a literal 0 to mean 'not downloading'.",
        offenders.join("\n  ")
    );
}

/// The right-hand side of a `speed_bytes_per_sec = ...` assignment, when that
/// value is neither a literal zero nor derived from a division.
///
/// `stmts` is the whole file so a bare identifier can be followed back to
/// where it was computed — `progress.rs` divides into a local and then assigns
/// it, which is correct and which a check on the assignment alone reads as a
/// bare size.
fn speed_assigned_without_a_division(stmts: &[(usize, String)], at: usize) -> Option<String> {
    let stmt = &stmts[at].1;
    let idx = stmt.find("speed_bytes_per_sec")?;
    let rest = stmt[idx + "speed_bytes_per_sec".len()..].trim_start();
    // Only an assignment; `==`, `>` and a struct-literal `:` are reads.
    let rhs = rest.strip_prefix('=')?;
    if rhs.starts_with('=') {
        return None;
    }
    let rhs = rhs.split(';').next()?.trim();
    if rhs.is_empty() || rhs == "0" || rhs.contains('/') {
        return None;
    }
    // A bare identifier: accept it if it was computed by a division not far
    // above. Anything else (a field, a call, an arithmetic expression with no
    // division) is a size being passed off as a rate.
    if rhs.chars().all(|c| c.is_alphanumeric() || c == '_') {
        let lo = at.saturating_sub(20);
        let assigns_by_division = stmts[lo..at].iter().any(|(_, prev)| {
            let flat: String = prev.chars().filter(|c| !c.is_whitespace()).collect();
            (flat.contains(&format!("let{rhs}=")) || flat.contains(&format!("{rhs}=")))
                && prev.contains('/')
        });
        if assigns_by_division {
            return None;
        }
    }
    Some(rhs.to_string())
}

/// Self-test: the scan must catch the shape it forbids, and pass the two it allows.
#[test]
fn the_download_speed_guard_catches_a_size_reported_as_a_rate() {
    let one = |l: &str| {
        let v = vec![(1usize, l.to_string())];
        speed_assigned_without_a_division(&v, 0)
    };
    // the defect
    assert!(one("entry.speed_bytes_per_sec = chunk_len;").is_some());
    // the allowed forms
    assert!(one("job.status.speed_bytes_per_sec = 0;").is_none());
    assert!(one("entry.speed_bytes_per_sec = (chunk_len as f64 / chunk_secs) as u64;").is_none());
    // reads, not writes
    assert!(one("if status.speed_bytes_per_sec > 0").is_none());
    assert!(one("\"speed_bytes_per_sec\": entry.speed_bytes_per_sec,").is_none());

    // a bare identifier computed by a division just above is fine …
    let ok = vec![
        (
            1usize,
            "let speed = (moved as f64 / dt) as u64;".to_string(),
        ),
        (2usize, "entry.speed_bytes_per_sec = speed;".to_string()),
    ];
    assert!(speed_assigned_without_a_division(&ok, 1).is_none());
    // … and the same shape without one is not.
    let bad = vec![
        (1usize, "let speed = chunk_len;".to_string()),
        (2usize, "entry.speed_bytes_per_sec = speed;".to_string()),
    ];
    assert!(speed_assigned_without_a_division(&bad, 1).is_some());
}

/// Resolving an API key must not ANNOUNCE a file write it does not perform.
///
/// `resolve_api_key` deliberately does not touch the data directory — only the
/// daemon's own startup calls `publish_api_key_file` — and
/// `tests/api_key_side_effects.rs` guards that behaviour, because the write
/// really did overwrite running nodes' keys twice (2026-07-31, 2026-08-01).
///
/// The BANNER outlived that fix. It kept printing `Saved to: <path>` and
/// `Recover anytime: cat <path>`, so every test that builds a `SharedState`
/// announced, on a developer's machine, that it had just overwritten the
/// api_key of whatever node they had running. The message is indistinguishable
/// from the regression it sits next to, and cost a real investigation on
/// 2026-08-30 before the file's mtime proved nothing had been written.
///
/// A message asserting something the code deliberately stopped doing is a stale
/// comment that the user can see.
#[test]
fn resolving_an_api_key_does_not_announce_a_file_write() {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("src/daemon/helpers.rs")).expect("helpers.rs");
    let body = fn_body(
        &src,
        "pub(super) fn resolve_api_key(config: &Config, db: &Database) -> String {",
    )
    .expect("resolve_api_key must still exist with this signature");

    // Scan STATEMENTS, not raw text: the comment above the banner quotes both
    // phrases to explain why they were removed, and a raw `contains` trips on
    // its own explanation — the trap a sibling test in this file already hit.
    let code: String = statements(body)
        .into_iter()
        .map(|(_, st)| st)
        .collect::<Vec<_>>()
        .join("\n");
    for claim in ["Saved to", "Recover anytime"] {
        assert!(
            !code.contains(claim),
            "`resolve_api_key` prints {claim:?}, but it does NOT write the api_key \
             file — `publish_api_key_file` does, from daemon startup only. Printing \
             this makes every SharedState-building test look like it just clobbered \
             a running node's key. Say what was actually done, or move the line to \
             where the write happens."
        );
    }
    // The banner must still exist — first-run users need the key, and deleting
    // it to satisfy the assertions above would be the wrong fix.
    assert!(
        code.contains("Generated new API key") && code.contains("KEY:"),
        "the first-run key banner has gone; users need to see the key they were given"
    );
}

/// Every test count the README quotes, as `(count, label)` pairs.
///
/// The README states the figure twice — once in the status banner near the top,
/// once in the disclosure note at the bottom — and nothing tied them together.
fn readme_test_counts(text: &str) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for (i, _) in text.match_indices(" lib tests + ") {
        let before: String = text[..i]
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let lib: u32 = match before.chars().rev().collect::<String>().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let after = &text[i + " lib tests + ".len()..];
        let integ: u32 = match after
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect::<String>()
            .parse()
        {
            Ok(v) => v,
            Err(_) => continue,
        };
        out.push((lib, integ));
    }
    out
}

/// The README's test counts must agree with each other AND with `CLAUDE.md`.
///
/// The README quoted the figure in two places and they **disagreed** — "1218
/// lib tests + 79 integration" in the status banner, "1169 lib tests + 75
/// integration" in the disclosure note at the bottom — while the real number
/// was 2155. Both had been wrong for months and nothing looked at either.
///
/// It matters more than an ordinary stale number because of where it sits: the
/// disclosure note offers the count as the evidence on which to "judge the
/// project on its technical merits". A figure produced for that purpose has to
/// be right, and one that contradicts itself two screens further up is worse
/// than none.
///
/// `CLAUDE.md` is the single source of truth (it is the file re-measured after
/// a run), so updating it is what this test asks for. Same shape as the MSRV
/// guard above, which exists because one version was promised in seven places.
#[test]
fn the_readme_test_counts_agree_with_each_other_and_with_claude_md() {
    let root = repo_root();
    let readme = std::fs::read_to_string(root.join("README.md")).expect("README.md");
    let claude = std::fs::read_to_string(root.join("CLAUDE.md")).expect("CLAUDE.md");

    let canonical_lib: u32 = claude
        .split("**")
        .find_map(|seg| seg.strip_suffix(" lib").and_then(|n| n.parse().ok()))
        .expect("CLAUDE.md must state the lib-test count as `**<n> lib**`");
    let canonical_integ: u32 = claude
        .split(" integration (")
        .next()
        .and_then(|s| {
            let tail: String = s.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
            tail.chars().rev().collect::<String>().parse().ok()
        })
        .expect("CLAUDE.md must state the integration count as `<n> integration (`");

    let found = readme_test_counts(&readme);
    assert!(
        !found.is_empty(),
        "README.md no longer states a test count in the `<n> lib tests + <n> integration tests`          form this guard reads. Either restore that wording or update the guard."
    );
    for (lib, integ) in &found {
        assert_eq!(
            (*lib, *integ),
            (canonical_lib, canonical_integ),
            "README.md claims {lib} lib + {integ} integration tests; CLAUDE.md says              {canonical_lib} + {canonical_integ}. The README states this figure in more than one              place and they must all match CLAUDE.md, which is the one re-measured after a run.              All README occurrences found: {found:?}"
        );
    }
}

/// The guard above must actually fire on the shape it forbids — including the
/// one rustfmt produces, which is what four guards in this file could not see
/// before 2026-08-30.
#[test]
fn the_whole_machine_sysinfo_guard_catches_the_shape_it_forbids() {
    let caught = |src: &str| !whole_machine_sysinfo_lines(src).is_empty();

    assert!(caught("let mut sys = System::new_all();"));
    assert!(caught("sys.refresh_all();"));
    assert!(caught("let mut sys = sysinfo::System::new_all();"));
    // Wrapped by rustfmt across lines — the form that blinded six guards.
    // A method chain is what rustfmt actually breaks, and `join_statement`
    // closes the " ." gap it leaves; it does NOT split a `::` path, so there
    // is no such shape to test for.
    assert!(caught(
        "self.shared_state\n    .system\n    .refresh_all();"
    ));
    assert!(caught("let mut sys =\n    System::new_all();"));

    // The targeted pattern every other site uses must stay quiet, or the guard
    // is noise and gets suppressed rather than obeyed.
    assert!(!caught(
        "let mut sys = System::new();\nsys.refresh_memory();"
    ));
    assert!(!caught(
        "sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);"
    ));
    // The same sweep spelled through `new_with_specifics`.
    assert!(caught(
        "System::new_with_specifics(sysinfo::RefreshKind::everything());"
    ));
    assert!(caught("let k = RefreshKind::everything();"));
    // ...but the NARROW kinds share that suffix and must not be caught. The
    // second is exactly right inside a pid-scoped `refresh_processes`.
    assert!(!caught(
        "sys.refresh_cpu_list(sysinfo::CpuRefreshKind::everything());"
    ));
    assert!(!caught(
        "sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), ProcessRefreshKind::everything());"
    ));
    // The one real `new_with_specifics` in the tree asks for nothing but a CPU
    // list, and must stay quiet.
    assert!(!caught(
        "System::new_with_specifics(RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing()));"
    ));
    // `Disks` genuinely needs its list refreshed and costs ~1 ms.
    assert!(!caught("sysinfo::Disks::new_with_refreshed_list();"));
    // A comment naming the call is not the call.
    assert!(!caught("// System::new_all() enumerates every process"));
}

/// `Content::read` with no `BufReader` or `Cursor` between them is the defect,
/// whatever anything is called. Naming was the first cut and it is only as good
/// as the naming — worse, it silently missed the `match File::open { Ok(mut f)
/// => Content::read(&mut f)` form, which is the shape one of the seven sites
/// actually had. A `Cursor` over a slice or an mmap is already in memory and
/// correctly passes; so does a reader built further up, which no longer has an
/// open file next to it.
///
/// "Opens a file" means `File::open` **or** `OpenOptions` — the second was
/// missing until 2026-08-30 and is the ordinary way to open a file when any
/// flag is wanted, so the guard was mute on a site it exists to catch. Both
/// directions are pinned below: a planted violation of each form must fail this
/// test, or it is only a guard against the one spelling someone happened to
/// think of.
#[test]
fn a_gguf_header_is_never_parsed_straight_off_an_unbuffered_file() {
    let root = repo_root();
    let mut offenders: Vec<String> = Vec::new();
    let mut stack = vec![root.join("src")];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            for line_no in unbuffered_gguf_parse_lines(&text) {
                offenders.push(format!(
                    "{}:{}",
                    p.strip_prefix(&root).unwrap_or(&p).display(),
                    line_no
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a GGUF header is being parsed off an unbuffered File — every tiny read \
         becomes a syscall, which cost 9.6 s of kernel time per \
         /api/admin/models request before this was caught:\n  {}\n\
         Use `inference::split::read_gguf_header(path)`, or wrap the handle in \
         a `BufReader` when the same handle is reused for tensor loads.",
        offenders.join("\n  ")
    );
}

/// Every pipeline coordinator that streams tokens must also send the TERMINAL
/// finish event.
///
/// The SSE encoder treats "no finish event arrived" as "this path never
/// streamed", and falls back to emitting the whole reply from the final
/// `InferenceOutput` — on top of the tokens it already sent. So a path that
/// streams but never finishes does not merely omit a marker, it DUPLICATES the
/// entire response.
///
/// Measured on the live swarm 2026-08-30 against the released v0.3.135:
/// `ngram_only_spec.rs` was the only coordinator missing it, and "Count 1 to 3"
/// came back as `1\n2\n3<|eot_id|>1\n2\n3` from every peer-held model with the
/// default settings. Its five siblings all had it, which is why nothing else
/// showed the fault — the one-invariant-N-paths shape this repo keeps paying
/// for.
#[test]
fn a_streaming_pipeline_path_sends_its_terminal_finish_event() {
    let dir = std::path::Path::new("src/inference/pipeline");
    let mut offenders = Vec::new();
    let mut checked = 0usize;
    for entry in std::fs::read_dir(dir).expect("pipeline dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).expect("read");
        let code = statements(&src)
            .into_iter()
            .map(|(_, s)| s)
            .collect::<Vec<_>>()
            .join("\n");
        // Only files that actually stream tokens are in scope. `StreamingTokenEvent`
        // is the one type used to push a token at a waiting client.
        if !code.contains("StreamingTokenEvent") {
            continue;
        }
        // `mod.rs` holds the shared emit helpers (`emit_streaming_batch`,
        // `emit_first_streaming_token`). They push content tokens by design;
        // ending the stream is the COORDINATOR's job, because only it knows
        // why generation stopped. Excluded on that basis, not to make the
        // scan pass.
        if path.file_name().and_then(|f| f.to_str()) == Some("mod.rs") {
            continue;
        }
        checked += 1;
        if !code.contains("finish_reason: Some(") {
            offenders.push(format!(
                "{} streams tokens but never sends a terminal finish event — the SSE \
                 encoder will re-emit the whole reply on top of the streamed tokens",
                path.display()
            ));
        }
    }
    assert!(
        checked >= 4,
        "expected to find several streaming pipeline paths, found {checked} — \
         the scan is no longer reaching them"
    );
    assert!(
        offenders.is_empty(),
        "streaming pipeline paths missing their terminal finish event:\n  {}",
        offenders.join("\n  ")
    );
}

/// Reply text reaches a streaming client through the shared emit helpers, never
/// through a hand-rolled `StreamingTokenEvent`.
///
/// Is the statement at `idx` inside an end-of-turn guard?
///
/// Looks back over the preceding statements for `!eos.contains(`, which is the
/// filter an inline decoder applies in place of the shared emit helpers. The
/// window is generous because the guard opens a loop body that does real work
/// (decode, stop-string search) before the send; it is bounded so a guard in a
/// different function cannot vouch for this one.
fn eos_guarded(stmts: &[(usize, String)], idx: usize) -> bool {
    const LOOKBACK: usize = 40;
    let start = idx.saturating_sub(LOOKBACK);
    stmts[start..idx]
        .iter()
        .any(|(_, s)| s.contains("!eos.contains("))
}

/// The end-of-turn lookback must see a guard that opens a loop body doing real
/// work before the send, and must NOT vouch for one too far away to be the same
/// block — otherwise the exclusion it replaces has just moved.
#[test]
fn the_eos_lookback_sees_a_real_guard_and_not_a_distant_one() {
    let stmt = |s: &str| (0usize, s.to_string());

    // The shape in `distributed.rs`: guard, then decode + stop-string work,
    // then the send.
    let mut near = vec![stmt("if !eos.contains(&tid) {")];
    for _ in 0..12 {
        near.push(stmt("let text = decoder.decode_tokens(&[tid]);"));
    }
    near.push(stmt(".send(StreamingTokenEvent {"));
    let idx = near.len() - 1;
    assert!(
        eos_guarded(&near, idx),
        "a guard 12 statements back must count"
    );

    // A guard in some earlier function cannot vouch for this send.
    let mut far = vec![stmt("if !eos.contains(&tid) {")];
    for _ in 0..80 {
        far.push(stmt("let unrelated = 1;"));
    }
    far.push(stmt(".send(StreamingTokenEvent {"));
    let idx = far.len() - 1;
    assert!(
        !eos_guarded(&far, idx),
        "a guard 80 statements away is a different block and must not count"
    );

    // No guard at all.
    let none = vec![stmt("let x = 1;"), stmt(".send(StreamingTokenEvent {")];
    assert!(!eos_guarded(&none, 1));
}

/// `emit_streaming_batch` / `emit_first_streaming_token` own the end-of-turn
/// filter. A coordinator that builds its own event with decoded text bypasses
/// that filter, which is how `<|eot_id|>` reached clients as reply text from
/// all three speculative paths at once (gotcha #414): each had copied the R105
/// truncation, which stops post-EOS junk but keeps EOS itself.
///
/// Terminal events (`text: String::new()`) are fine — they carry no content,
/// and sending one is separately required by
/// `a_streaming_pipeline_path_sends_its_terminal_finish_event`.
#[test]
fn streamed_reply_text_goes_through_the_shared_emit_helpers() {
    let dir = std::path::Path::new("src/inference/pipeline");
    let mut offenders = Vec::new();
    let mut checked = 0usize;
    for entry in std::fs::read_dir(dir).expect("pipeline dir") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let name = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
        // `mod.rs` DEFINES the helpers, so it is the one place that may build a
        // content event. `remote_generate.rs` forwards `t.text` verbatim from
        // the serving peer, which produced it through its own local executor —
        // there is no token id here to test, only text someone else decoded.
        // `distributed.rs` decodes inline and does not use the helper pattern
        // at all; it is checked against the guard it actually relies on
        // instead of being skipped — see `eos_guarded` below. This used to be
        // an exclusion with the hole written down beside it ("a NEW content
        // send added to that file outside its guard is not caught here"), and
        // a stated limitation is still a limitation.
        if name == "mod.rs" || name == "remote_generate.rs" {
            continue;
        }
        let decodes_inline = name == "distributed.rs";
        let src = std::fs::read_to_string(&path).expect("read");
        let stmts = statements(&src);
        for (i, (line, stmt)) in stmts.iter().enumerate() {
            if !stmt.contains("StreamingTokenEvent {") {
                continue;
            }
            checked += 1;
            // The `text:` field is within a couple of statements of the literal.
            let window: String = stmts[i..(i + 4).min(stmts.len())]
                .iter()
                .map(|(_, s)| s.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            let text_is_empty = window.contains("text: String::new()");
            if text_is_empty {
                continue;
            }
            // A file that decodes inline may carry reply text, but only from
            // inside the end-of-turn guard that does the filtering the helpers
            // would otherwise have done.
            if decodes_inline && eos_guarded(&stmts, i) {
                continue;
            }
            offenders.push(format!(
                "{}:{} builds a StreamingTokenEvent carrying reply text — \
                 route it through emit_streaming_batch so it inherits the \
                 end-of-turn filter{}",
                path.display(),
                line,
                if decodes_inline {
                    " (or put it inside the `!eos.contains` guard)"
                } else {
                    ""
                }
            ));
        }
    }
    assert!(
        checked >= 3,
        "expected several StreamingTokenEvent sites, found {checked} — scan is not reaching them"
    );
    assert!(
        offenders.is_empty(),
        "hand-rolled streaming content sends:\n  {}",
        offenders.join("\n  ")
    );
}

/// Does this statement compute a storage budget of its own, rather than ask
/// `model::auto_manage::storage_budget`?
///
/// The two spellings the codebase had grown: "half of the disk limit" and
/// "the auto-manage field if it is set". Both are the old rule, and any new
/// site reproducing either is a second accountant.
fn computes_its_own_storage_budget(statement: &str) -> bool {
    statement.contains("max_disk_mb / 2")
        || statement.contains("max_disk_mb/2")
        || statement.contains("max_storage_mb > 0")
}

/// The storage-budget guard must catch both spellings, wrapped or not, and
/// must not fire on the arithmetic-free reads that legitimately remain (the
/// config API clamps the field on write; the shard-storage listing prints it).
#[test]
fn the_storage_budget_guard_catches_a_second_accountant() {
    let half = "let budget_mb = live\n    .resources\n    .max_disk_mb / 2;";
    assert!(statements(half)
        .iter()
        .any(|(_, l)| computes_its_own_storage_budget(l)));
    let if_set =
        "let max = if shared.cfg().auto_manage.max_storage_mb > 0 {\n  x\n} else {\n  y\n};";
    assert!(statements(if_set)
        .iter()
        .any(|(_, l)| computes_its_own_storage_budget(l)));
    let clamp =
        "config.auto_manage.max_storage_mb = max_storage.clamp(1, MAX_AUTO_MANAGE_STORAGE_MB);";
    assert!(!statements(clamp)
        .iter()
        .any(|(_, l)| computes_its_own_storage_budget(l)));
    let print =
        "\"auto_manage_max_storage_mb\": state.shared_state.cfg().auto_manage.max_storage_mb,";
    assert!(!statements(print)
        .iter()
        .any(|(_, l)| computes_its_own_storage_budget(l)));
}

/// How much shard storage this node may hold is decided in ONE place —
/// `model::auto_manage::storage_budget` — and every consumer asks it.
///
/// There were three (gotcha #448). The download pass quartered the figure for
/// Minimal contribution; the prune pass's disk pressure and the pool page's
/// disk bar did not; the settings bar used the download rule but drew the cap
/// as headroom. A node holding 18 GB against a 50 GB setting was therefore OVER
/// budget for downloading and at 36% for pruning: it refused every download
/// and pruned nothing, indefinitely, and every surface a user could consult
/// said 50 GB. A tester built a careful phantom-reservation theory from that.
/// Two accountants for one disk wedge exactly where they disagree; a third
/// spelling of the rule anywhere in `src/` is that bug coming back.
#[test]
fn the_storage_budget_has_one_accountant() {
    let root = repo_root();
    let mut sources: Vec<(PathBuf, String)> = Vec::new();
    let mut stack = vec![root.join("src")];
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

    let mut offenders: Vec<String> = Vec::new();
    let mut helper_seen = false;
    for (path, text) in &sources {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        if rel == "src/model/auto_manage/mod.rs" {
            helper_seen = text.contains("pub fn storage_budget(");
            continue;
        }
        for (line_no, l) in statements(text) {
            if computes_its_own_storage_budget(&l) {
                offenders.push(format!("{rel}:{line_no}: {l}"));
            }
        }
    }
    assert!(
        helper_seen,
        "the single storage-budget accountant (`storage_budget`) has moved or been renamed; \
         update this guard so it keeps checking the right place"
    );
    assert!(
        offenders.is_empty(),
        "a storage budget is being computed outside `model::auto_manage::storage_budget`. \
         Ask `storage_budget_now(&state)` instead — two figures for one disk wedge the node \
         (gotcha #448).\n{}",
        offenders.join("\n")
    );
}

/// A retry only helps if the routing input has changed by the time it runs.
///
/// `dispatch_single` now re-runs the whole scheduler for an exhausted segment
/// (`should_retry_after` → `segment_ran_out_of_machines`), which is what makes
/// single-peer delegation's deliberate `standbys: vec![]` survivable. Without
/// barring the machines that just failed, that retry re-learns the same holders
/// and produces the identical plan — observed live, a peer that answered
/// `CUDA_ERROR_OUT_OF_MEMORY` was handed the same 34 layers again on the very
/// next attempt (gotcha #454).
///
/// The pairing is stated in `is_transient_remote_failure`'s own doc — *"the
/// blacklist is what makes the retry actually work"* — and a doc comment is
/// what this codebase keeps being caught by, so it is checked instead.
#[test]
fn an_exhausted_segment_bars_the_machines_that_just_failed_it() {
    let root = repo_root();
    let src = std::fs::read_to_string(root.join("src/inference/pipeline/distributed.rs"))
        .expect("read distributed.rs");
    let body = fn_body(&src, "fn failover_segment(").expect(
        "`failover_segment` has moved or been renamed; update this guard so it keeps checking          the right place",
    );
    assert!(
        exhaustion_arm_bars_its_tried_nodes(body),
        "the failover-exhaustion arm returns `SegmentFailoverExhausted` without calling          `blacklist_holder_for_request` for the nodes it tried. The router retries that error          by re-running the scheduler, so without the bar it re-picks the machine that just ran          out of memory (gotcha #454)."
    );
}

/// Is a `blacklist_holder_for_request` call present ahead of the
/// `SegmentFailoverExhausted` return, in the same arm?
///
/// Scanned over whole statements, not raw lines, so rustfmt wrapping the call
/// across four lines cannot retire the check (gotcha #413). The window is the
/// text before the return, since the bar has to happen first to be of any use.
fn exhaustion_arm_bars_its_tried_nodes(body: &str) -> bool {
    let Some(ret) = body.find("SegmentFailoverExhausted") else {
        return false;
    };
    statements(&body[..ret])
        .iter()
        .any(|(_, l)| l.contains("blacklist_holder_for_request("))
}

/// The guard above must be able to SEE the violation it exists to catch — a
/// scan that cannot fire is indistinguishable from one that finds nothing.
#[test]
fn the_exhaustion_guard_catches_a_return_that_bars_nobody() {
    let bad = r#"
            let Some(backup) = standby else {
                tracing::error!(failed_node = %failed_segment.node_id, "no standby");
                return Err(SwarmError::SegmentFailoverExhausted(exhausted_message(
                    failed_idx,
                    last_failure.as_deref(),
                )));
            };
"#;
    assert!(!exhaustion_arm_bars_its_tried_nodes(bad));

    // And it must still see the call when rustfmt has wrapped it.
    let good = r#"
            let Some(backup) = standby else {
                for node in std::iter::once(&failed_segment.node_id).chain(tried.iter()) {
                    self.shared_state
                        .blacklist_holder_for_request(request_id, node);
                }
                return Err(SwarmError::SegmentFailoverExhausted(exhausted_message(
                    failed_idx,
                    last_failure.as_deref(),
                )));
            };
"#;
    assert!(exhaustion_arm_bars_its_tried_nodes(good));
}

/// Lines that construct a `ShardAnnounce` by struct literal.
///
/// Comment lines are excluded: this file and the helper's own doc comment both
/// name the pattern, and a guard that trips over the documentation explaining
/// it is the trap a sibling test in this file already hit once.
fn bare_shard_announce_literals(src: &str) -> Vec<usize> {
    statements(src)
        .into_iter()
        .filter(|(_, s)| {
            let t = s.trim_start();
            !t.starts_with("//") && !t.starts_with("///") && !t.starts_with("*")
        })
        .filter(|(_, s)| s.contains("ShardAnnounce {"))
        .map(|(n, _)| n)
        .collect()
}

/// A shard announcement is built in ONE place (gotcha #406).
///
/// Every announcement must carry a per-shard build tag, so a receiver can tell
/// a holder of THIS GGUF build from a holder of a different one sharing the
/// same model id. Eight sites construct announcements, and a site that forgot
/// the field would send an announcement claiming nothing — indistinguishable
/// on the wire from an older peer, so the omission would be invisible rather
/// than merely wrong.
///
/// `model::manifest::shard_announce` is that place. Adding a field to
/// `ShardAnnounce` extends it, not the call sites — the same contract
/// `build_spec_verify_forward` keeps for `LayerForward`.
#[test]
fn shard_announce_is_built_in_one_place() {
    let root = repo_root();
    let mut stack = vec![root.join("src")];
    let mut offenders: Vec<String> = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            if !p.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            // The constructor itself.
            if rel == "src/model/manifest.rs" {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&p) else {
                continue;
            };
            for line in bare_shard_announce_literals(&text) {
                offenders.push(format!("{rel}:{line}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "ShardAnnounce must be built via model::manifest::shard_announce so every \
         announcement carries its per-shard build tag (gotcha #406). Bare literals at: {offenders:#?}"
    );
}

/// The guard above must be able to see the violation it exists to catch —
/// including the wrapped form rustfmt produces at depth, which is how four
/// other guards in this file came to be inert (gotcha #413).
#[test]
fn the_shard_announce_guard_catches_a_planted_literal() {
    let caught = |src: &str| !bare_shard_announce_literals(src).is_empty();

    assert!(caught(
        "let a = crate::types::ShardAnnounce { node_id, shards };"
    ));
    // Wrapped across lines by rustfmt at depth — `statements` rejoins it.
    assert!(caught(
        "let a = crate::types::SwarmMessage::ShardAnnounce(\n    crate::types::ShardAnnounce {\n        node_id,\n    },\n);"
    ));
    // The constructor call is the correct form and must stay quiet.
    assert!(!caught(
        "let a = crate::model::manifest::shard_announce(&reg, node_id, shards, vec![]);"
    ));
    // Matching on the enum variant is not construction.
    assert!(!caught("SwarmMessage::ShardAnnounce(announce) => {}"));
    // Prose naming the pattern is not the pattern.
    assert!(!caught(
        "// never write crate::types::ShardAnnounce { .. } here"
    ));
}

/// The frontend's core modules must actually BUILD their exports.
///
/// **v0.3.160 shipped a dashboard that never left "Connecting…"** on every
/// machine that updated (gotcha #488). A helper was added to the
/// `App.utils = { … }` export list while its definition sat inside another
/// function's body, so the name was not in scope when the object literal was
/// evaluated: `ReferenceError`, `App.utils` never assigned AT ALL, and every
/// component reading `App.utils.<anything>` died with it — `bindEvents` threw
/// before finishing, so nothing ever moved the page off its loading skeleton.
///
/// **Two checks could not see it, which is why this one evaluates the code.**
/// `node -c` passes: the file is syntactically perfect. And a source scan that
/// takes "module scope" to mean a particular indentation passes too — the
/// misplaced function was indented exactly like a top-level one, and the first
/// version of this guard was written that way and went GREEN against the
/// broken file. Only running it finds it.
///
/// **2026-09-12 — it now also CALLS a render function, because loading a module
/// cannot see a name that is only missing inside a function body.**
/// `dashboard.js::_renderHardware` read `data.network_traffic` while its
/// parameter was named `hw`: the file loads perfectly and throws
/// `ReferenceError: data is not defined` the first time the dashboard renders,
/// which is every two seconds on every node. Same failure as above, one level
/// deeper — and `node -c` passes on it too.
///
/// Skips (loudly) when `node` is absent so an infrastructure change cannot turn
/// a missing interpreter into a red build; GitHub runners ship one.
#[test]
fn the_frontend_core_modules_build_their_exports() {
    let root = repo_root();
    let script = root.join("examples/check_frontend_modules.js");
    assert!(
        script.exists(),
        "examples/check_frontend_modules.js is the only check that can catch a \
         frontend scope error — it must not be deleted"
    );
    let out = match std::process::Command::new("node").arg(&script).output() {
        Ok(o) => o,
        Err(e) => {
            eprintln!(
                "SKIPPED the_frontend_core_modules_build_their_exports: node not runnable ({e})"
            );
            return;
        }
    };
    assert!(
        out.status.success(),
        "a frontend core module failed to build its exports — the dashboard will \
         not start:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// **Both figures a peer ranks this node by must agree about where its models
/// run.** `ram_model_budget_mb` has keyed on `models_go_to_the_card` since
/// report #022 — a node that HAS a card and has been told not to use it still
/// gossips that card, because the card is really there, while its models load
/// into system memory. `est_tokens_per_sec_7b` never had the equivalent, so
/// such a node advertised its card's throughput: measured 35.6 tok/s broadcast
/// against the 4.95 its own scheduler was pricing the local candidate at.
///
/// That figure is GOSSIPED and `delegation_target` on the far side compares
/// against it, so the overstatement decides whether other machines hand it work.
/// The guard is here rather than as a unit test because the failure is the two
/// fields DRIFTING APART, which no single-field test can see.
#[test]
fn the_advertised_speed_and_memory_agree_about_where_models_run() {
    let src = std::fs::read_to_string("src/health/monitor.rs").expect("monitor.rs");
    let window = capability_speed_window(&src).expect(
        "the capability broadcast must still assign est_tokens_per_sec_7b — if this \
         moved, move the guard with it",
    );
    // Either inlined, or via a binding whose name says what it is — both are
    // the expression consulting the answer, which is the thing being pinned.
    assert!(
        window.contains("models_go_to_the_card") || window.contains("models_on_card"),
        "est_tokens_per_sec_7b must be keyed on where models actually run, the same \
         predicate ram_model_budget_mb uses, or a node told to stay off its card \
         advertises the card's speed to every peer.\n---\n{window}\n---"
    );
}

/// The window helper the guard above depends on: from the assignment to the end
/// of the expression that produces it, **with comment lines removed**.
///
/// The comments are not incidental. The first cut of this kept them, and the
/// prose above the assignment explains the rule by name — so the guard matched
/// its own documentation and passed with the code reverted. Caught by the null
/// control below, which is the second time in one session a guard has been
/// found passing for the wrong reason (gotcha #413, and #502).
fn capability_speed_window(src: &str) -> Option<String> {
    // The ASSIGNMENT EXPRESSION ONLY. Backing up to catch a predicate bound
    // just above it was the second wrong version: the binding line itself
    // mentions the predicate, so deleting its USE left the guard still
    // matching. What has to be proved is that the expression consults the
    // answer, not that the answer was computed nearby.
    let at = src.find("let est_tokens_per_sec_7b")?;
    let rest = &src[at..];
    let end = rest.find("\n\n").unwrap_or(rest.len());
    Some(
        rest[..end]
            .lines()
            .filter(|l| !l.trim_start().starts_with("//"))
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

/// The guard is only worth having if it can see the defect it exists for
/// (gotcha #413): a version keyed on the card's mere presence must fail it.
#[test]
fn the_advertised_speed_guard_catches_the_card_presence_form() {
    // The defect's REAL shape: prose that names the rule sitting above code that
    // does not apply it. Anything less than this is not the thing to plant —
    // the first version of the window matched the comment and passed.
    // The defect's REAL shape, and both ways an earlier version of this guard
    // was fooled: a comment that names the rule, AND the predicate computed on
    // the line above without the expression ever consulting it.
    let planted = "fn build() {\n\n    // Keyed on models_go_to_the_card, the same predicate the memory field uses.\n    let models_on_card = vram::models_go_to_the_card(&s);\n    let est_tokens_per_sec_7b = gpu_info\n        .as_ref()\n        .map(|g| estimate(g.memory_bandwidth_gbps, true))\n        .unwrap_or_else(|| estimate(ram_gbps(), false));\n\n}\n";
    let window = capability_speed_window(planted).expect("the planted form is found");
    assert!(
        !window.contains("models_go_to_the_card") && !window.contains("models_on_card"),
        "the scanner must report the pre-fix form as failing even when a comment \
         names the rule and the predicate is computed beside it, or it proves \
         nothing:\n---\n{window}\n---"
    );

    // ...and the real file's form passes, so the window is not simply always empty.
    let real = std::fs::read_to_string("src/health/monitor.rs").expect("monitor.rs");
    let real_window = capability_speed_window(&real).expect("found");
    assert!(
        real_window.contains("models_go_to_the_card") || real_window.contains("models_on_card"),
        "and the real file's form passes, so the window is not simply always \
         empty:\n---\n{real_window}\n---"
    );
}

/// Does this `failover_segment` body ask whether a stand-in could reproduce the
/// failed machine, BEFORE it goes looking for one?
fn failover_checks_state_before_searching(body: &str) -> bool {
    let Some(check) = body.find("failover_can_restore_state(") else {
        return false;
    };
    // The standby search is the thing that must not be reached first.
    match body.find("self\n                .assignment\n                .standbys") {
        Some(search) => check < search,
        // No search found (renamed or reshaped) — the ordering claim cannot be
        // made, so require only that the question is asked at all.
        None => true,
    }
}

/// Is the request's trace registered BEFORE the scheduler is asked for a route?
///
/// The scheduler writes its cost prediction through `active_traces`, keyed by
/// request id, and the write is a no-op when there is no entry. So the insert
/// has to precede the assembly call or every prediction is silently discarded.
fn trace_is_registered_before_scheduling(body: &str) -> bool {
    let insert = body.find("active_traces.insert(");
    let assemble = body
        .find("assemble_awaiting_dht(")
        .into_iter()
        .chain(body.find("assemble_pipeline_for("))
        .min();
    match (insert, assemble) {
        (Some(i), Some(a)) => i < a,
        // No assembly here means nothing to order against; no insert at all is
        // a different defect and the caller asserts on it separately.
        _ => false,
    }
}

/// The scheduler's cost prediction reaches the trace it is written into.
///
/// `note_predicted_route_cost` finds the trace by request id and does nothing
/// when the request is not in `active_traces`. `execute_request` used to insert
/// the trace AFTER assembling the pipeline, so every prediction the scheduler
/// recorded went nowhere — and this instrument had already shipped inert once
/// before (`45d11f56` wired five more of `assemble_pipeline_for`'s six returns,
/// all of which still wrote into a map the request was not yet in).
///
/// Measured on the live node 2026-09-09: v0.3.166, five peers, five hours of
/// uptime, a deliberately remote request that completed normally, and
/// `predicted_ms` absent from its own completion line.
///
/// Asserts on the ORDER in the source, because the unit test beside the
/// scheduler cannot: it inserts the trace itself before calling the scheduler,
/// which manufactures exactly the precondition production was failing to
/// provide (gotcha #502).
#[test]
fn the_route_prediction_reaches_a_trace_that_is_already_registered() {
    let root = repo_root();
    let rel = "src/inference/router/distributed_exec.rs";
    let src = std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
    let body = fn_body(&src, "async fn execute_request(")
        .unwrap_or_else(|| panic!("{rel}: execute_request not found — has it been renamed?"));

    assert!(
        body.contains("active_traces.insert("),
        "{rel}: `execute_request` no longer registers the request's trace at all — \
         the completion line and every prediction written into it are lost."
    );
    assert!(
        trace_is_registered_before_scheduling(body),
        "{rel}: `execute_request` registers the trace AFTER asking the scheduler for a \
         route.\n`note_predicted_route_cost` is a no-op for a request that is not in \
         `active_traces`, so every prediction the scheduler records is silently \
         discarded — which is how this instrument shipped inert twice.\nSee \
         docs/FUTURE_WORK.md item 3."
    );
}

/// The guard above must see the violation in the form it really took: the
/// insert present, but after the assembly.
#[test]
fn the_trace_registration_guard_catches_an_insert_after_assembly() {
    let violation = r#"
        let assignment = assemble_awaiting_dht(&scheduler, model_id, &local, request.id).await?;
        shared_state.active_pipelines.insert(request.id, assignment.clone());
        shared_state.active_traces.insert(request.id, trace.clone());
    "#;
    assert!(
        !trace_is_registered_before_scheduling(violation),
        "the guard cannot see a trace registered after the scheduler ran"
    );

    let correct = r#"
        shared_state.active_traces.insert(request.id, trace.clone());
        let assignment = assemble_awaiting_dht(&scheduler, model_id, &local, request.id).await?;
    "#;
    assert!(
        trace_is_registered_before_scheduling(correct),
        "the guard must accept the correct order"
    );
}

/// A reply already under way is never moved to a machine that cannot continue it.
///
/// `failover_segment` re-sends the current step alone, and the KV cache is keyed
/// by `(layer range, request id)` — so a stand-in holds nothing for this segment
/// and nothing rebuilds it. `split::executor` then takes `kv_offset` from the
/// cache rather than from `index_pos`, and no check disagrees, so the
/// replacement answers from the current token alone while the reply carries on.
///
/// Measured in `examples/failover_kv_probe.rs` on llama-3.2-3b: replacing 4 of
/// 28 layers takes the probability of the token the healthy machine would have
/// chosen from 0.997 to 0.119; half the model takes it to 0.005. Nothing errors
/// and nothing warns, which is why only a guard keeps it fixed.
///
/// The unit tests beside the predicate cannot pin this: they would all still
/// pass with the call removed from `failover_segment`, which is the one edit
/// that reintroduces the defect (gotcha #502 — assert the PATH, not the
/// outcome).
#[test]
fn a_reply_under_way_is_never_moved_to_a_machine_that_cannot_continue_it() {
    let root = repo_root();
    let rel = "src/inference/pipeline/distributed.rs";
    let src = std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
    let body = fn_body(&src, "    async fn failover_segment(")
        .unwrap_or_else(|| panic!("{rel}: failover_segment not found — has it been renamed?"));

    assert!(
        failover_checks_state_before_searching(body),
        "{rel}: `failover_segment` no longer asks `failover_can_restore_state` before \
         looking for a stand-in.\nA mid-reply takeover computes from the current token \
         alone — measured at P(healthy machine's token) = 0.119 replacing just 4 of 28 \
         layers — and nothing errors or warns.\nSee docs/FUTURE_WORK.md § \"A failover \
         after the prompt pass silently loses the failed segment's KV context\"."
    );
}

/// Does this `failover_segment` body send the replayed payload at the replayed
/// position, rather than the bare current step?
///
/// Both must move together. A replay carries every position from 0, so it is a
/// prompt pass to a stand-in with no cache; sending those bytes while still
/// naming the CURRENT `index_pos` would rotate every position wrongly and
/// produce a fluent, wrong continuation with nothing erroring — the same shape
/// of silent defect the replay exists to remove.
fn failover_sends_replay_and_position_together(body: &str) -> bool {
    // Every send of a payload in this function must use the replay-aware
    // binding, never the raw `activations` it was handed.
    //
    // Matched on an identifier BOUNDARY, not as a substring: `activations` is a
    // suffix of `send_activations`, so a plain `contains` can never tell the
    // two apart. The first version of this guard did exactly that and could not
    // fail; its null control is what said so (gotcha #502, and #503 on planting
    // the violation in the form it would really take).
    let raw_send = |hay: &str| -> bool {
        let needle = "activations.to_vec(";
        let bytes = hay.as_bytes();
        let mut from = 0usize;
        while let Some(rel) = hay[from..].find(needle) {
            let at = from + rel;
            let prev = if at == 0 { b' ' } else { bytes[at - 1] };
            // An identifier character before it means this is the tail of a
            // longer name, e.g. `send_activations`; anything else is the bare
            // binding, which is the defect.
            if !(prev == b'_' || prev.is_ascii_alphanumeric()) {
                return true;
            }
            from = at + needle.len();
        }
        false
    };
    !raw_send(body) && body.contains("replay_index_pos")
}

/// The replayed payload and the replayed position move together.
///
/// `assemble_replay` builds a forward covering positions `0..=current`, which is
/// only correct if it is SENT at position 0 — a stand-in holds no cache, so that
/// is what a fresh cache makes it. Sending it while naming the current position
/// instead rotates every position wrongly, and the reply stays fluent while
/// ceasing to be the model's, which is precisely the failure the replay was
/// built to end.
///
/// Asserts on the call site rather than on `assemble_replay`, whose own tests
/// would all still pass with the send left on the old binding (gotcha #502).
#[test]
fn a_replayed_failover_is_sent_from_the_position_it_was_assembled_for() {
    let root = repo_root();
    let rel = "src/inference/pipeline/distributed.rs";
    let src = std::fs::read_to_string(root.join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"));
    let body = fn_body(&src, "    async fn failover_segment(")
        .unwrap_or_else(|| panic!("{rel}: failover_segment not found — has it been renamed?"));

    assert!(
        failover_sends_replay_and_position_together(body),
        "{rel}: `failover_segment` sends a payload without the replay-aware binding, \
         or no longer names `replay_index_pos`.\nA replay covers positions 0..=current \
         and is only correct sent at position 0; sending it at the current position \
         rotates every position wrongly and the reply stays fluent while ceasing to be \
         the model's.\nSee `assemble_replay` and docs/invariants/scheduling.md."
    );
}

/// The guard above must SEE the violation, planted in the form it would really
/// take: the replay assembled and then sent on the old binding.
#[test]
fn the_replay_position_guard_catches_a_send_on_the_old_binding() {
    let violation = r#"
        let (replay_payload, replay_index_pos) = match replay_history {
            Some(ref history) => (Some(assemble_replay(history, activations)?), 0u32),
            None => (None, index_pos as u32),
        };
        loop {
            let forward = LayerForward {
                index_pos: index_pos as u32,
                activations: activations.to_vec(),
            };
        }
    "#;
    assert!(
        !failover_sends_replay_and_position_together(violation),
        "the guard cannot see a send left on the raw `activations` binding"
    );

    let correct = r#"
        let (replay_payload, replay_index_pos) = match replay_history {
            Some(ref history) => (Some(assemble_replay(history, activations)?), 0u32),
            None => (None, index_pos as u32),
        };
        let send_activations: &[u8] = replay_payload.as_deref().unwrap_or(activations);
        loop {
            let forward = LayerForward {
                index_pos: replay_index_pos,
                activations: send_activations.to_vec(),
            };
        }
    "#;
    assert!(
        failover_sends_replay_and_position_together(correct),
        "the guard must accept the correct shape"
    );
}

/// The guard above must be able to SEE the violation it exists to catch, in the
/// form it would really take: the check deleted, the search left in place.
#[test]
fn the_mid_reply_failover_guard_catches_a_body_that_searches_first() {
    let violation = r#"
    async fn failover_segment(
        &mut self,
        failed_idx: usize,
    ) -> Result<LayerResult, SwarmError> {
        loop {
            let standby = self
                .assignment
                .standbys
                .iter()
                .find(|s| standby_covers(s, failed_segment.layer_range))
                .cloned();
        }
}
"#;
    assert!(!failover_checks_state_before_searching(violation));

    // And the ordering half: asking only AFTER the search is no guard at all.
    let too_late = r#"
    async fn failover_segment(
        &mut self,
    ) -> Result<LayerResult, SwarmError> {
        loop {
            let standby = self
                .assignment
                .standbys
                .iter()
                .find(|s| standby_covers(s, failed_segment.layer_range))
                .cloned();
            if !failover_can_restore_state(sequence_num) {
                return Err(oops);
            }
        }
}
"#;
    assert!(!failover_checks_state_before_searching(too_late));

    let good = r#"
    async fn failover_segment(
        &mut self,
    ) -> Result<LayerResult, SwarmError> {
        if !failover_can_restore_state(sequence_num) {
            return Err(oops);
        }
        loop {
            let standby = self
                .assignment
                .standbys
                .iter()
                .find(|s| standby_covers(s, failed_segment.layer_range))
                .cloned();
        }
}
"#;
    assert!(failover_checks_state_before_searching(good));
}

/// `CLAUDE.md` states the vendored request-response crate's test count in TWO
/// places, and on 2026-09-09 they disagreed: the repository tree said 9, the
/// Testing section said 11, and the truth was 11.
///
/// That figure has no mechanical guard while the lib and integration counts do,
/// so it drifted silently for however long — and it is not a cosmetic number.
/// A workspace-root `cargo test` does NOT compile this crate; it is run as its
/// own step (`--manifest-path vendor/libp2p-request-response/Cargo.toml --lib`)
/// by CI and by the pre-push hook. A reader who believes there are 9 tests has
/// a wrong idea of what that separate step is protecting — which is the
/// SwarmLLM patch to connection selection, the thing gotcha #356 exists about.
///
/// Counted from the source rather than by running it, so this stays a cheap
/// repo-consistency check. Three things the count has to get right, all of
/// which a bare `text.matches("#[test]")` over `read_dir` gets wrong — the
/// "spellings it knows" rule applies to this guard as much as to the ones it
/// sits beside:
///
/// * **Async tests spell it differently.** `json.rs` and `cbor.rs` already use
///   `#[tokio::test]`, so a literal `#[test]` search undercounts the moment an
///   async test lands in `lib.rs`, and the guard then fails on correct docs.
/// * **`src/` has subdirectories.** `src/handler/protocol.rs` is invisible to a
///   non-recursive `read_dir`.
/// * **`--lib` builds with DEFAULT features.** `cbor` and `json` are opt-in and
///   their modules are `#[cfg(feature = ...)]`-gated in `lib.rs`, so their tests
///   are compiled by neither CI nor the pre-push hook. Counting them would make
///   the guard demand a number nobody can observe.
#[test]
fn the_vendored_request_response_test_count_is_stated_once_and_correctly() {
    let root = repo_root();
    let src = root.join("vendor/libp2p-request-response/src");
    let lib_rs = std::fs::read_to_string(src.join("lib.rs")).expect("vendored lib.rs");
    let gated = feature_gated_modules(&lib_rs);
    assert!(
        !gated.is_empty(),
        "expected `cbor`/`json` to be feature-gated in the vendored lib.rs — if that \
         changed, this guard's default-feature assumption needs re-checking"
    );

    let mut actual = 0usize;
    let mut stack = vec![src.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("vendored crate src/ must exist") {
            let path = entry.expect("read_dir entry").path();
            let stem = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            // A gated module contributes neither `src/<name>.rs` nor `src/<name>/`.
            if dir == src && gated.contains(&stem) {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("rs") {
                continue;
            }
            let text = std::fs::read_to_string(&path).expect("read vendored source");
            actual += count_test_attributes(&text);
        }
    }
    assert!(
        actual > 0,
        "found no #[test] in vendor/libp2p-request-response/src — the crate or this \
         guard's assumptions have changed"
    );

    // The annotated source tree moved to docs/ARCHITECTURE.md on 2026-09-16 (CLAUDE.md
    // loads every session). The property this guard exists for is unchanged: the count
    // is stated INDEPENDENTLY in two documents, so drift between them is caught.
    //
    // Both needles are matched against WHITESPACE-COLLAPSED text. These are prose files:
    // a markdown reflow that moves a line break into the middle of the phrase changes
    // nothing a reader would notice, and broke this guard the day the tree moved. Same
    // lesson as `statements()` for Rust — scan the statement, not the line.
    let squash = |s: &str| s.split_whitespace().collect::<Vec<_>>().join(" ");
    let claude = squash(&std::fs::read_to_string(root.join("CLAUDE.md")).expect("CLAUDE.md"));
    let arch = squash(
        &std::fs::read_to_string(root.join("docs/ARCHITECTURE.md")).expect("docs/ARCHITECTURE.md"),
    );
    let claims: Vec<usize> = [
        // tree line: "libp2p-request-response/ (<n> tests, `--lib`)"
        arch.split("libp2p-request-response/ (")
            .nth(1)
            .and_then(|s| s.split(" tests").next())
            .and_then(|n| n.trim().parse().ok()),
        // Testing section: "<n> in the vendored request-response patch"
        claude
            .split(" in the vendored request-response patch")
            .next()
            .and_then(|s| {
                let tail: String = s.chars().rev().take_while(|c| c.is_ascii_digit()).collect();
                tail.chars().rev().collect::<String>().parse().ok()
            }),
    ]
    .into_iter()
    .flatten()
    .collect();

    assert_eq!(
        claims.len(),
        2,
        "The vendored test count must be stated in BOTH docs/ARCHITECTURE.md's source tree \
         (`libp2p-request-response/ (<n> tests, ...`) and CLAUDE.md's Testing section \
         (`<n> in the vendored request-response patch`). Found: {claims:?}"
    );
    for claimed in &claims {
        assert_eq!(
            *claimed, actual,
            "The docs claim {claimed} tests in the vendored request-response patch; \
             the crate has {actual}. Both occurrences must match the source. All \
             claims found: {claims:?}"
        );
    }
}

/// Every attribute that makes `cargo test --lib` run a function: `#[test]` and
/// the runtime-qualified forms (`#[tokio::test]`, `#[async_std::test]`, …).
/// Pinned by `the_vendored_test_counter_sees_every_spelling_of_a_test`.
fn count_test_attributes(src: &str) -> usize {
    let mut n = 0;
    for line in src.lines() {
        let l = line.trim();
        if !l.starts_with("#[") || !l.ends_with("test]") {
            continue;
        }
        let inner = &l[2..l.len() - 1];
        // `test` itself, or `<path>::test` — but not `#[cfg(test)]`, which does
        // not end in `test]`, nor `#[should_panic]`-style unrelated attributes.
        if inner == "test" || inner.ends_with("::test") {
            n += 1;
        }
    }
    n
}

/// Module names `lib.rs` declares behind a `#[cfg(feature = "…")]`. Their tests
/// are not compiled by a default-feature `--lib` run, so they must not be
/// counted. Pinned by `the_vendored_test_counter_sees_every_spelling_of_a_test`.
fn feature_gated_modules(lib_rs: &str) -> Vec<String> {
    let lines: Vec<&str> = lib_rs.lines().collect();
    let mut out = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if !line.trim_start().starts_with("#[cfg(feature") {
            continue;
        }
        let Some(next) = lines.get(i + 1) else {
            continue;
        };
        let decl = next.trim();
        let Some(rest) = decl
            .strip_prefix("pub mod ")
            .or_else(|| decl.strip_prefix("mod "))
        else {
            continue;
        };
        if let Some(name) = rest.split([';', ' ', '{']).next() {
            if !name.is_empty() {
                out.push(name.to_string());
            }
        }
    }
    out
}

/// The guard above must fire on the shapes it forbids, planted in the form they
/// would really appear — a scan that finds nothing is indistinguishable from a
/// scan that CANNOT find anything (`.claude/rules/architecture.md` § "A
/// source-scanning guard is only as good as the spellings it knows").
#[test]
fn the_vendored_test_counter_sees_every_spelling_of_a_test() {
    // The plain form, indented as it really is inside `mod tests`.
    assert_eq!(count_test_attributes("    #[test]\n    fn a() {}\n"), 1);
    // The async form that `json.rs` and `cbor.rs` already use, and which the
    // first version of this guard could not see at all.
    assert_eq!(
        count_test_attributes("    #[tokio::test]\n    async fn a() {}\n"),
        1
    );
    assert_eq!(
        count_test_attributes("    #[async_std::test]\n    async fn a() {}\n"),
        1
    );
    // Things that are NOT a test function must not be counted.
    assert_eq!(count_test_attributes("#[cfg(test)]\nmod tests {}\n"), 0);
    assert_eq!(count_test_attributes("#[should_panic]\nfn a() {}\n"), 0);
    assert_eq!(
        count_test_attributes("// #[test] in a comment is still one\n"),
        0
    );
    // Mixed, the way a real file reads.
    assert_eq!(
        count_test_attributes("#[cfg(test)]\nmod t {\n    #[test]\n    fn a() {}\n    #[tokio::test]\n    async fn b() {}\n}\n"),
        2
    );

    // Feature-gated modules are excluded, in the exact shape the vendored
    // lib.rs writes them.
    let gated = feature_gated_modules(
        "#[cfg(feature = \"cbor\")]\npub mod cbor;\nmod codec;\nmod handler;\n#[cfg(feature = \"json\")]\npub mod json;\n",
    );
    assert_eq!(gated, vec!["cbor".to_string(), "json".to_string()]);
    // An ungated module is not excluded.
    assert!(feature_gated_modules("mod codec;\npub mod handler;\n").is_empty());
}

/// Recursively collect every `.rs` file under a directory.
fn rust_files_under(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|e| e == "rs") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// A peer earns trust for work that was CHECKED, and only for that.
///
/// Issue #21 (2026-09-10): `distributed_exec` credited `InferenceSuccess` to
/// every remote segment holder the moment the pipeline returned `Ok`, and only
/// then sampled a 5% check. Expected movement for a peer whose output is always
/// degenerate was `+0.01 + 0.05 x (-0.1) = +0.005` a request — it climbed to
/// the 1.0 ceiling, over the bar deciding who may read a plaintext prompt.
///
/// Two ways to bring that back, so the guard checks both: crediting from a NEW
/// path, and hoisting the credit back out of the `WellFormed` arm.
#[test]
fn inference_trust_is_credited_only_behind_the_well_formed_verdict() {
    let root = repo_root();
    const HOME: &str = "src/inference/router/spot_check.rs";

    let mut offenders = Vec::new();
    let mut credit_sites = 0usize;
    for path in rust_files_under(&root.join("src")) {
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        // The variant's own definition and the TrustManager unit tests name it
        // without paying anybody.
        if rel.ends_with("src/credit/trust.rs") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap_or_default();
        for (line, stmt) in statements(&src) {
            if !stmt.contains("TrustEvent::InferenceSuccess") {
                continue;
            }
            credit_sites += 1;
            if !rel.ends_with(HOME) {
                offenders.push(format!("{rel}:{line}  {stmt}"));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "TrustEvent::InferenceSuccess is credited outside {HOME}:\n  {}\n\n\
         Trust is paid for a result that `check_distributed_result` judged \
         well-formed, through `settle_participant_trust`. A second path that \
         pays for merely not erroring re-creates issue #21, in which a peer \
         returning garbage climbed to the trust ceiling.",
        offenders.join("\n  ")
    );

    // The scanner must be able to see the site it is guarding — a scan that
    // finds nothing is indistinguishable from one that cannot find anything.
    assert!(
        credit_sites >= 1,
        "the scanner found no InferenceSuccess credit anywhere; it has gone \
         blind (renamed variant? statements() no longer joining the chain?)"
    );

    // And the surviving site must sit behind the verdict, not above the match.
    let src = std::fs::read_to_string(root.join(HOME)).expect("read spot_check.rs");
    let body = fn_body(&src, "pub(super) fn settle_participant_trust(")
        .expect("settle_participant_trust must exist");
    let arm = body
        .find("ResultCheck::WellFormed =>")
        .expect("the WellFormed arm must exist");
    let credit = body
        .find("TrustEvent::InferenceSuccess")
        .expect("the credit must exist");
    assert!(
        credit > arm,
        "{HOME} credits InferenceSuccess before the WellFormed arm — that is \
         the pre-fix ordering, in which every participant was paid and the \
         check merely ran afterwards"
    );
}

/// A workflow step never carries both `uses:` and `shell:`, or `uses:` and `run:`.
///
/// GitHub rejects the whole file for either — `Unexpected value 'shell'`, and the
/// action fails to load before a single step runs. **Nothing local catches it.**
/// `actionlint` does not validate composite action files at all (it reads them as
/// workflows and complains that `on:` and `jobs:` are missing), and a YAML parse
/// only checks syntax, not the Actions schema.
///
/// So the only signal is a workflow that actually loads the file — and
/// `.github/actions/gpu-build-env/action.yml` is loaded by `cache-warm.yml` and
/// `release.yml` and by NOTHING in CI. On 2026-09-11 a conversion left a
/// `shell: bash` on a step that had become a `uses:`, CI went green across all
/// 14 jobs, and all three Cache warm jobs died at load. **A release tagged in
/// that window would have failed every GPU build**, which is the same shape as
/// the `cfg`-gated GPU code CLAUDE.md warns about: the path with the least
/// coverage is the one the release depends on.
#[test]
fn no_workflow_step_mixes_uses_with_shell_or_run() {
    let root = repo_root();
    let mut offenders = Vec::new();
    let mut steps_seen = 0usize;
    let mut dirs = vec![root.join(".github")];
    while let Some(d) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                dirs.push(p);
                continue;
            }
            if !p.extension().is_some_and(|e| e == "yml" || e == "yaml") {
                continue;
            }
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            let src = std::fs::read_to_string(&p).unwrap_or_default();
            let lines: Vec<&str> = src.lines().collect();
            let mut i = 0usize;
            while i < lines.len() {
                let indent = lines[i].len() - lines[i].trim_start().len();
                if !lines[i].trim_start().starts_with("- ") {
                    i += 1;
                    continue;
                }
                // A step block: this `- ` line plus every following line indented
                // deeper than it. A blank line does not end it; a `- ` or a
                // dedent does.
                let start = i;
                let mut j = i + 1;
                while j < lines.len() {
                    let l = lines[j];
                    if l.trim().is_empty() {
                        j += 1;
                        continue;
                    }
                    let ind = l.len() - l.trim_start().len();
                    if ind <= indent {
                        break;
                    }
                    j += 1;
                }
                // Keys at the step's own level: the `- ` line's own key, plus
                // lines indented exactly two further (`  key:`).
                let mut has_uses = false;
                let mut has_shell = false;
                let mut has_run = false;
                for l in &lines[start..j] {
                    let t = l.trim_start();
                    let t = t.strip_prefix("- ").unwrap_or(t);
                    if t.starts_with('#') {
                        continue;
                    }
                    let ind = l.len() - l.trim_start().len();
                    // Only the step's own keys, not keys nested under `with:`.
                    if ind > indent + 2 {
                        continue;
                    }
                    if t.starts_with("uses:") {
                        has_uses = true;
                    }
                    if t.starts_with("shell:") {
                        has_shell = true;
                    }
                    if t.starts_with("run:") {
                        has_run = true;
                    }
                }
                if has_uses {
                    steps_seen += 1;
                    if has_shell {
                        offenders.push(format!(
                            "{rel}:{}  `uses:` step also sets `shell:`",
                            start + 1
                        ));
                    }
                    if has_run {
                        offenders.push(format!(
                            "{rel}:{}  step sets both `uses:` and `run:`",
                            start + 1
                        ));
                    }
                }
                i = j;
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "a step mixes `uses:` with `shell:`/`run:`, which makes GitHub refuse the \
         whole file at load:\n  {}\n\n\
         A `uses:` step runs an action and takes no shell. Drop the `shell:` (and \
         split the step if it also needs a `run:`). Note that neither actionlint \
         nor a YAML parse can see this, and gpu-build-env is loaded only by \
         cache-warm and release — so CI going green means nothing here.",
        offenders.join("\n  ")
    );

    // The scan has to actually reach steps that use actions.
    assert!(
        steps_seen >= 10,
        "the scanner found only {steps_seen} `uses:` steps under .github — \
         it is not parsing step blocks correctly"
    );
}

/// Every apt command in CI goes through a retrying composite action.
///
/// A bare `sudo apt-get update` fails the whole job when ANY repository on the
/// runner image has a bad index — including third-party ones nothing here
/// installs from. On 2026-09-09 all three Linux release jobs died on the
/// runner's Google Chrome index and v0.3.168 published as a DRAFT with 12 of
/// its 25 assets, with nothing wrong with the code. The retry lives in one
/// composite action; this stops the one-liner coming back beside it.
///
/// **`apt-get install` is checked the same way, and was the gap this guard had.**
/// The retry was on `update` only, so six bare installs sat behind it — the half
/// that downloads tens of megabytes, up to ~3 GB for the CUDA toolkit, reaching
/// the same servers that fail the same transient ways (gotcha #544). A guard that
/// covers the cheap command and not the expensive one describes the fix that was
/// made rather than the invariant that was wanted.
#[test]
fn every_apt_command_goes_through_a_retrying_action() {
    let root = repo_root();
    // The two sanctioned homes for a bare apt command: the actions that exist to
    // retry them. Each one necessarily contains the command it wraps.
    // Written on a line that is already inside a retry loop. See the scan below.
    const RETRY_LOOP_MARKER: &str = "apt-retry-loop";
    const HOMES: [&str; 2] = [
        ".github/actions/apt-update/action.yml",
        ".github/actions/apt-install/action.yml",
    ];

    let mut offenders = Vec::new();
    let mut checked_files = 0usize;
    let mut dirs = vec![root.join(".github")];
    while let Some(d) = dirs.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                dirs.push(p);
                continue;
            }
            if !p.extension().is_some_and(|e| e == "yml" || e == "yaml") {
                continue;
            }
            let rel = p
                .strip_prefix(&root)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            if HOMES.iter().any(|h| rel.ends_with(h)) {
                continue;
            }
            checked_files += 1;
            let src = std::fs::read_to_string(&p).unwrap_or_default();
            for (i, line) in src.lines().enumerate() {
                let l = line.trim();
                if l.starts_with('#') {
                    continue;
                }
                // One sanctioned exception, and it has to be written at the
                // site: a composite action cannot `uses:` another one from
                // inside a `run:` block, and `gpu-build-env` computes its
                // package name, so its CUDA-toolkit install carries its own
                // three-attempt loop. Such a line must say so with the marker
                // below — an EXPLICIT, greppable opt-out rather than this
                // scanner trying to recognise a loop by its shape, which a
                // bare command could accidentally match.
                if l.contains(RETRY_LOOP_MARKER) {
                    continue;
                }
                if l.contains("apt-get update") || l.contains("apt-get install") {
                    offenders.push(format!("{rel}:{}  {l}", i + 1));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "a bare apt command is back in CI:\n  {}\n\n\
         Use `uses: ./.github/actions/apt-update` to refresh and \
         `uses: ./.github/actions/apt-install` to install. A single bad index on \
         any repository the runner image ships fails the whole job, and that \
         published v0.3.168 as a draft with 12 of 25 assets. Where a composite \
         action cannot be called — inside another composite's `run:` block — wrap \
         the command in the same three-attempt loop instead.",
        offenders.join("\n  ")
    );

    // The scan has to actually reach the workflow files.
    assert!(
        checked_files >= 3,
        "the scanner found only {checked_files} workflow files under .github — \
         it is not looking where the workflows are"
    );
    for home in HOMES {
        assert!(
            root.join(home).is_file(),
            "{home} is missing, so the call sites point at nothing"
        );
    }
}

/// Whether prompt privacy is in force is asked in ONE place, and it is not the
/// per-model map.
///
/// `SharedState::encrypted_pipeline_for` is the single answer. Reading
/// `encrypted_pipeline_models` directly sees only an EXPLICIT per-model toggle
/// and misses both automatic cases — an explicit global `encrypted_pipeline`,
/// and the default `encrypted_pipeline_auto`, which has switched privacy on
/// wherever a node holds both ends of a model since 2026-07-27.
///
/// Auto-manage prune did exactly that. Its skip looked like a guard and read
/// like one; it protected models a user had toggled by hand and silently left
/// unprotected the ones privacy was actually in force for, so prune would delete
/// an end shard and strand the setting — every request for that model then
/// failing at pipeline assembly, which is the state a live node reached on
/// 2026-08-09.
///
/// Two files may touch the map: `daemon/state` owns it, and the admin lifecycle
/// endpoints deliberately expose the EXPLICIT value as distinct from the
/// effective one. Everywhere else asks `encrypted_pipeline_for`, or
/// `privacy_explicitly_enabled_for` where the deliberateness is the point.
#[test]
fn prompt_privacy_is_never_re_derived_from_the_per_model_map() {
    let allowed = [
        // Owns the map: the field, the accessors, and the startup load.
        "src/daemon/state/mod.rs",
        // The admin endpoints that read and write the explicit choice itself.
        "src/api/admin_models/lifecycle.rs",
    ];
    let mut offenders = Vec::new();
    for path in rust_files_under(std::path::Path::new("src")) {
        let rel = path.to_string_lossy().replace('\\', "/");
        if allowed.iter().any(|a| rel.ends_with(a)) {
            continue;
        }
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (line, text) in statements(&src) {
            if !text.contains("encrypted_pipeline_models") {
                continue;
            }
            // WRITING the map is not re-deriving the setting — tests set the
            // explicit choice up that way, and that is what the map is for.
            // Only a READ can answer the question wrongly.
            if text.contains(".insert(") || text.contains(".remove(") {
                continue;
            }
            offenders.push(format!("{rel}:{line}: {}", text.trim()));
        }
    }
    assert!(
        offenders.is_empty(),
        "prompt privacy must be read through `SharedState::encrypted_pipeline_for` \
         (or `privacy_explicitly_enabled_for` where a DELIBERATE choice is the \
         question). Reading `encrypted_pipeline_models` directly misses the \
         automatic cases, which are the default:\n{}",
        offenders.join("\n")
    );
}

/// The scan above must actually be able to see the mistake it forbids —
/// including the shape rustfmt produces, which is how the original went
/// unnoticed. Planted here rather than trusted.
#[test]
fn the_privacy_map_guard_catches_the_form_the_defect_took() {
    let wrapped = "fn f() {\n    if self\n        .shared_state\n        .encrypted_pipeline_models\n        .get(&manifest.id)\n        .map(|v| *v)\n        .unwrap_or(false)\n    {\n        continue;\n    }\n}\n";
    assert!(
        statements(wrapped)
            .iter()
            .any(|(_, t)| t.contains("encrypted_pipeline_models")),
        "the guard cannot see a wrapped chain, which is the exact form prune had"
    );
    let inline = "fn f() { let on = s.encrypted_pipeline_models.get(&id); }\n";
    assert!(statements(inline)
        .iter()
        .any(|(_, t)| t.contains("encrypted_pipeline_models")));
    // A commented-out mention is not a violation.
    let commented = "fn f() {\n    // s.encrypted_pipeline_models.get(&id)\n    let a = 1;\n}\n";
    assert!(!statements(commented)
        .iter()
        .any(|(_, t)| t.contains("encrypted_pipeline_models")));
}

/// Is this `remove_session` call gated on the peer serving an active pipeline?
///
/// The whole guard, kept as its own function so its reach can be tested by
/// planting the violation rather than assumed — a scan that finds nothing and a
/// scan that *cannot* find anything read identically (gotcha #413).
fn session_removal_is_gated_on_an_active_pipeline(body: &str) -> bool {
    let mut saw_gate = false;
    for (_, stmt) in statements(body) {
        if stmt.contains("in_active_pipeline") && stmt.contains("remove_session") {
            return true;
        }
        // The gate and the call on separate statements: `if … in_active_pipeline
        // {` opening a block that contains the call.
        if stmt.starts_with("if ") && stmt.contains("in_active_pipeline") {
            saw_gate = true;
            continue;
        }
        if saw_gate && stmt.contains("session_manager.remove_session") {
            return true;
        }
        if stmt == "}" {
            saw_gate = false;
        }
    }
    false
}

/// **A disconnect retires the session, whether or not this node was
/// coordinating a request through the peer.**
///
/// It did not, and the exemption cost 29 consecutive undecryptable forwards to
/// one peer with zero successes (report #016). The peer SERVING the request
/// retires its own session on the same disconnect — `active_pipelines` is the
/// coordinator's map and holds nothing for work a node does for someone else
/// (gotcha #194) — and returns on a fresh static key, so a coordinator that
/// kept an ephemeral key seals what nothing can open, for as long as it holds
/// it. `establish_session` is idempotent, so the reconnect does not repair it.
///
/// The comment that stood here asserted the opposite ("reconnection will
/// refresh it"), which was true in April 2026 and stopped being true when
/// `establish_session` became idempotent. That is why this is a test and not a
/// comment: a claim about another function's behaviour goes stale silently.
#[test]
fn a_disconnect_retires_the_session_even_mid_pipeline() {
    let src = std::fs::read_to_string("src/network/manager/connections.rs").unwrap();
    let body = fn_body(&src, "fn handle_connection_closed")
        .expect("handle_connection_closed must exist — rename it and update this guard");
    assert!(
        body.contains("session_manager.remove_session"),
        "the disconnect must still retire the session"
    );
    assert!(
        !session_removal_is_gated_on_an_active_pipeline(body),
        "an active pipeline must not exempt a peer from the session being \
         retired — the peer retires its own either way (report #016)"
    );
}

/// The guard above must see the defect in the shape it actually had.
#[test]
fn the_active_pipeline_session_guard_catches_the_shape_it_replaced() {
    // The exact line this replaced.
    assert!(session_removal_is_gated_on_an_active_pipeline(
        "if !self.swarm.is_connected(&peer_id) && !in_active_pipeline {\n\
         self.shared_state.session_manager.remove_session(&node_id);\n}\n"
    ));
    // The gate as a separate enclosing block.
    assert!(session_removal_is_gated_on_an_active_pipeline(
        "if !in_active_pipeline {\n\
         self.shared_state.session_manager.remove_session(&node_id);\n\
         }\n"
    ));
    // Wrapped by rustfmt, which is the form that has blinded six guards.
    assert!(session_removal_is_gated_on_an_active_pipeline(
        "if !self.swarm.is_connected(&peer_id)\n    && !in_active_pipeline\n{\n\
         self.shared_state\n    .session_manager\n    .remove_session(&node_id);\n}\n"
    ));
    // The shape that is correct now must not trip it.
    assert!(!session_removal_is_gated_on_an_active_pipeline(
        "if !self.swarm.is_connected(&peer_id) {\n\
         self.shared_state.session_manager.remove_session(&node_id);\n\
         }\n\
         if !in_active_pipeline {\n    self.peer_to_node.remove(&peer_id);\n}\n"
    ));
}

/// **Send/Stop belong to the chat on screen, and so does the cancel behind
/// them** (report #015).
///
/// `send-btn` and `stop-btn` exist once in the document and every session
/// shares them, so a page-level "a reply is in flight" flag showed Stop in a
/// chat that had sent nothing — and the single page-level `AbortController`
/// meant pressing it cancelled a DIFFERENT chat's reply. Sending was blocked
/// there too, by the same flag.
///
/// Both now hang off `S.streaming[sessionId]`. The three things that change
/// which session is on screen must move the buttons with it, and nothing may
/// keep a controller of its own again.
#[test]
fn the_chat_send_and_stop_buttons_follow_the_session_on_screen() {
    let src = std::fs::read_to_string(repo_root().join("frontend/js/components/chat.js")).unwrap();

    // One writer of the flag, inside the refresher that reads the current
    // session. Anything else setting it is the page-level state coming back.
    let writes: Vec<&str> = src
        .lines()
        .filter(|l| l.contains("S.isStreaming =") || l.contains("S.isStreaming="))
        .collect();
    assert_eq!(
        writes.len(),
        1,
        "exactly one place may set the on-screen streaming flag, found: {writes:?}"
    );

    // A cancel must reach the session being looked at, never "the" reply.
    let stop = fn_body(&src, "stopGeneration: function()")
        .expect("stopGeneration must exist — rename it and update this guard");
    assert!(
        stop.contains("S.streaming[S.currentSessionId]"),
        "Stop must abort the reply in the session on screen, not a page-level one"
    );
    assert!(
        !src.contains("App.chat._abort"),
        "the page-level AbortController is what cancelled the wrong chat's reply"
    );

    // Everything that changes the session on screen refreshes the buttons.
    for f in [
        "switchSession: function(id)",
        "newSession: function()",
        "deleteSession: function(id, e)",
    ] {
        let body = fn_body(&src, f).unwrap_or_else(|| panic!("{f} must exist"));
        assert!(
            body.contains("_refreshStreamingUI"),
            "{f} changes which session is on screen and must move Send/Stop with it"
        );
    }
}

/// **Every stats surface that promises the traffic figure carries it.**
///
/// There are THREE, and they are separate payloads built in separate files:
/// `/api/admin/stats` (the dashboard's initial load), the WebSocket
/// `stats_update` tick, and `/v1/status` — which is what `swarmllm status`
/// reads, and the one a user is told to run when they are working out whether
/// this program is saturating their connection.
///
/// The figure reached the first two and not the third, and nothing noticed:
/// `describe_traffic` was tested against a hand-made JSON object, so the
/// CLI printed no Traffic line at all while the API served the numbers
/// (2026-09-12). One builder is not one surface — `network_traffic_json`
/// existed and was simply not called from the third place.
#[test]
fn every_stats_surface_carries_the_traffic_figure() {
    let root = repo_root();
    for (file, what) in [
        ("src/api/admin.rs", "GET /api/admin/stats"),
        ("src/api/websocket.rs", "the WebSocket stats_update tick"),
        (
            "src/api/openai/mod.rs",
            "GET /v1/status, which `swarmllm status` reads",
        ),
    ] {
        let src = std::fs::read_to_string(root.join(file)).unwrap();
        let joined: Vec<String> = statements(&src).into_iter().map(|(_, s)| s).collect();
        assert!(
            joined
                .iter()
                .any(|s| s.contains("\"network_traffic\"") && s.contains("network_traffic_json")),
            "{what} ({file}) must carry \"network_traffic\" from \
             `api::metrics::network_traffic_json` — a stats surface that omits it \
             reports no traffic at all, which reads as a silent node"
        );
    }
}

/// A method's body, brace-balanced from its signature.
///
/// [`fn_body`] stops at the first closing brace in column zero, which for a
/// method inside an `impl` is the impl's own — so it returns every method
/// below the one asked for. That is safe for a presence check and wrong for an
/// absence one, and the guard below is an absence check.
fn method_body<'a>(src: &'a str, signature: &str) -> Option<&'a str> {
    let start = src.find(signature)?;
    let rest = &src[start..];
    let open = rest.find('{')?;
    let mut depth = 0usize;
    for (i, ch) in rest.char_indices().skip(open) {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// **Which budget a live worker's growth is weighed against is the budget its
/// spawn charged — never a fact re-derived from placement.**
///
/// `WorkerHandle::placed_on_cpu_because` records why a model was DEMOTED. On a
/// machine with no graphics card nothing was demoted, so it reads `None`
/// exactly as it does for a worker holding a card, while `charges_ram` — which
/// knows "no card detected" and "this build has no CUDA" — put that worker on
/// the RAM budget. `charge_additional_segment` read placement, so every later
/// layer-range growth of a live worker on a GPU-less node was weighed by
/// `admit_to_gpu`, which returns `true` unconditionally when `vram_budget_mb`
/// is 0. Reported live 2026-09-13 (report #030) from a 16 GB Mac mini: 470 MB
/// through the real RAM gate at spawn, then ~11.5 GB waved through the
/// graphics gate across four growths of the same worker, and the machine
/// swapped. The anti-swap check was working and was simply never asked again.
///
/// Growth is the COMMON case on a swarm node — coverage is reassigned by
/// scheduling, failover and re-plans — so this is not a rare path.
#[test]
fn a_live_workers_growth_is_weighed_by_the_budget_its_spawn_charged() {
    let src =
        std::fs::read_to_string("src/inference/process_pool.rs").expect("read process_pool.rs");
    let body = method_body(&src, "    async fn charge_additional_segment(")
        .expect("charge_additional_segment was renamed — re-point this guard");

    assert!(
        body.contains("handle.holds_gpu_memory()"),
        "charge_additional_segment must ask `WorkerHandle::holds_gpu_memory`, \
         which reads `charged_against_ram` — the budget the spawn actually \
         charged"
    );
    for (line, stmt) in statements(body) {
        assert!(
            !stmt.contains("placed_on_cpu_because"),
            "line {line}: charge_additional_segment must not re-derive placement \
             — `placed_on_cpu_because` is `None` on a node with NO card, so this \
             sends every growth to the graphics gate, which has no ceiling to \
             check there. Ask `handle.holds_gpu_memory()`.\n  {stmt}"
        );
    }
}

/// The guard above must actually be able to fire, on the shape rustfmt
/// produces. Planted violation, per `.claude/rules/architecture.md` §
/// "A source-scanning guard is only as good as the spellings it knows".
#[test]
fn the_growth_accountant_guard_catches_a_re_derived_placement() {
    let planted = "impl ModelProcessPool {\n\
        \x20   async fn charge_additional_segment(\n\
        \x20       &self,\n\
        \x20   ) -> Result<(), SwarmError> {\n\
        \x20       let on_gpu = handle\n\
        \x20           .placed_on_cpu_because\n\
        \x20           .is_none();\n\
        \x20       if on_gpu {\n\
        \x20           self.admit_to_gpu(model_id, delta_mb);\n\
        \x20       }\n\
        \x20   }\n\
        \x20   fn something_else(&self) {\n\
        \x20       let _ = self.holds_gpu_memory();\n\
        \x20   }\n\
        }\n";
    let body = method_body(planted, "    async fn charge_additional_segment(")
        .expect("the extractor must find the method");

    assert!(
        !body.contains("holds_gpu_memory"),
        "the extractor must stop at the method's OWN closing brace — reaching \
         the next method makes the presence half of the guard pass on any file"
    );
    assert!(
        statements(body)
            .iter()
            .any(|(_, s)| s.contains(".placed_on_cpu_because.is_none()")),
        "the scanner must see the chain rustfmt has wrapped across three lines, \
         which is the shape the real code was written in"
    );
}

/// Name of the function containing a given 1-based line: the nearest `fn` at or
/// above it. Used to exempt a specific call site rather than a whole file.
fn enclosing_fn_name(src: &str, line: usize) -> Option<String> {
    let above: Vec<&str> = src.lines().take(line).collect();
    for raw in above.iter().rev() {
        let mut t = raw.trim_start();
        for prefix in [
            "pub(crate) ",
            "pub(super) ",
            "pub ",
            "async ",
            "const ",
            "unsafe ",
        ] {
            t = t.strip_prefix(prefix).unwrap_or(t);
        }
        let Some(name) = t.strip_prefix("fn ") else {
            continue;
        };
        let end = name
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(name.len());
        if end > 0 {
            return Some(name[..end].to_string());
        }
    }
    None
}

#[test]
fn the_enclosing_fn_helper_names_the_function_a_line_sits_in() {
    let src = "fn outer() {\n    let a = 1;\n}\n\npub async fn inner_one(&self) {\n    x.remove(&k);\n}\n";
    assert_eq!(enclosing_fn_name(src, 2).as_deref(), Some("outer"));
    assert_eq!(enclosing_fn_name(src, 6).as_deref(), Some("inner_one"));
}

/// A download's progress entry is not a progress bar — it is the state that
/// identifies a shard already being fetched.
///
/// `acquisition_progress` carries the per-shard `Downloading` marks
/// `is_shard_in_progress` reads, and that predicate is the only thing stopping
/// a second task appending to the same `shard_NNN.bin.tmp`. Removing a model's
/// entry is therefore a claim that nothing is in flight for it, and it was
/// being made unconditionally by every caller — including the per-shard
/// completion paths, which know only about their own shard.
///
/// Field report, 2026-09-13: an eleven-shard model auto-replicating three
/// shards at a time (`max_concurrent_downloads`) lost the whole entry five
/// seconds after the first shard landed, so the next auto-manage tick started
/// duplicate downloads of the two still running. Each duplicate discarded the
/// partial `.tmp` and restarted from byte zero — the same ~512 MB shard
/// re-downloading every five minutes for hours, on two machines, never landing.
///
/// So: `SharedState::remove_acquisition_if_idle` is the one place an entry is
/// removed for tidiness, and it asks first. The only sanctioned bare remove is
/// `delete_model`, where the user has deleted the model outright and there is
/// nothing left for a download to land in.
#[test]
fn a_finished_download_does_not_delete_the_progress_of_one_still_running() {
    let root = repo_root();
    let sanctioned: &[(&str, &str)] = &[
        // The helper itself.
        ("src/daemon/state/mod.rs", "remove_acquisition_if_idle"),
        // The user deleted the whole model; the guard above it is the
        // active-pipeline check, not this one.
        ("src/api/admin_models/lifecycle.rs", "delete_model"),
    ];

    let mut offenders: Vec<String> = Vec::new();
    for path in rust_files_under(&root.join("src")) {
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !src.contains("acquisition_progress") {
            continue;
        }
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        for (line, stmt) in statements(&src) {
            if !stmt.contains("acquisition_progress.remove(") {
                continue;
            }
            let enclosing = enclosing_fn_name(&src, line);
            let allowed = sanctioned
                .iter()
                .any(|(f, func)| *f == rel && Some(*func) == enclosing.as_deref());
            if !allowed {
                offenders.push(format!(
                    "{rel}:{line} (in {}): {stmt}",
                    enclosing.as_deref().unwrap_or("?")
                ));
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "acquisition_progress must be removed through \
         SharedState::remove_acquisition_if_idle, which refuses while a shard \
         of the model is still being written. Removing it un-guards those \
         downloads and a later cycle starts duplicates of them:\n  {}",
        offenders.join("\n  ")
    );
}

/// Null control for the scan above: plant the violation it exists to catch and
/// confirm the scanner can see it, through the line-wrapping rustfmt produces
/// at this nesting depth.
#[test]
fn the_acquisition_removal_scan_sees_a_chain_rustfmt_has_wrapped() {
    let planted = r#"
fn tidy_up(&self) {
    self.shared_state
        .models
        .acquisition_progress
        .remove(&model_id);
}
"#;
    let found = statements(planted)
        .into_iter()
        .any(|(_, s)| s.contains("acquisition_progress.remove("));
    assert!(
        found,
        "the statement scanner must join a wrapped chain back together, or the \
         guard above passes on code it cannot read"
    );
}

/// Stopping something is not the same as it breaking.
///
/// A download the user cancelled, or one the daemon stopped on its way down,
/// was recorded as `AcquisitionState::Failed { reason: "Cancelled by …" }` —
/// in three separate places, one of which logged "Download cancelled" on the
/// line directly above. `downloads.js` renders any `failed` state in red as
/// "Download failed — check Activity log for details", so somebody who pressed
/// Cancel was told something had gone wrong with the thing they had just
/// deliberately done, and sent to a log with nothing in it.
///
/// `AcquisitionState::Cancelled` exists for this. The reason (user vs
/// shutdown) still goes in the acquisition log, which is where the detail
/// belongs; the STATE must stop claiming failure.
///
/// Found by cancelling a real download and reading the screen, after the unit
/// tests for the same commit had all passed.
#[test]
fn a_cancelled_download_is_not_recorded_as_a_failed_one() {
    let root = repo_root();
    let mut offenders: Vec<String> = Vec::new();
    for path in rust_files_under(&root.join("src")) {
        let Ok(src) = std::fs::read_to_string(&path) else {
            continue;
        };
        if !src.contains("AcquisitionState::Failed") {
            continue;
        }
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let lines: Vec<&str> = src.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !line.contains("AcquisitionState::Failed") {
                continue;
            }
            // The reason literal may sit a line or two below the variant.
            let window = lines[i..lines.len().min(i + 4)].join("\n").to_lowercase();
            // Only a CONSTRUCTION carries a `reason:`. A match arm reads
            // `Failed { .. }` and may legitimately sit next to a `Cancelled`
            // arm — which is exactly what the first version of this guard
            // flagged, in the health monitor's cleanup.
            if window.contains("cancel") && window.contains("reason:") {
                offenders.push(format!("{rel}:{}: {}", i + 1, line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a cancellation is recorded as a failure — use \
         `AcquisitionState::Cancelled`, which the dashboard renders neutrally, \
         and keep the reason in the acquisition log:\n  {}",
        offenders.join("\n  ")
    );
}

/// `I18n.translatePage` does `el.textContent = t(key)` for every `[data-i18n]`
/// element. Setting `textContent` REPLACES everything inside the element — so
/// a translated element that WRAPS a control deletes that control on every page
/// load, in every language, including English.
///
/// This is not hypothetical. `<label data-i18n="settings.key_source_label">`
/// wrapped the `provider-key-source` `<select>`, so the Cloud Providers "Key
/// source" setting did not exist in the DOM for any user: `init.js` bound its
/// change handler to `null`, `settings.js` guarded its write with `if (sel)`,
/// and three translated options in 21 locales were strings nobody could ever
/// see. Nothing went red — the markup is valid, the translation is correct, and
/// the control simply is not there.
///
/// The failure is invisible by construction, which is exactly what a scan is
/// for. Text that must sit next to a control goes in a `<span data-i18n=…>`
/// beside it, never on the element that contains it.
#[test]
fn a_translated_element_never_wraps_a_control_it_would_delete() {
    let html = std::fs::read_to_string(repo_root().join("frontend/index.html"))
        .expect("cannot read frontend/index.html");

    let offenders = translated_elements_wrapping_a_control(&html);
    assert!(
        offenders.is_empty(),
        "these elements carry data-i18n AND contain a control, which \
         `translatePage`'s `textContent` write deletes on every page load:\n  {}\n\
         Move the text into a <span data-i18n=\"…\"> beside the control instead.",
        offenders.join("\n  ")
    );
}

/// Returns `tag@line -> contained control` for every element that carries
/// `data-i18n` and contains an interactive descendant.
///
/// Deliberately scans the WHOLE element body rather than a character window,
/// and reports the line so the offender is findable (see
/// `.claude/rules/architecture.md` § "A source-scanning guard is only as good
/// as the spellings it knows").
fn translated_elements_wrapping_a_control(html: &str) -> Vec<String> {
    const CONTAINERS: [&str; 6] = ["label", "span", "div", "p", "button", "summary"];
    const CONTROLS: [&str; 4] = ["<input", "<select", "<textarea", "<button"];

    let mut offenders = Vec::new();
    for tag in CONTAINERS {
        let open = format!("<{tag}");
        let close = format!("</{tag}>");
        let mut from = 0usize;
        while let Some(rel) = html[from..].find(&open) {
            let start = from + rel;
            from = start + open.len();
            // The opening tag's own attributes, up to its '>'.
            let Some(gt) = html[start..].find('>') else {
                break;
            };
            let attrs = &html[start..start + gt];
            if !attrs.contains("data-i18n=") {
                continue;
            }
            // Body from after '>' to the matching close, honouring nesting.
            let body_start = start + gt + 1;
            let mut depth = 1usize;
            let mut cursor = body_start;
            let body_end = loop {
                let next_open = html[cursor..].find(&open).map(|i| cursor + i);
                let next_close = html[cursor..].find(&close).map(|i| cursor + i);
                match (next_open, next_close) {
                    (_, None) => break html.len(),
                    (Some(o), Some(c)) if o < c => {
                        depth += 1;
                        cursor = o + open.len();
                    }
                    (_, Some(c)) => {
                        depth -= 1;
                        if depth == 0 {
                            break c;
                        }
                        cursor = c + close.len();
                    }
                }
            };
            let body = &html[body_start..body_end.min(html.len())];
            let found: Vec<&str> = CONTROLS
                .iter()
                .copied()
                .filter(|c| body.contains(c))
                .collect();
            if !found.is_empty() {
                let line = html[..start].matches('\n').count() + 1;
                let key = attrs
                    .split("data-i18n=\"")
                    .nth(1)
                    .and_then(|s| s.split('"').next())
                    .unwrap_or("?");
                offenders.push(format!(
                    "frontend/index.html:{line} <{tag} data-i18n=\"{key}\"> contains {}",
                    found.join(", ")
                ));
            }
        }
    }
    offenders
}

/// The scan above finds nothing today, and a scan that finds nothing is
/// indistinguishable from one that CANNOT find anything. Plant the exact defect
/// it exists to catch and require it to fire.
#[test]
fn the_translated_wrapper_scan_catches_the_defect_it_is_for() {
    let planted = r#"
        <div class="row">
          <label class="text-sm" data-i18n="settings.key_source_label">Key source:
            <select id="provider-key-source"><option value="auto">Auto</option></select>
          </label>
        </div>
    "#;
    let hits = translated_elements_wrapping_a_control(planted);
    assert_eq!(
        hits.len(),
        1,
        "the scan missed a planted <label data-i18n> wrapping a <select>: {hits:?}"
    );
    assert!(
        hits[0].contains("<select"),
        "wrong control reported: {hits:?}"
    );

    // And the corrected shape must NOT trip it, or the guard is unusable.
    let fixed = r#"
        <label class="text-sm">
          <span data-i18n="settings.key_source_label">Key source:</span>
          <select id="provider-key-source"><option value="auto">Auto</option></select>
        </label>
    "#;
    assert!(
        translated_elements_wrapping_a_control(fixed).is_empty(),
        "the guard fires on the correct markup"
    );
}

/// **A streamed failure the daemon NAMED must reach the user, and must not be
/// retried.**
///
/// The daemon already sends `{"error": {message, type}}` as an ordinary SSE
/// frame — the same shape the non-streaming path parses — but `onChunk` read
/// only `chunk.choices` and `chunk.usage`, so the frame matched no branch and
/// was dropped. The user was then shown "the model might still be loading"
/// for a request the daemon had explicitly refused, closed and refunded
/// (report #002). Gotcha #300 is the same defect one layer down, and its own
/// lesson was that the fix "landed where the report came from and nowhere
/// else": the backend learned to send a proper error frame, and the dashboard
/// was never taught to read it.
///
/// Two terms are load-bearing and pinned here:
///   * `onChunk` must look at `chunk.error` at all;
///   * the silent retry must be gated on NOT having one — the commonest such
///     refusal is "not enough memory right now", where re-sending is the last
///     thing the node needs.
#[test]
fn a_streamed_failure_is_shown_to_the_user_and_never_silently_retried() {
    let src = std::fs::read_to_string(repo_root().join("frontend/js/components/chat.js"))
        .expect("cannot read chat.js");

    assert!(
        src.contains("chunk.error"),
        "chat.js's stream handler never inspects `chunk.error`, so a terminal \
         failure frame from the daemon is dropped and reported to the user as \
         \"the model might still be loading\" (report #002)."
    );

    let retry_gate = src
        .lines()
        .find(|l| {
            l.contains("!fullContent") && l.contains("!reasoningContent") && l.contains("if ")
        })
        .unwrap_or_else(|| {
            panic!("could not find chat.js's silent-retry condition — if it moved, move this guard")
        });
    assert!(
        retry_gate.contains("!streamError"),
        "chat.js retries a request silently without excluding one the daemon \
         already refused:\n  {}\n\
         A terminal error frame is a final answer; re-sending it doubles the \
         load on a node that has just said it has none to spare.",
        retry_gate.trim()
    );
}

/// **An element that starts with the `hidden` CLASS can only be revealed by
/// removing that class.** `.hidden` is `display: none !important`, and no
/// inline `style.display` can beat an `!important` declaration, whatever value
/// it is set to.
///
/// This was FIVE dead surfaces at once, found from one report (#037): both pool
/// entry buttons — the only way into the My Devices feature for a node not yet
/// in a pool — plus the Claude subscription card in Settings and the chat image
/// preview, none of which could ever appear. The JS ran, threw nothing, and
/// logged nothing; only the computed style showed it.
///
/// It is a MIGRATION hazard, not a coding mistake: the handlers were written
/// against markup that used an inline `display`, and the elements were later
/// moved to the `hidden` utility class without the JS being brought along —
/// `pool-create-form` on 2026-04-07, `pool-join-form` on 2026-05-20, each
/// breaking its button on a different day. Gotcha #469 is the mirror image, and
/// its own note says this had "been hit and patched locally twice without
/// anyone naming the general rule". This is the rule, named.
#[test]
fn an_element_hidden_by_class_is_never_revealed_by_an_inline_display() {
    let html = std::fs::read_to_string(repo_root().join("frontend/index.html"))
        .expect("cannot read frontend/index.html");
    let js = read_frontend_js();

    let offenders = hidden_class_elements_driven_by_inline_display(&html, &js);
    assert!(
        offenders.is_empty(),
        "these elements carry the `hidden` class in the markup but are shown or \
         hidden with `style.display`, which `.hidden`'s `!important` always \
         beats — so they can never appear:\n  {}\n\
         Use `classList.add('hidden')` / `classList.remove('hidden')` instead, \
         as the other show/hide sites in this frontend do.",
        offenders.join("\n  ")
    );
}

/// Every `frontend/js/**/*.js` as `(path, source)`.
fn read_frontend_js() -> Vec<(String, String)> {
    fn walk(dir: &std::path::Path, out: &mut Vec<(String, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "js") {
                if let Ok(s) = std::fs::read_to_string(&p) {
                    out.push((p.display().to_string(), s));
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(&repo_root().join("frontend/js"), &mut out);
    out
}

/// Returns one entry per element that carries the bare `hidden` class in the
/// markup AND is reached by a `style.display` write in the frontend JS.
///
/// Two shapes are recognised, both taken from the real defects rather than
/// imagined: the direct chain `getElementById('x').style.display = …`, and the
/// far commoner `var el = document.getElementById('x'); … el.style.display = …`.
///
/// The second needs a proximity window, because a generic variable name (`card`,
/// `area`, `detail` — all three appeared) is reused across functions and
/// matching it file-wide would report unrelated code. 40 lines is sized from the
/// real cases, whose widest gap was 7 (`chat.js` 28 → 35).
fn hidden_class_elements_driven_by_inline_display(
    html: &str,
    js: &[(String, String)],
) -> Vec<String> {
    // ids whose markup class list contains `hidden` as a whole token
    let mut hidden_ids: Vec<String> = Vec::new();
    for tag in html.split('<') {
        let Some(gt) = tag.find('>') else { continue };
        let attrs = &tag[..gt];
        let Some(class) = attr_value(attrs, "class") else {
            continue;
        };
        if !class.split_whitespace().any(|c| c == "hidden") {
            continue;
        }
        if let Some(id) = attr_value(attrs, "id") {
            hidden_ids.push(id.to_string());
        }
    }

    let mut offenders = Vec::new();
    for id in &hidden_ids {
        let needle = format!("getElementById('{id}')");
        for (path, src) in js {
            let lines: Vec<&str> = src.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                if !line.contains(&needle) {
                    continue;
                }
                // (a) direct chain on this very line
                if line.contains(".style.display") {
                    offenders.push(format!("{path}:{} #{id} (direct)", i + 1));
                    continue;
                }
                // (b) captured into a variable, then written to nearby
                let Some(var) = line
                    .split('=')
                    .next()
                    .and_then(|lhs| lhs.split_whitespace().last())
                else {
                    continue;
                };
                if var.is_empty() {
                    continue;
                }
                let probe = format!("{var}.style.display");
                let end = (i + 40).min(lines.len());
                if let Some(off) = lines[i..end].iter().position(|l| l.contains(&probe)) {
                    offenders.push(format!("{path}:{} #{id} (via `{var}`)", i + off + 1));
                }
            }
        }
    }
    offenders.sort();
    offenders.dedup();
    offenders
}

/// Value of `name="…"` in an opening tag's attribute text.
fn attr_value<'a>(attrs: &'a str, name: &str) -> Option<&'a str> {
    let pat = format!("{name}=\"");
    let at = attrs.find(&pat)? + pat.len();
    let rest = &attrs[at..];
    Some(&rest[..rest.find('"')?])
}

/// The scan above finds nothing today, and a scan that finds nothing is
/// indistinguishable from one that CANNOT find anything (gotcha #413). Plant
/// both real shapes and require each to fire — and require the corrected shape
/// not to.
#[test]
fn the_hidden_class_scan_catches_both_shapes_of_the_defect() {
    let html = r#"
        <div id="pool-create-form" class="mt-2 hidden mx-auto"></div>
        <div id="image-preview-area" class="hidden bg-surface"></div>
        <div id="pool-active" style="display:none"></div>
    "#;

    let direct = vec![(
        "pool.js".to_string(),
        "document.getElementById('pool-create-form').style.display = '';".to_string(),
    )];
    let hits = hidden_class_elements_driven_by_inline_display(html, &direct);
    assert_eq!(hits.len(), 1, "missed the direct-chain shape: {hits:?}");
    assert!(hits[0].contains("pool-create-form"), "{hits:?}");

    let via_var = vec![(
        "chat.js".to_string(),
        "var area = document.getElementById('image-preview-area');\n\
         if (!area) return;\n\
         area.innerHTML = '';\n\
         area.style.display = 'flex';"
            .to_string(),
    )];
    let hits = hidden_class_elements_driven_by_inline_display(html, &via_var);
    assert_eq!(hits.len(), 1, "missed the via-variable shape: {hits:?}");
    assert!(hits[0].contains("image-preview-area"), "{hits:?}");

    // An element that does NOT carry the class is not this defect — `pool-active`
    // uses an inline display deliberately, and reporting it would make the guard
    // unusable.
    let inline_ok = vec![(
        "pool.js".to_string(),
        "var active = document.getElementById('pool-active');\n\
         active.style.display = 'none';"
            .to_string(),
    )];
    assert!(
        hidden_class_elements_driven_by_inline_display(html, &inline_ok).is_empty(),
        "the guard fires on an element that legitimately uses an inline display"
    );

    // And the corrected shape must not trip it.
    let fixed = vec![(
        "pool.js".to_string(),
        "document.getElementById('pool-create-form').classList.remove('hidden');".to_string(),
    )];
    assert!(
        hidden_class_elements_driven_by_inline_display(html, &fixed).is_empty(),
        "the guard fires on the corrected markup"
    );
}

/// A reasoning model's scratchpad is removed by `StreamingToolText::push`, and
/// whether a reply has one has nothing to do with whether the caller passed
/// `tools`. All four streaming surfaces nevertheless wrapped that call in
/// `if tools_requested { … } else { send the raw token }`, so an ordinary chat
/// message — no tools, which is nearly all real traffic — never reached the
/// filter and streamed the model's whole `<think>` block to the user as the
/// answer. Reproduced on v0.3.179 against qwen3-1.7b (report #032).
///
/// The type-level half of the fix is that `StreamingToolText` has no `Default`,
/// so a caller must state which kind of reply it is reading — the compiler
/// enforces that. This is the other half: the mode belongs in the CONSTRUCTOR,
/// never as a condition around `push`.
#[test]
fn the_scratchpad_filter_is_never_gated_on_whether_tools_were_requested() {
    let mut offenders = Vec::new();
    for rel in [
        "src/api/openai/streaming.rs",
        "src/api/anthropic/handlers.rs",
    ] {
        let src = std::fs::read_to_string(repo_root().join(rel))
            .unwrap_or_else(|e| panic!("cannot read {rel}: {e}"));
        for (line, body) in blocks_guarded_by(&src, "tools_requested") {
            if body.contains("buffered.push(") || body.contains(".push(&event.text)") {
                offenders.push(format!("{rel}:{line}"));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a streamed token reaches `StreamingToolText::push` only when tools were \
         requested, at:\n  {}\n\
         The buffer also strips a reasoning model's `<think>` preamble, so gating \
         it this way leaks the whole scratchpad on an ordinary chat message \
         (report #032). Pass the mode to `StreamingToolText::new` instead.",
        offenders.join("\n  ")
    );
}

/// Bodies of every `if <cond> {` block in `src`, as `(1-indexed line, body)`.
///
/// Brace-matched over the whole block, not a character window — a guard pinned
/// to a fixed lookahead goes blind the moment the block grows (see
/// `.claude/rules/architecture.md` § "A source-scanning guard is only as good as
/// the spellings it knows").
fn blocks_guarded_by(src: &str, cond: &str) -> Vec<(usize, String)> {
    let needle = format!("if {cond}");
    let bytes: Vec<char> = src.chars().collect();
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = src[from..].find(&needle) {
        let at = from + rel;
        from = at + needle.len();
        // Char index of the match, and the `{` that opens its block.
        let start_chars = src[..at].chars().count();
        let Some(open_rel) = bytes[start_chars..].iter().position(|c| *c == '{') else {
            continue;
        };
        // A `{` further away than the condition itself is a different construct.
        if open_rel > 120 {
            continue;
        }
        let open = start_chars + open_rel;
        let mut depth = 0usize;
        let mut end = open;
        for (i, c) in bytes.iter().enumerate().skip(open) {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i;
                        break;
                    }
                }
                _ => {}
            }
        }
        let line = src[..at].matches('\n').count() + 1;
        out.push((line, bytes[open..=end].iter().collect::<String>()));
    }
    out
}

/// The scan finds nothing today, so pin its reach by planting the exact defect —
/// and require the corrected shape not to trip it.
#[test]
fn the_tools_gate_scan_catches_the_defect_it_is_for() {
    let planted = r#"
        fn stream() {
            let mut buffered = StreamingToolText::default();
            if tools_requested {
                if let Some(safe) = buffered.push(&event.text) { send(safe); }
                continue;
            }
            send(event.text);
        }
    "#;
    let hits = blocks_guarded_by(planted, "tools_requested");
    assert_eq!(hits.len(), 1, "the scan did not find the planted block");
    assert!(
        hits[0].1.contains("buffered.push("),
        "the scan took the wrong block body: {:?}",
        hits[0].1
    );

    let fixed = r#"
        fn stream() {
            let mut buffered = StreamingToolText::new(tools_requested);
            if let Some(safe) = buffered.push(&event.text) { send(safe); }
        }
    "#;
    assert!(
        blocks_guarded_by(fixed, "tools_requested")
            .iter()
            .all(|(_, body)| !body.contains("buffered.push(")),
        "the guard fires on the corrected shape"
    );
}

/// A Windows GPU node updates itself by swapping the bare `.exe`, which leaves
/// the CUDA redist DLLs beside it at whatever version the user's original zip
/// install shipped (gotcha #162). That is safe for as long as the toolkit stays
/// within CUDA 12: `cudart64_12.dll` serves every 12.x. It is silently fatal
/// across a major bump — the new binary would `LoadLibraryW("cudart64_13.dll")`
/// against a folder holding only `cudart64_12.dll`, and every auto-updated
/// Windows GPU node would die at CUDA init with no fallback.
///
/// This never mattered before 2026-09-14, because `apply_update_with_version`
/// refused to install anything on Windows at all — the hazard had two locks on
/// it and one was load-bearing by accident. Now that Windows installs updates
/// like everywhere else, the pin is the only lock left, so it gets a guard
/// rather than a paragraph in a gotchas file.
///
/// If you are here because this test went red: bumping the Windows CUDA toolkit
/// means teaching `update.rs` to fetch the `-gpu` **zip** (which carries the
/// DLLs) instead of the bare exe, or gating the release so those users are sent
/// a fresh installer. `docs/FUTURE_WORK.md` § "Windows-GPU auto-update carries
/// stale CUDA redist DLLs" has the detail.
#[test]
fn the_windows_gpu_update_swaps_one_file_and_the_cuda_pin_still_allows_that() {
    let path = repo_root().join(".github/actions/gpu-build-env/action.yml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    // The `default:` on the line after the `cuda-windows-version:` block.
    let block = text
        .split_once("cuda-windows-version:")
        .unwrap_or_else(|| panic!("{} no longer declares cuda-windows-version", path.display()))
        .1;
    let default_line = block
        .lines()
        .find(|l| l.trim_start().starts_with("default:"))
        .unwrap_or_else(|| panic!("cuda-windows-version has no default in {}", path.display()));
    let pinned = default_line
        .trim()
        .trim_start_matches("default:")
        .trim()
        .trim_matches('\'')
        .trim_matches('"')
        .to_string();

    assert!(
        pinned.starts_with("12."),
        "the Windows CUDA toolkit is pinned to {pinned}, but the Windows updater still installs \
         the bare .exe and never refreshes the CUDA redist DLLs beside it. Crossing a CUDA major \
         strands every auto-updated Windows GPU node at CUDA init. Ship the -gpu zip from \
         update.rs, or gate the release, before changing this pin."
    );

    // The other half of the claim: that the updater really does swap a single
    // file. A guard that derives the state it checks from one side only would
    // keep passing if the swap grew to carry the DLLs and the pin then moved.
    let update =
        std::fs::read_to_string(repo_root().join("src/update.rs")).expect("read update.rs");
    assert!(
        update.contains("swap_by_moving_aside"),
        "src/update.rs no longer installs by swapping a single binary — if the -gpu path now \
         unpacks the zip, this guard and the CUDA pin comment above it are both stale"
    );
}

/// Every `vh` length in the stylesheets is paired with a `dvh` one.
///
/// `100vh` on iOS Safari means the viewport with the browser chrome COLLAPSED,
/// which is taller than what is on screen. Sizing the app shell from it made
/// the whole page scroll as one block on iPad — sidebar, chat header and
/// message list drifting together, with the input reachable only by scrolling
/// past everything above it (report #034, 2026-09-14). `dvh` tracks the
/// viewport actually visible.
///
/// The pattern is a `vh` declaration immediately followed by its `dvh` sibling:
/// a browser that does not know `dvh` drops the second and keeps today's
/// behaviour, and one that does takes the second. Which means a NEW `vh` added
/// on its own is invisible — it works everywhere the author can test and is
/// wrong on the one engine they cannot. That is what this catches.
///
/// If you are here because this went red: add `<same-property>: <n>dvh;`
/// directly after the `vh` line, or on the same line for a compact rule.
#[test]
fn every_viewport_height_in_css_has_a_dynamic_fallback_beside_it() {
    // Block comments hold prose about `100vh` and must not be scanned.
    fn strip_comments(src: &str) -> String {
        let mut out = String::with_capacity(src.len());
        let b = src.as_bytes();
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
                // Keep newlines so line numbers survive.
                let mut j = i + 2;
                while j + 1 < b.len() && !(b[j] == b'*' && b[j + 1] == b'/') {
                    if b[j] == b'\n' {
                        out.push('\n');
                    }
                    j += 1;
                }
                i = (j + 2).min(b.len());
            } else {
                out.push(b[i] as char);
                i += 1;
            }
        }
        out
    }

    // A `vh` length, but never `dvh`/`svh`/`lvh` (the `\D` guard) and never
    // `vw`.
    fn has_bare_vh(line: &str) -> bool {
        let bytes = line.as_bytes();
        let mut idx = 0;
        while let Some(found) = line[idx..].find("vh") {
            let at = idx + found;
            let prev = if at == 0 { b' ' } else { bytes[at - 1] };
            let after_ok = bytes.get(at + 2).is_none_or(|c| !c.is_ascii_alphanumeric());
            // A digit before `vh` is a length; a letter (d/s/l) is another unit.
            if prev.is_ascii_digit() && after_ok {
                return true;
            }
            idx = at + 2;
        }
        false
    }

    let dir = repo_root().join("frontend/css");
    let mut offenders = Vec::new();
    let mut paired = 0usize;

    for entry in std::fs::read_dir(&dir).expect("read frontend/css") {
        let path = entry.expect("dir entry").path();
        if path.extension().is_none_or(|e| e != "css") {
            continue;
        }
        let raw = std::fs::read_to_string(&path).expect("read css");
        let text = strip_comments(&raw);
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            if !has_bare_vh(line) {
                continue;
            }
            let next = lines.get(i + 1).copied().unwrap_or("");
            if line.contains("dvh") || next.contains("dvh") {
                paired += 1;
                continue;
            }
            offenders.push(format!(
                "{}:{}: {}",
                path.file_name().unwrap().to_string_lossy(),
                i + 1,
                line.trim()
            ));
        }
    }

    assert!(
        offenders.is_empty(),
        "a `vh` length with no `dvh` beside it sizes the layout from iOS \
         Safari's chrome-collapsed viewport, which is taller than the screen \
         (report #034). Add the `dvh` sibling directly after each:\n  {}",
        offenders.join("\n  ")
    );

    // The scanner must be able to SEE the pairs, or it passes by finding
    // nothing — the failure mode of every guard in this file (gotcha #413).
    assert!(
        paired >= 6,
        "expected to find the shell and modal `vh`/`dvh` pairs; found {paired}. \
         Either they were removed or this scanner stopped matching them."
    );
}

/// The UI calls a piece of a model a "part" and a machine a "computer",
/// in English, everywhere.
///
/// It used to call it three things at once — "shard", "part" and "piece" — and
/// not by area: `dashboard.info_shards` read "Parts" while the tip beside it
/// read "All shards available", and one activity feed contained both spellings
/// two lines apart. Every locale had inherited the same split, several with a
/// third word of their own. That is the kind of thing a reader does not report
/// as a bug; they just quietly conclude the thing is for someone else.
///
/// The rename is only worth anything if it stays done, and a new string using
/// the old word is invisible — it reads fine on its own, and nothing compares
/// it with its neighbours. Hence this.
///
/// Scope is deliberately the ENGLISH source only. The translations were brought
/// in line in the same change, but a locale is prose in another language and
/// pinning its vocabulary from here would be guessing.
///
/// If you are here because this went red: say "part". `{shard}` and `{shards}`
/// as PLACEHOLDER names are fine and are ignored — they are wire field names,
/// not words anybody reads.
#[test]
fn the_english_ui_uses_one_word_for_a_model_part_and_one_for_a_machine() {
    let path = repo_root().join("frontend/i18n/en.json");
    let raw = std::fs::read_to_string(&path).expect("read en.json");
    let map: serde_json::Map<String, serde_json::Value> =
        serde_json::from_str(&raw).expect("en.json is an object");

    // "pieces of work" and "pieces of other people's requests" are units of
    // WORK, not parts of a model — a different thing that keeps its own word.
    const NOT_ABOUT_MODELS: &[&str] = &["perf.served_detail", "dashboard.stat_forwards_tip"];

    // The machine a model runs on is a "computer", not a peer or a node.
    //
    // The pool feature is the ONE exception and keeps "device": machines you
    // own and manage in a pool are a different thing from anyone's machine on
    // the swarm, and "My Devices" is a nav destination with its own name. So
    // this forbids peer/node and deliberately says nothing about "device".
    const MACHINE_WORDS: &[&str] = &["peer", "peers", "node", "nodes"];

    let word = regex_lite_word;
    let mut offenders = Vec::new();
    let mut machines = Vec::new();
    for (key, value) in &map {
        let Some(text) = value.as_str() else { continue };
        if NOT_ABOUT_MODELS.contains(&key.as_str()) {
            continue;
        }
        // Placeholder names are wire identifiers, not prose.
        let prose = strip_placeholders(text);
        for banned in ["shard", "shards", "piece", "pieces"] {
            if word(&prose, banned) {
                offenders.push(format!("{key}: …{}…", prose.trim()));
                break;
            }
        }
        for banned in MACHINE_WORDS {
            if word(&prose, banned) {
                machines.push(format!("{key}: …{}…", prose.trim()));
                break;
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "the English UI must call a piece of a model a \"part\" — these say \
         something else:\n  {}",
        offenders.join("\n  ")
    );
    assert!(
        machines.is_empty(),
        "the English UI must call a machine a \"computer\" — \"peer\" and \"node\" \
         are the words this replaced. (The pool feature keeps \"device\" and is \
         not checked here.)\n  {}",
        machines.join("\n  ")
    );

    // The scan must be able to SEE the word it forbids, or it passes by being
    // blind (gotcha #413).
    assert!(
        regex_lite_word("a shard of the model", "shard"),
        "the word scanner stopped matching; it would pass on anything"
    );
    assert!(
        !regex_lite_word("shard_progress_label", "shard"),
        "must not match inside an identifier"
    );
}

/// Whole-word match without pulling in a regex engine: the surrounding bytes
/// must not be alphanumeric, so `shard` does not match inside `shard_label`.
fn regex_lite_word(haystack: &str, needle: &str) -> bool {
    let lower = haystack.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut from = 0;
    while let Some(at) = lower[from..].find(needle).map(|i| i + from) {
        let before_ok = at == 0 || !bytes[at - 1].is_ascii_alphanumeric() && bytes[at - 1] != b'_';
        let end = at + needle.len();
        let after_ok =
            end >= bytes.len() || (!bytes[end].is_ascii_alphanumeric() && bytes[end] != b'_');
        if before_ok && after_ok {
            return true;
        }
        from = at + 1;
    }
    false
}

/// A holder count a person reads is the count that could actually serve them.
///
/// `ModelRegistry::shard_holders` drops holders that positively claim a
/// different GGUF build — their bytes would fail our hash — while
/// `all_shard_entries` hands back the map RAW. Both of the functions scanned
/// here build a holder count that lands in the SAME dashboard row, so an
/// unfiltered one does not merely overstate: the number changes depending on
/// which writer touched the row last.
///
/// Measured on the live node 2026-09-14, before the fix: two peers announced
/// parts of `qwen2.5-coder-7b-instruct-q4-k-m`, neither could serve one, and
/// the card said three computers had it. That count picks the model health
/// sentence a non-technical user reads to decide whether the model keeps
/// working.
///
/// Deliberately NOT scanned: `api::admin_hf::count_unique_shard_holders`, which
/// answers a different question — "how many copies of this exist in the swarm",
/// for an HF search result the user has not downloaded yet. There the raw count
/// is the honest one, and for a model this node does not hold the filter is a
/// no-op anyway (an absent manifest means `BUILD_TAG_UNKNOWN`, which never
/// conflicts). See the comment at that helper.
#[test]
fn a_holder_count_shown_to_a_person_is_the_count_that_can_serve() {
    let sites = [
        (
            "src/api/admin_models/listing.rs",
            "pub async fn list_models(",
        ),
        ("src/api/websocket.rs", "async fn build_stats_message("),
    ];
    for (path, signature) in sites {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let body = fn_body(&src, signature)
            .unwrap_or_else(|| panic!("{path}: signature not found: {signature}"));
        assert!(
            counts_holders_through_the_filter(body),
            "{path}: {signature} builds a holder count from the RAW `all_shard_entries` list. \
             Ask `shard_holders(&shard_id)` for anything a person will read or a decision \
             will use — see docs/invariants/network.md § \"A holder record names a BUILD\"."
        );
    }
}

/// Does this function body take its holder count from the filtered accessor?
///
/// The shape scanned is the one both defects had: a `for` loop destructuring
/// `all_shard_entries()` into `(shard_id, holders)` and then counting the raw
/// `holders` binding. A binding that is deliberately unused (`_raw_holders`)
/// cannot be counted at all, which is why the fix renames it rather than
/// merely not calling `.len()` on it.
///
/// Statement-joined, so a chain rustfmt has wrapped across lines still reads
/// back as one expression (the trap that blinded six earlier guards).
fn counts_holders_through_the_filter(body: &str) -> bool {
    let stmts = statements(body);
    let Some(at) = stmts
        .iter()
        .position(|(_, s)| s.contains("all_shard_entries()"))
    else {
        // No raw iteration at all: nothing to get wrong.
        return true;
    };
    let binding = raw_holder_binding(&stmts[at].1);
    // The loop bodies here are 4-8 statements; 16 covers them with room and
    // stops before the next unrelated block.
    let window = &stmts[at..stmts.len().min(at + 16)];
    let counts_raw = binding.is_some_and(|name| {
        window
            .iter()
            .any(|(_, s)| s.contains(&format!("{name}.len()")))
    });
    let asks_filter = window.iter().any(|(_, s)| s.contains("shard_holders("));
    !counts_raw && asks_filter
}

/// The name a `for (.., holders) in .. all_shard_entries()` statement binds the
/// holder list to, or `None` when it is discarded (`_`, `_raw_holders`).
fn raw_holder_binding(stmt: &str) -> Option<String> {
    let open = stmt.find('(')? + 1;
    let close = stmt[open..].find(')')? + open;
    let name = stmt[open..close].split(',').nth(1)?.trim();
    if name.is_empty() || name.starts_with('_') {
        return None;
    }
    Some(name.to_string())
}

/// The scan above must fire on the real defect, not merely pass on the fix.
///
/// A guard that goes green the first time it is written has told you nothing
/// (gotchas #413/#503/#517), so both shapes of the defect are planted here.
#[test]
fn the_holder_count_guard_catches_both_shapes_of_the_defect() {
    let fixed = "
        for (shard_id, _raw) in registry.all_shard_entries() {
            let servable = registry.shard_holders(&shard_id).len();
            push(servable);
        }
    ";
    assert!(
        counts_holders_through_the_filter(fixed),
        "the fixed shape must pass"
    );

    // Shape 1: the count taken straight off the raw binding.
    let raw_count = "
        for (shard_id, holders) in registry.all_shard_entries() {
            push(holders.len());
        }
    ";
    assert!(
        !counts_holders_through_the_filter(raw_count),
        "a count off the raw holder list must fail"
    );

    // Shape 2: rustfmt has wrapped the chain, which is how six earlier guards
    // were blinded.
    let wrapped = "
        for (shard_id, holders) in registry
            .all_shard_entries()
        {
            push(
                holders
                    .len(),
            );
        }
    ";
    assert!(
        !counts_holders_through_the_filter(wrapped),
        "a wrapped chain must not hide the raw count"
    );

    // Shape 3: raw binding unused but the filtered accessor never asked.
    let neither = "
        for (shard_id, _raw) in registry.all_shard_entries() {
            push(guess());
        }
    ";
    assert!(
        !counts_holders_through_the_filter(neither),
        "discarding the raw list is not enough — the filtered count must be asked for"
    );
}

/// A node that KEEPS disagreeing bytes says so, on every path that decides to.
///
/// `ModelRegistry::mismatch_policy` answers "may we destroy our own copy?" and
/// `OnMismatch::KeepBytes` is the answer that means "no — keep serving these
/// bytes, the hash condemning them has no origin backing". `disputed_shards`
/// exists solely so that state is countable from OUTSIDE the log, because how
/// often it happens in the field is the open question that decides what to
/// build next (`docs/FUTURE_WORK.md` § "A disputed shard is kept but the
/// disagreement is never settled").
///
/// Three paths compute that policy and can land on `KeepBytes`. Until
/// 2026-09-14 only two recorded it, and the third was reached by exactly the
/// nodes most likely to be in this state — ones whose manifests came from peers
/// rather than from an origin. An isolated node logged the warning twice while
/// its own diagnostics reported "shards kept despite disagreeing (0) — none,
/// every checked shard matches its expected hash".
///
/// A count that reads zero while the thing it counts is happening is worse than
/// no count: it is read as evidence. The reading taken from the live node
/// earlier the same day had to be discarded because of this.
///
/// Sites that pass `KeepBytes` as a CONSTANT are deliberately not scanned —
/// `model::acquisition` and `auto_manage::download` use it to ask whether a
/// file is already there, which is a question, not an acceptance. The scan keys
/// on computing the policy, which is what distinguishes them.
#[test]
fn every_path_that_keeps_disagreeing_bytes_records_the_dispute() {
    let sites = [
        (
            "src/daemon/background.rs",
            "pub(super) fn spawn_shard_verification(",
        ),
        (
            "src/model/auto_manage/scan.rs",
            "pub async fn rescan_local_shards(",
        ),
        (
            "src/model/auto_manage/manager.rs",
            "async fn verify_pending_shards(",
        ),
    ];
    for (path, signature) in sites {
        let src = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let body = fn_body(&src, signature)
            .unwrap_or_else(|| panic!("{path}: signature not found: {signature}"));
        assert!(
            records_the_dispute_it_keeps(body),
            "{path}: {signature} decides to KEEP bytes that disagree with the swarm's \
             hash and never calls `note_shard_disputed`. The diagnostics report then \
             says zero while it is happening — see docs/invariants/network.md."
        );
    }
}

/// Does a body that can land on `KeepBytes` also record the dispute?
///
/// Keyed on COMPUTING the policy (`mismatch_policy(`), not on the enum name:
/// two call sites pass `KeepBytes` as a constant to ask whether a file is
/// already present, and recording a dispute there would be wrong.
fn records_the_dispute_it_keeps(body: &str) -> bool {
    let stmts = statements(body);
    let computes = stmts.iter().any(|(_, s)| s.contains("mismatch_policy("));
    if !computes {
        return true;
    }
    stmts
        .iter()
        .any(|(_, s)| s.contains("note_shard_disputed("))
}

/// The scan must fire on the real shape of the defect, not merely pass on the fix.
#[test]
fn the_dispute_recording_guard_catches_a_path_that_stays_silent() {
    let fixed = "
        let policy = registry.mismatch_policy(&sid, &info.hash);
        if policy == OnMismatch::KeepBytes {
            warn!(\"keeping our bytes\");
            self.shared_state.note_shard_disputed(&sid);
        }
    ";
    assert!(
        records_the_dispute_it_keeps(fixed),
        "the fixed shape must pass"
    );

    // The real defect: warns, keeps, records nothing.
    let silent = "
        let policy = registry.mismatch_policy(&sid, &info.hash);
        if policy == OnMismatch::KeepBytes {
            warn!(\"keeping our bytes\");
        }
    ";
    assert!(
        !records_the_dispute_it_keeps(silent),
        "a path that keeps disagreeing bytes silently must fail"
    );

    // rustfmt wraps the call across lines — the trap that blinded six guards.
    let wrapped = "
        let policy = registry
            .mismatch_policy(&sid, &info.hash);
        if policy == OnMismatch::KeepBytes {
            self.shared_state
                .note_shard_disputed(&sid);
        }
    ";
    assert!(
        records_the_dispute_it_keeps(wrapped),
        "a wrapped chain must still be seen as recording"
    );

    // A body that never computes the policy is out of scope, not a violation:
    // passing KeepBytes as a constant is how two call sites ask a question.
    let asks_a_question = "
        let ok = store.verify_shard(&mid, &info, OnMismatch::KeepBytes).is_ok();
    ";
    assert!(
        records_the_dispute_it_keeps(asks_a_question),
        "passing KeepBytes as a constant is a question, not an acceptance"
    );
}

/// A peer-connected entry in the user's activity list marks a TRANSITION.
///
/// libp2p re-pushes Identify constantly — `handle_identify_received` says so in
/// as many words at the top, where the foreign-peer INFO is gated on a set
/// insert for exactly that reason, and again 350 lines later where the "Peer
/// connected" log is gated on `connected_node_ids.insert`. The activity event
/// used to sit BETWEEN those two gates, ungated, so a peer that stayed
/// connected re-announced itself into the activity list indefinitely: three
/// consecutive identical "Computer connected: …" entries for one computer was
/// an ordinary sight, in a 100-entry ring that is also the replay a dashboard
/// opens on and the "recent activity" of the pasteable report.
///
/// This is the third time the same lesson has been learned for a log and not
/// carried to the surface a person reads (see `note_build_tag` and the shard
/// announcement feed, both 2026-09-14). The check is positional because that is
/// what the fix is: the emit must live INSIDE the transition gate.
#[test]
fn a_peer_connected_activity_entry_is_emitted_only_on_the_transition() {
    let src = std::fs::read_to_string("src/network/manager/identify.rs")
        .expect("identify.rs must be readable");
    let body = fn_body(&src, "pub(super) fn handle_identify_received(")
        .expect("handle_identify_received signature not found");
    let stmts = statements(body);
    let gate = stmts
        .iter()
        .position(|(_, s)| s.contains("connected_node_ids.insert("))
        .expect("the connected_node_ids transition gate must exist");
    let emit = stmts
        .iter()
        .position(|(_, s)| s.contains("\"peer_connected\""))
        .expect("the peer_connected activity event must exist");
    assert!(
        emit > gate,
        "src/network/manager/identify.rs: the `peer_connected` activity event is emitted \
         before the `connected_node_ids.insert` transition gate, so it fires on every \
         Identify rather than once per connection. Identify re-pushes constantly — see \
         docs/invariants/network.md."
    );
}

/// A failed fetch never destroys what the frontend already knew.
///
/// `frontend/js/core/data.js` is the single data-fetch choke point every
/// component is told to use, and `App.data.cache.*` is read DIRECTLY by a dozen
/// of them — the header status strip, the swarm tab, the network map, the
/// encryption toggle, the model-name helpers — as their source of truth between
/// loads. `fetch` does not reject on an HTTP error status, so a 401 from a
/// rotated key or a 503 during a restart arrives looking exactly like "this node
/// has nothing", and assigning that to the cache turned one transient failure
/// into "0 peers, no models" everywhere at once.
///
/// `notifications.js` also merges the WebSocket tick INTO `cache.stats`, so a
/// nulled cache additionally dropped every field that tick does not carry.
///
/// The fix is not to hide the failure — the `load*` helpers still RETURN what
/// the fetch produced, and `loadReachedDaemon` still tells a caller which it
/// was (item 69). It is that the CACHE holds on to the last thing it actually
/// knew. This scan pins that: every write to `cache.<field>` in a `load*`
/// helper is guarded by `loadReachedDaemon`.
#[test]
fn a_failed_fetch_never_overwrites_the_frontend_cache() {
    let src = std::fs::read_to_string("frontend/js/core/data.js")
        .expect("frontend/js/core/data.js must be readable");
    let mut unguarded = Vec::new();
    for (n, line) in src.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("//") || !t.starts_with("cache.") || !t.contains(" = ") {
            continue;
        }
        // The guard is written inline on the same statement:
        //   `if (loadReachedDaemon('stats')) cache.stats = stats;`
        // so an unguarded write is one whose line does not mention it.
        unguarded.push(format!("  line {}: {}", n + 1, t));
    }
    assert!(
        unguarded.is_empty(),
        "frontend/js/core/data.js writes the cache without checking the daemon was \
         reached, so one transient 401/503 replaces what a dozen components read \
         with an empty value:\n{}\nGuard each with `loadReachedDaemon('<key>')`.",
        unguarded.join("\n")
    );
}

/// The byte span of every `fn` body in `src`. Used to ask "does the function
/// containing this line also call X?", which a proximity window cannot answer
/// here: the widest real gap between a `finalize_reply_text` call and the
/// `InferenceOutput` it finalises is ~30 lines (`pipeline/distributed.rs`),
/// and a window that generous reaches into neighbouring functions. Taking the
/// whole body is the rule `arch-guards-and-tests.md` states, for the reason
/// the VRAM guard proved: a character window is brittle in both directions.
fn fn_bodies(src: &str) -> Vec<(usize, usize)> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = src[i..].find("fn ") {
        let at = i + rel;
        i = at + 3;
        // `fn` must begin a token, or `transfn (`-style names would match.
        if at > 0 && !bytes[at - 1].is_ascii_whitespace() {
            continue;
        }
        // Walk the signature to its opening brace. `;` first means a trait
        // method or an extern declaration, which has no body.
        let mut j = at;
        let mut parens = 0i32;
        let mut body_start = None;
        while j < bytes.len() {
            match bytes[j] {
                b'(' => parens += 1,
                b')' => parens -= 1,
                b'{' if parens <= 0 => {
                    body_start = Some(j);
                    break;
                }
                b';' if parens <= 0 => break,
                _ => {}
            }
            j += 1;
        }
        let Some(start) = body_start else { continue };
        let mut depth = 0i32;
        let mut k = start;
        while k < bytes.len() {
            match bytes[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        out.push((start, k));
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
    }
    out
}

/// The byte span of everything behind a `#[cfg(test)]`, whether it guards a
/// `mod` or a single `fn`. Test fixtures build `InferenceOutput`s freely and
/// must not be asked to finalise them — and the attribute sits on the
/// enclosing MODULE, so looking for it inside the function body finds
/// nothing. Four fixtures were reported as offenders before this existed.
fn cfg_test_spans(src: &str) -> Vec<(usize, usize)> {
    let bytes = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0usize;
    while let Some(rel) = src[i..].find("#[cfg(test)]") {
        let at = i + rel;
        i = at + 12;
        // First brace after the attribute opens the guarded item's body.
        let mut j = at;
        let mut parens = 0i32;
        let mut start = None;
        while j < bytes.len() {
            match bytes[j] {
                b'(' => parens += 1,
                b')' => parens -= 1,
                b'{' if parens <= 0 => {
                    start = Some(j);
                    break;
                }
                _ => {}
            }
            j += 1;
        }
        let Some(start) = start else { continue };
        let mut depth = 0i32;
        let mut k = start;
        while k < bytes.len() {
            match bytes[k] {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        out.push((start, k));
                        break;
                    }
                }
                _ => {}
            }
            k += 1;
        }
    }
    out
}

/// Every site in `text` that builds an `InferenceOutput` carrying content
/// without its enclosing function calling `finalize_reply_text`, as 1-indexed
/// line numbers. Separated from the test so the planted-violation self-test
/// below can drive it on a source string that is not on disk.
fn unfinalised_reply_sites(text: &str) -> Vec<usize> {
    let bodies = fn_bodies(text);
    let tests = cfg_test_spans(text);
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(rel) = text[from..].find("InferenceOutput {") {
        let at = from + rel;
        from = at + 1;
        if tests.iter().any(|(s, e)| *s < at && at < *e) {
            continue;
        }
        // The enclosing function is the SMALLEST body containing this offset.
        let Some((start, end)) = bodies
            .iter()
            .filter(|(s, e)| *s < at && at < *e)
            .min_by_key(|(s, e)| e - s)
            .copied()
        else {
            continue;
        };
        let body = &text[start..end];
        let literal_end = text[at..end].find('}').map(|o| at + o).unwrap_or(end);
        let literal = &text[at..literal_end];
        // A cancellation or a zero-`max_tokens` early return has no text to
        // finalise; a site with no `content` field at all is a signature or a
        // return type rather than a construction.
        if literal.contains("content: String::new()") || !literal.contains("content") {
            continue;
        }
        if !body.contains("finalize_reply_text") {
            out.push(text[..at].lines().count());
        }
    }
    out
}

/// **Every source of reply text ends by calling `finalize_reply_text`** — the
/// control-token scrub, the leading `<think>` reasoning block, stop truncation
/// and the newlines a removed marker strands. Its own doc comment has said so
/// since it was written, and saying so was not enough twice:
///
/// - `pipeline::remote_generate` never called it, so a reasoning model asked
///   over the swarm answered with its raw scratchpad while the same request
///   answered locally came back clean (gotcha #634).
/// - `finish_speculative` — the finaliser ALL THREE speculative coordinators
///   share — never called it either. It filtered EOS *ids*, which catches only
///   the markers a GGUF happened to declare, and applied no stops at all, so a
///   caller's `stop` was ignored on the DEFAULT distributed path (gotcha #643).
///
/// Both times the count in the prose was wrong — "three sources", then "five" —
/// and a count in a comment goes stale the moment a path is added. So the
/// property is asserted instead of the census: a function that builds an
/// `InferenceOutput` carrying content must also finalise it.
///
/// Exempt: `content: String::new()` (a cancellation or a zero-`max_tokens`
/// early return has nothing to finalise) and anything built through
/// `from_gen_result`, whose producers finalise before returning.
#[test]
fn every_reply_source_finalises_its_text() {
    let root = repo_root();
    let mut stack = vec![root.join("src")];
    let mut offenders: Vec<String> = Vec::new();
    let mut checked = 0usize;
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for e in rd.filter_map(|e| e.ok()) {
            let path = e.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if !path.extension().is_some_and(|x| x == "rs") {
                continue;
            }
            let rel = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            // A `tests.rs` beside its module is fixtures end to end; the
            // `#[cfg(test)]` span check covers in-file test modules.
            if rel.ends_with("/tests.rs") || rel.contains("/tests/") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            if !text.contains("InferenceOutput {") {
                continue;
            }
            checked += text.matches("InferenceOutput {").count();
            for line in unfinalised_reply_sites(&text) {
                offenders.push(format!("{rel}:{line}"));
            }
        }
    }
    assert!(
        checked >= 5,
        "only {checked} `InferenceOutput` constructions found anywhere in src/ \
         — the scan has stopped matching, which reads exactly like a rule \
         nobody breaks"
    );
    assert!(
        offenders.is_empty(),
        "these build an InferenceOutput carrying content without their function \
         finalising it — call crate::inference::finalize_reply_text (see \
         docs/invariants/api-surfaces.md § \"A reply is finalised on the \
         coordinator\"): {offenders:?}"
    );
}

/// The planted violation, kept. A repo-wide scan that finds nothing is
/// indistinguishable from one that CANNOT find anything (gotcha #413, where
/// four of five guards had been reporting success for months), so the
/// scanner's reach is pinned the way any other behaviour is.
///
/// The three shapes are the real ones: gotcha #643's actual pre-fix code, the
/// two exemptions, and a second construction added to a function that already
/// finalises elsewhere — which a file-level `contains` would have missed.
#[test]
fn the_finalisation_guard_catches_a_reply_source_that_skips_the_finaliser() {
    // #643's `finish_speculative`, as it actually stood: EOS ids filtered,
    // decoded, returned.
    let unfinalised = r#"
impl PipelineExecutor {
    fn finish_speculative(&self, generated: Vec<u32>) -> InferenceOutput {
        let clean: Vec<u32> = generated.into_iter().filter(|t| !eos.contains(t)).collect();
        let content = decoder.decode_tokens(&clean);
        InferenceOutput {
            request_id,
            content,
            matched_stop_sequence: None,
        }
    }
}
"#;
    assert_eq!(
        unfinalised_reply_sites(unfinalised).len(),
        1,
        "the guard must catch the shape it exists for"
    );

    // The same function, fixed.
    let finalised = unfinalised.replace(
        "let content = decoder.decode_tokens(&clean);",
        "let mut content = decoder.decode_tokens(&clean);\n        \
         crate::inference::finalize_reply_text(&mut content, stops);",
    );
    assert!(
        unfinalised_reply_sites(&finalised).is_empty(),
        "and must accept it once finalised"
    );

    // Exemption: an early return with no text in it.
    let empty = r#"
fn cancelled(&self) -> InferenceOutput {
    InferenceOutput {
        request_id,
        content: String::new(),
        finish_reason: "stop".to_string(),
    }
}
"#;
    assert!(
        unfinalised_reply_sites(empty).is_empty(),
        "a cancellation has nothing to finalise"
    );

    // A SECOND construction in a function that finalises for the first one is
    // still caught, because the check is per site — but a function-wide
    // `contains` cannot tell them apart, which is the limitation this records
    // rather than hides: both sites here sit in one body, so one
    // `finalize_reply_text` covers both. The guard's reach is the FUNCTION.
    let two_in_one = r#"
fn two(&self) -> InferenceOutput {
    crate::inference::finalize_reply_text(&mut a, stops);
    if x {
        return InferenceOutput { request_id, content: a };
    }
    InferenceOutput { request_id, content: b }
}
"#;
    assert!(
        unfinalised_reply_sites(two_in_one).is_empty(),
        "documented reach: one finaliser call covers its whole function — a \
         new reply source belongs in its own function, which is how both real \
         instances were shaped"
    );

    // And the scanner must actually be resolving bodies, not matching the
    // whole file: an unfinalised construction beside a finalising neighbour
    // is the case a file-level check gets wrong.
    let neighbours = format!(
        "fn ok() -> InferenceOutput {{\n    \
         crate::inference::finalize_reply_text(&mut c, stops);\n    \
         InferenceOutput {{ request_id, content: c }}\n}}\n{unfinalised}"
    );
    assert_eq!(
        unfinalised_reply_sites(&neighbours).len(),
        1,
        "a finalising function must not vouch for its neighbour"
    );

    // Test fixtures are exempt, and the attribute sits on the enclosing
    // MODULE — which is why looking for it inside the function body found
    // nothing and reported four of them as offenders.
    let fixture = format!("#[cfg(test)]\nmod t {{\n{unfinalised}\n}}\n");
    assert!(
        unfinalised_reply_sites(&fixture).is_empty(),
        "a #[cfg(test)] module's fixtures must not be asked to finalise"
    );
    // …and the exemption must not leak past the module's closing brace.
    let after = format!("#[cfg(test)]\nmod t {{\n    fn noop() {{}}\n}}\n{unfinalised}");
    assert_eq!(
        unfinalised_reply_sites(&after).len(),
        1,
        "the test exemption must end with the module"
    );
}
