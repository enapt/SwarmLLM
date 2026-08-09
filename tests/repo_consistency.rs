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

    // Every quoted "/api/..." literal in the router source.
    let mut routes: BTreeSet<String> = BTreeSet::new();
    for (i, _) in server.match_indices("\"/api/") {
        let rest = &server[i + 1..];
        if let Some(end) = rest.find('"') {
            routes.insert(normalise(&rest[..end]));
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
        for (i, line) in text.lines().enumerate() {
            let l = line.trim();
            if l.starts_with("//") || l.starts_with("///") {
                continue;
            }
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
                offenders.push(format!("{rel}:{}: {l}", i + 1));
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
        ".config.auto_manage.",
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
            let mut in_tests = false;
            for (i, line) in text.lines().enumerate() {
                let l = line.trim();
                if l.starts_with("#[cfg(test)]") {
                    in_tests = true;
                }
                if in_tests || l.starts_with("//") || l.starts_with("///") {
                    continue;
                }
                if mutable.iter().any(|m| l.contains(m)) {
                    offenders.push(format!("{rel}:{}: {l}", i + 1));
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
#[test]
fn per_request_state_is_released_in_one_place() {
    let root = repo_root();
    let guarded = [
        "active_pipelines",
        "active_traces",
        "request_holder_blacklist",
    ];
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
            let mut in_tests = false;
            for (i, line) in text.lines().enumerate() {
                let l = line.trim();
                if l.starts_with("#[cfg(test)]") {
                    in_tests = true;
                }
                if in_tests || l.starts_with("//") || l.starts_with("///") {
                    continue;
                }
                for name in guarded {
                    if l.contains(&format!("{name}.remove(")) {
                        offenders.push(format!("{rel}:{}: {l}", i + 1));
                    }
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "per-request state is removed outside `SharedState::release_request_state`.\n\
         A path that owns a pipeline must clear all three together — clearing a \
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
