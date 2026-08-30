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
                    if l.contains(&format!("{name}.remove(")) {
                        offenders.push(format!("{rel}:{line_no}: {l}"));
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
                for (i, line) in src.lines().enumerate() {
                    let l = line.trim();
                    if l.starts_with("//") || l.starts_with("///") {
                        continue;
                    }
                    // Reading the legacy field anywhere under the HTTP surface
                    // means something user-facing is describing update
                    // behaviour from a value that does not determine it.
                    if l.contains("updates.auto_update") || l.contains("AutoUpdateMode::") {
                        offenders.push(format!(
                            "{}:{}: {}",
                            p.strip_prefix(&root).unwrap_or(&p).display(),
                            i + 1,
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
        for (i, line) in text.lines().enumerate() {
            let l = line.trim();
            if l.starts_with("//") || l.starts_with("///") {
                continue;
            }
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
                offenders.push(format!("{rel}:{}: {l}", i + 1));
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
                for (i, line) in text.lines().enumerate() {
                    // Match the FREE-FUNCTION form too, not just `self.swarm`.
                    // This test read `self.swarm.dial(` only, and
                    // `discovery::bootstrap_peers` took `&mut Swarm` and called
                    // `swarm.dial(addr)` — so the one dial site that mattered
                    // was invisible to the audit that added this test, and went
                    // on dialling every cached address unconditionally and
                    // ungated for another release (gotcha #405).
                    let call = line.trim_start();
                    // Skip comments, or this trips over the doc comments that
                    // explain the rule — which is what happened the first time.
                    let is_comment = call.starts_with("//");
                    if !is_comment && call.contains("swarm.dial(") {
                        sites.push(format!("{}:{}", p.display(), i + 1));
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
            if !p.extension().is_some_and(|x| x == "md") {
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
        "markdown files contain NUL bytes, which makes grep skip them silently:\n  {}\n\
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
