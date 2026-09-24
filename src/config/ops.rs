//! Operational config: logging, UI, update, and HTTP API surfaces.
//!
//! Hosts `LoggingConfig` (level/format/file), `UiConfig` (browser),
//! `UpdateConfig` + `AutoUpdateMode`, and `ApiConfig` (api_key + rate
//! limits). UpdateConfig has its own Default impl since AutoUpdateMode
//! is non-trivial; LoggingConfig's Default lives here too rather than in
//! a separate impl block at the bottom.

use super::default_true;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default = "default_log_format")]
    pub format: String,
    #[serde(default)]
    pub file: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default = "default_true")]
    pub open_browser_on_start: bool,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            open_browser_on_start: true,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateConfig {
    /// What this node does about updates. `None` means the config predates this
    /// field, in which case [`UpdateConfig::effective_mode`] derives it from
    /// the legacy `auto_update` setting.
    ///
    /// Deliberately `Option` rather than a defaulted enum: a serde default only
    /// fills a key that is ABSENT, and the daemon serialises every field, so
    /// every existing config on disk already says `auto_update = "disabled"` —
    /// not because anyone chose it, but because it was the default when that
    /// file was written. A new key is the one thing those configs don't have,
    /// so it is the only way a new default can actually reach them. Same trap
    /// as `bootstrap_peers = []` (gotcha #198).
    #[serde(default)]
    pub mode: Option<UpdateMode>,
    /// Legacy setting, kept so old configs keep working and so a deliberate
    /// `auto_update = "all"` is not silently downgraded. Superseded by `mode`.
    #[serde(default = "default_auto_update")]
    pub auto_update: AutoUpdateMode,
    #[serde(default = "default_check_interval_hours")]
    pub check_interval_hours: u32,
    /// Offer pre-release builds. Defaults TRUE because every release this
    /// project has ever published is tagged `-alpha`: excluding pre-releases
    /// would mean a node never sees any update at all, which is how
    /// `auto_update = "stable"` came to be a setting that silently did nothing.
    #[serde(default = "default_true")]
    pub include_prereleases: bool,
    /// `swarmllm run --no-update-check`: this PROCESS never contacts GitHub,
    /// whatever any file says.
    ///
    /// A fact about how the process was started, so no file can carry it —
    /// `serde(skip)` — and [`crate::daemon::SharedState::apply_live_config`]
    /// carries it across every live-config swap, because a settings save
    /// rebuilds the live config from `config.toml` and would otherwise drop it
    /// at the first click. Command line beats file, as everywhere else.
    ///
    /// The flag used to set the legacy `auto_update = "disabled"`, which
    /// [`UpdateConfig::effective_mode`] has resolved to `Install` since
    /// v0.3.191 — so the one switch that promised no updates changed nothing,
    /// and a node started with it updated and restarted itself anyway.
    #[serde(skip)]
    pub off_for_this_process: bool,
}

/// What a node does when a newer release exists.
///
/// Ordered by how much it does on its own; each level includes the one before.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum UpdateMode {
    /// Never contact GitHub. Nothing is checked, downloaded or shown.
    Off,
    /// Check and tell the user. Nothing is downloaded until they ask.
    Notify,
    /// Check and download in the background; installing stays a click.
    Download,
    /// Check, download, install and restart once the node is idle.
    Install,
}

impl UpdateMode {
    /// Wire name, matching the `[updates] mode` config value.
    pub fn as_str(self) -> &'static str {
        match self {
            UpdateMode::Off => "off",
            UpdateMode::Notify => "notify",
            UpdateMode::Download => "download",
            UpdateMode::Install => "install",
        }
    }
}

impl UpdateConfig {
    /// The mode actually in force, migrating a pre-`mode` config.
    ///
    /// **A config that never chose resolves to `Install` (2026-09-19).** It was
    /// `Notify` until release signing landed, and `Notify` never installs — so
    /// a fleet of default nodes only moved when each operator acted. Measured
    /// on the live swarm that day: of five peers, two were still two releases
    /// behind after ~16 h and ~4.5 h of uptime, having checked roughly sixteen
    /// and four times each. They were not failing; they were waiting for a
    /// human who was never coming.
    ///
    /// **What made this safe to change is `crate::update_signature`, not a
    /// change of mind.** Unattended self-replacement was held back behind audit
    /// item C1 for as long as a release was authenticated only by a checksum
    /// published beside it, because anyone who could swap the binary could swap
    /// the checksum. Now an update must carry a signature from a key that never
    /// exists in CI. **If that verification is ever weakened, this default has
    /// to go back** — the two are one decision.
    ///
    /// The legacy field no longer distinguishes anything: `disabled` was the
    /// shipped default rather than a decision, and `stable`/`all` were opt-ins
    /// to *less* than the default now does. All three resolve the same way, and
    /// an explicit `mode` still wins — `off`, `notify` and `download` are how
    /// you opt out, in increasing order of what you keep.
    pub fn effective_mode(&self) -> UpdateMode {
        if self.off_for_this_process {
            return UpdateMode::Off;
        }
        match self.mode {
            Some(m) => m,
            None => match self.auto_update {
                AutoUpdateMode::Disabled | AutoUpdateMode::Stable | AutoUpdateMode::All => {
                    UpdateMode::Install
                }
            },
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AutoUpdateMode {
    Disabled,
    Stable,
    All,
}

impl Default for UpdateConfig {
    fn default() -> Self {
        Self {
            // MUST stay `None`, matching the `#[serde(default)]` on the field.
            // `None` is the designed "not explicitly set" state that
            // `effective_mode` resolves; hardcoding a mode here made the answer
            // depend on whether the `[updates]` *section* happened to exist — a
            // section with no `mode` key deserialized to `None` while a missing
            // section used this impl and got the hardcoded value. Leaving it
            // `None` means both routes go through `effective_mode` and cannot
            // disagree, whatever that function decides today.
            mode: None,
            auto_update: AutoUpdateMode::Disabled,
            check_interval_hours: default_check_interval_hours(),
            include_prereleases: true,
            off_for_this_process: false,
        }
    }
}

// This is the LEGACY field's default and it no longer decides anything on its
// own — `effective_mode` resolves every value of it to `Install`. It stays
// `Disabled` because it is written into every config file on disk and changing
// the literal would rewrite those files to no purpose.
//
// It used to carry the C1 posture: while a release was authenticated only by a
// SHA256 sidecar published beside it, default-disabled was the documented safe
// answer. Release signing (`src/update_signature.rs`) is what retired that
// argument on 2026-09-19 — not a reassessment of the risk.
//
// **The section is `[updates]`, plural.** This said `[update]` for a long time.
// An unknown section warns and is ignored, so anyone following it set nothing
// and got the defaults — the shape of a 2026-08-09 report from an operator who
// believed they had set `notify` and watched their node install twice anyway.
fn default_auto_update() -> AutoUpdateMode {
    AutoUpdateMode::Disabled
}

/// Hourly. Six hours was chosen when releases were rare; during alpha several
/// can ship in a day, so a six-hour window means a node usually reports an
/// update that is already superseded — and the operator updates by hand rather
/// than waiting. The check is one small GitHub request and a no-op when
/// current.
fn default_check_interval_hours() -> u32 {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApiConfig {
    /// Bearer token for API authentication. If empty, one is auto-generated on first run.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Rate limit (requests per minute) for `/v1/` and `/api/chat` endpoints.
    /// Default: 60.
    #[serde(default)]
    pub rate_limit_rpm: Option<u64>,
    /// Rate limit (requests per minute) for `/api/admin/` endpoints.
    /// Default: 200.
    #[serde(default)]
    pub rate_limit_admin_rpm: Option<u64>,
    /// Require Bearer auth on `/metrics` even from loopback.
    ///
    /// Default `false` — matches the Prometheus "metrics endpoints are
    /// unauthenticated" convention and keeps the existing dashboard's
    /// loopback scrape working without a token. When `true`, /metrics
    /// goes through the normal `auth_middleware` regardless of source
    /// IP, so a Prometheus scraper must set
    /// `Authorization: Bearer <api_key>` in its scrape config.
    ///
    /// R138 (closes R101/R102 deferrals about /metrics disclosing the
    /// credit balance on publicly-reachable nodes): operator-facing
    /// dial. Public nodes that expose port 8800 to the internet
    /// should set this to `true`.
    #[serde(default)]
    pub metrics_auth_required: bool,
    /// Hand the dashboard its API key when the browser reaches us over a
    /// Tailscale-style overlay (100.64.0.0/10, `fd7a:115c:a1e0::/48`).
    ///
    /// Default `true`, but it only takes effect when THIS node is itself on
    /// such an overlay — see `api::dashboard_trust::node_is_on_overlay`. We
    /// document running nodes over Tailscale, and a dashboard that 401s on
    /// the tailnet makes remote nodes unmanageable. Membership of a tailnet
    /// is an authenticated act (the device was authorised into it), which is
    /// a stronger claim than being on the same LAN.
    ///
    /// Set `false` on a node whose overlay you share with people you would
    /// not give admin access to.
    #[serde(default = "default_true")]
    pub dashboard_trust_overlay: bool,
    /// Hand the dashboard its API key when the browser reaches us from a
    /// private/LAN address (RFC1918, IPv6 ULA, link-local).
    ///
    /// Default `false` — a LAN is not an authenticated boundary, and this
    /// grants admin + inference to anything on it. It exists because a
    /// Tailscale *subnet router* masquerades by default, so traffic from a
    /// tailnet arrives from the router's own private address and is
    /// indistinguishable from any other LAN client (see
    /// `docs/book/src/operations/tailscale-wan.md`). Users in that topology
    /// turn this on deliberately, from the dashboard, once.
    #[serde(default)]
    pub dashboard_trust_lan: bool,
}

/// Written out by hand rather than derived.
///
/// `#[derive(Default)]` does NOT consult `#[serde(default = "...")]` — the two
/// are unrelated mechanisms. The serde attribute only fills a key missing from
/// the TOML being parsed, whereas a node starting with no config file at all
/// goes through `Default`. Deriving it therefore shipped
/// `dashboard_trust_overlay = false` to exactly the fresh installs the default
/// exists for, and wrote that `false` back to the generated config.toml where
/// it then looked deliberate.
impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            rate_limit_rpm: None,
            rate_limit_admin_rpm: None,
            metrics_auth_required: false,
            dashboard_trust_overlay: default_true(),
            dashboard_trust_lan: false,
        }
    }
}

fn default_log_level() -> String {
    "info".into()
}

fn default_log_format() -> String {
    "pretty".into()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            format: default_log_format(),
            file: None,
        }
    }
}

#[cfg(test)]
mod update_mode_tests {
    use super::*;

    /// The reason `mode` is an Option. Every config the daemon has ever written
    /// contains `auto_update = "disabled"` — the shipped default, not a choice —
    /// and that value suppressed the update check entirely, so nodes ran old
    /// builds with nothing ever saying so.
    ///
    /// It resolved to `Notify` from then until 2026-09-19, which told those
    /// nodes about releases without ever installing one. Now that a release
    /// must be signed, it resolves to `Install`.
    #[test]
    fn a_legacy_config_keeps_itself_up_to_date() {
        let cfg: UpdateConfig = toml::from_str("auto_update = \"disabled\"").unwrap();
        assert_eq!(cfg.mode, None, "old configs have no mode key");
        assert_eq!(cfg.effective_mode(), UpdateMode::Install);
    }

    /// `swarmllm run --no-update-check` set `auto_update = "disabled"`, and the
    /// test above is exactly why that did nothing: the legacy value resolves
    /// to `Install`. The flag has its own field now, which must beat every mode
    /// a file can hold — and which no file can hold.
    #[test]
    fn the_command_line_switch_turns_updates_off_whatever_the_file_says() {
        for mode in [
            None,
            Some(UpdateMode::Install),
            Some(UpdateMode::Download),
            Some(UpdateMode::Notify),
            Some(UpdateMode::Off),
        ] {
            let cfg = UpdateConfig {
                mode,
                off_for_this_process: true,
                ..Default::default()
            };
            assert_eq!(cfg.effective_mode(), UpdateMode::Off, "file mode {mode:?}");
        }

        let cfg = UpdateConfig {
            off_for_this_process: true,
            ..Default::default()
        };
        let written = toml::to_string(&cfg).unwrap();
        assert!(
            !written.contains("off_for_this_process"),
            "a command-line fact must not be written into config.toml"
        );
        let read: UpdateConfig = toml::from_str("off_for_this_process = true\n").unwrap();
        assert!(
            !read.off_for_this_process,
            "and a file must not be able to claim it"
        );
    }

    /// A legacy opt-in must never come out as LESS than a config that chose
    /// nothing at all. `stable`/`all` meant "update me automatically" and
    /// mapped to `Download` while the default was `Notify`; once the default
    /// became `Install`, leaving them at `Download` would have quietly demoted
    /// the people who had asked for this all along.
    #[test]
    fn a_legacy_opt_in_is_never_weaker_than_the_default() {
        let default_mode = UpdateConfig::default().effective_mode();
        for legacy in ["stable", "all"] {
            let cfg: UpdateConfig = toml::from_str(&format!("auto_update = \"{legacy}\"")).unwrap();
            assert!(
                cfg.effective_mode() >= default_mode,
                "legacy auto_update = {legacy} opted IN, so it must not resolve \
                 to less than the default ({default_mode:?})"
            );
            assert_eq!(cfg.effective_mode(), UpdateMode::Install);
        }
    }

    /// An explicit mode always wins, including the one that turns everything off.
    #[test]
    fn an_explicit_mode_overrides_the_legacy_field() {
        let cfg: UpdateConfig = toml::from_str("mode = \"off\"\nauto_update = \"all\"").unwrap();
        assert_eq!(cfg.effective_mode(), UpdateMode::Off);

        let cfg: UpdateConfig =
            toml::from_str("mode = \"install\"\nauto_update = \"disabled\"").unwrap();
        assert_eq!(cfg.effective_mode(), UpdateMode::Install);
    }

    /// Ordering is load-bearing: the loop gates downloading on `>= Download`
    /// and installing on `== Install`.
    #[test]
    fn modes_are_ordered_by_how_much_they_do() {
        assert!(UpdateMode::Off < UpdateMode::Notify);
        assert!(UpdateMode::Notify < UpdateMode::Download);
        assert!(UpdateMode::Download < UpdateMode::Install);
    }

    /// Every release is tagged `-alpha`; excluding pre-releases would mean
    /// never finding an update at all, which is what `auto_update = "stable"`
    /// silently did.
    #[test]
    fn prereleases_are_included_by_default() {
        let cfg = UpdateConfig::default();
        assert!(cfg.include_prereleases);
        assert_eq!(cfg.effective_mode(), UpdateMode::Install);
        // A fresh install checks often enough to matter when several releases
        // can ship in one day.
        assert!(cfg.check_interval_hours <= 1);
    }

    /// The reported update mode must be what the node DOES, not the legacy
    /// `auto_update` field it is derived from.
    ///
    /// `auto_update` defaults to `Disabled`, and `effective_mode` deliberately
    /// does not honour that literally. `GET /api/admin/version` reported the
    /// legacy field instead and therefore answered "disabled" on a node that
    /// was checking on schedule, with a populated `last_checked` sitting next
    /// to it (observed live 2026-08-10). Anyone asking "will this node tell me
    /// about a release?" got the wrong answer from the endpoint built to
    /// answer it — and the gap is wider now that the stock answer is that the
    /// node installs by itself.
    #[test]
    fn a_stock_install_reports_that_it_keeps_itself_updated() {
        let cfg = UpdateConfig::default();
        assert_eq!(cfg.auto_update, AutoUpdateMode::Disabled, "precondition");
        assert_eq!(
            cfg.effective_mode().as_str(),
            "install",
            "a default node installs — reporting the legacy field here is how \
             it came to claim updates were disabled"
        );
    }

    #[test]
    fn every_update_mode_has_a_distinct_wire_name() {
        let all = [
            UpdateMode::Off,
            UpdateMode::Notify,
            UpdateMode::Download,
            UpdateMode::Install,
        ];
        let names: Vec<&str> = all.iter().map(|m| m.as_str()).collect();
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "duplicate wire name: {names:?}");
        // The names are the `[updates] mode` config vocabulary; they must round
        // trip so the value reported is one a user can paste back into config.
        for (m, n) in all.iter().zip(&names) {
            let parsed: UpdateMode = toml::from_str(&format!("mode = \"{n}\""))
                .map(|c: UpdateConfig| c.mode.unwrap())
                .unwrap_or_else(|e| panic!("wire name {n} is not a valid config value: {e}"));
            assert_eq!(parsed, *m);
        }
    }
}
