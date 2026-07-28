//! Operational config: logging, UI, update, and HTTP API surfaces.
//!
//! Hosts `LoggingConfig` (level/format/file), `UiConfig` (browser/theme),
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
    #[serde(default = "default_theme")]
    pub theme: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            open_browser_on_start: true,
            theme: default_theme(),
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

impl UpdateConfig {
    /// The mode actually in force, migrating a pre-`mode` config.
    ///
    /// A legacy `auto_update` of `stable`/`all` was an explicit opt-in to
    /// automatic downloads and is preserved as `Download`. Legacy `disabled`
    /// becomes `Notify`, NOT `Off`: it was the shipped default rather than a
    /// decision, and it suppressed the update check entirely — so nodes went on
    /// running old builds with nothing ever telling anyone. `mode = "off"` is
    /// how you actually opt out now.
    pub fn effective_mode(&self) -> UpdateMode {
        match self.mode {
            Some(m) => m,
            None => match self.auto_update {
                AutoUpdateMode::Disabled => UpdateMode::Notify,
                AutoUpdateMode::Stable | AutoUpdateMode::All => UpdateMode::Download,
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
            mode: Some(UpdateMode::Notify),
            auto_update: AutoUpdateMode::Disabled,
            check_interval_hours: default_check_interval_hours(),
            include_prereleases: true,
        }
    }
}

// Auto-update default is `Disabled` per docs/ARCHITECTURE.md "Key Design
// Decisions" and the C1 deferred-item note: until binary signing is wired,
// every node opting in to auto-update is downloading SHA256-only-verified
// binaries from GitHub. Default-disabled is the documented safe posture;
// users opt-in via `[update] auto_update = "stable"` in config.toml.
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

fn default_theme() -> String {
    "dark".into()
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
    #[test]
    fn a_legacy_config_starts_getting_notified() {
        let cfg: UpdateConfig = toml::from_str("auto_update = \"disabled\"").unwrap();
        assert_eq!(cfg.mode, None, "old configs have no mode key");
        assert_eq!(cfg.effective_mode(), UpdateMode::Notify);
    }

    /// ...but a deliberate opt-in to automatic downloads must not be downgraded.
    #[test]
    fn a_legacy_opt_in_is_preserved() {
        for legacy in ["stable", "all"] {
            let cfg: UpdateConfig = toml::from_str(&format!("auto_update = \"{legacy}\"")).unwrap();
            assert_eq!(
                cfg.effective_mode(),
                UpdateMode::Download,
                "legacy auto_update = {legacy} opted in to downloading"
            );
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
        assert_eq!(cfg.effective_mode(), UpdateMode::Notify);
        // A fresh install checks often enough to matter when several releases
        // can ship in one day.
        assert!(cfg.check_interval_hours <= 1);
    }
}
