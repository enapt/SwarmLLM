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
    #[serde(default = "default_auto_update")]
    pub auto_update: AutoUpdateMode,
    #[serde(default = "default_check_interval_hours")]
    pub check_interval_hours: u32,
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
            auto_update: AutoUpdateMode::Stable,
            check_interval_hours: default_check_interval_hours(),
        }
    }
}

fn default_auto_update() -> AutoUpdateMode {
    AutoUpdateMode::Stable
}

fn default_check_interval_hours() -> u32 {
    6
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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
