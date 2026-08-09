use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::error::SwarmError;
// Re-export ContributionMode from swarmllm-types crate
pub use crate::types::ContributionMode;

/// Settings a running subsystem must **react** to, not merely re-read.
///
/// Most settings need nothing but [`crate::daemon::SharedState::cfg`]: the code
/// that acts on them reads the live config each time it runs. These are the
/// exceptions — each one sizes something built once, so somebody has to be told
/// to rebuild it (resize the concurrency limit, retime an interval, change a
/// cache's expiry).
///
/// **Keep this list to things with a consumer.** It previously carried five
/// fields that nothing anywhere read — `max_peers`, `contribution`,
/// `contribution_auto`, `max_gpu_vram_mb`, `session_timeout_secs` — while its
/// own doc comment said they "can be changed without restart". A struct that
/// announces an invariant it does not keep reads as verification and stops
/// anyone checking (gotcha #281).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationalParams {
    pub max_concurrent_requests: u32,
    pub auto_manage_interval_minutes: u32,
    pub max_batch_size: u32,
    pub batch_timeout_ms: u64,
    pub session_timeout_secs: u64,
}

impl OperationalParams {
    /// Extract operational params from a full Config.
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_concurrent_requests: config.inference.max_concurrent_requests,
            auto_manage_interval_minutes: config.auto_manage.interval_minutes,
            max_batch_size: config.inference.max_batch_size,
            batch_timeout_ms: config.inference.batch_timeout_ms,
            session_timeout_secs: config.inference.session_timeout_seconds,
        }
    }
}

/// Reload only operational (hot-reloadable) parameters from the config file.
/// Returns the new params or an error if the file cannot be read/parsed.
pub fn reload_operational_params(config_path: &Path) -> Result<OperationalParams, SwarmError> {
    if !config_path.exists() {
        return Err(SwarmError::Config(format!(
            "Config file not found: {}",
            config_path.display()
        )));
    }
    let contents = std::fs::read_to_string(config_path).map_err(SwarmError::Io)?;
    let config: Config = toml::from_str(&contents).map_err(|e| {
        SwarmError::Config(format!("Failed to parse {}: {e}", config_path.display()))
    })?;
    Ok(OperationalParams::from_config(&config))
}

/// Top-level configuration for the SwarmLLM daemon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub node: NodeConfig,
    #[serde(default)]
    pub resources: ResourceConfig,
    #[serde(default)]
    pub network: NetworkConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub ui: UiConfig,
    #[serde(default)]
    pub updates: UpdateConfig,
    #[serde(default)]
    pub pool: PoolConfig,
    #[serde(default)]
    pub api: ApiConfig,
    #[serde(default)]
    pub auto_manage: AutoManageConfig,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub identity: IdentityConfig,
    #[serde(default)]
    pub providers: ProvidersConfig,
}

/// Strip every value that is identical to the compiled default.
///
/// **Why this exists.** The daemon used to serialize the WHOLE `Config` on every
/// save, and `PUT /api/admin/config` is called by the setup wizard on "Start
/// SwarmLLM" — so every field landed on disk as an explicit value. A
/// `#[serde(default)]` only fills a key that is *missing*, which means that once
/// a field is written, no future change to its default can ever reach that
/// install. The user sees a config that looks deliberately chosen and has no way
/// to tell it apart from one they actually chose.
///
/// This has now caused three separate user-visible faults: `bootstrap_peers = []`
/// stranding every pre-2026-07-21 node with no bootstrap (gotcha #198), a
/// default-on dashboard flag shipping off to the fresh installs it was written
/// for, and `check_interval_hours = 6` keeping nodes on a six-hour update check
/// after the default became hourly.
///
/// Writing only what differs from the default keeps defaults *live*: a key the
/// user never set stays absent, so it follows the compiled value across
/// upgrades. A user who explicitly set a value equal to the default has it
/// dropped, and will likewise follow the default if it later changes — which is
/// the intended reading of "I did not override this".
///
/// Safe by construction: `empty_toml_parses_to_full_default` proves every field
/// has a serde default, so a fully pruned file always reloads to the same
/// effective config.
///
/// Returns true when `value` is left empty and the caller may drop it entirely.
pub(crate) fn prune_defaults(value: &mut toml::Value, defaults: &toml::Value) -> bool {
    let (v, d) = match (value, defaults) {
        (toml::Value::Table(v), toml::Value::Table(d)) => (v, d),
        _ => return false,
    };
    let mut drop_keys = Vec::new();
    for (k, val) in v.iter_mut() {
        let Some(dv) = d.get(k) else { continue };
        // Identical to the default, or a sub-table left empty once its own
        // defaults were pruned out.
        if val == dv || (val.is_table() && dv.is_table() && prune_defaults(val, dv)) {
            drop_keys.push(k.clone());
        }
    }
    for k in drop_keys {
        v.remove(&k);
    }
    v.is_empty()
}

/// Serialize a config, emitting only what differs from the compiled defaults.
pub fn to_minimal_toml(config: &Config) -> Result<String, SwarmError> {
    let mut value = toml::Value::try_from(config)
        .map_err(|e| SwarmError::Internal(format!("Failed to serialize config: {e}")))?;
    let defaults = toml::Value::try_from(Config::default())
        .map_err(|e| SwarmError::Internal(format!("Failed to serialize default config: {e}")))?;
    prune_defaults(&mut value, &defaults);
    toml::to_string_pretty(&value)
        .map_err(|e| SwarmError::Internal(format!("Failed to serialize config to TOML: {e}")))
}

/// Reset values that are stranded copies of a superseded default.
///
/// The pruning above stops this recurring, but it cannot help a config that
/// already carries the old value — that key is present, so it keeps winning.
/// Each entry here is a default the daemon itself wrote, which a later release
/// changed, and which was never exposed in any UI — so a value matching the old
/// default is overwhelmingly the daemon's, not a deliberate choice.
///
/// Keep this list SHORT and delete entries once the affected installs are gone.
/// Anything that a user could plausibly have chosen on purpose does not belong
/// here: silently changing a deliberate setting is worse than a stale default.
pub(crate) fn migrate_superseded_defaults(config: &mut Config, source: &str) {
    // Update checks moved 6h -> 1h when releases became several-per-day. There
    // is no UI for this field, so 6 on disk is the daemon's old default.
    const SUPERSEDED_UPDATE_INTERVAL_HOURS: u32 = 6;
    if config.updates.check_interval_hours == SUPERSEDED_UPDATE_INTERVAL_HOURS {
        let fresh = UpdateConfig::default().check_interval_hours;
        if fresh != SUPERSEDED_UPDATE_INTERVAL_HOURS {
            tracing::info!(
                from = SUPERSEDED_UPDATE_INTERVAL_HOURS,
                to = fresh,
                source,
                "Update-check interval was a stranded copy of an old default; using the current default"
            );
            config.updates.check_interval_hours = fresh;
        }
    }

    // `max_peers` was inert until 2026-08-04 — parsed and displayed, never
    // enforced. For most of the project's life the daemon also wrote every
    // field to disk, so an existing config almost certainly carries the old
    // default of 200 whether or not anyone chose it.
    //
    // Now that the key does something, leaving that value in place would read
    // as a deliberate choice and override the contribution-derived ceiling. It
    // cannot be one: nobody could have tuned a setting that had no effect. This
    // is exactly the case the migration list is for — a value that was the
    // daemon's, not the user's.
    const SUPERSEDED_MAX_PEERS: u32 = 200;
    if config.network.max_peers == Some(SUPERSEDED_MAX_PEERS) {
        tracing::info!(
            source,
            "max_peers was a stranded copy of an old default (and was never enforced \
             until now); resolving it from the contribution mode instead"
        );
        config.network.max_peers = None;
    }
}

/// Warn about keys in the config file that no longer (or never did) exist.
///
/// serde ignores unknown fields, which is what keeps an older binary able to
/// read a newer config — but it also means a typo, a key in the wrong section,
/// or a setting copied from a blog post does nothing at all and says nothing.
/// Reported 2026-07-29 by a user who set `disable_default_bootstrap` and
/// `enable_upnp`, saw "Loaded config" in the log, and reasonably concluded the
/// settings were wired up and broken.
///
/// `deny_unknown_fields` is deliberately NOT used: refusing to start because a
/// config mentions a key from a later release is worse than ignoring it. A
/// warning names the key and moves on.
///
/// Tables that are empty in the schema are treated as free-form maps
/// (`auto_manage.model_policies`, `identity`) and are not descended into,
/// since any key inside them is user data rather than a setting name.
pub(crate) fn collect_unknown_config_keys(
    file: &toml::Value,
    schema: &toml::Value,
    path: &str,
    out: &mut Vec<String>,
) {
    let (f, s) = match (file, schema) {
        (toml::Value::Table(f), toml::Value::Table(s)) => (f, s),
        _ => return,
    };
    if s.is_empty() {
        return; // free-form map — its keys are values, not setting names
    }
    for (k, v) in f {
        let full = if path.is_empty() {
            k.clone()
        } else {
            format!("{path}.{k}")
        };
        match s.get(k) {
            None => out.push(full),
            Some(sv) => collect_unknown_config_keys(v, sv, &full, out),
        }
    }
}

/// Check a parsed config file for keys the daemon does not recognise.
///
/// The schema is the user's OWN file round-tripped through `Config`, not
/// `Config::default()`. TOML cannot represent null, so serializing the
/// defaults omits every `Option` field that defaults to `None` — which made
/// the check report working settings as ignored. Observed 2026-07-29:
/// `inference.max_seq_len_override`, `api.rate_limit_rpm`, `logging.file` and
/// `node.region` were all announced as "being IGNORED" while taking effect,
/// whereas `inference.kv_cache_ttl_secs` (default `Some`) was not — telling a
/// user their API rate limit does nothing is worse than saying nothing.
///
/// Round-tripping inverts the test into exactly the question worth asking: a
/// key the daemon understood is populated by deserialization and therefore
/// survives re-serialization; one it ignored disappears. New `Option` fields
/// inherit the correct behaviour with no per-field maintenance.
pub(crate) fn unknown_config_keys(contents: &str) -> Vec<String> {
    let Ok(file) = toml::from_str::<toml::Value>(contents) else {
        return Vec::new(); // a parse error is reported by the caller with a better message
    };
    let Ok(parsed) = toml::from_str::<Config>(contents) else {
        return Vec::new();
    };
    let Ok(schema) = toml::Value::try_from(parsed) else {
        return Vec::new();
    };
    let mut unknown = Vec::new();
    collect_unknown_config_keys(&file, &schema, "", &mut unknown);
    unknown
}

/// Log every unrecognised key in `contents`.
pub(crate) fn warn_unknown_keys_in(contents: &str) {
    for key in unknown_config_keys(contents) {
        tracing::warn!(
            %key,
            "Unknown setting in config file — it is being IGNORED. Check the \
             spelling and which section it belongs under"
        );
    }
}

/// Provider config types (ProvidersConfig, ProviderEntry, CustomProvider,
/// ProviderKeySource) and the .env loader. Their Debug/Drop impls handle
/// API-key redaction and zeroization.
mod providers;
pub use providers::*;

/// Credit-economy + device-pool config: CreditRateConfig, PoolConfig.
mod credit;
pub use credit::*;

/// Network/transport config: NetworkConfig and is_wsl2 helper.
pub(crate) mod network;
pub use network::*;

/// Operational config: LoggingConfig, UiConfig, UpdateConfig +
/// AutoUpdateMode, ApiConfig.
mod ops;
pub use ops::*;

/// Node + resource + identity config: NodeConfig, ResourceConfig,
/// ResourceSchedule, IdentityConfig + resolve_data_dir helper.
mod node;
pub use node::*;

/// Inference / auto-shard-management / model-storage config:
/// InferenceConfig, AutoManageConfig + ModelAutoManagePolicy,
/// ModelConfig + shard-size constants.
mod inference;
pub use inference::*;

// ---- Defaults ----

/// Parse the `SWARMLLM_NETWORK_BOOTSTRAP_PEERS` env var value into a
/// list of multiaddrs. Splits on commas and any whitespace (newline,
/// tab, space) so users can paste either a comma list or a multi-line
/// `.env` value; empty entries are dropped.
pub(crate) fn parse_bootstrap_peers_env(value: &str) -> Vec<String> {
    value
        .split(|c: char| c == ',' || c.is_whitespace())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Read the HuggingFace API token from standard env vars (`HF_TOKEN` preferred,
/// `HUGGING_FACE_HUB_TOKEN` fallback). Returns `None` if unset or empty.
/// Lives here so all external-service credentials flow through `config`.
pub fn hf_api_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .filter(|t| !t.is_empty())
}

pub(super) fn default_true() -> bool {
    true
}

// ---- Impl defaults ----

impl Default for Config {
    fn default() -> Self {
        toml::from_str("").expect("empty TOML should parse to defaults")
    }
}

impl Config {
    /// Force every inference/model knob off for a bootstrap/relay anchor node.
    ///
    /// Called when `--anchor` is passed (or `[node] anchor_mode = true`) so the
    /// single flag is self-sufficient: the daemon won't load models, poll
    /// HuggingFace, acquire shards, auto-manage, or pop a browser — and the API
    /// binds to loopback (read from `node.anchor_mode` in the server). The P2P
    /// network stack (relay, AutoNAT, DCUtR, UPnP, DHT, gossip) is untouched.
    pub fn apply_anchor_mode(&mut self) {
        self.node.anchor_mode = true;
        self.auto_manage.enabled = false;
        self.auto_manage.hf_watcher_enabled = false;
        self.node.contribution_auto = false;
        self.ui.open_browser_on_start = false;
        // An anchor never loads a model, so no worker should ever spawn and
        // this is belt-and-braces. It is here because R146 flipped the
        // `gpu_layers` default from `0` to `-1` (auto) — which silently moved
        // anchors from "CPU only" to "use the GPU if there is one". Nothing
        // reads it on an anchor today, but the doc comment above promises
        // every inference knob is off, and a promise that depends on an
        // unrelated default staying put is not one worth making.
        self.inference.gpu_layers = 0;
        // Connection capacity is the one resource an anchor exists to spend.
        //
        // `contribution` defaults to `Minimal`, and since 2026-08-04 that
        // resolves to a 150-connection ceiling — a sensible figure for the
        // gaming PC the setting was written for, and completely wrong for a
        // bootstrap/relay node whose whole purpose is being reachable by as
        // much of the swarm as possible. `deploy/anchor/config.toml` sets no
        // contribution, so without this an anchor would silently inherit the
        // consumer cap and start refusing the peers it exists to serve.
        //
        // An explicit `max_peers` still wins: someone running an anchor on a
        // small VPS may well want a lower figure than this.
        if self.network.max_peers.is_none() {
            self.network.max_peers = Some(network::MAX_ESTABLISHED_CONNECTIONS_CEILING);
        }
    }

    /// Load config with priority: CLI overrides > env vars > config file > defaults.
    pub fn load_or_create(
        config_path: Option<&Path>,
        cli_port: Option<u16>,
        cli_data_dir: Option<&Path>,
        cli_model_path: Option<&Path>,
        cli_gpu_layers: Option<i32>,
        cli_bootstrap: Vec<String>,
    ) -> Result<Self, SwarmError> {
        tracing::debug!(
            config_path = ?config_path,
            cli_port = ?cli_port,
            cli_data_dir = ?cli_data_dir,
            "DIAG: config load_or_create starting"
        );

        // 1. Start with defaults
        let mut config = Self::default();

        // 1b. Apply data_dir overrides BEFORE config file lookup, so the config
        // is loaded from the correct data directory when SWARMLLM_NODE_DATA_DIR
        // or --data-dir is set. Without this, config.toml is always loaded from
        // the default data dir (~/.local/share/swarmllm/), ignoring per-node
        // overrides and applying stale settings (e.g. enable_autonat=true).
        if let Some(dir) = cli_data_dir {
            config.node.data_dir = dir.to_path_buf();
        } else if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
            config.node.data_dir = PathBuf::from(val);
        }

        // 2. Load from config file if it exists
        let path = config_path
            .map(PathBuf::from)
            .unwrap_or_else(|| config.node.data_dir.join("config.toml"));

        let mut config_text = String::new();
        if path.exists() {
            let contents = std::fs::read_to_string(&path).map_err(SwarmError::Io)?;
            config_text = contents.clone();
            config = toml::from_str(&contents).map_err(|e| {
                SwarmError::Config(format!("Failed to parse {}: {e}", path.display()))
            })?;
            warn_unknown_keys_in(&contents);
            migrate_superseded_defaults(&mut config, "config.toml");
            // Re-apply data_dir overrides since toml::from_str replaces the entire config
            if let Some(dir) = cli_data_dir {
                config.node.data_dir = dir.to_path_buf();
            } else if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
                config.node.data_dir = PathBuf::from(val);
            }
            tracing::info!(path = %path.display(), "Loaded config");
        }

        // 3. Apply environment variable overrides (SWARMLLM_ prefix)
        if let Ok(val) = std::env::var("SWARMLLM_NODE_LISTEN_PORT") {
            match val.parse() {
                Ok(port) => config.node.listen_port = port,
                Err(e) => tracing::warn!(
                    var = "SWARMLLM_NODE_LISTEN_PORT",
                    value = %val,
                    error = %e,
                    "Failed to parse env var, ignoring"
                ),
            }
        }
        if let Ok(val) = std::env::var("SWARMLLM_NODE_DATA_DIR") {
            config.node.data_dir = PathBuf::from(val);
        }
        if let Ok(val) = std::env::var("SWARMLLM_LOGGING_LEVEL") {
            match val.as_str() {
                "trace" | "debug" | "info" | "warn" | "error" => config.logging.level = val,
                _ => tracing::warn!(
                    value = %val,
                    "Ignoring invalid SWARMLLM_LOGGING_LEVEL (expected: trace/debug/info/warn/error)"
                ),
            }
        }
        if let Ok(val) = std::env::var("SWARMLLM_INFERENCE_MODEL_PATH") {
            config.inference.model_path = Some(PathBuf::from(val));
        }
        if let Ok(val) = std::env::var("SWARMLLM_INFERENCE_GPU_LAYERS") {
            match val.parse() {
                Ok(n) => config.inference.gpu_layers = n,
                Err(e) => tracing::warn!(
                    var = "SWARMLLM_INFERENCE_GPU_LAYERS",
                    value = %val,
                    error = %e,
                    "Failed to parse env var, ignoring"
                ),
            }
        }
        // Bearer token for the HTTP API. Advertised in `.env.example` for
        // Docker deployments where users need a deterministic key (vs. the
        // auto-generated 32-byte one) without baking it into config.toml.
        // Empty string is treated as "unset" so commenting `SWARMLLM_API_KEY=`
        // doesn't blank the persisted key.
        if let Ok(val) = std::env::var("SWARMLLM_API_KEY") {
            if !val.is_empty() {
                config.api.api_key = Some(val);
            }
        }
        // Bootstrap peers as a comma-, space-, or newline-separated list of
        // multiaddrs. Mirrors `--bootstrap` on the CLI for headless
        // deployments. Empty entries are filtered.
        if let Ok(val) = std::env::var("SWARMLLM_NETWORK_BOOTSTRAP_PEERS") {
            let peers = parse_bootstrap_peers_env(&val);
            if !peers.is_empty() {
                config.network.bootstrap_peers = peers;
            }
        }

        // 4. Apply CLI overrides (highest priority)
        if let Some(port) = cli_port {
            config.node.listen_port = port;
        }
        if let Some(dir) = cli_data_dir {
            config.node.data_dir = dir.to_path_buf();
        }
        if let Some(path) = cli_model_path {
            config.inference.model_path = Some(path.to_path_buf());
        }
        if let Some(n) = cli_gpu_layers {
            config.inference.gpu_layers = n;
        }
        if !cli_bootstrap.is_empty() {
            config.network.bootstrap_peers = cli_bootstrap;
        }

        // An empty bootstrap list means "not configured", not "no peers".
        //
        // Every config written before 2026-07-21 contains `bootstrap_peers = []`
        // because the daemon serialised it there when an empty list WAS the
        // default. Those users did not opt out of anything — and once the
        // built-in anchor landed, an explicit `[]` silently outranked it
        // (a serde default only fills a MISSING key), so the node started with
        // no bootstrap peers, no way to reach the DHT, and an empty peer list
        // with nothing saying why. Reads as "the update broke my networking".
        //
        // Runs after the env/CLI overrides so an explicitly supplied list still
        // wins; both of those already ignore an empty value, so reaching here
        // with an empty list means nothing anywhere asked for one.
        let opted_out = config.network.disable_default_bootstrap || config.node.anchor_mode;
        if config.network.bootstrap_peers.is_empty() && !opted_out {
            config.network.bootstrap_peers = network::default_bootstrap_peers();
            tracing::info!(
                count = config.network.bootstrap_peers.len(),
                "No bootstrap peers configured — using the built-in anchors. \
                 Set network.disable_default_bootstrap = true to run without any."
            );
        }

        // 5. Auto-detect WSL2 and apply safe network defaults — but ONLY in the
        // default NAT networking mode. In mirrored mode the VM shares the host's
        // interfaces and is a first-class LAN citizen (real IP, working QUIC /
        // mDNS / UPnP / AutoNAT / DCUtR), so the NAT-mode safe defaults would
        // strand it on the relay. Only overrides values the user didn't
        // explicitly set in config.toml.
        if network::is_wsl2() && network::wsl_networking_is_mirrored() {
            tracing::info!(
                "WSL2 mirrored networking detected — node is a first-class LAN citizen; \
                 keeping full networking (QUIC/mDNS/UPnP/AutoNAT/DCUtR), NAT-mode safe \
                 defaults NOT applied"
            );
            // ...but nothing will be able to CONNECT to it until the Windows
            // firewall allows the ports, and Windows never asks.
            //
            // Running the Windows build natively triggers the usual "allow this
            // app through the firewall?" prompt, so those users are covered by
            // the OS. A Linux binary under WSL gets no prompt at all: the
            // firewall is on by default, silently drops inbound, and the node
            // looks perfectly healthy from the inside — it holds a real LAN
            // address, advertises it correctly, and dials out fine. Only the
            // other machine sees the problem, as sends that never complete.
            //
            // Measured 2026-08-04 on exactly this setup: a peer 2ms away on the
            // same subnet could not open TCP 8810 or UDP 8800, one direction of
            // every request depended on the connection this node had dialled
            // out, and cross-machine requests died on the segment timeout after
            // 284 seconds. Opening the two ports fixed it outright.
            //
            // A warning is the right level: it is not an error here (outbound
            // works, and a node that only makes requests is fine), but it is
            // never what someone running a peer-to-peer node intends.
            // The warning itself is NOT emitted here. Whether the firewall is
            // actually blocking anything is not knowable at config-load time,
            // and emitting it unconditionally meant every mirrored-mode node saw
            // it on every start — including ones where the ports had already been
            // opened, which is telling someone to fix a problem they have fixed.
            // `health::monitor` raises it only once inbound has demonstrably
            // failed to arrive; see `maybe_warn_wsl_firewall`.
        } else if network::is_wsl2() {
            // Parse TOML into a Value to check which keys were explicitly set.
            // Raw string search (e.g., config_text.contains("enable_quic")) would
            // match commented-out keys or keys in string values — false positives.
            let explicit_network_keys: std::collections::HashSet<String> =
                toml::from_str::<toml::Value>(&config_text)
                    .ok()
                    .and_then(|v| v.get("network").and_then(|n| n.as_table().cloned()))
                    .map(|t| t.keys().cloned().collect())
                    .unwrap_or_default();
            let has = |key: &str| explicit_network_keys.contains(key);
            let net = &mut config.network;
            let mut adapted = Vec::new();
            if !has("enable_quic") {
                net.enable_quic = false;
                adapted.push("enable_quic=false");
            }
            if !has("enable_autonat") {
                net.enable_autonat = false;
                adapted.push("enable_autonat=false");
            }
            if !has("enable_dcutr") {
                net.enable_dcutr = false;
                adapted.push("enable_dcutr=false");
            }
            if !has("enable_upnp") {
                net.enable_upnp = false;
                adapted.push("enable_upnp=false");
            }
            if !has("enable_mdns") {
                net.enable_mdns = false;
                adapted.push("enable_mdns=false");
            }
            if !has("listen_address") {
                net.listen_address = "127.0.0.1".to_string();
                adapted.push("listen_address=127.0.0.1");
            }
            if !adapted.is_empty() {
                tracing::info!(
                    settings = adapted.join(", "),
                    "WSL2 detected: auto-applied safe network defaults (set explicitly in config.toml to override)"
                );
            }
        }

        // Validate
        config.validate()?;

        tracing::debug!(
            port = config.node.listen_port,
            data_dir = %config.node.data_dir.display(),
            "DIAG: config load_or_create complete"
        );

        Ok(config)
    }

    fn validate(&self) -> Result<(), SwarmError> {
        // Reject port 0 (DAE-M13)
        if self.node.listen_port == 0 {
            return Err(SwarmError::Config("listen_port must not be 0".to_string()));
        }

        // Warn on privileged ports (not fatal — user may have permissions)
        if self.node.listen_port < 1024 {
            tracing::warn!(
                port = self.node.listen_port,
                "Using privileged port — may require elevated permissions"
            );
        }

        // Validate inference config (DAE-I4)
        if self.inference.max_concurrent_requests < 1 {
            return Err(SwarmError::Config(
                "max_concurrent_requests must be >= 1".to_string(),
            ));
        }
        if self.inference.session_timeout_seconds == 0 {
            return Err(SwarmError::Config(
                "session_timeout_seconds must be > 0".to_string(),
            ));
        }
        if self.inference.max_batch_size == 0 {
            return Err(SwarmError::Config(
                "max_batch_size must be >= 1".to_string(),
            ));
        }
        if self.inference.speculative_gamma == 0 {
            return Err(SwarmError::Config(
                "speculative_gamma must be > 0".to_string(),
            ));
        }
        if !(0.0..1.0).contains(&self.inference.swift_skip_ratio) {
            return Err(SwarmError::Config(format!(
                "swift_skip_ratio must be in [0.0, 1.0) (got {}); a value >= 1.0 would cause empty layer ranges in the SWIFT draft pass",
                self.inference.swift_skip_ratio
            )));
        }
        if !(0.0..=1.0).contains(&self.inference.cross_node_prefix_trust_min) {
            return Err(SwarmError::Config(format!(
                "cross_node_prefix_trust_min must be in [0.0, 1.0] (got {}); values outside this range either trust everyone or no one",
                self.inference.cross_node_prefix_trust_min
            )));
        }
        if self.network.relay_max_circuits == 0 {
            return Err(SwarmError::Config(
                "relay_max_circuits must be > 0".to_string(),
            ));
        }
        if self.resources.schedule.reduced_hours_start > 23 {
            return Err(SwarmError::Config(
                "reduced_hours_start must be 0-23".to_string(),
            ));
        }
        if self.resources.schedule.reduced_hours_end > 23 {
            return Err(SwarmError::Config(
                "reduced_hours_end must be 0-23".to_string(),
            ));
        }

        // Validate model path exists if specified
        if let Some(ref path) = self.inference.model_path {
            if !path.exists() {
                return Err(SwarmError::Config(format!(
                    "Model file not found: {}",
                    path.display()
                )));
            }
        }

        // Validate model/shard config
        self.model.validate()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let config = Config::default();
        assert_eq!(config.node.listen_port, 8800);
        assert_eq!(config.resources.max_disk_mb, 50_000);
        assert_eq!(config.inference.session_timeout_seconds, 600);
    }

    #[test]
    fn anchor_mode_defaults_off() {
        assert!(!Config::default().node.anchor_mode);
        assert!(!NodeConfig::default().anchor_mode);
    }

    #[test]
    fn apply_anchor_mode_forces_inference_knobs_off() {
        let mut config = Config::default();
        // Simulate a node that would otherwise run inference + auto-manage.
        config.auto_manage.enabled = true;
        config.auto_manage.hf_watcher_enabled = true;
        config.node.contribution_auto = true;
        config.ui.open_browser_on_start = true;
        config.inference.gpu_layers = -1;

        config.apply_anchor_mode();

        assert!(config.node.anchor_mode, "anchor_mode must be set");
        assert!(!config.auto_manage.enabled, "auto-manage must be off");
        assert!(
            !config.auto_manage.hf_watcher_enabled,
            "HF watcher must be off"
        );
        assert!(
            !config.node.contribution_auto,
            "contribution_auto must be off"
        );
        assert_eq!(
            config.inference.gpu_layers, 0,
            "anchor must not claim a GPU — the default is -1 (auto) since R146"
        );
        assert!(
            !config.ui.open_browser_on_start,
            "browser open must be off on a headless anchor"
        );
        // The P2P stack is untouched — an anchor is still a full network peer.
        assert!(
            config.network.enable_relay,
            "relay must stay on for an anchor"
        );
        assert!(config.network.enable_autonat, "autonat must stay on");
    }

    #[test]
    fn parse_toml_with_overrides() {
        let toml_str = r#"
[node]
listen_port = 9000

[inference]
max_concurrent_requests = 5
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.node.listen_port, 9000);
        assert_eq!(config.inference.max_concurrent_requests, 5);
        // Defaults still apply for unset fields
        assert_eq!(config.resources.max_disk_mb, 50_000);
    }

    #[test]
    fn load_from_nonexistent_path_uses_defaults() {
        let config = Config::load_or_create(
            Some(Path::new("/nonexistent/config.toml")),
            None,
            None,
            None,
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(config.node.listen_port, 8800);
    }

    #[test]
    fn cli_overrides_take_priority() {
        let config = Config::load_or_create(
            Some(Path::new("/nonexistent")),
            Some(9999),
            None,
            None,
            None,
            vec![],
        )
        .unwrap();
        assert_eq!(config.node.listen_port, 9999);
    }

    #[test]
    fn config_parses_with_missing_sections() {
        // A minimal config (only [node]) must deserialize and rely on
        // serde defaults for every other section. This is the back-
        // compat invariant: old config.toml files written before new
        // sections existed must still load.
        let toml_str = r#"
[node]
listen_port = 8800
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.node.listen_port, 8800);
        // Sections that didn't exist in the file should fall back to
        // their Default impls, not panic.
        assert_eq!(config.resources.max_disk_mb, 50_000);
        assert_eq!(config.inference.max_concurrent_requests, 10);
    }

    #[test]
    fn config_ignores_unknown_top_level_field() {
        // `serde(default)` + non-`deny_unknown_fields` means a future
        // version of swarmllm that adds new sections is forward-compat
        // when read by an older binary. Verify behavior.
        let toml_str = r#"
[node]
listen_port = 8800

[future_section_we_dont_know_about]
some_setting = "value"
"#;
        // toml::from_str must not error on the unknown section. If a
        // future change adds #[serde(deny_unknown_fields)] to Config,
        // this test will fire and the author must update the contract
        // (or document the breakage).
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.node.listen_port, 8800);
    }

    #[test]
    fn config_old_field_names_still_parse() {
        // Sanity: a minimal file with only [api] api_key set parses
        // and the old field name is preserved through round-trip.
        // Catches accidental rename/removal of public config fields.
        let toml_str = r#"
[api]
api_key = "test-key-abc"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.api.api_key.as_deref(), Some("test-key-abc"));
    }

    /// The legacy-config case this normalisation exists for. Every config the
    /// daemon wrote before 2026-07-21 carries `bootstrap_peers = []`, which an
    /// explicit-value-beats-serde-default rule made outrank the built-in
    /// anchors — stranding the node with no way to find the swarm.
    #[test]
    fn empty_bootstrap_list_falls_back_to_the_built_in_anchors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "[network]\nbootstrap_peers = []\n").unwrap();

        let config = Config::load_or_create(Some(&path), None, None, None, None, vec![]).unwrap();
        assert!(
            !config.network.bootstrap_peers.is_empty(),
            "an empty list must be treated as unconfigured, not as opting out"
        );
        assert!(config
            .network
            .bootstrap_peers
            .iter()
            .any(|p| p.contains("swarmllm.duckdns.org")));
    }

    /// ...but saying so explicitly must still work, or a private/air-gapped
    /// swarm silently starts dialling the public anchors.
    #[test]
    fn disable_default_bootstrap_really_means_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[network]\nbootstrap_peers = []\ndisable_default_bootstrap = true\n",
        )
        .unwrap();

        let config = Config::load_or_create(Some(&path), None, None, None, None, vec![]).unwrap();
        assert!(config.network.bootstrap_peers.is_empty());
    }

    /// An anchor IS the bootstrap; making it dial itself is the one case the
    /// shipped `deploy/anchor/config.toml` was guarding against with `[]`.
    #[test]
    fn anchor_mode_does_not_get_the_default_anchors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[node]\nanchor_mode = true\n\n[network]\nbootstrap_peers = []\n",
        )
        .unwrap();

        let config = Config::load_or_create(Some(&path), None, None, None, None, vec![]).unwrap();
        assert!(config.network.bootstrap_peers.is_empty());
    }

    /// An explicitly configured list must survive untouched — the normalisation
    /// runs after the env/CLI overrides precisely so it cannot clobber one.
    #[test]
    fn an_explicit_bootstrap_list_is_left_alone() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "[network]\nbootstrap_peers = [\"/ip4/10.0.0.1/tcp/8810\"]\n",
        )
        .unwrap();

        let config = Config::load_or_create(Some(&path), None, None, None, None, vec![]).unwrap();
        assert_eq!(
            config.network.bootstrap_peers,
            vec!["/ip4/10.0.0.1/tcp/8810"]
        );

        // And a CLI list still wins over a file that had none.
        let path2 = dir.path().join("empty.toml");
        std::fs::write(&path2, "[network]\nbootstrap_peers = []\n").unwrap();
        let cli = Config::load_or_create(
            Some(&path2),
            None,
            None,
            None,
            None,
            vec!["/ip4/10.0.0.2/tcp/8810".to_string()],
        )
        .unwrap();
        assert_eq!(cli.network.bootstrap_peers, vec!["/ip4/10.0.0.2/tcp/8810"]);
    }

    #[test]
    fn bootstrap_peers_env_parses_commas_and_whitespace() {
        // Comma-separated.
        let parsed = parse_bootstrap_peers_env(
            "/ip4/1.1.1.1/udp/8800/quic-v1,/ip4/2.2.2.2/udp/8800/quic-v1",
        );
        assert_eq!(
            parsed,
            vec![
                "/ip4/1.1.1.1/udp/8800/quic-v1".to_string(),
                "/ip4/2.2.2.2/udp/8800/quic-v1".to_string(),
            ]
        );

        // Whitespace and newlines (multi-line .env value).
        let parsed = parse_bootstrap_peers_env(
            "/ip4/1.1.1.1/udp/8800/quic-v1\n/ip4/2.2.2.2/udp/8800/quic-v1",
        );
        assert_eq!(parsed.len(), 2);

        // Mixed separators with empty entries.
        let parsed = parse_bootstrap_peers_env(", ,/ip4/1.1.1.1/udp/8800/quic-v1, , ,");
        assert_eq!(parsed, vec!["/ip4/1.1.1.1/udp/8800/quic-v1".to_string()]);

        // Empty string yields no peers.
        assert!(parse_bootstrap_peers_env("").is_empty());
        assert!(parse_bootstrap_peers_env("   \n  ").is_empty());
    }

    #[test]
    fn default_shard_size_mb() {
        let config = Config::default();
        assert_eq!(config.model.shard_size_mb, 512);
        assert_eq!(config.model.shard_size_bytes(), 512 * 1024 * 1024);
    }

    #[test]
    fn parse_custom_shard_size() {
        let toml_str = r#"
[model]
shard_size_mb = 256
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.model.shard_size_mb, 256);
        assert_eq!(config.model.shard_size_bytes(), 256 * 1024 * 1024);
    }

    #[test]
    fn shard_size_validation_rejects_too_small() {
        let model_config = ModelConfig { shard_size_mb: 32 };
        assert!(model_config.validate().is_err());
    }

    #[test]
    fn shard_size_validation_rejects_too_large() {
        let model_config = ModelConfig {
            shard_size_mb: 4096,
        };
        assert!(model_config.validate().is_err());
    }

    #[test]
    fn shard_size_validation_accepts_valid() {
        let model_config = ModelConfig { shard_size_mb: 512 };
        assert!(model_config.validate().is_ok());
        let model_config = ModelConfig { shard_size_mb: 64 };
        assert!(model_config.validate().is_ok());
        let model_config = ModelConfig {
            shard_size_mb: 2048,
        };
        assert!(model_config.validate().is_ok());
    }

    #[test]
    fn shard_size_non_power_of_two_warns_but_passes() {
        let model_config = ModelConfig { shard_size_mb: 300 };
        // Should succeed (only a warning, not an error)
        assert!(model_config.validate().is_ok());
    }

    #[test]
    fn identity_region_defaults_to_none() {
        let config = Config::default();
        assert!(config.identity.region.is_none());
    }

    #[test]
    fn parse_identity_region() {
        let toml_str = r#"
[identity]
region = "US"
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.identity.region.as_deref(), Some("US"));
    }

    #[test]
    fn auto_relay_defaults_to_true() {
        let config = Config::default();
        assert!(config.network.auto_relay);
    }

    #[test]
    fn parse_auto_relay_disabled() {
        let toml_str = r#"
[network]
auto_relay = false
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(!config.network.auto_relay);
    }

    #[test]
    fn max_concurrent_downloads_defaults_to_3() {
        let config = Config::default();
        assert_eq!(config.auto_manage.max_concurrent_downloads, 3);
    }

    /// An anchor's whole job is being reachable, so it must not inherit the
    /// consumer connection cap. `deploy/anchor/config.toml` sets no
    /// contribution, so without the override in `apply_anchor_mode` an anchor
    /// resolves to Minimal and starts refusing the peers it exists to serve.
    #[test]
    fn anchor_mode_keeps_full_connection_capacity() {
        let mut config = Config::default();
        assert!(
            config
                .network
                .effective_max_connections(config.node.contribution.clone())
                < network::MAX_ESTABLISHED_CONNECTIONS_CEILING,
            "precondition: a default node is capped below the ceiling"
        );

        config.apply_anchor_mode();
        assert_eq!(
            config
                .network
                .effective_max_connections(config.node.contribution.clone()),
            network::MAX_ESTABLISHED_CONNECTIONS_CEILING
        );
    }

    /// Someone running an anchor on a small VPS may want fewer connections than
    /// the ceiling; anchor mode must not stamp over that.
    #[test]
    fn anchor_mode_respects_an_explicit_max_peers() {
        let mut config = Config::default();
        config.network.max_peers = Some(64);
        config.apply_anchor_mode();
        assert_eq!(config.network.max_peers, Some(64));
    }

    /// `max_peers = 200` on disk is the old daemon-written default for a key
    /// that was never enforced, so it cannot be a deliberate choice. Left in
    /// place it would override the contribution-derived ceiling forever.
    #[test]
    fn stranded_max_peers_default_is_migrated_away() {
        let mut config = Config::default();
        config.network.max_peers = Some(200);
        migrate_superseded_defaults(&mut config, "test");
        assert_eq!(config.network.max_peers, None);

        // A value that is not the old default is a real choice and stays.
        let mut chosen = Config::default();
        chosen.network.max_peers = Some(64);
        migrate_superseded_defaults(&mut chosen, "test");
        assert_eq!(chosen.network.max_peers, Some(64));
    }

    #[test]
    fn operational_params_from_config() {
        let config = Config::default();
        let params = OperationalParams::from_config(&config);
        assert_eq!(params.max_concurrent_requests, 10);
        assert_eq!(params.auto_manage_interval_minutes, 5);
        assert_eq!(params.max_batch_size, 1);
        assert_eq!(params.batch_timeout_ms, 50);
        assert_eq!(params.session_timeout_secs, 600);
    }

    /// This struct is for settings a subsystem has to be TOLD about, because
    /// each sizes something built once. Anything a caller can simply re-read
    /// belongs on the live config instead. Five fields once sat here with no
    /// consumer at all while the struct advertised itself as hot-reloadable;
    /// this pins the list so that cannot quietly come back.
    #[test]
    fn operational_params_carries_only_settings_with_a_consumer() {
        let json = serde_json::to_value(OperationalParams::from_config(&Config::default()))
            .expect("serializable");
        let mut fields: Vec<&str> = json
            .as_object()
            .expect("object")
            .keys()
            .map(|k| k.as_str())
            .collect();
        fields.sort_unstable();
        assert_eq!(
            fields,
            vec![
                "auto_manage_interval_minutes",
                "batch_timeout_ms",
                "max_batch_size",
                "max_concurrent_requests",
                "session_timeout_secs",
            ],
            "every field here must be consumed by a subsystem's reload arm — \
             if a new one is only ever read, put it on the live config instead"
        );
    }

    #[test]
    fn operational_params_equality() {
        let config = Config::default();
        let p1 = OperationalParams::from_config(&config);
        let p2 = OperationalParams::from_config(&config);
        assert_eq!(p1, p2);
    }

    #[test]
    fn reload_operational_params_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        let toml_str = r#"
[inference]
max_concurrent_requests = 20
max_batch_size = 4
session_timeout_seconds = 300

[auto_manage]
interval_minutes = 10

[network]
max_peers = 100
"#;
        std::fs::write(&config_path, toml_str).unwrap();
        let params = reload_operational_params(&config_path).unwrap();
        assert_eq!(params.max_concurrent_requests, 20);
        assert_eq!(params.max_batch_size, 4);
        assert_eq!(params.session_timeout_secs, 300);
        assert_eq!(params.auto_manage_interval_minutes, 10);
    }

    #[test]
    fn reload_operational_params_missing_file() {
        let result = reload_operational_params(Path::new("/nonexistent/config.toml"));
        assert!(result.is_err());
    }

    #[test]
    fn reload_operational_params_partial_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("config.toml");
        // Only override one field — rest should use defaults
        let toml_str = r#"
[inference]
max_concurrent_requests = 42
"#;
        std::fs::write(&config_path, toml_str).unwrap();
        let params = reload_operational_params(&config_path).unwrap();
        assert_eq!(params.max_concurrent_requests, 42);
        // Defaults for others
        assert_eq!(params.max_batch_size, 1);
    }

    /// An explicit ceiling is the user's own decision and overrides the
    /// contribution-derived fraction entirely, in either direction.
    #[test]
    fn vram_budget_explicit_cap_wins_over_contribution() {
        let rc = ResourceConfig {
            max_gpu_vram_mb: 4000,
            ..Default::default()
        };
        for mode in [
            swarmllm_types::ContributionMode::Minimal,
            swarmllm_types::ContributionMode::Moderate,
            swarmllm_types::ContributionMode::Maximum,
        ] {
            assert_eq!(rc.inference_vram_budget_mb(8000, mode), Some(4000));
        }
    }

    /// **The budget follows what the user agreed to contribute.**
    ///
    /// It was a flat 80% whatever they had chosen — and `Minimal` is the
    /// DEFAULT, so a stock install claimed 6.5 GB of an 8 GB card on machines
    /// that are mostly gaming PCs and home desktops.
    #[test]
    fn vram_budget_scales_with_contribution() {
        let rc = ResourceConfig::default(); // max_gpu_vram_mb = 0
        let minimal = rc
            .inference_vram_budget_mb(8000, swarmllm_types::ContributionMode::Minimal)
            .unwrap();
        let moderate = rc
            .inference_vram_budget_mb(8000, swarmllm_types::ContributionMode::Moderate)
            .unwrap();
        let maximum = rc
            .inference_vram_budget_mb(8000, swarmllm_types::ContributionMode::Maximum)
            .unwrap();

        assert!(
            minimal < moderate && moderate < maximum,
            "more contribution must mean more headroom, got {minimal}/{moderate}/{maximum}"
        );
        assert_eq!(maximum, 6400, "an explicit offer of the machine keeps 80%");
        assert!(
            minimal <= 4000,
            "the DEFAULT setting must leave most of the card to the person using              the computer, got {minimal} of 8000"
        );
    }

    /// RAM must follow the contribution level too. A limit that governs one
    /// kind of memory and not the other is a suggestion, not a limit — and RAM
    /// exhaustion is the worse failure, because it swaps and degrades the whole
    /// machine rather than just this daemon.
    #[test]
    fn ram_budget_scales_with_contribution() {
        let rc = ResourceConfig::default(); // max_ram_mb = 0
        for has_gpu in [true, false] {
            let min = rc
                .inference_ram_budget_mb(16000, has_gpu, swarmllm_types::ContributionMode::Minimal)
                .unwrap();
            let max = rc
                .inference_ram_budget_mb(16000, has_gpu, swarmllm_types::ContributionMode::Maximum)
                .unwrap();
            assert!(
                min < max,
                "has_gpu={has_gpu}: minimal ({min}) must claim less RAM than maximum ({max})"
            );
        }
        // The documented shapes still hold for a node that has offered itself.
        assert_eq!(
            rc.inference_ram_budget_mb(16000, true, swarmllm_types::ContributionMode::Maximum),
            Some(8000),
            "GPU node at maximum keeps the documented 50%"
        );
        assert_eq!(
            rc.inference_ram_budget_mb(16000, false, swarmllm_types::ContributionMode::Maximum),
            Some(12800),
            "CPU-only at maximum keeps the documented 80%"
        );
    }

    /// An explicit RAM ceiling is the user's own decision and still wins.
    #[test]
    fn ram_budget_explicit_cap_wins_over_contribution() {
        let rc = ResourceConfig {
            max_ram_mb: 3000,
            ..Default::default()
        };
        assert_eq!(
            rc.inference_ram_budget_mb(16000, true, swarmllm_types::ContributionMode::Minimal),
            Some(3000)
        );
    }

    /// Seeding shards to peers is pure contribution, so an unset cap must not
    /// mean "use the whole connection". It did — `0` was unlimited at every
    /// level, so a stock install could saturate a home uplink.
    #[test]
    fn shard_upload_is_capped_unless_the_machine_was_offered() {
        let rc = ResourceConfig::default(); // max_bandwidth_mbps = 0
        let min = rc.shard_upload_mbps(swarmllm_types::ContributionMode::Minimal);
        let mod_ = rc.shard_upload_mbps(swarmllm_types::ContributionMode::Moderate);
        let max = rc.shard_upload_mbps(swarmllm_types::ContributionMode::Maximum);

        assert!(min > 0, "the DEFAULT must not be unlimited, got {min}");
        assert!(
            mod_ > min,
            "more contribution must allow more, got {min}/{mod_}"
        );
        assert_eq!(max, 0, "offering the machine keeps unlimited");
    }

    /// CPU inference had no thread limit at all — candle's rayon pool defaults
    /// to every logical core. Measured on a 6-core node set to Minimal: a single
    /// request held 529-534% of 600%, with ~10% of the machine idle.
    #[test]
    fn cpu_threads_leave_the_machine_usable_at_the_default_level() {
        let rc = ResourceConfig::default(); // max_cpu_threads = 0
        let (phys, logical) = (8usize, 16usize); // a typical 2-way SMT desktop
        let min =
            rc.inference_cpu_threads(phys, logical, swarmllm_types::ContributionMode::Minimal);
        let mod_ =
            rc.inference_cpu_threads(phys, logical, swarmllm_types::ContributionMode::Moderate);
        let max =
            rc.inference_cpu_threads(phys, logical, swarmllm_types::ContributionMode::Maximum);

        assert!(
            min < phys,
            "the DEFAULT level must leave cores free, got {min} of {phys}"
        );
        assert!(
            min < mod_ && mod_ < max,
            "more contribution must allow more, got {min}/{mod_}/{max}"
        );

        // Never 0: rayon reads 0 as "pick the default", i.e. every logical core
        // — the exact behaviour being fixed. Single-core machines still get 1.
        for c in [
            swarmllm_types::ContributionMode::Minimal,
            swarmllm_types::ContributionMode::Moderate,
            swarmllm_types::ContributionMode::Maximum,
        ] {
            assert_eq!(rc.inference_cpu_threads(1, 2, c.clone()), 1);
            assert!(rc.inference_cpu_threads(0, 0, c) >= 1);
        }
    }

    /// **No contribution level may exceed the physical core count.**
    ///
    /// Swept on a Ryzen 7 5800H (8 physical / 16 logical), phi-3.5 Q4_K_M:
    /// 4 threads 2.26 tok/s, 6 -> 2.36, 8 -> 2.18, 12 -> 1.75, 16 -> 1.49.
    /// Throughput is flat to about the physical count and then falls off a
    /// cliff, because quantised inference is bound by memory bandwidth and two
    /// threads sharing one physical core contend rather than add.
    ///
    /// The first version scaled a fraction of LOGICAL cores, which made
    /// `Maximum` the slowest setting on the machine — someone offering their
    /// whole computer got 37% less throughput than the default. Offering more
    /// must never cost performance.
    #[test]
    fn no_contribution_level_oversubscribes_physical_cores() {
        let rc = ResourceConfig::default();
        for (phys, logical) in [(8usize, 16usize), (4, 8), (6, 6), (2, 4), (1, 2)] {
            for c in [
                swarmllm_types::ContributionMode::Minimal,
                swarmllm_types::ContributionMode::Moderate,
                swarmllm_types::ContributionMode::Maximum,
            ] {
                let t = rc.inference_cpu_threads(phys, logical, c.clone());
                assert!(
                    t <= phys,
                    "{c:?} asked for {t} threads on {phys} physical cores ({logical} logical) \
                     — past the physical count throughput only falls"
                );
                assert!(t >= 1, "{c:?} must never resolve to zero threads");
            }
        }
    }

    /// Maximum means the whole machine's real compute, which is its physical
    /// cores — not its hyper-threads.
    #[test]
    fn maximum_contribution_means_every_physical_core() {
        let rc = ResourceConfig::default();
        assert_eq!(
            rc.inference_cpu_threads(8, 16, swarmllm_types::ContributionMode::Maximum),
            8
        );
    }

    /// An explicit thread count is the owner's decision and wins in either
    /// direction, but is still clamped to the machine — more threads than cores
    /// only adds contention.
    #[test]
    fn an_explicit_cpu_thread_count_wins_over_contribution() {
        let rc = ResourceConfig {
            max_cpu_threads: 6,
            ..Default::default()
        };
        assert_eq!(
            rc.inference_cpu_threads(8, 16, swarmllm_types::ContributionMode::Minimal),
            6,
            "asking for MORE than the minimal default must be honoured"
        );
        assert_eq!(
            rc.inference_cpu_threads(8, 16, swarmllm_types::ContributionMode::Maximum),
            6,
            "asking for FEWER than the machine must be honoured"
        );
        // Clamped to LOGICAL, not physical: oversubscribing hyper-threads is
        // usually slower here, but it is a legitimate deliberate choice and
        // quietly overriding it would be worse than honouring it.
        let big = ResourceConfig {
            max_cpu_threads: 16,
            ..Default::default()
        };
        assert_eq!(
            big.inference_cpu_threads(8, 16, swarmllm_types::ContributionMode::Minimal),
            16,
            "an explicit request for the logical count is the user's call"
        );
        assert_eq!(
            big.inference_cpu_threads(2, 2, swarmllm_types::ContributionMode::Maximum),
            2,
            "but never more threads than the OS actually has"
        );
    }

    /// An explicit figure is the owner's decision and wins in either direction,
    /// including asking for MORE than the contribution default would give.
    #[test]
    fn an_explicit_bandwidth_cap_wins_over_contribution() {
        let rc = ResourceConfig {
            max_bandwidth_mbps: 500,
            ..Default::default()
        };
        for c in [
            swarmllm_types::ContributionMode::Minimal,
            swarmllm_types::ContributionMode::Maximum,
        ] {
            assert_eq!(rc.shard_upload_mbps(c), 500);
        }
    }

    #[test]
    fn vram_budget_no_gpu() {
        let rc = ResourceConfig::default();
        assert_eq!(
            rc.inference_vram_budget_mb(0, swarmllm_types::ContributionMode::Maximum),
            None
        );
    }

    #[test]
    fn ram_budget_explicit_cap() {
        let rc = ResourceConfig {
            max_ram_mb: 3000,
            ..Default::default()
        };
        // An explicit cap wins regardless of what hardware is present.
        assert_eq!(
            rc.inference_ram_budget_mb(16000, true, swarmllm_types::ContributionMode::Maximum),
            Some(3000)
        );
        assert_eq!(
            rc.inference_ram_budget_mb(16000, false, swarmllm_types::ContributionMode::Maximum),
            Some(3000)
        );
    }

    /// With a GPU, system RAM is support work — half the machine is generous.
    /// This is the figure `config/default.toml` has always documented.
    #[test]
    fn ram_budget_auto_is_half_the_machine_when_a_gpu_is_present() {
        let rc = ResourceConfig::default(); // max_ram_mb = 0
        assert_eq!(
            rc.inference_ram_budget_mb(16000, true, swarmllm_types::ContributionMode::Maximum),
            Some(8000)
        );
    }

    /// On a CPU-only node, serving models IS the machine's job, so half of it
    /// is a capability cut rather than headroom.
    #[test]
    fn ram_budget_auto_is_most_of_the_machine_when_there_is_no_gpu() {
        let rc = ResourceConfig::default();
        assert_eq!(
            rc.inference_ram_budget_mb(16000, false, swarmllm_types::ContributionMode::Maximum),
            Some(12800)
        );
    }

    /// The regression that a flat 50% default would have shipped: an 8 GB
    /// CPU-only node — a primary deployment target — must still admit
    /// `llama-3.2-3b-instruct-q4-k-m`, which estimates ~4575 MB on the CPU
    /// (4639 MB was logged for it on a real node's GPU path, and the two
    /// estimates differ only by the 64 MB process-overhead delta). At 50% the
    /// budget is 4096 MB and the model is refused despite such nodes serving
    /// it today.
    #[test]
    fn an_8gb_cpu_only_node_still_admits_a_3b_model() {
        let rc = ResourceConfig::default();
        const LLAMA_3B_CPU_ESTIMATE_MB: u64 = 4575;

        let cpu_only = rc
            .inference_ram_budget_mb(8192, false, swarmllm_types::ContributionMode::Maximum)
            .unwrap();
        assert!(
            cpu_only >= LLAMA_3B_CPU_ESTIMATE_MB,
            "budget {cpu_only} MB must still fit a 3B model at {LLAMA_3B_CPU_ESTIMATE_MB} MB"
        );

        // The flat-50% behaviour, which is still correct where a GPU does the
        // work, is exactly what would have refused this model on a CPU-only box.
        let with_gpu = rc
            .inference_ram_budget_mb(8192, true, swarmllm_types::ContributionMode::Maximum)
            .unwrap();
        assert!(
            with_gpu < LLAMA_3B_CPU_ESTIMATE_MB,
            "precondition: a flat 50% default ({with_gpu} MB) would have refused it"
        );
    }

    /// A machine we could not read must not have a limit invented for it.
    #[test]
    fn ram_budget_unknown_machine() {
        let rc = ResourceConfig::default();
        assert_eq!(
            rc.inference_ram_budget_mb(0, true, swarmllm_types::ContributionMode::Maximum),
            None
        );
        assert_eq!(
            rc.inference_ram_budget_mb(0, false, swarmllm_types::ContributionMode::Maximum),
            None
        );
    }
}

#[cfg(test)]
mod config_default_hygiene {
    /// An *empty section* must deserialize to the same thing as a *missing*
    /// section. These take different code paths — a missing section uses the
    /// section's `impl Default`, a present-but-empty one uses each field's
    /// `#[serde(default)]` — and they silently disagreed for `updates.mode`,
    /// where the struct default said `Some(Notify)` and the field default said
    /// `None`. That made the effective update mode depend on whether the
    /// `[updates]` header happened to be in the file.
    #[test]
    fn empty_section_matches_missing_section() {
        let defaults = toml::Value::try_from(crate::config::Config::default()).unwrap();
        let sections: Vec<String> = match &defaults {
            toml::Value::Table(t) => t.keys().cloned().collect(),
            _ => panic!("config must serialize to a table"),
        };
        assert!(!sections.is_empty());
        for section in sections {
            let text = format!("[{section}]\n");
            let parsed: crate::config::Config = toml::from_str(&text)
                .unwrap_or_else(|e| panic!("[{section}] alone must parse: {e}"));
            let got = toml::Value::try_from(&parsed).unwrap();
            assert_eq!(
                got.get(&section),
                defaults.get(&section),
                "an empty [{section}] must equal a missing one — a field's \
                 serde default disagrees with the section's impl Default"
            );
        }
    }

    /// A misspelled or misplaced key must be named in the log rather than
    /// silently ignored — reported by a user who set two real settings in the
    /// wrong place, saw "Loaded config", and concluded they were broken.
    #[test]
    fn unknown_keys_are_detected() {
        // A real key in the WRONG section, and an outright typo. Goes through
        // the real entry point: the previous version of this test rebuilt the
        // schema itself, the same way the code did, so it reproduced the
        // defect instead of catching it (see `option_settings_are_not_reported_as_unknown`).
        let unknown = crate::config::unknown_config_keys(
            "[node]\ndisable_default_bootstrap = true\n[network]\nenable_upnpp = false\n",
        );
        assert!(
            unknown.contains(&"node.disable_default_bootstrap".to_string()),
            "a real key in the wrong section is unknown there: {unknown:?}"
        );
        assert!(
            unknown.contains(&"network.enable_upnpp".to_string()),
            "a typo must be caught: {unknown:?}"
        );

        // A correct config must produce no warnings at all.
        let good = toml::to_string_pretty(&crate::config::Config::default()).unwrap();
        let none_expected = crate::config::unknown_config_keys(&good);
        assert!(
            none_expected.is_empty(),
            "a valid config must not warn: {none_expected:?}"
        );
    }

    /// Every `Option` setting that defaults to `None` was reported as "being
    /// IGNORED" while actually taking effect, because TOML cannot represent
    /// null so `Config::default()` serialized without the key. Observed live
    /// 2026-07-29 on `inference.max_seq_len_override`, which was applied (the
    /// loader logged the clamp) and denounced in the same run.
    ///
    /// Asserted per-field rather than on one example: the sibling
    /// `kv_cache_ttl_secs` defaults to `Some` and was never affected, so a
    /// single-case test proves nothing about the class.
    #[test]
    fn option_settings_are_not_reported_as_unknown() {
        for (section, key, value) in [
            ("inference", "max_seq_len_override", "4096"),
            ("api", "rate_limit_rpm", "500"),
            ("api", "rate_limit_admin_rpm", "60"),
            ("logging", "file", "\"/tmp/swarmllm.log\""),
            // `region` is the sole field of IdentityConfig, so `[identity]`
            // serialized to an EMPTY table whenever it was unset — which the
            // walker then treated as a free-form map and skipped entirely.
            // Round-tripping gives the section real content, so it is now
            // checked like any other.
            ("identity", "region", "\"eu-west\""),
            ("network", "gossip_network_id", "\"testnet\""),
            ("inference", "max_split_model_memory_mb", "2048"),
            ("inference", "draft_gpu_layers", "8"),
        ] {
            let text = format!("[{section}]\n{key} = {value}\n");
            let unknown = crate::config::unknown_config_keys(&text);
            assert!(
                unknown.is_empty(),
                "{section}.{key} is a real setting and must not be reported as \
                 unknown, but got: {unknown:?}"
            );
        }
    }

    /// The round-trip schema must not become a blanket amnesty: a typo inside
    /// a section whose only other content is an `Option` field still has to be
    /// caught.
    #[test]
    fn typos_beside_option_settings_are_still_caught() {
        let unknown = crate::config::unknown_config_keys(
            "[inference]\nmax_seq_len_override = 4096\nmax_seq_len_overide = 2048\n",
        );
        assert!(
            unknown.contains(&"inference.max_seq_len_overide".to_string()),
            "the misspelling must still be named: {unknown:?}"
        );
        assert!(
            !unknown.contains(&"inference.max_seq_len_override".to_string()),
            "the correct spelling must not be: {unknown:?}"
        );
    }

    /// A value stranded from a superseded default is reset; a deliberate one is
    /// left alone.
    #[test]
    fn superseded_update_interval_is_migrated() {
        let text = "[updates]\ncheck_interval_hours = 6\n";
        let mut c: crate::config::Config = toml::from_str(text).unwrap();
        assert_eq!(c.updates.check_interval_hours, 6);
        crate::config::migrate_superseded_defaults(&mut c, "test");
        assert_eq!(
            c.updates.check_interval_hours,
            crate::config::UpdateConfig::default().check_interval_hours,
            "a stranded copy of the old default must follow the current one"
        );

        let mut other: crate::config::Config =
            toml::from_str("[updates]\ncheck_interval_hours = 24\n").unwrap();
        crate::config::migrate_superseded_defaults(&mut other, "test");
        assert_eq!(
            other.updates.check_interval_hours, 24,
            "a value that was never a default must be left alone"
        );
    }

    /// Any config, once pruned, must reload to exactly the same effective
    /// values. This is the property that makes pruning safe: what we drop is
    /// only ever what the defaults will put back.
    #[test]
    fn pruned_config_round_trips_unchanged() {
        let mut original = crate::config::Config::default();
        original.node.listen_port = 9123;
        original.updates.check_interval_hours = 12;
        original.network.enable_mdns = !original.network.enable_mdns;
        original.inference.max_concurrent_requests += 7;

        let text = crate::config::to_minimal_toml(&original).expect("serialize");
        let reloaded: crate::config::Config = toml::from_str(&text).expect("reload pruned config");

        assert_eq!(
            toml::Value::try_from(&original).unwrap(),
            toml::Value::try_from(&reloaded).unwrap(),
            "pruned config must reload identically; wrote:\n{text}"
        );
    }

    /// The whole point: a field the user never set must not appear on disk, so
    /// a future change to its default still reaches this install.
    #[test]
    fn untouched_fields_are_not_written() {
        let mut c = crate::config::Config::default();
        c.node.listen_port = 9123;
        let text = crate::config::to_minimal_toml(&c).expect("serialize");

        assert!(
            text.contains("9123"),
            "the changed value must be written: {text}"
        );
        assert!(
            !text.contains("check_interval_hours"),
            "an untouched field must stay absent so its default stays live: {text}"
        );
    }

    /// A default-valued config should write essentially nothing.
    #[test]
    fn default_config_writes_nothing() {
        let text = crate::config::to_minimal_toml(&crate::config::Config::default()).unwrap();
        assert!(
            text.trim().is_empty(),
            "a config identical to defaults should emit no keys, got:\n{text}"
        );
    }

    /// **A config write must not discard provider settings.**
    ///
    /// Reported 2026-08-05: a node's `config.toml` shrank from ~3,992 bytes to
    /// 392 after a settings change, losing its `[providers]` section among
    /// others, and its cloud models went to zero. If the writer dropped
    /// non-default provider values that would be silent data loss, so this
    /// pins it: anything the operator actually set has to survive the round
    /// trip.
    #[test]
    fn a_config_write_preserves_provider_settings() {
        let mut config = super::Config::default();
        config.providers.key_source = crate::config::providers::ProviderKeySource::Dashboard;
        config.providers.custom = vec![crate::config::providers::CustomProvider {
            name: "my-endpoint".into(),
            base_url: "https://example.invalid/v1".into(),
            api_key: String::new(),
            default_model: Some("some-model".into()),
        }];

        let written = super::to_minimal_toml(&config).expect("serialize");
        assert!(
            written.contains("providers"),
            "the providers section vanished from a write:\n{written}"
        );

        let reloaded: super::Config = toml::from_str(&written).expect("reparse");
        assert_eq!(
            reloaded.providers.key_source,
            crate::config::providers::ProviderKeySource::Dashboard
        );
        assert_eq!(reloaded.providers.custom.len(), 1);
        assert_eq!(reloaded.providers.custom[0].name, "my-endpoint");
    }

    #[test]
    fn empty_toml_parses_to_full_default() {
        let parsed: crate::config::Config = toml::from_str("")
            .expect("an empty config must parse — every field needs a serde default");
        let d = crate::config::Config::default();
        assert_eq!(
            toml::Value::try_from(&parsed).unwrap(),
            toml::Value::try_from(&d).unwrap(),
            "empty config must equal Config::default()"
        );
    }
}
