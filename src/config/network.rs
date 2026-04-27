//! Network/transport configuration.
//!
//! Hosts `NetworkConfig` (peer-exchange, relay, mDNS, NAT/DCUtR, gossip
//! ID, encryption, tensor + prefix-KV compression, listen address, QUIC),
//! its `Default` impl, and the network-only default helpers (max_peers,
//! relay capacity, compression level/threshold, listen address). Also
//! exposes `is_wsl2` — used by Config::load_or_create to apply
//! WSL2-safe network overrides.

use super::default_true;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkConfig {
    #[serde(default)]
    pub bootstrap_peers: Vec<String>,
    #[serde(default = "default_true")]
    pub peer_exchange: bool,
    #[serde(default = "default_true")]
    pub enable_relay: bool,
    #[serde(default = "default_true")]
    pub enable_relay_client: bool,
    #[serde(default = "default_max_peers")]
    pub max_peers: u32,
    /// Maximum duration for a single relay circuit in seconds.
    #[serde(default = "default_relay_circuit_duration")]
    pub relay_max_circuit_duration_secs: u64,
    /// Maximum number of relay circuits this node will serve simultaneously.
    #[serde(default = "default_relay_max_circuits")]
    pub relay_max_circuits: usize,
    /// Automatically activate relay listener when NAT is detected as Private.
    #[serde(default = "default_true")]
    pub auto_relay: bool,
    /// Enable mDNS for automatic LAN peer discovery (default: true).
    #[serde(default = "default_true")]
    pub enable_mdns: bool,
    /// Gossip network ID for grouping nodes. All nodes sharing the same ID
    /// can decode each other's sealed gossip. Defaults to "swarmllm-mainnet-v1".
    /// Set to a custom value (e.g. "my-private-net") for private networks.
    #[serde(default)]
    pub gossip_network_id: Option<String>,
    /// Enable AutoNAT for NAT detection (default: true).
    /// Disable on WSL2 to prevent protocol negotiation noise that starves outbound substreams.
    #[serde(default = "default_true")]
    pub enable_autonat: bool,
    /// Enable DCUtR for hole punching (default: true).
    /// Disable on WSL2 to prevent protocol negotiation noise that starves outbound substreams.
    #[serde(default = "default_true")]
    pub enable_dcutr: bool,
    /// Enable E2E encryption for tensor forwards and control messages (default: true).
    #[serde(default = "default_true")]
    pub enable_encryption: bool,
    /// Enable zstd compression for tensor payloads sent over the network.
    /// Only payloads larger than `tensor_compress_threshold` bytes are compressed.
    #[serde(default = "default_true")]
    pub tensor_compression: bool,
    /// Enable zstd compression for cross-node prefix-KV snapshot payloads
    /// (Item 8 wire frames, tag 0x04). Off by default — only worth flipping
    /// when WAN measurements show wire size is the binding constraint
    /// (localhost's RTT-vs-wire trade is roughly neutral). Receivers always
    /// decompress regardless of this flag, so flipping it on a single peer
    /// doesn't require a coordinated upgrade.
    #[serde(default)]
    pub prefix_kv_compression: bool,
    /// Zstd compression level (1-22, default 1 for speed). Shared between
    /// tensor and prefix-KV compression.
    #[serde(default = "default_tensor_compress_level")]
    pub tensor_compress_level: i32,
    /// Minimum payload size in bytes before compression is applied (default 1024).
    /// Shared between tensor and prefix-KV.
    #[serde(default = "default_tensor_compress_threshold")]
    pub tensor_compress_threshold: usize,
    /// IP address to bind P2P listeners on (default: "0.0.0.0" = all interfaces).
    /// Set to "127.0.0.1" on WSL2 to prevent connections via unreliable NAT adapters.
    #[serde(default = "default_listen_address")]
    pub listen_address: String,
    /// Enable QUIC transport (default: true).
    /// Disable on WSL2 to prevent QUIC connection races with TCP (QUIC handshake is faster
    /// than TCP+Noise+Yamux, causing max_established_per_peer=1 to kill the TCP connection).
    #[serde(default = "default_true")]
    pub enable_quic: bool,
}

fn default_listen_address() -> String {
    "0.0.0.0".to_string()
}

/// Detect WSL2 by checking /proc/version for "microsoft" or "WSL".
pub(super) fn is_wsl2() -> bool {
    std::fs::read_to_string("/proc/version")
        .map(|v| {
            let lower = v.to_lowercase();
            lower.contains("microsoft") || lower.contains("wsl")
        })
        .unwrap_or(false)
}

fn default_max_peers() -> u32 {
    200
}

fn default_tensor_compress_level() -> i32 {
    1
}

fn default_tensor_compress_threshold() -> usize {
    1024
}

fn default_relay_circuit_duration() -> u64 {
    3600
}

fn default_relay_max_circuits() -> usize {
    16
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            bootstrap_peers: vec![],
            peer_exchange: true,
            enable_relay: true,
            enable_relay_client: true,
            max_peers: default_max_peers(),
            relay_max_circuit_duration_secs: default_relay_circuit_duration(),
            relay_max_circuits: default_relay_max_circuits(),
            auto_relay: true,
            enable_mdns: true,
            enable_autonat: true,
            enable_dcutr: true,
            enable_encryption: true,
            gossip_network_id: None,
            tensor_compression: true,
            prefix_kv_compression: false,
            tensor_compress_level: default_tensor_compress_level(),
            tensor_compress_threshold: default_tensor_compress_threshold(),
            listen_address: default_listen_address(),
            enable_quic: true,
        }
    }
}
