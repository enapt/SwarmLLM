//! Pool invite code v2 — `swarmpool://...` shareable bootstrap blob.
//!
//! The original 8-character invite code (still kept inside this blob as the
//! actual join token) only worked once both nodes were already on the same
//! libp2p swarm — it carried no information about how to *find* the inviter.
//! In a fully decentralized network that's fine (DHT bootstrap), but during
//! the bootstrap-before-decentralization phase the pool owner is often the
//! only address the joiner can reach. v2 fixes that by bundling the owner's
//! reachable listen multiaddrs alongside the join token, so a fresh node
//! anywhere with the code can:
//!
//! 1. Decode it.
//! 2. Dial one of the owner's multiaddrs over the wire (Tailscale, LAN,
//!    public IP, whatever's in the bundle).
//! 3. Once a libp2p connection lands, broadcast the existing
//!    `PoolMessage::JoinRequest { code_hash, ... }` over GossipSub.
//!
//! The owner's handler still matches `code_hash` against its
//! `invite_codes` map exactly as before — there is **no wire-protocol
//! change** for the pool-join itself. v2 is purely the rendezvous wrapper.
//!
//! ## Wire format
//!
//! - prefix: `swarmpool://`
//! - payload: base64url(no-pad) of `key (32B) || nonce (12B) || ciphertext`
//! - ciphertext: ChaCha20-Poly1305 of `serde_json::to_vec(&InviteCodePayload)`
//!
//! Encryption is here to prevent casual IP harvesting from anyone glancing
//! at a code pasted into a chat window. The key is embedded in the code, so
//! anyone with the full code can decrypt — same threat model as the existing
//! `swarm://` network code in `src/network/discovery.rs`.

use base64::Engine;
use chacha20poly1305::aead::{Aead, KeyInit, OsRng};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use crate::error::SwarmError;
use crate::types::NodeId;

/// `swarmpool://...` is the v2 prefix. Separate from `swarm://` (network-only
/// invite) so callers can tell at a glance which flow a pasted code drives.
pub const INVITE_PREFIX: &str = "swarmpool://";

/// Current wire-format version. Bump when changing the inner payload.
pub const INVITE_VERSION: u8 = 1;

/// Hard cap on the encoded length we'll attempt to decode. A typical blob is
/// ~300-500 chars; anything bigger is either garbage or an attempted DoS.
const MAX_ENCODED_LEN: usize = 2048;

/// Inner payload of a v2 invite code — what the recipient gets after
/// decryption.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InviteCodePayload {
    /// Wire-format version. Must equal `INVITE_VERSION` or decode fails fast.
    pub version: u8,
    /// The pool ID. Equals the owner's `NodeId`; the libp2p `PeerId` is
    /// derivable from this via `peer_id_to_node_id`'s inverse.
    pub pool_id: NodeId,
    /// Optional human-friendly pool name for the joining UI. Empty string if
    /// the owner hasn't named the pool.
    pub pool_name: String,
    /// The owner's reachable listen multiaddrs (each terminated with
    /// `/p2p/<peer_id>` so a remote dialer can verify the target identity).
    /// The joiner dials each in order until one connects.
    pub multiaddrs: Vec<String>,
    /// The 8-char short code that still drives the actual join wire protocol.
    /// The joiner hashes this and broadcasts `JoinRequest { code_hash, ... }`
    /// over GossipSub — same as the legacy 8-char path.
    pub code: String,
    /// Expiry as a unix-seconds timestamp. Mirrored from the underlying
    /// `PoolInviteCode.expires_at` so decoders can fail fast before dialing.
    pub expires_at_unix: i64,
}

impl InviteCodePayload {
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now().timestamp() > self.expires_at_unix
    }
}

/// Encode an invite code into the `swarmpool://...` wire form.
///
/// Returns `Err` only if JSON serialization of the payload fails — in
/// practice that means a bug, since all field types are infallibly
/// serializable.
pub fn encode_invite_code(payload: &InviteCodePayload) -> Result<String, SwarmError> {
    let plaintext = serde_json::to_vec(payload)
        .map_err(|e| SwarmError::Internal(format!("invite payload serialize: {e}")))?;

    let mut key_bytes = [0u8; 32];
    let mut nonce_bytes = [0u8; 12];
    OsRng.fill_bytes(&mut key_bytes);
    OsRng.fill_bytes(&mut nonce_bytes);

    let cipher = ChaCha20Poly1305::new_from_slice(&key_bytes).expect("valid key size");
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_slice())
        .expect("ChaCha20Poly1305 encryption cannot fail on a valid key");

    let mut packed = Vec::with_capacity(32 + 12 + ciphertext.len());
    packed.extend_from_slice(&key_bytes);
    packed.extend_from_slice(&nonce_bytes);
    packed.extend_from_slice(&ciphertext);
    let encoded = base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&packed);
    Ok(format!("{INVITE_PREFIX}{encoded}"))
}

/// Decode a `swarmpool://...` blob back to its inner payload.
///
/// Returns `SwarmError::Validation` for malformed input, expired codes, or
/// version mismatches — anything the user might paste that we can recognize
/// and report cleanly. Decryption failure also surfaces as `Validation`
/// (rather than `Internal`) because the most likely cause is a truncated
/// or mistyped code, not a daemon bug.
pub fn decode_invite_code(raw: &str) -> Result<InviteCodePayload, SwarmError> {
    let trimmed = raw.trim();
    let Some(encoded) = trimmed.strip_prefix(INVITE_PREFIX) else {
        return Err(SwarmError::Validation(format!(
            "Invite code must start with '{INVITE_PREFIX}'"
        )));
    };

    if encoded.is_empty() {
        return Err(SwarmError::Validation("Invite code is empty".into()));
    }
    if encoded.len() > MAX_ENCODED_LEN {
        return Err(SwarmError::Validation(format!(
            "Invite code too long (max {MAX_ENCODED_LEN} chars)"
        )));
    }

    let packed = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded.as_bytes())
        .map_err(|e| SwarmError::Validation(format!("Invite code is not valid base64url: {e}")))?;

    if packed.len() < 32 + 12 + 16 {
        return Err(SwarmError::Validation(
            "Invite code is too short to be valid".into(),
        ));
    }

    let key_bytes = &packed[..32];
    let nonce_bytes = &packed[32..44];
    let ciphertext = &packed[44..];

    let cipher = ChaCha20Poly1305::new_from_slice(key_bytes)
        .map_err(|e| SwarmError::Validation(format!("Invite code key invalid: {e}")))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|_| {
        SwarmError::Validation(
            "Invite code is corrupt or was edited — ask the inviter to regenerate".into(),
        )
    })?;

    let payload: InviteCodePayload = serde_json::from_slice(&plaintext).map_err(|e| {
        SwarmError::Validation(format!(
            "Invite code payload is not valid (newer daemon version?): {e}"
        ))
    })?;

    if payload.version != INVITE_VERSION {
        return Err(SwarmError::Validation(format!(
            "Unsupported invite code version {} (this daemon supports v{})",
            payload.version, INVITE_VERSION
        )));
    }
    if payload.is_expired() {
        return Err(SwarmError::Validation(
            "Invite code has expired — ask the inviter to generate a new one".into(),
        ));
    }
    if payload.code.len() != 8 {
        return Err(SwarmError::Validation(
            "Invite code payload is malformed (join token wrong length)".into(),
        ));
    }
    Ok(payload)
}

/// Quick sniff: does this string look like a v2 code? Used by the API layer
/// to route a pasted blob to either the v2 path or the legacy 8-char path.
pub fn looks_like_v2(raw: &str) -> bool {
    raw.trim().starts_with(INVITE_PREFIX)
}

/// Does at least one of these multiaddrs denote an address reachable from the
/// **open internet** (as opposed to LAN-only)?
///
/// Distinct from `network::manager::events::addr_is_remotely_reachable`, which
/// keeps LAN + CGNAT/Tailscale addresses (reachable on the same overlay). This
/// is the stricter test used to decide whether an invite code will actually
/// work for a stranger somewhere else on the internet — if it returns false,
/// the code is LAN-only and we warn the user rather than let it fail silently.
///
/// True for: global/public IPv4 & IPv6, any DNS name (`/dns4`, `/dns6`,
/// `/dnsaddr` — a dynamic-DNS anchor), and relay-circuit addresses
/// (`/p2p-circuit`). False for RFC1918 LAN, loopback, link-local, IPv6 ULA,
/// and CGNAT/Tailscale (100.64.0.0/10) — those only work locally or within a
/// private overlay.
pub fn any_internet_reachable(multiaddrs: &[String]) -> bool {
    multiaddrs
        .iter()
        .any(|s| match s.parse::<libp2p::Multiaddr>() {
            Ok(m) => multiaddr_is_internet_reachable(&m),
            Err(_) => false,
        })
}

/// Whether a multiaddr is reachable from the open internet.
///
/// Public IP or DNS name, or a relay circuit. Deliberately EXCLUDES private
/// ranges, link-local, loopback and CGNAT (100.64.0.0/10, which Tailscale hands
/// out) — those are reachable from somewhere, but not from the internet.
///
/// Shared with the AutoNAT handler, which must not accept a LAN peer's
/// successful probe of a LAN address as proof of internet reachability. See
/// `network::manager::events`.
pub(crate) fn multiaddr_is_internet_reachable(addr: &libp2p::Multiaddr) -> bool {
    use libp2p::multiaddr::Protocol;
    let mut via_relay = false;
    for proto in addr.iter() {
        match proto {
            Protocol::P2pCircuit => via_relay = true,
            Protocol::Dns(_) | Protocol::Dns4(_) | Protocol::Dns6(_) | Protocol::Dnsaddr(_) => {
                return true
            }
            Protocol::Ip4(ip) if is_public_ipv4(ip) => return true,
            Protocol::Ip6(ip) if is_public_ipv6(ip) => return true,
            _ => {}
        }
    }
    via_relay
}

fn is_public_ipv4(ip: std::net::Ipv4Addr) -> bool {
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_unspecified()
        || ip.is_broadcast()
        || is_cgnat_ipv4(ip))
}

/// RFC 6598 shared address space (100.64.0.0/10) — carrier-grade NAT and the
/// range Tailscale hands out. Reachable within the overlay, not the internet.
fn is_cgnat_ipv4(ip: std::net::Ipv4Addr) -> bool {
    let o = ip.octets();
    o[0] == 100 && (o[1] & 0xc0) == 0x40
}

fn is_public_ipv6(ip: std::net::Ipv6Addr) -> bool {
    let seg0 = ip.segments()[0];
    !(ip.is_loopback()
        || ip.is_unspecified()
        || (seg0 & 0xffc0) == 0xfe80  // link-local fe80::/10
        || (seg0 & 0xfe00) == 0xfc00) // unique-local fc00::/7
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A LAN peer confirming a LAN address does not make this node public.**
    ///
    /// AutoNAT servers are ordinary peers, so a node on the same subnet happily
    /// confirms an RFC1918 address. Treating that as "Public" makes a NAT'd node
    /// skip reserving a relay and sit unreachable from the internet while
    /// reporting otherwise. Observed live 2026-08-05: a node reporting
    /// `nat: Public` on the strength of confirmations for these exact addresses.
    #[test]
    fn a_confirmed_private_address_is_not_internet_reachable() {
        for addr in [
            "/ip4/192.168.1.53/tcp/8810",   // RFC1918, confirmed by a LAN peer
            "/ip4/10.255.255.254/tcp/8810", // RFC1918
            "/ip4/169.254.83.107/tcp/8810", // link-local — never routable
            "/ip4/172.17.0.1/tcp/8810",     // Docker bridge
            "/ip4/127.0.0.1/tcp/8810",      // loopback
            "/ip4/100.64.0.7/tcp/8810",     // CGNAT / Tailscale
        ] {
            let m: libp2p::Multiaddr = addr.parse().unwrap();
            assert!(
                !multiaddr_is_internet_reachable(&m),
                "{addr} must not count as internet-reachable"
            );
        }
    }

    /// The addresses that genuinely do mean public reachability.
    #[test]
    fn public_addresses_and_relay_circuits_are_internet_reachable() {
        for addr in [
            "/ip4/203.0.113.7/tcp/8810",
            "/dns4/anchor.example.net/tcp/8810",
        ] {
            let m: libp2p::Multiaddr = addr.parse().unwrap();
            assert!(
                multiaddr_is_internet_reachable(&m),
                "{addr} should count as internet-reachable"
            );
        }
    }

    fn sample_payload() -> InviteCodePayload {
        InviteCodePayload {
            version: INVITE_VERSION,
            pool_id: NodeId([7u8; 32]),
            pool_name: "Test Pool".into(),
            multiaddrs: vec![
                "/ip4/100.64.0.5/tcp/8810/p2p/12D3KooWFakeXXX".into(),
                "/ip4/192.168.1.5/udp/8800/quic-v1/p2p/12D3KooWFakeXXX".into(),
            ],
            code: "A3F7K2M9".into(),
            expires_at_unix: chrono::Utc::now().timestamp() + 3600,
        }
    }

    #[test]
    fn roundtrip_preserves_payload() {
        let original = sample_payload();
        let encoded = encode_invite_code(&original).unwrap();
        assert!(encoded.starts_with(INVITE_PREFIX));

        let decoded = decode_invite_code(&encoded).unwrap();
        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.pool_id, original.pool_id);
        assert_eq!(decoded.pool_name, original.pool_name);
        assert_eq!(decoded.multiaddrs, original.multiaddrs);
        assert_eq!(decoded.code, original.code);
        assert_eq!(decoded.expires_at_unix, original.expires_at_unix);
    }

    #[test]
    fn whitespace_around_code_is_tolerated() {
        let original = sample_payload();
        let encoded = encode_invite_code(&original).unwrap();
        let padded = format!("  \n{encoded}\t  ");
        let decoded = decode_invite_code(&padded).unwrap();
        assert_eq!(decoded.code, original.code);
    }

    #[test]
    fn missing_prefix_fails() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        let stripped = encoded.strip_prefix(INVITE_PREFIX).unwrap();
        let err = decode_invite_code(stripped).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn truncated_code_fails_cleanly() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        let truncated = &encoded[..encoded.len() - 5];
        let err = decode_invite_code(truncated).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn tampered_ciphertext_fails_authentication() {
        let encoded = encode_invite_code(&sample_payload()).unwrap();
        // Flip the last char of the base64url body to a different valid char.
        let mut bytes: Vec<char> = encoded.chars().collect();
        let last = bytes.last_mut().unwrap();
        *last = if *last == 'A' { 'B' } else { 'A' };
        let tampered: String = bytes.into_iter().collect();
        let err = decode_invite_code(&tampered).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn expired_code_rejected() {
        let mut payload = sample_payload();
        payload.expires_at_unix = chrono::Utc::now().timestamp() - 60;
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("expired"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn version_mismatch_rejected() {
        let mut payload = sample_payload();
        payload.version = INVITE_VERSION + 7;
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("version"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn malformed_join_token_length_rejected() {
        let mut payload = sample_payload();
        payload.code = "TOOSHORT".into(); // 8 — should pass
        let encoded = encode_invite_code(&payload).unwrap();
        decode_invite_code(&encoded).unwrap();

        payload.code = "WAYTOOLONG".into(); // 10
        let encoded = encode_invite_code(&payload).unwrap();
        let err = decode_invite_code(&encoded).unwrap_err();
        assert!(matches!(err, SwarmError::Validation(_)));
    }

    #[test]
    fn oversized_input_rejected_before_decode() {
        let huge = format!("{INVITE_PREFIX}{}", "A".repeat(MAX_ENCODED_LEN + 1));
        let err = decode_invite_code(&huge).unwrap_err();
        match err {
            SwarmError::Validation(msg) => assert!(msg.contains("too long"), "got: {msg}"),
            other => panic!("expected Validation, got {other:?}"),
        }
    }

    #[test]
    fn looks_like_v2_sniff() {
        assert!(looks_like_v2("swarmpool://abc"));
        assert!(looks_like_v2("  swarmpool://xyz  "));
        assert!(!looks_like_v2("A3F7K2M9"));
        assert!(!looks_like_v2("swarm://abc"));
        assert!(!looks_like_v2(""));
    }

    // A real, parseable peer id — the fixtures below carry a valid `/p2p/...`
    // suffix exactly like production `listen_multiaddrs` entries do, so the
    // classifier is exercised on strings that actually parse.
    fn pid() -> libp2p::PeerId {
        libp2p::PeerId::random()
    }

    #[test]
    fn internet_reachable_public_ipv4() {
        let p = pid();
        assert!(any_internet_reachable(&[format!(
            "/ip4/203.0.113.5/tcp/8810/p2p/{p}"
        )]));
        assert!(any_internet_reachable(&[format!(
            "/ip4/8.8.8.8/udp/8800/quic-v1/p2p/{p}"
        )]));
    }

    #[test]
    fn internet_reachable_dns_anchor() {
        // A dynamic-DNS anchor address is internet-reachable regardless of the
        // IP it currently resolves to.
        let p = pid();
        assert!(any_internet_reachable(&[format!(
            "/dns4/anchor.example.net/tcp/8810/p2p/{p}"
        )]));
        assert!(any_internet_reachable(&[format!(
            "/dns6/anchor.example.net/tcp/8810/p2p/{p}"
        )]));
    }

    #[test]
    fn internet_reachable_relay_circuit() {
        let (relay, me) = (pid(), pid());
        assert!(any_internet_reachable(&[format!(
            "/ip4/203.0.113.9/tcp/8810/p2p/{relay}/p2p-circuit/p2p/{me}"
        )]));
    }

    #[test]
    fn internet_reachable_public_ipv6() {
        let p = pid();
        assert!(any_internet_reachable(&[format!(
            "/ip6/2001:db8::1/tcp/8810/p2p/{p}"
        )]));
    }

    #[test]
    fn not_internet_reachable_lan_only() {
        // RFC1918 LAN, loopback, link-local, IPv6 ULA — none reachable from
        // the open internet.
        let p = pid();
        assert!(!any_internet_reachable(&[
            format!("/ip4/192.168.1.5/tcp/8810/p2p/{p}"),
            format!("/ip4/10.0.0.7/udp/8800/quic-v1/p2p/{p}"),
            format!("/ip4/127.0.0.1/tcp/8810/p2p/{p}"),
            format!("/ip6/fc00::1/tcp/8810/p2p/{p}"),
        ]));
    }

    #[test]
    fn not_internet_reachable_cgnat_tailscale() {
        // 100.64.0.0/10 — CGNAT / Tailscale. Works on the overlay, not the
        // open internet, so an invite with only these must warn.
        let p = pid();
        assert!(!any_internet_reachable(&[format!(
            "/ip4/100.64.10.5/tcp/8810/p2p/{p}"
        )]));
        assert!(!any_internet_reachable(&[format!(
            "/ip4/100.127.255.254/tcp/8810/p2p/{p}"
        )]));
    }

    #[test]
    fn internet_reachable_mixed_list_true_if_any() {
        // A LAN address alongside a public one still counts as reachable.
        let p = pid();
        assert!(any_internet_reachable(&[
            format!("/ip4/192.168.1.5/tcp/8810/p2p/{p}"),
            format!("/dns4/anchor.example.net/tcp/8810/p2p/{p}"),
        ]));
    }

    #[test]
    fn not_internet_reachable_empty_or_garbage() {
        assert!(!any_internet_reachable(&[]));
        assert!(!any_internet_reachable(&["not a multiaddr".into()]));
    }
}
