//! What UPnP has done for this node — so a router that stays silent is
//! explained, and a mapping this node holds is handed back when it stops.
//!
//! libp2p-upnp reports a mapping it made, a gateway it could not find and a
//! gateway behind another NAT. A router that REFUSES the mapping produces no
//! event at all: the crate logs it at `debug` and retries (from 0.6, with a
//! backoff, then gives up). The commonest refusal is a port another device on
//! the network already holds — a second node behind the same router — and a
//! tester ran two nodes that way for days, the second one silent, before
//! finding out (2026-10-08). So the silence itself is the signal: past
//! [`UPNP_QUIET_AFTER`] with UPnP on, nothing heard and no public address, the
//! router answered and would not map the port.

use std::time::{Duration, Instant};

/// How long UPnP may stay silent before the silence is explained. The gateway
/// search gives up within 10 s (`igd_next::SearchOptions::default`); a refused
/// mapping is retried at 30 s and 60 s more, so by three minutes three attempts
/// have been refused.
pub(super) const UPNP_QUIET_AFTER: Duration = Duration::from_secs(180);

/// How long a stopping node keeps its network running to hand its port
/// mappings back: the router's answer is one request on the local network.
pub(super) const UPNP_RELEASE_WAIT: Duration = Duration::from_millis(1500);

pub(super) struct UpnpWatch {
    enabled: bool,
    started: Instant,
    /// Any UPnP outcome at all — a mapping, its expiry, no gateway, a gateway
    /// behind another NAT. Each already says what happened.
    heard: bool,
    /// Mappings this node holds on the router now.
    mapped: u32,
    explained: bool,
}

impl UpnpWatch {
    pub(super) fn new(enabled: bool, started: Instant) -> Self {
        Self {
            enabled,
            started,
            heard: false,
            mapped: 0,
            explained: false,
        }
    }

    pub(super) fn note_mapped(&mut self) {
        self.heard = true;
        self.mapped = self.mapped.saturating_add(1);
    }

    pub(super) fn note_expired(&mut self) {
        self.heard = true;
        self.mapped = self.mapped.saturating_sub(1);
    }

    /// No gateway, or one behind another NAT: said by its own event.
    pub(super) fn note_answered(&mut self) {
        self.heard = true;
    }

    /// Does this node hold a mapping on the router that it should hand back?
    pub(super) fn holds_a_mapping(&self) -> bool {
        self.mapped > 0
    }

    /// Is now the moment to explain UPnP's silence? True ONCE: UPnP is on,
    /// nothing has been heard from it for [`UPNP_QUIET_AFTER`], and the node
    /// is not reachable some other way (a forwarded port, a public address).
    pub(super) fn silence_to_explain(&mut self, now: Instant, publicly_reachable: bool) -> bool {
        if !self.enabled || self.heard || self.explained || publicly_reachable {
            return false;
        }
        if now.saturating_duration_since(self.started) < UPNP_QUIET_AFTER {
            return false;
        }
        self.explained = true;
        true
    }
}

/// What the owner is told when the router would not map this node's ports.
pub(super) fn refused_mapping_message(udp_port: u16, tcp_port: u16) -> String {
    format!(
        "UPnP: the router did not open port {udp_port} (UDP) or {tcp_port} (TCP) for this \
         computer. The usual cause is another device on this network already holding those \
         ports — for example a second SwarmLLM node: a router gives a port to one device \
         only. This node is reached through a relay meanwhile. To make it reachable directly, \
         run it on a different port (`swarmllm run -p <port>`), or stop the other node and \
         restart this one."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn watch_started_ago(enabled: bool, ago: Duration) -> (UpnpWatch, Instant) {
        let now = Instant::now();
        let started = now.checked_sub(ago).expect("a clock old enough");
        (UpnpWatch::new(enabled, started), now)
    }

    /// The two-nodes-behind-one-router case: UPnP on, nothing heard, no public
    /// address — explained once, after the quiet period and not before.
    #[test]
    fn a_silent_router_is_explained_once_after_the_quiet_period() {
        let (mut early, now) = watch_started_ago(true, UPNP_QUIET_AFTER / 2);
        assert!(!early.silence_to_explain(now, false));

        let (mut w, now) = watch_started_ago(true, UPNP_QUIET_AFTER);
        assert!(w.silence_to_explain(now, false));
        assert!(!w.silence_to_explain(now, false), "once only");
    }

    /// Every case where something else already said what happened, or where
    /// there is nothing to explain.
    #[test]
    fn nothing_is_explained_where_upnp_spoke_or_was_not_needed() {
        let ago = UPNP_QUIET_AFTER * 2;

        let (mut mapped, now) = watch_started_ago(true, ago);
        mapped.note_mapped();
        assert!(!mapped.silence_to_explain(now, false));
        assert!(mapped.holds_a_mapping());
        mapped.note_expired();
        assert!(!mapped.holds_a_mapping());

        let (mut no_gateway, now) = watch_started_ago(true, ago);
        no_gateway.note_answered();
        assert!(!no_gateway.silence_to_explain(now, false));
        assert!(!no_gateway.holds_a_mapping());

        let (mut reachable, now) = watch_started_ago(true, ago);
        assert!(
            !reachable.silence_to_explain(now, true),
            "reachable another way"
        );

        let (mut off, now) = watch_started_ago(false, ago);
        assert!(!off.silence_to_explain(now, false), "UPnP turned off");
    }

    #[test]
    fn the_message_names_both_ports_and_the_way_out() {
        let m = refused_mapping_message(8800, 8810);
        assert!(m.contains("8800") && m.contains("8810"));
        assert!(m.contains("-p <port>"));
    }
}
