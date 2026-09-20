//! How much traffic this node is actually putting on the wire.
//!
//! A user on a shared home connection noticed it being hammered, looked for the
//! setting to fix it, found `resources.max_bandwidth_mbps` already set to 1, and
//! measured ~11 Mbps of traffic anyway (2026-09-11 suggestion). Two separate
//! problems, and this file is about the second one.
//!
//! The first: that setting throttles shard SERVING and nothing else — it is
//! enforced, in `network::manager::requests`, but only there. Gossip, DHT
//! maintenance, manifest announcements and inference traffic are all outside
//! it. The setting's own documentation says so; nothing the user could see did.
//!
//! The second is the one that made the first hard to discover. **There was no
//! way to ask this node how much traffic it was generating.** The only method
//! available was to stop the daemon and diff `/sys/class/net`, which is what
//! the reporter did — for a question the daemon is far better placed to answer
//! than its operator is.
//!
//! So the counting is turned on at the transport, where every byte of every
//! protocol passes, and the totals are read back here. libp2p already
//! implements the counting (`libp2p::metrics::BandwidthTransport` wraps the
//! muxer and counts reads and writes); what it does NOT offer is a way to read
//! its counters programmatically — they are registered into a Prometheus
//! registry, and `prometheus_client::Registry` exposes no iteration. So the
//! registry is encoded and the two totals read out of the text. That is the
//! whole reason this file exists rather than two atomics.
//!
//! The alternative was our own copy of the transport wrapper, and it was
//! rejected: the builder offers no phase where a transport can be wrapped by
//! hand, so a hand-rolled version would have to sit before the relay and DNS
//! layers and would count a different set of bytes depending on where it
//! landed. Upstream's wrapper is applied in exactly one place, after all of
//! them.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Cumulative bytes in and out since the daemon started.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BandwidthTotals {
    pub inbound_bytes: u64,
    pub outbound_bytes: u64,
}

/// The totals plus how fast they are moving, which is the figure anyone
/// actually asks for.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BandwidthSnapshot {
    pub inbound_bytes: u64,
    pub outbound_bytes: u64,
    /// Bytes per second over the interval between the last two refreshes, or
    /// `None` until there have been two.
    pub inbound_bytes_per_sec: Option<f64>,
    pub outbound_bytes_per_sec: Option<f64>,
}

/// The registry libp2p's transport counters write into, plus the reader.
pub struct BandwidthMeter {
    /// Held so the totals can be read after the swarm is built. The lock is
    /// taken for the duration of a build or an encode, never across an await.
    registry: Mutex<prometheus_client::registry::Registry>,
    /// Has the counting actually been switched on? Set when the swarm builder
    /// takes the registry.
    armed: AtomicBool,
    /// Has the "no counters found" warning been logged? Once is enough; this
    /// is read on every stats tick.
    warned_absent: AtomicBool,
    /// The last refresh, and what it computed. A rate needs two readings, and
    /// the pair must be taken by ONE caller at a steady cadence — two readers
    /// interleaving would each see part of the interval and both would be
    /// wrong. `refresh` is that caller (the health monitor's tick); everything
    /// else reads `current`.
    last: Mutex<Option<Refresh>>,
}

#[derive(Clone, Copy)]
struct Refresh {
    at: std::time::Instant,
    totals: BandwidthTotals,
    inbound_bytes_per_sec: Option<f64>,
    outbound_bytes_per_sec: Option<f64>,
}

impl Default for BandwidthMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl BandwidthMeter {
    pub fn new() -> Self {
        Self {
            registry: Mutex::new(prometheus_client::registry::Registry::default()),
            armed: AtomicBool::new(false),
            warned_absent: AtomicBool::new(false),
            last: Mutex::new(None),
        }
    }

    /// Take a reading and work out the rate since the previous one.
    ///
    /// Called from the health monitor's tick, and from nowhere else — see
    /// `last`. Cheap: one encode of a registry holding a handful of rows.
    pub fn refresh(&self) {
        let Some(totals) = self.totals() else {
            return;
        };
        let now = std::time::Instant::now();
        let mut last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        let rates = last.as_ref().and_then(|prev| {
            let secs = now.duration_since(prev.at).as_secs_f64();
            // A refresh twice in the same instant would divide by ~0 and
            // publish an absurd rate; nothing is lost by skipping it.
            (secs >= 0.5).then(|| {
                (
                    // `saturating_sub` because a counter can only rise, and if
                    // one ever appears to fall the honest answer is 0 rather
                    // than an enormous number from an underflow.
                    totals
                        .inbound_bytes
                        .saturating_sub(prev.totals.inbound_bytes) as f64
                        / secs,
                    totals
                        .outbound_bytes
                        .saturating_sub(prev.totals.outbound_bytes) as f64
                        / secs,
                )
            })
        });
        *last = Some(Refresh {
            at: now,
            totals,
            inbound_bytes_per_sec: rates.map(|r| r.0),
            outbound_bytes_per_sec: rates.map(|r| r.1),
        });
    }

    /// The most recent reading, or `None` when nothing is counting or nothing
    /// has been read yet.
    pub fn current(&self) -> Option<BandwidthSnapshot> {
        let last = self.last.lock().unwrap_or_else(|e| e.into_inner());
        last.as_ref().map(|r| BandwidthSnapshot {
            inbound_bytes: r.totals.inbound_bytes,
            outbound_bytes: r.totals.outbound_bytes,
            inbound_bytes_per_sec: r.inbound_bytes_per_sec,
            outbound_bytes_per_sec: r.outbound_bytes_per_sec,
        })
    }

    /// Hand the registry to the swarm builder, which registers the transport
    /// counters into it. Called once, while the swarm is being built.
    ///
    /// The closure shape is what keeps the lock out of the caller: the builder
    /// needs `&mut Registry` for one call, and nothing else may hold it then.
    pub fn arm<R>(&self, f: impl FnOnce(&mut prometheus_client::registry::Registry) -> R) -> R {
        let mut registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        let out = f(&mut registry);
        self.armed.store(true, Ordering::Relaxed);
        out
    }

    /// Bytes in and out since startup, or `None` when nothing is counting.
    ///
    /// `None` rather than zero, deliberately. A figure that reads 0.0 Mbps
    /// whether the node is silent or the counters were never wired is the
    /// shape of a gauge that cannot fire, and this one exists precisely
    /// because a user could not tell those two apart from outside.
    pub fn totals(&self) -> Option<BandwidthTotals> {
        if !self.armed.load(Ordering::Relaxed) {
            return None;
        }
        let mut text = String::with_capacity(1024);
        {
            let registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
            if prometheus_client::encoding::text::encode(&mut text, &registry).is_err() {
                return None;
            }
        }
        let totals = parse_bandwidth_totals(&text);
        // Absent has two causes and only one of them is a fault. Every node
        // starts in the other one: the health monitor's first tick runs before
        // any byte has crossed the transport, so the family has no rows yet.
        // Warning there fired on EVERY start, 51 ms in, announcing that the
        // figure "will be absent" about a figure that appears 30-60 s later —
        // and `warned_absent` is a one-shot, so nothing ever retracted it.
        // `metric_is_registered` is the discriminator; see its doc comment.
        if totals.is_none()
            && !metric_is_registered(&text)
            && !self.warned_absent.swap(true, Ordering::Relaxed)
        {
            tracing::warn!(
                "network traffic counters are armed but report nothing — the metric \
                 libp2p registers may have been renamed; the traffic figure will be absent"
            );
        }
        totals
    }
}

/// How many bytes of gossip this node has put on the wire, by topic.
///
/// The transport counters above answer "how much", and two field reports
/// (2026-09-11 and 2026-09-20) showed that is not the question an operator is
/// really asking — both had to work out WHICH traffic by elimination: inference
/// measured at zero, shard serving bounded by counting a log line, "the rest
/// must be announcements". The second report asked for the split by name.
///
/// Gossip is the half worth measuring first, because it turned out to be almost
/// all of it: an idle node was re-sending 825 KB of model manifests every 30 s
/// (fixed in v0.3.194). It also cannot be measured at the transport — libp2p's
/// counters label by transport stack (`/ip4/tcp`, `/ip4/udp/quic-v1`) and know
/// nothing about protocols above the muxer.
///
/// So this reads GossipSub's own per-topic counters, which count **every copy
/// put on the wire** — `send_message` records a publish AND a forward, once per
/// recipient — which is the figure that matches what leaves the interface, not
/// the number of distinct messages composed. The same encode-and-parse shape as
/// `BandwidthMeter`, and for the same reason: `prometheus_client::Registry`
/// cannot be iterated.
pub struct GossipMeter {
    registry: Mutex<prometheus_client::registry::Registry>,
    armed: AtomicBool,
}

/// Gossip bytes, in total and per topic.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GossipTotals {
    pub sent_bytes: u64,
    pub recv_bytes: u64,
    /// `(topic, sent, received)`, sorted by topic so a reader diffing two
    /// snapshots is not comparing a reordered list.
    pub by_topic: Vec<(String, u64, u64)>,
}

impl Default for GossipMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl GossipMeter {
    pub fn new() -> Self {
        Self {
            registry: Mutex::new(prometheus_client::registry::Registry::default()),
            armed: AtomicBool::new(false),
        }
    }

    /// Hand a prefixed sub-registry to GossipSub's `with_metrics`, once, while
    /// the behaviour is being built.
    ///
    /// The prefix is ours rather than upstream's because `with_metrics`
    /// registers into whatever registry it is given, under bare names
    /// (`topic_msg_sent_bytes`). Naming them here makes what we parse a
    /// decision we made rather than one we inherited.
    pub fn arm<R>(&self, f: impl FnOnce(&mut prometheus_client::registry::Registry) -> R) -> R {
        let mut registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        let out = f(registry.sub_registry_with_prefix(GOSSIP_METRIC_PREFIX));
        self.armed.store(true, Ordering::Relaxed);
        out
    }

    /// Bytes of gossip sent and received since startup, or `None` when nothing
    /// is counting.
    ///
    /// `None` rather than zero, for the reason `BandwidthMeter::totals` gives:
    /// a figure that reads 0 whether the node is silent or the counting was
    /// never wired cannot be told apart from outside, and that is the exact
    /// complaint this whole module exists to answer.
    ///
    /// **Deliberately warns about nothing.** `prometheus_client` writes a
    /// `Family`'s metadata as soon as it is registered and its ROWS only once a
    /// label set exists, so every node reads `None` here until its first gossip
    /// message — and a diagnostic that fired on that would fire on every start,
    /// which is precisely the bug gotcha #582 records against the sibling
    /// meter. There the warning was worth keeping because an upstream RENAME
    /// was a real risk; here the metric names are registered under our own
    /// prefix by our own call, so a rename cannot happen without this file
    /// changing, and `the_gossip_counters_are_read_out_of_the_shape_gossipsub_writes`
    /// pins the shape instead.
    pub fn totals(&self) -> Option<GossipTotals> {
        if !self.armed.load(Ordering::Relaxed) {
            return None;
        }
        let mut text = String::with_capacity(2048);
        {
            let registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
            if prometheus_client::encoding::text::encode(&mut text, &registry).is_err() {
                return None;
            }
        }
        parse_gossip_totals(&text)
    }

    /// The encoded registry, so a test can assert against what GossipSub
    /// ACTUALLY registers rather than against a shape written by hand.
    #[cfg(test)]
    fn encoded_for_test(&self) -> String {
        let mut text = String::new();
        let registry = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        let _ = prometheus_client::encoding::text::encode(&mut text, &registry);
        text
    }
}

/// Prefix this node registers GossipSub's metrics under.
const GOSSIP_METRIC_PREFIX: &str = "gossip";

/// The two rows read out of it, already carrying the prefix and OpenMetrics'
/// counter suffix.
const GOSSIP_SENT_METRIC: &str = "gossip_topic_msg_sent_bytes_total";
const GOSSIP_RECV_METRIC: &str = "gossip_topic_msg_recv_bytes_total";

/// Sum GossipSub's per-topic byte counters.
///
/// The label is `hash`, which for an `IdentTopic` — the only kind this node
/// publishes — is the topic name itself, so the breakdown is readable.
fn parse_gossip_totals(text: &str) -> Option<GossipTotals> {
    use std::collections::BTreeMap;
    let mut per_topic: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut saw_any = false;

    for line in text.lines() {
        let (rest, is_sent) = if let Some(rest) = line.strip_prefix(GOSSIP_SENT_METRIC) {
            (rest, true)
        } else if let Some(rest) = line.strip_prefix(GOSSIP_RECV_METRIC) {
            (rest, false)
        } else {
            continue;
        };
        let Some((labels, value)) = rest.rsplit_once(' ') else {
            continue;
        };
        let Ok(bytes) = value.trim().parse::<f64>() else {
            continue;
        };
        let topic = extract_topic_label(labels).unwrap_or("unknown").to_string();
        let slot = per_topic.entry(topic).or_insert((0, 0));
        if is_sent {
            slot.0 += bytes as u64;
        } else {
            slot.1 += bytes as u64;
        }
        saw_any = true;
    }

    if !saw_any {
        return None;
    }
    let mut totals = GossipTotals::default();
    for (topic, (sent, recv)) in per_topic {
        totals.sent_bytes += sent;
        totals.recv_bytes += recv;
        totals.by_topic.push((topic, sent, recv));
    }
    Some(totals)
}

/// Pull the topic out of `{hash="swarmllm/models"}`.
fn extract_topic_label(labels: &str) -> Option<&str> {
    let start = labels.find("hash=\"")? + "hash=\"".len();
    let rest = &labels[start..];
    let end = rest.find('"')?;
    Some(&rest[..end])
}

/// Metric libp2p registers its transport byte counters as.
///
/// `libp2p_metrics::BandwidthTransport` registers `bandwidth` with
/// `Unit::Bytes` under the `libp2p` sub-registry, which OpenMetrics renders as
/// this name with a `_total` suffix for a counter. Both halves are upstream's
/// choice, so this constant is a contract with a specific version of libp2p and
/// `the_totals_are_read_out_of_the_shape_libp2p_writes` pins it.
const BANDWIDTH_METRIC: &str = "libp2p_bandwidth_bytes_total";

/// The same metric without the counter's `_total` suffix, which is the name
/// OpenMetrics writes the `# HELP` / `# TYPE` / `# UNIT` metadata under.
///
/// Those lines appear as soon as the family is REGISTERED. The data rows do
/// not: `prometheus_client`'s `Family` encodes one row per label set, and
/// libp2p creates the first label set on the first byte that actually moves.
/// So an armed node that has not yet sent anything encodes metadata and
/// nothing else — which is what separates "registered and silent" from
/// "renamed", the only two ways the totals can come back absent.
const BANDWIDTH_METRIC_BASE: &str = "libp2p_bandwidth_bytes";

/// Is the metric registered under the name we expect, whether or not it has
/// recorded anything yet?
fn metric_is_registered(text: &str) -> bool {
    text.lines().any(|line| {
        ["# HELP ", "# TYPE ", "# UNIT "].iter().any(|prefix| {
            line.strip_prefix(prefix)
                .is_some_and(|rest| rest.starts_with(BANDWIDTH_METRIC_BASE))
        })
    })
}

/// Sum the per-protocol counters by direction.
///
/// One line per (protocol stack, direction) — `/ip4/tcp/yamux` and `/ip4/udp/quic`
/// are counted separately, and this does not care which: the question a user
/// asks is how much this node is sending, not over what.
fn parse_bandwidth_totals(text: &str) -> Option<BandwidthTotals> {
    let mut totals = BandwidthTotals::default();
    let mut saw_any = false;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix(BANDWIDTH_METRIC) else {
            continue;
        };
        // `{labels} value`, and a line with no labels at all is not one of ours.
        let Some((labels, value)) = rest.rsplit_once(' ') else {
            continue;
        };
        let Ok(bytes) = value.trim().parse::<f64>() else {
            continue;
        };
        let bytes = bytes as u64;
        // The label value is upstream's `Direction` enum, rendered by
        // `EncodeLabelValue` as the variant name.
        if labels.contains("direction=\"Inbound\"") {
            totals.inbound_bytes += bytes;
            saw_any = true;
        } else if labels.contains("direction=\"Outbound\"") {
            totals.outbound_bytes += bytes;
            saw_any = true;
        }
    }
    saw_any.then_some(totals)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact text libp2p produces, taken from its own encoder: one line per
    /// protocol stack and direction, with the counter's `_total` suffix and the
    /// `libp2p` sub-registry prefix.
    ///
    /// This is a contract with a specific upstream version — a rename there
    /// leaves the figure absent rather than wrong, which `totals` says out
    /// loud — so it is pinned rather than assumed.
    const UPSTREAM_SHAPE: &str = "\
# HELP libp2p_bandwidth_bytes Bandwidth usage by direction and transport protocols.
# UNIT libp2p_bandwidth_bytes bytes
libp2p_bandwidth_bytes_total{protocols=\"/ip4/tcp\",direction=\"Inbound\"} 1024
libp2p_bandwidth_bytes_total{protocols=\"/ip4/tcp\",direction=\"Outbound\"} 2048
libp2p_bandwidth_bytes_total{protocols=\"/ip4/udp/quic-v1\",direction=\"Inbound\"} 512
libp2p_bandwidth_bytes_total{protocols=\"/ip4/udp/quic-v1\",direction=\"Outbound\"} 256
# EOF
";

    /// The exact text GossipSub's own encoder produces once its metrics are
    /// registered under our prefix: one row per (topic, direction), the counter
    /// `_total` suffix, and `hash` as the label key because `TopicHash` derives
    /// `EncodeLabelSet` over its single `hash` field.
    ///
    /// Pinned rather than assumed, for the same reason as `UPSTREAM_SHAPE`
    /// above: this is a contract with a specific upstream version, and a change
    /// there leaves the split silently absent rather than wrong.
    const GOSSIP_SHAPE: &str = "\
# HELP gossip_topic_msg_sent_bytes Bytes from gossip messages sent to each topic.
# TYPE gossip_topic_msg_sent_bytes counter
gossip_topic_msg_sent_bytes_total{hash=\"swarmllm/models\"} 825000
gossip_topic_msg_sent_bytes_total{hash=\"swarmllm/health\"} 4096
# HELP gossip_topic_msg_recv_bytes Bytes from gossip messages received per topic.
# TYPE gossip_topic_msg_recv_bytes counter
gossip_topic_msg_recv_bytes_total{hash=\"swarmllm/models\"} 1650000
gossip_topic_msg_recv_bytes_total{hash=\"swarmllm/health\"} 2048
# EOF
";

    #[test]
    fn the_gossip_counters_are_read_out_of_the_shape_gossipsub_writes() {
        let got = parse_gossip_totals(GOSSIP_SHAPE).expect("the counters must be found");
        assert_eq!(got.sent_bytes, 825_000 + 4096, "every topic is summed");
        assert_eq!(got.recv_bytes, 1_650_000 + 2048);
        assert_eq!(
            got.by_topic,
            vec![
                ("swarmllm/health".to_string(), 4096, 2048),
                ("swarmllm/models".to_string(), 825_000, 1_650_000),
            ],
            "sorted by topic, and sent/received paired per topic — the breakdown \
             is the whole point, not just the total"
        );
    }

    /// Absent, not zero, for the same reason as the transport totals: a node
    /// that has sent no gossip yet and a node whose counting was never wired
    /// must not read the same. Every node is in the first state for its first
    /// seconds, because `prometheus_client` writes a `Family`'s rows only once
    /// a label set exists (gotcha #582).
    #[test]
    fn a_registry_without_gossip_counters_reads_as_absent() {
        assert_eq!(parse_gossip_totals(""), None);
        // Registered and silent — metadata present, no rows yet.
        assert_eq!(
            parse_gossip_totals(
                "# HELP gossip_topic_msg_sent_bytes Bytes sent.\n\
                 # TYPE gossip_topic_msg_sent_bytes counter\n# EOF\n"
            ),
            None
        );
        // A renamed metric must not be silently read as zero either.
        assert_eq!(
            parse_gossip_totals("gossip_topic_bytes_out_total{hash=\"x\"} 5\n"),
            None
        );
    }

    /// An unarmed meter reports nothing, whatever is in its registry.
    #[test]
    fn an_unarmed_gossip_meter_reports_nothing() {
        assert_eq!(GossipMeter::new().totals(), None);
    }

    /// **The contract with upstream, asserted against the REAL encoder.**
    ///
    /// Everything above parses a shape written by hand, which proves the parser
    /// and proves nothing about GossipSub. This arms a real `Behaviour` and
    /// reads back what it registered: if upstream renames the metric, changes
    /// the label key from `hash`, or stops registering the family, this fails
    /// here rather than leaving the split silently absent in the field.
    ///
    /// Asserts on the METADATA lines, not on rows — a `Family` has no rows
    /// until a message is actually sent, which needs a live mesh (gotcha #582).
    #[test]
    fn a_gossip_meter_armed_by_the_real_behaviour_registers_the_names_we_parse() {
        let meter = GossipMeter::new();
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        meter.arm(|registry| {
            let behaviour: libp2p::gossipsub::Behaviour = libp2p::gossipsub::Behaviour::new(
                libp2p::gossipsub::MessageAuthenticity::Signed(keypair),
                libp2p::gossipsub::Config::default(),
            )
            .expect("gossipsub builds with its own defaults");
            let _ = behaviour.with_metrics(registry, libp2p::gossipsub::MetricsConfig::default());
        });

        let text = meter.encoded_for_test();
        for base in [
            GOSSIP_SENT_METRIC.trim_end_matches("_total"),
            GOSSIP_RECV_METRIC.trim_end_matches("_total"),
        ] {
            assert!(
                text.lines().any(|l| {
                    ["# HELP ", "# TYPE ", "# UNIT "]
                        .iter()
                        .any(|p| l.strip_prefix(p).is_some_and(|r| r.starts_with(base)))
                }),
                "gossipsub no longer registers {base} — the traffic split would go \
                 silently absent. Registry was:\n{text}"
            );
        }
        // Registered and silent is the normal state here, and must read as
        // absent rather than as a node sending no gossip.
        assert_eq!(meter.totals(), None, "no rows until a message is sent");
    }

    /// The prefix this node registers under and the names it parses are one
    /// decision. If they drift the split goes silently absent — the failure
    /// mode with no symptom.
    #[test]
    fn the_gossip_metric_names_carry_the_prefix_this_node_registers() {
        assert!(GOSSIP_SENT_METRIC.starts_with(GOSSIP_METRIC_PREFIX));
        assert!(GOSSIP_RECV_METRIC.starts_with(GOSSIP_METRIC_PREFIX));
        assert!(GOSSIP_SENT_METRIC.ends_with("_total"));
        assert!(GOSSIP_RECV_METRIC.ends_with("_total"));
    }

    /// A label we cannot read must not be dropped silently — it lands under
    /// `unknown` and still contributes to the total, because a breakdown that
    /// quietly loses bytes is worse than one that admits it cannot name them.
    #[test]
    fn a_row_with_no_readable_topic_is_still_counted() {
        let got = parse_gossip_totals("gossip_topic_msg_sent_bytes_total{other=\"x\"} 512\n")
            .expect("the row is still ours");
        assert_eq!(got.sent_bytes, 512);
        assert_eq!(got.by_topic, vec![("unknown".to_string(), 512, 0)]);
    }

    #[test]
    fn the_totals_are_read_out_of_the_shape_libp2p_writes() {
        let got = parse_bandwidth_totals(UPSTREAM_SHAPE).expect("the counters must be found");
        assert_eq!(got.inbound_bytes, 1024 + 512, "both transports are summed");
        assert_eq!(got.outbound_bytes, 2048 + 256);
    }

    /// A registry with no bandwidth counters in it must read as absent, not as
    /// a node sending nothing. Telling someone chasing their bandwidth that the
    /// answer is zero, when the truth is that nobody is counting, is worse than
    /// saying nothing.
    #[test]
    fn a_registry_without_the_counters_reads_as_absent() {
        assert_eq!(parse_bandwidth_totals(""), None);
        assert_eq!(
            parse_bandwidth_totals("swarmllm_peers_connected 7\n# EOF\n"),
            None
        );
        // The metric present but renamed — the case the warning exists for.
        assert_eq!(
            parse_bandwidth_totals("libp2p_traffic_bytes_total{direction=\"Inbound\"} 5\n"),
            None
        );
    }

    /// An unarmed meter reports nothing, whatever is in its registry.
    #[test]
    fn an_unarmed_meter_reports_nothing() {
        let meter = BandwidthMeter::new();
        assert_eq!(meter.totals(), None);
    }

    /// Arming it makes upstream register its counters, and they read back —
    /// the wiring itself, not just the parser. Without this the two halves
    /// could each be right while agreeing on nothing.
    #[test]
    fn a_meter_armed_by_the_real_transport_reads_back() {
        let meter = BandwidthMeter::new();
        // What the swarm builder does with the registry, done directly: wrap a
        // transport, which is what registers the counters.
        meter.arm(|registry| {
            let _ = libp2p::metrics::BandwidthTransport::new(
                libp2p::core::transport::MemoryTransport::default(),
                registry,
            );
        });
        // A family with no observations yet encodes no rows, so the totals are
        // legitimately absent until a byte moves. What must hold here is that
        // arming is recorded and reading does not fail.
        let _ = meter.totals();
        assert!(meter.armed.load(Ordering::Relaxed));
    }

    /// The first read of every node's life finds the family registered and
    /// empty, and that must not be reported as upstream having renamed the
    /// metric. Asserted on the MECHANISM — the one-shot warning flag — because
    /// the return value is `None` either way, so a test on the totals alone
    /// cannot tell the two apart.
    #[test]
    fn an_armed_but_silent_meter_is_not_mistaken_for_a_rename() {
        let meter = BandwidthMeter::new();
        meter.arm(|registry| {
            let _ = libp2p::metrics::BandwidthTransport::new(
                libp2p::core::transport::MemoryTransport::default(),
                registry,
            );
        });
        // Nothing has been sent, so there are no rows to read.
        assert_eq!(
            meter.totals(),
            None,
            "a family with no label set yet has nothing to total"
        );
        assert!(
            !meter.warned_absent.load(Ordering::Relaxed),
            "a registered-but-silent metric is the normal first-seconds state, \
             not a rename — warning about it fires on every node start"
        );
    }

    /// The warning still has to fire for the case it exists for: the metric
    /// genuinely absent from the registry under the name we read.
    #[test]
    fn a_metric_that_is_not_there_is_still_reported() {
        assert!(
            !metric_is_registered("# HELP libp2p_traffic_bytes Renamed upstream.\n# EOF\n"),
            "a different name must not satisfy the registration check"
        );
        assert!(
            !metric_is_registered(""),
            "an empty registry registers nothing"
        );
        // Present with no rows — the shape the real encoder produces before any
        // traffic has moved.
        assert!(metric_is_registered(
            "# HELP libp2p_bandwidth_bytes Bandwidth usage by direction and transport protocols.\n\
             # TYPE libp2p_bandwidth_bytes counter\n\
             # UNIT libp2p_bandwidth_bytes bytes\n\
             # EOF\n"
        ));
        // A reporting metric is never mistaken for a missing one either.
        assert!(metric_is_registered(UPSTREAM_SHAPE));
    }

    /// The two constants describe one metric: OpenMetrics suffixes a counter's
    /// SAMPLES with `_total` and leaves its metadata on the base name. If they
    /// drift apart the registration check silently stops matching.
    #[test]
    fn the_sample_name_is_the_metadata_name_plus_the_counter_suffix() {
        assert_eq!(
            BANDWIDTH_METRIC,
            format!("{BANDWIDTH_METRIC_BASE}_total"),
            "the row name is the metadata name plus OpenMetrics' counter suffix"
        );
    }
}
