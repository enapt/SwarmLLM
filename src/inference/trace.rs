//! Per-request routing and performance record.
//!
//! One `RequestTrace` is built per inference request and is the SOLE input to
//! every observability surface: the completion log line, the response headers,
//! the diagnostics ring, the Prometheus histograms and the dashboard. Four
//! response paths each assembling their own timing struct is precisely the
//! "one invariant, N paths" defect documented in `.claude/rules/architecture.md`
//! — and observability is the worst place for it, because the drift is
//! invisible (nothing fails, the numbers are just quietly wrong).
//!
//! **Cost.** Phase boundaries only. The one per-token operation is a relaxed
//! atomic load in [`RequestTrace::mark_first_token`], which is why TTFT can be
//! stamped from the token channel without touching the hot path that
//! `.claude/rules` protects (`pipeline.rs`, `split/executor.rs::forward`,
//! `forward_through_segments`).
//!
//! **Naming.** Durations follow OpenTelemetry's GenAI semantic conventions
//! (`time_to_first_token`, `time_per_output_token`, `request.duration`) so
//! collectors and Grafana need no translation layer. Swarm-specific dimensions
//! (route shape, segment, peer, region) have no OTel equivalent and keep
//! `swarm`-prefixed names.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::types::NodeId;

/// How a request was served. Derived from the pipeline assignment, never
/// guessed from segment count — a one-segment remote pipeline and a one-segment
/// local one are different routes, and that distinction is exactly what a user
/// asking "why was that slow" needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Route {
    /// Whole model on this node, single segment.
    #[default]
    Local,
    /// Multiple segments, all on this node (locally sharded model).
    Split,
    /// At least one segment on a remote peer.
    Distributed,
    /// At least one remote segment reached through an application-level relay.
    Relayed,
    /// Proxied to a cloud provider; no swarm segments involved.
    Cloud,
}

impl Route {
    pub fn as_str(self) -> &'static str {
        match self {
            Route::Local => "local",
            Route::Split => "split",
            Route::Distributed => "distributed",
            Route::Relayed => "relayed",
            Route::Cloud => "cloud",
        }
    }
}

/// How a segment's activations reached the node that computed them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Transport {
    /// Computed on this node — no wire involved.
    #[default]
    Local,
    /// Direct connection to the peer.
    Direct,
    /// Through an application-level relay (roughly one extra RTT each way).
    Relayed,
}

/// Terminal state of a request.
#[derive(Debug, Clone, PartialEq, Eq, Default, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Outcome {
    /// Still running. A trace left in this state was dropped without
    /// completing, which is itself worth seeing in the ring.
    #[default]
    Pending,
    Ok,
    /// `error.type` in OTel terms — the SwarmError variant name, not the
    /// message, so it stays a bounded label.
    Error(String),
    Cancelled,
}

impl Outcome {
    pub fn as_str(&self) -> &str {
        match self {
            Outcome::Pending => "pending",
            Outcome::Ok => "ok",
            Outcome::Error(_) => "error",
            Outcome::Cancelled => "cancelled",
        }
    }
}

/// One pipeline segment's contribution.
///
/// `elapsed_ms` is this segment's share of the request. In a pipeline the
/// segments are SERIALISED — every token traverses each in turn — so these sum
/// toward the total and identify the bottleneck hop. They are emphatically not
/// per-segment throughput: see `docs/FUTURE_WORK.md` § Observability on why
/// "tokens per second per node" is not directly measurable here.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SegmentTrace {
    pub index: u16,
    pub node_id: String,
    pub is_local: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<String>,
    pub layer_start: u32,
    pub layer_end: u32,
    pub shard_indices: Vec<u32>,
    pub transport: Transport,
    /// Wall time attributed to this segment, once known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub elapsed_ms: Option<u32>,
    /// Bytes of activation handed to the next hop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation_bytes: Option<u32>,
}

impl SegmentTrace {
    /// Short display id, matching what the peer list and admin UI already show.
    pub fn short_node(&self) -> &str {
        &self.node_id[..self.node_id.len().min(8)]
    }
}

#[derive(Default)]
struct TraceInner {
    t_dequeued: Option<Instant>,
    t_assembled: Option<Instant>,
    t_finished: Option<Instant>,
    route: Route,
    segments: Vec<SegmentTrace>,
    prompt_tokens: u32,
    completion_tokens: u32,
    outcome: Outcome,
}

/// The per-request record. Cheap to clone behind an `Arc`.
pub struct RequestTrace {
    pub request_id: uuid::Uuid,
    pub model: String,
    /// OTel `gen_ai.operation.name`: chat | text_completion | embeddings | responses.
    pub operation: &'static str,
    /// Origin for every duration in the trace.
    t_admitted: Instant,
    /// Microseconds from admission to the first emitted token; 0 = not yet.
    /// An atomic rather than part of `inner` so the token channel can stamp it
    /// with a relaxed load per token instead of taking a lock.
    ttft_us: AtomicU64,
    inner: Mutex<TraceInner>,
}

impl RequestTrace {
    pub fn new(request_id: uuid::Uuid, model: impl Into<String>, operation: &'static str) -> Self {
        Self {
            request_id,
            model: model.into(),
            operation,
            t_admitted: Instant::now(),
            ttft_us: AtomicU64::new(0),
            inner: Mutex::new(TraceInner::default()),
        }
    }

    /// Poisoning can only happen if a thread panicked mid-update. The trace is
    /// diagnostic; losing a field is strictly better than turning an unrelated
    /// panic into a second one on a request that would otherwise succeed.
    fn lock(&self) -> std::sync::MutexGuard<'_, TraceInner> {
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Request left the queue and began dispatch.
    pub fn mark_dequeued(&self) {
        self.lock().t_dequeued = Some(Instant::now());
    }

    /// Pipeline assembled. Records the route and the segment layout in one
    /// call so the two can never disagree.
    pub fn mark_assembled(&self, route: Route, segments: Vec<SegmentTrace>) {
        let mut g = self.lock();
        g.t_assembled = Some(Instant::now());
        g.route = route;
        g.segments = segments;
    }

    /// First token reached the client. Idempotent and lock-free; safe to call
    /// on every token, which is what makes it correct from the channel wrapper
    /// rather than from each of the seven emit sites.
    pub fn mark_first_token(&self) {
        if self.ttft_us.load(Ordering::Relaxed) != 0 {
            return;
        }
        // Clamp to >=1us so 0 keeps meaning "unset" for a token that arrives
        // within the timer's resolution.
        let us = (self.t_admitted.elapsed().as_micros() as u64).max(1);
        let _ = self
            .ttft_us
            .compare_exchange(0, us, Ordering::Relaxed, Ordering::Relaxed);
    }

    /// Attach timing to a segment already recorded by `mark_assembled`.
    pub fn record_segment_timing(&self, index: u16, elapsed_ms: u32, activation_bytes: u32) {
        let mut g = self.lock();
        if let Some(seg) = g.segments.iter_mut().find(|s| s.index == index) {
            seg.elapsed_ms = Some(elapsed_ms);
            seg.activation_bytes = Some(activation_bytes);
        }
    }

    /// Request finished. `outcome` is terminal; later calls are ignored so a
    /// cancellation racing a completion cannot rewrite history.
    pub fn mark_finished(&self, outcome: Outcome, prompt_tokens: u32, completion_tokens: u32) {
        let mut g = self.lock();
        if g.t_finished.is_some() {
            return;
        }
        g.t_finished = Some(Instant::now());
        g.outcome = outcome;
        g.prompt_tokens = prompt_tokens;
        g.completion_tokens = completion_tokens;
    }

    /// Immutable view for rendering. Taken once per surface.
    pub fn snapshot(&self) -> TraceSnapshot {
        let g = self.lock();
        let ttft_us = self.ttft_us.load(Ordering::Relaxed);
        let total_ms = g
            .t_finished
            .unwrap_or_else(Instant::now)
            .duration_since(self.t_admitted)
            .as_millis() as u64;
        let ttft_ms = (ttft_us > 0).then(|| (ttft_us as f64 / 1000.0).round() as u64);

        // Decode is everything after the first token. Without a first-token
        // stamp (non-streaming paths that never emit incrementally) there is no
        // honest split, so report None rather than pretending total == decode.
        let decode_ms = ttft_ms.map(|t| total_ms.saturating_sub(t));
        // OTel `time_per_output_token` explicitly excludes the first token.
        let tpot_ms = match (decode_ms, g.completion_tokens) {
            (Some(d), n) if n > 1 => Some(d as f64 / (n - 1) as f64),
            _ => None,
        };
        let tok_per_sec = if total_ms > 0 && g.completion_tokens > 0 {
            Some(g.completion_tokens as f64 * 1000.0 / total_ms as f64)
        } else {
            None
        };

        TraceSnapshot {
            request_id: self.request_id,
            model: self.model.clone(),
            operation: self.operation,
            route: g.route,
            queue_ms: g
                .t_dequeued
                .map(|t| t.duration_since(self.t_admitted).as_millis() as u64),
            sched_ms: match (g.t_dequeued, g.t_assembled) {
                (Some(a), Some(b)) => Some(b.duration_since(a).as_millis() as u64),
                _ => None,
            },
            ttft_ms,
            decode_ms,
            tpot_ms,
            total_ms,
            tok_per_sec,
            prompt_tokens: g.prompt_tokens,
            completion_tokens: g.completion_tokens,
            outcome: g.outcome.clone(),
            segments: g.segments.clone(),
        }
    }
}

/// A rendered, immutable view of a [`RequestTrace`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct TraceSnapshot {
    pub request_id: uuid::Uuid,
    pub model: String,
    pub operation: &'static str,
    pub route: Route,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sched_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<f64>,
    pub total_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tok_per_sec: Option<f64>,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub outcome: Outcome,
    pub segments: Vec<SegmentTrace>,
}

impl TraceSnapshot {
    /// Comma-joined short node ids in segment order, e.g. `0718d8b9,96842635`.
    pub fn nodes_csv(&self) -> String {
        self.segments
            .iter()
            .map(|s| s.short_node())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Comma-joined regions, `??` where a peer declared none (region is a
    /// voluntary field). Empty string when no segment declared one at all, so
    /// the header can be omitted entirely rather than sent as noise.
    pub fn regions_csv(&self) -> String {
        if !self.segments.iter().any(|s| s.region.is_some()) {
            return String::new();
        }
        self.segments
            .iter()
            .map(|s| s.region.as_deref().unwrap_or("??"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Number of remote hops. 0 for a purely local request.
    pub fn remote_segments(&self) -> usize {
        self.segments.iter().filter(|s| !s.is_local).count()
    }

    /// The single greppable completion line. This is what makes a log file
    /// analysable without reconstructing a route from a dozen interleaved DIAG
    /// lines across two machines.
    pub fn log_line(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::with_capacity(256);
        let _ = write!(
            s,
            "request_id={} route={} segments={} model={}",
            self.request_id,
            self.route.as_str(),
            self.segments.len(),
            self.model
        );
        if !self.segments.is_empty() {
            let _ = write!(s, " nodes={}", self.nodes_csv());
        }
        let regions = self.regions_csv();
        if !regions.is_empty() {
            let _ = write!(s, " regions={regions}");
        }
        for (k, v) in [
            ("queue_ms", self.queue_ms),
            ("sched_ms", self.sched_ms),
            ("ttft_ms", self.ttft_ms),
            ("decode_ms", self.decode_ms),
        ] {
            if let Some(v) = v {
                let _ = write!(s, " {k}={v}");
            }
        }
        let _ = write!(
            s,
            " total_ms={} prompt_tokens={} tokens={}",
            self.total_ms, self.prompt_tokens, self.completion_tokens
        );
        if let Some(t) = self.tok_per_sec {
            let _ = write!(s, " tok_per_sec={t:.1}");
        }
        if let Some(t) = self.tpot_ms {
            let _ = write!(s, " tpot_ms={t:.1}");
        }
        for seg in &self.segments {
            if let Some(ms) = seg.elapsed_ms {
                let _ = write!(s, " seg{}_ms={}", seg.index, ms);
            }
        }
        let bytes: u32 = self
            .segments
            .iter()
            .filter_map(|s| s.activation_bytes)
            .sum();
        if bytes > 0 {
            let _ = write!(s, " activation_bytes={bytes}");
        }
        let _ = write!(s, " outcome={}", self.outcome.as_str());
        if let Outcome::Error(ref kind) = self.outcome {
            let _ = write!(s, " error_type={kind}");
        }
        s
    }

    /// `Server-Timing` header value (W3C). Browsers render this natively in
    /// devtools and `PerformanceServerTiming` exposes it to JS, which is why
    /// durations go here rather than in bespoke `x-swarm-*-ms` headers.
    ///
    /// Only durations known before the body flushes are usable on a streaming
    /// response; callers pass `streaming` to omit the rest. `desc` values are
    /// restricted to `[A-Za-z0-9 _-]` so they never need quoting or escaping.
    pub fn server_timing(&self, streaming: bool) -> String {
        let mut parts: Vec<String> = Vec::with_capacity(self.segments.len() + 4);
        if let Some(v) = self.queue_ms {
            parts.push(format!("queue;dur={v}"));
        }
        if let Some(v) = self.sched_ms {
            parts.push(format!("sched;dur={v}"));
        }
        if !streaming {
            if let Some(v) = self.ttft_ms {
                parts.push(format!("ttft;dur={v}"));
            }
            if let Some(v) = self.decode_ms {
                parts.push(format!("decode;dur={v}"));
            }
            parts.push(format!("total;dur={}", self.total_ms));
            for seg in &self.segments {
                if let Some(ms) = seg.elapsed_ms {
                    parts.push(format!(
                        "seg{};dur={};desc=\"{} L{}-{}\"",
                        seg.index,
                        ms,
                        seg.short_node(),
                        seg.layer_start,
                        seg.layer_end
                    ));
                }
            }
        }
        parts.join(", ")
    }
}

/// Build the segment list from a pipeline assignment.
///
/// Lives here rather than in the router so the mapping from "what the scheduler
/// decided" to "what we report" exists once. `local_node_id` decides `is_local`;
/// `region_of` resolves a peer's declared region (voluntary, so `None` is normal).
///
/// The `relayed` flag returned by `to_parts` must mean **this hop actually goes
/// through a relay**, not "this peer supports relaying". Passing an eligibility
/// check labelled directly-connected LAN peers as relayed — see the call site in
/// `router/distributed_exec.rs`.
pub fn segments_from_assignment<'a, S, F>(
    segments: impl Iterator<Item = &'a S>,
    local_node_id: &NodeId,
    to_parts: impl Fn(&'a S) -> (NodeId, u32, u32, Vec<u32>, bool),
    region_of: F,
) -> Vec<SegmentTrace>
where
    S: 'a,
    F: Fn(&NodeId) -> Option<String>,
{
    segments
        .enumerate()
        .map(|(i, s)| {
            let (node_id, layer_start, layer_end, shard_indices, relayed) = to_parts(s);
            let is_local = &node_id == local_node_id;
            SegmentTrace {
                index: i as u16,
                region: if is_local { None } else { region_of(&node_id) },
                node_id: node_id.to_string(),
                is_local,
                layer_start,
                layer_end,
                shard_indices,
                transport: match (is_local, relayed) {
                    (true, _) => Transport::Local,
                    (false, true) => Transport::Relayed,
                    (false, false) => Transport::Direct,
                },
                elapsed_ms: None,
                activation_bytes: None,
            }
        })
        .collect()
}

/// Response headers describing how a request was served.
///
/// Returned as name/value pairs rather than written into a specific response
/// type so the OpenAI, Anthropic, Responses and MCP paths all attach the same
/// set through one function — the alternative is four hand-built header blocks
/// that drift, which is the defect `.claude/rules/architecture.md` documents.
///
/// Routing identity goes in `x-swarm-*`; durations go in the W3C
/// `Server-Timing` header, which browser devtools renders natively and
/// `PerformanceServerTiming` exposes to JS.
///
/// `streaming` MUST be true for SSE responses: headers flush before the body,
/// so time-to-first-token and decode time are not yet known and are omitted
/// rather than sent as zeros. Those ride the final SSE usage event instead.
pub fn response_headers(snap: &TraceSnapshot, streaming: bool) -> Vec<(&'static str, String)> {
    let mut out = Vec::with_capacity(5);
    out.push(("x-swarm-route", snap.route.as_str().to_string()));
    out.push(("x-swarm-segments", snap.segments.len().to_string()));
    // Remote segments, i.e. how many OTHER machines were involved. Distinct
    // from `segments`: a two-segment pipeline with one local segment used one
    // peer, and reporting "2 peers" would be wrong. Sent explicitly rather than
    // left for a client to derive, since the local/remote split is not
    // recoverable from the node list.
    out.push(("x-swarm-peers", snap.remote_segments().to_string()));
    if !snap.segments.is_empty() {
        out.push(("x-swarm-nodes", snap.nodes_csv()));
    }
    let regions = snap.regions_csv();
    if !regions.is_empty() {
        out.push(("x-swarm-regions", regions));
    }
    let timing = snap.server_timing(streaming);
    if !timing.is_empty() {
        out.push(("server-timing", timing));
    }
    out
}

/// OTel `error.type` for a failure — the variant name, never the message.
///
/// Messages carry request ids, peer ids, offsets and file paths, so using one
/// as a metric label produces unbounded cardinality and will eventually take
/// down a scrape. Variant names are a closed set.
pub fn error_kind(err: &crate::error::SwarmError) -> &'static str {
    use crate::error::SwarmError as E;
    match err {
        E::Config(_) => "Config",
        E::CreditError(_) => "CreditError",
        E::Database(_) => "Database",
        E::DecryptionFailed => "DecryptionFailed",
        E::Encryption(_) => "Encryption",
        E::Inference(_) => "Inference",
        E::InferenceTimeout(_) => "InferenceTimeout",
        E::InsufficientCapacity(_) => "InsufficientCapacity",
        E::InsufficientCredits { .. } => "InsufficientCredits",
        E::InsufficientDisk { .. } => "InsufficientDisk",
        E::Internal(_) => "Internal",
        E::InvalidNickname(_) => "InvalidNickname",
        E::InvalidSignature => "InvalidSignature",
        E::Io(_) => "Io",
        E::Keystore(_) => "Keystore",
        E::ModelNotAvailable(_) => "ModelNotAvailable",
        E::Network(_) => "Network",
        E::NoModelLoaded => "NoModelLoaded",
        E::NoSession(_) => "NoSession",
        E::NonceOverflow => "NonceOverflow",
        E::NotFound(_) => "NotFound",
        E::PeerNotFound(_) => "PeerNotFound",
        E::PipelineError(_) => "PipelineError",
        E::PrivateModeUnavailable { .. } => "PrivateModeUnavailable",
        E::ProviderError { .. } => "ProviderError",
        E::Serialization(_) => "Serialization",
        E::ServiceUnavailable(_) => "ServiceUnavailable",
        E::ShardIntegrity { .. } => "ShardIntegrity",
        E::ShardNotFound(_) => "ShardNotFound",
        E::Unauthorized(_) => "Unauthorized",
        E::Validation(_) => "Validation",
        E::VisionEncoderUnavailable(_) => "VisionEncoderUnavailable",
    }
}

/// The one-segment, all-local layout, for paths that never build a pipeline
/// assignment (the local-complete fast path). Keeps those traces the same shape
/// as routed ones so every surface renders them identically.
pub fn local_segment(node_id: &NodeId, layer_range: (u32, u32)) -> Vec<SegmentTrace> {
    vec![SegmentTrace {
        index: 0,
        node_id: node_id.to_string(),
        is_local: true,
        region: None,
        layer_start: layer_range.0,
        layer_end: layer_range.1,
        shard_indices: Vec::new(),
        transport: Transport::Local,
        elapsed_ms: None,
        activation_bytes: None,
    }]
}

/// Classify a route from its segments. Single definition so the log line, the
/// header and the UI can never disagree about what "distributed" means.
pub fn classify_route(segments: &[SegmentTrace]) -> Route {
    if segments.iter().any(|s| s.transport == Transport::Relayed) {
        return Route::Relayed;
    }
    if segments.iter().any(|s| !s.is_local) {
        return Route::Distributed;
    }
    if segments.len() > 1 {
        return Route::Split;
    }
    Route::Local
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(index: u16, node: &str, local: bool, transport: Transport) -> SegmentTrace {
        SegmentTrace {
            index,
            node_id: node.to_string(),
            is_local: local,
            region: None,
            layer_start: 0,
            layer_end: 8,
            shard_indices: vec![0],
            transport,
            elapsed_ms: None,
            activation_bytes: None,
        }
    }

    #[test]
    fn route_classification_distinguishes_local_from_one_remote_segment() {
        // The whole point of Route: both have exactly one segment.
        assert_eq!(
            classify_route(&[seg(0, "aaaa", true, Transport::Local)]),
            Route::Local
        );
        assert_eq!(
            classify_route(&[seg(0, "bbbb", false, Transport::Direct)]),
            Route::Distributed
        );
    }

    #[test]
    fn multiple_local_segments_are_split_not_distributed() {
        assert_eq!(
            classify_route(&[
                seg(0, "aaaa", true, Transport::Local),
                seg(1, "aaaa", true, Transport::Local),
            ]),
            Route::Split
        );
    }

    #[test]
    fn relay_outranks_plain_distributed() {
        assert_eq!(
            classify_route(&[
                seg(0, "aaaa", true, Transport::Local),
                seg(1, "bbbb", false, Transport::Relayed),
            ]),
            Route::Relayed
        );
    }

    #[test]
    fn first_token_stamp_is_idempotent() {
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_first_token();
        let first = t.snapshot().ttft_ms;
        std::thread::sleep(std::time::Duration::from_millis(12));
        t.mark_first_token();
        assert_eq!(
            t.snapshot().ttft_ms,
            first,
            "second stamp must not move TTFT"
        );
    }

    #[test]
    fn tpot_excludes_the_first_token() {
        // OTel time_per_output_token is defined over tokens AFTER the first.
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_first_token();
        t.mark_finished(Outcome::Ok, 10, 5);
        let s = t.snapshot();
        let decode = s.decode_ms.expect("decode known once TTFT is stamped");
        let tpot = s.tpot_ms.expect("tpot needs >1 completion token");
        assert!(
            (tpot - decode as f64 / 4.0).abs() < 1e-9,
            "divide by n-1, not n"
        );
    }

    #[test]
    fn no_first_token_stamp_means_no_invented_decode_split() {
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_finished(Outcome::Ok, 10, 5);
        let s = t.snapshot();
        assert!(s.ttft_ms.is_none());
        assert!(s.decode_ms.is_none(), "must not claim total == decode");
        assert!(s.tpot_ms.is_none());
    }

    #[test]
    fn finish_is_terminal() {
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_finished(Outcome::Ok, 1, 2);
        t.mark_finished(Outcome::Cancelled, 99, 99);
        let s = t.snapshot();
        assert_eq!(s.outcome, Outcome::Ok);
        assert_eq!(s.completion_tokens, 2);
    }

    #[test]
    fn streaming_server_timing_omits_post_body_durations() {
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_dequeued();
        t.mark_assembled(Route::Local, vec![seg(0, "aaaa", true, Transport::Local)]);
        t.mark_first_token();
        t.mark_finished(Outcome::Ok, 4, 9);
        let s = t.snapshot();

        let streaming = s.server_timing(true);
        assert!(streaming.contains("queue;dur="), "queue is known pre-body");
        assert!(
            !streaming.contains("ttft"),
            "headers flush before the body — TTFT cannot be there: {streaming}"
        );
        assert!(!streaming.contains("total"), "got: {streaming}");

        let complete = s.server_timing(false);
        assert!(complete.contains("ttft;dur="));
        assert!(complete.contains("total;dur="));
    }

    #[test]
    fn headers_omit_post_body_timings_on_a_streaming_response() {
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_dequeued();
        t.mark_assembled(
            Route::Distributed,
            vec![
                seg(0, "0718d8b987a4975a", true, Transport::Local),
                seg(1, "9684263580c6660f", false, Transport::Direct),
            ],
        );
        t.mark_first_token();
        t.mark_finished(Outcome::Ok, 4, 9);
        let s = t.snapshot();

        let h: std::collections::HashMap<_, _> = response_headers(&s, true).into_iter().collect();
        assert_eq!(h["x-swarm-route"], "distributed");
        assert_eq!(h["x-swarm-segments"], "2");
        assert_eq!(h["x-swarm-nodes"], "0718d8b9,96842635");
        // Headers flush before the body — TTFT is not known yet.
        assert!(
            !h["server-timing"].contains("ttft"),
            "{:?}",
            h["server-timing"]
        );

        let h2: std::collections::HashMap<_, _> = response_headers(&s, false).into_iter().collect();
        assert!(h2["server-timing"].contains("ttft;dur="));
    }

    #[test]
    fn header_values_are_valid_http_header_values() {
        // Guards the Server-Timing desc quoting: an invalid value would be
        // silently dropped at the axum boundary and the header would vanish.
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_assembled(
            Route::Distributed,
            vec![seg(0, "9684263580c6660f", false, Transport::Direct)],
        );
        t.record_segment_timing(0, 900, 1234);
        t.mark_finished(Outcome::Ok, 4, 9);
        for (name, value) in response_headers(&t.snapshot(), false) {
            assert!(
                axum::http::HeaderValue::from_str(&value).is_ok(),
                "{name}: {value:?} is not a valid header value"
            );
        }
    }

    #[test]
    fn log_line_carries_route_nodes_and_outcome() {
        let t = RequestTrace::new(uuid::Uuid::nil(), "llama-3.2-1b", "chat");
        t.mark_dequeued();
        t.mark_assembled(
            Route::Distributed,
            vec![
                seg(0, "0718d8b987a4975a", true, Transport::Local),
                seg(1, "9684263580c6660f", false, Transport::Direct),
            ],
        );
        t.record_segment_timing(1, 900, 39188);
        t.mark_finished(Outcome::Ok, 22, 3);
        let line = t.snapshot().log_line();
        assert!(line.contains("route=distributed"), "{line}");
        assert!(line.contains("segments=2"), "{line}");
        assert!(line.contains("nodes=0718d8b9,96842635"), "{line}");
        assert!(line.contains("seg1_ms=900"), "{line}");
        assert!(line.contains("activation_bytes=39188"), "{line}");
        assert!(line.contains("outcome=ok"), "{line}");
    }

    #[test]
    fn regions_are_omitted_entirely_when_undeclared() {
        // region is voluntary; an all-None swarm should not emit ",,,".
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_assembled(
            Route::Distributed,
            vec![seg(0, "bbbb", false, Transport::Direct)],
        );
        assert_eq!(t.snapshot().regions_csv(), "");
        assert!(!t.snapshot().log_line().contains("regions="));
    }

    #[test]
    fn error_outcome_records_a_bounded_label() {
        let t = RequestTrace::new(uuid::Uuid::nil(), "m", "chat");
        t.mark_finished(Outcome::Error("PipelineError".into()), 5, 0);
        let line = t.snapshot().log_line();
        assert!(line.contains("outcome=error"), "{line}");
        assert!(line.contains("error_type=PipelineError"), "{line}");
    }
}
