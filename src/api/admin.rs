use crate::api::server::JsonBody;
use axum::extract::State;
use axum::Json;
use serde::Deserialize;

/// Timeout for considering a peer "healthy" — peers not seen within this window
/// are marked as unhealthy in the admin dashboard.
const PEER_HEALTHY_TIMEOUT_SECS: i64 = 90;
/// Maximum length of an invite code string accepted by the join-network endpoint.
const MAX_INVITE_CODE_LEN: usize = 4096;

/// Upper bound for `max_concurrent_requests` accepted via the config API.
const MAX_CONCURRENT_REQUESTS_CAP: u32 = 256;
/// Upper bound for `max_bandwidth_mbps` accepted via the config API.
const MAX_BANDWIDTH_MBPS_CAP: u64 = 100_000;
/// Lower bound for `max_disk_mb` accepted via the config API.
const MIN_DISK_MB: u64 = 100;
/// Upper bound for `max_disk_mb` accepted via the config API.
const MAX_DISK_MB: u64 = 10_000_000;
/// Upper bound for `auto_manage_max_storage_mb` accepted via the config API.
const MAX_AUTO_MANAGE_STORAGE_MB: u64 = 10_000_000;
/// Upper bound for `batch_timeout_ms` accepted via the config API.
const MAX_BATCH_TIMEOUT_MS: u64 = 60_000;
/// Maximum bytes accepted for the `status` filter on `list_responses`.
const MAX_STATUS_FILTER_BYTES: usize = 256;

/// Serialize a peer registry entry to JSON. Used by both REST and WebSocket.
///
/// When `include_addresses` is true, includes `addresses` and `last_seen` fields
/// (REST API returns these; WebSocket omits them for bandwidth).
pub fn serialize_peer_to_json(
    peer: &crate::types::PeerInfo,
    state: &crate::daemon::state::SharedState,
    include_addresses: bool,
) -> serde_json::Value {
    let timeout = chrono::Duration::seconds(PEER_HEALTHY_TIMEOUT_SECS);
    let now = chrono::Utc::now();
    let healthy = now.signed_duration_since(peer.last_seen) < timeout;
    let hosted_shards_count = peer
        .capability
        .as_ref()
        .map(|c| c.hosted_shards.len())
        .unwrap_or(0);
    let hosted_models: Vec<String> = peer
        .capability
        .as_ref()
        .map(|c| {
            let mut models: Vec<String> = c
                .hosted_shards
                .iter()
                .map(|s| s.model_id.0.clone())
                // A peer on an older build still gossips a backup-copy name in
                // its capability; never surface it in this peer's hosted_models.
                .filter(|id| !crate::model::manifest::is_backup_artifact_id(id))
                .collect();
            models.sort_unstable();
            models.dedup();
            models
        })
        .unwrap_or_default();
    let nickname = state
        .nickname_registry
        .get(&peer.node_id)
        .map(|r| r.nickname.clone());
    // Is this peer a member of our device pool? `try_read` avoids blocking the
    // peer-list build on a contended pool_state write; on contention we treat it
    // as "not a pool member" (the frontend falls back to LAN/Remote labelling,
    // which is never wrong, just less specific).
    let is_pool_member = state
        .credits
        .pool_state
        .try_read()
        .ok()
        .and_then(|g| {
            g.as_ref()
                .map(|ps| ps.members.iter().any(|m| m.node_id == peer.node_id))
        })
        .unwrap_or(false);
    // Version + uptime are already gossiped in every NodeCapabilityUpdate and
    // are non-revealing — surfacing them makes cross-node diagnostics far easier
    // (e.g. spotting a peer still on an older build, or one that keeps
    // restarting). Absent when the peer hasn't sent a capability update yet.
    let peer_version = peer
        .capability
        .as_ref()
        .map(|c| c.version.clone())
        .filter(|v| !v.is_empty());
    let uptime_seconds = peer.capability.as_ref().map(|c| c.uptime_seconds);
    let mut obj = serde_json::json!({
        "node_id": format!("{}", peer.node_id),
        "nickname": nickname,
        "latency_ms": peer.latency_ms,
        "trust_score": peer.trust_score,
        "healthy": healthy,
        "gpu": peer.capability.as_ref().and_then(|c| c.gpu.as_ref().map(|g| &g.name)),
        "version": peer_version,
        "uptime_seconds": uptime_seconds,
        "hosted_models": hosted_models,
        "hosted_shards": hosted_shards_count,
        "is_lan_peer": peer.is_lan_peer,
        "is_pool_member": is_pool_member,
        // A dedicated bootstrap/relay node. Surfaced so the dashboard can label
        // it — an anchor holds no shards and serves no inference by design, so
        // in a peer list it is otherwise indistinguishable from a broken node.
        "is_anchor": peer
            .capability
            .as_ref()
            .map(|c| c.anchor_mode)
            .unwrap_or(false),
    });
    if include_addresses {
        if let Some(o) = obj.as_object_mut() {
            o.insert("addresses".into(), serde_json::json!(peer.addresses));
            o.insert(
                "last_seen".into(),
                serde_json::json!(peer.last_seen.to_rfc3339()),
            );
        }
    }
    obj
}

// Re-export sub-module handlers so server.rs routes continue to use `admin::handler_name`
pub use super::admin_hf::*;
pub use super::admin_models::*;
pub use super::admin_providers::*;

use crate::api::server::AppState;
use crate::config::ContributionMode;
use crate::error::ApiError;

/// GET /api/admin/swarm/capacity — collective hardware + serveable-models snapshot.
///
/// Designed for the "what can my swarm run?" dashboard header. Refreshes
/// the snapshot inline so the response always reflects the current peer
/// set (cheap — single pass over the registries). Non-technical-friendly
/// fields: every value is human-renderable without further interpretation.
pub async fn swarm_capacity(State(state): State<AppState>) -> Json<serde_json::Value> {
    crate::daemon::state::refresh_swarm_capacity(&state.shared_state);
    let snap = state.shared_state.metrics.swarm_capacity.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/wishlist — ranked list of models the swarm wants.
///
/// R111. The wishlist is the user-visible face of auto-manage: instead of
/// the daemon downloading models in mysterious silence, the user sees a
/// ranked queue with status badges and human-readable "why" tags. Refreshed
/// on demand so manual browsing always sees fresh data.
pub async fn wishlist(State(state): State<AppState>) -> Json<serde_json::Value> {
    crate::model::auto_manage::refresh_wishlist(&state.shared_state);
    let snap = state.shared_state.models.wishlist.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/quant-recommendations — per-family quant choice
/// recommendations (R133).
///
/// Groups models in the local registry by inferred base name (model name
/// with the quant suffix stripped) and surfaces the highest-quality
/// variant that fits the swarm's aggregate VRAM with reasonable
/// replication. Read-only — the recommender does NOT auto-switch which
/// quant the auto-manage system downloads. Frontend can render
/// "We're hosting Q4_K_M because the swarm only has X TB; with N more
/// nodes we'd switch to Q5_K_M."
pub async fn quant_recommendations(State(state): State<AppState>) -> Json<serde_json::Value> {
    crate::model::auto_manage::quant::refresh_quant_recommendations(&state.shared_state);
    let snap = state.shared_state.models.quant_recommendations.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/diagnostics — a pasteable plain-text snapshot for bug reports.
///
/// Exists because the two external bug reports this project has received both
/// required the reporter to dig through `journalctl` by hand and hand-copy the
/// parts that mattered. The interesting state is already in the daemon; asking
/// a user to extract it manually loses detail and wastes their time.
///
/// **Redaction is the whole design constraint.** This output is meant to be
/// pasted into a public issue or chat, so it must never carry anything that
/// grants access: no API key, no invite code, no pool secret, no file paths
/// (which leak usernames). Node and peer ids are already public on the wire, so
/// they stay — they are what makes a report traceable. Adding a field here
/// means checking it against that rule first.
/// GET /api/admin/performance — routing and performance data as JSON.
///
/// The machine-readable sibling of `diagnostics`, for the dashboard. Pulled on
/// demand rather than pushed on the 2s WebSocket stats tick: the per-peer and
/// per-request detail here is exactly the high-cardinality data that must NOT
/// go anywhere retained (see `docs/FUTURE_WORK.md` § Observability), and a
/// panel nobody has open should cost nothing.
pub async fn performance(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    use std::sync::atomic::Ordering;
    let ss = &state.shared_state;
    let m = &ss.metrics;

    // Newest first — matches the text rendering and what a reader expects.
    let mut recent = ss.recent_traces_snapshot();
    recent.reverse();

    let layers = m.layers_served.load(Ordering::Relaxed);
    let micros = m.segment_serve_micros.load(Ordering::Relaxed);

    // Hourly trend, oldest first. Aggregates only, and the series is capped at
    // a week, so this stays a few KB regardless of traffic.
    let hourly: Vec<serde_json::Value> = ss
        .perf_history
        .snapshot()
        .into_iter()
        .map(|b| {
            serde_json::json!({
                "hour_start_ms": b.hour_start_ms,
                "requests": b.requests,
                "errors": b.errors,
                "avg_total_ms": b.avg_total_ms(),
                "avg_ttft_ms": b.avg_ttft_ms(),
                "avg_tok_per_sec": b.avg_tok_per_sec(),
                "by_route": b.by_route,
            })
        })
        .collect();

    axum::Json(serde_json::json!({
        "recent": recent,
        "active": ss.active_request_rows(),
        "hourly": hourly,
        "peers": ss.peer_performance_rows(),
        "served": {
            "segments": m.segments_served.load(Ordering::Relaxed),
            "layers": layers,
            "compute_secs": micros as f64 / 1_000_000.0,
            "activation_bytes_out": m.segment_bytes_out.load(Ordering::Relaxed),
            // The comparable figure: what a peer's scheduler ranks us on.
            "ms_per_layer": if layers > 0 {
                Some((micros as f64 / 1000.0) / layers as f64)
            } else {
                None
            },
        },
    }))
}

pub async fn diagnostics(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    use std::fmt::Write as _;
    let ss = &state.shared_state;
    let mut out = String::new();

    let _ = writeln!(out, "SwarmLLM diagnostics");
    let _ = writeln!(out, "version: {}", env!("CARGO_PKG_VERSION"));
    let _ = writeln!(out, "node:    {}", ss.identity.node_id());

    {
        let stats = ss.metrics.node_stats.read().await;
        let up = (chrono::Utc::now() - stats.uptime_start)
            .num_seconds()
            .max(0);
        let _ = writeln!(out, "uptime:  {}h{}m", up / 3600, (up % 3600) / 60);
        let _ = writeln!(
            out,
            "nat:     {}",
            stats.nat_status.as_deref().unwrap_or("unknown")
        );
        let _ = writeln!(out, "requests: {}", stats.requests_made);
    }

    let _ = writeln!(out, "anchor_mode: {}", ss.config.node.anchor_mode);
    // In-flight request bookkeeping, side by side because the interesting case
    // is when they disagree with an idle node.
    //
    // `active_traces` is the oracle behind `model_is_in_use` — the guard that
    // decides whether a model's files may be deleted. It is removed by RAII on
    // every completion path and has no cap and no sweep behind that, so a
    // single leaked entry would refuse deletion **forever**, and the user would
    // see a 503 saying the model is busy on a node serving nobody, with nothing
    // anywhere to explain it. Nothing exposed the count until now, so that
    // question had no answer.
    //
    // Expect both zero on an idle node. Non-zero with no traffic is the bug.
    let _ = writeln!(
        out,
        "in_flight: {} traces, {} pipelines",
        ss.active_traces.len(),
        ss.active_pipelines.len()
    );
    let _ = writeln!(
        out,
        "tensor_parallel: {}  gpu_layers: {}",
        ss.config.inference.tensor_parallel, ss.config.inference.gpu_layers
    );
    // Read the runtime atomic, not the startup-frozen config. Turning
    // auto-manage off is a live toggle (`PUT /api/admin/config`), so printing
    // the config value reported it as still ON after it had been switched off —
    // which is exactly the wrong answer for a diagnostics page someone consults
    // when trying to work out why shards are moving on their own.
    let _ = writeln!(
        out,
        "auto_manage: {}  shard_size_mb: {}",
        ss.models
            .auto_manage_enabled
            .load(std::sync::atomic::Ordering::Relaxed),
        ss.config.model.shard_size_mb
    );

    // How this node is reachable. An anchor advertising only private
    // addresses looks healthy from every other angle, so this belongs next to
    // the NAT status rather than behind a separate endpoint.
    {
        let addrs = ss.listen_multiaddrs.load();
        let _ = writeln!(out, "\n-- reachable at ({}) --", addrs.len());
        for a in addrs.iter().take(10) {
            let _ = writeln!(out, "  {a}");
        }
        if addrs.is_empty() {
            let _ = writeln!(out, "  (none — no invite code can be minted)");
        }
    }

    let _ = writeln!(out, "\n-- peers ({}) --", ss.connected_node_ids.len());
    for entry in ss.connected_node_ids.iter().take(20) {
        let _ = writeln!(out, "  {}", entry.key());
    }

    // Recent inference failures. The first thing to look at when a user says
    // "inference doesn't work", and previously unavailable without asking them
    // to re-run with -v and reproduce.
    {
        let failures = ss.recent_failures_snapshot();
        let _ = writeln!(
            out,
            "\n-- recent inference failures ({}) --",
            failures.len()
        );
        if failures.is_empty() {
            let _ = writeln!(out, "  (none since start)");
        }
        // Newest first — that is what someone debugging just-now wants.
        for f in failures.iter().rev() {
            let where_ = match &f.served_by {
                Some(peer) => format!("peer {}", &peer[..peer.len().min(16)]),
                None => "locally".to_string(),
            };
            let _ = writeln!(
                out,
                "  {} {} [{}] served {} after {}ms\n      {}",
                f.at.format("%H:%M:%SZ"),
                &f.request_id[..f.request_id.len().min(8)],
                f.model,
                where_,
                f.elapsed_ms,
                f.error
            );
        }
        // Repeated failures against one peer is the signature we have
        // historically taken rounds to spot; count it explicitly.
        let mut per_peer: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for f in &failures {
            if let Some(p) = &f.served_by {
                *per_peer.entry(p.as_str()).or_default() += 1;
            }
        }
        let mut worst: Vec<_> = per_peer.into_iter().filter(|(_, n)| *n >= 2).collect();
        worst.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
        for (peer, n) in worst {
            let _ = writeln!(
                out,
                "  NOTE: {n} of these were served by peer {} — suspect that peer, not this node",
                &peer[..peer.len().min(16)]
            );
        }
    }

    // Completed requests. The successful sibling of the failure ring above:
    // "why was that slow" needs the route and the per-phase split, and
    // reconstructing those from interleaved DIAG lines across two machines is
    // what has repeatedly cost hours.
    {
        let traces = ss.recent_traces_snapshot();
        let _ = writeln!(out, "\n-- recent requests ({}) --", traces.len());
        if traces.is_empty() {
            let _ = writeln!(out, "  (none since start)");
        }
        // Newest first — the request someone is asking about just now.
        for t in traces.iter().rev().take(15) {
            let ttft = t
                .ttft_ms
                .map(|v| format!("{v}ms"))
                .unwrap_or_else(|| "-".into());
            let rate = t
                .tok_per_sec
                .map(|v| format!("{v:.1} tok/s"))
                .unwrap_or_else(|| "-".into());
            let _ = writeln!(
                out,
                "  {} [{}] {} {}  total={}ms ttft={} {}  {}",
                &t.request_id.to_string()[..8],
                t.model,
                t.route.as_str(),
                if t.segments.len() > 1 {
                    format!("x{}", t.segments.len())
                } else {
                    String::new()
                },
                t.total_ms,
                ttft,
                rate,
                t.outcome.as_str(),
            );
            // Only worth a second line when the work actually left this node.
            if t.remote_segments() > 0 {
                for s in &t.segments {
                    let ms = s
                        .elapsed_ms
                        .map(|v| format!("{v}ms"))
                        .unwrap_or_else(|| "-".into());
                    let _ = writeln!(
                        out,
                        "      seg{} {} L{}-{} {} {}{}",
                        s.index,
                        if s.is_local { "local" } else { s.short_node() },
                        s.layer_start,
                        s.layer_end,
                        ms,
                        match s.transport {
                            crate::inference::trace::Transport::Relayed => "relayed",
                            crate::inference::trace::Transport::Direct => "direct",
                            crate::inference::trace::Transport::Local => "",
                        },
                        s.region
                            .as_deref()
                            .map(|r| format!(" {r}"))
                            .unwrap_or_default(),
                    );
                }
            }
        }
    }

    // What this node did FOR the swarm. Distinct from everything above, which
    // is all about requests this node made.
    {
        use std::sync::atomic::Ordering;
        let m = &ss.metrics;
        let segments = m.segments_served.load(Ordering::Relaxed);
        let layers = m.layers_served.load(Ordering::Relaxed);
        let micros = m.segment_serve_micros.load(Ordering::Relaxed);
        let bytes = m.segment_bytes_out.load(Ordering::Relaxed);
        let _ = writeln!(out, "\n-- served for others --");
        if segments == 0 {
            let _ = writeln!(out, "  (no segments served yet)");
        } else {
            let _ = writeln!(
                out,
                "  segments={segments} layers={layers} compute={:.1}s activations_out={:.1}MB",
                micros as f64 / 1_000_000.0,
                bytes as f64 / (1024.0 * 1024.0),
            );
            // Per-layer cost is the comparable figure: it is what a peer's
            // scheduler uses to rank us, so it is the number to watch.
            let _ = writeln!(
                out,
                "  {:.1} ms per layer served",
                (micros as f64 / 1000.0) / layers.max(1) as f64
            );
        }
    }

    // Per-peer serving performance. `hedge_tracker` has carried EWMA latency
    // with variance per (model, segment, holder) since R136 and nothing could
    // read it; this is the first surface that does. Answers "which peer is
    // dragging the pipeline" directly instead of by inference from failures.
    {
        let rows = ss.peer_performance_rows();
        let _ = writeln!(out, "\n-- peer serving performance ({}) --", rows.len());
        if rows.is_empty() {
            let _ = writeln!(out, "  (no remote segments served yet)");
        } else {
            let _ = writeln!(
                out,
                "  {:<18} {:>6} {:>9} {:>9} {:>7}  region",
                "peer", "rtt", "ms/layer", "ewma", "samples"
            );
        }
        for r in rows {
            let _ = writeln!(
                out,
                "  {:<18} {:>6} {:>9} {:>9} {:>7}  {}",
                &r.node_id[..r.node_id.len().min(16)],
                r.rtt_ms
                    .map(|v| format!("{v}ms"))
                    .unwrap_or_else(|| "-".into()),
                r.ms_per_layer
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "-".into()),
                r.ewma_ms
                    .map(|v| format!("{v:.0}ms"))
                    .unwrap_or_else(|| "-".into()),
                r.samples,
                r.region.as_deref().unwrap_or("-"),
            );
        }
    }

    // NAT traversal. "Is this node stuck behind the relay?" is the first
    // question worth asking when remote inference is slow or failing, and it
    // used to be unanswerable — DCUtR emitted no logs at all.
    {
        use std::sync::atomic::Ordering;
        let ok = ss.hole_punch_successes.load(Ordering::Relaxed);
        let failed = ss.hole_punch_failures.load(Ordering::Relaxed);
        let public = ss.publicly_reachable.load(Ordering::Relaxed);
        let _ = writeln!(out, "\n-- NAT traversal --");
        let _ = writeln!(out, "  publicly reachable: {public}");
        let _ = writeln!(
            out,
            "  donating relay capacity: {}",
            ss.relay_forwarding_enabled()
        );
        let _ = writeln!(out, "  hole punches: {ok} succeeded / {failed} failed");
        // The reading of "0 attempts" depends entirely on whether this node is
        // public, so say which case applies rather than listing possibilities.
        // A public node NEVER needs to hole-punch — peers dial it directly — so
        // 0/0 there is the healthy steady state, not a pending result.
        if public && ok == 0 && failed == 0 {
            let _ = writeln!(
                out,
                "  (none needed — this node is reachable directly, so peers dial \
                 it without hole punching)"
            );
        } else if ok == 0 && failed == 0 {
            let _ = writeln!(
                out,
                "  (no attempts yet — nothing to punch until a peer is seen that \
                 can only be reached through a relay)"
            );
        } else if ok == 0 && !ss.relay_routes.is_empty() {
            let _ = writeln!(
                out,
                "  (never escaped the relay — expected behind symmetric NAT/CGNAT; \
                 traffic still flows, with one extra hop)"
            );
        } else if ok == 0 {
            // Attempts failed, but nothing is currently being routed through a
            // relay — so the failures are historical, not the present state.
            // Reporting "never escaped the relay" here would be actively
            // misleading: the peer in question is very likely reachable
            // directly now, which is the outcome we wanted.
            let _ = writeln!(
                out,
                "  ({failed} attempt(s) failed, but nothing is currently routed \
                 through a relay — those peers are reachable directly now)"
            );
        }
        // A peer needs this same version for a hole punch to complete, since the
        // connection limit that used to block it applies to both ends. Say so,
        // because "0 succeeded" against older peers is expected, not a fault.
        if failed > 0 && ok == 0 {
            let _ = writeln!(
                out,
                "  note: both ends need v0.3.21 or newer for a direct connection \
                 to form"
            );
        }
    }

    // Remembered peer addresses, raw vs. what is actually worth dialling.
    // A gap between the two is the signature of a cache holding entries that
    // can never connect — other peers' loopback and private addresses, or a
    // relay circuit through our own id. Reported as both numbers because the
    // raw count alone cannot distinguish "cache is clean" from "filter is
    // silently doing nothing".
    {
        let cached = crate::network::peer_cache::load_peer_cache(&ss.db);
        match crate::network::transport::node_id_to_peer_id(ss.identity.node_id()) {
            Some(me) => {
                let local_addrs = ss.listen_multiaddrs.load();
                let dialable =
                    crate::network::peer_cache::filter_dialable(&cached, &me, &local_addrs);
                let _ = writeln!(
                    out,
                    "\n-- peer cache --\n  {} stored, {} dialable",
                    cached.len(),
                    dialable.len()
                );
                for a in dialable.iter().take(10) {
                    let _ = writeln!(out, "  {a}");
                }
            }
            None => {
                let _ = writeln!(
                    out,
                    "\n-- peer cache --\n  {} stored (could not derive local peer id)",
                    cached.len()
                );
            }
        }
    }

    let _ = writeln!(out, "\n-- models --");
    let local = ss.identity.node_id().clone();
    for m in ss.model_registry.models() {
        let held: Vec<u32> = ss.model_registry.local_shard_indices_in(&m, &local);
        let tag = if crate::model::reference::is_reference_model(&m.id.0) {
            " [reference]"
        } else {
            ""
        };
        let _ = writeln!(
            out,
            "  {}{}: {}/{} shards local {:?}, {} layers",
            m.id.0,
            tag,
            held.len(),
            m.shard_count,
            held,
            m.num_layers
        );
    }

    {
        let bal = ss.credits.credit_balance.read().await;
        let _ = writeln!(out, "\n-- credits --\n  balance: {}", bal.balance);
    }

    // Recent events carry the failure that prompted the report. English
    // messages are intentional: a maintainer reading a pasted report should not
    // have to reverse a translation.
    let _ = writeln!(out, "\n-- recent activity --");
    {
        let history = ss.events.activity_history.lock();
        for ev in history.iter().rev().take(30) {
            let _ = writeln!(out, "  [{}/{}] {}", ev.category, ev.kind, ev.message);
        }
    }

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; charset=utf-8",
        )],
        out,
    )
}

/// GET /api/admin/reference-models — the pinned models used for benchmarking
/// and network testing (`docs/REFERENCE_MODELS.md`).
///
/// Read-only. Acquiring one goes through the normal
/// `POST /api/admin/hf/download-shards` path like any other model — there is
/// no special fetch, and nothing here is downloaded unless a user asks for it.
///
/// `held` reports whether this node already has the model, so the dashboard can
/// distinguish "offer this" from "you have this" without a second round trip.
pub async fn reference_models(State(state): State<AppState>) -> Json<serde_json::Value> {
    let entries: Vec<serde_json::Value> = crate::model::reference::REFERENCE_MODELS
        .iter()
        .map(|m| {
            // "Held" has to mean this node actually hosts shards, not merely
            // that it has heard of the model. A manifest arrives by gossip the
            // moment any peer announces the model, so checking for one marked
            // every reference model as installed on a node holding none of it.
            let local = state.shared_state.identity.node_id();
            let held = state
                .shared_state
                .model_registry
                .get_manifest(&crate::types::ModelId(m.model_id.to_string()))
                .map(|manifest| {
                    !state
                        .shared_state
                        .model_registry
                        .local_shard_indices_in(&manifest, local)
                        .is_empty()
                })
                .unwrap_or(false);
            serde_json::json!({
                "tier": m.tier,
                "model_id": m.model_id,
                "repo_id": m.repo_id,
                "filename": m.filename,
                "size_mb": m.size_mb,
                "shards": m.shards,
                "held": held,
            })
        })
        .collect();
    Json(serde_json::json!({ "models": entries }))
}

/// GET /api/admin/foreign-pool-catalog — R134 — cross-pool model
/// availability discovery surface. Returns the cached signals from
/// `PoolModelAvailability` gossip, grouped by pool, with stale entries
/// trimmed against `FOREIGN_POOL_CATALOG_MAX_AGE_MS`. Pure discovery —
/// does NOT bind routing decisions. Useful for the admin UI's
/// "models the swarm knows about but this pool doesn't host yet" tile.
pub async fn foreign_pool_catalog(State(state): State<AppState>) -> Json<serde_json::Value> {
    use std::collections::BTreeMap;
    let now_ms = crate::types::unix_now_ms();
    state.shared_state.credits.trim_stale_foreign_pool_catalog(
        now_ms,
        crate::daemon::dispatch::FOREIGN_POOL_CATALOG_MAX_AGE_MS,
    );
    // Group by pool_id for the response shape.
    let mut by_pool: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
    for entry in state.shared_state.credits.foreign_pool_catalog.iter() {
        let (pool, model) = entry.key();
        by_pool
            .entry(format!("{pool}"))
            .or_default()
            .push(serde_json::json!({
                "model_id": model.0,
                "received_at_ms": *entry.value(),
            }));
    }
    let pools: Vec<serde_json::Value> = by_pool
        .into_iter()
        .map(|(pool_id, models)| serde_json::json!({"pool_id": pool_id, "models": models}))
        .collect();
    Json(serde_json::json!({
        "pools": pools,
        "computed_at_ms": now_ms,
    }))
}

/// GET /api/admin/hf/trending — latest HuggingFace trending-GGUF snapshot
/// captured by the background HfWatcher (R112). Surfaces the same data the
/// wishlist scorer consumes so the frontend can render a "trending now"
/// view without re-querying HF.
pub async fn hf_trending(State(state): State<AppState>) -> Json<serde_json::Value> {
    let snap = state.shared_state.models.hf_trending_cache.load_full();
    Json(serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/swarm/capacity-plan — what-if scenarios.
///
/// R113. Drives the dashboard's "if N more contributors joined with X GB
/// each, you'd unlock Y" message — the educational layer that turns the
/// product's value prop ("contribute and run huge models together") into
/// a concrete next step. Three baked scenarios (small/medium/large) +
/// a headline_target showing the closest aspirational upgrade.
pub async fn swarm_capacity_plan(State(state): State<AppState>) -> Json<serde_json::Value> {
    let plan = crate::daemon::state::compute_capacity_plan(&state.shared_state);
    Json(serde_json::to_value(&plan).unwrap_or_else(|_| serde_json::json!({})))
}

/// GET /api/admin/storage/breakdown — disk allocation summary for the
/// stacked-bar UI. Replaces the dual "Max Disk" / "Max Auto-Download Storage"
/// settings with a single bar showing total / used / auto-manage-budget /
/// free. R110.
///
/// Numbers are pre-converted to MB so the frontend doesn't have to handle
/// byte→MB rounding (avoids `49.99 GB` rendering when user typed `50 GB`).
pub async fn storage_breakdown(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config = state.config.clone();
    let local_node_id = state.shared_state.identity.node_id().clone();
    let mgr = &state.shared_state.models;

    // Bytes currently held on disk by this node.
    let mut used_bytes: u64 = 0;
    let mut held_shards: u32 = 0;
    for entry in state.shared_state.model_registry.models() {
        for shard in &entry.shards {
            let sid = crate::types::ShardId {
                model_id: entry.id.clone(),
                index: shard.index,
            };
            let holders = state.shared_state.model_registry.shard_holders(&sid);
            if holders.contains(&local_node_id) {
                used_bytes = used_bytes.saturating_add(shard.size_bytes);
                held_shards += 1;
            }
        }
    }

    // What auto-manage will try to grow to. Shared with the scheduler
    // via `model::auto_manage::compute_budget_max_bytes` so the two
    // can't drift if the ContributionMode scaling changes.
    let live = state.shared_state.cfg();
    let auto_target_bytes = crate::model::auto_manage::compute_budget_max_bytes(
        live.auto_manage.max_storage_mb,
        live.resources.max_disk_mb,
        // Live level: `scoring.rs` sizes the real budget the same way, so
        // reading the boot-time config here would show the user a storage
        // target the scheduler is no longer working towards.
        &state.shared_state.contribution(),
        crate::model::auto_manage::free_disk_bytes_for(&config.node.data_dir),
    );
    let total_bytes = live
        .resources
        .max_disk_mb
        .saturating_mul(1024)
        .saturating_mul(1024);
    let auto_target_capped = auto_target_bytes.min(total_bytes);
    // Free = max(0, total - used). When used > total (rare — happens if
    // user shrinks Max Disk after already having more on disk), we report
    // 0 free and let the UI show a "you're over your budget" hint.
    let free_bytes = total_bytes.saturating_sub(used_bytes);

    let auto_enabled = mgr
        .auto_manage_enabled
        .load(std::sync::atomic::Ordering::Relaxed);

    Json(serde_json::json!({
        // All values in MB to match the slider units; UI converts to GB
        // for display. Single-source-of-truth: never present "max_disk_mb"
        // and "auto_manage_max_storage_mb" as independent inputs again.
        "total_mb": total_bytes / (1024 * 1024),
        "used_mb": used_bytes / (1024 * 1024),
        "free_mb": free_bytes / (1024 * 1024),
        "auto_target_mb": auto_target_capped / (1024 * 1024),
        "held_shards": held_shards,
        "auto_manage_enabled": auto_enabled,
        "contribution": match config.node.contribution {
            ContributionMode::Minimal => "minimal",
            ContributionMode::Moderate => "moderate",
            ContributionMode::Maximum => "maximum",
        },
    }))
}

/// GET /api/admin/stats — Full dashboard stats snapshot.
pub async fn stats(State(state): State<AppState>) -> Json<serde_json::Value> {
    let node_id = hex::encode(state.shared_state.identity.node_id().0);

    // Snapshot the locked values into stack copies and drop the guards BEFORE
    // the sysinfo spawn_blocking await. Holding RwLock guards across the
    // blocking-call .await would otherwise park concurrent writers
    // (apply_credit on the inference hot path, the health monitor) for the
    // duration of the /proc/* scan.
    let (uptime_start, requests_made) = {
        let stats = state.shared_state.metrics.node_stats.read().await;
        (stats.uptime_start, stats.requests_made)
    };
    let (tier, credit_json) = {
        let credit = state.shared_state.credits.credit_balance.read().await;
        (
            crate::credit::priority::PriorityCalculator::tier_name(credit.balance),
            super::credit_summary_json(&credit),
        )
    };
    let uptime_seconds = (chrono::Utc::now() - uptime_start).num_seconds().max(0) as u64;

    // Count only shards held locally (not all tracked shards network-wide)
    let hosted_shards = crate::api::metrics::count_local_shards(&state.shared_state);

    // Hardware detection — sysinfo does blocking filesystem reads (/proc/*)
    let ss = state.shared_state.clone();
    let hardware = tokio::task::spawn_blocking(move || detect_hardware(&ss))
        .await
        .unwrap_or_else(|_| serde_json::json!({}));

    // Inference performance metrics from latency samples
    let inference_perf = match crate::api::metrics::compute_latency_stats(&state.shared_state) {
        Some(ls) => serde_json::json!({
            "total_requests": ls.total_requests,
            "avg_latency_ms": ls.avg_ms,
            "min_latency_ms": ls.min_ms,
            "max_latency_ms": ls.max_ms,
            "p50_latency_ms": ls.p50_ms,
            "p95_latency_ms": ls.p95_ms,
            "p99_latency_ms": ls.p99_ms,
            "samples": ls.count,
        }),
        None => serde_json::json!({
            "total_requests": state.shared_state.metrics.inference_requests_total
                .load(std::sync::atomic::Ordering::Relaxed),
            "avg_latency_ms": null,
            "samples": 0,
        }),
    };

    // SWARM-SPEC layer metrics (R136): hedge + prefetch tracker
    // snapshots. Empty / zero counters until those layers see real
    // traffic with their feature flags enabled.
    // R137: + L1 n-gram hit/miss lifetime counters so operators can
    // tell whether the cascade is actually firing on their workload mix.
    let ngram_hits = state
        .shared_state
        .metrics
        .ngram_hits
        .load(std::sync::atomic::Ordering::Relaxed);
    let ngram_misses = state
        .shared_state
        .metrics
        .ngram_misses
        .load(std::sync::atomic::Ordering::Relaxed);
    let ngram_total = ngram_hits + ngram_misses;
    let ngram_hit_rate = if ngram_total > 0 {
        ngram_hits as f64 / ngram_total as f64
    } else {
        0.0
    };
    let swarm_spec_metrics = serde_json::json!({
        "hedge": state.shared_state.metrics.hedge_tracker.metrics(),
        "prefetch": state.shared_state.metrics.prefetch_orchestrator.metrics(),
        "ngram": {
            "hits": ngram_hits,
            "misses": ngram_misses,
            "total": ngram_total,
            "hit_rate": (ngram_hit_rate * 10000.0).round() / 10000.0,
        },
    });

    Json(serde_json::json!({
        "node_id": node_id,
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_seconds": uptime_seconds,
        "tier": tier,
        "peers": state.shared_state.connected_node_ids.len(),
        "requests_served": state.shared_state.metrics.requests_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        "forwards_served": state.shared_state.metrics.forwards_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        "requests_made": requests_made,
        "active_requests": state.shared_state.active_pipelines.len(),
        "hosted_shards": hosted_shards,
        "credits": credit_json,
        "hardware": hardware,
        "inference": inference_perf,
        "swarm_spec": swarm_spec_metrics,
    }))
}

/// GET /api/admin/config — Return current configuration.
///
/// Reads the persisted config from disk so this always reflects the latest saved
/// values (including those applied by `PUT /api/admin/config`). Falls back to the
/// in-memory startup config if the file cannot be read.
pub async fn get_config(State(state): State<AppState>) -> Json<serde_json::Value> {
    let config_path = state.config.node.data_dir.join("config.toml");
    // Hot-path: dashboard polls this. Read off the async runtime so the
    // sync read doesn't block a Tokio worker (R98). update_config at line
    // ~296 already wraps writes in spawn_blocking — match that pattern.
    let config = tokio::task::spawn_blocking(move || std::fs::read_to_string(&config_path))
        .await
        .ok()
        .and_then(|r| r.ok())
        .and_then(|s| toml::from_str::<crate::config::Config>(&s).ok())
        .unwrap_or_else(|| state.config.clone());
    let config = &config;
    let contribution = match config.node.contribution {
        ContributionMode::Minimal => "minimal",
        ContributionMode::Moderate => "moderate",
        ContributionMode::Maximum => "maximum",
    };
    // Include claude_subscription config if the feature is enabled
    #[cfg(feature = "claude-subscription")]
    let claude_sub = {
        let providers = state.shared_state.metrics.providers_config.try_read();
        providers.ok().and_then(|p| {
            p.claude_subscription.as_ref().map(|s| {
                serde_json::json!({
                    "enabled": s.enabled,
                })
            })
        })
    };
    #[cfg(not(feature = "claude-subscription"))]
    let claude_sub: Option<serde_json::Value> = None;

    // Provider configuration, so a node whose cloud access disappears has
    // something to look at. NEVER the key — name, whether it is configured, and
    // where it came from.
    //
    // Reported 2026-08-05: a node went from 102 cloud models to zero with no
    // error and no log line, and this endpoint exposed nothing provider-adjacent
    // at all. `provider-health` lists only providers that ARE configured, so at
    // zero it returns an empty array — the same non-information as
    // `cloud_models: []`. `source` is what actually diagnoses it: a key that
    // came from the environment vanishes when a restart does not inherit it,
    // which from outside looks identical to one that was never set.
    let providers_summary: Vec<serde_json::Value> = state
        .shared_state
        .metrics
        .providers_config
        .try_read()
        .map(|p| {
            p.configured_summary()
                .into_iter()
                .map(|(name, configured, source)| {
                    serde_json::json!({
                        "name": name,
                        "configured": configured,
                        "key_source": source,
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let mut result = serde_json::json!({
        "providers": providers_summary,
        "contribution": contribution,
        "contribution_auto": config.node.contribution_auto,
        "max_concurrent_requests": config.inference.max_concurrent_requests,
        "max_bandwidth_mbps": config.resources.max_bandwidth_mbps,
        "max_disk_mb": config.resources.max_disk_mb,
        "max_gpu_vram_mb": config.resources.max_gpu_vram_mb,
        "listen_port": config.node.listen_port,
        "session_timeout_seconds": config.inference.session_timeout_seconds,
        "auto_manage_shards": state.shared_state.models.auto_manage_enabled.load(std::sync::atomic::Ordering::Relaxed),
        "auto_manage_max_storage_mb": config.auto_manage.max_storage_mb,
        "shard_size_mb": config.model.shard_size_mb,
        "max_batch_size": config.inference.max_batch_size,
        "batch_timeout_ms": config.inference.batch_timeout_ms,
        // R137: surface the runtime values (not the startup-frozen config)
        // so the dashboard reflects post-PUT state immediately.
        "allow_cross_pool_inference": state.shared_state.cfg().pool.allow_cross_pool_inference,
        "share_model_catalog": state.shared_state.cfg().pool.share_model_catalog,
        // Runtime value, same reason. Paired with `dashboard_on_overlay` so the
        // settings panel can say whether the overlay path is actually in play
        // on this node rather than just offering an abstract switch.
        "dashboard_trust_lan": state.shared_state.cfg().api.dashboard_trust_lan,
        "dashboard_trust_overlay": config.api.dashboard_trust_overlay,
        "dashboard_on_overlay": crate::api::dashboard_trust::node_is_on_overlay(&state.shared_state),
        // Effective, not raw: a config predating the `mode` key reports what it
        // actually does rather than a null the settings panel can't render.
        "update_mode": match config.updates.effective_mode() {
            crate::config::UpdateMode::Off => "off",
            crate::config::UpdateMode::Notify => "notify",
            crate::config::UpdateMode::Download => "download",
            crate::config::UpdateMode::Install => "install",
        },
    });
    if let Some(cs) = claude_sub {
        result["claude_subscription"] = cs;
    }
    Json(result)
}

/// PUT /api/admin/config — Update configuration at runtime.
pub async fn update_config(
    State(state): State<AppState>,
    JsonBody(body): JsonBody<ConfigUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Persist the updated config to the config TOML file.
    // Note: most config changes take effect after daemon restart.
    let config_path = state.config.node.data_dir.join("config.toml");

    // Build a partial config update
    let mut config = state.config.clone();

    if let Some(contribution) = &body.contribution {
        let mode = match contribution.as_str() {
            "minimal" => ContributionMode::Minimal,
            "moderate" => ContributionMode::Moderate,
            "maximum" => ContributionMode::Maximum,
            other => {
                return Err(ApiError(crate::error::SwarmError::Validation(format!(
                    "Unknown contribution mode '{other}' (expected: minimal, moderate, maximum)"
                ))));
            }
        };
        config.node.contribution = mode.clone();
        // The level itself goes live with the whole config at the end of this
        // handler. What needs doing *here* is the part re-reading cannot do:

        // Inference thread count is handed to a worker when it spawns, so
        // re-deriving it here means the next worker picks up the new level.
        // Workers already running keep the pool they were given — recycling a
        // live worker would drop whatever it is currently answering.
        let (physical_cores, logical_cores) = crate::config::ResourceConfig::detect_cpu_topology();
        let threads =
            config
                .resources
                .inference_cpu_threads(physical_cores, logical_cores, mode.clone());
        state
            .shared_state
            .model_process_pool
            .set_cpu_threads(threads);
        tracing::info!(
            contribution = ?mode,
            inference_cpu_threads = threads,
            "Contribution level changed — applied to the running daemon"
        );
    }
    if let Some(auto) = body.contribution_auto {
        config.node.contribution_auto = auto;
    }
    if let Some(max_reqs) = body.max_concurrent_requests {
        config.inference.max_concurrent_requests = max_reqs.clamp(1, MAX_CONCURRENT_REQUESTS_CAP);
    }
    if let Some(bw) = body.max_bandwidth_mbps {
        config.resources.max_bandwidth_mbps = bw.clamp(1, MAX_BANDWIDTH_MBPS_CAP);
    }
    if let Some(disk) = body.max_disk_mb {
        config.resources.max_disk_mb = disk.clamp(MIN_DISK_MB, MAX_DISK_MB);
    }
    if let Some(vram) = body.max_gpu_vram_mb {
        // 0 = auto (80% of detected VRAM). Cap at 1 TB so a stray UI
        // value can't disable VRAM accounting entirely on the dashboard
        // side; the inference path will still honor whatever this is.
        config.resources.max_gpu_vram_mb = vram.min(1_048_576);
    }
    if let Some(auto_manage) = body.auto_manage_shards {
        config.auto_manage.enabled = auto_manage;
        // Update the runtime atomic so AutoShardManager picks it up immediately
        state
            .shared_state
            .models
            .auto_manage_enabled
            .store(auto_manage, std::sync::atomic::Ordering::Release);
        if auto_manage {
            // Wake the AutoShardManager so it evaluates promptly
            state.shared_state.models.auto_manage_notify.notify_one();
        }
        state.shared_state.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "system",
                "config_updated",
                format!(
                    "Auto-manage {}",
                    if auto_manage { "enabled" } else { "disabled" }
                ),
            )
            .with_toast("info", 4000),
        );
    }
    if let Some(max_storage) = body.auto_manage_max_storage_mb {
        config.auto_manage.max_storage_mb = max_storage.clamp(1, MAX_AUTO_MANAGE_STORAGE_MB);
    }
    if let Some(shard_size) = body.shard_size_mb {
        if !(crate::config::SHARD_SIZE_MIN_MB..=crate::config::SHARD_SIZE_MAX_MB)
            .contains(&shard_size)
        {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "shard_size_mb must be between {} and {} (got {})",
                crate::config::SHARD_SIZE_MIN_MB,
                crate::config::SHARD_SIZE_MAX_MB,
                shard_size
            ))));
        }
        config.model.shard_size_mb = shard_size;
    }
    if let Some(batch_size) = body.max_batch_size {
        config.inference.max_batch_size = batch_size.max(1);
    }
    if let Some(timeout) = body.batch_timeout_ms {
        config.inference.batch_timeout_ms = timeout.clamp(1, MAX_BATCH_TIMEOUT_MS);
    }
    if let Some(allow) = body.allow_cross_pool_inference {
        // Goes live with the whole config at the end of this handler;
        // `cross_pool_extras` reads it on the next request.
        config.pool.allow_cross_pool_inference = allow;
    }
    if let Some(share) = body.share_model_catalog {
        // Read by `HealthMonitor::broadcast_pool_model_availability` on the
        // next gossip tick (≤30s by default).
        config.pool.share_model_catalog = share;
    }
    if let Some(ref m) = body.update_mode {
        let parsed = match m.as_str() {
            "off" => Some(crate::config::UpdateMode::Off),
            "notify" => Some(crate::config::UpdateMode::Notify),
            "download" => Some(crate::config::UpdateMode::Download),
            "install" => Some(crate::config::UpdateMode::Install),
            _ => None,
        };
        match parsed {
            Some(mode) => config.updates.mode = Some(mode),
            None => {
                return Err(ApiError(crate::error::SwarmError::Validation(format!(
                    "unknown update mode {m:?} (expected off, notify, download or install)"
                ))))
            }
        }
    }
    if let Some(trust_lan) = body.dashboard_trust_lan {
        // Goes live with the whole config at the end of this handler — which
        // matters especially here, since this setting's whole purpose is to
        // make a dashboard reachable that currently isn't.
        config.api.dashboard_trust_lan = trust_lan;
        tracing::info!(
            enabled = trust_lan,
            "Dashboard local-network trust changed via admin API"
        );
    }

    // Write updated config to disk, emitting ONLY what differs from the
    // compiled defaults. Serializing the whole struct here is what stranded
    // three separate defaults on existing installs — see
    // `config::prune_defaults` for the full reasoning.
    let toml_str = crate::config::to_minimal_toml(&config).map_err(ApiError)?;

    let cp = config_path.clone();
    let cp_for_err = config_path.clone();
    tokio::task::spawn_blocking(move || {
        if let Some(parent) = cp.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&cp, toml_str)
    })
    .await
    .map_err(|e| {
        tracing::error!(error = %e, "Config save task panicked");
        ApiError(crate::error::SwarmError::Internal(
            "Failed to save configuration".into(),
        ))
    })?
    .map_err(|e| {
        // OS/disk failure (permission denied, disk full, ENOTDIR, etc)
        // — not a code bug. ServiceUnavailable (503) surfaces the right
        // semantics to the client (transient, retryable) and the
        // structured error log + response body carry the path so the
        // operator can triage.
        ApiError(crate::error::SwarmError::ServiceUnavailable(format!(
            "Failed to write config to {}: {e}",
            cp_for_err.display()
        )))
    })?;

    tracing::info!(path = %config_path.display(), "Configuration saved");

    // Make the saved config the LIVE one. This is what turns "saved" into
    // "applied": every runtime path reads `shared_state.cfg()`, so one store
    // here covers each setting rather than needing a per-setting mirror added
    // whenever somebody notices another one doing nothing.
    state.shared_state.apply_live_config(config.clone());

    // ...and wake the subsystems that must *react* rather than merely re-read:
    // the router resizes its concurrency and batch limits, auto-manage retimes
    // its interval. Re-reading alone cannot resize a semaphore already built.
    state
        .shared_state
        .apply_config_reload(crate::config::OperationalParams::from_config(&config));

    Ok(Json(serde_json::json!({ "status": "ok" })))
}

/// POST /api/admin/config/reload — Hot-reload operational parameters from config file.
///
/// Re-reads config.toml and makes it the live config, so every setting the
/// running node consults takes effect immediately, and pushes the few that a
/// subsystem must be told about (concurrency limit, batch size and window,
/// auto-manage interval, session timeout).
///
/// The response separates `applied` from `restart_required`. It used to list
/// `max_peers` among the applied set; nothing consumed it, and nothing could —
/// libp2p's connection limits are fixed when the swarm is built.
pub async fn reload_config(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let config_path = state.config.node.data_dir.join("config.toml");
    tracing::info!(
        path = %config_path.display(),
        "Config reload requested via API"
    );

    // Map config-file errors to HTTP-appropriate variants:
    //   missing file → 404 NotFound (the dashboard hasn't saved yet)
    //   parse / IO error → 400 Validation (broken file content)
    // SwarmError::Config is reserved for startup-only errors per the rule in
    // .claude/rules/completeness.md and would otherwise leak the unhelpful
    // "invalid_request_error" type for what is really a config-file issue.
    let params = crate::config::reload_operational_params(&config_path).map_err(|e| match e {
        crate::error::SwarmError::Config(msg) if msg.starts_with("Config file not found") => {
            ApiError(crate::error::SwarmError::NotFound(msg))
        }
        crate::error::SwarmError::Config(msg) => {
            ApiError(crate::error::SwarmError::Validation(msg))
        }
        crate::error::SwarmError::Io(io_err) => ApiError(crate::error::SwarmError::Validation(
            format!("Config file IO error: {io_err}"),
        )),
        other => ApiError(other),
    })?;

    let old = crate::config::OperationalParams::from_config(&state.config);
    let changed = params != old;

    state.shared_state.apply_config_reload(params.clone());

    if changed {
        tracing::info!(?params, "Config reloaded with changes via API");
    } else {
        tracing::info!(path = %config_path.display(), "Config reloaded via API — no changes detected");
    }

    // Report what this call actually did, split from what it could not do.
    // The previous response listed `max_peers` and the contribution/VRAM
    // settings alongside the rest as though reloading had applied them; none of
    // them was wired to anything, and `max_peers` cannot be — libp2p's
    // connection limits are fixed when the swarm is built.
    let live = state.shared_state.cfg();
    Ok(Json(serde_json::json!({
        "status": "ok",
        "changed": changed,
        // Re-read by whoever acts on them, or pushed to a subsystem that had to
        // resize something. Either way: in force now.
        "applied": {
            "max_concurrent_requests": params.max_concurrent_requests,
            "auto_manage_interval_minutes": params.auto_manage_interval_minutes,
            "max_batch_size": params.max_batch_size,
            "batch_timeout_ms": params.batch_timeout_ms,
            "session_timeout_secs": params.session_timeout_secs,
            "contribution": live.node.contribution,
            "contribution_auto": live.node.contribution_auto,
            "max_gpu_vram_mb": live.resources.max_gpu_vram_mb,
            "max_disk_mb": live.resources.max_disk_mb,
            "max_bandwidth_mbps": live.resources.max_bandwidth_mbps,
        },
        // Read once at startup. Saying so is the point: a caller that cannot
        // tell these apart has no way to know the reload was partial.
        "restart_required": {
            "max_peers": live
                .network
                .effective_max_connections(live.node.contribution.clone()),
        }
    })))
}

/// GET /api/admin/peers — List connected peers.
pub async fn list_peers(State(state): State<AppState>) -> Json<Vec<serde_json::Value>> {
    let peers: Vec<serde_json::Value> = state
        .shared_state
        .peer_registry
        .iter()
        .map(|entry| serialize_peer_to_json(entry.value(), &state.shared_state, true))
        .collect();

    Json(peers)
}

/// GET /api/admin/credits — Credit details.
pub async fn credit_info(State(state): State<AppState>) -> Json<serde_json::Value> {
    // Snapshot the balance and drop the read lock before computing escrow
    // so the credit hot-path doesn't park behind us.
    let (
        balance,
        lifetime_earned,
        lifetime_spent,
        lifetime_refunded,
        net_spent,
        books_balance,
        last_updated,
        tier,
    ) = {
        let credit = state.shared_state.credits.credit_balance.read().await;
        (
            credit.balance,
            credit.lifetime_earned,
            credit.lifetime_spent,
            credit.lifetime_refunded,
            credit.net_spent(),
            credit.books_balance(),
            credit.last_updated.to_rfc3339(),
            crate::credit::priority::PriorityCalculator::tier_name(credit.balance),
        )
    };
    let escrow_held = state.shared_state.credits.escrow_manager.pending_total();
    let escrow_pending = state.shared_state.credits.escrow_manager.pending_count();

    Json(serde_json::json!({
        "balance": balance,
        "lifetime_earned": lifetime_earned,
        "lifetime_spent": lifetime_spent,
        // `lifetime_spent` is GROSS reservations and never decreases, so
        // `earned - spent` does not equal `balance`. Report the refunds and the
        // net so the arithmetic closes without the reader having to guess —
        // and because refunds/spent IS this node's request failure rate.
        "lifetime_refunded": lifetime_refunded,
        "net_spent": net_spent,
        "books_balance": books_balance,
        "tier": tier,
        "last_updated": last_updated,
        "escrow_held": escrow_held,
        "escrow_pending_count": escrow_pending,
    }))
}

/// GET /api/admin/api-key — Return the current API key.
/// This endpoint requires authentication itself (Bearer token).
/// GET /api/admin/credits/transactions — the individual movements behind the totals.
///
/// **The totals were all there was.** A node reported 205,170 spent and 204,880
/// refunded against zero requests made or served, and asked, reasonably, whether
/// anything could show the transactions behind those figures. Nothing could:
/// only the running counters were kept.
///
/// Worth knowing when reading this: `lifetime_refunded` is partly synthetic.
/// `backfill_historical_refunds` attributes any otherwise-unexplained gap to
/// refunds, so `earned - spent + refunded == balance` closes by construction and
/// is not evidence that the movements were understood. This log is.
pub async fn credit_transactions(State(state): State<AppState>) -> Json<serde_json::Value> {
    let entries = state.shared_state.db.credit_log();
    let bal = state.shared_state.credits.credit_balance.read().await;
    Json(serde_json::json!({
        "transactions": entries,
        "count": entries.len(),
        "note": "Bounded log of recent balance movements, oldest first. Entries \
                 predating this feature are absent — a node with large totals and \
                 an empty log has simply not moved credits since upgrading.",
        "totals": {
            "balance": bal.balance,
            "lifetime_earned": bal.lifetime_earned,
            "lifetime_spent": bal.lifetime_spent,
            "lifetime_refunded": bal.lifetime_refunded,
        },
    }))
}

/// POST /api/admin/api-key/rotate — issue a new key and invalidate the old one.
///
/// **There was no way to do this.** The key is generated once and kept in the
/// database; `data/api_key` is only a published copy, so deleting that file
/// republishes the same value byte for byte. An operator who believed they had
/// rotated a leaked key had not, and the only real remedy was destroying the
/// node's database — which also destroys its identity and its credit balance.
/// Reported 2026-08-09 by someone running a node reachable from the internet,
/// where it matters most.
///
/// The new key takes effect on the next daemon start, because the running
/// server holds the current one in `SharedState`, which is immutable by design.
/// Saying so plainly is the honest option: silently issuing a key that does not
/// yet work would be worse than asking for a restart.
pub async fn rotate_api_key(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let key = hex::encode(bytes);

    state
        .shared_state
        .db
        .put_json("config", "api_key", &key)
        .map_err(ApiError)?;
    // Publish it alongside, so the file and the database never disagree — a
    // stale file is how someone concludes rotation did nothing.
    crate::daemon::publish_api_key_file(&state.shared_state.config.node.data_dir, &key);

    tracing::warn!("API key rotated by admin request — restart this node for it to take effect");
    state.shared_state.emit_activity(
        crate::daemon::state::ActivityEvent::new(
            "security",
            "api_key_rotated",
            "A new API key was issued. Restart this node to start using it — \
             anything holding the old key keeps working until then."
                .to_string(),
        )
        .with_toast("warning", 10000),
    );

    Ok(Json(serde_json::json!({
        "api_key": key,
        "active": false,
        "message": "New key saved. Restart this node for it to take effect; \
                    the previous key keeps working until you do.",
    })))
}

pub async fn get_api_key(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "api_key": state.shared_state.api_key,
    }))
}

// ---- Request types ----

#[derive(Debug, Deserialize)]
pub struct ConfigUpdate {
    pub contribution: Option<String>,
    pub contribution_auto: Option<bool>,
    pub max_concurrent_requests: Option<u32>,
    pub max_bandwidth_mbps: Option<u64>,
    pub max_disk_mb: Option<u64>,
    pub max_gpu_vram_mb: Option<u64>,
    pub auto_manage_shards: Option<bool>,
    pub auto_manage_max_storage_mb: Option<u64>,
    pub shard_size_mb: Option<u64>,
    pub max_batch_size: Option<u32>,
    pub batch_timeout_ms: Option<u64>,
    /// R137: hot-reloadable cross-pool inference fallback toggle.
    /// Persisted to config TOML + mirrored to
    /// the live config.
    pub allow_cross_pool_inference: Option<bool>,
    /// R137: hot-reloadable cross-pool model catalog gossip toggle.
    /// Persisted to config TOML + mirrored to
    /// the live config.
    pub share_model_catalog: Option<bool>,
    /// Hand the dashboard its API key to browsers on a private/LAN address.
    /// Persisted to config TOML + mirrored to `state.dashboard_trust_lan` so
    /// it applies on the next page load rather than the next restart.
    pub dashboard_trust_lan: Option<bool>,
    /// "off" | "notify" | "download" | "install". Takes effect on restart —
    /// the UpdateChecker task is spawned (or not) at startup.
    pub update_mode: Option<String>,
}

/// POST /api/admin/shutdown — Gracefully shut down the node.
/// Only accepts requests from localhost (127.0.0.1 or ::1) for safety.
pub async fn shutdown_node(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if !addr.ip().is_loopback() {
        return Err(ApiError(crate::error::SwarmError::Unauthorized(
            "Shutdown only allowed from localhost".into(),
        )));
    }
    tracing::info!(addr = %addr, "Shutdown requested via API");

    // Signal all subsystems to shut down via the watch channel.
    // The daemon.rs supervisor loop will handle graceful draining,
    // peer cache saving, DB flushing, and process exit.
    state.shared_state.shutdown();

    Ok(Json(serde_json::json!({ "status": "shutting_down" })))
}
// ---- Hardware detection ----

fn detect_hardware(shared_state: &crate::daemon::SharedState) -> serde_json::Value {
    use sysinfo::System;

    let mut sys = System::new_all();
    sys.refresh_all();

    let total_ram_mb = sys.total_memory() / (1024 * 1024);
    let used_ram_mb = sys.used_memory() / (1024 * 1024);

    // Per-process memory (RSS) — actual memory this node is using
    let process_rss_mb = {
        let pid = sysinfo::Pid::from_u32(std::process::id());
        sys.process(pid)
            .map(|p| p.memory() / (1024 * 1024))
            .unwrap_or(0)
    };

    let cpu_name = sys
        .cpus()
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown".to_string());
    let cpu_cores = sys.cpus().len();

    // Disk info — use sysinfo disks
    let disks = sysinfo::Disks::new_with_refreshed_list();
    let (mut total_disk_mb, mut available_disk_mb) = (0u64, 0u64);
    for disk in disks.list() {
        total_disk_mb += disk.total_space() / (1024 * 1024);
        available_disk_mb += disk.available_space() / (1024 * 1024);
    }
    let used_disk_mb = total_disk_mb.saturating_sub(available_disk_mb);

    // GPU info from llama.cpp device detection (set at startup)
    // Falls back to nvidia-smi when gpu_info is None (e.g. non-CUDA build)
    let (gpu_name, gpu_vram_mb, gpu_vram_used_mb) = match &shared_state.gpu_info {
        Some(gpu) => {
            // Query live VRAM usage via nvidia-smi for an up-to-date reading
            let used = crate::model::auto_manage::vram::query_gpu_vram_used();
            (
                Some(gpu.name.clone()),
                Some(gpu.vram_total_mb),
                used.or(Some(gpu.vram_total_mb.saturating_sub(gpu.vram_free_mb))),
            )
        }
        None => {
            let (name, total) = detect_gpu_nvidia_smi();
            let used = crate::model::auto_manage::vram::query_gpu_vram_used();
            (name, total, used)
        }
    };

    // gpu_inference: true only when llama.cpp actually bound to the GPU device
    let gpu_inference = shared_state.gpu_info.is_some();
    let inference_backend = shared_state.gpu_info.as_ref().map(|g| g.backend.clone());

    let (memory_bandwidth_gbps, est_tokens_per_sec_7b) = match &shared_state.gpu_info {
        Some(gpu) => {
            let bw = crate::model::auto_manage::vram::gpu_memory_bandwidth_gbps(&gpu.name);
            let tps = crate::model::auto_manage::vram::estimate_tokens_per_sec_7b(bw, true);
            (Some(bw), Some(tps))
        }
        None => (None, None),
    };

    serde_json::json!({
        "gpu_name": gpu_name,
        "gpu_vram_mb": gpu_vram_mb,
        "gpu_vram_used_mb": gpu_vram_used_mb,
        "gpu_inference": gpu_inference,
        "inference_backend": inference_backend,
        // `inference_backend` is a property of the BUILD. These are the models
        // that are NOT on the GPU right now despite it — pinned there by a GPU
        // OOM. Without this the API reported "CUDA" while everything ran on the
        // CPU at roughly a tenth of the speed.
        "models_on_cpu_fallback": shared_state.model_process_pool.cpu_pinned_model_ids(),
        "memory_bandwidth_gbps": memory_bandwidth_gbps,
        "est_tokens_per_sec_7b": est_tokens_per_sec_7b,
        "total_ram_mb": total_ram_mb,
        "used_ram_mb": used_ram_mb,
        "process_rss_mb": process_rss_mb,
        "available_disk_mb": available_disk_mb,
        "total_disk_mb": total_disk_mb,
        "used_disk_mb": used_disk_mb,
        "cpu_name": cpu_name,
        "cpu_cores": cpu_cores,
    })
}

/// Fallback GPU detection via nvidia-smi when llama.cpp gpu_info is unavailable.
pub(crate) use crate::model::auto_manage::vram::detect_gpu_nvidia_smi;

/// POST /api/admin/rescan-shards — Scan the models directory for new shard files.
///
/// Discovers shard files that were added to disk since the last scan (e.g. by
/// manual copy), registers them in the model registry, reloads affected models,
/// and re-announces shards to the network. No restart needed.
pub async fn rescan_shards(State(state): State<AppState>) -> Json<serde_json::Value> {
    let network_tx = state.network_tx.clone();
    let outcome =
        crate::model::auto_manage::rescan_local_shards(&state.shared_state, network_tx.as_ref())
            .await;
    // `skipped_*` is additive — `status`, `models_updated` and `count` keep
    // their meaning. Without it a rescan reported `count: 0` whether it had
    // nothing to do or had deliberately passed over a shard sitting on disk,
    // and those are very different answers to "why is this shard not served?".
    Json(serde_json::json!({
        "status": "ok",
        "models_updated": outcome.changed.iter().map(|m| &m.0).collect::<Vec<_>>(),
        "count": outcome.changed.len(),
        "skipped_outside_shard_range": outcome.skipped_outside_shard_range,
    }))
}

/// GET /api/admin/network-map — Aggregated region data for the world heatmap.
///
/// Returns `{ regions: { "US": { total: N, models: { "model-id": count } }, ... } }`
/// based on self-reported region in peer capabilities.
pub async fn network_map(State(state): State<AppState>) -> Json<serde_json::Value> {
    use std::collections::HashMap;

    let mut regions: HashMap<String, (u64, HashMap<String, u64>)> = HashMap::new();

    // Always include our own node on the map.
    // Use auto-detected region (IP geolocation), configured region, or "??" as fallback.
    {
        let code = state
            .shared_state
            .effective_region()
            .await
            .unwrap_or_else(|| "??".into())
            .to_uppercase();
        let entry = regions.entry(code).or_insert_with(|| (0, HashMap::new()));
        entry.0 += 1;
        // Add our hosted models (never a backup-copy name).
        let node_id = state.shared_state.identity.node_id();
        for (shard_id, holders) in state.shared_state.model_registry.all_shard_entries() {
            if holders.contains(node_id)
                && !crate::model::manifest::is_backup_artifact_id(&shard_id.model_id.0)
            {
                *entry.1.entry(shard_id.model_id.0.clone()).or_insert(0) += 1;
            }
        }
    }

    // Aggregate peer regions from capabilities.
    // Peers without capability/region info are placed in our own region (most peers
    // on a LAN share the same region) or "??" as fallback.
    let self_region = state
        .shared_state
        .effective_region()
        .await
        .unwrap_or_else(|| "??".into())
        .to_uppercase();
    for peer in state.shared_state.peer_registry.iter() {
        let (region_code, hosted_shards) = match peer.value().capability {
            Some(ref cap) => {
                let code = cap.region.as_deref().unwrap_or(&self_region).to_uppercase();
                (code, &cap.hosted_shards[..])
            }
            None => (self_region.clone(), &[][..]),
        };
        let entry = regions
            .entry(region_code)
            .or_insert_with(|| (0, HashMap::new()));
        entry.0 += 1;
        // Count distinct models this peer hosts (a peer on an older build still
        // gossips a backup-copy name in its capability — keep it out of the
        // swarm-wide region aggregation too).
        let mut peer_models = std::collections::HashSet::new();
        for shard in hosted_shards {
            if !crate::model::manifest::is_backup_artifact_id(&shard.model_id.0) {
                peer_models.insert(shard.model_id.0.clone());
            }
        }
        for model_id in peer_models {
            *entry.1.entry(model_id).or_insert(0) += 1;
        }
    }

    // Collect all known model IDs for coverage gap detection
    let all_models: Vec<String> = state
        .shared_state
        .model_registry
        .models()
        .iter()
        .map(|m| m.id.0.clone())
        .collect();

    let pool_size = state.shared_state.peer_registry.len() + 1;
    let min_replicas = state.shared_state.cfg().auto_manage.min_replicas as usize;

    // Build JSON with regional demand, coverage gaps, and replication targets
    let region_json: serde_json::Map<String, serde_json::Value> = regions
        .into_iter()
        .map(|(code, (total, models))| {
            let models_json: serde_json::Map<String, serde_json::Value> = models
                .into_iter()
                .map(|(k, v)| (k, serde_json::json!(v)))
                .collect();

            // Per-model demand rates for this region
            let mut demand: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
            for entry in state.shared_state.region_demand.iter() {
                let (model_id, region) = entry.key();
                if region.eq_ignore_ascii_case(&code) {
                    demand.insert(model_id.0.clone(), serde_json::json!(*entry.value()));
                }
            }

            // Coverage gaps: models where this region has 0 holders
            let coverage_gaps: Vec<&str> = all_models
                .iter()
                .filter(|m| !models_json.contains_key(m.as_str()))
                .map(|m| m.as_str())
                .collect();

            // Per-model replication target for this region
            let mut replication_target: serde_json::Map<String, serde_json::Value> =
                serde_json::Map::new();
            for model_id_str in models_json.keys() {
                let model_id = crate::types::ModelId(model_id_str.clone());
                let request_count = state
                    .shared_state
                    .models
                    .model_request_counts
                    .get(&model_id)
                    .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0);
                let global_floor = if pool_size <= 1 {
                    min_replicas
                } else {
                    let log2_pool = (pool_size as f64).log2().ceil() as usize;
                    let max_replicas = (pool_size / 3).max(1);
                    log2_pool.clamp(min_replicas.min(max_replicas), max_replicas)
                };
                let demand_factor = match request_count {
                    0 => 1.0,
                    1..=5 => 1.5,
                    6..=20 => 2.0,
                    21..=100 => 2.5,
                    _ => 3.0,
                };
                let target = (global_floor as f64 * demand_factor).ceil() as usize;
                replication_target.insert(
                    model_id_str.clone(),
                    serde_json::json!(target.min(pool_size).max(1)),
                );
            }

            (
                code,
                serde_json::json!({
                    "total": total,
                    "models": models_json,
                    "demand": demand,
                    "coverage_gaps": coverage_gaps,
                    "replication_target": replication_target,
                }),
            )
        })
        .collect();

    Json(serde_json::json!({ "regions": region_json }))
}

/// GET /api/admin/network-code — Return this node's network invite code.
///
/// Returns a shareable invite code that other nodes can use to connect.
/// The code encodes the node's QUIC listening address.
pub async fn network_code(State(state): State<AppState>) -> Json<serde_json::Value> {
    let port = state.config.node.listen_port;
    let peer_count = state.shared_state.peer_registry.len();

    // Build the QUIC listen address with the node's peer ID
    let signing_key_bytes = state.shared_state.identity.signing_key_bytes();
    let peer_id_str = match crate::network::transport::ed25519_to_libp2p_keypair(signing_key_bytes)
    {
        Ok(kp) => kp.public().to_peer_id().to_string(),
        Err(_) => {
            return Json(serde_json::json!({
                "error": "Failed to derive peer ID"
            }))
        }
    };

    // Pick a real IP by scanning peer addresses that other nodes see for us,
    // or fall back to detecting the local machine's non-loopback IP. Cap the
    // peer scan at NETWORK_CODE_PEER_SCAN_CAP — a public-facing IP is almost
    // always advertised by the first few peers, and at 10k-peer scale the
    // unbounded inner loop becomes a notable per-request hot path on the
    // dashboard's invite-code refresh.
    const NETWORK_CODE_PEER_SCAN_CAP: usize = 64;
    const NETWORK_CODE_ADDR_PER_PEER_CAP: usize = 16;
    let best_ip = {
        // Try to find a non-loopback IP from peers' addresses for our node
        let mut found_ip = None;
        for peer in state
            .shared_state
            .peer_registry
            .iter()
            .take(NETWORK_CODE_PEER_SCAN_CAP)
        {
            for addr in peer.addresses.iter().take(NETWORK_CODE_ADDR_PER_PEER_CAP) {
                if addr.starts_with("/ip4/") {
                    let parts: Vec<&str> = addr.split('/').collect();
                    if parts.len() >= 3 {
                        let ip = parts[2];
                        if ip != "127.0.0.1" && ip != "0.0.0.0" && ip != "10.255.255.254" {
                            found_ip = Some(ip.to_string());
                            break;
                        }
                    }
                }
            }
            if found_ip.is_some() {
                break;
            }
        }
        found_ip.unwrap_or_else(|| {
            // Fall back: try to detect local non-loopback IP via UDP socket trick
            std::net::UdpSocket::bind("0.0.0.0:0")
                .and_then(|s| {
                    s.connect("8.8.8.8:80")?;
                    s.local_addr()
                })
                .map(|a| a.ip().to_string())
                .unwrap_or_else(|_| "127.0.0.1".to_string())
        })
    };

    // Use TCP address (port+10) — more reliable across environments (WSL2, Docker, NAT).
    // QUIC on WSL2 often fails with handshake timeouts on the virtual adapter IP.
    let tcp_port = port + 10;
    let multiaddr_str = format!("/ip4/{best_ip}/tcp/{tcp_port}/p2p/{peer_id_str}");
    let code = if let Ok(addr) = multiaddr_str.parse::<libp2p::Multiaddr>() {
        crate::network::discovery::encode_network_code(&addr)
    } else {
        multiaddr_str.clone()
    };

    // Determine network phase
    let phase = if peer_count == 0 {
        "seedling" // no peers — solo node
    } else {
        "established" // 1+ peers — connected to network
    };

    Json(serde_json::json!({
        "code": code,
        "node_id": format!("{}", state.shared_state.identity.node_id()),
        "peer_id": peer_id_str,
        // The node's current reachable dial addresses (listeners ∪ confirmed
        // external addresses), each terminated with /p2p/<peer_id>. For a node
        // with `network.external_address` set (e.g. an anchor with a DuckDNS
        // host) this is the exact string to drop into other nodes'
        // `bootstrap_peers`. Empty until the swarm has bound + confirmed addrs.
        "listen_multiaddrs": state.shared_state.listen_multiaddrs.load().as_ref().clone(),
        "phase": phase,
        "peer_count": peer_count,
    }))
}

/// POST /api/admin/join-network — Join the network using an invite code.
///
/// Accepts a network invite code (swarm://...) or raw multiaddr and dials the peer.
pub async fn join_network(
    State(state): State<AppState>,
    JsonBody(body): JsonBody<JoinNetworkRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if body.code.len() > MAX_INVITE_CODE_LEN {
        return Err(ApiError(crate::error::SwarmError::Validation(format!(
            "Invite code too long (max {} chars)",
            MAX_INVITE_CODE_LEN
        ))));
    }
    let addr_str = crate::network::discovery::decode_network_code(&body.code)
        .map_err(|e| ApiError(crate::error::SwarmError::Validation(e.to_string())))?;

    // Validate the multiaddr
    let _addr: libp2p::Multiaddr = addr_str.parse().map_err(|e: libp2p::multiaddr::Error| {
        ApiError(crate::error::SwarmError::Validation(format!(
            "Invalid address in invite code: {e}"
        )))
    })?;

    // SEC: Reject private / loopback / link-local / cloud-metadata addresses.
    // Without this check, an attacker with the API key could supply a multiaddr
    // pointing at internal services (e.g. 169.254.169.254 IMDS, 127.0.0.1
    // services on the host) and the daemon would attempt P2P-layer dials —
    // P2P-layer SSRF — and persist the address to peer cache for re-dialing.
    if crate::network::helpers::is_non_public_addr(&addr_str) {
        return Err(ApiError(crate::error::SwarmError::Validation(
            "Address resolves to a private/loopback/link-local IP — refusing to dial".into(),
        )));
    }

    tracing::info!(addr = %addr_str, "Joining network via invite code");

    // Save to peer cache so it persists across restarts
    let mut cached = crate::network::peer_cache::load_peer_cache(&state.shared_state.db);
    if !cached.contains(&addr_str) {
        cached.push(addr_str.clone());
        crate::network::peer_cache::save_peer_cache(&state.shared_state.db, &cached);
    }

    // Dial immediately if network manager is available
    if let Some(ref tx) = state.network_tx {
        let _ = tx
            .send(crate::types::NetworkCommand::DialAddress(addr_str.clone()))
            .await;
    }

    Ok(Json(serde_json::json!({
        "status": "ok",
        "address": addr_str,
        "message": "Connecting to peer..."
    })))
}

#[derive(Deserialize)]
pub struct JoinNetworkRequest {
    pub code: String,
}
// ---- Resource Schedule API ----

/// GET /api/admin/schedule — Get current resource schedule.
pub async fn get_schedule(State(state): State<AppState>) -> Json<serde_json::Value> {
    let schedule = state.shared_state.models.resource_schedule.read().await;
    Json(serde_json::json!({
        "enabled": schedule.enabled,
        "reduced_hours_start": schedule.reduced_hours_start,
        "reduced_hours_end": schedule.reduced_hours_end,
        "reduced_contribution": schedule.reduced_contribution,
        "prune_aggressiveness": schedule.prune_aggressiveness,
    }))
}

#[derive(Debug, Deserialize)]
pub struct ScheduleUpdate {
    pub enabled: Option<bool>,
    pub reduced_hours_start: Option<u32>,
    pub reduced_hours_end: Option<u32>,
    pub reduced_contribution: Option<String>,
    pub prune_aggressiveness: Option<String>,
}

/// PUT /api/admin/schedule — Update resource schedule at runtime (persisted to redb).
pub async fn update_schedule(
    State(state): State<AppState>,
    JsonBody(body): JsonBody<ScheduleUpdate>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // Clone current schedule, validate + apply updates without holding the write lock
    let mut new_schedule = state
        .shared_state
        .models
        .resource_schedule
        .read()
        .await
        .clone();

    if let Some(enabled) = body.enabled {
        new_schedule.enabled = enabled;
    }
    if let Some(start) = body.reduced_hours_start {
        if start > 23 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "reduced_hours_start must be 0-23".to_string(),
            )));
        }
        new_schedule.reduced_hours_start = start;
    }
    if let Some(end) = body.reduced_hours_end {
        if end > 23 {
            return Err(ApiError(crate::error::SwarmError::Validation(
                "reduced_hours_end must be 0-23".to_string(),
            )));
        }
        new_schedule.reduced_hours_end = end;
    }
    if let Some(ref contribution) = body.reduced_contribution {
        match contribution.as_str() {
            "minimal" | "moderate" | "maximum" => {
                new_schedule.reduced_contribution = contribution.clone();
            }
            _ => {
                return Err(ApiError(crate::error::SwarmError::Validation(
                    "reduced_contribution must be 'minimal', 'moderate', or 'maximum'".to_string(),
                )));
            }
        }
    }
    if let Some(ref aggressiveness) = body.prune_aggressiveness {
        match aggressiveness.as_str() {
            "normal" | "aggressive" | "conservative" => {
                new_schedule.prune_aggressiveness = aggressiveness.clone();
            }
            _ => {
                return Err(ApiError(crate::error::SwarmError::Validation(
                    "prune_aggressiveness must be 'normal', 'aggressive', or 'conservative'"
                        .to_string(),
                )));
            }
        }
    }

    // Persist to DB (no write lock held)
    if let Err(e) = state
        .shared_state
        .db
        .put_json("resource_schedule", "current", &new_schedule)
    {
        tracing::warn!(error = %e, "Failed to persist resource schedule — will revert on restart");
    }

    tracing::debug!(
        enabled = new_schedule.enabled,
        prune_aggressiveness = %new_schedule.prune_aggressiveness,
        "DIAG: schedule updated"
    );

    let result = serde_json::json!({
        "status": "ok",
        "enabled": new_schedule.enabled,
        "reduced_hours_start": new_schedule.reduced_hours_start,
        "reduced_hours_end": new_schedule.reduced_hours_end,
        "reduced_contribution": new_schedule.reduced_contribution,
        "prune_aggressiveness": new_schedule.prune_aggressiveness,
    });

    // Briefly acquire write lock to commit
    *state.shared_state.models.resource_schedule.write().await = new_schedule;

    Ok(Json(result))
}

// ============================================================================
// V6 (responses_api_v2): Responses dashboard endpoint
// ============================================================================

/// Query parameters for `GET /api/admin/responses`.
#[derive(Debug, Deserialize)]
pub struct AdminResponsesQuery {
    /// Filter by status (queued | in_progress | completed | failed |
    /// cancelled | incomplete). Repeatable via comma list. Empty / unset
    /// returns every status.
    #[serde(default)]
    pub status: Option<String>,
    /// Cap the number of records returned. Default 100, max 500.
    #[serde(default)]
    pub limit: Option<u32>,
}

/// `GET /api/admin/responses?status=...&limit=...` — list stored
/// `/v1/responses` records for the dashboard. Sorted newest first.
///
/// Streams the underlying redb tree so memory stays O(limit) rather
/// than O(total_records). The full preview JSON is only built for the
/// records that survive the bounded top-k pass.
pub async fn list_responses(
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    State(state): State<crate::api::server::AppState>,
    axum::extract::Query(params): axum::extract::Query<AdminResponsesQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    // SEC: input_preview shows a 100-char prefix of the user's prompt. With a
    // shared cluster API key, exposing it to non-loopback callers would leak
    // every other user's prompts. Loopback callers (local dashboard) get the
    // full preview; remote API-key holders see metadata only.
    let include_preview = addr.ip().is_loopback();
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    /// Heap entry that orders by `created_at` only — the record itself
    /// doesn't implement Ord. Sort order is REVERSED (`older` compares
    /// as greater) so a default max-heap behaves as a min-heap of
    /// newest survivors: `peek()` returns the oldest kept candidate,
    /// `pop()` evicts it.
    struct HeapEntry {
        created_at: i64,
        rec: crate::api::openai::responses::store::ResponsesRecord,
    }
    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.created_at == other.created_at
        }
    }
    impl Eq for HeapEntry {}
    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            other.created_at.cmp(&self.created_at)
        }
    }

    // Cap the raw query string before splitting so a caller can't pass a
    // megabyte-long `?status=` and force `.split(',')` to materialise an
    // arbitrarily large Vec — the only valid values are a handful of
    // short ASCII enum strings, 256 bytes is generous.
    let status_filter: Option<Vec<String>> = match params.status {
        Some(s) if s.len() > MAX_STATUS_FILTER_BYTES => {
            return Err(ApiError(crate::error::SwarmError::Validation(format!(
                "status filter too long ({} bytes, max {MAX_STATUS_FILTER_BYTES})",
                s.len()
            ))));
        }
        Some(s) => {
            let tokens: Vec<String> = s
                .split(',')
                .map(|t| t.trim().to_lowercase())
                .filter(|t| !t.is_empty())
                .collect();
            // Reject typos: a `?status=complete` filter would silently match
            // zero records and look like an empty result instead of an error.
            const VALID: &[&str] = &[
                "queued",
                "in_progress",
                "completed",
                "failed",
                "cancelled",
                "incomplete",
            ];
            for tok in &tokens {
                if !VALID.contains(&tok.as_str()) {
                    return Err(ApiError(crate::error::SwarmError::Validation(format!(
                        "unknown status filter '{tok}': must be one of queued, in_progress, completed, failed, cancelled, incomplete"
                    ))));
                }
            }
            Some(tokens)
        }
        None => None,
    };
    let limit = params.limit.unwrap_or(100).clamp(1, 500) as usize;

    // Whether a live background-streaming task is in flight for this id.
    let live_ids: std::collections::HashSet<String> =
        crate::api::openai::responses::background::BACKGROUND_STATE
            .iter()
            .map(|e| e.key().clone())
            .collect();

    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(limit + 1);

    state
        .db
        .for_each_json::<crate::api::openai::responses::store::ResponsesRecord, _>(
            crate::api::openai::responses::store::TREE,
            |_subkey, rec| {
                if let Some(filter) = &status_filter {
                    let s = serde_json::to_string(&rec.response.status)
                        .unwrap_or_default()
                        .trim_matches('"')
                        .to_string();
                    if !filter.iter().any(|f| f == &s) {
                        return;
                    }
                }
                if heap.len() < limit {
                    heap.push(HeapEntry {
                        created_at: rec.created_at,
                        rec,
                    });
                } else if let Some(top) = heap.peek() {
                    if rec.created_at > top.created_at {
                        heap.pop();
                        heap.push(HeapEntry {
                            created_at: rec.created_at,
                            rec,
                        });
                    }
                }
            },
        )
        .map_err(ApiError)?;

    // into_sorted_vec yields ascending by `Ord` — which we reversed —
    // so the result is oldest → newest. Reverse for newest-first.
    let mut kept: Vec<HeapEntry> = heap.into_sorted_vec();
    kept.reverse();

    let data: Vec<serde_json::Value> = kept
        .into_iter()
        .map(|HeapEntry { rec, .. }| {
            let live = live_ids.contains(&rec.id);
            let preview = if include_preview {
                match &rec.request.input {
                    crate::api::openai::responses::types::ResponsesInput::Text(s) => {
                        truncate_preview(s)
                    }
                    crate::api::openai::responses::types::ResponsesInput::Items(items) => items
                        .first()
                        .and_then(|item| match item {
                            crate::api::openai::responses::types::InputItem::Typed(
                                crate::api::openai::responses::types::TypedInputItem::Message(m),
                            ) => match &m.content {
                                crate::api::openai::responses::types::InputMessageContent::Text(
                                    t,
                                ) => Some(truncate_preview(t)),
                                _ => None,
                            },
                            _ => None,
                        })
                        .unwrap_or_default(),
                }
            } else {
                String::new()
            };
            let output_preview = if include_preview {
                rec.response.output_text.as_deref().map(truncate_preview)
            } else {
                None
            };
            serde_json::json!({
                "id": rec.id,
                "created_at": rec.created_at,
                "expires_at": rec.expires_at,
                "model": rec.response.model,
                "status": rec.response.status,
                "background": rec.response.background.unwrap_or(false),
                "live": live,
                "input_preview": preview,
                "output_text_preview": output_preview,
                "usage": {
                    "input_tokens": rec.response.usage.input_tokens,
                    "output_tokens": rec.response.usage.output_tokens,
                },
            })
        })
        .collect();

    Ok(Json(serde_json::json!({
        "object": "list",
        "data": data,
        "total": data.len(),
    })))
}

fn truncate_preview(s: &str) -> String {
    const MAX: usize = 120;
    if s.chars().count() <= MAX {
        return s.to_string();
    }
    let mut out = String::with_capacity(MAX);
    for (i, ch) in s.chars().enumerate() {
        if i >= MAX {
            break;
        }
        out.push(ch);
    }
    out.push('…');
    out
}
