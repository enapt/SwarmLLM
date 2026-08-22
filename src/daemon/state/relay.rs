//! NETWORKING_PLAN Phase 1 — learned relay-route table + relay-side rate
//! limiting on the root `SharedState`.
//!
//! These back the application-level inference relay: a NAT'd node routes an
//! inference message through a mutually-reachable relay peer, and the two
//! endpoints learn a reverse route so replies and later turns flow back the
//! same way. The routing decision lives entirely in the NetworkManager's
//! directed-send path (`network/manager/commands.rs`) — daemon send code is
//! unchanged.

use std::time::{Duration, Instant};

use crate::types::NodeId;

/// How long a learned reverse route stays usable without being refreshed. A
/// route is refreshed every time another envelope arrives from that origin, so
/// an active session keeps it warm; this bounds staleness for an idle peer.
pub const RELAY_ROUTE_TTL_SECS: u64 = 300;

/// Relay-side forward rate limit window.
pub const RELAY_FORWARD_WINDOW_SECS: u64 = 10;

/// Max messages one origin may push through us as a relay per window. Generous
/// enough for fast token streaming (~200 msg/s sustained), low enough to blunt
/// a flood. Phase 3 replaces this coarse cap with credit-metered relaying.
pub const RELAY_FORWARD_MAX_PER_WINDOW: u32 = 2000;

/// A learned route to a target that we cannot reach directly: send to
/// `relay_peer_bytes`, which forwards to the target. `target_node` is kept so
/// the send path can seal the inner message for the target's X25519 key.
#[derive(Clone, Debug)]
pub struct RelayRoute {
    pub relay_peer_bytes: Vec<u8>,
    pub target_node: NodeId,
    pub learned_at: Instant,
}

/// Sliding-window forward counter for one origin (relay side).
#[derive(Debug)]
pub struct RelayForwardCounter {
    pub count: u32,
    pub window_start: Instant,
}

/// Relay features a peer has *demonstrably* used by sending us a relayed message
/// addressed to us: a `RelayedTensor` proves `features::TENSOR_RELAY`, a
/// `RelayedEnvelope` proves `features::RELAY`. This is direct, first-hand proof
/// the peer speaks the protocol — stronger and fresher than the gossiped
/// `NodeCapability.features`, which has a cold-start window where our entry for
/// a relay-only peer is still `capability: None` (the capability-gossip handler
/// is update-only, so it can't populate an entry that didn't exist yet). Without
/// this proof the return path would refuse to relay a computed result back to a
/// coordinator that JUST relayed a forward to us, dropping the first result and
/// timing the request out until a capability-gossip round lands.
#[derive(Debug, Clone)]
pub struct RelayProvenFeatures {
    pub features: u64,
    pub proven_at: Instant,
}

impl super::SharedState {
    /// Record that `origin` is reachable via the relay peer a `RelayedEnvelope`
    /// just arrived from. Keyed by `origin`'s peer-id bytes so the send path can
    /// look it up by the target peer bytes it already carries.
    pub fn learn_relay_route(&self, origin: &NodeId, relay_peer_bytes: Vec<u8>) {
        let Some(target_peer_id) = crate::network::transport::node_id_to_peer_id(origin) else {
            return;
        };
        let target_bytes = target_peer_id.to_bytes();
        // A route through the target itself is degenerate (and a self-route is
        // meaningless) — ignore both.
        if relay_peer_bytes == target_bytes {
            return;
        }
        self.relay_routes.insert(
            target_bytes,
            RelayRoute {
                relay_peer_bytes,
                target_node: origin.clone(),
                learned_at: Instant::now(),
            },
        );
    }

    /// Release every per-request entry a completed request leaves behind.
    ///
    /// `active_pipelines`, `active_traces` and `request_holder_blacklist` are
    /// keyed by request id and share one lifetime, so they are cleared together
    /// — five call sites removed all three by hand, and the invariant was held
    /// only by three adjacent lines and a comment saying so. Missing one is
    /// silent and unbounded: `active_traces` is the oracle behind
    /// `model_is_in_use`, so a stranded entry would refuse to delete that model
    /// for the daemon's lifetime, and the blacklist would keep barring a peer
    /// that was only ever meant to be skipped for one request.
    ///
    /// Does NOT touch the concurrency counter or the queue notify: those belong
    /// to the dispatch path that owns the slot, and the router requires the two
    /// to happen together (see `.claude/rules/architecture.md` § Inference
    /// Router Queue).
    pub fn release_request_state(&self, request_id: &uuid::Uuid) {
        self.active_pipelines.remove(request_id);
        self.active_traces.remove(request_id);
        self.request_holder_blacklist.remove(request_id);
    }

    /// Record a failed inference for the diagnostics ring buffer.
    ///
    /// Oldest-first eviction at [`super::MAX_RECENT_FAILURES`]. Best-effort: a
    /// poisoned lock is skipped rather than propagated, because losing a
    /// diagnostic record must never affect the request path.
    pub fn record_request_failure(&self, failure: super::RequestFailure) {
        if let Ok(mut buf) = self.recent_failures.lock() {
            if buf.len() >= super::MAX_RECENT_FAILURES {
                buf.pop_front();
            }
            buf.push_back(failure);
        }
    }

    /// Snapshot of recent failures, oldest first.
    pub fn recent_failures_snapshot(&self) -> Vec<super::RequestFailure> {
        self.recent_failures
            .lock()
            .map(|b| b.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Emit a completed request's trace to every observability surface.
    ///
    /// Called once, from the router's completion arm. Writes the single
    /// greppable `DIAG: request complete` line and pushes the snapshot into the
    /// ring that `GET /api/admin/diagnostics` renders — the successful sibling
    /// of [`Self::record_request_failure`], which is what made "why did that
    /// break" answerable but left "why was that slow" unanswerable.
    pub fn publish_request_trace(&self, trace: &crate::inference::trace::RequestTrace) {
        let snap = trace.snapshot();
        tracing::info!("DIAG: request complete {}", snap.log_line());

        // Prometheus. Fed from the same snapshot as the log line so a metric
        // can never disagree with what the logs say happened.
        *self
            .metrics
            .requests_by_route
            .entry((snap.route.as_str(), outcome_label(&snap.outcome)))
            .or_insert(0) += 1;

        // The grand total and the latency histogram live HERE, beside the
        // per-route counter, not in the callers.
        //
        // There are three call sites (the router's completion arm, the
        // distributed executor, and the OpenAI local-complete fast path).
        // `requests_by_route` was inside this function so every path fed it,
        // while the total and the latency samples were recorded *outside* at
        // one site only — so a scrape could show
        // `requests_by_route{local,ok} 1` next to `inference_requests_total 0`
        // and an empty histogram, for the same request. Reported 2026-07-30
        // as two separate findings ("a dead counter", "the histogram is never
        // populated"); they are one placement mistake.
        self.metrics
            .inference_requests_total
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if matches!(snap.outcome, crate::inference::trace::Outcome::Ok) {
            record_duration_sample(
                &self.metrics.inference_latency_samples,
                &self.metrics.inference_latency_total_count,
                &self.metrics.inference_latency_total_micros,
                snap.total_ms as f64 / 1000.0,
            );
        }
        if let Some(ms) = snap.ttft_ms {
            record_duration_sample(
                &self.metrics.ttft_samples,
                &self.metrics.ttft_total_count,
                &self.metrics.ttft_total_micros,
                ms as f64 / 1000.0,
            );
        }
        if let Some(ms) = snap.tpot_ms {
            record_duration_sample(
                &self.metrics.tpot_samples,
                &self.metrics.tpot_total_count,
                &self.metrics.tpot_total_micros,
                ms / 1000.0,
            );
        }

        // Hourly rollup. Persisted only when the hour rolls over — a redb
        // transaction per request would put disk I/O on the completion path.
        let now_ms = chrono::Utc::now().timestamp_millis().max(0) as u64;
        if self.perf_history.record(&snap, now_ms) {
            self.perf_history.persist(&self.db);
        }

        if let Ok(mut buf) = self.recent_traces.lock() {
            if buf.len() >= super::MAX_RECENT_TRACES {
                buf.pop_front();
            }
            buf.push_back(snap);
        }
    }

    /// Bar a holder from serving THIS request again.
    ///
    /// Pairs with [`Self::retract_shard_holder_claims_for_range`]: the retraction
    /// corrects our registry, this makes the correction stick for the retry even
    /// if the DHT re-advertises the holder in the meantime.
    pub fn blacklist_holder_for_request(&self, request_id: uuid::Uuid, holder: &NodeId) {
        self.request_holder_blacklist
            .entry(request_id)
            .or_default()
            .insert(holder.clone());
    }

    /// Is this holder barred from serving `request_id`?
    pub fn holder_blacklisted_for_request(&self, request_id: uuid::Uuid, holder: &NodeId) -> bool {
        self.request_holder_blacklist
            .get(&request_id)
            .is_some_and(|s| s.contains(holder))
    }

    /// Retract a holder's claims over the layer span it was asked to serve.
    ///
    /// A segment can cover SEVERAL shards, and `PipelineSegment::shard_id` names
    /// only the first of them — so retracting that one id is wrong whenever the
    /// failure was in a later shard. Observed live 2026-07-26: a whole-model
    /// segment (layers 0..16) failed on `blk.10`, i.e. shard 2, and the
    /// single-id version retracted shard 0, which the holder genuinely had. That
    /// is the "pushes work off a healthy peer" failure this whole mechanism has
    /// to avoid.
    ///
    /// The honest scope is "you were asked for this span and reported missing
    /// data", so every shard of the model overlapping `layer_range` loses its
    /// claim. That is broader than strictly necessary — the holder may well have
    /// some of them — but it is safe in the direction that matters: the holder's
    /// own next `ShardAnnounce` re-asserts whatever it really has, so an
    /// over-broad retraction costs at most one announce interval of not routing
    /// there, whereas retracting the wrong shard leaves the bad claim in place
    /// and keeps failing.
    pub fn retract_shard_holder_claims_for_range(
        &self,
        model_id: &crate::types::ModelId,
        holder: &NodeId,
        layer_range: (u32, u32),
        reason: &str,
    ) {
        for shard_id in self
            .model_registry
            .shards_overlapping_layers(model_id, layer_range)
        {
            self.retract_shard_holder_claim(&shard_id, holder, reason);
        }
    }

    /// Drop ONE holder's claim on ONE shard.
    ///
    /// Prefer [`Self::retract_shard_holder_claims_for_range`] from a failure
    /// path: a segment usually covers several shards and the failing one is not
    /// necessarily the segment's first.
    ///
    /// Holder claims arrive by gossip, so our registry can outlive the truth. The
    /// claim is re-established the moment that peer announces the shard again, so
    /// an over-eager retraction is self-correcting, while a missed one keeps
    /// failing every request routed there until the next announcement.
    pub fn retract_shard_holder_claim(
        &self,
        shard_id: &crate::types::ShardId,
        holder: &NodeId,
        reason: &str,
    ) {
        // Never retract our own claim from a remote error: our local shard
        // inventory is ground truth, not something a peer's message can revise.
        if holder == self.identity.node_id() {
            return;
        }
        // Durable, not just for this request: the DHT keeps advertising this
        // holder for hours, and the merge that reads it cannot remove anyone —
        // so a retraction that is not remembered is undone within seconds and
        // every later request is routed here again (gotcha #364).
        self.model_registry.retract_shard_holder(shard_id, holder);
        let remaining = self.model_registry.shard_holders(shard_id).len();
        tracing::info!(
            model = %shard_id.model_id,
            shard = shard_id.index,
            holder = %holder,
            remaining_holders = remaining,
            reason,
            "Retracted stale shard-holder claim after a missing-shard error"
        );
        self.emit_activity(
            crate::daemon::state::ActivityEvent::new(
                "network",
                "shard_holder_retracted",
                format!(
                    "{} no longer has shard {} of {} — re-routing",
                    &holder.to_string()[..8.min(holder.to_string().len())],
                    shard_id.index,
                    self.model_registry.display_name(&shard_id.model_id),
                ),
            )
            .with_model(shard_id.model_id.0.clone())
            .with_node(holder.to_string()),
        );
    }

    /// Attribute a segment's wall time and activation size to the in-flight
    /// trace, if one is registered. A no-op when the request has already
    /// finished or was never traced, so pipeline code can call it
    /// unconditionally without knowing whether tracing is active.
    pub fn record_segment_timing(
        &self,
        request_id: uuid::Uuid,
        segment_index: u16,
        elapsed_ms: u32,
        activation_bytes: u32,
    ) {
        if let Some(t) = self.active_traces.get(&request_id) {
            t.record_segment_timing(segment_index, elapsed_ms, activation_bytes);
        }
    }

    /// Snapshot of recent request traces, oldest first.
    pub fn recent_traces_snapshot(&self) -> Vec<crate::inference::trace::TraceSnapshot> {
        self.recent_traces
            .lock()
            .map(|b| b.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Per-peer serving performance, worst first.
    ///
    /// Joins the three places peer speed is already known — the health-ping
    /// round trip, the per-layer EMA the Parallax router uses, and the hedge
    /// tracker's EWMA — into the single table an operator actually wants. Each
    /// existed before this; none was readable from outside the scheduler.
    ///
    /// Only peers that have actually served something appear: a row of dashes
    /// for every peer in the registry would bury the handful that matter.
    /// In-flight requests that have not produced a token yet, with what they
    /// are doing and how long is left.
    ///
    /// Only requests carrying a progress snapshot appear: one that is already
    /// streaming has TTFT and a token count, which say more than a phase label
    /// would. Sorted longest-running first — if a user is looking here at all,
    /// it is because something feels stuck, and that is the row they want.
    pub fn active_request_rows(&self) -> Vec<serde_json::Value> {
        let mut rows: Vec<(u64, serde_json::Value)> = self
            .active_traces
            .iter()
            .filter_map(|e| {
                let trace = e.value();
                let p = trace.progress()?;
                Some((
                    p.elapsed_ms,
                    serde_json::json!({
                        "request_id": trace.request_id.to_string(),
                        "model": trace.model,
                        "operation": trace.operation,
                        "phase": p.phase,
                        "done": p.done,
                        "total": p.total,
                        "percent": p.percent,
                        "eta_ms": p.eta_ms,
                        "elapsed_ms": p.elapsed_ms,
                    }),
                ))
            })
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.0));
        rows.into_iter().map(|(_, v)| v).collect()
    }

    pub fn peer_performance_rows(&self) -> Vec<super::PeerPerformanceRow> {
        let hedge: std::collections::HashMap<crate::types::NodeId, (f32, u32)> = self
            .metrics
            .hedge_tracker
            .latency_by_holder()
            .into_iter()
            .map(|(n, ewma, samples)| (n, (ewma, samples)))
            .collect();

        let mut ids: std::collections::HashSet<crate::types::NodeId> =
            hedge.keys().cloned().collect();
        ids.extend(self.metrics.peer_speed.iter().map(|e| e.key().clone()));

        let mut rows: Vec<super::PeerPerformanceRow> = ids
            .into_iter()
            .map(|node| {
                let peer = self.peer_registry.get(&node);
                let (ewma_ms, samples) = hedge
                    .get(&node)
                    .map(|(e, s)| (Some(*e), *s))
                    .unwrap_or((None, 0));
                let speed = self.metrics.peer_speed.get(&node);
                super::PeerPerformanceRow {
                    node_id: node.to_string(),
                    rtt_ms: peer.as_ref().and_then(|p| p.latency_ms),
                    ms_per_layer: speed.as_ref().and_then(|s| s.ranking_ms_per_layer()),
                    // Reported per KB rather than per byte so the number is
                    // legible in a table instead of being all leading zeros.
                    prefill_ms_per_layer_kb: speed
                        .as_ref()
                        .and_then(|s| s.predict_ms(super::WorkKind::Prefill, 1, 1024)),
                    prefill_samples: speed.as_ref().map(|s| s.prefill_samples()).unwrap_or(0),
                    decode_samples: speed.as_ref().map(|s| s.decode_samples()).unwrap_or(0),
                    ewma_ms,
                    samples,
                    region: peer
                        .as_ref()
                        .and_then(|p| p.capability.as_ref().and_then(|c| c.region.clone())),
                }
            })
            .collect();

        // Slowest first: the reason to open this table is to find the drag.
        rows.sort_by(|a, b| {
            b.ewma_ms
                .unwrap_or(0.0)
                .total_cmp(&a.ewma_ms.unwrap_or(0.0))
        });
        rows
    }

    /// Whether this node should act as an application-level inference relay.
    ///
    /// The single source of truth for the relay-forwarder decision — the DHT
    /// relay-service registration, the `relay_capable` capability advertisement,
    /// and the inbound forward gates must all agree, or peers route through a
    /// node that then refuses to forward.
    ///
    /// Three ways to qualify:
    ///  - `--anchor` mode (a dedicated bootstrap/relay node),
    ///  - explicit `network.relay_forwarding = true` opt-in, or
    ///  - NETWORKING_PLAN Phase 3 auto-donation: the node is confirmed reachable
    ///    from the open internet AND has not opted out via
    ///    `network.relay_forwarding_auto = false`.
    ///
    /// The auto path exists because Phase 3's "any public node can opt in as a
    /// relay" was previously only a config flag that nothing ever set, so the
    /// swarm's entire relay capacity was whichever nodes ran `--anchor` — a
    /// single point of failure and a hard throughput ceiling for every NAT'd
    /// pair.
    pub fn relay_forwarding_enabled(&self) -> bool {
        if self.config.node.anchor_mode || self.config.network.relay_forwarding {
            return true;
        }
        self.config.network.relay_forwarding_auto
            && self
                .publicly_reachable
                .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Whether inference for `node` can be routed through an application-level
    /// relay, even though we hold no libp2p connection to it.
    ///
    /// NETWORKING_PLAN §4 Phase 1 calls for the router to prefer
    /// `direct → relayed-through-a-relay → fail`, via a "reachable-via-relay"
    /// tier in the scheduler "so a NAT'd holder is still usable". Without it the
    /// scheduler filtered purely on `connected_node_ids` (populated only by a
    /// libp2p Identify), so the app-relay could only ever *substitute* the data
    /// path for a peer we were already connected to — it could never make an
    /// unconnectable peer usable, which is the case it exists for.
    ///
    /// A peer qualifies when it speaks the relay protocol AND we can name a
    /// relay that reaches it:
    ///  - a **fresh learned route** (it has already relayed to us — definitive,
    ///    since the route was observed working), or
    ///  - a relay it advertises a reservation with (`relay_reservations`) that
    ///    we are also connected to — the forward is then guaranteed to land.
    ///
    /// Conservative by construction: an unknown peer, a peer with no relay
    /// feature, or one sharing no relay with us returns false, so this only ever
    /// *adds* candidates that have a concrete path.
    pub fn peer_reachable_via_relay(&self, node: &NodeId) -> bool {
        use crate::types::features;

        let Some(info) = self.peer_registry.get(node) else {
            return false;
        };

        // Must be able to decode a relayed envelope. Gossiped capability first,
        // then the proof recorded when it actually relayed something to us
        // (covers the cold-start window before capability gossip lands).
        let advertises = info
            .capability
            .as_ref()
            .is_some_and(|c| features::supports(c.features, features::RELAY));
        if !advertises && !self.relay_feature_proven(node, features::RELAY) {
            return false;
        }

        // A route we have already seen work.
        if let Some(bytes) = info.peer_id_bytes.as_deref() {
            if self.relay_route_for_peer(bytes).is_some() {
                return true;
            }
        }

        // Otherwise: does the target share a relay with us? Its advertised
        // reservations, intersected with the relays we are connected to.
        let Some(reservations) = info
            .capability
            .as_ref()
            .map(|c| c.relay_reservations.clone())
        else {
            return false;
        };
        drop(info);

        reservations.iter().any(|relay| {
            self.connected_node_ids.contains(relay)
                && self
                    .peer_registry
                    .get(relay)
                    .and_then(|p| p.capability.as_ref().map(|c| c.relay_capable))
                    .unwrap_or(false)
        })
    }

    /// Fresh learned relay route for a target peer (by peer-id bytes), or None
    /// if unknown or expired.
    pub fn relay_route_for_peer(&self, target_peer_bytes: &[u8]) -> Option<RelayRoute> {
        let entry = self.relay_routes.get(target_peer_bytes)?;
        if entry.learned_at.elapsed() > Duration::from_secs(RELAY_ROUTE_TTL_SECS) {
            return None;
        }
        Some(entry.clone())
    }

    /// Relay-side rate check: is `origin` under its forward budget for the
    /// current window? Increments the counter when it returns true.
    pub fn relay_forward_allowed(&self, origin_peer_bytes: &[u8]) -> bool {
        let now = Instant::now();
        let window = Duration::from_secs(RELAY_FORWARD_WINDOW_SECS);
        let mut entry = self
            .relay_forward_counters
            .entry(origin_peer_bytes.to_vec())
            .or_insert(RelayForwardCounter {
                count: 0,
                window_start: now,
            });
        if now.duration_since(entry.window_start) > window {
            entry.count = 0;
            entry.window_start = now;
        }
        if entry.count >= RELAY_FORWARD_MAX_PER_WINDOW {
            return false;
        }
        entry.count += 1;
        true
    }

    /// Record that `peer` has demonstrably used relay `features` (it sent us a
    /// relayed message addressed to us). ORs the newly-proven bits into any
    /// existing proof and refreshes the timestamp, so an active relay session
    /// keeps the proof warm. Cheap direct proof that sidesteps the capability-
    /// gossip cold-start window on the relay send path's feature gates.
    pub fn record_relay_proven_features(&self, peer: &NodeId, features: u64) {
        let now = Instant::now();
        self.relay_proven_features
            .entry(peer.clone())
            .and_modify(|e| {
                e.features |= features;
                e.proven_at = now;
            })
            .or_insert(RelayProvenFeatures {
                features,
                proven_at: now,
            });
    }

    /// Whether `peer` has proven support for ALL of `needed` within the freshness
    /// window (same TTL as a learned route — an active relay session re-proves it
    /// on every inbound message, so it never goes stale mid-session). Lets the
    /// relay send path trust a peer that just relayed to us even before its
    /// capability gossip arrives.
    pub fn relay_feature_proven(&self, peer: &NodeId, needed: u64) -> bool {
        self.relay_proven_features
            .get(peer)
            .filter(|e| e.proven_at.elapsed() <= Duration::from_secs(RELAY_ROUTE_TTL_SECS))
            .is_some_and(|e| crate::types::features::supports(e.features, needed))
    }

    /// Sweep expired relay routes + idle forward counters + stale proven-feature
    /// proofs. Wired to the HealthMonitor tick so all three maps stay bounded
    /// under peer churn.
    pub fn sweep_stale_relay_state(&self) {
        let route_ttl = Duration::from_secs(RELAY_ROUTE_TTL_SECS);
        self.relay_routes
            .retain(|_, r| r.learned_at.elapsed() <= route_ttl);
        let counter_ttl = Duration::from_secs(RELAY_FORWARD_WINDOW_SECS * 2);
        self.relay_forward_counters
            .retain(|_, c| c.window_start.elapsed() <= counter_ttl);
        self.relay_proven_features
            .retain(|_, e| e.proven_at.elapsed() <= route_ttl);
    }
}

/// Static label for an outcome. Keeps the Prometheus label set closed —
/// `Outcome::Error` carries a variant name that is bounded, but the label
/// itself collapses to "error" so the series count stays at routes × 4.
fn outcome_label(outcome: &crate::inference::trace::Outcome) -> &'static str {
    use crate::inference::trace::Outcome as O;
    match outcome {
        O::Pending => "pending",
        O::Ok => "ok",
        O::Error(_) => "error",
        O::Cancelled => "cancelled",
    }
}

/// Push a duration into a bounded ring plus its monotonic counters.
///
/// The ring is age- and size-bounded and supplies the bucket distribution; the
/// atomics supply `_count`/`_sum`, which MUST never fall or Prometheus'
/// `rate()`/`increase()` break when the ring wraps (R105).
fn record_duration_sample(
    ring: &std::sync::RwLock<std::collections::VecDeque<(Instant, f64)>>,
    count: &std::sync::atomic::AtomicU64,
    micros: &std::sync::atomic::AtomicU64,
    secs: f64,
) {
    use std::sync::atomic::Ordering;
    count.fetch_add(1, Ordering::Relaxed);
    micros.fetch_add((secs * 1_000_000.0).round() as u64, Ordering::Relaxed);
    if let Ok(mut ring) = ring.write() {
        let cutoff = Instant::now() - crate::api::metrics::LATENCY_SAMPLE_MAX_AGE;
        while ring.front().is_some_and(|(t, _)| *t < cutoff) {
            ring.pop_front();
        }
        if ring.len() >= 1000 {
            ring.pop_front();
        }
        ring.push_back((Instant::now(), secs));
    }
}

/// How much of a request this node served for a peer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServeKind {
    /// One segment of a multi-segment pipeline — we computed a layer range and
    /// handed the activations on. Counts as a forward, not as a request.
    Segment,
    /// The whole decode, start to finish, via the remote-generate fast path.
    /// This is what serving a peer looks like when one machine holds the model,
    /// so it counts as BOTH a request served and a forward.
    WholeRequest,
}

/// One unit of inference work performed for a peer.
///
/// A struct rather than positional arguments so a call site reads as a
/// description of the work, and so adding a dimension later does not silently
/// reorder anything at the two sites that build it.
#[derive(Clone, Copy, Debug)]
pub struct PeerServe {
    pub kind: ServeKind,
    /// Transformer layers computed.
    pub layers: u32,
    pub elapsed_ms: u64,
    /// Activation bytes sent back. The fast path streams tokens instead, so it
    /// passes 0.
    pub activation_bytes: u64,
    /// Tokens to bill for. Clamped here — see `MAX_CREDITABLE_TOKENS`.
    pub tokens: u32,
}

/// Ceiling on the tokens a single serve can bill for.
///
/// SEC: the requester chooses the token count on the `LayerForward` path, so
/// without a cap `token_count = u32::MAX` mints ~43B credits on the serving
/// node at the next flush. 8192 covers any realistic single forward — model
/// context lengths reach 32K-128K but those arrive as several forwards.
const MAX_CREDITABLE_TOKENS: u32 = 8192;

impl super::SharedState {
    /// Record inference work this node performed **for a peer** — the single
    /// place serving is counted and paid.
    ///
    /// Every path that does work for someone else funnels through here:
    /// `dispatch::layer_forward` (a pipeline segment) and
    /// `dispatch::remote_generate` (the whole decode). That is deliberate.
    /// These counters and the credit earn used to live in
    /// `track_forward_participation`, which only the segment path called, so a
    /// node serving through the fast path — the common case, since it is how a
    /// machine holding a whole model answers a peer — reported serving nothing
    /// and was paid nothing, while the requester was still debited. Measured
    /// 2026-08-09: a cross-node request cost the requester 430 credits and
    /// moved neither counter nor the balance on the node that did the work.
    ///
    /// So: do NOT count or bill serving at a call site. A new serving path that
    /// forgets to call this is invisible and unpaid, which is exactly the
    /// failure this consolidation exists to make impossible.
    ///
    /// Conversely, work this node does for *itself* — its own chat, its own
    /// segment inside a pipeline it coordinates — must NOT come through here.
    /// The dashboard promises "requests your computer handled for others" and
    /// "earns credits"; counting local work there told a user who only ever
    /// talks to themselves that they were serving the swarm.
    ///
    /// Relaxed atomics: these are counters read by a scrape, and this sits on
    /// the per-decode-step hot path of a serving node.
    pub fn record_peer_serve(&self, work: PeerServe) {
        use std::sync::atomic::Ordering::Relaxed;
        let m = &self.metrics;
        m.segments_served.fetch_add(1, Relaxed);
        m.layers_served.fetch_add(work.layers as u64, Relaxed);
        m.segment_serve_micros
            .fetch_add(work.elapsed_ms * 1000, Relaxed);
        m.segment_bytes_out
            .fetch_add(work.activation_bytes, Relaxed);

        // Both kinds are a forward computed for someone else; only the fast
        // path completed a whole request.
        m.forwards_served_atomic.fetch_add(1, Relaxed);
        if work.kind == ServeKind::WholeRequest {
            m.requests_served_atomic.fetch_add(1, Relaxed);
        }

        let tokens = work.tokens.clamp(1, MAX_CREDITABLE_TOKENS) as i64;
        let earned = crate::credit::ledger::RATE_INFERENCE_SERVE.saturating_mul(tokens);
        self.credits.pending_credit_earn.fetch_add(earned, Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a SharedState for relay-tier tests.
    #[cfg(test)]
    fn test_state(config: crate::config::Config) -> std::sync::Arc<crate::daemon::SharedState> {
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use tokio::sync::Mutex;

        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = std::sync::Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(config, identity, db, executor, None);
        state
    }

    /// A retraction must scope to the shards the holder was actually asked to
    /// serve. Retracting `segment.shard_id` alone dropped shard 0 for a failure
    /// in shard 2 (observed live 2026-07-26) — the wrong claim, leaving the bad
    /// one in place while penalising a shard the holder really had.
    #[test]
    fn retraction_scopes_to_the_shards_overlapping_the_failed_span() {
        let state = test_state(crate::config::Config::default());
        let model_id = crate::types::ModelId("m".to_string());
        let holder = crate::types::NodeId([9u8; 32]);

        // Three shards: 0..10, 10..16, 16..24.
        let shards: Vec<crate::types::ShardInfo> = [(0u32, 0u32, 10u32), (1, 10, 16), (2, 16, 24)]
            .iter()
            .map(|&(index, a, b)| crate::types::ShardInfo {
                index,
                layer_range: (a, b),
                size_bytes: 1u64,
                hash: [0u8; 32],
                tensors: vec![],
            })
            .collect();
        let manifest = crate::types::ModelManifest {
            id: model_id.clone(),
            name: "m".into(),
            architecture: crate::types::ModelArchitecture::Llama,
            num_layers: 24,
            num_params_billions: 1.0,
            quantization: crate::types::Quantization::Q4KM,
            total_size_bytes: 3,
            shard_count: 3,
            shards: shards.clone(),
            tokenizer_hash: [0u8; 32],
            manifest_hash: [0u8; 32],
            publisher: holder.clone(),
            publish_date: chrono::Utc::now(),
            license: String::new(),
            mmproj: None,
        };
        state.model_registry.register_manifest(manifest);
        for s in &shards {
            state.model_registry.record_shard_holder(
                crate::types::ShardId {
                    model_id: model_id.clone(),
                    index: s.index,
                },
                holder.clone(),
            );
        }

        // Failure serving layers 10..16 → only shard 1 overlaps.
        state.retract_shard_holder_claims_for_range(&model_id, &holder, (10, 16), "test");

        let held = |idx: u32| {
            state
                .model_registry
                .shard_holders(&crate::types::ShardId {
                    model_id: model_id.clone(),
                    index: idx,
                })
                .contains(&holder)
        };
        assert!(held(0), "shard 0 does not overlap 10..16 — must be kept");
        assert!(!held(1), "shard 1 overlaps 10..16 — must be retracted");
        assert!(held(2), "shard 2 does not overlap 10..16 — must be kept");
    }

    /// Retraction alone is futile — the DHT re-advertises the holder and the
    /// retry picks it again. The blacklist is what makes a retry change the
    /// outcome, and it is scoped per request so one bad data point cannot ban a
    /// peer globally.
    #[test]
    fn blacklist_is_scoped_to_one_request() {
        let state = test_state(crate::config::Config::default());
        let holder = crate::types::NodeId([7u8; 32]);
        let a = uuid::Uuid::new_v4();
        let b = uuid::Uuid::new_v4();

        state.blacklist_holder_for_request(a, &holder);

        assert!(state.holder_blacklisted_for_request(a, &holder));
        assert!(
            !state.holder_blacklisted_for_request(b, &holder),
            "a different request must not inherit the ban"
        );
        let other = crate::types::NodeId([8u8; 32]);
        assert!(
            !state.holder_blacklisted_for_request(a, &other),
            "only the failing holder is barred"
        );

        // Cleanup mirrors active_traces; after removal the peer is usable again.
        state.request_holder_blacklist.remove(&a);
        assert!(!state.holder_blacklisted_for_request(a, &holder));
    }

    /// Our own inventory is ground truth; a peer's error must never revise it.
    #[test]
    fn retraction_never_drops_our_own_claim() {
        let state = test_state(crate::config::Config::default());
        let model_id = crate::types::ModelId("m".to_string());
        let me = state.identity.node_id().clone();
        let shard_id = crate::types::ShardId {
            model_id: model_id.clone(),
            index: 0,
        };
        state
            .model_registry
            .record_shard_holder(shard_id.clone(), me.clone());

        state.retract_shard_holder_claim(&shard_id, &me, "test");

        assert!(
            state.model_registry.shard_holders(&shard_id).contains(&me),
            "a remote error must not retract our own local shard claim"
        );
    }

    /// Insert a peer with the given capability into the registry.
    #[cfg(test)]
    fn insert_peer(
        state: &crate::daemon::SharedState,
        node: &crate::types::NodeId,
        capability: Option<crate::types::NodeCapability>,
    ) {
        use crate::types::PeerInfo;
        state.peer_registry.insert(
            node.clone(),
            PeerInfo {
                node_id: node.clone(),
                addresses: vec![],
                capability,
                last_seen: chrono::Utc::now(),
                latency_ms: Some(50),
                trust_score: 0.5,
                peer_id_bytes: None,
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );
    }

    #[cfg(test)]
    fn capability_with(
        features: u64,
        relay_capable: bool,
        reservations: Vec<crate::types::NodeId>,
    ) -> crate::types::NodeCapability {
        crate::types::NodeCapability {
            node_id: crate::types::NodeId([0u8; 32]),
            gpu: None,
            cpu: None,
            ram_total_mb: 0,
            ram_available_mb: 0,
            disk_available_mb: 0,
            bandwidth_mbps: 0.0,
            hosted_shards: vec![],
            max_contribution: crate::types::ContributionLevel::Moderate,
            uptime_seconds: 0,
            version: String::new(),
            region: None,
            est_tokens_per_sec_7b: 0.0,
            os: None,
            observed_latencies: vec![],
            relay_capable,
            protocol_version: 0,
            features,
            relay_reservations: reservations,
            anchor_mode: false,
        }
    }

    /// The relay-donation decision must come out the same everywhere it is
    /// consulted, so all three qualifying paths are pinned here.
    #[test]
    fn relay_forwarding_enabled_covers_anchor_explicit_and_auto() {
        use std::sync::atomic::Ordering;

        // Plain NAT'd node: not an anchor, no opt-in, not publicly reachable.
        let mut config = crate::config::Config::default();
        config.node.anchor_mode = false;
        config.network.relay_forwarding = false;
        let state = test_state(config);
        assert!(
            !state.relay_forwarding_enabled(),
            "a NAT'd node must never advertise itself as a relay"
        );

        // Becoming publicly reachable auto-enables donation (Phase 3).
        state.publicly_reachable.store(true, Ordering::Relaxed);
        assert!(state.relay_forwarding_enabled());

        // ...unless the operator opted out of auto-donation.
        let mut config = crate::config::Config::default();
        config.network.relay_forwarding_auto = false;
        let state = test_state(config);
        state.publicly_reachable.store(true, Ordering::Relaxed);
        assert!(
            !state.relay_forwarding_enabled(),
            "relay_forwarding_auto = false must stop a public node donating"
        );

        // Explicit opt-in wins even without confirmed reachability.
        let mut config = crate::config::Config::default();
        config.network.relay_forwarding = true;
        config.network.relay_forwarding_auto = false;
        let state = test_state(config);
        assert!(state.relay_forwarding_enabled());

        // Anchor mode always forwards.
        let mut config = crate::config::Config::default();
        config.node.anchor_mode = true;
        config.network.relay_forwarding_auto = false;
        let state = test_state(config);
        assert!(state.relay_forwarding_enabled());
    }

    /// The scheduler's relay tier must only admit peers with a concrete path,
    /// or it re-introduces the dead-peer hang the liveness filter prevents.
    #[test]
    fn peer_reachable_via_relay_requires_a_real_path() {
        use crate::types::{features, NodeId};

        let state = test_state(crate::config::Config::default());
        let target = NodeId([7u8; 32]);
        let relay = NodeId([8u8; 32]);

        // Completely unknown peer.
        assert!(!state.peer_reachable_via_relay(&target));

        // Known, but advertises no relay support → cannot decode an envelope.
        insert_peer(
            &state,
            &target,
            Some(capability_with(0, false, vec![relay.clone()])),
        );
        assert!(
            !state.peer_reachable_via_relay(&target),
            "a peer without the RELAY feature must not be offered a relayed request"
        );

        // Speaks relay, names a relay — but we aren't connected to that relay,
        // so there is no path and it must stay unschedulable.
        insert_peer(
            &state,
            &target,
            Some(capability_with(features::RELAY, false, vec![relay.clone()])),
        );
        assert!(!state.peer_reachable_via_relay(&target));

        // Connecting to the shared relay completes the path.
        insert_peer(
            &state,
            &relay,
            Some(capability_with(features::RELAY, true, vec![])),
        );
        state.connected_node_ids.insert(relay.clone());
        assert!(
            state.peer_reachable_via_relay(&target),
            "a shared, connected, relay-capable node makes the target reachable"
        );

        // A connected node that is NOT relay-capable does not count.
        let bystander = NodeId([9u8; 32]);
        insert_peer(
            &state,
            &target,
            Some(capability_with(
                features::RELAY,
                false,
                vec![bystander.clone()],
            )),
        );
        insert_peer(
            &state,
            &bystander,
            Some(capability_with(features::RELAY, false, vec![])),
        );
        state.connected_node_ids.insert(bystander);
        assert!(!state.peer_reachable_via_relay(&target));
    }

    /// Cold start: a peer that has actually relayed to us is reachable even
    /// before its capability gossip lands (which is when `capability` is None).
    #[test]
    fn peer_reachable_via_relay_accepts_a_proven_route() {
        use crate::types::{features, NodeId};

        let state = test_state(crate::config::Config::default());
        let target = NodeId([4u8; 32]);
        let peer_bytes = vec![1u8, 2, 3, 4];

        // Registry entry with no capability at all — the cold-start shape
        // created by `ensure_relayed_origin_known`.
        use crate::types::PeerInfo;
        state.peer_registry.insert(
            target.clone(),
            PeerInfo {
                node_id: target.clone(),
                addresses: vec![],
                capability: None,
                last_seen: chrono::Utc::now(),
                latency_ms: None,
                trust_score: 0.5,
                peer_id_bytes: Some(peer_bytes.clone()),
                active_request_count: 0,
                first_seen: 0,
                verified_transaction_count: 0,
                is_lan_peer: false,
            },
        );
        assert!(!state.peer_reachable_via_relay(&target));

        // It relayed to us (proving RELAY) and we learned the route back.
        state.record_relay_proven_features(&target, features::RELAY);
        state.relay_routes.insert(
            peer_bytes,
            RelayRoute {
                relay_peer_bytes: vec![9u8; 8],
                target_node: target.clone(),
                learned_at: Instant::now(),
            },
        );
        assert!(
            state.peer_reachable_via_relay(&target),
            "a proven, freshly-learned route is the strongest reachability signal"
        );
    }

    #[test]
    fn relay_proven_features_record_check_and_isolate() {
        use crate::config::Config;
        use crate::identity::Identity;
        use crate::inference::executor::ModelExecutor;
        use crate::storage::db::Database;
        use crate::types::{features, NodeId};
        use std::sync::Arc;
        use tokio::sync::Mutex;

        let config = Config::default();
        let identity = Identity::generate();
        let temp = tempfile::tempdir().unwrap();
        let db = Database::open(temp.path()).unwrap();
        let executor = Arc::new(Mutex::new(ModelExecutor::new()));
        let (state, _, _) = crate::daemon::SharedState::new(config, identity, db, executor, None);

        let peer = NodeId([9u8; 32]);
        // Unknown peer → nothing proven.
        assert!(!state.relay_feature_proven(&peer, features::TENSOR_RELAY));

        // Receiving a relayed tensor proves TENSOR_RELAY...
        state.record_relay_proven_features(&peer, features::TENSOR_RELAY);
        assert!(state.relay_feature_proven(&peer, features::TENSOR_RELAY));
        // ...but a TENSOR_RELAY proof does NOT imply RELAY (distinct bits).
        assert!(!state.relay_feature_proven(&peer, features::RELAY));

        // Recording RELAY too ORs the bits — both now proven for that peer.
        state.record_relay_proven_features(&peer, features::RELAY);
        assert!(state.relay_feature_proven(&peer, features::RELAY));
        assert!(state.relay_feature_proven(&peer, features::TENSOR_RELAY));

        // A different peer is unaffected by another's proof.
        assert!(!state.relay_feature_proven(&NodeId([1u8; 32]), features::TENSOR_RELAY));

        // A stale proof (older than the route TTL) no longer counts, and the
        // sweep drops it. Guarded because a just-booted host's monotonic clock
        // may not be able to represent an instant that far in the past.
        if let Some(old) =
            Instant::now().checked_sub(Duration::from_secs(RELAY_ROUTE_TTL_SECS + 30))
        {
            state.relay_proven_features.insert(
                peer.clone(),
                RelayProvenFeatures {
                    features: features::TENSOR_RELAY,
                    proven_at: old,
                },
            );
            assert!(!state.relay_feature_proven(&peer, features::TENSOR_RELAY));
            state.sweep_stale_relay_state();
            assert!(!state.relay_proven_features.contains_key(&peer));
        }
    }

    #[test]
    fn forward_rate_limit_trips_and_resets() {
        // Pure window-counter logic, exercised without a full SharedState.
        let mut counter = RelayForwardCounter {
            count: RELAY_FORWARD_MAX_PER_WINDOW,
            window_start: Instant::now(),
        };
        // At cap → blocked.
        assert!(counter.count >= RELAY_FORWARD_MAX_PER_WINDOW);
        // After the window elapses the caller resets; emulate that.
        counter.count = 0;
        counter.window_start = Instant::now();
        assert!(counter.count < RELAY_FORWARD_MAX_PER_WINDOW);
    }

    /// The fast path — one machine holding a whole model answering a peer — is
    /// how most serving actually happens, and it recorded nothing at all.
    /// Measured live 2026-08-09: a cross-node request debited the requester 430
    /// credits while the node that did all 28 layers of work showed
    /// `requests_served = 0`, `forwards_served = 0` and an unchanged balance.
    #[test]
    fn serving_a_whole_request_for_a_peer_is_counted_and_paid() {
        let state = test_state(crate::config::Config::default());
        let m = &state.metrics;
        let relaxed = std::sync::atomic::Ordering::Relaxed;

        state.record_peer_serve(PeerServe {
            kind: ServeKind::WholeRequest,
            layers: 28,
            elapsed_ms: 5540,
            activation_bytes: 0,
            tokens: 43,
        });

        assert_eq!(
            m.requests_served_atomic.load(relaxed),
            1,
            "a whole request served for a peer must count as a request served"
        );
        assert_eq!(
            m.forwards_served_atomic.load(relaxed),
            1,
            "it is also a forward computed for someone else"
        );
        assert_eq!(
            state.credits.pending_credit_earn.load(relaxed),
            crate::credit::ledger::RATE_INFERENCE_SERVE * 43,
            "the node that did the work must be paid for it"
        );
        // The segment telemetry that already worked must keep working.
        assert_eq!(m.segments_served.load(relaxed), 1);
        assert_eq!(m.layers_served.load(relaxed), 28);
    }

    /// A segment is a forward, not a request — the requester assembles the
    /// request from several of them, and counting each as a whole request would
    /// inflate a multi-segment serve by the number of hops.
    #[test]
    fn serving_a_segment_counts_as_a_forward_but_not_a_request() {
        let state = test_state(crate::config::Config::default());
        let m = &state.metrics;
        let relaxed = std::sync::atomic::Ordering::Relaxed;

        state.record_peer_serve(PeerServe {
            kind: ServeKind::Segment,
            layers: 12,
            elapsed_ms: 40,
            activation_bytes: 8192,
            tokens: 1,
        });

        assert_eq!(m.forwards_served_atomic.load(relaxed), 1);
        assert_eq!(
            m.requests_served_atomic.load(relaxed),
            0,
            "a segment is one hop of a request, not a request"
        );
        assert_eq!(
            state.credits.pending_credit_earn.load(relaxed),
            crate::credit::ledger::RATE_INFERENCE_SERVE,
        );
        assert_eq!(m.segment_bytes_out.load(relaxed), 8192);
    }

    /// SEC: the requester picks `token_count` on the `LayerForward` path, so an
    /// unclamped bill lets a peer mint credits on every node it can reach.
    #[test]
    fn billing_is_clamped_against_a_forged_token_count() {
        let state = test_state(crate::config::Config::default());
        let relaxed = std::sync::atomic::Ordering::Relaxed;

        state.record_peer_serve(PeerServe {
            kind: ServeKind::Segment,
            layers: 1,
            elapsed_ms: 1,
            activation_bytes: 0,
            tokens: u32::MAX,
        });

        assert_eq!(
            state.credits.pending_credit_earn.load(relaxed),
            crate::credit::ledger::RATE_INFERENCE_SERVE * MAX_CREDITABLE_TOKENS as i64,
            "a forged token count must bill at the cap, not at face value"
        );
    }

    /// Zero tokens still represents real work — the floor keeps a serve from
    /// being free, which is what an `elapsed_ms` of real compute deserves.
    #[test]
    fn a_zero_token_serve_still_bills_the_one_token_floor() {
        let state = test_state(crate::config::Config::default());
        let relaxed = std::sync::atomic::Ordering::Relaxed;

        state.record_peer_serve(PeerServe {
            kind: ServeKind::WholeRequest,
            layers: 4,
            elapsed_ms: 12,
            activation_bytes: 0,
            tokens: 0,
        });

        assert_eq!(
            state.credits.pending_credit_earn.load(relaxed),
            crate::credit::ledger::RATE_INFERENCE_SERVE,
        );
    }
}
