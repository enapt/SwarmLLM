use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use futures::{SinkExt, StreamExt};

use crate::api::server::AppState;
use crate::daemon::SharedState;

/// Interval between stats push messages to WebSocket clients.
const WS_STATS_INTERVAL_SECS: u64 = 2;
/// Interval between WebSocket ping frames for liveness detection.
const WS_PING_INTERVAL_SECS: u64 = 30;
/// Maximum time since last pong before considering a connection dead.
/// Must be > WS_PING_INTERVAL_SECS to allow one full ping cycle.
const WS_PONG_TIMEOUT_SECS: u64 = 35;
/// Maximum concurrent WebSocket connections (prevents resource exhaustion).
const MAX_WS_CONNECTIONS: usize = 100;
/// Time-to-live for a WebSocket upgrade ticket. Client obtains via
/// `POST /api/admin/ws-ticket` (Bearer-authed) then immediately opens the
/// socket — 30s is ample for a round trip + constructor latency without
/// leaving a useful replay window.
const WS_TICKET_TTL: Duration = Duration::from_secs(30);

#[derive(serde::Deserialize)]
pub struct WsQuery {
    /// Short-lived single-use ticket from `POST /api/admin/ws-ticket`.
    /// Required because browsers cannot set an `Authorization` header on
    /// WebSocket upgrades — the ticket-in-URL round trip is how we keep
    /// Bearer-only auth on `/api/admin/ws`.
    #[serde(default)]
    pub t: Option<String>,
}

/// POST /api/admin/ws-ticket — issue a short-lived WS upgrade ticket.
///
/// Bearer-authed via the normal middleware. Returns `{"ticket": "<hex>"}`
/// which the frontend passes as `/api/admin/ws?t=<ticket>` on the next
/// upgrade. Ticket is single-use (atomic `remove` on consume) with a
/// `WS_TICKET_TTL` expiry. Uses 32 bytes of OS randomness — the ticket
/// is a lookup key, not a self-contained credential, so no HMAC / JWT
/// signing overhead.
pub async fn issue_ticket(State(state): State<AppState>) -> impl IntoResponse {
    use rand::RngCore;
    let mut bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    let ticket = hex::encode(bytes);
    state
        .shared_state
        .events
        .ws_tickets
        .insert(ticket.clone(), std::time::Instant::now());
    // Opportunistically prune expired entries so the map can't grow
    // unbounded from unused tickets (browser tab closed before WS open).
    let now = std::time::Instant::now();
    state
        .shared_state
        .events
        .ws_tickets
        .retain(|_, issued| now.duration_since(*issued) < WS_TICKET_TTL);
    axum::Json(serde_json::json!({ "ticket": ticket })).into_response()
}

/// GET /api/admin/ws — WebSocket handler for real-time dashboard updates.
///
/// Authentication: requires a valid single-use ticket in `?t=<hex>` (issued
/// by `POST /api/admin/ws-ticket`). The ticket is consumed atomically by
/// `DashMap::remove` and rejected if older than `WS_TICKET_TTL`.
///
/// Also validates the `Origin` header as defense in depth against DNS
/// rebinding attacks — a malicious page cannot rebind to `127.0.0.1` and
/// open this WebSocket because the ticket is obtained through an
/// authenticated POST first.
pub async fn handler(
    ws: WebSocketUpgrade,
    axum::extract::ConnectInfo(addr): axum::extract::ConnectInfo<std::net::SocketAddr>,
    axum::extract::Query(q): axum::extract::Query<WsQuery>,
    headers: axum::http::HeaderMap,
    State(state): State<AppState>,
) -> impl IntoResponse {
    // Origin validation — defense in depth against cross-site WS hijacking.
    // Browsers always send Origin on WebSocket upgrades (unlike same-origin
    // GETs), so we can require a match when it's present. Missing Origin
    // is allowed (non-browser clients like the Rust integration tests
    // don't send it).
    if let Some(origin) = headers.get(axum::http::header::ORIGIN) {
        let origin_str = origin.to_str().unwrap_or("");
        let port = state.config.node.listen_port;
        let allowed = [
            format!("http://localhost:{port}"),
            format!("http://127.0.0.1:{port}"),
            format!("http://[::1]:{port}"),
        ];
        if !allowed.iter().any(|a| a == origin_str) {
            tracing::warn!(
                origin = %origin_str,
                remote = %addr,
                "WebSocket connection rejected: origin not allowed"
            );
            return axum::http::StatusCode::FORBIDDEN.into_response();
        }
    }

    // Ticket validation — single-use, time-bounded.
    let ticket = match q.t {
        Some(t) if !t.is_empty() => t,
        _ => {
            tracing::warn!(remote = %addr, "WebSocket rejected: missing ticket");
            return axum::http::StatusCode::UNAUTHORIZED.into_response();
        }
    };
    let Some((_, issued)) = state.shared_state.events.ws_tickets.remove(&ticket) else {
        tracing::warn!(remote = %addr, "WebSocket rejected: unknown or already-consumed ticket");
        return axum::http::StatusCode::UNAUTHORIZED.into_response();
    };
    if issued.elapsed() >= WS_TICKET_TTL {
        tracing::warn!(remote = %addr, "WebSocket rejected: ticket expired");
        return axum::http::StatusCode::UNAUTHORIZED.into_response();
    }

    let shared = state.shared_state.clone();
    ws.on_upgrade(move |socket| handle_socket(socket, shared))
}

/// RAII guard to ensure ws_connection_count is decremented even on panic.
struct WsCountGuard(Arc<SharedState>);
impl Drop for WsCountGuard {
    fn drop(&mut self) {
        self.0
            .metrics
            .ws_connection_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

async fn handle_socket(socket: WebSocket, shared_state: Arc<SharedState>) {
    // Enforce a global WebSocket connection limit to prevent resource exhaustion.
    // Incrementing inside handle_socket (not before on_upgrade) prevents counter
    // leaks when the HTTP upgrade itself fails.
    let current = shared_state
        .metrics
        .ws_connection_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if current >= MAX_WS_CONNECTIONS {
        shared_state
            .metrics
            .ws_connection_count
            .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        tracing::warn!(
            current = current,
            max = MAX_WS_CONNECTIONS,
            "WebSocket connection limit reached — dropping"
        );
        return;
    }
    // RAII guard ensures decrement on panic or any exit path
    let _ws_guard = WsCountGuard(shared_state.clone());
    tracing::debug!(subsystem = "websocket", "DIAG: client connected");
    let (mut sender, mut receiver) = socket.split();

    // Track last pong timestamp for dead connection detection
    let last_pong = std::sync::Arc::new(tokio::sync::Mutex::new(tokio::time::Instant::now()));
    let last_pong_push = last_pong.clone();

    // Spawn a task to push stats every 2 seconds + ping every 30 seconds + activity events
    let push_state = shared_state.clone();
    let mut dashboard_rx = shared_state.events.dashboard_tx.subscribe();
    let mut activity_rx = shared_state.events.activity_tx.subscribe();
    let mut push_task = tokio::spawn(async move {
        // Send the current peer list immediately on connect so the dashboard
        // populates without waiting for the first peer_list_changed event.
        let initial_peers = build_peer_list_message(&push_state);
        let _ = sender
            .send(Message::Text(
                serde_json::to_string(&initial_peers)
                    .unwrap_or_default()
                    .into(),
            ))
            .await;

        // Replay activity history so new clients see startup events
        {
            let events: Vec<_> = {
                let history = push_state.events.activity_history.lock();
                history.iter().cloned().collect()
            };
            for event in events {
                let msg = serde_json::json!({
                    "type": "activity_event",
                    "data": event,
                });
                let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                if sender.send(Message::Text(msg_str.into())).await.is_err() {
                    break;
                }
            }
        }

        let mut stats_interval = tokio::time::interval(Duration::from_secs(WS_STATS_INTERVAL_SECS));
        let mut ping_interval = tokio::time::interval(Duration::from_secs(WS_PING_INTERVAL_SECS));
        let mut tick_count: u64 = 0;
        loop {
            tokio::select! {
                _ = stats_interval.tick() => {
                    let msg = get_or_build_stats_message(&push_state).await;
                    if sender.send(Message::Text((*msg).clone().into())).await.is_err() {
                        break;
                    }
                }
                _ = ping_interval.tick() => {
                    tick_count += 1;
                    // Skip only the first tick (fires immediately)
                    if tick_count == 1 {
                        continue;
                    }
                    // Check if last pong was received within a reasonable window.
                    // Ping fires every 30s; allow up to 35s since last pong to detect
                    // dead connections within ~1 ping cycle instead of ~2.
                    let last = *last_pong_push.lock().await;
                    if last.elapsed() > Duration::from_secs(WS_PONG_TIMEOUT_SECS) {
                        tracing::debug!("WebSocket client failed pong check — closing");
                        break;
                    }
                    let ping_data: bytes::Bytes =
                        chrono::Utc::now().timestamp().to_le_bytes().to_vec().into();
                    if sender.send(Message::Ping(ping_data)).await.is_err() {
                        break;
                    }
                }
                signal = dashboard_rx.recv() => {
                    let sig = match signal {
                        Ok(s) => s,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(
                                dropped = n,
                                "WebSocket client lagged on dashboard channel — re-syncing"
                            );
                            // Force the client to re-fetch authoritative state.
                            // Without this, missed PeersChanged / ModelsChanged
                            // signals leave the dashboard stale until the next
                            // 2s stats tick or page reload.
                            let resync = serde_json::to_string(&serde_json::json!({
                                "type": "models_changed"
                            }))
                            .unwrap_or_default();
                            if sender.send(Message::Text(resync.into())).await.is_err() {
                                break;
                            }
                            continue;
                        }
                        Err(_) => break, // channel closed (daemon shutdown)
                    };
                    let msg_str = match sig {
                        crate::daemon::state::DashboardSignal::ModelsChanged => {
                            serde_json::to_string(&serde_json::json!({"type": "models_changed"})).unwrap_or_default()
                        }
                        crate::daemon::state::DashboardSignal::PeersChanged => {
                            let peers = build_peer_list_message(&push_state);
                            serde_json::to_string(&peers).unwrap_or_default()
                        }
                        crate::daemon::state::DashboardSignal::UpdateAvailable(info) => {
                            serde_json::to_string(&serde_json::json!({
                                "type": "update_available",
                                "data": {
                                    "current_version": info.current_version,
                                    "latest_version": info.latest_version,
                                    "changelog": info.changelog,
                                    "published_at": info.published_at,
                                    "downloaded": info.downloaded,
                                }
                            })).unwrap_or_default()
                        }
                    };
                    if sender.send(Message::Text(msg_str.into())).await.is_err() {
                        break;
                    }
                }
                activity = activity_rx.recv() => {
                    let event = match activity {
                        Ok(e) => e,
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                            tracing::warn!(
                                dropped = n,
                                "WebSocket client lagged on activity channel — events dropped"
                            );
                            continue;
                        }
                        Err(_) => break,
                    };
                    let msg = serde_json::json!({
                        "type": "activity_event",
                        "data": event,
                    });
                    let msg_str = serde_json::to_string(&msg).unwrap_or_default();
                    if sender.send(Message::Text(msg_str.into())).await.is_err() {
                        break;
                    }
                }
            }
        }
    });

    // Race push_task against receiver loop — either side exiting cleans up the other.
    // Without this, if push_task exits first (sender dropped), the receiver loop blocks
    // indefinitely waiting for the client to send a close frame.
    let recv_loop = async {
        while let Some(msg) = receiver.next().await {
            match msg {
                Ok(Message::Close(_)) | Err(_) => break,
                Ok(Message::Pong(_)) => {
                    *last_pong.lock().await = tokio::time::Instant::now();
                }
                _ => {}
            }
        }
    };

    tokio::select! {
        _ = &mut push_task => {
            tracing::debug!(subsystem = "websocket", "DIAG: push_task exited first");
        }
        _ = recv_loop => {
            push_task.abort();
            tracing::debug!(subsystem = "websocket", "DIAG: receiver loop exited first");
        }
    }
    // _ws_guard drop handles decrement
    tracing::debug!(subsystem = "websocket", "DIAG: client disconnected");
}

/// Build a `peer_list` WS message with the current peer registry snapshot.
/// Same data shape as GET /api/admin/peers so the frontend can share one render path.
fn build_peer_list_message(state: &SharedState) -> serde_json::Value {
    let peers: Vec<serde_json::Value> = state
        .peer_registry
        .iter()
        .map(|entry| crate::api::admin::serialize_peer_to_json(entry.value(), state, false))
        .collect();

    serde_json::json!({
        "type": "peer_list",
        "data": { "peers": peers }
    })
}

/// TTL for the shared stats JSON cache. A client ticking within this window
/// of the last build reuses the cached string instead of re-scanning registries.
/// Set slightly below the 2s WS_STATS_INTERVAL_SECS so the first client per
/// tick rebuilds while subsequent clients in the same tick hit the hot cache.
const STATS_CACHE_TTL_MS: u128 = 1500;

/// Return a cached stats message if still fresh, otherwise rebuild and cache.
///
/// Called by every WebSocket client every 2s. With N connected clients the
/// per-client O(shards+peers) scans collapse into at most one rebuild per
/// 1.5s, shared across all clients.
async fn get_or_build_stats_message(state: &SharedState) -> std::sync::Arc<String> {
    {
        let cache = state.metrics.stats_cache.lock();
        if let Some((built_at, msg)) = cache.as_ref() {
            if built_at.elapsed().as_millis() < STATS_CACHE_TTL_MS {
                return msg.clone();
            }
        }
    }
    // Stampede guard: when the cache expires and multiple WS push tasks tick
    // at the same TTL boundary, only one performs the rebuild. Losers return
    // the stale value (acceptable: it's at most STATS_CACHE_TTL_MS old).
    use std::sync::atomic::Ordering;
    struct BuildFlag<'a>(&'a std::sync::atomic::AtomicBool);
    impl Drop for BuildFlag<'_> {
        fn drop(&mut self) {
            self.0.store(false, Ordering::Release);
        }
    }
    let _flag = match state.metrics.stats_building.compare_exchange(
        false,
        true,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => BuildFlag(&state.metrics.stats_building),
        Err(_) => {
            // Another task is rebuilding — return the stale value. On very
            // first call (no cache yet), fall through and build ourselves to
            // unblock — the race is bounded to the first tick per daemon.
            if let Some(stale) = state
                .metrics
                .stats_cache
                .lock()
                .as_ref()
                .map(|(_, m)| m.clone())
            {
                return stale;
            }
            BuildFlag(&state.metrics.stats_building)
        }
    };
    let msg = std::sync::Arc::new(build_stats_message(state).await);
    *state.metrics.stats_cache.lock() = Some((std::time::Instant::now(), msg.clone()));
    // _flag Drop clears the atomic — panic-safe.
    msg
}

async fn build_stats_message(state: &SharedState) -> String {
    // Snapshot the locked values into stack copies and drop the guards BEFORE
    // the DashMap full scans + serde_json construction below. Holding RwLock
    // guards across hundreds of microseconds of synchronous work parks the
    // inference hot path's apply_credit writers and the health monitor's
    // node_stats writer, every 2 seconds, per connected WS client.
    let (uptime_start, requests_made) = {
        let stats = state.metrics.node_stats.read().await;
        (stats.uptime_start, stats.requests_made)
    };
    let credit_json = {
        let credit = state.credits.credit_balance.read().await;
        crate::api::credit_summary_json(&credit)
    };
    let local_node_id = state.identity.node_id().clone();

    // Collect active acquisition progress with per-shard detail
    let acquisitions: Vec<serde_json::Value> = state
        .models
        .acquisition_progress
        .iter()
        .map(|entry| crate::api::admin_models::serialize_acquisition_to_json(entry.value(), state))
        .collect();

    // Build shard registry. Since the stats message is shared across all
    // clients via stats_cache, always include the full registry — per-client
    // diffing is no longer possible without per-client cache state.
    let mut per_model: HashMap<String, Vec<(u32, bool, usize)>> = HashMap::new();
    for (shard_id, holders) in state.model_registry.all_shard_entries() {
        let model_id = shard_id.model_id.0.clone();
        let local = holders.contains(&local_node_id);
        per_model
            .entry(model_id)
            .or_default()
            .push((shard_id.index, local, holders.len()));
    }
    let shard_registry_val: serde_json::Value = per_model
        .iter()
        .map(|(model_id, shards)| {
            let shard_arr: Vec<serde_json::Value> = shards
                .iter()
                .map(|(index, local, holder_count)| {
                    let mid = crate::types::ModelId(model_id.clone());
                    let in_vram = if *local {
                        state.is_shard_in_vram(&mid, *index)
                    } else {
                        false
                    };
                    serde_json::json!({
                        "index": index,
                        "local": local,
                        "holders": holder_count,
                        "in_vram": in_vram,
                    })
                })
                .collect();
            (model_id.clone(), serde_json::Value::Array(shard_arr))
        })
        .collect::<serde_json::Map<String, serde_json::Value>>()
        .into();

    // Derive LAN count from the authoritative peer_registry rather than the
    // monotonically-incremented atomic counter, which can overcount after
    // reconnect cycles on platforms without mDNS-expire events (WSL2).
    let lan_peers = state
        .peer_registry
        .iter()
        .filter(|entry| entry.is_lan_peer)
        .count();

    // Use connected_node_ids as the connectivity oracle (gotcha #86).
    // peer_registry is intentionally preserved across mid-pipeline disconnects
    // for reconnect attempts, so it overcounts live peers. R105 fixed the
    // Prometheus metric the same way; the WS payload was missed.
    let peers_connected = state.connected_node_ids.len() as u32;

    // R110: refresh + serialise the swarm capacity snapshot. Cheap (single
    // pass over registries), runs once per stats build (every ~2s per WS
    // client, but coalesced via the cache stampede guard so we only build
    // one snapshot per cache TTL). Folded into stats_update so the
    // dashboard header gets it without a new WS message type.
    crate::daemon::state::refresh_swarm_capacity(state);
    let capacity_json = {
        let snap = state.metrics.swarm_capacity.load_full();
        serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({}))
    };
    // R111: piggyback the wishlist on the same stats payload. Same
    // coalescing benefits — every dashboard render gets a fresh ranked
    // queue without spamming a new WS message type or polling REST.
    crate::model::auto_manage::refresh_wishlist(state);
    crate::model::auto_manage::quant::refresh_quant_recommendations(state);
    let wishlist_json = {
        let snap = state.models.wishlist.load_full();
        serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({}))
    };
    let quant_recs_json = {
        let snap = state.models.quant_recommendations.load_full();
        serde_json::to_value(&*snap).unwrap_or_else(|_| serde_json::json!({}))
    };
    // R134.5: size-bounded snapshot of the inter-pool model availability
    // signal (R134) — group by pool, trim stale, cap to keep the stats
    // payload small. Full data still reachable via REST.
    let foreign_pool_catalog_json = {
        use std::collections::BTreeMap;
        let now_ms = crate::types::unix_now_ms();
        state.credits.trim_stale_foreign_pool_catalog(
            now_ms,
            crate::daemon::dispatch::FOREIGN_POOL_CATALOG_MAX_AGE_MS,
        );
        const WS_POOL_CATALOG_MAX_POOLS: usize = 30;
        const WS_POOL_CATALOG_MAX_MODELS_PER_POOL: usize = 12;
        let mut by_pool: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for entry in state.credits.foreign_pool_catalog.iter() {
            let (pool, model) = entry.key();
            by_pool
                .entry(format!("{pool}"))
                .or_default()
                .push(model.0.clone());
        }
        let pools: Vec<serde_json::Value> = by_pool
            .into_iter()
            .take(WS_POOL_CATALOG_MAX_POOLS)
            .map(|(pool_id, mut models)| {
                models.sort();
                models.truncate(WS_POOL_CATALOG_MAX_MODELS_PER_POOL);
                serde_json::json!({ "pool_id": pool_id, "models": models })
            })
            .collect();
        serde_json::json!({ "pools": pools })
    };

    // Network mode flags. Folded into stats_update so the unified Network
    // Status banner can render the right named state (connecting / global /
    // private / lan / solo / offline) without a separate fetch.
    let private_mode = state
        .credits
        .private_mode
        .load(std::sync::atomic::Ordering::Relaxed);
    let offline_mode = state
        .credits
        .offline_mode
        .load(std::sync::atomic::Ordering::Relaxed);
    let allow_lan = state.config.pool.private_mode_allow_lan;

    let mut data = serde_json::json!({
        "peers": peers_connected,
        "lan_peers": lan_peers,
        "credits": credit_json,
        "active_requests": state.active_pipelines.len(),
        "requests_served": state.metrics.requests_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        "requests_made": requests_made,
        "forwards_served": state.metrics.forwards_served_atomic.load(std::sync::atomic::Ordering::Relaxed),
        // CORRECTNESS (R105): clamp to 0. NTP-step backwards on a freshly-
        // booted node would otherwise produce a negative uptime in the WS
        // payload (the REST handler at admin.rs:113 already clamps).
        "uptime_seconds": (chrono::Utc::now() - uptime_start).num_seconds().max(0),
        "acquisitions": acquisitions,
        "swarm_capacity": capacity_json,
        "wishlist": wishlist_json,
        "quant_recommendations": quant_recs_json,
        "foreign_pool_catalog": foreign_pool_catalog_json,
        // SWARM-SPEC Layer 2/3 metrics (R136): hedge + prefetch
        // tracker snapshots so operators can observe layer behavior.
        "swarm_spec": {
            "hedge": state.metrics.hedge_tracker.metrics(),
            "prefetch": state.metrics.prefetch_orchestrator.metrics(),
        },
        "network_mode": {
            "private_mode": private_mode,
            "offline_mode": offline_mode,
            "private_mode_allow_lan": allow_lan,
        },
    });

    data["shard_registry"] = shard_registry_val;

    // Peer shard download progress (from gossip)
    {
        let mut peer_dl: Vec<serde_json::Value> = Vec::new();
        for entry in state.models.peer_shard_downloads.iter() {
            let shard_id = entry.key();
            for (nid, pct) in entry.value().iter() {
                peer_dl.push(serde_json::json!({
                    "model_id": shard_id.model_id.0,
                    "shard_index": shard_id.index,
                    "node_id": format!("{}", nid),
                    "progress_pct": pct,
                }));
            }
        }
        if !peer_dl.is_empty() {
            data["peer_downloads"] = serde_json::json!(peer_dl);
        }
    }

    // Region summary for network map — includes demand rates and coverage gaps
    {
        let mut region_counts: HashMap<String, u64> = HashMap::new();
        if let Some(ref region) = state.config.identity.region {
            *region_counts.entry(region.to_uppercase()).or_insert(0) += 1;
        }
        for peer in state.peer_registry.iter() {
            if let Some(ref cap) = peer.value().capability {
                if let Some(ref region) = cap.region {
                    *region_counts.entry(region.to_uppercase()).or_insert(0) += 1;
                }
            }
        }
        if !region_counts.is_empty() {
            data["region_summary"] = serde_json::to_value(&region_counts).unwrap_or_default();
        }

        // Per-region demand rates
        let mut region_demand: HashMap<String, HashMap<String, f64>> = HashMap::new();
        for entry in state.region_demand.iter() {
            let (model_id, region) = entry.key();
            region_demand
                .entry(region.clone())
                .or_default()
                .insert(model_id.0.clone(), *entry.value());
        }
        if !region_demand.is_empty() {
            data["region_demand"] = serde_json::to_value(&region_demand).unwrap_or_default();
        }

        // Coverage gaps: per-region list of models with 0 holders
        let all_model_ids: Vec<String> = state
            .model_registry
            .models()
            .iter()
            .map(|m| m.id.0.clone())
            .collect();
        if !all_model_ids.is_empty() && !region_counts.is_empty() {
            let mut gaps: HashMap<String, Vec<String>> = HashMap::new();
            for region_code in region_counts.keys() {
                let mut region_gaps = Vec::new();
                for model_id in &all_model_ids {
                    let key = (region_code.clone(), crate::types::ModelId(model_id.clone()));
                    let has_holders = state
                        .region_shard_summaries
                        .get(&key)
                        .map(|s| s.shard_counts.iter().any(|(_, c)| *c > 0))
                        .unwrap_or(false);
                    if !has_holders {
                        region_gaps.push(model_id.clone());
                    }
                }
                if !region_gaps.is_empty() {
                    gaps.insert(region_code.clone(), region_gaps);
                }
            }
            if !gaps.is_empty() {
                data["region_coverage_gaps"] = serde_json::to_value(&gaps).unwrap_or_default();
            }
        }
    }

    let msg = serde_json::json!({
        "type": "stats_update",
        "data": data,
    });

    serde_json::to_string(&msg).unwrap_or_default()
}
